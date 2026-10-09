from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime

import torch
import wandb

from models import dnnlib
from models.torch_utils import distributed as dist
from training_diffusionpde import training_loop
from utils.yaml_config import Config, process_arguments
from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from posteriorbench.training_source import is_training_field_source

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")


def resolve_resume_paths(state_path):
    state_path = os.path.abspath(os.fspath(state_path))
    match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(state_path))
    if not match or not os.path.isfile(state_path):
        raise ValueError("--resume must point to an existing training-state-<nimg>.pt")
    resume_nimg = int(match.group(1))
    snapshot_path = os.path.join(
        os.path.dirname(state_path),
        f"network-snapshot-{resume_nimg}.pkl",
    )
    if not os.path.isfile(snapshot_path):
        raise ValueError(
            "The matching EMA snapshot must be adjacent to the training state: "
            f"{snapshot_path}"
        )
    return state_path, snapshot_path, resume_nimg


def main():
    args = process_arguments()
    conf = Config(args)

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    if conf["seed"] is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        conf.update("seed", int(seed))

    if dist.get_rank() == 0:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "PosteriorBench"),
            config=conf.to_dict(),
            name=conf["name"],
            mode=conf["wandb"],
        )
        if wandb.run is not None:
            wandb.run.log_code(root=".")

    c = dnnlib.EasyDict()
    if not is_training_field_source(conf["data"]):
        raise ValueError(
            "Canonical DiffusionPDE training requires a materialized Hugging "
            "Face PDEFieldDataset_hf source"
        )
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name="training_diffusionpde.dataset.PDEDiffusionDataset",
        path=conf["data"],
        resolution=conf["resolution"],
        use_labels=conf["cond"],
        xflip=conf["xflip"],
        cache=conf["cache"],
        normalization_scale=conf.get("normalization_scale", 0.5),
    )
    c.data_loader_kwargs = dnnlib.EasyDict(
        num_workers=conf["workers"],
        pin_memory=True,
        prefetch_factor=2,
    )
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=conf["lr"],
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution
        c.dataset_kwargs.max_size = len(dataset_obj)
        dataset_shape = as_spatial_shape(dataset_obj.resolution)
        if conf["cond"] and not dataset_obj.has_labels:
            raise ValueError("--cond=True requires labels specified in dataset metadata")
        del dataset_obj
    except IOError as err:
        raise ValueError(f"--data: {err}")

    if conf["arch"] == "ddpmpp":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif conf["arch"] == "ncsnpp":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )
    elif conf["arch"] == "adm":
        c.network_kwargs.update(
            model_type="DhariwalUNet",
            model_channels=192,
            channel_mult=[1, 2, 3, 4],
        )
    else:
        raise ValueError(f"Invalid DiffusionPDE architecture: {conf['arch']}")

    if conf["precond"] == "vp":
        c.network_kwargs.class_name = "training_diffusionpde.networks.VPPrecond"
        c.loss_kwargs.class_name = "training_diffusionpde.loss.VPLoss"
    elif conf["precond"] == "ve":
        c.network_kwargs.class_name = "training_diffusionpde.networks.VEPrecond"
        c.loss_kwargs.class_name = "training_diffusionpde.loss.VELoss"
    elif conf["precond"] == "edm":
        c.network_kwargs.class_name = "training_diffusionpde.networks.EDMPrecond"
        c.loss_kwargs.class_name = "training_diffusionpde.loss.EDMLoss"
    else:
        raise ValueError(f"Invalid DiffusionPDE preconditioning: {conf['precond']}")

    if conf["cbase"] is not None:
        c.network_kwargs.model_channels = conf["cbase"]
    if conf["cres"] is not None:
        c.network_kwargs.channel_mult = conf["cres"]
    if float(conf["augment"]) > 0:
        c.augment_kwargs = dnnlib.EasyDict(
            class_name="training_diffusionpde.augment.AugmentPipe",
            p=conf["augment"],
        )
        c.augment_kwargs.update(
            xflip=1e8,
            yflip=1,
            scale=1,
            rotate_frac=1,
            aniso=1,
            translate_frac=1,
        )
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=conf["dropout"], use_fp16=conf["fp16"])
    nn_resolution = conf.get("nn_resolution", None)
    if nn_resolution is not None:
        nn_shape = as_spatial_shape(nn_resolution, name="nn_resolution")
        if nn_shape[0] > dataset_shape[0] or nn_shape[1] > dataset_shape[1]:
            raise ValueError(
                f"nn_resolution={spatial_to_config(nn_shape)} must fit within data resolution {spatial_to_config(dataset_shape)}"
            )
        c.network_kwargs.update(img_resolution=spatial_to_config(nn_shape))
    else:
        c.network_kwargs.update(img_resolution=spatial_to_config(dataset_shape))

    c.total_kimg = max(int(conf["duration"] * 1000), 1)
    c.lr_rampup_kimg = int(conf["lr_rampup"] * 1000)
    c.ema_halflife_kimg = int(conf["ema"] * 1000)
    c.update(batch_size=conf["batch"], batch_gpu=conf["batch_gpu"])
    c.update(loss_scaling=conf["ls"], cudnn_benchmark=conf["bench"], allow_tf32=conf.get("tf32", False))
    c.update(
        kimg_per_tick=conf["tick"],
        snapshot_ticks=conf["snap"],
        state_dump_ticks=conf["dump"],
    )
    c.cond = conf["cond"]
    c.seed = conf["seed"]

    if conf["resume"]:
        state_path, snapshot_path, resume_nimg = resolve_resume_paths(conf["resume"])
        c.resume_pkl = snapshot_path
        c.resume_nimg = resume_nimg
        c.resume_state_dump = state_path

    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        start_time = wandb.run.start_time if wandb.run is not None else datetime.now().timestamp()
        formatted_time = datetime.fromtimestamp(start_time).strftime("%m%d_%H%M%S")
        run_id = wandb.run.id if wandb.run is not None else "local"
        desc = f"{formatted_time}-{conf['name']}-{run_id}"
        c.run_dir = os.path.join(conf["outdir"], desc)
        assert not os.path.exists(c.run_dir)

    dist.print0()
    dist.print0("DiffusionPDE training options:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset name:            {dataset_name}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    dist.print0(f"Network architecture:    {conf['arch']}")
    dist.print0(f"Preconditioning & loss:  {conf['precond']}")
    dist.print0(f"Number of GPUs:          {dist.get_world_size()}")
    dist.print0(f"Batch size:              {c.batch_size}")
    dist.print0(f"Mixed-precision:         {c.network_kwargs.use_fp16}")
    dist.print0(f"TF32 enabled:            {c.allow_tf32}")
    dist.print0()

    if conf["dry_run"]:
        dist.print0("Dry run; exiting.")
        if dist.get_rank() == 0:
            wandb.finish()
        return

    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    training_loop.training_loop(**c)

    if dist.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
