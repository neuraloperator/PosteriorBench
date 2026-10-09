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
from training_fundps import training_loop
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
    resume_paths = resolve_resume_paths(conf["resume"]) if conf["resume"] else None

    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    if conf["seed"] is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        conf.update("seed", int(seed))

    run_dir = None
    if dist.get_rank() == 0:
        run_id = (
            "local"
            if conf["wandb"] == "disabled"
            else os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
        )
        formatted_time = datetime.now().strftime("%m%d_%H%M%S")
        desc = f"{formatted_time}-{conf['name']}-{run_id}"
        run_dir = os.path.join(conf["outdir"], desc)
        if os.path.exists(run_dir):
            raise FileExistsError(f"Output directory already exists: {run_dir}")
        if conf["wandb"] != "disabled" or not conf["dry_run"]:
            os.makedirs(run_dir, exist_ok=False)
        if conf["wandb"] != "disabled":
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "PosteriorBench"),
                config=conf.to_dict(),
                dir=run_dir,
                id=run_id,
                name=conf["name"],
                mode=conf["wandb"],
            )
            if wandb.run is not None:
                wandb.run.log_code(root=".")

    c = dnnlib.EasyDict()
    if not is_training_field_source(conf["data"]):
        raise ValueError(
            "Canonical Fun-DDPS prior training requires a materialized Hugging "
            "Face PDEFieldDataset_hf source"
        )
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name="training_funddps.dataset.FunDDPSPriorDataset",
        path=conf["data"],
        resolution=conf["resolution"],
        use_labels=conf["cond"],
        xflip=conf["xflip"],
        cache=conf["cache"],
        target_channels=conf.get("target_channels", None),
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
    c.sampler_kwargs = dnnlib.EasyDict(
        class_name="training_fundps.noise_samplers.RBFKernel",
        scale=conf["rbf_scale"],
    )

    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        dataset_channels = dataset_obj.channels
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
    elif conf["arch"] == "ddpmpp-uno":
        c.network_kwargs.update(
            model_type="SongUNO",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
            cond=conf["cond"],
            attn_resolutions=conf["attn_resolutions"],
            num_blocks=conf["num_blocks"],
            fmult=conf["fmult"],
            rank=conf["rank"],
        )
    else:
        raise ValueError(f"Invalid Fun-DDPS prior architecture: {conf['arch']}")

    if conf["precond"] == "vp":
        c.network_kwargs.class_name = "training_fundps.networks.VPPrecond"
        c.loss_kwargs.class_name = "training_fundps.loss.VPLoss"
    elif conf["precond"] == "ve":
        c.network_kwargs.class_name = "training_fundps.networks.VEPrecond"
        c.loss_kwargs.class_name = "training_fundps.loss.VELoss"
    elif conf["precond"] == "edm":
        c.network_kwargs.class_name = "training_fundps.networks.EDMPrecond"
        c.loss_kwargs.class_name = (
            "training_fundps.loss.EDMLossWithSampler"
            if conf["arch"] == "ddpmpp-uno"
            else "training_fundps.loss.EDMLoss"
        )
    else:
        raise ValueError(f"Invalid Fun-DDPS prior preconditioning: {conf['precond']}")

    if conf["cbase"] is not None:
        c.network_kwargs.model_channels = conf["cbase"]
    if conf["cres"] is not None:
        c.network_kwargs.channel_mult = conf["cres"]
    c.network_kwargs.update(dropout=conf["dropout"], use_fp16=conf["fp16"])
    if conf["nn_resolution"] is not None:
        nn_shape = as_spatial_shape(conf["nn_resolution"], name="nn_resolution")
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
    c.update(
        loss_scaling=conf["ls"],
        cudnn_benchmark=conf["bench"],
        allow_tf32=conf.get("tf32", False),
    )
    c.update(
        kimg_per_tick=conf["tick"],
        snapshot_ticks=conf["snap"],
        state_dump_ticks=conf["dump"],
    )
    c.cond = conf["cond"]
    c.seed = conf["seed"]

    if resume_paths is not None:
        state_path, snapshot_path, resume_nimg = resume_paths
        c.resume_pkl = snapshot_path
        c.resume_nimg = resume_nimg
        c.resume_state_dump = state_path

    if dist.get_rank() != 0:
        c.run_dir = None
    else:
        c.run_dir = run_dir

    dist.print0()
    dist.print0("Fun-DDPS prior training options:")
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f"Output directory:        {c.run_dir}")
    dist.print0(f"Dataset name:            {dataset_name}")
    dist.print0(f"Dataset path:            {c.dataset_kwargs.path}")
    dist.print0(f"Prior channels:          {dataset_channels}")
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
        if dist.get_rank() == 0 and wandb.run is not None:
            wandb.finish()
        return

    dist.print0("Creating output directory...")
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    training_loop.training_loop(**c)

    if dist.get_rank() == 0 and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
