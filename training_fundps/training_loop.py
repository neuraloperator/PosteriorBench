# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import numpy as np
import torch
from models import dnnlib
import wandb
from models.torch_utils import distributed as dist
from models.torch_utils import training_stats
from models.torch_utils import misc
from posteriorbench.compat import install_legacy_checkpoint_aliases

# ----------------------------------------------------------------------------


def _load_training_state(path):
    """Load a trusted local full-state checkpoint for optimizer resume."""
    install_legacy_checkpoint_aliases()
    return torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )


def training_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    sampler_kwargs={},  # Options for noise sampler.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_nimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    allow_tf32=False,  # Enable TF32 tensor cores for FP32 conv/matmul?
    device=torch.device("cuda"),
    cond=True,  # If True, use conditional diffusion.
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # subclass of training_fundps.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    image_shape = list(dataset_obj.image_shape)

    # Construct network.
    dist.print0("Constructing network...")
    interface_kwargs = dict(img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    dist.print0("Number of params: {}".format(misc.count_parameters(net)))
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, *image_shape], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device) if cond else None
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Construct loss function.
    sampler_kwargs.in_channels = dataset_obj.num_channels
    sampler_kwargs.Ln1 = image_shape[-2]
    sampler_kwargs.Ln2 = image_shape[-1]
    # Have to specify `device` here since it is not serialisable
    sampler_kwargs.device = device
    loss_kwargs.sampler = dnnlib.util.construct_class_by_name(**sampler_kwargs)
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)  # training_fundps.loss.(VP|VE|EDM)Loss

    # Setup optimizer.
    dist.print0("Setting up optimizer...")
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)  # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None  # training_fundps.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()  # rank 0 goes first
        install_legacy_checkpoint_aliases()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()  # other ranks follow
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=ema, require_all=False)
        del data  # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        # Full training states contain trusted local optimizer and persistent
        # model objects, not a weights-only state dict. PyTorch 2.6 changed the
        # torch.load default to weights_only=True, so resume must opt out.
        data = _load_training_state(resume_state_dump)
        misc.copy_params_and_buffers(src_module=data["net"], dst_module=net, require_all=True)
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = resume_nimg
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32)
                labels = labels.to(device) if cond else None
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        target_lr = optimizer_kwargs["lr"] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for g in optimizer.param_groups:
            g["lr"] = target_lr
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        dist.print0(" ".join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.get_rank() == 0:
                network_snapshot_path = os.path.join(run_dir, f"network-snapshot-{cur_nimg}.pkl")
                with open(network_snapshot_path, "wb") as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f"training-state-{cur_nimg}.pt"))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            log_dict = training_stats.default_collector.as_dict()
            wandb.log(
                {
                    "loss": log_dict["Loss/loss"]["mean"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=cur_nimg,
            )
            stats_jsonl.write(json.dumps(dict(log_dict, timestamp=time.time())) + "\n")
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0("Exiting...")


# ----------------------------------------------------------------------------
