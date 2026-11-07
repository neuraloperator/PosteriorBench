from __future__ import annotations

import copy
import json
import os
import pickle
import time

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime metric
    psutil = None

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional for local tests
    wandb = None

from models import dnnlib
from models.torch_utils import distributed as dist
from models.torch_utils import misc, training_stats


def _load_training_state(path):
    """Load a trusted local full-state checkpoint for optimizer resume."""
    return torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )


def _wandb_log(payload: dict[str, float], step: int) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(payload, step=step)


def training_loop(
    run_dir=".",
    dataset_kwargs={},
    data_loader_kwargs={},
    network_kwargs={},
    loss_kwargs={},
    optimizer_kwargs={},
    augment_kwargs=None,
    seed=0,
    batch_size=512,
    batch_gpu=None,
    total_kimg=20000,
    ema_halflife_kimg=50,
    ema_rampup_ratio=0.05,
    lr_rampup_kimg=10000,
    loss_scaling=1,
    kimg_per_tick=50,
    snapshot_ticks=50,
    state_dump_ticks=500,
    resume_pkl=None,
    resume_state_dump=None,
    resume_nimg=0,
    cudnn_benchmark=True,
    allow_tf32=False,
    device=torch.device("cuda"),
    cond=False,
):
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    dist.print0("Loading dataset...")
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed,
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            **data_loader_kwargs,
        )
    )
    image_shape = list(dataset_obj.image_shape)

    dist.print0("Constructing network...")
    interface_kwargs = dict(
        img_channels=dataset_obj.num_channels,
        label_dim=dataset_obj.label_dim,
    )
    if "img_resolution" not in network_kwargs:
        interface_kwargs["img_resolution"] = image_shape[-2]
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)
    dist.print0("Number of params: {}".format(misc.count_parameters(net)))
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros(
                [batch_gpu, *image_shape],
                device=device,
            )
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device) if cond else None
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    dist.print0("Setting up optimizer...")
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    augment_pipe = (
        dnnlib.util.construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data["ema"], dst_module=ema, require_all=False)
        del data
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = _load_training_state(resume_state_dump)
        misc.copy_params_and_buffers(src_module=data["net"], dst_module=net, require_all=True)
        optimizer.load_state_dict(data["optimizer_state"])
        del data

    dist.print0(f"Training for {total_kimg} kimg...")
    dist.print0()
    cur_nimg = int(resume_nimg)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32)
                labels = labels.to(device) if cond else None
                loss = loss_fn(
                    net=ddp,
                    images=images,
                    labels=labels,
                    augment_pipe=augment_pipe,
                )
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        target_lr = optimizer_kwargs["lr"] * min(
            cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
        )
        for group in optimizer.param_groups:
            group["lr"] = target_lr
        for parameter in net.parameters():
            if parameter.grad is not None:
                torch.nan_to_num(
                    parameter.grad,
                    nan=0,
                    posinf=1e5,
                    neginf=-1e5,
                    out=parameter.grad,
                )
        optimizer.step()

        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        if psutil is not None:
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        if device.type == "cuda":
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats(device)
        dist.print0(" ".join(fields))

        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0("Aborting...")

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value
            if dist.get_rank() == 0:
                path = os.path.join(run_dir, f"network-snapshot-{cur_nimg}.pkl")
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            del data

        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.get_rank() == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg}.pt"),
            )

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            log_dict = training_stats.default_collector.as_dict()
            payload = {"lr": optimizer.param_groups[0]["lr"]}
            if "Loss/loss" in log_dict:
                payload["loss"] = log_dict["Loss/loss"]["mean"]
            _wandb_log(payload, step=cur_nimg)
            stats_jsonl.write(json.dumps(dict(log_dict, timestamp=time.time())) + "\n")
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0()
    dist.print0("Exiting...")
