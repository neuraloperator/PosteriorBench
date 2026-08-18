from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader

from training_mcdropout.dataset import MCDropoutDataset
from training_mcdropout.model import (
    create_mcdropout_model,
    normalized_mse_loss,
    relative_l2_loss,
    set_mc_dropout_mode,
)
from utils.yaml_config import Config, process_arguments


def _loader_kwargs(workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def _as_plain_dict(value: object) -> dict[str, object]:
    if isinstance(value, Config):
        return value.to_dict()
    return dict(value)


def _allocate_run(conf: Config) -> tuple[str, Path]:
    if conf["wandb"] == "disabled":
        run_id = "local"
    else:
        run_id = os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
    formatted_time = datetime.now().strftime("%m%d_%H%M%S")
    return run_id, Path(conf["outdir"]) / f"{formatted_time}-{conf['name']}-{run_id}"


def _stats_payload(dataset: MCDropoutDataset) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "resolution": dataset.resolution,
        "target_channels": list(dataset.target_channels),
        "condition_channels": list(dataset.condition_channels),
        "observed_channel": dataset.observed_channel,
        "normalization_scale": dataset.normalization_scale,
        "target": dataset.target_statistics,
        "observed": dataset.observed_statistics,
        "observation": {
            "mode": dataset.observation_mode,
            "sensor_count": dataset.sensor_count,
            "observation_resolution": dataset.observation_shape,
            "fixed_columns": list(dataset.fixed_columns),
        },
    }


def _load_checkpoint(path: os.PathLike[str] | str) -> dict[str, Any]:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"--resume must point to an existing checkpoint: {path}")
    checkpoint = torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a state dictionary: {path}")
    return checkpoint


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    images_seen: int,
    train_loss: float,
    val_metrics: dict[str, float] | None,
    best_val_metric: float,
    history: list[dict[str, float | int]],
    model_spec: object,
    config: dict[str, object],
    normalization: dict[str, object],
    saved_snapshots: list[int],
) -> None:
    torch.save(
        {
            "schema_version": "posteriorbench.mcdropout.v1",
            "step": int(step),
            "images_seen": int(images_seen),
            "train_loss": float(train_loss),
            "val_metrics": val_metrics,
            "best_val_metric": float(best_val_metric),
            "model_spec": vars(model_spec),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "config": config,
            "normalization": normalization,
            "saved_snapshots": [int(value) for value in saved_snapshots],
        },
        path,
    )


def _apply_discrete_postprocess(
    x: torch.Tensor,
    channels: tuple[str, ...],
    discrete_values: dict[str, object],
) -> torch.Tensor:
    if not discrete_values:
        return x
    result = x.clone()
    for field, values in discrete_values.items():
        if field not in channels:
            raise ValueError(f"Unknown discrete field '{field}' for channels {channels}")
        if len(values) != 2:
            raise ValueError("Only binary discrete postprocessing is currently supported")
        low, high = sorted(float(value) for value in values)
        threshold = (low + high) / 2.0
        channel = channels.index(field)
        result[:, channel] = torch.where(
            result[:, channel] > threshold,
            torch.as_tensor(high, device=x.device, dtype=x.dtype),
            torch.as_tensor(low, device=x.device, dtype=x.dtype),
        )
    return result


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [int(device.index)]
    return [torch.cuda.current_device()]


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: MCDropoutDataset,
    device: torch.device,
    *,
    mc_samples: int,
    seed: int,
    discrete_values: dict[str, object],
) -> dict[str, float]:
    if mc_samples <= 0:
        raise ValueError("validation_mc_samples must be positive")
    channels = tuple(dataset.target_channels)
    rel_l2_values: list[float] = []
    mse_values: list[float] = []
    sigma_t_rel_l2_values: list[float] = []

    with torch.random.fork_rng(devices=_fork_rng_devices(device), enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        set_mc_dropout_mode(model)
        for condition, target in loader:
            condition = condition.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction_sum = torch.zeros_like(target)
            for _ in range(mc_samples):
                prediction_sum += model(condition)
            prediction_mean = prediction_sum / float(mc_samples)
            mse_values.append(float(normalized_mse_loss(prediction_mean, target).item()))
            prediction_physical = dataset.denormalize_target_tensor(prediction_mean)
            target_physical = dataset.denormalize_target_tensor(target)
            prediction_physical = _apply_discrete_postprocess(
                prediction_physical,
                channels,
                discrete_values,
            )
            rel_l2_values.extend(
                float(value)
                for value in relative_l2_loss(
                    prediction_physical,
                    target_physical,
                    reduction="none",
                ).detach().cpu()
            )
            if dataset.name == "light_transport" and len(channels) == 2:
                prediction_sigma_t = 0.5 * (
                    prediction_physical[:, 0:1] + prediction_physical[:, 1:2]
                )
                target_sigma_t = 0.5 * (
                    target_physical[:, 0:1] + target_physical[:, 1:2]
                )
                sigma_t_rel_l2_values.extend(
                    float(value)
                    for value in relative_l2_loss(
                        prediction_sigma_t,
                        target_sigma_t,
                        reduction="none",
                    ).detach().cpu()
                )

    metrics = {
        "val_mean_rel_l2": float(sum(rel_l2_values) / max(len(rel_l2_values), 1)),
        "val_normalized_mse": float(sum(mse_values) / max(len(mse_values), 1)),
    }
    if sigma_t_rel_l2_values:
        metrics["val_sigma_t_mean_rel_l2"] = float(
            sum(sigma_t_rel_l2_values) / len(sigma_t_rel_l2_values)
        )
    return metrics


def _build_scheduler(
    conf: Config,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, bool]:
    scheduler_name = str(conf["scheduler"]).lower()
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(conf["lr"]),
            total_steps=max_steps,
            pct_start=float(conf["warmup_fraction"]),
            div_factor=float(conf["div_factor"]),
            final_div_factor=float(conf["final_div_factor"]),
        )
        return scheduler, True
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=float(conf["min_lr"]),
        )
        return scheduler, True
    if scheduler_name == "none":
        return None, False
    raise ValueError("scheduler must be one of: onecycle, cosine, none")


def main() -> None:
    args = process_arguments()
    conf = Config(args)
    raw_conf = conf.to_dict()
    resume_checkpoint = (
        _load_checkpoint(conf["resume"])
        if conf.get("resume", None)
        else None
    )

    seed = int(conf["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    allow_tf32 = bool(conf.get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    device = torch.device(conf.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    common_dataset_kwargs = {
        "path": conf["data"],
        "resolution": conf["resolution"],
        "validation_size": conf["validation_size"],
        "target_channels": conf.get("target_channels", None),
        "observed_channel": conf.get("observed_channel", None),
        "normalization_scale": conf.get("normalization_scale", 1.0),
        "observation_mode": conf.get("observation_mode", None),
        "sensor_count": conf.get("sensor_count", 128),
        "observation_resolution": conf.get("observation_resolution", None),
        "fixed_columns": conf.get("fixed_columns", None),
        "sensor_seed": conf.get("sensor_seed", seed + 17),
    }
    train_dataset = MCDropoutDataset(
        **common_dataset_kwargs,
        split="train",
        randomize_sensors=True,
    )
    val_dataset = MCDropoutDataset(
        **common_dataset_kwargs,
        split="val",
        max_size=conf.get("validation_eval_size", None),
        randomize_sensors=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(conf["batch"]),
        shuffle=True,
        drop_last=True,
        **_loader_kwargs(int(conf["workers"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(conf["batch"]),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(int(conf["workers"])),
    )
    if len(train_loader) == 0:
        raise ValueError("MC-dropout training loader is empty; reduce batch or max_size")

    model_config = _as_plain_dict(conf.get("model_config", {}))
    model, model_spec = create_mcdropout_model(
        conf["model"],
        in_channels=len(train_dataset.condition_channels),
        out_channels=len(train_dataset.target_channels),
        resolution=train_dataset.resolution,
        config=model_config,
    )
    model = model.to(device)

    run_id, run_dir = _allocate_run(conf)
    if conf["wandb"] != "disabled" or not conf["dry_run"]:
        run_dir.mkdir(parents=True, exist_ok=False)
    if conf["wandb"] != "disabled":
        wandb.init(
            project=str(conf.get("wandb_project", "PosteriorBench-MCDropout")),
            config=raw_conf,
            dir=str(run_dir),
            id=run_id,
            name=conf["name"],
            mode=conf["wandb"],
        )
        if wandb.run is not None:
            run_id = wandb.run.id
            wandb.run.log_code(root=".")

    max_images = int(conf["max_images"])
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    batch_size = int(conf["batch"])
    max_steps = int(math.ceil(max_images / batch_size))
    normalization = _stats_payload(train_dataset)
    options = {
        **raw_conf,
        "device": str(device),
        "run_dir": str(run_dir),
        "dataset": train_dataset.name,
        "train_size": len(train_dataset),
        "heldout_validation_size": int(conf["validation_size"]),
        "validation_eval_size": len(val_dataset),
        "steps_per_epoch": len(train_loader),
        "max_steps": max_steps,
        "model_spec": vars(model_spec),
    }
    print(json.dumps(options, indent=2))
    if conf["dry_run"]:
        print("Dry run; exiting.")
        if wandb.run is not None:
            wandb.finish()
        return

    (run_dir / "training_options.json").write_text(json.dumps(options, indent=2))
    (run_dir / "normalization_stats.json").write_text(json.dumps(normalization, indent=2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(conf["lr"]),
        weight_decay=float(conf["weight_decay"]),
        betas=(float(conf["beta1"]), float(conf["beta2"])),
    )
    scheduler, scheduler_per_batch = _build_scheduler(conf, optimizer, max_steps)
    use_amp = bool(conf["fp16"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: list[dict[str, float | int]] = []
    step = 0
    images_seen = 0
    last_loss = float("nan")
    best_val_metric = float("inf")
    saved_snapshots: set[int] = set()

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("scheduler_state_dict", None)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict", None)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        history = list(checkpoint.get("history", []))
        step = int(checkpoint["step"])
        images_seen = int(checkpoint["images_seen"])
        last_loss = float(checkpoint.get("train_loss", last_loss))
        best_val_metric = float(checkpoint.get("best_val_metric", best_val_metric))
        saved_snapshots = {int(value) for value in checkpoint.get("saved_snapshots", [])}
        print(
            f"Resuming MC-dropout FNO from {conf['resume']} at "
            f"step={step} images_seen={images_seen}"
        )

    validation_every_images = int(conf["validation_every_images"])
    log_every_images = int(conf["log_every_images"])
    if validation_every_images <= 0 or log_every_images <= 0:
        raise ValueError("validation_every_images and log_every_images must be positive")
    next_validation = ((images_seen // validation_every_images) + 1) * validation_every_images
    next_log = ((images_seen // log_every_images) + 1) * log_every_images
    snapshot_images = sorted(int(value) for value in conf.get("snapshot_images", []))
    discrete_values = _as_plain_dict(conf.get("postprocess.discrete_values", {}))
    val_metrics: dict[str, float] | None = None

    epoch = 0
    while images_seen < max_images:
        epoch += 1
        model.train()
        for condition, target in train_loader:
            if images_seen >= max_images:
                break
            condition = condition.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(condition)
                loss = normalized_mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_per_batch:
                scheduler.step()

            step += 1
            images_seen += int(condition.shape[0])
            last_loss = float(loss.detach().cpu())

            if images_seen >= next_log or step == 1:
                record = {
                    "step": step,
                    "images_seen": images_seen,
                    "epoch": epoch,
                    "train_normalized_mse": last_loss,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(record)
                print(
                    f"step={step} images_seen={images_seen} "
                    f"train_normalized_mse={last_loss:.6e}"
                )
                if wandb.run is not None:
                    wandb.log(record, step=step)
                while images_seen >= next_log:
                    next_log += log_every_images

            should_validate = images_seen >= next_validation or images_seen >= max_images
            if should_validate:
                val_metrics = _validate(
                    model,
                    val_loader,
                    val_dataset,
                    device,
                    mc_samples=int(conf["validation_mc_samples"]),
                    seed=int(conf["validation_seed"]),
                    discrete_values=discrete_values,
                )
                val_metric = float(val_metrics["val_mean_rel_l2"])
                improved = val_metric < best_val_metric - float(conf["min_delta"])
                if improved:
                    best_val_metric = val_metric
                    _save_checkpoint(
                        run_dir / "mcdropout_best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=step,
                        images_seen=images_seen,
                        train_loss=last_loss,
                        val_metrics=val_metrics,
                        best_val_metric=best_val_metric,
                        history=history,
                        model_spec=model_spec,
                        config=raw_conf,
                        normalization=normalization,
                        saved_snapshots=sorted(saved_snapshots),
                    )
                record = {
                    "step": step,
                    "images_seen": images_seen,
                    "epoch": epoch,
                    "best_val_mean_rel_l2": best_val_metric,
                    **val_metrics,
                }
                history.append(record)
                print(
                    f"validation images_seen={images_seen} "
                    f"val_mean_rel_l2={val_metric:.6e} best={best_val_metric:.6e}"
                )
                if wandb.run is not None:
                    wandb.log(record, step=step)
                _save_checkpoint(
                    run_dir / "mcdropout_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    images_seen=images_seen,
                    train_loss=last_loss,
                    val_metrics=val_metrics,
                    best_val_metric=best_val_metric,
                    history=history,
                    model_spec=model_spec,
                    config=raw_conf,
                    normalization=normalization,
                    saved_snapshots=sorted(saved_snapshots),
                )
                while images_seen >= next_validation:
                    next_validation += validation_every_images
                model.train()

            for snapshot in snapshot_images:
                if images_seen >= snapshot and snapshot not in saved_snapshots:
                    _save_checkpoint(
                        run_dir / f"mcdropout_snapshot_{snapshot}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=step,
                        images_seen=images_seen,
                        train_loss=last_loss,
                        val_metrics=val_metrics,
                        best_val_metric=best_val_metric,
                        history=history,
                        model_spec=model_spec,
                        config=raw_conf,
                        normalization=normalization,
                        saved_snapshots=sorted(saved_snapshots | {snapshot}),
                    )
                    saved_snapshots.add(snapshot)

    (run_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
