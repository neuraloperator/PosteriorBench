from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from torch.utils.data import DataLoader

from training_fundps.dataset import PDEFieldDataset
from training_eci.flow import clean_eci_state_dict, create_eci_model
from utils.yaml_config import Config, process_arguments


def _loader_kwargs(workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def _dataset_stats_payload(dataset: PDEFieldDataset, normalization_scale: float) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "resolution": dataset.resolution,
        "channels": list(dataset.channels),
        "normalization_scale": float(normalization_scale),
        "statistics": dataset.statistics,
    }


def _images_from_batch(batch: object, device: torch.device) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    return batch.to(device=device, dtype=torch.float32, non_blocking=True)


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
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    loss: float,
    history: list[dict[str, float | int]],
    model_spec: object,
    config: dict[str, object],
    normalization: dict[str, object],
) -> None:
    torch.save(
        {
            "schema_version": "posteriorbench.eci.v1",
            "step": int(step),
            "loss": float(loss),
            "model_spec": vars(model_spec),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "config": config,
            "normalization": normalization,
        },
        path,
    )


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
    normalization_scale = float(conf.get("normalization_scale", 0.5))
    if normalization_scale != 0.5:
        raise ValueError(
            "ECI currently reuses the PDE field-source normalization and requires "
            "normalization_scale=0.5"
        )
    dataset = PDEFieldDataset(
        conf["data"],
        resolution=conf["resolution"],
        max_size=conf.get("max_size", None),
        shuffle=bool(conf.get("shuffle_subset", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(conf["batch"]),
        shuffle=True,
        drop_last=True,
        **_loader_kwargs(int(conf["workers"])),
    )
    if len(loader) == 0:
        raise ValueError("ECI training loader is empty; reduce batch or max_size")

    model_config = _as_plain_dict(conf.get("model_config", {}))
    model, model_spec = create_eci_model(
        channels=dataset.num_channels,
        resolution=dataset.resolution,
        config=model_config,
    )
    model = model.to(device).train()

    run_id, run_dir = _allocate_run(conf)
    if conf["wandb"] != "disabled" or not conf["dry_run"]:
        run_dir.mkdir(parents=True, exist_ok=False)
    if conf["wandb"] != "disabled":
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "PosteriorBench"),
            config=raw_conf,
            dir=str(run_dir),
            id=run_id,
            name=conf["name"],
            mode=conf["wandb"],
        )
        if wandb.run is not None:
            run_id = wandb.run.id
            wandb.run.log_code(root=".")

    normalization = _dataset_stats_payload(dataset, normalization_scale)
    options = {
        **raw_conf,
        "device": str(device),
        "run_dir": str(run_dir),
        "dataset": dataset.name,
        "channels": list(dataset.channels),
        "train_size": len(dataset),
        "steps_per_epoch": len(loader),
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
    use_amp = bool(conf["fp16"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    max_steps = int(conf["max_steps"])
    log_freq = int(conf["log_freq"])
    save_freq = int(conf["save_freq"])
    history: list[dict[str, float | int]] = []
    step = 0
    last_loss = float("nan")

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(clean_eci_state_dict(checkpoint["model_state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict", None)
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        history = list(checkpoint.get("history", []))
        step = int(checkpoint["step"])
        last_loss = float(checkpoint.get("loss", last_loss))
        print(f"Resuming ECI from {conf['resume']} at step {step}")

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            images = _images_from_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = model.get_loss(images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            last_loss = float(loss.detach().cpu())
            if step % log_freq == 0 or step == 1:
                record = {"step": step, "train_loss": last_loss}
                history.append(record)
                print(f"step={step} train_loss={last_loss:.6f}")
                if wandb.run is not None:
                    wandb.log(record)

            if step % save_freq == 0 or step == max_steps:
                _save_checkpoint(
                    run_dir / "eci_latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    loss=last_loss,
                    history=history,
                    model_spec=model_spec,
                    config=raw_conf,
                    normalization=normalization,
                )
                if step % int(conf["snapshot_freq"]) == 0 or step == max_steps:
                    _save_checkpoint(
                        run_dir / f"eci_step_{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        step=step,
                        loss=last_loss,
                        history=history,
                        model_spec=model_spec,
                        config=raw_conf,
                        normalization=normalization,
                    )

    (run_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
