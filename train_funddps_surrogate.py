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

from training_funddps.dataset import FunDDPSSurrogateDataset
from training_funddps.surrogate import create_surrogate_model, relative_l2_loss
from utils.yaml_config import Config, process_arguments


def _loader_kwargs(workers: int) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def _stats_payload(dataset: FunDDPSSurrogateDataset) -> dict[str, object]:
    return {
        "dataset": dataset.name,
        "resolution": dataset.resolution,
        "input_channels": list(dataset.input_channels),
        "output_channels": list(dataset.output_channels),
        "normalization_scale": dataset.normalization_scale,
        "input": dataset.input_statistics,
        "output": dataset.output_statistics,
    }


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    history: list[dict[str, float | int]],
    model_spec: object,
    config: dict[str, object],
    normalization: dict[str, object],
) -> None:
    checkpoint = {
        "schema_version": "posteriorbench.funddps.surrogate.v1",
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "model_spec": vars(model_spec),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict(),
        "patience_counter": int(patience_counter),
        "history": history,
        "config": config,
        "normalization": normalization,
    }
    torch.save(checkpoint, path)


@torch.no_grad()
def _validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for x, y in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.float32, non_blocking=True)
        losses.append(float(relative_l2_loss(model(x), y).item()))
    return float(sum(losses) / max(len(losses), 1))


def main():
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
    train_dataset = FunDDPSSurrogateDataset(
        conf["data"],
        resolution=conf["resolution"],
        split="train",
        validation_size=conf["validation_size"],
        input_channels=conf.get("input_channels", None),
        output_channels=conf.get("output_channels", None),
        normalization_scale=conf.get("normalization_scale", 1.0),
    )
    val_dataset = FunDDPSSurrogateDataset(
        conf["data"],
        resolution=conf["resolution"],
        split="val",
        validation_size=conf["validation_size"],
        input_channels=conf.get("input_channels", None),
        output_channels=conf.get("output_channels", None),
        normalization_scale=conf.get("normalization_scale", 1.0),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(conf["batch"]),
        shuffle=True,
        drop_last=False,
        **_loader_kwargs(int(conf["workers"])),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(conf["batch"]),
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(int(conf["workers"])),
    )

    model_config = conf.get("model_config", {})
    model, model_spec = create_surrogate_model(
        conf["model"],
        in_channels=len(train_dataset.input_channels),
        out_channels=len(train_dataset.output_channels),
        resolution=train_dataset.resolution,
        config=model_config,
    )
    model = model.to(device)

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

    options = {
        **raw_conf,
        "device": str(device),
        "run_dir": str(run_dir),
        "train_size": len(train_dataset),
        "validation_size": len(val_dataset),
        "model_spec": vars(model_spec),
    }
    print(json.dumps(options, indent=2))
    if conf["dry_run"]:
        print("Dry run; exiting.")
        if wandb.run is not None:
            wandb.finish()
        return

    (run_dir / "training_options.json").write_text(json.dumps(options, indent=2))
    normalization = _stats_payload(train_dataset)
    (run_dir / "normalization_stats.json").write_text(json.dumps(normalization, indent=2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(conf["lr"]),
        weight_decay=float(conf["weight_decay"]),
    )
    scheduler_name = str(conf["scheduler"]).lower()
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(conf["lr"]),
            total_steps=int(conf["epochs"]) * max(len(train_loader), 1),
            pct_start=float(conf["warmup_fraction"]),
            div_factor=float(conf["div_factor"]),
            final_div_factor=float(conf["final_div_factor"]),
        )
        scheduler_per_batch = True
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(conf["epochs"]),
            eta_min=float(conf["min_lr"]),
        )
        scheduler_per_batch = False
    elif scheduler_name == "none":
        scheduler = None
        scheduler_per_batch = False
    else:
        raise ValueError("scheduler must be one of: onecycle, cosine, none")

    use_amp = bool(conf["fp16"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1

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
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming Fun-DDPS surrogate from {conf['resume']} at epoch {start_epoch}")

    for epoch in range(start_epoch, int(conf["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = relative_l2_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_per_batch:
                scheduler.step()
            train_losses.append(float(loss.item()))

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        should_validate = (
            epoch % int(conf["validation_frequency"]) == 0
            or epoch == int(conf["epochs"])
        )
        if should_validate:
            val_loss = _validate(model, val_loader, device)
            improved = val_loss < best_val_loss - float(conf["min_delta"])
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            record = {
                "epoch": epoch,
                "train_rel_l2": train_loss,
                "val_rel_l2": val_loss,
                "best_val_rel_l2": best_val_loss,
            }
            history.append(record)

            if improved:
                _save_checkpoint(
                    run_dir / "surrogate_best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    history=history,
                    model_spec=model_spec,
                    config=raw_conf,
                    normalization=normalization,
                )

            if epoch % int(conf["save_frequency"]) == 0:
                _save_checkpoint(
                    run_dir / f"surrogate_epoch_{epoch}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    history=history,
                    model_spec=model_spec,
                    config=raw_conf,
                    normalization=normalization,
                )
            print(
                f"epoch={epoch} train_rel_l2={train_loss:.6f} "
                f"val_rel_l2={val_loss:.6f} best={best_val_loss:.6f}"
            )
            if wandb.run is not None:
                wandb.log(record)
            if patience_counter >= int(conf["patience"]):
                print(f"Early stopping at epoch {epoch}")
                break
        else:
            print(f"epoch={epoch} train_rel_l2={train_loss:.6f}")
            if wandb.run is not None:
                wandb.log({"epoch": epoch, "train_rel_l2": train_loss})

        if scheduler is not None and not scheduler_per_batch:
            scheduler.step()

    (run_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
