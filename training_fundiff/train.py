from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from training_fundiff.checkpoint import (
    load_checkpoint,
    package_paths,
    save_checkpoint,
    write_json,
)
from training_fundiff.data import (
    ChannelStats,
    FundiffBatches,
    make_coord_grid,
    sample_query_batch,
)
from training_fundiff.models import Decoder, DiT, Encoder, decode_at_coords
from posteriorbench.spatial import as_spatial_shape, spatial_to_config


class TrainState(train_state.TrainState):
    pass


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else tuple(value)


def _config_dict(config: Any) -> dict[str, Any]:
    return config.to_dict() if hasattr(config, "to_dict") else dict(config)


def _encoder_from_config(config: dict[str, Any]) -> Encoder:
    kwargs = dict(config)
    kwargs["patch_size"] = _as_tuple(kwargs["patch_size"])
    kwargs["grid_size"] = _as_tuple(kwargs["grid_size"])
    return Encoder(**kwargs)


def _decoder_from_config(config: dict[str, Any]) -> Decoder:
    return Decoder(**dict(config))


def _dit_from_config(config: dict[str, Any]) -> DiT:
    return DiT(**dict(config))


def _stats_from_config(config: dict[str, Any], key: str) -> ChannelStats:
    raw = dict(config[key])
    return ChannelStats(
        channels=tuple(raw["channels"]),
        mean=tuple(float(value) for value in raw["mean"]),
        std=tuple(float(value) for value in raw["std"]),
        scale=float(raw.get("scale", 0.5)),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "fundiff"


def _create_run_dir(config: dict[str, Any], stage: str) -> Path:
    if config.get("run_dir"):
        run_dir = Path(config["run_dir"])
    else:
        name = _safe_name(str(config.get("name") or f"{config['dataset']}-fundiff-{stage}"))
        timestamp = time.strftime("%m%d_%H%M%S")
        run_dir = Path(config.get("outdir", "exps")) / f"{timestamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_metric(config: dict[str, Any], stage: str, step: int, loss: float) -> None:
    run_dir = config.get("run_dir")
    if not run_dir:
        return
    _append_jsonl(
        Path(run_dir) / "metrics.jsonl",
        {"stage": stage, "step": int(step), "loss": float(loss), "time_unix": time.time()},
    )


def _record_checkpoint_event(config: dict[str, Any], stage: str, step: int, path: Path) -> None:
    run_dir = config.get("run_dir")
    if not run_dir:
        return
    _append_jsonl(
        Path(run_dir) / "checkpoints.jsonl",
        {
            "stage": stage,
            "step": int(step),
            "checkpoint": str(path),
            "time_unix": time.time(),
        },
    )


def _batch_source(config: dict[str, Any]) -> FundiffBatches:
    return FundiffBatches(
        config["data"],
        target_channels=config["target_channels"],
        condition_channels=config["condition_channels"],
        target_stats=_stats_from_config(config, "target_normalization"),
        condition_stats=_stats_from_config(config, "condition_normalization"),
        resolution=config["resolution"],
        observation_operator=str(config["observation_operator"]),
        num_sensors=int(config.get("num_sensors", 128)),
        sensor_columns=config.get("sensor_columns", None),
        low_resolution=config.get("condition_low_resolution", None),
    )


def _optimizer(config: dict[str, Any]) -> optax.GradientTransformation:
    lr = optax.warmup_exponential_decay_schedule(
        init_value=float(config["lr"].get("init_value", 0.0)),
        peak_value=float(config["lr"]["peak_value"]),
        warmup_steps=int(config["lr"].get("warmup_steps", 2000)),
        transition_steps=int(config["lr"].get("transition_steps", 2000)),
        decay_rate=float(config["lr"].get("decay_rate", 0.9)),
    )
    return optax.chain(
        optax.clip_by_global_norm(float(config["optim"].get("clip_norm", 1.0))),
        optax.adamw(lr, weight_decay=float(config["optim"].get("weight_decay", 1e-5))),
    )


def _init_autoencoder_state(
    *,
    config: dict[str, Any],
    encoder: Encoder,
    decoder: Decoder,
    in_channels: int,
    out_channels: int,
    seed: int,
) -> TrainState:
    height, width = as_spatial_shape(config["resolution"])
    key_encoder, key_decoder = jax.random.split(jax.random.PRNGKey(seed), 2)
    x = jnp.ones((1, height, width, in_channels), dtype=jnp.float32)
    encoder_params = encoder.init(key_encoder, x)
    z = encoder.apply(encoder_params, x)
    decoder_params = decoder.init(
        key_decoder,
        z,
        jnp.ones((2,), dtype=jnp.float32),
    )
    params = {"encoder": encoder_params, "decoder": decoder_params}
    return TrainState.create(apply_fn=lambda *_: None, params=params, tx=_optimizer(config))


def _init_dit_state(
    *,
    config: dict[str, Any],
    dit: DiT,
    seed: int,
) -> TrainState:
    key = jax.random.PRNGKey(seed)
    num_latents = int(config["model"]["target_encoder"]["num_latents"])
    emb_dim = int(config["model"]["target_encoder"]["emb_dim"])
    x = jnp.ones((1, num_latents, emb_dim), dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    c = jnp.ones((1, num_latents, emb_dim), dtype=jnp.float32)
    params = dit.init(key, x, t, c)
    return TrainState.create(apply_fn=lambda *_: None, params=params, tx=_optimizer(config))


def _create_autoencoder_step(encoder: Encoder, decoder: Decoder):
    @jax.jit
    def train_step(
        state: TrainState,
        inputs: jnp.ndarray,
        coords: jnp.ndarray,
        values: jnp.ndarray,
    ) -> tuple[TrainState, jnp.ndarray]:
        def loss_fn(params: dict[str, Any]) -> jnp.ndarray:
            z = encoder.apply(params["encoder"], inputs)
            pred = decode_at_coords(decoder, params["decoder"], z, coords)
            return jnp.mean((pred - values) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return train_step


def _create_dit_step(dit: DiT, target_encoder: Encoder, condition_encoder: Encoder):
    @jax.jit
    def train_step(
        state: TrainState,
        target_encoder_params: Any,
        condition_encoder_params: Any,
        target_input: jnp.ndarray,
        condition_input: jnp.ndarray,
        rng_key: jnp.ndarray,
    ) -> tuple[TrainState, jnp.ndarray]:
        z_target = target_encoder.apply(target_encoder_params, target_input)
        z_condition = condition_encoder.apply(condition_encoder_params, condition_input)
        noise_key, time_key = jax.random.split(rng_key, 2)
        z_noise = jax.random.normal(noise_key, z_target.shape, dtype=jnp.float32)
        t = jax.random.uniform(time_key, (z_target.shape[0], 1, 1), dtype=jnp.float32)
        z_t = t * z_target + (1.0 - t) * z_noise
        velocity = z_target - z_noise

        def loss_fn(params: Any) -> jnp.ndarray:
            pred = dit.apply(params, z_t, t.reshape(-1), z_condition)
            return jnp.mean((pred - velocity) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    return train_step


def _metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "posteriorbench.fundiff.package.v1",
        "method": "fundiff",
        "dataset": config["dataset"],
        "resolution": spatial_to_config(as_spatial_shape(config["resolution"])),
        "target_channels": list(config["target_channels"]),
        "condition_channels": list(config["condition_channels"]),
        "observation_operator": config["observation_operator"],
        "num_sensors": int(config.get("num_sensors", 128)),
        "sensor_columns": config.get("sensor_columns", None),
        "condition_low_resolution": config.get("condition_low_resolution", None),
        "model": config["model"],
        "target_normalization": config["target_normalization"],
        "condition_normalization": config["condition_normalization"],
    }


def _save_metadata(config: dict[str, Any], stage: str, step: int) -> None:
    paths = package_paths(config["package_dir"])
    metadata_path = paths["metadata"]
    metadata = _metadata(config)
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text())
            if isinstance(existing, dict):
                metadata.update({key: value for key, value in existing.items() if key == "stages"})
        except json.JSONDecodeError:
            pass
    stages = dict(metadata.get("stages", {}))
    stages[stage] = {"step": int(step)}
    metadata["stages"] = stages
    write_json(metadata_path, metadata)


def _maybe_wandb(config: dict[str, Any], stage: str):
    if str(config.get("wandb", "disabled")) == "disabled":
        return None
    import wandb

    run_name = config.get("name") or f"{config['dataset']}-fundiff-{stage}"
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "PosteriorBench"),
        config=config,
        name=run_name,
        mode=str(config.get("wandb", "online")),
    )


def train_target_fae(config: dict[str, Any]) -> None:
    source = _batch_source(config)
    encoder = _encoder_from_config(config["model"]["target_encoder"])
    decoder = _decoder_from_config(config["model"]["target_decoder"])
    state = _init_autoencoder_state(
        config=config,
        encoder=encoder,
        decoder=decoder,
        in_channels=len(config["target_channels"]),
        out_channels=len(config["target_channels"]),
        seed=int(config["seed"]),
    )
    if config.get("dry_run", False):
        print(json.dumps({**config, "num_samples": source.num_samples}, indent=2))
        return

    train_step = _create_autoencoder_step(encoder, decoder)
    coords = make_coord_grid(config["resolution"])
    rng = np.random.default_rng(int(config["seed"]))
    run = _maybe_wandb(config, "target_fae")
    factors = tuple(int(x) for x in config.get("input_downsample_factors", [1]))
    for step in range(1, int(config["max_steps"]) + 1):
        indices = source.random_indices(int(config["batch"]), rng)
        target = source.read_target(indices)
        factor = int(rng.choice(factors))
        inputs = target[:, ::factor, ::factor]
        query_coords, query_values = sample_query_batch(
            target,
            coords,
            int(config["num_queries"]),
            rng,
        )
        state, loss = train_step(
            state,
            jnp.asarray(inputs),
            jnp.asarray(query_coords),
            jnp.asarray(query_values),
        )
        if step % int(config["log_interval"]) == 0:
            loss_value = float(loss.item())
            print(f"stage=target_fae step={step} loss={loss_value:.6e}")
            _record_metric(config, "target_fae", step, loss_value)
            if run is not None:
                run.log({"loss": loss_value}, step=step)
        if step % int(config["save_interval"]) == 0 or step == int(config["max_steps"]):
            checkpoint_path = package_paths(config["package_dir"])["target_fae"]
            payload = {
                "schema_version": "posteriorbench.fundiff.autoencoder.v1",
                "stage": "target_fae",
                "step": step,
                "params": state.params,
                "config": config,
            }
            save_checkpoint(checkpoint_path, payload)
            _save_metadata(config, "target_fae", step)
            _record_checkpoint_event(config, "target_fae", step, checkpoint_path)
    if run is not None:
        run.finish()


def train_condition_fae(config: dict[str, Any]) -> None:
    source = _batch_source(config)
    encoder = _encoder_from_config(config["model"]["condition_encoder"])
    decoder = _decoder_from_config(config["model"]["condition_decoder"])
    state = _init_autoencoder_state(
        config=config,
        encoder=encoder,
        decoder=decoder,
        in_channels=2,
        out_channels=len(config["condition_channels"]),
        seed=int(config["seed"]),
    )
    if config.get("dry_run", False):
        print(json.dumps({**config, "num_samples": source.num_samples}, indent=2))
        return

    train_step = _create_autoencoder_step(encoder, decoder)
    coords = make_coord_grid(config["resolution"])
    rng = np.random.default_rng(int(config["seed"]) + 1)
    run = _maybe_wandb(config, "condition_fae")
    for step in range(1, int(config["max_steps"]) + 1):
        indices = source.random_indices(int(config["batch"]), rng)
        inputs = source.read_condition(indices, rng)
        observed = source.read_observed_full(indices)
        query_coords, query_values = sample_query_batch(
            observed,
            coords,
            int(config["num_queries"]),
            rng,
        )
        state, loss = train_step(
            state,
            jnp.asarray(inputs),
            jnp.asarray(query_coords),
            jnp.asarray(query_values),
        )
        if step % int(config["log_interval"]) == 0:
            loss_value = float(loss.item())
            print(f"stage=condition_fae step={step} loss={loss_value:.6e}")
            _record_metric(config, "condition_fae", step, loss_value)
            if run is not None:
                run.log({"loss": loss_value}, step=step)
        if step % int(config["save_interval"]) == 0 or step == int(config["max_steps"]):
            checkpoint_path = package_paths(config["package_dir"])["condition_fae"]
            payload = {
                "schema_version": "posteriorbench.fundiff.autoencoder.v1",
                "stage": "condition_fae",
                "step": step,
                "params": state.params,
                "config": config,
            }
            save_checkpoint(checkpoint_path, payload)
            _save_metadata(config, "condition_fae", step)
            _record_checkpoint_event(config, "condition_fae", step, checkpoint_path)
    if run is not None:
        run.finish()


def train_dit(config: dict[str, Any]) -> None:
    source = _batch_source(config)
    if config.get("dry_run", False):
        print(json.dumps({**config, "num_samples": source.num_samples}, indent=2))
        return

    paths = package_paths(config["package_dir"])
    target_ckpt = load_checkpoint(paths["target_fae"])
    condition_ckpt = load_checkpoint(paths["condition_fae"])
    target_encoder = _encoder_from_config(config["model"]["target_encoder"])
    condition_encoder = _encoder_from_config(config["model"]["condition_encoder"])
    dit = _dit_from_config(config["model"]["dit"])
    state = _init_dit_state(config=config, dit=dit, seed=int(config["seed"]))

    target_encoder_params = target_ckpt["params"]["encoder"]
    condition_encoder_params = condition_ckpt["params"]["encoder"]
    train_step = _create_dit_step(dit, target_encoder, condition_encoder)
    rng = np.random.default_rng(int(config["seed"]) + 2)
    key = jax.random.PRNGKey(int(config["seed"]) + 3)
    run = _maybe_wandb(config, "dit")
    for step in range(1, int(config["max_steps"]) + 1):
        indices = source.random_indices(int(config["batch"]), rng)
        target = source.read_target(indices)
        condition = source.read_condition(indices, rng)
        key, subkey = jax.random.split(key)
        state, loss = train_step(
            state,
            target_encoder_params,
            condition_encoder_params,
            jnp.asarray(target),
            jnp.asarray(condition),
            subkey,
        )
        if step % int(config["log_interval"]) == 0:
            loss_value = float(loss.item())
            print(f"stage=dit step={step} loss={loss_value:.6e}")
            _record_metric(config, "dit", step, loss_value)
            if run is not None:
                run.log({"loss": loss_value}, step=step)
        if step % int(config["save_interval"]) == 0 or step == int(config["max_steps"]):
            checkpoint_path = paths["dit"]
            payload = {
                "schema_version": "posteriorbench.fundiff.dit.v1",
                "stage": "dit",
                "step": step,
                "params": state.params,
                "config": config,
            }
            save_checkpoint(checkpoint_path, payload)
            _save_metadata(config, "dit", step)
            _record_checkpoint_event(config, "dit", step, checkpoint_path)
    if run is not None:
        run.finish()


def train_from_config(config: Any) -> None:
    conf = _config_dict(config)
    stage = str(conf["stage"])
    if not bool(conf.get("dry_run", False)):
        run_dir = _create_run_dir(conf, stage)
        conf["run_dir"] = str(run_dir)
        Path(conf["package_dir"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(conf["package_dir"]) / f"training_options_{stage}.json", conf)
        write_json(run_dir / "training_options.json", conf)
        write_json(
            run_dir / "run_metadata.json",
            {
                "schema_version": "posteriorbench.fundiff.run.v1",
                "method": "fundiff",
                "dataset": conf["dataset"],
                "stage": stage,
                "package_dir": conf["package_dir"],
                "started_at_unix": time.time(),
            },
        )
    if stage == "target_fae":
        train_target_fae(conf)
    elif stage == "condition_fae":
        train_condition_fae(conf)
    elif stage == "dit":
        train_dit(conf)
    else:
        raise ValueError("FunDiff stage must be one of: target_fae, condition_fae, dit")


def sample_ode(
    dit: DiT,
    params: Any,
    z0: jnp.ndarray,
    condition: jnp.ndarray,
    num_steps: int,
) -> jnp.ndarray:
    z = z0
    dt = 1.0 / float(num_steps)
    for index in range(num_steps):
        t = jnp.ones((z.shape[0],), dtype=jnp.float32) * (index / float(num_steps))
        z = z + dit.apply(params, z, t, condition) * dt
    return z
