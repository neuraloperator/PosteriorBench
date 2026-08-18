from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from training_fundiff.checkpoint import load_checkpoint, package_paths, read_json
from training_fundiff.data import (
    ChannelStats,
    area_condition_from_observation,
    denormalize_fields,
    make_coord_grid,
    sparse_condition_from_coords,
)


def validate_fundiff_channels(
    dataset_name: str,
    target_fields: tuple[str, ...],
    channels: tuple[str, ...],
) -> None:
    if channels != target_fields:
        raise ValueError(
            f"FunDiff {dataset_name} target channels must be {target_fields}, got {channels}"
        )


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else tuple(value)


def _encoder_from_config(config: dict[str, Any]):
    from training_fundiff.models import Encoder

    kwargs = dict(config)
    kwargs["patch_size"] = _as_tuple(kwargs["patch_size"])
    kwargs["grid_size"] = _as_tuple(kwargs["grid_size"])
    return Encoder(**kwargs)


def _decoder_from_config(config: dict[str, Any]):
    from training_fundiff.models import Decoder

    return Decoder(**dict(config))


def _dit_from_config(config: dict[str, Any]):
    from training_fundiff.models import DiT

    return DiT(**dict(config))


def _profile_stats(profile) -> ChannelStats:
    norm = profile.normalization
    return ChannelStats(
        channels=profile.channels,
        mean=norm.mean,
        std=norm.std,
        scale=norm.scale,
    )


def _condition_stats(guidance: dict[str, Any]) -> ChannelStats:
    raw = dict(guidance.get("condition_normalization", {}))
    if not raw:
        raise ValueError("FunDiff profile guidance must include condition_normalization")
    return ChannelStats(
        channels=tuple(raw["channels"]),
        mean=tuple(float(value) for value in raw["mean"]),
        std=tuple(float(value) for value in raw["std"]),
        scale=float(raw.get("scale", 0.5)),
    )


class FunDiffAdapter(MethodAdapter):
    def load(self) -> None:
        if self.profile.method != "fundiff":
            raise ValueError(f"FunDiff adapter cannot use method profile '{self.profile.method}'")
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match '{self.dataset.name}'"
            )
        validate_fundiff_channels(
            self.dataset.name,
            self.dataset.target_fields,
            self.profile.channels,
        )
        if bool(self.profile.guidance.get("pending", False)) and not bool(
            self.profile.guidance.get("allow_pending", False)
        ):
            raise ValueError(
                f"FunDiff profile for dataset '{self.dataset.name}' is still pending; "
                "run dry-runs only until trained checkpoints are smoke-tested"
            )

        paths = package_paths(self.checkpoint)
        if not paths["root"].is_dir():
            raise FileNotFoundError(f"FunDiff checkpoint package not found: {paths['root']}")
        self.metadata = read_json(paths["metadata"])
        if self.metadata.get("dataset") != self.dataset.name:
            raise ValueError(
                f"FunDiff package dataset '{self.metadata.get('dataset')}' does not match "
                f"'{self.dataset.name}'"
            )
        if tuple(self.metadata.get("target_channels", ())) != self.profile.channels:
            raise ValueError(
                f"FunDiff package target channels {self.metadata.get('target_channels')} "
                f"do not match profile {self.profile.channels}"
            )

        self.target_checkpoint = load_checkpoint(paths["target_fae"])
        self.condition_checkpoint = load_checkpoint(paths["condition_fae"])
        self.dit_checkpoint = load_checkpoint(paths["dit"])

        model_config = dict(self.metadata["model"])
        self.target_encoder = _encoder_from_config(model_config["target_encoder"])
        self.target_decoder = _decoder_from_config(model_config["target_decoder"])
        self.condition_encoder = _encoder_from_config(model_config["condition_encoder"])
        self.dit = _dit_from_config(model_config["dit"])

        self.target_encoder_params = self.target_checkpoint["params"]["encoder"]
        self.target_decoder_params = self.target_checkpoint["params"]["decoder"]
        self.condition_encoder_params = self.condition_checkpoint["params"]["encoder"]
        self.dit_params = self.dit_checkpoint["params"]

        self.spatial_shape = as_spatial_shape(self.profile.resolution or self.metadata["resolution"])
        self.resolution = spatial_to_config(self.spatial_shape)
        self.target_stats = _profile_stats(self.profile)
        self.condition_stats = _condition_stats(self.profile.guidance)
        self.observation_operator = str(self.profile.guidance.get("observation_operator"))
        if self.observation_operator != self.dataset.observation_operator:
            raise ValueError(
                f"Profile observation operator '{self.observation_operator}' does not match "
                f"dataset operator '{self.dataset.observation_operator}'"
            )
        self.num_steps = int(self.profile.sampling.get("num_steps", 100))
        if self.num_steps <= 0:
            raise ValueError("FunDiff sampling.num_steps must be positive")
        self.decode_chunk = int(self.profile.sampling.get("decode_chunk", 1024))
        self.postprocess = dict(self.profile.postprocess.get("discrete_values", {}))

    def _condition_for_case(self, case: BenchmarkCase) -> np.ndarray:
        observed = case.fields[case.observed_field]
        if self.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError("Sparse FunDiff condition requires sensor_coords")
            return sparse_condition_from_coords(
                observed,
                case.sensor_coords,
                self.condition_stats,
                self.spatial_shape,
            )
        if self.observation_operator == "area_average":
            return area_condition_from_observation(
                observed,
                self.condition_stats,
                self.spatial_shape,
            )
        raise ValueError(f"Unsupported FunDiff observation operator '{self.observation_operator}'")

    def _decode_grid(self, z) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from training_fundiff.models import decode_at_coords

        coords = make_coord_grid(self.spatial_shape)
        chunks = []
        for start in range(0, len(coords), self.decode_chunk):
            coord_chunk = jnp.asarray(coords[start : start + self.decode_chunk])
            chunk = decode_at_coords(
                self.target_decoder,
                self.target_decoder_params,
                z,
                coord_chunk,
            )
            chunks.append(np.asarray(jax.device_get(chunk)))
        values = np.concatenate(chunks, axis=1)
        return values.reshape(z.shape[0], self.spatial_shape[0], self.spatial_shape[1], -1)

    def _apply_postprocess(self, fields: np.ndarray) -> np.ndarray:
        result = fields.copy()
        for field, values in self.postprocess.items():
            if field not in self.profile.channels:
                raise ValueError(f"Unknown postprocess field '{field}' for {self.profile.channels}")
            if len(values) != 2:
                raise ValueError("FunDiff only supports binary discrete postprocess values")
            low, high = sorted(float(value) for value in values)
            threshold = (low + high) / 2.0
            channel = self.profile.channels.index(field)
            result[..., channel] = np.where(result[..., channel] > threshold, high, low)
        return result

    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        import jax
        import jax.numpy as jnp
        from training_fundiff.train import sample_ode

        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        condition_single = self._condition_for_case(case)
        outputs = []
        generated = 0
        key = jax.random.PRNGKey(int(seed))
        while generated < num_samples:
            current = min(batch_size, num_samples - generated)
            condition = np.repeat(condition_single, current, axis=0)
            z_condition = self.condition_encoder.apply(
                self.condition_encoder_params,
                jnp.asarray(condition, dtype=jnp.float32),
            )
            key, subkey = jax.random.split(key)
            z0 = jax.random.normal(subkey, z_condition.shape, dtype=jnp.float32)
            z = sample_ode(
                self.dit,
                self.dit_params,
                z0,
                z_condition,
                self.num_steps,
            )
            decoded = self._decode_grid(z)
            decoded = denormalize_fields(decoded, self.target_stats)
            outputs.append(self._apply_postprocess(decoded))
            generated += current

        samples = np.concatenate(outputs, axis=0)[:num_samples]
        return {
            field: samples[:, :, :, index].astype(np.float32)
            for index, field in enumerate(self.profile.channels)
        }
