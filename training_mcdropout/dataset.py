from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from posteriorbench.training_source import TrainingFieldSource, open_training_field_source


TARGET_CHANNELS_BY_DATASET = {
    "ccs": ("x",),
    "poisson": ("f",),
    "darcy": ("a",),
    "light_transport": ("sigma_t1", "sigma_t2"),
}

OBSERVED_CHANNEL_BY_DATASET = {
    "ccs": "y",
    "poisson": "phi",
    "darcy": "u",
    "light_transport": "u",
}

OBSERVATION_MODE_BY_DATASET = {
    "ccs": "fixed_columns",
    "poisson": "sparse_points",
    "darcy": "sparse_points",
    "light_transport": "area_average",
}

CONDITION_CHANNELS = ("observed_value", "mask")


def _as_channels(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(channel) for channel in value)


def _resolve_default_channels(
    metadata: dict[str, object],
    requested: Iterable[str] | str | None,
) -> tuple[str, ...]:
    channels = _as_channels(requested)
    if channels is not None:
        return channels
    dataset = str(metadata["dataset"])
    if dataset not in TARGET_CHANNELS_BY_DATASET:
        raise ValueError(
            f"Dataset '{dataset}' has no default MC-dropout target channels; "
            "set target_channels explicitly"
        )
    return TARGET_CHANNELS_BY_DATASET[dataset]


def _resolve_observed_channel(
    metadata: dict[str, object],
    requested: str | None,
) -> str:
    if requested is not None:
        return str(requested)
    dataset = str(metadata["dataset"])
    if dataset not in OBSERVED_CHANNEL_BY_DATASET:
        raise ValueError(
            f"Dataset '{dataset}' has no default MC-dropout observed channel; "
            "set observed_channel explicitly"
        )
    return OBSERVED_CHANNEL_BY_DATASET[dataset]


def _validate_channels(
    source: TrainingFieldSource,
    metadata: dict[str, object],
    channels: tuple[str, ...],
) -> tuple[int, int, int]:
    if not channels:
        raise ValueError("At least one channel is required")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Channels must be unique, got {channels}")
    shapes = source.channel_shapes(channels)
    if any(len(shape) != 3 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError(f"Expected matching [N,H,W] channels, got {shapes}")
    num_samples, raw_height, raw_width = shapes[0]
    metadata_count = int(metadata["num_samples"])
    metadata_resolution = tuple(metadata["resolution"])
    if metadata_count != num_samples or metadata_resolution != (raw_height, raw_width):
        raise ValueError(
            "source/metadata shape mismatch: "
            f"source={shapes[0]}, metadata=({metadata_count}, {metadata_resolution})"
        )
    return int(num_samples), int(raw_height), int(raw_width)


def _channel_stats(
    metadata: dict[str, object],
    channels: tuple[str, ...],
) -> dict[str, list[float]]:
    statistics = metadata["statistics"]
    stats = {
        "mean": [float(statistics[channel]["mean"]) for channel in channels],
        "std": [float(statistics[channel]["std"]) for channel in channels],
    }
    if not np.all(np.isfinite(stats["mean"])):
        raise ValueError("Channel means must be finite")
    if not np.all(np.isfinite(stats["std"])) or np.any(np.asarray(stats["std"]) <= 0):
        raise ValueError("Channel standard deviations must be finite and positive")
    return stats


def _normalization_arrays(
    stats: dict[str, list[float]],
    normalization_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_scale must be finite and positive")
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    multiplier = (scale / np.asarray(stats["std"], dtype=np.float32))[:, None, None]
    return mean, multiplier


def _build_indices(
    num_samples: int,
    split: str,
    validation_size: int,
    max_size: int | None,
) -> np.ndarray:
    validation_size = int(validation_size)
    if validation_size < 0 or validation_size >= num_samples:
        raise ValueError(
            f"validation_size must be in [0, {num_samples}), got {validation_size}"
        )
    train_end = num_samples - validation_size
    split = str(split)
    if split == "train":
        indices = np.arange(0, train_end, dtype=np.int64)
    elif split in {"val", "validation"}:
        if validation_size == 0:
            raise ValueError("validation_size must be positive for validation split")
        indices = np.arange(train_end, num_samples, dtype=np.int64)
    elif split == "all":
        indices = np.arange(0, num_samples, dtype=np.int64)
    else:
        raise ValueError("split must be one of: train, val, validation, all")
    if max_size is not None:
        max_size = int(max_size)
        if max_size < 0 or max_size > len(indices):
            raise ValueError(f"Invalid max_size={max_size} for split of {len(indices)}")
        indices = indices[:max_size]
    return indices


def _stride_downsample(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = field.shape[-2:]
    target_h, target_w = target_shape
    if (height, width) == target_shape:
        return field
    if height % target_h != 0 or width % target_w != 0:
        raise ValueError(f"Cannot stride-downsample {(height, width)} to {target_shape}")
    stride_h = height // target_h
    stride_w = width // target_w
    return field[..., ::stride_h, ::stride_w]


def _area_average(field: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = field.shape[-2:]
    out_h, out_w = output_shape
    if in_h % out_h != 0 or in_w % out_w != 0:
        raise ValueError(f"Cannot area-average {(in_h, in_w)} to {output_shape}")
    block_h = in_h // out_h
    block_w = in_w // out_w
    return field.reshape(out_h, block_h, out_w, block_w).mean(axis=(1, 3))


def _nearest_upsample(field: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = field.shape[-2:]
    out_h, out_w = output_shape
    if (in_h, in_w) == output_shape:
        return field
    if out_h % in_h != 0 or out_w % in_w != 0:
        raise ValueError(f"Cannot nearest-upsample {(in_h, in_w)} to {output_shape}")
    return np.repeat(np.repeat(field, out_h // in_h, axis=-2), out_w // in_w, axis=-1)


class _SourceMixin:
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rngs"] = {}
        return state


class MCDropoutDataset(_SourceMixin, Dataset):
    """Paired observation-to-target dataset for direct FNO MC dropout."""

    def __init__(
        self,
        path,
        resolution=None,
        split="train",
        validation_size=5000,
        max_size=None,
        target_channels=None,
        observed_channel=None,
        normalization_scale=1.0,
        observation_mode=None,
        sensor_count=128,
        observation_resolution=None,
        fixed_columns=None,
        randomize_sensors=None,
        sensor_seed=0,
    ):
        self._source = open_training_field_source(path)
        self._metadata = self._source.metadata
        self._name = str(self._metadata["dataset"])
        self._target_channels = _resolve_default_channels(self._metadata, target_channels)
        self._observed_channel = _resolve_observed_channel(
            self._metadata,
            observed_channel,
        )
        all_channels = self._target_channels + (self._observed_channel,)
        num_samples, raw_height, raw_width = _validate_channels(
            self._source,
            self._metadata,
            all_channels,
        )
        raw_shape = (raw_height, raw_width)
        self._spatial_shape = raw_shape if resolution is None else as_spatial_shape(resolution)
        if raw_height % self._spatial_shape[0] != 0 or raw_width % self._spatial_shape[1] != 0:
            raise ValueError(
                f"Requested resolution {self._spatial_shape} must divide raw "
                f"resolution {raw_shape}"
            )
        self._indices = _build_indices(
            num_samples,
            split,
            int(validation_size),
            max_size,
        )
        self._normalization_scale = float(normalization_scale)
        self._target_stats = _channel_stats(self._metadata, self._target_channels)
        self._observed_stats = _channel_stats(self._metadata, (self._observed_channel,))
        self._target_mean, self._target_scale = _normalization_arrays(
            self._target_stats,
            self._normalization_scale,
        )
        self._observed_mean, self._observed_scale = _normalization_arrays(
            self._observed_stats,
            self._normalization_scale,
        )

        default_mode = OBSERVATION_MODE_BY_DATASET.get(self._name)
        self._observation_mode = str(observation_mode or default_mode)
        if self._observation_mode not in {"sparse_points", "fixed_columns", "area_average"}:
            raise ValueError(
                "observation_mode must be one of: sparse_points, fixed_columns, area_average"
            )
        self._sensor_count = int(sensor_count)
        if self._sensor_count <= 0:
            raise ValueError("sensor_count must be positive")
        if observation_resolution is None:
            self._observation_shape = (16, 16) if self._name == "light_transport" else self._spatial_shape
        else:
            self._observation_shape = as_spatial_shape(observation_resolution)
        self._fixed_columns = tuple(int(col) for col in (fixed_columns or (0, 50)))
        self._randomize_sensors = (
            split == "train" and self._observation_mode == "sparse_points"
            if randomize_sensors is None
            else bool(randomize_sensors)
        )
        self._sensor_seed = int(sensor_seed)
        self._rngs: dict[int, np.random.Generator] = {}

    def _rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = -1 if worker is None else int(worker.id)
        rng = self._rngs.get(worker_id)
        if rng is None:
            offset = 0 if worker_id < 0 else 1000003 * (worker_id + 1)
            rng = np.random.default_rng(self._sensor_seed + offset)
            self._rngs[worker_id] = rng
        return rng

    def _sparse_coords(self, sample_idx: int) -> np.ndarray:
        height, width = self._spatial_shape
        total = height * width
        if self._sensor_count > total:
            raise ValueError(
                f"sensor_count={self._sensor_count} exceeds grid size {total}"
            )
        rng = (
            self._rng()
            if self._randomize_sensors
            else np.random.default_rng(self._sensor_seed + int(sample_idx))
        )
        flat = rng.choice(total, size=self._sensor_count, replace=False)
        return np.stack([flat // width, flat % width], axis=1).astype(np.int64)

    def _condition_from_sparse(self, observed: np.ndarray, sample_idx: int) -> np.ndarray:
        coords = self._sparse_coords(sample_idx)
        value = np.zeros(self._spatial_shape, dtype=np.float32)
        mask = np.zeros(self._spatial_shape, dtype=np.float32)
        normalized = (observed - self._observed_mean[0]) * self._observed_scale[0]
        value[coords[:, 0], coords[:, 1]] = normalized[coords[:, 0], coords[:, 1]]
        mask[coords[:, 0], coords[:, 1]] = 1.0
        return np.stack([value, mask], axis=0)

    def _condition_from_columns(self, observed: np.ndarray) -> np.ndarray:
        value = np.zeros(self._spatial_shape, dtype=np.float32)
        mask = np.zeros(self._spatial_shape, dtype=np.float32)
        normalized = (observed - self._observed_mean[0]) * self._observed_scale[0]
        for column in self._fixed_columns:
            if column < 0 or column >= self._spatial_shape[1]:
                raise ValueError(
                    f"Fixed observation column {column} is outside width {self._spatial_shape[1]}"
                )
            value[:, column] = normalized[:, column]
            mask[:, column] = 1.0
        return np.stack([value, mask], axis=0)

    def _condition_from_area_average(self, observed: np.ndarray) -> np.ndarray:
        pooled = _area_average(observed, self._observation_shape)
        normalized = (pooled - self._observed_mean[0]) * self._observed_scale[0]
        upsampled = _nearest_upsample(normalized.astype(np.float32), self._spatial_shape)
        mask = np.ones(self._spatial_shape, dtype=np.float32)
        return np.stack([upsampled, mask], axis=0)

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample_idx = int(self._indices[int(idx)])
        target = self._source.read_channels([sample_idx], self._target_channels)[0]
        observed = self._source.read_channels([sample_idx], (self._observed_channel,))[0, 0]
        target = _stride_downsample(target, self._spatial_shape)
        observed = _stride_downsample(observed, self._spatial_shape)
        target = (target - self._target_mean) * self._target_scale
        if self._observation_mode == "sparse_points":
            condition = self._condition_from_sparse(observed, sample_idx)
        elif self._observation_mode == "fixed_columns":
            condition = self._condition_from_columns(observed)
        elif self._observation_mode == "area_average":
            condition = self._condition_from_area_average(observed)
        else:
            raise AssertionError("validated observation_mode became invalid")
        return (
            condition.astype(np.float32, copy=False),
            target.astype(np.float32, copy=False),
        )

    def denormalize_target_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != len(self._target_channels):
            raise ValueError(
                f"Expected [N,{len(self._target_channels)},H,W], got {tuple(x.shape)}"
            )
        mean = torch.as_tensor(self._target_stats["mean"], device=x.device, dtype=x.dtype).reshape(1, -1, 1, 1)
        std = torch.as_tensor(self._target_stats["std"], device=x.device, dtype=x.dtype).reshape(1, -1, 1, 1)
        return x * (std / self._normalization_scale) + mean

    @property
    def name(self):
        return self._name

    @property
    def target_channels(self):
        return self._target_channels

    @property
    def observed_channel(self):
        return self._observed_channel

    @property
    def condition_channels(self):
        return CONDITION_CHANNELS

    @property
    def target_statistics(self):
        return self._target_stats

    @property
    def observed_statistics(self):
        return self._observed_stats

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @property
    def observation_mode(self):
        return self._observation_mode

    @property
    def observation_shape(self):
        return spatial_to_config(self._observation_shape)

    @property
    def fixed_columns(self):
        return self._fixed_columns

    @property
    def sensor_count(self):
        return self._sensor_count

    @property
    def resolution(self):
        return spatial_to_config(self._spatial_shape)

    @property
    def spatial_shape(self):
        return self._spatial_shape
