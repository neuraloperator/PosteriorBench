from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from posteriorbench.training_source import open_training_field_source


@dataclass(frozen=True)
class ChannelStats:
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: float = 0.5

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.mean) or len(self.channels) != len(self.std):
            raise ValueError("ChannelStats requires one mean/std per channel")
        if any(value <= 0 for value in self.std):
            raise ValueError("ChannelStats standard deviations must be positive")

    def index(self, channel: str) -> int:
        if channel not in self.channels:
            raise ValueError(f"Unknown channel '{channel}' for stats {self.channels}")
        return self.channels.index(channel)

    def select(self, channels: Sequence[str]) -> "ChannelStats":
        indices = [self.index(channel) for channel in channels]
        return ChannelStats(
            channels=tuple(channels),
            mean=tuple(self.mean[index] for index in indices),
            std=tuple(self.std[index] for index in indices),
            scale=self.scale,
        )


def normalize_fields(x: np.ndarray, stats: ChannelStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(1, 1, 1, -1)
    return (x.astype(np.float32) - mean) * (float(stats.scale) / std)


def denormalize_fields(x: np.ndarray, stats: ChannelStats) -> np.ndarray:
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(1, 1, 1, -1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(1, 1, 1, -1)
    return x.astype(np.float32) * (std / float(stats.scale)) + mean


def make_coord_grid(resolution: object) -> np.ndarray:
    height, width = as_spatial_shape(resolution)
    coords_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    coords_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(coords_y, coords_x, indexing="ij")
    return np.stack([yy.reshape(-1), xx.reshape(-1)], axis=-1)


def sample_query_batch(
    fields: np.ndarray,
    coords: np.ndarray,
    num_queries: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if fields.ndim != 4:
        raise ValueError(f"Expected fields [B,H,W,C], got {fields.shape}")
    total = fields.shape[1] * fields.shape[2]
    if num_queries > total:
        raise ValueError(f"num_queries={num_queries} exceeds grid size {total}")
    query_index = rng.choice(total, size=num_queries, replace=False)
    values = fields.reshape(fields.shape[0], total, fields.shape[-1])[:, query_index]
    return coords[query_index], values.astype(np.float32)


def _resize_square_nearest(x: np.ndarray, resolution: object) -> np.ndarray:
    target_h, target_w = as_spatial_shape(resolution)
    if x.shape[1] == target_h and x.shape[2] == target_w:
        return x
    if x.shape[1] > target_h or x.shape[2] > target_w:
        step_y = x.shape[1] // target_h
        step_x = x.shape[2] // target_w
        if x.shape[1] != target_h * step_y or x.shape[2] != target_w * step_x:
            raise ValueError(f"Cannot downsample shape {x.shape[1:3]} to {(target_h, target_w)}")
        return x[:, ::step_y, ::step_x]
    repeat_y = target_h // x.shape[1]
    repeat_x = target_w // x.shape[2]
    if x.shape[1] * repeat_y != target_h or x.shape[2] * repeat_x != target_w:
        raise ValueError(f"Cannot upsample shape {x.shape[1:3]} to {(target_h, target_w)}")
    return np.repeat(np.repeat(x, repeat_y, axis=1), repeat_x, axis=2)


def random_sparse_condition(
    observed_normalized: np.ndarray,
    num_sensors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if observed_normalized.ndim != 4 or observed_normalized.shape[-1] != 1:
        raise ValueError("Sparse condition expects observed fields [B,H,W,1]")
    batch, height, width, _ = observed_normalized.shape
    total = height * width
    if num_sensors <= 0 or num_sensors > total:
        raise ValueError(f"Invalid num_sensors={num_sensors} for grid {height}x{width}")
    flat_mask = np.zeros((batch, total, 1), dtype=np.float32)
    for index in range(batch):
        sensor_index = rng.choice(total, size=num_sensors, replace=False)
        flat_mask[index, sensor_index, 0] = 1.0
    mask = flat_mask.reshape(batch, height, width, 1)
    value = observed_normalized * mask
    return np.concatenate([value, mask], axis=-1).astype(np.float32)


def column_sparse_condition(
    observed_normalized: np.ndarray,
    columns: Sequence[int],
) -> np.ndarray:
    if observed_normalized.ndim != 4 or observed_normalized.shape[-1] != 1:
        raise ValueError("Column sparse condition expects observed fields [B,H,W,1]")
    _, _, width, _ = observed_normalized.shape
    column_array = np.asarray(columns, dtype=np.int64)
    if column_array.ndim != 1 or column_array.size == 0:
        raise ValueError("sensor_columns must be a non-empty 1D sequence")
    if np.any(column_array < 0) or np.any(column_array >= width):
        raise ValueError(f"sensor_columns {column_array.tolist()} out of bounds for width {width}")
    if len(np.unique(column_array)) != len(column_array):
        raise ValueError(f"sensor_columns must be unique, got {column_array.tolist()}")
    mask = np.zeros_like(observed_normalized, dtype=np.float32)
    mask[:, :, column_array, 0] = 1.0
    value = observed_normalized * mask
    return np.concatenate([value, mask], axis=-1).astype(np.float32)


def sparse_condition_from_coords(
    observed: np.ndarray,
    sensor_coords: np.ndarray,
    stats: ChannelStats,
    resolution: object,
) -> np.ndarray:
    if observed.ndim != 2:
        raise ValueError(f"Observed field must have shape [H,W], got {observed.shape}")
    if len(stats.channels) != 1:
        raise ValueError("Sparse condition currently supports one observed channel")
    coords = np.asarray(sensor_coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"sensor_coords must have shape [N,2], got {coords.shape}")
    source_h, source_w = observed.shape
    target_coords = coords.copy()
    target_h, target_w = as_spatial_shape(resolution)
    if (source_h, source_w) != (target_h, target_w):
        scale_y = target_h / source_h
        scale_x = target_w / source_w
        target_coords[:, 0] = np.clip(np.floor(coords[:, 0] * scale_y), 0, target_h - 1)
        target_coords[:, 1] = np.clip(np.floor(coords[:, 1] * scale_x), 0, target_w - 1)

    condition = np.zeros((target_h, target_w, 2), dtype=np.float32)
    mean = float(stats.mean[0])
    std = float(stats.std[0])
    scale = float(stats.scale)
    values = (observed[coords[:, 0], coords[:, 1]].astype(np.float32) - mean) * (scale / std)
    condition[target_coords[:, 0], target_coords[:, 1], 0] = values
    condition[target_coords[:, 0], target_coords[:, 1], 1] = 1.0
    return condition[None]


def area_average_repeat(x: np.ndarray, low_resolution: object) -> np.ndarray:
    if x.ndim != 4 or x.shape[-1] != 1:
        raise ValueError(f"Expected [B,H,W,1], got {x.shape}")
    batch, height, width, channels = x.shape
    low_h, low_w = as_spatial_shape(low_resolution)
    if height % low_h != 0 or width % low_w != 0:
        raise ValueError(f"Resolution {(height, width)} is not divisible by {(low_h, low_w)}")
    block_h = height // low_h
    block_w = width // low_w
    low = x.reshape(batch, low_h, block_h, low_w, block_w, channels).mean(
        axis=(2, 4)
    )
    return np.repeat(np.repeat(low, block_h, axis=1), block_w, axis=2).astype(np.float32)


def area_condition_from_observation(
    observed_low: np.ndarray,
    stats: ChannelStats,
    resolution: object,
) -> np.ndarray:
    if observed_low.ndim != 2:
        raise ValueError(f"Low-resolution observation must have shape [H,W], got {observed_low.shape}")
    if len(stats.channels) != 1:
        raise ValueError("Area condition currently supports one observed channel")
    low = observed_low[None, :, :, None].astype(np.float32)
    low = normalize_fields(low, stats)
    value = _resize_square_nearest(low, resolution)
    mask = np.ones_like(value, dtype=np.float32)
    return np.concatenate([value, mask], axis=-1).astype(np.float32)


class FundiffBatches:
    def __init__(
        self,
        path: str | Path,
        *,
        target_channels: Sequence[str],
        condition_channels: Sequence[str],
        target_stats: ChannelStats,
        condition_stats: ChannelStats,
        resolution: int,
        observation_operator: str,
        num_sensors: int = 128,
        sensor_columns: Sequence[int] | None = None,
        low_resolution: int | None = None,
    ):
        self.path = Path(path)
        self.source = open_training_field_source(path)
        self.target_channels = tuple(target_channels)
        self.condition_channels = tuple(condition_channels)
        self.target_stats = target_stats.select(self.target_channels)
        self.condition_stats = condition_stats.select(self.condition_channels)
        self.spatial_shape = as_spatial_shape(resolution)
        self.resolution = spatial_to_config(self.spatial_shape)
        self.observation_operator = str(observation_operator)
        self.num_sensors = int(num_sensors)
        self.sensor_columns = None if sensor_columns is None else tuple(int(x) for x in sensor_columns)
        self.low_resolution = low_resolution
        self.num_samples = int(self.source.num_samples)

    def _read_channels(self, indices: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        fields = self.source.read_channels(indices, tuple(channels))
        return np.moveaxis(fields, 1, -1).astype(np.float32, copy=False)

    def random_indices(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(self.num_samples, size=batch_size, replace=False)
        return np.sort(indices)

    def read_target(self, indices: np.ndarray) -> np.ndarray:
        fields = self._read_channels(indices, self.target_channels)
        fields = _resize_square_nearest(fields, self.spatial_shape)
        return normalize_fields(fields, self.target_stats)

    def read_condition(self, indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        observed = self._read_channels(indices, self.condition_channels)
        observed = _resize_square_nearest(observed, self.spatial_shape)
        observed = normalize_fields(observed, self.condition_stats)
        if self.observation_operator == "sparse_points":
            if self.sensor_columns is not None:
                return column_sparse_condition(observed, self.sensor_columns)
            return random_sparse_condition(observed, self.num_sensors, rng)
        if self.observation_operator == "area_average":
            low_resolution = self.low_resolution
            if low_resolution is None:
                raise ValueError("area_average condition requires low_resolution")
            value = area_average_repeat(observed, low_resolution)
            mask = np.ones_like(value, dtype=np.float32)
            return np.concatenate([value, mask], axis=-1).astype(np.float32)
        raise ValueError(f"Unsupported observation_operator '{self.observation_operator}'")

    def read_observed_full(self, indices: np.ndarray) -> np.ndarray:
        observed = self._read_channels(indices, self.condition_channels)
        observed = _resize_square_nearest(observed, self.spatial_shape)
        return normalize_fields(observed, self.condition_stats)
