from __future__ import annotations

from typing import Iterable

import numpy as np
from torch.utils.data import Dataset

from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from posteriorbench.training_source import TrainingFieldSource, open_training_field_source


TARGET_CHANNELS_BY_DATASET = {
    "ccs": ("x",),
    "poisson": ("f",),
    "darcy": ("a",),
    "light_transport": ("sigma_t1", "sigma_t2"),
}

OBSERVED_CHANNELS_BY_DATASET = {
    "ccs": ("y",),
    "poisson": ("phi",),
    "darcy": ("u",),
    "light_transport": ("u",),
}


def _as_channels(value: Iterable[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(channel) for channel in value)


def _resolve_default_channels(
    metadata: dict[str, object],
    requested: Iterable[str] | str | None,
    defaults: dict[str, tuple[str, ...]],
    role: str,
) -> tuple[str, ...]:
    channels = _as_channels(requested)
    if channels is not None:
        return channels
    dataset = str(metadata["dataset"])
    if dataset not in defaults:
        raise ValueError(
            f"Dataset '{dataset}' has no default Fun-DDPS {role} channels; "
            "set them explicitly in the config"
        )
    return defaults[dataset]


def _validate_channels(
    source: TrainingFieldSource,
    metadata: dict[str, object],
    channels: tuple[str, ...],
) -> tuple[list[tuple[int, ...]], int, int, int]:
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
    return shapes, int(num_samples), int(raw_height), int(raw_width)


def _normalization_arrays(
    metadata: dict[str, object],
    channels: tuple[str, ...],
    normalization_scale: float,
) -> tuple[dict[str, list[float]], np.ndarray, np.ndarray]:
    statistics = metadata["statistics"]
    stats = {
        "mean": [float(statistics[channel]["mean"]) for channel in channels],
        "std": [float(statistics[channel]["std"]) for channel in channels],
    }
    if not np.all(np.isfinite(stats["mean"])):
        raise ValueError("Channel means must be finite")
    if not np.all(np.isfinite(stats["std"])) or np.any(np.asarray(stats["std"]) <= 0):
        raise ValueError("Channel standard deviations must be finite and positive")
    scale = float(normalization_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("normalization_scale must be finite and positive")
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    multiplier = (scale / np.asarray(stats["std"], dtype=np.float32))[:, None, None]
    return stats, mean, multiplier


def _build_indices(
    num_samples: int,
    offset: int | None,
    max_size: int | None,
    shuffle: bool,
) -> np.ndarray:
    start = 0 if offset is None else int(offset)
    if start < 0 or start > num_samples:
        raise ValueError(f"Invalid offset {start} for {num_samples} samples")
    indices = np.arange(start, num_samples, dtype=np.int64)
    if max_size is not None:
        max_size = int(max_size)
        if max_size < 0:
            raise ValueError("max_size must be nonnegative or None")
        if max_size > len(indices):
            raise ValueError(
                f"max_size={max_size} exceeds {len(indices)} samples after offset"
            )
        if max_size < len(indices):
            if shuffle:
                indices = np.random.choice(indices, max_size, replace=False)
            else:
                indices = indices[:max_size]
    elif shuffle:
        print("Warning: shuffle=True has no effect when max_size=None")
    return indices


class _SourceMixin:
    def __getstate__(self):
        return self.__dict__.copy()


class FunDDPSPriorDataset(_SourceMixin, Dataset):
    """Target-only field-source view for Fun-DDPS prior diffusion training."""

    def __init__(
        self,
        path,
        offset=None,
        resolution=None,
        max_size=None,
        shuffle=False,
        use_labels=False,
        xflip=False,
        cache=False,
        target_channels=None,
        normalization_scale=0.5,
    ):
        if use_labels:
            raise ValueError("Fun-DDPS prior training sources do not provide labels")
        if xflip:
            raise ValueError("Horizontal flips are not supported for PDE field data")
        if cache:
            print("Warning: FunDDPSPriorDataset uses lazy Hugging Face dataset reads; ignoring cache=True")

        self._source = open_training_field_source(path)
        self._metadata = self._source.metadata
        self._name = str(self._metadata["dataset"])
        self._channels = _resolve_default_channels(
            self._metadata,
            target_channels,
            TARGET_CHANNELS_BY_DATASET,
            "target",
        )
        _, num_samples, raw_height, raw_width = _validate_channels(
            self._source,
            self._metadata,
            self._channels,
        )

        raw_shape = (raw_height, raw_width)
        self._spatial_shape = raw_shape if resolution is None else as_spatial_shape(resolution)
        if (
            raw_height % self._spatial_shape[0] != 0
            or raw_width % self._spatial_shape[1] != 0
        ):
            raise ValueError(
                f"Requested resolution {self._spatial_shape} must divide raw "
                f"resolution {raw_shape}"
            )
        self._downsample = (
            raw_height // self._spatial_shape[0],
            raw_width // self._spatial_shape[1],
        )
        self._indices = _build_indices(num_samples, offset, max_size, shuffle)

        self._normalization_scale = float(normalization_scale)
        self._stats, self._mean, self._scale = _normalization_arrays(
            self._metadata,
            self._channels,
            self._normalization_scale,
        )
        self._raw_shape = [
            len(self._indices),
            len(self._channels),
            self._spatial_shape[0],
            self._spatial_shape[1],
        ]

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample_idx = int(self._indices[int(idx)])
        image = self._source.read_channels([sample_idx], self._channels)[0]
        if self._downsample != (1, 1):
            image = image[:, :: self._downsample[0], :: self._downsample[1]]
        image = (image - self._mean) * self._scale
        return image.astype(np.float32, copy=False), np.zeros(0, dtype=np.float32)

    @property
    def name(self):
        return self._name

    @property
    def channels(self):
        return self._channels

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @property
    def statistics(self):
        return self._stats

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        return len(self._channels)

    @property
    def resolution(self):
        return spatial_to_config(self._spatial_shape)

    @property
    def spatial_shape(self):
        return self._spatial_shape

    @property
    def label_shape(self):
        return [0]

    @property
    def label_dim(self):
        return 0

    @property
    def has_labels(self):
        return False

    @property
    def has_onehot_labels(self):
        return False


class FunDDPSSurrogateDataset(_SourceMixin, Dataset):
    """Paired target-to-observed-field source view for Fun-DDPS surrogate training."""

    def __init__(
        self,
        path,
        resolution=None,
        split="train",
        validation_size=5000,
        max_size=None,
        input_channels=None,
        output_channels=None,
        normalization_scale=1.0,
    ):
        self._source = open_training_field_source(path)
        self._metadata = self._source.metadata
        self._name = str(self._metadata["dataset"])
        self._input_channels = _resolve_default_channels(
            self._metadata,
            input_channels,
            TARGET_CHANNELS_BY_DATASET,
            "input",
        )
        self._output_channels = _resolve_default_channels(
            self._metadata,
            output_channels,
            OBSERVED_CHANNELS_BY_DATASET,
            "output",
        )
        all_channels = self._input_channels + self._output_channels
        _, num_samples, raw_height, raw_width = _validate_channels(
            self._source,
            self._metadata,
            all_channels,
        )

        raw_shape = (raw_height, raw_width)
        self._spatial_shape = raw_shape if resolution is None else as_spatial_shape(resolution)
        if (
            raw_height % self._spatial_shape[0] != 0
            or raw_width % self._spatial_shape[1] != 0
        ):
            raise ValueError(
                f"Requested resolution {self._spatial_shape} must divide raw "
                f"resolution {raw_shape}"
            )
        self._downsample = (
            raw_height // self._spatial_shape[0],
            raw_width // self._spatial_shape[1],
        )

        split = str(split)
        validation_size = int(validation_size)
        if validation_size < 0 or validation_size >= num_samples:
            raise ValueError(
                f"validation_size must be in [0, {num_samples}), got {validation_size}"
            )
        train_end = num_samples - validation_size
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
        self._indices = indices

        self._normalization_scale = float(normalization_scale)
        self._input_stats, self._input_mean, self._input_scale = _normalization_arrays(
            self._metadata,
            self._input_channels,
            self._normalization_scale,
        )
        self._output_stats, self._output_mean, self._output_scale = _normalization_arrays(
            self._metadata,
            self._output_channels,
            self._normalization_scale,
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample_idx = int(self._indices[int(idx)])
        x = self._source.read_channels([sample_idx], self._input_channels)[0]
        y = self._source.read_channels([sample_idx], self._output_channels)[0]
        if self._downsample != (1, 1):
            x = x[:, :: self._downsample[0], :: self._downsample[1]]
            y = y[:, :: self._downsample[0], :: self._downsample[1]]
        x = (x - self._input_mean) * self._input_scale
        y = (y - self._output_mean) * self._output_scale
        return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

    @property
    def name(self):
        return self._name

    @property
    def input_channels(self):
        return self._input_channels

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def input_statistics(self):
        return self._input_stats

    @property
    def output_statistics(self):
        return self._output_stats

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @property
    def resolution(self):
        return spatial_to_config(self._spatial_shape)

    @property
    def spatial_shape(self):
        return self._spatial_shape
