import numpy as np
from torch.utils.data import Dataset

from posteriorbench.spatial import as_spatial_shape, spatial_to_config
from posteriorbench.training_source import open_training_field_source
from training_fundps.dataset_utils import DatasetNormalizer


class PDEFieldDataset(Dataset):
    """Raw physical-space field tuples stored in a materialized Hugging Face dataset.

    The adjacent PosteriorBench metadata file defines the ordered model channels
    and full-pool statistics. Samples are normalized to the same scale used by
    the official FunDPS preprocessing: ``(x - mean) * 0.5 / std``.
    """

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
    ):
        if use_labels:
            raise ValueError("PDEFieldDataset_hf training sources do not provide labels")
        if xflip:
            raise ValueError("Horizontal flips are not supported")
        if cache:
            print("Warning: PDEFieldDataset uses lazy Hugging Face dataset reads; ignoring cache=True")

        self._source = open_training_field_source(path)
        self._metadata = self._source.metadata
        self._name = str(self._metadata["dataset"])
        self._channels = tuple(self._metadata["channels"])
        if not self._channels:
            raise ValueError("Training metadata must define at least one model channel")
        if len(set(self._channels)) != len(self._channels):
            raise ValueError(f"Training channels must be unique, got {self._channels}")

        shapes = self._source.channel_shapes(self._channels)
        if any(len(shape) != 3 for shape in shapes) or len(set(shapes)) != 1:
            raise ValueError(f"Expected matching [N,H,W] channels, got {shapes}")

        num_samples, raw_height, raw_width = shapes[0]
        raw_shape = (int(raw_height), int(raw_width))
        metadata_count = int(self._metadata["num_samples"])
        metadata_resolution = tuple(self._metadata["resolution"])
        if metadata_count != num_samples or metadata_resolution != raw_shape:
            raise ValueError(
                "source/metadata shape mismatch: "
                f"source={shapes[0]}, metadata=({metadata_count}, {metadata_resolution})"
            )

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
        self._indices = indices

        statistics = self._metadata["statistics"]
        self._stats = {
            "mean": [float(statistics[channel]["mean"]) for channel in self._channels],
            "std": [float(statistics[channel]["std"]) for channel in self._channels],
        }
        self._normalization_scale = 0.5
        if not np.all(np.isfinite(self._stats["mean"])):
            raise ValueError("Channel means must be finite")
        if not np.all(np.isfinite(self._stats["std"])) or np.any(np.asarray(self._stats["std"]) <= 0):
            raise ValueError("Channel standard deviations must be finite and positive")

        self._mean = np.asarray(self._stats["mean"], dtype=np.float32)[:, None, None]
        self._scale = (
            self._normalization_scale / np.asarray(self._stats["std"], dtype=np.float32)
        )[:, None, None]
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

    def create_normalizer(self):
        return DatasetNormalizer(self._name, self._stats)

    def denormalize(self, x_normalized):
        return self.create_normalizer().denormalize(x_normalized)

    @property
    def name(self):
        return self._name

    @property
    def channels(self):
        return self._channels

    @property
    def statistics(self):
        return self._stats

    @property
    def normalization_scale(self):
        return self._normalization_scale

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
