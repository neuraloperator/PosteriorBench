from collections.abc import Mapping

import torch

from posteriorbench.artifacts import NormalizationProfile


class ModelNormalizer:
    """Checkpoint-specific affine normalization used by a method adapter."""

    def __init__(
        self,
        profile: NormalizationProfile,
        discrete_values: Mapping[str, list[float]] | None = None,
    ):
        self.channels = profile.channels
        self.mean = torch.tensor(profile.mean).reshape(1, -1, 1, 1)
        self.std = torch.tensor(profile.std).reshape(1, -1, 1, 1)
        self.scale = profile.scale
        self.discrete_values = dict(discrete_values or {})

    def _check_shape(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [N,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} channels {self.channels}, got {x.shape[1]}"
            )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) * (self.scale / std)

    def normalize_channel(self, x: torch.Tensor, channel: str) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(
                f"Expected one field with shape [N,1,H,W], got {tuple(x.shape)}"
            )
        if channel not in self.channels:
            raise ValueError(f"Unknown channel '{channel}' for {self.channels}")
        index = self.channels.index(channel)
        mean = self.mean[:, index : index + 1].to(device=x.device, dtype=x.dtype)
        std = self.std[:, index : index + 1].to(device=x.device, dtype=x.dtype)
        return (x - mean) * (self.scale / std)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * (std / self.scale) + mean

    def transform(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        if denormalize:
            x = self.denormalize(x)
        return self.apply_postprocess(x)

    def apply_postprocess(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shape(x)
        result = x.clone()
        for field, values in self.discrete_values.items():
            if field not in self.channels:
                raise ValueError(f"Unknown discrete field '{field}' for channels {self.channels}")
            if len(values) != 2:
                raise ValueError("Only binary discrete postprocessing is currently supported")
            low, high = sorted(float(value) for value in values)
            threshold = (low + high) / 2
            channel = self.channels.index(field)
            result[:, channel] = torch.where(
                result[:, channel] > threshold,
                torch.as_tensor(high, device=x.device, dtype=x.dtype),
                torch.as_tensor(low, device=x.device, dtype=x.dtype),
            )
        return result
