from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords, spatial_to_config


def _validate_checkpoint_normalization(checkpoint: dict[str, object], profile) -> None:
    raw = checkpoint.get("normalization")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("ECI checkpoint normalization must be a dictionary")
    dataset = raw.get("dataset")
    if dataset is not None and str(dataset) != profile.dataset:
        raise ValueError(
            f"ECI checkpoint normalization dataset '{dataset}' does not match "
            f"profile dataset '{profile.dataset}'"
        )
    resolution = raw.get("resolution")
    if (
        resolution is not None
        and profile.resolution is not None
        and as_spatial_shape(resolution) != as_spatial_shape(profile.resolution)
    ):
        raise ValueError(
            f"ECI checkpoint normalization resolution {resolution} does not match "
            f"profile resolution {profile.resolution}"
        )
    channels = tuple(raw.get("channels", ()))
    if channels and channels != tuple(profile.channels):
        raise ValueError(
            f"ECI checkpoint normalization channels {channels} do not match "
            f"profile channels {profile.channels}"
        )
    scale = raw.get("normalization_scale")
    if scale is not None and not np.isclose(float(scale), profile.normalization.scale):
        raise ValueError(
            f"ECI checkpoint normalization scale {scale} does not match "
            f"profile scale {profile.normalization.scale}"
        )
    statistics = raw.get("statistics")
    if isinstance(statistics, dict):
        mean = statistics.get("mean")
        std = statistics.get("std")
        if mean is not None and not np.allclose(mean, profile.normalization.mean):
            raise ValueError("ECI checkpoint normalization mean does not match profile")
        if std is not None and not np.allclose(std, profile.normalization.std):
            raise ValueError("ECI checkpoint normalization std does not match profile")


class SparseValueConstraint:
    def __init__(self, channel_index: int, value: torch.Tensor, mask: torch.Tensor):
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("SparseValueConstraint value must have shape [1,1,H,W]")
        if mask.shape != value.shape:
            raise ValueError("SparseValueConstraint mask/value shape mismatch")
        self.channel_index = int(channel_index)
        self.value = value
        self.mask = mask.to(dtype=torch.bool)

    def adjust(self, x1: torch.Tensor) -> torch.Tensor:
        result = x1.clone()
        prediction = result[:, self.channel_index : self.channel_index + 1]
        value = self.value.to(device=x1.device, dtype=x1.dtype).expand_as(prediction)
        mask = self.mask.to(device=x1.device).expand_as(prediction)
        result[:, self.channel_index : self.channel_index + 1] = torch.where(
            mask,
            value,
            prediction,
        )
        return result


class AreaAverageConstraint:
    def __init__(self, channel_index: int, value: torch.Tensor, model_resolution: object):
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError("AreaAverageConstraint value must have shape [1,1,H,W]")
        self.channel_index = int(channel_index)
        self.value = value
        self.model_shape = as_spatial_shape(model_resolution)
        obs_h, obs_w = value.shape[-2:]
        if self.model_shape[0] % obs_h != 0 or self.model_shape[1] % obs_w != 0:
            raise ValueError(
                f"Model resolution {self.model_shape} must be divisible by "
                f"observation shape {(obs_h, obs_w)}"
            )
        self.observation_size = (int(obs_h), int(obs_w))

    def adjust(self, x1: torch.Tensor) -> torch.Tensor:
        result = x1.clone()
        field = result[:, self.channel_index : self.channel_index + 1]
        pooled = F.adaptive_avg_pool2d(field, self.observation_size)
        value = self.value.to(device=x1.device, dtype=x1.dtype).expand_as(pooled)
        delta = value - pooled
        delta = F.interpolate(delta, size=field.shape[-2:], mode="nearest")
        result[:, self.channel_index : self.channel_index + 1] = field + delta
        return result


def _field_tensor(field: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(field).unsqueeze(0).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )


class ECIAdapter(MethodAdapter):
    """Unified adapter for ECI hard-constrained FFM sampling."""

    def load(self) -> None:
        if self.profile.method != "eci":
            raise ValueError(f"ECI adapter cannot use method profile '{self.profile.method}'")
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match "
                f"'{self.dataset.name}'"
            )
        self.dataset.validate_model_channels(self.profile.channels)
        if bool(self.profile.guidance.get("pending", False)) and not bool(
            self.profile.guidance.get("allow_pending", False)
        ):
            raise ValueError(
                f"ECI profile for dataset '{self.dataset.name}' is still pending; "
                "run only --dry-run until a real checkpoint is trained and checked"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"ECI checkpoint not found: {self.checkpoint}")

        self.torch_device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint, map_location=self.torch_device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("ECI checkpoint must be a dictionary")
        if isinstance(checkpoint.get("model"), torch.nn.Module):
            self.flow = checkpoint["model"].to(self.torch_device).eval()
        elif "model_state_dict" in checkpoint:
            from training_eci.flow import clean_eci_state_dict, create_eci_model

            config = dict(checkpoint.get("config", {}))
            model_config = dict(config.get("model_config", {}))
            self.flow, _ = create_eci_model(
                channels=len(self.profile.channels),
                resolution=self.profile.resolution,
                config=model_config,
            )
            _validate_checkpoint_normalization(checkpoint, self.profile)
            self.flow.load_state_dict(clean_eci_state_dict(checkpoint["model_state_dict"]))
            self.flow = self.flow.to(self.torch_device).eval()
        else:
            raise ValueError("ECI checkpoint must contain 'model' or 'model_state_dict'")

        self.model_shape = as_spatial_shape(self.profile.resolution)
        self.model_resolution = spatial_to_config(self.model_shape)
        spec = checkpoint.get("model_spec", {})
        if isinstance(spec, dict) and spec:
            if int(spec.get("channels", len(self.profile.channels))) != len(self.profile.channels):
                raise ValueError(
                    f"ECI checkpoint channels {spec.get('channels')} do not match "
                    f"profile channels {self.profile.channels}"
                )
            if as_spatial_shape(spec.get("resolution", self.model_resolution)) != self.model_shape:
                raise ValueError(
                    f"ECI checkpoint resolution {spec.get('resolution')} does not match "
                    f"profile resolution {self.model_resolution}"
                )

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        self.sampling = self.profile.sampling
        self.observed_channel = self.profile.channels.index(self.dataset.observed_field)

    def _build_constraint(self, case: BenchmarkCase) -> object:
        observed = _field_tensor(case.fields[case.observed_field], self.torch_device)
        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = scale_sensor_coords(
                case.sensor_coords,
                case.observation_resolution,
                self.model_shape,
                unique=True,
            )
            if observed.shape[-2:] != self.model_shape:
                observed = F.interpolate(
                    observed,
                    size=self.model_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            observed = self.normalizer.normalize_channel(observed, case.observed_field)
            mask = torch.zeros_like(observed, dtype=torch.bool)
            index = torch.as_tensor(coords, device=self.torch_device, dtype=torch.long)
            mask[0, 0, index[:, 0], index[:, 1]] = True
            return SparseValueConstraint(self.observed_channel, observed, mask)
        if case.observation_operator == "area_average":
            observed = self.normalizer.normalize_channel(observed, case.observed_field)
            return AreaAverageConstraint(
                self.observed_channel,
                observed,
                self.model_resolution,
            )
        raise ValueError(f"Unsupported observation operator: {case.observation_operator}")

    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")

        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=self.torch_device).manual_seed(seed)
        constraint = self._build_constraint(case)

        n_step = int(self.sampling.get("n_step", 200))
        n_mix = int(self.sampling.get("n_mix", 1))
        resample_step = self.sampling.get("resample_step", None)
        generated: list[torch.Tensor] = []
        count = 0
        while count < num_samples:
            current_batch = min(batch_size, num_samples - count)
            prediction = self.flow.eci_sample(
                batch_size=current_batch,
                channels=len(self.profile.channels),
                resolution=self.model_resolution,
                n_step=n_step,
                n_mix=n_mix,
                resample_step=resample_step,
                constraint=constraint,
                device=self.torch_device,
                dtype=torch.float32,
                generator=generator,
            )
            physical = self.normalizer.denormalize(prediction.to(torch.float32))
            physical = self.normalizer.apply_postprocess(physical)
            if physical.shape[-2:] != case.resolution:
                physical = F.interpolate(
                    physical,
                    size=case.resolution,
                    mode="bilinear",
                    align_corners=False,
                )
                physical = self.normalizer.apply_postprocess(physical)
            generated.append(physical.cpu())
            count += current_batch

        samples = torch.cat(generated, dim=0)[:num_samples].numpy()
        return {
            field: samples[:, self.profile.channels.index(field)]
            for field in case.target_fields
        }
