from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords
from training_mcdropout.model import create_mcdropout_model, set_mc_dropout_mode


def validate_mcdropout_channels(
    dataset_name: str,
    target_fields: tuple[str, ...],
    channels: tuple[str, ...],
) -> None:
    if channels != target_fields:
        raise ValueError(
            f"MC-dropout {dataset_name} target channels must be {target_fields}, "
            f"got {channels}"
        )


def _condition_stats(guidance: dict[str, Any]) -> dict[str, object]:
    raw = dict(guidance.get("condition_normalization", {}))
    if not raw:
        raise ValueError("MC-dropout profile guidance must include condition_normalization")
    mean = tuple(float(value) for value in raw["mean"])
    std = tuple(float(value) for value in raw["std"])
    if len(mean) != 1 or len(std) != 1:
        raise ValueError("MC-dropout currently expects one observed-field channel")
    if std[0] <= 0:
        raise ValueError("condition_normalization std must be positive")
    return {
        "channels": tuple(raw["channels"]),
        "mean": mean,
        "std": std,
        "scale": float(raw.get("scale", 1.0)),
    }


def _nearest_upsample(field: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = field.shape[-2:]
    out_h, out_w = output_shape
    if (in_h, in_w) == output_shape:
        return field.astype(np.float32, copy=False)
    if out_h % in_h != 0 or out_w % in_w != 0:
        tensor = torch.from_numpy(field).reshape(1, 1, in_h, in_w).to(torch.float32)
        return (
            F.interpolate(tensor, size=output_shape, mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .numpy()
            .astype(np.float32, copy=False)
        )
    return np.repeat(np.repeat(field, out_h // in_h, axis=-2), out_w // in_w, axis=-1).astype(
        np.float32,
        copy=False,
    )


class MCDropoutAdapter(MethodAdapter):
    """Unified adapter for direct FNO MC-dropout posterior samples."""

    def load(self) -> None:
        if self.profile.method != "mcdropout":
            raise ValueError(
                f"MC-dropout adapter cannot use method profile '{self.profile.method}'"
            )
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match "
                f"'{self.dataset.name}'"
            )
        validate_mcdropout_channels(
            self.dataset.name,
            self.dataset.target_fields,
            self.profile.channels,
        )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"MC-dropout checkpoint not found: {self.checkpoint}")

        self.torch_device = torch.device(self.device)
        checkpoint = torch.load(
            self.checkpoint,
            map_location=self.torch_device,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("MC-dropout checkpoint must be a dictionary")
        if checkpoint.get("schema_version") != "posteriorbench.mcdropout.v1":
            raise ValueError(
                "Expected schema_version='posteriorbench.mcdropout.v1', got "
                f"{checkpoint.get('schema_version')!r}"
            )

        normalization = dict(checkpoint.get("normalization", {}))
        target_channels = tuple(normalization.get("target_channels", ()))
        if target_channels and target_channels != self.profile.channels:
            raise ValueError(
                f"Checkpoint target channels {target_channels} do not match "
                f"profile {self.profile.channels}"
            )
        self.condition_channels = tuple(
            normalization.get("condition_channels", ("observed_value", "mask"))
        )
        if self.condition_channels != ("observed_value", "mask"):
            raise ValueError(
                "MC-dropout checkpoint condition channels must be "
                "('observed_value', 'mask')"
            )

        config = dict(checkpoint.get("config", {}))
        model_config = dict(config.get("model_config", {}))
        spec = dict(checkpoint.get("model_spec", {}))
        resolution = spec.get("resolution", self.profile.resolution)
        self.model_shape = as_spatial_shape(resolution)
        self.model, _ = create_mcdropout_model(
            str(config.get("model", spec.get("model", "fno"))),
            in_channels=int(spec.get("in_channels", len(self.condition_channels))),
            out_channels=len(self.profile.channels),
            resolution=self.model_shape,
            config=model_config,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.torch_device)
        set_mc_dropout_mode(self.model)

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        self.sampling = dict(self.profile.sampling)
        self.guidance = dict(self.profile.guidance)
        self.condition_stats = _condition_stats(self.guidance)
        observed_channels = tuple(self.condition_stats["channels"])
        if observed_channels != (self.dataset.observed_field,):
            raise ValueError(
                f"Profile condition channels {observed_channels} do not match "
                f"observed field '{self.dataset.observed_field}'"
            )
        self.observation_operator = str(self.guidance.get("observation_operator"))
        if self.observation_operator != self.dataset.observation_operator:
            raise ValueError(
                f"Profile observation operator '{self.observation_operator}' does not "
                f"match dataset operator '{self.dataset.observation_operator}'"
            )

    def _normalize_observed(self, observed: np.ndarray) -> np.ndarray:
        mean = float(self.condition_stats["mean"][0])
        std = float(self.condition_stats["std"][0])
        scale = float(self.condition_stats["scale"])
        return ((observed.astype(np.float32) - mean) * (scale / std)).astype(np.float32)

    def _condition_sparse(self, case: BenchmarkCase) -> torch.Tensor:
        if case.sensor_coords is None:
            raise ValueError(f"{case.dataset} requires stored sensor coordinates")
        observed = case.fields[case.observed_field]
        if observed.shape != self.model_shape:
            tensor = torch.from_numpy(observed).reshape(1, 1, *observed.shape).to(torch.float32)
            observed = (
                F.interpolate(
                    tensor,
                    size=self.model_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )
        coords = scale_sensor_coords(
            case.sensor_coords,
            case.observation_resolution,
            self.model_shape,
            unique=True,
        )
        normalized = self._normalize_observed(observed)
        value = np.zeros(self.model_shape, dtype=np.float32)
        mask = np.zeros(self.model_shape, dtype=np.float32)
        value[coords[:, 0], coords[:, 1]] = normalized[coords[:, 0], coords[:, 1]]
        mask[coords[:, 0], coords[:, 1]] = 1.0
        condition = np.stack([value, mask], axis=0)
        return torch.from_numpy(condition).unsqueeze(0).to(
            device=self.torch_device,
            dtype=torch.float32,
        )

    def _condition_area_average(self, case: BenchmarkCase) -> torch.Tensor:
        observed = case.fields[case.observed_field]
        normalized = self._normalize_observed(observed)
        value = _nearest_upsample(normalized, self.model_shape)
        mask = np.ones(self.model_shape, dtype=np.float32)
        condition = np.stack([value, mask], axis=0)
        return torch.from_numpy(condition).unsqueeze(0).to(
            device=self.torch_device,
            dtype=torch.float32,
        )

    def _condition_for_case(self, case: BenchmarkCase) -> torch.Tensor:
        if self.observation_operator == "sparse_points":
            return self._condition_sparse(case)
        if self.observation_operator == "area_average":
            return self._condition_area_average(case)
        raise ValueError(f"Unsupported observation operator: {self.observation_operator}")

    def generate(
        self,
        case: BenchmarkCase,
        num_samples: int,
        batch_size: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        torch.manual_seed(seed)
        if self.torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        condition_single = self._condition_for_case(case)
        generated: list[torch.Tensor] = []
        count = 0
        set_mc_dropout_mode(self.model)
        with torch.no_grad():
            while count < num_samples:
                current_batch = min(batch_size, num_samples - count)
                condition = condition_single.expand(current_batch, -1, -1, -1).contiguous()
                prediction = self.model(condition)
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
            field: samples[:, self.profile.channels.index(field)].astype(
                np.float32,
                copy=False,
            )
            for field in case.target_fields
        }
