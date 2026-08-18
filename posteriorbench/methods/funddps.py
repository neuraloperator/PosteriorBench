from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from posteriorbench.compat import (
    install_legacy_checkpoint_aliases,
    load_pickle_with_torch_storage_map,
)
from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords, spatial_to_config
from training_fundps.noise_samplers import RBFKernel


def validate_funddps_channels(
    dataset_name: str,
    target_fields: tuple[str, ...],
    channels: tuple[str, ...],
    method_label: str = "Fun-DDPS",
) -> None:
    if channels != target_fields:
        raise ValueError(
            f"{method_label} {dataset_name} prior expects target channels "
            f"{target_fields}, got {channels}"
        )


def _edm_sigma_steps(
    net: torch.nn.Module,
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("Fun-DDPS sampling requires at least two steps")
    sigma_min = max(float(sigma_min), float(getattr(net, "sigma_min", sigma_min)))
    sigma_max = min(float(sigma_max), float(getattr(net, "sigma_max", sigma_max)))
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([net.round_sigma(steps), torch.zeros_like(steps[:1])])


def _stats_tensors(
    stats: dict[str, object],
    channels: tuple[str, ...],
    scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_values = stats.get("mean")
    std_values = stats.get("std")
    if mean_values is None or std_values is None:
        raise ValueError("Surrogate normalization stats must contain mean/std")
    if len(mean_values) != len(channels) or len(std_values) != len(channels):
        raise ValueError(
            f"Surrogate stats for {channels} must have one mean/std per channel"
        )
    mean = torch.as_tensor(mean_values, device=device, dtype=dtype).reshape(1, -1, 1, 1)
    std = torch.as_tensor(std_values, device=device, dtype=dtype).reshape(1, -1, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("Surrogate standard deviations must be positive")
    return mean, std / float(scale)


def _normalize_raw(
    x_raw: torch.Tensor,
    stats: dict[str, object],
    channels: tuple[str, ...],
    scale: float,
) -> torch.Tensor:
    mean, std_over_scale = _stats_tensors(
        stats,
        channels,
        scale,
        x_raw.device,
        x_raw.dtype,
    )
    return (x_raw - mean) / std_over_scale


def _denormalize_surrogate(
    x_normalized: torch.Tensor,
    stats: dict[str, object],
    channels: tuple[str, ...],
    scale: float,
) -> torch.Tensor:
    mean, std_over_scale = _stats_tensors(
        stats,
        channels,
        scale,
        x_normalized.device,
        x_normalized.dtype,
    )
    return x_normalized * std_over_scale + mean


class FunDDPSSurrogate(torch.nn.Module):
    """Neural-operator surrogate with PosteriorBench normalization bridges."""

    def __init__(
        self,
        model: torch.nn.Module,
        normalization: dict[str, object],
        device: torch.device,
    ):
        super().__init__()
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.input_channels = tuple(normalization["input_channels"])
        self.output_channels = tuple(normalization["output_channels"])
        self.scale = float(normalization.get("normalization_scale", 1.0))
        self.input_stats = dict(normalization["input"])
        self.output_stats = dict(normalization["output"])
        self.spatial_shape = as_spatial_shape(normalization["resolution"])
        self.resolution = spatial_to_config(self.spatial_shape)

    def forward_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        x_surrogate = _normalize_raw(
            x_raw,
            self.input_stats,
            self.input_channels,
            self.scale,
        )
        y_surrogate = self.model(x_surrogate.to(torch.float32)).to(x_raw.dtype)
        return _denormalize_surrogate(
            y_surrogate,
            self.output_stats,
            self.output_channels,
            self.scale,
        )

    def normalize_output_channel(self, x_raw: torch.Tensor, channel: str) -> torch.Tensor:
        if channel not in self.output_channels:
            raise ValueError(
                f"Unknown surrogate output channel '{channel}' for {self.output_channels}"
            )
        index = self.output_channels.index(channel)
        stats = {
            "mean": [self.output_stats["mean"][index]],
            "std": [self.output_stats["std"][index]],
        }
        return _normalize_raw(x_raw, stats, (channel,), self.scale)


def _load_surrogate(path: Path, device: torch.device) -> FunDDPSSurrogate:
    if not path.exists():
        raise FileNotFoundError(f"Fun-DDPS surrogate checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Fun-DDPS surrogate checkpoint must be a dictionary")

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Fun-DDPS surrogate checkpoint is missing normalization metadata")

    if isinstance(checkpoint.get("model"), torch.nn.Module):
        model = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        from training_funddps.surrogate import create_surrogate_model

        spec = dict(checkpoint.get("model_spec", {}))
        config = dict(checkpoint.get("config", {}))
        model_config = dict(config.get("model_config", {}))
        model_name = spec.get("model", config.get("model", "fno"))
        model, _ = create_surrogate_model(
            model_name,
            in_channels=int(spec.get("in_channels", len(normalization["input_channels"]))),
            out_channels=int(spec.get("out_channels", len(normalization["output_channels"]))),
            resolution=spec.get("resolution", normalization["resolution"]),
            config=model_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError(
            "Fun-DDPS surrogate checkpoint must contain either 'model' or "
            "'model_state_dict'"
        )

    return FunDDPSSurrogate(model, normalization, device)


class FunDDPSAdapter(MethodAdapter):
    """Unified adapter for decoupled Fun-DDPS posterior sampling."""

    method_name = "funddps"
    method_label = "Fun-DDPS"

    def load(self) -> None:
        if self.profile.method != self.method_name:
            raise ValueError(
                f"{self.method_label} adapter cannot use method profile "
                f"'{self.profile.method}'"
            )
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match "
                f"'{self.dataset.name}'"
            )
        validate_funddps_channels(
            self.dataset.name,
            self.dataset.target_fields,
            self.profile.channels,
            method_label=self.method_label,
        )
        if bool(self.profile.guidance.get("pending", False)) and not bool(
            self.profile.guidance.get("allow_pending", False)
        ):
            raise ValueError(
                f"Guidance is not finalized for {self.method_label} dataset "
                f"'{self.dataset.name}'; "
                "run only --dry-run or use a non-pending profile for experiments"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Prior checkpoint not found: {self.checkpoint}")

        self.torch_device = torch.device(self.device)
        install_legacy_checkpoint_aliases()
        checkpoint = load_pickle_with_torch_storage_map(
            self.checkpoint,
            map_location=torch.device("cpu"),
        )
        if "ema" not in checkpoint:
            raise KeyError(f"Fun-DDPS prior checkpoint {self.checkpoint} has no 'ema' network")
        self.net = checkpoint["ema"].to(self.torch_device)
        self.net.eval()

        self.model_shape = as_spatial_shape(self.profile.resolution or self.net.img_resolution)
        self.model_resolution = spatial_to_config(self.model_shape)
        if int(self.net.img_channels) != len(self.profile.channels):
            raise ValueError(
                f"Prior checkpoint has {self.net.img_channels} channels, profile has "
                f"{len(self.profile.channels)}"
            )

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        surrogate_config = dict(self.profile.guidance.get("surrogate", {}))
        surrogate_path = Path(surrogate_config.get("checkpoint", ""))
        if not surrogate_path.is_absolute():
            surrogate_path = self.profile.path.parent.parent.parent / surrogate_path
        self.surrogate = _load_surrogate(surrogate_path, self.torch_device)
        if self.surrogate.input_channels != self.profile.channels:
            raise ValueError(
                f"Surrogate inputs {self.surrogate.input_channels} do not match "
                f"prior channels {self.profile.channels}"
            )
        if self.dataset.observed_field not in self.surrogate.output_channels:
            raise ValueError(
                f"Surrogate outputs {self.surrogate.output_channels} do not contain "
                f"observed field '{self.dataset.observed_field}'"
            )

        self.sampling = self.profile.sampling
        self.guidance = self.profile.guidance
        field_weights = self.guidance.get("field_weights", {})
        self.observation_weight = float(
            self.guidance.get(
                "observation_weight",
                field_weights.get(self.dataset.observed_field, 0.0),
            )
        )
        self.loss_type = str(self.guidance.get("loss_type", "mse"))
        self.normalize_observation = bool(self.guidance.get("normalize", False))

    def _prepare_case_observation(self, case: BenchmarkCase) -> dict[str, object]:
        output_shape = self.surrogate.spatial_shape
        observed = torch.from_numpy(case.fields[case.observed_field]).unsqueeze(0).unsqueeze(0)
        observed = observed.to(device=self.torch_device, dtype=torch.float32)
        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = scale_sensor_coords(
                case.sensor_coords,
                case.observation_resolution,
                output_shape,
                unique=True,
            )
            if observed.shape[-2:] != output_shape:
                observed = F.interpolate(
                    observed,
                    size=output_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            mask = torch.zeros_like(observed)
            index = torch.as_tensor(coords, device=self.torch_device, dtype=torch.long)
            mask[0, 0, index[:, 0], index[:, 1]] = 1
            if self.normalize_observation:
                observed = self.surrogate.normalize_output_channel(observed, case.observed_field)
            return {"observed": observed, "mask": mask}
        if case.observation_operator == "area_average":
            if self.normalize_observation:
                observed = self.surrogate.normalize_output_channel(observed, case.observed_field)
            return {"observed": observed, "size": tuple(observed.shape[-2:])}
        raise ValueError(f"Unsupported observation operator: {case.observation_operator}")

    def _field_loss(self, prediction: torch.Tensor, prepared: dict[str, object]) -> torch.Tensor:
        observed = prepared["observed"]
        if not isinstance(observed, torch.Tensor):
            raise TypeError("prepared observation is malformed")
        if "mask" in prepared:
            mask = prepared["mask"]
            if not isinstance(mask, torch.Tensor):
                raise TypeError("prepared sparse mask is malformed")
            diff = (prediction - observed) * mask
            denom = mask.sum().clamp_min(1)
        else:
            size = prepared["size"]
            if not isinstance(size, tuple):
                raise TypeError("prepared area-average size is malformed")
            diff = F.adaptive_avg_pool2d(prediction, size) - observed
            denom = torch.as_tensor(diff[0].numel(), device=diff.device, dtype=diff.dtype)
        if self.loss_type == "mse":
            return diff.square().sum() / denom
        if self.loss_type in {"l2", "norm"}:
            return torch.linalg.vector_norm(diff.flatten(1), dim=1).mean()
        raise ValueError(f"Unsupported Fun-DDPS loss_type '{self.loss_type}'")

    def _surrogate_prediction(self, x_denoised: torch.Tensor) -> torch.Tensor:
        x_raw = self.normalizer.denormalize(x_denoised.to(torch.float32))
        if x_raw.shape[-2:] != self.surrogate.spatial_shape:
            x_raw = F.interpolate(
                x_raw,
                size=self.surrogate.spatial_shape,
                mode="bilinear",
                align_corners=False,
            )
        output = self.surrogate.forward_raw(x_raw)
        index = self.surrogate.output_channels.index(self.dataset.observed_field)
        prediction = output[:, index : index + 1]
        if self.normalize_observation:
            prediction = self.surrogate.normalize_output_channel(
                prediction,
                self.dataset.observed_field,
            )
        return prediction

    def _sample_latents(
        self,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        shape = [
            batch_size,
            len(self.profile.channels),
            self.model_shape[0],
            self.model_shape[1],
        ]
        init_latents = str(self.sampling.get("init_latents", "gaussian"))
        if init_latents == "rbf":
            sampler = RBFKernel(
                len(self.profile.channels),
                self.model_shape[0],
                self.model_shape[1],
                scale=float(self.sampling.get("rbf_scale", 0.05)),
                device=self.torch_device,
            )
            return sampler.sample(batch_size)
        if init_latents == "gaussian":
            return torch.randn(shape, device=self.torch_device, generator=generator)
        raise ValueError(f"Unsupported Fun-DDPS init_latents '{init_latents}'")

    def _sample_batch(
        self,
        case: BenchmarkCase,
        batch_size: int,
        generator: torch.Generator,
        prepared: dict[str, object],
    ) -> torch.Tensor:
        iterations = int(self.sampling.get("iterations", 500))
        sigma_steps = _edm_sigma_steps(
            self.net,
            iterations,
            float(self.sampling.get("sigma_min", 0.002)),
            float(self.sampling.get("sigma_max", 80.0)),
            float(self.sampling.get("rho", 7.0)),
            self.torch_device,
        )
        x_next = self._sample_latents(batch_size, generator).to(torch.float64) * sigma_steps[0]
        class_labels = None

        for index, (sigma_cur, sigma_next) in enumerate(zip(sigma_steps[:-1], sigma_steps[1:])):
            x_cur = x_next.detach().clone().requires_grad_(self.observation_weight != 0)
            sigma_cur = self.net.round_sigma(sigma_cur)
            x_denoised = self.net(x_cur, sigma_cur, class_labels=class_labels).to(torch.float64)
            d_cur = (x_cur - x_denoised) / sigma_cur
            proposal = x_cur + (sigma_next - sigma_cur) * d_cur

            if index < iterations - 1:
                sigma_next = self.net.round_sigma(sigma_next)
                x_denoised_2 = self.net(proposal, sigma_next, class_labels=class_labels).to(torch.float64)
                d_prime = (proposal - x_denoised_2) / sigma_next
                proposal = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

            if self.observation_weight != 0:
                observed_prediction = self._surrogate_prediction(x_denoised)
                loss = self._field_loss(observed_prediction, prepared)
                grad = torch.autograd.grad(self.observation_weight * loss, x_cur)[0]
                proposal = proposal - grad
            x_next = proposal.detach()

        physical = self.normalizer.denormalize(x_next.to(torch.float32))
        return self.normalizer.apply_postprocess(physical)

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
        prepared = self._prepare_case_observation(case)

        generated: list[torch.Tensor] = []
        count = 0
        while count < num_samples:
            current_batch = min(batch_size, num_samples - count)
            prediction = self._sample_batch(case, current_batch, generator, prepared)
            if prediction.shape[-2:] != case.resolution:
                prediction = F.interpolate(
                    prediction,
                    size=case.resolution,
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = self.normalizer.apply_postprocess(prediction)
            generated.append(prediction.cpu())
            count += current_batch

        samples = torch.cat(generated, dim=0)[:num_samples].numpy()
        return {
            channel: samples[:, index]
            for index, channel in enumerate(self.profile.channels)
        }
