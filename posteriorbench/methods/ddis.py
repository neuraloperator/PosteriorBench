from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from typing import Callable

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
from posteriorbench.spatial import (
    as_spatial_shape,
    scale_sensor_coords as _scale_coords,
    spatial_to_config,
)
from training_fundps.noise_samplers import RBFKernel


def validate_ddis_channels(
    dataset_name: str,
    target_fields: tuple[str, ...],
    channels: tuple[str, ...],
) -> None:
    if channels != target_fields:
        raise ValueError(
            f"DDIS {dataset_name} prior expects target channels "
            f"{target_fields}, got {channels}"
        )


def _scale_sensor_coords(
    coords: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    return _scale_coords(coords, source_shape, target_shape, unique=True)


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
        raise ValueError("DDIS surrogate normalization stats must contain mean/std")
    if len(mean_values) != len(channels) or len(std_values) != len(channels):
        raise ValueError(
            f"DDIS surrogate stats for {channels} must have one mean/std per channel"
        )
    mean = torch.as_tensor(mean_values, device=device, dtype=dtype).reshape(1, -1, 1, 1)
    std = torch.as_tensor(std_values, device=device, dtype=dtype).reshape(1, -1, 1, 1)
    if torch.any(std <= 0):
        raise ValueError("DDIS surrogate standard deviations must be positive")
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


def _create_surrogate_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    resolution: int,
    config: dict[str, object],
) -> torch.nn.Module:
    name = str(model_name).lower()
    if name == "fno_pad":
        try:
            from training_ddis.surrogate import create_ddis_surrogate_model
        except ImportError as error:
            raise ImportError(
                "DDIS FNO_pad surrogate loading requires training_ddis.surrogate."
            ) from error

        model, _ = create_ddis_surrogate_model(
            name,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution=resolution,
            config=config,
        )
        return model

    try:
        import neuralop.models as neuralop_models
    except ImportError as error:
        raise ImportError(
            "DDIS surrogate loading requires neuralop when the checkpoint stores "
            "only a model_state_dict."
        ) from error

    if name == "fno":
        return neuralop_models.FNO(
            n_modes=tuple(config.get("n_modes", (64, 64))),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 64)),
            n_layers=int(config.get("n_layers", 4)),
            domain_padding=config.get("domain_padding", None),
            domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
        )
    raise ValueError("DDIS surrogate model must be one of: fno_pad, fno")


class DDISSurrogate(torch.nn.Module):
    """DDIS forward-operator surrogate with PosteriorBench normalization bridges."""

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

    def normalize_output_channel(
        self,
        x_raw: torch.Tensor,
        channel: str,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        if channel not in self.output_channels:
            raise ValueError(
                f"Unknown DDIS surrogate output channel '{channel}' for "
                f"{self.output_channels}"
            )
        index = self.output_channels.index(channel)
        stats = {
            "mean": [self.output_stats["mean"][index]],
            "std": [self.output_stats["std"][index]],
        }
        return _normalize_raw(x_raw, stats, (channel,), self.scale if scale is None else scale)


def _load_ddis_surrogate(path: Path, device: torch.device) -> DDISSurrogate:
    if not path.exists():
        raise FileNotFoundError(f"DDIS surrogate checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("DDIS surrogate checkpoint must be a dictionary")

    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("DDIS surrogate checkpoint is missing normalization metadata")

    if isinstance(checkpoint.get("model"), torch.nn.Module):
        model = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        spec = dict(checkpoint.get("model_spec", {}))
        config = dict(checkpoint.get("config", {}))
        model_config = dict(config.get("model_config", {}))
        model = _create_surrogate_model(
            str(spec.get("model", config.get("model", "fno"))),
            in_channels=int(spec.get("in_channels", len(normalization["input_channels"]))),
            out_channels=int(
                spec.get("out_channels", len(normalization["output_channels"]))
            ),
            resolution=spec.get("resolution", normalization["resolution"]),
            config=model_config,
        )
        model.load_state_dict(_clean_ddis_state_dict(checkpoint["model_state_dict"]))
    else:
        raise ValueError(
            "DDIS surrogate checkpoint must contain either 'model' or "
            "'model_state_dict'"
        )
    return DDISSurrogate(model, normalization, device)


def _clean_ddis_state_dict(state_dict: object) -> dict[str, object]:
    if not isinstance(state_dict, Mapping):
        raise ValueError("DDIS surrogate model_state_dict must be a mapping")
    return {
        str(key): value
        for key, value in state_dict.items()
        if str(key) != "_metadata"
    }


def _daps_sigma_steps(config: dict[str, object], sigma_max: float | None = None) -> np.ndarray:
    num_steps = int(config.get("num_steps", 100))
    if num_steps < 1:
        raise ValueError("DDIS DAPS scheduler requires num_steps >= 1")
    schedule = str(config.get("schedule", "linear"))
    rho = float(config.get("rho", 7.0))
    if rho <= 0:
        raise ValueError("DDIS DAPS scheduler requires rho > 0")
    sigma_hi = float(sigma_max if sigma_max is not None else config.get("sigma_max", 10.0))
    sigma_min = float(config.get("sigma_min", 0.01))
    sigma_final = float(config.get("sigma_final", 0.0))
    if sigma_hi <= 0 or sigma_min <= 0:
        raise ValueError("DDIS DAPS scheduler requires positive sigma_max/sigma_min")

    if schedule == "linear":
        sigma_fn: Callable[[np.ndarray], np.ndarray] = lambda t: t
        sigma_inv: Callable[[float], float] = lambda sigma: sigma
    elif schedule == "sqrt":
        sigma_fn = np.sqrt
        sigma_inv = lambda sigma: sigma**2
    else:
        raise ValueError(f"Unsupported DDIS DAPS schedule '{schedule}'")

    grid = np.linspace(0.0, 1.0, num_steps)
    time_steps = (
        sigma_hi ** (1.0 / rho)
        + grid * (sigma_min ** (1.0 / rho) - sigma_hi ** (1.0 / rho))
    ) ** rho
    time_steps = np.append(time_steps, sigma_inv(sigma_final))
    return sigma_fn(time_steps).astype(np.float64)


class DDISAdapter(MethodAdapter):
    """Unified adapter for DDIS decoupled DAPS posterior sampling."""

    def load(self) -> None:
        if self.profile.method != "ddis":
            raise ValueError(
                f"DDIS adapter cannot use method profile '{self.profile.method}'"
            )
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match "
                f"'{self.dataset.name}'"
            )
        validate_ddis_channels(
            self.dataset.name,
            self.dataset.target_fields,
            self.profile.channels,
        )
        if bool(self.profile.guidance.get("pending", False)) and not bool(
            self.profile.guidance.get("allow_pending", False)
        ):
            raise ValueError(
                f"Guidance is not finalized for DDIS dataset '{self.dataset.name}'; "
                "run only --dry-run or use a non-pending profile for experiments"
            )
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"DDIS prior checkpoint not found: {self.checkpoint}")

        self.torch_device = torch.device(self.device)
        install_legacy_checkpoint_aliases()
        checkpoint = load_pickle_with_torch_storage_map(
            self.checkpoint,
            map_location=torch.device("cpu"),
        )
        if "ema" not in checkpoint:
            raise KeyError(f"DDIS prior checkpoint {self.checkpoint} has no 'ema' network")
        self.net = checkpoint["ema"].to(self.torch_device)
        self.net.eval()

        self.model_shape = as_spatial_shape(self.profile.resolution or self.net.img_resolution)
        self.model_resolution = spatial_to_config(self.model_shape)
        if int(self.net.img_channels) != len(self.profile.channels):
            raise ValueError(
                f"DDIS prior checkpoint has {self.net.img_channels} channels, "
                f"profile has {len(self.profile.channels)}"
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
        self.surrogate = _load_ddis_surrogate(surrogate_path, self.torch_device)
        if self.surrogate.input_channels != self.profile.channels:
            raise ValueError(
                f"DDIS surrogate inputs {self.surrogate.input_channels} do not "
                f"match prior channels {self.profile.channels}"
            )
        if self.dataset.observed_field not in self.surrogate.output_channels:
            raise ValueError(
                f"DDIS surrogate outputs {self.surrogate.output_channels} do not "
                f"contain observed field '{self.dataset.observed_field}'"
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
        self.observation_normalization_scale = float(
            self.guidance.get("observation_normalization_scale", self.surrogate.scale)
        )
        if self.observation_normalization_scale <= 0:
            raise ValueError("DDIS observation_normalization_scale must be positive")

    def _prepare_case_observation(self, case: BenchmarkCase) -> dict[str, object]:
        output_shape = self.surrogate.spatial_shape
        observed = torch.from_numpy(case.fields[case.observed_field]).unsqueeze(0).unsqueeze(0)
        observed = observed.to(device=self.torch_device, dtype=torch.float32)
        if case.observation_operator == "sparse_points":
            if case.sensor_coords is None:
                raise ValueError(f"{case.dataset} requires stored sensor coordinates")
            coords = _scale_sensor_coords(
                case.sensor_coords,
                case.observation_resolution,
                output_shape,
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
                observed = self.surrogate.normalize_output_channel(
                    observed,
                    case.observed_field,
                    scale=self.observation_normalization_scale,
                )
            return {"observed": observed, "mask": mask}
        if case.observation_operator == "area_average":
            if self.normalize_observation:
                observed = self.surrogate.normalize_output_channel(
                    observed,
                    case.observed_field,
                    scale=self.observation_normalization_scale,
                )
            return {"observed": observed, "size": tuple(observed.shape[-2:])}
        raise ValueError(f"Unsupported DDIS observation operator: {case.observation_operator}")

    def _field_loss(self, prediction: torch.Tensor, prepared: dict[str, object]) -> torch.Tensor:
        observed = prepared["observed"]
        if not isinstance(observed, torch.Tensor):
            raise TypeError("prepared DDIS observation is malformed")
        if "mask" in prepared:
            mask = prepared["mask"]
            if not isinstance(mask, torch.Tensor):
                raise TypeError("prepared DDIS sparse mask is malformed")
            diff = (prediction - observed) * mask
            denom = mask.sum().clamp_min(1)
        else:
            size = prepared["size"]
            if not isinstance(size, tuple):
                raise TypeError("prepared DDIS area-average size is malformed")
            diff = F.adaptive_avg_pool2d(prediction, size) - observed
            denom = torch.as_tensor(diff[0].numel(), device=diff.device, dtype=diff.dtype)
        if self.loss_type == "mse":
            return diff.square().sum() / denom
        if self.loss_type in {"l1", "mae"}:
            return diff.abs().sum() / denom
        if self.loss_type in {"l2", "norm"}:
            return torch.linalg.vector_norm(diff.flatten(1), dim=1).mean()
        raise ValueError(f"Unsupported DDIS loss_type '{self.loss_type}'")

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
                scale=self.observation_normalization_scale,
            )
        return prediction

    def _make_noise_sampler(self) -> RBFKernel | None:
        init_latents = str(self.sampling.get("init_latents", "rbf"))
        if init_latents == "rbf":
            return RBFKernel(
                len(self.profile.channels),
                self.model_shape[0],
                self.model_shape[1],
                scale=float(self.sampling.get("rbf_scale", 0.05)),
                device=self.torch_device,
            )
        if init_latents in {"gaussian", "white_noise"}:
            return None
        raise ValueError(f"Unsupported DDIS init_latents '{init_latents}'")

    def _sample_noise(
        self,
        batch_size: int,
        generator: torch.Generator,
        noise_sampler: RBFKernel | None,
    ) -> torch.Tensor:
        if noise_sampler is not None:
            return noise_sampler.sample(batch_size)
        shape = (
            batch_size,
            len(self.profile.channels),
            self.model_shape[0],
            self.model_shape[1],
        )
        return torch.randn(shape, device=self.torch_device, generator=generator)

    def _reverse_diffusion(
        self,
        x_current: torch.Tensor,
        scheduler_config: dict[str, object],
        sigma_max: float,
    ) -> torch.Tensor:
        sigma_steps = _daps_sigma_steps(scheduler_config, sigma_max=sigma_max)
        x_next = x_current
        class_labels = None
        with torch.no_grad():
            for index, (sigma_cur_value, sigma_next_value) in enumerate(
                zip(sigma_steps[:-1], sigma_steps[1:])
            ):
                sigma_cur = torch.as_tensor(
                    sigma_cur_value,
                    dtype=torch.float64,
                    device=self.torch_device,
                )
                sigma_next = torch.as_tensor(
                    sigma_next_value,
                    dtype=torch.float64,
                    device=self.torch_device,
                )
                sigma_cur = self.net.round_sigma(sigma_cur)
                x_denoised = self.net(x_next, sigma_cur, class_labels=class_labels).to(
                    torch.float64
                )
                d_cur = (x_next - x_denoised) / sigma_cur
                proposal = x_next + (sigma_next - sigma_cur) * d_cur

                if index < len(sigma_steps) - 2:
                    sigma_next_rounded = self.net.round_sigma(sigma_next)
                    x_denoised_next = self.net(
                        proposal,
                        sigma_next_rounded,
                        class_labels=class_labels,
                    ).to(torch.float64)
                    d_prime = (proposal - x_denoised_next) / sigma_next_rounded
                    proposal = x_next + (sigma_next_rounded - sigma_cur) * (
                        0.5 * d_cur + 0.5 * d_prime
                    )
                x_next = proposal
        return x_next

    @staticmethod
    def _langevin_lr(config: dict[str, object], step_ratio: float) -> float:
        base_lr = float(config.get("lr", 1.0e-4))
        lr_min_ratio = float(config.get("lr_min_ratio", 0.01))
        lr_rho = float(config.get("lr_rho", 1.0))
        if base_lr <= 0 or lr_min_ratio <= 0 or lr_rho <= 0:
            raise ValueError("DDIS Langevin lr, lr_min_ratio, and lr_rho must be positive")
        multiplier = (
            1.0 ** (1.0 / lr_rho)
            + step_ratio * (lr_min_ratio ** (1.0 / lr_rho) - 1.0 ** (1.0 / lr_rho))
        ) ** lr_rho
        return base_lr * multiplier

    def _langevin_dynamics(
        self,
        x0hat: torch.Tensor,
        prepared: dict[str, object],
        sigma: float,
        annealing_step: int,
        annealing_steps: int,
        config: dict[str, object],
        generator: torch.Generator,
        noise_sampler: RBFKernel | None,
    ) -> torch.Tensor:
        num_steps = int(config.get("num_steps", 20))
        if num_steps < 0:
            raise ValueError("DDIS Langevin num_steps must be nonnegative")
        if num_steps == 0:
            return x0hat.detach()
        tau = float(config.get("tau", 0.001))
        eta = float(config.get("eta", 0.1))
        if tau <= 0:
            raise ValueError("DDIS Langevin tau must be positive")
        ratio = annealing_step / max(annealing_steps, 1)
        current_lr = self._langevin_lr(config, ratio)
        sigma_scale = max(float(sigma), 1.0e-12)

        x0_reference = x0hat.detach()
        x = x0_reference.clone()
        for _ in range(num_steps):
            x = x.detach().requires_grad_(True)
            prior_loss = (x - x0_reference).square().sum()
            loss = prior_loss / (2.0 * sigma_scale**2)
            if self.observation_weight != 0:
                observed_prediction = self._surrogate_prediction(x)
                obs_loss = self._field_loss(observed_prediction, prepared)
                loss = loss + self.observation_weight * obs_loss / (2.0 * tau**2)

            grad = torch.autograd.grad(loss, x)[0]
            with torch.no_grad():
                x = x - current_lr * grad
                if eta != 0:
                    noise = self._sample_noise(
                        x.shape[0],
                        generator,
                        noise_sampler,
                    ).to(device=x.device, dtype=x.dtype)
                    x = x + noise * math.sqrt(2.0 * current_lr) * eta
            if not torch.isfinite(x).all():
                raise FloatingPointError("DDIS Langevin dynamics produced non-finite values")
        return x.detach()

    def _sample_batch(
        self,
        case: BenchmarkCase,
        batch_size: int,
        generator: torch.Generator,
        prepared: dict[str, object],
    ) -> torch.Tensor:
        del case
        annealing_config = dict(
            self.sampling.get(
                "annealing",
                {
                    "num_steps": 100,
                    "sigma_max": 10.0,
                    "sigma_min": 0.01,
                    "sigma_final": 0.0,
                    "rho": 7.0,
                },
            )
        )
        diffusion_config = dict(
            self.sampling.get(
                "diffusion",
                {
                    "num_steps": 5,
                    "sigma_min": 0.001,
                    "sigma_final": 0.0,
                    "rho": 7.0,
                },
            )
        )
        langevin_config = dict(
            self.sampling.get(
                "langevin",
                {
                    "num_steps": 20,
                    "lr": 1.0e-4,
                    "lr_min_ratio": 0.01,
                    "lr_rho": 1.0,
                    "eta": 0.1,
                    "tau": 0.001,
                },
            )
        )

        noise_sampler = self._make_noise_sampler()
        sigma_steps = _daps_sigma_steps(annealing_config)
        x_next = (
            self._sample_noise(batch_size, generator, noise_sampler).to(torch.float64)
            * sigma_steps[0]
        )
        annealing_steps = len(sigma_steps) - 1

        for step, (sigma_cur, sigma_next) in enumerate(
            zip(sigma_steps[:-1], sigma_steps[1:])
        ):
            x0hat = self._reverse_diffusion(
                x_next,
                diffusion_config,
                sigma_max=float(sigma_cur),
            )
            x0_guided = self._langevin_dynamics(
                x0hat,
                prepared,
                sigma=float(sigma_cur),
                annealing_step=step,
                annealing_steps=annealing_steps,
                config=langevin_config,
                generator=generator,
                noise_sampler=noise_sampler,
            )
            noise = self._sample_noise(batch_size, generator, noise_sampler).to(
                device=self.torch_device,
                dtype=torch.float64,
            )
            x_next = x0_guided + noise * float(sigma_next)

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
