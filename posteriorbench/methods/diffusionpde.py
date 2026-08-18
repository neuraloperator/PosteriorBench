from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from posteriorbench.datasets.base import BenchmarkCase
from posteriorbench.compat import load_pickle_with_torch_storage_map
from posteriorbench.methods.base import MethodAdapter
from posteriorbench.normalization import ModelNormalizer
from posteriorbench.spatial import as_spatial_shape, scale_sensor_coords


def _as_field_tensor(
    field: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.from_numpy(field).unsqueeze(0).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
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
        raise ValueError("DiffusionPDE sampling requires at least two steps")
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return torch.cat([net.round_sigma(steps), torch.zeros_like(steps[:1])])


def _darcy_pde_loss(physical: torch.Tensor) -> torch.Tensor:
    a = physical[:, 0:1].to(torch.float64)
    u = physical[:, 1:2].to(torch.float64)
    device = physical.device
    deriv_x = torch.tensor(
        [[-1, 0, 1]], dtype=torch.float64, device=device
    ).view(1, 1, 1, 3) / 2
    deriv_y = torch.tensor(
        [[-1], [0], [1]], dtype=torch.float64, device=device
    ).view(1, 1, 3, 1) / 2
    grad_x = F.conv2d(u, deriv_x, padding=(0, 1))
    grad_y = F.conv2d(u, deriv_y, padding=(1, 0))
    residual = (
        F.conv2d(a * grad_x, deriv_x, padding=(0, 1))
        + F.conv2d(a * grad_y, deriv_y, padding=(1, 0))
        + 1
    )
    return torch.linalg.vector_norm(residual.flatten(1), dim=1).mean() / (u.shape[-1] * u.shape[-2])


def _poisson_pde_loss(physical: torch.Tensor) -> torch.Tensor:
    source = physical[:, 0:1].to(torch.float64)
    solution = physical[:, 1:2].to(torch.float64)
    size = solution.shape[-1]
    h = 1 / (size - 1)
    padded = F.pad(solution, (1, 1, 1, 1), mode="constant", value=0)
    laplacian = (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4 * solution
    ) / h**2
    residual = laplacian - source
    residual = residual.clone()
    residual[:, :, 0, :] = 0
    residual[:, :, -1, :] = 0
    residual[:, :, :, 0] = 0
    residual[:, :, :, -1] = 0
    return torch.linalg.vector_norm(residual.flatten(1), dim=1).mean() / ((size - 1) * (size - 1))


class DiffusionPDEAdapter(MethodAdapter):
    def load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        if self.profile.method != "diffusionpde":
            raise ValueError(
                f"DiffusionPDE adapter cannot use method profile '{self.profile.method}'"
            )
        if self.profile.dataset != self.dataset.name:
            raise ValueError(
                f"Profile dataset '{self.profile.dataset}' does not match '{self.dataset.name}'"
            )
        self.dataset.validate_model_channels(self.profile.channels)
        if bool(self.profile.guidance.get("pending", False)):
            raise ValueError(
                f"Guidance is not finalized for dataset '{self.dataset.name}'; "
                "run only --dry-run until a real checkpoint quick scan fixes it"
            )

        self.torch_device = torch.device(self.device)
        checkpoint = load_pickle_with_torch_storage_map(
            self.checkpoint,
            map_location=torch.device("cpu"),
        )
        if "ema" not in checkpoint:
            raise KeyError(f"DiffusionPDE checkpoint {self.checkpoint} has no 'ema' network")
        self.net = checkpoint["ema"].to(self.torch_device)
        self.net.eval()

        self.model_shape = as_spatial_shape(self.profile.resolution or self.net.img_resolution)
        checkpoint_shape = as_spatial_shape(self.net.img_resolution)
        if checkpoint_shape != self.model_shape:
            raise ValueError(
                "DiffusionPDE checkpoints are fixed-resolution; "
                f"checkpoint={checkpoint_shape}, profile={self.model_shape}"
            )
        if int(self.net.img_channels) != len(self.profile.channels):
            raise ValueError(
                f"Checkpoint has {self.net.img_channels} channels, profile has "
                f"{len(self.profile.channels)}"
            )

        discrete_values = self.profile.postprocess.get("discrete_values", {})
        self.normalizer = ModelNormalizer(
            self.profile.normalization,
            discrete_values=discrete_values,
        )
        self.sampling = self.profile.sampling
        raw_init_latents = self.sampling.get("init_latents", "gaussian")
        if not isinstance(raw_init_latents, str):
            raise TypeError("DiffusionPDE sampling.init_latents must be a string")
        self.init_latents = raw_init_latents.strip().lower()
        if self.init_latents != "gaussian":
            raise ValueError(
                "DiffusionPDE currently supports sampling.init_latents='gaussian' only, "
                f"got {raw_init_latents!r}"
            )
        self.guidance = self.profile.guidance
        self.observed_channel = self.profile.channels.index(self.dataset.observed_field)
        field_weights = self.guidance.get("field_weights", {})
        unexpected_weights = {
            channel: float(weight)
            for channel, weight in field_weights.items()
            if channel != self.dataset.observed_field and float(weight) != 0
        }
        if unexpected_weights:
            raise ValueError(
                "DiffusionPDE benchmark conditioning may only weight the observed "
                f"field '{self.dataset.observed_field}', got {unexpected_weights}"
            )
        self.observation_weight = float(field_weights.get(self.dataset.observed_field, 0.0))
        self.pde_weight = float(self.guidance.get("pde_weight", 0.0))
        if self.dataset.name == "light_transport" and self.pde_weight != 0:
            raise ValueError("Light Transport DiffusionPDE must use pde_weight=0")
        self.loss_type = str(self.guidance.get("loss_type", "l2"))
        self.normalize_observation = bool(self.guidance.get("normalize", False))
        schedule = self.guidance.get("schedule", {})
        self.observation_fraction = float(schedule.get("observation_fraction", 0.8))
        self.late_observation_scale = float(schedule.get("late_observation_scale", 0.1))

    def _prepare_case_observation(self, case: BenchmarkCase) -> dict[str, torch.Tensor | tuple[int, int]]:
        observed = _as_field_tensor(case.fields[case.observed_field], self.torch_device)
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
            mask = torch.zeros_like(observed)
            index = torch.as_tensor(coords, device=self.torch_device, dtype=torch.long)
            mask[0, 0, index[:, 0], index[:, 1]] = 1
            if self.normalize_observation:
                observed = self.normalizer.normalize_channel(observed, case.observed_field)
            return {"observed": observed, "mask": mask}
        if case.observation_operator == "area_average":
            if self.normalize_observation:
                observed = self.normalizer.normalize_channel(observed, case.observed_field)
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
        raise ValueError(f"Unsupported DiffusionPDE loss_type '{self.loss_type}'")

    def _observation_loss(
        self,
        physical: torch.Tensor,
        case: BenchmarkCase,
        prepared: dict[str, object],
    ) -> torch.Tensor:
        prediction = physical[:, self.observed_channel : self.observed_channel + 1]
        if self.normalize_observation:
            prediction = self.normalizer.normalize_channel(prediction, case.observed_field)
        return self._field_loss(prediction, prepared)

    def _pde_loss(self, physical: torch.Tensor) -> torch.Tensor:
        if self.pde_weight == 0:
            return physical.sum() * 0
        if self.dataset.name == "darcy":
            return _darcy_pde_loss(physical)
        if self.dataset.name == "poisson":
            return _poisson_pde_loss(physical)
        return physical.sum() * 0

    def _sample_batch(
        self,
        case: BenchmarkCase,
        prepared: dict[str, object],
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        iterations = int(self.sampling.get("iterations", 2000))
        sigma_steps = _edm_sigma_steps(
            self.net,
            iterations,
            float(self.sampling.get("sigma_min", 0.002)),
            float(self.sampling.get("sigma_max", 80.0)),
            float(self.sampling.get("rho", 7.0)),
            self.torch_device,
        )
        latents = torch.randn(
            [batch_size, len(self.profile.channels), *self.model_shape],
            device=self.torch_device,
            generator=generator,
        )
        x_next = latents * sigma_steps[0]
        class_labels = None

        for index, (sigma_cur, sigma_next) in enumerate(zip(sigma_steps[:-1], sigma_steps[1:])):
            x_cur = x_next.detach().clone().requires_grad_(True)
            sigma_cur = self.net.round_sigma(sigma_cur)
            denoised = self.net(x_cur, sigma_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / sigma_cur
            proposal = x_cur + (sigma_next - sigma_cur) * d_cur

            guided_denoised = denoised
            if index < iterations - 1:
                sigma_next = self.net.round_sigma(sigma_next)
                guided_denoised = self.net(proposal, sigma_next, class_labels=class_labels)
                d_prime = (proposal - guided_denoised) / sigma_next
                proposal = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_prime)

            physical = self.normalizer.denormalize(guided_denoised)
            obs_loss = self._observation_loss(physical, case, prepared)
            pde_loss = self._pde_loss(physical)
            if index <= self.observation_fraction * iterations:
                total_loss = self.observation_weight * obs_loss
            else:
                total_loss = (
                    self.late_observation_scale * self.observation_weight * obs_loss
                    + self.pde_weight * pde_loss
                )
            if torch.is_tensor(total_loss) and bool(total_loss.requires_grad):
                grad = torch.autograd.grad(total_loss, x_cur)[0]
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
            prediction = self._sample_batch(case, prepared, current_batch, generator)
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
