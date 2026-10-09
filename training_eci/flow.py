from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from posteriorbench.spatial import as_spatial_shape, spatial_to_config


@dataclass(frozen=True)
class ECIModelSpec:
    backbone: str
    channels: int
    resolution: int | list[int]
    n_modes: tuple[int, int]
    emb_channels: int
    hidden_channels: int
    n_layers: int
    lifting_channel_ratio: int
    projection_channel_ratio: int
    domain_padding: float | int | None
    domain_padding_mode: str
    base_noise: str
    kernel_length: float
    kernel_variance: float
    n_params: int


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def clean_eci_state_dict(state_dict: object) -> dict[str, object]:
    if not isinstance(state_dict, Mapping):
        raise ValueError("ECI model_state_dict must be a mapping")
    return {str(key): value for key, value in state_dict.items() if str(key) != "_metadata"}


def get_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_positions: int = 2000,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps.reshape(1)
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}")
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.new_zeros((timesteps.shape[0], 0))
    scale = np.log(max_positions) / max(half_dim - 1, 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -scale
    )
    embedding = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1), mode="constant")
    return embedding


class ECIFNOVectorField(nn.Module):
    """FNO vector field for functional flow matching.

    The neural-operator backbone is the official `neuralop.models.FNO`. This
    wrapper only adds the scalar flow time embedding and keeps the NO layers in
    the library implementation.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_modes: tuple[int, int] = (32, 32),
        emb_channels: int = 32,
        hidden_channels: int = 64,
        n_layers: int = 4,
        lifting_channel_ratio: int = 4,
        projection_channel_ratio: int = 4,
        domain_padding: float | int | None = 0.1,
        domain_padding_mode: str = "one-sided",
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if emb_channels < 0:
            raise ValueError("emb_channels must be nonnegative")
        try:
            from neuralop.models import FNO
        except ImportError as error:
            raise ImportError(
                "ECI training/inference requires neuralop. Install the project "
                "environment with neuraloperator before running ECI."
            ) from error

        self.channels = int(channels)
        self.emb_channels = int(emb_channels)
        self.model = FNO(
            n_modes=tuple(int(mode) for mode in n_modes),
            in_channels=self.channels + self.emb_channels,
            out_channels=self.channels,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
            lifting_channel_ratio=int(lifting_channel_ratio),
            projection_channel_ratio=int(projection_channel_ratio),
            positional_embedding="grid",
            domain_padding=domain_padding,
            domain_padding_mode=str(domain_padding_mode),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ECI vector field expects [B,C,H,W], got {tuple(x.shape)}")
        batch_size, _, height, width = x.shape
        if t.ndim == 0 or t.numel() == 1:
            t = torch.full((batch_size,), float(t.item()), device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype).reshape(-1)
        if t.shape[0] != batch_size:
            raise ValueError(f"t has batch {t.shape[0]}, expected {batch_size}")
        if self.emb_channels:
            t_emb = get_time_embedding(t, self.emb_channels)
            t_emb = t_emb.to(dtype=x.dtype).reshape(batch_size, -1, 1, 1)
            t_emb = t_emb.expand(-1, -1, height, width)
            x = torch.cat([x, t_emb], dim=1)
        return self.model(x)


def make_grid(
    dims: tuple[int, ...],
    device: torch.device,
    start: float | tuple[float, ...] = 0.0,
    end: float | tuple[float, ...] = 1.0,
) -> torch.Tensor:
    ndim = len(dims)
    if ndim <= 0:
        raise ValueError("GP prior dimensions must be non-empty")
    if not isinstance(start, (tuple, list)):
        start = (float(start),) * ndim
    if not isinstance(end, (tuple, list)):
        end = (float(end),) * ndim
    if len(start) != ndim or len(end) != ndim:
        raise ValueError("GP prior start/end must match the number of dimensions")
    if ndim == 1:
        return torch.linspace(start[0], end[0], dims[0], dtype=torch.float32, device=device).unsqueeze(-1)
    axes = [
        torch.linspace(start[idx], end[idx], dims[idx], dtype=torch.float32, device=device)
        for idx in range(ndim)
    ]
    mesh = torch.meshgrid(axes, indexing="ij")
    return torch.stack(mesh, dim=-1).view(-1, ndim)


class GPPrior:
    """Official ECI GP prior wrapper, adapted to PosteriorBench tensors."""

    def __init__(
        self,
        kernel: str | None = "matern",
        *,
        lengthscale: float | None = 0.001,
        variance: float | None = 1.0,
    ):
        self.kernel = "matern" if kernel is None else str(kernel).lower()
        if self.kernel == "gaussian":
            self.kernel = "randn"
        self.lengthscale = 0.001 if lengthscale is None else float(lengthscale)
        self.variance = 1.0 if variance is None else float(variance)
        if self.lengthscale <= 0:
            raise ValueError("ECI GP kernel_length must be positive")
        if self.variance <= 0:
            raise ValueError("ECI GP kernel_variance must be positive")
        if self.kernel not in {"matern", "randn", "rand"}:
            raise ValueError(f"Unknown ECI base_noise/kernel: {kernel}")
        self._gp = None

    def _build_gp(self, device: torch.device):
        try:
            import gpytorch
        except ImportError as error:
            raise ImportError(
                "Canonical ECI GP base noise requires gpytorch. Install the "
                "project requirements before running ECI training or inference."
            ) from error

        class _ExactPrior(gpytorch.models.ExactGP):
            def __init__(self, lengthscale: float, variance: float):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                super().__init__(None, None, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, eps=1e-10)
                base_kernel.lengthscale = torch.as_tensor([lengthscale], device=device)
                self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
                self.covar_module.outputscale = torch.as_tensor([variance], device=device)

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x),
                    self.covar_module(x),
                )

        gp = _ExactPrior(self.lengthscale, self.variance).to(device)
        gp.eval()
        return gp

    def sample(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch_size, *dims = shape
        if self.kernel == "randn":
            return torch.randn(shape, device=device, dtype=dtype, generator=generator)
        if self.kernel == "rand":
            return torch.rand(shape, device=device, dtype=dtype, generator=generator)

        if len(dims) == 3:
            channels, height, width = (int(dim) for dim in dims)
            spatial_dims = (height, width)
            n_samples = batch_size * channels
        else:
            spatial_dims = tuple(int(dim) for dim in dims)
            n_samples = batch_size
        grid = make_grid(spatial_dims, device=device)
        if self._gp is None or next(self._gp.parameters()).device != device:
            self._gp = self._build_gp(device)
        distribution = self._gp(grid)
        samples = distribution.sample(sample_shape=torch.Size([n_samples]))
        return samples.reshape(batch_size, *dims).to(dtype=dtype)


class ECINoiseSampler:
    def __init__(
        self,
        mode: str = "matern",
        *,
        kernel_length: float = 0.001,
        kernel_variance: float = 1.0,
    ):
        self.prior = GPPrior(
            kernel=mode,
            lengthscale=kernel_length,
            variance=kernel_variance,
        )

    def sample(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.prior.sample(shape, device=device, dtype=dtype, generator=generator)


class ECIFlowModel(nn.Module):
    def __init__(
        self,
        vector_field: nn.Module,
        *,
        base_noise: str = "matern",
        kernel_length: float = 0.001,
        kernel_variance: float = 1.0,
    ):
        super().__init__()
        self.vector_field = vector_field
        self.noise_sampler = ECINoiseSampler(
            base_noise,
            kernel_length=kernel_length,
            kernel_variance=kernel_variance,
        )

    def sample_base_noise(
        self,
        shape: tuple[int, int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.noise_sampler.sample(shape, device=device, dtype=dtype, generator=generator)

    def get_loss(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ECI training expects [B,C,H,W], got {tuple(x.shape)}")
        batch_size = x.shape[0]
        noise = self.sample_base_noise(
            tuple(x.shape),
            device=x.device,
            dtype=x.dtype,
        )
        t = torch.rand(batch_size, dtype=x.dtype, device=x.device)
        t_view = t.reshape(batch_size, 1, 1, 1)
        xt = (1 - t_view) * noise + t_view * x
        target_vector = x - noise
        predicted_vector = self.vector_field(t, xt)
        return (predicted_vector - target_vector).square().mean()

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        channels: int,
        resolution: object,
        n_step: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        height, width = as_spatial_shape(resolution)
        x = self.sample_base_noise(
            (batch_size, channels, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        for step in range(n_step):
            t = torch.as_tensor(step / n_step, device=device, dtype=dtype)
            x = x + self.vector_field(t, x) / n_step
        return x

    @torch.no_grad()
    def eci_sample(
        self,
        *,
        batch_size: int,
        channels: int,
        resolution: object,
        n_step: int,
        n_mix: int,
        resample_step: int | None,
        constraint: Any,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        if n_mix <= 0:
            raise ValueError("n_mix must be positive")
        height, width = as_spatial_shape(resolution)
        shape = (batch_size, channels, height, width)
        noise = self.sample_base_noise(shape, device=device, dtype=dtype, generator=generator)
        x = noise
        dt = 1.0 / n_step
        counter = 0
        if resample_step is None or int(resample_step) == 0:
            resample_interval = n_step * n_mix + 1
        else:
            resample_interval = int(resample_step)
            if resample_interval < 0:
                raise ValueError("resample_step must be nonnegative or null")

        for step in range(n_step):
            t_value = step / n_step
            t = torch.as_tensor(t_value, device=device, dtype=dtype)
            for mix in range(n_mix):
                counter += 1
                if counter % resample_interval == 0:
                    noise = self.sample_base_noise(
                        shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                vector = self.vector_field(t, x)
                x1 = x + vector * (1 - t)
                x1 = constraint.adjust(x1)
                next_t = t_value if mix < n_mix - 1 else t_value + dt
                x = x1 * next_t + noise * (1 - next_t)
        return x


def create_eci_model(
    *,
    channels: int,
    resolution: object,
    config: dict[str, Any],
) -> tuple[ECIFlowModel, ECIModelSpec]:
    backbone = str(config.get("backbone", "fno")).lower()
    if backbone != "fno":
        raise ValueError("ECI canonical implementation supports backbone='fno'")
    n_modes = tuple(int(mode) for mode in config.get("n_modes", (32, 32)))
    if len(n_modes) != 2:
        raise ValueError("ECI currently expects 2D FNO n_modes")
    spatial_shape = as_spatial_shape(resolution)
    vector_field = ECIFNOVectorField(
        channels=channels,
        n_modes=n_modes,
        emb_channels=int(config.get("emb_channels", 32)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        n_layers=int(config.get("n_layers", 4)),
        lifting_channel_ratio=int(config.get("lifting_channel_ratio", 4)),
        projection_channel_ratio=int(config.get("projection_channel_ratio", 4)),
        domain_padding=config.get("domain_padding", 0.1),
        domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
    )
    model = ECIFlowModel(
        vector_field,
        base_noise=str(config.get("base_noise", "matern")),
        kernel_length=float(config.get("kernel_length", 0.001)),
        kernel_variance=float(config.get("kernel_variance", 1.0)),
    )
    spec = ECIModelSpec(
        backbone=backbone,
        channels=int(channels),
        resolution=spatial_to_config(spatial_shape),
        n_modes=n_modes,
        emb_channels=int(config.get("emb_channels", 32)),
        hidden_channels=int(config.get("hidden_channels", 64)),
        n_layers=int(config.get("n_layers", 4)),
        lifting_channel_ratio=int(config.get("lifting_channel_ratio", 4)),
        projection_channel_ratio=int(config.get("projection_channel_ratio", 4)),
        domain_padding=config.get("domain_padding", 0.1),
        domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
        base_noise=str(config.get("base_noise", "matern")),
        kernel_length=float(config.get("kernel_length", 0.001)),
        kernel_variance=float(config.get("kernel_variance", 1.0)),
        n_params=count_parameters(model),
    )
    return model, spec
