from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from posteriorbench.spatial import as_spatial_shape, spatial_to_config


@dataclass(frozen=True)
class DDISSurrogateSpec:
    model: str
    implementation: str
    in_channels: int
    out_channels: int
    resolution: int | list[int]
    n_params: int


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


class FNOPad(torch.nn.Module):
    """Official DDIS FNO_pad: reflect-pad, apply FNO, then crop."""

    def __init__(
        self,
        *,
        n_modes: tuple[int, int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        from neuralop.models.fno import FNO

        self.model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_y = x.shape[-2] // 2
        pad_x = x.shape[-1] // 2
        x_pad = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        y_pad = self.model(x_pad)
        return y_pad[..., pad_y:-pad_y, pad_x:-pad_x]


def create_ddis_surrogate_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    resolution: object,
    config: dict[str, object],
) -> tuple[torch.nn.Module, DDISSurrogateSpec]:
    name = str(model_name).lower()
    spatial_shape = as_spatial_shape(resolution)
    try:
        import neuralop.models as neuralop_models
    except ImportError as error:
        raise ImportError(
            "DDIS surrogate training requires neuralop. Install the project "
            "environment with neuralop before running this entry point."
        ) from error

    if name == "fno_pad":
        model = FNOPad(
            n_modes=tuple(config.get("n_modes", (64, 64))),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 64)),
            n_layers=int(config.get("n_layers", 4)),
        )
        implementation = type(model).__name__
    elif name == "fno":
        model = neuralop_models.FNO(
            n_modes=tuple(config.get("n_modes", (64, 64))),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 64)),
            n_layers=int(config.get("n_layers", 4)),
            domain_padding=config.get("domain_padding", None),
            domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
        )
        implementation = type(model).__name__
    else:
        raise ValueError("DDIS surrogate model must be one of: fno_pad, fno")

    spec = DDISSurrogateSpec(
        model=name,
        implementation=implementation,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        resolution=spatial_to_config(spatial_shape),
        n_params=count_parameters(model),
    )
    return model, spec


def relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    batch_size = prediction.shape[0]
    pred_flat = prediction.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1)
    per_sample = diff_norm / (target_norm + eps)
    reduction = str(reduction).lower()
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError("relative_l2_loss reduction must be one of: mean, sum, none")
