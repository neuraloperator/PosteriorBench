from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from posteriorbench.spatial import as_spatial_shape, spatial_to_config


@dataclass(frozen=True)
class MCDropoutSpec:
    model: str
    implementation: str
    in_channels: int
    out_channels: int
    resolution: int | list[int]
    n_modes: tuple[int, int]
    hidden_channels: int
    n_layers: int
    channel_mlp_dropout: float
    n_params: int


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def create_mcdropout_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    resolution: object,
    config: dict[str, Any],
) -> tuple[torch.nn.Module, MCDropoutSpec]:
    name = str(model_name).lower()
    if name != "fno":
        raise ValueError("MC-dropout baseline currently supports only model='fno'")
    try:
        from neuralop.models import FNO
    except ImportError as error:
        raise ImportError(
            "MC-dropout FNO training requires neuralop. Install the project "
            "environment with neuralop before running this entry point."
        ) from error

    spatial_shape = as_spatial_shape(resolution)
    n_modes = tuple(int(mode) for mode in config.get("n_modes", (32, 32)))
    if len(n_modes) != 2:
        raise ValueError("MC-dropout FNO expects 2D n_modes")
    dropout = float(config.get("channel_mlp_dropout", config.get("dropout", 0.1)))
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("channel_mlp_dropout must be in [0, 1)")

    model = FNO(
        n_modes=n_modes,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        hidden_channels=int(config.get("hidden_channels", 128)),
        n_layers=int(config.get("n_layers", 6)),
        lifting_channel_ratio=int(config.get("lifting_channel_ratio", 2)),
        projection_channel_ratio=int(config.get("projection_channel_ratio", 1)),
        positional_embedding=str(config.get("positional_embedding", "grid")),
        channel_mlp_dropout=dropout,
        channel_mlp_expansion=float(config.get("channel_mlp_expansion", 0.5)),
        channel_mlp_skip=str(config.get("channel_mlp_skip", "soft-gating")),
        fno_skip=str(config.get("fno_skip", "linear")),
        domain_padding=config.get("domain_padding", 0.1),
        domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
    )
    spec = MCDropoutSpec(
        model=name,
        implementation=type(model).__name__,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        resolution=spatial_to_config(spatial_shape),
        n_modes=n_modes,
        hidden_channels=int(config.get("hidden_channels", 128)),
        n_layers=int(config.get("n_layers", 6)),
        channel_mlp_dropout=dropout,
        n_params=count_parameters(model),
    )
    return model, spec


def set_mc_dropout_mode(model: torch.nn.Module) -> None:
    """Evaluate all layers except dropout, which remains stochastic."""

    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def normalized_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)


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
