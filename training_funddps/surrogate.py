from __future__ import annotations

from dataclasses import dataclass

import torch

from posteriorbench.spatial import as_spatial_shape, spatial_to_config


@dataclass(frozen=True)
class SurrogateSpec:
    model: str
    implementation: str
    in_channels: int
    out_channels: int
    resolution: int | list[int]
    n_params: int


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def create_surrogate_model(
    model_name: str,
    *,
    in_channels: int,
    out_channels: int,
    resolution: object,
    config: dict[str, object],
) -> tuple[torch.nn.Module, SurrogateSpec]:
    name = str(model_name).lower()
    if name == "localfno":
        name = "localno"
    spatial_shape = as_spatial_shape(resolution)

    try:
        import neuralop.models as neuralop_models
    except ImportError as error:
        raise ImportError(
            "Fun-DDPS surrogate training requires neuralop. Install the "
            "project environment with neuralop before running this entry point."
        ) from error

    if name == "fno":
        model = neuralop_models.FNO(
            n_modes=tuple(config.get("n_modes", (16, 16))),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 48)),
            projection_channel_ratio=int(config.get("projection_channel_ratio", 1)),
            n_layers=int(config.get("n_layers", 4)),
            domain_padding=config.get("domain_padding", 0.1),
            domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
        )
        implementation = type(model).__name__
    elif name == "uno":
        model = neuralop_models.UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 96)),
            projection_channels=int(config.get("projection_channels", 96)),
            n_layers=int(config.get("n_layers", 4)),
            domain_padding=config.get("domain_padding", 0.1),
        )
        implementation = type(model).__name__
    elif name == "localno":
        local_no_class = getattr(
            neuralop_models,
            "LocalNO",
            getattr(neuralop_models, "LocalFNO", None),
        )
        if local_no_class is None:
            raise ImportError(
                "The installed neuralop package exposes neither LocalNO nor "
                "LocalFNO. Use model='fno' or model='uno', or install an "
                "official neuralop version that provides a local NO model."
            )
        default_in_shape = config.get("default_in_shape", spatial_shape)
        model = local_no_class(
            n_modes=tuple(config.get("n_modes", (16, 16))),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(config.get("hidden_channels", 48)),
            default_in_shape=tuple(int(value) for value in default_in_shape),
            n_layers=int(config.get("n_layers", 4)),
            disco_layers=config.get("disco_layers", True),
            diff_layers=config.get("diff_layers", False),
            disco_kernel_shape=list(config.get("disco_kernel_shape", [4, 4])),
            domain_length=list(config.get("domain_length", [1.0, 1.0])),
            lifting_channel_ratio=int(config.get("lifting_channel_ratio", 2)),
            projection_channel_ratio=int(config.get("projection_channel_ratio", 2)),
            domain_padding=config.get("domain_padding", None),
            domain_padding_mode=str(config.get("domain_padding_mode", "one-sided")),
        )
        implementation = type(model).__name__
    else:
        raise ValueError(
            "Fun-DDPS surrogate model must be one of: fno, uno, localno/localfno"
        )

    spec = SurrogateSpec(
        model=name,
        implementation=implementation,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        resolution=spatial_to_config(spatial_shape),
        n_params=count_parameters(model),
    )
    return model, spec


def relative_l2_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    batch_size = prediction.shape[0]
    pred_flat = prediction.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    diff_norm = torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target_flat, ord=2, dim=1)
    return (diff_norm / (target_norm + eps)).mean()
