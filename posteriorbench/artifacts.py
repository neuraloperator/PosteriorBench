from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NormalizationProfile:
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: float


@dataclass(frozen=True)
class CheckpointProfile:
    path: Path
    method: str
    dataset: str
    channels: tuple[str, ...]
    resolution: int | list[int] | tuple[int, int] | None
    normalization: NormalizationProfile
    sampling: dict[str, Any]
    guidance: dict[str, Any]
    postprocess: dict[str, Any]


def load_checkpoint_profile(path: str | Path) -> CheckpointProfile:
    profile_path = Path(path)
    raw = yaml.safe_load(profile_path.read_text())

    for key in ("method", "dataset", "model", "normalization"):
        if key not in raw:
            raise ValueError(f"Checkpoint profile is missing required key '{key}': {profile_path}")

    model = raw["model"]
    channels = tuple(model["channels"])
    normalization = raw["normalization"]
    norm_channels = tuple(normalization.get("channels", channels))
    mean = tuple(float(x) for x in normalization["mean"])
    std = tuple(float(x) for x in normalization["std"])

    if norm_channels != channels:
        raise ValueError(
            f"Normalization channels {norm_channels} do not match model channels {channels}"
        )
    if len(mean) != len(channels) or len(std) != len(channels):
        raise ValueError("Normalization mean/std must have one value per model channel")
    if any(value <= 0 for value in std):
        raise ValueError("Normalization std values must be positive")

    return CheckpointProfile(
        path=profile_path,
        method=str(raw["method"]).lower(),
        dataset=str(raw["dataset"]).lower(),
        channels=channels,
        resolution=model.get("resolution"),
        normalization=NormalizationProfile(
            channels=norm_channels,
            mean=mean,
            std=std,
            scale=float(normalization.get("scale", 1.0)),
        ),
        sampling=dict(raw.get("sampling", {})),
        guidance=dict(raw.get("guidance", {})),
        postprocess=dict(raw.get("postprocess", {})),
    )
