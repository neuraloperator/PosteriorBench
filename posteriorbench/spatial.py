from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np


SpatialShape = tuple[int, int]


def as_spatial_shape(value: object, *, name: str = "resolution") -> SpatialShape:
    """Normalize an int or ``[height, width]`` value into a spatial shape."""

    if isinstance(value, Integral):
        size = int(value)
        if size <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return size, size
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or [height, width], got {value!r}")
        height, width = (int(item) for item in value)
        if height <= 0 or width <= 0:
            raise ValueError(f"{name} values must be positive, got {value!r}")
        return height, width
    raise TypeError(f"{name} must be an int or [height, width], got {type(value).__name__}")


def spatial_to_config(shape: SpatialShape) -> int | list[int]:
    height, width = as_spatial_shape(shape, name="shape")
    return height if height == width else [height, width]


def nominal_resolution(value: object) -> int:
    """Return the height used by legacy square-only network constructors."""

    return as_spatial_shape(value)[0]


def scale_sensor_coords(
    coords: np.ndarray,
    source_shape: SpatialShape,
    target_shape: SpatialShape,
    *,
    unique: bool = False,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Sensor coordinates must have shape [N,2], got {coords.shape}")
    source_h, source_w = as_spatial_shape(source_shape, name="source_shape")
    target_h, target_w = as_spatial_shape(target_shape, name="target_shape")
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Invalid source shape {(source_h, source_w)}")

    if (source_h, source_w) == (target_h, target_w):
        scaled = coords.copy()
    else:
        scaled = np.empty_like(coords)
        scaled[:, 0] = np.floor(coords[:, 0] * (target_h / source_h)).astype(np.int64)
        scaled[:, 1] = np.floor(coords[:, 1] * (target_w / source_w)).astype(np.int64)
    scaled[:, 0] = np.clip(scaled[:, 0], 0, target_h - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, target_w - 1)
    return np.unique(scaled, axis=0) if unique else scaled
