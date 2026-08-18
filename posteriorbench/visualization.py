from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from posteriorbench.datasets.base import BenchmarkCase, ReferencePosterior
from posteriorbench.evaluation.metrics import normalize_weights


DEFAULT_VISUALIZATION_SEED = 0


def weighted_sample_indices(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = normalize_weights(weights)
    if count <= 0:
        raise ValueError("count must be positive")
    if len(weights) <= count:
        return np.arange(len(weights), dtype=np.int64)

    positive = int(np.count_nonzero(weights > 0))
    if positive < count:
        raise ValueError(
            f"Cannot draw {count} distinct weighted samples from only "
            f"{positive} positive-weight samples"
        )
    return rng.choice(len(weights), size=count, replace=False, p=weights)


def _plot_style(
    case: BenchmarkCase,
    field: str,
    displayed: list[np.ndarray],
) -> tuple[object, object | None, float | None, float | None]:
    values = np.concatenate([array.ravel() for array in displayed])
    if case.dataset == "darcy" and field == "a":
        unique = np.unique(values)
        if len(unique) <= 2:
            low = float(unique.min())
            high = float(unique.max())
            if low == high:
                high = low + 1.0
            midpoint = (low + high) / 2
            cmap = ListedColormap(["#2c7bb6", "#d7191c"])
            norm = BoundaryNorm([low - 1e-6, midpoint, high + 1e-6], cmap.N)
            return cmap, norm, None, None
    return "viridis", None, float(values.min()), float(values.max())


def write_posterior_samples_figure(
    path: str | Path,
    case: BenchmarkCase,
    reference: ReferencePosterior,
    generated: np.ndarray,
    generated_weights: np.ndarray,
    field: str | None = None,
    seed: int = DEFAULT_VISUALIZATION_SEED,
) -> dict[str, list[int]]:
    if field is None:
        field = case.target_field
    if field not in case.target_fields:
        raise ValueError(f"Unknown target field '{field}' for {case.target_fields}")
    generated = np.asarray(generated)
    if generated.ndim != 3:
        raise ValueError(
            f"Generated target samples must have shape [N,H,W], got {generated.shape}"
        )
    if generated.shape[1:] != case.resolution:
        raise ValueError(
            f"Generated visualization resolution {generated.shape[1:]} does not "
            f"match case resolution {case.resolution}"
        )

    rng = np.random.default_rng(seed)
    reference_indices = weighted_sample_indices(reference.weights, 3, rng)
    generated_indices = weighted_sample_indices(generated_weights, 4, rng)
    reference_samples = reference.fields[field]
    displayed = [case.fields[field]]
    displayed.extend(reference_samples[index] for index in reference_indices)
    displayed.extend(generated[index] for index in generated_indices)
    cmap, norm, vmin, vmax = _plot_style(case, field, displayed)

    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels: list[tuple[int, int, np.ndarray, str]] = [
        (
            0,
            0,
            case.fields[field],
            f"Latent truth {field}_ref",
        )
    ]
    panels.extend(
        (0, column, reference_samples[index], f"Reference posterior #{index}")
        for column, index in enumerate(reference_indices, start=1)
    )
    panels.extend(
        (1, column, generated[index], f"Generated posterior #{index}")
        for column, index in enumerate(generated_indices)
    )

    image = None
    for row, column, values, title in panels:
        image = axes[row, column].imshow(
            values,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )
        axes[row, column].set_title(title)
        axes[row, column].axis("off")

    occupied = {(row, column) for row, column, _, _ in panels}
    for row in range(2):
        for column in range(4):
            if (row, column) not in occupied:
                axes[row, column].axis("off")

    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.8, label=field)
    figure.suptitle(
        f"{case.case_id}: reference and generated {field} samples",
        fontsize=16,
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "reference_indices": reference_indices.astype(int).tolist(),
        "generated_indices": generated_indices.astype(int).tolist(),
    }
