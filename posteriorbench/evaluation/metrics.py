import numpy as np
from scipy.fft import idctn


DEFAULT_N_PROJECTIONS = 128
DEFAULT_PROJECTION_SEED = 0
DEFAULT_PROJECTION_GRF_ALPHA = 2.0
DEFAULT_PROJECTION_GRF_TAU = 3.0
SPECTRAL_RELATIVE_ERROR_EPS = 1e-10


def normalize_weights(
    weights: np.ndarray,
    expected_length: int | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"Weights must have shape [N], got {weights.shape}")
    if expected_length is not None and weights.shape != (expected_length,):
        raise ValueError(
            f"Weights must have shape ({expected_length},), got {weights.shape}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite")
    if np.any(weights < 0):
        raise ValueError("Weights must be nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return weights / total


def _pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2 * (x @ y.T), 0.0)


def weighted_mmd(
    predicted: np.ndarray,
    reference: np.ndarray,
    pred_weights: np.ndarray,
    ref_weights: np.ndarray,
) -> float:
    x = predicted.reshape(len(predicted), -1).astype(np.float64)
    y = reference.reshape(len(reference), -1).astype(np.float64)
    wx = normalize_weights(pred_weights, len(predicted))
    wy = normalize_weights(ref_weights, len(reference))
    d_xx = _pairwise_sq_dist(x, x)
    d_yy = _pairwise_sq_dist(y, y)
    d_xy = _pairwise_sq_dist(x, y)
    bandwidth = float(np.median(d_xy))
    if bandwidth <= 0:
        bandwidth = 1.0

    scales = (0.2, 0.5, 1.0, 2.0, 5.0)

    def kernel(distances: np.ndarray) -> np.ndarray:
        return sum(
            np.exp(-distances / (scale * bandwidth)) for scale in scales
        ) / len(scales)

    value = (
        np.sum(np.outer(wx, wx) * kernel(d_xx))
        + np.sum(np.outer(wy, wy) * kernel(d_yy))
        - 2 * np.sum(np.outer(wx, wy) * kernel(d_xy))
    )
    return float(np.sqrt(max(value, 0.0)))


def weighted_wasserstein_1d(
    x: np.ndarray,
    y: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    wx = normalize_weights(wx, len(x))
    wy = normalize_weights(wy, len(y))
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    x, wx = x[order_x], wx[order_x]
    y, wy = y[order_y], wy[order_y]
    values = np.unique(np.concatenate([x, y]))
    if len(values) < 2:
        return 0.0
    cdf_x = np.searchsorted(x, values[:-1], side="right")
    cdf_y = np.searchsorted(y, values[:-1], side="right")
    cum_x = np.concatenate([[0.0], np.cumsum(wx)])
    cum_y = np.concatenate([[0.0], np.cumsum(wy)])
    return float(
        np.sum(
            np.abs(cum_x[cdf_x] - cum_y[cdf_y])
            * np.diff(values)
        )
    )


def sample_grf_projection(
    height: int,
    width: int,
    rng: np.random.Generator,
    alpha: float = DEFAULT_PROJECTION_GRF_ALPHA,
    tau: float = DEFAULT_PROJECTION_GRF_TAU,
) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError(f"Projection shape must be positive, got {(height, width)}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    xi = rng.normal(size=(height, width))
    ky = np.arange(height, dtype=np.float64)
    kx = np.arange(width, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(kx, ky, indexing="xy")
    coef = tau ** (alpha - 1) * (
        np.pi**2 * (grid_x**2 + grid_y**2) + tau**2
    ) ** (-alpha / 2)
    coefficients = np.sqrt(height * width) * coef * xi
    coefficients[0, 0] = 0.0
    direction = idctn(coefficients, type=2, norm="ortho")
    direction /= np.linalg.norm(direction.ravel()) + 1e-12
    return direction.astype(np.float64, copy=False)


def sliced_wasserstein(
    predicted: np.ndarray,
    reference: np.ndarray,
    pred_weights: np.ndarray,
    ref_weights: np.ndarray,
    n_projections: int = DEFAULT_N_PROJECTIONS,
    seed: int = DEFAULT_PROJECTION_SEED,
    grf_alpha: float = DEFAULT_PROJECTION_GRF_ALPHA,
    grf_tau: float = DEFAULT_PROJECTION_GRF_TAU,
) -> float:
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    if grf_alpha <= 0:
        raise ValueError("grf_alpha must be positive")
    if grf_tau <= 0:
        raise ValueError("grf_tau must be positive")
    x = predicted.reshape(len(predicted), -1).astype(np.float64)
    y = reference.reshape(len(reference), -1).astype(np.float64)
    _, height, width = predicted.shape
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_projections):
        direction = sample_grf_projection(
            height,
            width,
            rng,
            alpha=grf_alpha,
            tau=grf_tau,
        ).reshape(-1)
        distances.append(
            weighted_wasserstein_1d(
                x @ direction,
                y @ direction,
                pred_weights,
                ref_weights,
            )
        )
    return float(np.mean(distances))


def radially_averaged_power_spectrum(
    samples: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weights = normalize_weights(weights, len(samples))
    _, height, width = samples.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    grid_x, grid_y = np.meshgrid(fx, fy)
    radius = np.sqrt(grid_x**2 + grid_y**2)
    frequency_step = 1 / min(height, width)
    bins = np.arange(0, 0.5 + frequency_step, frequency_step)
    indices = np.digitize(radius.ravel(), bins) - 1
    n_bins = len(bins) - 1

    spectra = []
    for sample in samples:
        power = np.abs(np.fft.fftshift(np.fft.fft2(sample))) ** 2
        valid = (indices >= 0) & (indices < n_bins)
        sums = np.bincount(
            indices[valid],
            weights=power.ravel()[valid],
            minlength=n_bins,
        )
        counts = np.bincount(indices[valid], minlength=n_bins)
        spectra.append(sums / np.maximum(counts, 1))
    return np.average(np.asarray(spectra), axis=0, weights=weights)


def spectral_relative_geomean(pred_spectrum: np.ndarray, ref_spectrum: np.ndarray) -> float:
    relative_error = (
        np.abs(pred_spectrum - ref_spectrum)
        / (np.abs(ref_spectrum) + SPECTRAL_RELATIVE_ERROR_EPS)
    )
    valid_mask = (
        np.isfinite(relative_error)
        & np.isfinite(ref_spectrum)
        & (ref_spectrum > SPECTRAL_RELATIVE_ERROR_EPS)
    )
    if not np.any(valid_mask):
        return float("nan")
    valid_errors = relative_error[valid_mask]
    return float(
        np.exp(np.mean(np.log(valid_errors + SPECTRAL_RELATIVE_ERROR_EPS)))
    )


def evaluate_distribution(
    predicted: np.ndarray,
    reference: np.ndarray,
    pred_weights: np.ndarray,
    ref_weights: np.ndarray,
    n_projections: int = DEFAULT_N_PROJECTIONS,
    projection_seed: int = DEFAULT_PROJECTION_SEED,
    projection_grf_alpha: float = DEFAULT_PROJECTION_GRF_ALPHA,
    projection_grf_tau: float = DEFAULT_PROJECTION_GRF_TAU,
    swd_normalization_mean: float | None = None,
    swd_normalization_std: float | None = None,
) -> dict[str, float]:
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    if predicted.ndim != 3 or reference.ndim != 3:
        raise ValueError(
            "Predicted and reference samples must have shape [N,H,W], got "
            f"{predicted.shape} and {reference.shape}"
        )
    if len(predicted) == 0 or len(reference) == 0:
        raise ValueError("Predicted and reference sample sets must be non-empty")
    if predicted.shape[1:] != reference.shape[1:]:
        raise ValueError(
            f"Sample resolution mismatch: {predicted.shape} vs {reference.shape}"
        )
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(reference)):
        raise ValueError("Predicted and reference samples must be finite")
    if (swd_normalization_mean is None) != (swd_normalization_std is None):
        raise ValueError(
            "Both swd_normalization_mean and swd_normalization_std must be provided"
        )
    if swd_normalization_mean is not None:
        if not np.isfinite(swd_normalization_mean):
            raise ValueError("swd_normalization_mean must be finite")
        if not np.isfinite(swd_normalization_std) or swd_normalization_std <= 0:
            raise ValueError("swd_normalization_std must be finite and positive")

    pred_weights = normalize_weights(pred_weights, len(predicted))
    ref_weights = normalize_weights(ref_weights, len(reference))
    pred_mean = np.average(predicted, axis=0, weights=pred_weights)
    ref_mean = np.average(reference, axis=0, weights=ref_weights)
    pred_std = np.sqrt(
        np.average((predicted - pred_mean) ** 2, axis=0, weights=pred_weights)
    )
    ref_std = np.sqrt(
        np.average((reference - ref_mean) ** 2, axis=0, weights=ref_weights)
    )
    pred_spectrum = radially_averaged_power_spectrum(predicted, pred_weights)
    ref_spectrum = radially_averaged_power_spectrum(reference, ref_weights)
    swd_predicted = predicted
    swd_reference = reference
    if swd_normalization_mean is not None:
        mean = float(swd_normalization_mean)
        std = float(swd_normalization_std)
        swd_predicted = (predicted.astype(np.float64) - mean) / std
        swd_reference = (reference.astype(np.float64) - mean) / std

    return {
        "mean_rel_l2": float(
            np.linalg.norm(pred_mean - ref_mean)
            / (np.linalg.norm(ref_mean) + 1e-12)
        ),
        "std_rel_l2": float(
            np.linalg.norm(pred_std - ref_std)
            / (np.linalg.norm(ref_std) + 1e-12)
        ),
        "mmd": weighted_mmd(
            predicted,
            reference,
            pred_weights,
            ref_weights,
        ),
        "swd": sliced_wasserstein(
            swd_predicted,
            swd_reference,
            pred_weights,
            ref_weights,
            n_projections=n_projections,
            seed=projection_seed,
            grf_alpha=projection_grf_alpha,
            grf_tau=projection_grf_tau,
        ),
        "spectral_rel_geomean": spectral_relative_geomean(pred_spectrum, ref_spectrum),
    }
