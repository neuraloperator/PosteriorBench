"""NumPy/SciPy implementation of PosteriorBench paper Eqs. (9)--(15).

One physical risk channel is evaluated. References can have unequal weights;
predictions normally have equal weights. Population variances are used. SWD
uses one frozen set of 128 unit-norm, zero-DC periodic GRF directions (alpha=2,
tau=3); both inputs use the same training-only channel normalization. Spectral
error is the geometric mean in Eq. (15). These are paper-based diagnostics,
not an executed official CLI.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


METRICS = ('mean_relative_l2', 'std_relative_l2', 'mmd', 'swd', 'spectral_error')


def ensemble(fields, weights=None):
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 3 or not len(fields) or not np.isfinite(fields).all():
        raise ValueError('Expected finite [samples, rows, columns] fields.')
    if weights is None:
        weights = np.ones(len(fields), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(fields),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid ensemble weights.')
    return fields, weights / weights.sum()


def moments(fields, weights):
    mean = np.einsum('n,nij->ij', weights, fields)
    variance = np.einsum('n,nij->ij', weights, (fields - mean) ** 2)
    return mean, np.sqrt(np.maximum(variance, 0.))


def relative_l2(a, b):
    denominator = float(np.linalg.norm(b))
    if denominator <= 1e-14:
        return float('nan')
    return float(np.linalg.norm(a - b) / denominator)


def squared_distances(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0.)


def weighted_mmd(a, b, wa, wb, bandwidth=None):
    cross = squared_distances(a, b)
    if bandwidth is None:
        bandwidth = float(np.median(cross))
        if bandwidth <= 1e-14:
            positive = cross[cross > 1e-14]
            bandwidth = float(np.median(positive)) if len(positive) else 1.
    elif not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Fixed MMD bandwidth must be positive and finite.')
    def kernel(distances):
        return np.mean([np.exp(-distances / (bandwidth * s)) for s in (.2, .5, 1., 2., 5.)], axis=0)
    kaa, kbb = kernel(squared_distances(a, a)), kernel(squared_distances(b, b))
    # Self-distances are exactly zero, without dot-product cancellation noise.
    np.fill_diagonal(kaa, 1.)
    np.fill_diagonal(kbb, 1.)
    score = float(wa @ kaa @ wa + wb @ kbb @ wb - 2 * wa @ kernel(cross) @ wb)
    return float(np.sqrt(max(score, 0.)))


def grf_directions(shape=(128, 128), count=128, seed=88021, alpha=2., tau=3.):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(shape[0]) * shape[0]
    fx = np.fft.rfftfreq(shape[1]) * shape[1]
    filt = (tau * tau + 4 * np.pi**2 * (fy[:, None]**2 + fx[None, :]**2)) ** (-alpha / 2)
    filt[0, 0] = 0.
    noise = rng.normal(size=(count, *shape))
    vectors = np.fft.irfft2(np.fft.rfft2(noise, axes=(-2, -1)) * filt,
                           s=shape, axes=(-2, -1)).reshape(count, -1)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def weighted_swd(a, b, wa, wb, directions, mean, std):
    if std <= 0 or not np.isfinite(std):
        raise ValueError('A shared positive normalization scale is required.')
    aa = ((a.reshape(len(a), -1) - mean) / std) @ directions.T
    bb = ((b.reshape(len(b), -1) - mean) / std) @ directions.T
    return float(np.mean([wasserstein_distance(x, y, wa, wb) for x, y in zip(aa.T, bb.T)]))


def radial_spectra(fields):
    fields = np.asarray(fields, dtype=np.float64)
    h, w = fields.shape[-2:]
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    bins = np.floor(np.sqrt(fy[:, None]**2 + fx[None, :]**2)).astype(int).ravel()
    counts = np.bincount(bins)
    power = np.abs(np.fft.fft2(fields, axes=(-2, -1), norm='ortho')) ** 2
    return np.stack([np.bincount(bins, weights=p.ravel(), minlength=len(counts)) / counts for p in power])


def spectral_discrepancy(sa, sb, valid_bins=None):
    """Eq. (15) on ensemble spectra, optionally with pre-calibrated valid bins."""
    sa, sb = np.asarray(sa), np.asarray(sb)
    valid = sb > max(float(sb.max()) * 1e-14, 1e-30) if valid_bins is None else np.asarray(valid_bins, dtype=bool)
    if valid.shape != sb.shape or sa.shape != sb.shape:
        raise ValueError('Spectrum and frequency-mask shapes differ.')
    if not valid.any():
        return float('nan')
    if (sb[valid] <= 0).any():
        return float('nan')
    difference = np.abs(sa[valid] - sb[valid])
    if np.all(difference == 0):
        return 0.
    relative = difference / sb[valid]
    if np.any(relative == 0):
        return 0.
    return float(np.exp(np.mean(np.log(relative))))


def spectral_error(a, b, wa, wb, valid_bins=None):
    sa, sb = wa @ radial_spectra(a), wb @ radial_spectra(b)
    return spectral_discrepancy(sa, sb, valid_bins)


def compare(a, b, directions, normalization, wa=None, wb=None, mmd_bandwidth=None, spectral_valid_bins=None):
    a, wa = ensemble(a, wa)
    b, wb = ensemble(b, wb)
    if a.shape[1:] != b.shape[1:]:
        raise ValueError('Ensemble field shapes differ.')
    ma, sa = moments(a, wa)
    mb, sb = moments(b, wb)
    return dict(mean_relative_l2=relative_l2(ma, mb), std_relative_l2=relative_l2(sa, sb),
                mmd=weighted_mmd(a, b, wa, wb, bandwidth=mmd_bandwidth),
                swd=weighted_swd(a, b, wa, wb, directions, **normalization),
                spectral_error=spectral_error(a, b, wa, wb, valid_bins=spectral_valid_bins))
