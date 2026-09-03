"""Small robust-statistics / covariance helpers used across the pipeline.

Everything that touches an inverse covariance goes through here so that a
single jittered, symmetrised inverse is used everywhere (PLAN 2 §4.1 gating,
§3.2 propagation, PLAN 1 stage 6 information-filter fusion).
"""

from __future__ import annotations

import numpy as np

#: chi-square quantiles for 2 degrees of freedom (world plane gating).
CHI2_2DOF = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210, 0.997: 11.829}
#: PLAN 2 §4.1: "Gate hợp lệ: C_distance < chi2(2; 0.99) = 9.21".
CHI2_2DOF_99 = 9.210

MAD_TO_SIGMA = 1.4826  # consistency constant for a normal distribution


def symmetrize(matrix) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    return 0.5 * (m + m.T)


def psd_inverse(matrix, jitter: float = 1e-12) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix, jittered if singular."""
    m = symmetrize(matrix)
    n = m.shape[0]
    for scale in (0.0, jitter, jitter * 1e3, jitter * 1e6, jitter * 1e9):
        try:
            return np.linalg.inv(m + scale * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(m)


def mahalanobis_sq(residual, covariance) -> float:
    """r^T S^-1 r  (PLAN 2 §4.1)."""
    r = np.asarray(residual, dtype=float).reshape(-1)
    return float(r @ psd_inverse(covariance) @ r)


def gate_ok(residual, covariance, confidence: float = 0.99) -> bool:
    return mahalanobis_sq(residual, covariance) < CHI2_2DOF.get(confidence, CHI2_2DOF_99)


def inflate_isotropic(covariance, rho: float) -> np.ndarray:
    """Sigma + rho^2 I — the seam / border uncertainty budget (PLAN 2 §3.2)."""
    m = symmetrize(covariance)
    return m + float(rho) ** 2 * np.eye(m.shape[0])


def information_fuse(means, covariances) -> tuple[np.ndarray, np.ndarray]:
    """Information-filter fusion of independent Gaussian estimates.

    Sigma_f = (sum Sigma_i^-1)^-1 ,  mu_f = Sigma_f sum Sigma_i^-1 mu_i
    (PLAN 1 stage 6 / Phase 2 "Fusion chỉ dùng Euclidean, bỏ covariance" = Fail).
    """
    means = [np.asarray(m, dtype=float).reshape(-1) for m in means]
    if not means:
        raise ValueError("information_fuse needs at least one estimate")
    dim = means[0].size
    info = np.zeros((dim, dim))
    weighted = np.zeros(dim)
    for mu, cov in zip(means, covariances):
        inv = psd_inverse(cov)
        info += inv
        weighted += inv @ mu
    cov_f = psd_inverse(info)
    return cov_f @ weighted, symmetrize(cov_f)


def ellipse_semi_axes(covariance, confidence: float = 0.99) -> np.ndarray:
    """Semi-axis lengths of the confidence ellipse (descending)."""
    vals = np.linalg.eigvalsh(symmetrize(covariance))
    vals = np.clip(vals, 0.0, None)
    k = CHI2_2DOF.get(confidence, CHI2_2DOF_99)
    return np.sqrt(vals * k)[::-1]


def positional_sigma(covariance) -> float:
    """Scalar summary of a 2-D positional covariance: sqrt(trace/2)."""
    m = symmetrize(covariance)
    return float(np.sqrt(max(np.trace(m), 0.0) / m.shape[0]))


def mad(values, scale: float = MAD_TO_SIGMA) -> float:
    """Median absolute deviation, scaled to a sigma estimate (PLAN 2 §1.4).

    Robust to the vehicle motion that would drag a plain standard deviation.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0
    med = float(np.median(v))
    return float(np.median(np.abs(v - med)) * scale)


def percentiles(values, qs=(50, 95, 100)) -> dict[int, float]:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(v, q)) for q in qs}


def clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))
