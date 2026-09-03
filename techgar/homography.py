"""Planar homography calibration and uncertainty propagation (PLAN 2 §3.1-§3.2).

Two things this module refuses to do quietly:

* it never reports a 4-point calibration as "zero error" — with 8 DOF and
  exactly 4 correspondences the fit is an exact solution, so the residual is
  structurally 0 and proves nothing (PLAN 1 Phase 0 Fail criterion, PLAN 2 §3.2
  warning).  ``dof_redundancy = 2n - 8`` is reported alongside every residual;
* it never approximates the projection Jacobian by finite differences — the
  analytic form is written out explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .linalg import symmetrize


def _normalizing_transform(points: np.ndarray) -> np.ndarray:
    """Hartley isotropic normalisation: centroid at origin, mean |p| = sqrt(2)."""
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([[scale, 0.0, -scale * centroid[0]],
                     [0.0, scale, -scale * centroid[1]],
                     [0.0, 0.0, 1.0]])


def estimate_homography(pixel_points, world_points) -> np.ndarray:
    """Normalised DLT.  Needs >= 4 correspondences; > 4 gives a real residual."""
    p = np.asarray(pixel_points, dtype=float)
    q = np.asarray(world_points, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("pixel/world point sets must both be (n, 2)")
    if len(p) < 4:
        raise ValueError("homography needs at least 4 correspondences")
    tp, tq = _normalizing_transform(p), _normalizing_transform(q)
    ph = np.column_stack([p, np.ones(len(p))]) @ tp.T
    qh = np.column_stack([q, np.ones(len(q))]) @ tq.T
    rows = []
    for (u, v, _), (x, y, _) in zip(ph, qh):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h_norm = vt[-1].reshape(3, 3)
    h = np.linalg.inv(tq) @ h_norm @ tp
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def project_points(h: np.ndarray, points) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, dtype=float))
    homo = np.column_stack([p, np.ones(len(p))]) @ np.asarray(h, dtype=float).T
    w = homo[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = homo[:, :2] / w
    return out.reshape(np.shape(points)) if np.ndim(points) == 2 else out[0]


def projection_jacobian(h: np.ndarray, u: float, v: float) -> np.ndarray:
    """Explicit d(X, Y)/d(u, v) of the homography at one pixel (PLAN 2 §3.2)."""
    h = np.asarray(h, dtype=float)
    w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if abs(w) < 1e-12:
        w = 1e-12
    x = (h[0, 0] * u + h[0, 1] * v + h[0, 2]) / w
    y = (h[1, 0] * u + h[1, 1] * v + h[1, 2]) / w
    return np.array([[(h[0, 0] - x * h[2, 0]) / w, (h[0, 1] - x * h[2, 1]) / w],
                     [(h[1, 0] - y * h[2, 0]) / w, (h[1, 1] - y * h[2, 1]) / w]])


@dataclass
class HomographyCalibration:
    """A calibration plus the honest quality report that goes with it."""

    camera_id: str
    h: np.ndarray
    pixel_points: np.ndarray
    world_points: np.ndarray
    residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sigma_calib: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    sigma_floor: float = 0.02

    @property
    def n_points(self) -> int:
        return len(self.pixel_points)

    @property
    def dof_redundancy(self) -> int:
        """Redundant degrees of freedom: 2n measurements - 8 homography DOF."""
        return 2 * self.n_points - 8

    @property
    def overfit(self) -> bool:
        return self.dof_redundancy <= 0

    @property
    def rms_residual(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2))) if self.residuals.size else 0.0

    @property
    def max_residual(self) -> float:
        return float(self.residuals.max()) if self.residuals.size else 0.0

    def project(self, points) -> np.ndarray:
        return project_points(self.h, points)

    def unproject(self, world_points) -> np.ndarray:
        return project_points(np.linalg.inv(self.h), world_points)

    def jacobian(self, u: float, v: float) -> np.ndarray:
        return projection_jacobian(self.h, u, v)

    def propagate(self, sigma_pixel, u: float, v: float, sigma_parallax=None) -> np.ndarray:
        """Sigma_w = J Sigma_p J^T + Sigma_calib + Sigma_parallax (PLAN 2 §3.2)."""
        j = self.jacobian(u, v)
        cov = j @ symmetrize(sigma_pixel) @ j.T + self.sigma_calib
        if sigma_parallax is not None:
            cov = cov + symmetrize(np.atleast_2d(sigma_parallax))
        return symmetrize(cov)

    def report(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "n_points": self.n_points,
            "dof_redundancy": self.dof_redundancy,
            "overfit_exact_solution": self.overfit,
            "rms_residual_world": self.rms_residual,
            "max_residual_world": self.max_residual,
            "sigma_calib_trace": float(np.trace(self.sigma_calib)),
            "residual_is_meaningful": not self.overfit,
        }


def calibrate(camera_id: str, pixel_points, world_points, sigma_floor: float = 0.02
              ) -> HomographyCalibration:
    """Fit H and measure it.  ``n = 4`` yields ``overfit=True`` and a floor sigma."""
    p = np.asarray(pixel_points, dtype=float)
    q = np.asarray(world_points, dtype=float)
    h = estimate_homography(p, q)
    projected = project_points(h, p)
    errors = projected - q
    residuals = np.linalg.norm(errors, axis=1)
    calib = HomographyCalibration(camera_id, h, p, q, residuals, np.zeros((2, 2)), sigma_floor)
    if calib.overfit:
        # An exact fit cannot estimate its own error: fall back to a declared floor.
        calib.sigma_calib = (sigma_floor ** 2) * np.eye(2)
    else:
        n_eff = max(1, calib.n_points - 4)      # 8 parameters over 2n equations
        calib.sigma_calib = symmetrize(errors.T @ errors / n_eff)
        floor = (sigma_floor ** 2) * np.eye(2)
        if np.trace(calib.sigma_calib) < np.trace(floor):
            calib.sigma_calib = calib.sigma_calib + floor
    return calib
