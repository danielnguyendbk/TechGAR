"""Virtual oblique camera over the floor plane (validation rig, not production).

Provides the *ground truth* pixel<->world mapping the pipeline is never told
about: calibration must be re-estimated from noisy correspondences exactly as it
would be on site (PLAN 1 Phase 0 work items 4-5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import clip_polygon, ensure_ccw


def _look_at_rotation(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rows = camera x (right), y (down), z (forward) expressed in world axes."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    norm = np.linalg.norm(right)
    if norm < 1e-9:                     # nadir view: pick an arbitrary right
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / norm
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


@dataclass
class VirtualCamera:
    camera_id: str
    position: np.ndarray
    target: np.ndarray
    focal: float = 500.0
    width: int = 640
    height: int = 480
    rotation: np.ndarray = field(init=False)
    intrinsics: np.ndarray = field(init=False)
    h_world_to_pixel: np.ndarray = field(init=False)
    h_pixel_to_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.target = np.asarray(self.target, dtype=float)
        self.rotation = _look_at_rotation(self.position, self.target)
        self.intrinsics = np.array([[self.focal, 0.0, self.width / 2.0],
                                    [0.0, self.focal, self.height / 2.0],
                                    [0.0, 0.0, 1.0]])
        r = self.rotation
        translation = -r @ self.position
        self.h_world_to_pixel = self.intrinsics @ np.column_stack([r[:, 0], r[:, 1], translation])
        self.h_pixel_to_world = np.linalg.inv(self.h_world_to_pixel)

    # --- projection ---------------------------------------------------------
    def project_3d(self, points_xyz) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_xyz, dtype=float))
        cam = (p - self.position) @ self.rotation.T
        depth = np.clip(cam[:, 2], 1e-6, None)
        pix = (self.intrinsics @ np.column_stack([cam[:, 0], cam[:, 1], depth]).T).T
        return pix[:, :2] / pix[:, 2:3]

    def depth_3d(self, points_xyz) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_xyz, dtype=float))
        return ((p - self.position) @ self.rotation.T)[:, 2]

    def project_floor(self, points_xy) -> np.ndarray:
        p = np.atleast_2d(np.asarray(points_xy, dtype=float))
        return self.project_3d(np.column_stack([p, np.zeros(len(p))]))

    def floor_fov_polygon(self, floor_bounds=None) -> np.ndarray:
        corners = np.array([[0.0, 0.0], [self.width, 0.0],
                            [self.width, self.height], [0.0, self.height]])
        homo = np.column_stack([corners, np.ones(4)]) @ self.h_pixel_to_world.T
        if np.any(homo[:, 2] <= 1e-9):
            raise ValueError(f"camera {self.camera_id}: image corner above the horizon")
        quad = ensure_ccw(homo[:, :2] / homo[:, 2:3])
        if floor_bounds is None:
            return quad
        x0, y0, x1, y1 = floor_bounds
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        clipped = clip_polygon(quad, rect)
        return quad if clipped is None else clipped

    def sees_floor_point(self, point_xy, margin: float = 0.0) -> bool:
        pix = self.project_floor(point_xy)[0]
        return bool(margin <= pix[0] <= self.width - margin
                    and margin <= pix[1] <= self.height - margin)

    # --- calibration --------------------------------------------------------
    def calibration_points(self, count: int = 12, noise_px: float = 0.6, rng=None,
                           floor_bounds=None) -> tuple[np.ndarray, np.ndarray]:
        """Marker survey: world points inside the FOV, measured in pixels with noise.

        ``count > 4`` on purpose — PLAN 2 §3.2 forbids validating a homography
        with the 4 points that determine it exactly.
        """
        rng = np.random.default_rng(7) if rng is None else rng
        quad = self.floor_fov_polygon(floor_bounds)
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        world = []
        side = int(np.ceil(np.sqrt(count))) + 1
        for gx in np.linspace(x0 + 0.08 * (x1 - x0), x1 - 0.08 * (x1 - x0), side):
            for gy in np.linspace(y0 + 0.08 * (y1 - y0), y1 - 0.08 * (y1 - y0), side):
                if self.sees_floor_point((gx, gy), margin=12.0):
                    world.append((gx, gy))
        if len(world) < count:
            raise ValueError(f"camera {self.camera_id}: only {len(world)} usable markers")
        idx = rng.choice(len(world), size=count, replace=False)
        world_pts = np.asarray([world[i] for i in sorted(idx)], dtype=float)
        pixel_pts = self.project_floor(world_pts) + rng.normal(0.0, noise_px, size=(count, 2))
        return pixel_pts, world_pts
