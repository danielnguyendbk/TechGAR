"""Per-camera survey profile — what commissioning measures, in one object.

A planar homography alone cannot describe a vehicle's *height*, yet height is
what makes a blob's near edge project past the vehicle's real footprint.  The
survey therefore also records:

* ``ground_direction`` — the pixel direction from the top of a plumb-line marker
  to its base.  The extreme of a blob along this direction provably lies on the
  floor, so it is the only anchor that a planar homography may trust
  (PLAN 1 stage 5 logic 1: "ground-contact point cho motion mặt sàn").
* ``camera_ground_point`` / ``parallax_gain`` — the world point under the camera
  and h_ref / H_cam, which give the known parallax budget of PLAN 1 stage 5
  logic 5 and Sigma_parallax of PLAN 2 §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homography import HomographyCalibration


@dataclass
class CameraProfile:
    camera_id: str
    calibration: HomographyCalibration
    width: int
    height: int
    ground_direction: np.ndarray
    camera_ground_point: np.ndarray
    vehicle_dimensions: tuple[float, float] = (4.6, 1.9)
    vehicle_height: float = 1.5
    expected_vehicle_area: float = 1500.0
    parallax_gain: float = 0.075
    anchor_bias: float = 0.0
    anchor_bias_sigma: float = 0.0

    def __post_init__(self) -> None:
        direction = np.asarray(self.ground_direction, dtype=float).reshape(2)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError(f"{self.camera_id}: ground_direction must be non-zero")
        self.ground_direction = direction / norm
        self.camera_ground_point = np.asarray(self.camera_ground_point, dtype=float).reshape(2)

    @property
    def image_diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def parallax_offset(self, world_position) -> float:
        """Distance by which a roof point over-projects towards the camera."""
        distance = float(np.linalg.norm(np.asarray(world_position, dtype=float)
                                        - self.camera_ground_point))
        return self.parallax_gain * distance

    def footprint_depth(self, observed_edge: float) -> float:
        """Extent perpendicular to the observed ground edge, from the vehicle model.

        The ground edge of a vehicle is its side when it drives along the camera's
        ground direction (observed length ~ L, depth ~ W) and its front/rear when
        it turns into a slot (observed ~ W, depth ~ L).  Interpolating between the
        two regimes instead of switching keeps the reconstructed footprint
        continuous while a vehicle turns, which the slot stability test in
        PLAN 2 §5.4 depends on.
        """
        length, width = self.vehicle_dimensions
        span = length - width
        if abs(span) < 1e-6:
            return width
        u = float(np.clip((observed_edge - width) / span, 0.0, 1.0))
        return width * u + length * (1.0 - u)

    def world_ground_direction(self, pixel) -> np.ndarray:
        """The world direction that the pixel ground direction maps to locally."""
        pixel = np.asarray(pixel, dtype=float)
        jacobian = self.calibration.jacobian(pixel[0], pixel[1])
        direction = jacobian @ self.ground_direction
        norm = float(np.linalg.norm(direction))
        return np.array([0.0, -1.0]) if norm < 1e-12 else direction / norm

    def border_distance(self, pixel) -> float:
        u, v = float(pixel[0]), float(pixel[1])
        return min(u, v, self.width - u, self.height - v)

    def away_direction(self, world_position) -> np.ndarray:
        """Unit world vector pointing from the camera towards ``world_position``."""
        delta = np.asarray(world_position, dtype=float) - self.camera_ground_point
        norm = float(np.linalg.norm(delta))
        return np.array([1.0, 0.0]) if norm < 1e-9 else delta / norm

    def correct_anchor(self, world_position) -> np.ndarray:
        """Apply the surveyed systematic anchor bias (see ``anchor_bias``).

        The ground band of a blob sits a fixed distance inside the true contact
        edge because the fail-open shadow rule protects a one-pixel halo around
        the vehicle boundary.  The offset is a measured constant of the install,
        so commissioning removes it and reports the residual scatter as
        ``anchor_bias_sigma`` (which feeds Sigma_calib, never a hidden fudge).
        """
        position = np.asarray(world_position, dtype=float)
        if self.anchor_bias == 0.0:
            return position
        return position + self.anchor_bias * self.away_direction(position)
