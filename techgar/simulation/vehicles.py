"""Scripted vehicles for the synthetic rig."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    t: float
    x: float
    y: float
    heading: float | None = None


@dataclass
class SimVehicle:
    """A rigid box on the floor plane following a piecewise-linear schedule."""

    vehicle_id: str
    waypoints: list[Waypoint]
    length: float = 4.6
    width: float = 1.9
    height: float = 1.5
    color: tuple[int, int, int] = (190, 190, 195)

    def __post_init__(self) -> None:
        self.waypoints = sorted(self.waypoints, key=lambda w: w.t)
        if len(self.waypoints) < 2:
            raise ValueError(f"{self.vehicle_id}: need at least two waypoints")

    @property
    def t_start(self) -> float:
        return self.waypoints[0].t

    @property
    def t_end(self) -> float:
        return self.waypoints[-1].t

    def present(self, t: float) -> bool:
        return self.t_start - 1e-9 <= t <= self.t_end + 1e-9

    def position(self, t: float) -> np.ndarray:
        wps = self.waypoints
        if t <= wps[0].t:
            return np.array([wps[0].x, wps[0].y])
        if t >= wps[-1].t:
            return np.array([wps[-1].x, wps[-1].y])
        for a, b in zip(wps, wps[1:]):
            if a.t <= t <= b.t:
                span = b.t - a.t
                u = 0.0 if span <= 0 else (t - a.t) / span
                return np.array([a.x + u * (b.x - a.x), a.y + u * (b.y - a.y)])
        return np.array([wps[-1].x, wps[-1].y])

    def velocity(self, t: float, dt: float = 0.02) -> np.ndarray:
        t0 = max(self.t_start, t - dt)
        t1 = min(self.t_end, t + dt)
        if t1 <= t0:
            return np.zeros(2)
        return (self.position(t1) - self.position(t0)) / (t1 - t0)

    def heading(self, t: float) -> float:
        for wp in self.waypoints:
            if abs(wp.t - t) < 1e-9 and wp.heading is not None:
                return wp.heading
        v = self.velocity(t)
        if np.linalg.norm(v) < 1e-6:
            # Standing still: reuse the last moving direction so a parked car
            # keeps its orientation instead of snapping to zero.
            for probe in np.arange(t, self.t_start - 1e-9, -0.1):
                v = self.velocity(float(probe))
                if np.linalg.norm(v) >= 1e-6:
                    break
        if np.linalg.norm(v) < 1e-6:
            return 0.0
        return float(np.arctan2(v[1], v[0]))

    def footprint(self, t: float) -> np.ndarray:
        """Ground-contact rectangle (world units), oriented along the heading."""
        c = self.position(t)
        theta = self.heading(t)
        forward = np.array([np.cos(theta), np.sin(theta)])
        left = np.array([-np.sin(theta), np.cos(theta)])
        hl, hw = self.length / 2.0, self.width / 2.0
        return np.array([c + hl * forward + hw * left,
                         c + hl * forward - hw * left,
                         c - hl * forward - hw * left,
                         c - hl * forward + hw * left])

    def box_faces(self, t: float) -> list[np.ndarray]:
        """Bottom face + roof + four sides as 3-D polygons, for rendering."""
        base = self.footprint(t)
        bottom = np.column_stack([base, np.zeros(len(base))])
        roof = np.column_stack([base, np.full(len(base), self.height)])
        faces = [bottom, roof]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(np.array([bottom[i], bottom[j], roof[j], roof[i]]))
        return faces


@dataclass
class Occluder:
    """Structural blind spot: a pillar/wall box standing on the floor."""

    occluder_id: str
    centre: tuple[float, float]
    size: tuple[float, float] = (0.8, 0.8)
    height: float = 2.6
    color: tuple[int, int, int] = (95, 95, 100)

    def footprint(self) -> np.ndarray:
        cx, cy = self.centre
        sx, sy = self.size[0] / 2.0, self.size[1] / 2.0
        return np.array([[cx - sx, cy - sy], [cx + sx, cy - sy],
                         [cx + sx, cy + sy], [cx - sx, cy + sy]])

    def box_faces(self) -> list[np.ndarray]:
        base = self.footprint()
        bottom = np.column_stack([base, np.zeros(4)])
        roof = np.column_stack([base, np.full(4, self.height)])
        faces = [roof]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(np.array([bottom[i], bottom[j], roof[j], roof[i]]))
        return faces
