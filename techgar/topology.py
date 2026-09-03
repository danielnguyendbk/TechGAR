"""Camera topology: directed transitions, exit/entry corridors, overlap zones.

Implements PLAN 2 §3.3-§3.5.  The rule this module exists to enforce: Camera 2
may only inherit a Camera 1 identity through a *calibrated directed corridor*,
never through global appearance search (PLAN 1 stage 7 logic 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contracts import TopologyRegion
from .geometry import inflate_polygon, point_in_polygon, point_polygon_distance


@dataclass
class CameraZone:
    """Calibrated world regions belonging to one camera."""

    camera_id: str
    fov_polygon: np.ndarray
    exit_polygons: dict[str, np.ndarray] = field(default_factory=dict)   # per successor
    entry_polygons: dict[str, np.ndarray] = field(default_factory=dict)  # per predecessor
    high_uncertainty_polygons: list[np.ndarray] = field(default_factory=list)
    entry_gate: np.ndarray | None = None
    exit_gate: np.ndarray | None = None

    def contains(self, position) -> bool:
        return point_in_polygon(position, self.fov_polygon)

    def in_exit_corridor(self, position, successor: str | None = None,
                         tolerance: float = 0.0) -> bool:
        polys = ([self.exit_polygons[successor]] if successor in self.exit_polygons
                 else list(self.exit_polygons.values()))
        return any(point_polygon_distance(position, p) <= tolerance for p in polys)

    def in_entry_corridor(self, position, predecessor: str | None = None,
                          tolerance: float = 0.0) -> bool:
        polys = ([self.entry_polygons[predecessor]] if predecessor in self.entry_polygons
                 else list(self.entry_polygons.values()))
        return any(point_polygon_distance(position, p) <= tolerance for p in polys)

    def extra_uncertainty(self, position) -> float:
        for poly in self.high_uncertainty_polygons:
            if point_in_polygon(position, poly):
                return 1.0
        return 0.0


@dataclass
class TopologyEdge:
    """A directed arc G_ij of the topology graph (PLAN 2 §3.4)."""

    source: str
    target: str
    dt_min: float = 0.0
    dt_max: float = 4.0
    dt_expected: float = 0.6
    v_max: float = 12.0

    def time_feasible(self, dt: float) -> bool:
        return self.dt_min - 1e-9 <= dt <= self.dt_max + 1e-9

    def displacement_feasible(self, displacement: float, dt: float, rho_seam: float = 0.0) -> bool:
        """||w_j - w_i|| <= v_max * dt + rho_seam (PLAN 2 §3.4)."""
        return displacement <= self.v_max * max(dt, 0.0) + rho_seam + 1e-9


@dataclass
class HandoffCheck:
    feasible: bool
    reason: str = ""
    edge: TopologyEdge | None = None


@dataclass
class CameraTopology:
    zones: dict[str, CameraZone] = field(default_factory=dict)
    edges: dict[tuple[str, str], TopologyEdge] = field(default_factory=dict)
    overlaps: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    #: Facility exit polygons (world).  Leaving through one of these is the only
    #: evidence that retires an identity and closes a session (PLAN 1 stage 8
    #: logic 5, Phase 5 work item 4).
    exit_lines: dict[str, np.ndarray] = field(default_factory=dict)
    entry_gates: dict[str, np.ndarray] = field(default_factory=dict)
    exit_gates: dict[str, np.ndarray] = field(default_factory=dict)

    def in_exit_line(self, position, camera_id: str | None = None) -> str | None:
        items = ([(camera_id, self.exit_lines[camera_id])] if camera_id in self.exit_lines
                 else list(self.exit_lines.items()))
        for name, polygon in items:
            if polygon is not None and point_in_polygon(position, polygon):
                return name
        return None

    # --- structure ----------------------------------------------------------
    def successors(self, camera_id: str) -> list[str]:
        return [tgt for (src, tgt) in self.edges if src == camera_id]

    def edge(self, source: str, target: str) -> TopologyEdge | None:
        return self.edges.get((source, target))

    def overlap_polygon(self, cam_a: str, cam_b: str) -> np.ndarray | None:
        poly = self.overlaps.get((cam_a, cam_b))
        if poly is None:
            poly = self.overlaps.get((cam_b, cam_a))
        return poly

    # --- queries ------------------------------------------------------------
    def region_of(self, camera_id: str, position) -> TopologyRegion:
        zone = self.zones.get(camera_id)
        if zone is None:
            return TopologyRegion.NORMAL
        for other in self.zones:
            if other == camera_id:
                continue
            poly = self.overlap_polygon(camera_id, other)
            if poly is not None and point_in_polygon(position, poly):
                return TopologyRegion.OVERLAP
        if zone.in_exit_corridor(position):
            return TopologyRegion.HANDOFF_EXIT
        if zone.in_entry_corridor(position):
            return TopologyRegion.HANDOFF_ENTRY
        return TopologyRegion.NORMAL

    def in_overlap(self, cam_a: str, cam_b: str, position, expansion: float = 0.0) -> bool:
        """PLAN 2 §3.3: w in Omega_expanded, expanded by *uncertainty*."""
        poly = self.overlap_polygon(cam_a, cam_b)
        if poly is None:
            return False
        if expansion <= 0.0:
            return point_in_polygon(position, poly)
        return point_polygon_distance(position, poly) <= expansion

    def expanded_overlap(self, cam_a: str, cam_b: str, expansion: float) -> np.ndarray | None:
        poly = self.overlap_polygon(cam_a, cam_b)
        return None if poly is None else inflate_polygon(poly, expansion)

    def check_handoff(self, source_camera: str, source_position, target_camera: str,
                      target_position, dt: float, rho_seam: float = 0.0,
                      tolerance: float = 0.0, source_confirmed: bool = False) -> HandoffCheck:
        """The full PLAN 2 §3.4 conjunction, with the failing clause named.

        ``tolerance`` widens the corridors by the candidate's own positional
        uncertainty (PLAN 2 §3.3: expansion is decided by uncertainty, not by an
        arbitrary global radius); ``source_confirmed`` records that the identity was
        *observed* inside the exit corridor earlier, which is the evidence the plan
        actually asks for — a filtered estimate that has since drifted a few
        centimetres past the polygon edge must not veto a valid handoff.
        """
        if source_camera == target_camera:
            return HandoffCheck(False, "same_camera")
        edge = self.edge(source_camera, target_camera)
        if edge is None:
            return HandoffCheck(False, "no_directed_arc")
        zone_src = self.zones.get(source_camera)
        zone_tgt = self.zones.get(target_camera)
        if zone_src is None or zone_tgt is None:
            return HandoffCheck(False, "unknown_camera", edge)
        effective_tolerance = max(tolerance, 0.25)
        if not source_confirmed and not zone_src.in_exit_corridor(source_position, target_camera,
                                                                 effective_tolerance):
            return HandoffCheck(False, "source_not_in_exit_polygon", edge)
        if not zone_tgt.in_entry_corridor(target_position, source_camera, effective_tolerance):
            return HandoffCheck(False, "target_not_in_entry_polygon", edge)
        if not edge.time_feasible(dt):
            return HandoffCheck(False, "time_infeasible", edge)
        displacement = float(np.linalg.norm(np.asarray(target_position, dtype=float)
                                            - np.asarray(source_position, dtype=float)))
        if not edge.displacement_feasible(displacement, dt, rho_seam):
            return HandoffCheck(False, "displacement_infeasible", edge)
        return HandoffCheck(True, "ok", edge)
