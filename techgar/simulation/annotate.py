"""Ground-truth annotation for the synthetic rig (PLAN 3 §1 schema).

The evaluator consumes only what this module produces.  Nothing here ever looks
at a predicted Global ID — PLAN 3's golden rule ("evaluator KHÔNG BAO GIỜ dùng
Global ID dự đoán làm ground truth").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..geometry import bbox_to_polygon, clip_polygon, polygon_area, polygon_coverage, polygon_iou

V_PARKED_TRUE = 0.35          # world units / s below which a vehicle is standing
GT_SLOT_COVERAGE = 0.60       # PLAN 2 §5.1 Coverage = |V ∩ S| / |V|
GT_APPROACH_COVERAGE = 0.25


class GTPhase(str, Enum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    MERGED_WITH_OTHER_VEHICLE = "MERGED_WITH_OTHER_VEHICLE"
    OUTSIDE_CAMERA = "OUTSIDE_CAMERA"
    IN_HANDOFF_ZONE = "IN_HANDOFF_ZONE"
    PARKING_APPROACH = "PARKING_APPROACH"
    PARKED = "PARKED"
    DEPARTING = "DEPARTING"
    EXITED = "EXITED"


@dataclass
class GroundTruthRecord:
    frame_index: int
    timestamp: float
    physical_vehicle_id: str
    camera_id: str
    bbox: np.ndarray | None
    world_anchor: np.ndarray
    footprint: np.ndarray
    phase: GTPhase
    visibility_state: str
    occlusion_state: str
    slot_id: str | None = None
    handoff_source: str | None = None
    handoff_target: str | None = None
    visible_fraction: float = 1.0

    @property
    def observable(self) -> bool:
        """True when a correct system is expected to report this vehicle."""
        return self.phase not in (GTPhase.OUTSIDE_CAMERA, GTPhase.EXITED)


@dataclass
class HandoffTruth:
    physical_vehicle_id: str
    source_camera: str
    target_camera: str
    t_last_source: float
    t_first_target: float


@dataclass
class SlotTruth:
    timestamp: float
    slot_id: str
    physical_vehicle_id: str | None
    coverage: float = 0.0
    changed: bool = False


@dataclass
class Annotator:
    layout: object
    vehicles: list = field(default_factory=list)

    def _image_rect(self, camera) -> np.ndarray:
        return bbox_to_polygon((0.0, 0.0, float(camera.width), float(camera.height)))

    def silhouette(self, camera, vehicle, t: float) -> np.ndarray:
        return camera.project_floor(vehicle.footprint(t))

    def visible_fraction(self, camera, vehicle, t: float) -> float:
        sil = self.silhouette(camera, vehicle, t)
        total = polygon_area(sil)
        if total <= 1e-9:
            return 0.0
        inside = clip_polygon(sil, self._image_rect(camera))
        frac = 0.0 if inside is None else polygon_area(inside) / total
        for blind in self.layout.blind_regions.get(camera.camera_id, []):
            hidden = clip_polygon(sil, camera.project_floor(blind))
            if hidden is not None:
                frac -= polygon_area(hidden) / total
        return float(max(0.0, min(1.0, frac)))

    def merged_with(self, camera, vehicle, t: float) -> str | None:
        mine = self.silhouette(camera, vehicle, t)
        for other in self.vehicles:
            if other.vehicle_id == vehicle.vehicle_id or not other.present(t):
                continue
            if polygon_iou(mine, self.silhouette(camera, other, t)) >= 0.08:
                return other.vehicle_id
        return None

    def slot_of(self, vehicle, t: float, threshold: float = GT_SLOT_COVERAGE) -> tuple[str | None, float]:
        best, best_cov = None, 0.0
        footprint = vehicle.footprint(t)
        for slot_id, poly in self.layout.slots.items():
            cov = polygon_coverage(footprint, poly)
            if cov > best_cov:
                best, best_cov = slot_id, cov
        return (best, best_cov) if best_cov >= threshold else (None, best_cov)

    def phase(self, camera, vehicle, t: float, visible_fraction: float) -> tuple[GTPhase, str | None]:
        speed = float(np.linalg.norm(vehicle.velocity(t)))
        slot_id, coverage = self.slot_of(vehicle, t)
        if slot_id is not None and speed <= V_PARKED_TRUE:
            return GTPhase.PARKED, slot_id
        if visible_fraction < 0.25:
            phase = GTPhase.OUTSIDE_CAMERA if not camera.sees_floor_point(
                vehicle.position(t)) else GTPhase.OCCLUDED
            return phase, slot_id
        if self.merged_with(camera, vehicle, t) is not None:
            return GTPhase.MERGED_WITH_OTHER_VEHICLE, slot_id
        if slot_id is not None:
            return GTPhase.DEPARTING, slot_id
        if coverage >= GT_APPROACH_COVERAGE:
            return GTPhase.PARKING_APPROACH, None
        zone = self.layout.topology.zones.get(camera.camera_id)
        pos = vehicle.position(t)
        if zone is not None and (zone.in_exit_corridor(pos) or zone.in_entry_corridor(pos)):
            return GTPhase.IN_HANDOFF_ZONE, None
        return GTPhase.VISIBLE, None

    def record(self, frame_index: int, camera, vehicle, t: float) -> GroundTruthRecord:
        frac = self.visible_fraction(camera, vehicle, t)
        phase, slot_id = self.phase(camera, vehicle, t, frac)
        sil = self.silhouette(camera, vehicle, t)
        bbox = None
        if frac > 0.0:
            bbox = np.array([sil[:, 0].min(), sil[:, 1].min(), sil[:, 0].max(), sil[:, 1].max()])
        merged = self.merged_with(camera, vehicle, t)
        return GroundTruthRecord(
            frame_index=frame_index, timestamp=t, physical_vehicle_id=vehicle.vehicle_id,
            camera_id=camera.camera_id, bbox=bbox, world_anchor=vehicle.position(t),
            footprint=vehicle.footprint(t), phase=phase,
            visibility_state="visible" if frac >= 0.6 else "partial" if frac > 0.0 else "hidden",
            occlusion_state="merged" if merged else ("occluded" if frac < 0.6 else "clear"),
            slot_id=slot_id, visible_fraction=frac)
