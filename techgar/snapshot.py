"""Stage 10 — immutable runtime snapshot for the frontend.

The frontend is a *consumer*: it renders what this snapshot says and never decides
identity (PLAN 1 §2, §6.6).  Two display rules the plan is explicit about:

* a vehicle that missed one update keeps its marker and its Global ID; after the
  display hold it changes visual state to ``temporarily_missing`` — it never
  re-appears under a different Global ID (stage 10 Pass/Fail);
* a parked vehicle is always represented by its slot plus its Global ID, even
  though it has stopped producing motion evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .states import DisplayState, GlobalVehicleState, LifecycleState, ParkingSlotState
from .units import WORLD_FRAME_DOC, WORLD_FRAME_NAME

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class VehicleView:
    global_id: int
    display_state: DisplayState
    lifecycle_state: LifecycleState
    world_position: tuple[float, float]
    uncertainty: float
    velocity: tuple[float, float]
    camera_id: str
    slot_id: str | None
    session_ids: tuple[str, ...]
    last_observed: float
    footprint: tuple[tuple[float, float], ...] = ()
    observed: bool = True
    stale_seconds: float = 0.0
    display_hold_seconds: float = 6.0


@dataclass(frozen=True)
class SlotView:
    slot_id: str
    occupancy_state: str
    owning_global_id: int | None
    overlap_score: float
    dwell_duration: float
    confirmation_confidence: float
    vision_occupied: bool = False


@dataclass(frozen=True)
class RuntimeSnapshot:
    sequence: int
    timestamp: float
    world_frame: str
    vehicles: tuple[VehicleView, ...]
    slots: tuple[SlotView, ...]
    camera_health: dict
    identity_events: tuple[dict, ...]
    latency: dict = field(default_factory=dict)
    overload: bool = False
    gps_used: bool = False
    published_at: float | None = None
    schema_version: str = SCHEMA_VERSION
    slot_layout: tuple[dict, ...] = ()

    def vehicle(self, global_id: int) -> VehicleView | None:
        for view in self.vehicles:
            if view.global_id == global_id:
                return view
        return None

    def to_dict(self) -> dict:
        """Return the versioned wire contract consumed by PLAN 4/5/6."""
        published_at = self.timestamp if self.published_at is None else self.published_at
        cameras = {}
        for camera_id, health in self.camera_health.items():
            value = dict(health) if isinstance(health, dict) else {"online": bool(health)}
            last = value.get("last_timestamp")
            value.setdefault("online", bool(value.get("frames", 0)) and last is not None
                             and self.timestamp - float(last) <= 2.0)
            cameras[camera_id] = value
        payload = {
            "schema_version": self.schema_version,
            "frame_index": self.sequence,
            "published_at": published_at,
            "sequence": self.sequence, "timestamp": self.timestamp,
            "world_frame": {"name": self.world_frame, "description": WORLD_FRAME_DOC},
            "gps_used": self.gps_used, "overload": self.overload,
            "vehicles": [{
                "state": v.display_state.value,
                "observed": v.observed,
                "parked_slot_id": v.slot_id,
                "stale_seconds": v.stale_seconds,
                "display_hold_seconds": v.display_hold_seconds,
                "position": list(v.world_position),
                "global_id": v.global_id, "display_state": v.display_state.value,
                "lifecycle_state": v.lifecycle_state.value,
                "world_position": list(v.world_position), "uncertainty": v.uncertainty,
                "velocity": list(v.velocity), "camera_id": v.camera_id, "slot_id": v.slot_id,
                "session_ids": list(v.session_ids), "last_observed": v.last_observed,
                "footprint": [list(point) for point in v.footprint],
            } for v in self.vehicles],
            "parking_slots": [{
                "slot_id": s.slot_id, "occupancy_state": s.occupancy_state,
                "occupied": s.occupancy_state == "occupied",
                "status": s.occupancy_state,
                "owning_global_id": s.owning_global_id, "overlap_score": s.overlap_score,
                "dwell_duration": s.dwell_duration,
                "confirmation_confidence": s.confirmation_confidence,
                "vision_occupied": s.vision_occupied,
            } for s in self.slots],
            "slots": [{
                "slot_id": s.slot_id, "occupancy_state": s.occupancy_state,
                "owning_global_id": s.owning_global_id, "overlap_score": s.overlap_score,
                "dwell_duration": s.dwell_duration,
                "confirmation_confidence": s.confirmation_confidence,
                "vision_occupied": s.vision_occupied,
            } for s in self.slots],
            "slot_layout": list(self.slot_layout),
            "cameras": cameras,
            "camera_health": self.camera_health,
            "identity_events": list(self.identity_events),
            "latency": self.latency,
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)


class SnapshotPublisher:
    def __init__(self, display_hold: float = 6.0, event_tail: int = 12) -> None:
        self.display_hold = display_hold
        self.event_tail = event_tail
        self.sequence = 0
        self.last: RuntimeSnapshot | None = None

    def display_state(self, state: GlobalVehicleState, now: float) -> DisplayState:
        if state.lifecycle_state is LifecycleState.PARKED:
            return DisplayState.PARKED
        missing = state.missing_duration(now)
        if state.lifecycle_state in (LifecycleState.TEMPORARILY_MISSING, LifecycleState.OCCLUDED):
            return (DisplayState.TEMPORARILY_MISSING if missing <= self.display_hold
                    else DisplayState.HIDDEN)
        if state.lifecycle_state is LifecycleState.ACTIVE:
            return DisplayState.OBSERVED
        return DisplayState.HIDDEN

    def publish(self, identities: list[GlobalVehicleState], slots: dict[str, ParkingSlotState],
                timestamp: float, camera_health: dict, events, latency: dict | None = None,
                overload: bool = False, published_at: float | None = None,
                slot_layout: dict[str, np.ndarray] | None = None) -> RuntimeSnapshot:
        self.sequence += 1
        vehicles = []
        for state in identities:
            if not state.lifecycle_state.is_published:
                continue
            display = self.display_state(state, timestamp)
            footprint = ()
            if state.latest_footprint is not None:
                footprint = tuple(tuple(float(c) for c in point)
                                  for point in np.asarray(state.latest_footprint))
            vehicles.append(VehicleView(
                global_id=state.global_id, display_state=display,
                lifecycle_state=state.lifecycle_state,
                world_position=(float(state.latest_world_position[0]),
                                float(state.latest_world_position[1])),
                uncertainty=float(state.uncertainty),
                velocity=(float(state.velocity[0]), float(state.velocity[1])),
                camera_id=state.latest_camera, slot_id=state.slot_id,
                session_ids=tuple(state.session_ids),
                last_observed=float(state.last_observed_timestamp), footprint=footprint,
                observed=display is DisplayState.OBSERVED,
                stale_seconds=float(state.missing_duration(timestamp)),
                display_hold_seconds=float(self.display_hold)))
        slot_views = tuple(SlotView(
            slot_id=s.slot_id, occupancy_state=s.occupancy_state.value,
            owning_global_id=s.owning_global_id, overlap_score=float(s.overlap_score),
            dwell_duration=float(s.dwell_duration),
            confirmation_confidence=float(s.confirmation_confidence),
            vision_occupied=bool(getattr(s, "vision_occupied", False)))
            for s in slots.values())
        event_views = tuple({"event_id": e.event_id, "timestamp": e.timestamp,
                             "frame_sequence": e.frame_sequence, "type": e.event_type.value,
                             "global_id": e.global_id, "detail": e.detail}
                            for e in list(events)[-self.event_tail:])
        snapshot = RuntimeSnapshot(
            sequence=self.sequence, timestamp=timestamp, world_frame=WORLD_FRAME_NAME,
            vehicles=tuple(sorted(vehicles, key=lambda v: v.global_id)), slots=slot_views,
            camera_health=camera_health, identity_events=event_views,
            latency=dict(latency or {}), overload=overload, gps_used=False,
            published_at=published_at,
            slot_layout=tuple({"slot_id": slot_id, "camera_id": "shared",
                               "polygon": np.asarray(polygon, dtype=float).tolist()}
                              for slot_id, polygon in (slot_layout or {}).items()))
        self.last = snapshot
        return snapshot
