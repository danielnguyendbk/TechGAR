"""Phase 5 — session protection.

A session is bound to a Global ID *and* to a persistent vehicle fingerprint, so an
audited identity re-map can move it without any user action (no second QR scan).
Deleting a session requires a confirmed physical exit: a short tracking gap must
never close a session while the vehicle is still inside the facility
(PLAN 1 Phase 5, PLAN 3 §9 "Session survival = 100%").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .appearance import cosine_distance
from .config_world import SessionConfig
from .registry import GlobalIdentityRegistry
from .states import IdentityEventType, LifecycleState


@dataclass
class VehicleFingerprint:
    appearance: np.ndarray | None = None
    footprint_area: float = 0.0
    aspect: float = 1.0
    entry_timestamp: float = 0.0
    entry_camera: str = ""
    last_slot: str | None = None
    last_position: np.ndarray | None = None

    def distance(self, other: "VehicleFingerprint") -> float:
        appearance = cosine_distance(self.appearance, other.appearance)
        area = abs(np.log(max(self.footprint_area, 1e-6) / max(other.footprint_area, 1e-6)))
        return float(appearance + 0.25 * area)


@dataclass
class SessionRecord:
    session_id: str
    global_id: int | None
    fingerprint: VehicleFingerprint
    created_at: float
    closed_at: float | None = None
    state: str = "active"
    remaps: list[tuple[float, int | None, int]] = field(default_factory=list)
    orphaned_at: float | None = None

    @property
    def resolvable(self) -> bool:
        return self.state in ("active", "orphan")


class SessionRegistry:
    def __init__(self, registry: GlobalIdentityRegistry,
                 config: SessionConfig | None = None) -> None:
        self.registry = registry
        self.config = config or SessionConfig()
        self.sessions: dict[str, SessionRecord] = {}
        self.blocked_deletions: list[tuple[float, str, str]] = []

    # --- binding ------------------------------------------------------------
    def bind(self, session_id: str, global_id: int, timestamp: float,
             frame_sequence: int) -> SessionRecord:
        state = self.registry.get(global_id)
        fingerprint = VehicleFingerprint(
            appearance=None if state is None else (state.appearance_gallery.centroid
                                                  if state.appearance_gallery else None),
            footprint_area=0.0 if state is None else state.footprint_area,
            aspect=1.0 if state is None else state.footprint_aspect,
            entry_timestamp=timestamp,
            entry_camera="" if state is None else state.latest_camera,
            last_slot=None if state is None else state.slot_id,
            last_position=None if state is None else np.asarray(state.latest_world_position))
        record = SessionRecord(session_id, global_id, fingerprint, timestamp)
        self.sessions[session_id] = record
        self.registry.bind_session(global_id, session_id, timestamp, frame_sequence)
        return record

    def resolve(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def global_id_for(self, session_id: str) -> int | None:
        record = self.sessions.get(session_id)
        return None if record is None or record.state == "closed" else record.global_id

    def sessions_for(self, global_id: int) -> list[str]:
        return [s.session_id for s in self.sessions.values()
                if s.global_id == global_id and s.state != "closed"]

    def accessible(self, session_id: str) -> bool:
        """True when the session still resolves to a live identity or a held orphan."""
        record = self.sessions.get(session_id)
        if record is None or record.state == "closed":
            return False
        if record.state == "orphan":
            return True
        state = self.registry.identities.get(record.global_id)
        return state is not None and state.lifecycle_state.is_live

    # --- maintenance --------------------------------------------------------
    def remap(self, session_id: str, new_global_id: int, timestamp: float, frame_sequence: int,
              evidence: dict | None = None) -> bool:
        record = self.sessions.get(session_id)
        if record is None or record.state == "closed":
            return False
        previous = record.global_id
        record.remaps.append((timestamp, previous, new_global_id))
        record.global_id = new_global_id
        record.state = "active"
        record.orphaned_at = None
        self.registry.bind_session(new_global_id, session_id, timestamp, frame_sequence)
        self.registry.events.append(timestamp, frame_sequence, IdentityEventType.SESSION_REMAP,
                                    new_global_id, detail=f"{session_id}:{previous}->{new_global_id}",
                                    evidence=evidence or {})
        return True

    def sweep(self, timestamp: float, frame_sequence: int) -> list[str]:
        """Detach sessions whose identity vanished, then try to re-attach them."""
        orphaned = []
        for record in self.sessions.values():
            if record.state != "active" or record.global_id is None:
                continue
            state = self.registry.identities.get(record.global_id)
            if state is not None and state.lifecycle_state.is_live:
                if state.appearance_gallery is not None and state.appearance_gallery.centroid is not None:
                    record.fingerprint.appearance = state.appearance_gallery.centroid
                record.fingerprint.footprint_area = state.footprint_area or record.fingerprint.footprint_area
                record.fingerprint.last_slot = state.slot_id or record.fingerprint.last_slot
                record.fingerprint.last_position = np.asarray(state.latest_world_position)
                continue
            retired = self.registry.retired_identities.get(record.global_id)
            if retired is not None and retired.lifecycle_state is LifecycleState.RETIRED \
                    and self._exited(record.global_id):
                self.close(record.session_id, timestamp, frame_sequence, exit_confirmed=True)
                continue
            record.state = "orphan"
            record.orphaned_at = timestamp
            orphaned.append(record.session_id)
        self.reattach(timestamp, frame_sequence)
        return orphaned

    def _exited(self, global_id: int) -> bool:
        return any(e.event_type is IdentityEventType.EXIT and e.global_id == global_id
                   for e in self.registry.events)

    def reattach(self, timestamp: float, frame_sequence: int, max_distance: float = 0.55) -> int:
        """Re-bind orphan sessions using fingerprint + slot + topology evidence."""
        reattached = 0
        for record in self.sessions.values():
            if record.state != "orphan":
                continue
            best, best_score = None, float("inf")
            for state in self.registry.live():
                if state.session_ids:
                    continue
                candidate = VehicleFingerprint(
                    appearance=state.appearance_gallery.centroid if state.appearance_gallery else None,
                    footprint_area=state.footprint_area, aspect=state.footprint_aspect)
                score = record.fingerprint.distance(candidate)
                if record.fingerprint.last_slot and state.slot_id == record.fingerprint.last_slot:
                    score *= 0.4                      # slot ownership is strong evidence
                if record.fingerprint.last_position is not None:
                    gap = float(np.linalg.norm(np.asarray(state.latest_world_position)
                                               - record.fingerprint.last_position))
                    score += 0.05 * gap
                if score < best_score:
                    best, best_score = state.global_id, score
            if best is not None and best_score <= max_distance:
                self.remap(record.session_id, best, timestamp, frame_sequence,
                           evidence={"fingerprint_distance": best_score, "reason": "reattach"})
                reattached += 1
        return reattached

    def close(self, session_id: str, timestamp: float, frame_sequence: int,
              exit_confirmed: bool = False) -> bool:
        record = self.sessions.get(session_id)
        if record is None:
            return False
        if self.config.require_exit_for_delete and not exit_confirmed:
            self.blocked_deletions.append((timestamp, session_id, "exit_not_confirmed"))
            self.registry.events.append(timestamp, frame_sequence,
                                        IdentityEventType.SESSION_DELETE_BLOCKED,
                                        record.global_id, detail=session_id)
            return False
        record.state = "closed"
        record.closed_at = timestamp
        self.registry.events.append(timestamp, frame_sequence, IdentityEventType.SESSION_CLOSE,
                                    record.global_id, detail=session_id)
        return True
