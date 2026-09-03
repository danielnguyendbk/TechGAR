"""Phase F1 — Unified RuntimeCoordinator facade.

Single source of truth uniting:
- Pipeline tracking & slot engine
- SQLite WAL persistence store
- QR token lifecycle manager
- Atomic slot reservation manager
- State machine v2 validation:
  WAITING_FOR_SCAN -> SELECTING_SPOT -> NAVIGATING -> PARKED -> EXIT_NAVIGATION -> CLOSED
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .persistence import PersistenceStore
from .qr import QRTokenManager
from .reservation import ReservationManager, SlotConflictError
from .snapshot import RuntimeSnapshot, SCHEMA_VERSION
from .units import WORLD_FRAME_NAME


VALID_TRANSITIONS: dict[str, set[str]] = {
    "WAITING_FOR_SCAN": {"SELECTING_SPOT", "NAVIGATING", "CLOSED"},
    "SELECTING_SPOT": {"NAVIGATING", "CLOSED"},
    "NAVIGATING": {"PARKED", "SELECTING_SPOT", "EXIT_NAVIGATION", "CLOSED"},
    "PARKED": {"EXIT_NAVIGATION", "CLOSED"},
    "EXIT_NAVIGATION": {"CLOSED"},  # Backwards transition to PARKED is strictly forbidden
    "CLOSED": set(),
}


@dataclass
class CoordinatorSession:
    session_id: str
    state: str = "WAITING_FOR_SCAN"
    global_vehicle_id: int | None = None
    target_spot_id: str | None = None
    parked_spot_id: str | None = None
    qr_token_hash: str | None = None
    qr_expires_at: float | None = None
    claimed_at: float | None = None
    parked_at: float | None = None
    exit_started_at: float | None = None
    closed_at: float | None = None
    revision: int = 1
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "state": self.state,
            "globalVehicleId": self.global_vehicle_id,
            "targetSpotId": self.target_spot_id,
            "parkedSpotId": self.parked_spot_id,
            "qrExpiresAt": self.qr_expires_at,
            "claimedAt": self.claimed_at,
            "parkedAt": self.parked_at,
            "exitStartedAt": self.exit_started_at,
            "closedAt": self.closed_at,
            "revision": self.revision,
            "updatedAt": self.updated_at,
        }


class RuntimeCoordinator:
    """The unified runtime facade. Replaces the fragmented in-memory stores."""

    def __init__(self, pipeline=None, store: PersistenceStore | None = None,
                 site_id: str = "default_site") -> None:
        self.pipeline = pipeline
        self.site_id = str(site_id)
        self.store = store or PersistenceStore(":memory:", site_id=site_id)
        self.qr_manager = QRTokenManager(self.store, site_id=site_id)
        self.reservation_manager = ReservationManager(self.store)
        self.sessions: dict[str, CoordinatorSession] = {}
        self.gate_points: list[tuple[float, float]] = []
        self._snapshot: RuntimeSnapshot | None = None
        self._lock = threading.RLock()

    # --- Snapshots -----------------------------------------------------------

    def set_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        with self._lock:
            if self._snapshot is not None and snapshot.sequence < self._snapshot.sequence:
                raise ValueError("snapshot sequence must be monotonic")
            self._snapshot = snapshot

    def current_snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            if self.pipeline is not None and self.pipeline.publisher.last is not None:
                return self.pipeline.publisher.last
            if self._snapshot is not None:
                return self._snapshot
            return RuntimeSnapshot(
                sequence=0,
                timestamp=time.time(),
                world_frame=WORLD_FRAME_NAME,
                vehicles=(),
                slots=(),
                camera_health={"cam1": {"online": True}, "cam2": {"online": True}},
                identity_events=(),
                published_at=time.time(),
            )

    # --- Session State Transitions (Validated) --------------------------------

    def _transition_session(self, session: CoordinatorSession, new_state: str) -> None:
        if new_state == session.state:
            return
        allowed = VALID_TRANSITIONS.get(session.state, set())
        if new_state not in allowed:
            raise ValueError(f"Invalid state transition: '{session.state}' -> '{new_state}'")
        session.state = new_state
        session.revision += 1
        session.updated_at = time.time()
        if new_state == "PARKED":
            session.parked_at = session.updated_at
        elif new_state == "EXIT_NAVIGATION":
            session.exit_started_at = session.updated_at
        elif new_state == "CLOSED":
            session.closed_at = session.updated_at

    # --- QR & Session Claim ---------------------------------------------------

    def claim_via_token(self, token: str) -> CoordinatorSession:
        """Driver scans kiosk QR code. Token is validated and claimed."""
        with self._lock:
            session_id, global_vehicle_id = self.qr_manager.claim_token(token)
            existing = self.sessions.get(session_id)
            if existing is not None:
                return existing

            session = CoordinatorSession(
                session_id=session_id,
                state="SELECTING_SPOT",
                global_vehicle_id=global_vehicle_id,
                claimed_at=time.time(),
            )
            self.sessions[session_id] = session

            if self.pipeline is not None and global_vehicle_id is not None:
                self.pipeline.sessions.bind(
                    session_id, global_vehicle_id, session.claimed_at,
                    getattr(self.pipeline, "_frame_sequence", 0),
                )
            return session

    def get_session(self, session_id: str) -> CoordinatorSession | None:
        with self._lock:
            return self.sessions.get(session_id)

    # --- Slot Reservation ----------------------------------------------------

    def select_spot_reservation(self, session_id: str, spot_id: str) -> CoordinatorSession:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None or session.state == "CLOSED":
                raise KeyError(session_id)

            # Validate slot exists if snapshot has slots
            valid_slots = {slot.slot_id for slot in self.current_snapshot().slots}
            if valid_slots and spot_id not in valid_slots:
                raise ValueError(f"Unknown parking spot: {spot_id}")

            # Acquire atomic lease
            self.reservation_manager.acquire_lease(
                session_id=session_id,
                slot_id=spot_id,
                global_vehicle_id=session.global_vehicle_id,
            )

            session.target_spot_id = spot_id
            self._transition_session(session, "NAVIGATING")
            return session

    def cancel_reservation(self, session_id: str) -> CoordinatorSession:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            self.reservation_manager.release_lease(session_id)
            session.target_spot_id = None
            if session.state == "NAVIGATING":
                self._transition_session(session, "SELECTING_SPOT")
            return session

    # --- Slot Engine Internal Integration -------------------------------------

    def on_slot_parked(self, global_id: int, physical_slot_id: str) -> None:
        """Internal callback invoked strictly by Stage 9 Slot Engine."""
        with self._lock:
            for session in self.sessions.values():
                if session.global_vehicle_id == global_id and session.state in ("SELECTING_SPOT", "NAVIGATING"):
                    # Auto swap reservation if parked in different slot
                    self.reservation_manager.auto_swap_on_parked(session.session_id, physical_slot_id)
                    session.parked_spot_id = physical_slot_id
                    session.target_spot_id = physical_slot_id
                    self._transition_session(session, "PARKED")
                    break

    def request_exit(self, session_id: str) -> CoordinatorSession:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None or session.state == "CLOSED":
                raise KeyError(session_id)
            self._transition_session(session, "EXIT_NAVIGATION")
            return session

    def on_physical_exit(self, global_id: int) -> None:
        """Internal callback triggered when vehicle crosses exit gate."""
        with self._lock:
            for session in self.sessions.values():
                if session.global_vehicle_id == global_id and session.state != "CLOSED":
                    self.reservation_manager.release_lease(session.session_id)
                    self._transition_session(session, "CLOSED")

    # --- Resets (Soft & Close-all) --------------------------------------------

    def soft_reset(self) -> dict[str, Any]:
        """Soft reset preserves GID sequence, parked vehicles, and sessions."""
        with self._lock:
            now = time.time()
            seq = getattr(self.pipeline, "_frame_sequence", 0) + 1 if self.pipeline else 1
            if self.pipeline is not None:
                res = self.pipeline.registry.soft_reset(now, seq)
            else:
                res = {"soft_reset": True}
            return res

    def close_all(self, confirm: bool = False) -> dict[str, Any]:
        """Two-step confirmed hard reset: closes active sessions & resets in-memory tracks,

        Preserves GID sequence counter per the plan invariants.
        """
        if not confirm:
            raise ValueError("close-all requires explicit confirm=True")
        with self._lock:
            now = time.time()
            seq = getattr(self.pipeline, "_frame_sequence", 0) + 1 if self.pipeline else 1
            if self.pipeline is not None:
                retired = self.pipeline.registry.reset(now, seq)
            else:
                retired = 0
            # Close all active sessions
            for s in self.sessions.values():
                if s.state != "CLOSED":
                    s.state = "CLOSED"
                    s.closed_at = now
            return {"closed_all": True, "retired_identities": retired}
