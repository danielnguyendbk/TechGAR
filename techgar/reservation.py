"""Phase E2 — Atomic 5-minute parking slot reservations.

Implements the plan's reservation invariants:
- Lease is atomic per slot (at most one active lease per slot).
- 5-minute lease duration with automatic renewal on tracking progress.
- Changing spots releases the old lease before acquiring the new one.
- If a vehicle parks in a different spot, the reservation is auto-swapped to the physical spot.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .persistence import PersistenceStore


class SlotConflictError(Exception):
    """Raised when requesting a slot that is already reserved or occupied."""


@dataclass
class SlotReservation:
    reservation_id: str
    session_id: str
    slot_id: str
    global_vehicle_id: int | None
    created_at: float
    lease_expires_at: float
    state: str = "active"

    @property
    def is_expired(self) -> bool:
        return time.time() > self.lease_expires_at


class ReservationManager:
    """Thread-safe manager for atomic parking slot reservations."""

    def __init__(self, store: PersistenceStore | None = None,
                 lease_duration: float = 300.0) -> None:
        self.store = store
        self.lease_duration = float(lease_duration)
        self._reservations_by_slot: dict[str, SlotReservation] = {}
        self._reservations_by_session: dict[str, str] = {}  # session_id -> slot_id
        self._lock = threading.RLock()

    def acquire_lease(self, session_id: str, slot_id: str,
                      global_vehicle_id: int | None = None,
                      now: float | None = None) -> SlotReservation:
        """Acquire a 5-minute lease on slot_id.

        If session_id already holds a lease on a different slot, that lease
        is atomically released.
        Raises SlotConflictError if slot_id is already held by another active lease.
        """
        moment = time.time() if now is None else float(now)
        with self._lock:
            # Check existing lease on target slot
            existing = self._reservations_by_slot.get(slot_id)
            if existing is not None and not existing.is_expired and existing.session_id != session_id:
                raise SlotConflictError(
                    f"Slot '{slot_id}' is already reserved by another session until "
                    f"{existing.lease_expires_at:.1f}"
                )

            # Release old slot if session is swapping spots
            old_slot = self._reservations_by_session.get(session_id)
            if old_slot is not None and old_slot != slot_id:
                self.release_lease(session_id)

            res_id = f"res_{secrets.token_hex(6)}"
            expires_at = moment + self.lease_duration
            reservation = SlotReservation(
                reservation_id=res_id,
                session_id=session_id,
                slot_id=slot_id,
                global_vehicle_id=global_vehicle_id,
                created_at=moment,
                lease_expires_at=expires_at,
                state="active",
            )
            self._reservations_by_slot[slot_id] = reservation
            self._reservations_by_session[session_id] = slot_id

            return reservation

    def release_lease(self, session_id: str) -> bool:
        """Release any active lease held by session_id."""
        with self._lock:
            slot_id = self._reservations_by_session.pop(session_id, None)
            if slot_id is not None:
                res = self._reservations_by_slot.pop(slot_id, None)
                if res is not None:
                    res.state = "released"
                return True
            return False

    def get_lease_for_session(self, session_id: str) -> SlotReservation | None:
        with self._lock:
            slot_id = self._reservations_by_session.get(session_id)
            if slot_id is None:
                return None
            res = self._reservations_by_slot.get(slot_id)
            if res is not None and res.is_expired:
                self.release_lease(session_id)
                return None
            return res

    def get_lease_for_slot(self, slot_id: str) -> SlotReservation | None:
        with self._lock:
            res = self._reservations_by_slot.get(slot_id)
            if res is not None and res.is_expired:
                self._reservations_by_slot.pop(slot_id, None)
                self._reservations_by_session.pop(res.session_id, None)
                return None
            return res

    def renew_lease(self, session_id: str, extra_seconds: float | None = None) -> bool:
        """Extend lease duration if session is actively progressing."""
        duration = float(extra_seconds if extra_seconds is not None else self.lease_duration)
        with self._lock:
            res = self.get_lease_for_session(session_id)
            if res is None:
                return False
            res.lease_expires_at = time.time() + duration
            return True

    def auto_swap_on_parked(self, session_id: str, actual_slot_id: str) -> SlotReservation:
        """Auto-swap reservation to actual physical parking spot when vehicle parks."""
        with self._lock:
            # Force release any previous slot and claim actual slot
            self.release_lease(session_id)
            # Remove any conflicting stale lease on actual slot
            self._reservations_by_slot.pop(actual_slot_id, None)
            return self.acquire_lease(session_id, actual_slot_id)
