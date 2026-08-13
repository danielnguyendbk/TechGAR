"""
Canonical Session model for TechGAR parking system.

Session lifecycle:
  WAITING_FOR_SCAN    →  Vehicle at gate, driver hasn't scanned QR yet
  SELECTING_SPOT      →  Driver scanned QR, choosing a parking spot
  NAVIGATING_TO_SPOT  →  Vehicle is moving towards the chosen spot
  PARKED              →  Vehicle is parked (track may be lost, session persists)
  EXIT_NAVIGATION     →  Driver clicked "exit" – navigating to the exit gate
  CLOSED              →  Vehicle has exited the lot, session complete

Key design decisions (review fixes #8, #10, #12):
  - sessionId is a random token (NOT track_id)
  - parkedSpotId is set by the Binder, not by target occupancy
  - Session is ONLY closed by an EXIT event, never by lost-track
"""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class SessionState(str, Enum):
    WAITING_FOR_SCAN = "WAITING_FOR_SCAN"
    SELECTING_SPOT = "SELECTING_SPOT"
    NAVIGATING_TO_SPOT = "NAVIGATING_TO_SPOT"
    PARKED = "PARKED"
    EXIT_NAVIGATION = "EXIT_NAVIGATION"
    CLOSED = "CLOSED"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _generate_session_id(length: int = 8) -> str:
    """Generate a short random session token (e.g. 'K8F4W9X2')."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass
class Session:
    sessionId: str = field(default_factory=_generate_session_id)
    globalVehicleId: Optional[int] = None
    state: SessionState = SessionState.WAITING_FOR_SCAN

    targetSpotId: Optional[str] = None
    parkedSpotId: Optional[str] = None
    entryGateId: Optional[str] = None

    createdAt: str = field(default_factory=_now_iso)
    claimedAt: Optional[str] = None
    parkedAt: Optional[str] = None
    exitStartedAt: Optional[str] = None
    closedAt: Optional[str] = None

    def claim(self) -> bool:
        """Driver scanned QR. Only valid from WAITING_FOR_SCAN."""
        if self.state != SessionState.WAITING_FOR_SCAN:
            return False
        self.state = SessionState.SELECTING_SPOT
        self.claimedAt = _now_iso()
        return True

    def select_spot(self, spot_id: Optional[str]) -> bool:
        """Driver chose a target spot (or None to deselect)."""
        if self.state not in (SessionState.SELECTING_SPOT, SessionState.NAVIGATING_TO_SPOT):
            return False
        if spot_id:
            self.state = SessionState.NAVIGATING_TO_SPOT
            self.targetSpotId = spot_id
        else:
            self.state = SessionState.SELECTING_SPOT
            self.targetSpotId = None
        return True

    def set_parked(self, parked_spot_id: str) -> bool:
        """Vision Binder confirmed vehicle is parked at this spot."""
        if self.state in (SessionState.CLOSED,):
            return False
        self.state = SessionState.PARKED
        self.parkedSpotId = parked_spot_id
        self.parkedAt = _now_iso()
        return True

    def start_exit(self) -> bool:
        """Driver clicked 'retrieve car'. Spot remains occupied until vision confirms departure."""
        if self.state != SessionState.PARKED:
            return False
        self.state = SessionState.EXIT_NAVIGATION
        self.exitStartedAt = _now_iso()
        # NOTE: parkedSpotId stays — spot is still occupied until vision confirms
        return True

    def close(self) -> bool:
        """Vehicle exited the lot (confirmed by vision EXIT event)."""
        if self.state == SessionState.CLOSED:
            return False
        self.state = SessionState.CLOSED
        self.closedAt = _now_iso()
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        data = dict(data)  # copy
        if "state" in data:
            data["state"] = SessionState(data["state"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
