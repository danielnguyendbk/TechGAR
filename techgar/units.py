"""Canonical units and the clock contract (PLAN 1 Phase 0, work item 1).

Two rules the whole system obeys:

1. **Time is the primary axis.**  Every timestamp, TTL, grace window and
   evidence window is a float number of *seconds* taken from a monotonic
   source.  Frame counts are never used for lifecycle decisions
   (PLAN 1 §6.2, PLAN 2 §2.5).
2. **Space is the calibrated floor plane.**  Positions are 2-vectors in
   *world units* on the parking floor.  There is no latitude/longitude and no
   GPS input anywhere in the codebase (PLAN 1 §1.1 problem 5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# --- time -------------------------------------------------------------------
SECOND = 1.0
MILLISECOND = 1.0e-3


def ms(value: float) -> float:
    """Milliseconds -> canonical seconds."""
    return value * MILLISECOND


def to_ms(seconds: float) -> float:
    """Canonical seconds -> milliseconds (reporting only)."""
    return seconds / MILLISECOND


# --- space ------------------------------------------------------------------
WORLD_UNIT = 1.0  # one calibrated unit on the floor plane (metre by default)
CENTIMETRE = 0.01
METRE = 1.0

#: Name of the world coordinate frame, published in every snapshot so that
#: consumers can never confuse it with a geographic frame (rubric A).
WORLD_FRAME_NAME = "techgar.floor_plane.local"
WORLD_FRAME_DOC = (
    "Right-handed 2-D frame on the parking floor plane. Origin = calibration "
    "marker 0. +X along the main drive lane, +Y towards the slot rows. "
    "Units: world units (metres by default). No geographic reference exists."
)


@dataclass
class Clock:
    """Monotonic clock with an injectable value, so tests own time exactly."""

    _offset: float = 0.0
    _manual: float | None = field(default=None, repr=False)

    def now(self) -> float:
        if self._manual is not None:
            return self._manual + self._offset
        return time.perf_counter() + self._offset

    def set(self, value: float) -> None:
        """Pin the clock (test / replay mode)."""
        self._manual = value

    def advance(self, dt: float) -> float:
        if self._manual is None:
            raise RuntimeError("advance() requires a pinned clock; call set() first")
        self._manual += dt
        return self.now()
