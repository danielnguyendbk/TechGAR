"""
Canonical Vehicle model for TechGAR parking system.

Key design decisions (review fixes #6, #9, #29, #37):
  - Uses globalVehicleId as the single cross-camera identity
  - trackingState has a clear lifecycle: TRACKING → PARKED → TRACKING_EXIT → EXITED
  - position can be null when vehicle is PARKED (motion tracker lost it)
  - Frontend uses displayPosition which falls back to spot center
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class VehicleTrackingState(str, Enum):
    TRACKING = "TRACKING"
    PARKED = "PARKED"
    TRACKING_EXIT = "TRACKING_EXIT"
    EXITED = "EXITED"


@dataclass
class Vehicle:
    globalVehicleId: int
    trackingState: VehicleTrackingState = VehicleTrackingState.TRACKING

    # null when vehicle is parked (motion tracker lost it)
    position: Optional[dict] = None  # {"x": float, "y": float}

    cameraIds: List[str] = field(default_factory=list)
    parkedSpotId: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "globalVehicleId": self.globalVehicleId,
            "trackingState": self.trackingState.value,
            "position": self.position,
            "cameraIds": self.cameraIds,
            "parkedSpotId": self.parkedSpotId,
        }
