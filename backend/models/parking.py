"""
Canonical Parking Spot model for TechGAR parking system.

Key design decisions (review fixes #7, #30, #31, #39):
  - vehicleId is the globalVehicleId from the Binder
  - vehicleId=null + occupied=true means pre-existing car (not tracked from entry)
  - visionOccupied: raw detection from ParkingDetector
  - trackingOccupied: true if Binder has bound a vehicle to this slot
  - Final occupied = visionOccupied OR trackingOccupied
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParkingSpot:
    spotId: str
    occupied: bool = False
    vehicleId: Optional[int] = None  # globalVehicleId from Binder

    visionOccupied: bool = False    # raw from ParkingDetector
    trackingOccupied: bool = False  # from SlotVehicleBinder

    def to_dict(self) -> dict:
        return {
            "spotId": self.spotId,
            "occupied": self.occupied,
            "vehicleId": self.vehicleId,
            "visionOccupied": self.visionOccupied,
            "trackingOccupied": self.trackingOccupied,
        }
