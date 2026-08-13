"""
Runtime state for the TechGAR backend.

This is the single source of truth that replaces shared JSON files.
Vision pushes updates here; backend API reads from here; frontend queries the API.

Review fixes #18 (no writing to frontend/public), #26 (no shared filesystem),
#45 (separate business state from debug).
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from models.session import Session, SessionState
from models.vehicle import Vehicle, VehicleTrackingState
from models.parking import ParkingSpot


class RuntimeState:
    """Thread-safe in-memory state for the parking system."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}             # sessionId -> Session
        self._vehicles: Dict[int, Vehicle] = {}              # globalVehicleId -> Vehicle
        self._spots: Dict[str, ParkingSpot] = {}             # spotId -> ParkingSpot
        self._vehicle_session_map: Dict[int, str] = {}       # globalVehicleId -> sessionId
        self._events: List[dict] = []                        # recent events

    # ── Session management ──

    def create_session(self, global_vehicle_id: int, entry_gate_id: str = "ENTRY_1") -> Session:
        with self._lock:
            session = Session(
                globalVehicleId=global_vehicle_id,
                entryGateId=entry_gate_id,
            )
            self._sessions[session.sessionId] = session
            self._vehicle_session_map[global_vehicle_id] = session.sessionId
            self._add_event("session_created", {
                "sessionId": session.sessionId,
                "globalVehicleId": global_vehicle_id,
            })
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_session_by_vehicle(self, global_vehicle_id: int) -> Optional[Session]:
        with self._lock:
            sid = self._vehicle_session_map.get(global_vehicle_id)
            return self._sessions.get(sid) if sid else None

    def get_all_sessions(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: s.to_dict() for sid, s in self._sessions.items()}

    def get_waiting_sessions(self, gate_id: Optional[str] = None) -> List[Session]:
        """Get sessions waiting for QR scan, optionally filtered by gate."""
        with self._lock:
            result = [
                s for s in self._sessions.values()
                if s.state == SessionState.WAITING_FOR_SCAN
            ]
            if gate_id:
                result = [s for s in result if s.entryGateId == gate_id]
            # Sort by creation time so kiosk picks the latest
            result.sort(key=lambda s: s.createdAt)
            return result

    def claim_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.claim():
                return s
            return None

    def select_spot(self, session_id: str, spot_id: Optional[str]) -> Optional[Session]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.select_spot(spot_id):
                return s
            return None

    def set_session_parked(self, session_id: str, parked_spot_id: str) -> Optional[Session]:
        """Called when Binder confirms vehicle is in a slot."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.set_parked(parked_spot_id):
                self._add_event("session_parked", {
                    "sessionId": session_id,
                    "parkedSpotId": parked_spot_id,
                })
                return s
            return None

    def start_exit(self, session_id: str) -> Optional[Session]:
        """User clicked 'retrieve car'. Spot stays occupied."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.start_exit():
                self._add_event("session_exit_started", {
                    "sessionId": session_id,
                    "parkedSpotId": s.parkedSpotId,
                })
                return s
            return None

    def close_session(self, session_id: str) -> Optional[Session]:
        """Only called on EXIT event from vision. Never on lost track."""
        with self._lock:
            s = self._sessions.get(session_id)
            if s and s.close():
                self._add_event("session_closed", {"sessionId": session_id})
                return s
            return None

    # ── Vehicle management ──

    def update_vehicle(self, global_vehicle_id: int, position: Optional[dict],
                       camera_ids: List[str], parked_spot_id: Optional[str] = None) -> Vehicle:
        with self._lock:
            v = self._vehicles.get(global_vehicle_id)
            if v is None:
                v = Vehicle(globalVehicleId=global_vehicle_id)
                self._vehicles[global_vehicle_id] = v

            v.position = position
            v.cameraIds = camera_ids

            if parked_spot_id:
                v.trackingState = VehicleTrackingState.PARKED
                v.parkedSpotId = parked_spot_id
            elif position is not None:
                if v.trackingState == VehicleTrackingState.PARKED:
                    v.trackingState = VehicleTrackingState.TRACKING_EXIT
                elif v.trackingState == VehicleTrackingState.EXITED:
                    pass  # keep
                else:
                    v.trackingState = VehicleTrackingState.TRACKING

            return v

    def mark_vehicle_exited(self, global_vehicle_id: int):
        with self._lock:
            v = self._vehicles.get(global_vehicle_id)
            if v:
                v.trackingState = VehicleTrackingState.EXITED
                v.position = None

    def get_vehicle(self, global_vehicle_id: int) -> Optional[Vehicle]:
        with self._lock:
            return self._vehicles.get(global_vehicle_id)

    def get_all_vehicles(self) -> Dict[int, dict]:
        with self._lock:
            return {vid: v.to_dict() for vid, v in self._vehicles.items()}

    # ── Parking spot management ──

    def update_spot(self, spot_id: str, vision_occupied: bool,
                    tracking_occupied: bool, vehicle_id: Optional[int] = None) -> ParkingSpot:
        with self._lock:
            spot = self._spots.get(spot_id)
            if spot is None:
                spot = ParkingSpot(spotId=spot_id)
                self._spots[spot_id] = spot

            spot.visionOccupied = vision_occupied
            spot.trackingOccupied = tracking_occupied
            spot.occupied = vision_occupied or tracking_occupied
            spot.vehicleId = vehicle_id

            # Auto-bind session: if a vehicle is parked in this slot,
            # find its session and update parkedSpotId
            if vehicle_id is not None and tracking_occupied:
                sid = self._vehicle_session_map.get(vehicle_id)
                if sid:
                    s = self._sessions.get(sid)
                    if s and s.state not in (SessionState.PARKED, SessionState.CLOSED):
                        s.set_parked(spot_id)

            return spot

    def clear_spot_vehicle(self, spot_id: str):
        """Vision confirmed vehicle left slot."""
        with self._lock:
            spot = self._spots.get(spot_id)
            if spot:
                spot.vehicleId = None
                spot.trackingOccupied = False
                spot.occupied = spot.visionOccupied

    def get_all_spots(self) -> Dict[str, dict]:
        with self._lock:
            return {sid: s.to_dict() for sid, s in self._spots.items()}

    def get_spot(self, spot_id: str) -> Optional[ParkingSpot]:
        with self._lock:
            return self._spots.get(spot_id)

    # ── Events ──

    def _add_event(self, event_type: str, data: dict):
        """Add an event (already under lock)."""
        from datetime import datetime, timezone
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            **data,
        }
        self._events.append(event)
        # Keep last 200 events
        if len(self._events) > 200:
            self._events = self._events[-200:]

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self._events[-limit:])

    # ── Aggregated state for API ──

    def get_full_state(self) -> dict:
        """Complete state snapshot for dashboard/debug."""
        with self._lock:
            return {
                "vehicles": {vid: v.to_dict() for vid, v in self._vehicles.items()},
                "spots": {sid: s.to_dict() for sid, s in self._spots.items()},
                "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
                "events": list(self._events[-50:]),
            }


# Singleton
runtime_state = RuntimeState()
