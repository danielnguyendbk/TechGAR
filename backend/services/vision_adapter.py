"""
Vision State Adapter — translates raw Vision/AI output into canonical backend models.

Review fix #46: Backend business logic should not depend on AI internal schema.
This adapter normalizes:
  - track_id → globalVehicleId
  - local_track_id → dropped (backend doesn't need it)
  - Vision parking_status → canonical ParkingSpot updates
  - Vision vehicle_positions → canonical Vehicle updates

Review fix #9: Naming standardization (globalVehicleId, localTrackId)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from state.runtime_state import runtime_state


def process_vision_vehicles(vision_data: dict) -> None:
    """
    Process raw vehicle position data from vision output and update runtime state.

    Expected input format (from global_vehicle_registry.json or per-camera vehicle_positions):
    {
        "global_vehicles": {
            "7": {
                "global_id": 7,
                "position": {"x": 524, "y": 318},
                "camera_ids": ["cam3"],
                "parked_in_slot": "A03" | null,
                ...
            }
        }
    }
    """
    global_vehicles = vision_data.get("global_vehicles", {})

    for vid_str, v_data in global_vehicles.items():
        global_id = int(vid_str)
        position = v_data.get("position")
        camera_ids = v_data.get("camera_ids", [])
        parked_slot = v_data.get("parked_in_slot")

        runtime_state.update_vehicle(
            global_vehicle_id=global_id,
            position=position,
            camera_ids=camera_ids,
            parked_spot_id=parked_slot,
        )


def process_vision_parking(vision_data: dict) -> None:
    """
    Process raw parking slot data from vision output and update runtime state.

    Expected input format (from parking_status or global_vehicle_registry):
    {
        "parking_slots": {
            "A03": {
                "occupied": true,
                "vehicle_id": 7 | null,
                "vision_occupied": true,
                "tracking_occupied": true
            }
        }
    }
    """
    slots = vision_data.get("parking_slots", {})

    for spot_id, slot_data in slots.items():
        vision_occ = slot_data.get("vision_occupied", slot_data.get("occupied", False))
        tracking_occ = slot_data.get("tracking_occupied", False)
        vehicle_id = slot_data.get("vehicle_id")

        runtime_state.update_spot(
            spot_id=spot_id,
            vision_occupied=vision_occ,
            tracking_occupied=tracking_occ,
            vehicle_id=vehicle_id,
        )


def process_vision_entry_event(global_vehicle_id: int, gate_id: str = "ENTRY_1") -> Optional[str]:
    """
    Vision detected a vehicle crossing the ENTRY line.
    Creates a new session and returns the sessionId.

    Review fix #25, #33, #34: Vehicle must cross ENTRY ROI in correct direction.
    """
    # Check if this vehicle already has an active session
    existing = runtime_state.get_session_by_vehicle(global_vehicle_id)
    if existing and existing.state.value not in ("CLOSED",):
        return existing.sessionId

    session = runtime_state.create_session(global_vehicle_id, gate_id)
    print(f"[VISION_ADAPTER] Vehicle entered: globalVehicleId={global_vehicle_id} "
          f"→ session={session.sessionId}")
    return session.sessionId


def process_vision_exit_event(global_vehicle_id: int) -> Optional[str]:
    """
    Vision detected a vehicle crossing the EXIT line.
    Closes the associated session.

    Review fix #27, #35: Only EXIT event closes a session, never lost-track.
    """
    session = runtime_state.get_session_by_vehicle(global_vehicle_id)
    if session:
        # Clear the parked spot
        if session.parkedSpotId:
            runtime_state.clear_spot_vehicle(session.parkedSpotId)
        runtime_state.close_session(session.sessionId)
        runtime_state.mark_vehicle_exited(global_vehicle_id)
        print(f"[VISION_ADAPTER] Vehicle exited: globalVehicleId={global_vehicle_id} "
              f"→ session={session.sessionId} CLOSED")
        return session.sessionId
    return None


def process_vision_parked_event(global_vehicle_id: int, spot_id: str) -> None:
    """
    Vision Binder confirmed vehicle stopped in a slot.
    Updates spot and auto-binds session.

    Review fix #4, #11, #18: Use Binder's actual binding, not target slot occupancy.
    """
    runtime_state.update_spot(
        spot_id=spot_id,
        vision_occupied=True,
        tracking_occupied=True,
        vehicle_id=global_vehicle_id,
    )

    # Find and update session
    session = runtime_state.get_session_by_vehicle(global_vehicle_id)
    if session and session.state.value not in ("PARKED", "CLOSED"):
        runtime_state.set_session_parked(session.sessionId, spot_id)
        print(f"[VISION_ADAPTER] Vehicle parked: globalVehicleId={global_vehicle_id} "
              f"in {spot_id} → session={session.sessionId}")


def process_vision_left_slot_event(global_vehicle_id: int, spot_id: str) -> None:
    """
    Vision Binder confirmed vehicle left a slot.
    Clears the slot binding. Does NOT close the session.

    Review fix #17: Slot cleared only when vision confirms, not on user click.
    """
    runtime_state.clear_spot_vehicle(spot_id)
    print(f"[VISION_ADAPTER] Vehicle left slot: globalVehicleId={global_vehicle_id} "
          f"from {spot_id}")
