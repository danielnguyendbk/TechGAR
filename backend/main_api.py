"""
FastAPI backend for TechGAR Smart Parking System.

Review fixes:
  #33 — Replaced HTTPServer with FastAPI + Uvicorn
  #32 — Proper HTTP status codes (404, 409, 400, 200)
  #18 — No writing to frontend/public
  #19 — No 127.0.0.1 fallback in frontend
  #20 — Consolidated endpoints, reduced polling
  #21 — Per-session endpoint returns only that session's vehicle
  #25 — No hardcoded y>700 for entry detection
  #26 — Frontend reads from API, not shared JSON
  #28 — Polling endpoints for dashboard
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup ──
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from state.runtime_state import runtime_state
from services.vision_adapter import (
    process_vision_vehicles,
    process_vision_parking,
    process_vision_entry_event,
    process_vision_exit_event,
    process_vision_parked_event,
    process_vision_left_slot_event,
)

app = FastAPI(
    title="TechGAR Smart Parking API",
    version="2.0.0",
    description="Canonical API — vision engine is the only source of tracking data.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════
#  Request / Response Models
# ══════════════════════════════════════════════

class ClaimRequest(BaseModel):
    sessionId: str

class SelectSpotRequest(BaseModel):
    sessionId: str
    spotId: Optional[str] = None

class ExitRequest(BaseModel):
    sessionId: str

class VisionEntryEvent(BaseModel):
    globalVehicleId: int
    gateId: str = "ENTRY_1"

class VisionExitEvent(BaseModel):
    globalVehicleId: int

class VisionParkedEvent(BaseModel):
    globalVehicleId: int
    spotId: str

class VisionLeftSlotEvent(BaseModel):
    globalVehicleId: int
    spotId: str

class VisionBulkUpdate(BaseModel):
    """For polling vision output files."""
    global_vehicles: Optional[dict] = None
    parking_slots: Optional[dict] = None


# ══════════════════════════════════════════════
#  Session API — Frontend calls these
# ══════════════════════════════════════════════

@app.get("/api/sessions")
def get_all_sessions():
    """Dashboard: get all sessions."""
    return runtime_state.get_all_sessions()


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Personal page: get a specific session + its vehicle."""
    session = runtime_state.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    result = session.to_dict()

    # Include vehicle data for this session (fix #21: only this vehicle)
    if session.globalVehicleId is not None:
        vehicle = runtime_state.get_vehicle(session.globalVehicleId)
        if vehicle:
            result["vehicle"] = vehicle.to_dict()
        else:
            # Vehicle may not be actively tracked; provide display position from parked spot
            result["vehicle"] = {
                "globalVehicleId": session.globalVehicleId,
                "trackingState": "PARKED" if session.parkedSpotId else "TRACKING",
                "position": None,
                "parkedSpotId": session.parkedSpotId,
            }

    return result


@app.get("/api/sessions/waiting")
def get_waiting_sessions(gate_id: Optional[str] = None):
    """QR Kiosk: get sessions waiting for scan, sorted by creation time."""
    sessions = runtime_state.get_waiting_sessions(gate_id)
    return [s.to_dict() for s in sessions]


@app.post("/api/session/claim")
def claim_session(req: ClaimRequest):
    """User scanned QR → claim session."""
    session = runtime_state.claim_session(req.sessionId)
    if not session:
        existing = runtime_state.get_session(req.sessionId)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Session {req.sessionId} not found")
        raise HTTPException(status_code=409, detail=f"Session not in WAITING_FOR_SCAN state (current: {existing.state.value})")
    return {"ok": True, "sessionId": session.sessionId, "state": session.state.value}


@app.post("/api/session/select")
def select_spot(req: SelectSpotRequest):
    """User chose a target spot."""
    session = runtime_state.select_spot(req.sessionId, req.spotId)
    if not session:
        existing = runtime_state.get_session(req.sessionId)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Session {req.sessionId} not found")
        raise HTTPException(status_code=409, detail=f"Cannot select spot in state {existing.state.value}")
    return {
        "ok": True,
        "sessionId": session.sessionId,
        "spotId": req.spotId,
        "state": session.state.value,
    }


@app.post("/api/session/exit")
def start_exit(req: ExitRequest):
    """User clicked 'retrieve car'. Spot remains occupied until vision confirms."""
    session = runtime_state.start_exit(req.sessionId)
    if not session:
        existing = runtime_state.get_session(req.sessionId)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Session {req.sessionId} not found")
        raise HTTPException(status_code=409, detail=f"Cannot start exit in state {existing.state.value}")
    return {
        "ok": True,
        "sessionId": session.sessionId,
        "state": session.state.value,
        "parkedSpotId": session.parkedSpotId,
    }


# ══════════════════════════════════════════════
#  Parking & Vehicle API — Frontend calls these
# ══════════════════════════════════════════════

@app.get("/api/parking")
def get_parking_status():
    """Get all parking spot states."""
    spots = runtime_state.get_all_spots()
    total = len(spots)
    free = sum(1 for s in spots.values() if not s["occupied"])
    return {
        "total": total,
        "free": free,
        "occupied": total - free,
        "spots": spots,
    }


@app.get("/api/vehicles")
def get_all_vehicles():
    """Dashboard: get all tracked vehicles."""
    return runtime_state.get_all_vehicles()


@app.get("/api/state")
def get_full_state():
    """Debug/Dashboard: complete runtime state."""
    return runtime_state.get_full_state()


@app.get("/api/events")
def get_events(limit: int = 50):
    """Get recent events."""
    return runtime_state.get_recent_events(limit)


# ══════════════════════════════════════════════
#  Vision Events API — Vision engine pushes events here
# ══════════════════════════════════════════════

@app.post("/api/vision/entry")
def vision_entry(ev: VisionEntryEvent):
    """Vision detected vehicle crossing ENTRY line."""
    session_id = process_vision_entry_event(ev.globalVehicleId, ev.gateId)
    return {"ok": True, "sessionId": session_id}


@app.post("/api/vision/exit")
def vision_exit(ev: VisionExitEvent):
    """Vision detected vehicle crossing EXIT line."""
    session_id = process_vision_exit_event(ev.globalVehicleId)
    return {"ok": True, "sessionId": session_id}


@app.post("/api/vision/parked")
def vision_parked(ev: VisionParkedEvent):
    """Vision Binder confirmed vehicle parked in slot."""
    process_vision_parked_event(ev.globalVehicleId, ev.spotId)
    return {"ok": True}


@app.post("/api/vision/left_slot")
def vision_left_slot(ev: VisionLeftSlotEvent):
    """Vision Binder confirmed vehicle left slot."""
    process_vision_left_slot_event(ev.globalVehicleId, ev.spotId)
    return {"ok": True}


@app.post("/api/vision/bulk_update")
def vision_bulk_update(data: VisionBulkUpdate):
    """Bulk update from vision output files (for file-watching mode)."""
    if data.global_vehicles:
        process_vision_vehicles({"global_vehicles": data.global_vehicles})
    if data.parking_slots:
        process_vision_parking({"parking_slots": data.parking_slots})
    return {"ok": True}


# ══════════════════════════════════════════════
#  Vision File Watcher (background thread)
# ══════════════════════════════════════════════

def _watch_vision_output(vision_output_dir: str, interval: float = 0.5):
    """Background thread that reads vision output files and pushes into runtime state."""
    output_dir = Path(vision_output_dir)
    registry_file = output_dir / "global_vehicle_registry.json"
    last_frame_idx = -1

    print(f"[VISION WATCHER] Watching: {output_dir}")

    while True:
        try:
            if registry_file.exists():
                with registry_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                frame_idx = data.get("frame_index", 0)
                if frame_idx <= last_frame_idx:
                    time.sleep(interval)
                    continue
                last_frame_idx = frame_idx

                # Process parking slots
                parking_slots = data.get("parking_slots", {})
                if parking_slots:
                    process_vision_parking({"parking_slots": parking_slots})

                # Process global vehicles
                global_vehicles_raw = data.get("global_vehicles", {})
                if global_vehicles_raw:
                    process_vision_vehicles({"global_vehicles": global_vehicles_raw})

                # Process events from vision
                events = data.get("parking_events", [])
                for event in events:
                    etype = event.get("type", "")
                    gvid = event.get("global_vehicle_id")
                    spot_id = event.get("slot_id")

                    if etype == "vehicle_stopped_in_slot" and gvid is not None and spot_id:
                        process_vision_parked_event(gvid, spot_id)
                    elif etype == "vehicle_left_slot" and gvid is not None and spot_id:
                        process_vision_left_slot_event(gvid, spot_id)

        except Exception as e:
            print(f"[VISION WATCHER] Error: {e}")

        time.sleep(interval)


def start_vision_watcher(vision_output_dir: str = None):
    """Start the vision file watcher in a background thread."""
    if vision_output_dir is None:
        # Default: look for vision output relative to backend
        vision_output_dir = str(BACKEND_DIR / "vision" / "runtime_output")

    # Also check main_detect (for backward compatibility)
    alt_dir = BACKEND_DIR.parent / "main_detect" / "runtime_output"
    if not Path(vision_output_dir).exists() and alt_dir.exists():
        vision_output_dir = str(alt_dir)

    thread = threading.Thread(
        target=_watch_vision_output,
        args=(vision_output_dir,),
        daemon=True,
    )
    thread.start()
    return thread


# ══════════════════════════════════════════════
#  Entry Gate Watcher (simulates entry detection)
# ══════════════════════════════════════════════

def _watch_entry_gate(vision_output_dir: str, interval: float = 0.3):
    """
    Watches vehicle positions to detect entry events.

    Review fix #25: In production, this should use ROI line crossing from vision.
    For demo, we watch vehicle positions for new globalVehicleIds near the gate area.
    """
    output_dir = Path(vision_output_dir)
    known_vehicle_ids: set = set()

    while True:
        try:
            # Check all vehicle position files
            for cam_file in sorted(output_dir.glob("vehicle_positions_cam*.json")):
                try:
                    with cam_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                active = data.get("active_vehicles", {})
                for vid_str, v_data in active.items():
                    global_id = v_data.get("track_id", int(vid_str))
                    if global_id not in known_vehicle_ids:
                        known_vehicle_ids.add(global_id)
                        pos = v_data.get("position", {})
                        # For demo: any new vehicle gets a session
                        # In production: only vehicles crossing ENTRY ROI line
                        process_vision_entry_event(global_id, "ENTRY_1")

        except Exception as e:
            print(f"[ENTRY WATCHER] Error: {e}")

        time.sleep(interval)


def start_entry_watcher(vision_output_dir: str = None):
    """Start entry gate watcher in background."""
    if vision_output_dir is None:
        vision_output_dir = str(BACKEND_DIR / "vision" / "runtime_output")
    alt_dir = BACKEND_DIR.parent / "main_detect" / "runtime_output"
    if not Path(vision_output_dir).exists() and alt_dir.exists():
        vision_output_dir = str(alt_dir)

    thread = threading.Thread(
        target=_watch_entry_gate,
        args=(vision_output_dir,),
        daemon=True,
    )
    thread.start()
    return thread


# ══════════════════════════════════════════════
#  Startup
# ══════════════════════════════════════════════

@app.on_event("startup")
def on_startup():
    """Start background watchers on FastAPI startup."""
    start_vision_watcher()
    start_entry_watcher()
    print("=" * 60)
    print("  🅿️  TechGAR Smart Parking Backend v2.0")
    print("  FastAPI + Uvicorn")
    print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
