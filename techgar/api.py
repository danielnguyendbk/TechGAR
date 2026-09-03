"""HTTP boundary for the runtime, session and operator contracts.

The tracking pipeline remains transport-agnostic.  This module is the thin,
validated boundary used by PLAN 4: it exposes a versioned immutable snapshot,
idempotent session actions and explicitly confirmed identity reset operations.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .snapshot import RuntimeSnapshot, SCHEMA_VERSION
from .units import WORLD_FRAME_NAME


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_vehicle_id: int | None = None


class SelectSpotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spot_id: str = Field(min_length=1, max_length=64)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_sessions: bool = False


class GateConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[tuple[float, float]] = Field(min_length=6, max_length=6)


class ReservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spot_id: str = Field(min_length=1, max_length=64)


class CloseAllRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: bool = False


class TokenClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1)


@dataclass
class RuntimeSession:
    session_id: str
    state: str = "WAITING"
    global_vehicle_id: int | None = None
    target_spot_id: str | None = None
    parked_spot_id: str | None = None
    claimed_at: float | None = None
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "state": self.state,
            "globalVehicleId": self.global_vehicle_id,
            "targetSpotId": self.target_spot_id,
            "parkedSpotId": self.parked_spot_id,
            "claimedAt": self.claimed_at,
            "updatedAt": self.updated_at,
        }


class RuntimeService:
    """Thread-safe state adapter around an optional :class:`TechgarPipeline`."""

    def __init__(self, pipeline=None) -> None:
        self.pipeline = pipeline
        self.sessions: dict[str, RuntimeSession] = {}
        self.gate_points: list[tuple[float, float]] = []
        self._snapshot: RuntimeSnapshot | None = None
        self._lock = threading.RLock()

    def set_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        with self._lock:
            if self._snapshot is not None and snapshot.sequence < self._snapshot.sequence:
                raise ValueError("snapshot frame_index must be monotonic")
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
                camera_health={"C1": {"online": False}, "C2": {"online": False}},
                identity_events=(),
                published_at=time.time(),
            )

    def _first_visible_global_id(self) -> int | None:
        vehicles = [vehicle for vehicle in self.current_snapshot().vehicles
                    if vehicle.display_state.value != "hidden"]
        return min((vehicle.global_id for vehicle in vehicles), default=None)

    def claim(self, session_id: str, requested_global_id: int | None = None) -> RuntimeSession:
        with self._lock:
            existing = self.sessions.get(session_id)
            if existing is not None and existing.claimed_at is not None:
                return existing
            record = existing or RuntimeSession(session_id=session_id)
            record.global_vehicle_id = (requested_global_id if requested_global_id is not None
                                        else self._first_visible_global_id())
            record.claimed_at = time.time()
            record.updated_at = record.claimed_at
            record.state = "WAITING"
            self.sessions[session_id] = record
            if self.pipeline is not None and record.global_vehicle_id is not None:
                self.pipeline.sessions.bind(session_id, record.global_vehicle_id,
                                            record.claimed_at, self.pipeline._frame_sequence)
            return record

    def select_spot(self, session_id: str, spot_id: str) -> RuntimeSession:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None or record.claimed_at is None:
                raise KeyError(session_id)
            valid = {slot.slot_id for slot in self.current_snapshot().slots}
            if valid and spot_id not in valid:
                raise ValueError(f"unknown parking spot {spot_id}")
            record.target_spot_id = spot_id
            record.state = "NAVIGATING"
            record.updated_at = time.time()
            return record

    def mark_parked(self, session_id: str) -> RuntimeSession:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.parked_spot_id = record.target_spot_id
            record.state = "PARKED"
            record.updated_at = time.time()
            return record

    def request_exit(self, session_id: str) -> RuntimeSession:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.state = "EXIT_NAVIGATION"
            record.updated_at = time.time()
            return record

    def reset(self, include_sessions: bool) -> dict:
        with self._lock:
            if self.pipeline is not None:
                result = self.pipeline.reset_identities(include_sessions=include_sessions)
            else:
                snapshot = self.current_snapshot()
                result = {
                    "reset": True,
                    "retired_identities": len(snapshot.vehicles),
                    "include_sessions": include_sessions,
                }
                self._snapshot = RuntimeSnapshot(
                    sequence=snapshot.sequence + 1,
                    timestamp=time.time(),
                    world_frame=snapshot.world_frame,
                    vehicles=(),
                    slots=snapshot.slots,
                    camera_health=snapshot.camera_health,
                    identity_events=snapshot.identity_events,
                    published_at=time.time(),
                    slot_layout=snapshot.slot_layout,
                )
            if include_sessions:
                self.sessions.clear()
            else:
                for record in self.sessions.values():
                    record.global_vehicle_id = None
                    record.state = "WAITING"
                    record.updated_at = time.time()
            return result

    def soft_reset(self) -> dict:
        with self._lock:
            if self.pipeline is not None:
                return self.pipeline.registry.soft_reset(time.time(), getattr(self.pipeline, "_frame_sequence", 0) + 1)
            return {"soft_reset": True, "parked_kept": 0, "recovery_pending": 0, "retired": 0}

    def close_all(self, confirm: bool = False) -> dict:
        if not confirm:
            raise ValueError("close-all requires explicit confirm=True")
        return self.reset(include_sessions=True)

    def cancel_reservation(self, session_id: str) -> RuntimeSession:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None:
                raise KeyError(session_id)
            record.target_spot_id = None
            if record.state == "NAVIGATING":
                record.state = "WAITING"
            record.updated_at = time.time()
            return record


def create_app(service: RuntimeService | None = None) -> FastAPI:
    runtime = service or RuntimeService()
    app = FastAPI(title="TechGAR Runtime API", version=SCHEMA_VERSION)
    app.state.runtime = runtime
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "schema_version": SCHEMA_VERSION, "timestamp": time.time()}

    @app.get("/api/ready")
    def ready():
        return {"ready": True, "timestamp": time.time()}

    @app.post("/api/runtime/soft-reset")
    def soft_reset():
        return runtime.soft_reset()

    @app.post("/api/runtime/close-all")
    def close_all(request: CloseAllRequest):
        try:
            return runtime.close_all(confirm=request.confirm)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from None

    @app.get("/api/runtime/snapshot")
    def get_snapshot():
        return runtime.current_snapshot().to_dict()

    @app.post("/api/runtime/reset-identities")
    def reset_identities(request: ResetRequest):
        return runtime.reset(request.include_sessions)

    @app.post("/api/runtime/gates")
    def configure_gates(request: GateConfigRequest):
        runtime.gate_points = [(float(x), float(y)) for x, y in request.points]
        return {"saved": True, "points": runtime.gate_points}

    @app.get("/api/runtime/cameras/{camera_id}.mjpg")
    async def camera_stream(camera_id: str):
        health = runtime.current_snapshot().to_dict().get("cameras", {}).get(camera_id)
        if health is None:
            raise HTTPException(status_code=404, detail="unknown camera")
        if not health.get("online", False):
            raise HTTPException(status_code=503, detail="camera offline")
        # Standards-compliant one-pixel JPEG heartbeat.  A production capture
        # adapter replaces this generator with encoded subscriber-gated frames.
        frame = base64.b64decode(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////"
            "////////////////////////2wBDAf//////////////////////////////////////"
            "////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
            "AAAAAAAAAAb/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAA"
            "AAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgB"
            "AwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAA"
            "AAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//"
            "9oADAMBAAIAAwAAABAf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPxB//8QAFBE"
            "BAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAA"
            "gBAQABPxB//9k="
        )

        async def stream():
            while True:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/sessions/waiting")
    def waiting_sessions():
        return [record.as_dict() for record in runtime.sessions.values()
                if record.state == "WAITING"]

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        record = runtime.sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="session ended")
        return record.as_dict()

    @app.post("/api/sessions/{session_id}/claim")
    def claim_session(session_id: str, request: ClaimRequest):
        return runtime.claim(session_id, request.global_vehicle_id).as_dict()

    @app.put("/api/sessions/{session_id}/reservation")
    def make_reservation(session_id: str, request: ReservationRequest):
        try:
            return runtime.select_spot(session_id, request.spot_id).as_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="session ended") from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.delete("/api/sessions/{session_id}/reservation")
    def cancel_reservation(session_id: str):
        try:
            return runtime.cancel_reservation(session_id).as_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="session ended") from None

    @app.post("/api/sessions/{session_id}/select-spot")
    def select_spot(session_id: str, request: SelectSpotRequest):
        try:
            return runtime.select_spot(session_id, request.spot_id).as_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="session ended") from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

    @app.post("/api/sessions/{session_id}/parked")
    def mark_parked(session_id: str):
        try:
            return runtime.mark_parked(session_id).as_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="session ended") from None

    @app.post("/api/sessions/{session_id}/exit")
    def request_exit(session_id: str):
        try:
            return runtime.request_exit(session_id).as_dict()
        except KeyError:
            raise HTTPException(status_code=404, detail="session ended") from None

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("techgar.api:app", host="127.0.0.1", port=8001, reload=False)

