"""Local two-camera replay dashboard for checking imported DroidCam assets."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .replay import (ReplaySite, build_replay_pipeline, iter_decoded_pairs,
                     load_replay_site, process_pair, roi_mask)
from .replay_output import ReplayOutputWriter


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenCV is required. Run: .venv\\Scripts\\python.exe -m pip install -e .[video]"
        ) from exc
    return cv2


def _put_text(image: np.ndarray, text: str, point: tuple[int, int], color,
              scale: float = 0.55, thickness: int = 1) -> None:
    cv2 = _cv2()
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def annotate_frame(site: ReplaySite, camera_id: str, frame: np.ndarray, result,
                   frame_index: int) -> np.ndarray:
    """Render imported geometry plus current Stage 1-10 observations."""
    cv2 = _cv2()
    canvas = frame.copy()
    roi = np.rint(site.roi_polygons[camera_id]).astype(np.int32)
    cv2.polylines(canvas, [roi], True, (0, 215, 255), 3, cv2.LINE_AA)

    snapshot = result.snapshot if result is not None else None
    slot_states = ({slot.slot_id: slot for slot in snapshot.slots}
                   if snapshot is not None else {})
    inverse_scale = 1.0 / site.processing_scale
    for slot_id, polygon in site.pixel_slots[camera_id].items():
        state = slot_states.get(slot_id)
        status = state.occupancy_state if state is not None else "empty"
        is_occupied = (status == "occupied") or (state is not None and getattr(state, "vision_occupied", False))
        color = ((40, 50, 245) if is_occupied else
                 (0, 165, 255) if status in ("claim_pending", "releasing") else
                 (50, 220, 80))
        points = np.rint(polygon * inverse_scale).astype(np.int32)
        cv2.polylines(canvas, [points], True, color, 2, cv2.LINE_AA)
        centre = np.rint(polygon.mean(axis=0) * inverse_scale).astype(int)
        owner = f" G{state.owning_global_id}" if state and state.owning_global_id else ""
        _put_text(canvas, f"{slot_id}{owner}", (int(centre[0]) - 18, int(centre[1])),
                  color, 0.38, 1)

    detections = result.detections.get(camera_id, []) if result is not None else []
    for detection in detections:
        box = np.rint(np.asarray(detection.bbox) * inverse_scale).astype(int)
        color = (255, 220, 0) if not detection.occlusion_group_candidate else (200, 80, 255)
        cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), color, 2, cv2.LINE_AA)
        anchor = np.rint(np.asarray(detection.ground_anchor) * inverse_scale).astype(int)
        cv2.circle(canvas, tuple(anchor), 5, color, -1, cv2.LINE_AA)
        _put_text(canvas, f"DET {detection.confidence:.2f}",
                  (int(box[0]), max(70, int(box[1]) - 5)), color, 0.42, 1)

    observations = ([item for item in result.observations if item.camera_id == camera_id]
                    if result is not None else [])
    for observation in observations:
        box_value = (observation.measured_bbox if observation.measured_bbox is not None
                     else observation.predicted_bbox)
        box = np.rint(np.asarray(box_value) * inverse_scale).astype(int)
        _put_text(canvas, f"L{observation.local_track_id}:{observation.state.value}",
                  (int(box[0]), min(canvas.shape[0] - 8, int(box[3]) + 16)),
                  (255, 255, 0), 0.42, 1)

    if snapshot is not None:
        calibration = site.profiles[camera_id].calibration
        for vehicle in snapshot.vehicles:
            if vehicle.display_state.value not in ("active", "parked"):
                continue
            point = calibration.unproject(np.asarray(vehicle.world_position, dtype=float))
            point = np.rint(point * inverse_scale).astype(int)
            if 0 <= point[0] < canvas.shape[1] and 0 <= point[1] < canvas.shape[0]:
                cv2.circle(canvas, tuple(point), 13, (255, 0, 210), 3, cv2.LINE_AA)
                _put_text(canvas, f"G{vehicle.global_id} {vehicle.display_state.value}",
                          (int(point[0]) + 15, int(point[1]) - 8), (255, 0, 210), 0.52, 2)

    global_count = len([v for v in snapshot.vehicles if v.display_state.value in ("active", "parked")]) if snapshot is not None else 0
    occupied = (sum((slot.occupancy_state == "occupied" or getattr(slot, "vision_occupied", False))
                    for slot in snapshot.slots)
                if snapshot is not None else 0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (18, 20, 24), -1)
    _put_text(canvas,
              f"{site.dataset_id} | {camera_id} | frame {frame_index}/{len(site.timestamps)} "
              f"| det {len(detections)} | global {global_count} | occupied {occupied}",
              (14, 24), (245, 245, 245), 0.55, 1)
    _put_text(canvas, "BOOTSTRAP 4-POINT CALIBRATION - NOT COMMISSIONED",
              (14, 50), (0, 165, 255), 0.55, 2)
    return canvas


class ReplayPlayer:
    def __init__(self, site: ReplaySite, speed: float = 4.0, loop: bool = True,
                 jpeg_quality: int = 78, output_root: str | Path | None = None,
                 max_frames: int | None = None) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.site = site
        self.speed = float(speed)
        self.loop = bool(loop)
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.output_root = None if output_root is None else Path(output_root).resolve()
        self.max_frames = max_frames if max_frames is None else max(1, int(max_frames))
        self.runtime_id = uuid4().hex
        self.pipeline = build_replay_pipeline(site)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: dict[str, bytes] = {}
        self._jpeg_sequence = 0
        self._status: dict[str, Any] = {
            "state": "created", "frame_index": 0, "total_frames": len(site.timestamps),
            "cycle": 0, "processing_ms": 0.0, "detections": {}, "global_vehicles": 0,
            "occupied_slots": 0, "error": None,
            "output_directory": None, "camera_skew_ms": 0.0,
        }

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="techgar-replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        self._restart.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def pause(self) -> None:
        self._pause.set()
        with self._lock:
            self._status["state"] = "paused"

    def resume(self) -> None:
        self._pause.clear()
        with self._lock:
            self._status["state"] = "running"

    def restart(self) -> None:
        self._restart.set()
        self._pause.clear()

    def status(self) -> dict[str, Any]:
        with self._lock:
            value = dict(self._status)
            value.update({
                "dataset": self.site.dataset_id,
                "calibration_profile": self.site.calibration_profile,
                "processing_scale": self.site.processing_scale,
                "playback_speed": self.speed,
                "bootstrap_warning": self.site.bootstrap_warning,
                "runtime_id": self.runtime_id,
                "source_mode": "replay",
                "max_frames": self.max_frames,
            })
            return value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.pipeline.publisher.last is None:
                return {"sequence": 0, "vehicles": [], "slots": [],
                        "warning": self.site.bootstrap_warning}
            value = self.pipeline.publisher.last.to_dict()
            value["warning"] = self.site.bootstrap_warning
            value["dataset"] = self.site.dataset_id
            value["calibration_profile"] = self.site.calibration_profile
            value["runtime_id"] = self.runtime_id
            value["source_mode"] = "replay"
            value["camera_skew_ms"] = self._status.get("camera_skew_ms", 0.0)
            value["pending_handoffs"] = [
                {"global_id": state.global_id,
                 "source_camera": state.latest_camera,
                 "target_camera": state.exit_pending_to}
                for state in self.pipeline.registry.identities.values()
                if state.exit_pending_to is not None
            ]
            return value

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return json.loads(self.pipeline.registry.events.to_json(limit=max(1, min(limit, 1000))))

    def jpeg(self, camera_id: str) -> tuple[int, bytes | None]:
        with self._lock:
            return self._jpeg_sequence, self._jpeg.get(camera_id)

    def _wait_if_paused(self) -> bool:
        while self._pause.is_set() and not self._stop.is_set():
            self._stop.wait(0.1)
        return self._stop.is_set()

    def _run(self) -> None:
        cv2 = _cv2()
        cycle = 0
        writer: ReplayOutputWriter | None = None
        try:
            while not self._stop.is_set():
                cycle += 1
                self.pipeline = build_replay_pipeline(self.site)
                writer = (ReplayOutputWriter(self.site, self.output_root, self.speed,
                                             runtime_id=self.runtime_id)
                          if self.output_root is not None else None)
                masks = {camera_id: roi_mask(self.site, camera_id)
                         for camera_id in self.site.camera_ids}
                self._restart.clear()
                previous_capture = 0.0
                with self._lock:
                    self._status.update({"state": "running", "cycle": cycle,
                                         "frame_index": 0, "error": None,
                                         "output_directory": (str(writer.directory)
                                                              if writer else None)})
                for timestamp, frames in iter_decoded_pairs(self.site, limit=self.max_frames):
                    if self._stop.is_set() or self._restart.is_set():
                        break
                    if self._wait_if_paused():
                        break
                    delay = max(0.0, timestamp.capture_time - previous_capture) / self.speed
                    previous_capture = timestamp.capture_time
                    if self._stop.wait(delay):
                        break
                    started = time.perf_counter()
                    result = process_pair(self.site, self.pipeline, timestamp, frames, masks)
                    encoded: dict[str, bytes] = {}
                    annotated: dict[str, np.ndarray] = {}
                    for camera_id, frame in frames.items():
                        overlay = annotate_frame(self.site, camera_id, frame, result,
                                                 timestamp.frame_index)
                        annotated[camera_id] = overlay
                        ok, payload = cv2.imencode(
                            ".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        )
                        if ok:
                            encoded[camera_id] = payload.tobytes()
                    snapshot = result.snapshot if result is not None else None
                    with self._lock:
                        self._jpeg.update(encoded)
                        self._jpeg_sequence += 1
                        self._status.update({
                            "frame_index": timestamp.frame_index,
                            "processing_ms": round((time.perf_counter() - started) * 1000.0, 1),
                            "detections": ({camera: len(result.detections.get(camera, []))
                                            for camera in self.site.camera_ids}
                                           if result is not None else {}),
                            "global_vehicles": len(snapshot.vehicles) if snapshot else 0,
                            "occupied_slots": (sum(slot.occupancy_state == "occupied"
                                                   for slot in snapshot.slots)
                                               if snapshot else 0),
                            "camera_skew_ms": round(
                                (max(timestamp.camera_times.values())
                                 - min(timestamp.camera_times.values())) * 1000.0, 3),
                        })
                    if writer is not None:
                        writer.write(timestamp, annotated, result,
                                     time.perf_counter() - started, self.pipeline)
                if writer is not None:
                    run_status = ("stopped_by_user" if self._stop.is_set() else
                                  "restarted" if self._restart.is_set() else "completed")
                    writer.finish(self.pipeline, run_status)
                    writer = None
                if self._stop.is_set():
                    break
                if self._restart.is_set():
                    continue
                if not self.loop:
                    with self._lock:
                        self._status["state"] = "completed"
                    break
                self._stop.wait(0.4)
        except Exception as exc:  # keep the diagnostic available to the dashboard
            if writer is not None:
                try:
                    writer.finish(self.pipeline, "failed", f"{type(exc).__name__}: {exc}")
                except Exception:
                    pass
            with self._lock:
                self._status.update({"state": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if self._stop.is_set():
                with self._lock:
                    self._status["state"] = "stopped"


def _dashboard_html(site: ReplaySite) -> str:
    camera_cards = "".join(
        f'<section class="camera"><h2>{camera_id}</h2>'
        f'<img src="/api/demo/cameras/{camera_id}.mjpg" alt="{camera_id} replay"></section>'
        for camera_id in site.camera_ids
    )
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TechGAR Replay — {site.dataset_id}</title>
<style>
body{{margin:0;background:#0d1117;color:#e6edf3;font:15px system-ui,sans-serif}}
header{{padding:14px 20px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
h1{{font-size:20px;margin:0}} button{{padding:8px 14px;border:0;border-radius:7px;background:#238636;color:white;cursor:pointer}}
button:nth-of-type(2){{background:#1f6feb}} button:nth-of-type(3){{background:#9e6a03}}
.warning{{color:#f0b72f;font-weight:700}} main{{padding:14px;display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.camera{{background:#161b22;border:1px solid #30363d;border-radius:10px;overflow:hidden}} h2{{font-size:15px;margin:0;padding:9px 12px}}
img{{display:block;width:100%;height:auto;background:#000}} pre{{margin:14px;padding:12px;background:#161b22;border:1px solid #30363d;border-radius:8px;white-space:pre-wrap}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>TechGAR two-camera replay: {site.dataset_id}</h1>
<button onclick="control('resume')">Chạy</button><button onclick="control('pause')">Tạm dừng</button>
<button onclick="control('restart')">Chạy lại từ đầu</button>
<span class="warning">4-point bootstrap — chưa phải production calibration</span></header>
<main>{camera_cards}</main><pre id="status">Đang khởi động…</pre>
<script>
async function control(action){{await fetch('/api/demo/control/'+action,{{method:'POST'}});}}
async function poll(){{try{{let r=await fetch('/api/demo/status');let x=await r.json();
document.getElementById('status').textContent=`Trạng thái: ${{x.state}} | frame ${{x.frame_index}}/${{x.max_frames||x.total_frames}} | tốc độ ${{x.playback_speed}}x | xử lý ${{x.processing_ms}} ms\nDetection: ${{JSON.stringify(x.detections)}} | Global vehicles: ${{x.global_vehicles}} | Occupied slots: ${{x.occupied_slots}}${{x.output_directory?'\nĐang lưu: '+x.output_directory:''}}${{x.error?'\nLỗi: '+x.error:''}}`;}}catch(e){{}}}}
setInterval(poll,500);poll();</script></body></html>"""


def create_demo_app(player: ReplayPlayer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        player.start()
        yield
        player.stop()

    app = FastAPI(title="TechGAR local replay", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _dashboard_html(player.site)

    @app.get("/api/demo/status")
    def status():
        return player.status()

    @app.get("/api/runtime/status")
    def runtime_status():
        return player.status()

    @app.get("/api/runtime/snapshot")
    def snapshot():
        return player.snapshot()

    @app.get("/api/runtime/events")
    def events(limit: int = 100):
        return player.events(limit)

    @app.post("/api/demo/control/{action}")
    def control(action: str):
        if action == "pause":
            player.pause()
        elif action == "resume":
            player.resume()
        elif action == "restart":
            player.restart()
        else:
            raise HTTPException(status_code=422, detail="action must be pause/resume/restart")
        return player.status()

    @app.get("/api/demo/cameras/{camera_id}.mjpg")
    async def camera_stream(camera_id: str):
        if camera_id not in player.site.camera_ids:
            raise HTTPException(status_code=404, detail="unknown camera")

        async def stream():
            last_sequence = -1
            while True:
                sequence, payload = player.jpeg(camera_id)
                if payload is not None and sequence != last_sequence:
                    last_sequence = sequence
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n"
                           + payload + b"\r\n")
                await asyncio.sleep(0.025)

        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/runtime/cameras/{camera_id}.jpg")
    def camera_jpeg(camera_id: str):
        if camera_id not in player.site.camera_ids:
            raise HTTPException(status_code=404, detail="unknown camera")
        _, payload = player.jpeg(camera_id)
        if payload is None:
            raise HTTPException(status_code=503, detail="frame not ready")
        return Response(payload, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/runtime/cameras/{camera_id}.mjpg")
    async def runtime_camera_stream(camera_id: str):
        return await camera_stream(camera_id)

    return app


def smoke(site: ReplaySite, frames: int) -> dict[str, Any]:
    pipeline = build_replay_pipeline(site)
    masks = {camera_id: roi_mask(site, camera_id) for camera_id in site.camera_ids}
    last = None
    processed = 0
    started = time.perf_counter()
    for timestamp, images in iter_decoded_pairs(site, limit=frames):
        last = process_pair(site, pipeline, timestamp, images, masks)
        processed += 1
    snapshot = last.snapshot if last is not None else None
    return {
        "dataset": site.dataset_id,
        "frames_processed": processed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "global_vehicles": len(snapshot.vehicles) if snapshot else 0,
        "occupied_slots": (sum(slot.occupancy_state == "occupied" for slot in snapshot.slots)
                           if snapshot else 0),
        "performance": pipeline.performance_report(),
        "warning": site.bootstrap_warning,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synchronized TechGAR DroidCam replay")
    parser.add_argument("--manifest", default="config/site_manifest.json")
    parser.add_argument("--dataset", default="droidcam_shared_vd_18")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--processing-scale", type=float, default=None)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--save", action="store_true",
                        help="write a new immutable output run directory")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="process only the first N synchronized pairs")
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args(argv)
    site = load_replay_site(args.manifest, args.dataset, args.calibration,
                            args.processing_scale)
    if args.smoke_frames:
        print(json.dumps(smoke(site, args.smoke_frames), indent=2))
        return 0
    import uvicorn
    player = ReplayPlayer(site, speed=args.speed, loop=not args.no_loop,
                          output_root=args.output_root if args.save else None,
                          max_frames=args.max_frames)
    uvicorn.run(create_demo_app(player), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
