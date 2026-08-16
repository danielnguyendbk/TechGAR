"""Record one reproducible TechGAR experiment session.

The recorder accepts a phone/USB camera, an IP-camera URL, or a video file and
writes synchronized raw/debug videos, frame predictions, timestamps, resource
measurements, and ground-truth templates into one timestamped directory.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional, Union

import cv2
import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from techgar.direction_detector import DirectionDetector
from techgar.motion_tracker import MotionVehicleTracker
from techgar.parking_detector import ParkingDetector
from techgar.slot_vehicle_binder import SlotVehicleBinder

try:
    import psutil
except ImportError:  # The session remains usable; CPU/RAM columns are blank.
    psutil = None


Source = Union[int, str]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Khong the ghi JSON cho kieu {type(value).__name__}")


def _write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, default=_json_default)
    temporary.replace(path)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_source(args: argparse.Namespace) -> tuple[Source, str, str]:
    if args.camera is not None:
        return int(args.camera), "camera", f"camera:{args.camera}"
    if args.stream_url:
        return args.stream_url, "stream", args.stream_url
    video_path = _resolve_project_path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Khong tim thay video: {video_path}")
    return str(video_path), "video", str(video_path)


def _safe_session_name(value: Optional[str]) -> str:
    if not value:
        return "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not safe:
        raise ValueError("--session-name phai co it nhat mot ky tu hop le")
    return safe


def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Khong tao duoc video: {path}")
    return writer


def _track_json(track: object, binder: Optional[SlotVehicleBinder]) -> dict:
    track_id = int(track.track_id)
    return {
        "global_id": track_id,
        "local_track_id": track_id,
        "id_scope": "single_camera",
        "status": track.status.value,
        "visible_this_frame": int(track.consecutive_invisible_count) == 0,
        "position": {
            "x": int(track.cx),
            "y": int(track.cy),
            "reference": "bbox_bottom_center",
        },
        "bbox": {
            "x": int(track.x),
            "y": int(track.y),
            "w": int(track.w),
            "h": int(track.h),
        },
        "area": float(track.area),
        "age": int(track.age),
        "visible_count": int(track.total_visible_count),
        "invisible_count": int(track.consecutive_invisible_count),
        "parked_in_slot": binder.get_slot_for_vehicle(track_id) if binder else None,
    }


def _build_prediction(
    session_id: str,
    tracker: MotionVehicleTracker,
    binder: Optional[SlotVehicleBinder],
    source_time_ms: float,
    elapsed_ms: float,
    wall_time_iso: str,
    direction_events: list[dict],
) -> dict:
    confirmed = {
        str(track_id): _track_json(track, binder)
        for track_id, track in tracker.confirmed_tracks.items()
    }
    tentative = {
        str(track_id): _track_json(track, binder)
        for track_id, track in tracker.observable_tracks.items()
        if track.status.value == "tentative"
    }
    binder_events = []
    if binder is not None:
        binder_events = [
            event for event in binder.events
            if int(event.get("frame", -1)) == tracker.frame_index
        ]
    return {
        "schema_version": 1,
        "session_id": session_id,
        "frame_idx": int(tracker.frame_index),
        "source_time_ms": round(float(source_time_ms), 3),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "wall_time_iso": wall_time_iso,
        "active_vehicle_count": len(tracker.active_tracks),
        "confirmed_vehicles": confirmed,
        "tentative_vehicles": tentative,
        "parking_slots": binder.to_json() if binder is not None else {},
        "direction_events": direction_events,
        "parking_events": binder_events,
    }


def _draw_debug_frame(
    frame: np.ndarray,
    tracker: MotionVehicleTracker,
    parking_detector: Optional[ParkingDetector],
    slot_results: Optional[list],
    direction_detector: Optional[DirectionDetector],
    source_time_ms: float,
    processing_ms: float,
) -> np.ndarray:
    debug = tracker.draw_tracks(frame, tracker.all_tracks, show_non_active=True)
    if parking_detector is not None and slot_results is not None:
        debug = parking_detector.draw_results(debug, slot_results)
    if direction_detector is not None:
        direction_detector.draw_roi_lines(debug)

    cv2.rectangle(debug, (0, 0), (min(debug.shape[1], 560), 48), (0, 0, 0), -1)
    cv2.putText(
        debug,
        f"frame={tracker.frame_index}  t={source_time_ms / 1000.0:.3f}s",
        (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
    )
    cv2.putText(
        debug,
        f"active={len(tracker.active_tracks)}  process={processing_ms:.1f} ms",
        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1,
    )
    return debug


def _create_ground_truth_templates(session_dir: Path) -> None:
    with (session_dir / "ground_truth_slots.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes"])
    with (session_dir / "ground_truth_events.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["event_id", "vehicle_id", "slot_id", "event_type", "frame_idx", "notes"])


def record(args: argparse.Namespace) -> Path:
    source, source_kind, source_description = _resolve_source(args)
    session_id = _safe_session_name(args.session_name)
    output_root = _resolve_project_path(args.output_root)
    session_dir = output_root / session_id
    if session_dir.exists():
        raise FileExistsError(f"Session da ton tai, hay doi --session-name: {session_dir}")
    session_dir.mkdir(parents=True)
    _create_ground_truth_templates(session_dir)

    cap = cv2.VideoCapture(source)
    if source_kind in ("camera", "stream"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon: {source_description}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Nguon khong cung cap kich thuoc frame hop le")
    if not math.isfinite(source_fps) or source_fps < 1.0 or source_fps > 240.0:
        source_fps = 30.0
    record_fps = float(args.record_fps) if args.record_fps > 0 else source_fps

    parking_detector: Optional[ParkingDetector] = None
    binder: Optional[SlotVehicleBinder] = None
    slots_path: Optional[Path] = None
    if not args.disable_parking:
        slots_path = _resolve_project_path(args.slots_file)
        if not slots_path.is_file():
            cap.release()
            raise FileNotFoundError(f"Khong tim thay ROI o do: {slots_path}")
        parking_detector = ParkingDetector(
            str(slots_path),
            smoothing_frames=args.parking_smoothing,
        )
        binder = SlotVehicleBinder(
            release_grace_frames=args.slot_release_grace,
            stop_seconds=args.slot_stop_seconds,
            exit_seconds=args.slot_exit_seconds,
            min_vehicle_overlap=args.slot_min_vehicle_overlap,
            strong_vehicle_overlap=args.slot_strong_vehicle_overlap,
            stationary_radius_ratio=args.slot_stationary_radius_ratio,
            stationary_drift_ratio=args.slot_stationary_drift_ratio,
            recovery_expand_ratio=args.slot_recovery_expand_ratio,
        )

    direction_detector: Optional[DirectionDetector] = None
    roi_path = _resolve_project_path(args.roi_file)
    if roi_path.is_file():
        direction_detector = DirectionDetector.from_json(str(roi_path))

    tracker = MotionVehicleTracker(
        min_visible_count=args.min_visible_count,
        lost_track_ttl=args.lost_track_ttl,
        history_len=args.history_len,
        min_area=args.motion_min_area,
        max_distance=args.motion_max_distance,
        min_confirm_displacement=args.motion_min_displacement,
        motion_frame_gap=args.motion_frame_gap,
        motion_threshold=args.motion_threshold,
        motion_min_ratio=args.motion_min_ratio,
        slot_binder=binder,
    )

    raw_writer = _open_video_writer(session_dir / "raw_video.mp4", record_fps, (width, height))
    debug_writer = _open_video_writer(session_dir / "debug_video.mp4", record_fps, (width, height))

    timestamp_file = (session_dir / "frame_timestamps.csv").open("w", newline="", encoding="utf-8-sig")
    performance_file = (session_dir / "performance.csv").open("w", newline="", encoding="utf-8-sig")
    prediction_file = (session_dir / "predictions.jsonl").open("w", encoding="utf-8")
    timestamp_writer = csv.writer(timestamp_file)
    performance_writer = csv.writer(performance_file)
    timestamp_writer.writerow([
        "frame_idx", "source_time_ms", "elapsed_ms", "capture_unix_ns", "wall_time_iso",
    ])
    performance_writer.writerow([
        "frame_idx", "capture_ms", "tracking_ms", "parking_ms", "render_ms",
        "recording_ms", "total_processing_ms", "instant_processing_fps",
        "process_cpu_percent", "process_ram_mb",
    ])

    started_wall = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    next_parking_time_s = 0.0
    slot_results: Optional[list] = None
    process = psutil.Process() if psutil is not None else None
    if process is not None:
        process.cpu_percent(interval=None)
    last_cpu: Optional[float] = None
    last_ram: Optional[float] = None
    status = "running"
    error_message: Optional[str] = None
    processed_frames = 0

    metadata = {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "started_at": started_wall.isoformat(timespec="milliseconds"),
        "source": {
            "kind": source_kind,
            "description": source_description,
            "width": width,
            "height": height,
            "source_fps": source_fps,
            "record_fps": record_fps,
        },
        "algorithm": {
            "tracker": "motion_mog2_kalman_lapjv_hsv",
            "id_scope": "single_camera",
            "parking_enabled": parking_detector is not None,
            "slots_file": str(slots_path) if slots_path else None,
            "direction_roi_file": str(roi_path) if roi_path.is_file() else None,
        },
        "arguments": vars(args),
        "files": {
            "raw_video": "raw_video.mp4",
            "debug_video": "debug_video.mp4",
            "predictions": "predictions.jsonl",
            "timestamps": "frame_timestamps.csv",
            "performance": "performance.csv",
            "ground_truth_slots": "ground_truth_slots.csv",
            "ground_truth_events": "ground_truth_events.csv",
        },
    }
    _write_json_atomic(session_dir / "session_info.json", metadata)

    try:
        while True:
            frame_started = time.perf_counter()
            capture_started = frame_started
            ok, frame = cap.read()
            capture_ended = time.perf_counter()
            if not ok:
                status = "completed"
                break

            capture_unix_ns = time.time_ns()
            wall_time_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
            elapsed_s = time.monotonic() - started_monotonic
            if source_kind == "video":
                source_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if source_time_s <= 0:
                    source_time_s = (processed_frames + 1) / max(source_fps, 1.0)
            else:
                source_time_s = elapsed_s

            tracking_started = time.perf_counter()
            tracks, _, _ = tracker.process_frame(frame)
            direction_events = (
                direction_detector.update(tracker.active_tracks, tracker.frame_index)
                if direction_detector is not None else []
            )
            tracking_ended = time.perf_counter()

            parking_started = tracking_ended
            if parking_detector is not None and binder is not None:
                interval_s = 1.0 / max(args.parking_fps, 0.01)
                if slot_results is None or source_time_s >= next_parking_time_s:
                    slot_results = parking_detector.detect(frame, apply_smoothing=True)
                    binder.update_vision(slot_results, tracker.frame_index, source_time_s)
                    next_parking_time_s = source_time_s + interval_s
                # Confirmed + LOST retain measured bbox long enough to decide a stop.
                binder.update_tracks(tracker.confirmed_tracks, tracker.frame_index, source_time_s)
            parking_ended = time.perf_counter()

            processing_before_render_ms = (parking_ended - frame_started) * 1000.0
            render_started = parking_ended
            debug_frame = _draw_debug_frame(
                frame, tracker, parking_detector, slot_results, direction_detector,
                source_time_s * 1000.0, processing_before_render_ms,
            )
            render_ended = time.perf_counter()

            prediction = _build_prediction(
                session_id=session_id,
                tracker=tracker,
                binder=binder,
                source_time_ms=source_time_s * 1000.0,
                elapsed_ms=elapsed_s * 1000.0,
                wall_time_iso=wall_time_iso,
                direction_events=direction_events,
            )

            recording_started = time.perf_counter()
            raw_writer.write(frame)
            debug_writer.write(debug_frame)
            prediction_file.write(json.dumps(
                prediction, ensure_ascii=False, separators=(",", ":"), default=_json_default,
            ) + "\n")
            timestamp_writer.writerow([
                tracker.frame_index,
                f"{source_time_s * 1000.0:.3f}",
                f"{elapsed_s * 1000.0:.3f}",
                capture_unix_ns,
                wall_time_iso,
            ])
            recording_ended = time.perf_counter()

            total_ms = (recording_ended - frame_started) * 1000.0
            if process is not None and (tracker.frame_index == 1 or tracker.frame_index % args.resource_sample_every == 0):
                last_cpu = float(process.cpu_percent(interval=None))
                last_ram = float(process.memory_info().rss / (1024.0 * 1024.0))
            performance_writer.writerow([
                tracker.frame_index,
                f"{(capture_ended - capture_started) * 1000.0:.3f}",
                f"{(tracking_ended - tracking_started) * 1000.0:.3f}",
                f"{(parking_ended - parking_started) * 1000.0:.3f}",
                f"{(render_ended - render_started) * 1000.0:.3f}",
                f"{(recording_ended - recording_started) * 1000.0:.3f}",
                f"{total_ms:.3f}",
                f"{1000.0 / max(total_ms, 0.001):.3f}",
                "" if last_cpu is None else f"{last_cpu:.2f}",
                "" if last_ram is None else f"{last_ram:.2f}",
            ])

            processed_frames += 1
            if processed_frames % args.flush_every == 0:
                timestamp_file.flush()
                performance_file.flush()
                prediction_file.flush()

            if not args.no_display:
                cv2.imshow("TechGAR experiment recorder", debug_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    status = "stopped_by_user"
                    break

            if args.max_frames > 0 and processed_frames >= args.max_frames:
                status = "completed_max_frames"
                break

            if source_kind == "video" and not args.as_fast_as_possible:
                target_fps = args.playback_fps if args.playback_fps > 0 else source_fps
                remaining = (1.0 / max(target_fps, 0.01)) - (time.perf_counter() - frame_started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        status = "stopped_by_user"
    except Exception as exc:
        status = "failed"
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cap.release()
        raw_writer.release()
        debug_writer.release()
        timestamp_file.close()
        performance_file.close()
        prediction_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        metadata.update({
            "status": status,
            "error": error_message,
            "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "processed_frames": processed_frames,
            "duration_wall_seconds": round(time.monotonic() - started_monotonic, 3),
        })
        _write_json_atomic(session_dir / "session_info.json", metadata)

    return session_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ghi mot session thuc nghiem TechGAR day du video, JSONL va CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Chi so camera USB/virtual webcam, vi du 0")
    source.add_argument("--stream-url", help="URL MJPEG/RTSP tu dien thoai")
    source.add_argument("--video", default="data/carPark.mp4", help="Video dau vao")
    parser.add_argument("--output-root", default="experiment_test/output")
    parser.add_argument("--session-name", help="Ten session; bo trong de tu tao theo thoi gian")
    parser.add_argument("--slots-file", default="config/parking_slots.json")
    parser.add_argument("--roi-file", default="config/roi_lines.json")
    parser.add_argument("--disable-parking", action="store_true")
    parser.add_argument("--parking-fps", type=float, default=2.0)
    parser.add_argument("--parking-smoothing", type=int, default=5)
    parser.add_argument("--record-fps", type=float, default=0.0, help="0 = lay FPS cua nguon")
    parser.add_argument("--playback-fps", type=float, default=0.0, help="0 = toc do goc khi doc video")
    parser.add_argument("--as-fast-as-possible", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = chay den het/nhan Q")
    parser.add_argument("--flush-every", type=int, default=30)
    parser.add_argument("--resource-sample-every", type=int, default=10)

    parser.add_argument("--min-visible-count", type=int, default=3)
    parser.add_argument("--lost-track-ttl", type=int, default=90)
    parser.add_argument("--history-len", type=int, default=90)
    parser.add_argument("--motion-min-area", type=int, default=650)
    parser.add_argument("--motion-max-distance", type=float, default=180.0)
    parser.add_argument("--motion-min-displacement", type=float, default=12.0)
    parser.add_argument("--motion-frame-gap", type=int, default=3)
    parser.add_argument("--motion-threshold", type=int, default=25)
    parser.add_argument("--motion-min-ratio", type=float, default=0.08)

    parser.add_argument("--slot-release-grace", type=int, default=90)
    parser.add_argument("--slot-stop-seconds", type=float, default=1.0)
    parser.add_argument("--slot-exit-seconds", type=float, default=0.5)
    parser.add_argument("--slot-min-vehicle-overlap", type=float, default=0.35)
    parser.add_argument("--slot-strong-vehicle-overlap", type=float, default=0.60)
    parser.add_argument("--slot-stationary-radius-ratio", type=float, default=0.06)
    parser.add_argument("--slot-stationary-drift-ratio", type=float, default=0.10)
    parser.add_argument("--slot-recovery-expand-ratio", type=float, default=0.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.parking_fps <= 0:
        raise SystemExit("--parking-fps phai > 0")
    if args.flush_every <= 0 or args.resource_sample_every <= 0:
        raise SystemExit("--flush-every va --resource-sample-every phai > 0")
    session_dir = record(args)
    print(f"\nDa ghi xong session: {session_dir}")
    print(f"Kiem tra bang: python experiment_test/validate_session.py --session \"{session_dir}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
