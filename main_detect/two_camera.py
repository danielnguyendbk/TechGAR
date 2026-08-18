"""Run TechGAR tracking and parking on two real camera streams.

This entrypoint deliberately leaves ``main.py`` unchanged: main.py remains the
four-crop simulator, while this file accepts two independent MJPEG/RTSP feeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from techgar.cross_camera_manager import CrossCameraManager
from techgar.latest_frame_capture import LatestFrameCapture
from techgar.live_roi_editor import LiveROIEditor, MAIN_WINDOW
from techgar.motion_tracker import MotionVehicleTracker
from techgar.parking_detector import ParkingDetector
from techgar.slot_vehicle_binder import SlotVehicleBinder


def save_json(path: Path, payload: dict, *, tolerate_lock: bool = False) -> bool:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        for attempt in range(4):
            try:
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt == 3:
                    if tolerate_lock:
                        return False
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def load_calibration(path: Path) -> tuple[dict, dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    transforms = data.get("camera_transforms", {})
    if set(transforms) != {"cam1", "cam2"}:
        raise ValueError("calibration phai co camera_transforms cho cam1 va cam2")
    matrices = {}
    for camera_id, matrix in transforms.items():
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (3, 3) or abs(np.linalg.det(value)) < 1e-12:
            raise ValueError(f"Homography khong hop le cho {camera_id}")
        matrices[camera_id] = value

    adjacency = {}
    for item in data.get("edge_adjacency", []):
        adjacency[(item["source_camera"], item["exit_edge"])] = item["target_camera"]
    required = {("cam1", "right"): "cam2", ("cam2", "left"): "cam1"}
    if adjacency != required:
        raise ValueError("edge_adjacency phai la cam1:right -> cam2 va cam2:left -> cam1")

    polygon = np.asarray(data.get("overlap_world_polygon", []), dtype=np.float32)
    if polygon.shape != (4, 2):
        raise ValueError("overlap_world_polygon phai co dung 4 diem [x, y]")
    return matrices, adjacency, {("cam1", "cam2"): polygon}


def detector_parameters(detector: ParkingDetector) -> dict:
    return {
        "base_gamma": float(detector.base_gamma),
        "base_clahe": float(detector.base_clahe),
        "clahe_grid": int(detector.clahe_grid),
        "ratio_thr": float(detector.ratio_thr),
        "edge_thr": float(detector.edge_thr),
        "use_edge_recheck": bool(detector.use_edge_recheck),
        "border_ignore_ratio": float(detector.border_ignore_ratio),
        "line_min_span_ratio": float(detector.line_min_span_ratio),
        "line_max_thickness_ratio": float(detector.line_max_thickness_ratio),
        "core_scale": float(detector.core_scale),
        "core_ratio_threshold": float(detector.core_ratio_threshold),
        "core_component_threshold": float(detector.core_component_threshold),
    }


def apply_detector_parameters(detector: ParkingDetector, values: dict) -> None:
    detector.base_gamma = max(0.1, float(values.get("base_gamma", detector.base_gamma)))
    detector.base_clahe = max(0.1, float(values.get("base_clahe", detector.base_clahe)))
    detector.clahe_grid = max(2, int(values.get("clahe_grid", detector.clahe_grid)))
    detector.ratio_thr = min(1.0, max(0.01, float(values.get("ratio_thr", detector.ratio_thr))))
    detector.edge_thr = min(1.0, max(0.01, float(values.get("edge_thr", detector.edge_thr))))
    detector.use_edge_recheck = bool(values.get("use_edge_recheck", detector.use_edge_recheck))
    detector.configure_roi_filter(values)
    # The real two-camera runner uses threshold pixels only, even with an old profile.
    detector.use_edge_recheck = False


def select_moving_tracks(
    active_tracks: dict,
    local_to_global: dict[int, int],
    parked_global_ids: set[int],
    canonicalize: Callable[[int], int],
) -> tuple[dict, dict[int, int]]:
    """Return visible non-parked tracks and their canonical Global IDs."""
    canonical_parked = {canonicalize(global_id) for global_id in parked_global_ids}
    moving_tracks = {}
    shown_ids = {}
    for local_id, track in active_tracks.items():
        global_id = local_to_global.get(local_id)
        if global_id is None:
            continue
        global_id = canonicalize(global_id)
        if global_id in canonical_parked:
            continue
        moving_tracks[local_id] = track
        shown_ids[local_id] = global_id
    return moving_tracks, shown_ids


def process_parking_frame(
    detector: ParkingDetector,
    frame: np.ndarray,
    include_debug: bool,
) -> tuple[list, tuple[np.ndarray, np.ndarray] | None]:
    """Run slow parking work outside the live display loop."""
    results = detector.detect(frame, apply_smoothing=True)
    debug = detector.build_debug_images(frame) if include_debug else None
    return results, debug


class DetectorTuningPanel:
    WINDOW = "Parking detector settings"
    FIELDS = (
        ("Gamma x10", "base_gamma", 10.0, 60),
        ("CLAHE x10", "base_clahe", 10.0, 60),
        ("Grid", "clahe_grid", 1.0, 32),
        ("Ratio %", "ratio_thr", 100.0, 100),
    )

    def __init__(self, detectors: dict[str, ParkingDetector], profile_path: Path):
        self.detectors = detectors
        self.profile_path = profile_path
        if profile_path.is_file():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            for camera_id, detector in detectors.items():
                apply_detector_parameters(detector, profile.get(camera_id, {}))
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 560, 440)
        for camera_id, detector in detectors.items():
            prefix = camera_id.upper()
            for label, attribute, scale, maximum in self.FIELDS:
                value = int(round(getattr(detector, attribute) * scale))
                cv2.createTrackbar(f"{prefix} {label}", self.WINDOW, value, maximum, lambda _value: None)

    def apply(self) -> None:
        for camera_id, detector in self.detectors.items():
            prefix = camera_id.upper()
            values = {}
            for label, attribute, scale, _maximum in self.FIELDS:
                values[attribute] = cv2.getTrackbarPos(f"{prefix} {label}", self.WINDOW) / scale
            apply_detector_parameters(detector, values)

    def snapshot(self) -> dict:
        return {camera_id: detector_parameters(detector) for camera_id, detector in self.detectors.items()}

    def save(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(self.profile_path, self.snapshot())
        print(f"Da luu detector profile: {self.profile_path}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hai camera that: tracking, parking va Global ID")
    parser.add_argument("--cam1-url", required=True)
    parser.add_argument("--cam2-url", required=True)
    parser.add_argument("--slots-cam1", required=True)
    parser.add_argument("--slots-cam2", required=True)
    parser.add_argument("--calibration", required=True, help="JSON homography va overlap da hieu chinh")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runtime_output_two_camera"))
    parser.add_argument("--session-dir", help="Thu muc ghi video/JSONL hai camera; phai chua ton tai")
    parser.add_argument("--detector-profile", default=str(PROJECT_ROOT / "config" / "two_camera_detector.local.json"))
    parser.add_argument("--no-parking-debug", action="store_true", help="An cua so threshold pixel de giam tai")
    parser.add_argument("--parking-fps", type=float, default=2.0)
    parser.add_argument("--json-fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--min-visible-count", type=int, default=4)
    parser.add_argument("--lost-track-ttl", type=int, default=90)
    parser.add_argument("--motion-min-area", type=int, default=900)
    parser.add_argument("--motion-max-distance", type=float, default=180.0)
    parser.add_argument("--motion-min-displacement", type=float, default=12.0)
    parser.add_argument("--handoff-ttl", type=int, default=45)
    parser.add_argument("--handoff-match-distance", type=float, default=100.0)
    parser.add_argument("--handoff-appearance-threshold", type=float, default=0.45)
    parser.add_argument("--handoff-lookahead-frames", type=int, default=16)
    parser.add_argument("--handoff-prediction-radius", type=float, default=90.0)
    parser.add_argument("--handoff-min-direction-cosine", type=float, default=0.25)
    return parser


def run(args: argparse.Namespace) -> None:
    calibration_path = Path(args.calibration).resolve()
    transforms, adjacency, overlap_regions = load_calibration(calibration_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dir = Path(args.session_dir).resolve() if args.session_dir else None
    if session_dir is not None and session_dir.exists():
        raise FileExistsError(f"Session da ton tai: {session_dir}")

    captures = {}
    writers = {}
    timestamps_file = performance_file = predictions_file = None
    status = "failed"
    frame_index = 0
    session_created = False
    started_at = datetime.now().astimezone()
    tuning_panel = None
    roi_editor = None
    parking_executor = None
    try:
        camera_urls = {"cam1": args.cam1_url, "cam2": args.cam2_url}
        for camera_id, url in camera_urls.items():
            captures[camera_id] = LatestFrameCapture(url).start()

        frames = {}
        capture_sequences = {}
        for camera_id, capture in captures.items():
            frame, sequence = capture.read_latest(timeout=10.0)
            frames[camera_id] = frame
            capture_sequences[camera_id] = sequence

        sizes = {camera_id: (frame.shape[1], frame.shape[0]) for camera_id, frame in frames.items()}
        if session_dir is not None:
            session_dir.mkdir(parents=True)
            session_created = True
            for camera_id, (width, height) in sizes.items():
                fps = float(captures[camera_id].get(cv2.CAP_PROP_FPS)) or 25.0
                writers[camera_id] = (
                    cv2.VideoWriter(str(session_dir / f"raw_{camera_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)),
                    cv2.VideoWriter(str(session_dir / f"debug_{camera_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)),
                )
            timestamps_file = (session_dir / "frame_timestamps.csv").open("w", newline="", encoding="utf-8-sig")
            performance_file = (session_dir / "performance.csv").open("w", newline="", encoding="utf-8-sig")
            predictions_file = (session_dir / "predictions.jsonl").open("w", encoding="utf-8")
            csv.writer(timestamps_file).writerow(["frame_idx", "capture_unix_ns", "wall_time_iso"])
            csv.writer(performance_file).writerow(["frame_idx", "total_processing_ms"])
            for filename, header in (("ground_truth_slots.csv", ["camera_id", "slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes"]), ("ground_truth_events.csv", ["event_id", "global_id", "source_camera", "target_camera", "event_type", "frame_idx", "notes"])):
                with (session_dir / filename).open("w", newline="", encoding="utf-8-sig") as ground_truth:
                    csv.writer(ground_truth).writerow(header)
        manager = CrossCameraManager(
            camera_sizes=sizes,
            camera_crops={camera_id: (0, 0, *size) for camera_id, size in sizes.items()},
            camera_transforms=transforms,
            edge_adjacency=adjacency,
            overlap_regions=overlap_regions,
            handoff_ttl=args.handoff_ttl,
            match_distance=args.handoff_match_distance,
            appearance_threshold=args.handoff_appearance_threshold,
            lookahead_frames=args.handoff_lookahead_frames,
            prediction_radius=args.handoff_prediction_radius,
            min_direction_cosine=args.handoff_min_direction_cosine,
        )
        slot_files = {"cam1": args.slots_cam1, "cam2": args.slots_cam2}
        detectors = {
            camera_id: ParkingDetector(slot_file, use_edge_recheck=False)
            for camera_id, slot_file in slot_files.items()
        }
        parking_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="parking")
        profile_path = Path(args.detector_profile).resolve()
        if args.no_display:
            if profile_path.is_file():
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                for camera_id, detector in detectors.items():
                    apply_detector_parameters(detector, profile.get(camera_id, {}))
        else:
            tuning_panel = DetectorTuningPanel(detectors, profile_path)
        binders = {camera_id: SlotVehicleBinder() for camera_id in frames}
        if not args.no_display:
            roi_editor = LiveROIEditor(slot_files, sizes)
            cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(MAIN_WINDOW, roi_editor.handle_mouse)
        trackers = {
            camera_id: MotionVehicleTracker(
                min_visible_count=args.min_visible_count,
                lost_track_ttl=args.lost_track_ttl,
                min_area=args.motion_min_area,
                max_distance=args.motion_max_distance,
                min_confirm_displacement=args.motion_min_displacement,
                slot_binder=None,
            )
            for camera_id in frames
        }
        slot_results = {camera_id: [] for camera_id in frames}
        threshold_debug = {}
        last_parking_at = 0.0
        last_json_at = 0.0
        last_json_warning_at = 0.0
        parking_futures = {}
        parking_job_frame_index = 0
        parking_job_timestamp = 0.0

        while True:
            started = time.perf_counter()
            if frame_index:
                for camera_id, capture in captures.items():
                    frame, sequence = capture.read_latest(
                        after_sequence=capture_sequences[camera_id], timeout=5.0
                    )
                    frames[camera_id] = frame
                    capture_sequences[camera_id] = sequence
            frame_index += 1
            now = time.monotonic()
            if tuning_panel is not None and not parking_futures:
                tuning_panel.apply()

            for camera_id, tracker in trackers.items():
                _, _, expired = tracker.process_frame(frames[camera_id])
                for local_id, track in tracker.newly_lost_tracks:
                    manager.notify_track_lost(camera_id, local_id, track, frame_index)
                for local_id, track in expired:
                    manager.notify_track_expired(camera_id, local_id, track.cx, track.cy, track.w, track.h, getattr(track, "appearance", None), frame_index)

            observable = {camera_id: tracker.observable_tracks for camera_id, tracker in trackers.items()}
            for camera_id, tracks in observable.items():
                for local_id, track in tracks.items():
                    if manager.get_global_id(camera_id, local_id) is None:
                        recovered = binders[camera_id].try_recover_id(
                            camera_id=camera_id,
                            bbox=(track.x, track.y, track.w, track.h),
                            appearance=getattr(track, "appearance", None),
                        )
                        if recovered is not None:
                            manager.bind_external_id(camera_id, local_id, recovered, frame_index, source="parking_slot_release")
            global_ids = manager.update_all_tracks(observable, frame_index)

            for camera_id, binder in binders.items():
                binder.remap_vehicle_ids(manager.canonical_global_id)
                global_tracks = {
                    global_id: trackers[camera_id].confirmed_tracks[local_id]
                    for local_id, global_id in global_ids.get(camera_id, {}).items()
                    if local_id in trackers[camera_id].confirmed_tracks
                }
                binder.update_tracks(global_tracks, frame_index, now)

            if parking_futures and all(future.done() for future in parking_futures.values()):
                for camera_id, future in parking_futures.items():
                    slot_results[camera_id], debug_images = future.result()
                    binders[camera_id].update_vision(
                        slot_results[camera_id],
                        parking_job_frame_index,
                        parking_job_timestamp,
                        camera_id=camera_id,
                    )
                    if debug_images is not None:
                        threshold_debug[camera_id] = debug_images
                parking_futures = {}

            if (
                not parking_futures
                and now - last_parking_at >= 1.0 / max(args.parking_fps, 0.01)
            ):
                include_parking_debug = not args.no_display and not args.no_parking_debug
                parking_futures = {
                    camera_id: parking_executor.submit(
                        process_parking_frame,
                        detector,
                        frames[camera_id],
                        include_parking_debug,
                    )
                    for camera_id, detector in detectors.items()
                }
                parking_job_frame_index = frame_index
                parking_job_timestamp = now
                last_parking_at = now

            confirmed = {camera_id: trackers[camera_id].confirmed_tracks for camera_id in trackers}
            registry = manager.to_json(confirmed)
            if now - last_json_at >= 1.0 / max(args.json_fps, 0.01):
                timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                json_write_ok = True
                for camera_id in frames:
                    json_write_ok &= save_json(output_dir / f"parking_status_{camera_id}.json", {
                        "timestamp": timestamp, "frame_index": frame_index, "camera_id": camera_id,
                        "parking_slots": binders[camera_id].to_json(camera_id=camera_id),
                    }, tolerate_lock=True)
                json_write_ok &= save_json(output_dir / "global_vehicle_registry.json", {
                    "timestamp": timestamp, "frame_index": frame_index,
                    "calibration": str(calibration_path), **registry,
                }, tolerate_lock=True)
                if not json_write_ok and now - last_json_warning_at >= 5.0:
                    print("Canh bao: JSON runtime dang bi khoa; bo qua mot lan cap nhat.")
                    last_json_warning_at = now
                last_json_at = now

            tracking_frames = {}
            debug_frames = {}
            if not args.no_display or writers:
                parked_global_ids = {
                    global_id
                    for binder in binders.values()
                    for global_id in binder.get_all_parked_vehicle_ids()
                }
                for camera_id in ("cam1", "cam2"):
                    moving_tracks, shown_ids = select_moving_tracks(
                        trackers[camera_id].active_tracks,
                        global_ids.get(camera_id, {}),
                        parked_global_ids,
                        manager.canonical_global_id,
                    )
                    debug = trackers[camera_id].draw_tracks(
                        frames[camera_id],
                        moving_tracks,
                        id_overrides=shown_ids,
                        confirmed_color=(255, 0, 0),
                        confirmed_label="moving",
                        point_color=(255, 0, 0),
                    )
                    cv2.putText(
                        debug,
                        f"Frame: {frame_index}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )
                    tracking_frames[camera_id] = debug
                    debug_frames[camera_id] = detectors[camera_id].draw_results(
                        debug, slot_results[camera_id]
                    )
            if writers:
                for camera_id, (raw_writer, debug_writer) in writers.items():
                    raw_writer.write(frames[camera_id])
                    debug_writer.write(debug_frames[camera_id])
                capture_ns = time.time_ns()
                timestamps_file.write(f"{frame_index},{capture_ns},{datetime.now().astimezone().isoformat(timespec='milliseconds')}\n")
                predictions_file.write(json.dumps({
                    "schema_version": 2, "frame_idx": frame_index,
                    "cameras": {camera_id: {"confirmed_vehicles": list(global_ids.get(camera_id, {}).values()), "parking_slots": binders[camera_id].to_json(camera_id=camera_id)} for camera_id in frames},
                    "global_registry": registry,
                }, ensure_ascii=False) + "\n")
                performance_file.write(f"{frame_index},{(time.perf_counter() - started) * 1000.0:.3f}\n")
            if not args.no_display:
                camera_views = {}
                for camera_id in ("cam1", "cam2"):
                    source = tracking_frames[camera_id] if roi_editor.enabled else debug_frames[camera_id]
                    view = cv2.resize(source, (640, 360))
                    if roi_editor.enabled:
                        view = roi_editor.render_camera(camera_id, view, slot_results[camera_id])
                    camera_views[camera_id] = view
                cv2.imshow(MAIN_WINDOW, roi_editor.compose_main_view(camera_views))
                if not args.no_parking_debug and set(threshold_debug) == {"cam1", "cam2"}:
                    threshold_rows = []
                    for camera_id in ("cam1", "cam2"):
                        raw_view, filtered_view = threshold_debug[camera_id]
                        threshold_rows.append(np.hstack([
                            cv2.resize(raw_view, (480, 270)),
                            cv2.resize(filtered_view, (480, 270)),
                        ]))
                    cv2.imshow(
                        "B/W pixels - raw | filtered (cam1 top, cam2 bottom)",
                        np.vstack(threshold_rows),
                    )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord("S")):
                    if parking_futures:
                        for future in parking_futures.values():
                            future.result()
                        parking_futures = {}
                    if tuning_panel is not None:
                        tuning_panel.apply()
                        tuning_panel.save()
                    saved_camera_ids = roi_editor.save()
                    for camera_id in saved_camera_ids:
                        old_detector = detectors[camera_id]
                        parameters = detector_parameters(old_detector)
                        detector = ParkingDetector(
                            slot_files[camera_id],
                            smoothing_frames=old_detector.smoothing_frames,
                            use_edge_recheck=False,
                        )
                        apply_detector_parameters(detector, parameters)
                        detectors[camera_id] = detector
                        slot_results[camera_id] = detector.detect(
                            frames[camera_id], apply_smoothing=False
                        )
                        binders[camera_id].retain_slot_ids(set(detector.slot_ids))
                        binders[camera_id].update_vision(
                            slot_results[camera_id], frame_index, now, camera_id=camera_id
                        )
                        if not args.no_parking_debug:
                            threshold_debug[camera_id] = detector.build_debug_images(frames[camera_id])
                        print(f"Da luu va ap dung ROI moi cho {camera_id}")
                elif key in (ord("q"), ord("Q")):
                    saved_camera_ids = roi_editor.save()
                    if saved_camera_ids:
                        print("Da luu ROI truoc khi thoat: " + ", ".join(sorted(saved_camera_ids)))
                    break
                elif key == 27:
                    break
                else:
                    roi_editor.handle_key(key)
            if args.max_frames and frame_index >= args.max_frames:
                break
            if time.perf_counter() - started < 0.001:
                time.sleep(0.001)
        status = "completed_max_frames" if args.max_frames else "stopped_by_user"
    except KeyboardInterrupt:
        status = "stopped_by_user"
    finally:
        if parking_executor is not None:
            parking_executor.shutdown(wait=True)
        if tuning_panel is not None:
            tuning_panel.apply()
            tuning_panel.save()
        for capture in captures.values():
            capture.release()
        for raw_writer, debug_writer in writers.values():
            raw_writer.release()
            debug_writer.release()
        for output in (timestamps_file, performance_file, predictions_file):
            if output is not None:
                output.close()
        if session_created:
            save_json(session_dir / "session_info.json", {
                "schema_version": 2,
                "status": status,
                "started_at": started_at.isoformat(timespec="milliseconds"),
                "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "processed_frames": frame_index,
                "camera_ids": ["cam1", "cam2"],
                "calibration": str(calibration_path),
                "detector_parameters": tuning_panel.snapshot() if tuning_panel is not None else {
                    camera_id: detector_parameters(detector) for camera_id, detector in detectors.items()
                } if "detectors" in locals() else {},
                "files": {"raw_cam1": "raw_cam1.mp4", "raw_cam2": "raw_cam2.mp4", "debug_cam1": "debug_cam1.mp4", "debug_cam2": "debug_cam2.mp4"},
            })
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(make_parser().parse_args())
