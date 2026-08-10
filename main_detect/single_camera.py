"""Single-camera entry point for live/video tracking and parking fusion.

Ví dụ:
  python single_camera.py --video data/carPark.mp4 --no-display
  python single_camera.py --camera 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from techgar.direction_detector import DirectionDetector
from techgar.motion_tracker import MotionVehicleTracker
from techgar.vehicle_tracker import VehicleTracker

# Parking slot integration (optional)
try:
    from techgar.parking_detector import ParkingDetector
    from techgar.slot_vehicle_binder import SlotVehicleBinder
    _HAS_PARKING = True
except ImportError:
    _HAS_PARKING = False

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


def load_homography(path: Optional[str]) -> Optional[np.ndarray]:
    """Đọc ma trận 3x3 từ ``[[...], [...], [...]]`` hoặc ``{"homography": ...}``."""
    if not path:
        return None
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as file:
        value = json.load(file)
    matrix = value.get("homography", value) if isinstance(value, dict) else value
    homography = np.asarray(matrix, dtype=np.float32)
    if homography.shape != (3, 3):
        raise ValueError(f"{input_path} phải chứa ma trận homography 3x3")
    return homography


def resolve_video_path(value: str, project_root: Path) -> str:
    """Resolve a video from the current directory or submission root."""
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, project_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Không tìm thấy video: {value}. Đã kiểm tra: {checked}")


def vehicle_to_json(track) -> dict:
    position = {"x": track.cx, "y": track.cy, "reference": "bbox_bottom_center"}
    if track.ground_point is not None:
        position["ground_x"] = track.ground_point[0]
        position["ground_y"] = track.ground_point[1]
    return {
        "track_id": track.track_id,
        "status": track.status.value,
        "position": position,
        "bbox": {"x": track.x, "y": track.y, "w": track.w, "h": track.h},
        "confidence": round(track.confidence, 4),
        "vehicle_class": track.class_name,
        "area": int(track.area),
        "age": track.age,
        "visible_count": track.total_visible_count,
        "invisible_count": track.consecutive_invisible_count,
        "last_seen_frame": track.last_seen_frame,
        "direction_events": track.direction_events,
        "trail": [{"x": x, "y": y} for x, y in track.history[-30:]],
    }


def build_positions_json(
    tracker,
    frame_w: int,
    frame_h: int,
    binder=None,
) -> dict:
    """Chỉ đưa xe nhìn thấy thật vào ``active_vehicles``; không lẫn dự đoán cũ."""
    active = {}
    for track_id, track in tracker.active_tracks.items():
        vj = vehicle_to_json(track)
        # Thêm thông tin ô đỗ nếu có binder
        if binder is not None:
            slot_id = binder.get_slot_for_vehicle(track_id)
            vj["parked_in_slot"] = slot_id
        active[str(track_id)] = vj

    lost = {
        str(track_id): vehicle_to_json(track)
        for track_id, track in tracker.confirmed_tracks.items()
        if track.status.value == "lost"
    }
    exited = {
        str(track_id): {
            "track_id": track_id,
            "entered_frame": track.entered_frame,
            "exited_frame": track.exited_frame,
            "last_known_position": {"x": track.cx, "y": track.cy, "reference": "bbox_bottom_center"},
            "total_visible_frames": track.total_visible_count,
            "direction_events": track.direction_events,
        }
        for track_id, track in tracker.exited_tracks.items()
    }

    result = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "frame_index": tracker.frame_index,
        "frame_size": {"width": frame_w, "height": frame_h},
        "position_reference": "image pixel; x/y = bottom-center of vehicle bounding box",
        "active_count": len(active),
        "lost_count": len(lost),
        "exited_count": len(exited),
        "active_vehicles": active,
        "lost_vehicles": lost,
        "exited_vehicles": exited,
    }

    # Thêm parking_slots nếu có binder
    if binder is not None:
        result["parking_slots"] = binder.to_json()

    return result


def save_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def run(args: argparse.Namespace) -> None:
    source = args.camera if args.camera is not None else args.video
    cap = cv2.VideoCapture(source)
    if args.camera is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video/camera: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = Path(args.output)

    # ── Khởi tạo Parking Detector + Binder (optional) ──
    parking_detector = None
    binder = None
    if _HAS_PARKING and args.slots_file:
        slots_path = Path(args.slots_file)
        if not slots_path.exists():
            base = Path(__file__).resolve().parent.parent
            alt = base / args.slots_file
            if alt.exists():
                slots_path = alt
        if slots_path.exists():
            parking_detector = ParkingDetector(
                slots_file=str(slots_path),
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
            print(f"\U0001f17f\ufe0f Parking: {parking_detector.slot_count} slots from {slots_path.name}")
        else:
            print(f"\u26a0\ufe0f Slots file not found: {args.slots_file} (parking detection disabled)")
    elif not _HAS_PARKING and args.slots_file:
        print("\u26a0\ufe0f parking_detector/slot_vehicle_binder not importable (parking detection disabled)")

    common_tracker_args = {
        "min_visible_count": args.min_visible_count,
        "lost_track_ttl": args.lost_track_ttl,
        "history_len": args.history_len,
        "homography": load_homography(args.homography),
    }
    if args.backend == "yolo":
        tracker = VehicleTracker(
            model_path=args.model,
            tracker_config=args.tracker,
            confidence=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            **common_tracker_args,
        )
    else:
        tracker = MotionVehicleTracker(
            min_area=args.motion_min_area,
            max_distance=args.motion_max_distance,
            min_confirm_displacement=args.motion_min_displacement,
            motion_frame_gap=args.motion_frame_gap,
            motion_threshold=args.motion_threshold,
            motion_min_ratio=args.motion_min_ratio,
            slot_binder=binder,  # Truyền binder vào motion tracker
            **common_tracker_args,
        )
    direction_detector = DirectionDetector.from_json(
        args.roi,
        decision_frames=args.direction_frames,
        min_after_distance=args.direction_distance,
    )

    print(f"Nguồn: {source} | {frame_w}x{frame_h} | source FPS: {source_fps:.2f}")
    if args.backend == "yolo":
        print(f"Backend: YOLO | Model: {Path(args.model).resolve().name} | Tracker: {Path(args.tracker).resolve().name}")
    else:
        print("Backend: motion + Kalman + global assignment + HSV Re-ID")
    print(f"JSON: {output_path.resolve()}")
    last_json_at = 0.0
    last_parking_at = 0.0
    last_report_at = time.perf_counter()
    report_frames = 0
    target_fps = source_fps if args.realtime else args.playback_fps
    frame_period = 1.0 / target_fps if target_fps > 0 else 0.0
    parking_interval = 1.0 / args.parking_fps if args.parking_fps > 0 else 1.0
    last_slot_results = None

    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop and args.camera is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            tracks, debug_mask, _ = tracker.process_frame(frame)
            direction_detector.update(tracker.active_tracks, tracker.frame_index)

            # ── Parking detection (chạy ở tần suất thấp hơn) ──
            now = time.monotonic()
            if parking_detector is not None and binder is not None:
                if args.camera is not None:
                    tracking_timestamp_s = now
                else:
                    tracking_timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if tracking_timestamp_s <= 0:
                        tracking_timestamp_s = tracker.frame_index / max(source_fps, 1.0)
                # Movement/stop evidence must be sampled on every source frame,
                # not only when the slower ensemble detector runs.
                binder.update_tracks(
                    # Motion detections disappear shortly after a car stops.
                    # LOST tracks retain the last measured bbox (not a Kalman
                    # prediction), allowing the one-second stationary window
                    # to complete while lost_track_ttl is still active.
                    tracker.confirmed_tracks,
                    tracker.frame_index,
                    tracking_timestamp_s,
                )
                if now - last_parking_at >= parking_interval:
                    last_slot_results = parking_detector.detect(frame, apply_smoothing=True)
                    binder.update_vision(
                        last_slot_results,
                        tracker.frame_index,
                        tracking_timestamp_s,
                    )
                    last_parking_at = now

            if now - last_json_at >= 1.0 / args.json_fps:
                save_json_atomic(
                    build_positions_json(tracker, frame_w, frame_h, binder=binder),
                    output_path,
                )
                last_json_at = now

            report_frames += 1
            if args.verbose and now - last_report_at >= 1.0:
                effective_fps = report_frames / (time.perf_counter() - last_report_at)
                parked_info = ""
                if binder is not None:
                    parked_count = len(binder.get_all_parked_vehicle_ids())
                    parked_info = f" parked={parked_count}"
                print(
                    f"frame={tracker.frame_index} active={len(tracker.active_tracks)} "
                    f"lost={len(tracker.confirmed_tracks) - len(tracker.active_tracks)}"
                    f"{parked_info} fps={effective_fps:.1f}"
                )
                last_report_at, report_frames = time.perf_counter(), 0

            if not args.no_display:
                display = tracker.draw_tracks(frame, tracks, show_non_active=args.show_debug_tracks)
                direction_detector.draw_roi_lines(display)
                # Vẽ overlay parking slots nếu có
                if parking_detector is not None and last_slot_results is not None:
                    display = parking_detector.draw_results(display, last_slot_results)
                cv2.imshow("TechGAR vehicle tracker", display)
                cv2.imshow("Debug mask", debug_mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.max_frames and tracker.frame_index >= args.max_frames:
                break
            if frame_period:
                remaining = frame_period - (time.perf_counter() - frame_started_at)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_json_atomic(
            build_positions_json(tracker, frame_w, frame_h, binder=binder),
            output_path,
        )


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Realtime vehicle tracking: YOLO + BoT-SORT/Re-ID")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--video", "-v", help="Đường dẫn video")
    source.add_argument("--camera", "-c", type=int, help="Camera ID")
    parser.add_argument("--model", default=str(root / "models" / "yolov8n.pt"), help="YOLO .pt model (chỉ cần khi --backend yolo)")
    parser.add_argument("--tracker", default=str(root / "config" / "botsort_parking_reid.yaml"), help="BoT-SORT tracker YAML")
    parser.add_argument("--backend", choices=("motion", "yolo"), default="motion", help="motion cho camera top-down; yolo cho model đã fine-tune")
    parser.add_argument("--roi", default=str(root / "config" / "roi_lines.json"), help="ROI lines JSON")
    parser.add_argument("--homography", help="JSON ma trận 3x3 chuyển pixel sang sơ đồ/mét")
    parser.add_argument("--output", "-o", default=str(root / "runtime_output" / "vehicle_positions.json"))
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    parser.add_argument("--device", help="auto/CUDA device, ví dụ 0 hoặc cpu")
    parser.add_argument("--min-visible-count", type=int, default=4, help="Detection liên tiếp để xác nhận xe")
    parser.add_argument("--lost-track-ttl", type=int, default=90, help="Số frame giữ vị trí lost trong JSON/debug")
    parser.add_argument("--motion-min-area", type=int, default=900, help="Diện tích foreground tối thiểu khi dùng backend motion")
    parser.add_argument("--motion-max-distance", type=float, default=180.0, help="Gate Kalman (pixel) khi dùng backend motion")
    parser.add_argument("--motion-min-displacement", type=float, default=12.0, help="Pixel tối thiểu trước khi xác nhận track motion")
    parser.add_argument("--motion-frame-gap", type=int, default=3, help="Khoảng cách frame để kiểm tra chuyển động")
    parser.add_argument("--motion-threshold", type=int, default=25, help="Ngưỡng khác biệt pixel để được coi là chuyển động")
    parser.add_argument("--motion-min-ratio", type=float, default=0.08, help="Tỷ lệ pixel phải chuyển động trong bbox")
    parser.add_argument("--history-len", type=int, default=90)
    parser.add_argument("--direction-frames", type=int, default=8)
    parser.add_argument("--direction-distance", type=float, default=35.0, help="Pixel phải đi thêm trước khi chốt hướng")
    parser.add_argument("--json-fps", type=float, default=10.0, help="Tần suất ghi JSON; không làm chậm inference")
    parser.add_argument("--realtime", action="store_true", help="Phát video đúng FPS gốc")
    parser.add_argument("--playback-fps", type=float, default=0.0, help="Giới hạn FPS phát video (0 = nhanh nhất)")
    parser.add_argument("--max-frames", type=int, default=0, help="Chỉ xử lý N frame (0 = toàn bộ nguồn)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--show-debug-tracks", action="store_true", help="Hiện cả tentative/lost")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # Parking slot integration
    parser.add_argument("--slots-file", default=str(root / "config" / "parking_slots.json"), help="JSON polygon ô đỗ; truyền chuỗi rỗng để tắt")
    parser.add_argument("--parking-fps", type=float, default=2.0, help="Tần suất chạy parking detection (lần/giây)")
    parser.add_argument("--parking-smoothing", type=int, default=5, help="Số frame đồng thuận để đổi trạng thái ô đỗ")
    parser.add_argument("--slot-release-grace", type=int, default=45, help="Số frame giữ vehicle_id sau khi ô trống")
    parser.add_argument("--slot-stop-seconds", type=float, default=1.0, help="Số giây xe phải đứng ổn định trước khi gán ô")
    parser.add_argument("--slot-exit-seconds", type=float, default=0.5, help="Số giây cùng ID phải nằm ngoài ROI trước khi gỡ tracking override")
    parser.add_argument("--slot-min-vehicle-overlap", type=float, default=0.35, help="Overlap bbox tối thiểu khi tâm xe nằm trong ROI")
    parser.add_argument("--slot-strong-vehicle-overlap", type=float, default=0.60, help="Overlap đủ mạnh để nhận ROI dù tâm bbox nằm ngoài")
    parser.add_argument("--slot-stationary-radius-ratio", type=float, default=0.06, help="Bán kính rung tối đa / đường chéo bbox")
    parser.add_argument("--slot-stationary-drift-ratio", type=float, default=0.10, help="Độ trôi đầu-cuối tối đa / đường chéo bbox")
    parser.add_argument("--slot-recovery-expand-ratio", type=float, default=0.15, help="Tỷ lệ mở rộng ROI để nhận lại ID xe rời ô")
    args = parser.parse_args()
    if args.json_fps <= 0:
        parser.error("--json-fps phải lớn hơn 0")
    if args.max_frames < 0:
        parser.error("--max-frames không được âm")
    if args.playback_fps < 0:
        parser.error("--playback-fps không được âm")
    if args.slot_stop_seconds <= 0 or args.slot_exit_seconds <= 0:
        parser.error("--slot-stop-seconds và --slot-exit-seconds phải > 0")
    if not 0 <= args.slot_min_vehicle_overlap <= 1 or not 0 <= args.slot_strong_vehicle_overlap <= 1:
        parser.error("Các ngưỡng slot overlap phải nằm trong [0, 1]")
    if args.slot_min_vehicle_overlap > args.slot_strong_vehicle_overlap:
        parser.error("--slot-min-vehicle-overlap không được lớn hơn --slot-strong-vehicle-overlap")
    if args.video is None and args.camera is None:
        fallback = root / "data" / "carPark.mp4"
        if not fallback.exists():
            parser.error("Cần --video hoặc --camera")
        args.video = str(fallback)
    elif args.video is not None:
        try:
            args.video = resolve_video_path(args.video, root)
        except FileNotFoundError as error:
            parser.error(str(error))
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"Lỗi tracker: {error}", file=sys.stderr)
        raise SystemExit(1)
