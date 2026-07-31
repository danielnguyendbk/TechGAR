"""Entry point realtime: YOLO detection + BoT-SORT/Re-ID + toạ độ xe.

Ví dụ:
  python tracker_main.py --video ../dataset/carPark.mp4 --no-display
  python tracker_main.py --camera 0 --device 0
  python tracker_main.py --video ../dataset/carPark.mp4 --homography homography.json
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

from direction_detector import DirectionDetector
from motion_tracker import MotionVehicleTracker
from vehicle_tracker import VehicleTracker

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
    """Tìm video tương đối từ cwd hoặc từ thư mục gốc project.

    Nhờ đó khi đang ở ``detect_car_update`` vẫn dùng được
    ``--video dataset/output2_video.mp4`` thay vì phải tự thêm ``..``.
    """
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


def build_positions_json(tracker: VehicleTracker, frame_w: int, frame_h: int) -> dict:
    """Chỉ đưa xe nhìn thấy thật vào ``active_vehicles``; không lẫn dự đoán cũ."""
    active = {str(track_id): vehicle_to_json(track) for track_id, track in tracker.active_tracks.items()}
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
    return {
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
        # Webcam/IP camera: không tích backlog frame cũ khi inference chậm.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video/camera: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = Path(args.output)
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
    last_report_at = time.perf_counter()
    report_frames = 0
    target_fps = source_fps if args.realtime else args.playback_fps
    frame_period = 1.0 / target_fps if target_fps > 0 else 0.0

    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop and args.camera is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            tracks, debug_mask = tracker.process_frame(frame)
            # Hướng chỉ được tính từ detection thật, không dùng vị trí predicted/lost.
            direction_detector.update(tracker.active_tracks, tracker.frame_index)
            now = time.monotonic()
            if now - last_json_at >= 1.0 / args.json_fps:
                save_json_atomic(build_positions_json(tracker, frame_w, frame_h), output_path)
                last_json_at = now

            report_frames += 1
            if args.verbose and now - last_report_at >= 1.0:
                effective_fps = report_frames / (time.perf_counter() - last_report_at)
                print(
                    f"frame={tracker.frame_index} active={len(tracker.active_tracks)} "
                    f"lost={len(tracker.confirmed_tracks) - len(tracker.active_tracks)} "
                    f"processing_fps={effective_fps:.1f}"
                )
                last_report_at, report_frames = time.perf_counter(), 0

            if not args.no_display:
                display = tracker.draw_tracks(frame, tracks, show_non_active=args.show_debug_tracks)
                direction_detector.draw_roi_lines(display)
                cv2.imshow("TechGAR vehicle tracker", display)
                cv2.imshow("YOLO tracked areas", debug_mask)
                key = cv2.waitKey(1) & 0xFF  # Không throttle pipeline bằng FPS nguồn.
                if key in (ord("q"), 27):
                    break
            if args.max_frames and tracker.frame_index >= args.max_frames:
                break
            # Video file: giới hạn tốc độ phát để quan sát dễ hơn. Mặc định 0 là
            # xử lý nhanh nhất có thể, phù hợp chạy headless/production.
            if frame_period:
                remaining = frame_period - (time.perf_counter() - frame_started_at)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_json_atomic(build_positions_json(tracker, frame_w, frame_h), output_path)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    root = base.parent
    parser = argparse.ArgumentParser(description="Realtime vehicle tracking: YOLO + BoT-SORT/Re-ID")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--video", "-v", help="Đường dẫn video")
    source.add_argument("--camera", "-c", type=int, help="Camera ID")
    parser.add_argument("--model", default=str(root / "yolov8n.pt"), help="YOLO .pt model")
    parser.add_argument("--tracker", default=str(base / "botsort_parking_reid.yaml"), help="BoT-SORT tracker YAML")
    parser.add_argument("--backend", choices=("motion", "yolo"), default="motion", help="motion cho camera top-down; yolo cho model đã fine-tune")
    parser.add_argument("--roi", default=str(base / "roi_lines.json"), help="ROI lines JSON")
    parser.add_argument("--homography", help="JSON ma trận 3x3 chuyển pixel sang sơ đồ/mét")
    parser.add_argument("--output", "-o", default=str(base / "vehicle_positions.json"))
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
    parser.add_argument("--show-debug-tracks", action="store_true", help="Hiện cả tentative/lost; mặc định chỉ hiện xe đang di chuyển")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.json_fps <= 0:
        parser.error("--json-fps phải lớn hơn 0")
    if args.max_frames < 0:
        parser.error("--max-frames không được âm")
    if args.playback_fps < 0:
        parser.error("--playback-fps không được âm")
    if args.video is None and args.camera is None:
        fallback = root / "dataset" / "carPark.mp4"
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
