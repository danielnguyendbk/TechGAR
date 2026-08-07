"""
tracker_main.py — Entry point v2: chạy tracker + direction detection, xuất JSON.

Cách chạy:
  python tracker_main.py --video ../dataset/carPark.mp4
  python tracker_main.py --video ../dataset/carPark.mp4 --no-display --verbose
  python tracker_main.py --video ../dataset/carPark.mp4 --roi roi_lines.json
  python tracker_main.py --camera 0

Output:
  vehicle_positions.json — tọa độ xe (confirmed + exited), hướng đi, trail.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from vehicle_tracker import VehicleTracker, TrackStatus
from direction_detector import DirectionDetector

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def build_positions_json(
    tracker: VehicleTracker,
    direction_detector: DirectionDetector,
    frame_w: int,
    frame_h: int,
) -> dict:
    """
    Tạo dict JSON mô tả tọa độ hiện tại + lịch sử xe.

    Bao gồm:
      - active_vehicles: xe confirmed đang trong bãi
      - exited_vehicles: xe đã rời bãi (lưu lại lịch sử)
    """
    confirmed = tracker.confirmed_tracks
    exited = tracker.exited_tracks

    active_vehicles = {}
    for tid, t in confirmed.items():
        active_vehicles[str(tid)] = {
            "track_id": tid,
            "status": t.status.value,
            "center": {"x": t.cx, "y": t.cy},
            "bbox": {"x": t.x, "y": t.y, "w": t.w, "h": t.h},
            "area": int(t.area),
            "age": t.age,
            "visible_count": t.total_visible_count,
            "invisible_count": t.consecutive_invisible_count,
            "direction_events": t.direction_events,
            "trail": [{"x": px, "y": py} for px, py in t.history[-20:]],  # Giới hạn 20 điểm
        }

    exited_vehicles = {}
    for tid, t in exited.items():
        exited_vehicles[str(tid)] = {
            "track_id": tid,
            "entered_frame": t.entered_frame,
            "exited_frame": t.exited_frame,
            "last_known_position": {"x": t.cx, "y": t.cy},
            "total_visible_frames": t.total_visible_count,
            "direction_events": t.direction_events,
        }

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index": tracker.frame_index,
        "frame_size": {"width": frame_w, "height": frame_h},
        "active_count": len(active_vehicles),
        "total_tracked": len(active_vehicles) + len(exited_vehicles),
        "active_vehicles": active_vehicles,
        "exited_vehicles": exited_vehicles,
    }


def save_json_atomic(data: dict, path: Path):
    """Ghi JSON an toàn bằng cách ghi vào file tạm trước rồi rename."""
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def run(args):
    # ── Mở video / camera ──
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"Camera #{args.camera}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video

    if not cap.isOpened():
        print(f"❌ Không mở được: {source_name}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ Nguồn: {source_name}")
    print(f"   Kích thước: {frame_w}x{frame_h} | FPS: {fps:.1f} | Tổng frames: {total_frames}")

    # ── Khởi tạo tracker ──
    tracker = VehicleTracker(
        history=args.bg_history,
        var_threshold=args.bg_var_threshold,
        gamma=args.gamma,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        min_area=args.min_area,
        min_width=args.min_width,
        min_height=args.min_height,
        merge_distance=args.merge_distance,
        merge_size_ratio=args.merge_size_ratio,
        age_threshold=args.age_threshold,
        min_visible_count=args.min_visible_count,
        invisible_for_too_long=args.invisible_too_long,
        max_distance=args.max_distance,
        reid_distance=args.reid_distance,
        reid_max_frames=args.reid_max_frames,
    )

    # ── Khởi tạo direction detector ──
    direction_detector = DirectionDetector.from_json(
        args.roi,
        decision_frames=args.direction_frames,
    )

    output_path = Path(args.output)
    json_interval = max(1, int(fps / args.json_fps))

    print(f"   JSON output: {output_path.resolve()}")
    print(f"   JSON update mỗi {json_interval} frames (~{args.json_fps} lần/giây)")
    print(f"   {'Chế độ headless' if args.no_display else 'Nhấn Q để thoát'}")
    print("─" * 50)

    loop_video = args.loop

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop_video and args.camera is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker = VehicleTracker(
                    history=args.bg_history,
                    var_threshold=args.bg_var_threshold,
                    gamma=args.gamma,
                    clahe_clip=args.clahe_clip,
                    clahe_grid=args.clahe_grid,
                    min_area=args.min_area,
                    min_width=args.min_width,
                    min_height=args.min_height,
                    merge_distance=args.merge_distance,
                    merge_size_ratio=args.merge_size_ratio,
                    age_threshold=args.age_threshold,
                    min_visible_count=args.min_visible_count,
                    invisible_for_too_long=args.invisible_too_long,
                    max_distance=args.max_distance,
                    reid_distance=args.reid_distance,
                    reid_max_frames=args.reid_max_frames,
                )
                direction_detector = DirectionDetector.from_json(
                    args.roi, decision_frames=args.direction_frames
                )
                print("🔄 Loop video — reset tracker")
                continue
            else:
                print("📹 Video kết thúc.")
                break

        # ── Tracking ──
        tracks, fg_mask = tracker.process_frame(frame)

        # ── Direction detection ──
        new_events = direction_detector.update(tracker.confirmed_tracks, tracker.frame_index)

        # ── Ghi JSON ──
        if tracker.frame_index % json_interval == 0:
            positions = build_positions_json(tracker, direction_detector, frame_w, frame_h)
            save_json_atomic(positions, output_path)

            if args.verbose:
                confirmed = len(tracker.confirmed_tracks)
                tentative = len(tracker.all_tracks) - confirmed
                exited = len(tracker.exited_tracks)
                print(
                    f"  Frame {tracker.frame_index:>6d} | "
                    f"Confirmed: {confirmed} | Tentative: {tentative} | "
                    f"Exited: {exited}"
                )

        # ── Hiển thị ──
        if not args.no_display:
            display = tracker.draw_tracks(frame, tracks)
            direction_detector.draw_roi_lines(display)

            info = f"Frame: {tracker.frame_index} | {output_path.name}"
            cv2.putText(display, info, (10, frame_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("TechGAR Vehicle Tracker v2", display)
            cv2.imshow("FG Mask", fg_mask)

            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q") or key == 27:
                print("⏹️ Dừng.")
                break

    # ── Cleanup ──
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    positions = build_positions_json(tracker, direction_detector, frame_w, frame_h)
    save_json_atomic(positions, output_path)
    print(f"✅ JSON cuối: {output_path.resolve()}")
    print(f"   Confirmed: {len(tracker.confirmed_tracks)} | Exited: {len(tracker.exited_tracks)}")


def main():
    parser = argparse.ArgumentParser(
        description="TechGAR Vehicle Tracker v2 — Particle Filter + Direction Detection"
    )

    # Nguồn video
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", "-v", type=str, help="Đường dẫn video")
    src.add_argument("--camera", "-c", type=int, default=None, help="Camera ID")

    # Tham số tracker
    parser.add_argument("--min-area", type=int, default=800,
                        help="Diện tích contour tối thiểu (pixel²)")
    parser.add_argument("--min-width", type=int, default=25,
                        help="Chiều rộng bbox tối thiểu (pixel)")
    parser.add_argument("--min-height", type=int, default=20,
                        help="Chiều cao bbox tối thiểu (pixel)")
    parser.add_argument("--max-distance", type=float, default=120.0,
                        help="Khoảng cách tối đa cho Hungarian assignment")
    parser.add_argument("--age-threshold", type=int, default=8,
                        help="Số frame tối thiểu để track được xét confirm")
    parser.add_argument("--min-visible-count", type=int, default=8,
                        help="Số frame visible tối thiểu để confirm")
    parser.add_argument("--invisible-too-long", type=int, default=30,
                        help="Số frame mất dấu trước khi xóa track")
    parser.add_argument("--bg-history", type=int, default=500,
                        help="MOG2 background history")
    parser.add_argument("--bg-var-threshold", type=int, default=50,
                        help="MOG2 variance threshold")

    # Tiền xử lý ảnh
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (< 1 = sáng hơn, > 1 = tối hơn, 1.0 = tắt)")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit (0 = tắt CLAHE)")
    parser.add_argument("--clahe-grid", type=int, default=8,
                        help="CLAHE grid size")

    # Gộp box
    parser.add_argument("--merge-distance", type=float, default=60.0,
                        help="Khoảng cách tâm tối đa để gộp 2 box (pixel)")
    parser.add_argument("--merge-size-ratio", type=float, default=0.5,
                        help="Tỷ lệ area/median dưới ngưỡng này = box bất thường")

    # Re-ID
    parser.add_argument("--reid-distance", type=float, default=100.0,
                        help="Khoảng cách pixel tối đa để nhận lại xe cũ")
    parser.add_argument("--reid-max-frames", type=int, default=60,
                        help="Xe mất tối đa N frame vẫn có thể nhận lại")

    # Direction detection
    parser.add_argument("--roi", type=str, default="roi_lines.json",
                        help="File ROI lines JSON")
    parser.add_argument("--direction-frames", type=int, default=10,
                        help="Số frame sau cắt vạch để quyết định hướng")

    # Output
    parser.add_argument("--output", "-o", type=str, default="vehicle_positions.json",
                        help="File JSON output")
    parser.add_argument("--json-fps", type=float, default=5.0,
                        help="Số lần cập nhật JSON/giây")

    # Display
    parser.add_argument("--no-display", action="store_true", help="Chạy headless")
    parser.add_argument("--loop", action="store_true", help="Lặp video")
    parser.add_argument("--verbose", action="store_true", help="Log chi tiết")

    args = parser.parse_args()

    if args.video is None and args.camera is None:
        ref_video = Path(__file__).parent / "_ref_repo" / "video.mp4"
        if ref_video.exists():
            args.video = str(ref_video)
            print(f"ℹ️ Dùng video mẫu: {ref_video}")
        else:
            parser.error("Cần --video hoặc --camera")

    run(args)


if __name__ == "__main__":
    main()
