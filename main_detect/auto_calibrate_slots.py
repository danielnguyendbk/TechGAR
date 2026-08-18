"""Auto-calibrate parking slot polygons to match actual camera view.

Workflow:
  1. Capture 1 frame from each camera (empty parking lot required).
  2. Find white parking lines via adaptive thresholding.
  3. Snap each polygon vertex to the nearest white-line pixel (within search_radius).
  4. Validate: run ParkingDetector on the calibrated frame — all slots must be "empty".
  5. Save updated JSON + debug images.

Usage:
  python auto_calibrate_slots.py \
    --cam1-url "http://192.168.100.53:4747/video/force/1280x720" \
    --cam2-url "http://192.168.100.137:4747/video/force/1280x720" \
    --slots-cam1 config/parking_slots_cam1.json \
    --slots-cam2 config/parking_slots_cam2.json \
    --search-radius 15
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def capture_frame(url: str, retries: int = 3, timeout_ms: int = 10000) -> np.ndarray:
    """Capture a single frame from camera URL."""
    for attempt in range(retries):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        if cap.isOpened():
            # Skip a few frames to get a stable image
            for _ in range(5):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                print(f"  ✅ Chụp thành công ({frame.shape[1]}x{frame.shape[0]})")
                return frame
        cap.release()
        print(f"  ⚠️ Thử lần {attempt + 1}/{retries}...")
        time.sleep(1)
    raise RuntimeError(f"Không thể chụp frame từ {url}")


def find_line_pixels(frame: np.ndarray) -> np.ndarray:
    """Find white parking line pixels using adaptive thresholding.
    
    Returns a binary mask where white line pixels = 255.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Global threshold for bright white lines
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Method 2: Adaptive threshold to catch lines under varying lighting
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, -8
    )
    # -8 means pixel must be 8 brighter than local mean → catches white lines
    
    # Combine both
    combined = cv2.bitwise_or(bright_mask, adaptive)
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return combined


def snap_vertex_to_line(
    vertex: Tuple[int, int],
    line_mask: np.ndarray,
    search_radius: int,
    h: int,
    w: int,
) -> Tuple[int, int, float]:
    """Snap a polygon vertex to the nearest white-line pixel.
    
    Returns (new_x, new_y, distance_moved).
    """
    vx, vy = vertex
    best_dist = float("inf")
    best_point = (vx, vy)
    
    # Search in a square region around the vertex
    y_start = max(0, vy - search_radius)
    y_end = min(h, vy + search_radius + 1)
    x_start = max(0, vx - search_radius)
    x_end = min(w, vx + search_radius + 1)
    
    # Extract the search region
    roi = line_mask[y_start:y_end, x_start:x_end]
    white_points = np.argwhere(roi == 255)  # (row, col) = (y, x)
    
    if len(white_points) == 0:
        return vx, vy, 0.0
    
    # Convert to global coordinates
    white_points[:, 0] += y_start  # y
    white_points[:, 1] += x_start  # x
    
    # Find nearest white pixel
    distances = np.sqrt(
        (white_points[:, 1].astype(float) - vx) ** 2
        + (white_points[:, 0].astype(float) - vy) ** 2
    )
    min_idx = np.argmin(distances)
    best_dist = float(distances[min_idx])
    best_point = (int(white_points[min_idx, 1]), int(white_points[min_idx, 0]))
    
    return best_point[0], best_point[1], best_dist


def calibrate_slots(
    frame: np.ndarray,
    slots_data: dict,
    search_radius: int = 15,
) -> Tuple[dict, np.ndarray, dict]:
    """Calibrate slot polygons by snapping vertices to detected white lines.
    
    Returns (updated_data, debug_image, stats).
    """
    h, w = frame.shape[:2]
    line_mask = find_line_pixels(frame)
    
    updated = copy.deepcopy(slots_data)
    debug = frame.copy()
    
    # Draw line mask overlay (blue tint for detected lines)
    line_overlay = np.zeros_like(debug)
    line_overlay[line_mask == 255] = (255, 100, 0)  # Blue
    debug = cv2.addWeighted(debug, 0.7, line_overlay, 0.3, 0)
    
    total_vertices = 0
    moved_vertices = 0
    total_distance = 0.0
    max_distance = 0.0
    
    for slot in updated["slots"]:
        poly = slot.get("polygon", [])
        if not poly:
            continue
            
        for i, point in enumerate(poly):
            if isinstance(point, dict):
                ox, oy = int(point["x"]), int(point["y"])
            else:
                ox, oy = int(point[0]), int(point[1])
            
            nx, ny, dist = snap_vertex_to_line((ox, oy), line_mask, search_radius, h, w)
            total_vertices += 1
            
            if dist > 0:
                moved_vertices += 1
                total_distance += dist
                max_distance = max(max_distance, dist)
                
                # Draw: old position (red) → new position (green)
                cv2.circle(debug, (ox, oy), 4, (0, 0, 255), -1)
                cv2.circle(debug, (nx, ny), 4, (0, 255, 0), -1)
                cv2.line(debug, (ox, oy), (nx, ny), (0, 255, 255), 1)
            else:
                cv2.circle(debug, (ox, oy), 3, (200, 200, 200), -1)
            
            # Update the point
            if isinstance(point, dict):
                point["x"] = nx
                point["y"] = ny
            else:
                poly[i] = [nx, ny]
        
        # Update center
        if "center" in slot and poly:
            if isinstance(poly[0], dict):
                xs = [p["x"] for p in poly]
                ys = [p["y"] for p in poly]
            else:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
            slot["center"] = {"x": int(np.mean(xs)), "y": int(np.mean(ys))}
        
        # Draw updated polygon (green)
        if isinstance(poly[0], dict):
            pts = np.array([[p["x"], p["y"]] for p in poly], np.int32)
        else:
            pts = np.array(poly, np.int32)
        cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
        
        # Draw slot ID
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        cv2.putText(debug, slot["id"], (cx - 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    stats = {
        "total_vertices": total_vertices,
        "moved_vertices": moved_vertices,
        "avg_distance": round(total_distance / max(1, moved_vertices), 2),
        "max_distance": round(max_distance, 2),
    }
    
    return updated, debug, stats


def validate_empty(frame: np.ndarray, slots_file: str) -> Tuple[int, int, List[str]]:
    """Validate that all slots show as 'empty' using ParkingDetector.
    
    Returns (total_slots, empty_count, list_of_occupied_ids).
    """
    from src.techgar.parking_detector import ParkingDetector
    
    detector = ParkingDetector(
        slots_file=slots_file,
        smoothing_frames=0,  # No smoothing for single-frame validation
    )
    results = detector.detect(frame, apply_smoothing=False)
    
    occupied_ids = [r.slot_id for r in results if r.occupied]
    empty_count = sum(1 for r in results if not r.occupied)
    
    return len(results), empty_count, occupied_ids


def main():
    parser = argparse.ArgumentParser(
        description="Auto-calibrate parking slot ROIs to match camera position"
    )
    parser.add_argument("--cam1-url", required=True, help="Camera 1 URL")
    parser.add_argument("--cam2-url", required=True, help="Camera 2 URL")
    parser.add_argument("--slots-cam1", default="config/parking_slots_cam1.json")
    parser.add_argument("--slots-cam2", default="config/parking_slots_cam2.json")
    parser.add_argument("--search-radius", type=int, default=15,
                        help="Max pixels to search for nearest line (default: 15)")
    parser.add_argument("--output-dir", default="config",
                        help="Directory to save calibrated JSON files")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating backup of original files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cameras = {
        "cam1": {"url": args.cam1_url, "slots_file": args.slots_cam1},
        "cam2": {"url": args.cam2_url, "slots_file": args.slots_cam2},
    }
    
    for cam_id, cam_info in cameras.items():
        print(f"\n{'='*60}")
        print(f"  📷 {cam_id.upper()}")
        print(f"{'='*60}")
        
        # Step 1: Capture frame
        print(f"\n[1/4] Chụp frame từ {cam_id}...")
        frame = capture_frame(cam_info["url"])
        
        # Save raw capture
        capture_path = output_dir / f"calibration_capture_{cam_id}.png"
        cv2.imwrite(str(capture_path), frame)
        print(f"  💾 Ảnh gốc: {capture_path}")
        
        # Step 2: Load existing slots
        print(f"\n[2/4] Đọc file slot: {cam_info['slots_file']}")
        slots_path = Path(cam_info["slots_file"])
        with slots_path.open("r", encoding="utf-8") as f:
            slots_data = json.load(f)
        print(f"  📋 {len(slots_data['slots'])} ô đỗ")
        
        # Step 3: Calibrate
        print(f"\n[3/4] Calibrate (search_radius={args.search_radius}px)...")
        updated_data, debug_img, stats = calibrate_slots(
            frame, slots_data, search_radius=args.search_radius
        )
        
        print(f"  📊 Kết quả:")
        print(f"     Tổng đỉnh: {stats['total_vertices']}")
        print(f"     Đã dịch:   {stats['moved_vertices']}")
        print(f"     TB dịch:   {stats['avg_distance']} px")
        print(f"     Max dịch:  {stats['max_distance']} px")
        
        # Save debug image
        debug_path = output_dir / f"calibration_debug_{cam_id}.png"
        cv2.imwrite(str(debug_path), debug_img)
        print(f"  🖼️ Debug: {debug_path}")
        
        # Step 4: Save calibrated JSON
        if not args.no_backup:
            backup_path = slots_path.with_suffix(".backup.json")
            if not backup_path.exists():
                shutil.copy2(slots_path, backup_path)
                print(f"\n  📦 Backup: {backup_path}")
        
        # Update imageWidth/imageHeight to match captured frame
        updated_data["imageWidth"] = frame.shape[1]
        updated_data["imageHeight"] = frame.shape[0]
        
        with slots_path.open("w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Đã ghi: {slots_path}")
        
        # Step 5: Validate
        print(f"\n[4/4] Kiểm tra tất cả ô phải trống...")
        total, empty, occupied = validate_empty(frame, str(slots_path))
        
        if not occupied:
            print(f"  ✅ PASS: {empty}/{total} ô đều trống!")
        else:
            print(f"  ⚠️ CẢNH BÁO: {len(occupied)}/{total} ô báo occupied sai:")
            for sid in occupied:
                print(f"     - {sid}")
            print(f"  → Có thể cần tăng search_radius hoặc kiểm tra lại ánh sáng.")
    
    print(f"\n{'='*60}")
    print(f"  🎉 Hoàn tất calibration cả 2 camera!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
