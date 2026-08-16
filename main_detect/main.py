"""TechGAR main demo — tracking, Global ID and parking on four virtual cameras.

Chia video gốc thành 4 phần (top-left, top-right, bottom-left, bottom-right),
tự động tạo file parking_slots cho mỗi camera (tọa độ đã shift),
và vẽ dấu cộng ở giữa để minh họa vùng cắt.

Cách dùng:
  python main.py
  python main.py --preview
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow the submission to run directly without installing it as a package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# ══════════════════════════════════════════════════════
#  Camera Simulator
# ══════════════════════════════════════════════════════

class CameraSimulator:
    """Giả lập nhiều camera bằng cách cắt 1 frame thành các vùng.

    Mỗi camera có:
      - crop region (x1, y1, x2, y2) trên frame gốc
      - danh sách slots thuộc về camera đó (tọa độ đã shift về local)
      - output JSON riêng
    """

    def __init__(
        self,
        source_path: str,
        slots_file: str,
        overlap: int = 0,
        output_dir: Optional[str] = None,
    ):
        self.source_path = source_path
        # Predictive handoff no longer needs duplicated pixels at the border.
        # Keeping this at zero avoids two local motion trackers seeing and
        # temporarily splitting the same car into two global candidates.
        self.overlap = overlap

        # Load video info
        video_src = int(source_path) if source_path.isdigit() else source_path
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được video: {source_path}")
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Tính điểm cắt (giữa frame)
        self.mid_x = self.frame_w // 2
        self.mid_y = self.frame_h // 2

        # Vùng crop cho mỗi camera. Overlap is optional for diagnostics only.
        self.cameras: Dict[str, dict] = {
            "cam1": {
                "name": "Top-Left",
                "crop": (0, 0, self.mid_x + overlap, self.mid_y + overlap),
            },
            "cam2": {
                "name": "Top-Right",
                "crop": (self.mid_x - overlap, 0, self.frame_w, self.mid_y + overlap),
            },
            "cam3": {
                "name": "Bottom-Left",
                "crop": (0, self.mid_y - overlap, self.mid_x + overlap, self.frame_h),
            },
            "cam4": {
                "name": "Bottom-Right",
                "crop": (self.mid_x - overlap, self.mid_y - overlap, self.frame_w, self.frame_h),
            },
        }

        # Tính kích thước mỗi camera
        for cam_id, cam in self.cameras.items():
            x1, y1, x2, y2 = cam["crop"]
            cam["width"] = x2 - x1
            cam["height"] = y2 - y1

        # Output dir
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load và phân chia slots
        self._load_and_split_slots(slots_file)

    def _load_and_split_slots(self, slots_file: str) -> None:
        """Load slots, phân loại vào camera dựa trên center, shift tọa độ."""
        path = Path(slots_file)
        if not path.exists():
            # Tìm từ thư mục cha
            alt = Path(__file__).resolve().parent.parent / slots_file
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy slots file: {slots_file}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        all_slots = data["slots"]
        ref_w = data["imageWidth"]
        ref_h = data["imageHeight"]

        # Scale factor (slots file có thể dùng kích thước khác video)
        sx = self.frame_w / ref_w
        sy = self.frame_h / ref_h

        for cam_id, cam in self.cameras.items():
            cam["slots"] = []
            cam["slots_raw"] = []  # Giữ bản gốc để debug

        for slot in all_slots:
            # Scale center về video coords
            center_x = slot["center"]["x"] * sx
            center_y = slot["center"]["y"] * sy

            # Xác định camera nào chứa center
            cam_id = self._classify_point(center_x, center_y)

            # Scale và shift polygon về tọa độ local camera
            crop = self.cameras[cam_id]["crop"]
            local_polygon = []
            for p in slot["polygon"]:
                lx = p["x"] * sx - crop[0]
                ly = p["y"] * sy - crop[1]
                local_polygon.append({"x": round(lx), "y": round(ly)})

            local_center = {
                "x": round(center_x - crop[0]),
                "y": round(center_y - crop[1]),
            }

            local_slot = {
                "id": slot["id"],
                "type": slot.get("type", "polygon"),
                "polygon": local_polygon,
                "center": local_center,
                "status": slot.get("status", "empty"),
            }
            self.cameras[cam_id]["slots"].append(local_slot)
            self.cameras[cam_id]["slots_raw"].append(slot)

        # Tạo file JSON cho mỗi camera
        for cam_id, cam in self.cameras.items():
            cam_slots_file = self.output_dir / f"parking_slots_{cam_id}.json"
            cam_data = {
                "imageWidth": cam["width"],
                "imageHeight": cam["height"],
                "camera_id": cam_id,
                "camera_name": cam["name"],
                "source_crop": list(cam["crop"]),
                "slots": cam["slots"],
            }
            with cam_slots_file.open("w", encoding="utf-8") as f:
                json.dump(cam_data, f, ensure_ascii=False, indent=2)
            cam["slots_file"] = str(cam_slots_file)
            print(f"  📷 {cam_id} ({cam['name']}): {len(cam['slots'])} slots → {cam_slots_file.name}")

    def _classify_point(self, x: float, y: float) -> str:
        """Phân loại point vào camera nào dựa trên quadrant."""
        if x < self.mid_x and y < self.mid_y:
            return "cam1"
        elif x >= self.mid_x and y < self.mid_y:
            return "cam2"
        elif x < self.mid_x and y >= self.mid_y:
            return "cam3"
        else:
            return "cam4"

    def get_camera_frame(self, frame: np.ndarray, cam_id: str) -> np.ndarray:
        """Cắt frame gốc thành frame cho 1 camera."""
        x1, y1, x2, y2 = self.cameras[cam_id]["crop"]
        return frame[y1:y2, x1:x2].copy()

    def get_all_camera_frames(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Cắt frame gốc thành 4 frame cho 4 camera."""
        return {cam_id: self.get_camera_frame(frame, cam_id) for cam_id in self.cameras}

    def draw_split_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ dấu cộng ở giữa frame để minh họa vùng cắt 4 camera."""
        out = frame.copy()
        # Đường kẻ dọc giữa
        cv2.line(out, (self.mid_x, 0), (self.mid_x, self.frame_h), (0, 255, 255), 2)
        # Đường kẻ ngang giữa
        cv2.line(out, (0, self.mid_y), (self.frame_w, self.mid_y), (0, 255, 255), 2)

        # Nhãn camera
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = {
            "cam1": (10, 30),
            "cam2": (self.mid_x + 10, 30),
            "cam3": (10, self.mid_y + 30),
            "cam4": (self.mid_x + 10, self.mid_y + 30),
        }
        for cam_id, (lx, ly) in labels.items():
            cam = self.cameras[cam_id]
            text = f"{cam_id}: {cam['name']} ({len(cam['slots'])} slots)"
            cv2.putText(out, text, (lx, ly), font, 0.55, (0, 255, 255), 2)

        # Dấu cộng ở tâm
        size = 20
        cv2.line(out, (self.mid_x - size, self.mid_y), (self.mid_x + size, self.mid_y), (0, 0, 255), 3)
        cv2.line(out, (self.mid_x, self.mid_y - size), (self.mid_x, self.mid_y + size), (0, 0, 255), 3)

        return out

    def draw_camera_with_slots(self, cam_frame: np.ndarray, cam_id: str) -> np.ndarray:
        """Vẽ polygon ô đỗ lên frame camera."""
        out = cam_frame.copy()
        cam = self.cameras[cam_id]
        for slot in cam["slots"]:
            pts = np.array([[p["x"], p["y"]] for p in slot["polygon"]], np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 1)
            cx, cy = slot["center"]["x"], slot["center"]["y"]
            cv2.putText(out, slot["id"], (cx - 12, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Header
        text = f"{cam_id}: {cam['name']} | {len(cam['slots'])} slots"
        cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return out


# ══════════════════════════════════════════════════════
#  Preview mode — xem trước 4 camera trên 1 cửa sổ
# ══════════════════════════════════════════════════════

def run_preview(sim: CameraSimulator, source: str, loop: bool = True, playback_fps: float = 0.0):
    """Chạy preview 4 camera song song trên 1 cửa sổ."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được: {source}")

    print("\n🎬 Preview mode — nhấn Q để thoát")
    frame_period = 1.0 / playback_fps if playback_fps > 0 else 0.0
    while True:
        frame_started_at = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        # Frame gốc với dấu cộng
        overview = sim.draw_split_overlay(frame)
        overview_small = cv2.resize(overview, (550, 360))

        # 4 camera views
        cam_frames = sim.get_all_camera_frames(frame)
        views = {}
        for cam_id, cf in cam_frames.items():
            views[cam_id] = sim.draw_camera_with_slots(cf, cam_id)

        # Resize cả 4 về cùng kích thước
        target_w, target_h = 400, 280
        resized = {cid: cv2.resize(v, (target_w, target_h)) for cid, v in views.items()}

        # Ghép 2x2
        top_row = np.hstack([resized["cam1"], resized["cam2"]])
        bot_row = np.hstack([resized["cam3"], resized["cam4"]])
        grid = np.vstack([top_row, bot_row])

        cv2.imshow("Multi-Camera Overview", overview_small)
        cv2.imshow("4 Cameras", grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if frame_period:
            remaining = frame_period - (time.perf_counter() - frame_started_at)
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════
#  Production mode — parking detection + vehicle tracking trên mỗi camera
# ══════════════════════════════════════════════════════

def _save_json_atomic(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except PermissionError:
        pass


def run_detection(sim: CameraSimulator, source: str, args):
    """Chạy ensemble parking detection + motion tracking trên 4 camera."""
    from techgar.parking_detector import ParkingDetector
    from techgar.motion_tracker import MotionVehicleTracker
    from techgar.slot_vehicle_binder import SlotVehicleBinder
    from techgar.cross_camera_manager import CrossCameraManager

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được: {source}")

    camera_sizes = {
        cam_id: (cam["width"], cam["height"])
        for cam_id, cam in sim.cameras.items()
    }
    manager = CrossCameraManager(
        camera_sizes=camera_sizes,
        camera_crops={cam_id: tuple(cam["crop"]) for cam_id, cam in sim.cameras.items()},
        edge_margin=40,
        handoff_ttl=args.handoff_ttl,
        match_distance=args.handoff_match_distance,
        appearance_threshold=args.handoff_appearance_threshold,
        lookahead_frames=args.handoff_lookahead_frames,
        prediction_radius=args.handoff_prediction_radius,
        min_direction_cosine=args.handoff_min_direction_cosine,
    )

    # ── Khởi tạo detector + tracker + binder cho mỗi camera ──
    detectors: Dict[str, ParkingDetector] = {}
    trackers: Dict[str, MotionVehicleTracker] = {}
    binders: Dict[str, SlotVehicleBinder] = {}
    parking_binder = SlotVehicleBinder(
        release_grace_frames=args.slot_release_grace,
        bind_confirmations=args.slot_bind_confirmations,
        stop_seconds=args.slot_stop_seconds,
        exit_seconds=args.slot_exit_seconds,
        min_vehicle_overlap=args.slot_min_vehicle_overlap,
        strong_vehicle_overlap=args.slot_strong_vehicle_overlap,
        stationary_radius_ratio=args.slot_stationary_radius_ratio,
        stationary_drift_ratio=args.slot_stationary_drift_ratio,
        recovery_expand_ratio=args.slot_recovery_expand_ratio,
    )

    for cam_id, cam in sim.cameras.items():
        if not cam["slots"]:
            print(f"  ⚠️ {cam_id} không có slot nào, bỏ qua")
            continue

        # Parking detector (ensemble)
        detectors[cam_id] = ParkingDetector(
            slots_file=cam["slots_file"],
            base_gamma=args.base_gamma,
            base_clahe=args.base_clahe,
            clahe_grid=args.clahe_grid,
            ratio_thr=args.ratio_thr,
            edge_thr=args.edge_thr,
            smoothing_frames=args.parking_smoothing,
        )

        # Slot-Vehicle binder
        # All four views share one binder in the source-video coordinate
        # system.  Therefore one global ID cannot be assigned to two slots in
        # different crops.
        binders[cam_id] = parking_binder

        # Motion vehicle tracker (mỗi camera có tracker riêng)
        trackers[cam_id] = MotionVehicleTracker(
            min_visible_count=args.min_visible_count,
            lost_track_ttl=args.lost_track_ttl,
            min_area=args.motion_min_area,
            max_distance=args.motion_max_distance,
            min_confirm_displacement=args.motion_min_displacement,
            # Binder works with global IDs below.  Passing it to the local
            # tracker would incorrectly reuse a cam-local ID in another view.
            slot_binder=None,
        )

        print(f"  ✅ {cam_id}: {len(cam['slots'])} slots | detector + tracker + binder")

    frame_idx = 0
    last_parking_at = 0.0
    last_json_at = 0.0
    parking_interval = 1.0 / args.parking_fps
    json_interval = 1.0 / args.json_fps
    last_slot_results: Dict[str, list] = {}
    target_fps = sim.fps if args.realtime else args.playback_fps
    frame_period = 1.0 / target_fps if target_fps > 0 else 0.0

    print(f"\n🚀 Chạy trên {len(detectors)} camera")
    print(f"   Tracking: mỗi frame | Parking detection: mỗi {parking_interval:.2f}s")
    print(f"   JSON output: mỗi {json_interval:.2f}s")
    print("   Nhấn Q để thoát\n")

    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            now = time.monotonic()
            cam_frames = sim.get_all_camera_frames(frame)

            # ── 1. Vehicle tracking (mỗi frame, mỗi camera) ──
            expired_per_cam = {}
            for cam_id, tracker in trackers.items():
                cf = cam_frames[cam_id]
                _, _, expired_tracks = tracker.process_frame(cf)
                expired_per_cam[cam_id] = [tid for tid, _ in expired_tracks]

                for tid, track in tracker.newly_lost_tracks:
                    manager.notify_track_lost(cam_id, tid, track, frame_idx)

                # Báo cho manager biết track nào vừa expire (để tạo handoff)
                for tid, track in expired_tracks:
                    manager.notify_track_expired(
                        cam_id=cam_id,
                        local_track_id=tid,
                        cx=track.cx, cy=track.cy,
                        bbox_w=track.w, bbox_h=track.h,
                        appearance=getattr(track, "appearance", None),
                        frame_idx=frame_idx,
                    )

            # Cập nhật toàn bộ camera cùng lúc. Quan trọng: manager nhìn thấy
            # xe ở camera nguồn và camera đích trong cùng frame để handoff ngay.
            # Handoff sees tentative observations so a fast vehicle receives its
            # old global ID before local confirmation.  UI/JSON stay confirmed-only.
            all_observable_tracks = {cam_id: tracker.observable_tracks for cam_id, tracker in trackers.items()}

            # A car leaving a parking slot can first appear as a tentative local
            # track. Recover its *global* ID before handoff/new-ID assignment.
            for cam_id, observations in all_observable_tracks.items():
                if cam_id not in binders:
                    continue
                crop = sim.cameras[cam_id]["crop"]
                for local_id, track in observations.items():
                    if manager.get_global_id(cam_id, local_id) is not None:
                        continue
                    recovered_global_id = parking_binder.try_recover_id(
                        position=(track.cx, track.cy),
                        camera_id=cam_id,
                        bbox=(track.x, track.y, track.w, track.h),
                        appearance=getattr(track, "appearance", None),
                        coordinate_offset=(crop[0], crop[1]),
                    )
                    if recovered_global_id is not None:
                        manager.bind_external_id(
                            cam_id, local_id, recovered_global_id, frame_idx,
                            source="parking_slot_release",
                        )
            global_ids_per_cam = manager.update_all_tracks(all_observable_tracks, frame_idx)
            all_active_tracks = {cam_id: tracker.active_tracks for cam_id, tracker in trackers.items()}

            # Canonical global observations, translated from crop-local pixels
            # back to the common source-video coordinate system.
            parking_binder.remap_vehicle_ids(manager.canonical_global_id)
            global_active_tracks = {}
            for cam_id, tracker in trackers.items():
                crop = sim.cameras[cam_id]["crop"]
                id_map = global_ids_per_cam.get(cam_id, {})
                for local_id, track in tracker.confirmed_tracks.items():
                    # A LOST motion track keeps its last real measurement. Use
                    # it to complete stop detection, but prefer a visible
                    # observation of the same global ID in another camera.
                    global_id = id_map.get(local_id) or manager.get_global_id(cam_id, local_id)
                    if global_id is None:
                        continue
                    candidate = {
                        "bbox": (track.x + crop[0], track.y + crop[1], track.w, track.h),
                        "appearance": getattr(track, "appearance", None),
                        "camera_id": cam_id,
                        "area": float(track.area),
                        "visible": track.consecutive_invisible_count == 0,
                    }
                    previous = global_active_tracks.get(global_id)
                    if (
                        previous is None
                        or (candidate["visible"] and not previous["visible"])
                        or (candidate["visible"] == previous["visible"] and candidate["area"] > previous["area"])
                    ):
                        global_active_tracks[global_id] = candidate
            tracking_timestamp_s = frame_idx / max(sim.fps, 1.0)
            parking_binder.update_tracks(
                global_active_tracks,
                frame_idx,
                tracking_timestamp_s,
            )

            # A pending parking-slot departure is no longer uncertain once its
            # global ID is observed anywhere in the four-camera system.
            active_global_ids = {
                global_id
                for cam_id, tracks in all_observable_tracks.items()
                for local_id in tracks
                if (global_id := global_ids_per_cam.get(cam_id, {}).get(local_id)) is not None
            }
            parking_binder.resolve_pending_global_ids(active_global_ids)

            # ── 2. Parking detection (tần suất thấp hơn) ──
            if now - last_parking_at >= parking_interval:
                for cam_id, detector in detectors.items():
                    cf = cam_frames[cam_id]
                    results = detector.detect(cf, apply_smoothing=True)
                    last_slot_results[cam_id] = results

                    crop = sim.cameras[cam_id]["crop"]
                    parking_binder.update_vision(
                        results,
                        frame_idx,
                        tracking_timestamp_s,
                        camera_id=cam_id,
                        coordinate_offset=(crop[0], crop[1]),
                    )
                last_parking_at = now

            # ── 3. Ghi JSON output ──
            if now - last_json_at >= json_interval:
                for cam_id in detectors:
                    cam = sim.cameras[cam_id]
                    tracker = trackers[cam_id]
                    binder = parking_binder
                    slot_results = last_slot_results.get(cam_id, [])
                    id_map = global_ids_per_cam.get(cam_id, {})
                    slot_states = binder.to_json(camera_id=cam_id)

                    # parking_status_camX.json
                    parking_output = {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "frame_index": frame_idx,
                        "camera_id": cam_id,
                        "camera_name": cam["name"],
                        "total": len(slot_states),
                        "free": sum(1 for state in slot_states.values() if not state["occupied"]),
                        "occupied": sum(1 for state in slot_states.values() if state["occupied"]),
                        "slots": slot_states,
                    }
                    _save_json_atomic(
                        parking_output,
                        sim.output_dir / f"parking_status_{cam_id}.json",
                    )

                    # vehicle_positions_camX.json
                    active_vehicles = {}
                    for tid, track in tracker.active_tracks.items():
                        global_tid = id_map.get(tid)
                        if global_tid is None:
                            continue
                        slot_id = binder.get_slot_for_vehicle(global_tid)
                        vehicle_entry = {
                            "track_id": global_tid,
                            "local_track_id": tid,
                            "status": track.status.value,
                            "position": {"x": track.cx, "y": track.cy},
                            "bbox": {"x": track.x, "y": track.y, "w": track.w, "h": track.h},
                            "area": int(track.area),
                            "age": track.age,
                            "parked_in_slot": slot_id,
                        }
                        current = active_vehicles.get(str(global_tid))
                        if current is None or vehicle_entry["area"] > current["area"]:
                            active_vehicles[str(global_tid)] = vehicle_entry

                    vehicle_output = {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "frame_index": frame_idx,
                        "camera_id": cam_id,
                        "camera_name": cam["name"],
                        "active_count": len(active_vehicles),
                        "parked_count": len(binder.get_all_parked_vehicle_ids(camera_id=cam_id)),
                        "active_vehicles": active_vehicles,
                        "parking_slots": slot_states,
                    }
                    _save_json_atomic(
                        vehicle_output,
                        sim.output_dir / f"vehicle_positions_{cam_id}.json",
                    )

                _save_json_atomic(
                    {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "frame_index": frame_idx,
                        "map_coordinate_space": {
                            "name": "source_video_pixel",
                            "width": sim.frame_w,
                            "height": sim.frame_h,
                            "description": "Common coordinate system used by all four virtual camera crops",
                        },
                        "parking_slots": parking_binder.to_json(),
                        "parking_events": parking_binder.events,
                        **manager.to_json(all_active_tracks),
                    },
                    sim.output_dir / "global_vehicle_registry.json",
                )

                last_json_at = now

                if args.verbose:
                    total_active = sum(len(t.active_tracks) for t in trackers.values())
                    all_slot_states = parking_binder.to_json()
                    total_parked = len(parking_binder.get_all_parked_vehicle_ids())
                    total_free = sum(1 for state in all_slot_states.values() if not state["occupied"])
                    total_occ = sum(1 for state in all_slot_states.values() if state["occupied"])
                    print(
                        f"  frame={frame_idx} "
                        f"vehicles={total_active} parked={total_parked} "
                        f"free={total_free} occupied={total_occ}"
                    )

            # ── 4. Display ──
            if not args.no_display:
                target_w, target_h = 400, 280
                views = []
                for cam_id in ["cam1", "cam2", "cam3", "cam4"]:
                    cf = cam_frames[cam_id]
                    # Vẽ tracking boxes
                    if cam_id in trackers:
                        id_map = global_ids_per_cam.get(cam_id, {})
                        # A same-camera motion echo may share one global ID;
                        # draw only its strongest box so the operator never
                        # sees one vehicle twice.
                        shown_tracks = {}
                        shown_global_ids = set()
                        for local_id, track in sorted(
                            trackers[cam_id].active_tracks.items(),
                            key=lambda item: item[1].area,
                            reverse=True,
                        ):
                            global_id = id_map.get(local_id, local_id)
                            if global_id in shown_global_ids:
                                continue
                            shown_global_ids.add(global_id)
                            shown_tracks[local_id] = track
                        cf = trackers[cam_id].draw_tracks(
                            cf,
                            tracks=shown_tracks,
                            id_overrides=id_map,
                        )
                    # Vẽ parking slots overlay
                    if cam_id in detectors and cam_id in last_slot_results:
                        cf = detectors[cam_id].draw_results(cf, last_slot_results[cam_id])
                    views.append(cv2.resize(cf, (target_w, target_h)))

                top_row = np.hstack([views[0], views[1]])
                bot_row = np.hstack([views[2], views[3]])
                grid = np.vstack([top_row, bot_row])

                overview = sim.draw_split_overlay(frame)
                overview_small = cv2.resize(overview, (400, 260))

                cv2.imshow("4 Cameras - Tracking + Parking", grid)
                cv2.imshow("Overview", overview_small)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                break
            if frame_period:
                remaining = frame_period - (time.perf_counter() - frame_started_at)
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Hoàn tất.")
        print(f"   Parking: parking_status_cam1..4.json")
        print(f"   Tracking: vehicle_positions_cam1..4.json")


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

def parse_args():
    root = PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Multi-camera parking simulator")
    parser.add_argument("--video", "-v", default=str(root / "data" / "carPark.mp4"))
    parser.add_argument("--slots", "-s", default=str(root / "config" / "parking_slots.json"),
                        help="File JSON chứa 69 ô đỗ (tọa độ video gốc)")
    parser.add_argument("--overlap", type=int, default=0,
                        help="Pixel overlap giữa 2 camera kề nhau (mặc định 0 để tránh ID trùng)")
    parser.add_argument("--output-dir", default=str(root / "runtime_output"),
                        help="Thư mục chứa output JSON")

    parser.add_argument("--preview", action="store_true",
                        help="Chỉ xem trước 4 camera (không chạy detection)")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--realtime", action="store_true", help="Phat video theo FPS goc")
    parser.add_argument("--playback-fps", type=float, default=0.0, help="Gioi han FPS phat (0 = nhanh nhat)")

    parser.add_argument("--parking-fps", type=float, default=2.0)
    parser.add_argument("--parking-smoothing", type=int, default=5)

    # Ensemble tuning (giá trị mặc định từ ensemble_test.py trackbar)
    parser.add_argument("--base-gamma", type=float, default=2.4, help="Base Gamma cho ensemble")
    parser.add_argument("--base-clahe", type=float, default=2.0, help="Base CLAHE clip cho ensemble")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE grid size")
    parser.add_argument("--ratio-thr", type=float, default=0.20, help="Ngưỡng pixel ratio trống/đầy")
    parser.add_argument("--edge-thr", type=float, default=0.25, help="Ngưỡng edge ratio pass 2")

    # Vehicle tracking
    parser.add_argument("--json-fps", type=float, default=5.0, help="Tần suất ghi JSON output")
    parser.add_argument("--min-visible-count", type=int, default=4, help="Detection liên tiếp để xác nhận xe")
    parser.add_argument("--lost-track-ttl", type=int, default=90, help="Số frame giữ track lost")
    parser.add_argument("--handoff-ttl", type=int, default=45, help="Số frame chờ xe xuất hiện ở camera kề")
    parser.add_argument("--handoff-match-distance", type=float, default=100.0, help="Khoảng cách tối đa khi ghép handoff (pixel toàn cục)")
    parser.add_argument("--handoff-appearance-threshold", type=float, default=0.45, help="Ngưỡng khác biệt HSV tối đa khi ghép handoff")
    parser.add_argument("--handoff-lookahead-frames", type=int, default=16, help="Số frame dự đoán trước khi xe chạm ranh camera")
    parser.add_argument("--handoff-prediction-radius", type=float, default=90.0, help="Sai số tối đa của vị trí handoff dự đoán (pixel)")
    parser.add_argument("--handoff-min-direction-cosine", type=float, default=0.25, help="Cosine hướng tối thiểu giữa xe nguồn/đích khi handoff")
    parser.add_argument("--motion-min-area", type=int, default=900, help="Diện tích foreground tối thiểu")
    parser.add_argument("--motion-max-distance", type=float, default=180.0, help="Gate Kalman (pixel)")
    parser.add_argument("--motion-min-displacement", type=float, default=12.0, help="Pixel tối thiểu để confirm")
    parser.add_argument("--slot-release-grace", type=int, default=90, help="Số frame giữ global vehicle_id sau khi ô trống")
    parser.add_argument("--slot-bind-confirmations", type=int, default=2, help="Số lần detector phải xác nhận trước khi gán xe vào ô")
    parser.add_argument("--slot-stop-seconds", type=float, default=1.0, help="Số giây xe phải đứng ổn định trước khi gán ô")
    parser.add_argument("--slot-exit-seconds", type=float, default=0.5, help="Số giây cùng Global ID ở ngoài ROI trước khi gỡ override")
    parser.add_argument("--slot-min-vehicle-overlap", type=float, default=0.35, help="Overlap bbox tối thiểu khi tâm xe nằm trong ROI")
    parser.add_argument("--slot-strong-vehicle-overlap", type=float, default=0.60, help="Overlap đủ mạnh khi tâm bbox nằm ngoài ROI")
    parser.add_argument("--slot-stationary-radius-ratio", type=float, default=0.06, help="Bán kính rung tối đa / đường chéo bbox")
    parser.add_argument("--slot-stationary-drift-ratio", type=float, default=0.10, help="Độ trôi đầu-cuối tối đa / đường chéo bbox")
    parser.add_argument("--slot-recovery-expand-ratio", type=float, default=0.15, help="Tỷ lệ mở rộng ROI để nhận lại ID xe rời ô")

    args = parser.parse_args()
    if args.overlap < 0:
        parser.error("--overlap khong duoc am")
    if args.playback_fps < 0:
        parser.error("--playback-fps khong duoc am")
    if args.handoff_lookahead_frames < 1:
        parser.error("--handoff-lookahead-frames phai >= 1")
    if args.handoff_prediction_radius <= 0:
        parser.error("--handoff-prediction-radius phai > 0")
    if not -1.0 <= args.handoff_min_direction_cosine <= 1.0:
        parser.error("--handoff-min-direction-cosine phai nam trong [-1, 1]")
    if args.slot_release_grace < 1:
        parser.error("--slot-release-grace phai >= 1")
    if args.slot_bind_confirmations < 1:
        parser.error("--slot-bind-confirmations phai >= 1")
    if args.slot_stop_seconds <= 0 or args.slot_exit_seconds <= 0:
        parser.error("--slot-stop-seconds va --slot-exit-seconds phai > 0")
    if not 0 <= args.slot_min_vehicle_overlap <= 1 or not 0 <= args.slot_strong_vehicle_overlap <= 1:
        parser.error("Cac nguong slot overlap phai nam trong [0, 1]")
    if args.slot_min_vehicle_overlap > args.slot_strong_vehicle_overlap:
        parser.error("--slot-min-vehicle-overlap khong duoc lon hon --slot-strong-vehicle-overlap")
    if args.slot_stationary_radius_ratio <= 0 or args.slot_stationary_drift_ratio <= 0:
        parser.error("Cac nguong stationary ratio phai > 0")
    if args.slot_recovery_expand_ratio < 0:
        parser.error("--slot-recovery-expand-ratio khong duoc am")
    return args


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  🅿️  Multi-Camera Parking Simulator")
    print("=" * 60)

    sim = CameraSimulator(
        source_path=args.video,
        slots_file=args.slots,
        overlap=args.overlap,
        output_dir=args.output_dir,
    )

    print(f"\n📐 Video: {sim.frame_w}x{sim.frame_h} | Cắt tại ({sim.mid_x}, {sim.mid_y})")
    for cam_id, cam in sim.cameras.items():
        crop = cam["crop"]
        print(f"  {cam_id} ({cam['name']}): crop=({crop[0]},{crop[1]})-({crop[2]},{crop[3]}) "
              f"→ {cam['width']}x{cam['height']} | {len(cam['slots'])} slots")

    if args.preview:
        preview_fps = sim.fps if args.realtime else args.playback_fps
        run_preview(sim, args.video, loop=args.loop, playback_fps=preview_fps)
    else:
        run_detection(sim, args.video, args)
