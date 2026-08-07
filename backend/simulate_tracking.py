"""
simulate_tracking.py - Bộ mô phỏng 5 xe di chuyển siêu mượt, chuẩn thời gian thực (Real-time Monotonic Path)

Khắc phục triệt để:
  1. Tọa độ di chuyển tuần tự đơn điệu theo thời gian thực (không bị giật/nhảy lùi thứ tự).
  2. Lọc trail đút nút cách đều (downsampled trail) giúp đường vẽ SVG cực kỳ mịn màng.
  3. Đảm bảo đúng 100% lọc xe theo Session ID: Mỗi session chỉ hiển thị duy nhất xe của nó.
"""

import json
import time
import math
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
try:
    from session_manager import create_session, update_session_state
except ImportError:
    create_session = None
    update_session_state = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

OUTPUT_JSON = BASE_DIR.parent / "frontend" / "public" / "vehicle_positions.json"

# Cấu hình 5 xe với tuyến đường chính xác
VEHICLE_CONFIGS = [
    {
        "track_id": 1,
        "session_id": "PARK_SIM_001",
        "target_spot": "D08",
        "start_delay": 0,
        "inbound": [(997, 850), (997, 600), (997, 286), (850, 286), (572, 286), (572, 230)],
        "outbound": [(572, 230), (572, 286), (850, 286), (997, 286), (997, 35)]
    },
    {
        "track_id": 2,
        "session_id": "PARK_SIM_002",
        "target_spot": "B12",
        "start_delay": 12,
        "inbound": [(997, 850), (997, 750), (997, 618), (850, 618), (752, 618), (752, 660)],
        "outbound": [(752, 660), (752, 618), (850, 618), (997, 618), (997, 35)]
    },
    {
        "track_id": 3,
        "session_id": "PARK_SIM_003",
        "target_spot": "C04",
        "start_delay": 24,
        "inbound": [(997, 850), (997, 650), (997, 452), (600, 452), (320, 452), (320, 410)],
        "outbound": [(320, 410), (320, 452), (600, 452), (997, 452), (997, 35)]
    },
    {
        "track_id": 4,
        "session_id": "PARK_SIM_004",
        "target_spot": "A05",
        "start_delay": 36,
        "inbound": [(997, 850), (997, 780), (600, 780), (380, 780), (380, 740)],
        "outbound": [(380, 740), (380, 780), (600, 780), (997, 780), (997, 35)]
    },
    {
        "track_id": 5,
        "session_id": "PARK_SIM_005",
        "target_spot": "E02",
        "start_delay": 48,
        "inbound": [(997, 850), (997, 450), (997, 120), (500, 120), (200, 120), (200, 80)],
        "outbound": [(200, 80), (200, 120), (500, 120), (997, 120), (997, 35)]
    }
]

PARK_DURATION = 30.0  # Đỗ xe 30s
INBOUND_DURATION = 16.0 # 16 giây di chuyển từ cổng vào ô đỗ (tốc độ xe bò tự nhiên)
OUTBOUND_DURATION = 12.0 # 12 giây di chuyển từ ô đỗ ra cổng

def get_path_position(waypoints, progress: float):
    """Tính toán tọa độ liên tục (monotonically progressive) theo tỷ lệ progress [0.0 ... 1.0]"""
    progress = max(0.0, min(1.0, progress))
    if progress == 0.0:
        return waypoints[0]
    if progress == 1.0:
        return waypoints[-1]

    # Tính tổng chiều dài tuyến đường
    total_len = 0.0
    lens = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        l = math.hypot(dx, dy)
        lens.append(l)
        total_len += l

    target_dist = progress * total_len
    accum_dist = 0.0

    for i, l in enumerate(lens):
        if accum_dist + l >= target_dist:
            seg_progress = (target_dist - accum_dist) / l if l > 0 else 0
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            x = p1[0] + (p2[0] - p1[0]) * seg_progress
            y = p1[1] + (p2[1] - p1[1]) * seg_progress
            return (round(x, 1), round(y, 1))
        accum_dist += l

    return waypoints[-1]

def save_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for _ in range(3):
            try:
                temp_path.replace(path)
                return
            except (PermissionError, FileNotFoundError):
                time.sleep(0.02)
    except Exception:
        pass

def main():
    print("=" * 70)
    print(" === BO MOP HONG 5 XE DI CHUYEN SIEU MUOT (MONOTONIC TIME-BASED) ===")
    print("=" * 70)
    print(f"Output JSON: {OUTPUT_JSON.resolve()}")

    sim_start_time = time.time()
    vehicle_states = {}

    for cfg in VEHICLE_CONFIGS:
        t_id = cfg["track_id"]
        vehicle_states[t_id] = {
            "config": cfg,
            "created": False,
            "trail": [],
            "last_pos": None,
        }

    try:
        while True:
            now = time.time()
            elapsed_total = now - sim_start_time
            active_vehicles = {}
            all_finished = True

            for t_id, state in vehicle_states.items():
                cfg = state["config"]
                t_delay = cfg["start_delay"]

                if elapsed_total < t_delay:
                    all_finished = False
                    continue

                t_rel = elapsed_total - t_delay

                # Lần đầu xuất hiện -> Tạo Session
                if not state["created"]:
                    state["created"] = True
                    print(f"🚥 Xe #{t_id} CHINH THUC VAO CONG -> Tao Session: {cfg['session_id']}")
                    if create_session:
                        create_session(cfg["session_id"], cfg["target_spot"], t_id)
                        update_session_state(cfg["session_id"], "NAVIGATING_TO_SPOT")

                # Phase 1: Inbound (0s -> INBOUND_DURATION)
                if t_rel < INBOUND_DURATION:
                    all_finished = False
                    progress = t_rel / INBOUND_DURATION
                    pos = get_path_position(cfg["inbound"], progress)
                    status = "moving"

                # Phase 2: Parked (INBOUND_DURATION -> INBOUND_DURATION + PARK_DURATION)
                elif t_rel < INBOUND_DURATION + PARK_DURATION:
                    all_finished = False
                    pos = cfg["inbound"][-1]
                    status = "parked"
                    # Cập nhật trạng thái đỗ
                    if update_session_state and t_rel - INBOUND_DURATION < 0.2:
                        update_session_state(cfg["session_id"], "PARKED", parked_spot_id=cfg["target_spot"])

                # Phase 3: Outbound (INBOUND_DURATION + PARK_DURATION -> INBOUND + PARK + OUTBOUND)
                elif t_rel < INBOUND_DURATION + PARK_DURATION + OUTBOUND_DURATION:
                    all_finished = False
                    progress = (t_rel - INBOUND_DURATION - PARK_DURATION) / OUTBOUND_DURATION
                    pos = get_path_position(cfg["outbound"], progress)
                    status = "exiting"
                    if update_session_state and progress < 0.05:
                        update_session_state(cfg["session_id"], "EXIT_NAVIGATION")

                else:
                    # Hoàn thành phiên đỗ
                    if update_session_state and state.get("active_last", True):
                        update_session_state(cfg["session_id"], "CLOSED")
                        state["active_last"] = False
                    continue

                # Lưu vết đường đi (trail) cách nhau ít nhất 12px để đường rẽ đẹp mịn
                last_p = state["last_pos"]
                if last_p is None or math.hypot(pos[0] - last_p[0], pos[1] - last_p[1]) >= 12.0:
                    state["trail"].append({"x": pos[0], "y": pos[1]})
                    state["last_pos"] = pos
                    if len(state["trail"]) > 20:
                        state["trail"].pop(0)

                active_vehicles[str(t_id)] = {
                    "track_id": t_id,
                    "status": status,
                    "position": {"x": pos[0], "y": pos[1]},
                    "trail": state["trail"] if status != "parked" else [],
                    "vehicle_class": "car",
                    "confidence": 0.98,
                }

            json_data = {
                "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "source": "simulation",
                "frame_size": {"width": 1200, "height": 900},
                "active_count": len(active_vehicles),
                "active_vehicles": active_vehicles,
            }

            save_json_atomic(json_data, OUTPUT_JSON)

            if all_finished:
                print("\n==> Tat ca 5 xe da di ra khoi bai! Reset chu ky sau 5s...")
                time.sleep(5.0)
                sim_start_time = time.time()
                for t_id, state in vehicle_states.items():
                    state["created"] = False
                    state["trail"] = []
                    state["last_pos"] = None
                    state["active_last"] = True

            time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nDa dung mo phong.")

if __name__ == "__main__":
    main()
