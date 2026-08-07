"""
session_manager.py  –  Quản lý vòng đời Phiên Đỗ Xe (Parking Session)

Vòng đời trạng thái:
  WAITING_FOR_SCAN    →  xe ở cổng, chờ tài xế quét QR
  SELECTING_SPOT      →  tài xế đã quét QR, đang chọn ô đỗ
  NAVIGATING_TO_SPOT  →  xe đang di chuyển vào ô đỗ
  PARKED              →  xe đã đỗ, track mất dấu nhưng session vẫn sống
  EXIT_NAVIGATION     →  xe đang di chuyển ra khỏi bãi
  CLOSED              →  xe đã ra khỏi bãi, session kết thúc

Session ID: sử dụng số đơn giản (1, 2, 3, ...) trùng với track_id
  → URL: /?session=1  /?session=2  ...

Identity tracking:
  vehicleTrackId  –  track_id gốc (cố định, không mất khi đỗ)
  activeTrackId   –  track_id đang active trên camera (null khi đỗ, có thể khác nếu Re-ID)

Cách chạy (ở thư mục gốc TechGAR, môi trường ảo .venv):
  python backend/session_manager.py --create --track 1
  python backend/session_manager.py --list
  python backend/session_manager.py --claim 1
  python backend/session_manager.py --select-spot 1 --spot D08
  python backend/session_manager.py --set-parked 1 --spot D08
  python backend/session_manager.py --set-exit 1
  python backend/session_manager.py --close 1
  python backend/session_manager.py --watch   # chạy nền, tự động cập nhật trạng thái
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
#  ĐƯỜNG DẪN FILE
# ──────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPT_DIR.parent

# File session được đặt trong frontend/public để React đọc trực tiếp
SESSIONS_FILE   = ROOT_DIR / "frontend" / "public" / "navigation_sessions.json"
# File vị trí xe từ tracker của An
POSITIONS_FILE  = ROOT_DIR / "backend" / "detect_car_update" / "vehicle_positions.json"
# File trạng thái ô đỗ xe
STATUS_FILE     = ROOT_DIR / "frontend" / "public" / "parking_status.json"

# ──────────────────────────────────────────────
#  CẤU HÌNH
# ──────────────────────────────────────────────
WATCH_INTERVAL      = 2.0    # giây - tần suất kiểm tra trạng thái
PARKED_TTL_FRAMES   = 60     # số frame không có chuyển động gần ô đỗ → chuyển PARKED
REACTIVATION_RADIUS = 80     # pixel - vùng tìm track mới khi xe rời ô đỗ


# ──────────────────────────────────────────────
#  TIỆN ÍCH
# ──────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_write(path: Path, data: dict) -> None:
    """Ghi file JSON an toàn (atomic) – tránh React đọc file đang ghi dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for _ in range(5):
            try:
                tmp.replace(path)
                return
            except Exception:
                time.sleep(0.05)
    except Exception:
        pass


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# ──────────────────────────────────────────────
#  QUẢN LÝ SESSION
# ──────────────────────────────────────────────
def load_sessions() -> dict:
    return load_json(SESSIONS_FILE)


def save_sessions(sessions: dict) -> None:
    atomic_write(SESSIONS_FILE, sessions)


def create_session(track_id: int) -> str:
    """
    Tạo phiên mới khi xe vào cổng. Session ID = track_id (đơn giản).
    State bắt đầu: WAITING_FOR_SCAN (chưa có target, chưa claimed).
    Trả về session_id (string).
    """
    sessions = load_sessions()
    sid = str(track_id)

    # Nếu session cũ của track này chưa CLOSED, đóng nó trước
    if sid in sessions and sessions[sid].get("state") != "CLOSED":
        sessions[sid]["state"] = "CLOSED"
        sessions[sid]["closedAt"] = now_iso()

    sessions[sid] = {
        "sessionId":        sid,
        "state":            "WAITING_FOR_SCAN",
        "targetSpotId":     None,
        "parkedSpotId":     None,
        "vehicleTrackId":   track_id,    # Identity cố định
        "activeTrackId":    track_id,    # Track đang active trên camera
        "claimed":          False,
        "lastKnownPosition": None,
        "createdAt":        now_iso(),
        "claimedAt":        None,
        "spotSelectedAt":   None,
        "parkedAt":         None,
        "exitStartedAt":    None,
        "closedAt":         None,
    }
    save_sessions(sessions)
    print(f"[OK] Tao session: {sid}  (track #{track_id})  state=WAITING_FOR_SCAN")
    return sid


def claim_session(session_id: str) -> None:
    """User đã quét QR → claim session. Chuyển WAITING_FOR_SCAN → SELECTING_SPOT."""
    sessions = load_sessions()
    if session_id not in sessions:
        print(f"Khong tim thay session: {session_id}")
        return
    s = sessions[session_id]
    if s["state"] != "WAITING_FOR_SCAN":
        print(f"Session {session_id} khong o trang thai WAITING_FOR_SCAN (hien: {s['state']})")
        return
    s["state"]     = "SELECTING_SPOT"
    s["claimed"]   = True
    s["claimedAt"] = now_iso()
    save_sessions(sessions)
    print(f"[CLAIMED] {session_id}  ->  SELECTING_SPOT")


def select_spot(session_id: str, spot_id: Optional[str]) -> None:
    """User chọn ô đỗ (hoặc hủy chọn nếu spot_id is None)."""
    sessions = load_sessions()
    if session_id not in sessions:
        print(f"Khong tim thay session: {session_id}")
        return
    s = sessions[session_id]
    if not spot_id:
        s["state"]        = "SELECTING_SPOT"
        s["targetSpotId"] = None
        print(f"[UNSELECT] {session_id}  ->  SELECTING_SPOT (Da huy o do)")
    else:
        s["state"]          = "NAVIGATING_TO_SPOT"
        s["targetSpotId"]   = spot_id
        s["claimed"]        = True
        s["spotSelectedAt"] = now_iso()
        if s.get("claimedAt") is None:
            s["claimedAt"]  = now_iso()
        print(f"[SELECT] {session_id}  ->  NAVIGATING_TO_SPOT  target={spot_id}")
    save_sessions(sessions)


def set_parked(session_id: str, parked_spot: str) -> None:
    """
    Xe đã đỗ vào ô. Chuyển trạng thái PARKED và XÓA targetSpotId.
    Giữ vehicleTrackId, chỉ xóa activeTrackId.
    """
    sessions = load_sessions()
    if session_id not in sessions:
        print(f"Khong tim thay session: {session_id}")
        return
    s = sessions[session_id]
    s["state"]          = "PARKED"
    s["parkedSpotId"]   = parked_spot
    s["targetSpotId"]   = None       # Xóa ô chọn mục tiêu cũ
    s["activeTrackId"]  = None       # Track mất dấu khi đỗ
    s["parkedAt"]       = now_iso()
    save_sessions(sessions)
    print(f"[PARKED] {session_id}  ->  PARKED tai {parked_spot}")


def set_exit_navigation(session_id: str, new_track_id: Optional[int] = None) -> None:
    """Người dùng bấm 'Lấy xe ra'. Chuyển EXIT_NAVIGATION, XÓA targetSpotId cũ."""
    sessions = load_sessions()
    if session_id not in sessions:
        print(f"Khong tim thay session: {session_id}")
        return
    s = sessions[session_id]
    s["state"]          = "EXIT_NAVIGATION"
    s["targetSpotId"]   = None       # Xóa targetSpotId cũ để chỉ dẫn đường lối ra
    s["activeTrackId"]  = new_track_id
    s["exitStartedAt"]  = now_iso()
    save_sessions(sessions)
    print(f"[EXIT] {session_id}  ->  EXIT_NAVIGATION  track={new_track_id}")


def close_session(session_id: str) -> None:
    """Xe đã ra khỏi bãi. Đóng session."""
    sessions = load_sessions()
    if session_id not in sessions:
        print(f"Khong tim thay session: {session_id}")
        return
    s = sessions[session_id]
    s["state"]         = "CLOSED"
    s["closedAt"]      = now_iso()
    s["activeTrackId"] = None
    save_sessions(sessions)
    print(f"[CLOSED] {session_id}  ->  CLOSED")


def update_track_position(session_id: str, track_id: int, position: dict) -> None:
    """Cập nhật vị trí cuối biết của xe trong session (last_known_position)."""
    sessions = load_sessions()
    if session_id not in sessions:
        return
    sessions[session_id]["activeTrackId"]      = track_id
    sessions[session_id]["lastKnownPosition"]  = position
    save_sessions(sessions)


def list_sessions() -> None:
    sessions = load_sessions()
    if not sessions:
        print("Chua co session nao.")
        return
    print(f"{'SESSION':<10} {'STATE':<22} {'TARGET':<8} {'PARKED':<8} {'V-TRACK':<8} {'A-TRACK':<8} {'CLAIMED'}")
    print("-" * 85)
    for sid, s in sessions.items():
        target  = s.get('targetSpotId')  or '-'
        parked  = s.get('parkedSpotId')  or '-'
        v_track = s.get('vehicleTrackId')
        a_track = s.get('activeTrackId')
        claimed = 'Yes' if s.get('claimed') else 'No'
        v_s = str(v_track) if v_track is not None else '-'
        a_s = str(a_track) if a_track is not None else '-'
        print(f"{sid:<10} {s['state']:<22} {target:<8} {parked:<8} {v_s:<8} {a_s:<8} {claimed}")


# ──────────────────────────────────────────────
#  CHẾ ĐỘ WATCH – Tự động đồng bộ track_id vào session
# ──────────────────────────────────────────────
def _find_nearest_track(position: dict, vehicles: dict, radius: float) -> Optional[int]:
    """Tìm track_id gần nhất với tọa độ đã cho, trong bán kính radius pixel."""
    best_id, best_dist = None, float("inf")
    for tid_str, v in vehicles.items():
        pos = v.get("position", {})
        dx = pos.get("x", 0) - position.get("x", 0)
        dy = pos.get("y", 0) - position.get("y", 0)
        dist = (dx**2 + dy**2) ** 0.5
        if dist < radius and dist < best_dist:
            best_dist = dist
            best_id = int(tid_str)
    return best_id


def watch_loop() -> None:
    """
    Vòng lặp nền: đọc vehicle_positions.json và parking_status.json mỗi 2 giây,
    tự động cập nhật vị trí xe và chuyển trạng thái session NAVIGATING → PARKED
    khi ô đỗ mục tiêu chuyển sang 'occupied'.
    """
    print(f"[WATCH] Session Watcher dang chay... (Ctrl+C de dung)")
    print(f"  Doc tu:  {POSITIONS_FILE}")
    print(f"  Ghi vao: {SESSIONS_FILE}")

    while True:
        try:
            sessions  = load_sessions()
            vehicles  = load_json(POSITIONS_FILE).get("active_vehicles", {})
            park_status = load_json(STATUS_FILE).get("slots", {})
            changed   = False

            for sid, s in sessions.items():
                if s["state"] in ("CLOSED",):
                    continue

                track_id = s.get("activeTrackId")

                # --- Cập nhật last_known_position nếu track đang active ---
                if track_id is not None and str(track_id) in vehicles:
                    pos = vehicles[str(track_id)].get("position", {})
                    s["lastKnownPosition"] = {"x": pos.get("x"), "y": pos.get("y")}
                    changed = True

                # --- Tự động chuyển NAVIGATING_TO_SPOT → PARKED ---
                # Điều kiện: ô mục tiêu chuyển sang 'occupied' và track mất dấu
                if s["state"] == "NAVIGATING_TO_SPOT":
                    target = s.get("targetSpotId", "")
                    slot_status = park_status.get(target, {}).get("status", "")
                    if slot_status == "occupied":
                        # Xác nhận bằng cách kiểm tra track có gần ô không
                        last_pos = s.get("lastKnownPosition") or {}
                        print(f"  [AUTO-PARKED] {sid}: o {target} occupied -> chuyen PARKED")
                        s["state"]          = "PARKED"
                        s["parkedSpotId"]   = target
                        s["activeTrackId"]  = None  # Track mất dấu
                        # vehicleTrackId giữ nguyên
                        s["parkedAt"]       = now_iso()
                        changed = True

                # --- Tự động gắn track mới khi EXIT_NAVIGATION ---
                # Điều kiện: activeTrackId chưa có, ô đỗ vừa chuyển empty
                if s["state"] == "EXIT_NAVIGATION" and s.get("activeTrackId") is None:
                    parked_spot = s.get("parkedSpotId", "")
                    slot_status = park_status.get(parked_spot, {}).get("status", "")
                    if slot_status == "empty" and vehicles:
                        # Tìm track mới xuất hiện gần ô đỗ cũ
                        last_pos = s.get("lastKnownPosition") or {"x": 0, "y": 0}
                        candidate = _find_nearest_track(last_pos, vehicles, REACTIVATION_RADIUS)
                        if candidate is not None:
                            print(f"  [RELINK] {sid}: tai lien ket voi track #{candidate}")
                            s["activeTrackId"] = candidate
                            changed = True

            if changed:
                save_sessions(sessions)

        except Exception as e:
            print(f"Watcher loi: {e}")

        time.sleep(WATCH_INTERVAL)


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TechGAR Parking Session Manager")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--create",       action="store_true",   help="Tao session moi")
    g.add_argument("--list",         action="store_true",   help="Liet ke tat ca session")
    g.add_argument("--claim",        metavar="SESSION_ID",  help="Claim session (user quet QR)")
    g.add_argument("--select-spot",  metavar="SESSION_ID",  help="Chon o do cho session")
    g.add_argument("--set-parked",   metavar="SESSION_ID",  help="Chuyen session sang PARKED")
    g.add_argument("--set-exit",     metavar="SESSION_ID",  help="Chuyen session sang EXIT_NAVIGATION")
    g.add_argument("--close",        metavar="SESSION_ID",  help="Dong session")
    g.add_argument("--watch",        action="store_true",   help="Chay vong lap tu dong cap nhat")
    p.add_argument("--spot",   default="P001", help="O muc tieu khi --select-spot hoac --set-parked")
    p.add_argument("--track",  type=int, default=1, help="Track ID kem theo khi --create")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.create:
        create_session(args.track)
    elif args.list:
        list_sessions()
    elif args.claim:
        claim_session(args.claim)
    elif args.select_spot:
        select_spot(args.select_spot, args.spot)
    elif args.set_parked:
        set_parked(args.set_parked, args.spot)
    elif args.set_exit:
        set_exit_navigation(args.set_exit, args.track)
    elif args.close:
        close_session(args.close)
    elif args.watch:
        watch_loop()
