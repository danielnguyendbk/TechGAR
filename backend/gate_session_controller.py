"""
gate_session_controller.py - Kẻ Gác Cổng (Gate Session Watcher) & API Server

Nhiệm vụ:
1. Đứng nhìn (watch) file Tracking Feed (ví dụ: vehicle_positions_sample.json).
2. Khi thấy track_id mới xuất hiện gần cổng -> Tạo Session (WAITING_FOR_SCAN, chưa có target).
3. Tích hợp HTTP API Server trên cổng 8000 để Frontend gửi Yêu cầu:
   - POST /api/session/claim    : User quét QR (WAITING_FOR_SCAN -> SELECTING_SPOT)
   - POST /api/session/select   : User chọn ô đỗ (SELECTING_SPOT -> NAVIGATING_TO_SPOT)
   - POST /api/session/exit     : User bấm lấy xe ra (PARKED -> EXIT_NAVIGATION)
   - GET  /api/sessions         : Đọc tất cả session
4. [Sample mode] Tự động cập nhật parking_status_sample.json khi xe đỗ / rời ô.
"""

import json
import time
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
try:
    from session_manager import (
        create_session, claim_session, select_spot, load_sessions, save_sessions,
        set_parked, set_exit_navigation, close_session, now_iso
    )
except ImportError as e:
    print(f"Khong tim thay session_manager.py! Loi: {e}")
    sys.exit(1)

# ── Đường dẫn file ──
PARKING_STATUS_SAMPLE = BASE_DIR.parent / "frontend" / "public" / "parking_status_sample.json"
SESSIONS_FILE = BASE_DIR.parent / "frontend" / "public" / "navigation_sessions.json"


def load_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json_atomic(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        for _ in range(3):
            try:
                temp_path.replace(path)
                return
            except Exception:
                time.sleep(0.02)
    except Exception:
        pass


def update_sample_parking_status(spot_id: str, status: str):
    """
    Cập nhật parking_status_sample.json khi simulator xe đỗ/rời ô.
    Giúp đồng bộ occupancy trong sample mode.
    """
    data = load_json_safe(PARKING_STATUS_SAMPLE)
    if "slots" not in data:
        data["slots"] = {}
    data["slots"][spot_id] = {
        "status": status,
        "confidence": 0.99,
    }
    data["timestamp"] = now_iso()
    save_json_atomic(data, PARKING_STATUS_SAMPLE)


# ──────────────────────────────────────────────
#  HTTP API SERVER (Cổng 8000)
# ──────────────────────────────────────────────
class SessionAPIRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/sessions") or self.path.startswith("/navigation_sessions"):
            sessions = load_sessions()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(sessions, ensure_ascii=False, indent=2).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            payload = json.loads(body_bytes.decode('utf-8'))
        except Exception:
            payload = {}

        sid = payload.get("sessionId")
        spot_id = payload.get("spotId")

        print(f"[API HTTP] {self.path} - Payload: {payload}")

        if self.path == "/api/session/claim":
            if sid:
                claim_session(sid)
                self._respond_json({"ok": True, "sessionId": sid, "state": "SELECTING_SPOT"})
            else:
                self._respond_json({"error": "Missing sessionId"}, status=400)

        elif self.path == "/api/session/select":
            if sid:
                select_spot(sid, spot_id)
                new_state = "NAVIGATING_TO_SPOT" if spot_id else "SELECTING_SPOT"
                self._respond_json({"ok": True, "sessionId": sid, "spotId": spot_id, "state": new_state})
            else:
                self._respond_json({"error": "Missing sessionId"}, status=400)

        elif self.path == "/api/session/exit":
            if sid:
                sessions = load_sessions()
                session = sessions.get(sid, {})
                parked_spot = session.get("parkedSpotId")
                set_exit_navigation(sid)
                if parked_spot:
                    update_sample_parking_status(parked_spot, "empty")
                self._respond_json({"ok": True, "sessionId": sid, "state": "EXIT_NAVIGATION"})
            else:
                self._respond_json({"error": "Missing sessionId"}, status=400)

        else:
            self._respond_json({"error": "Route not found"}, status=404)

    def do_PUT(self):
        # Hỗ trợ ghi đè file navigation_sessions trực tiếp nếu frontend PUT
        content_length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            data = json.loads(body_bytes.decode('utf-8'))
            save_sessions(data)
            self._respond_json({"ok": True})
        except Exception as e:
            self._respond_json({"error": str(e)}, status=500)

    def _respond_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        # Ẩn bớt log http request định kỳ để terminal đỡ rác
        if "GET /api/sessions" in format % args:
            return
        super().log_message(format, *args)


def start_api_server(port: int = 8000):
    server = HTTPServer(("0.0.0.0", port), SessionAPIRequestHandler)
    print(f"[API SERVER] Listening on http://0.0.0.0:{port}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="vehicle_positions_sample.json",
                        help="File tracking feed de watch")
    parser.add_argument("--port", type=int, default=8000, help="Port cho HTTP API Server")
    args = parser.parse_args()

    # Khởi chạy HTTP API Server ở background thread
    api_thread = threading.Thread(target=start_api_server, args=(args.port,), daemon=True)
    api_thread.start()
    
    feed_path = BASE_DIR.parent / "frontend" / "public" / args.source
    print("=" * 60)
    print(" [GATE CONTROLLER] GATE SESSION CONTROLLER & API DANG CHAY")
    print(f" Theo doi file: {args.source}")
    print(f" HTTP API Endpoint: http://localhost:{args.port}/api/session")
    print("=" * 60)
    
    # State tracking: track_id -> session_id
    active_tracks = {}
    
    try:
        while True:
            if not feed_path.exists():
                time.sleep(0.5)
                continue
                
            try:
                with open(feed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                time.sleep(0.1)
                continue
                
            vehicles = data.get("active_vehicles", {})
            current_track_ids = set(int(k) for k in vehicles.keys())
            
            # ── Xử lý xe mới vào cổng ──
            for t_id_str, v_data in vehicles.items():
                t_id = int(t_id_str)
                pos = v_data.get("position", {"x": 0, "y": 0})
                
                # Cổng vào khoảng (997, 850) – kiểm tra y > 700
                if t_id not in active_tracks:
                    if pos["y"] > 700:
                        # Tạo session KHÔNG có target – chỉ track_id
                        session_id = create_session(track_id=t_id)
                        print(f"[GATE] Phat hien xe moi #{t_id} o cong! "
                              f"Tao Session: {session_id} (WAITING_FOR_SCAN)")
                        active_tracks[t_id] = session_id

            # ── [SAMPLE MODE] Xử lý cập nhật trạng thái từ simulated status ──
            for t_id_str, v_data in vehicles.items():
                t_id = int(t_id_str)
                if t_id not in active_tracks:
                    continue
                
                session_id = active_tracks[t_id]
                sessions = load_sessions()
                session = sessions.get(session_id, {})
                s_state = session.get("state")
                
                # Sample source báo xe "parked" -> lấy ĐÚNG ô thực tế từ camera/simulator
                if v_data.get("status") == "parked" and s_state != "PARKED" and s_state != "CLOSED":
                    real_parked_spot = v_data.get("parked_spot_id")
                    if real_parked_spot:
                        print(f"[PARK] Xe #{t_id} da do THUC TE tai o {real_parked_spot} -> PARKED")
                        set_parked(session_id, real_parked_spot)
                        update_sample_parking_status(real_parked_spot, "occupied")
                
                # Sample source báo xe "exiting" -> giải phóng ô đỗ thực tế
                elif v_data.get("status") == "exiting" and s_state in ("PARKED", "EXIT_NAVIGATION"):
                    real_parked_spot = session.get("parkedSpotId") or v_data.get("parked_spot_id")
                    print(f"[EXIT] Xe #{t_id} bat dau roi o do {real_parked_spot} -> EXIT_NAVIGATION")
                    set_exit_navigation(session_id, t_id)
                    if real_parked_spot:
                        update_sample_parking_status(real_parked_spot, "empty")
                    
            # ── Xử lý xe đã đi khỏi bãi (mất track) ──
            lost_tracks = list(set(active_tracks.keys()) - current_track_ids)
            for t_id in lost_tracks:
                session_id = active_tracks[t_id]
                print(f"[CLOSE] Xe #{t_id} da roi khoi bai -> Dong Session {session_id}")
                close_session(session_id)
                del active_tracks[t_id]
                
            time.sleep(0.3)
            
    except KeyboardInterrupt:
        print("Stopped Gate Controller.")

if __name__ == "__main__":
    main()
