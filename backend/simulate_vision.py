import json
import time
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "vision" / "runtime_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Toạ độ Camera mô phỏng (tương ứng với bản đồ SVG của Frontend) ──
# Frontend camToMap: mapX = (camX / 1100) * 1200, mapY = (camY / 720) * 900
MAIN_ROAD_X = 914
ENTRY_Y = 750 # Đổi thành 750 (dưới viền) để xe trượt từ ngoài vào
EXIT_Y = -50 # Đổi thành -50 để xe trượt hẳn ra ngoài màn hình rồi mới biến mất

LANE_Y = {
    "A": 615,
    "B": 482,
    "C": 349,
    "D": 216,
    "E": 84
}

def get_spot_cam_x(number):
    """Tính tọa độ X trên camera dựa theo chỉ số ô (1-15)"""
    col_index = number - 1
    map_x = 92 + col_index * 56
    return int((map_x / 1200) * 1100)

class ParkingSimulator:
    def __init__(self):
        self.frame = 1
        self.slots = {}
        self.vehicles = {} # id -> data
    
    def emit_frame(self, events=None):
        if events is None: events = []
        
        # global_vehicle_registry.json
        reg_data = {
            "frame_index": self.frame,
            "parking_slots": self.slots,
            "global_vehicles": {
                str(v_id): {
                    "global_id": v_id,
                    "position": {"x": v["x"], "y": v["y"]},
                    "camera_ids": ["cam1"],
                    "parked_in_slot": v.get("parked_slot")
                }
                for v_id, v in self.vehicles.items()
            },
            "parking_events": events
        }
        with open(OUTPUT_DIR / "global_vehicle_registry.json", "w", encoding="utf-8") as f:
            json.dump(reg_data, f, ensure_ascii=False)
            
        # vehicle_positions_cam1.json (chỉ chứa xe ĐANG CHẠY, không phải xe đã tắt máy)
        cam_data = {
            "active_vehicles": {
                str(v_id): {"track_id": v_id, "position": {"x": v["x"], "y": v["y"]}}
                for v_id, v in self.vehicles.items() if not v.get("is_parked")
            }
        }
        with open(OUTPUT_DIR / "vehicle_positions_cam1.json", "w", encoding="utf-8") as f:
            json.dump(cam_data, f, ensure_ascii=False)
            
        self.frame += 1

    def move_vehicle(self, v_id, target_x, target_y, steps=10, sleep_time=0.1):
        """Di chuyển xe mượt mà tới tọa độ đích"""
        start_x = self.vehicles[v_id]["x"]
        start_y = self.vehicles[v_id]["y"]
        for i in range(1, steps + 1):
            self.vehicles[v_id]["x"] = int(start_x + (target_x - start_x) * (i / steps))
            self.vehicles[v_id]["y"] = int(start_y + (target_y - start_y) * (i / steps))
            self.emit_frame()
            time.sleep(sleep_time)

    def run_scenario(self):
        print("🚀 Khởi động script giả lập 5 xe (Mock Vision) - CHẠY LẶP LẠI LIÊN TỤC...")
        # Khởi tạo frame rỗng
        self.emit_frame()
        time.sleep(1)

        cycle_count = 1
        while True:
            print(f"\n==============================================")
            print(f"🔄 BẮT ĐẦU CHU KỲ MỚI (LẦN {cycle_count})")
            print(f"==============================================")

            # Định nghĩa 5 kịch bản xe (ID tăng dần theo chu kỳ để tạo xe mới hoàn toàn)
            base_id = cycle_count * 100
            scenarios = [
                {"id": base_id + 11, "spot": "A02", "zone": "A", "num": 2},
                {"id": base_id + 22, "spot": "B05", "zone": "B", "num": 5},
                {"id": base_id + 33, "spot": "C08", "zone": "C", "num": 8},
                {"id": base_id + 44, "spot": "D11", "zone": "D", "num": 11},
                {"id": base_id + 55, "spot": "E14", "zone": "E", "num": 14},
            ]

            print("--- 🚗 GIAI ĐOẠN 1: 5 XE LẦN LƯỢT ĐI VÀO BÃI ---")
            for sc in scenarios:
                vid = sc["id"]
                zone = sc["zone"]
                spot_id = sc["spot"]
                spot_x = get_spot_cam_x(sc["num"])
                lane_y = LANE_Y[zone]

                print(f"\n=> Xe #{vid} tiến vào cổng...")
                self.vehicles[vid] = {"x": MAIN_ROAD_X, "y": ENTRY_Y}
                self.emit_frame()
                time.sleep(2) # Đợi Frontend hiện QR

                print(f"=> Xe #{vid} chạy thẳng tới Khu {zone}...")
                self.move_vehicle(vid, MAIN_ROAD_X, lane_y, steps=15, sleep_time=0.08)

                print(f"=> Xe #{vid} rẽ trái vào làn Khu {zone} tới ô {spot_id}...")
                self.move_vehicle(vid, spot_x, lane_y, steps=20, sleep_time=0.08)

                print(f"=> Xe #{vid} lùi vào ô đỗ {spot_id}...")
                # Lùi lên một chút để mô phỏng vào chuồng
                self.move_vehicle(vid, spot_x, lane_y - 20, steps=5, sleep_time=0.1)
                
                # Khai báo đã đỗ
                self.vehicles[vid]["parked_slot"] = spot_id
                self.slots[spot_id] = {"occupied": True, "vehicle_id": vid, "vision_occupied": True, "tracking_occupied": True}
                self.emit_frame([{"type": "vehicle_stopped_in_slot", "global_vehicle_id": vid, "slot_id": spot_id}])
                time.sleep(1)

                print(f"=> Xe #{vid} tắt máy (Lost track).")
                self.vehicles[vid]["is_parked"] = True
                # Sau khi đỗ, AI thực tế sẽ mất track xe đó (vì xe nằm yên, tắt máy)
                # Frontend sẽ giữ nó lại nhờ parked_slot
                self.emit_frame()
                time.sleep(2)

            print("\n--- 🏁 BÃI ĐÃ NHẬN ĐỦ 5 XE. NGHỈ 5 GIÂY TRƯỚC KHI ĐI RA ---")
            time.sleep(5)

            print("\n--- 🚗 GIAI ĐOẠN 2: 5 XE LẦN LƯỢT RỜI BÃI ---")
            for sc in scenarios:
                vid = sc["id"]
                zone = sc["zone"]
                spot_id = sc["spot"]
                spot_x = get_spot_cam_x(sc["num"])
                lane_y = LANE_Y[zone]

                print(f"\n=> Xe #{vid} rời khỏi ô {spot_id}...")
                self.vehicles[vid]["is_parked"] = False
                self.vehicles[vid]["parked_slot"] = None
                self.slots[spot_id] = {"occupied": False, "vehicle_id": None}
                self.emit_frame([{"type": "vehicle_left_slot", "global_vehicle_id": vid, "slot_id": spot_id}])
                time.sleep(1)

                print(f"=> Xe #{vid} lùi ra làn Khu {zone}...")
                self.move_vehicle(vid, spot_x, lane_y, steps=5, sleep_time=0.1)

                print(f"=> Xe #{vid} chạy ra đường chính...")
                self.move_vehicle(vid, MAIN_ROAD_X, lane_y, steps=20, sleep_time=0.08)

                print(f"=> Xe #{vid} đi thẳng ra lối ra (Exit)...")
                self.move_vehicle(vid, MAIN_ROAD_X, EXIT_Y, steps=20, sleep_time=0.08)

                print(f"=> Xe #{vid} đã qua cổng ra.")
                self.emit_frame([{"type": "vehicle_exited", "global_vehicle_id": vid}])
                time.sleep(0.5)
                del self.vehicles[vid]
                self.emit_frame()
                time.sleep(1)

            print(f"\n🎉 HOÀN TẤT CHU KỲ {cycle_count}! Chuẩn bị vòng lặp mới...")
            cycle_count += 1
            time.sleep(3)

if __name__ == "__main__":
    sim = ParkingSimulator()
    sim.run_scenario()
