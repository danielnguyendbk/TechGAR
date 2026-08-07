"""
draw_roi_lines.py — Tool vẽ ROI lines (vạch ngã rẽ) trên video/ảnh.

Cách dùng:
  python draw_roi_lines.py --video ../dataset/carPark.mp4
  python draw_roi_lines.py --image frame.jpg
  python draw_roi_lines.py --video ../dataset/carPark.mp4 --output my_roi.json

Thao tác:
  - Click chuột TRÁI: đặt điểm (2 click = 1 đường)
  - Nhấn U: Undo (xóa đường vừa vẽ)
  - Nhấn S: Save vào file JSON
  - Nhấn Q/ESC: Thoát
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class ROIDrawer:
    def __init__(self, frame: np.ndarray, output_path: str = "roi_lines.json"):
        self.frame = frame.copy()
        self.original = frame.copy()
        self.output_path = output_path
        self.lines = []           # Danh sách đường đã vẽ: [{"p1": (x,y), "p2": (x,y)}]
        self.current_point = None  # Điểm đầu tiên của đường đang vẽ
        self.line_count = 0

        # Load existing lines if file exists
        self._load_existing()

    def _load_existing(self):
        path = Path(self.output_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("lines", []):
                self.lines.append({
                    "id": item["id"],
                    "name": item.get("name", item["id"]),
                    "p1": tuple(item["p1"]),
                    "p2": tuple(item["p2"]),
                })
            self.line_count = len(self.lines)
            print(f"[ROI] Loaded {self.line_count} đường từ {self.output_path}")

    def _redraw(self):
        self.frame = self.original.copy()

        # Vẽ tất cả đường đã hoàn thành
        for line in self.lines:
            cv2.line(self.frame, line["p1"], line["p2"], (255, 0, 255), 2)
            mid_x = (line["p1"][0] + line["p2"][0]) // 2
            mid_y = (line["p1"][1] + line["p2"][1]) // 2
            cv2.putText(self.frame, line["name"], (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Vẽ điểm đang chờ
        if self.current_point is not None:
            cv2.circle(self.frame, self.current_point, 5, (0, 255, 0), -1)

        # Info
        cv2.putText(self.frame,
                    f"ROI Lines: {len(self.lines)} | Click 2 diem = 1 duong | S=Save U=Undo Q=Quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point is None:
                # Điểm đầu
                self.current_point = (x, y)
                print(f"  Điểm 1: ({x}, {y}) — click tiếp điểm 2")
            else:
                # Điểm cuối → hoàn thành 1 đường
                self.line_count += 1
                line_id = f"junction_{self.line_count}"
                self.lines.append({
                    "id": line_id,
                    "name": line_id,
                    "p1": self.current_point,
                    "p2": (x, y),
                })
                print(f"  ✅ Đường {line_id}: {self.current_point} → ({x}, {y})")
                self.current_point = None

            self._redraw()

    def save(self):
        data = {
            "lines": [
                {
                    "id": line["id"],
                    "name": line["name"],
                    "p1": list(line["p1"]),
                    "p2": list(line["p2"]),
                }
                for line in self.lines
            ]
        }
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Đã lưu {len(self.lines)} đường vào {self.output_path}")

    def run(self):
        win_name = "Draw ROI Lines"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self._mouse_callback)

        self._redraw()

        print("─" * 50)
        print("Click TRÁI: đặt 2 điểm = 1 đường ngã rẽ")
        print("U: Undo | S: Save | Q/ESC: Thoát")
        print("─" * 50)

        while True:
            cv2.imshow(win_name, self.frame)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key in (ord("s"), ord("S")):
                self.save()
            elif key in (ord("u"), ord("U")):
                if self.lines:
                    removed = self.lines.pop()
                    print(f"  ↩️ Undo: xóa {removed['id']}")
                    self.current_point = None
                    self._redraw()
                elif self.current_point is not None:
                    self.current_point = None
                    self._redraw()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Vẽ ROI lines (vạch ngã rẽ) trên video/ảnh")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", "-v", type=str, help="Đường dẫn video (lấy frame đầu tiên)")
    src.add_argument("--image", "-i", type=str, help="Đường dẫn ảnh")
    parser.add_argument("--output", "-o", type=str, default="roi_lines.json",
                        help="File JSON output (mặc định: roi_lines.json)")
    parser.add_argument("--frame", type=int, default=50,
                        help="Video: lấy frame thứ N (mặc định: 50, bỏ qua frame đầu khi BG chưa ổn)")

    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"❌ Không đọc được ảnh: {args.image}")
            return
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"❌ Không mở được video: {args.video}")
            return
        # Nhảy đến frame chỉ định
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"❌ Không đọc được frame {args.frame}")
            return
        print(f"📷 Lấy frame #{args.frame} từ video")

    drawer = ROIDrawer(frame, output_path=args.output)
    drawer.run()


if __name__ == "__main__":
    main()
