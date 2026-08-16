"""Interactive four-corner parking-slot polygon editor.

This is the only parking-space ROI editor kept in the clean submission.  It
reads/writes the same JSON schema used by ``ParkingDetector`` and produces no
legacy pickle or result files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Polygon = List[Point]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def order_points(points: List[Point]) -> Polygon:
    values = np.asarray(points, dtype=np.float32)
    center = np.mean(values, axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles)]
    top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -top_left, axis=0)
    return [tuple(int(value) for value in point) for point in ordered]


class ParkingSpacePicker:
    def __init__(self, frame: np.ndarray, output_path: Path, reset: bool = False, initial_row: str | None = None):
        self.original = frame.copy()
        self.height, self.width = frame.shape[:2]
        self.output_path = output_path
        self.slots: List[tuple[str, Polygon]] = []
        self.pending: List[Point] = []
        if not reset:
            self._load()
        self.current_row = self._normalize_row(initial_row) if initial_row else self._prompt_row()

    @staticmethod
    def _normalize_row(value: str) -> str:
        row = value.strip().upper()
        if not re.fullmatch(r"[A-Z]", row):
            raise ValueError("Ma day phai la mot chu cai A-Z")
        return row

    def _prompt_row(self) -> str:
        while True:
            try:
                value = input("Chon day hien tai (A-Z, mac dinh A): ").strip() or "A"
                return self._normalize_row(value)
            except ValueError as exc:
                print(exc)

    def _next_slot_id(self) -> str:
        pattern = re.compile(rf"^{re.escape(self.current_row)}(\d+)$")
        highest = 0
        for slot_id, _ in self.slots:
            match = pattern.fullmatch(slot_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"{self.current_row}{highest + 1:02d}"

    def _load(self) -> None:
        if not self.output_path.is_file():
            return
        with self.output_path.open(encoding="utf-8") as file:
            data = json.load(file)
        ref_width = max(1, int(data.get("imageWidth", self.width)))
        ref_height = max(1, int(data.get("imageHeight", self.height)))
        scale_x, scale_y = self.width / ref_width, self.height / ref_height
        for slot in data.get("slots", []):
            polygon = slot.get("polygon", [])
            if len(polygon) != 4:
                continue
            self.slots.append((str(slot.get("id", f"P{len(self.slots) + 1:03d}")), [
                (round(float(point["x"]) * scale_x), round(float(point["y"]) * scale_y))
                for point in polygon
            ]))
        print(f"Loaded {len(self.slots)} parking slots from {self.output_path}")

    def save(self) -> None:
        slots = []
        for slot_id, polygon in self.slots:
            points = np.asarray(polygon, dtype=np.float32)
            slots.append({
                "id": slot_id,
                "type": "polygon",
                "polygon": [{"x": int(x), "y": int(y)} for x, y in polygon],
                "center": {
                    "x": int(round(float(np.mean(points[:, 0])))),
                    "y": int(round(float(np.mean(points[:, 1])))),
                },
                "status": "empty",
            })
        data = {"imageWidth": self.width, "imageHeight": self.height, "slots": slots}
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temporary.replace(self.output_path)
        print(f"Saved {len(slots)} parking slots to {self.output_path}")

    @staticmethod
    def _contains(point: Point, polygon: Polygon) -> bool:
        contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def _mouse(self, event, x, y, _flags, _parameter) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pending) < 4:
            self.pending.append((x, y))
            print(f"Point {len(self.pending)}/4: ({x}, {y})")
        elif event == cv2.EVENT_RBUTTONDOWN:
            for index, (slot_id, polygon) in enumerate(self.slots):
                if self._contains((x, y), polygon):
                    self.slots.pop(index)
                    print(f"Removed slot {slot_id}")
                    break

    def _render(self) -> np.ndarray:
        image = self.original.copy()
        for slot_id, polygon in self.slots:
            points = np.asarray(polygon, dtype=np.int32)
            cv2.polylines(image, [points], True, (255, 0, 255), 2)
            center = tuple(np.mean(points, axis=0).astype(int))
            cv2.putText(image, slot_id, center, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        for index, point in enumerate(self.pending, start=1):
            cv2.circle(image, point, 5, (0, 255, 255), -1)
            cv2.putText(image, str(index), (point[0] + 7, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for index in range(1, len(self.pending)):
            cv2.line(image, self.pending[index - 1], self.pending[index], (0, 255, 0), 2)
        if len(self.pending) == 4:
            cv2.line(image, self.pending[-1], self.pending[0], (0, 200, 0), 2)
        cv2.rectangle(image, (0, 0), (min(image.shape[1], 820), 38), (0, 0, 0), -1)
        status = (
            f"Slots={len(self.slots)} Row={self.current_row} Next={self._next_slot_id()} Pending={len(self.pending)}/4 | "
            "Left=point Right=delete A=accept O=row D=cancel Z=undo S=save Q=save+quit"
        )
        cv2.putText(image, status, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        return image

    def run(self) -> None:
        window = "TechGAR Parking Space Picker"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse)
        while True:
            cv2.imshow(window, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                self.save()
                break
            if key == ord("s"):
                self.save()
            elif key == ord("a"):
                if len(self.pending) == 4:
                    slot_id = self._next_slot_id()
                    self.slots.append((slot_id, order_points(self.pending)))
                    print(f"Added slot {slot_id}")
                    self.pending.clear()
                else:
                    print("Four points are required before accepting a slot")
            elif key == ord("d"):
                self.pending.clear()
            elif key == ord("z"):
                if self.pending:
                    self.pending.pop()
                elif self.slots:
                    slot_id, _ = self.slots.pop()
                    print(f"Removed slot {slot_id}")
            elif key == ord("c"):
                self.pending.clear()
                self.slots.clear()
            elif key == ord("o"):
                if self.pending:
                    print("Hay A/D de xu ly 4 diem dang ve truoc khi doi day")
                else:
                    self.current_row = self._prompt_row()
                    print(f"Da chuyen sang day {self.current_row}; slot tiep theo: {self._next_slot_id()}")
        cv2.destroyAllWindows()


def load_frame(image_path: Path | None, video_path: Path | None, frame_index: int) -> np.ndarray:
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return frame
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw four-corner parking-slot polygons")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Input video")
    source.add_argument("--image", type=Path, help="Input image")
    parser.add_argument("--frame", type=int, default=50, help="Video frame used for calibration")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "config" / "parking_slots.json")
    parser.add_argument("--reset", action="store_true", help="Start empty instead of loading the output JSON")
    parser.add_argument("--row", help="Day bat dau A-Z; neu bo trong se hoi trong terminal")
    args = parser.parse_args()
    if args.video is None and args.image is None:
        args.video = PROJECT_ROOT / "data" / "carPark.mp4"
    return args


def main() -> None:
    args = parse_args()
    picker = ParkingSpacePicker(load_frame(args.image, args.video, args.frame), args.output, reset=args.reset, initial_row=args.row)
    picker.run()


if __name__ == "__main__":
    main()
