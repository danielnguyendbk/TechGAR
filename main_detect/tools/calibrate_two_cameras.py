"""Create a two-camera calibration from four matching overlap corners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(url: str, camera_id: str) -> np.ndarray:
    capture = cv2.VideoCapture(url)
    try:
        ok, frame = capture.read()
        if not capture.isOpened() or not ok:
            raise RuntimeError(f"Khong mo duoc {camera_id}: {url}")
        return frame
    finally:
        capture.release()


def select_points(window: str, frame: np.ndarray) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        preview = frame.copy()
        for index, point in enumerate(points, start=1):
            cv2.circle(preview, point, 7, (0, 255, 255), -1)
            cv2.putText(preview, str(index), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(preview, "Click 4 corners clockwise. Right-click undo. Enter confirm.", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        if key in (13, 10) and len(points) == 4:
            return points
        if key == ord("r"):
            points.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hieu chinh overlap cam1/cam2 bang bon goc chung")
    parser.add_argument("--cam1-url", required=True)
    parser.add_argument("--cam2-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cam1 = read_frame(args.cam1_url, "cam1")
    cam2 = read_frame(args.cam2_url, "cam2")
    try:
        print("Chon 4 goc cua vung overlap tren cam1, theo chieu kim dong ho.")
        cam1_points = select_points("Calibration cam1", cam1)
        print("Chon DUNG 4 diem vat ly do tren cam2, cung thu tu.")
        cam2_points = select_points("Calibration cam2", cam2)
    finally:
        cv2.destroyAllWindows()

    homography, mask = cv2.findHomography(np.float32(cam2_points), np.float32(cam1_points), method=0)
    if homography is None or mask is None:
        raise RuntimeError("Khong tinh duoc homography")
    payload = {
        "camera_transforms": {
            "cam1": np.eye(3).tolist(),
            "cam2": homography.tolist(),
        },
        "edge_adjacency": [
            {"source_camera": "cam1", "exit_edge": "right", "target_camera": "cam2"},
            {"source_camera": "cam2", "exit_edge": "left", "target_camera": "cam1"},
        ],
        "overlap_world_polygon": [[int(x), int(y)] for x, y in cam1_points],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Da ghi calibration: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
