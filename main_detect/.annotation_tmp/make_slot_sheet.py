import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--slots", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.slots).read_text(encoding="utf-8-sig"))
    slot = next(item for item in config["slots"] if item["id"].upper() == args.slot.upper())
    polygon = np.asarray([(point["x"], point["y"]) for point in slot["polygon"]], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    pad_x = max(50, w // 2)
    pad_y = max(50, h // 2)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1 = min(int(config["imageWidth"]), x + w + pad_x)
    y1 = min(int(config["imageHeight"]), y + h + pad_y)

    capture = cv2.VideoCapture(args.video)
    panels = []
    for frame_number in range(args.start, args.end + 1, args.step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = capture.read()
        if not ok:
            continue
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 3)
        crop = frame[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (255, 255, 255), 2)
        cv2.putText(crop, f"{args.slot} f{frame_number}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop, f"{args.slot} f{frame_number}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(crop)
    capture.release()

    if not panels:
        raise RuntimeError("No frames could be read")
    panel_h = max(panel.shape[0] for panel in panels)
    panel_w = max(panel.shape[1] for panel in panels)
    rows = (len(panels) + args.cols - 1) // args.cols
    sheet = np.zeros((rows * panel_h, args.cols * panel_w, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, col = divmod(index, args.cols)
        sheet[row * panel_h:row * panel_h + panel.shape[0], col * panel_w:col * panel_w + panel.shape[1]] = panel
    cv2.imwrite(args.output, sheet)


if __name__ == "__main__":
    main()
