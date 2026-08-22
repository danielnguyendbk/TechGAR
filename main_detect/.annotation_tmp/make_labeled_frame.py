import argparse
import json
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--slots", required=True)
parser.add_argument("--frame", type=int, required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

config = json.loads(Path(args.slots).read_text(encoding="utf-8-sig"))
capture = cv2.VideoCapture(args.video)
capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame - 1)
ok, image = capture.read()
capture.release()
if not ok:
    raise RuntimeError(f"Cannot read frame {args.frame}")

for slot in config["slots"]:
    polygon = np.asarray(
        [(point["x"], point["y"]) for point in slot["polygon"]],
        dtype=np.int32,
    )
    cv2.polylines(image, [polygon], True, (0, 255, 255), 3)
    center = polygon.mean(axis=0).astype(int)
    cv2.putText(
        image,
        slot["id"],
        (int(center[0]) - 16, int(center[1]) + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        slot["id"],
        (int(center[0]) - 16, int(center[1]) + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

cv2.imwrite(args.output, image)
