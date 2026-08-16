"""Validate synchronization and readability of a recorded experiment session."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2


REQUIRED_FILES = (
    "session_info.json",
    "raw_video.mp4",
    "debug_video.mp4",
    "predictions.jsonl",
    "frame_timestamps.csv",
    "performance.csv",
    "ground_truth_slots.csv",
    "ground_truth_events.csv",
)


def _video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Khong doc duoc video: {path.name}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
    capture.release()
    return count


def _csv_frame_ids(path: Path) -> list[int]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return [int(row["frame_idx"]) for row in csv.DictReader(source)]


def validate(session: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (session / name).is_file():
            errors.append(f"Thieu file {name}")
    if errors:
        return errors, {}

    try:
        metadata = json.loads((session / "session_info.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"session_info.json khong hop le: {exc}"], {}

    prediction_ids: list[int] = []
    try:
        with (session / "predictions.jsonl").open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "parking_slots" not in record or "confirmed_vehicles" not in record:
                    errors.append(f"Dong JSONL {line_number} thieu truong du doan")
                prediction_ids.append(int(record["frame_idx"]))
    except Exception as exc:
        errors.append(f"predictions.jsonl khong hop le: {exc}")

    try:
        timestamp_ids = _csv_frame_ids(session / "frame_timestamps.csv")
        performance_ids = _csv_frame_ids(session / "performance.csv")
    except Exception as exc:
        errors.append(f"CSV khong hop le: {exc}")
        timestamp_ids, performance_ids = [], []

    try:
        raw_frames = _video_frames(session / "raw_video.mp4")
        debug_frames = _video_frames(session / "debug_video.mp4")
    except Exception as exc:
        errors.append(str(exc))
        raw_frames, debug_frames = 0, 0

    expected = int(metadata.get("processed_frames", 0))
    counts = {
        "metadata": expected,
        "raw_video": raw_frames,
        "debug_video": debug_frames,
        "predictions": len(prediction_ids),
        "timestamps": len(timestamp_ids),
        "performance": len(performance_ids),
    }
    if expected <= 0:
        errors.append("Session khong co frame nao")
    if len(set(counts.values())) != 1:
        errors.append("So frame/dong khong dong bo: " + json.dumps(counts, ensure_ascii=False))

    expected_ids = list(range(1, expected + 1))
    for label, frame_ids in (
        ("predictions", prediction_ids),
        ("frame_timestamps", timestamp_ids),
        ("performance", performance_ids),
    ):
        if frame_ids != expected_ids:
            errors.append(f"frame_idx trong {label} khong lien tuc tu 1 den {expected}")

    for filename in ("ground_truth_slots.csv", "ground_truth_events.csv"):
        with (session / filename).open("r", newline="", encoding="utf-8-sig") as source:
            if not next(csv.reader(source), None):
                errors.append(f"{filename} khong co header")
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra bo file cua mot TechGAR session")
    parser.add_argument("--session", required=True, type=Path)
    args = parser.parse_args()
    session = args.session.resolve()
    if not session.is_dir():
        print(f"FAIL: Khong tim thay session: {session}")
        return 2
    errors, counts = validate(session)
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: Tat ca file deu doc duoc va dong bo theo frame.")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
