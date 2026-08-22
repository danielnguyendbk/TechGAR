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

V3_PREDICTION_FIELDS = {
    "schema_version", "frame_idx", "capture_unix_ns", "wall_time_iso",
    "camera_timestamps_ns", "camera_skew_ms", "observations", "slots",
    "gid_aliases", "identity_events", "parking_events", "parking_recovery",
    "parked_identity_reservations",
}
V3_OBSERVATION_FIELDS = {
    "observation_uid", "camera_id", "local_track_id", "raw_gid",
    "canonical_gid", "gid_aliases", "bbox", "anchor_pixel", "anchor_world",
    "track_state", "association_state", "invisible_count", "assignment_cost",
    "fragment_visible_count", "first_observation_frame", "identity_state",
    "slot_ownership",
}
V3_SLOT_FIELDS = {
    "camera_id", "slot_id", "occupied", "raw_vehicle_gid",
    "canonical_vehicle_gid", "vision_occupied", "tracking_occupied",
    "decision_source", "tracking_state", "vehicle_overlap", "stopped_for_ms",
    "recovery_state", "recovery_global_id", "recovery_age_ms",
    "recovery_radius_px", "recovery_candidate_count",
}
V3_EVENT_FIELDS = {
    "event_uid", "source", "event_type", "frame_idx", "canonical_gid",
    "raw_gid", "details",
}
V3_GT_HEADERS = {
    "ground_truth_slots.csv": [
        "schema_version", "camera_id", "slot_id", "start_frame", "end_frame",
        "occupied", "physical_vehicle_id", "identity_required", "notes",
    ],
    "ground_truth_events.csv": [
        "schema_version", "event_id", "physical_vehicle_id", "event_type",
        "start_frame", "end_frame", "source_camera", "target_camera",
        "source_slot_id", "target_slot_id", "preferred_delay_frames",
        "max_delay_frames", "required", "critical", "notes",
    ],
    "ground_truth_identity.csv": [
        "schema_version", "observation_id", "physical_vehicle_id", "frame_idx",
        "camera_id", "anchor_x", "anchor_y", "slot_id", "phase", "required",
        "notes",
    ],
}
V3_EVENT_TYPES = {
    "vehicle_appeared", "slot_enter", "parked", "departure_started",
    "slot_leave", "camera_handoff", "temporary_occlusion", "vehicle_exited",
}


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


def _validate_two_camera(session: Path, metadata: dict) -> tuple[list[str], dict]:
    videos = (
        "raw_cam1.mp4", "raw_cam2.mp4", "debug_cam1.mp4", "debug_cam2.mp4",
    )
    required = (
        *((() if metadata.get("analysis_only") else videos)),
        "predictions.jsonl", "frame_timestamps.csv", "performance.csv",
        "ground_truth_slots.csv", "ground_truth_events.csv",
    )
    errors = [f"Thieu file {name}" for name in required if not (session / name).is_file()]
    if errors:
        return errors, {}
    prediction_ids = []
    try:
        with (session / "predictions.jsonl").open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if set(record.get("cameras", {})) != {"cam1", "cam2"} or "global_registry" not in record:
                    errors.append(f"Dong JSONL {line_number} thieu du lieu hai camera/Global ID")
                prediction_ids.append(int(record["frame_idx"]))
    except Exception as exc:
        errors.append(f"predictions.jsonl khong hop le: {exc}")
    try:
        timestamp_ids = _csv_frame_ids(session / "frame_timestamps.csv")
        performance_ids = _csv_frame_ids(session / "performance.csv")
        video_counts = {
            name: _video_frames(session / name)
            for name in videos
            if (session / name).is_file()
        }
    except Exception as exc:
        return errors + [f"Khong doc duoc session hai camera: {exc}"], {}
    expected = int(metadata.get("processed_frames", 0))
    counts = {"metadata": expected, "predictions": len(prediction_ids), "timestamps": len(timestamp_ids), "performance": len(performance_ids), **video_counts}
    if expected <= 0:
        errors.append("Session khong co frame nao")
    if len(set(counts.values())) != 1:
        errors.append("So frame/dong khong dong bo: " + json.dumps(counts, ensure_ascii=False))
    expected_ids = list(range(1, expected + 1))
    for label, frame_ids in (("predictions", prediction_ids), ("frame_timestamps", timestamp_ids), ("performance", performance_ids)):
        if frame_ids != expected_ids:
            errors.append(f"frame_idx trong {label} khong lien tuc tu 1 den {expected}")
    return errors, counts


def _is_bool_text(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "false"}


def _validate_v3_csv(session: Path, filename: str, errors: list[str]) -> None:
    path = session / filename
    expected_header = V3_GT_HEADERS[filename]
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != expected_header:
                errors.append(
                    f"{filename} sai header schema v3: can {','.join(expected_header)}"
                )
                return
            seen_ids: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                prefix = f"{filename}:{line_number}"
                if row.get("schema_version") != "3":
                    errors.append(f"{prefix}: schema_version phai la 3")
                if filename == "ground_truth_slots.csv":
                    if row.get("camera_id") not in {"cam1", "cam2"}:
                        errors.append(f"{prefix}: camera_id phai la cam1/cam2")
                    if not _is_bool_text(row.get("occupied")) or not _is_bool_text(row.get("identity_required")):
                        errors.append(f"{prefix}: occupied/identity_required phai la true/false")
                    try:
                        start, end = int(row["start_frame"]), int(row["end_frame"])
                        if start < 1 or end < start:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(f"{prefix}: khoang frame khong hop le")
                    if row.get("identity_required", "").lower() == "true" and not row.get("physical_vehicle_id"):
                        errors.append(f"{prefix}: identity_required=true nhung thieu physical_vehicle_id")
                elif filename == "ground_truth_events.csv":
                    if not _is_bool_text(row.get("required")):
                        errors.append(f"{prefix}: required phai la true/false")
                    event_id = row.get("event_id", "")
                    if not event_id or event_id in seen_ids:
                        errors.append(f"{prefix}: event_id rong hoac bi lap")
                    seen_ids.add(event_id)
                    if row.get("event_type") not in V3_EVENT_TYPES:
                        errors.append(f"{prefix}: event_type khong hop le")
                    if not _is_bool_text(row.get("critical")):
                        errors.append(f"{prefix}: critical phai la true/false")
                    try:
                        start, end = int(row["start_frame"]), int(row["end_frame"])
                        preferred = int(row["preferred_delay_frames"])
                        maximum = int(row["max_delay_frames"])
                        if start < 1 or end < start or preferred < 0 or maximum < preferred:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(f"{prefix}: frame/delay khong hop le")
                else:
                    if not _is_bool_text(row.get("required")):
                        errors.append(f"{prefix}: required phai la true/false")
                    observation_id = row.get("observation_id", "")
                    if not observation_id or observation_id in seen_ids:
                        errors.append(f"{prefix}: observation_id rong hoac bi lap")
                    seen_ids.add(observation_id)
                    if not row.get("physical_vehicle_id"):
                        errors.append(f"{prefix}: thieu physical_vehicle_id")
                    if row.get("camera_id") not in {"cam1", "cam2"}:
                        errors.append(f"{prefix}: camera_id phai la cam1/cam2")
                    try:
                        if int(row["frame_idx"]) < 1:
                            raise ValueError
                    except (TypeError, ValueError):
                        errors.append(f"{prefix}: frame_idx khong hop le")
                    anchors = (row.get("anchor_x", "").strip(), row.get("anchor_y", "").strip())
                    if bool(anchors[0]) != bool(anchors[1]):
                        errors.append(f"{prefix}: anchor_x va anchor_y phai cung co hoac cung trong")
                    if not row.get("slot_id") and not all(anchors):
                        errors.append(f"{prefix}: checkpoint phai co slot_id hoac cap anchor_x/anchor_y")
    except Exception as exc:
        errors.append(f"{filename} khong hop le: {exc}")


def _validate_v3(session: Path, metadata: dict) -> tuple[list[str], dict]:
    videos = ("raw_cam1.mp4", "raw_cam2.mp4", "debug_cam1.mp4", "debug_cam2.mp4")
    required = (
        *((() if metadata.get("analysis_only") else videos)),
        "predictions.jsonl", "frame_timestamps.csv", "performance.csv",
        *V3_GT_HEADERS.keys(),
    )
    errors = [f"Thieu file {name}" for name in required if not (session / name).is_file()]
    if errors:
        return errors, {}

    prediction_ids: list[int] = []
    event_uids: set[str] = set()
    observation_uids: set[str] = set()
    try:
        with (session / "predictions.jsonl").open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                prefix = f"predictions.jsonl:{line_number}"
                if record.get("schema_version") != 3:
                    errors.append(f"{prefix}: schema_version phai la 3")
                missing = V3_PREDICTION_FIELDS - set(record)
                if missing:
                    errors.append(f"{prefix}: thieu field {sorted(missing)}")
                    continue
                frame_idx = int(record["frame_idx"])
                prediction_ids.append(frame_idx)
                if not isinstance(record["capture_unix_ns"], int) or isinstance(record["capture_unix_ns"], bool) or record["capture_unix_ns"] < 0:
                    errors.append(f"{prefix}: capture_unix_ns phai la so nguyen >= 0")
                if not isinstance(record["wall_time_iso"], str):
                    errors.append(f"{prefix}: wall_time_iso phai la chuoi")
                timestamps = record["camera_timestamps_ns"]
                if not isinstance(timestamps, dict) or not all(
                    isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for key, value in (timestamps.items() if isinstance(timestamps, dict) else ())
                ):
                    errors.append(f"{prefix}: camera_timestamps_ns khong hop le")
                skew = record["camera_skew_ms"]
                if not isinstance(skew, (int, float)) or isinstance(skew, bool) or skew < 0:
                    errors.append(f"{prefix}: camera_skew_ms phai la so >= 0")
                for field in (
                    "observations", "slots", "gid_aliases", "identity_events",
                    "parking_events", "parking_recovery", "parked_identity_reservations",
                ):
                    if not isinstance(record[field], list):
                        errors.append(f"{prefix}: {field} phai la list")
                for observation in record["observations"]:
                    missing_observation = V3_OBSERVATION_FIELDS - set(observation)
                    if missing_observation:
                        errors.append(f"{prefix}: observation thieu {sorted(missing_observation)}")
                    uid = str(observation.get("observation_uid", ""))
                    if not uid or uid in observation_uids:
                        errors.append(f"{prefix}: observation_uid rong hoac bi lap: {uid}")
                    observation_uids.add(uid)
                    if observation.get("camera_id") not in {"cam1", "cam2"}:
                        errors.append(f"{prefix}: observation camera_id khong hop le")
                    if observation.get("canonical_gid") is not None and not isinstance(observation.get("canonical_gid"), int):
                        errors.append(f"{prefix}: observation canonical_gid phai la int/null")
                    bbox = observation.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4 or not all(
                        isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox
                    ):
                        errors.append(f"{prefix}: observation bbox phai co 4 so")
                    anchor = observation.get("anchor_pixel")
                    if not isinstance(anchor, dict) or not {"x", "y", "reference"}.issubset(anchor) or not all(
                        isinstance(anchor.get(axis), (int, float)) and not isinstance(anchor.get(axis), bool)
                        for axis in ("x", "y")
                    ):
                        errors.append(f"{prefix}: observation anchor_pixel khong hop le")
                slot_keys: set[tuple[str, str]] = set()
                for slot in record["slots"]:
                    missing_slot = V3_SLOT_FIELDS - set(slot)
                    if missing_slot:
                        errors.append(f"{prefix}: slot thieu {sorted(missing_slot)}")
                    if not all(isinstance(slot.get(field), bool) for field in (
                        "occupied", "vision_occupied", "tracking_occupied",
                    )):
                        errors.append(f"{prefix}: trang thai occupied cua slot phai la bool")
                    key = (str(slot.get("camera_id")), str(slot.get("slot_id")))
                    if key in slot_keys:
                        errors.append(f"{prefix}: slot bi lap {key[0]}/{key[1]}")
                    slot_keys.add(key)
                for alias in record["gid_aliases"]:
                    if not isinstance(alias, dict) or not {"alias_gid", "canonical_gid"}.issubset(alias):
                        errors.append(f"{prefix}: gid_aliases item khong hop le")
                for event_field in ("identity_events", "parking_events"):
                    for event in record[event_field]:
                        missing_event = V3_EVENT_FIELDS - set(event)
                        if missing_event or not isinstance(event.get("details"), dict):
                            errors.append(f"{prefix}: {event_field} item khong hop le")
                        uid = str(event.get("event_uid", ""))
                        if not uid or uid in event_uids:
                            errors.append(f"{prefix}: event_uid rong hoac bi lap: {uid}")
                        event_uids.add(uid)
                        if event_field == "parking_events" and event.get("camera_id") not in {"cam1", "cam2"}:
                            errors.append(f"{prefix}: parking event thieu camera_id hop le")
                for reservation in record["parked_identity_reservations"]:
                    if not isinstance(reservation, dict) or not {
                        "canonical_gid", "camera_id", "slot_id", "state", "bbox",
                    }.issubset(reservation):
                        errors.append(f"{prefix}: parked_identity_reservations item khong hop le")
                        continue
                    if not isinstance(reservation["canonical_gid"], int):
                        errors.append(f"{prefix}: reservation canonical_gid phai la int")
                    if reservation["camera_id"] not in {"cam1", "cam2"} or not isinstance(reservation["slot_id"], str):
                        errors.append(f"{prefix}: reservation camera/slot khong hop le")
    except Exception as exc:
        errors.append(f"predictions.jsonl khong hop le: {exc}")

    try:
        timestamp_ids = _csv_frame_ids(session / "frame_timestamps.csv")
        performance_ids = _csv_frame_ids(session / "performance.csv")
        video_counts = {
            name: _video_frames(session / name)
            for name in videos if (session / name).is_file()
        }
    except Exception as exc:
        return errors + [f"Khong doc duoc session schema v3: {exc}"], {}

    expected = int(metadata.get("processed_frames", 0))
    counts = {
        "metadata": expected, "predictions": len(prediction_ids),
        "timestamps": len(timestamp_ids), "performance": len(performance_ids),
        **video_counts,
    }
    if expected <= 0:
        errors.append("Session khong co frame nao")
    if len(set(counts.values())) != 1:
        errors.append("So frame/dong khong dong bo: " + json.dumps(counts, ensure_ascii=False))
    expected_ids = list(range(1, expected + 1))
    for label, frame_ids in (
        ("predictions", prediction_ids), ("frame_timestamps", timestamp_ids),
        ("performance", performance_ids),
    ):
        if frame_ids != expected_ids:
            errors.append(f"frame_idx trong {label} khong lien tuc tu 1 den {expected}")
    for filename in V3_GT_HEADERS:
        _validate_v3_csv(session, filename, errors)
    return errors, counts


def validate(session: Path) -> tuple[list[str], dict]:
    metadata_path = session / "session_info.json"
    if not metadata_path.is_file():
        return ["Thieu file session_info.json"], {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"session_info.json khong hop le: {exc}"], {}

    schema_version = int(metadata.get("schema_version", 1))
    if schema_version == 3:
        return _validate_v3(session, metadata)
    if schema_version == 2:
        return _validate_two_camera(session, metadata)

    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (session / name).is_file():
            errors.append(f"Thieu file {name}")
    if errors:
        return errors, {}

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
