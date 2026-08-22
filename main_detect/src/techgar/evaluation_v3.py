"""Schema-v3 evaluator for practical TechGAR identity lifecycles.

The evaluator deliberately treats a human vehicle label (for example ``M02_V1``)
as a different namespace from a numeric Global ID.  A one-to-one maximum-weight
matching is learned from manually placed identity checkpoints before any identity
or slot-owner score is calculated.

Only schema version 3 is accepted.  This is intentional: silently interpreting a
schema-v2 recording would make identity metrics look valid while essential data is
missing.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 3
EVALUATOR_VERSION = "3.0"

SLOT_COLUMNS = (
    "schema_version", "camera_id", "slot_id", "start_frame", "end_frame",
    "occupied", "physical_vehicle_id", "identity_required", "notes",
)
EVENT_COLUMNS = (
    "schema_version", "event_id", "physical_vehicle_id", "event_type",
    "start_frame", "end_frame", "source_camera", "target_camera",
    "source_slot_id", "target_slot_id", "preferred_delay_frames",
    "max_delay_frames", "required", "critical", "notes",
)
IDENTITY_COLUMNS = (
    "schema_version", "observation_id", "physical_vehicle_id", "frame_idx",
    "camera_id", "anchor_x", "anchor_y", "slot_id", "phase", "required",
    "notes",
)
GROUND_EVENT_TYPES = {
    "vehicle_appeared", "slot_enter", "parked", "departure_started",
    "slot_leave", "camera_handoff", "temporary_occlusion", "vehicle_exited",
}
CAMERA_RE = re.compile(r"^cam[1-9][0-9]*$")

PREDICTION_TOP_FIELDS = {
    "schema_version", "frame_idx", "capture_unix_ns", "wall_time_iso",
    "camera_timestamps_ns", "camera_skew_ms", "observations", "slots",
    "gid_aliases", "identity_events", "parking_events", "parking_recovery",
    "parked_identity_reservations",
}
OBSERVATION_FIELDS = {
    "observation_uid", "camera_id", "local_track_id", "raw_gid",
    "canonical_gid", "gid_aliases", "bbox", "anchor_pixel", "anchor_world",
    "track_state", "association_state", "invisible_count", "assignment_cost",
    "fragment_visible_count", "first_observation_frame", "identity_state",
    "slot_ownership",
}
SLOT_PREDICTION_FIELDS = {
    "camera_id", "slot_id", "occupied", "raw_vehicle_gid",
    "canonical_vehicle_gid", "vision_occupied", "tracking_occupied",
    "decision_source", "tracking_state", "vehicle_overlap", "stopped_for_ms",
    "recovery_state", "recovery_global_id", "recovery_age_ms",
    "recovery_radius_px", "recovery_candidate_count",
}
EVENT_PREDICTION_FIELDS = {
    "event_uid", "source", "event_type", "frame_idx", "canonical_gid",
    "raw_gid", "details",
}


class EvaluationValidationError(ValueError):
    """Raised when schema-v3 experiment data is incomplete or inconsistent."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        preview = "\n".join(f"- {item}" for item in self.errors[:25])
        if len(self.errors) > 25:
            preview += f"\n- ... and {len(self.errors) - 25} more"
        super().__init__(f"Schema v3 validation failed:\n{preview}")


@dataclass(frozen=True)
class EvaluatorConfig:
    identity_anchor_max_distance_px: float = 120.0
    phantom_max_frames: int = 25
    short_flicker_max_frames: int = 12
    wrong_owner_critical_frames: int = 5
    slot_binding_preferred_frames: int = 25
    slot_binding_max_frames: int = 75
    recovery_preferred_frames: int = 25
    recovery_max_frames: int = 125
    occupancy_delay_preferred_frames: int = 12
    occupancy_delay_max_frames: int = 75
    sustained_state_frames: int = 3


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


def _nullable_int(value: Any) -> bool:
    return value is None or _is_int(value)


def _parse_bool(value: str, location: str, errors: List[str]) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        errors.append(f"{location}: expected true/false, got {value!r}")
    return normalized == "true"


def _parse_int(value: str, location: str, errors: List[str], *, optional: bool = False) -> Optional[int]:
    if optional and not value.strip():
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"{location}: expected integer, got {value!r}")
        return None


def _parse_float(value: str, location: str, errors: List[str], *, optional: bool = False) -> Optional[float]:
    if optional and not value.strip():
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
        return result
    except (TypeError, ValueError):
        errors.append(f"{location}: expected finite number, got {value!r}")
        return None


def _validate_camera(value: str, location: str, errors: List[str], *, optional: bool = False) -> None:
    if optional and not value:
        return
    if not CAMERA_RE.fullmatch(value):
        errors.append(f"{location}: expected camera id like cam1, got {value!r}")


def _read_csv(path: Path, columns: Sequence[str], errors: List[str]) -> List[dict]:
    if not path.exists():
        errors.append(f"missing required file: {path.name}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        if actual != tuple(columns):
            errors.append(
                f"{path.name}: columns must exactly be {list(columns)}, got {list(actual)}"
            )
            return []
        return [dict(row) for row in reader]


def _load_ground_truth(session_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    errors: List[str] = []
    slot_rows = _read_csv(session_dir / "ground_truth_slots.csv", SLOT_COLUMNS, errors)
    event_rows = _read_csv(session_dir / "ground_truth_events.csv", EVENT_COLUMNS, errors)
    identity_rows = _read_csv(session_dir / "ground_truth_identity.csv", IDENTITY_COLUMNS, errors)

    slots: List[dict] = []
    seen_intervals: MutableMapping[Tuple[str, str], List[Tuple[int, int, int]]] = defaultdict(list)
    identity_slot_vehicles = set()
    for index, row in enumerate(slot_rows, start=2):
        loc = f"ground_truth_slots.csv:{index}"
        if row["schema_version"].strip() != "3":
            errors.append(f"{loc}: schema_version must be 3")
        cam = row["camera_id"].strip()
        slot_id = row["slot_id"].strip()
        _validate_camera(cam, f"{loc}.camera_id", errors)
        if not slot_id:
            errors.append(f"{loc}.slot_id: must not be empty")
        start = _parse_int(row["start_frame"], f"{loc}.start_frame", errors)
        end = _parse_int(row["end_frame"], f"{loc}.end_frame", errors)
        occupied = _parse_bool(row["occupied"], f"{loc}.occupied", errors)
        required = _parse_bool(row["identity_required"], f"{loc}.identity_required", errors)
        physical = row["physical_vehicle_id"].strip() or None
        if start is not None and end is not None and (start < 0 or end < start):
            errors.append(f"{loc}: invalid frame interval {start}..{end}")
        if required and (not occupied or not physical):
            errors.append(
                f"{loc}: identity_required=true requires occupied=true and physical_vehicle_id"
            )
        if not occupied and physical:
            errors.append(f"{loc}: a free interval cannot own physical_vehicle_id {physical!r}")
        item = {
            "camera_id": cam, "slot_id": slot_id, "start_frame": start or 0,
            "end_frame": end or 0, "occupied": occupied,
            "physical_vehicle_id": physical, "identity_required": required,
            "notes": row["notes"],
        }
        slots.append(item)
        if start is not None and end is not None:
            seen_intervals[(cam, slot_id)].append((start, end, index))
        if required and physical:
            identity_slot_vehicles.add(physical)

    for key, intervals in seen_intervals.items():
        previous_end = -1
        previous_line = None
        for start, end, line in sorted(intervals):
            if start <= previous_end:
                errors.append(
                    f"ground_truth_slots.csv:{line}: overlaps previous interval for "
                    f"{key[0]}/{key[1]} (line {previous_line})"
                )
            previous_end, previous_line = max(previous_end, end), line

    events: List[dict] = []
    event_ids = set()
    for index, row in enumerate(event_rows, start=2):
        loc = f"ground_truth_events.csv:{index}"
        if row["schema_version"].strip() != "3":
            errors.append(f"{loc}: schema_version must be 3")
        event_id = row["event_id"].strip()
        physical = row["physical_vehicle_id"].strip()
        event_type = row["event_type"].strip()
        if not event_id or event_id in event_ids:
            errors.append(f"{loc}.event_id: must be non-empty and unique")
        event_ids.add(event_id)
        if not physical:
            errors.append(f"{loc}.physical_vehicle_id: must not be empty")
        if event_type not in GROUND_EVENT_TYPES:
            errors.append(f"{loc}.event_type: unsupported value {event_type!r}")
        start = _parse_int(row["start_frame"], f"{loc}.start_frame", errors)
        end = _parse_int(row["end_frame"], f"{loc}.end_frame", errors)
        preferred = _parse_int(
            row["preferred_delay_frames"], f"{loc}.preferred_delay_frames", errors
        )
        maximum = _parse_int(row["max_delay_frames"], f"{loc}.max_delay_frames", errors)
        source_camera = row["source_camera"].strip() or None
        target_camera = row["target_camera"].strip() or None
        _validate_camera(source_camera or "", f"{loc}.source_camera", errors, optional=True)
        _validate_camera(target_camera or "", f"{loc}.target_camera", errors, optional=True)
        required = _parse_bool(row["required"], f"{loc}.required", errors)
        critical = _parse_bool(row["critical"], f"{loc}.critical", errors)
        if start is not None and end is not None and (start < 0 or end < start):
            errors.append(f"{loc}: invalid frame interval {start}..{end}")
        if preferred is not None and preferred < 0:
            errors.append(f"{loc}.preferred_delay_frames: must be >= 0")
        if maximum is not None and (maximum < 0 or (preferred is not None and maximum < preferred)):
            errors.append(f"{loc}.max_delay_frames: must be >= preferred_delay_frames")
        if event_type == "camera_handoff" and (not source_camera or not target_camera):
            errors.append(f"{loc}: camera_handoff requires source_camera and target_camera")
        events.append({
            "event_id": event_id, "physical_vehicle_id": physical,
            "event_type": event_type, "start_frame": start or 0,
            "end_frame": end or 0, "source_camera": source_camera,
            "target_camera": target_camera,
            "source_slot_id": row["source_slot_id"].strip() or None,
            "target_slot_id": row["target_slot_id"].strip() or None,
            "preferred_delay_frames": preferred or 0,
            "max_delay_frames": maximum or 0, "required": required,
            "critical": critical, "notes": row["notes"],
        })

    identities: List[dict] = []
    observation_ids = set()
    checkpoint_vehicles = set()
    for index, row in enumerate(identity_rows, start=2):
        loc = f"ground_truth_identity.csv:{index}"
        if row["schema_version"].strip() != "3":
            errors.append(f"{loc}: schema_version must be 3")
        observation_id = row["observation_id"].strip()
        physical = row["physical_vehicle_id"].strip()
        camera_id = row["camera_id"].strip()
        slot_id = row["slot_id"].strip() or None
        phase = row["phase"].strip()
        frame_idx = _parse_int(row["frame_idx"], f"{loc}.frame_idx", errors)
        anchor_x = _parse_float(row["anchor_x"], f"{loc}.anchor_x", errors, optional=True)
        anchor_y = _parse_float(row["anchor_y"], f"{loc}.anchor_y", errors, optional=True)
        required = _parse_bool(row["required"], f"{loc}.required", errors)
        if not observation_id or observation_id in observation_ids:
            errors.append(f"{loc}.observation_id: must be non-empty and unique")
        observation_ids.add(observation_id)
        if not physical:
            errors.append(f"{loc}.physical_vehicle_id: must not be empty")
        _validate_camera(camera_id, f"{loc}.camera_id", errors)
        if frame_idx is not None and frame_idx < 0:
            errors.append(f"{loc}.frame_idx: must be >= 0")
        if bool(anchor_x is None) != bool(anchor_y is None):
            errors.append(f"{loc}: anchor_x and anchor_y must both be present or both blank")
        if slot_id is None and (anchor_x is None or anchor_y is None):
            errors.append(f"{loc}: moving checkpoint requires anchor_x and anchor_y")
        if not phase:
            errors.append(f"{loc}.phase: must not be empty")
        identities.append({
            "observation_id": observation_id, "physical_vehicle_id": physical,
            "frame_idx": frame_idx or 0, "camera_id": camera_id,
            "anchor_x": anchor_x, "anchor_y": anchor_y, "slot_id": slot_id,
            "phase": phase, "required": required, "notes": row["notes"],
        })
        checkpoint_vehicles.add(physical)

    missing_checkpoint = sorted(identity_slot_vehicles - checkpoint_vehicles)
    if missing_checkpoint:
        errors.append(
            "identity-required slot vehicles have no identity checkpoint: "
            + ", ".join(missing_checkpoint)
        )
    scored_event_vehicles = {
        event["physical_vehicle_id"]
        for event in events
        if event["required"]
        and event["event_type"] in {"camera_handoff", "departure_started", "slot_leave"}
    }
    missing_event_checkpoint = sorted(scored_event_vehicles - checkpoint_vehicles)
    if missing_event_checkpoint:
        errors.append(
            "identity-scored event vehicles have no identity checkpoint: "
            + ", ".join(missing_event_checkpoint)
        )
    if errors:
        raise EvaluationValidationError(errors)
    return slots, events, identities


def _missing_fields(value: Mapping[str, Any], fields: Iterable[str]) -> List[str]:
    return sorted(set(fields) - set(value))


def _validate_prediction_frame(data: Any, line_no: int, errors: List[str]) -> None:
    loc = f"predictions.jsonl:{line_no}"
    if not isinstance(data, dict):
        errors.append(f"{loc}: each line must be a JSON object")
        return
    if data.get("schema_version") != 3:
        errors.append(f"{loc}.schema_version: only schema 3 is supported")
    missing = _missing_fields(data, PREDICTION_TOP_FIELDS)
    if missing:
        errors.append(f"{loc}: missing fields {missing}")
        return
    if not _is_int(data.get("frame_idx")) or data["frame_idx"] < 0:
        errors.append(f"{loc}.frame_idx: expected non-negative integer")
    if not _is_int(data.get("capture_unix_ns")) or data["capture_unix_ns"] < 0:
        errors.append(f"{loc}.capture_unix_ns: expected non-negative integer")
    if not isinstance(data.get("wall_time_iso"), str):
        errors.append(f"{loc}.wall_time_iso: expected string")
    timestamps = data.get("camera_timestamps_ns")
    if not isinstance(timestamps, dict) or not all(
        isinstance(key, str) and _is_int(value) and value >= 0
        for key, value in (timestamps.items() if isinstance(timestamps, dict) else ())
    ):
        errors.append(f"{loc}.camera_timestamps_ns: expected object of non-negative integers")
    if not _is_number(data.get("camera_skew_ms")) or data["camera_skew_ms"] < 0:
        errors.append(f"{loc}.camera_skew_ms: expected non-negative number")
    for field in (
        "observations", "slots", "gid_aliases", "identity_events",
        "parking_events", "parking_recovery", "parked_identity_reservations",
    ):
        if not isinstance(data.get(field), list):
            errors.append(f"{loc}.{field}: expected list")

    for idx, item in enumerate(data.get("observations", []) if isinstance(data.get("observations"), list) else []):
        item_loc = f"{loc}.observations[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_loc}: expected object")
            continue
        absent = _missing_fields(item, OBSERVATION_FIELDS)
        if absent:
            errors.append(f"{item_loc}: missing fields {absent}")
            continue
        if not isinstance(item["observation_uid"], str) or not item["observation_uid"]:
            errors.append(f"{item_loc}.observation_uid: expected non-empty string")
        _validate_camera(str(item["camera_id"]), f"{item_loc}.camera_id", errors)
        if not _is_int(item["local_track_id"]):
            errors.append(f"{item_loc}.local_track_id: expected integer")
        for key in ("raw_gid", "canonical_gid"):
            if not _nullable_int(item[key]):
                errors.append(f"{item_loc}.{key}: expected integer or null")
        if not isinstance(item["gid_aliases"], list) or not all(_is_int(v) for v in item["gid_aliases"]):
            errors.append(f"{item_loc}.gid_aliases: expected integer list")
        if not isinstance(item["bbox"], list) or len(item["bbox"]) != 4 or not all(_is_number(v) for v in item["bbox"]):
            errors.append(f"{item_loc}.bbox: expected four numbers")
        anchor = item["anchor_pixel"]
        if not isinstance(anchor, dict) or not {"x", "y", "reference"}.issubset(anchor) or not _is_number(anchor.get("x")) or not _is_number(anchor.get("y")):
            errors.append(f"{item_loc}.anchor_pixel: expected x/y/reference object")
        world = item["anchor_world"]
        if world is not None and (
            not isinstance(world, dict)
            or not {"x", "y", "unit", "reference"}.issubset(world)
            or not _is_number(world.get("x")) or not _is_number(world.get("y"))
        ):
            errors.append(f"{item_loc}.anchor_world: expected x/y/unit/reference object or null")
        if not _is_int(item["invisible_count"]) or item["invisible_count"] < 0:
            errors.append(f"{item_loc}.invisible_count: expected non-negative integer")
        if not _is_int(item["fragment_visible_count"]) or item["fragment_visible_count"] < 0:
            errors.append(f"{item_loc}.fragment_visible_count: expected non-negative integer")
        if not _is_int(item["first_observation_frame"]) or item["first_observation_frame"] < 0:
            errors.append(f"{item_loc}.first_observation_frame: expected non-negative integer")
        for key in ("track_state", "association_state", "identity_state"):
            if not isinstance(item[key], str):
                errors.append(f"{item_loc}.{key}: expected string")
        if item["assignment_cost"] is not None and not isinstance(item["assignment_cost"], dict):
            errors.append(f"{item_loc}.assignment_cost: expected object or null")
        owner = item["slot_ownership"]
        if owner is not None and (
            not isinstance(owner, dict)
            or not {"camera_id", "slot_id", "state"}.issubset(owner)
        ):
            errors.append(f"{item_loc}.slot_ownership: expected camera/slot/state object or null")
        elif isinstance(owner, dict):
            _validate_camera(str(owner["camera_id"]), f"{item_loc}.slot_ownership.camera_id", errors)
            if not isinstance(owner["slot_id"], str) or not owner["slot_id"]:
                errors.append(f"{item_loc}.slot_ownership.slot_id: expected non-empty string")
            if not isinstance(owner["state"], str):
                errors.append(f"{item_loc}.slot_ownership.state: expected string")

    for idx, item in enumerate(data.get("slots", []) if isinstance(data.get("slots"), list) else []):
        item_loc = f"{loc}.slots[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{item_loc}: expected object")
            continue
        absent = _missing_fields(item, SLOT_PREDICTION_FIELDS)
        if absent:
            errors.append(f"{item_loc}: missing fields {absent}")
            continue
        _validate_camera(str(item["camera_id"]), f"{item_loc}.camera_id", errors)
        if not isinstance(item["slot_id"], str) or not item["slot_id"]:
            errors.append(f"{item_loc}.slot_id: expected non-empty string")
        for key in ("occupied", "vision_occupied", "tracking_occupied"):
            if not isinstance(item[key], bool):
                errors.append(f"{item_loc}.{key}: expected boolean")
        for key in ("raw_vehicle_gid", "canonical_vehicle_gid", "recovery_global_id"):
            if not _nullable_int(item[key]):
                errors.append(f"{item_loc}.{key}: expected integer or null")
        for key in ("decision_source", "tracking_state", "recovery_state"):
            if not isinstance(item[key], str):
                errors.append(f"{item_loc}.{key}: expected string")
        if not _is_number(item["vehicle_overlap"]):
            errors.append(f"{item_loc}.vehicle_overlap: expected number")
        if not _is_int(item["stopped_for_ms"]) or item["stopped_for_ms"] < 0:
            errors.append(f"{item_loc}.stopped_for_ms: expected non-negative integer")
        if not _is_int(item["recovery_age_ms"]) or item["recovery_age_ms"] < 0:
            errors.append(f"{item_loc}.recovery_age_ms: expected non-negative integer")
        if not _is_number(item["recovery_radius_px"]) or item["recovery_radius_px"] < 0:
            errors.append(f"{item_loc}.recovery_radius_px: expected non-negative number")
        if not _is_int(item["recovery_candidate_count"]) or item["recovery_candidate_count"] < 0:
            errors.append(f"{item_loc}.recovery_candidate_count: expected non-negative integer")

    for idx, item in enumerate(data.get("gid_aliases", []) if isinstance(data.get("gid_aliases"), list) else []):
        item_loc = f"{loc}.gid_aliases[{idx}]"
        if not isinstance(item, dict) or not {"alias_gid", "canonical_gid"}.issubset(item):
            errors.append(f"{item_loc}: expected alias_gid/canonical_gid object")
        elif not _is_int(item["alias_gid"]) or not _is_int(item["canonical_gid"]):
            errors.append(f"{item_loc}: alias and canonical GIDs must be integers")

    for list_name in ("identity_events", "parking_events"):
        for idx, item in enumerate(data.get(list_name, []) if isinstance(data.get(list_name), list) else []):
            item_loc = f"{loc}.{list_name}[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{item_loc}: expected object")
                continue
            absent = _missing_fields(item, EVENT_PREDICTION_FIELDS)
            if absent:
                errors.append(f"{item_loc}: missing fields {absent}")
                continue
            if not isinstance(item["event_uid"], str) or not item["event_uid"]:
                errors.append(f"{item_loc}.event_uid: expected non-empty string")
            if not isinstance(item["source"], str) or not item["source"]:
                errors.append(f"{item_loc}.source: expected non-empty string")
            if not isinstance(item["event_type"], str) or not item["event_type"]:
                errors.append(f"{item_loc}.event_type: expected non-empty string")
            if not _is_int(item["frame_idx"]):
                errors.append(f"{item_loc}.frame_idx: expected integer")
            for key in ("canonical_gid", "raw_gid"):
                if not _nullable_int(item[key]):
                    errors.append(f"{item_loc}.{key}: expected integer or null")
            if not isinstance(item["details"], dict):
                errors.append(f"{item_loc}.details: expected object")
            if list_name == "parking_events":
                if "camera_id" not in item:
                    errors.append(f"{item_loc}.camera_id: required for parking event")
                else:
                    _validate_camera(str(item["camera_id"]), f"{item_loc}.camera_id", errors)

    for field in ("parking_recovery", "parked_identity_reservations"):
        for idx, item in enumerate(data.get(field, []) if isinstance(data.get(field), list) else []):
            if not isinstance(item, dict):
                errors.append(f"{loc}.{field}[{idx}]: expected object")
                continue
            if field == "parked_identity_reservations":
                item_loc = f"{loc}.{field}[{idx}]"
                absent = _missing_fields(
                    item, {"canonical_gid", "camera_id", "slot_id", "state", "bbox"}
                )
                if absent:
                    errors.append(f"{item_loc}: missing fields {absent}")
                    continue
                if not _is_int(item["canonical_gid"]):
                    errors.append(f"{item_loc}.canonical_gid: expected integer")
                _validate_camera(str(item["camera_id"]), f"{item_loc}.camera_id", errors)
                if not isinstance(item["slot_id"], str) or not item["slot_id"]:
                    errors.append(f"{item_loc}.slot_id: expected non-empty string")
                if not isinstance(item["state"], str):
                    errors.append(f"{item_loc}.state: expected string")
                if item["bbox"] is not None and (
                    not isinstance(item["bbox"], list)
                    or len(item["bbox"]) != 4
                    or not all(_is_number(value) for value in item["bbox"])
                ):
                    errors.append(f"{item_loc}.bbox: expected four numbers or null")


def _load_predictions(path: Path) -> List[dict]:
    if not path.exists():
        raise EvaluationValidationError([f"missing required file: {path.name}"])
    errors: List[str] = []
    frames: List[dict] = []
    seen_frames = set()
    seen_observations = set()
    seen_events = set()
    previous_frame = -1
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"predictions.jsonl:{line_no}: invalid JSON ({exc.msg})")
                continue
            _validate_prediction_frame(data, line_no, errors)
            if not isinstance(data, dict) or not _is_int(data.get("frame_idx")):
                continue
            frame_idx = data["frame_idx"]
            if frame_idx in seen_frames:
                errors.append(f"predictions.jsonl:{line_no}: duplicate frame_idx {frame_idx}")
            if frame_idx <= previous_frame:
                errors.append(f"predictions.jsonl:{line_no}: frame_idx must be strictly increasing")
            previous_frame = frame_idx
            seen_frames.add(frame_idx)
            for observation in data.get("observations", []):
                if not isinstance(observation, dict):
                    continue
                uid = observation.get("observation_uid")
                if uid in seen_observations:
                    errors.append(f"predictions.jsonl:{line_no}: repeated observation_uid {uid!r}")
                seen_observations.add(uid)
            for list_name in ("identity_events", "parking_events"):
                for event in data.get(list_name, []):
                    if not isinstance(event, dict):
                        continue
                    uid = event.get("event_uid")
                    if uid in seen_events:
                        errors.append(
                            f"predictions.jsonl:{line_no}: event_uid {uid!r} repeated; "
                            "schema 3 requires delta events"
                        )
                    seen_events.add(uid)
                    if _is_int(event.get("frame_idx")) and event["frame_idx"] > frame_idx:
                        errors.append(
                            f"predictions.jsonl:{line_no}: event {uid!r} is from a future frame"
                        )
            slots = [
                (item.get("camera_id"), item.get("slot_id"))
                for item in data.get("slots", []) if isinstance(item, dict)
            ]
            if len(slots) != len(set(slots)):
                errors.append(f"predictions.jsonl:{line_no}: duplicate camera/slot entry")
            frames.append(data)
    if not frames:
        errors.append("predictions.jsonl: no prediction frames")
    else:
        actual_frames = [frame["frame_idx"] for frame in frames]
        expected_frames = list(range(1, len(frames) + 1))
        if actual_frames != expected_frames:
            errors.append(
                "predictions.jsonl: frame_idx must be contiguous from 1 "
                f"through {len(frames)}; got bounds {actual_frames[0]}..{actual_frames[-1]}"
            )
    if errors:
        raise EvaluationValidationError(errors)
    return frames


def _validate_session_info(path: Path, frames: Sequence[dict]) -> dict:
    errors: List[str] = []
    if not path.exists():
        raise EvaluationValidationError([f"missing required file: {path.name}"])
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationValidationError([f"{path.name}: invalid JSON ({exc})"]) from exc
    if not isinstance(data, dict):
        errors.append(f"{path.name}: expected a JSON object")
    else:
        if data.get("schema_version") != 3:
            errors.append(f"{path.name}.schema_version: must be 3")
        processed = data.get("processed_frames")
        if not _is_int(processed) or processed < 1:
            errors.append(f"{path.name}.processed_frames: expected positive integer")
        elif processed != len(frames):
            errors.append(
                f"{path.name}.processed_frames={processed} but predictions contain {len(frames)} frames"
            )
        if frames and (
            frames[0]["frame_idx"] != 1
            or not _is_int(processed)
            or frames[-1]["frame_idx"] != processed
        ):
            errors.append(
                f"{path.name}: prediction bounds must be 1..processed_frames"
            )
    if errors:
        raise EvaluationValidationError(errors)
    return data


def _event_value(event: Mapping[str, Any], key: str) -> Any:
    if key in event:
        return event[key]
    details = event.get("details")
    return details.get(key) if isinstance(details, dict) else None


def _build_canonicalizer(frames: Sequence[dict]):
    directed: MutableMapping[int, set] = defaultdict(set)

    def add_alias(alias: Any, canonical: Any) -> None:
        if alias is None or canonical is None:
            return
        alias_gid, canonical_gid = int(alias), int(canonical)
        directed.setdefault(alias_gid, set())
        directed.setdefault(canonical_gid, set())
        if alias_gid != canonical_gid:
            directed[alias_gid].add(canonical_gid)

    for frame in frames:
        for item in frame["gid_aliases"]:
            add_alias(item["alias_gid"], item["canonical_gid"])
        for observation in frame["observations"]:
            canonical = observation.get("canonical_gid")
            raw = observation.get("raw_gid")
            if canonical is not None:
                add_alias(canonical, canonical)
                add_alias(raw, canonical)
                for alias in observation.get("gid_aliases", []):
                    add_alias(alias, canonical)
        for slot in frame["slots"]:
            canonical = slot.get("canonical_vehicle_gid")
            raw = slot.get("raw_vehicle_gid")
            if canonical is not None:
                add_alias(canonical, canonical)
                add_alias(raw, canonical)
        for event in (*frame["identity_events"], *frame["parking_events"]):
            canonical = event.get("canonical_gid")
            raw = event.get("raw_gid")
            if canonical is not None:
                add_alias(canonical, canonical)
                add_alias(raw, canonical)
            if event.get("event_type") == "global_id_merged":
                superseded = _event_value(event, "superseded_global_id")
                if _is_int(superseded) and canonical is not None:
                    add_alias(superseded, canonical)

    resolved: Dict[int, int] = {}
    visiting: List[int] = []

    def resolve(value: int) -> int:
        if value in resolved:
            return resolved[value]
        if value in visiting:
            cycle = visiting[visiting.index(value):] + [value]
            raise EvaluationValidationError([
                "predictions.jsonl: directed GID alias cycle: "
                + " -> ".join(str(item) for item in cycle)
            ])
        visiting.append(value)
        roots = {resolve(target) for target in directed.get(value, set())}
        visiting.pop()
        if len(roots) > 1:
            raise EvaluationValidationError([
                f"predictions.jsonl: conflicting canonical targets for alias {value}: "
                + ", ".join(str(item) for item in sorted(roots))
            ])
        result = next(iter(roots)) if roots else value
        resolved[value] = result
        return result

    for gid in list(directed):
        resolve(gid)

    def canonicalize(value: Any) -> Optional[int]:
        if value is None:
            return None
        gid = int(value)
        return resolved.get(gid, gid)

    return canonicalize


def _maximum_weight_mapping(weights: Mapping[str, Counter]) -> Dict[str, Optional[int]]:
    """Return an exact one-to-one maximum-weight row/column assignment."""
    physical_ids = sorted(weights)
    gids = sorted({gid for counter in weights.values() for gid in counter})
    if not physical_ids:
        return {}
    if not gids:
        return {physical: None for physical in physical_ids}
    size = max(len(physical_ids), len(gids))
    max_weight = max((max(counter.values(), default=0) for counter in weights.values()), default=0)
    cost = [[float(max_weight) for _ in range(size)] for _ in range(size)]
    for row, physical in enumerate(physical_ids):
        for col, gid in enumerate(gids):
            cost[row][col] = float(max_weight - weights[physical].get(gid, 0))

    # Hungarian algorithm for a square minimization matrix.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        col0 = 0
        min_value = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float("inf")
            col1 = 0
            for col in range(1, size + 1):
                if used[col]:
                    continue
                current = cost[row0 - 1][col - 1] - u[row0] - v[col]
                if current < min_value[col]:
                    min_value[col] = current
                    way[col] = col0
                if min_value[col] < delta:
                    delta, col1 = min_value[col], col
            for col in range(size + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    min_value[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break
    assignment = [-1] * size
    for col in range(1, size + 1):
        if p[col]:
            assignment[p[col] - 1] = col - 1
    result: Dict[str, Optional[int]] = {}
    for row, physical in enumerate(physical_ids):
        col = assignment[row]
        gid = gids[col] if 0 <= col < len(gids) else None
        result[physical] = gid if gid is not None and weights[physical].get(gid, 0) > 0 else None
    return result


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification(total: float, scores: Mapping[str, Optional[float]], critical_count: int) -> str:
    if critical_count or total < 70.0:
        return "FAIL"
    core = [
        scores.get("identity_continuity_handoff"),
        scores.get("slot_identity_ownership"),
        scores.get("departure_recovery"),
    ]
    if total >= 85.0 and all(value is not None and value >= 95.0 for value in core):
        return "PASS"
    return "CONDITIONAL"


def _normalized_delay_score(delay: Optional[int], preferred: int, maximum: int) -> float:
    if delay is None:
        return 0.0
    if delay <= preferred:
        return 1.0
    if maximum <= preferred or delay >= maximum:
        return 0.0
    return (maximum - delay) / (maximum - preferred)


def _non_null_runs(timeline: Sequence[Tuple[int, Optional[int]]]) -> List[Tuple[int, int, int, int]]:
    """Return consecutive same-value runs, excluding null values."""
    runs: List[Tuple[int, int, int, int]] = []
    start = last = value = None
    for frame_idx, current in list(timeline) + [
        ((timeline[-1][0] + 2) if timeline else 0, None)
    ]:
        continues = (
            start is not None
            and current == value
            and last is not None
            and frame_idx == last + 1
        )
        if current is not None and start is None:
            start = last = frame_idx
            value = int(current)
        elif current is not None and continues:
            last = frame_idx
        else:
            if start is not None and last is not None and value is not None:
                runs.append((start, last, value, last - start + 1))
            if current is not None:
                start = last = frame_idx
                value = int(current)
            else:
                start = last = value = None
    return runs


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    occupied_precision = _safe_ratio(tp, tp + fp)
    occupied_recall = _safe_ratio(tp, tp + fn)
    occupied_f1 = _safe_ratio(2 * occupied_precision * occupied_recall, occupied_precision + occupied_recall)
    free_precision = _safe_ratio(tn, tn + fn)
    free_recall = _safe_ratio(tn, tn + fp)
    free_f1 = _safe_ratio(2 * free_precision * free_recall, free_precision + free_recall)
    balanced = (occupied_recall + free_recall) / 2.0
    return {
        "total_slot_frames": tp + fp + fn + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "occupied_precision": occupied_precision,
        "occupied_recall": occupied_recall,
        "occupied_f1": occupied_f1,
        "free_precision": free_precision,
        "free_recall": free_recall,
        "free_f1": free_f1,
        "false_free_rate": _safe_ratio(fn, tp + fn),
        "false_occupied_rate": _safe_ratio(fp, tn + fp),
        "balanced_accuracy": balanced,
        "occupancy_score": 100.0 * (0.60 * occupied_f1 + 0.40 * balanced),
    }


def _add_error(errors: List[dict], code: str, message: str, *, critical: bool = False, **context: Any) -> None:
    item = {"code": code, "severity": "critical" if critical else "error", "message": message}
    item.update({key: value for key, value in context.items() if value is not None})
    errors.append(item)


def _round_tree(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_tree(item) for item in value]
    return value


def evaluate_session(
    session_dir: Path | str,
    *,
    fps: float = 30.0,
    config: Optional[EvaluatorConfig] = None,
    write_outputs: bool = True,
) -> dict:
    """Validate and evaluate one schema-v3 experiment session."""
    session_dir = Path(session_dir)
    config = config or EvaluatorConfig()
    if fps <= 0:
        raise ValueError("fps must be > 0")
    gt_slots, gt_events, gt_identity = _load_ground_truth(session_dir)
    frames = _load_predictions(session_dir / "predictions.jsonl")
    session_info = _validate_session_info(session_dir / "session_info.json", frames)
    canonicalize = _build_canonicalizer(frames)
    frame_by_idx = {frame["frame_idx"]: frame for frame in frames}
    frame_indices = sorted(frame_by_idx)

    observations_by_frame: Dict[int, List[dict]] = {}
    slots_by_frame: Dict[int, Dict[Tuple[str, str], dict]] = {}
    reservations_by_frame: Dict[int, List[dict]] = {}
    all_identity_events: List[dict] = []
    all_parking_events: List[dict] = []
    visible_frames_by_gid: MutableMapping[int, set] = defaultdict(set)
    bound_gids = set()
    for frame in frames:
        frame_idx = frame["frame_idx"]
        normalized_observations = []
        for observation in frame["observations"]:
            item = dict(observation)
            item["canonical_gid"] = canonicalize(item.get("canonical_gid"))
            item["raw_gid"] = canonicalize(item.get("raw_gid"))
            normalized_observations.append(item)
            gid = item["canonical_gid"]
            if gid is not None and item.get("invisible_count") == 0:
                visible_frames_by_gid[gid].add(frame_idx)
            if gid is not None and item.get("slot_ownership") is not None:
                bound_gids.add(gid)
        observations_by_frame[frame_idx] = normalized_observations
        normalized_slots = {}
        for slot in frame["slots"]:
            item = dict(slot)
            item["canonical_vehicle_gid"] = canonicalize(item.get("canonical_vehicle_gid"))
            item["recovery_global_id"] = canonicalize(item.get("recovery_global_id"))
            normalized_slots[(item["camera_id"], item["slot_id"])] = item
            if item["canonical_vehicle_gid"] is not None:
                bound_gids.add(item["canonical_vehicle_gid"])
        slots_by_frame[frame_idx] = normalized_slots
        normalized_reservations = []
        for raw_reservation in frame["parked_identity_reservations"]:
            reservation = dict(raw_reservation)
            reservation["canonical_gid"] = canonicalize(
                reservation.get("canonical_gid")
            )
            normalized_reservations.append(reservation)
            if reservation["canonical_gid"] is not None:
                bound_gids.add(reservation["canonical_gid"])
        reservations_by_frame[frame_idx] = normalized_reservations
        for target, source_items in (
            (all_identity_events, frame["identity_events"]),
            (all_parking_events, frame["parking_events"]),
        ):
            for raw_event in source_items:
                event = dict(raw_event)
                event["canonical_gid"] = canonicalize(event.get("canonical_gid"))
                event["raw_gid"] = canonicalize(event.get("raw_gid"))
                target.append(event)

    phantom_gids = {
        gid for gid, visible_frames in visible_frames_by_gid.items()
        if len(visible_frames) < config.phantom_max_frames and gid not in bound_gids
    }
    long_unbound_gids = {
        gid: {
            "visible_frame_count": len(visible_frames),
            "first_frame": min(visible_frames),
            "last_frame": max(visible_frames),
        }
        for gid, visible_frames in visible_frames_by_gid.items()
        if len(visible_frames) >= config.phantom_max_frames and gid not in bound_gids
    }

    # Legacy sessions may omit a configured slot only during a short prefix
    # warm-up.  Once a slot appears, every later labeled frame must contain it.
    # Allowed warm-up samples are excluded, never interpreted as "free".
    missing_labeled_slot_frames = 0
    slot_coverage_errors = []
    labeled_keys = sorted({
        (interval["camera_id"], interval["slot_id"]) for interval in gt_slots
    })
    for key in labeled_keys:
        present_frames = [
            frame_idx for frame_idx in frame_indices if key in slots_by_frame[frame_idx]
        ]
        if not present_frames:
            slot_coverage_errors.append(
                f"predictions.jsonl: labeled slot {key[0]}/{key[1]} never appears"
            )
            continue
        first_present = present_frames[0]
        if first_present - 1 > 10:
            slot_coverage_errors.append(
                f"predictions.jsonl: slot {key[0]}/{key[1]} first appears at frame "
                f"{first_present}; prefix warm-up exceeds 10 frames"
            )
        for frame_idx in frame_indices:
            if frame_idx >= first_present and key not in slots_by_frame[frame_idx]:
                slot_coverage_errors.append(
                    f"predictions frame {frame_idx}: labeled slot {key[0]}/{key[1]} "
                    "is missing after warm-up"
                )
                if len(slot_coverage_errors) >= 25:
                    break
        if len(slot_coverage_errors) >= 25:
            break
        for interval in (
            item for item in gt_slots
            if (item["camera_id"], item["slot_id"]) == key
        ):
            for frame_idx in frame_indices:
                if not interval["start_frame"] <= frame_idx <= interval["end_frame"]:
                    continue
                if key in slots_by_frame[frame_idx]:
                    continue
                if frame_idx < first_present and frame_idx <= 10:
                    missing_labeled_slot_frames += 1
    if slot_coverage_errors:
        raise EvaluationValidationError(slot_coverage_errors)
    checkpoint_frame_errors = [
        f"ground_truth_identity observation {item['observation_id']!r}: "
        f"frame {item['frame_idx']} is absent from predictions"
        for item in gt_identity
        if item["frame_idx"] not in frame_by_idx
    ]
    if checkpoint_frame_errors:
        raise EvaluationValidationError(checkpoint_frame_errors)

    def checkpoint_gid(checkpoint: dict) -> Tuple[Optional[int], Optional[dict], Optional[float]]:
        frame_idx = checkpoint["frame_idx"]
        if frame_idx not in frame_by_idx:
            return None, None, None
        if checkpoint["slot_id"]:
            slot = slots_by_frame[frame_idx].get((checkpoint["camera_id"], checkpoint["slot_id"]))
            return (
                slot.get("canonical_vehicle_gid") if slot else None,
                slot,
                0.0 if slot else None,
            )
        candidates = []
        for observation in observations_by_frame[frame_idx]:
            if observation["camera_id"] != checkpoint["camera_id"] or observation.get("invisible_count") != 0:
                continue
            anchor = observation["anchor_pixel"]
            distance = math.hypot(
                float(anchor["x"]) - float(checkpoint["anchor_x"]),
                float(anchor["y"]) - float(checkpoint["anchor_y"]),
            )
            if distance <= config.identity_anchor_max_distance_px:
                candidates.append((distance, observation))
        if not candidates:
            return None, None, None
        distance, observation = min(candidates, key=lambda item: item[0])
        gid = observation.get("canonical_gid")
        return (None if gid in phantom_gids else gid), observation, distance

    checkpoint_results = []
    # Parked checkpoints are evaluated only after mapping.  Mapping/identity collision
    # evidence must come from required, independently anchored observations;
    # otherwise a wrong slot owner could name itself as the expected identity.
    mapping_evidence: MutableMapping[str, Counter] = defaultdict(Counter)
    checkpoint_claims: MutableMapping[str, Counter] = defaultdict(Counter)
    checkpoint_claim_evidence: MutableMapping[int, MutableMapping[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scored_physical_ids = {
        item["physical_vehicle_id"] for item in gt_identity if item["required"]
    } | {
        item["physical_vehicle_id"] for item in gt_slots if item["identity_required"]
    }
    for physical in scored_physical_ids:
        mapping_evidence.setdefault(physical, Counter())
        checkpoint_claims.setdefault(physical, Counter())
    for checkpoint in gt_identity:
        gid, matched, distance = checkpoint_gid(checkpoint)
        result = dict(checkpoint)
        result.update({
            "predicted_gid": gid,
            "matched_observation_uid": matched.get("observation_uid") if isinstance(matched, dict) else None,
            "distance_px": distance,
        })
        checkpoint_results.append(result)
        independent = checkpoint["required"] and not checkpoint["slot_id"]
        if gid is not None and independent:
            checkpoint_claims[checkpoint["physical_vehicle_id"]][gid] += 1
            mapping_evidence[checkpoint["physical_vehicle_id"]][gid] += 1
            checkpoint_claim_evidence[gid][checkpoint["physical_vehicle_id"]].append({
                "observation_id": checkpoint["observation_id"],
                "frame_idx": checkpoint["frame_idx"],
                "camera_id": checkpoint["camera_id"],
            })

    physical_to_gid = _maximum_weight_mapping(mapping_evidence)
    errors: List[dict] = []
    for gid, diagnostic in sorted(long_unbound_gids.items()):
        _add_error(
            errors, "long_unbound_gid",
            f"Canonical GID {gid} remained visible for {diagnostic['visible_frame_count']} "
            "frames without ever owning or reserving a slot",
            frame_idx=diagnostic["first_frame"], end_frame=diagnostic["last_frame"],
            predicted_gid=gid, duration_frames=diagnostic["visible_frame_count"],
        )
    for physical, gid in sorted(physical_to_gid.items()):
        if gid is None:
            _add_error(
                errors, "unmapped_physical_vehicle",
                f"{physical} has no required independent checkpoint matched to a valid GID",
                physical_vehicle_id=physical,
            )

    # One canonical GID observed at checkpoints belonging to different physical
    # vehicles is always an identity collision, regardless of assignment outcome.
    gid_to_physical = defaultdict(set)
    for physical, counter in checkpoint_claims.items():
        for gid, count in counter.items():
            if count:
                gid_to_physical[gid].add(physical)
    for gid, physicals in sorted(gid_to_physical.items()):
        if len(physicals) > 1:
            evidence = {
                physical: sorted(
                    checkpoint_claim_evidence[gid][physical],
                    key=lambda item: (item["frame_idx"], item["camera_id"]),
                )
                for physical in sorted(physicals)
            }
            evidence_frames = sorted({
                item["frame_idx"]
                for items in evidence.values()
                for item in items
            })
            evidence_cameras = sorted({
                item["camera_id"]
                for items in evidence.values()
                for item in items
            })
            _add_error(
                errors, "gid_shared_between_vehicles",
                f"Canonical GID {gid} is observed for multiple physical vehicles",
                critical=True, predicted_gid=gid,
                physical_vehicle_ids=sorted(physicals),
                frame_idx=evidence_frames[0] if evidence_frames else None,
                end_frame=(
                    evidence_frames[-1]
                    if len(evidence_frames) > 1 else None
                ),
                camera_ids=evidence_cameras,
                checkpoint_evidence=evidence,
            )

    per_vehicle_checkpoints: MutableMapping[str, List[dict]] = defaultdict(list)
    for result in checkpoint_results:
        physical = result["physical_vehicle_id"]
        expected = physical_to_gid.get(physical)
        predicted = result["predicted_gid"]
        required = result["required"]
        correct = predicted is not None and expected is not None and predicted == expected
        result["expected_gid"] = expected
        result["correct"] = correct
        per_vehicle_checkpoints[physical].append(result)
        if required and predicted is not None and expected is not None and predicted != expected:
            _add_error(
                errors, "wrong_gid_at_checkpoint",
                f"{physical} has GID {predicted}, expected canonical GID {expected}",
                critical=True, physical_vehicle_id=physical,
                frame_idx=result["frame_idx"], camera_id=result["camera_id"],
                slot_id=result["slot_id"], expected_gid=expected,
                predicted_gid=predicted,
            )
        elif required and predicted is None:
            _add_error(
                errors, "missing_gid_at_checkpoint",
                f"{physical} has no valid Global ID at a required checkpoint",
                physical_vehicle_id=physical, frame_idx=result["frame_idx"],
                camera_id=result["camera_id"], slot_id=result["slot_id"],
                expected_gid=expected,
            )

    handoff_results = []
    handoffs_by_vehicle: MutableMapping[str, List[dict]] = defaultdict(list)
    for event in gt_events:
        if event["event_type"] != "camera_handoff" or not event["required"]:
            continue
        expected = physical_to_gid.get(event["physical_vehicle_id"])
        maximum = event["max_delay_frames"]
        deadline = max(event["end_frame"], event["start_frame"] + maximum)
        found_frame = None
        if expected is not None:
            for frame_idx in frame_indices:
                if not event["start_frame"] <= frame_idx <= deadline:
                    continue
                if any(
                    observation["camera_id"] == event["target_camera"]
                    and observation.get("invisible_count") == 0
                    and observation.get("canonical_gid") == expected
                    for observation in observations_by_frame[frame_idx]
                ):
                    found_frame = frame_idx
                    break
            if found_frame is None:
                for prediction_event in all_identity_events:
                    if not event["start_frame"] <= prediction_event["frame_idx"] <= deadline:
                        continue
                    if (
                        prediction_event.get("canonical_gid") == expected
                        and _event_value(prediction_event, "source_camera") == event["source_camera"]
                        and _event_value(prediction_event, "target_camera") == event["target_camera"]
                        and "handoff" in prediction_event.get("event_type", "")
                    ):
                        found_frame = prediction_event["frame_idx"]
                        break
        delay = found_frame - event["start_frame"] if found_frame is not None else None
        result = {
            "event_id": event["event_id"], "physical_vehicle_id": event["physical_vehicle_id"],
            "source_camera": event["source_camera"], "target_camera": event["target_camera"],
            "expected_gid": expected, "matched_frame": found_frame, "delay_frames": delay,
            "success": found_frame is not None,
            "delay_score": _normalized_delay_score(
                delay, event["preferred_delay_frames"], maximum
            ),
        }
        handoff_results.append(result)
        handoffs_by_vehicle[event["physical_vehicle_id"]].append(result)
        if found_frame is None:
            _add_error(
                errors, "missed_camera_handoff",
                f"No canonical GID {expected} observation appeared in {event['target_camera']} "
                f"within the handoff deadline",
                physical_vehicle_id=event["physical_vehicle_id"],
                frame_idx=event["start_frame"], camera_id=event["target_camera"],
                expected_gid=expected,
            )

    identity_vehicle_results = []
    identity_vehicle_scores = []
    all_identity_vehicles = sorted(set(per_vehicle_checkpoints) | set(handoffs_by_vehicle))
    for physical in all_identity_vehicles:
        required_checkpoints = [item for item in per_vehicle_checkpoints[physical] if item["required"]]
        checkpoint_score = (
            _safe_ratio(sum(bool(item["correct"]) for item in required_checkpoints), len(required_checkpoints))
            if required_checkpoints else None
        )
        vehicle_handoffs = handoffs_by_vehicle[physical]
        handoff_score = (
            _safe_ratio(sum(bool(item["success"]) for item in vehicle_handoffs), len(vehicle_handoffs))
            if vehicle_handoffs else None
        )
        if checkpoint_score is not None and handoff_score is not None:
            score = 100.0 * (0.70 * checkpoint_score + 0.30 * handoff_score)
        elif checkpoint_score is not None:
            score = 100.0 * checkpoint_score
        elif handoff_score is not None:
            score = 100.0 * handoff_score
        else:
            continue
        identity_vehicle_scores.append(score)
        identity_vehicle_results.append({
            "physical_vehicle_id": physical,
            "expected_gid": physical_to_gid.get(physical),
            "required_checkpoints": len(required_checkpoints),
            "correct_checkpoints": sum(bool(item["correct"]) for item in required_checkpoints),
            "required_handoffs": len(vehicle_handoffs),
            "correct_handoffs": sum(bool(item["success"]) for item in vehicle_handoffs),
            "score": score,
        })
    identity_score = statistics.mean(identity_vehicle_scores) if identity_vehicle_scores else None

    # Slot ownership is averaged by parking lifecycle instead of by frame, so a
    # long static interval cannot hide one short but dangerous wrong-owner event.
    slot_lifecycles = []
    binding_delay_items = []
    for interval in gt_slots:
        if not interval["identity_required"]:
            continue
        physical = interval["physical_vehicle_id"]
        expected = physical_to_gid.get(physical)
        timeline = []
        for frame_idx in frame_indices:
            if interval["start_frame"] <= frame_idx <= interval["end_frame"]:
                slot_prediction = slots_by_frame[frame_idx].get(
                    (interval["camera_id"], interval["slot_id"])
                )
                if slot_prediction is not None:
                    timeline.append((frame_idx, slot_prediction["canonical_vehicle_gid"]))
        first_correct = next(
            (frame_idx for frame_idx, owner in timeline if expected is not None and owner == expected),
            None,
        )
        delay = first_correct - interval["start_frame"] if first_correct is not None else None
        if delay is None or delay > config.slot_binding_max_frames:
            _add_error(
                errors, "missed_slot_binding",
                f"{interval['camera_id']}/{interval['slot_id']} did not bind canonical "
                f"GID {expected} within {config.slot_binding_max_frames} frames",
                physical_vehicle_id=physical, frame_idx=interval["start_frame"],
                camera_id=interval["camera_id"], slot_id=interval["slot_id"],
                expected_gid=expected,
            )
        binding_delay_items.append({
            "kind": "slot_binding", "physical_vehicle_id": physical,
            "frame_idx": interval["start_frame"], "delay_frames": delay,
            "preferred_frames": config.slot_binding_preferred_frames,
            "max_frames": config.slot_binding_max_frames,
        })
        evaluation_start = min(interval["end_frame"], interval["start_frame"] + config.slot_binding_max_frames)
        steady = [(frame, owner) for frame, owner in timeline if frame >= evaluation_start]
        if not steady and timeline:
            steady = [timeline[-1]]
        steady_score = _safe_ratio(
            sum(expected is not None and owner == expected for _, owner in steady), len(steady)
        )
        binding_score = float(delay is not None and delay <= config.slot_binding_max_frames)
        lifecycle_score = 100.0 * (0.80 * steady_score + 0.20 * binding_score)

        wrong_runs = []
        run_start = None
        run_gid = None
        run_last = None
        for frame_idx, owner in timeline + [(interval["end_frame"] + 2, None)]:
            wrong = expected is not None and owner is not None and owner != expected
            consecutive = run_last is not None and frame_idx == run_last + 1 and owner == run_gid
            if wrong and run_start is None:
                run_start, run_last, run_gid = frame_idx, frame_idx, owner
            elif wrong and consecutive:
                run_last = frame_idx
            else:
                if run_start is not None:
                    length = run_last - run_start + 1
                    wrong_runs.append((run_start, run_last, run_gid, length))
                if wrong:
                    run_start, run_last, run_gid = frame_idx, frame_idx, owner
                else:
                    run_start = run_last = run_gid = None
        for start, end, wrong_gid, length in wrong_runs:
            _add_error(
                errors,
                "wrong_slot_owner" if length >= config.wrong_owner_critical_frames else "transient_wrong_slot_owner",
                f"{interval['camera_id']}/{interval['slot_id']} stored GID {wrong_gid} "
                f"instead of {expected} for {length} consecutive frames",
                critical=length >= config.wrong_owner_critical_frames,
                physical_vehicle_id=physical, frame_idx=start,
                end_frame=end, camera_id=interval["camera_id"], slot_id=interval["slot_id"],
                expected_gid=expected, predicted_gid=wrong_gid,
                duration_frames=length,
            )

        reservation_timeline = []
        for frame_idx, _ in timeline:
            matching = [
                reservation for reservation in reservations_by_frame[frame_idx]
                if reservation["camera_id"] == interval["camera_id"]
                and reservation["slot_id"] == interval["slot_id"]
            ]
            reservation_timeline.append((
                frame_idx,
                matching[0]["canonical_gid"] if matching else None,
            ))
        reservation_runs = []
        reservation_start = reservation_last = reservation_gid = None
        for frame_idx, owner in reservation_timeline + [(interval["end_frame"] + 2, None)]:
            wrong = expected is not None and owner is not None and owner != expected
            consecutive = (
                reservation_last is not None
                and frame_idx == reservation_last + 1
                and owner == reservation_gid
            )
            if wrong and reservation_start is None:
                reservation_start = reservation_last = frame_idx
                reservation_gid = owner
            elif wrong and consecutive:
                reservation_last = frame_idx
            else:
                if reservation_start is not None:
                    reservation_runs.append((
                        reservation_start, reservation_last, reservation_gid,
                        reservation_last - reservation_start + 1,
                    ))
                if wrong:
                    reservation_start = reservation_last = frame_idx
                    reservation_gid = owner
                else:
                    reservation_start = reservation_last = reservation_gid = None
        for start, end, wrong_gid, length in reservation_runs:
            _add_error(
                errors,
                "wrong_slot_reservation" if length >= config.wrong_owner_critical_frames else "transient_wrong_slot_reservation",
                f"Reservation for {interval['camera_id']}/{interval['slot_id']} held GID "
                f"{wrong_gid} instead of {expected} for {length} frames",
                critical=length >= config.wrong_owner_critical_frames,
                physical_vehicle_id=physical, frame_idx=start, end_frame=end,
                camera_id=interval["camera_id"], slot_id=interval["slot_id"],
                expected_gid=expected, predicted_gid=wrong_gid,
                duration_frames=length,
            )
        slot_lifecycles.append({
            "physical_vehicle_id": physical, "camera_id": interval["camera_id"],
            "slot_id": interval["slot_id"], "start_frame": interval["start_frame"],
            "end_frame": interval["end_frame"], "expected_gid": expected,
            "first_correct_owner_frame": first_correct, "binding_delay_frames": delay,
            "steady_correct_frames": sum(expected is not None and owner == expected for _, owner in steady),
            "steady_total_frames": len(steady), "score": lifecycle_score,
        })
    slot_score = statistics.mean(item["score"] for item in slot_lifecycles) if slot_lifecycles else None

    # A free GT slot must not silently become owned by an unrelated identity.
    # The just-departed vehicle may leave its own GID stale briefly; that is a
    # release-delay error, not a wrong-identity critical error.
    intervals_by_key: MutableMapping[Tuple[str, str], List[dict]] = defaultdict(list)
    for interval in gt_slots:
        intervals_by_key[(interval["camera_id"], interval["slot_id"])].append(interval)
    release_delay_items = []
    free_slot_binding_runs = []
    for key, raw_intervals in intervals_by_key.items():
        ordered_intervals = sorted(raw_intervals, key=lambda item: item["start_frame"])
        for interval_index, interval in enumerate(ordered_intervals):
            if interval["occupied"]:
                continue
            previous = ordered_intervals[interval_index - 1] if interval_index else None
            previous_physical = None
            previous_expected = None
            if previous is not None and previous["occupied"] and previous["physical_vehicle_id"]:
                previous_physical = previous["physical_vehicle_id"]
                previous_expected = physical_to_gid.get(previous_physical)

            owner_timeline = []
            reservation_sets = []
            for frame_idx in frame_indices:
                if not interval["start_frame"] <= frame_idx <= interval["end_frame"]:
                    continue
                slot = slots_by_frame[frame_idx].get(key)
                if slot is None:
                    continue
                owner_timeline.append((frame_idx, slot["canonical_vehicle_gid"]))
                reservation_sets.append((
                    frame_idx,
                    {
                        reservation["canonical_gid"]
                        for reservation in reservations_by_frame[frame_idx]
                        if reservation["camera_id"] == key[0]
                        and reservation["slot_id"] == key[1]
                        and reservation["canonical_gid"] is not None
                    },
                ))

            if previous_physical is not None:
                clear_confirmed = None
                clear_streak = 0
                for frame_idx, owner in owner_timeline:
                    if owner is None:
                        clear_streak += 1
                        if clear_streak >= config.sustained_state_frames:
                            clear_confirmed = frame_idx
                            break
                    else:
                        clear_streak = 0
                delay = (
                    clear_confirmed - interval["start_frame"]
                    if clear_confirmed is not None else None
                )
                release_delay_items.append({
                    "kind": "slot_release",
                    "physical_vehicle_id": previous_physical,
                    "camera_id": key[0], "slot_id": key[1],
                    "frame_idx": interval["start_frame"], "delay_frames": delay,
                    "preferred_frames": 0,
                    "max_frames": config.slot_binding_max_frames,
                })

            def report_free_run(kind: str, start: int, end: int, gid: int, length: int) -> None:
                if length < config.wrong_owner_critical_frames:
                    return
                stale = previous_expected is not None and gid == previous_expected
                if stale:
                    code = "stale_release_binding" if kind == "owner" else "stale_release_reservation"
                    _add_error(
                        errors, code,
                        f"Free slot {key[0]}/{key[1]} retained departed GID {gid} "
                        f"as {kind} for {length} frames",
                        physical_vehicle_id=previous_physical, frame_idx=start,
                        end_frame=end, camera_id=key[0], slot_id=key[1],
                        predicted_gid=gid, duration_frames=length,
                    )
                elif previous_physical is not None and previous_expected is None:
                    _add_error(
                        errors, "binding_on_unmapped_free_slot",
                        f"Free slot {key[0]}/{key[1]} retained GID {gid}, but the "
                        "departed physical vehicle is unmapped",
                        physical_vehicle_id=previous_physical, frame_idx=start,
                        end_frame=end, camera_id=key[0], slot_id=key[1],
                        predicted_gid=gid, duration_frames=length,
                    )
                else:
                    code = "binding_on_gt_free_slot" if kind == "owner" else "reservation_on_gt_free_slot"
                    _add_error(
                        errors, code,
                        f"GT-free slot {key[0]}/{key[1]} held unrelated GID {gid} "
                        f"as {kind} for {length} frames",
                        critical=True, physical_vehicle_id=previous_physical,
                        frame_idx=start, end_frame=end, camera_id=key[0],
                        slot_id=key[1], expected_gid=previous_expected,
                        predicted_gid=gid, duration_frames=length,
                    )
                free_slot_binding_runs.append({
                    "kind": kind, "camera_id": key[0], "slot_id": key[1],
                    "start_frame": start, "end_frame": end,
                    "canonical_gid": gid, "duration_frames": length,
                    "stale_departed_gid": stale,
                })

            for start, end, gid, length in _non_null_runs(owner_timeline):
                report_free_run("owner", start, end, gid, length)

            reservation_gids = sorted({
                gid for _, gids in reservation_sets for gid in gids
            })
            for gid in reservation_gids:
                timeline = [
                    (frame_idx, gid if gid in gids else None)
                    for frame_idx, gids in reservation_sets
                ]
                for start, end, _, length in _non_null_runs(timeline):
                    report_free_run("reservation", start, end, gid, length)

    # Different canonical identities reserving one slot simultaneously is
    # invalid regardless of the slot's occupied/free GT state.
    duplicate_reservation_runs = []
    for key in intervals_by_key:
        start = last = None
        gids_seen = set()
        for frame_idx in frame_indices + [frame_indices[-1] + 2]:
            current = {
                reservation["canonical_gid"]
                for reservation in reservations_by_frame.get(frame_idx, [])
                if reservation["camera_id"] == key[0]
                and reservation["slot_id"] == key[1]
                and reservation["canonical_gid"] is not None
            }
            duplicated = len(current) >= 2
            if duplicated and start is None:
                start = last = frame_idx
                gids_seen = set(current)
            elif duplicated and last is not None and frame_idx == last + 1:
                last = frame_idx
                gids_seen.update(current)
            else:
                if start is not None and last is not None:
                    length = last - start + 1
                    if length >= config.wrong_owner_critical_frames:
                        _add_error(
                            errors, "duplicate_slot_reservations",
                            f"{key[0]}/{key[1]} had different-GID reservations "
                            f"{sorted(gids_seen)} for {length} frames",
                            critical=True, frame_idx=start, end_frame=last,
                            camera_id=key[0], slot_id=key[1],
                            predicted_gids=sorted(gids_seen), duration_frames=length,
                        )
                        duplicate_reservation_runs.append({
                            "camera_id": key[0], "slot_id": key[1],
                            "start_frame": start, "end_frame": last,
                            "canonical_gids": sorted(gids_seen),
                            "duration_frames": length,
                        })
                if duplicated:
                    start = last = frame_idx
                    gids_seen = set(current)
                else:
                    start = last = None
                    gids_seen = set()

    # Select departure_started as the authoritative recovery event.  A slot_leave
    # event is used only when that lifecycle has no nearby departure_started row.
    departures = [
        event for event in gt_events
        if event["required"]
        and event["event_type"] == "departure_started"
        and event["source_camera"]
        and event["source_slot_id"]
    ]
    for candidate in (
        event for event in gt_events
        if event["required"]
        and event["event_type"] == "slot_leave"
        and event["source_camera"]
        and event["source_slot_id"]
    ):
        if not any(
            event["physical_vehicle_id"] == candidate["physical_vehicle_id"]
            and event.get("source_slot_id") == candidate.get("source_slot_id")
            and abs(event["start_frame"] - candidate["start_frame"]) <= config.recovery_max_frames
            for event in departures
        ):
            departures.append(candidate)

    recovery_results = []
    recovery_delay_items = []
    for event in departures:
        physical = event["physical_vehicle_id"]
        expected = physical_to_gid.get(physical)
        preferred = event["preferred_delay_frames"]
        maximum = event["max_delay_frames"]
        deadline = max(event["end_frame"], event["start_frame"] + maximum)
        target_cameras = {event["target_camera"]} if event["target_camera"] else None
        cleared_frame = None
        source_key = (
            (event["source_camera"], event["source_slot_id"])
            if event["source_camera"] and event["source_slot_id"]
            else None
        )
        clear_run_start_frame = None
        if source_key is not None:
            clear_streak = 0
            for frame_idx in frame_indices:
                if not event["start_frame"] <= frame_idx <= deadline:
                    continue
                slot = slots_by_frame[frame_idx].get(source_key)
                if slot is not None and slot["canonical_vehicle_gid"] is None:
                    clear_streak += 1
                    if clear_streak == 1:
                        clear_run_start_frame = frame_idx
                    if clear_streak >= config.sustained_state_frames:
                        cleared_frame = frame_idx
                        break
                else:
                    clear_streak = 0
                    clear_run_start_frame = None

        recovery_event_frame = None
        for prediction_event in all_parking_events:
            if prediction_event.get("event_type") != "parked_id_recovered":
                continue
            if not event["start_frame"] <= prediction_event["frame_idx"] <= deadline:
                continue
            if cleared_frame is None or prediction_event["frame_idx"] < cleared_frame:
                continue
            predicted_slot = _event_value(prediction_event, "slot_id")
            predicted_camera = prediction_event.get("camera_id") or _event_value(prediction_event, "camera_id")
            same_slot = not event["source_slot_id"] or predicted_slot == event["source_slot_id"]
            same_camera = not event["source_camera"] or not predicted_camera or predicted_camera == event["source_camera"]
            predicted_gid = prediction_event.get("canonical_gid")
            if same_slot and same_camera and expected is not None and predicted_gid == expected:
                recovery_event_frame = prediction_event["frame_idx"]
                break

        recovered_frame = recovery_event_frame
        if expected is not None and cleared_frame is not None:
            for frame_idx in frame_indices:
                if not event["start_frame"] <= frame_idx <= deadline:
                    continue
                if frame_idx < cleared_frame:
                    continue
                if recovered_frame is not None and frame_idx >= recovered_frame:
                    break
                for observation in observations_by_frame[frame_idx]:
                    if (
                        observation.get("invisible_count") != 0
                        or observation.get("canonical_gid") != expected
                        or (target_cameras is not None and observation["camera_id"] not in target_cameras)
                    ):
                        continue
                    cross_camera = bool(
                        event["source_camera"]
                        and observation["camera_id"] != event["source_camera"]
                    )
                    if cross_camera:
                        recovered_frame = frame_idx
                        break
                    ownership = observation.get("slot_ownership")
                    still_bound_source = bool(
                        isinstance(ownership, dict)
                        and ownership.get("camera_id") == source_key[0]
                        and ownership.get("slot_id") == source_key[1]
                    )
                    # A still-parked local track is not departure recovery.  On
                    # the same camera, require the observation to have released
                    # the old binding after the source slot's sustained clear.
                    if not still_bound_source:
                        recovered_frame = frame_idx
                        break
                if recovered_frame == frame_idx:
                    break
        recovered_delay = recovered_frame - event["start_frame"] if recovered_frame is not None else None

        if source_key is not None:
            wrong_run_start = wrong_run_last = wrong_run_gid = None
            for frame_idx in [idx for idx in frame_indices if event["start_frame"] <= idx <= deadline] + [deadline + 2]:
                slot = slots_by_frame.get(frame_idx, {}).get(source_key)
                owner = slot.get("canonical_vehicle_gid") if slot else None
                wrong = expected is not None and owner is not None and owner != expected
                consecutive = wrong_run_last is not None and frame_idx == wrong_run_last + 1 and owner == wrong_run_gid
                if wrong and wrong_run_start is None:
                    wrong_run_start = wrong_run_last = frame_idx
                    wrong_run_gid = owner
                elif wrong and consecutive:
                    wrong_run_last = frame_idx
                else:
                    if wrong_run_start is not None:
                        length = wrong_run_last - wrong_run_start + 1
                        if length >= config.wrong_owner_critical_frames:
                            _add_error(
                                errors, "wrong_departure_slot_owner",
                                f"Departed slot {source_key[0]}/{source_key[1]} was rebound to GID "
                                f"{wrong_run_gid} for {length} frames",
                                critical=True, physical_vehicle_id=physical,
                                frame_idx=wrong_run_start, end_frame=wrong_run_last,
                                camera_id=source_key[0], slot_id=source_key[1],
                                expected_gid=expected, predicted_gid=wrong_run_gid,
                                duration_frames=length,
                            )
                    if wrong:
                        wrong_run_start = wrong_run_last = frame_idx
                        wrong_run_gid = owner
                    else:
                        wrong_run_start = wrong_run_last = wrong_run_gid = None

        for prediction_event in all_parking_events:
            if prediction_event.get("event_type") != "parked_id_recovered":
                continue
            if not event["start_frame"] <= prediction_event["frame_idx"] <= deadline:
                continue
            predicted_slot = _event_value(prediction_event, "slot_id")
            predicted_camera = prediction_event.get("camera_id") or _event_value(prediction_event, "camera_id")
            same_slot = not event["source_slot_id"] or predicted_slot == event["source_slot_id"]
            same_camera = not event["source_camera"] or not predicted_camera or predicted_camera == event["source_camera"]
            predicted_gid = prediction_event.get("canonical_gid")
            if same_slot and same_camera and expected is not None and predicted_gid is not None and predicted_gid != expected:
                _add_error(
                    errors, "wrong_departure_recovery_gid",
                    f"Departure from {event['source_slot_id']} recovered GID {predicted_gid}, expected {expected}",
                    critical=True, physical_vehicle_id=physical,
                    frame_idx=prediction_event["frame_idx"], camera_id=event["source_camera"],
                    slot_id=event["source_slot_id"], expected_gid=expected,
                    predicted_gid=predicted_gid,
                )

        recovery_ok = recovered_frame is not None
        clear_ok = source_key is None or cleared_frame is not None
        if not recovery_ok:
            _add_error(
                errors, "missed_departure_recovery",
                f"{physical} did not recover canonical GID {expected} within {maximum} frames",
                physical_vehicle_id=physical, frame_idx=event["start_frame"],
                camera_id=event["target_camera"] or event["source_camera"],
                slot_id=event["source_slot_id"], expected_gid=expected,
            )
        if not clear_ok:
            _add_error(
                errors, "departure_slot_not_cleared",
                f"{event['source_camera']}/{event['source_slot_id']} did not clear its owner "
                f"within {maximum} frames",
                physical_vehicle_id=physical, frame_idx=event["start_frame"],
                camera_id=event["source_camera"], slot_id=event["source_slot_id"],
                expected_gid=expected,
            )
        score = 100.0 * ((0.80 if recovery_ok else 0.0) + (0.20 if clear_ok else 0.0))
        result = {
            "event_id": event["event_id"], "physical_vehicle_id": physical,
            "source_camera": event["source_camera"], "source_slot_id": event["source_slot_id"],
            "target_camera": event["target_camera"], "expected_gid": expected,
            "recovered_frame": recovered_frame, "recovery_delay_frames": recovered_delay,
            "slot_clear_run_start_frame": clear_run_start_frame,
            "slot_cleared_frame": cleared_frame, "recovered_correct_gid": recovery_ok,
            "slot_cleared": clear_ok, "score": score,
        }
        recovery_results.append(result)
        recovery_delay_items.append({
            "kind": "departure_recovery", "physical_vehicle_id": physical,
            "frame_idx": event["start_frame"], "delay_frames": recovered_delay,
            "preferred_frames": preferred, "max_frames": maximum,
        })
    recovery_score = statistics.mean(item["score"] for item in recovery_results) if recovery_results else None

    # Occupancy raw and practical timelines.
    labeled_by_key: MutableMapping[Tuple[str, str], List[dict]] = defaultdict(list)
    for interval in gt_slots:
        labeled_by_key[(interval["camera_id"], interval["slot_id"])].append(interval)
    occupancy_samples: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    raw_counts = [0, 0, 0, 0]  # tp, fp, fn, tn
    for frame_idx in frame_indices:
        for key, intervals in labeled_by_key.items():
            interval = next(
                (item for item in intervals if item["start_frame"] <= frame_idx <= item["end_frame"]),
                None,
            )
            if interval is None:
                continue
            slot = slots_by_frame[frame_idx].get(key)
            if slot is None:
                continue
            gt_occupied = interval["occupied"]
            predicted = slot["occupied"]
            if gt_occupied and predicted:
                raw_counts[0] += 1
            elif not gt_occupied and predicted:
                raw_counts[1] += 1
            elif gt_occupied and not predicted:
                raw_counts[2] += 1
            else:
                raw_counts[3] += 1
            occupancy_samples[key].append({
                "frame_idx": frame_idx, "gt": gt_occupied, "raw": predicted,
                "practical": predicted, "owner": slot["canonical_vehicle_gid"],
            })

    ignored_flickers = []
    harmful_flickers = []
    for key, samples in occupancy_samples.items():
        segment_start = 0
        while segment_start < len(samples):
            segment_end = segment_start + 1
            while (
                segment_end < len(samples)
                and samples[segment_end]["frame_idx"] == samples[segment_end - 1]["frame_idx"] + 1
            ):
                segment_end += 1
            segment = samples[segment_start:segment_end]
            runs = []
            run_start = 0
            for index in range(1, len(segment) + 1):
                if index == len(segment) or segment[index]["raw"] != segment[run_start]["raw"]:
                    runs.append((run_start, index))
                    run_start = index
            for run_index, (start, end) in enumerate(runs):
                if run_index == 0 or run_index == len(runs) - 1:
                    continue
                before_start, before_end = runs[run_index - 1]
                after_start, after_end = runs[run_index + 1]
                surrounding = segment[before_start]["raw"]
                if surrounding != segment[after_start]["raw"] or segment[start]["raw"] == surrounding:
                    continue
                length_frames = segment[end - 1]["frame_idx"] - segment[start]["frame_idx"] + 1
                owner_values = {
                    item["owner"] for item in segment[before_end - 1:end]
                } | {segment[after_start]["owner"]}
                context = {
                    "camera_id": key[0], "slot_id": key[1],
                    "start_frame": segment[start]["frame_idx"],
                    "end_frame": segment[end - 1]["frame_idx"],
                    "duration_frames": length_frames,
                }
                if length_frames < config.short_flicker_max_frames and len(owner_values) == 1:
                    for item in segment[start:end]:
                        item["practical"] = surrounding
                    ignored_flickers.append(context)
                elif any(item["raw"] != item["gt"] for item in segment[start:end]):
                    harmful_flickers.append(context)
            segment_start = segment_end

    practical_counts = [0, 0, 0, 0]
    for samples in occupancy_samples.values():
        for item in samples:
            if item["gt"] and item["practical"]:
                practical_counts[0] += 1
            elif not item["gt"] and item["practical"]:
                practical_counts[1] += 1
            elif item["gt"] and not item["practical"]:
                practical_counts[2] += 1
            else:
                practical_counts[3] += 1
    raw_occupancy = _metrics(*raw_counts)
    practical_occupancy = _metrics(*practical_counts)
    occupancy_score = practical_occupancy["occupancy_score"]

    occupancy_delay_items = []
    occupancy_transition_results = []
    for key, intervals in labeled_by_key.items():
        ordered = sorted(intervals, key=lambda item: item["start_frame"])
        sample_map = {item["frame_idx"]: item for item in occupancy_samples[key]}
        for previous, current in zip(ordered, ordered[1:]):
            if previous["occupied"] == current["occupied"]:
                continue
            found = None
            search_end = current["start_frame"] + config.occupancy_delay_max_frames
            for frame_idx in range(current["start_frame"], search_end + 1):
                window = [sample_map.get(frame_idx + offset) for offset in range(config.sustained_state_frames)]
                if all(item is not None and item["practical"] == current["occupied"] for item in window):
                    found = frame_idx
                    break
            delay = found - current["start_frame"] if found is not None else None
            occupancy_transition_results.append({
                "camera_id": key[0], "slot_id": key[1],
                "frame_idx": current["start_frame"], "expected_occupied": current["occupied"],
                "matched_frame": found, "delay_frames": delay,
            })
            occupancy_delay_items.append({
                "kind": "occupancy_transition", "camera_id": key[0],
                "slot_id": key[1], "frame_idx": current["start_frame"],
                "delay_frames": delay,
                "preferred_frames": config.occupancy_delay_preferred_frames,
                "max_frames": config.occupancy_delay_max_frames,
            })

    handoff_delay_items = [
        {
            "kind": "camera_handoff", "physical_vehicle_id": item["physical_vehicle_id"],
            "frame_idx": next(
                event["start_frame"] for event in gt_events if event["event_id"] == item["event_id"]
            ),
            "delay_frames": item["delay_frames"],
            "preferred_frames": next(
                event["preferred_delay_frames"] for event in gt_events if event["event_id"] == item["event_id"]
            ),
            "max_frames": next(
                event["max_delay_frames"]
                for event in gt_events if event["event_id"] == item["event_id"]
            ),
        }
        for item in handoff_results
    ]
    delay_items = (
        binding_delay_items + release_delay_items + recovery_delay_items
        + handoff_delay_items + occupancy_delay_items
    )
    for item in delay_items:
        item["score"] = _normalized_delay_score(
            item["delay_frames"], item["preferred_frames"], item["max_frames"]
        ) * 100.0
    delay_score = statistics.mean(item["score"] for item in delay_items) if delay_items else 100.0
    stability_denominator = max(1, len(occupancy_transition_results) * 2)
    occupancy_stability_score = 100.0 * max(
        0.0, 1.0 - len(harmful_flickers) / stability_denominator
    )
    long_unbound_penalty = min(10.0, 2.0 * len(long_unbound_gids))
    stale_release_count = sum(
        bool(item["stale_departed_gid"]) for item in free_slot_binding_runs
    )
    stale_release_penalty = min(10.0, float(stale_release_count))
    stability_score = max(
        0.0,
        occupancy_stability_score - long_unbound_penalty - stale_release_penalty,
    )
    delay_stability_score = 0.75 * delay_score + 0.25 * stability_score

    scores: Dict[str, Optional[float]] = {
        "identity_continuity_handoff": identity_score,
        "slot_identity_ownership": slot_score,
        "departure_recovery": recovery_score,
        "occupancy": occupancy_score,
        "delay_stability": delay_stability_score,
    }
    weights = {
        "identity_continuity_handoff": 0.35,
        "slot_identity_ownership": 0.30,
        "departure_recovery": 0.15,
        "occupancy": 0.15,
        "delay_stability": 0.05,
    }
    available_weight = sum(weights[key] for key, value in scores.items() if value is not None)
    uncapped = (
        sum(weights[key] * value for key, value in scores.items() if value is not None) / available_weight
        if available_weight else 0.0
    )
    critical_errors = [item for item in errors if item["severity"] == "critical"]
    total_score = min(49.0, uncapped) if critical_errors else uncapped
    result = {
        "schema_version": 3,
        "evaluator_version": EVALUATOR_VERSION,
        "session": session_dir.name,
        "fps": fps,
        "processed_frames": session_info["processed_frames"],
        "classification": _classification(total_score, scores, len(critical_errors)),
        "practical_system_score": total_score,
        "uncapped_practical_system_score": uncapped,
        "score_cap_applied": bool(critical_errors and uncapped > 49.0),
        "critical_error_count": len(critical_errors),
        "critical_errors": critical_errors,
        "errors": errors,
        "scores": scores,
        "weights": weights,
        "identity": {
            "physical_to_canonical_gid": physical_to_gid,
            "phantom_gids_ignored": sorted(phantom_gids),
            "long_unbound_gids": [
                {"canonical_gid": gid, **diagnostic}
                for gid, diagnostic in sorted(long_unbound_gids.items())
            ],
            "checkpoints": checkpoint_results,
            "handoffs": handoff_results,
            "per_vehicle": identity_vehicle_results,
        },
        "slot_ownership": {
            "lifecycles": slot_lifecycles,
            "free_slot_binding_runs": free_slot_binding_runs,
            "duplicate_reservation_runs": duplicate_reservation_runs,
        },
        "departure_recovery": {"events": recovery_results},
        "occupancy": {
            "raw": raw_occupancy,
            "practical": practical_occupancy,
            "missing_labeled_slot_frames_excluded": missing_labeled_slot_frames,
            "ignored_short_flickers": ignored_flickers,
            "harmful_flickers": harmful_flickers,
            "transitions": occupancy_transition_results,
        },
        "delay_stability": {
            "delay_score": delay_score,
            "occupancy_stability_score": occupancy_stability_score,
            "long_unbound_gid_penalty": long_unbound_penalty,
            "stale_release_penalty": stale_release_penalty,
            "stability_score": stability_score,
            "combined_score": delay_stability_score,
            "delay_items": delay_items,
        },
    }
    rounded = _round_tree(result)
    if write_outputs:
        (session_dir / "evaluation_results_v3.json").write_text(
            json.dumps(rounded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (session_dir / "evaluation_report_v3.md").write_text(
            render_markdown_report(rounded), encoding="utf-8"
        )
    return rounded


def _markdown_cell(value: Any) -> str:
    """Render one report-table cell without dropping useful error context."""
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(str(item) for item in value)
        return rendered.replace("|", "\\|").replace("\n", " ") or "n/a"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_error_table(items: Sequence[Mapping[str, Any]]) -> List[str]:
    """Build the same complete context table for critical and ordinary errors."""
    lines = [
        "| Code | Frame(s) | Physical vehicle(s) | Camera(s) | Slot | "
        "Expected GID | Predicted GID(s) | Duration | Details |",
        "|---|---:|---|---|---|---:|---:|---:|---|",
    ]
    for item in items:
        start = item.get("frame_idx")
        end = item.get("end_frame")
        if start is None and end is None:
            frame_range = "n/a"
        elif start is None:
            frame_range = str(end)
        elif end is not None and end != start:
            frame_range = f"{start}–{end}"
        else:
            frame_range = str(start)

        vehicles = item.get("physical_vehicle_id")
        if vehicles in (None, ""):
            vehicles = item.get("physical_vehicle_ids")
        cameras = item.get("camera_id")
        if cameras in (None, ""):
            cameras = item.get("camera_ids")
        predicted = item.get("predicted_gid")
        if predicted in (None, ""):
            predicted = item.get("predicted_gids")
        duration = item.get("duration_frames")
        duration_display = "n/a" if duration in (None, "") else f"{duration} frames"

        lines.append(
            "| " + " | ".join([
                _markdown_cell(item.get("code")),
                _markdown_cell(frame_range),
                _markdown_cell(vehicles),
                _markdown_cell(cameras),
                _markdown_cell(item.get("slot_id")),
                _markdown_cell(item.get("expected_gid")),
                _markdown_cell(predicted),
                _markdown_cell(duration_display),
                _markdown_cell(item.get("message")),
            ]) + " |"
        )
    return lines


def render_markdown_report(result: Mapping[str, Any]) -> str:
    scores = result["scores"]
    lines = [
        f"# TechGAR Practical System Report — {result['session']}",
        "",
        f"- Kết luận: **{result['classification']}**",
        f"- Practical System Score: **{result['practical_system_score']:.2f}/100**",
        f"- Điểm trước giới hạn critical: {result['uncapped_practical_system_score']:.2f}/100",
        f"- Critical errors: **{result['critical_error_count']}**",
        "",
        "## Điểm thành phần",
        "",
        "| Thành phần | Trọng số | Điểm |",
        "|---|---:|---:|",
    ]
    labels = {
        "identity_continuity_handoff": "Identity continuity + handoff",
        "slot_identity_ownership": "Đúng chủ sở hữu ô",
        "departure_recovery": "Departure recovery / ReID",
        "occupancy": "Occupied / free",
        "delay_stability": "Delay + stability",
    }
    for key, label in labels.items():
        value = scores.get(key)
        display = "N/A" if value is None else f"{value:.2f}"
        lines.append(f"| {label} | {result['weights'][key] * 100:.0f}% | {display} |")
    raw = result["occupancy"]["raw"]
    practical = result["occupancy"]["practical"]
    lines.extend([
        "",
        "## Occupancy",
        "",
        "| Metric | Raw | Practical |",
        "|---|---:|---:|",
        f"| Occupied F1 | {raw['occupied_f1']:.4f} | {practical['occupied_f1']:.4f} |",
        f"| Balanced accuracy | {raw['balanced_accuracy']:.4f} | {practical['balanced_accuracy']:.4f} |",
        f"| False-free rate | {raw['false_free_rate']:.4f} | {practical['false_free_rate']:.4f} |",
        f"| False-occupied rate | {raw['false_occupied_rate']:.4f} | {practical['false_occupied_rate']:.4f} |",
        "",
        "## Critical errors",
        "",
    ])
    if not result["critical_errors"]:
        lines.append("Không có critical error.")
    else:
        lines.extend(_markdown_error_table(result["critical_errors"]))
    noncritical = [item for item in result.get("errors", []) if item.get("severity") != "critical"]
    lines.extend(["", "## Non-critical misses / delays", ""])
    if not noncritical:
        lines.append("Không có lỗi non-critical.")
    else:
        lines.extend(_markdown_error_table(noncritical))
    lines.extend([
        "",
        "## Quy tắc kết luận",
        "",
        "Một critical error làm phiên FAIL và giới hạn điểm tối đa 49. "
        "PASS yêu cầu tổng điểm ≥85 và cả identity, slot ownership, recovery đều ≥95.",
        "",
    ])
    return "\n".join(lines)


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict:
    """Aggregate sessions by lifecycle units rather than slot-frame count."""
    if not results:
        raise ValueError("at least one session result is required")

    identity_units = [
        item["score"] for result in results for item in result["identity"]["per_vehicle"]
    ]
    slot_units = [
        item["score"] for result in results for item in result["slot_ownership"]["lifecycles"]
    ]
    recovery_units = [
        item["score"] for result in results for item in result["departure_recovery"]["events"]
    ]
    occupancy_units = [result["scores"]["occupancy"] for result in results if result["scores"]["occupancy"] is not None]
    delay_units = [result["scores"]["delay_stability"] for result in results if result["scores"]["delay_stability"] is not None]

    def mean_or_none(items: Sequence[float]) -> Optional[float]:
        return statistics.mean(items) if items else None

    scores = {
        "identity_continuity_handoff": mean_or_none(identity_units),
        "slot_identity_ownership": mean_or_none(slot_units),
        "departure_recovery": mean_or_none(recovery_units),
        "occupancy": mean_or_none(occupancy_units),
        "delay_stability": mean_or_none(delay_units),
    }
    weights = dict(results[0]["weights"])
    available_weight = sum(weights[key] for key, value in scores.items() if value is not None)
    uncapped = sum(weights[key] * value for key, value in scores.items() if value is not None) / available_weight
    critical_errors = [
        {"session": result["session"], **item}
        for result in results for item in result["critical_errors"]
    ]
    score = min(49.0, uncapped) if critical_errors else uncapped
    return _round_tree({
        "schema_version": 3,
        "evaluator_version": EVALUATOR_VERSION,
        "session": "aggregate",
        "sessions": [result["session"] for result in results],
        "classification": _classification(score, scores, len(critical_errors)),
        "practical_system_score": score,
        "uncapped_practical_system_score": uncapped,
        "score_cap_applied": bool(critical_errors and uncapped > 49.0),
        "critical_error_count": len(critical_errors),
        "critical_errors": critical_errors,
        "scores": scores,
        "weights": weights,
        "aggregation_units": {
            "identity_vehicle_lifecycles": len(identity_units),
            "slot_lifecycles": len(slot_units),
            "departure_events": len(recovery_units),
            "occupancy_sessions": len(occupancy_units),
            "delay_sessions": len(delay_units),
        },
    })
