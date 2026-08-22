"""Migrate a two-camera experiment session from prediction schema 2 to 3.

The migration is deliberately separated from ground-truth annotation.  It can
upgrade the *shape* of legacy ground-truth CSV files, but it never invents an
identity checkpoint from predictions.  Human-reviewed checkpoints belong in
``ground_truth_identity.csv``.

All output is staged before commit.  The original prediction stream is kept as
``predictions.schema2.jsonl`` and every replaced metadata/ground-truth file is
also backed up.  If any replace fails, the original session is restored.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 3
PHYSICAL_VEHICLE_RE = re.compile(r"\bM\d+_V\d+\b", re.IGNORECASE)
SLOT_RE = re.compile(r"\b[A-Z]\d{2}\b", re.IGNORECASE)

SLOT_FIELDS = (
    "schema_version", "camera_id", "slot_id", "start_frame", "end_frame",
    "occupied", "physical_vehicle_id", "identity_required", "notes",
)
EVENT_FIELDS = (
    "schema_version", "event_id", "physical_vehicle_id", "event_type",
    "start_frame", "end_frame", "source_camera", "target_camera",
    "source_slot_id", "target_slot_id", "preferred_delay_frames",
    "max_delay_frames", "required", "critical", "notes",
)
IDENTITY_FIELDS = (
    "schema_version", "observation_id", "physical_vehicle_id", "frame_idx",
    "camera_id", "anchor_x", "anchor_y", "slot_id", "phase", "required",
    "notes",
)


def _as_gid(value: Any) -> int | str | None:
    if value is None or value == "" or str(value).lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _gid_key(value: Any) -> str | None:
    value = _as_gid(value)
    return None if value is None else str(value)


def _normalize_camera(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"1", "cam1", "camera1"}:
        return "cam1"
    if text in {"2", "cam2", "camera2"}:
        return "cam2"
    return text


def _bool_text(value: Any, *, default: bool = False) -> str:
    if value is None or value == "":
        return "true" if default else "false"
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "false"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: JSON khong hop le: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"{path} rong")
    return records


def _collect_aliases(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for record in records:
        registry = record.get("global_registry") or {}
        for alias, canonical in (registry.get("retired_global_ids") or {}).items():
            aliases[str(alias)] = _as_gid(canonical)
    # Collapse alias chains using the final graph observed over the whole run.
    for alias in list(aliases):
        aliases[alias] = _resolve_gid(aliases[alias], aliases)
    return aliases


def _resolve_gid(value: Any, aliases: Mapping[str, Any]) -> int | str | None:
    current = _as_gid(value)
    seen: set[str] = set()
    while current is not None and str(current) in aliases and str(current) not in seen:
        seen.add(str(current))
        current = _as_gid(aliases[str(current)])
    return current


def _aliases_for(canonical_gid: Any, aliases: Mapping[str, Any]) -> list[int | str]:
    canonical = _resolve_gid(canonical_gid, aliases)
    if canonical is None:
        return []
    values: list[int | str] = [canonical]
    for alias in aliases:
        if _resolve_gid(alias, aliases) == canonical and _as_gid(alias) not in values:
            values.append(_as_gid(alias))
    return sorted(values, key=lambda item: (str(type(item)), str(item)))


def _canonicalize_detail_gids(value: Any, aliases: Mapping[str, Any], key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            name: _canonicalize_detail_gids(item, aliases, name)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_detail_gids(item, aliases, key) for item in value]
    if key.endswith("global_id") or key.endswith("_gid"):
        return _resolve_gid(value, aliases)
    return value


def _event_payload(
    event: Mapping[str, Any], *, source: str, fallback_frame: int,
    aliases: Mapping[str, Any], camera_id: str | None = None,
) -> dict[str, Any]:
    raw_gid = _as_gid(event.get("global_id", event.get("vehicle_id")))
    frame_idx = int(event.get("frame", event.get("frame_idx", fallback_frame)))
    details = {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key not in {"type", "event_type", "frame", "frame_idx", "global_id", "vehicle_id"}
    }
    details = _canonicalize_detail_gids(details, aliases)
    identity = {
        "source": source,
        "event_type": str(event.get("type", event.get("event_type", "unknown"))),
        "frame_idx": frame_idx,
        "raw_gid": raw_gid,
        "details": details,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:20]
    payload: dict[str, Any] = {
        "event_uid": f"evt-{digest}",
        **identity,
        "canonical_gid": _resolve_gid(raw_gid, aliases),
    }
    if camera_id is not None:
        payload["camera_id"] = camera_id
    return payload


def _number_event_occurrence(
    payload: dict[str, Any], occurrences: dict[str, int],
) -> dict[str, Any]:
    """Keep distinct byte-identical entries in one legacy rolling snapshot.

    A schema-2 JSON snapshot has lost Python object identity, so an event that
    is dropped and replaced by a byte-identical event cannot be reconstructed
    perfectly.  Numbering equal entries still preserves every simultaneous
    occurrence while allowing repeated rolling-history snapshots to dedupe.
    """
    base_uid = payload["event_uid"]
    occurrence = occurrences.get(base_uid, 0) + 1
    occurrences[base_uid] = occurrence
    payload["event_uid"] = f"{base_uid}-{occurrence:03d}"
    return payload


def _world_lookup(record: Mapping[str, Any], aliases: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    registry = record.get("global_registry") or {}
    unit = str(registry.get("world_unit") or "cm")
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for gid, vehicle in (registry.get("active_global_vehicles") or {}).items():
        canonical = _resolve_gid(gid, aliases)
        for observation in vehicle.get("observations", []):
            camera_id = _normalize_camera(observation.get("camera_id"))
            local_id = observation.get("local_track_id")
            position = observation.get("global_position")
            if local_id is None or not isinstance(position, dict):
                continue
            lookup[(camera_id, int(local_id))] = {
                "x": position.get("x"), "y": position.get("y"), "unit": unit,
                "reference": "shared_ground_plane", "canonical_gid": canonical,
            }
    return lookup


def _slot_rows(record: Mapping[str, Any], aliases: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for camera_id, camera in (record.get("cameras") or {}).items():
        camera_id = _normalize_camera(camera_id)
        for slot_id, slot in (camera.get("parking_slots") or {}).items():
            raw_gid = _as_gid(slot.get("vehicle_id"))
            recovery_gid = _as_gid(slot.get("recovery_global_id"))
            result.append({
                "camera_id": camera_id,
                "slot_id": str(slot_id),
                "occupied": bool(slot.get("occupied", False)),
                "raw_vehicle_gid": raw_gid,
                "canonical_vehicle_gid": _resolve_gid(raw_gid, aliases),
                "vision_occupied": bool(slot.get("vision_occupied", False)),
                "tracking_occupied": bool(slot.get("tracking_occupied", False)),
                "decision_source": str(slot.get("decision_source", "unknown")),
                "tracking_state": str(slot.get("tracking_state", "unknown")),
                "vehicle_overlap": float(slot.get("vehicle_overlap", 0.0) or 0.0),
                "stopped_for_ms": int(slot.get("stopped_for_ms", 0) or 0),
                "recovery_state": str(slot.get("recovery_state", "none")),
                "recovery_global_id": _resolve_gid(recovery_gid, aliases),
                "recovery_age_ms": int(slot.get("recovery_age_ms", 0) or 0),
                "recovery_radius_px": float(slot.get("recovery_radius_px", 0.0) or 0.0),
                "recovery_candidate_count": int(slot.get("recovery_candidate_count", 0) or 0),
            })
    return result


def _observation_rows(
    record: Mapping[str, Any], slots: list[dict[str, Any]], aliases: Mapping[str, Any],
) -> list[dict[str, Any]]:
    registry = record.get("global_registry") or {}
    lifecycle = registry.get("identity_lifecycle") or {}
    world = _world_lookup(record, aliases)
    result: list[dict[str, Any]] = []
    frame_idx = int(record["frame_idx"])
    for camera_id, camera in (record.get("cameras") or {}).items():
        camera_id = _normalize_camera(camera_id)
        for track in camera.get("local_tracks") or []:
            local_id = int(track["local_track_id"])
            raw_gid = _as_gid(track.get("global_id"))
            canonical_gid = _resolve_gid(raw_gid, aliases)
            bbox = list(track.get("bbox") or [])
            center = track.get("center")
            if not center and len(bbox) == 4:
                center = [float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0]
            if not center:
                # Telemetry from supported schema-2 writers always has a bbox,
                # but keep migrated schema strict if a hand-edited stream does not.
                center = [0.0, 0.0]
            owner = next(({
                "camera_id": item["camera_id"], "slot_id": item["slot_id"],
                "state": "bound" if item["occupied"] else "reserved",
            } for item in slots if canonical_gid is not None and item["canonical_vehicle_gid"] == canonical_gid), None)
            life = lifecycle.get(str(canonical_gid), lifecycle.get(str(raw_gid), {})) if raw_gid is not None else {}
            result.append({
                "observation_uid": f"{frame_idx}:{camera_id}:{local_id}",
                "camera_id": camera_id,
                "local_track_id": local_id,
                "raw_gid": raw_gid,
                "canonical_gid": canonical_gid,
                "gid_aliases": _aliases_for(canonical_gid, aliases),
                "bbox": bbox,
                "anchor_pixel": {
                    "x": center[0], "y": center[1], "reference": "tracker_center",
                },
                "anchor_world": world.get((camera_id, local_id)),
                "track_state": str(track.get("state", "unknown")),
                "association_state": str(track.get("association_state", "unknown")),
                "invisible_count": int(track.get("invisible_count", 0) or 0),
                "assignment_cost": track.get("assignment_cost") or {},
                "fragment_visible_count": int(track.get("fragment_visible_count", 0) or 0),
                "first_observation_frame": int(track.get("first_observation_frame") or frame_idx),
                "identity_state": str(
                    life.get("state", "unknown") if isinstance(life, dict) else "unknown"
                ) if canonical_gid is not None else "unassigned",
                "slot_ownership": owner,
            })
    return result


def _reservations(record: Mapping[str, Any], aliases: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = record.get("parked_identity_reservations") or {}
    result: list[dict[str, Any]] = []
    for gid, reservation in source.items():
        item = copy.deepcopy(reservation)
        item["canonical_gid"] = _resolve_gid(gid, aliases)
        item["camera_id"] = _normalize_camera(item.get("camera_id", item.get("camera")))
        item["slot_id"] = item.get("slot_id")
        item.setdefault("state", "reserved")
        item.setdefault("bbox", None)
        result.append(item)
    return result


def convert_prediction_records(
    records: list[dict[str, Any]], timing_by_frame: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert loaded schema-2 records to schema 3 without using ground truth."""
    if not records or any(int(record.get("schema_version", 0)) != 2 for record in records):
        raise ValueError("Nguon predictions phai chi chua schema_version=2")
    aliases = _collect_aliases(records)
    alias_rows = [
        {"alias_gid": _as_gid(alias), "canonical_gid": _resolve_gid(alias, aliases)}
        for alias in sorted(aliases, key=str)
    ]
    seen_identity_events: set[str] = set()
    seen_parking_events: set[str] = set()
    output: list[dict[str, Any]] = []
    timing_by_frame = timing_by_frame or {}

    for record in records:
        frame_idx = int(record["frame_idx"])
        timing = timing_by_frame.get(frame_idx, {})
        slots = _slot_rows(record, aliases)
        identity_events: list[dict[str, Any]] = []
        parking_events: list[dict[str, Any]] = []

        manager_events = (record.get("global_registry") or {}).get("recent_events") or []
        manager_occurrences: dict[str, int] = {}
        for event in manager_events:
            payload = _number_event_occurrence(_event_payload(
                event, source="global_manager", fallback_frame=frame_idx, aliases=aliases,
            ), manager_occurrences)
            if payload["event_uid"] not in seen_identity_events:
                seen_identity_events.add(payload["event_uid"])
                identity_events.append(payload)

        for camera_id, camera in (record.get("cameras") or {}).items():
            camera_id = _normalize_camera(camera_id)
            association_occurrences: dict[str, int] = {}
            for event in camera.get("association_events") or []:
                event_with_frame = dict(event)
                event_with_frame.setdefault("frame", frame_idx)
                payload = _number_event_occurrence(_event_payload(
                    event_with_frame, source=f"motion_tracker:{camera_id}",
                    fallback_frame=frame_idx, aliases=aliases, camera_id=camera_id,
                ), association_occurrences)
                if payload["event_uid"] not in seen_identity_events:
                    seen_identity_events.add(payload["event_uid"])
                    identity_events.append(payload)
            parking_occurrences: dict[str, int] = {}
            for event in camera.get("recent_parking_events") or []:
                payload = _number_event_occurrence(_event_payload(
                    event, source=f"slot_binder:{camera_id}", fallback_frame=frame_idx,
                    aliases=aliases, camera_id=camera_id,
                ), parking_occurrences)
                if payload["event_uid"] not in seen_parking_events:
                    seen_parking_events.add(payload["event_uid"])
                    parking_events.append(payload)

        output.append({
            "schema_version": SCHEMA_VERSION,
            "frame_idx": frame_idx,
            "capture_unix_ns": _optional_int(timing.get("capture_unix_ns")),
            "wall_time_iso": timing.get("wall_time_iso"),
            "camera_timestamps_ns": record.get("camera_timestamps_ns") or {},
            "camera_skew_ms": record.get("camera_skew_ms"),
            "observations": _observation_rows(record, slots, aliases),
            "slots": slots,
            "gid_aliases": copy.deepcopy(alias_rows),
            "identity_events": identity_events,
            "parking_events": parking_events,
            "parking_recovery": copy.deepcopy(record.get("parking_recovery") or []),
            "parked_identity_reservations": _reservations(record, aliases),
        })
    return output


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _load_timing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Migration schema v3 can frame_timestamps.csv de tao capture_unix_ns/wall_time_iso: {path}"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"frame_idx", "capture_unix_ns", "wall_time_iso"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} thieu cot: {', '.join(sorted(missing))}")
        result = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_idx = int(row["frame_idx"])
                capture_unix_ns = int(row["capture_unix_ns"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_number}: timing khong hop le") from exc
            if frame_idx < 1 or capture_unix_ns < 0 or not row["wall_time_iso"].strip():
                raise ValueError(f"{path.name}:{line_number}: timing khong hop le")
            result[frame_idx] = row
        return result


def _csv_text(rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return "\ufeff" + buffer.getvalue()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _upgrade_slots(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if rows and set(SLOT_FIELDS).issubset(rows[0]):
        return rows
    output: list[dict[str, Any]] = []
    for row in rows:
        notes = row.get("notes", "")
        physical_id = row.get("physical_vehicle_id") or row.get("vehicle_id") or ""
        if not physical_id:
            match = PHYSICAL_VEHICLE_RE.search(notes)
            physical_id = match.group(0).upper() if match else ""
        output.append({
            "schema_version": SCHEMA_VERSION,
            "camera_id": _normalize_camera(row.get("camera_id")),
            "slot_id": row.get("slot_id", ""),
            "start_frame": row.get("start_frame", ""),
            "end_frame": row.get("end_frame", ""),
            "occupied": _bool_text(row.get("occupied")),
            "physical_vehicle_id": physical_id,
            "identity_required": _bool_text(bool(physical_id)),
            "notes": notes,
        })
    return output


def _event_defaults(event_type: str) -> tuple[int, int, bool]:
    if event_type in {"departure_started", "slot_leave"}:
        return 25, 125, True
    if event_type == "camera_handoff":
        return 25, 125, True
    if event_type in {"slot_enter", "parked"}:
        return 25, 75, True
    return 25, 75, False


def _upgrade_events(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if rows and set(EVENT_FIELDS).issubset(rows[0]):
        return rows
    output: list[dict[str, Any]] = []
    for row in rows:
        event_type = row.get("event_type", "")
        preferred, maximum, critical = _event_defaults(event_type)
        frame = row.get("frame_idx", row.get("start_frame", ""))
        notes = row.get("notes", "")
        slot_match = SLOT_RE.search(notes)
        slot_id = slot_match.group(0).upper() if slot_match else ""
        source_slot = slot_id if event_type in {"slot_leave", "departure_started"} else ""
        target_slot = slot_id if event_type in {"slot_enter", "parked"} else ""
        output.append({
            "schema_version": SCHEMA_VERSION,
            "event_id": row.get("event_id", ""),
            "physical_vehicle_id": row.get("physical_vehicle_id") or row.get("global_id", ""),
            "event_type": event_type,
            "start_frame": frame,
            "end_frame": row.get("end_frame") or frame,
            "source_camera": _normalize_camera(row.get("source_camera")),
            "target_camera": _normalize_camera(row.get("target_camera")),
            "source_slot_id": source_slot,
            "target_slot_id": target_slot,
            "preferred_delay_frames": preferred,
            "max_delay_frames": maximum,
            "required": "true",
            "critical": _bool_text(critical),
            "notes": notes,
        })
    return output


def _jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)


def _validate_staged_predictions(records: list[dict[str, Any]]) -> None:
    required = {
        "schema_version", "frame_idx", "observations", "slots", "gid_aliases",
        "identity_events", "parking_events", "parking_recovery",
        "parked_identity_reservations",
    }
    frame_ids: list[int] = []
    event_ids: set[str] = set()
    for record in records:
        missing = required - set(record)
        if missing or record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Prediction v3 staged khong hop le; thieu={sorted(missing)}")
        if not isinstance(record.get("capture_unix_ns"), int) or record["capture_unix_ns"] < 0:
            raise ValueError("Prediction v3 staged thieu capture_unix_ns hop le")
        if not isinstance(record.get("wall_time_iso"), str) or not record["wall_time_iso"]:
            raise ValueError("Prediction v3 staged thieu wall_time_iso hop le")
        timestamps = record.get("camera_timestamps_ns")
        if not isinstance(timestamps, dict) or not all(
            isinstance(key, str) and isinstance(value, int) and value >= 0
            for key, value in timestamps.items()
        ):
            raise ValueError("Prediction v3 staged co camera_timestamps_ns khong hop le")
        if not isinstance(record.get("camera_skew_ms"), (int, float)) or record["camera_skew_ms"] < 0:
            raise ValueError("Prediction v3 staged co camera_skew_ms khong hop le")
        frame_ids.append(int(record["frame_idx"]))
        for field in ("identity_events", "parking_events"):
            for event in record[field]:
                uid = event["event_uid"]
                if uid in event_ids:
                    raise ValueError(f"Event UID bi lap: {uid}")
                event_ids.add(uid)
    if frame_ids != list(range(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise ValueError("frame_idx prediction khong lien tuc")


def migrate_session(session: Path, *, dry_run: bool = False) -> dict[str, Any]:
    session = session.resolve()
    prediction_path = session / "predictions.jsonl"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    records = _read_jsonl(prediction_path)
    source_schema = int(records[0].get("schema_version", 0))
    if source_schema == SCHEMA_VERSION:
        return {"session": str(session), "status": "already_v3", "frames": len(records)}
    if source_schema != 2:
        raise ValueError(f"Chi ho tro migrate predictions schema 2; gap schema {source_schema}")

    converted = convert_prediction_records(records, _load_timing(session / "frame_timestamps.csv"))
    _validate_staged_predictions(converted)
    slots = _upgrade_slots(_read_csv(session / "ground_truth_slots.csv"))
    events = _upgrade_events(_read_csv(session / "ground_truth_events.csv"))
    identities = _read_csv(session / "ground_truth_identity.csv")
    if identities and not set(IDENTITY_FIELDS).issubset(identities[0]):
        raise ValueError("ground_truth_identity.csv ton tai nhung khong dung schema v3")

    info_path = session / "session_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    info["schema_version"] = SCHEMA_VERSION
    info.setdefault("files", {})
    info["files"].update({
        "predictions": "predictions.jsonl",
        "predictions_schema2_backup": "predictions.schema2.jsonl",
        "ground_truth_slots": "ground_truth_slots.csv",
        "ground_truth_events": "ground_truth_events.csv",
        "ground_truth_identity": "ground_truth_identity.csv",
    })

    staged_contents = {
        "predictions.jsonl": _jsonl_text(converted),
        "ground_truth_slots.csv": _csv_text(slots, SLOT_FIELDS),
        "ground_truth_events.csv": _csv_text(events, EVENT_FIELDS),
        "ground_truth_identity.csv": _csv_text(identities, IDENTITY_FIELDS),
        "session_info.json": json.dumps(info, indent=2, ensure_ascii=False) + "\n",
    }
    result = {
        "session": str(session), "status": "dry_run" if dry_run else "migrated",
        "frames": len(converted), "identity_checkpoints": len(identities),
        "identity_events": sum(len(item["identity_events"]) for item in converted),
        "parking_events": sum(len(item["parking_events"]) for item in converted),
    }
    if dry_run:
        return result

    backup_names = {
        "predictions.jsonl": "predictions.schema2.jsonl",
        "ground_truth_slots.csv": "ground_truth_slots.legacy.csv",
        "ground_truth_events.csv": "ground_truth_events.legacy.csv",
        "ground_truth_identity.csv": "ground_truth_identity.pre_v3.csv",
        "session_info.json": "session_info.schema2.json",
    }
    existing_backups = [name for name in backup_names.values() if (session / name).exists()]
    if existing_backups:
        raise FileExistsError(
            "Khong ghi de backup migration da ton tai: " + ", ".join(existing_backups)
        )

    stage_dir = Path(tempfile.mkdtemp(prefix=".schema3-stage-", dir=session))
    rollback_dir = Path(tempfile.mkdtemp(prefix=".schema3-rollback-", dir=session))
    replaced: list[str] = []
    try:
        for name, content in staged_contents.items():
            target = stage_dir / name
            target.write_text(content, encoding="utf-8", newline="")
            # Windows requires a writable descriptor for fsync/FlushFileBuffers.
            with target.open("rb+") as handle:
                os.fsync(handle.fileno())
        for name, backup_name in backup_names.items():
            source = session / name
            if source.exists():
                shutil.copy2(source, rollback_dir / name)
                shutil.copy2(source, session / backup_name)
        for name in staged_contents:
            os.replace(stage_dir / name, session / name)
            replaced.append(name)
    except Exception:
        for name in replaced:
            rollback = rollback_dir / name
            target = session / name
            if rollback.exists():
                os.replace(rollback, target)
            elif target.exists():
                target.unlink()
        for backup_name in backup_names.values():
            backup = session / backup_name
            if backup.exists():
                backup.unlink()
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(rollback_dir, ignore_errors=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate TechGAR two-camera experiment sessions to schema v3",
    )
    parser.add_argument("sessions", nargs="+", type=Path, help="Thu muc session can migrate")
    parser.add_argument("--dry-run", action="store_true", help="Kiem tra/chuyen trong RAM, khong ghi file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for session in args.sessions:
        result = migrate_session(session, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
