import csv
import json
from pathlib import Path

from experiment_test.migrate_schema_v3 import migrate_session


def _legacy_record(frame, *, alias=False):
    manager_events = [{
        "type": "global_id_created", "frame": 1, "global_id": 9,
        "camera": "cam1", "local_track_id": 7,
    }]
    if alias:
        manager_events.append({
            "type": "global_id_merged", "frame": 2, "global_id": 9,
            "kept_global_id": 4,
        })
    return {
        "schema_version": 2,
        "frame_idx": frame,
        "camera_timestamps_ns": {"cam1": frame * 10, "cam2": frame * 10 + 1},
        "camera_skew_ms": 0.001,
        "cameras": {
            "cam1": {
                "confirmed_vehicles": [9],
                "local_tracks": [{
                    "local_track_id": 7,
                    "global_id": 9 if frame == 1 else 4,
                    "bbox": [10, 20, 30, 40],
                    "center": [25, 60],
                    "state": "confirmed",
                    "invisible_count": 0,
                    "association_state": "matched",
                    "assignment_cost": {},
                    "fragment_visible_count": 8,
                    "first_observation_frame": 1,
                }],
                "association_events": [],
                "parking_slots": {
                    "A01": {
                        "occupied": True,
                        "vehicle_id": 9,
                        "vision_occupied": True,
                        "tracking_occupied": True,
                    }
                },
                "recent_parking_events": [{
                    "type": "vehicle_stopped_in_slot", "frame": 1,
                    "global_id": 9, "slot_id": "A01",
                }],
            },
            "cam2": {
                "confirmed_vehicles": [], "local_tracks": [],
                "association_events": [], "parking_slots": {},
                "recent_parking_events": [],
            },
        },
        "parking_recovery": [],
        "parked_identity_reservations": {},
        "global_registry": {
            "world_unit": "cm",
            "retired_global_ids": {"9": 4} if alias else {},
            "active_global_vehicles": {},
            "identity_lifecycle": {"4": {"state": "active"}},
            "recent_events": manager_events,
        },
    }


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _write_timing(path: Path, frame_count: int):
    _write_csv(
        path,
        ["frame_idx", "capture_unix_ns", "wall_time_iso"],
        [[frame, frame * 100, f"2026-08-22T12:00:{frame:02d}+07:00"] for frame in range(1, frame_count + 1)],
    )


def test_migration_is_backed_up_and_uses_final_alias_without_inventing_checkpoints(tmp_path):
    records = [_legacy_record(1), _legacy_record(2, alias=True)]
    (tmp_path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8",
    )
    (tmp_path / "session_info.json").write_text(
        json.dumps({"schema_version": 2, "processed_frames": 2, "files": {}}),
        encoding="utf-8",
    )
    _write_csv(
        tmp_path / "frame_timestamps.csv",
        ["frame_idx", "capture_unix_ns", "wall_time_iso"],
        [[1, 100, "t1"], [2, 200, "t2"]],
    )
    _write_csv(
        tmp_path / "ground_truth_slots.csv",
        ["camera_id", "slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes"],
        [[1, "A01", 1, 2, "true", "", "M01_V1 is parked"]],
    )
    _write_csv(
        tmp_path / "ground_truth_events.csv",
        ["event_id", "global_id", "source_camera", "target_camera", "event_type", "frame_idx", "notes"],
        [["E1", "M01_V1", "cam1", "cam1", "slot_enter", 1, "enters A01"]],
    )

    result = migrate_session(tmp_path)

    assert result["status"] == "migrated"
    assert (tmp_path / "predictions.schema2.jsonl").read_text(encoding="utf-8") == (
        "".join(json.dumps(row) + "\n" for row in records)
    )
    migrated = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["schema_version"] for item in migrated] == [3, 3]
    assert migrated[0]["observations"][0]["canonical_gid"] == 4
    assert migrated[0]["slots"][0]["canonical_vehicle_gid"] == 4
    assert sum(len(item["parking_events"]) for item in migrated) == 1
    assert len({
        event["event_uid"]
        for item in migrated
        for key in ("identity_events", "parking_events")
        for event in item[key]
    }) == sum(
        len(item[key]) for item in migrated
        for key in ("identity_events", "parking_events")
    )

    with (tmp_path / "ground_truth_slots.csv").open(encoding="utf-8-sig") as source:
        slot = next(csv.DictReader(source))
    assert slot["schema_version"] == "3"
    assert slot["camera_id"] == "cam1"
    assert slot["physical_vehicle_id"] == "M01_V1"
    assert slot["identity_required"] == "true"

    with (tmp_path / "ground_truth_identity.csv").open(encoding="utf-8-sig") as source:
        identities = list(csv.DictReader(source))
    assert identities == []  # Manual raw-video annotation remains mandatory.
    assert json.loads((tmp_path / "session_info.json").read_text())["schema_version"] == 3


def test_migration_dry_run_does_not_modify_session(tmp_path):
    source = json.dumps(_legacy_record(1)) + "\n"
    (tmp_path / "predictions.jsonl").write_text(source, encoding="utf-8")
    _write_timing(tmp_path / "frame_timestamps.csv", 1)
    _write_csv(tmp_path / "ground_truth_slots.csv", [
        "camera_id", "slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes",
    ], [])
    _write_csv(tmp_path / "ground_truth_events.csv", [
        "event_id", "global_id", "source_camera", "target_camera", "event_type", "frame_idx", "notes",
    ], [])

    result = migrate_session(tmp_path, dry_run=True)

    assert result["status"] == "dry_run"
    assert (tmp_path / "predictions.jsonl").read_text(encoding="utf-8") == source
    assert not (tmp_path / "predictions.schema2.jsonl").exists()


def test_conversion_preserves_simultaneous_identical_legacy_events(tmp_path):
    record = _legacy_record(1)
    duplicate = {
        "type": "local_track_lost", "frame": 1, "global_id": 9,
        "camera": "cam1", "local_track_id": 7,
    }
    record["global_registry"]["recent_events"] = [dict(duplicate), dict(duplicate)]
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8",
    )
    _write_timing(tmp_path / "frame_timestamps.csv", 1)
    _write_csv(tmp_path / "ground_truth_slots.csv", [
        "camera_id", "slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes",
    ], [])
    _write_csv(tmp_path / "ground_truth_events.csv", [
        "event_id", "global_id", "source_camera", "target_camera", "event_type", "frame_idx", "notes",
    ], [])

    migrate_session(tmp_path)

    migrated = json.loads(
        (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert len(migrated["identity_events"]) == 2
    assert len({event["event_uid"] for event in migrated["identity_events"]}) == 2


def test_migration_rejects_missing_timing_before_replacing_files(tmp_path):
    source = json.dumps(_legacy_record(1)) + "\n"
    (tmp_path / "predictions.jsonl").write_text(source, encoding="utf-8")
    _write_csv(tmp_path / "ground_truth_slots.csv", [
        "camera_id", "slot_id", "start_frame", "end_frame", "occupied", "vehicle_id", "notes",
    ], [])
    _write_csv(tmp_path / "ground_truth_events.csv", [
        "event_id", "global_id", "source_camera", "target_camera", "event_type", "frame_idx", "notes",
    ], [])

    try:
        migrate_session(tmp_path)
    except FileNotFoundError as exc:
        assert "frame_timestamps.csv" in str(exc)
    else:
        raise AssertionError("migration must reject a session without frame timing")

    assert (tmp_path / "predictions.jsonl").read_text(encoding="utf-8") == source
    assert not (tmp_path / "predictions.schema2.jsonl").exists()
