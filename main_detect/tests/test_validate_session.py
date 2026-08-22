import json
import csv

from experiment_test.validate_session import validate


def test_schema_two_uses_two_camera_required_files(tmp_path):
    (tmp_path / "session_info.json").write_text(
        json.dumps({"schema_version": 2, "processed_frames": 0}),
        encoding="utf-8",
    )

    errors, _ = validate(tmp_path)

    assert "Thieu file raw_cam1.mp4" in errors
    assert "Thieu file raw_video.mp4" not in errors


def test_analysis_only_two_camera_session_does_not_require_copied_videos(tmp_path):
    (tmp_path / "session_info.json").write_text(
        json.dumps({
            "schema_version": 2,
            "processed_frames": 0,
            "analysis_only": True,
        }),
        encoding="utf-8",
    )

    errors, _ = validate(tmp_path)

    assert not any(".mp4" in error for error in errors)


def _write_csv(path, header, rows=()):
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _valid_v3_session(tmp_path):
    (tmp_path / "session_info.json").write_text(json.dumps({
        "schema_version": 3,
        "processed_frames": 1,
        "analysis_only": True,
    }), encoding="utf-8")
    prediction = {
        "schema_version": 3,
        "frame_idx": 1,
        "capture_unix_ns": 1,
        "wall_time_iso": "2026-08-22T12:00:00+07:00",
        "camera_timestamps_ns": {"cam1": 1, "cam2": 2},
        "camera_skew_ms": 0.001,
        "observations": [{
            "observation_uid": "1:cam1:1", "camera_id": "cam1",
            "local_track_id": 1, "raw_gid": 7, "canonical_gid": 7,
            "gid_aliases": [7], "bbox": [0, 0, 10, 10],
            "anchor_pixel": {"x": 5, "y": 10, "reference": "tracker_center"},
            "anchor_world": None, "track_state": "confirmed",
            "association_state": "matched", "invisible_count": 0,
            "assignment_cost": {}, "fragment_visible_count": 5,
            "first_observation_frame": 1, "identity_state": "active",
            "slot_ownership": {"camera_id": "cam1", "slot_id": "A01", "state": "occupied"},
        }],
        "slots": [{
            "camera_id": "cam1", "slot_id": "A01", "occupied": True,
            "raw_vehicle_gid": 7, "canonical_vehicle_gid": 7,
            "vision_occupied": True, "tracking_occupied": True,
            "decision_source": "vision_and_tracking", "tracking_state": "parked",
            "vehicle_overlap": 0.8, "stopped_for_ms": 1000,
            "recovery_state": "none", "recovery_global_id": None,
            "recovery_age_ms": 0, "recovery_radius_px": 0.0,
            "recovery_candidate_count": 0,
        }],
        "gid_aliases": [],
        "identity_events": [{
            "event_uid": "identity-1", "source": "global_manager",
            "event_type": "global_id_created", "frame_idx": 1,
            "canonical_gid": 7, "raw_gid": 7, "details": {},
        }],
        "parking_events": [], "parking_recovery": [],
        "parked_identity_reservations": [{
            "canonical_gid": 7, "camera_id": "cam1", "slot_id": "A01",
            "state": "parked", "bbox": [0, 0, 10, 10],
        }],
    }
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8",
    )
    _write_csv(tmp_path / "frame_timestamps.csv", ["frame_idx"], [[1]])
    _write_csv(tmp_path / "performance.csv", ["frame_idx"], [[1]])
    _write_csv(tmp_path / "ground_truth_slots.csv", [
        "schema_version", "camera_id", "slot_id", "start_frame", "end_frame",
        "occupied", "physical_vehicle_id", "identity_required", "notes",
    ], [[3, "cam1", "A01", 1, 1, "true", "V1", "true", ""]])
    _write_csv(tmp_path / "ground_truth_events.csv", [
        "schema_version", "event_id", "physical_vehicle_id", "event_type",
        "start_frame", "end_frame", "source_camera", "target_camera",
        "source_slot_id", "target_slot_id", "preferred_delay_frames",
        "max_delay_frames", "required", "critical", "notes",
    ], [[3, "E1", "V1", "parked", 1, 1, "cam1", "cam1", "", "A01", 0, 75, "true", "true", ""]])
    _write_csv(tmp_path / "ground_truth_identity.csv", [
        "schema_version", "observation_id", "physical_vehicle_id", "frame_idx",
        "camera_id", "anchor_x", "anchor_y", "slot_id", "phase", "required", "notes",
    ], [[3, "O1", "V1", 1, "cam1", "", "", "A01", "parked", "true", ""]])
    return prediction


def test_schema_three_validates_flat_predictions_and_all_ground_truth_files(tmp_path):
    _valid_v3_session(tmp_path)

    errors, counts = validate(tmp_path)

    assert errors == []
    assert counts["predictions"] == 1


def test_schema_three_rejects_duplicate_event_uid(tmp_path):
    prediction = _valid_v3_session(tmp_path)
    prediction["parking_events"] = [{
        **prediction["identity_events"][0], "source": "slot_binder:cam1",
    }]
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8",
    )

    errors, _ = validate(tmp_path)

    assert any("event_uid" in error and "bi lap" in error for error in errors)
