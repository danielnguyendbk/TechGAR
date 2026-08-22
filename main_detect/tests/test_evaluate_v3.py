import csv
import json

import pytest

from evaluate import _render_aggregate_report
from techgar.evaluation_v3 import (
    EVENT_COLUMNS,
    IDENTITY_COLUMNS,
    SLOT_COLUMNS,
    EvaluationValidationError,
    EvaluatorConfig,
    aggregate_results,
    evaluate_session,
    render_markdown_report,
)


def _write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _slot_gt(start, end, occupied, *, physical="", required=False, slot="A01", camera="cam1"):
    return {
        "schema_version": 3,
        "camera_id": camera,
        "slot_id": slot,
        "start_frame": start,
        "end_frame": end,
        "occupied": str(bool(occupied)).lower(),
        "physical_vehicle_id": physical,
        "identity_required": str(bool(required)).lower(),
        "notes": "",
    }


def _identity_gt(observation_id, physical, frame, camera="cam1", *, x=10, y=10, slot="", phase="moving", required=True):
    return {
        "schema_version": 3,
        "observation_id": observation_id,
        "physical_vehicle_id": physical,
        "frame_idx": frame,
        "camera_id": camera,
        "anchor_x": "" if slot else x,
        "anchor_y": "" if slot else y,
        "slot_id": slot,
        "phase": phase,
        "required": str(bool(required)).lower(),
        "notes": "",
    }


def _event_gt(event_id, physical, event_type, start, end=None, *, source="", target="", source_slot="", target_slot="", preferred=0, maximum=5, required=True, critical=True):
    return {
        "schema_version": 3,
        "event_id": event_id,
        "physical_vehicle_id": physical,
        "event_type": event_type,
        "start_frame": start,
        "end_frame": start if end is None else end,
        "source_camera": source,
        "target_camera": target,
        "source_slot_id": source_slot,
        "target_slot_id": target_slot,
        "preferred_delay_frames": preferred,
        "max_delay_frames": maximum,
        "required": str(bool(required)).lower(),
        "critical": str(bool(critical)).lower(),
        "notes": "",
    }


def _observation(frame, camera="cam1", gid=1, *, x=10, y=10, local=1, aliases=None, invisible=0, owner=None):
    return {
        "observation_uid": f"frame-{frame}:{camera}:{local}",
        "camera_id": camera,
        "local_track_id": local,
        "raw_gid": gid,
        "canonical_gid": gid,
        "gid_aliases": list(aliases or ([] if gid is None else [gid])),
        "bbox": [x - 5, y - 5, 10, 10],
        "anchor_pixel": {"x": x, "y": y, "reference": "tracker_center"},
        "anchor_world": None,
        "track_state": "confirmed",
        "association_state": "matched",
        "invisible_count": invisible,
        "assignment_cost": {},
        "fragment_visible_count": frame,
        "first_observation_frame": 1,
        "identity_state": "active" if gid is not None else "unassigned",
        "slot_ownership": owner,
    }


def _slot_prediction(occupied=False, gid=None, *, camera="cam1", slot="A01"):
    return {
        "camera_id": camera,
        "slot_id": slot,
        "occupied": occupied,
        "raw_vehicle_gid": gid,
        "canonical_vehicle_gid": gid,
        "vision_occupied": occupied,
        "tracking_occupied": occupied,
        "decision_source": "test",
        "tracking_state": "occupied" if occupied else "free",
        "vehicle_overlap": 1.0 if occupied else 0.0,
        "stopped_for_ms": 0,
        "recovery_state": "none",
        "recovery_global_id": None,
        "recovery_age_ms": 0,
        "recovery_radius_px": 0.0,
        "recovery_candidate_count": 0,
    }


def _prediction_event(uid, event_type, frame, gid, *, parking=False, camera="cam1", details=None):
    result = {
        "event_uid": uid,
        "source": f"slot_binder:{camera}" if parking else "global_manager",
        "event_type": event_type,
        "frame_idx": frame,
        "canonical_gid": gid,
        "raw_gid": gid,
        "details": dict(details or {}),
    }
    if parking:
        result["camera_id"] = camera
    return result


def _reservation(gid, *, camera="cam1", slot="A01"):
    return {
        "canonical_gid": gid,
        "camera_id": camera,
        "slot_id": slot,
        "state": "parked",
        "bbox": [0, 0, 10, 10],
    }


def _frame(frame, *, observations=None, slots=None, aliases=None, identity_events=None, parking_events=None, reservations=None):
    return {
        "schema_version": 3,
        "frame_idx": frame,
        "capture_unix_ns": frame,
        "wall_time_iso": "2026-08-22T00:00:00+07:00",
        "camera_timestamps_ns": {"cam1": frame, "cam2": frame},
        "camera_skew_ms": 0.0,
        "observations": list(observations or []),
        "slots": [_slot_prediction()] if slots is None else list(slots),
        "gid_aliases": list(aliases or []),
        "identity_events": list(identity_events or []),
        "parking_events": list(parking_events or []),
        "parking_recovery": [],
        "parked_identity_reservations": list(reservations or []),
    }


def _session(tmp_path, frames, *, slots=None, events=None, identities=None):
    _write_csv(
        tmp_path / "ground_truth_slots.csv",
        SLOT_COLUMNS,
        slots if slots is not None else [_slot_gt(frames[0]["frame_idx"], frames[-1]["frame_idx"], False)],
    )
    _write_csv(tmp_path / "ground_truth_events.csv", EVENT_COLUMNS, events or [])
    _write_csv(tmp_path / "ground_truth_identity.csv", IDENTITY_COLUMNS, identities or [])
    with (tmp_path / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(frame) + "\n")
    (tmp_path / "session_info.json").write_text(
        json.dumps({
            "schema_version": 3,
            "processed_frames": len(frames),
        }),
        encoding="utf-8",
    )
    return tmp_path


FAST_CONFIG = EvaluatorConfig(phantom_max_frames=1)


def test_physical_label_does_not_need_to_equal_numeric_gid(tmp_path):
    frames = [
        _frame(1, observations=[_observation(1, gid=17)]),
        _frame(2, observations=[_observation(2, gid=17)]),
    ]
    session = _session(
        tmp_path,
        frames,
        identities=[_identity_gt("cp1", "CAR_BLUE", 1), _identity_gt("cp2", "CAR_BLUE", 2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_BLUE": 17}
    assert result["scores"]["identity_continuity_handoff"] == 100.0
    assert result["critical_error_count"] == 0


def test_alias_merge_is_canonicalized_before_identity_scoring(tmp_path):
    first = _observation(1, gid=2, aliases=[2])
    second = _observation(2, gid=1, aliases=[1, 2])
    frames = [
        _frame(1, observations=[first]),
        _frame(2, observations=[second], aliases=[{"alias_gid": 2, "canonical_gid": 1}]),
    ]
    session = _session(
        tmp_path,
        frames,
        identities=[_identity_gt("before", "CAR_A", 1), _identity_gt("after", "CAR_A", 2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_A": 1}
    assert result["critical_error_count"] == 0
    assert result["scores"]["identity_continuity_handoff"] == 100.0


def test_directed_alias_can_keep_a_larger_canonical_gid(tmp_path):
    observation = _observation(1, gid=7, aliases=[1, 7])
    observation["raw_gid"] = 1
    frames = [_frame(
        1,
        observations=[observation],
        aliases=[{"alias_gid": 1, "canonical_gid": 7}],
    )]
    session = _session(
        tmp_path, frames, identities=[_identity_gt("cp", "CAR_A", 1)]
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_A": 7}


@pytest.mark.parametrize("aliases, message", [
    ([{"alias_gid": 1, "canonical_gid": 2}, {"alias_gid": 2, "canonical_gid": 1}], "alias cycle"),
    ([{"alias_gid": 1, "canonical_gid": 2}, {"alias_gid": 1, "canonical_gid": 3}], "conflicting canonical"),
])
def test_directed_alias_cycle_or_conflict_is_rejected(tmp_path, aliases, message):
    session = _session(tmp_path, [_frame(1, aliases=aliases)])

    with pytest.raises(EvaluationValidationError, match=message):
        evaluate_session(session, config=FAST_CONFIG, write_outputs=False)


def test_optional_checkpoint_cannot_change_mapping_or_create_collision(tmp_path):
    frames = [
        _frame(1, observations=[_observation(1, gid=1, x=100, y=100)]),
        _frame(2, observations=[
            _observation(2, gid=1, x=10, y=10, local=1),
            _observation(2, gid=2, x=100, y=100, local=2),
        ]),
    ]
    session = _session(
        tmp_path,
        frames,
        identities=[
            _identity_gt("optional", "CAR_B", 1, x=100, y=100, required=False),
            _identity_gt("a", "CAR_A", 2, x=10, y=10),
            _identity_gt("b", "CAR_B", 2, x=100, y=100),
        ],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_A": 1, "CAR_B": 2}
    assert not any(item["code"] == "gid_shared_between_vehicles" for item in result["errors"])


def test_same_gid_for_two_physical_vehicles_is_critical(tmp_path):
    frames = [_frame(1, observations=[
        _observation(1, gid=5, x=10, y=10, local=1),
        _observation(1, gid=5, x=100, y=100, local=2),
    ])]
    session = _session(
        tmp_path,
        frames,
        identities=[
            _identity_gt("a", "CAR_A", 1, x=10, y=10),
            _identity_gt("b", "CAR_B", 1, x=100, y=100),
        ],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    collision = [
        item for item in result["critical_errors"]
        if item["code"] == "gid_shared_between_vehicles"
    ]
    assert collision and collision[0]["predicted_gid"] == 5
    assert collision[0]["frame_idx"] == 1
    assert collision[0]["physical_vehicle_ids"] == ["CAR_A", "CAR_B"]
    assert result["classification"] == "FAIL"


def test_markdown_error_table_keeps_full_owner_and_collision_context():
    owner_error = {
        "code": "wrong_slot_owner",
        "severity": "critical",
        "message": "Owner | changed unexpectedly",
        "frame_idx": 9,
        "end_frame": 14,
        "physical_vehicle_id": "CAR_A",
        "camera_id": "cam1",
        "slot_id": "A01",
        "expected_gid": 1,
        "predicted_gid": 2,
        "duration_frames": 6,
    }
    collision_error = {
        "code": "gid_shared_between_vehicles",
        "severity": "critical",
        "message": "One canonical GID belongs to two vehicles",
        "frame_idx": 300,
        "end_frame": 601,
        "physical_vehicle_ids": ["M04_V1", "M04_V4"],
        "camera_ids": ["cam1", "cam2"],
        "predicted_gid": 3,
    }
    metric = {
        "occupied_f1": 0.9,
        "balanced_accuracy": 0.8,
        "false_free_rate": 0.1,
        "false_occupied_rate": 0.2,
    }
    result = {
        "session": "demo",
        "classification": "FAIL",
        "practical_system_score": 49.0,
        "uncapped_practical_system_score": 90.0,
        "critical_error_count": 2,
        "critical_errors": [owner_error, collision_error],
        "errors": [owner_error, collision_error],
        "scores": {
            "identity_continuity_handoff": 90.0,
            "slot_identity_ownership": 90.0,
            "departure_recovery": 90.0,
            "occupancy": 90.0,
            "delay_stability": 90.0,
        },
        "weights": {
            "identity_continuity_handoff": 0.35,
            "slot_identity_ownership": 0.30,
            "departure_recovery": 0.15,
            "occupancy": 0.15,
            "delay_stability": 0.05,
        },
        "occupancy": {"raw": metric, "practical": metric},
    }

    report = render_markdown_report(result)

    assert "| 9–14 | CAR_A | cam1 | A01 | 1 | 2 | 6 frames |" in report
    assert "Owner \\| changed unexpectedly" in report
    assert (
        "| 300–601 | M04_V1, M04_V4 | cam1, cam2 | n/a | n/a | 3 | n/a |"
        in report
    )


def test_aggregate_markdown_keeps_session_and_full_critical_context():
    result = {
        "classification": "FAIL",
        "practical_system_score": 49.0,
        "critical_error_count": 1,
        "sessions": ["m_04"],
        "scores": {
            "identity_continuity_handoff": 0.0,
            "slot_identity_ownership": 0.0,
            "departure_recovery": 0.0,
            "occupancy": 90.0,
            "delay_stability": 80.0,
        },
        "aggregation_units": {
            "identity_vehicle_lifecycles": 2,
            "slot_lifecycles": 1,
            "departure_events": 1,
            "occupancy_sessions": 1,
            "delay_sessions": 1,
        },
        "critical_errors": [{
            "session": "m_04",
            "code": "gid_shared_between_vehicles",
            "severity": "critical",
            "message": "One canonical GID belongs to two vehicles",
            "frame_idx": 300,
            "end_frame": 601,
            "physical_vehicle_ids": ["M04_V1", "M04_V4"],
            "camera_ids": ["cam1", "cam2"],
            "predicted_gid": 3,
            "duration_frames": 302,
        }],
    }

    report = _render_aggregate_report(result)

    assert "| Session | Code | Frame(s) | Physical vehicle(s) |" in report
    assert (
        "| m_04 | gid_shared_between_vehicles | 300–601 | "
        "M04_V1, M04_V4 | cam1, cam2 | n/a | n/a | 3 | 302 frames |"
        in report
    )


def test_correct_camera_handoff_scores_full_identity(tmp_path):
    frames = [
        _frame(1, observations=[_observation(1, "cam1", 7)]),
        _frame(2, observations=[_observation(2, "cam1", 7)]),
        _frame(3, observations=[_observation(3, "cam2", 7, x=20, y=20)]),
        _frame(4, observations=[_observation(4, "cam2", 7, x=20, y=20)]),
    ]
    session = _session(
        tmp_path,
        frames,
        identities=[
            _identity_gt("source", "CAR_A", 1),
            _identity_gt("target", "CAR_A", 3, camera="cam2", x=20, y=20),
        ],
        events=[_event_gt("h1", "CAR_A", "camera_handoff", 2, 3, source="cam1", target="cam2", preferred=1, maximum=3)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["handoffs"][0]["success"] is True
    assert result["scores"]["identity_continuity_handoff"] == 100.0
    assert result["critical_error_count"] == 0


def test_wrong_gid_after_handoff_is_critical_and_caps_score(tmp_path):
    frames = [
        _frame(1, observations=[_observation(1, "cam1", 7)]),
        _frame(2, observations=[_observation(2, "cam1", 7)]),
        _frame(3, observations=[_observation(3, "cam1", 7)]),
        _frame(4, observations=[_observation(4, "cam2", 8, x=20, y=20)]),
    ]
    session = _session(
        tmp_path,
        frames,
        identities=[
            _identity_gt("s1", "CAR_A", 1),
            _identity_gt("s2", "CAR_A", 2),
            _identity_gt("s3", "CAR_A", 3),
            _identity_gt("target", "CAR_A", 4, camera="cam2", x=20, y=20),
        ],
        events=[_event_gt("h1", "CAR_A", "camera_handoff", 3, 4, source="cam1", target="cam2", maximum=2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "wrong_gid_at_checkpoint" for item in result["critical_errors"])
    assert result["classification"] == "FAIL"
    assert result["practical_system_score"] <= 49.0


def test_passing_vehicle_cannot_take_slot_owner_for_five_frames(tmp_path):
    frames = []
    for frame in range(1, 16):
        observations = [_observation(frame, gid=1)] if frame <= 3 else []
        owner = 1 if frame < 9 else (2 if frame <= 14 else 1)
        frames.append(_frame(
            frame,
            observations=observations,
            slots=[_slot_prediction(frame >= 4, owner if frame >= 4 else None)],
        ))
    session = _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 3, False),
            _slot_gt(4, 15, True, physical="CAR_PARKED", required=True),
        ],
        identities=[
            _identity_gt("cp1", "CAR_PARKED", 1),
            _identity_gt("cp2", "CAR_PARKED", 2),
            _identity_gt("cp3", "CAR_PARKED", 3),
        ],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    wrong = [item for item in result["critical_errors"] if item["code"] == "wrong_slot_owner"]
    assert wrong and wrong[0]["duration_frames"] == 6
    assert result["classification"] == "FAIL"


def test_parked_checkpoint_cannot_circularly_define_expected_gid(tmp_path):
    frames = [
        _frame(frame, slots=[_slot_prediction(True, 7 if frame <= 4 else 8)])
        for frame in range(1, 11)
    ]
    session = _session(
        tmp_path,
        frames,
        slots=[_slot_gt(1, 10, True, physical="CAR_A", required=True)],
        identities=[_identity_gt(
            "parked", "CAR_A", 2, slot="A01", phase="parked"
        )],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_A": None}
    assert result["scores"]["slot_identity_ownership"] == 0.0
    assert any(item["code"] == "unmapped_physical_vehicle" for item in result["errors"])
    assert result["critical_error_count"] == 0


def test_wrong_slot_reservation_for_five_frames_is_critical(tmp_path):
    frames = []
    for frame in range(1, 12):
        reservations = []
        if 6 <= frame <= 10:
            reservations = [{
                "canonical_gid": 2,
                "camera_id": "cam1",
                "slot_id": "A01",
                "state": "parked",
                "bbox": [0, 0, 10, 10],
            }]
        frames.append(_frame(
            frame,
            observations=[_observation(frame, gid=1)] if frame <= 2 else [],
            slots=[_slot_prediction(frame >= 3, 1 if frame >= 3 else None)],
            reservations=reservations,
        ))
    session = _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 2, False),
            _slot_gt(3, 11, True, physical="CAR_A", required=True),
        ],
        identities=[_identity_gt("cp1", "CAR_A", 1), _identity_gt("cp2", "CAR_A", 2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "wrong_slot_reservation" for item in result["critical_errors"])
    assert result["practical_system_score"] <= 49.0


def _departure_session(tmp_path, *, wrong_recovery_event=False):
    frames = []
    for frame in range(1, 21):
        observations = []
        if frame in (1, 2) or frame >= 10:
            observations = [_observation(frame, gid=1)]
        owner = 1 if frame <= 9 else None
        parking_events = []
        if frame == 10:
            parking_events.append(_prediction_event(
                "recover-1", "parked_id_recovered", frame,
                2 if wrong_recovery_event else 1,
                parking=True, details={"slot_id": "A01"},
            ))
        frames.append(_frame(
            frame,
            observations=observations,
            slots=[_slot_prediction(frame <= 9, owner)],
            parking_events=parking_events,
        ))
    return _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 9, True, physical="CAR_A", required=True),
            _slot_gt(10, 20, False),
        ],
        identities=[_identity_gt("cp1", "CAR_A", 1), _identity_gt("cp2", "CAR_A", 2)],
        events=[_event_gt(
            "depart-1", "CAR_A", "departure_started", 10,
            source="cam1", source_slot="A01", preferred=1, maximum=5,
        )],
    )


def test_departure_recovery_requires_same_gid_and_cleared_slot(tmp_path):
    result = evaluate_session(
        _departure_session(tmp_path), config=FAST_CONFIG, write_outputs=False
    )

    event = result["departure_recovery"]["events"][0]
    assert event["recovered_correct_gid"] is True
    assert event["slot_cleared"] is True
    assert result["scores"]["departure_recovery"] == 100.0
    assert result["critical_error_count"] == 0


def test_wrong_departure_recovery_event_is_critical(tmp_path):
    result = evaluate_session(
        _departure_session(tmp_path, wrong_recovery_event=True),
        config=FAST_CONFIG,
        write_outputs=False,
    )

    assert any(
        item["code"] == "wrong_departure_recovery_gid"
        for item in result["critical_errors"]
    )
    assert result["classification"] == "FAIL"


def test_still_parked_track_is_not_counted_as_departure_recovery(tmp_path):
    owner = {"camera_id": "cam1", "slot_id": "A01", "state": "occupied"}
    frames = [
        _frame(
            frame,
            observations=[_observation(frame, gid=1, owner=owner)],
            slots=[_slot_prediction(True, 1)],
        )
        for frame in range(1, 16)
    ]
    session = _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 9, True, physical="CAR_A", required=True),
            _slot_gt(10, 15, False),
        ],
        identities=[_identity_gt("cp1", "CAR_A", 1)],
        events=[_event_gt(
            "depart-1", "CAR_A", "departure_started", 10,
            source="cam1", source_slot="A01", maximum=5,
        )],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    recovery = result["departure_recovery"]["events"][0]
    assert recovery["recovered_correct_gid"] is False
    assert recovery["slot_cleared"] is False
    assert result["scores"]["departure_recovery"] == 0.0
    assert any(item["code"] == "missed_departure_recovery" for item in result["errors"])


def test_departure_outside_a_slot_is_not_scored_as_parking_recovery(tmp_path):
    frames = [_frame(1, observations=[_observation(1, gid=1)])]
    session = _session(
        tmp_path,
        frames,
        identities=[_identity_gt("cp", "CAR_A", 1)],
        events=[_event_gt(
            "move-from-waiting", "CAR_A", "departure_started", 1,
            source="cam1", source_slot="", maximum=5,
        )],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["departure_recovery"]["events"] == []
    assert result["scores"]["departure_recovery"] is None


def _recovery_clear_session(tmp_path, signal, *, stable_clear):
    frames = []
    signal_frame = 12 if stable_clear else 10
    for frame in range(1, 15):
        if frame <= 9:
            owner = 1
        elif stable_clear:
            owner = None
        else:
            owner = 1 if frame == 11 else None
        observations = []
        if frame == 1:
            observations.append(_observation(frame, gid=1))
        if frame == signal_frame and signal in {"same", "cross"}:
            observations.append(_observation(
                frame,
                camera="cam1" if signal == "same" else "cam2",
                gid=1,
                x=20,
                y=20,
                local=2,
            ))
        parking_events = []
        if frame == signal_frame and signal == "event":
            parking_events.append(_prediction_event(
                "recover", "parked_id_recovered", frame, 1,
                parking=True, details={"slot_id": "A01"},
            ))
        frames.append(_frame(
            frame,
            observations=observations,
            slots=[_slot_prediction(frame <= 9, owner)],
            parking_events=parking_events,
        ))
    return _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 9, True, physical="CAR_A", required=True),
            _slot_gt(10, 14, False),
        ],
        identities=[_identity_gt("cp", "CAR_A", 1)],
        events=[_event_gt(
            "depart", "CAR_A", "departure_started", 10,
            source="cam1", source_slot="A01", preferred=0, maximum=4,
        )],
    )


@pytest.mark.parametrize("signal", ["same", "cross", "event"])
def test_recovery_signal_requires_three_frame_source_slot_clear(tmp_path, signal):
    result = evaluate_session(
        _recovery_clear_session(tmp_path, signal, stable_clear=False),
        config=FAST_CONFIG,
        write_outputs=False,
    )

    recovery = result["departure_recovery"]["events"][0]
    assert recovery["slot_cleared_frame"] == 14
    assert recovery["recovered_correct_gid"] is False


@pytest.mark.parametrize("signal", ["same", "cross", "event"])
def test_recovery_signal_after_sustained_clear_is_accepted(tmp_path, signal):
    result = evaluate_session(
        _recovery_clear_session(tmp_path, signal, stable_clear=True),
        config=FAST_CONFIG,
        write_outputs=False,
    )

    recovery = result["departure_recovery"]["events"][0]
    assert recovery["slot_cleared_frame"] == 12
    assert recovery["recovered_frame"] == 12
    assert recovery["recovered_correct_gid"] is True


def test_null_gid_loses_identity_points_but_is_not_critical(tmp_path):
    frames = [_frame(1, observations=[_observation(1, gid=None)])]
    session = _session(
        tmp_path, frames, identities=[_identity_gt("cp", "CAR_A", 1)]
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert result["scores"]["identity_continuity_handoff"] == 0.0
    assert result["critical_error_count"] == 0


def test_short_unbound_phantom_gid_is_ignored(tmp_path):
    frames = []
    for frame in range(1, 31):
        observations = [_observation(frame, gid=9)] if frame <= 3 else []
        frames.append(_frame(frame, observations=observations))
    session = _session(
        tmp_path, frames, identities=[_identity_gt("cp", "CAR_A", 2)]
    )

    result = evaluate_session(session, write_outputs=False)

    assert result["identity"]["phantom_gids_ignored"] == [9]
    assert result["identity"]["physical_to_canonical_gid"] == {"CAR_A": None}
    assert result["critical_error_count"] == 0


def _flicker_session(tmp_path, *, owner_lost):
    frames = []
    for frame in range(1, 21):
        occupied = not 6 <= frame <= 10
        owner = None if owner_lost and not occupied else 1
        frames.append(_frame(
            frame,
            slots=[_slot_prediction(occupied, owner)],
        ))
    return _session(
        tmp_path,
        frames,
        slots=[_slot_gt(1, 20, True)],
    )


def test_short_occupancy_flicker_is_ignored_when_owner_is_stable(tmp_path):
    result = evaluate_session(
        _flicker_session(tmp_path, owner_lost=False),
        config=FAST_CONFIG,
        write_outputs=False,
    )

    assert result["occupancy"]["raw"]["fn"] == 5
    assert result["occupancy"]["practical"]["fn"] == 0
    assert len(result["occupancy"]["ignored_short_flickers"]) == 1


def test_flicker_that_loses_slot_owner_is_not_ignored(tmp_path):
    result = evaluate_session(
        _flicker_session(tmp_path, owner_lost=True),
        config=FAST_CONFIG,
        write_outputs=False,
    )

    assert result["occupancy"]["practical"]["fn"] == 5
    assert result["occupancy"]["ignored_short_flickers"] == []
    assert len(result["occupancy"]["harmful_flickers"]) == 1


def test_binding_on_initial_gt_free_slot_is_critical(tmp_path):
    frames = [
        _frame(frame, slots=[_slot_prediction(False, 5)])
        for frame in range(1, 7)
    ]
    session = _session(tmp_path, frames, slots=[_slot_gt(1, 6, False)])

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "binding_on_gt_free_slot" for item in result["critical_errors"])


def test_departed_expected_gid_on_free_slot_is_stale_not_critical(tmp_path):
    frames = []
    for frame in range(1, 11):
        frames.append(_frame(
            frame,
            observations=[_observation(frame, gid=1)] if frame <= 2 else [],
            slots=[_slot_prediction(frame <= 3, 1)],
        ))
    session = _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 3, True, physical="CAR_A", required=True),
            _slot_gt(4, 10, False),
        ],
        identities=[_identity_gt("cp1", "CAR_A", 1), _identity_gt("cp2", "CAR_A", 2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "stale_release_binding" for item in result["errors"])
    assert not any(item["code"] == "binding_on_gt_free_slot" for item in result["critical_errors"])
    release = [item for item in result["delay_stability"]["delay_items"] if item["kind"] == "slot_release"]
    assert release and release[0]["delay_frames"] is None
    assert result["delay_stability"]["stale_release_penalty"] == 1.0


def test_unrelated_gid_after_departure_on_free_slot_is_critical(tmp_path):
    frames = []
    for frame in range(1, 11):
        owner = 1 if frame <= 3 else 2
        frames.append(_frame(
            frame,
            observations=[_observation(frame, gid=1)] if frame <= 2 else [],
            slots=[_slot_prediction(frame <= 3, owner)],
        ))
    session = _session(
        tmp_path,
        frames,
        slots=[
            _slot_gt(1, 3, True, physical="CAR_A", required=True),
            _slot_gt(4, 10, False),
        ],
        identities=[_identity_gt("cp1", "CAR_A", 1), _identity_gt("cp2", "CAR_A", 2)],
    )

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "binding_on_gt_free_slot" for item in result["critical_errors"])


def test_reservation_on_initial_gt_free_slot_is_critical(tmp_path):
    frames = [
        _frame(frame, reservations=[_reservation(5)])
        for frame in range(1, 7)
    ]
    session = _session(tmp_path, frames, slots=[_slot_gt(1, 6, False)])

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    assert any(item["code"] == "reservation_on_gt_free_slot" for item in result["critical_errors"])


def test_different_gid_duplicate_reservations_are_critical_after_five_frames(tmp_path):
    frames = [
        _frame(frame, reservations=[_reservation(1), _reservation(2)])
        for frame in range(1, 6)
    ]
    session = _session(tmp_path, frames, slots=[_slot_gt(1, 5, False)])

    result = evaluate_session(session, config=FAST_CONFIG, write_outputs=False)

    duplicate = [item for item in result["critical_errors"] if item["code"] == "duplicate_slot_reservations"]
    assert duplicate and duplicate[0]["predicted_gids"] == [1, 2]


def test_long_unbound_gid_is_reported_and_lightly_penalizes_stability(tmp_path):
    frames = [
        _frame(frame, observations=[_observation(frame, gid=9)])
        for frame in range(1, 26)
    ]
    session = _session(tmp_path, frames)

    result = evaluate_session(session, write_outputs=False)

    assert result["identity"]["phantom_gids_ignored"] == []
    assert result["identity"]["long_unbound_gids"][0]["canonical_gid"] == 9
    assert any(item["code"] == "long_unbound_gid" for item in result["errors"])
    assert result["delay_stability"]["long_unbound_gid_penalty"] == 2.0
    assert result["delay_stability"]["stability_score"] == 98.0


def test_schema_two_prediction_is_rejected(tmp_path):
    _write_csv(tmp_path / "ground_truth_slots.csv", SLOT_COLUMNS, [])
    _write_csv(tmp_path / "ground_truth_events.csv", EVENT_COLUMNS, [])
    _write_csv(tmp_path / "ground_truth_identity.csv", IDENTITY_COLUMNS, [])
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"schema_version": 2, "frame_idx": 1}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "session_info.json").write_text(
        json.dumps({"schema_version": 3, "processed_frames": 1}),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="only schema 3"):
        evaluate_session(tmp_path, write_outputs=False)


def test_prediction_frames_must_be_contiguous_from_one(tmp_path):
    session = _session(tmp_path, [_frame(2)])

    with pytest.raises(EvaluationValidationError, match="contiguous from 1"):
        evaluate_session(session, write_outputs=False)


def test_session_info_must_match_prediction_count(tmp_path):
    session = _session(tmp_path, [_frame(1), _frame(2)])
    (session / "session_info.json").write_text(
        json.dumps({"schema_version": 3, "processed_frames": 3}),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationValidationError, match="predictions contain 2"):
        evaluate_session(session, write_outputs=False)


def test_slot_prefix_warmup_up_to_ten_frames_is_excluded(tmp_path):
    frames = [
        _frame(frame, slots=[] if frame <= 5 else [_slot_prediction()])
        for frame in range(1, 9)
    ]
    session = _session(
        tmp_path, frames, slots=[_slot_gt(1, 8, False)]
    )

    result = evaluate_session(session, write_outputs=False)

    assert result["occupancy"]["missing_labeled_slot_frames_excluded"] == 5
    assert result["occupancy"]["raw"]["total_slot_frames"] == 3


def test_slot_missing_after_first_appearance_is_rejected(tmp_path):
    frames = [
        _frame(1),
        _frame(2),
        _frame(3, slots=[]),
        _frame(4),
    ]
    session = _session(tmp_path, frames, slots=[_slot_gt(1, 4, False)])

    with pytest.raises(EvaluationValidationError, match="missing after warm-up"):
        evaluate_session(session, write_outputs=False)


def test_slot_prefix_warmup_over_ten_frames_is_rejected(tmp_path):
    frames = [
        _frame(frame, slots=[] if frame <= 11 else [_slot_prediction()])
        for frame in range(1, 13)
    ]
    session = _session(tmp_path, frames, slots=[_slot_gt(1, 12, False)])

    with pytest.raises(EvaluationValidationError, match="exceeds 10 frames"):
        evaluate_session(session, write_outputs=False)


def test_outputs_and_lifecycle_aggregate_are_written_and_unweighted(tmp_path):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = evaluate_session(
        _session(first_dir, [_frame(1)]), config=FAST_CONFIG, write_outputs=True
    )
    second = evaluate_session(
        _session(second_dir, [_frame(1), _frame(2)]),
        config=FAST_CONFIG,
        write_outputs=True,
    )

    aggregate = aggregate_results([first, second])

    assert (first_dir / "evaluation_results_v3.json").exists()
    assert (first_dir / "evaluation_report_v3.md").exists()
    assert aggregate["aggregation_units"]["occupancy_sessions"] == 2
    assert aggregate["sessions"] == ["one", "two"]
