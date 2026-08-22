import json

from techgar.prediction_writer import PREDICTION_SCHEMA_VERSION, PredictionV3Builder


class FakeManager:
    def canonical_global_id(self, global_id):
        return {9: 4}.get(int(global_id), int(global_id))


class FakeTracker:
    def __init__(self):
        self.association_events = [{
            "type": "local_reacquired",
            "local_track_id": 7,
            "cost": 0.12,
        }]

    def local_track_telemetry(self, global_ids):
        return [{
            "local_track_id": 7,
            "global_id": global_ids.get(7),
            "bbox": [10, 20, 30, 40],
            "center": [25, 40],
            "state": "confirmed",
            "invisible_count": 0,
            "association_state": "matched",
            "assignment_cost": {"total": 0.12},
            "fragment_visible_count": 8,
            "first_observation_frame": 2,
        }]


class FakeBinder:
    def __init__(self):
        self.events = [{
            "type": "vehicle_stopped_in_slot",
            "frame": 8,
            "global_id": 9,
            "slot_id": "A01",
        }]

    def to_json(self, camera_id=None):
        return {
            "A01": {
                "occupied": True,
                "vehicle_id": 9,
                "vision_occupied": True,
                "tracking_occupied": True,
                "decision_source": "vision_and_tracking",
                "tracking_state": "parked",
                "vehicle_overlap": 0.7,
                "stopped_for_ms": 1200,
                "recovery_state": "none",
                "recovery_age_ms": 0,
                "recovery_radius_px": 0,
            }
        }


def make_registry(manager_events):
    return {
        "world_unit": "cm",
        "retired_global_ids": {"9": 4},
        "active_global_vehicles": {
            "4": {
                "global_id": 4,
                "observations": [{
                    "camera_id": "cam1",
                    "local_track_id": 7,
                    "shared_map_anchor": {
                        "x": 25.0,
                        "y": 57.0,
                        "reference": "bottom_center",
                    },
                    "global_position": {"x": 100.5, "y": 70.25},
                }],
            }
        },
        "identity_lifecycle": {"4": {"state": "active"}},
        "recent_events": manager_events,
    }


def build(builder, tracker, binder, registry, frame_idx=10):
    return builder.build_frame(
        frame_idx=frame_idx,
        capture_unix_ns=123456,
        wall_time_iso="2026-08-22T12:00:00+07:00",
        camera_timestamps_ns={"cam1": 100, "cam2": 105},
        camera_skew_ms=0.000005,
        trackers={"cam1": tracker},
        global_ids={"cam1": {7: 9}},
        binders={"cam1": binder},
        manager=FakeManager(),
        registry=registry,
        parking_recovery=[],
        parked_identity_reservations={
            9: {
                "camera_id": "cam1",
                "slot_id": "A01",
                "state": "parked",
                "bbox": (10, 20, 30, 40),
                "appearance": object(),
            }
        },
    )


def test_schema_v3_flattens_canonical_identity_and_slot_ownership():
    builder = PredictionV3Builder()
    tracker = FakeTracker()
    binder = FakeBinder()
    manager_events = [{
        "type": "global_id_merged",
        "frame": 9,
        "global_id": 9,
        "kept_global_id": 4,
    }]

    payload = build(builder, tracker, binder, make_registry(manager_events))

    assert payload["schema_version"] == PREDICTION_SCHEMA_VERSION == 3
    assert "cameras" not in payload
    assert "global_registry" not in payload
    assert payload["gid_aliases"] == [{"alias_gid": 9, "canonical_gid": 4}]

    observation = payload["observations"][0]
    assert observation["raw_gid"] == 9
    assert observation["canonical_gid"] == 4
    assert observation["gid_aliases"] == [4, 9]
    assert observation["anchor_pixel"] == {
        "x": 25,
        "y": 40,
        "reference": "tracker_center",
    }
    assert observation["anchor_world"] == {
        "x": 100.5,
        "y": 70.25,
        "unit": "cm",
        "reference": "bottom_center",
    }
    assert observation["slot_ownership"] == {
        "camera_id": "cam1",
        "slot_id": "A01",
        "state": "occupied",
    }

    slot = payload["slots"][0]
    assert slot["raw_vehicle_gid"] == 9
    assert slot["canonical_vehicle_gid"] == 4
    assert payload["parked_identity_reservations"] == [{
        "canonical_gid": 4,
        "camera_id": "cam1",
        "slot_id": "A01",
        "state": "parked",
        "bbox": [10, 20, 30, 40],
    }]
    json.dumps(payload)


def test_rolling_manager_and_parking_events_are_written_once():
    builder = PredictionV3Builder()
    tracker = FakeTracker()
    binder = FakeBinder()
    manager_events = [{
        "type": "handoff_started",
        "frame": 9,
        "global_id": 9,
        "source_camera": "cam1",
        "target_camera": "cam2",
    }]
    registry = make_registry(manager_events)

    first = build(builder, tracker, binder, registry, frame_idx=10)
    tracker.association_events = []
    second = build(builder, tracker, binder, registry, frame_idx=11)

    assert len(first["identity_events"]) == 2  # manager + current tracker delta
    assert len(first["parking_events"]) == 1
    assert second["identity_events"] == []
    assert second["parking_events"] == []

    manager_events.append({
        "type": "handoff_matched",
        "frame": 12,
        "global_id": 9,
        "source_camera": "cam1",
        "target_camera": "cam2",
    })
    binder.events.append({
        "type": "vehicle_left_slot",
        "frame": 12,
        "global_id": 9,
        "slot_id": "A01",
    })
    third = build(builder, tracker, binder, registry, frame_idx=12)

    assert [event["event_type"] for event in third["identity_events"]] == [
        "handoff_matched"
    ]
    assert [event["event_type"] for event in third["parking_events"]] == [
        "vehicle_left_slot"
    ]
    event = third["identity_events"][0]
    assert event["canonical_gid"] == 4
    assert event["raw_gid"] == 9
    assert event["details"] == {
        "source_camera": "cam1",
        "target_camera": "cam2",
    }

    all_uids = {
        item["event_uid"]
        for payload in (first, second, third)
        for key in ("identity_events", "parking_events")
        for item in payload[key]
    }
    event_count = sum(
        len(payload[key])
        for payload in (first, second, third)
        for key in ("identity_events", "parking_events")
    )
    assert len(all_uids) == event_count


def test_rolling_delta_preserves_identical_distinct_events_after_truncation():
    builder = PredictionV3Builder()
    tracker = FakeTracker()
    tracker.association_events = []
    binder = FakeBinder()
    binder.events = []
    duplicate = {"type": "same", "frame": 1, "global_id": 4}
    events = [dict(duplicate), dict(duplicate)]
    registry = make_registry(events)

    first = build(builder, tracker, binder, registry, frame_idx=1)
    assert len(first["identity_events"]) == 2

    # Simulate a bounded history dropping the first event and appending a new,
    # byte-identical occurrence. The suffix/prefix overlap emits exactly one.
    events.pop(0)
    events.append(dict(duplicate))
    second = build(builder, tracker, binder, registry, frame_idx=2)
    assert len(second["identity_events"]) == 1
    assert second["identity_events"][0]["event_type"] == "same"
