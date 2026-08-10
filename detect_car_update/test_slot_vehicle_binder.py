from types import SimpleNamespace

import numpy as np

from slot_vehicle_binder import SlotVehicleBinder


def slot_result(slot_id="P001", occupied=False, x=0, y=0, size=100):
    return SimpleNamespace(
        slot_id=slot_id,
        occupied=occupied,
        polygon=np.asarray(
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size]],
            dtype=np.int32,
        ),
        center=(x + size // 2, y + size // 2),
        vehicle_id=None,
    )


def track(x=20, y=20, w=60, h=60):
    return {"bbox": (x, y, w, h), "appearance": None}


def settle(binder, global_id=2, start=0.0, x=20, slot_seconds=1.2):
    frame = 1
    timestamp = start
    while timestamp <= start + slot_seconds:
        binder.update_tracks({global_id: track(x=x)}, frame, timestamp)
        frame += 1
        timestamp += 1.0 / 30.0
    return frame, timestamp


def test_tracking_can_override_false_free_vision_after_one_second():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    result = slot_result(occupied=False)
    binder.update_vision([result], 0, 0.0)

    settle(binder)

    state = binder.get_slot_state("P001")
    assert state["vision_occupied"] is False
    assert state["tracking_occupied"] is True
    assert state["occupied"] is True
    assert state["vehicle_id"] == 2
    assert state["decision_source"] == "tracking_override"
    assert result.occupied is True
    assert result.vehicle_id == 2


def test_visual_occupancy_remains_primary_without_track_id():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)

    state = binder.get_slot_state("P001")
    assert state["occupied"] is True
    assert state["vehicle_id"] is None
    assert state["decision_source"] == "vision"


def test_vehicle_passing_through_slot_does_not_bind():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0)

    for frame in range(46):
        x = -50 + frame * 4
        binder.update_tracks({7: track(x=x)}, frame + 1, frame / 30.0)

    state = binder.get_slot_state("P001")
    assert state["tracking_occupied"] is False
    assert state["vehicle_id"] is None


def test_parked_binding_survives_track_disappearance():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0)
    frame, timestamp = settle(binder, global_id=3)

    for _ in range(180):
        binder.update_tracks({}, frame, timestamp)
        frame += 1
        timestamp += 1.0 / 30.0

    state = binder.get_slot_state("P001")
    assert state["tracking_occupied"] is True
    assert state["vehicle_id"] == 3


def test_same_id_outside_roi_releases_only_tracking_override():
    binder = SlotVehicleBinder(stop_seconds=1.0, exit_seconds=0.5)
    result = slot_result(occupied=False)
    binder.update_vision([result], 0, 0.0)
    frame, timestamp = settle(binder, global_id=4)

    for _ in range(20):
        binder.update_tracks({4: track(x=150)}, frame, timestamp)
        frame += 1
        timestamp += 1.0 / 30.0

    state = binder.get_slot_state("P001")
    assert state["tracking_occupied"] is False
    assert state["occupied"] is False
    assert state["vehicle_id"] is None


def test_vision_red_stays_red_after_tracked_vehicle_leaves():
    binder = SlotVehicleBinder(stop_seconds=1.0, exit_seconds=0.5)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)
    frame, timestamp = settle(binder, global_id=5)

    for _ in range(20):
        binder.update_tracks({5: track(x=150)}, frame, timestamp)
        frame += 1
        timestamp += 1.0 / 30.0

    state = binder.get_slot_state("P001")
    assert state["tracking_occupied"] is False
    assert state["vision_occupied"] is True
    assert state["occupied"] is True
    assert state["decision_source"] == "vision"


def test_batch_assignment_allows_only_one_vehicle_per_slot():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0)

    for frame in range(38):
        binder.update_tracks(
            {10: track(x=18), 11: track(x=22)},
            frame + 1,
            frame / 30.0,
        )

    parked = binder.get_all_parked_vehicle_ids()
    assert len(parked) == 1
    assert binder.get_slot_state("P001")["vehicle_id"] in {10, 11}


def test_new_detection_inside_parked_slot_recovers_old_global_id():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0, camera_id="cam3")
    settle(binder, global_id=12)

    recovered = binder.try_recover_id(
        camera_id="cam3",
        bbox=(25, 25, 55, 55),
        position=(50, 50),
    )
    assert recovered == 12


def test_global_id_merge_remaps_parked_binding():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0)
    settle(binder, global_id=4)

    binder.remap_vehicle_ids(lambda global_id: 2 if global_id == 4 else global_id)

    assert binder.get_slot_for_vehicle(4) is None
    assert binder.get_slot_for_vehicle(2) == "P001"
    assert binder.get_slot_state("P001")["vehicle_id"] == 2


def test_global_id_merge_cannot_leave_canonical_id_in_two_slots():
    binder = SlotVehicleBinder(stop_seconds=1.0)
    left = slot_result("P001", occupied=False, x=0)
    right = slot_result("P002", occupied=False, x=120)
    binder.update_vision([left, right], 0, 0.0)

    for frame in range(38):
        binder.update_tracks(
            {2: track(x=20), 4: track(x=140)},
            frame + 1,
            frame / 30.0,
        )
    assert binder.get_slot_state("P001")["vehicle_id"] == 2
    assert binder.get_slot_state("P002")["vehicle_id"] == 4

    binder.remap_vehicle_ids(lambda global_id: 2 if global_id == 4 else global_id)

    assigned = [
        slot_id
        for slot_id in ("P001", "P002")
        if binder.get_slot_state(slot_id)["vehicle_id"] == 2
    ]
    assert len(assigned) == 1
    assert binder.get_slot_for_vehicle(2) == assigned[0]
