from types import SimpleNamespace

import numpy as np

from techgar.slot_vehicle_binder import SlotVehicleBinder, VehicleParkingState


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


def track(x=20, y=20, w=60, h=60, appearance=None, **extra):
    return {"bbox": (x, y, w, h), "appearance": appearance, **extra}


def appearance(bin_index=0):
    histogram = np.zeros((16, 16), dtype=np.float32)
    histogram.flat[bin_index] = 1.0
    return histogram


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


def test_vision_primary_track_expiry_cannot_auto_park_into_green_slot():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)
    timestamp = 0.0
    for frame in range(1, 20):
        binder.update_tracks({41: track()}, frame, timestamp)
        timestamp += 1.0 / 30.0

    # A raw green update arrives before the weak stop candidate expires.
    result.occupied = False
    binder.update_vision([result], 20, timestamp)
    binder.notify_track_expired(41, 21, timestamp + 0.01)

    state = binder.get_slot_state("P001")
    assert state["occupied"] is False
    assert state["tracking_occupied"] is False
    assert state["vehicle_id"] is None
    rejected = [
        event
        for event in binder.events
        if event.get("reason") == "vision_primary_slot_empty"
    ]
    assert rejected


def test_vision_primary_arrival_claim_binds_when_motion_stops_before_one_second():
    binder = SlotVehicleBinder(
        policy="vision_primary",
        arrival_min_samples=3,
        arrival_vision_confirmations=2,
    )
    result = slot_result(occupied=False)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    binder.update_tracks({1: track(x=-70)}, 1, 0.0, camera_id="cam1")
    for frame, (timestamp, x) in enumerate(
        ((0.10, -20), (0.20, -10), (0.30, 0)), start=2
    ):
        binder.update_tracks({1: track(x=x)}, frame, timestamp, camera_id="cam1")

    result.occupied = True
    binder.update_vision([result], 5, 0.40, camera_id="cam1")
    assert binder.notify_track_lost(1, 6, 0.45) is None
    binder.update_vision([result], 7, 0.90, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["vehicle_id"] == 1
    assert state["tracking_occupied"] is True
    assert binder.get_identity_reservations()[0]["global_id"] == 1
    assert any(
        event["type"] == "slot_arrival_claim_confirmed"
        for event in binder.events
    )


def test_stationary_noise_without_inward_trajectory_cannot_claim_red_slot():
    binder = SlotVehicleBinder(
        policy="vision_primary",
        arrival_min_samples=3,
        arrival_vision_confirmations=2,
    )
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)
    for frame, timestamp in enumerate((0.10, 0.20, 0.30), start=1):
        binder.update_tracks({9: track(x=20)}, frame, timestamp)
    binder.notify_track_lost(9, 4, 0.35)
    binder.update_vision([result], 5, 0.75)

    assert binder.get_slot_state("P001")["vehicle_id"] is None
    assert any(
        event.get("reason") == "no_inward_trajectory"
        for event in binder.events
    )


def test_observable_vehicle_leaving_roi_cannot_commit_arrival_claim():
    binder = SlotVehicleBinder(
        policy="vision_primary",
        arrival_min_samples=3,
        arrival_vision_confirmations=2,
        arrival_absence_seconds=0.20,
    )
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    binder.update_tracks({2: track(x=-60)}, 1, 0.0, camera_id="cam1")
    for frame, (timestamp, x) in enumerate(
        ((0.10, -20), (0.20, -10), (0.30, 0)), start=2
    ):
        binder.update_tracks({2: track(x=x)}, frame, timestamp, camera_id="cam1")

    # The same live track drives out of the ROI. This is not the motion-loss
    # event used as evidence that a vehicle stopped.
    binder.update_tracks({2: track(x=180)}, 5, 0.60, camera_id="cam1")
    binder.update_vision([result], 6, 0.70, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["vehicle_id"] is None
    assert state["tracking_occupied"] is False
    assert any(
        event.get("reason") == "vehicle_left_roi_before_track_lost"
        for event in binder.events
    )


def test_reidentified_fragment_during_lost_grace_cancels_parking_commit():
    binder = SlotVehicleBinder(
        policy="vision_primary",
        arrival_min_samples=3,
        arrival_vision_confirmations=2,
        arrival_lost_commit_delay_seconds=0.35,
    )
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    binder.update_tracks({2: track(x=-60)}, 1, 0.0, camera_id="cam1")
    for frame, (timestamp, x) in enumerate(
        ((0.10, -20), (0.20, -10), (0.30, 0)), start=2
    ):
        binder.update_tracks({2: track(x=x)}, frame, timestamp, camera_id="cam1")

    assert binder.notify_track_lost(2, 5, 0.32) is None
    # The local tracker fragmented, but dormant Re-ID restored the same GID
    # before the LOST commit grace elapsed.
    binder.update_tracks({2: track(x=5)}, 6, 0.50, camera_id="cam1")
    binder.update_vision([result], 7, 0.80, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["vehicle_id"] is None
    assert state["tracking_occupied"] is False
    assert any(
        event["type"] == "slot_arrival_claim_resumed"
        for event in binder.events
    )


def test_short_roi_overlap_gaps_keep_one_arrival_claim_until_lost():
    binder = SlotVehicleBinder(
        policy="vision_primary",
        arrival_min_samples=3,
        arrival_vision_confirmations=2,
        arrival_absence_seconds=0.75,
        arrival_lost_commit_delay_seconds=0.35,
    )
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    binder.update_tracks({2: track(x=-60)}, 1, 0.0, camera_id="cam1")
    binder.update_tracks({2: track(x=-15)}, 2, 0.10, camera_id="cam1")
    binder.update_tracks({2: track(x=-5)}, 3, 0.20, camera_id="cam1")
    # One short geometry miss must not split the arrival into unrelated
    # one-sample claims.
    binder.update_tracks({2: track(x=180)}, 4, 0.45, camera_id="cam1")
    binder.update_tracks({2: track(x=0)}, 5, 0.60, camera_id="cam1")
    assert binder.notify_track_lost(2, 6, 0.65) is None
    binder.update_vision([result], 7, 1.05, camera_id="cam1")

    assert binder.get_slot_state("P001")["vehicle_id"] == 2


def test_unbound_fragment_opens_only_its_parked_slot_predeparture_token():
    binder = SlotVehicleBinder(policy="vision_primary")
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    state = binder._vehicle_states.setdefault(2, VehicleParkingState(global_id=2))
    state.last_bbox = (20, 20, 60, 60)
    state.last_appearance = appearance(2)
    binder._bind_vehicle(2, "P001", 1, 0.8, 1000)

    protected = binder.prepare_predeparture_tokens(
        {("cam1", 7): track(x=20, appearance=appearance(2))},
        1.1,
        camera_id="cam1",
    )

    assert protected == {("cam1", 7)}
    token = binder.export_recovery_tokens(1.1)[0]
    assert token["global_id"] == 2
    assert token["predeparture"] is True


def test_legacy_track_expiry_keeps_existing_auto_park_behavior():
    binder = SlotVehicleBinder(policy="legacy", stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=False)], 0, 0.0)
    timestamp = 0.0
    for frame in range(1, 20):
        binder.update_tracks({41: track()}, frame, timestamp)
        timestamp += 1.0 / 30.0

    binder.notify_track_expired(41, 21, timestamp)

    assert binder.get_slot_state("P001")["vehicle_id"] == 41


def test_vision_primary_strong_overlap_dwell_binds_despite_centroid_jitter():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=True)], 0, 0.0)

    for index in range(36):
        # The car is stable with sparse 20px detector outliers.  r95 remains
        # too noisy for the strict stationary rule, while robust dwell should
        # retain the dominant stable center.
        binder.update_tracks(
            {51: track(x=28 if index % 7 == 0 else 8, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )

    state = binder.get_slot_state("P001")
    assert state["vehicle_id"] == 51
    assert state["tracking_occupied"] is True
    confirmed = [
        event
        for event in binder.events
        if event["type"] == "strong_overlap_dwell_confirmed"
    ]
    assert confirmed
    assert confirmed[-1]["dwell_ms"] >= 1150


def test_alternating_fresh_shadow_never_strong_binds_or_becomes_sticky():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)

    for index in range(40):
        binder.update_tracks(
            {53: track(x=0 if index % 2 == 0 else 20, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )
    assert binder.get_slot_state("P001")["vehicle_id"] is None

    binder.notify_track_expired(53, 41, 1.4)
    result.occupied = False
    for index in range(42, 60):
        binder.update_vision([result], index, 1.4 + (index - 41) / 30.0)

    state = binder.get_slot_state("P001")
    assert state["occupied"] is False
    assert state["tracking_occupied"] is False
    assert state["vehicle_id"] is None
    assert state["recovery_state"] == "none"


def test_lost_stale_bboxes_never_advance_dwell_or_stationary_commit():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)

    # Five real observations are below min_stop_samples.  Repeating the final
    # LOST bbox for another second must not create fresh parking evidence.
    for index in range(5):
        binder.update_tracks(
            {54: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=0)},
            index + 1,
            index / 30.0,
        )
    for index in range(5, 45):
        binder.update_tracks(
            {54: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=1)},
            index + 1,
            index / 30.0,
        )

    assert binder.get_slot_state("P001")["vehicle_id"] is None
    binder.notify_track_expired(54, 46, 1.55)
    result.occupied = False
    for index in range(47, 65):
        binder.update_tracks({}, index, 1.55 + (index - 46) / 30.0)
        binder.update_vision([result], index, 1.55 + (index - 46) / 30.0)

    state = binder.get_slot_state("P001")
    assert state["occupied"] is False
    assert state["tracking_occupied"] is False
    assert state["vehicle_id"] is None
    assert state["recovery_state"] == "none"


def test_fresh_stationary_windows_cannot_straddle_long_lost_gap():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0)
    frame = 1

    # Seven fresh observations cover only 0.20s.
    for timestamp in np.linspace(0.0, 0.20, 7):
        binder.update_tracks(
            {55: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=0)},
            frame,
            float(timestamp),
        )
        frame += 1

    # Twenty-two duplicated LOST observations bridge the apparent timeline to
    # 0.93s, but none of them may extend a stationary/dwell endpoint.
    for timestamp in np.linspace(0.23, 0.93, 22):
        binder.update_tracks(
            {55: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=1)},
            frame,
            float(timestamp),
        )
        frame += 1

    # Another seven fresh samples cover only 0.20s.  Without contiguous-suffix
    # filtering, last-first would incorrectly look like a 1.17s stop.
    for timestamp in np.linspace(0.97, 1.17, 7):
        binder.update_tracks(
            {55: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=0)},
            frame,
            float(timestamp),
        )
        frame += 1

    assert binder.get_slot_state("P001")["vehicle_id"] is None
    binder.notify_track_expired(55, frame, 1.18)
    result.occupied = False
    binder.update_vision([result], frame + 1, 1.20)
    binder.update_tracks({}, frame + 2, 1.23)
    state = binder.get_slot_state("P001")
    assert state["occupied"] is False
    assert state["vehicle_id"] is None
    assert state["recovery_state"] == "none"
    assert binder.export_recovery_tokens(1.23) == []

    # A later genuinely continuous interval remains able to bind normally.
    result.occupied = True
    binder.update_vision([result], frame + 3, 1.30)
    for offset in range(39):
        binder.update_tracks(
            {55: track(x=8, y=10, w=80, h=80, consecutive_invisible_count=0)},
            frame + 4 + offset,
            1.30 + offset / 30.0,
        )
    assert binder.get_slot_state("P001")["vehicle_id"] == 55


def test_strong_overlap_dwell_resets_when_vehicle_passes_or_gid_changes():
    binder = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=True)], 0, 0.0)

    for index in range(18):
        binder.update_tracks(
            {51: track(x=index % 2 * 20, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )
    # No active observation for one frame resets continuous dwell.
    binder.update_tracks({}, 19, 19 / 30.0)
    for index in range(20, 38):
        binder.update_tracks(
            {52: track(x=index % 2 * 20, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )

    assert binder.get_slot_state("P001")["vehicle_id"] is None


def test_weak_overlap_jitter_and_missing_vision_do_not_bind():
    weak = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    weak.update_vision([slot_result(occupied=True)], 0, 0.0)
    for index in range(40):
        # Horizontal overlap is 55%; vertical jitter prevents the ordinary
        # stationary rule, and it never satisfies the 60% strong dwell.
        weak.update_tracks(
            {61: track(x=78, y=0 if index % 2 == 0 else 40, w=40, h=60)},
            index + 1,
            index / 30.0,
        )
    assert weak.get_slot_state("P001")["vehicle_id"] is None

    green = SlotVehicleBinder(policy="vision_primary", stop_seconds=1.0)
    green.update_vision([slot_result(occupied=False)], 0, 0.0)
    for index in range(40):
        green.update_tracks(
            {62: track(x=index % 2 * 20, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )
    assert green.get_slot_state("P001")["vehicle_id"] is None


def test_competing_gid_cannot_take_strong_overlap_occupied_binding():
    binder, _, frame, timestamp, _ = parked_vision_primary_binder(global_id=30)
    for index in range(40):
        binder.update_tracks(
            {31: track(x=index % 2 * 20, y=10, w=80, h=80)},
            frame + index,
            timestamp + index / 30.0,
        )

    assert binder.get_slot_state("P001")["vehicle_id"] == 30
    assert binder.get_slot_for_vehicle(31) is None


def test_legacy_jitter_remains_stationary_only():
    binder = SlotVehicleBinder(policy="legacy", stop_seconds=1.0)
    binder.update_vision([slot_result(occupied=True)], 0, 0.0)
    for index in range(40):
        binder.update_tracks(
            {71: track(x=index % 2 * 20, y=10, w=80, h=80)},
            index + 1,
            index / 30.0,
        )

    assert binder.get_slot_state("P001")["vehicle_id"] is None


def parked_vision_primary_binder(global_id=30, **binder_kwargs):
    binder = SlotVehicleBinder(
        policy="vision_primary",
        stop_seconds=1.0,
        **binder_kwargs,
    )
    result = slot_result(occupied=True)
    binder.update_vision([result], 0, 0.0, camera_id="cam1")
    frame = 1
    timestamp = 0.0
    descriptor = appearance()
    while timestamp <= 1.2:
        binder.update_tracks(
            {global_id: track(appearance=descriptor)},
            frame,
            timestamp,
            camera_id="cam1",
        )
        frame += 1
        timestamp += 1.0 / 30.0
    assert binder.get_slot_state("P001")["vehicle_id"] == global_id
    return binder, result, frame, timestamp, descriptor


def confirm_departure(binder, result, frame, timestamp):
    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")
    binder.update_vision([result], frame + 1, timestamp + 0.1, camera_id="cam1")
    return frame + 2, timestamp + 0.1


def test_vision_primary_changes_public_colour_immediately_but_keeps_gid_token():
    binder, result, frame, timestamp, _ = parked_vision_primary_binder()

    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["occupied"] is False
    assert state["vehicle_id"] is None
    assert state["recovery_global_id"] == 30
    assert state["recovery_state"] == "provisional"
    assert result.occupied is False
    assert result.vehicle_id is None


def test_false_empty_restores_binding_without_losing_gid():
    binder, result, frame, timestamp, _ = parked_vision_primary_binder()
    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")

    result.occupied = True
    binder.update_vision([result], frame + 1, timestamp + 0.1, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["occupied"] is True
    assert state["vehicle_id"] == 30
    assert state["recovery_state"] == "none"
    assert binder.get_slot_for_vehicle(30) == "P001"


def test_last_bbox_survives_false_empty_and_second_departure_recovery():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()

    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")
    result.occupied = True
    binder.update_vision([result], frame + 1, timestamp + 0.1, camera_id="cam1")

    # Depart again without another tracker observation.  Observation history
    # was cleared by the false empty, so token size must come from last_bbox.
    result.occupied = False
    binder.update_vision([result], frame + 2, timestamp + 0.2, camera_id="cam1")
    binder.update_vision([result], frame + 3, timestamp + 0.3, camera_id="cam1")
    token = binder.export_recovery_tokens(timestamp + 0.3)[0]
    assert token["last_bbox"] == (20.0, 20.0, 60.0, 60.0)

    batch = None
    for offset, y in enumerate((30, 33, 36), start=1):
        batch = binder.batch_recover_ids(
            {58: track(y=y, appearance=descriptor)},
            frame + 3 + offset,
            timestamp + 0.3 + offset * 0.04,
            camera_id="cam1",
        )
    assert batch is not None
    assert batch.recovered_ids == {58: 30}


def test_two_empty_samples_then_red_at_one_second_restores_old_binding():
    binder, result, frame, timestamp, _ = parked_vision_primary_binder()
    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")
    binder.update_vision([result], frame + 1, timestamp + 0.5, camera_id="cam1")
    assert binder.get_slot_state("P001")["recovery_state"] == "searching"

    result.occupied = True
    binder.update_vision([result], frame + 2, timestamp + 1.0, camera_id="cam1")

    state = binder.get_slot_state("P001")
    assert state["occupied"] is True
    assert state["vehicle_id"] == 30
    assert state["recovery_state"] == "none"


def test_safe_batch_recovery_requires_three_frames_and_outward_motion():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    first = binder.batch_recover_ids(
        {58: track(y=30, appearance=descriptor)},
        frame,
        timestamp + 0.03,
        camera_id="cam1",
    )
    assert first.recovered_ids == {}
    assert first.protected_local_keys == {58}

    second = binder.batch_recover_ids(
        {58: track(y=33, appearance=descriptor)},
        frame + 1,
        timestamp + 0.07,
        camera_id="cam1",
    )
    assert second.recovered_ids == {}
    assert second.protected_local_keys == {58}

    third = binder.batch_recover_ids(
        {58: track(y=36, appearance=descriptor)},
        frame + 2,
        timestamp + 0.11,
        camera_id="cam1",
    )
    assert third.recovered_ids == {58: 30}
    assert third.protected_local_keys == set()
    assert binder.export_recovery_tokens(timestamp + 0.11) == []


def test_departure_evidence_continues_across_replaced_local_track_ids():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    batch = None
    for offset, (local_id, y) in enumerate(
        ((58, 30), (59, 33), (60, 36)),
        start=1,
    ):
        batch = binder.batch_recover_ids(
            {local_id: track(y=y, appearance=descriptor)},
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
        )

    assert batch is not None
    assert batch.recovered_ids == {60: 30}
    continued = [
        event
        for event in binder.events
        if event["type"] == "departure_candidate_fragment_continued"
    ]
    assert len(continued) == 2


def test_recent_departure_candidate_survives_one_occupied_vision_rebound():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()

    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")
    binder.batch_recover_ids(
        {58: track(y=30, appearance=descriptor)},
        frame + 1,
        timestamp + 0.04,
        camera_id="cam1",
    )

    result.occupied = True
    binder.update_vision(
        [result], frame + 2, timestamp + 0.08, camera_id="cam1"
    )

    tokens = binder.export_recovery_tokens(timestamp + 0.08)
    assert len(tokens) == 1
    assert tokens[0]["global_id"] == 30
    assert tokens[0]["predeparture"] is True
    assert binder.get_slot_state("P001")["vehicle_id"] == 30
    assert any(
        event["type"] == "departure_token_rearmed_after_vision_rebound"
        for event in binder.events
    )


def test_qualified_predeparture_evidence_survives_long_local_fragment_gap():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()

    # First empty sample opens a provisional token. Three observations pass
    # every gate except the required second empty vision confirmation.
    result.occupied = False
    binder.update_vision([result], frame, timestamp, camera_id="cam1")
    for offset, y in enumerate((30, 33, 36), start=1):
        batch = binder.batch_recover_ids(
            {58: track(y=y, appearance=descriptor)},
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
        )
    assert batch.diagnostics[58]["reason"] == "departure_not_yet_confirmed"

    # Vision briefly reports occupied and the local fragment disappears for
    # more than the ordinary 0.75 s evidence TTL.
    result.occupied = True
    binder.update_vision(
        [result], frame + 4, timestamp + 0.16, camera_id="cam1"
    )
    result.occupied = False
    binder.update_vision(
        [result], frame + 12, timestamp + 1.0, camera_id="cam1"
    )
    binder.update_vision(
        [result], frame + 13, timestamp + 1.1, camera_id="cam1"
    )

    recovered = binder.batch_recover_ids(
        {91: track(y=100, appearance=descriptor)},
        frame + 14,
        timestamp + 1.11,
        camera_id="cam1",
    )
    assert recovered.recovered_ids == {91: 30}, recovered.diagnostics


def test_one_in_slot_predeparture_sample_can_continue_after_fast_gap():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()

    protected = binder.prepare_predeparture_tokens(
        {69: track(y=20, appearance=descriptor)},
        timestamp,
        camera_id="cam1",
    )
    assert protected == {69}
    first = binder.batch_recover_ids(
        {69: track(y=20, appearance=descriptor)},
        frame,
        timestamp,
        camera_id="cam1",
    )
    assert first.recovered_ids == {}

    # The fast vehicle disappears before three samples and reappears outside
    # the ordinary ROI expansion with a new local ID after > 0.75 seconds.
    result.occupied = False
    binder.update_vision(
        [result], frame + 1, timestamp + 0.90, camera_id="cam1"
    )
    binder.update_vision(
        [result], frame + 2, timestamp + 1.00, camera_id="cam1"
    )
    second = binder.batch_recover_ids(
        {70: track(y=145, appearance=descriptor)},
        frame + 3,
        timestamp + 1.01,
        camera_id="cam1",
    )
    recovered = binder.batch_recover_ids(
        {70: track(y=150, appearance=descriptor)},
        frame + 4,
        timestamp + 1.05,
        camera_id="cam1",
    )

    assert second.recovered_ids == {}
    assert recovered.recovered_ids == {70: 30}, recovered.diagnostics


def test_identical_appearance_tiny_drift_never_consumes_token():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    for offset, y in enumerate((30, 31, 32), start=1):
        batch = binder.batch_recover_ids(
            {77: track(y=y, appearance=descriptor)},
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
        )

    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {77}
    assert batch.diagnostics[77]["reason"] == "insufficient_outward_evidence"
    assert batch.diagnostics[77]["moved_px"] == 2.0
    assert binder.export_recovery_tokens(timestamp + 0.2)[0]["global_id"] == 30


def test_pre_token_cross_camera_history_cannot_authorize_nearby_track():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)
    token_created_at = binder.export_recovery_tokens(timestamp)[0]["created_at_s"]

    batch = None
    # This unrelated track existed before the slot became empty.  Its old
    # origin would make the full historical trajectory look strongly outward,
    # while all post-token observations actually move back toward the slot.
    for offset, current_y in enumerate((95.0, 93.0, 91.0), start=1):
        batch = binder.batch_recover_ids(
            {
                ("cam2", 88): track(
                    x=500,
                    y=300,
                    appearance=descriptor,
                    camera_id="cam2",
                    recovery_position=(50.0, current_y),
                    recovery_first_position=(50.0, 60.0),
                    recovery_first_timestamp_s=token_created_at - 1.0,
                    recovery_size_ratio=1.0,
                )
            },
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
            allow_cross_camera=True,
        )

    assert batch is not None
    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {("cam2", 88)}
    assert batch.diagnostics[("cam2", 88)]["reason"] == "insufficient_outward_evidence"
    assert binder.export_recovery_tokens(timestamp + 0.2)[0]["global_id"] == 30


def test_noise_or_shadow_never_consumes_departure_token():
    binder, result, frame, timestamp, _ = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    for offset, y in enumerate((30, 38, 46), start=1):
        batch = binder.batch_recover_ids(
            {77: track(y=y, appearance=None)},
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
        )
        assert batch.recovered_ids == {}
        assert 77 in batch.protected_local_keys

    tokens = binder.export_recovery_tokens(timestamp + 0.2)
    assert len(tokens) == 1
    assert tokens[0]["global_id"] == 30


def test_intermittent_shadow_evidence_resets_after_gap():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder(
        recovery_evidence_frames=3
    )
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    binder.batch_recover_ids(
        {77: track(y=30, appearance=descriptor)},
        frame,
        timestamp + 0.03,
        camera_id="cam1",
    )
    binder.batch_recover_ids(
        {77: track(y=36, appearance=descriptor)},
        frame + 8,
        timestamp + 0.70,
        camera_id="cam1",
    )
    batch = binder.batch_recover_ids(
        {77: track(y=42, appearance=descriptor)},
        frame + 9,
        timestamp + 0.74,
        camera_id="cam1",
    )

    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {77}
    assert batch.diagnostics[77]["evidence_frames"] == 2
    assert binder.export_recovery_tokens(timestamp + 0.74)[0]["global_id"] == 30


def test_one_lucky_shadow_histogram_cannot_unlock_gid_later():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder(
        recovery_evidence_frames=3
    )
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)
    wrong_descriptor = appearance(200)

    for offset, (y, sample) in enumerate(
        ((30, descriptor), (36, wrong_descriptor), (42, wrong_descriptor)),
        start=1,
    ):
        batch = binder.batch_recover_ids(
            {77: track(y=y, appearance=sample)},
            frame + offset,
            timestamp + offset * 0.04,
            camera_id="cam1",
        )

    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {77}
    assert batch.diagnostics[77]["reason"] == "appearance_missing_or_mismatch"
    assert binder.export_recovery_tokens(timestamp + 0.2)[0]["global_id"] == 30


def test_ambiguous_candidates_are_protected_and_do_not_consume_gid():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    binder.batch_recover_ids(
        {
            58: track(x=18, y=30, appearance=descriptor),
            59: track(x=22, y=30, appearance=descriptor),
        },
        frame,
        timestamp + 0.03,
        camera_id="cam1",
    )
    binder.batch_recover_ids(
        {
            58: track(x=18, y=33, appearance=descriptor),
            59: track(x=22, y=33, appearance=descriptor),
        },
        frame + 1,
        timestamp + 0.07,
        camera_id="cam1",
    )
    batch = binder.batch_recover_ids(
        {
            58: track(x=18, y=36, appearance=descriptor),
            59: track(x=22, y=36, appearance=descriptor),
        },
        frame + 2,
        timestamp + 0.11,
        camera_id="cam1",
    )

    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {58, 59}
    assert batch.ambiguous_local_keys == {58, 59}
    assert binder.export_recovery_tokens(timestamp + 0.11)[0]["global_id"] == 30


def test_expanding_gate_protects_candidate_before_it_can_match():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    batch = binder.batch_recover_ids(
        {
            58: track(
                x=120,
                y=20,
                appearance=descriptor,
                recovery_position=(150.0, 50.0),
                recovery_first_position=(150.0, 50.0),
            )
        },
        frame,
        timestamp,
        camera_id="cam1",
    )

    assert batch.recovered_ids == {}
    assert batch.protected_local_keys == {58}
    assert batch.diagnostics[58]["reason"] == "waiting_for_expanding_gate"


def test_cross_camera_transformed_points_can_recover_without_fake_bbox_geometry():
    binder, result, frame, timestamp, descriptor = parked_vision_primary_binder()
    frame, timestamp = confirm_departure(binder, result, frame, timestamp)

    binder.batch_recover_ids(
        {
            ("cam2", 59): track(
                x=500,
                y=300,
                appearance=descriptor,
                camera_id="cam2",
                recovery_position=(50.0, 90.0),
                recovery_first_position=(50.0, 90.0),
                recovery_size_ratio=1.0,
            )
        },
        frame,
        timestamp + 0.03,
        camera_id="cam1",
        allow_cross_camera=True,
    )
    binder.batch_recover_ids(
        {
            ("cam2", 59): track(
                x=503,
                y=305,
                appearance=descriptor,
                camera_id="cam2",
                recovery_position=(50.0, 93.0),
                recovery_first_position=(50.0, 90.0),
                recovery_size_ratio=1.0,
            )
        },
        frame + 1,
        timestamp + 0.07,
        camera_id="cam1",
        allow_cross_camera=True,
    )
    batch = binder.batch_recover_ids(
        {
            ("cam2", 59): track(
                x=505,
                y=310,
                appearance=descriptor,
                camera_id="cam2",
                recovery_position=(50.0, 96.0),
                recovery_first_position=(50.0, 90.0),
                recovery_size_ratio=1.0,
            )
        },
        frame + 2,
        timestamp + 0.11,
        camera_id="cam1",
        allow_cross_camera=True,
    )

    assert batch.recovered_ids == {("cam2", 59): 30}


def test_token_expiry_is_timestamp_based_and_remap_preserves_token():
    binder, result, frame, timestamp, _ = parked_vision_primary_binder(global_id=30)
    _, timestamp = confirm_departure(binder, result, frame, timestamp)

    binder.remap_vehicle_ids(lambda global_id: 12 if global_id == 30 else global_id)
    assert binder.export_recovery_tokens(timestamp)[0]["global_id"] == 12

    # A huge frame count with little elapsed time cannot expire it.
    assert binder.export_recovery_tokens(timestamp + 4.8)[0]["global_id"] == 12
    assert binder.export_recovery_tokens(timestamp + 5.0) == []
    expiry = [event for event in binder.events if event["type"] == "parked_id_recovery_expired"]
    assert expiry[-1]["global_id"] == 12
    assert expiry[-1]["retained_in_global_gallery"] is True
