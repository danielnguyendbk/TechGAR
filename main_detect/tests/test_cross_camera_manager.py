"""Focused regression tests for predictive multi-camera handoff."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from techgar.cross_camera_manager import CrossCameraManager, HandoffEntry
from techgar.tracklet_descriptor import AppearanceTracklet


@dataclass
class DummyTrack:
    cx: int
    cy: int
    w: int = 42
    h: int = 24
    history: list = None
    status: str = "confirmed"
    appearance: object = None

    def __post_init__(self):
        if self.history is None:
            self.history = [(self.cx, self.cy)]
        if self.appearance is None:
            self.appearance = np.ones((16, 16), dtype=np.float32)

    @property
    def x(self):
        return self.cx - self.w // 2

    @property
    def y(self):
        return self.cy - self.h


def make_manager():
    # Same crop geometry as multi_camera_sim.py for a 1100x720 source frame.
    return CrossCameraManager(
        camera_sizes={"cam1": (570, 380), "cam2": (570, 380), "cam3": (570, 380), "cam4": (570, 380)},
        camera_crops={"cam1": (0, 0, 570, 380), "cam2": (530, 0, 1100, 380), "cam3": (0, 340, 570, 720), "cam4": (530, 340, 1100, 720)},
        lookahead_frames=16,
        prediction_radius=90,
        appearance_threshold=0.45,
        min_direction_cosine=0.25,
    )


def fast_left_track(x, status="confirmed"):
    return DummyTrack(x, 200, history=[(x + 60, 200), (x + 45, 200), (x + 30, 200), (x + 15, 200), (x, 200)], status=status)


def one_hot_histogram(bin_index: int) -> np.ndarray:
    histogram = np.zeros((16, 16), dtype=np.float32)
    histogram.flat[bin_index] = 1.0
    return histogram


def attach_tracklet(track: DummyTrack, *histograms: np.ndarray) -> DummyTrack:
    descriptor = AppearanceTracklet(max_samples=8, sample_interval=1)
    for frame_idx, histogram in enumerate(histograms, start=1):
        descriptor.update(histogram, frame_idx)
    track.appearance_tracklet = descriptor
    return track


def fragment_track(
    cx: int,
    cy: int,
    observations: int,
    *,
    origin=(0, 0),
    w: int = 42,
    h: int = 24,
) -> DummyTrack:
    value = DummyTrack(cx, cy, w=w, h=h)
    value.fragment_visible_count = observations
    value.first_observation_point = origin
    value.first_observation_frame = 1
    return value


def test_brand_new_global_id_waits_for_current_fragment_evidence():
    manager = make_manager()

    for frame_idx in range(1, 5):
        current = fragment_track(
            100 + frame_idx * 5,
            200,
            frame_idx,
            origin=(100, 200),
        )
        ids = manager.update_all_tracks({"cam1": {7: current}}, frame_idx)
        assert 7 not in ids["cam1"]

    current = fragment_track(130, 200, 5, origin=(100, 200))
    ids = manager.update_all_tracks({"cam1": {7: current}}, 5)
    assert ids["cam1"][7] == 1
    assert any(
        event["type"]
        == "new_global_id_deferred_insufficient_fragment_evidence"
        for event in manager.to_json({})["recent_events"]
    )


def test_small_bbox_on_recent_vehicle_path_never_gets_second_gid():
    manager = make_manager()
    primary = DummyTrack(
        220,
        200,
        w=100,
        h=100,
        history=[(50, 200), (100, 200), (160, 200), (220, 200)],
    )
    assert manager.update_all_tracks({"cam1": {1: primary}}, 1)["cam1"][1] == 1

    fragment = fragment_track(
        100,
        200,
        8,
        origin=(40, 200),
        w=30,
        h=30,
    )
    ids = manager.update_all_tracks(
        {"cam1": {1: primary, 2: fragment}}, 2
    )

    assert ids["cam1"] == {1: 1}
    assert any(
        event["type"] == "new_global_id_deferred_partial_echo"
        and event["local_track_id"] == 2
        for event in manager.to_json({})["recent_events"]
    )


def test_small_echo_waits_even_when_primary_fragment_has_no_gid_yet():
    manager = CrossCameraManager(
        camera_sizes={"cam1": (570, 380)},
        camera_crops={"cam1": (0, 0, 570, 380)},
        new_identity_min_observations=8,
    )
    primary = fragment_track(
        220,
        200,
        5,
        origin=(180, 200),
        w=100,
        h=100,
    )
    primary.history = [(80, 200), (140, 200), (220, 200)]
    echo = fragment_track(
        80,
        200,
        8,
        origin=(20, 200),
        w=30,
        h=30,
    )

    ids = manager.update_all_tracks(
        {"cam1": {1: primary, 2: echo}}, 1
    )

    assert ids["cam1"] == {}
    assert any(
        event["type"] == "new_global_id_deferred_partial_echo"
        and event.get("primary_global_id_pending") is True
        for event in manager.to_json({})["recent_events"]
    )


def test_tiny_motion_tail_farther_than_bbox_stays_idless_near_primary_path():
    manager = make_manager()
    primary = fragment_track(
        220, 200, 8, origin=(120, 200), w=100, h=100
    )
    primary.history = [(120, 200), (170, 200), (220, 200)]
    tail = fragment_track(
        370, 205, 8, origin=(300, 205), w=30, h=30
    )

    ids = manager.update_all_tracks(
        {"cam1": {1: primary, 2: tail}}, 1
    )

    assert ids["cam1"] == {1: 1}
    assert any(
        event["type"] == "new_global_id_deferred_partial_echo"
        and event["local_track_id"] == 2
        for event in manager.to_json({})["recent_events"]
    )


def test_protected_unbound_primary_still_blocks_tiny_tail_gid_birth():
    manager = make_manager()
    primary = fragment_track(
        220, 200, 12, origin=(180, 200), w=100, h=160
    )
    primary.history = [(180, 200), (200, 200), (220, 200)]
    tail = fragment_track(
        345, 150, 9, origin=(300, 150), w=32, h=44
    )

    ids = manager.update_all_tracks(
        {"cam1": {70: primary, 71: tail}},
        1,
        protected_local_keys={("cam1", 70)},
    )

    assert ids["cam1"] == {}
    assert any(
        event["type"] == "new_global_id_deferred_partial_echo"
        and event["local_track_id"] == 71
        and event.get("primary_global_id_pending") is True
        for event in manager.to_json({})["recent_events"]
    )


def test_unusually_small_track_needs_long_independent_trajectory_for_new_gid():
    manager = make_manager()
    established = DummyTrack(100, 200, w=120, h=100)
    manager.update_all_tracks({"cam1": {1: established}}, 1)
    for frame_idx in range(2, 35):
        established = DummyTrack(100 + frame_idx, 200, w=120, h=100)
        manager.update_all_tracks({"cam1": {1: established}}, frame_idx)
    manager.update_all_tracks({"cam1": {}}, 35)

    small = fragment_track(
        520, 60, 6, origin=(450, 60), w=50, h=45
    )
    ids = manager.update_all_tracks({"cam1": {9: small}}, 36)

    assert 9 not in ids["cam1"]
    assert any(
        event["type"]
        == "new_global_id_deferred_insufficient_fragment_evidence"
        and event.get("reason") == "unusual_size_without_long_trajectory"
        for event in manager.to_json({})["recent_events"]
    )


def test_fast_cam4_to_cam3_tentative_receives_old_global_id():
    manager = make_manager()
    source = fast_left_track(60)
    assert manager.update_all_tracks({"cam4": {1: source}}, 1)["cam4"][1] == 1

    # Frame 2: source has reached the left edge and opens a predictive handoff.
    source = fast_left_track(30)
    manager.update_all_tracks({"cam4": {1: source}}, 2)

    # Frame 5: source is gone. cam3 has only a tentative first destination track,
    # but it is at the extrapolated world position and travelling left/inward.
    target = fast_left_track(515, status="tentative")
    ids = manager.update_all_tracks({"cam3": {9: target}}, 5)
    assert ids["cam3"][9] == 1
    assert manager.get_global_id("cam3", 9) == 1


def test_open_handoff_is_refreshed_while_source_track_remains_live():
    manager = make_manager()
    manager.handoff_ttl = 2
    manager.update_all_tracks({"cam4": {1: fast_left_track(60)}}, 1)
    manager.update_all_tracks({"cam4": {1: fast_left_track(30)}}, 2)
    assert manager.to_json({})["pending_handoffs"][0]["updated_at_frame"] == 2

    # This observation no longer satisfies the edge-opening condition, but it
    # is still the same live source track. Its existing transfer record must
    # follow the newest position instead of ageing from frame 2.
    away_from_edge = DummyTrack(
        180,
        200,
        history=[(180, 200), (180, 200)],
        status="confirmed",
    )
    manager.update_all_tracks({"cam4": {1: away_from_edge}}, 3)
    pending = manager.to_json({})["pending_handoffs"]
    assert pending[0]["updated_at_frame"] == 3


def test_same_camera_dormant_reid_rebinds_pending_handoff_source():
    manager = make_manager()
    source = fast_left_track(60)
    manager.update_all_tracks({"cam4": {1: source}}, 1)
    source = fast_left_track(30)
    manager.update_all_tracks({"cam4": {1: source}}, 2)
    manager.notify_track_expired(
        "cam4",
        1,
        source.cx,
        source.cy,
        source.w,
        source.h,
        source.appearance,
        3,
    )

    recovered_source = fast_left_track(32)
    ids = manager.update_all_tracks({"cam4": {9: recovered_source}}, 4)

    assert ids["cam4"][9] == 1
    pending = manager.to_json({})["pending_handoffs"]
    assert pending[0]["global_id"] == 1
    assert pending[0]["source_local_track_id"] == 9
    assert pending[0]["updated_at_frame"] == 4
    assert any(
        event["type"] == "handoff_source_rebound"
        for event in manager.to_json({})["recent_events"]
    )


def test_custom_mask_entry_corridor_uses_bbox_pixels_not_world_centimetres():
    polygon = [
        {"x": 0, "y": 0},
        {"x": 200, "y": 0},
        {"x": 200, "y": 200},
        {"x": 0, "y": 200},
    ]
    manager = CrossCameraManager(
        camera_sizes={"cam1": (200, 200), "cam2": (200, 200)},
        camera_crops={"cam1": (0, 0, 200, 200), "cam2": (0, 0, 200, 200)},
        camera_transforms={"cam1": np.eye(3), "cam2": np.eye(3)},
        edge_adjacency={("cam1", "1"): "cam2"},
        custom_masks={
            "cam1": {"polygon": polygon, "handoff_edge": "1"},
            "cam2": {"polygon": polygon, "handoff_edge": "1"},
        },
        prediction_radius=25.0,
        edge_margin=40,
        shared_map_anchor="bottom_center",
    )
    appearance = one_hot_histogram(0)
    target = DummyTrack(
        100,
        82,
        w=42,
        h=24,
        history=[(100, 72), (100, 77), (100, 82)],
        status="tentative",
        appearance=appearance,
    )
    entry = HandoffEntry(
        global_id=1,
        source_cam="cam1",
        source_local_track_id=1,
        target_cam="cam2",
        exit_edge="1",
        last_world=(100.0, 82.0),
        velocity_world=(0.0, 5.0),
        bbox_size=(42, 24),
        appearance=appearance,
        appearance_samples=(appearance,),
        created_at_frame=1,
        updated_at_frame=3,
    )

    cost, reason, details = manager._candidate_cost(entry, "cam2", target, 3)

    assert details["entry_depth"] == pytest.approx(82.0)
    assert reason == "ok"
    assert cost is not None


def test_one_handoff_cannot_be_assigned_to_two_nearby_targets():
    manager = make_manager()
    manager.update_all_tracks({"cam4": {1: fast_left_track(60)}}, 1)
    manager.update_all_tracks({"cam4": {1: fast_left_track(30)}}, 2)

    best = fast_left_track(515, status="tentative")
    nearby = fast_left_track(450, status="tentative")
    ids = manager.update_all_tracks({"cam3": {9: best, 10: nearby}}, 5)
    assert ids["cam3"][9] == 1
    assert 10 not in ids["cam3"]


def test_large_boundary_blob_size_change_can_still_match_with_strong_motion_evidence():
    manager = make_manager()
    manager.update_all_tracks({"cam4": {1: fast_left_track(60)}}, 1)
    manager.update_all_tracks({"cam4": {1: fast_left_track(30)}}, 2)
    target = fast_left_track(515, status="tentative")
    target.w, target.h = 90, 55  # clipped/merged blob at the target crop edge
    assert manager.update_all_tracks({"cam3": {9: target}}, 5)["cam3"][9] == 1


def test_extremely_close_handoff_uses_bounded_relaxed_appearance():
    manager = make_manager()
    base = one_hot_histogram(0)
    shifted = np.zeros((16, 16), dtype=np.float32)
    shifted.flat[0] = 0.5
    shifted.flat[1] = 0.5
    shifted = cv2.normalize(shifted, shifted)
    target = fast_left_track(515, status="tentative")
    target.appearance = shifted
    entry = HandoffEntry(
        global_id=1,
        source_cam="cam4",
        source_local_track_id=1,
        target_cam="cam3",
        exit_edge="left",
        last_world=(545.0, 540.0),
        velocity_world=(-10.0, 0.0),
        bbox_size=(target.w, target.h),
        appearance=base,
        appearance_samples=(base,),
        created_at_frame=2,
        updated_at_frame=2,
    )

    cost, reason, details = manager._candidate_cost(entry, "cam3", target, 5)

    assert 0.45 < details["appearance_distance"] <= 0.60
    assert details["adaptive_appearance"] is True
    assert reason == "ok"
    assert cost is not None


def test_slot_release_recovers_global_id_before_new_allocation():
    manager = make_manager()
    # Global #1 already belongs to a vehicle that has just left a parking slot.
    first = DummyTrack(120, 160)
    assert manager.update_all_tracks({"cam3": {2: first}}, 1)["cam3"][2] == 1

    # Local IDs are not global: a newly created cam3 local #9 must receive #1,
    # not an allocated #2, after the slot-binder verifies its origin.
    leaving_slot = DummyTrack(135, 165, status="tentative")
    manager.bind_external_id("cam3", 9, 1, 20, source="parking_slot_release")
    ids = manager.update_all_tracks({"cam3": {9: leaving_slot}}, 20)
    assert ids["cam3"][9] == 1
    assert manager.to_json({"cam3": {9: leaving_slot}})["next_global_id"] == 2


def test_parked_reservation_detaches_local_track_and_blocks_id_theft():
    manager = make_manager()
    parked = DummyTrack(120, 160)
    assert manager.update_all_tracks({"cam3": {2: parked}}, 1)["cam3"][2] == 1
    reservations = manager.sync_parked_reservations(
        [{
            "global_id": 1,
            "slot_id": "F01",
            "camera_id": "cam3",
            "state": "parked",
            "bbox": (100, 130, 42, 24),
        }],
        2,
    )
    detached = manager.detach_parked_local_tracks(2)

    assert reservations[1]["slot_id"] == "F01"
    assert detached == [("cam3", 2, 1)]
    assert manager.get_global_id("cam3", 2) is None
    with pytest.raises(ValueError, match="is parked"):
        manager.bind_external_id("cam3", 9, 1, 3, source="dormant_reid")

    assert manager.bind_external_id(
        "cam3", 9, 1, 4, source="parking_departure_token"
    ) == 1
    assert manager.parked_global_ids == set()


def test_same_camera_motion_echo_keeps_one_global_id_and_map_observation():
    manager = make_manager()
    primary = fast_left_track(250)
    assert manager.update_all_tracks({"cam3": {1: primary}}, 1)["cam3"][1] == 1
    echo = fast_left_track(262)
    ids = manager.update_all_tracks({"cam3": {1: primary, 2: echo}}, 2)
    assert 2 not in ids["cam3"]
    assert 2 not in manager.update_all_tracks({"cam3": {1: primary, 2: echo}}, 3)["cam3"]
    ids = manager.update_all_tracks({"cam3": {1: primary, 2: echo}}, 4)
    assert ids["cam3"][2] == 1
    registry = manager.to_json({"cam3": {1: primary, 2: echo}})
    assert registry["next_global_id"] == 2
    assert registry["map_vehicles"]["1"]["observation_count"] == 1


def test_recently_lost_identity_does_not_consume_an_existing_nearby_id():
    manager = make_manager()
    original = fast_left_track(250)
    later = fast_left_track(420)
    ids = manager.update_all_tracks({"cam3": {1: original, 2: later}}, 1)
    assert ids["cam3"] == {1: 1, 2: 2}

    manager.notify_track_lost("cam3", 1, original, 2)
    # Local #2 is already a durable identity. Proximity to recently-lost #1 is
    # insufficient proof, even across several frames.
    continuation = fast_left_track(235)
    ids = manager.update_all_tracks({"cam3": {2: continuation}}, 3)
    assert ids["cam3"][2] == 2
    manager.update_all_tracks({"cam3": {2: continuation}}, 4)
    ids = manager.update_all_tracks({"cam3": {2: continuation}}, 5)
    assert ids["cam3"][2] == 2
    events = manager.to_json({"cam3": {2: continuation}})["recent_events"]
    assert not any(
        e["type"] == "global_id_merged" and e.get("superseded_global_id") == 2
        for e in events
    )


def test_two_existing_nearby_slow_boxes_do_not_merge_from_proximity_alone():
    manager = make_manager()
    first = DummyTrack(140, 180, history=[(138, 180), (139, 180), (140, 180)])
    second = DummyTrack(330, 180, history=[(328, 180), (329, 180), (330, 180)])
    ids = manager.update_all_tracks({"cam3": {1: first, 2: second}}, 1)
    assert ids["cam3"] == {1: 1, 2: 2}

    # Both already have IDs, then the slow-motion old/new boxes become close.
    first = DummyTrack(220, 180, history=[(218, 180), (219, 180), (220, 180)])
    second = DummyTrack(255, 180, history=[(253, 180), (254, 180), (255, 180)])
    ids = manager.update_all_tracks({"cam3": {1: first, 2: second}}, 2)
    assert ids["cam3"] == {1: 1, 2: 2}
    manager.update_all_tracks({"cam3": {1: first, 2: second}}, 3)
    ids = manager.update_all_tracks({"cam3": {1: first, 2: second}}, 4)
    assert ids["cam3"] == {1: 1, 2: 2}

    registry = manager.to_json({"cam3": {1: first, 2: second}})
    assert registry["retired_global_ids"] == {}
    assert set(registry["map_vehicles"]) == {"1", "2"}


def test_small_touching_partial_echo_does_not_allocate_second_global_id():
    manager = make_manager()
    primary = DummyTrack(250, 200, w=134, h=85)
    ids = manager.update_all_tracks({"cam3": {1: primary}}, 1)
    assert ids["cam3"][1] == 1

    fragment = DummyTrack(
        335,
        200,
        w=54,
        h=39,
        # A lamp/edge-only crop can have unrelated HSV even though it is a
        # geometrically tiny piece touching the full vehicle bbox.
        appearance=one_hot_histogram(100),
    )
    ids = manager.update_all_tracks(
        {"cam3": {1: primary, 2: fragment}}, 2
    )

    assert ids["cam3"] == {1: 1}
    assert manager.get_global_id("cam3", 2) is None
    assert manager.to_json({})["next_global_id"] == 2
    assert any(
        event["type"] == "new_global_id_deferred_partial_echo"
        for event in manager.to_json({})["recent_events"]
    )


def test_overlap_observations_share_one_global_id():
    manager = make_manager()
    # Stationary tracks ensure this case exercises overlap de-duplication, not
    # the predictive handoff path.
    cam4_track = DummyTrack(30, 200, history=[(30, 200), (30, 200)])
    assert manager.update_all_tracks({"cam4": {1: cam4_track}}, 1)["cam4"][1] == 1
    # World x=560 is inside the 40 pixel overlap: cam4 local 30, cam3 local 560.
    cam3_track = DummyTrack(560, 200, history=[(560, 200), (560, 200)])
    ids = manager.update_all_tracks({"cam4": {1: cam4_track}, "cam3": {2: cam3_track}}, 2)
    assert ids["cam3"][2] == 1
    registry = manager.to_json({"cam4": {1: cam4_track}, "cam3": {2: cam3_track}})
    assert list(registry["map_vehicles"]) == ["1"]
    assert registry["map_vehicles"]["1"]["observation_count"] == 2


def make_real_two_camera_manager(shared_map_anchor="bbox_center"):
    # cam2 pixels are translated into cam1/world pixels after calibration.
    return CrossCameraManager(
        camera_sizes={"cam1": (640, 480), "cam2": (640, 480)},
        camera_crops={"cam1": (0, 0, 640, 480), "cam2": (0, 0, 640, 480)},
        camera_transforms={
            "cam1": np.eye(3),
            "cam2": np.array([[1.0, 0.0, 500.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        },
        edge_adjacency={("cam1", "right"): "cam2", ("cam2", "left"): "cam1"},
        overlap_regions={("cam1", "cam2"): np.array([[500, 0], [640, 0], [640, 480], [500, 480]], dtype=np.float32)},
        match_distance=15.0,
        lookahead_frames=16,
        prediction_radius=90,
        appearance_threshold=0.45,
        min_direction_cosine=0.25,
        shared_map_anchor=shared_map_anchor,
    )


def test_reverse_handoff_cannot_steal_id_already_active_in_target_camera():
    manager = make_real_two_camera_manager()
    appearance = one_hot_histogram(0)
    source = DummyTrack(
        560,
        220,
        history=[(540, 220), (550, 220), (560, 220)],
        appearance=appearance,
    )
    existing_target = DummyTrack(
        60,
        220,
        history=[(40, 220), (50, 220), (60, 220)],
        appearance=appearance,
    )
    ids = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {1: existing_target}}, 1
    )
    assert ids == {"cam1": {1: 1}, "cam2": {1: 1}}

    # This neighbouring bbox is spatially and visually plausible for a
    # reverse handoff. It must stay unbound because G#1 already has a live
    # owner in cam2.
    neighbour = DummyTrack(
        72,
        220,
        history=[(62, 220), (67, 220), (72, 220)],
        status="tentative",
        appearance=appearance,
    )
    ids = manager.update_all_tracks(
        {
            "cam1": {1: source},
            "cam2": {1: existing_target, 2: neighbour},
        },
        2,
    )

    assert ids["cam2"][1] == 1
    assert 2 not in ids["cam2"]
    assert manager.get_global_id("cam2", 2) is None
    assert not manager.to_json({})["pending_handoffs"]


def test_cross_camera_dormant_reid_uses_last_position_when_velocity_is_bad():
    manager = make_real_two_camera_manager()
    appearance = one_hot_histogram(0)
    source = DummyTrack(
        40,
        220,
        history=[(20, 220), (30, 220), (40, 220)],
        appearance=appearance,
    )
    assert manager.update_all_tracks(
        {"cam2": {1: source}}, 1, {"cam2": 0.0}
    )["cam2"][1] == 1
    source = DummyTrack(
        60,
        220,
        history=[(20, 220), (40, 220), (60, 220)],
        appearance=appearance,
    )
    manager.update_all_tracks({"cam2": {1: source}}, 2, {"cam2": 0.1})
    # Exercise dormant Re-ID rather than a normal pending handoff.
    manager._handoffs.clear()
    manager.notify_track_expired(
        "cam2", 1, source.cx, source.cy, source.w, source.h,
        source.appearance, 3, timestamp_s=0.1,
    )

    # Same shared-map position in cam1. Extrapolating the noisy 200 px/s
    # terminal velocity would miss it badly, while the last reliable point is
    # exact and the strict appearance gate agrees.
    target = DummyTrack(
        560,
        220,
        # Deliberately opposite to the noisy source velocity. Strong colour,
        # short elapsed time and the last-position gate still prove identity.
        history=[(564, 220), (562, 220), (560, 220)],
        appearance=appearance,
    )
    ids = manager.update_all_tracks(
        {"cam1": {9: target}}, 6, {"cam1": 0.6}
    )

    assert ids["cam1"][9] == 1
    event = next(
        event
        for event in manager.to_json({})["recent_events"]
        if event["type"] == "dormant_global_id_recovered"
    )
    assert event["last_position_distance"] < event["extrapolated_distance"]
    assert any(
        event["type"] == "dormant_direction_override_strong_reid"
        for event in manager.to_json({})["recent_events"]
    )


def opposing_view_histogram(bin_index: int) -> np.ndarray:
    """Histogram about 0.743 away from bin zero (poor opposing view)."""
    histogram = np.zeros((16, 16), dtype=np.float32)
    histogram.flat[0] = 0.2
    histogram.flat[bin_index] = 0.8
    return cv2.normalize(histogram, histogram)


def test_real_camera_shared_map_uses_bbox_center_not_opposite_vehicle_ends():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 220, w=42, h=40)
    cam2 = DummyTrack(62, 240, w=42, h=80)

    # Bottom-centres differ by about 20 cm after calibration, while bbox
    # centres represent the same physical vehicle within 2 cm.
    bottom_distance = np.linalg.norm(np.subtract(
        manager._world("cam1", (cam1.cx, cam1.cy)),
        manager._world("cam2", (cam2.cx, cam2.cy)),
    ))
    assert bottom_distance > 20.0

    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 1)
    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    positions = [
        observation["global_position"]
        for observation in registry["active_global_vehicles"]["1"]["observations"]
    ]
    assert positions == [{"x": 560.0, "y": 200.0}, {"x": 562.0, "y": 200.0}]
    assert all(
        observation["shared_map_anchor"]["reference"] == "bbox_center"
        for observation in registry["active_global_vehicles"]["1"]["observations"]
    )


def test_unique_tight_cross_camera_pair_merges_ids_with_adaptive_appearance():
    manager = make_real_two_camera_manager()
    base = one_hot_histogram(0)
    opposing = opposing_view_histogram(1)
    cam1 = attach_tracklet(
        DummyTrack(560, 220, h=40, appearance=base),
        base,
        base,
    )
    cam2 = attach_tracklet(
        DummyTrack(62, 240, h=80, appearance=opposing),
        opposing,
        opposing,
    )

    first = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}}, 1
    )
    assert first == {"cam1": {1: 1}, "cam2": {7: 2}}
    second = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}}, 2
    )
    assert second == {"cam1": {1: 1}, "cam2": {7: 2}}
    ids = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}}, 3
    )

    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    assert registry["retired_global_ids"] == {"2": 1}
    assert registry["identity_lifecycle"]["1"]["appearance_sample_count"] == 2
    assert any(
        event["type"] == "global_id_merged"
        and event.get("reason")
        in {"explicit_predictive_handoff", "unique_cross_camera_overlap"}
        for event in registry["recent_events"]
    )


def test_existing_id_is_bound_before_new_id_allocation_in_overlap():
    manager = make_real_two_camera_manager()
    base = one_hot_histogram(0)
    opposing = opposing_view_histogram(1)
    cam1 = attach_tracklet(
        DummyTrack(560, 220, h=40, appearance=base),
        base,
        base,
    )
    assert manager.update_all_tracks({"cam1": {1: cam1}}, 1)["cam1"][1] == 1

    cam2 = attach_tracklet(
        DummyTrack(62, 240, h=80, appearance=opposing),
        opposing,
        opposing,
    )
    first_target_frame = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}},
        2,
    )
    assert first_target_frame == {"cam1": {1: 1}, "cam2": {}}
    ids = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}},
        3,
    )

    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    assert registry["next_global_id"] == 2
    assert registry["retired_global_ids"] == {}
    assert any(
        event["type"]
        in {
            "handoff_matched",
            "handoff_matched_target_gallery",
            "cross_camera_deferred_claim_matched",
        }
        for event in registry["recent_events"]
    )


def test_unique_overlap_candidate_waits_for_tracklet_before_allocating_id():
    manager = make_real_two_camera_manager()
    base = one_hot_histogram(0)
    source = attach_tracklet(
        DummyTrack(560, 220, h=40, appearance=base),
        base,
        base,
    )
    manager.update_all_tracks({"cam1": {1: source}}, 1)

    uncertain = DummyTrack(62, 240, h=80, appearance=one_hot_histogram(5))
    frame_two = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: uncertain}},
        2,
    )
    assert frame_two == {"cam1": {1: 1}, "cam2": {}}
    assert manager.to_json({"cam1": {1: source}, "cam2": {7: uncertain}})[
        "next_global_id"
    ] == 2

    opposing = opposing_view_histogram(1)
    clearer = attach_tracklet(
        DummyTrack(62, 240, h=80, appearance=opposing),
        opposing,
        opposing,
    )
    frame_three = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: clearer}},
        3,
    )
    assert frame_three == {"cam1": {1: 1}, "cam2": {7: 1}}
    events = manager.to_json(
        {"cam1": {1: source}, "cam2": {7: clearer}}
    )["recent_events"]
    assert any(event["type"] == "cross_camera_assignment_deferred" for event in events)
    assert any(
        event["type"]
        in {
            "cross_camera_unbound_matched",
            "cross_camera_deferred_claim_matched",
            "handoff_matched",
            "handoff_matched_target_gallery",
        }
        for event in events
    )


def test_deferred_different_vehicle_eventually_receives_new_id():
    manager = make_real_two_camera_manager()
    manager.cross_camera_defer_frames = 2
    source = DummyTrack(560, 220, h=40, appearance=one_hot_histogram(0))
    different = DummyTrack(62, 240, h=80, appearance=one_hot_histogram(5))
    manager.update_all_tracks({"cam1": {1: source}}, 1)

    assert manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: different}},
        2,
    )["cam2"] == {}
    assert manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: different}},
        3,
    )["cam2"] == {}
    assert manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: different}},
        4,
    )["cam2"] == {7: 2}


def test_adaptive_appearance_does_not_merge_beyond_strong_spatial_gate():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 220, h=40, appearance=one_hot_histogram(0))
    cam2 = DummyTrack(
        68,
        240,
        h=80,
        appearance=opposing_view_histogram(1),
    )

    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 1)

    assert ids == {"cam1": {1: 1}, "cam2": {7: 2}}
    assert manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})[
        "retired_global_ids"
    ] == {}


def test_ambiguous_cross_camera_neighbours_are_not_merged():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 220, h=40, appearance=one_hot_histogram(0))
    left = DummyTrack(
        58,
        240,
        h=80,
        appearance=opposing_view_histogram(1),
    )
    right = DummyTrack(
        62,
        240,
        h=80,
        appearance=opposing_view_histogram(2),
    )

    tracks = {"cam1": {1: cam1}, "cam2": {7: left, 8: right}}
    ids = manager.update_all_tracks(tracks, 1)
    manager.update_all_tracks(tracks, 2)
    ids = manager.update_all_tracks(tracks, 3)

    assert len(set(ids["cam1"].values()) | set(ids["cam2"].values())) == 3
    registry = manager.to_json(
        {"cam1": {1: cam1}, "cam2": {7: left, 8: right}}
    )
    assert registry["retired_global_ids"] == {}


def test_fast_handoff_keeps_evidence_after_source_disappears():
    manager = make_real_two_camera_manager()
    source_view = one_hot_histogram(0)
    target_view = opposing_view_histogram(1)
    source = attach_tracklet(
        DummyTrack(
            560,
            220,
            h=40,
            history=[(540, 220), (550, 220), (560, 220)],
            appearance=source_view,
        ),
        source_view,
        source_view,
    )
    assert manager.update_all_tracks({"cam1": {1: source}}, 1)["cam1"][1] == 1

    # Shared-map residual is 6 px/cm. Cross-view HSV is deliberately poor, so
    # the first destination observation must be deferred, not given G#2.
    target = attach_tracklet(
        DummyTrack(
            66,
            240,
            h=80,
            history=[(62, 240), (64, 240), (66, 240)],
            status="tentative",
            appearance=target_view,
        ),
        target_view,
        target_view,
    )
    assert manager.update_all_tracks({"cam2": {7: target}}, 2) == {
        "cam2": {}
    }
    ids = manager.update_all_tracks({"cam2": {7: target}}, 3)

    assert ids == {"cam2": {7: 1}}
    assert manager.to_json({})["next_global_id"] == 2
    assert any(
        event["type"] == "handoff_candidate_deferred"
        for event in manager.to_json({})["recent_events"]
    )


def test_explicit_handoff_reconciles_premature_target_gid():
    manager = make_real_two_camera_manager()
    source_view = one_hot_histogram(0)
    target_view = opposing_view_histogram(1)
    source = attach_tracklet(
        DummyTrack(
            560,
            220,
            h=40,
            history=[(540, 220), (550, 220), (560, 220)],
            appearance=source_view,
        ),
        source_view,
        source_view,
    )
    manager.update_all_tracks({"cam1": {1: source}}, 1)
    target = attach_tracklet(
        DummyTrack(
            66,
            240,
            h=80,
            history=[(62, 240), (64, 240), (66, 240)],
            appearance=target_view,
        ),
        target_view,
        target_view,
    )
    manager.bind_external_id("cam2", 7, 2, 2, source="test")

    first = manager.update_all_tracks({"cam2": {7: target}}, 2)
    assert first == {"cam2": {7: 2}}
    second = manager.update_all_tracks({"cam2": {7: target}}, 3)

    assert second == {"cam2": {7: 1}}
    assert manager.canonical_global_id(2) == 1
    assert any(
        event["type"] == "global_id_merged"
        and event.get("reason") == "explicit_predictive_handoff"
        for event in manager.to_json({})["recent_events"]
    )


def test_established_bound_gid_cannot_steal_handoff_from_unbound_target():
    manager = make_real_two_camera_manager()
    appearance = one_hot_histogram(0)
    source = DummyTrack(60, 240, h=80, appearance=appearance)
    wrong_existing = DummyTrack(560, 220, h=40, appearance=appearance)
    correct_unbound = DummyTrack(564, 220, h=40, appearance=appearance)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.bind_external_id("cam1", 2, 2, 1, source="test")
    manager.update_all_tracks(
        {"cam2": {1: source}, "cam1": {2: wrong_existing}}, 1
    )
    manager._handoffs = [
        HandoffEntry(
            global_id=1,
            source_cam="cam2",
            source_local_track_id=1,
            target_cam="cam1",
            exit_edge="overlap",
            last_world=(560.0, 200.0),
            velocity_world=(0.0, 0.0),
            bbox_size=(42, 80),
            appearance=appearance,
            appearance_samples=(appearance,),
            created_at_frame=19,
            updated_at_frame=19,
        )
    ]

    ids = manager.update_all_tracks(
        {
            "cam2": {1: source},
            "cam1": {2: wrong_existing, 3: correct_unbound},
        },
        20,
    )

    assert ids["cam1"] == {2: 2, 3: 1}
    assert manager.canonical_global_id(2) == 2
    assert any(
        event["type"] == "handoff_bound_identity_rejected"
        and event["reason"] == "both_global_ids_established"
        for event in manager.to_json({})["recent_events"]
    )


def test_handoff_merge_cannot_create_two_real_tracks_in_same_camera():
    manager = make_real_two_camera_manager()
    first_view = one_hot_histogram(0)
    second_view = one_hot_histogram(8)
    source = DummyTrack(60, 240, h=80, appearance=first_view)
    other_same_camera = DummyTrack(260, 240, h=80, appearance=second_view)
    # It looks plausible to the incoming G#1, while G#2 already owns a clearly
    # different live car in cam2. The rejection must therefore come from the
    # same-camera owner invariant, not the appearance gate.
    target = DummyTrack(560, 220, h=40, appearance=first_view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.bind_external_id("cam2", 9, 2, 1, source="test")
    manager.bind_external_id("cam1", 2, 2, 2, source="test")
    manager.update_all_tracks(
        {"cam2": {1: source, 9: other_same_camera}, "cam1": {2: target}},
        2,
    )
    manager._handoffs = [
        HandoffEntry(
            global_id=1,
            source_cam="cam2",
            source_local_track_id=1,
            target_cam="cam1",
            exit_edge="overlap",
            last_world=(560.0, 200.0),
            velocity_world=(0.0, 0.0),
            bbox_size=(42, 80),
            appearance=first_view,
            appearance_samples=(first_view,),
            created_at_frame=2,
            updated_at_frame=2,
        )
    ]

    ids = manager.update_all_tracks(
        {"cam2": {1: source, 9: other_same_camera}, "cam1": {2: target}},
        3,
    )

    assert ids["cam2"] == {1: 1, 9: 2}
    assert ids["cam1"] == {2: 2}
    assert manager.canonical_global_id(1) == 1
    assert manager.canonical_global_id(2) == 2
    assert any(
        event["type"] == "handoff_bound_identity_rejected"
        and event["reason"] == "same_camera_live_owner_conflict"
        for event in manager.to_json({})["recent_events"]
    )


def test_destination_camera_gallery_accepts_moderate_view_jitter():
    manager = make_real_two_camera_manager()
    base = one_hot_histogram(0)
    jittered = np.zeros((16, 16), dtype=np.float32)
    jittered.flat[0] = 0.72
    jittered.flat[1] = 0.28
    jittered = cv2.normalize(jittered, jittered)
    target = DummyTrack(60, 240, h=80, appearance=jittered)
    entry = HandoffEntry(
        global_id=1,
        source_cam="cam1",
        source_local_track_id=1,
        target_cam="cam2",
        exit_edge="overlap",
        last_world=(560.0, 200.0),
        velocity_world=(0.0, 0.0),
        bbox_size=(42, 40),
        appearance=one_hot_histogram(9),
        appearance_samples=(one_hot_histogram(9),),
        target_appearance_samples=(base,),
        target_bbox_size=(42, 80),
        created_at_frame=1,
        updated_at_frame=1,
    )

    cost, reason, details = manager._candidate_cost(
        entry, "cam2", target, 1
    )

    assert 0.33 < details["appearance_distance"] <= 0.45
    assert details["appearance_reference"] == "target_camera"
    assert reason == "ok"
    assert cost is not None


def test_short_dormant_return_uses_target_gallery_and_speed_margin():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    jittered_cam2 = np.zeros((16, 16), dtype=np.float32)
    jittered_cam2.flat[4] = 0.72
    jittered_cam2.flat[5] = 0.28
    jittered_cam2 = cv2.normalize(jittered_cam2, jittered_cam2)

    first_cam2 = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.update_all_tracks({"cam2": {1: first_cam2}}, 1, {"cam2": 0.0})
    cam1 = DummyTrack(560, 220, h=40, appearance=cam1_view)
    manager.bind_external_id("cam1", 2, 1, 2, source="test")
    manager.update_all_tracks({"cam1": {2: cam1}}, 2, {"cam1": 0.1})
    manager.notify_track_lost("cam1", 2, cam1, 3, timestamp_s=0.2)
    manager._handoffs.clear()

    # Forty shared-map units is just outside the calibrated 35-unit dormant
    # radius, but valid for a short fast transfer with destination history.
    returning = DummyTrack(
        100,
        240,
        h=80,
        history=[(96, 240), (98, 240), (100, 240)],
        appearance=jittered_cam2,
    )
    ids = manager.update_all_tracks(
        {"cam2": {9: returning}}, 8, {"cam2": 1.0}
    )

    assert ids == {"cam2": {9: 1}}
    event = next(
        event
        for event in manager.to_json({})["recent_events"]
        if event["type"] == "dormant_global_id_recovered"
    )
    assert event["appearance_reference"] == "target_camera"


def test_long_dormant_cross_camera_reid_requires_destination_gallery():
    manager = make_real_two_camera_manager()
    manager.identity_retention_seconds = 60.0
    black = one_hot_histogram(0)
    white = one_hot_histogram(8)
    source = DummyTrack(560, 220, appearance=black)
    manager.update_all_tracks({"cam1": {1: source}}, 1, {"cam1": 0.0})
    manager.notify_track_lost("cam1", 1, source, 2, timestamp_s=0.1)
    manager._handoffs.clear()

    stale_candidate = DummyTrack(
        60,
        240,
        h=80,
        history=[(58, 240), (59, 240), (60, 240)],
        appearance=white,
    )
    ids = manager.update_all_tracks(
        {"cam2": {9: stale_candidate}}, 220, {"cam2": 27.2}
    )

    assert ids == {"cam2": {9: 2}}
    assert manager.get_global_id("cam2", 9) != 1
    assert any(
        event["type"] == "dormant_reid_rejected_stale"
        for event in manager.to_json({})["recent_events"]
    )


def test_long_return_recovers_from_same_destination_camera_gallery():
    manager = make_real_two_camera_manager()
    manager.identity_retention_seconds = 60.0
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    first_cam1 = DummyTrack(560, 220, h=40, appearance=cam1_view)
    manager.update_all_tracks({"cam1": {1: first_cam1}}, 1, {"cam1": 0.0})

    cam2 = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 7, 1, 2, source="test")
    manager.update_all_tracks({"cam2": {7: cam2}}, 2, {"cam2": 0.1})
    manager.notify_track_lost("cam2", 7, cam2, 3, timestamp_s=0.2)
    manager._handoffs.clear()

    returning = DummyTrack(
        560,
        220,
        h=40,
        history=[(558, 220), (559, 220), (560, 220)],
        appearance=cam1_view,
    )
    ids = manager.update_all_tracks(
        {"cam1": {9: returning}}, 270, {"cam1": 33.2}
    )

    assert ids == {"cam1": {9: 1}}
    event = next(
        event
        for event in manager.to_json({})["recent_events"]
        if event["type"] == "dormant_global_id_recovered"
    )
    assert event["appearance_reference"] == "target_camera"


def test_camera_specific_gallery_is_kept_separate_by_view():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(7)
    first = DummyTrack(560, 220, appearance=cam1_view)
    manager.update_all_tracks({"cam1": {1: first}}, 1)
    second = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 7, 1, 2, source="test")
    manager.update_all_tracks({"cam2": {7: second}}, 2)

    lifecycle = manager.to_json({})["identity_lifecycle"]["1"]
    assert lifecycle["camera_appearance_sample_counts"] == {
        "cam1": 1,
        "cam2": 1,
    }


def test_successful_handoff_collapses_old_camera_specific_alias():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = opposing_view_histogram(1)

    old_cam1 = attach_tracklet(
        DummyTrack(560, 220, h=40, appearance=cam1_view),
        cam1_view,
        cam1_view,
    )
    manager.update_all_tracks({"cam1": {1: old_cam1}}, 1, {"cam1": 0.0})
    manager.notify_track_lost(
        "cam1", 1, old_cam1, 2, timestamp_s=0.1
    )
    manager._handoffs.clear()

    # Simulate the consequence of an earlier failed transfer: the same car is
    # currently G#2 in cam2 while its old G#1 record remains dormant in cam1.
    current_cam2 = attach_tracklet(
        DummyTrack(
            60,
            240,
            h=80,
            history=[(55, 240), (58, 240), (60, 240)],
            appearance=cam2_view,
        ),
        cam2_view,
        cam2_view,
    )
    manager.bind_external_id("cam2", 7, 2, 3, source="test")
    manager.update_all_tracks(
        {"cam2": {7: current_cam2}}, 3, {"cam2": 1.0}
    )

    returning = attach_tracklet(
        DummyTrack(
            560,
            220,
            h=40,
            history=[(558, 220), (559, 220), (560, 220)],
            status="tentative",
            appearance=cam1_view,
        ),
        cam1_view,
        cam1_view,
    )
    assert manager.update_all_tracks(
        {"cam1": {9: returning}}, 4, {"cam1": 1.1}
    ) == {"cam1": {}}
    ids = manager.update_all_tracks(
        {"cam1": {9: returning}}, 5, {"cam1": 1.2}
    )

    assert ids == {"cam1": {9: 1}}
    assert manager.canonical_global_id(2) == 1
    assert any(
        event["type"] == "handoff_reconciled_dormant_alias"
        for event in manager.to_json({})["recent_events"]
    )


def test_close_cross_camera_tracks_outside_overlap_are_not_merged():
    manager = make_real_two_camera_manager()
    manager.overlap_regions = {
        ("cam1", "cam2"): np.array(
            [[0, 0], [20, 0], [20, 20], [0, 20]],
            dtype=np.float32,
        )
    }
    cam1 = DummyTrack(560, 220, h=40, appearance=one_hot_histogram(0))
    cam2 = DummyTrack(
        62,
        240,
        h=80,
        appearance=opposing_view_histogram(1),
    )

    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 1)

    assert ids == {"cam1": {1: 1}, "cam2": {7: 2}}


def test_two_valid_multiview_global_ids_are_never_cross_merged():
    manager = make_real_two_camera_manager()
    first_view = one_hot_histogram(0)
    second_view = one_hot_histogram(1)
    tracks = {
        "cam1": {
            1: DummyTrack(550, 220, h=40, appearance=first_view),
            2: DummyTrack(555, 220, h=40, appearance=second_view),
        },
        "cam2": {
            3: DummyTrack(50, 240, h=80, appearance=second_view),
            4: DummyTrack(55, 240, h=80, appearance=first_view),
        },
    }
    manager.bind_external_id("cam1", 1, 1, 0, source="test")
    manager.bind_external_id("cam2", 3, 1, 0, source="test")
    manager.bind_external_id("cam1", 2, 2, 0, source="test")
    manager.bind_external_id("cam2", 4, 2, 0, source="test")

    ids = manager.update_all_tracks(tracks, 1)

    assert ids == {
        "cam1": {1: 1, 2: 2},
        "cam2": {3: 1, 4: 2},
    }
    assert manager.canonical_global_id(1) == 1
    assert manager.canonical_global_id(2) == 2
    assert manager.to_json(tracks)["retired_global_ids"] == {}


def test_missing_appearance_does_not_bind_cross_camera_track():
    manager = make_real_two_camera_manager()
    source = DummyTrack(560, 220, h=40)
    source.appearance = None
    manager.update_all_tracks({"cam1": {1: source}}, 1)
    target = DummyTrack(62, 240, h=80)
    target.appearance = None

    ids = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: target}},
        2,
    )

    assert ids == {"cam1": {1: 1}, "cam2": {}}
    assert manager.get_global_id("cam2", 7) is None


def test_relaxed_appearance_requires_two_tracklet_samples():
    manager = make_real_two_camera_manager()
    source = DummyTrack(560, 220, h=40, appearance=one_hot_histogram(0))
    manager.update_all_tracks({"cam1": {1: source}}, 1)
    target = DummyTrack(
        62,
        240,
        h=80,
        appearance=opposing_view_histogram(1),
    )

    ids = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {7: target}},
        2,
    )

    assert ids == {"cam1": {1: 1}, "cam2": {}}
    events = manager.to_json(
        {"cam1": {1: source}, "cam2": {7: target}}
    )["recent_events"]
    assert any(
        event["type"] == "cross_camera_assignment_deferred"
        and event["appearance_distance"] > manager.appearance_threshold
        for event in events
    )


def test_bottom_center_anchor_remains_available_for_ground_homography():
    manager = make_real_two_camera_manager(shared_map_anchor="bottom_center")
    cam1 = DummyTrack(560, 220, h=40)
    cam2 = DummyTrack(62, 240, h=80)

    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 1)

    assert ids == {"cam1": {1: 1}, "cam2": {7: 2}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    assert all(
        observation["shared_map_anchor"]["reference"] == "bottom_center"
        for vehicle in registry["active_global_vehicles"].values()
        for observation in vehicle["observations"]
    )


def test_two_real_cameras_deduplicate_a_vehicle_in_calibrated_overlap():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 200, history=[(550, 200), (560, 200)])
    assert manager.update_all_tracks({"cam1": {1: cam1}}, 1)["cam1"][1] == 1

    # cam2 x=60 maps to the same world x=560 via its configured homography.
    cam2 = DummyTrack(60, 200, history=[(50, 200), (60, 200)])
    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 2)
    assert ids["cam2"][7] == 1


def test_two_real_cameras_keep_handoff_id_from_cam1_to_cam2():
    manager = make_real_two_camera_manager()
    source = DummyTrack(620, 200, history=[(560, 200), (590, 200), (620, 200)])
    assert manager.update_all_tracks({"cam1": {1: source}}, 1)["cam1"][1] == 1
    manager.update_all_tracks({"cam1": {1: source}}, 2)

    # The source leaves cam1. cam2's tentative local track appears at predicted world x=650.
    target = DummyTrack(150, 200, history=[(120, 200), (150, 200)], status="tentative")
    ids = manager.update_all_tracks({"cam2": {8: target}}, 3)
    assert ids["cam2"][8] == 1


def test_two_new_simultaneous_tracks_allocate_only_one_global_id():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 200, history=[(550, 200), (560, 200)])
    cam2 = DummyTrack(60, 200, history=[(50, 200), (60, 200)])

    ids = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}}, 1
    )

    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    assert manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})["next_global_id"] == 2


def test_dormant_identity_recovers_in_both_camera_directions():
    manager = make_real_two_camera_manager()
    source = DummyTrack(560, 200, history=[(560, 200), (560, 200)])
    assert manager.update_all_tracks(
        {"cam1": {1: source}}, 1, {"cam1": 1.0}
    )["cam1"][1] == 1

    manager.notify_track_lost("cam1", 1, source, 2, timestamp_s=1.1)
    target = DummyTrack(
        60, 200, history=[(60, 200), (60, 200), (60, 200)]
    )
    assert manager.update_all_tracks(
        {"cam2": {8: target}}, 3, {"cam2": 1.4}
    )["cam2"][8] == 1

    manager.notify_track_lost("cam2", 8, target, 4, timestamp_s=1.5)
    returning = DummyTrack(
        560, 200, history=[(560, 200), (560, 200), (560, 200)]
    )
    assert manager.update_all_tracks(
        {"cam1": {9: returning}}, 5, {"cam1": 1.8}
    )["cam1"][9] == 1
    events = manager.to_json({"cam1": {9: returning}})["recent_events"]
    recovered = [event for event in events if event["type"] == "dormant_global_id_recovered"]
    assert [(event["source_camera"], event["target_camera"]) for event in recovered] == [
        ("cam1", "cam2"),
        ("cam2", "cam1"),
    ]


def test_global_tracklet_gallery_survives_handoff_and_reverse_return():
    manager = make_real_two_camera_manager()
    first_view = one_hot_histogram(1)
    transition_view = one_hot_histogram(2)
    source = attach_tracklet(
        DummyTrack(560, 200, history=[(560, 200), (560, 200)]),
        first_view,
        transition_view,
    )
    # Simulate the old single-descriptor path choosing the wrong endpoint.
    source.appearance = first_view
    assert manager.update_all_tracks(
        {"cam1": {1: source}}, 1, {"cam1": 1.0}
    )["cam1"][1] == 1
    manager.notify_track_lost("cam1", 1, source, 2, timestamp_s=1.1)

    target = attach_tracklet(
        DummyTrack(60, 200, history=[(60, 200), (60, 200), (60, 200)]),
        transition_view,
    )
    assert manager.update_all_tracks(
        {"cam2": {8: target}}, 3, {"cam2": 1.4}
    )["cam2"][8] == 1
    manager.notify_track_lost("cam2", 8, target, 4, timestamp_s=1.5)

    returning = attach_tracklet(
        DummyTrack(560, 200, history=[(560, 200), (560, 200), (560, 200)]),
        first_view,
    )
    assert manager.update_all_tracks(
        {"cam1": {9: returning}}, 5, {"cam1": 1.8}
    )["cam1"][9] == 1
    lifecycle = manager.to_json({"cam1": {9: returning}})["identity_lifecycle"]["1"]
    assert lifecycle["appearance_sample_count"] == 2


def test_dormant_identity_is_not_recovered_after_retention_window():
    manager = make_real_two_camera_manager()
    manager.identity_retention_seconds = 0.5
    source = DummyTrack(560, 200, history=[(560, 200), (560, 200)])
    manager.update_all_tracks({"cam1": {1: source}}, 1, {"cam1": 1.0})
    manager.notify_track_lost("cam1", 1, source, 2, timestamp_s=1.1)

    target = DummyTrack(60, 200, history=[(60, 200), (60, 200)])
    ids = manager.update_all_tracks({"cam2": {8: target}}, 3, {"cam2": 2.0})

    assert ids["cam2"][8] == 2
    lifecycle = manager.to_json({"cam2": {8: target}})["identity_lifecycle"]
    assert lifecycle["1"]["state"] == "expired"


def test_dormant_identity_recovers_in_the_same_camera_after_parking_gap():
    manager = make_real_two_camera_manager()
    parked = DummyTrack(300, 260, history=[(300, 260), (300, 260)])
    assert manager.update_all_tracks(
        {"cam1": {1: parked}}, 1, {"cam1": 1.0}
    )["cam1"][1] == 1
    manager.notify_track_lost("cam1", 1, parked, 2, timestamp_s=1.1)
    manager.notify_track_expired(
        "cam1", 1, parked.cx, parked.cy, parked.w, parked.h,
        parked.appearance, 3, timestamp_s=2.0,
    )

    moving_again = DummyTrack(
        315, 260, history=[(305, 260), (310, 260), (315, 260)]
    )
    ids = manager.update_all_tracks(
        {"cam1": {9: moving_again}}, 4, {"cam1": 2.4}
    )

    assert ids["cam1"][9] == 1
    events = manager.to_json({"cam1": {9: moving_again}})["recent_events"]
    assert any(
        event["type"] == "dormant_global_id_recovered"
        and event["source_camera"] == event["target_camera"] == "cam1"
        for event in events
    )


def test_one_frame_noise_cannot_consume_dormant_id_but_mature_track_can():
    manager = make_real_two_camera_manager()
    parked = DummyTrack(300, 260, history=[(300, 260)] * 3)
    assert manager.update_all_tracks(
        {"cam1": {1: parked}}, 1, {"cam1": 1.0}
    )["cam1"][1] == 1
    manager.notify_track_lost("cam1", 1, parked, 2, timestamp_s=1.1)

    matching_noise = DummyTrack(305, 260, history=[(305, 260)])
    first = manager.update_all_tracks(
        {"cam1": {9: matching_noise}}, 3, {"cam1": 1.2}
    )
    assert first == {"cam1": {}}
    assert manager.get_global_id("cam1", 9) is None
    assert manager.to_json({"cam1": {9: matching_noise}})["next_global_id"] == 2

    persistent = DummyTrack(
        315,
        260,
        history=[(305, 260), (310, 260), (315, 260)],
    )
    recovered = manager.update_all_tracks(
        {"cam1": {9: persistent}}, 4, {"cam1": 1.3}
    )
    assert recovered == {"cam1": {9: 1}}


def test_overlap_zone_opens_handoff_without_relying_on_image_edge_name():
    manager = make_real_two_camera_manager()
    source = DummyTrack(550, 200, history=[(540, 200), (550, 200)])
    manager.update_all_tracks({"cam1": {1: source}}, 1)
    source = DummyTrack(570, 200, history=[(550, 200), (570, 200)])
    target = DummyTrack(
        70, 200, history=[(60, 200), (70, 200)], status="tentative"
    )

    ids = manager.update_all_tracks(
        {"cam1": {1: source}, "cam2": {8: target}}, 2
    )

    assert ids["cam2"][8] == 1
    events = manager.to_json({"cam1": {1: source}, "cam2": {8: target}})["recent_events"]
    assert any(
        event["type"] == "handoff_opened" and event["edge"] == "overlap"
        for event in events
    )


def test_identity_exits_only_after_local_track_expires_in_exit_zone():
    manager = make_real_two_camera_manager()
    manager.exit_zones = {
        "cam1": [np.array([[520, 160], [600, 160], [600, 240], [520, 240]], dtype=np.float32)]
    }
    source = DummyTrack(560, 200, history=[(560, 200), (560, 200)])
    manager.update_all_tracks({"cam1": {1: source}}, 1, {"cam1": 1.0})
    manager.notify_track_lost("cam1", 1, source, 2, timestamp_s=1.1)

    before_expiry = manager.to_json({})["identity_lifecycle"]["1"]
    assert before_expiry["state"] == "dormant"
    manager.notify_track_expired(
        "cam1", 1, 560, 200, source.w, source.h, source.appearance, 3,
        timestamp_s=1.2,
    )

    lifecycle = manager.to_json({})["identity_lifecycle"]
    assert lifecycle["1"]["state"] == "exited"
    target = DummyTrack(60, 200, history=[(60, 200), (60, 200)])
    assert manager.update_all_tracks(
        {"cam2": {8: target}}, 4, {"cam2": 1.3}
    )["cam2"][8] == 2


def test_protected_track_skips_new_id_then_allocates_when_unprotected():
    manager = make_manager()
    candidate = DummyTrack(250, 180)

    protected = manager.update_all_tracks(
        {"cam3": {9: candidate}},
        1,
        protected_local_keys={("cam3", 9)},
    )

    assert protected == {"cam3": {}}
    assert manager.get_global_id("cam3", 9) is None
    assert manager.to_json({"cam3": {9: candidate}})["next_global_id"] == 1

    released = manager.update_all_tracks({"cam3": {9: candidate}}, 2)
    assert released == {"cam3": {9: 1}}


def test_protected_track_cannot_consume_pending_handoff():
    manager = make_manager()
    manager.update_all_tracks({"cam4": {1: fast_left_track(60)}}, 1)
    manager.update_all_tracks({"cam4": {1: fast_left_track(30)}}, 2)
    target = fast_left_track(515, status="tentative")

    protected = manager.update_all_tracks(
        {"cam3": {9: target}},
        5,
        protected_local_keys={("cam3", 9)},
    )
    assert protected == {"cam3": {}}
    assert manager.get_global_id("cam3", 9) is None

    released = manager.update_all_tracks({"cam3": {9: target}}, 6)
    assert released == {"cam3": {9: 1}}


def test_protected_track_cannot_consume_dormant_identity():
    manager = make_real_two_camera_manager()
    parked = DummyTrack(300, 260, history=[(300, 260), (300, 260)])
    manager.update_all_tracks(
        {"cam1": {1: parked}}, 1, {"cam1": 1.0}
    )
    manager.notify_track_lost(
        "cam1", 1, parked, 2, timestamp_s=1.1
    )
    returning = DummyTrack(
        310,
        260,
        history=[(305, 260), (308, 260), (310, 260)],
    )

    protected = manager.update_all_tracks(
        {"cam1": {9: returning}},
        3,
        {"cam1": 1.3},
        protected_local_keys={("cam1", 9)},
    )
    assert protected == {"cam1": {}}
    assert manager.get_global_id("cam1", 9) is None

    released = manager.update_all_tracks(
        {"cam1": {9: returning}}, 4, {"cam1": 1.4}
    )
    assert released == {"cam1": {9: 1}}


def test_protected_track_skips_simultaneous_and_existing_overlap_matching():
    manager = make_real_two_camera_manager()
    cam1 = DummyTrack(560, 200, history=[(550, 200), (560, 200)])
    cam2 = DummyTrack(60, 200, history=[(50, 200), (60, 200)])

    protected = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}},
        1,
        protected_local_keys={("cam2", 7)},
    )
    assert protected == {"cam1": {1: 1}, "cam2": {}}
    assert manager.get_global_id("cam2", 7) is None

    released = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}}, 2
    )
    assert released == {"cam1": {1: 1}, "cam2": {7: 1}}


def test_protected_track_skips_same_camera_duplicate_matching():
    manager = make_manager()
    primary = fast_left_track(250)
    manager.update_all_tracks({"cam3": {1: primary}}, 1)
    echo = fast_left_track(262)

    protected = manager.update_all_tracks(
        {"cam3": {1: primary, 2: echo}},
        2,
        protected_local_keys={("cam3", 2)},
    )
    assert protected == {"cam3": {1: 1}}
    assert manager.get_global_id("cam3", 2) is None

    released = manager.update_all_tracks(
        {"cam3": {1: primary, 2: echo}}, 3
    )
    assert released == {"cam3": {1: 1}}
    manager.update_all_tracks({"cam3": {1: primary, 2: echo}}, 4)
    released = manager.update_all_tracks(
        {"cam3": {1: primary, 2: echo}}, 5
    )
    assert released == {"cam3": {1: 1, 2: 1}}


def test_explicit_external_binding_overrides_call_scoped_protection():
    manager = make_manager()
    candidate = DummyTrack(250, 180, status="tentative")
    assert manager.bind_external_id(
        "cam3", 9, 41, 1, source="verified_slot_recovery"
    ) == 41

    ids = manager.update_all_tracks(
        {"cam3": {9: candidate}},
        1,
        protected_local_keys={("cam3", 9)},
    )

    assert ids == {"cam3": {9: 41}}
    assert manager.get_global_id("cam3", 9) == 41


def test_short_destination_history_overrides_reversed_motion_vector():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    jittered_cam2 = np.zeros((16, 16), dtype=np.float32)
    jittered_cam2.flat[4] = 0.72
    jittered_cam2.flat[5] = 0.28
    jittered_cam2 = cv2.normalize(jittered_cam2, jittered_cam2)

    old_cam2 = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.update_all_tracks({"cam2": {1: old_cam2}}, 1, {"cam2": 0.0})

    source = DummyTrack(
        560,
        220,
        h=40,
        history=[(540, 220), (550, 220), (560, 220)],
        appearance=cam1_view,
    )
    manager.bind_external_id("cam1", 2, 1, 2, source="test")
    manager.update_all_tracks({"cam1": {2: source}}, 2, {"cam1": 0.1})
    manager.notify_track_lost("cam1", 2, source, 3, timestamp_s=0.2)
    manager._handoffs.clear()

    # The motion blob points backwards, but the vehicle is at the last shared
    # position and agrees with this GID's prior cam2 appearance.
    returning = DummyTrack(
        60,
        240,
        h=80,
        history=[(80, 240), (70, 240), (60, 240)],
        appearance=jittered_cam2,
    )
    ids = manager.update_all_tracks(
        {"cam2": {9: returning}}, 6, {"cam2": 0.7}
    )

    assert ids == {"cam2": {9: 1}}
    assert any(
        event["type"] == "dormant_direction_override_strong_reid"
        for event in manager.to_json({})["recent_events"]
    )


def test_recent_dormant_ambiguity_blocks_unrelated_handoff_from_stealing_track():
    manager = make_real_two_camera_manager()
    target_view = one_hot_histogram(4)
    jittered = np.zeros((16, 16), dtype=np.float32)
    jittered.flat[4] = 0.78
    jittered.flat[5] = 0.22
    jittered = cv2.normalize(jittered, jittered)
    manager._global_created_frames[1] = 1
    manager._handoffs = [
        HandoffEntry(
            global_id=1,
            source_cam="cam1",
            source_local_track_id=1,
            target_cam="cam2",
            exit_edge="overlap",
            last_world=(560.0, 200.0),
            velocity_world=(0.0, 0.0),
            bbox_size=(42, 40),
            appearance=one_hot_histogram(9),
            appearance_samples=(one_hot_histogram(9),),
            target_appearance_samples=(target_view,),
            target_bbox_size=(42, 80),
            created_at_frame=1,
            updated_at_frame=1,
        )
    ]
    manager._ambiguous_local_identities[("cam2", 9)] = (4, {8})
    uncertain = DummyTrack(
        60,
        240,
        h=80,
        history=[(58, 240), (59, 240), (60, 240)],
        appearance=jittered,
    )

    blocked = manager.update_all_tracks(
        {"cam2": {9: uncertain}}, 2, {"cam2": 0.2}
    )

    assert blocked == {"cam2": {}}
    assert any(
        event["type"] == "handoff_rejected_recent_identity_ambiguity"
        for event in manager.to_json({})["recent_events"]
    )


def test_same_gid_cannot_remain_on_two_non_echo_tracks_in_one_camera():
    manager = make_real_two_camera_manager()
    owner_view = one_hot_histogram(0)
    wrong_view = one_hot_histogram(8)
    owner = DummyTrack(120, 220, appearance=owner_view)
    assert manager.update_all_tracks({"cam1": {1: owner}}, 1)["cam1"] == {
        1: 1
    }
    intruder = DummyTrack(460, 220, appearance=wrong_view)
    manager.bind_external_id("cam1", 2, 1, 2, source="test")

    ids = manager.update_all_tracks(
        {"cam1": {1: owner, 2: intruder}}, 2
    )

    assert ids == {"cam1": {1: 1}}
    assert manager.get_global_id("cam1", 2) is None
    assert any(
        event["type"] == "same_camera_global_conflict_detached"
        for event in manager.to_json({})["recent_events"]
    )


def test_same_camera_moderate_near_miss_waits_instead_of_creating_new_gid():
    manager = make_real_two_camera_manager()
    base = one_hot_histogram(0)
    jittered = np.zeros((16, 16), dtype=np.float32)
    jittered.flat[0] = 0.72
    jittered.flat[1] = 0.28
    jittered = cv2.normalize(jittered, jittered)
    old = DummyTrack(300, 240, appearance=base)
    assert manager.update_all_tracks(
        {"cam1": {1: old}}, 1, {"cam1": 0.0}
    )["cam1"][1] == 1
    manager.notify_track_lost("cam1", 1, old, 2, timestamp_s=0.1)

    fragment = DummyTrack(
        304,
        240,
        history=[(302, 240), (303, 240), (304, 240)],
        appearance=jittered,
    )
    ids = manager.update_all_tracks(
        {"cam1": {9: fragment}}, 3, {"cam1": 0.3}
    )

    assert 0.30 < manager._appearance_distance(old, fragment) <= 0.45
    assert ids == {"cam1": {}}
    assert manager.to_json({})["next_global_id"] == 2
    assert any(
        event["type"] == "new_global_id_deferred_dormant_near_miss"
        for event in manager.to_json({})["recent_events"]
    )


def test_recent_ambiguity_owner_beats_alternate_dormant_identity():
    manager = make_real_two_camera_manager()
    view = one_hot_histogram(0)

    first_cam2 = DummyTrack(60, 240, h=80, appearance=view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.update_all_tracks({"cam2": {1: first_cam2}}, 1, {"cam2": 0.0})
    first_cam1 = DummyTrack(
        560,
        220,
        h=40,
        history=[(540, 220), (550, 220), (560, 220)],
        appearance=view,
    )
    manager.bind_external_id("cam1", 2, 1, 2, source="test")
    manager.update_all_tracks({"cam1": {2: first_cam1}}, 2, {"cam1": 0.1})
    manager.notify_track_lost("cam1", 2, first_cam1, 3, timestamp_s=0.2)
    manager._handoffs.clear()

    # A second dormant identity is deliberately made equally plausible.
    second_cam2 = DummyTrack(60, 240, h=80, appearance=view)
    manager.bind_external_id("cam2", 3, 2, 3, source="test")
    manager.update_all_tracks({"cam2": {3: second_cam2}}, 3, {"cam2": 0.2})
    second_cam1 = DummyTrack(560, 220, h=40, appearance=view)
    manager.bind_external_id("cam1", 4, 2, 4, source="test")
    manager.update_all_tracks({"cam1": {4: second_cam1}}, 4, {"cam1": 0.3})
    manager.notify_track_lost("cam1", 4, second_cam1, 5, timestamp_s=0.4)
    manager._handoffs.clear()

    candidate = DummyTrack(
        60,
        240,
        h=80,
        history=[(80, 240), (70, 240), (60, 240)],
        appearance=view,
    )
    manager._ambiguous_local_identities[("cam2", 9)] = (9, {1})
    ids = manager.update_all_tracks(
        {"cam2": {9: candidate}}, 8, {"cam2": 3.0}
    )
    assert ids == {"cam2": {9: 1}}
    assert any(
        event["type"] == "dormant_reid_rejected_recent_identity_ambiguity"
        and event["global_id"] == 2
        for event in manager.to_json({})["recent_events"]
    )


def test_short_cross_camera_return_accepts_54_units_with_target_history():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    first_cam2 = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.update_all_tracks({"cam2": {1: first_cam2}}, 1, {"cam2": 0.0})
    cam1 = DummyTrack(560, 220, h=40, appearance=cam1_view)
    manager.bind_external_id("cam1", 2, 1, 2, source="test")
    manager.update_all_tracks({"cam1": {2: cam1}}, 2, {"cam1": 0.1})
    manager.notify_track_lost("cam1", 2, cam1, 3, timestamp_s=0.2)
    manager._handoffs.clear()

    returning = DummyTrack(
        114,
        240,
        h=80,
        history=[(110, 240), (112, 240), (114, 240)],
        appearance=cam2_view,
    )
    ids = manager.update_all_tracks(
        {"cam2": {9: returning}}, 8, {"cam2": 1.0}
    )

    assert ids == {"cam2": {9: 1}}


def test_mature_two_camera_identity_survives_long_blind_region():
    manager = make_real_two_camera_manager()
    manager.identity_retention_seconds = 60.0
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    cam1 = attach_tracklet(
        DummyTrack(560, 220, h=40, appearance=cam1_view),
        cam1_view,
        cam1_view,
        cam1_view,
        cam1_view,
    )
    cam2 = attach_tracklet(
        DummyTrack(60, 240, h=80, appearance=cam2_view),
        cam2_view,
        cam2_view,
        cam2_view,
        cam2_view,
    )
    assert manager.update_all_tracks(
        {"cam1": {1: cam1}}, 1, {"cam1": 0.0}
    ) == {"cam1": {1: 1}}
    manager.bind_external_id("cam2", 2, 1, 2, source="test")
    manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {2: cam2}},
        200,
        {"cam1": 0.0, "cam2": 0.0},
    )
    assert manager._identity_is_established(manager._identities[1])
    manager.notify_track_lost("cam1", 1, cam1, 201, timestamp_s=0.1)
    manager.notify_track_lost("cam2", 2, cam2, 201, timestamp_s=0.1)
    manager._handoffs.clear()

    # The car crosses a real blind region for longer than the base 60-second
    # TTL and reappears 54 shared-map units away in a previously seen camera.
    returning = attach_tracklet(
        DummyTrack(
            614,
            220,
            h=40,
            history=[(620, 220), (617, 220), (614, 220)],
            appearance=cam1_view,
        ),
        cam1_view,
        cam1_view,
    )
    ids = manager.update_all_tracks(
        {"cam1": {9: returning}}, 830, {"cam1": 78.1}
    )

    assert ids == {"cam1": {9: 1}}


def test_destination_gallery_overrides_bad_direction_for_five_second_gap():
    manager = make_real_two_camera_manager()
    cam1_view = one_hot_histogram(0)
    cam2_view = one_hot_histogram(4)
    jittered_cam2 = np.zeros((16, 16), dtype=np.float32)
    jittered_cam2.flat[4] = 0.80
    jittered_cam2.flat[5] = 0.20
    jittered_cam2 = cv2.normalize(jittered_cam2, jittered_cam2)
    old_cam2 = DummyTrack(60, 240, h=80, appearance=cam2_view)
    manager.bind_external_id("cam2", 1, 1, 1, source="test")
    manager.update_all_tracks({"cam2": {1: old_cam2}}, 1, {"cam2": 0.0})
    source = DummyTrack(
        560,
        220,
        h=40,
        history=[(540, 220), (550, 220), (560, 220)],
        appearance=cam1_view,
    )
    manager.bind_external_id("cam1", 2, 1, 2, source="test")
    manager.update_all_tracks({"cam1": {2: source}}, 2, {"cam1": 0.1})
    manager.notify_track_lost("cam1", 2, source, 3, timestamp_s=0.2)
    manager._handoffs.clear()
    returning = DummyTrack(
        114,
        240,
        h=80,
        history=[(130, 240), (122, 240), (114, 240)],
        appearance=jittered_cam2,
    )

    ids = manager.update_all_tracks(
        {"cam2": {9: returning}}, 40, {"cam2": 4.9}
    )

    assert ids == {"cam2": {9: 1}}
