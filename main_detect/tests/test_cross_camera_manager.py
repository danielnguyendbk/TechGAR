"""Focused regression tests for predictive multi-camera handoff."""

from dataclasses import dataclass

import cv2
import numpy as np

from techgar.cross_camera_manager import CrossCameraManager
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


def test_same_camera_motion_echo_keeps_one_global_id_and_map_observation():
    manager = make_manager()
    primary = fast_left_track(250)
    assert manager.update_all_tracks({"cam3": {1: primary}}, 1)["cam3"][1] == 1
    echo = fast_left_track(262)
    ids = manager.update_all_tracks({"cam3": {1: primary, 2: echo}}, 2)
    assert ids["cam3"][2] == 1
    registry = manager.to_json({"cam3": {1: primary, 2: echo}})
    assert registry["next_global_id"] == 2
    assert registry["map_vehicles"]["1"]["observation_count"] == 1


def test_lost_track_continuation_merges_later_id_back_to_original_id():
    manager = make_manager()
    original = fast_left_track(250)
    later = fast_left_track(420)
    ids = manager.update_all_tracks({"cam3": {1: original, 2: later}}, 1)
    assert ids["cam3"] == {1: 1, 2: 2}

    manager.notify_track_lost("cam3", 1, original, 2)
    # The local #2 tracker is now the only visible continuation at the
    # predicted position of #1. It must be rewritten to global #1.
    continuation = fast_left_track(235)
    ids = manager.update_all_tracks({"cam3": {2: continuation}}, 3)
    assert ids["cam3"][2] == 1
    events = manager.to_json({"cam3": {2: continuation}})["recent_events"]
    assert any(e["type"] == "global_id_merged" and e["superseded_global_id"] == 2 for e in events)


def test_two_existing_nearby_slow_boxes_merge_and_retire_larger_id():
    manager = make_manager()
    first = DummyTrack(140, 180, history=[(138, 180), (139, 180), (140, 180)])
    second = DummyTrack(330, 180, history=[(328, 180), (329, 180), (330, 180)])
    ids = manager.update_all_tracks({"cam3": {1: first, 2: second}}, 1)
    assert ids["cam3"] == {1: 1, 2: 2}

    # Both already have IDs, then the slow-motion old/new boxes become close.
    first = DummyTrack(220, 180, history=[(218, 180), (219, 180), (220, 180)])
    second = DummyTrack(255, 180, history=[(253, 180), (254, 180), (255, 180)])
    ids = manager.update_all_tracks({"cam3": {1: first, 2: second}}, 2)
    assert ids["cam3"] == {1: 1, 2: 1}

    # Even an external subsystem referencing retired #2 cannot revive it.
    assert manager.bind_external_id("cam4", 9, 2, 3, source="test") == 1
    registry = manager.to_json({"cam3": {1: first, 2: second}})
    assert registry["retired_global_ids"] == {"2": 1}
    assert set(registry["map_vehicles"]) == {"1"}


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

    ids = manager.update_all_tracks({"cam1": {1: cam1}, "cam2": {7: cam2}}, 1)

    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    assert registry["retired_global_ids"] == {"2": 1}
    assert registry["identity_lifecycle"]["1"]["appearance_sample_count"] == 2
    evidence = next(
        event
        for event in registry["recent_events"]
        if event["type"] == "cross_camera_duplicate_matched"
    )
    assert evidence["world_distance"] == 2.0
    assert evidence["appearance_distance"] > manager.appearance_threshold
    assert evidence["appearance_threshold"] == manager.relaxed_appearance_threshold
    assert evidence["adaptive_appearance"] is True


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
    ids = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: cam2}},
        2,
    )

    assert ids == {"cam1": {1: 1}, "cam2": {7: 1}}
    registry = manager.to_json({"cam1": {1: cam1}, "cam2": {7: cam2}})
    assert registry["next_global_id"] == 2
    assert registry["retired_global_ids"] == {}
    assert any(
        event["type"] == "cross_camera_unbound_matched"
        and event["adaptive_appearance"] is True
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
    assert any(event["type"] == "cross_camera_unbound_matched" for event in events)


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

    ids = manager.update_all_tracks(
        {"cam1": {1: cam1}, "cam2": {7: left, 8: right}},
        1,
    )

    assert len(set(ids["cam1"].values()) | set(ids["cam2"].values())) == 3
    registry = manager.to_json(
        {"cam1": {1: cam1}, "cam2": {7: left, 8: right}}
    )
    assert registry["retired_global_ids"] == {}


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
        60, 200, history=[(60, 200), (60, 200)], status="tentative"
    )
    assert manager.update_all_tracks(
        {"cam2": {8: target}}, 3, {"cam2": 1.4}
    )["cam2"][8] == 1

    manager.notify_track_lost("cam2", 8, target, 4, timestamp_s=1.5)
    returning = DummyTrack(
        560, 200, history=[(560, 200), (560, 200)], status="tentative"
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
        DummyTrack(60, 200, history=[(60, 200), (60, 200)], status="tentative"),
        transition_view,
    )
    assert manager.update_all_tracks(
        {"cam2": {8: target}}, 3, {"cam2": 1.4}
    )["cam2"][8] == 1
    manager.notify_track_lost("cam2", 8, target, 4, timestamp_s=1.5)

    returning = attach_tracklet(
        DummyTrack(560, 200, history=[(560, 200), (560, 200)], status="tentative"),
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
        315, 260, history=[(305, 260), (315, 260)], status="tentative"
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
