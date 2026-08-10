"""Focused regression tests for predictive multi-camera handoff."""

from dataclasses import dataclass

import numpy as np

from cross_camera_manager import CrossCameraManager


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
