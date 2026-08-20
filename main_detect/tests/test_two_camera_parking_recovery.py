from types import SimpleNamespace

import numpy as np

from techgar.slot_vehicle_binder import RecoveryBatchResult
from two_camera import (
    _project_points_between_cameras,
    build_recovery_track_payload,
    build_recovery_priority_regions,
    cancel_observed_recovery_tokens,
    collect_binder_global_tracks,
    recover_departing_vehicle_ids,
)


class FakeManager:
    def __init__(self):
        self.bindings = {}

    def get_global_id(self, camera_id, local_id):
        return self.bindings.get((camera_id, local_id))

    def bind_external_id(self, camera_id, local_id, global_id, frame_idx, source):
        self.bindings[(camera_id, local_id)] = global_id

    @staticmethod
    def canonical_global_id(global_id):
        return int(global_id)


class FakeBinder:
    def __init__(self, token, result=None):
        self.token = token
        self.result = result or RecoveryBatchResult()
        self.received = None
        self.cancelled = []
        self.false_empty_grace_seconds = 1.25

    def export_recovery_tokens(self, _timestamp_s):
        return [self.token] if self.token is not None else []

    def batch_recover_ids(self, candidates, *_args, **_kwargs):
        self.received = dict(candidates)
        return self.result

    def cancel_recovery_for_global_id(self, global_id, reason):
        self.cancelled.append((global_id, reason))
        self.token = None

    def recovery_priority_regions(self, _timestamp_s):
        return [] if self.token is None else [self.token["polygon"]]


def token(slot_id="E07", global_id=30, center=(50.0, 50.0)):
    return {
        "slot_id": slot_id,
        "global_id": global_id,
        "polygon": np.asarray(
            [[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32
        ),
        "center": center,
    }


def track_at(x=40, y=35):
    hist = np.zeros((32, 32), dtype=np.float32)
    hist[0, 0] = 1.0
    value = SimpleNamespace(
        bbox=(x, y, 20, 30),
        appearance=hist,
        first_observation_bbox=(x, y, 20, 30),
        first_observation_timestamp_s=8.5,
    )
    return value


def test_camera_projection_round_trip_uses_shared_world_plane():
    transforms = {
        "cam1": np.eye(3, dtype=np.float64),
        "cam2": np.asarray([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    }

    in_cam1 = _project_points_between_cameras(
        [(120, 40)], "cam2", "cam1", transforms
    )
    round_trip = _project_points_between_cameras(
        in_cam1, "cam1", "cam2", transforms
    )

    assert np.allclose(in_cam1[0], (220, 40))
    assert np.allclose(round_trip[0], (120, 40))


def test_cross_camera_payload_contains_transformed_recovery_geometry():
    transforms = {
        "cam1": np.eye(3, dtype=np.float64),
        "cam2": np.asarray([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    }

    payload = build_recovery_track_payload(
        track_at(10, 20),
        "cam2",
        "cam1",
        transforms,
        shared_map_anchor="bbox_center",
    )

    assert payload["camera_id"] == "cam2"
    assert np.allclose(payload["recovery_position"], (120, 35))
    assert payload["recovery_bbox"][0] == 110
    assert payload["recovery_first_bbox"] is None
    assert payload["recovery_size_ratio"] == 1.0
    assert payload["recovery_first_timestamp_s"] == 8.5
    assert payload["recovery_anchor"] == "bbox_center"


def test_same_camera_payload_uses_one_explicit_anchor_for_point_and_bbox():
    transforms = {"cam1": np.eye(3, dtype=np.float64)}
    value = SimpleNamespace(
        bbox=(40, 30, 20, 30),
        first_observation_bbox=(30, 20, 16, 24),
        appearance=None,
    )

    payload = build_recovery_track_payload(
        value,
        "cam1",
        "cam1",
        transforms,
        shared_map_anchor="bbox_center",
    )

    assert payload["recovery_bbox"] == (40.0, 30.0, 20.0, 30.0)
    assert payload["recovery_first_bbox"] == (30, 20, 16, 24)
    assert np.allclose(payload["recovery_position"], (50, 45))
    assert np.allclose(payload["recovery_first_position"], (38, 32))


def test_projective_payload_projects_anchor_from_each_matching_bbox():
    transforms = {
        "cam1": np.eye(3, dtype=np.float64),
        "cam2": np.asarray(
            [[1.0, 0.08, 25.0], [0.03, 1.0, 12.0], [0.001, 0.0004, 1.0]],
            dtype=np.float64,
        ),
    }
    value = SimpleNamespace(
        bbox=(80, 40, 30, 50),
        first_observation_bbox=(55, 35, 24, 42),
        appearance=None,
    )

    payload = build_recovery_track_payload(
        value, "cam2", "cam1", transforms, shared_map_anchor="bbox_center"
    )
    expected = _project_points_between_cameras(
        [(95, 65), (67, 56)], "cam2", "cam1", transforms
    )

    assert np.allclose(payload["recovery_position"], expected[0])
    assert np.allclose(payload["recovery_first_position"], expected[1])
    point = payload["recovery_position"]
    x, y, width, height = payload["recovery_bbox"]
    assert x <= point[0] <= x + width
    assert y <= point[1] <= y + height
    # Image-box corners are not ground-plane points.  Keep the projected
    # anchor/history, but do not treat a projected historical rectangle as a
    # metrically meaningful cross-camera size constraint.
    assert payload["recovery_first_bbox"] is None
    assert payload["recovery_size_ratio"] == 1.0


def test_bottom_center_anchor_is_available_for_ground_contact_calibration():
    transforms = {"cam1": np.eye(3, dtype=np.float64)}

    payload = build_recovery_track_payload(
        track_at(10, 20),
        "cam1",
        "cam1",
        transforms,
        shared_map_anchor="bottom_center",
    )

    assert payload["recovery_anchor"] == "bottom_center"
    assert np.allclose(payload["recovery_position"], (20, 50))
    assert np.allclose(payload["recovery_first_position"], (20, 50))


def test_unknown_recovery_anchor_is_rejected():
    transforms = {"cam1": np.eye(3, dtype=np.float64)}

    try:
        build_recovery_track_payload(
            track_at(), "cam1", "cam1", transforms, shared_map_anchor="centroid"
        )
    except ValueError as exc:
        assert "bbox_center" in str(exc)
    else:
        raise AssertionError("unknown recovery anchor should be rejected")


def test_verified_batch_recovery_binds_external_id_before_manager_allocation():
    manager = FakeManager()
    recovered = RecoveryBatchResult(recovered_ids={("cam1", 7): 30})
    binder = FakeBinder(token(), recovered)
    binders = {"cam1": binder, "cam2": FakeBinder(None)}
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    protected, _diagnostics = recover_departing_vehicle_ids(
        {"cam1": {7: track_at()}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
    )

    assert manager.bindings[("cam1", 7)] == 30
    assert protected == set()
    assert binder.received is not None


def test_recovery_pipeline_passes_calibration_anchor_to_binder_payload():
    manager = FakeManager()
    binder = FakeBinder(token())
    binders = {"cam1": binder, "cam2": FakeBinder(None)}
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    recover_departing_vehicle_ids(
        {"cam1": {7: track_at()}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
        shared_map_anchor="bottom_center",
    )

    assert binder.received is not None
    payload = binder.received[("cam1", 7)]
    assert payload["recovery_anchor"] == "bottom_center"
    assert np.allclose(payload["recovery_position"], (50, 65))


def test_dispatcher_routes_fast_fragment_using_saved_token_continuation():
    manager = FakeManager()
    saved_token = token()
    saved_token["continuation_evidence"] = [
        {
            "local_key": ("cam1", 69),
            "last_center": (75.0, 65.0),
            "last_seen_s": 9.0,
            "qualified_predeparture": False,
            "originated_in_slot": True,
        }
    ]
    binder = FakeBinder(saved_token)
    binders = {"cam1": binder, "cam2": FakeBinder(None)}
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    # Bottom center (210, 65) is outside the ordinary 45%-diagonal token
    # radius, but is a plausible continuation of the one fragment observed
    # inside the slot before a fast dropout.
    recover_departing_vehicle_ids(
        {"cam1": {70: track_at(200, 35)}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
    )

    assert binder.received is not None
    assert ("cam1", 70) in binder.received


def test_candidate_between_two_token_owners_stays_protected_and_unbound():
    manager = FakeManager()
    binders = {
        "cam1": FakeBinder(token("E07", 30)),
        "cam2": FakeBinder(token("C08", 31)),
    }
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    protected, diagnostics = recover_departing_vehicle_ids(
        {"cam1": {7: track_at()}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
    )

    assert protected == {("cam1", 7)}
    assert manager.bindings == {}
    assert diagnostics[0]["type"] == "slot_recovery_owner_ambiguous"
    assert all(binder.received is None for binder in binders.values())


def test_token_gid_already_visible_cannot_be_given_to_second_local_track():
    manager = FakeManager()
    manager.bindings[("cam1", 3)] = 30
    binder = FakeBinder(
        token(),
        RecoveryBatchResult(recovered_ids={("cam1", 7): 30}),
    )
    binders = {"cam1": binder, "cam2": FakeBinder(None)}
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    protected, diagnostics = recover_departing_vehicle_ids(
        {"cam1": {3: track_at(10, 10), 7: track_at()}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
    )

    assert manager.bindings == {("cam1", 3): 30}
    assert binder.received is None
    assert protected == set()
    assert diagnostics == []


def test_duplicate_cross_camera_tokens_keep_only_newest_owner():
    manager = FakeManager()
    older = token("E07", 30)
    older.update({"created_at_s": 8.0, "confirmed_empty": True})
    newer = token("C08", 30)
    newer.update({"created_at_s": 9.0, "confirmed_empty": True})
    cam1_binder = FakeBinder(older)
    cam2_binder = FakeBinder(
        newer,
        RecoveryBatchResult(recovered_ids={("cam1", 7): 30}),
    )
    binders = {"cam1": cam1_binder, "cam2": cam2_binder}
    transforms = {camera_id: np.eye(3) for camera_id in binders}

    protected, diagnostics = recover_departing_vehicle_ids(
        {"cam1": {7: track_at()}, "cam2": {}},
        manager,
        binders,
        transforms,
        100,
        {"cam1": 10.0, "cam2": 10.0},
        0.45,
        shared_map_anchor="bbox_center",
    )

    assert manager.bindings[("cam1", 7)] == 30
    assert protected == set()
    assert cam1_binder.received is None
    assert cam1_binder.cancelled == [(30, "duplicate_cross_camera_token")]
    assert cam2_binder.received is not None
    assert diagnostics[0]["type"] == "slot_recovery_duplicate_token_cancelled"


def test_priority_region_from_other_camera_is_projected_into_target_view():
    binders = {
        "cam1": FakeBinder(token(center=(50.0, 50.0))),
        "cam2": FakeBinder(token("C08", 31, center=(25.0, 25.0))),
    }
    transforms = {
        "cam1": np.eye(3),
        "cam2": np.asarray([[1, 0, 100], [0, 1, 0], [0, 0, 1]], dtype=float),
    }

    regions = build_recovery_priority_regions(
        "cam1",
        binders,
        transforms,
        {"cam1": 10.0, "cam2": 10.0},
    )

    own = next(item for item in regions if item["source_camera"] == "cam1")
    cross = next(item for item in regions if item["source_camera"] == "cam2")
    assert np.allclose(own["polygon"][0], (20, 20))
    assert np.allclose(cross["polygon"][0], (120, 20))


def test_binder_receives_retained_confirmed_track_after_current_detection_is_lost():
    manager = FakeManager()
    manager.bindings[("cam1", 4)] = 30
    stale = track_at()
    stale.consecutive_invisible_count = 4
    stale.last_seen_frame = 96
    fresh = track_at(50, 40)
    fresh.consecutive_invisible_count = 0
    fresh.last_seen_frame = 100
    manager.bindings[("cam1", 5)] = 30
    tracker = SimpleNamespace(confirmed_tracks={4: stale, 5: fresh})

    selected = collect_binder_global_tracks("cam1", tracker, manager)

    assert selected == {30: fresh}


def test_observed_identity_cancels_mature_same_camera_and_cross_camera_tokens():
    manager = FakeManager()
    same_token = token()
    same_token.update({"confirmed_empty": True, "age_ms": 1300})
    cross_token = token("E08", 31)
    cross_token.update({"confirmed_empty": False, "age_ms": 100})
    same = FakeBinder(same_token)
    cross = FakeBinder(cross_token)

    cancel_observed_recovery_tokens(
        {"cam1": {7: 30}, "cam2": {8: 31}},
        {"cam1": same, "cam2": cross},
        manager,
        {"cam1": 10.0, "cam2": 10.0},
    )

    assert same.cancelled == [(30, "identity_still_tracked_after_empty_grace")]
    # Token owner is cam2 and GID31 is only visible there, so provisional
    # evidence remains available for false-empty restoration.
    assert cross.cancelled == []

    cross.token = cross_token
    cancel_observed_recovery_tokens(
        {"cam1": {9: 31}, "cam2": {}},
        {"cam1": same, "cam2": cross},
        manager,
        {"cam1": 10.1, "cam2": 10.1},
    )
    assert cross.cancelled == [(31, "identity_observed_cross_camera")]
