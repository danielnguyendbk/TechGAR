import json
from pathlib import Path

import numpy as np

from techgar.motion_tracker import MotionVehicleTracker
from techgar.latest_frame_capture import LatestFrameCapture
from techgar.vehicle_tracker import TrackStatus, TrackedVehicle
from two_camera import (
    load_calibration,
    save_json,
    select_moving_tracks,
    synchronize_live_frames,
)


def make_track(track_id: int = 1) -> TrackedVehicle:
    return TrackedVehicle(
        track_id=track_id,
        cx=30,
        cy=50,
        bbox=(20, 30, 20, 20),
        area=400.0,
        status=TrackStatus.CONFIRMED,
    )


def test_select_moving_tracks_excludes_canonical_parked_ids():
    tracks = {1: make_track(1), 2: make_track(2), 3: make_track(3)}
    local_to_global = {1: 7, 2: 8, 3: 9}

    moving, shown_ids = select_moving_tracks(
        tracks,
        local_to_global,
        parked_global_ids={7},
        canonicalize=lambda global_id: 7 if global_id == 8 else global_id,
    )

    assert set(moving) == {3}
    assert shown_ids == {3: 9}


def test_draw_tracks_uses_blue_for_two_camera_moving_overlay():
    tracker = MotionVehicleTracker()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    track = make_track(4)

    output = tracker.draw_tracks(
        frame,
        {4: track},
        id_overrides={4: 17},
        confirmed_color=(255, 0, 0),
        confirmed_label="moving",
        point_color=(255, 0, 0),
    )

    assert tuple(output[30, 20]) == (255, 0, 0)
    assert tuple(output[50, 30]) == (255, 0, 0)


def test_draw_tracks_does_not_restore_internal_tracks_for_empty_selection():
    tracker = MotionVehicleTracker()
    tracker._tracks = {1: make_track(1)}
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    output = tracker.draw_tracks(frame, {})

    assert np.array_equal(output, frame)


def test_save_json_writes_complete_payload(tmp_path):
    output = tmp_path / "status.json"

    assert save_json(output, {"frame_index": 12})
    assert json.loads(output.read_text(encoding="utf-8")) == {"frame_index": 12}


def test_save_json_can_skip_a_locked_runtime_update(tmp_path, monkeypatch):
    output = tmp_path / "status.json"
    output.write_text("{}", encoding="utf-8")

    def locked_replace(_self, _target):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", locked_replace)

    assert not save_json(output, {"frame_index": 13}, tolerate_lock=True)
    assert not list(tmp_path.glob("*.tmp"))


def test_load_calibration_reads_optional_exit_zones(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "camera_transforms": {
            "cam1": np.eye(3).tolist(),
            "cam2": np.eye(3).tolist(),
        },
        "edge_adjacency": [
            {"source_camera": "cam1", "exit_edge": "right", "target_camera": "cam2"},
            {"source_camera": "cam2", "exit_edge": "left", "target_camera": "cam1"},
        ],
        "overlap_world_polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
        "exit_zones": [{
            "camera": "cam2",
            "polygon": [[500, 100], [600, 100], [600, 200], [500, 200]],
        }],
    }), encoding="utf-8")

    _transforms, _adjacency, _overlap, exit_zones = load_calibration(calibration)

    assert list(exit_zones) == ["cam2"]
    assert exit_zones["cam2"][0].shape == (4, 2)


def test_load_calibration_accepts_multi_vertex_world_overlap(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "world": {"unit": "cm"},
        "camera_transforms": {
            "cam1": np.eye(3).tolist(),
            "cam2": np.eye(3).tolist(),
        },
        "edge_adjacency": [
            {"source_camera": "cam1", "exit_edge": "right", "target_camera": "cam2"},
            {"source_camera": "cam2", "exit_edge": "left", "target_camera": "cam1"},
        ],
        "overlap_world_polygon": [
            [10, 0], [20, 0], [25, 5], [20, 10], [10, 10], [5, 5]
        ],
    }), encoding="utf-8")

    _transforms, _adjacency, overlap, _exit_zones = load_calibration(calibration)

    assert overlap[("cam1", "cam2")].shape == (6, 2)


def test_synchronize_live_frames_advances_the_older_camera():
    older = LatestFrameCapture.__new__(LatestFrameCapture)
    newer = LatestFrameCapture.__new__(LatestFrameCapture)
    replacement = np.ones((2, 2, 3), dtype=np.uint8)
    older.read_latest_timed = lambda after_sequence, timeout: (
        replacement, after_sequence + 1, 1_250_000_000
    )
    captures = {"cam1": older, "cam2": newer}
    frames = {
        "cam1": np.zeros((2, 2, 3), dtype=np.uint8),
        "cam2": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    sequences = {"cam1": 3, "cam2": 4}
    timestamps = {"cam1": 1_000_000_000, "cam2": 1_300_000_000}

    synchronize_live_frames(
        captures, frames, sequences, timestamps,
        max_skew_ms=100.0, max_catchup_reads=2,
    )

    assert sequences["cam1"] == 4
    assert timestamps["cam1"] == 1_250_000_000
    assert np.array_equal(frames["cam1"], replacement)
