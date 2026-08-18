import json
from pathlib import Path

import numpy as np

from techgar.motion_tracker import MotionVehicleTracker
from techgar.vehicle_tracker import TrackStatus, TrackedVehicle
from two_camera import save_json, select_moving_tracks


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
