import csv
from pathlib import Path

import numpy as np
import pytest

from two_camera import ReplaySession


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.read_calls = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.read_calls += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def make_session(tmp_path: Path, rows: list[dict]) -> Path:
    session = tmp_path / "source_session"
    session.mkdir()
    for camera_id in ("cam1", "cam2"):
        (session / f"raw_{camera_id}.mp4").write_bytes(b"fake")
    fieldnames = [
        "frame_idx",
        "capture_unix_ns",
        "wall_time_iso",
        "cam1_monotonic_ns",
        "cam2_monotonic_ns",
        "camera_skew_ms",
    ]
    with (session / "frame_timestamps.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return session


def timing_row(frame_idx: int, cam1_ns: int, cam2_ns: int) -> dict:
    return {
        "frame_idx": frame_idx,
        "capture_unix_ns": 1_000_000_000 + frame_idx,
        "wall_time_iso": f"2026-08-18T19:45:{frame_idx:02d}+07:00",
        "cam1_monotonic_ns": cam1_ns,
        "cam2_monotonic_ns": cam2_ns,
        "camera_skew_ms": abs(cam1_ns - cam2_ns) / 1_000_000,
    }


def test_replay_reads_atomic_pairs_with_recorded_timestamps_and_clean_eof(tmp_path):
    session_path = make_session(tmp_path, [
        timing_row(1, 1_000_000_000, 1_010_000_000),
        timing_row(2, 1_080_000_000, 1_090_000_000),
    ])
    fake_captures = {
        "cam1": FakeCapture([np.full((2, 2), 11), np.full((2, 2), 12)]),
        "cam2": FakeCapture([np.full((2, 2), 21), np.full((2, 2), 22)]),
    }

    def capture_factory(source):
        camera_id = Path(source).stem.removeprefix("raw_")
        return fake_captures[camera_id]

    replay = ReplaySession(session_path, capture_factory=capture_factory)
    first = replay.read_pair()
    second = replay.read_pair()

    assert int(first[0]["cam1"][0, 0]) == 11
    assert int(first[0]["cam2"][0, 0]) == 21
    assert first[1] == {"cam1": 1, "cam2": 1}
    assert first[2] == {"cam1": 1_000_000_000, "cam2": 1_010_000_000}
    assert first[3].capture_unix_ns == 1_000_000_001
    assert second[1] == {"cam1": 2, "cam2": 2}
    assert second[2] == {"cam1": 1_080_000_000, "cam2": 1_090_000_000}

    assert replay.read_pair() is None
    assert replay.read_pair() is None
    assert fake_captures["cam1"].read_calls == 3
    assert fake_captures["cam2"].read_calls == 3
    replay.release()
    assert all(capture.released for capture in fake_captures.values())


def test_replay_rejects_video_that_ends_before_timestamp_rows(tmp_path):
    session_path = make_session(tmp_path, [
        timing_row(1, 1_000_000_000, 1_010_000_000),
        timing_row(2, 1_080_000_000, 1_090_000_000),
    ])
    fake_captures = {
        "cam1": FakeCapture([np.zeros((1, 1))]),
        "cam2": FakeCapture([np.zeros((1, 1)), np.zeros((1, 1))]),
    }

    def capture_factory(source):
        return fake_captures[Path(source).stem.removeprefix("raw_")]

    replay = ReplaySession(session_path, capture_factory=capture_factory)
    replay.read_pair()
    with pytest.raises(RuntimeError, match="cam1.*ket thuc som.*frame 2"):
        replay.read_pair()
    replay.release()


def test_replay_rejects_extra_video_frames_without_timestamp(tmp_path):
    session_path = make_session(
        tmp_path, [timing_row(1, 1_000_000_000, 1_010_000_000)]
    )
    fake_captures = {
        "cam1": FakeCapture([np.zeros((1, 1)), np.ones((1, 1))]),
        "cam2": FakeCapture([np.zeros((1, 1))]),
    }

    def capture_factory(source):
        return fake_captures[Path(source).stem.removeprefix("raw_")]

    replay = ReplaySession(session_path, capture_factory=capture_factory)
    replay.read_pair()
    with pytest.raises(RuntimeError, match="nhieu frame.*cam1"):
        replay.read_pair()
    replay.release()


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [
                timing_row(1, 1_000_000_000, 1_010_000_000),
                timing_row(3, 1_080_000_000, 1_090_000_000),
            ],
            "phai lien tuc",
        ),
        (
            [
                timing_row(1, 1_000_000_000, 1_010_000_000),
                timing_row(2, 999_999_999, 1_090_000_000),
            ],
            "cam1.*di lui",
        ),
    ],
)
def test_replay_rejects_invalid_timeline(tmp_path, rows, message):
    session_path = make_session(tmp_path, rows)
    with pytest.raises(ValueError, match=message):
        ReplaySession(session_path, capture_factory=lambda _source: None)


def test_replay_accepts_repeated_camera_timestamp_from_latest_frame_capture(tmp_path):
    session_path = make_session(tmp_path, [
        timing_row(1, 1_000_000_000, 1_010_000_000),
        timing_row(2, 1_080_000_000, 1_010_000_000),
    ])
    captures = {
        "cam1": FakeCapture([np.zeros((1, 1)), np.zeros((1, 1))]),
        "cam2": FakeCapture([np.zeros((1, 1)), np.zeros((1, 1))]),
    }

    replay = ReplaySession(
        session_path,
        capture_factory=lambda source: captures[
            Path(source).stem.removeprefix("raw_")
        ],
    )

    assert replay.read_pair()[2]["cam2"] == 1_010_000_000
    assert replay.read_pair()[2]["cam2"] == 1_010_000_000
    replay.release()
