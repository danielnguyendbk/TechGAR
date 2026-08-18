import time

import numpy as np
import pytest

from techgar.latest_frame_capture import LatestFrameCapture


class FakeCapture:
    def __init__(self, _source, fail=False):
        self.fail = fail
        self.released = False
        self.reading = False
        self.released_during_read = False
        self.counter = 0

    def isOpened(self):
        return True

    def set(self, _property_id, _value):
        return True

    def get(self, _property_id):
        return 25.0

    def read(self):
        self.reading = True
        try:
            time.sleep(0.003)
            if self.released or self.fail:
                return False, None
            self.counter += 1
            return True, np.full((2, 2, 3), self.counter % 255, dtype=np.uint8)
        finally:
            self.reading = False

    def release(self):
        self.released_during_read = self.reading
        self.released = True


def test_latest_frame_capture_drops_frames_while_consumer_is_busy():
    fake = FakeCapture("stream")
    capture = LatestFrameCapture("stream", capture_factory=lambda _source: fake).start()
    try:
        first_frame, first_sequence = capture.read_latest(timeout=0.5)
        time.sleep(0.04)
        latest_frame, latest_sequence = capture.read_latest(
            after_sequence=first_sequence, timeout=0.5
        )

        assert latest_sequence >= first_sequence + 3
        assert int(latest_frame[0, 0, 0]) == latest_sequence % 255
        assert not np.array_equal(first_frame, latest_frame)
    finally:
        capture.release()
    assert not fake.released_during_read


def test_latest_frame_capture_times_out_when_stream_has_no_frames():
    fake = FakeCapture("stream", fail=True)
    capture = LatestFrameCapture("stream", capture_factory=lambda _source: fake).start()
    try:
        with pytest.raises(TimeoutError, match="Khong co frame moi"):
            capture.read_latest(timeout=0.05)
    finally:
        capture.release()


def test_latest_frame_capture_exposes_monotonic_decode_timestamp():
    fake = FakeCapture("stream")
    capture = LatestFrameCapture("stream", capture_factory=lambda _source: fake).start()
    try:
        _frame, first_sequence, first_timestamp = capture.read_latest_timed(timeout=0.5)
        _frame, second_sequence, second_timestamp = capture.read_latest_timed(
            after_sequence=first_sequence, timeout=0.5
        )

        assert second_sequence > first_sequence
        assert first_timestamp > 0
        assert second_timestamp >= first_timestamp
    finally:
        capture.release()
