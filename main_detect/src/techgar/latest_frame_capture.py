"""Background OpenCV capture that drops stale frames."""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2


class LatestFrameCapture:
    """Continuously drain a stream and expose only its newest decoded frame."""

    def __init__(
        self,
        source,
        capture_factory: Callable = cv2.VideoCapture,
    ) -> None:
        self.source = source
        self._capture = capture_factory(source)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Khong mo duoc stream: {source}")

        self._condition = threading.Condition()
        self._latest_frame = None
        self._latest_timestamp_ns = 0
        self._sequence = 0
        self._stopped = False
        self._consecutive_failures = 0
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"latest-frame-{id(self)}",
            daemon=True,
        )

    def start(self) -> "LatestFrameCapture":
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return
            try:
                ok, frame = self._capture.read()
            except cv2.error:
                with self._condition:
                    if self._stopped:
                        return
                    self._consecutive_failures += 1
                    self._condition.notify_all()
                time.sleep(0.02)
                continue
            with self._condition:
                if self._stopped:
                    return
                if ok and frame is not None:
                    self._latest_frame = frame
                    self._latest_timestamp_ns = time.monotonic_ns()
                    self._sequence += 1
                    self._consecutive_failures = 0
                    self._condition.notify_all()
                    continue
                self._consecutive_failures += 1
                self._condition.notify_all()
            time.sleep(0.02)

    def _read_latest_item(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ):
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                not self._stopped
                and (self._latest_frame is None or self._sequence <= after_sequence)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Khong co frame moi tu stream sau {timeout:.1f}s: {self.source}"
                    )
                self._condition.wait(remaining)
            if self._latest_frame is None:
                raise RuntimeError(f"Stream da dung truoc khi co frame: {self.source}")
            return self._latest_frame, self._sequence, self._latest_timestamp_ns

    def read_latest(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ):
        """Wait for a newer frame and return the backward-compatible pair."""
        frame, sequence, _timestamp_ns = self._read_latest_item(
            after_sequence=after_sequence, timeout=timeout
        )
        return frame, sequence

    def read_latest_timed(
        self,
        after_sequence: int = -1,
        timeout: float = 5.0,
    ):
        """Return a frame, stream sequence and monotonic decode timestamp."""
        return self._read_latest_item(
            after_sequence=after_sequence, timeout=timeout
        )

    def get(self, property_id: int) -> float:
        return float(self._capture.get(property_id))

    @property
    def skipped_decode_failures(self) -> int:
        with self._condition:
            return self._consecutive_failures

    def release(self) -> None:
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        # FFmpeg can abort the process if release() races with VideoCapture.read().
        # A daemon reader that is still blocked is safer to leave for process teardown.
        if not self._thread.is_alive():
            self._capture.release()
