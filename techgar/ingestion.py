"""Stage 1 — dual-stream ingestion with latest-frame buffering.

Rules from PLAN 1 stage 1:

* one *latest-frame* slot per camera — never a growing queue (logic 4-5);
* frames are paired by timestamp proximity, never by arrival order (logic 6-7);
* two frames 2.28 s apart are not a simultaneous observation and must never end
  up in the same pair (stage 1 Fail case).

A camera that stalls therefore cannot block the other: its partner is emitted on
its own once no partner can arrive inside the skew policy, and cross-camera
fusion simply does not run for that cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config_vision import IngestionConfig
from .contracts import FrameRecord, SynchronizedFramePair


@dataclass
class StreamHealth:
    camera_id: str
    frames_received: int = 0
    frames_dropped_stale: int = 0
    frames_replaced: int = 0
    decode_failures: int = 0
    last_timestamp: float | None = None
    last_sequence: int | None = None

    @property
    def alive(self) -> bool:
        return self.frames_received > 0


@dataclass
class IngestionStats:
    pairs: int = 0
    complete_pairs: int = 0
    single_emissions: int = 0
    skew_rejections: int = 0
    max_skew: float = 0.0


class DualStreamIngestion:
    def __init__(self, camera_ids, config: IngestionConfig | None = None) -> None:
        self.camera_ids = tuple(camera_ids)
        self.config = config or IngestionConfig()
        self.health = {cam: StreamHealth(cam) for cam in self.camera_ids}
        self.stats = IngestionStats()
        self._latest: dict[str, FrameRecord | None] = {cam: None for cam in self.camera_ids}
        self._consumed: dict[str, bool] = {cam: True for cam in self.camera_ids}
        self._newest_timestamp = float("-inf")
        self._pair_sequence = 0

    # --- input --------------------------------------------------------------
    def submit(self, frame: FrameRecord) -> bool:
        """Store the frame as the camera's latest.  Returns False if dropped."""
        if frame.camera_id not in self._latest:
            raise KeyError(f"unknown camera {frame.camera_id}")
        health = self.health[frame.camera_id]
        health.frames_received += 1
        if not frame.decode_ok:
            health.decode_failures += 1
            return False
        if (self.config.drop_stale if hasattr(self.config, "drop_stale") else True):
            age = self._newest_timestamp - frame.timestamp
            if age > self.config.stale_frame_age:
                health.frames_dropped_stale += 1
                return False
        previous = self._latest[frame.camera_id]
        if previous is not None and not self._consumed[frame.camera_id]:
            # The unprocessed frame is superseded: drop it rather than queue it.
            health.frames_replaced += 1
        self._latest[frame.camera_id] = frame
        self._consumed[frame.camera_id] = False
        health.last_timestamp = frame.timestamp
        health.last_sequence = frame.sequence
        self._newest_timestamp = max(self._newest_timestamp, frame.timestamp)
        return True

    # --- pairing ------------------------------------------------------------
    def _emit(self, group: dict[str, FrameRecord], reason: str = "") -> SynchronizedFramePair:
        for camera_id in group:
            self._consumed[camera_id] = True
        timestamps = [frame.timestamp for frame in group.values()]
        skew = float(max(timestamps) - min(timestamps))
        self._pair_sequence += 1
        self.stats.pairs += 1
        self.stats.max_skew = max(self.stats.max_skew, skew)
        if len(group) >= 2:
            self.stats.complete_pairs += 1
        else:
            self.stats.single_emissions += 1
        return SynchronizedFramePair(frames=dict(group), timestamp_skew=skew,
                                     pair_sequence=self._pair_sequence,
                                     accepted=skew <= self.config.max_pair_skew,
                                     reject_reason=reason)

    def try_pair(self) -> SynchronizedFramePair | None:
        pending = [(cam, frame) for cam, frame in self._latest.items()
                   if frame is not None and not self._consumed[cam]]
        if not pending:
            return None
        cfg = self.config
        if len(pending) == len(self.camera_ids):
            newest = max(frame.timestamp for _, frame in pending)
            group = {cam: frame for cam, frame in pending
                     if newest - frame.timestamp <= cfg.max_pair_skew}
            if len(group) == len(pending):
                return self._emit(group)
            # Skew outside policy: release the older frame on its own so that it is
            # processed, but never as a simultaneous observation of the newer one.
            self.stats.skew_rejections += 1
            camera_id, frame = min(pending, key=lambda item: item[1].timestamp)
            return self._emit({camera_id: frame},
                              reason=f"skew_{newest - frame.timestamp:.3f}s_exceeds_policy")
        camera_id, frame = min(pending, key=lambda item: item[1].timestamp)
        if self._newest_timestamp - frame.timestamp > cfg.max_pair_skew:
            return self._emit({camera_id: frame}, reason="no_partner_within_skew")
        return None

    def flush(self) -> list[SynchronizedFramePair]:
        """Emit whatever is still buffered (end of stream)."""
        pairs = []
        while True:
            pending = [(cam, frame) for cam, frame in self._latest.items()
                       if frame is not None and not self._consumed[cam]]
            if not pending:
                break
            newest = max(frame.timestamp for _, frame in pending)
            group = {cam: frame for cam, frame in pending
                     if newest - frame.timestamp <= self.config.max_pair_skew}
            pairs.append(self._emit(group))
        return pairs
