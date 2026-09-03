"""Phase 6 — performance instrumentation and overload policy.

Stage timing, bounded queues, latency percentiles, a subscriber gate for video
encoding, and the one rule that overrides all of them: under overload the system
may drop frames, raise uncertainty and report more temporarily-missing vehicles,
but it may **never** mint a Global ID (PLAN 1 Phase 6 / PLAN 3 §6).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config_world import PerfConfig
from .linalg import percentiles


@dataclass
class StageTimer:
    """Per-stage wall-clock accumulator."""

    samples: dict[str, list[float]] = field(default_factory=dict)
    _open: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._open[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        started = self._open.pop(stage, None)
        if started is None:
            return 0.0
        elapsed = time.perf_counter() - started
        self.samples.setdefault(stage, []).append(elapsed)
        return elapsed

    def measure(self, stage: str):
        timer = self

        class _Scope:
            def __enter__(self):
                timer.start(stage)
                return timer

            def __exit__(self, *exc):
                timer.stop(stage)
                return False

        return _Scope()

    def total(self, stage: str) -> float:
        return float(sum(self.samples.get(stage, ())))

    def mean(self, stage: str) -> float:
        values = self.samples.get(stage, ())
        return float(np.mean(values)) if values else 0.0

    def report(self) -> dict[str, dict[str, float]]:
        out = {}
        for stage, values in self.samples.items():
            stats = percentiles(values, (50, 95, 100))
            out[stage] = {"count": len(values), "mean": float(np.mean(values)),
                          "median": stats[50], "p95": stats[95], "max": stats[100],
                          "total": float(np.sum(values))}
        return out

    def slowest(self) -> tuple[str, float]:
        report = self.report()
        if not report:
            return "", 0.0
        stage = max(report, key=lambda key: report[key]["total"])
        return stage, report[stage]["total"]


class BoundedQueue:
    """Fixed-capacity queue that drops the *oldest* item — never unbounded backlog."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: deque = deque()
        self.dropped = 0

    def push(self, item) -> bool:
        if len(self._items) >= self.capacity:
            self._items.popleft()
            self.dropped += 1
            self._items.append(item)
            return False
        self._items.append(item)
        return True

    def pop(self):
        return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    @property
    def saturated(self) -> bool:
        return len(self._items) >= self.capacity


@dataclass
class LatencyTracker:
    """End-to-end latency, PLAN 3 §6:  L = t_render - t_capture."""

    samples: list[float] = field(default_factory=list)
    processing: list[float] = field(default_factory=list)

    def record(self, capture_timestamp: float, published_timestamp: float,
               processing_seconds: float) -> float:
        latency = max(0.0, published_timestamp - capture_timestamp)
        self.samples.append(latency)
        self.processing.append(processing_seconds)
        return latency

    def report(self) -> dict:
        stats = percentiles(self.samples, (50, 95, 100))
        work = percentiles(self.processing, (50, 95, 100))
        return {"count": len(self.samples), "median": stats[50], "p95": stats[95],
                "max": stats[100], "processing_median": work[50], "processing_p95": work[95],
                "processing_max": work[100]}


@dataclass
class OverloadMonitor:
    """Declares overload from measured stage time, and remembers that it did."""

    config: PerfConfig = field(default_factory=PerfConfig)
    active: bool = False
    episodes: int = 0
    frames_skipped: int = 0
    uncertainty_gain: float = 1.0

    def observe(self, processing_seconds: float, queue_saturated: bool = False) -> bool:
        over = processing_seconds > self.config.overload_stage_budget or queue_saturated
        if over and not self.active:
            self.episodes += 1
        self.active = bool(over)
        self.uncertainty_gain = (self.config.overload_uncertainty_gain if self.active else 1.0)
        return self.active

    def force(self, active: bool = True) -> None:
        """Test / stress hook: assert overload without waiting for real slowness."""
        if active and not self.active:
            self.episodes += 1
        self.active = bool(active)
        self.uncertainty_gain = (self.config.overload_uncertainty_gain if self.active else 1.0)

    def note_skip(self) -> None:
        self.frames_skipped += 1


@dataclass
class SubscriberGate:
    """Visualisation encoding runs only when somebody is watching (Phase 6.7)."""

    subscribers: int = 0
    encodes: int = 0
    skipped: int = 0

    def subscribe(self) -> int:
        self.subscribers += 1
        return self.subscribers

    def unsubscribe(self) -> int:
        self.subscribers = max(0, self.subscribers - 1)
        return self.subscribers

    def should_encode(self) -> bool:
        if self.subscribers > 0:
            self.encodes += 1
            return True
        self.skipped += 1
        return False


@dataclass
class ThroughputMeter:
    pairs: int = 0
    first: float | None = None
    last: float | None = None
    instantaneous: list[float] = field(default_factory=list)

    def tick(self, timestamp: float) -> None:
        if self.first is None:
            self.first = timestamp
        elif self.last is not None and timestamp > self.last:
            self.instantaneous.append(1.0 / (timestamp - self.last))
        self.last = timestamp
        self.pairs += 1

    @property
    def fps(self) -> float:
        if self.first is None or self.last is None or self.last <= self.first:
            return 0.0
        # N samples contain N-1 measured intervals.  Dividing by N would
        # systematically overstate throughput on short commissioning runs.
        return float(max(0, self.pairs - 1) / (self.last - self.first))

    def report(self) -> dict:
        window = 5
        sustained = []
        for index in range(len(self.instantaneous) - window + 1):
            sustained.append(float(np.mean(self.instantaneous[index:index + window])))
        return {"pairs": self.pairs, "mean_fps": self.fps,
                "min_sustained_fps": float(min(sustained)) if sustained else self.fps,
                "max_fps": float(max(self.instantaneous)) if self.instantaneous else 0.0}
