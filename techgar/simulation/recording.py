"""Recording assembly: lazy frame streams plus precomputed ground truth.

Frames are rendered on demand (a 20 s two-camera recording would otherwise be
hundreds of megabytes) but deterministically: the same recording iterated twice
yields byte-identical frames, because every frame seeds its own noise from
``(camera_id, sequence)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..contracts import FrameRecord
from ..geometry import polygon_coverage
from .annotate import (V_PARKED_TRUE, Annotator, GroundTruthRecord, GT_SLOT_COVERAGE, HandoffTruth,
                       SlotTruth)
from .render import Renderer, RenderOptions


@dataclass
class RecordingOptions:
    fps: float = 12.0
    tail: float = 0.6
    skew: dict[str, float] = field(default_factory=lambda: {"C1": 0.0, "C2": 0.035})
    jitter: float = 0.003
    stalls: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    flicker: list[tuple[float, float, float]] = field(default_factory=list)
    render: RenderOptions = field(default_factory=RenderOptions)
    seed: int = 17
    slot_transition_tolerance: float = 1.2
    duration: float | None = None


@dataclass
class FrameSpec:
    camera_id: str
    sequence: int
    timestamp: float
    brightness_gain: float = 1.0


@dataclass
class Recording:
    name: str
    layout: object
    vehicles: list
    options: RecordingOptions
    specs: list[FrameSpec]
    ground_truth: list[GroundTruthRecord]
    slot_truth: list[SlotTruth]
    handoffs: list[HandoffTruth]
    timeline: np.ndarray

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return self.layout.camera_ids

    def iter_frames(self) -> Iterator[FrameRecord]:
        renderers = {cam_id: Renderer(cam, self.layout.slots, seed=self.options.seed + i)
                     for i, (cam_id, cam) in enumerate(sorted(self.layout.cameras.items()))}
        for spec in self.specs:
            renderer = renderers[spec.camera_id]
            camera_seed = sum((index + 1) * ord(char)
                              for index, char in enumerate(spec.camera_id))
            renderer.rng = np.random.default_rng(
                (self.options.seed * 1_000_003 + spec.sequence * 97 + camera_seed) % (2 ** 32))
            opts = RenderOptions(**{**self.options.render.__dict__,
                                    "brightness_gain": spec.brightness_gain})
            image = renderer.render(self.vehicles, spec.timestamp, opts,
                                    self.layout.blind_regions.get(spec.camera_id, []))
            camera = self.layout.cameras[spec.camera_id]
            yield FrameRecord(spec.camera_id, spec.sequence, spec.timestamp,
                              camera.width, camera.height, True, image)

    def gt_for(self, vehicle_id: str) -> list[GroundTruthRecord]:
        return [g for g in self.ground_truth if g.physical_vehicle_id == vehicle_id]


def _stalled(t: float, windows) -> bool:
    return any(a <= t <= b for a, b in windows)


def _brightness(t: float, flicker) -> float:
    for t0, t1, gain in flicker:
        if t0 <= t <= t1:
            return gain
    return 1.0


def _slot_truth(layout, vehicles, timeline, tolerance: float) -> list[SlotTruth]:
    rows: list[SlotTruth] = []
    for t in timeline:
        for slot_id, poly in layout.slots.items():
            owner, best = None, 0.0
            for vehicle in vehicles:
                if not vehicle.present(t):
                    continue
                cov = polygon_coverage(vehicle.footprint(t), poly)
                speed = float(np.linalg.norm(vehicle.velocity(t)))
                if cov >= GT_SLOT_COVERAGE and speed <= V_PARKED_TRUE and cov > best:
                    owner, best = vehicle.vehicle_id, cov
            rows.append(SlotTruth(float(t), slot_id, owner, best))
    by_slot: dict[str, list[SlotTruth]] = {}
    for row in rows:
        by_slot.setdefault(row.slot_id, []).append(row)
    for series in by_slot.values():
        changes = [series[i].timestamp for i in range(1, len(series))
                   if series[i].physical_vehicle_id != series[i - 1].physical_vehicle_id]
        for row in series:
            row.changed = any(abs(row.timestamp - c) <= tolerance for c in changes)
    return rows


def _handoff_truth(ground_truth, camera_ids) -> list[HandoffTruth]:
    events: list[HandoffTruth] = []
    per_vehicle: dict[str, list[GroundTruthRecord]] = {}
    for rec in ground_truth:
        if rec.visible_fraction >= 0.5:
            per_vehicle.setdefault(rec.physical_vehicle_id, []).append(rec)
    for vehicle_id, records in per_vehicle.items():
        records.sort(key=lambda r: r.timestamp)
        current = records[0].camera_id
        last_seen = records[0].timestamp
        for rec in records[1:]:
            if rec.camera_id != current:
                events.append(HandoffTruth(vehicle_id, current, rec.camera_id, last_seen,
                                           rec.timestamp))
                current = rec.camera_id
            if rec.camera_id == current:
                last_seen = rec.timestamp
    return events


def build_recording(name: str, layout, vehicles, options: RecordingOptions | None = None
                    ) -> Recording:
    options = options or RecordingOptions()
    duration = options.duration
    if duration is None:
        duration = max(v.t_end for v in vehicles) + options.tail
    rng = np.random.default_rng(options.seed)
    specs: list[FrameSpec] = []
    period = 1.0 / options.fps
    for cam_id in layout.camera_ids:
        skew = options.skew.get(cam_id, 0.0)
        stalls = options.stalls.get(cam_id, [])
        seq = 0
        k = 0
        while True:
            t = k * period + skew + float(rng.normal(0.0, options.jitter))
            k += 1
            if t > duration:
                break
            if t < 0 or _stalled(t, stalls):
                continue
            specs.append(FrameSpec(cam_id, seq, round(t, 6), _brightness(t, options.flicker)))
            seq += 1
    specs.sort(key=lambda s: (s.timestamp, s.camera_id))
    annotator = Annotator(layout, list(vehicles))
    ground_truth = [annotator.record(spec.sequence, layout.cameras[spec.camera_id], vehicle,
                                    spec.timestamp)
                    for spec in specs for vehicle in vehicles if vehicle.present(spec.timestamp)]
    timeline = np.arange(0.0, duration, period)
    slot_truth = _slot_truth(layout, vehicles, timeline, options.slot_transition_tolerance)
    handoffs = _handoff_truth(ground_truth, layout.camera_ids)
    return Recording(name, layout, list(vehicles), options, specs, ground_truth, slot_truth,
                     handoffs, timeline)
