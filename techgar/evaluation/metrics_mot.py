"""MOT metrics — PLAN 3 §3: IDSW, fragmentation, IDF1, MOTA, stratified P/R."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..assignment import solve_assignment
from .matching import FrameMatch


@dataclass
class MotMetrics:
    frames: int = 0
    gt_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    id_switches: int = 0
    fragmentation: dict[str, int] = field(default_factory=dict)
    idf1: float = 0.0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0
    mota: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    strata: dict[str, dict[str, float]] = field(default_factory=dict)
    switch_details: list[tuple[float, str, int, int]] = field(default_factory=list)
    position_error_mean: float = 0.0
    position_error_p95: float = 0.0

    @property
    def max_fragmentation(self) -> int:
        return max(self.fragmentation.values(), default=0)

    def as_dict(self) -> dict:
        return {"frames": self.frames, "gt": self.gt_count, "tp": self.true_positives,
                "fp": self.false_positives, "fn": self.false_negatives,
                "idsw": self.id_switches, "max_fragmentation": self.max_fragmentation,
                "fragmentation": dict(self.fragmentation), "idf1": self.idf1,
                "mota": self.mota, "precision": self.precision, "recall": self.recall,
                "position_error_mean": self.position_error_mean,
                "position_error_p95": self.position_error_p95, "strata": self.strata}


def _idf1(frames: list[FrameMatch]) -> tuple[float, int, int, int]:
    """Optimal one-to-one identity mapping, then IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)."""
    overlap: dict[tuple[str, int], int] = {}
    gt_frames: dict[str, int] = {}
    pred_frames: dict[int, int] = {}
    for frame in frames:
        for vehicle_id, global_id in frame.matches.items():
            overlap[(vehicle_id, global_id)] = overlap.get((vehicle_id, global_id), 0) + 1
        for vehicle_id in list(frame.matches) + frame.false_negatives:
            gt_frames[vehicle_id] = gt_frames.get(vehicle_id, 0) + 1
        for global_id in list(frame.matches.values()) + frame.false_positives:
            pred_frames[global_id] = pred_frames.get(global_id, 0) + 1
    total_gt = sum(gt_frames.values())
    total_pred = sum(pred_frames.values())
    if not overlap:
        return 0.0, 0, total_pred, total_gt
    vehicles = sorted({key[0] for key in overlap})
    globals_ = sorted({key[1] for key in overlap})
    cost = np.zeros((len(vehicles), len(globals_)))
    for (vehicle_id, global_id), count in overlap.items():
        cost[vehicles.index(vehicle_id), globals_.index(global_id)] = -count
    idtp = -int(sum(cost[i, j] for i, j in solve_assignment(cost)))
    idfp = max(0, total_pred - idtp)
    idfn = max(0, total_gt - idtp)
    denominator = 2 * idtp + idfp + idfn
    return (2.0 * idtp / denominator if denominator else 0.0), idtp, idfp, idfn


def mot_metrics(frames: list[FrameMatch]) -> MotMetrics:
    metrics = MotMetrics(frames=len(frames))
    last_seen: dict[str, int] = {}
    assigned: dict[str, set[int]] = {}
    errors: list[float] = []
    strata_counts: dict[str, dict[str, int]] = {}
    for frame in frames:
        metrics.true_positives += frame.true_positives
        metrics.false_positives += len(frame.false_positives)
        metrics.false_negatives += len(frame.false_negatives)
        metrics.gt_count += frame.true_positives + len(frame.false_negatives)
        errors.extend(frame.distances.values())
        for vehicle_id, global_id in frame.matches.items():
            assigned.setdefault(vehicle_id, set()).add(global_id)
            previous = last_seen.get(vehicle_id)
            if previous is not None and previous != global_id:
                metrics.id_switches += 1
                metrics.switch_details.append((frame.timestamp, vehicle_id, previous, global_id))
            last_seen[vehicle_id] = global_id
        for vehicle_id in frame.false_negatives:
            phase = frame.truth_phase.get(vehicle_id, "unknown")
            bucket = strata_counts.setdefault(phase, {"tp": 0, "fn": 0})
            bucket["fn"] += 1
        for vehicle_id in frame.matches:
            phase = frame.truth_phase.get(vehicle_id, "unknown")
            bucket = strata_counts.setdefault(phase, {"tp": 0, "fn": 0})
            bucket["tp"] += 1
    metrics.fragmentation = {vehicle_id: len(ids) for vehicle_id, ids in assigned.items()}
    metrics.idf1, metrics.idtp, metrics.idfp, metrics.idfn = _idf1(frames)
    denominator = max(metrics.gt_count, 1)
    metrics.mota = 1.0 - (metrics.false_negatives + metrics.false_positives
                          + metrics.id_switches) / denominator
    detected = metrics.true_positives + metrics.false_positives
    metrics.precision = metrics.true_positives / detected if detected else 0.0
    metrics.recall = metrics.true_positives / denominator
    if errors:
        metrics.position_error_mean = float(np.mean(errors))
        metrics.position_error_p95 = float(np.percentile(errors, 95))
    metrics.strata = {
        phase: {"tp": counts["tp"], "fn": counts["fn"],
                "recall": counts["tp"] / max(counts["tp"] + counts["fn"], 1)}
        for phase, counts in sorted(strata_counts.items())}
    return metrics
