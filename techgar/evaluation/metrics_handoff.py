"""Cross-camera handoff metrics for PLAN 3 §5."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .matching import FrameMatch, Prediction


@dataclass
class HandoffMetrics:
    expected: int = 0
    matched: int = 0
    identity_correct: int = 0
    invalid: int = 0
    accuracy: float = 0.0
    identity_accuracy: float = 0.0
    invalid_rate: float = 0.0
    latency_mean: float = 0.0
    latency_p95: float = 0.0
    latencies: list[float] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "expected": self.expected,
            "matched": self.matched,
            "identity_correct": self.identity_correct,
            "invalid": self.invalid,
            "accuracy": self.accuracy,
            "identity_accuracy": self.identity_accuracy,
            "invalid_rate": self.invalid_rate,
            "latency_mean": self.latency_mean,
            "latency_p95": self.latency_p95,
            "latencies": list(self.latencies),
            "details": list(self.details),
        }


def handoff_metrics(oracle, frames: list[FrameMatch],
                    prediction_frames: list[tuple[float, list[Prediction]]],
                    tolerance: float = 1.0) -> HandoffMetrics:
    """Compare predicted camera transitions to independently annotated handoffs."""
    metrics = HandoffMetrics(expected=len(oracle.handoffs()))
    by_time = {round(timestamp, 6): predictions
               for timestamp, predictions in prediction_frames}
    observations: dict[str, list[tuple[float, int, str]]] = {}
    for frame in frames:
        predictions = by_time.get(round(frame.timestamp, 6), [])
        by_gid = {prediction.global_id: prediction for prediction in predictions}
        for vehicle_id, global_id in frame.matches.items():
            prediction = by_gid.get(global_id)
            if prediction is not None and prediction.camera_id:
                observations.setdefault(vehicle_id, []).append(
                    (frame.timestamp, global_id, prediction.camera_id)
                )

    expected_keys = {
        (truth.physical_vehicle_id, truth.source_camera, truth.target_camera)
        for truth in oracle.handoffs()
    }
    for truth in oracle.handoffs():
        series = sorted(observations.get(truth.physical_vehicle_id, []))
        sources = [row for row in series
                   if row[2] == truth.source_camera
                   and row[0] <= truth.t_first_target + tolerance]
        targets = [row for row in series
                   if row[2] == truth.target_camera
                   and row[0] >= truth.t_last_source - tolerance]
        detail = {
            "vehicle": truth.physical_vehicle_id,
            "source": truth.source_camera,
            "target": truth.target_camera,
            "matched": False,
            "identity_correct": False,
        }
        if sources and targets:
            source = max(sources, key=lambda row: row[0])
            target = min(targets, key=lambda row: row[0])
            if target[0] >= source[0]:
                metrics.matched += 1
                detail["matched"] = True
                detail["source_gid"] = source[1]
                detail["target_gid"] = target[1]
                if source[1] == target[1]:
                    metrics.identity_correct += 1
                    detail["identity_correct"] = True
                latency = max(0.0, target[0] - truth.t_first_target)
                metrics.latencies.append(latency)
                detail["latency"] = latency
        metrics.details.append(detail)

    # Count observed camera transitions not present in ground truth.  Consecutive
    # duplicate camera reports are collapsed before transition extraction.
    for vehicle_id, series in observations.items():
        previous = None
        for timestamp, _, camera in sorted(series):
            if previous is not None and camera != previous:
                if (vehicle_id, previous, camera) not in expected_keys:
                    metrics.invalid += 1
            previous = camera

    metrics.accuracy = metrics.matched / max(metrics.expected, 1)
    metrics.identity_accuracy = metrics.identity_correct / max(metrics.expected, 1)
    transitions = metrics.matched + metrics.invalid
    metrics.invalid_rate = metrics.invalid / max(transitions, 1)
    if metrics.latencies:
        metrics.latency_mean = float(np.mean(metrics.latencies))
        metrics.latency_p95 = float(np.percentile(metrics.latencies, 95))
    return metrics

