"""Geometric matching of predictions to physical vehicles.

PLAN 3's golden rule: "evaluator KHÔNG BAO GIỜ dùng Global ID dự đoán làm ground
truth".  Matching is therefore a one-to-one assignment on *world distance* alone;
the predicted Global ID is only read afterwards, to count identity statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..assignment import solve_assignment


@dataclass
class Prediction:
    timestamp: float
    global_id: int
    position: np.ndarray
    display_state: str = "observed"
    slot_id: str | None = None
    camera_id: str = ""


@dataclass
class FrameMatch:
    timestamp: float
    matches: dict[str, int] = field(default_factory=dict)          # vehicle -> global id
    distances: dict[str, float] = field(default_factory=dict)
    false_positives: list[int] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    truth_phase: dict[str, str] = field(default_factory=dict)

    @property
    def true_positives(self) -> int:
        return len(self.matches)


def match_frame(truth_states: dict, predictions: list[Prediction], max_distance: float
                ) -> FrameMatch:
    """One instant: Hungarian on Euclidean world distance, gated by max_distance."""
    timestamp = predictions[0].timestamp if predictions else 0.0
    match = FrameMatch(timestamp=timestamp)
    vehicles = sorted(truth_states)
    match.truth_phase = {v: truth_states[v].phase.value for v in vehicles}
    if not vehicles:
        match.false_positives = [p.global_id for p in predictions]
        return match
    if not predictions:
        match.false_negatives = list(vehicles)
        return match
    cost = np.full((len(vehicles), len(predictions)), np.inf)
    for i, vehicle_id in enumerate(vehicles):
        truth = truth_states[vehicle_id].position
        for j, prediction in enumerate(predictions):
            distance = float(np.linalg.norm(np.asarray(prediction.position) - truth))
            if distance <= max_distance:
                cost[i, j] = distance
    pairs = solve_assignment(cost)
    matched_predictions = set()
    for i, j in pairs:
        vehicle_id = vehicles[i]
        match.matches[vehicle_id] = predictions[j].global_id
        match.distances[vehicle_id] = float(cost[i, j])
        matched_predictions.add(j)
    match.false_negatives = [v for i, v in enumerate(vehicles) if v not in match.matches]
    match.false_positives = [p.global_id for j, p in enumerate(predictions)
                             if j not in matched_predictions]
    return match


def match_timeline(oracle, prediction_frames: list[tuple[float, list[Prediction]]],
                   max_distance: float = 3.0) -> list[FrameMatch]:
    """Match every published snapshot against ground truth at the same instant."""
    frames = []
    for timestamp, predictions in prediction_frames:
        truth = oracle.observable_at(timestamp)
        frame = match_frame(truth, predictions, max_distance)
        frame.timestamp = timestamp
        frames.append(frame)
    return frames


def dominant_mapping(frames: list[FrameMatch]) -> dict[str, int]:
    """Majority predicted Global ID per physical vehicle (for ownership checks)."""
    tally: dict[str, dict[int, int]] = {}
    for frame in frames:
        for vehicle_id, global_id in frame.matches.items():
            tally.setdefault(vehicle_id, {}).setdefault(global_id, 0)
            tally[vehicle_id][global_id] += 1
    return {vehicle_id: max(counts, key=counts.get) for vehicle_id, counts in tally.items()}


def snapshots_to_predictions(results, include_hidden: bool = False
                             ) -> list[tuple[float, list[Prediction]]]:
    """Extract published vehicles from a list of pipeline StepResults."""
    frames = []
    for result in results:
        snapshot = result.snapshot
        if snapshot is None:
            continue
        predictions = []
        for view in snapshot.vehicles:
            if not include_hidden and view.display_state.value == "hidden":
                continue
            predictions.append(Prediction(
                timestamp=snapshot.timestamp, global_id=view.global_id,
                position=np.asarray(view.world_position, dtype=float),
                display_state=view.display_state.value, slot_id=view.slot_id,
                camera_id=view.camera_id))
        frames.append((snapshot.timestamp, predictions))
    return frames
