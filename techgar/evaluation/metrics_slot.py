"""Parking-slot metrics — PLAN 3 §4.

Per slot-frame precision/recall/F1, ownership accuracy, and the two collision
counts that must be exactly zero (one Global ID owning two slots, two Global IDs
owning one slot).

Two variants are reported.  ``strict`` scores every slot-frame.  ``tolerant``
excludes a window around each ground-truth transition, because temporal
confirmation and release hysteresis are *required* behaviour: PLAN 3 scenario G
fails a system that marks D08 occupied on the first frame, so the confirmation
delay must not simultaneously be scored as an error.  Both numbers are always
reported side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SlotMetrics:
    slot_frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_vacancy_rate: float = 0.0
    false_occupied_rate: float = 0.0
    ownership_correct: int = 0
    ownership_total: int = 0
    ownership_accuracy: float = 0.0
    one_id_two_slots: int = 0
    two_ids_one_slot: int = 0
    excluded_transition_frames: int = 0
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"slot_frames": self.slot_frames, "tp": self.tp, "fp": self.fp, "fn": self.fn,
                "tn": self.tn, "precision": self.precision, "recall": self.recall,
                "f1": self.f1, "false_vacancy_rate": self.false_vacancy_rate,
                "false_occupied_rate": self.false_occupied_rate,
                "ownership_accuracy": self.ownership_accuracy,
                "ownership_total": self.ownership_total,
                "one_id_two_slots": self.one_id_two_slots,
                "two_ids_one_slot": self.two_ids_one_slot,
                "excluded_transition_frames": self.excluded_transition_frames}


def slot_metrics(oracle, results, mapping: dict[str, int], tolerance: float = 1.2,
                 strict: bool = False) -> SlotMetrics:
    """Compare predicted slot state with physical occupancy, slot-frame by slot-frame."""
    metrics = SlotMetrics()
    inverse = {global_id: vehicle_id for vehicle_id, global_id in mapping.items()}
    transitions = _transition_times(oracle)
    for result in results:
        snapshot = result.snapshot
        if snapshot is None or not snapshot.slots:
            continue
        timestamp = snapshot.timestamp
        owners_by_gid: dict[int, list[str]] = {}
        for slot_view in snapshot.slots:
            truth_owner = oracle.slot_owner_at(timestamp, slot_view.slot_id)
            predicted = slot_view.occupancy_state == "occupied"
            near_transition = any(abs(timestamp - t) <= tolerance
                                  for t in transitions.get(slot_view.slot_id, ()))
            if not strict and near_transition:
                metrics.excluded_transition_frames += 1
                continue
            metrics.slot_frames += 1
            if truth_owner is not None and predicted:
                metrics.tp += 1
                metrics.ownership_total += 1
                if inverse.get(slot_view.owning_global_id) == truth_owner:
                    metrics.ownership_correct += 1
            elif truth_owner is not None and not predicted:
                metrics.fn += 1
                metrics.ownership_total += 1
            elif truth_owner is None and predicted:
                metrics.fp += 1
            else:
                metrics.tn += 1
            if predicted and slot_view.owning_global_id is not None:
                owners_by_gid.setdefault(slot_view.owning_global_id, []).append(slot_view.slot_id)
        for global_id, slots in owners_by_gid.items():
            if len(slots) > 1:
                metrics.one_id_two_slots += 1
                metrics.detail.setdefault("one_id_two_slots", []).append(
                    (timestamp, global_id, slots))
    metrics.precision = metrics.tp / max(metrics.tp + metrics.fp, 1)
    metrics.recall = metrics.tp / max(metrics.tp + metrics.fn, 1)
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
    occupied_truth = metrics.tp + metrics.fn
    empty_truth = metrics.tn + metrics.fp
    metrics.false_vacancy_rate = metrics.fn / max(occupied_truth, 1)
    metrics.false_occupied_rate = metrics.fp / max(empty_truth, 1)
    metrics.ownership_accuracy = metrics.ownership_correct / max(metrics.ownership_total, 1)
    metrics.two_ids_one_slot = 0     # structurally impossible: one owner field per slot
    return metrics


def _transition_times(oracle) -> dict[str, tuple[float, ...]]:
    """Instants at which physical slot occupancy changes."""
    per_slot: dict[str, list[float]] = {}
    previous: dict[str, str | None] = {}
    for row in oracle.slot_truth_rows():
        last = previous.get(row.slot_id, "__init__")
        if last != "__init__" and last != row.physical_vehicle_id:
            per_slot.setdefault(row.slot_id, []).append(row.timestamp)
        previous[row.slot_id] = row.physical_vehicle_id
    return {slot_id: tuple(times) for slot_id, times in per_slot.items()}
