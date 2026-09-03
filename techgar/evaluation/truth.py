"""Ground-truth oracle — the only thing the evaluator is allowed to trust.

Wraps a synthetic recording and answers, for any instant: where each physical
vehicle is, whether a correct system is expected to report it, which slot it
physically occupies, and which camera handoffs really happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import polygon_coverage
from ..simulation.annotate import GT_SLOT_COVERAGE, GTPhase, V_PARKED_TRUE, Annotator


@dataclass
class TruthState:
    vehicle_id: str
    position: np.ndarray
    footprint: np.ndarray
    observable: bool
    phase: GTPhase
    slot_id: str | None
    cameras: tuple[str, ...]


class TruthOracle:
    """Ground truth sampled at arbitrary instants (the simulator is analytic)."""

    def __init__(self, recording, observable_fraction: float = 0.25) -> None:
        self.recording = recording
        self.layout = recording.layout
        self.vehicles = {v.vehicle_id: v for v in recording.vehicles}
        self.annotator = Annotator(recording.layout, list(recording.vehicles))
        self.observable_fraction = observable_fraction

    # --- per-instant truth --------------------------------------------------
    def states_at(self, timestamp: float) -> dict[str, TruthState]:
        out: dict[str, TruthState] = {}
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.present(timestamp):
                continue
            visible_cameras = []
            best_fraction = 0.0
            phase = GTPhase.OUTSIDE_CAMERA
            slot_id = None
            for camera_id, camera in self.layout.cameras.items():
                fraction = self.annotator.visible_fraction(camera, vehicle, timestamp)
                if fraction >= self.observable_fraction:
                    visible_cameras.append(camera_id)
                if fraction >= best_fraction:
                    best_fraction = fraction
                    phase, slot_id = self.annotator.phase(camera, vehicle, timestamp, fraction)
            gt_slot, coverage = self.annotator.slot_of(vehicle, timestamp)
            speed = float(np.linalg.norm(vehicle.velocity(timestamp)))
            parked = gt_slot is not None and speed <= V_PARKED_TRUE
            # A parked vehicle must stay in the identity map even with no motion
            # evidence at all (PLAN 3 §9 absolute rejection condition).
            observable = bool(visible_cameras) or parked
            out[vehicle_id] = TruthState(
                vehicle_id=vehicle_id, position=vehicle.position(timestamp),
                footprint=vehicle.footprint(timestamp), observable=observable,
                phase=GTPhase.PARKED if parked else phase,
                slot_id=gt_slot if parked else slot_id,
                cameras=tuple(sorted(visible_cameras)))
        return out

    def observable_at(self, timestamp: float) -> dict[str, TruthState]:
        return {k: v for k, v in self.states_at(timestamp).items() if v.observable}

    # --- slots --------------------------------------------------------------
    def slot_owner_at(self, timestamp: float, slot_id: str) -> str | None:
        polygon = self.layout.slots[slot_id]
        best, coverage = None, 0.0
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.present(timestamp):
                continue
            value = polygon_coverage(vehicle.footprint(timestamp), polygon)
            speed = float(np.linalg.norm(vehicle.velocity(timestamp)))
            if value >= GT_SLOT_COVERAGE and speed <= V_PARKED_TRUE and value > coverage:
                best, coverage = vehicle_id, value
        return best

    def slot_truth_rows(self):
        return self.recording.slot_truth

    # --- handoffs -----------------------------------------------------------
    def handoffs(self):
        return self.recording.handoffs

    def valid_handoff(self, vehicle_id: str, source: str, target: str, dt: float) -> bool:
        """Whether the *topology* permits this transition at all (PLAN 2 §3.4)."""
        edge = self.layout.topology.edge(source, target)
        return edge is not None and edge.time_feasible(dt)
