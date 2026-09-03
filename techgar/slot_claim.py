"""Slot arrival evidence: the sliding temporal window of PLAN 2 §5.4.

    o_t = (IoU_t, Coverage_t, D_center,t, delta_d_t, v_t, q_t)

Arrival confirms only when all five conditions hold at once: enough IoU samples,
enough Coverage samples, an inward displacement of at least tau_inward, a stable
centroid, and a speed below v_parked in the tail of the window.  A single frame
with a loose bbox and IoU = 0.95 confirms nothing (PLAN 2 §5.6 Fail case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config_world import SlotConfig


@dataclass
class SlotObservation:
    timestamp: float
    iou: float
    coverage: float
    centre_distance: float
    delta_d: float
    speed: float
    quality: float
    centroid: np.ndarray
    evidence: float


@dataclass
class ClaimEvidence:
    n_iou: int = 0
    n_coverage: int = 0
    max_delta_d: float = 0.0
    centroid_variance: float = 0.0
    tail_speed: float = 0.0
    duration: float = 0.0
    evidence: float = 0.0
    centred: bool = False
    samples: int = 0

    def as_dict(self) -> dict:
        return {"n_iou": self.n_iou, "n_coverage": self.n_coverage,
                "max_delta_d": self.max_delta_d, "centroid_variance": self.centroid_variance,
                "tail_speed": self.tail_speed, "duration": self.duration,
                "evidence": self.evidence, "centred": self.centred, "samples": self.samples}


@dataclass
class SlotClaim:
    """One (vehicle, slot) arrival hypothesis."""

    slot_id: str
    global_id: int
    opened_at: float
    observations: list[SlotObservation] = field(default_factory=list)
    d_outside: float | None = None
    last_evidence_at: float = 0.0
    confirmed_at: float | None = None

    def add(self, observation: SlotObservation) -> None:
        if self.d_outside is None:
            # d_outside = distance from the first valid candidate position (PLAN 2 §5.3)
            self.d_outside = observation.centre_distance
        if self.observations:
            # Use the observed footprint displacement for the parking-speed gate.
            # A constant-velocity filter necessarily carries momentum for a short
            # time after a vehicle stops; treating that stale latent velocity as a
            # fresh measurement prevents a genuine arrival from ever completing
            # its dwell window.  The centroid delta is the speed actually evidenced
            # by consecutive slot observations and remains valid when tracking is
            # temporarily interrupted and the last positive footprint is held.
            previous = self.observations[-1]
            dt = observation.timestamp - previous.timestamp
            if dt > 1e-6:
                observation.speed = float(
                    np.linalg.norm(observation.centroid - previous.centroid) / dt)
        observation.delta_d = float(self.d_outside - observation.centre_distance)
        self.observations.append(observation)
        self.last_evidence_at = observation.timestamp

    def prune(self, now: float, window: float) -> None:
        cutoff = now - window
        self.observations = [o for o in self.observations if o.timestamp >= cutoff]

    def age(self, now: float) -> float:
        return max(0.0, now - self.opened_at)

    def idle(self, now: float) -> float:
        return max(0.0, now - self.last_evidence_at)

    def evidence(self, config: SlotConfig) -> ClaimEvidence:
        result = ClaimEvidence(samples=len(self.observations))
        if not self.observations:
            return result
        result.n_iou = sum(1 for o in self.observations if o.iou >= config.tau_iou)
        result.n_coverage = sum(1 for o in self.observations if o.coverage >= config.tau_coverage)
        result.max_delta_d = float(max(o.delta_d for o in self.observations))
        tail_start = self.observations[-1].timestamp - config.window * config.tail_fraction
        tail = [o for o in self.observations if o.timestamp >= tail_start] or self.observations[-1:]
        # The complete window intentionally includes the inward approach, so its
        # centroid variance cannot also represent the final stopped-state test.
        # Measure stability over the same tail interval used by the parked-speed
        # gate: movement is required earlier, stability is required at the end.
        centroids = np.asarray([o.centroid for o in tail], dtype=float)
        if len(centroids) >= 2:
            result.centroid_variance = float(np.sum(centroids.var(axis=0)))
        # Camera/anchor jitter produces isolated displacement spikes even after
        # a vehicle has stopped.  Taking the maximum makes one noisy sample veto
        # the entire parking window and couples confirmation latency to sensor
        # noise.  The median is the robust estimate of sustained motion; the
        # independent centroid-variance gate still rejects unstable/transiting
        # trajectories.
        result.tail_speed = float(np.median([o.speed for o in tail]))
        result.duration = self.observations[-1].timestamp - self.observations[0].timestamp
        recent = self.observations[-3:]
        result.evidence = float(np.mean([o.evidence for o in recent]))
        result.centred = bool(self.observations[-1].centre_distance <= config.tau_center)
        return result

    def satisfied(self, config: SlotConfig) -> tuple[bool, ClaimEvidence, list[str]]:
        """The five conditions of PLAN 2 §5.4 plus the dwell requirement."""
        evidence = self.evidence(config)
        missing: list[str] = []
        if not config.enable_temporal_window:
            # Ablation: instantaneous confirmation, i.e. exactly the failure mode the
            # temporal window exists to prevent.
            ok = evidence.evidence >= config.tau_confirm and evidence.samples >= 1
            return ok, evidence, [] if ok else ["instant_evidence"]
        if evidence.n_iou < config.n_iou:
            missing.append(f"n_iou={evidence.n_iou}<{config.n_iou}")
        if evidence.n_coverage < config.n_coverage:
            missing.append(f"n_coverage={evidence.n_coverage}<{config.n_coverage}")
        if evidence.max_delta_d < config.tau_inward:
            missing.append(f"inward={evidence.max_delta_d:.2f}<{config.tau_inward}")
        if evidence.centroid_variance > config.sigma2_stable:
            missing.append(f"unstable={evidence.centroid_variance:.3f}")
        if evidence.tail_speed > config.v_parked:
            missing.append(f"speed={evidence.tail_speed:.2f}>{config.v_parked}")
        if evidence.duration < config.dwell_confirm:
            missing.append(f"dwell={evidence.duration:.2f}<{config.dwell_confirm}")
        if evidence.evidence < config.tau_confirm:
            missing.append(f"evidence={evidence.evidence:.2f}<{config.tau_confirm}")
        return (not missing), evidence, missing
