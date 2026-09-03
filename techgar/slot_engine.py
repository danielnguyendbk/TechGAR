"""Stage 9 — spatial-temporal parking slot occupancy engine (PLAN 2 §5).

Evidence per (vehicle, slot) is IoU + Coverage + centroid distance + inward
motion, accumulated over a *time* window, confirmed through a dwell requirement,
and held by a three-level hysteresis (enter > confirm > release).  Ownership is
resolved by a global one-to-one assignment, which is what structurally forbids
"one Global ID owning two slots" and "two Global IDs owning one slot"
(PLAN 3 §4 targets).

A parked vehicle stops moving, so its motion evidence disappears: releasing a slot
therefore requires *positive* departure evidence, never merely the absence of
observations (PLAN 2 §5.7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assignment import solve_assignment
from .config_world import SlotConfig
from .geometry import polygon_area, polygon_centroid, polygon_coverage, polygon_iou
from .slot_claim import ClaimEvidence, SlotClaim, SlotObservation
from .states import ParkingSlotState, SlotOccupancy


@dataclass
class VehicleFootprintView:
    """What the slot engine is allowed to know about a vehicle."""

    global_id: int
    footprint: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    quality: float = 1.0
    observed: bool = True
    parked_hint: bool = False


@dataclass
class VisionSlotVote:
    """One camera's pixel-content verdict for a physical slot (stage 9 fusion)."""

    slot_id: str
    occupied: bool
    evidence: float = 0.0
    ready: bool = False


@dataclass
class SlotEvent:
    timestamp: float
    slot_id: str
    global_id: int | None
    kind: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)


class SlotOccupancyEngine:
    def __init__(self, slots: dict[str, np.ndarray], config: SlotConfig | None = None) -> None:
        self.config = config or SlotConfig()
        self.slots = {slot_id: np.asarray(poly, dtype=float) for slot_id, poly in slots.items()}
        self.slot_area = {slot_id: polygon_area(poly) for slot_id, poly in self.slots.items()}
        self.slot_centre = {slot_id: polygon_centroid(poly) for slot_id, poly in self.slots.items()}
        self.states: dict[str, ParkingSlotState] = {slot_id: ParkingSlotState(slot_id)
                                                    for slot_id in self.slots}
        self.claims: dict[tuple[str, int], SlotClaim] = {}
        self.approach_distance: dict[tuple[str, int], tuple[float, float]] = {}
        self.events: list[SlotEvent] = []
        # Vision fusion (PLAN 1 stage 9): the latest pixel-content verdict per
        # slot, independent of tracking.  A parked vehicle keeps its motion
        # evidence at zero, so occupancy must survive on vision alone.
        self.vision_votes: dict[str, VisionSlotVote] = {}
        self.vision_confirm_streak: dict[str, int] = {}
        self.vision_release_streak: dict[str, int] = {}

    # --- evidence -----------------------------------------------------------
    def _observe(self, vehicle: VehicleFootprintView, slot_id: str,
                 timestamp: float) -> SlotObservation | None:
        slot = self.slots[slot_id]
        coverage = polygon_coverage(vehicle.footprint, slot)
        if coverage <= 0.0:
            return None
        iou = polygon_iou(vehicle.footprint, slot)
        vehicle_area = polygon_area(vehicle.footprint)
        slot_area = self.slot_area[slot_id]
        ideal = (min(vehicle_area, slot_area) / max(vehicle_area, slot_area)
                 if max(vehicle_area, slot_area) > 0 else 1.0)
        iou_normalised = float(np.clip(iou / max(ideal, 1e-6), 0.0, 1.0))
        centroid = polygon_centroid(vehicle.footprint)
        distance = float(np.linalg.norm(centroid - self.slot_centre[slot_id]))
        speed = float(np.linalg.norm(vehicle.velocity))
        evidence = 0.6 * coverage + 0.4 * iou_normalised
        return SlotObservation(timestamp=timestamp, iou=iou_normalised, coverage=coverage,
                               centre_distance=distance, delta_d=0.0, speed=speed,
                               quality=vehicle.quality, centroid=centroid, evidence=evidence)

    def _gather(self, vehicles: list[VehicleFootprintView], timestamp: float
                ) -> dict[tuple[str, int], SlotObservation]:
        gathered: dict[tuple[str, int], SlotObservation] = {}
        for vehicle in vehicles:
            for slot_id in self.slots:
                observation = self._observe(vehicle, slot_id, timestamp)
                if observation is None or observation.evidence < self.config.tau_release * 0.5:
                    continue
                gathered[(slot_id, vehicle.global_id)] = observation
        return gathered

    # --- main update --------------------------------------------------------
    def update_vision(self, votes: dict[str, VisionSlotVote] | None,
                      timestamp: float) -> None:
        """Absorb the pixel-content occupancy verdicts (PLAN 1 stage 9 fusion).

        Vision runs independently of tracking and may mark a slot occupied
        with no owning identity (a vehicle parked before startup, or one whose
        track has expired).  Vision may also *veto* a tracking-driven release
        when the vehicle is still visually present — that veto is what keeps
        the slot colour stable while the vehicle sits motionless.
        """
        self.vision_votes = dict(votes or {})
        for slot_id, vote in self.vision_votes.items():
            if slot_id not in self.states:
                continue
            if not vote.ready:
                continue
            if vote.occupied:
                self.vision_confirm_streak[slot_id] = self.vision_confirm_streak.get(slot_id, 0) + 1
                self.vision_release_streak.pop(slot_id, None)
            else:
                self.vision_release_streak[slot_id] = self.vision_release_streak.get(slot_id, 0) + 1
                self.vision_confirm_streak.pop(slot_id, None)
            state = self.states[slot_id]
            state.vision_occupied = bool(
                self.vision_confirm_streak.get(slot_id, 0) >= self.config.vision_confirm_frames)

    def update(self, vehicles: list[VehicleFootprintView], timestamp: float,
               frame_sequence: int = 0) -> list[SlotEvent]:
        cfg = self.config
        self.events = []
        observed = self._gather(vehicles, timestamp)
        seen_vehicles = {v.global_id for v in vehicles}

        for key, observation in observed.items():
            claim = self.claims.get(key)
            if claim is None:
                if observation.evidence < cfg.tau_enter:
                    prior = self.approach_distance.get(key)
                    furthest = observation.centre_distance if prior is None else max(
                        observation.centre_distance, prior[0])
                    self.approach_distance[key] = (furthest, timestamp)
                    continue
                claim = SlotClaim(key[0], key[1], timestamp)
                precursor = self.approach_distance.pop(key, None)
                if precursor is not None and timestamp - precursor[1] <= cfg.window:
                    claim.d_outside = precursor[0]
                self.claims[key] = claim
                self.events.append(SlotEvent(timestamp, key[0], key[1], "claim_open"))
            claim.add(observation)
            claim.prune(timestamp, cfg.window)

        for key, (_, last_seen) in list(self.approach_distance.items()):
            if timestamp - last_seen > cfg.window:
                self.approach_distance.pop(key, None)

        for key, claim in list(self.claims.items()):
            claim.prune(timestamp, cfg.window)
            state = self.states[key[0]]
            owner_missing = key[1] not in seen_vehicles
            grace = cfg.claim_grace if owner_missing else cfg.window
            if state.owning_global_id == key[1]:
                continue
            if claim.idle(timestamp) > grace or not claim.observations:
                self.claims.pop(key, None)
                self.events.append(SlotEvent(timestamp, key[0], key[1], "claim_expired",
                                             detail=f"idle={claim.idle(timestamp):.2f}s"))

        confirmations = self._confirmations(timestamp)
        self._assign(confirmations, timestamp, frame_sequence)
        self._release_pass(observed, seen_vehicles, timestamp)
        self._vision_pass(timestamp, frame_sequence)
        for slot_id, state in self.states.items():
            state.last_update_timestamp = timestamp
        return self.events

    # --- vision fusion pass ---------------------------------------------------
    def _vision_pass(self, timestamp: float, frame_sequence: int) -> None:
        """Apply pixel-content verdicts after the tracking pass (stage 9).

        Order matters: the tracking pass runs first so a *confirmed* owner is
        never displaced by vision.  Vision then (a) keeps occupied any slot it
        sees a vehicle in — including slots with no identity at all — and
        (b) blocks a tracking-driven release while the vehicle is visually
        still there.
        """
        if not self.config.enable_vision_fusion:
            return
        for slot_id, state in self.states.items():
            vision_occupied = state.vision_occupied
            if vision_occupied:
                if state.occupancy_state in (SlotOccupancy.EMPTY,
                                             SlotOccupancy.CLAIM_PENDING):
                    state.occupancy_state = SlotOccupancy.OCCUPIED
                    # No owner: vision alone cannot mint an identity, so the
                    # slot is occupied *anonymously* until tracking claims it.
                    if state.owning_global_id is None:
                        self.events.append(SlotEvent(
                            timestamp, slot_id, None, "vision_occupied",
                            detail="pixel-content occupancy without identity"))
                elif state.occupancy_state is SlotOccupancy.RELEASING:
                    # Tracking wanted to release; vision vetoes (§5.7 spirit:
                    # absence of tracking evidence is not departure evidence).
                    state.releasing_since = None
                    state.occupancy_state = SlotOccupancy.OCCUPIED
                    self.events.append(SlotEvent(
                        timestamp, slot_id, state.owning_global_id, "vision_veto_release",
                        detail="vehicle still visually present"))
            elif (state.occupancy_state is SlotOccupancy.OCCUPIED
                  and state.owning_global_id is None
                  and self.vision_release_streak.get(slot_id, 0)
                  >= self.config.vision_release_frames):
                # An anonymously vision-occupied slot empties again on
                # sustained vision-empty evidence (the vehicle left before
                # tracking ever saw it).
                state.occupancy_state = SlotOccupancy.EMPTY
                state.occupied_since = None
                state.confirmation_confidence = 0.0
                self.events.append(SlotEvent(
                    timestamp, slot_id, None, "vision_released",
                    detail="pixel-content empty"))

    def _confirmations(self, timestamp: float
                       ) -> dict[tuple[str, int], tuple[ClaimEvidence, float]]:
        ready: dict[tuple[str, int], tuple[ClaimEvidence, float]] = {}
        for key, claim in self.claims.items():
            state = self.states[key[0]]
            if (state.occupancy_state is SlotOccupancy.OCCUPIED
                    and state.owning_global_id is not None
                    and state.owning_global_id != key[1]):
                continue                       # a confirmed owner is not displaced by a claim
            satisfied, evidence, missing = claim.satisfied(self.config)
            state.overlap_score = max(state.overlap_score * 0.5, evidence.evidence)
            if state.occupancy_state is SlotOccupancy.EMPTY and evidence.evidence >= self.config.tau_enter:
                state.occupancy_state = SlotOccupancy.CLAIM_PENDING
                state.claim_global_id = key[1]
                state.claim_started_at = claim.opened_at
            if satisfied:
                ready[key] = (evidence, evidence.evidence)
            elif missing and state.occupancy_state is SlotOccupancy.CLAIM_PENDING:
                state.confirmation_confidence = float(np.clip(evidence.evidence, 0.0, 1.0))
        return ready

    def _assign(self, confirmations, timestamp: float, frame_sequence: int) -> None:
        """One-to-one ownership resolution with a deterministic margin."""
        if not confirmations:
            return
        slots = sorted({key[0] for key in confirmations})
        vehicles = sorted({key[1] for key in confirmations})
        cost = np.full((len(vehicles), len(slots)), np.inf)
        for (slot_id, global_id), (_, score) in confirmations.items():
            cost[vehicles.index(global_id), slots.index(slot_id)] = -score
        for i, j in solve_assignment(cost):
            global_id, slot_id = vehicles[i], slots[j]
            score = -float(cost[i, j])
            competing = [-float(cost[i, k]) for k in range(len(slots))
                         if k != j and np.isfinite(cost[i, k])]
            if competing and score - max(competing) < self.config.score_margin:
                self.events.append(SlotEvent(timestamp, slot_id, global_id, "ambiguous",
                                             detail="competing_slot_within_margin"))
                continue
            state = self.states[slot_id]
            if state.occupancy_state is SlotOccupancy.OCCUPIED and state.owning_global_id == global_id:
                state.releasing_since = None
                continue
            if state.owning_global_id not in (None, global_id):
                self.events.append(SlotEvent(timestamp, slot_id, global_id, "ownership_conflict",
                                             detail=f"held_by_{state.owning_global_id}"))
                continue
            for other_id, other in self.states.items():
                if other_id != slot_id and other.owning_global_id == global_id:
                    other.owning_global_id = None
                    other.occupancy_state = SlotOccupancy.EMPTY
                    self.events.append(SlotEvent(timestamp, other_id, global_id, "released",
                                                 detail="moved_to_" + slot_id))
            claim = self.claims.get((slot_id, global_id))
            evidence, _ = confirmations[(slot_id, global_id)]
            state.occupancy_state = SlotOccupancy.OCCUPIED
            state.owning_global_id = global_id
            state.occupied_since = timestamp
            state.releasing_since = None
            state.confirmation_confidence = float(np.clip(evidence.evidence, 0.0, 1.0))
            state.dwell_duration = evidence.duration
            if claim is not None:
                claim.confirmed_at = timestamp
            self.events.append(SlotEvent(timestamp, slot_id, global_id, "occupied",
                                         detail="confirmed", evidence=evidence.as_dict()))

    def _release_pass(self, observed, seen_vehicles, timestamp: float) -> None:
        cfg = self.config
        for slot_id, state in self.states.items():
            owner = state.owning_global_id
            if state.occupancy_state is SlotOccupancy.CLAIM_PENDING and owner is None:
                pending = [k for k in self.claims if k[0] == slot_id]
                if not pending:
                    state.occupancy_state = SlotOccupancy.EMPTY
                    state.claim_global_id = None
                    state.transit_events += 1
                    self.events.append(SlotEvent(timestamp, slot_id, None, "transit",
                                                 detail="claim_dropped_without_confirmation"))
                continue
            if owner is None:
                continue
            observation = observed.get((slot_id, owner))
            if owner not in seen_vehicles:
                # PLAN 2 §5.7: no observation is not evidence of departure.
                state.dwell_duration = timestamp - (state.occupied_since or timestamp)
                continue
            evidence = 0.0 if observation is None else observation.evidence
            state.overlap_score = evidence
            state.dwell_duration = timestamp - (state.occupied_since or timestamp)
            if evidence >= cfg.tau_confirm or not cfg.enable_hysteresis and evidence >= cfg.tau_release:
                state.releasing_since = None
                state.occupancy_state = SlotOccupancy.OCCUPIED
                continue
            if evidence < cfg.tau_release:
                if state.releasing_since is None:
                    state.releasing_since = timestamp
                    state.occupancy_state = SlotOccupancy.RELEASING
                    self.events.append(SlotEvent(timestamp, slot_id, owner, "releasing",
                                                 detail=f"evidence={evidence:.2f}"))
                elif timestamp - state.releasing_since >= (cfg.release_duration
                                                           if cfg.enable_hysteresis else 0.0):
                    self.events.append(SlotEvent(timestamp, slot_id, owner, "released",
                                                 detail=f"evidence={evidence:.2f}"))
                    state.occupancy_state = SlotOccupancy.EMPTY
                    state.owning_global_id = None
                    state.occupied_since = None
                    state.releasing_since = None
                    state.confirmation_confidence = 0.0
                    self.claims.pop((slot_id, owner), None)

    # --- helpers ------------------------------------------------------------
    def owner_of(self, slot_id: str) -> int | None:
        return self.states[slot_id].owning_global_id

    def slot_of(self, global_id: int) -> str | None:
        for slot_id, state in self.states.items():
            if state.owning_global_id == global_id:
                return slot_id
        return None

    def occupied_slots(self) -> dict[str, int]:
        return {slot_id: state.owning_global_id for slot_id, state in self.states.items()
                if state.occupancy_state is SlotOccupancy.OCCUPIED
                and state.owning_global_id is not None}
