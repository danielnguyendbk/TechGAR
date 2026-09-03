"""Stage 7 — topology-constrained cross-camera association.

The matrix is built over (live identities x fused observations), topology-invalid
candidates are excluded before solving (PLAN 2 §4.5), the assignment is global and
one-to-one (§4), and every accepted match must clear the margin rule (§4.7) *and*
the Re-ID acceptance rule (§6.2).  When it does not, the observation is deferred:
the established Global ID is kept and the decision is postponed, never replaced by
a fresh identity because the best candidate was momentarily uncertain
(PLAN 1 stage 7 logic 5-7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assignment import solve_assignment
from .config_world import AssociationConfig, IdentityConfig
from .cost import CostComponents, IdentityView, compute_cost
from .states import LifecycleState
from .topology import CameraTopology
from .world_contracts import AssociationDecision, DecisionType, FusedWorldDetection


@dataclass
class AssociationOutcome:
    decisions: list[AssociationDecision] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    components: dict[tuple[int, int], CostComponents] = field(default_factory=dict)
    identity_order: list[int] = field(default_factory=list)
    observation_order: list[int] = field(default_factory=list)

    def decision_for(self, observation_id: int) -> AssociationDecision | None:
        for decision in self.decisions:
            if decision.observation_id == observation_id:
                return decision
        return None

    def feasible_identities(self, observation_id: int) -> list[int]:
        return [gid for (gid, oid), c in self.components.items()
                if oid == observation_id and c.feasible]

    def best_score(self, observation_id: int, exclude: int | None = None) -> tuple[int | None, float]:
        best, value = None, 0.0
        for (gid, oid), c in self.components.items():
            if oid != observation_id or gid == exclude or not c.feasible:
                continue
            if c.identity_score > value:
                best, value = gid, c.identity_score
        return best, value


class TopologyConstrainedAssociator:
    def __init__(self, topology: CameraTopology, config: AssociationConfig | None = None,
                 identity_config: IdentityConfig | None = None, rho_seam: float = 0.0) -> None:
        self.topology = topology
        self.config = config or AssociationConfig()
        self.identity_config = identity_config or IdentityConfig()
        self.rho_seam = rho_seam

    def associate(self, identities: list[IdentityView],
                  observations: list[FusedWorldDetection]) -> AssociationOutcome:
        cfg = self.config
        outcome = AssociationOutcome()
        outcome.identity_order = [i.global_id for i in identities]
        outcome.observation_order = [o.observation_id for o in observations]
        if not observations:
            return outcome
        matrix = np.full((len(identities), len(observations)), np.inf)
        for i, identity in enumerate(identities):
            for j, observation in enumerate(observations):
                components = compute_cost(identity, observation, self.topology, cfg,
                                          self.identity_config, self.rho_seam)
                outcome.components[(identity.global_id, observation.observation_id)] = components
                if components.feasible:
                    matrix[i, j] = components.total
        outcome.matrix = matrix
        pairs = solve_assignment(matrix)
        matched_observations = set()
        for i, j in pairs:
            identity, observation = identities[i], observations[j]
            components = outcome.components[(identity.global_id, observation.observation_id)]
            margin, competitor = self._margin(matrix, i, j, identities)
            _, best_other_score = outcome.best_score(observation.observation_id,
                                                     exclude=identity.global_id)
            score = components.identity_score
            competing = tuple(gid for gid in outcome.feasible_identities(observation.observation_id)
                              if gid != identity.global_id)
            if margin < cfg.margin_min:
                outcome.decisions.append(self._defer(
                    observation, f"margin_{margin:.2f}<{cfg.margin_min:.2f}", competing, score,
                    margin, components, competitor))
                continue
            if score < self.identity_config.tau_accept:
                outcome.decisions.append(self._defer(
                    observation, f"score_{score:.2f}<tau_accept", competing, score, margin,
                    components, competitor))
                continue
            if score - best_other_score < self.identity_config.tau_margin and competing:
                outcome.decisions.append(self._defer(
                    observation, "score_margin", competing, score, margin, components, competitor))
                continue
            decision_type = DecisionType.CONTINUITY
            if not components.same_camera:
                decision_type = DecisionType.HANDOFF
            elif identity.lifecycle in (LifecycleState.TEMPORARILY_MISSING,
                                        LifecycleState.OCCLUDED):
                decision_type = DecisionType.REACQUIRE
            matched_observations.add(observation.observation_id)
            outcome.decisions.append(AssociationDecision(
                observation_id=observation.observation_id, timestamp=observation.timestamp,
                frame_sequence=observation.frame_sequence, decision_type=decision_type,
                assigned_global_id=identity.global_id, confidence=float(observation.fusion_confidence),
                identity_score=score, margin=float(margin), competing_global_ids=competing,
                cost_breakdown=components.as_dict()))
        for observation in observations:
            if observation.observation_id in matched_observations:
                continue
            if outcome.decision_for(observation.observation_id) is not None:
                continue
            best_gid, best_score = outcome.best_score(observation.observation_id)
            outcome.decisions.append(AssociationDecision(
                observation_id=observation.observation_id, timestamp=observation.timestamp,
                frame_sequence=observation.frame_sequence,
                decision_type=DecisionType.NEW_CANDIDATE, assigned_global_id=None,
                confidence=float(observation.fusion_confidence), identity_score=best_score,
                competing_global_ids=tuple(
                    outcome.feasible_identities(observation.observation_id)),
                defer_reason="" if best_gid is None else f"best_candidate_{best_gid}"))
        return outcome

    @staticmethod
    def _defer(observation, reason, competing, score, margin, components,
               competitor) -> AssociationDecision:
        return AssociationDecision(
            observation_id=observation.observation_id, timestamp=observation.timestamp,
            frame_sequence=observation.frame_sequence, decision_type=DecisionType.DEFER,
            assigned_global_id=None, confidence=float(observation.fusion_confidence),
            identity_score=score, margin=float(margin), competing_global_ids=competing,
            defer_reason=reason, cost_breakdown={**components.as_dict(),
                                                 "competitor": competitor})

    @staticmethod
    def _margin(matrix: np.ndarray, row: int, col: int,
                identities: list[IdentityView]) -> tuple[float, int | None]:
        """M = C_2 - C_1 over the best competing feasible candidate (PLAN 2 §4.7)."""
        selected = matrix[row, col]
        best, competitor = float("inf"), None
        for i in range(matrix.shape[0]):
            if i == row:
                continue
            value = matrix[i, col]
            if value >= selected and value < best:
                best, competitor = value, identities[i].global_id
        for j in range(matrix.shape[1]):
            if j == col:
                continue
            value = matrix[row, j]
            if value >= selected and value < best:
                best, competitor = value, identities[row].global_id
        if not np.isfinite(best):
            return float("inf"), None
        return float(best - selected), competitor
