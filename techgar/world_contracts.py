"""Phase 0 data contracts, world-side half.

Every world observation carries a covariance — Phase 0 exit criterion
"Mọi world observation chứa covariance".  The validators at the bottom are run
by the Phase 0 report tool and by the pipeline in ``strict`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .contracts import ContractViolation, TopologyRegion


class DecisionType(str, Enum):
    """PLAN 1 stage 7 ``decision_type``."""

    CONTINUITY = "continuity"          # same camera, same identity
    HANDOFF = "handoff"                # topology-valid cross-camera transfer
    REACQUIRE = "reacquire"            # identity recovered after a gap
    DEFER = "defer"                    # ambiguous: keep old GID, decide later
    NEW_CANDIDATE = "new_candidate"    # eligible to mint (registry decides)
    QUARANTINE = "quarantine"          # conflicting evidence, do not merge


@dataclass
class WorldDetection:
    """PLAN 1 stage 5 output (single camera, world frame)."""

    camera_id: str
    timestamp: float
    frame_sequence: int
    world_position: np.ndarray        # (2,)
    world_covariance: np.ndarray      # (2, 2)
    world_footprint: np.ndarray       # (4, 2)
    source_pixel_position: np.ndarray
    topology_region: TopologyRegion
    local_track_id: int
    quality: float
    confidence: float
    world_velocity: np.ndarray | None = None
    footprint_area: float = 0.0
    footprint_aspect: float = 1.0
    appearance: np.ndarray | None = None
    latent: bool = False
    partial: bool = False
    occlusion_group_id: int | None = None
    observation_id: int = -1

    def __post_init__(self) -> None:
        self.world_position = np.asarray(self.world_position, dtype=float).reshape(2)
        self.world_covariance = np.asarray(self.world_covariance, dtype=float).reshape(2, 2)
        validate_world_detection(self)


@dataclass
class FusedWorldDetection:
    """PLAN 1 stage 6 output — the unit the identity layer reasons about."""

    timestamp: float
    frame_sequence: int
    position: np.ndarray
    covariance: np.ndarray
    footprint: np.ndarray
    contributing_cameras: tuple[str, ...]
    contributing_observations: tuple[int, ...]
    fusion_confidence: float
    topology_region: TopologyRegion
    local_track_ids: tuple[tuple[str, int], ...]
    quality: float
    velocity: np.ndarray | None = None
    footprint_area: float = 0.0
    footprint_aspect: float = 1.0
    appearance: np.ndarray | None = None
    latent: bool = False
    partial: bool = False
    occlusion_group_id: int | None = None
    observation_id: int = -1

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(2)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(2, 2)
        if not np.isfinite(self.covariance).all():
            raise ContractViolation("fused detection with non-finite covariance")

    @property
    def primary_camera(self) -> str:
        return self.contributing_cameras[0]


@dataclass
class AssociationDecision:
    """PLAN 1 stage 7 output, fully auditable (rubric A: 'auditable + timestamp')."""

    observation_id: int
    timestamp: float
    frame_sequence: int
    decision_type: DecisionType
    assigned_global_id: int | None = None
    confidence: float = 0.0
    identity_score: float = 0.0
    margin: float = float("inf")
    competing_global_ids: tuple[int, ...] = ()
    defer_reason: str = ""
    cost_breakdown: dict = field(default_factory=dict)


def validate_world_detection(det: WorldDetection) -> None:
    if not det.camera_id:
        raise ContractViolation("world detection without camera id")
    if not np.isfinite(det.timestamp):
        raise ContractViolation("world detection without timestamp")
    cov = np.asarray(det.world_covariance, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        raise ContractViolation("world detection covariance must be a finite 2x2 matrix")
    if np.trace(cov) <= 0.0:
        raise ContractViolation("world detection covariance must be positive definite")
    eigs = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if eigs.min() <= 0.0:
        raise ContractViolation("world detection covariance is not positive definite")


def validate_pair(pair) -> None:
    for cam, frame in pair.frames.items():
        if frame.camera_id != cam:
            raise ContractViolation("frame keyed under the wrong camera id")
        if not np.isfinite(frame.timestamp):
            raise ContractViolation("frame without timestamp in pair")


def validate_decision(decision: AssociationDecision) -> None:
    if not np.isfinite(decision.timestamp):
        raise ContractViolation("association decision without timestamp")
    if decision.decision_type in (DecisionType.CONTINUITY, DecisionType.HANDOFF,
                                  DecisionType.REACQUIRE) and decision.assigned_global_id is None:
        raise ContractViolation(f"{decision.decision_type} decision without a global id")
