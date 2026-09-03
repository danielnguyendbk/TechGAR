"""Identity / slot state contracts and the append-only audit event record.

PLAN 1 stage 8 defines the lifecycle; PLAN 3 rubric A awards points for
"Quyết định danh tính auditable + timestamp", which :class:`IdentityEvent`
guarantees structurally: an event cannot exist without a timestamp and a frame
sequence number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .contracts import ContractViolation


class LifecycleState(str, Enum):
    """PLAN 1 stage 8 logic 3."""

    PROVISIONAL = "provisional"                  # candidate, not user facing
    ACTIVE = "active"
    TEMPORARILY_MISSING = "temporarily_missing"
    OCCLUDED = "occluded"
    PARKED = "parked"
    EXIT_CONFIRMED = "exit_confirmed"
    RETIRED = "retired"
    RECOVERY_PENDING = "recovery_pending"          # after restart, awaiting re-observation
    UNKNOWN = "unknown"                            # mid-lot detection, cannot mint GID

    @property
    def is_live(self) -> bool:
        """Still a candidate for association (may receive observations)."""
        return self in (LifecycleState.PROVISIONAL, LifecycleState.ACTIVE,
                        LifecycleState.TEMPORARILY_MISSING, LifecycleState.OCCLUDED,
                        LifecycleState.PARKED, LifecycleState.RECOVERY_PENDING)

    @property
    def is_published(self) -> bool:
        """Visible to the frontend (PLAN 1 stage 10 logic 2)."""
        return self in (LifecycleState.ACTIVE, LifecycleState.TEMPORARILY_MISSING,
                        LifecycleState.OCCLUDED, LifecycleState.PARKED,
                        LifecycleState.RECOVERY_PENDING)


class DisplayState(str, Enum):
    OBSERVED = "observed"
    TEMPORARILY_MISSING = "temporarily_missing"
    PARKED = "parked"
    HIDDEN = "hidden"


class SlotOccupancy(str, Enum):
    EMPTY = "empty"
    CLAIM_PENDING = "claim_pending"   # arrival evidence accumulating (transit-safe)
    OCCUPIED = "occupied"
    RELEASING = "releasing"           # below release threshold, timer running


class IdentityEventType(str, Enum):
    MINT = "mint"
    ACTIVATE = "activate"
    MATCH = "match"
    HANDOFF = "handoff"
    DEFER = "defer"
    MISSING = "missing"
    OCCLUDED = "occluded"
    RECOVER = "recover"
    PARK = "park"
    UNPARK = "unpark"
    EXIT = "exit"
    RETIRE = "retire"
    QUARANTINE = "quarantine"
    COLLISION = "collision"
    ALIAS = "alias"
    MINT_BLOCKED = "mint_blocked"
    OVERLOAD = "overload"
    SESSION_BIND = "session_bind"
    SESSION_REMAP = "session_remap"
    SESSION_CLOSE = "session_close"
    SESSION_DELETE_BLOCKED = "session_delete_blocked"
    RESET = "reset"


@dataclass(frozen=True)
class IdentityEvent:
    event_id: int
    timestamp: float
    frame_sequence: int
    event_type: IdentityEventType
    global_id: int | None
    detail: str = ""
    camera_id: str = ""
    evidence: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ContractViolation("identity event without a timestamp")
        if self.frame_sequence is None:
            raise ContractViolation("identity event without a frame sequence")


@dataclass
class GlobalVehicleState:
    """PLAN 1 stage 8 output.  Lifecycle is independent of any local track."""

    global_id: int
    lifecycle_state: LifecycleState
    created_at: float
    last_observed_timestamp: float
    latest_world_position: np.ndarray
    latest_world_covariance: np.ndarray
    latest_camera: str = ""
    latest_footprint: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    #: The *measured* position of the last observation (PLAN 2 §3.4 uses the
    #: displacement between consecutive observations for the speed bound; the
    #: Kalman-filtered estimate lags the measurement and would demand the
    #: vehicle travel slower than v_max).
    last_observed_position: np.ndarray | None = None
    footprint_area: float = 0.0
    footprint_aspect: float = 1.0
    slot_id: str | None = None
    session_ids: tuple[str, ...] = ()
    observation_count: int = 0
    missing_since: float | None = None
    last_camera_seen_at: dict[str, float] = field(default_factory=dict)
    exit_pending_to: str | None = None
    quarantined: bool = False
    origin_world_position: np.ndarray | None = None
    max_displacement: float = 0.0
    kinematics: Any = field(default=None, repr=False, compare=False)
    appearance_gallery: Any = field(default=None, repr=False, compare=False)

    @property
    def uncertainty(self) -> float:
        from .linalg import positional_sigma
        return positional_sigma(self.latest_world_covariance)

    def missing_duration(self, now: float) -> float:
        return max(0.0, now - self.last_observed_timestamp)


@dataclass
class ParkingSlotState:
    """PLAN 1 stage 9 output."""

    slot_id: str
    occupancy_state: SlotOccupancy = SlotOccupancy.EMPTY
    owning_global_id: int | None = None
    overlap_score: float = 0.0
    dwell_duration: float = 0.0
    confirmation_confidence: float = 0.0
    last_update_timestamp: float = 0.0
    occupied_since: float | None = None
    releasing_since: float | None = None
    claim_global_id: int | None = None
    claim_started_at: float | None = None
    transit_events: int = 0
    #: Pixel-content occupancy verdict (stage 9 vision fusion channel),
    #: independent of tracking.  Occupied colour may survive on this alone.
    vision_occupied: bool = False
