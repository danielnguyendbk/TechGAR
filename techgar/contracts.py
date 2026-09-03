"""Phase 0 data contracts, camera-side half (PLAN 1 Phase 0 work items 2 & 6).

Hard rules encoded here — a record that breaks one cannot be constructed:

* every detection carries a timestamp **and** a camera id;
* every state has an explicit enum, never a bare string;
* nothing in this module knows about Global IDs (only the registry mints those).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ContractViolation(ValueError):
    """Raised when a record would break a Phase 0 data contract."""


class TopologyRegion(str, Enum):
    NORMAL = "normal"
    OVERLAP = "overlap"
    HANDOFF_EXIT = "handoff_exit_corridor"
    HANDOFF_ENTRY = "handoff_entry_corridor"


class LocalTrackState(str, Enum):
    """PLAN 1 stage 4: visible -> temporarily_missed -> occluded -> merged ->
    re-acquiring -> retired."""

    VISIBLE = "visible"
    TEMPORARILY_MISSED = "temporarily_missed"
    OCCLUDED = "occluded"
    MERGED = "merged"
    RE_ACQUIRING = "re_acquiring"
    RETIRED = "retired"


class MeasurementSource(str, Enum):
    DETECTION = "detection"          # tier-1/tier-2 detection association
    TEMPLATE = "template"            # PLAN 1 Phase 1.6 recovery measurement
    COAST = "coast"                  # prediction only, no measurement


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame plus its provenance (PLAN 1 stage 1, logic step 3)."""

    camera_id: str
    sequence: int
    timestamp: float
    width: int
    height: int
    decode_ok: bool = True
    image: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ContractViolation("frame without camera id")
        if not np.isfinite(self.timestamp):
            raise ContractViolation("frame without a finite timestamp")


@dataclass(frozen=True)
class SynchronizedFramePair:
    """Timestamp-paired frames (PLAN 1 stage 1 outputs)."""

    frames: dict[str, FrameRecord]
    timestamp_skew: float
    pair_sequence: int
    accepted: bool
    reject_reason: str = ""

    @property
    def timestamp(self) -> float:
        """Pair reference time = latest contributing capture timestamp."""
        return max(f.timestamp for f in self.frames.values())

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.frames))


@dataclass
class EnvironmentQuality:
    """PLAN 1 stage 2 output + PLAN 2 §1.4 threshold telemetry."""

    camera_id: str
    timestamp: float
    noise_sigma: float          # sigma^noise_t, MAD based
    luminance: float            # L_t
    luminance_delta: float      # |L_t - L_{t-1}|
    illumination_event: float   # E^illumination_t in [0, 1]
    changed_fraction: float
    threshold: float            # tau_t actually used
    unstable: bool = False      # PLAN 1 stage 2 logic 9

    @property
    def quality(self) -> float:
        """1.0 = pristine, 0.0 = unusable; feeds R_t in PLAN 2 §2.4."""
        penalty = min(1.0, 0.6 * self.illumination_event + 0.4 * min(1.0, self.changed_fraction * 4.0))
        return float(max(0.0, 1.0 - penalty))


@dataclass
class LocalDetection:
    """PLAN 1 stage 3 output.  ``footprint_pixels`` is the *ground contact* quad
    (bottom band of the blob), which is what may be projected by a planar
    homography without height-induced parallax."""

    camera_id: str
    timestamp: float
    frame_sequence: int
    bbox: np.ndarray                     # (x0, y0, x1, y1) pixels
    confidence: float
    local_center: np.ndarray             # (2,) pixels, blob centroid
    ground_anchor: np.ndarray            # (2,) pixels, bottom-centre contact point
    footprint_pixels: np.ndarray         # (4, 2) pixels
    mask_area: float
    quality_score: float
    occlusion_group_candidate: bool = False
    partial: bool = False          # blob touches the image border: incomplete evidence
    internal_motion_peaks: int = 1
    covered_predictions: tuple[int, ...] = ()
    appearance: np.ndarray | None = None
    detection_id: int = -1

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ContractViolation("detection without camera id")
        if not np.isfinite(self.timestamp):
            raise ContractViolation("detection without timestamp")
        self.bbox = np.asarray(self.bbox, dtype=float)
        self.local_center = np.asarray(self.local_center, dtype=float)
        self.ground_anchor = np.asarray(self.ground_anchor, dtype=float)
        self.footprint_pixels = np.asarray(self.footprint_pixels, dtype=float)


@dataclass
class LocalTrackObservation:
    """PLAN 1 stage 4 output — a *proposal*, never an identity."""

    local_track_id: int
    camera_id: str
    timestamp: float
    frame_sequence: int
    predicted_bbox: np.ndarray
    measured_bbox: np.ndarray | None
    state: LocalTrackState
    missed_duration: float
    motion_vector: np.ndarray
    appearance_reference: np.ndarray | None
    footprint_pixels: np.ndarray
    ground_anchor: np.ndarray
    confidence: float
    quality: float
    source: MeasurementSource = MeasurementSource.DETECTION
    occlusion_group_id: int | None = None
    latent: bool = False                  # inside an unresolved occlusion group
    covariance: np.ndarray | None = None  # pixel-space positional covariance
    extras: dict = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return self.measured_bbox is not None
