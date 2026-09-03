"""Camera-local track record and its time-based state machine (PLAN 1 stage 4).

Every transition is a function of *elapsed seconds*, never of a frame count
(PLAN 2 §2.5): at 12 FPS three missed frames are 250 ms and must leave the track
recoverable, while the same three frames at 2 FPS are 1.5 s and must not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .appearance import AppearanceGallery
from .config_vision import LocalTrackConfig
from .contracts import LocalDetection, LocalTrackState, MeasurementSource
from .kalman import LagAwareKalman


@dataclass
class OcclusionGroup:
    """A detection covering several tracks — kept explicit, never resolved by force."""

    group_id: int
    camera_id: str
    detection_id: int
    track_ids: tuple[int, ...]
    bbox: np.ndarray
    started_at: float
    last_seen: float

    def duration(self, now: float) -> float:
        return max(0.0, now - self.started_at)


@dataclass
class LocalTrack:
    local_track_id: int
    camera_id: str
    filter: LagAwareKalman
    created_at: float
    last_observed: float
    state: LocalTrackState = LocalTrackState.VISIBLE
    observations: int = 1
    bbox: np.ndarray = field(default_factory=lambda: np.zeros(4))
    gallery: AppearanceGallery = field(default_factory=AppearanceGallery)
    template: np.ndarray | None = None
    last_detection: LocalDetection | None = None
    occlusion_group_id: int | None = None
    recoveries: int = 0
    blind_recoveries: int = 0
    latent: bool = False
    last_detection_at: float = 0.0
    partial_last: bool = False
    at_border: bool = False
    last_measurement_source: MeasurementSource = MeasurementSource.DETECTION

    def __post_init__(self) -> None:
        if self.last_detection_at <= 0.0:
            self.last_detection_at = self.last_observed

    # --- geometry -----------------------------------------------------------
    @property
    def center(self) -> np.ndarray:
        return self.filter.position

    def predicted_bbox(self) -> np.ndarray:
        cx, cy = self.filter.position
        w, h = self.filter.size
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    @property
    def area(self) -> float:
        w, h = self.filter.size
        return float(max(w, 1.0) * max(h, 1.0))

    def missed_duration(self, now: float) -> float:
        return max(0.0, now - self.last_observed)

    def detection_gap(self, now: float) -> float:
        """Time since the last *real* detection.

        Template recoveries deliberately do not reset this: a match against static
        scenery must never be able to keep a departed vehicle alive for ever
        (that is how a ghost track is born).
        """
        return max(0.0, now - self.last_detection_at)

    # --- lifecycle ----------------------------------------------------------
    def mark_observed(self, timestamp: float, bbox, detection: LocalDetection | None = None
                      ) -> None:
        self.last_observed = float(timestamp)
        self.last_detection_at = float(timestamp)
        self.blind_recoveries = 0
        self.bbox = np.asarray(bbox, dtype=float)
        self.observations += 1
        self.state = LocalTrackState.VISIBLE
        self.last_measurement_source = MeasurementSource.DETECTION
        self.latent = False
        self.occlusion_group_id = None
        if detection is not None:
            self.last_detection = detection
            self.partial_last = bool(detection.partial)

    def mark_merged(self, group_id: int, timestamp: float) -> None:
        """PLAN 2 §7: coast on prediction and freeze the gallery."""
        self.state = LocalTrackState.MERGED
        self.occlusion_group_id = group_id
        self.latent = True
        self.gallery.freeze("occlusion_group")
        self.bbox = self.predicted_bbox()
        del timestamp

    def refresh_state(self, now: float, config: LocalTrackConfig) -> LocalTrackState:
        """Time-based transition, applied when no measurement arrived."""
        gap = self.detection_gap(now)
        retire_after = config.t_retire_border if (self.at_border or self.partial_last) \
            else config.t_retire
        if gap > retire_after:
            self.state = LocalTrackState.RETIRED
            return self.state
        if self.state == LocalTrackState.MERGED:
            return self.state
        missed = self.missed_duration(now)
        if missed <= config.t_missed:
            self.state = LocalTrackState.VISIBLE
        elif missed <= config.t_occluded:
            self.state = LocalTrackState.TEMPORARILY_MISSED
        elif missed <= config.t_reacquire:
            self.state = LocalTrackState.OCCLUDED
        elif missed <= config.t_retire:
            self.state = LocalTrackState.RE_ACQUIRING
        else:
            self.state = LocalTrackState.RETIRED
        return self.state

    @property
    def alive(self) -> bool:
        return self.state != LocalTrackState.RETIRED

    def recoverable(self, now: float, config: LocalTrackConfig) -> bool:
        """A live re-acquisition hypothesis blocks new-track creation (stage 4 logic 8)."""
        return self.alive and self.detection_gap(now) <= config.t_retire

    def recovery_allowed(self, now: float, config: LocalTrackConfig) -> bool:
        """Template recovery is bounded in time, in count and away from the border."""
        if self.template is None or self.at_border or self.partial_last:
            return False
        if self.blind_recoveries >= config.max_blind_recoveries:
            return False
        return self.detection_gap(now) <= config.t_reacquire
