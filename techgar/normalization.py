"""Stage 2 — environmental normalisation and foreground evidence.

Composes PLAN 2 §1: adaptive threshold map, background evidence, dual-stage AND
gate, shadow rejection, speckle suppression, and the environmental-instability
verdict that PLAN 1 stage 2 demands instead of a frame full of vehicle-shaped
detections when the lighting flickers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .background import BackgroundModel
from .config_vision import (BackgroundConfig, MotionConfig, ShadowConfig, ThresholdConfig)
from .contracts import EnvironmentQuality, FrameRecord
from .motion import DifferenceResult, TemporalDifferenceEvidence, dual_stage_gate
from .shadow import shadow_mask
from .threshold import AdaptiveThresholdEngine, luminance

STRUCTURE = np.ones((3, 3), dtype=bool)


@dataclass
class NormalizedFrame:
    camera_id: str
    timestamp: float
    frame_sequence: int
    gray: np.ndarray
    color: np.ndarray | None
    foreground: np.ndarray
    shadow: np.ndarray
    candidates: np.ndarray          # low-confidence evidence kept, not deleted
    background: np.ndarray
    tau_map: np.ndarray
    quality: EnvironmentQuality
    difference: DifferenceResult | None = None


class EnvironmentalNormalizer:
    """One per camera (PLAN 1 stage 2 takes per-camera configuration)."""

    def __init__(self, camera_id: str, threshold: ThresholdConfig | None = None,
                 background: BackgroundConfig | None = None, motion: MotionConfig | None = None,
                 shadow: ShadowConfig | None = None) -> None:
        self.camera_id = camera_id
        self.motion_config = motion or MotionConfig()
        self.shadow_config = shadow or ShadowConfig()
        self.threshold_engine = AdaptiveThresholdEngine(threshold)
        self.background_model = BackgroundModel(background)
        self.difference_engine = TemporalDifferenceEvidence(self.motion_config)

    def reset(self) -> None:
        self.threshold_engine.reset()
        self.background_model.reset()
        self.difference_engine.reset()

    def process(self, frame: FrameRecord) -> NormalizedFrame:
        cfg = self.motion_config
        gray = luminance(frame.image)
        color = (np.asarray(frame.image, dtype=np.float32) if frame.image is not None
                 and np.ndim(frame.image) == 3 else None)
        threshold = self.threshold_engine.update(gray)
        background_mask = self.background_model.evidence(gray, threshold.tau_map)
        difference = self.difference_engine.evidence(gray, frame.timestamp, threshold.tau_map)
        self.difference_engine.push(frame.timestamp, gray)

        gated = dual_stage_gate(background_mask, difference, cfg.enable_frame_difference)
        unstable = threshold.changed_fraction > cfg.instability_fraction
        if not self.background_model.ready:
            gated = np.zeros_like(gated)

        shadow_result = shadow_mask(gray, self.background_model.gray if self.background_model.gray
                                    is not None else gray, color, self.background_model.color,
                                    gated, self.shadow_config)
        foreground = gated & ~shadow_result.shadow

        strong = ndi.binary_opening(foreground, structure=STRUCTURE,
                                    iterations=max(1, cfg.open_iterations))
        if unstable:
            # A whole-frame change is an environment event, not a vehicle: demand
            # morphological support before anything reaches the detector.
            strong = ndi.binary_opening(strong, structure=STRUCTURE, iterations=2)
        candidates = (gated | shadow_result.shadow) & ~strong

        self.background_model.update(gray, foreground=gated, color=color)
        quality = EnvironmentQuality(
            camera_id=frame.camera_id, timestamp=frame.timestamp,
            noise_sigma=threshold.noise_sigma, luminance=threshold.luminance,
            luminance_delta=threshold.luminance_delta,
            illumination_event=threshold.illumination_event,
            changed_fraction=threshold.changed_fraction, threshold=threshold.tau_global,
            unstable=unstable)
        return NormalizedFrame(
            camera_id=frame.camera_id, timestamp=frame.timestamp, frame_sequence=frame.sequence,
            gray=gray, color=color, foreground=strong, shadow=shadow_result.shadow,
            candidates=candidates, background=self.background_model.gray, tau_map=threshold.tau_map,
            quality=quality, difference=difference)
