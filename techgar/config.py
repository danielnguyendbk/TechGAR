"""Single import site for configuration.

``from techgar.config import TechgarConfig`` gives the whole tuning surface;
components take only the sub-config they need so that nothing can silently
reach across layers.
"""

from __future__ import annotations

from .config_vision import (BackgroundConfig, DetectionConfig, IngestionConfig, KalmanConfig,
                            LocalTrackConfig, MotionConfig, ShadowConfig, ThresholdConfig)
from .config_world import (AblationFlags, AssociationConfig, FusionConfig, IdentityConfig,
                           PerfConfig, ProjectionConfig, SessionConfig, SlotConfig, TechgarConfig)

#: PLAN 3 §7 — the four ablation configurations.
ABLATIONS = {
    "full": AblationFlags(name="full"),
    "no_frame_difference": AblationFlags(name="no_frame_difference", frame_difference=False),
    "no_prediction": AblationFlags(name="no_prediction", prediction=False),
    "no_topology": AblationFlags(name="no_topology", topology=False),
}

__all__ = [
    "ABLATIONS", "AblationFlags", "AssociationConfig", "BackgroundConfig", "DetectionConfig",
    "FusionConfig", "IdentityConfig", "IngestionConfig", "KalmanConfig", "LocalTrackConfig",
    "MotionConfig", "PerfConfig", "ProjectionConfig", "SessionConfig", "ShadowConfig",
    "SlotConfig", "TechgarConfig", "ThresholdConfig",
]
