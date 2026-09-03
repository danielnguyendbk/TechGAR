"""End-to-end recording runner and metric aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import TechgarConfig
from ..pipeline import TechgarPipeline
from ..simulation.layouts import build_profiles
from .matching import dominant_mapping, match_timeline, snapshots_to_predictions
from .metrics_handoff import HandoffMetrics, handoff_metrics
from .metrics_mot import MotMetrics, mot_metrics
from .metrics_slot import SlotMetrics, slot_metrics
from .truth import TruthOracle


@dataclass
class RunResult:
    recording: object
    pipeline: TechgarPipeline
    steps: list
    oracle: TruthOracle
    frame_matches: list
    identity_mapping: dict[str, int]
    mot: MotMetrics
    slots: SlotMetrics
    slots_strict: SlotMetrics
    handoff: HandoffMetrics
    performance: dict

    def as_dict(self) -> dict:
        return {
            "recording": self.recording.name,
            "frames": len(self.steps),
            "identity_mapping": dict(self.identity_mapping),
            "mot": self.mot.as_dict(),
            "slots": self.slots.as_dict(),
            "slots_strict": self.slots_strict.as_dict(),
            "handoff": self.handoff.as_dict(),
            "performance": self.performance,
        }


def run_recording(recording, config: TechgarConfig | None = None,
                  max_match_distance: float = 3.0) -> RunResult:
    """Run pixels through stages 1-10 and score against isolated ground truth."""
    cfg = config or TechgarConfig()
    cfg.identity.v_max_world = float(recording.layout.v_max)
    profiles = build_profiles(recording.layout)
    pipeline = TechgarPipeline(profiles, recording.layout.topology,
                               recording.layout.slots, cfg)
    steps = list(pipeline.run(recording.iter_frames()))
    oracle = TruthOracle(recording)
    prediction_frames = snapshots_to_predictions(steps)
    frames = match_timeline(oracle, prediction_frames, max_match_distance)
    mapping = dominant_mapping(frames)
    return RunResult(
        recording=recording,
        pipeline=pipeline,
        steps=steps,
        oracle=oracle,
        frame_matches=frames,
        identity_mapping=mapping,
        mot=mot_metrics(frames),
        slots=slot_metrics(oracle, steps, mapping, strict=False),
        slots_strict=slot_metrics(oracle, steps, mapping, strict=True),
        handoff=handoff_metrics(oracle, frames, prediction_frames),
        performance=pipeline.performance_report(),
    )

