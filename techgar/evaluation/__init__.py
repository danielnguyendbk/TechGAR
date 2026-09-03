"""Evaluation harness: PLAN 3 metric suite, scenario suite, ablations, rubric.

The golden rule of PLAN 3 is enforced structurally here: nothing in this package
uses a predicted Global ID as ground truth.  Predictions are matched to physical
vehicles purely by world geometry, and only then are identity statistics computed
from that mapping.
"""

from .ablation import AblationOutcome, run_ablation_suite
from .harness import RunResult, run_recording
from .metrics_handoff import HandoffMetrics, handoff_metrics
from .metrics_mot import MotMetrics, mot_metrics
from .metrics_slot import SlotMetrics, slot_metrics
from .rubric import RubricScore, score_rubric
from .scenarios import SCENARIOS, ScenarioResult, run_all_scenarios, run_scenario
from .truth import TruthOracle

__all__ = [
    "AblationOutcome", "HandoffMetrics", "MotMetrics", "RubricScore", "RunResult", "SCENARIOS",
    "ScenarioResult", "SlotMetrics", "TruthOracle", "handoff_metrics", "mot_metrics",
    "run_ablation_suite", "run_all_scenarios", "run_recording", "run_scenario", "score_rubric",
    "slot_metrics",
]
