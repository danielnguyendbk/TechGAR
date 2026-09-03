"""Reproducible ablation runner for PLAN 3 §7."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ABLATIONS, TechgarConfig
from .scenarios import ScenarioResult, run_all_scenarios


@dataclass
class AblationOutcome:
    name: str
    scenarios: list[ScenarioResult]
    passed: int
    failed: int
    pass_rate: float
    degradation_from_full: float = 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "degradation_from_full": self.degradation_from_full,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
        }


def run_ablation_suite(base: TechgarConfig | None = None) -> list[AblationOutcome]:
    """Run the same A-I suite with one mechanism removed at a time."""
    root = base or TechgarConfig()
    outcomes = []
    for name, flags in ABLATIONS.items():
        config = root.apply_ablation(flags)
        scenarios = run_all_scenarios(config)
        passed = sum(result.passed for result in scenarios)
        outcomes.append(AblationOutcome(
            name=name,
            scenarios=scenarios,
            passed=passed,
            failed=len(scenarios) - passed,
            pass_rate=passed / max(len(scenarios), 1),
        ))
    baseline = next((outcome.pass_rate for outcome in outcomes if outcome.name == "full"), 0.0)
    for outcome in outcomes:
        outcome.degradation_from_full = baseline - outcome.pass_rate
    return outcomes

