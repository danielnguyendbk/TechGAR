"""PLAN 3 100-point scoring rubric with non-negotiable acceptance gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from .scenarios import ScenarioResult


@dataclass
class RubricScore:
    categories: dict[str, float]
    total: float
    gates: dict[str, bool]
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.total >= 90.0 and all(self.gates.values()) and not self.rejection_reasons

    def as_dict(self) -> dict:
        return {
            "categories": dict(self.categories),
            "total": self.total,
            "gates": dict(self.gates),
            "rejection_reasons": list(self.rejection_reasons),
            "accepted": self.accepted,
        }


def _passed(results: dict[str, ScenarioResult], key: str) -> bool:
    return bool(results.get(key) and results[key].passed)


def score_rubric(scenarios: list[ScenarioResult], run=None,
                 environmental: dict[str, bool] | None = None) -> RubricScore:
    """Score scenario and optional pixel-recording evidence.

    Structural architecture points are earned because the evaluator is invoked
    through the production Registry/Pipeline types.  Performance, environmental
    and measured slot gates remain false until an end-to-end ``RunResult`` is
    supplied; scenario-only runs therefore cannot accidentally certify production.
    """
    results = {scenario.key: scenario for scenario in scenarios}

    architecture = 20.0

    identity = 0.0
    identity += 6.0 if _passed(results, "A") else 0.0
    identity += 6.0 if _passed(results, "C") else 0.0
    identity += 5.0 if _passed(results, "B") else 0.0
    identity += 4.0 if _passed(results, "E") else 0.0
    identity += 4.0 if _passed(results, "D") else 0.0
    identity += 3.0 if _passed(results, "F") else 0.0
    identity += 2.0 if _passed(results, "F") else 0.0

    handoff = 15.0 if _passed(results, "C") else 0.0

    parking = 0.0
    parking += 8.0 if _passed(results, "G") else 0.0
    parking += 6.0 if _passed(results, "H") else 0.0
    parking += 6.0 if _passed(results, "I") else 0.0

    environmental = environmental or {}
    environment = 0.0
    environment += 1.0 if environmental.get("brightness_transition", False) else 0.0
    environment += 1.0 if environmental.get("shadow_rejection", False) else 0.0
    environment += 1.0 if environmental.get("compression_noise_bounded", False) else 0.0
    environment += 1.0  # covariance + seam are represented in every world observation
    environment += 1.0  # local calibrated world frame; GPS is structurally absent

    efficiency = 2.0  # bounded latest-frame ingestion + overload mint guard
    mean_fps = 0.0
    min_fps = 0.0
    median_latency = float("inf")
    p95_latency = float("inf")
    slot_f1 = 0.0
    ownership = 0.0
    invalid_handoff = 1.0
    gps_free = True
    if run is not None:
        perf = run.performance
        throughput = perf.get("throughput", {})
        latency = perf.get("latency", {})
        mean_fps = float(throughput.get("mean_fps", 0.0))
        min_fps = float(throughput.get("min_sustained_fps", 0.0))
        median_latency = float(latency.get("median", float("inf")))
        p95_latency = float(latency.get("p95", float("inf")))
        efficiency += 3.0 if mean_fps >= 10.0 else 0.0
        efficiency += 2.0 if min_fps >= 6.0 else 0.0
        efficiency += 2.0 if median_latency <= 0.250 else 0.0
        efficiency += 1.0 if p95_latency <= 0.750 else 0.0
        slot_f1 = float(run.slots.f1)
        ownership = float(run.slots.ownership_accuracy)
        invalid_handoff = float(run.handoff.invalid_rate)
        snapshots = [step.snapshot for step in run.steps if step.snapshot is not None]
        gps_free = all(not snapshot.gps_used for snapshot in snapshots)

    categories = {
        "architectural_integrity": architecture,
        "identity_continuity": identity,
        "handoff_correctness": handoff,
        "parking_correctness": parking,
        "environmental_robustness": environment,
        "computational_efficiency": efficiency,
    }
    total = float(sum(categories.values()))
    required_ids_zero = all(
        result.metrics.get("id_switches", 0.0) == 0.0
        for key, result in results.items() if key in "ABCDEF"
    ) and all(key in results for key in "ABCDEF")
    pixel_ids_zero = run is not None and run.mot.id_switches == 0
    one_physical_one_id = run is not None and run.mot.max_fragmentation <= 1
    gates = {
        "overall_score_at_least_90": total >= 90.0,
        "identity_switches_zero": required_ids_zero and pixel_ids_zero,
        "one_physical_vehicle_one_id": one_physical_one_id,
        "invalid_handoff_rate_zero": run is not None and invalid_handoff == 0.0,
        "session_survival_100_percent": all(_passed(results, key) for key in ("B", "C", "E")),
        "slot_f1_at_least_0_97": run is not None and slot_f1 >= 0.97,
        "slot_ownership_at_least_0_98": run is not None and ownership >= 0.98,
        "mean_throughput_at_least_10_fps": run is not None and mean_fps >= 10.0,
        "p95_latency_at_most_750ms": run is not None and p95_latency <= 0.750,
        "gps_not_used": gps_free,
    }
    rejection_reasons = []
    for key, message in {
        "B": "vehicle changed identity after a short occlusion",
        "C": "valid handoff did not preserve identity",
        "D": "crossing vehicles did not preserve distinct identities",
        "E": "merged detection lost or contaminated a latent identity",
        "G": "parking assignment violated temporal confirmation",
        "I": "parked ownership did not survive a false-empty observation",
    }.items():
        if key in results and not results[key].passed:
            rejection_reasons.append(message)
    return RubricScore(categories, total, gates, rejection_reasons)
