"""Executable PLAN 3 scenario, ablation and rubric gates."""

from techgar.evaluation import (
    run_ablation_suite,
    run_all_scenarios,
    score_rubric,
)
from techgar.evaluation.cli import _pixel_parking_report
from techgar.evaluation.environment import run_environmental_checks


def test_all_mandatory_scenarios_pass():
    results = run_all_scenarios()
    assert [result.key for result in results] == list("ABCDEFGHI")
    assert all(result.passed for result in results), {
        result.key: result.checks for result in results if not result.passed
    }


def test_scenario_only_rubric_cannot_certify_production():
    score = score_rubric(run_all_scenarios())
    assert score.total == 89.0
    assert not score.accepted
    assert not score.gates["slot_f1_at_least_0_97"]
    assert not score.gates["mean_throughput_at_least_10_fps"]


def test_ablation_suite_uses_same_nine_scenarios():
    outcomes = run_ablation_suite()
    assert {outcome.name for outcome in outcomes} == {
        "full", "no_frame_difference", "no_prediction", "no_topology"
    }
    assert all(len(outcome.scenarios) == 9 for outcome in outcomes)
    assert next(item for item in outcomes if item.name == "full").passed == 9


def test_positive_pixel_parking_closes_synthetic_acceptance_gates():
    """Pixels must preserve one ID and reach the measured slot hard gates."""
    _, commissioning, run = _pixel_parking_report()
    assert commissioning.passed
    assert run.mot.id_switches == 0
    assert run.mot.max_fragmentation == 1
    assert run.slots.f1 >= 0.97
    assert run.slots.ownership_accuracy >= 0.98
    assert run.slots.one_id_two_slots == 0
    assert run.slots.two_ids_one_slot == 0

    rubric = score_rubric(
        run_all_scenarios(),
        run=run,
        environmental=run_environmental_checks(),
    )
    assert rubric.accepted, rubric.as_dict()
