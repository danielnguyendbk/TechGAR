"""Support-module tests for stage 7/8: cost components, assignment, events."""

from __future__ import annotations

import numpy as np
import pytest

from techgar.assignment import greedy_assignment, solve_assignment
from techgar.cost import direction_cost, geometry_cost, time_cost
from techgar.identity_events import IdentityEventLog
from techgar.states import IdentityEventType


# ---------------------------------------------------------------------------
# Assignment solver (PLAN 2 §4: global one-to-one, forbidden pairs excluded).
# ---------------------------------------------------------------------------

def test_assignment_optimal_not_greedy():
    """The canonical case where greedy is sub-optimal."""
    cost = np.array([
        [1.0, 2.0],
        [0.5, 100.0],
    ])
    optimal = solve_assignment(cost)
    # Optimal total: 2.0 + 0.5 = 2.5 (row 1 takes col 0, row 0 takes col 1).
    assert sorted(optimal) == [(0, 1), (1, 0)]
    greedy = greedy_assignment(cost)
    # Greedy grabs (1,0)=0.5 then (0,1)=2.0 → same here; use a case where they differ.
    cost2 = np.array([
        [1.0, 1.1],
        [1.0, 100.0],
    ])
    assert sorted(solve_assignment(cost2)) == [(0, 1), (1, 0)]


def test_assignment_excludes_forbidden_pairs():
    cost = np.array([
        [1.0, np.inf],
        [np.inf, 2.0],
    ])
    pairs = solve_assignment(cost)
    assert pairs == [(0, 0), (1, 1)]


def test_assignment_never_uses_forbidden_even_when_cheap():
    cost = np.array([
        [0.1, np.inf],
        [0.2, 0.3],
    ])
    pairs = solve_assignment(cost)
    assert (0, 0) in pairs and (1, 1) in pairs


def test_assignment_all_forbidden_is_empty():
    cost = np.full((2, 2), np.inf)
    assert solve_assignment(cost) == []


def test_assignment_rectangular_more_columns():
    cost = np.array([[5.0, 1.0, 4.0]])
    assert solve_assignment(cost) == [(0, 1)]


def test_assignment_rectangular_more_rows():
    cost = np.array([[5.0], [1.0], [4.0]])
    assert solve_assignment(cost) == [(1, 0)]


# ---------------------------------------------------------------------------
# Cost components (PLAN 2 §4).
# ---------------------------------------------------------------------------

def test_direction_cost_parallel_is_zero_opposite_is_one():
    assert direction_cost(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(0.0)
    assert direction_cost(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(1.0)


def test_direction_cost_unknown_is_half():
    assert direction_cost(None, np.array([1.0, 0.0])) == 0.5
    assert direction_cost(np.zeros(2), np.array([1.0, 0.0])) == 0.5


def test_geometry_cost_symmetric_and_zero_for_equal():
    assert geometry_cost(4.0, 4.0, 1.0, 1.0, 0.5) == 0.0
    value = geometry_cost(8.0, 4.0, 2.0, 1.0, 0.5)
    assert value == pytest.approx(np.log(2.0) + 0.5 * np.log(2.0))


def test_time_cost_hard_window():
    assert not np.isfinite(time_cost(-0.1, 0.5, 0.0, 4.0, soft=True))
    assert not np.isfinite(time_cost(5.0, 0.5, 0.0, 4.0, soft=True))
    assert time_cost(0.5, 0.5, 0.0, 4.0, soft=True) == 0.0
    assert time_cost(1.0, 0.5, 0.0, 4.0, soft=True) == pytest.approx(np.log(2.0))
    assert time_cost(2.0, 0.5, 0.0, 4.0, soft=False) == 0.0


# ---------------------------------------------------------------------------
# Append-only event log (rubric A).
# ---------------------------------------------------------------------------

def test_event_log_is_append_only():
    log = IdentityEventLog()
    log.append(0.0, 1, IdentityEventType.MINT, 1)
    log.append(0.1, 2, IdentityEventType.MATCH, 1, detail="t")
    with pytest.raises(AttributeError):
        log.pop()  # type: ignore[attr-defined]
    assert len(log) == 2


def test_event_log_requires_timestamp_and_frame():
    log = IdentityEventLog()
    from techgar.contracts import ContractViolation
    with pytest.raises(ContractViolation):
        log.append(float("nan"), 1, IdentityEventType.MINT, 1)
    with pytest.raises(ContractViolation):
        log.append(0.0, None, IdentityEventType.MINT, 1)  # type: ignore[arg-type]


def test_event_log_queries():
    log = IdentityEventLog()
    log.append(0.0, 1, IdentityEventType.MINT, 1)
    log.append(0.5, 5, IdentityEventType.HANDOFF, 1)
    log.append(0.6, 6, IdentityEventType.RETIRE, 2)
    assert len(log.for_global_id(1)) == 2
    assert log.count_of(IdentityEventType.HANDOFF) == 1
    assert len(log.since(0.55)) == 1
    assert log.tail(1)[0].event_type is IdentityEventType.RETIRE
