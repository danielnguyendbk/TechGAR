"""Topology-constrained association tests — PLAN 1 stage 7.

Covers:
* topology gate removes invalid handoff candidates from the matrix (PLAN 2 §4.5);
* assignment margin defers ambiguous matches instead of guessing (§4.7);
* every accepted match satisfies the Re-ID acceptance rule (§6.2);
* the catastrophic-failure proofs of §4.8/§4.9 have regression tests:
  distance never dominates (identity swap under projection error) and
  direction is present but never hard-rejects a U-turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import Rig, one_hot

from techgar.config_world import AssociationConfig, IdentityConfig
from techgar.states import LifecycleState
from techgar.world_contracts import DecisionType


# ---------------------------------------------------------------------------
# Topology gating.
# ---------------------------------------------------------------------------

def test_cross_camera_requires_directed_corridor(rig: Rig):
    """cam1 identity in cam2's *unrelated* region → infeasible, not merely costly."""
    rig.drive(10.0, 15.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    views = rig.registry.views(0.10)
    observation = rig.observation(80.0, 55.0, 0.10, "cam2", velocity=(0.0, 0.0))
    outcome = rig.associator.associate(views, [observation])
    components = outcome.components[(views[0].global_id, observation.observation_id)]
    assert not components.feasible
    assert "topology" in components.reason


def test_cross_camera_in_corridor_is_feasible(rig: Rig):
    rig.drive(42.0, 15.0, timestamp=0.00, camera="cam1", velocity=(4.0, 0.0))
    # Position the identity's *prediction* near the exit, then present cam2
    # inside the entry corridor.
    views = rig.registry.views(0.30)
    observation = rig.observation(44.5, 15.0, 0.30, "cam2", velocity=(4.0, 0.0))
    outcome = rig.associator.associate(views, [observation])
    components = outcome.components[(views[0].global_id, observation.observation_id)]
    assert components.feasible, components.reason


def test_time_window_blocks_late_handoff(rig: Rig):
    rig.drive(42.0, 15.0, timestamp=0.00, camera="cam1", velocity=(4.0, 0.0))
    views = rig.registry.views(5.00)   # beyond dt_max = 4.0
    observation = rig.observation(44.5, 15.0, 5.00, "cam2", velocity=(4.0, 0.0))
    outcome = rig.associator.associate(views, [observation])
    components = outcome.components[(views[0].global_id, observation.observation_id)]
    assert not components.feasible


def test_speed_bound_blocks_teleport(rig: Rig):
    """v_max_world = 12 wu/s: 40 units in 0.5 s is infeasible for a NEW match."""
    rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    views = rig.registry.views(0.50)
    observation = rig.observation(50.0, 10.0, 0.50, "cam1", velocity=(0.0, 0.0))
    outcome = rig.associator.associate(views, [observation])
    components = outcome.components[(views[0].global_id, observation.observation_id)]
    assert not components.feasible
    assert components.reason == "speed_bound"


def test_topology_ablation_allows_global_match(topology):
    """Ablation D: with the topology switch off, the unrelated region matches.

    This is the *degraded* configuration of PLAN 3 §7 — the test proves the
    switch changes behaviour: with topology ON the pair is infeasible
    (topology gate), with topology OFF it becomes feasible (invalid handoff
    rate becomes possible — exactly the penalty the ablation study expects).
    Positions are kept within the speed bound so topology is the *only*
    differing gate.
    """
    from conftest import make_fast_rig

    # Identity in cam1; observation in cam2's unrelated region (not in any
    # cam2 entry corridor), but within v_max·dt + rho of the last position.
    # Instant-maturity rig keeps the identity where the test put it.
    rig = make_fast_rig(topology)
    rig.drive(30.0, 15.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    views = rig.registry.views(0.10)
    observation = rig.observation(31.0, 15.0, 0.10, "cam2", velocity=(0.0, 0.0))

    # Baseline: topology ON → infeasible by the topology gate.
    outcome_on = rig.associator.associate(views, [observation])
    components_on = outcome_on.components[(views[0].global_id, observation.observation_id)]
    assert not components_on.feasible
    assert "topology" in components_on.reason, components_on.reason

    # Degraded: topology OFF → the same pair becomes feasible (no gate).
    from techgar.config_world import AssociationConfig
    degraded = Rig(topology, association_config=AssociationConfig(enable_topology=False))
    degraded.drive(30.0, 15.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    views_off = degraded.registry.views(0.10)
    outcome_off = degraded.associator.associate(views_off, [observation])
    components_off = outcome_off.components[(views_off[0].global_id, observation.observation_id)]
    assert components_off.feasible, (
        f"topology ablation must remove the infeasibility, got {components_off.reason}")


# ---------------------------------------------------------------------------
# Margin rule (PLAN 2 §4.7).
# ---------------------------------------------------------------------------

def test_equidistant_ambiguous_pair_defers(rig: Rig):
    """Two identical appearances equidistant from the identity → defer, not guess."""
    appearance = one_hot(3)
    rig.drive(20.0, 10.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0),
              appearance=appearance)
    left = rig.observation(16.0, 10.0, 0.10, "cam1", velocity=(0.0, 0.0),
                           appearance=appearance)
    right = rig.observation(24.0, 10.0, 0.10, "cam1", velocity=(0.0, 0.0),
                            appearance=appearance)
    decisions, result = rig.step(left, right, timestamp=0.10)
    matched = [d for d in decisions.values()
               if d.decision_type in (DecisionType.CONTINUITY, DecisionType.HANDOFF,
                                      DecisionType.REACQUIRE)]
    # Either no match at all, or every match cleared the margin; the pair may
    # never both silently claim the identity.
    assert len(matched) <= 1
    deferred = [d for d in decisions.values() if d.decision_type is DecisionType.DEFER]
    if matched:
        assert all(d.margin >= rig.associator.config.margin_min for d in matched)


def test_clear_winner_matches(rig: Rig):
    appearance = one_hot(5)
    rig.drive(20.0, 10.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0),
              appearance=appearance)
    near = rig.observation(20.5, 10.0, 0.10, "cam1", velocity=(0.0, 0.0),
                           appearance=appearance)
    far = rig.observation(26.0, 10.0, 0.10, "cam1", velocity=(0.0, 0.0),
                          appearance=one_hot(9))
    decisions, _ = rig.step(near, far, timestamp=0.10)
    near_decision = decisions.get(near.observation_id)
    assert near_decision is not None
    assert near_decision.decision_type is DecisionType.CONTINUITY


# ---------------------------------------------------------------------------
# PLAN 2 §4.8 — distance must never dominate (identity-swap regression).
# ---------------------------------------------------------------------------

def test_projection_error_cannot_swap_two_identities(rig: Rig):
    """Two vehicles 30 units apart; a 25-unit projection error must not swap.

    Vehicle A at (100,100), B at (100,130).  With distance-only costs the
    swapped pairing A→B is cheaper; the six-component cost with appearance
    and direction must prevent it.
    """
    rig.drive(100.0, 100.0, timestamp=0.00, camera="cam1", velocity=(5.0, 0.0),
              appearance=one_hot(1))
    rig.drive(100.0, 130.0, timestamp=0.00, camera="cam2", velocity=(5.0, 0.0),
              appearance=one_hot(2))
    a_gid, b_gid = (s.global_id for s in sorted(rig.registry.live(),
                                                key=lambda s: s.global_id))
    # Next frame: both displaced by the same physical motion; the *projection*
    # jitters A to (125,100) — near B's corridor.
    a_next = rig.observation(125.0, 100.0, 0.10, "cam1", velocity=(5.0, 0.0),
                             appearance=one_hot(1))
    b_next = rig.observation(100.0, 130.0, 0.10, "cam2", velocity=(5.0, 0.0),
                             appearance=one_hot(2))
    decisions, _ = rig.step(a_next, b_next, timestamp=0.10)
    a_decision = decisions.get(a_next.observation_id)
    b_decision = decisions.get(b_next.observation_id)
    # Whichever observation matched, it must be its own identity (or defer);
    # an accepted cross-assignment A→B / B→A is the catastrophic failure.
    if a_decision is not None and a_decision.assigned_global_id is not None:
        assert a_decision.assigned_global_id == a_gid
    if b_decision is not None and b_decision.assigned_global_id is not None:
        assert b_decision.assigned_global_id == b_gid


# ---------------------------------------------------------------------------
# PLAN 2 §4.9 — direction is never removed, never hard-rejects.
# ---------------------------------------------------------------------------

def test_opposite_direction_is_penalised_but_u_turn_survives(rig: Rig):
    """A vehicle that legitimately reverses must stay matchable.

    Direction enters the cost weighted by reliability; a stationary-snapshot
    velocity below direction_min_speed zeroes the weight, so the U-turn cost
    comes from position/appearance, not from a hard direction rejection.
    """
    rig.drive(30.0, 20.0, timestamp=0.00, camera="cam1", velocity=(4.0, 0.0))
    # Velocity estimate decays: at 0.10 s the reliability may already be low.
    decision, _, _ = rig.drive(29.6, 20.0, timestamp=0.10, camera="cam1",
                               velocity=(-4.0, 0.0))
    # Either matched back (acceptable) or deferred with a recorded reason —
    # but never minted as a new identity.
    assert decision is None or decision.assigned_global_id is not None
    live = rig.registry.live()
    assert len(live) == 1


def test_direction_cost_downweighted_when_unreliable():
    from techgar.association import AssociationConfig
    from techgar.cost import IdentityView, direction_cost, direction_reliability
    from techgar.states import LifecycleState

    from conftest import make_observation

    config = AssociationConfig(direction_min_speed=0.30)
    identity = IdentityView(
        global_id=1, position=np.zeros(2), covariance=np.eye(2) * 0.01,
        velocity=np.array([0.05, 0.0]),          # below min speed → unreliable
        area=1.0, aspect=1.0, gallery=None, last_camera="cam1",
        last_position=np.zeros(2), last_timestamp=0.0,
        lifecycle=LifecycleState.ACTIVE)
    # A real observation (direction_reliability reads observation.velocity).
    observation = make_observation(0.2, 0.0, 0.10, "cam1", 99, velocity=(4.0, 0.0))
    assert direction_reliability(identity, observation, 0.1, config) == 0.0
    # A perpendicular vector: direction cost is strictly positive.
    assert direction_cost(np.array([0.05, 0.0]), np.array([0.0, 4.0])) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Re-ID acceptance rule (PLAN 2 §6.2).
# ---------------------------------------------------------------------------

def test_accepted_matches_clear_tau_accept(rig: Rig):
    rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    decision, _, _ = rig.drive(10.5, 10.0, timestamp=0.10, camera="cam1",
                               velocity=(0.0, 0.0))
    assert decision is not None
    assert decision.decision_type is DecisionType.CONTINUITY
    assert decision.identity_score >= rig.registry.config.tau_accept


def test_unmatched_observation_is_new_candidate(rig: Rig):
    rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    # A blob in a region no live identity can physically reach.
    decision, result, observation = rig.drive(95.0, 50.0, timestamp=0.10,
                                              camera="cam1")
    if decision is not None:
        assert decision.decision_type is DecisionType.NEW_CANDIDATE
        assert decision.assigned_global_id is None


def test_owned_local_track_is_deferred_not_reassigned_or_reminted(rig: Rig):
    """A Local ID binding is an invariant, not another tunable cost term."""
    _, result, first = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    gid = result.minted[0]
    assert rig.registry.owner_of_local_track("cam1", first.observation_id) == gid
    displaced = rig.observation(80.0, 50.0, 0.10, "cam1")
    displaced.local_track_ids = (("cam1", first.observation_id),)
    constraints = rig.registry.owner_constraints([displaced])
    outcome = rig.associator.associate(
        rig.registry.views(0.10), [displaced], owner_constraints=constraints
    )
    decision = outcome.decision_for(displaced.observation_id)
    assert decision.decision_type is DecisionType.DEFER
    assert decision.competing_global_ids == (gid,)


def test_registry_quarantines_silent_local_owner_transfer(rig: Rig):
    """The registry is a second safety boundary even if a caller omits constraints."""
    first = rig.observation(10.0, 10.0, 0.00, "cam1")
    second = rig.observation(30.0, 10.0, 0.00, "cam1")
    _, initial = rig.step(first, second, timestamp=0.00)
    assert len(initial.minted) == 2
    owner = rig.registry.owner_of_local_track("cam1", first.observation_id)
    displaced = rig.observation(30.1, 10.0, 0.10, "cam1")
    displaced.local_track_ids = (("cam1", first.observation_id),)
    # Deliberately omit owner constraints to exercise the registry backstop.
    outcome = rig.associator.associate(rig.registry.views(0.10), [displaced])
    ingest = rig.registry.ingest([displaced], outcome, 0.10, 1)
    assert displaced.observation_id in ingest.quarantined
    assert rig.registry.owner_of_local_track("cam1", first.observation_id) == owner


def test_recent_active_binding_blocks_same_camera_local_id_hop(rig: Rig):
    _, result, first = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    gid = result.minted[0]
    fragment = rig.observation(10.1, 10.0, 0.10, "cam1")
    forbidden = rig.registry.forbidden_binding_pairs([fragment], 0.10)
    outcome = rig.associator.associate(
        rig.registry.views(0.10), [fragment],
        owner_constraints=rig.registry.owner_constraints([fragment]),
        forbidden_pairs=forbidden,
    )
    ingest = rig.registry.ingest([fragment], outcome, 0.10, 1)
    assert ingest.matched == {}
    assert ingest.minted == []
    assert rig.registry.active_owner_of_local_track("cam1", first.observation_id) == gid


# ---------------------------------------------------------------------------
# Occlusion-group hooks (PLAN 2 §7).
# ---------------------------------------------------------------------------

def test_note_occlusion_moves_identity_to_occluded(rig: Rig):
    """An ACTIVE identity in an occlusion group becomes OCCLUDED + gallery frozen."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0))
    gid = rig.active_gid()
    last_observation = rig.registry.get(gid).last_observed_timestamp
    # The local-track id that owns the last match: look it up from the log.
    owner_keys = [key for key, owner in rig.registry._local_track_owner.items()
                  if owner == gid]
    assert owner_keys, "expected the identity to own at least one local track"
    camera_id, local_track_id = owner_keys[-1]
    touched = rig.registry.note_occlusion(camera_id, [local_track_id],
                                          timestamp=last_observation + 0.10,
                                          frame_sequence=10)
    assert gid in touched
    assert rig.registry.get(gid).lifecycle_state is LifecycleState.OCCLUDED
    # The gallery froze: merged blobs must not contaminate appearance.
    assert rig.registry.get(gid).appearance_gallery.frozen


def test_occlusion_blocker_prevents_mint_of_split_fragment(rig: Rig):
    """While an occlusion group is pending, a nearby new blob must not mint."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0))
    gid = rig.active_gid()
    last_observation = rig.registry.get(gid).last_observed_timestamp
    owner_keys = [key for key, owner in rig.registry._local_track_owner.items()
                  if owner == gid]
    camera_id, local_track_id = owner_keys[-1]
    rig.registry.note_occlusion(camera_id, [local_track_id],
                                timestamp=last_observation + 0.10, frame_sequence=10)
    decision, result, _ = rig.drive(11.0, 10.0, timestamp=last_observation + 0.20,
                                    camera="cam1")
    assert result.minted == []


def test_appearance_gallery_stays_frozen_during_occlusion(rig: Rig):
    appearance = one_hot(4)
    rig.drive_n([(10.0, 10.0), (10.5, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0), appearance=appearance)
    gid = rig.active_gid()
    last_observation = rig.registry.get(gid).last_observed_timestamp
    owner_keys = [key for key, owner in rig.registry._local_track_owner.items()
                  if owner == gid]
    camera_id, local_track_id = owner_keys[-1]
    rig.registry.note_occlusion(camera_id, [local_track_id],
                                timestamp=last_observation + 0.10, frame_sequence=10)
    gallery = rig.registry.get(gid).appearance_gallery
    samples_before = len(gallery.samples)
    # A merged observation arriving during the freeze cannot add samples.
    merged = rig.observation(10.7, 10.0, last_observation + 0.15, "cam1",
                             appearance=one_hot(20), latent=True)
    rig.step(merged, timestamp=last_observation + 0.15)
    assert len(gallery.samples) == samples_before
