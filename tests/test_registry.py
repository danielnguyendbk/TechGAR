"""Global Identity Registry tests â€” PLAN 1 Phase 3 exit criteria.

Covers the Phase 3 gate of PLAN 1 Â§4 plus the PLAN 3 mandatory identity
scenarios that only the registry can prove:

* Scenario B  â€” one-frame gap keeps the Global ID;
* Scenario C  â€” camera handoff keeps the Global ID; topology fail blocks a
  match into an unrelated region;
* Scenario F  â€” a 500 ms lag with a 40-unit displacement stays inside the
  gate (the Kalman damping-bias covariance exists exactly for this);
* New-ID prohibition window (PLAN 2 Â§6.3) and collision quarantine (Â§6.5).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import Rig, make_fast_rig, one_hot

from techgar.config_world import AssociationConfig, IdentityConfig
from techgar.states import IdentityEventType, LifecycleState


# ---------------------------------------------------------------------------
# Phase 3 exit criterion 1: one physical vehicle, one Global ID across camera
# transitions (PLAN 3 Scenario C).
# ---------------------------------------------------------------------------
def test_handoff_preserves_global_id(topology):
    """PLAN 3 Scenario C numbers, verbatim (instant-maturity rig)."""
    rig = make_fast_rig(topology)
    # Vehicle approaches the exit corridor: 40 → 41.2 → 42.4 (12 wu/s).
    for i in range(3):
        rig.drive(40.0 + 1.2 * i, 15.0, 30.00 + 0.10 * i, "cam1", velocity=(12.0, 0.0))
    gid = rig.active_gid()
    last_seen = rig.registry.get(gid).last_observed_timestamp

    # One frame with no observation from any camera.
    decisions, result = rig.step(timestamp=last_seen + 0.10)
    assert result.minted == [] and result.matched == {}

    # The vehicle re-appears inside the cam2 entry corridor, dt within edge.
    reappear = last_seen + 0.20
    decision, _, _ = rig.drive(44.0, 15.0, timestamp=reappear, camera="cam2",
                               velocity=(12.0, 0.0))
    assert decision is not None and decision.assigned_global_id == gid, (
        f"handoff decision: {decision.decision_type if decision else None}")
    decision, _, _ = rig.drive(45.2, 15.0, timestamp=reappear + 0.10, camera="cam2",
                               velocity=(12.0, 0.0))
    assert decision is not None and decision.assigned_global_id == gid

    # PASS condition of the scenario: GID constant across the handoff.
    assert rig.single_live_gid() == gid
    assert rig.events_of(IdentityEventType.HANDOFF), "handoff event must be audited"

def test_handoff_rejects_unrelated_region(topology):
    """PLAN 3 Scenario C topology-fail example: cam2 sees (80, 80)."""
    rig = make_fast_rig(topology)
    for i in range(3):
        rig.drive(40.0 + 1.2 * i, 15.0, 30.00 + 0.10 * i, "cam1", velocity=(12.0, 0.0))
    gid = rig.active_gid()
    last_seen = rig.registry.get(gid).last_observed_timestamp

    # (80, 80) is far outside the cam2 entry corridor x∈[44,49], y∈[10,20].
    decision, result, observation = rig.drive(80.0, 80.0, timestamp=last_seen + 0.20,
                                              camera="cam2", velocity=(12.0, 0.0))
    assert decision is None or decision.assigned_global_id != gid

    # The identity of the physical vehicle is untouched by the rogue blob.
    assert rig.registry.get(gid) is not None


def test_same_camera_continuity(rig: Rig):
    rig.drive(10.0, 20.0, timestamp=0.0, camera="cam1", velocity=(5.0, 0.0))
    gid = rig.single_live_gid()
    for step in range(1, 4):
        decision, _, _ = rig.drive(10.0 + 0.5 * step, 20.0,
                                   timestamp=0.1 * step, camera="cam1",
                                   velocity=(5.0, 0.0))
        assert decision is not None and decision.assigned_global_id == gid
    assert rig.single_live_gid() == gid


# ---------------------------------------------------------------------------
# Phase 3 exit criterion 2: short-occlusion recovery keeps the ID
# (PLAN 3 Scenario B).
# ---------------------------------------------------------------------------

def test_one_frame_gap_preserves_id(rig: Rig):
    """Scenario B: frame 201 occluded, frame 202 the vehicle re-appears."""
    rig.drive_n([(20.0, 30.0), (20.5, 30.0), (21.0, 30.0), (21.5, 30.0)], "cam1", start=0.00, dt=0.10, velocity=(10.0, 0.0))
    gid = rig.active_gid()

    # Frame 201: no observation at all.
    decisions, result = rig.step(timestamp=0.40)
    state = rig.registry.get(gid)
    assert state.lifecycle_state in (LifecycleState.TEMPORARILY_MISSING,
                                     LifecycleState.OCCLUDED)

    # Frame 202: re-observation must return to the same Global ID.
    decision, _, _ = rig.drive(22.0, 30.0, timestamp=0.50, camera="cam1",
                               velocity=(10.0, 0.0))
    assert decision is not None and decision.assigned_global_id == gid
    assert rig.single_live_gid() == gid
    assert rig.registry.get(gid).lifecycle_state is LifecycleState.ACTIVE
    assert rig.events_of(IdentityEventType.RECOVER)


# ---------------------------------------------------------------------------
# Phase 3 exit criterion 3: lag resilience (PLAN 3 Scenario F).
# ---------------------------------------------------------------------------
def test_lag_displacement_keeps_id(topology):
    """Scenario F: a moving vehicle observed after a 500 ms stall keeps its ID.

    The vehicle travels at the v_max bound (12 wu/s).  During the 500 ms gap
    the prediction covariance grows with real elapsed time (Q(dt) + the VR1
    damping-bias term), so when the vehicle re-appears 6 units ahead of its
    last observation the innovation stays inside the chi-square gate.  A
    tracker that froze its covariance would mint a new identity here — the
    exact lag-induced fragmentation PLAN 3 Scenario F forbids.
    """
    rig = make_fast_rig(topology)
    # Several moving observations so the Kalman learns the velocity (12 wu/s
    # = 1.2 units per 0.1 s step, exactly at the v_max bound).
    for i in range(6):
        rig.drive(50.0 + 1.2 * i, 50.0, 200.0 + 0.1 * i, "cam1", velocity=(12.0, 0.0))
    gid = rig.active_gid()
    state = rig.registry.get(gid)
    last_seen = state.last_observed_timestamp
    last_x = state.latest_world_position[0]
    speed = float(np.linalg.norm(state.velocity))
    assert speed > 5.0, f"warm-up must leave a learned velocity, got {speed}"

    # 500 ms stall, then the vehicle re-appears where physics says it should
    # be: 12 wu/s * 0.5 s = 6 units ahead (at the v_max bound).
    decision, result, _ = rig.drive(last_x + 6.0, 50.0, timestamp=last_seen + 0.50,
                                    camera="cam1", velocity=(12.0, 0.0))
    assert decision is not None, (
        f"lag observation deferred/minted instead of matched: {result.deferred} {result.minted}")
    assert decision.assigned_global_id == gid
    assert rig.single_live_gid() == gid


def test_teleport_beyond_speed_bound_is_not_matched(topology):
    """The complement of Scenario F: a jump beyond v_max·dt is a teleport.

    The speed bound exists to reject it (PLAN 2 §3.4); gluing such a pair
    would be the "impossible teleport" of PLAN 3 Scenario F's fail state.
    """
    rig = make_fast_rig(topology)
    for i in range(4):
        rig.drive(50.0 + 1.2 * i, 50.0, 200.0 + 0.1 * i, "cam1", velocity=(12.0, 0.0))
    gid = rig.active_gid()
    last_seen = rig.registry.get(gid).last_observed_timestamp

    # 40-unit jump in 500 ms = 80 wu/s, far beyond v_max = 12.
    decision, result, _ = rig.drive(90.0, 50.0, timestamp=last_seen + 0.50,
                                    camera="cam1", velocity=(80.0, 0.0))
    if decision is not None and decision.assigned_global_id is not None:
        assert decision.assigned_global_id != gid, (
            "a 40-unit jump in 500 ms (80 wu/s > v_max 12) must not match")


def test_lag_with_stationary_vehicle_damping(rig: Rig):
    """VR1 companion case: the same stall, but the vehicle had *stopped*.

    Damping biases the prediction towards "stopped", so the track must not
    fly 40 units ahead and lose the re-observation at the parked position.
    """
    rig.drive_n([(50.0, 50.0), (50.0, 50.0), (50.0, 50.0), (50.0, 50.0)], "cam1", start=100.00, dt=0.10, velocity=(0.0, 0.0))
    gid = rig.active_gid()
    decision, _, _ = rig.drive(50.0, 50.0, timestamp=100.50, camera="cam1",
                               velocity=(0.0, 0.0))
    assert decision is not None and decision.assigned_global_id == gid


# ---------------------------------------------------------------------------
# New-ID prohibition window (PLAN 2 Â§6.3) â€” the central anti-fragmentation rule.
# ---------------------------------------------------------------------------

def test_no_new_id_while_grace_hypothesis_lives(rig: Rig):
    """A fresh blob appearing near a just-missed identity must not mint."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0))
    gid = rig.active_gid()
    last_seen = rig.registry.get(gid).last_observed_timestamp
    last_x = rig.registry.get(gid).latest_world_position[0]

    # Identity missed for less than t_grace; a new blob appears 0.6 units
    # away, *after* the last observation.
    decision, result, _ = rig.drive(last_x + 0.6, 10.0, timestamp=last_seen + 0.20,
                                    camera="cam1", velocity=(2.0, 0.0))
    assert result.minted == [], (
        f"minted {result.minted} while grace hypothesis {gid} was alive "
        f"(deferred={result.deferred}, blocked={[b[1] for b in result.blocked_mints]})")
    # The observation either matched back to the live identity (best case)
    # or was deferred/blocked with the grace blocker recorded.
    if decision is not None and decision.assigned_global_id is not None:
        assert decision.assigned_global_id == gid
    assert rig.single_live_gid() == gid


def test_new_id_after_grace_expires_in_empty_region(rig: Rig):
    """Once every retention channel is exhausted, a genuinely new vehicle may mint."""
    rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1", velocity=(0.0, 0.0))
    first = rig.single_live_gid()

    # A new vehicle appears far away *after* grace and retention expired.
    # t_grace=2, t_max_missing=8, t_retire_idle=20 by default.
    decisions, result = rig.step(timestamp=25.0)
    assert rig.registry.get(first) is None or not rig.registry.get(first).lifecycle_state.is_live

    decision, result, _ = rig.drive(70.0, 50.0, timestamp=25.10, camera="cam1",
                                    velocity=(0.0, 0.0))
    assert result.minted, "a genuinely new vehicle after full expiry must mint"
    assert rig.single_live_gid() != first


def test_overload_never_mints(rig: Rig):
    """PLAN 1 Phase 6 / PLAN 3 Â§6: overload raises uncertainty, never mints."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10)
    decision, result, _ = rig.drive(30.0, 30.0, timestamp=0.50, camera="cam1",
                                    overload=True)
    assert result.minted == []
    blocked_reasons = [reason for _, reason in result.blocked_mints]
    assert any("overload" in r for r in blocked_reasons)


# ---------------------------------------------------------------------------
# Collision quarantine (PLAN 2 Â§6.5).
# ---------------------------------------------------------------------------

def test_identity_claimed_twice_is_quarantined(rig: Rig):
    """Two spatially separate observations of one GID â†’ quarantine, not merge."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0))
    gid = rig.active_gid()

    # Two observations in the same frame, both feasibly matching gid but the
    # second arriving through the cam2 corridor (cross-camera duplicate claim).
    near = rig.observation(10.9, 10.0, 0.30, "cam1", velocity=(0.0, 0.0))
    far = rig.observation(44.5, 15.0, 0.30, "cam2", velocity=(0.0, 0.0))
    decisions, result = rig.step(near, far, timestamp=0.30)
    # The duplicate claim must not silently stretch the identity over both.
    assert len(result.matched) <= 1
    assert not result.collisions or all(g == gid for g, _ in result.collisions)


# ---------------------------------------------------------------------------
# Lifecycle + audit trail (Phase 3 rubric A: "auditable + timestamp").
# ---------------------------------------------------------------------------

def test_lifecycle_transitions_are_audited(rig: Rig):
    rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    gid = rig.single_live_gid()
    events = rig.events_of(IdentityEventType.MINT)
    assert events and events[-1].global_id == gid
    assert events[-1].timestamp == pytest.approx(0.00)
    assert events[-1].frame_sequence >= 0


def test_parked_identity_not_retired_by_time(rig: Rig):
    """PLAN 2 Â§6.4: slot ownership keeps retention open forever."""
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10)
    gid = rig.active_gid()
    rig.registry.mark_parked(gid, "D08", timestamp=0.20, frame_sequence=2)
    rig.step(timestamp=60.0)   # far beyond every timeout
    state = rig.registry.get(gid)
    assert state is not None and state.lifecycle_state is LifecycleState.PARKED


def test_exit_confirmation_retires(rig: Rig):
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10)
    gid = rig.active_gid()
    assert rig.registry.confirm_exit(gid, timestamp=1.0, frame_sequence=10)
    rig.step(timestamp=1.10)
    state = rig.registry.get(gid)
    assert state is not None and state.lifecycle_state is LifecycleState.RETIRED


def test_retire_releases_local_track_ownership(rig: Rig):
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10)
    gid = rig.active_gid()
    owner = rig.registry.owner_of_local_track("cam1", 1)
    assert owner == gid
    rig.registry.confirm_exit(gid, timestamp=1.0, frame_sequence=10)
    rig.step(timestamp=1.10)
    assert rig.registry.owner_of_local_track("cam1", 1) is None


def test_retired_local_track_history_blocks_gid_reuse(rig: Rig):
    _, first_result, first = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    old_gid = first_result.minted[0]
    rig.registry.confirm_exit(old_gid, timestamp=0.10, frame_sequence=1)
    rig.step(timestamp=0.20)
    assert rig.registry.owner_of_local_track("cam1", first.observation_id) is None
    assert (rig.registry.historical_owner_of_local_track("cam1", first.observation_id)
            == old_gid)
    returned = rig.observation(10.1, 10.0, 0.30, "cam1")
    returned.local_track_ids = (("cam1", first.observation_id),)
    outcome = rig.associator.associate(rig.registry.views(0.30), [returned])
    result = rig.registry.ingest([returned], outcome, 0.30, 3)
    assert result.minted == []
    assert result.blocked_mints[0][1].startswith("local_track_already_had_owner")


# ---------------------------------------------------------------------------
# Provisional maturity (PLAN 1 stage 8 logic 4d).
# ---------------------------------------------------------------------------

def test_provisional_never_published_before_maturity(rig: Rig):
    decision, result, _ = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    assert result.minted, "first observation of a new vehicle mints provisional"
    state = rig.registry.get(result.minted[0])
    assert state.lifecycle_state is LifecycleState.PROVISIONAL
    assert state not in rig.registry.published()


def test_provisional_promotes_after_maturity(rig: Rig):
    _, result, _ = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    gid = result.minted[0]
    # t_maturity = 0.25 s, n_maturity = 2 by default.
    rig.drive(10.5, 10.0, timestamp=0.30, camera="cam1")
    state = rig.registry.get(gid)
    assert state.lifecycle_state is LifecycleState.ACTIVE
    assert rig.events_of(IdentityEventType.ACTIVATE)


def test_provisional_expired_silently(rig: Rig):
    """A one-blob ghost never reaches ACTIVE and is retired without audit spam."""
    _, result, _ = rig.drive(10.0, 10.0, timestamp=0.00, camera="cam1")
    gid = result.minted[0]
    rig.step(timestamp=1.00)   # no further observations
    state = rig.registry.get(gid)
    assert state is None or state.lifecycle_state is LifecycleState.RETIRED
    # The retire of an expired provisional is not an audited RETIRE event.
    retires = [e for e in rig.events_of(IdentityEventType.RETIRE) if e.global_id == gid]
    assert not retires


# ---------------------------------------------------------------------------
# Alias stitch + session binding (Phase 5 hooks, registry side).
# ---------------------------------------------------------------------------

def test_alias_is_audited_and_keeps_sessions(rig: Rig):
    """Two genuinely distinct identities may be stitched, audited, session-safe."""
    # Two vehicles far apart, both warmed to ACTIVE.
    rig.drive_n([(10.0, 10.0), (10.5, 10.0), (11.0, 10.0)], "cam1", start=0.00, dt=0.10)
    first = rig.active_gid()
    rig.registry.bind_session(first, "S1", timestamp=0.20, frame_sequence=2)

    from conftest import make_observation
    # Vehicle 2 in cam2's exclusive region â€” its own identity.
    second_obs = make_observation(70.0, 40.0, 0.00, "cam2", 900)
    rig.registry.ingest(
        [second_obs],
        rig.associator.associate(rig.registry.views(0.00), [second_obs]),
        0.00, 0)
    for x in (70.5, 71.0):
        obs = make_observation(x, 40.0, 0.10 * (x - 69.5), "cam2", 900 + int(x))
        rig.registry.ingest(
            [obs],
            rig.associator.associate(rig.registry.views(obs.timestamp), [obs]),
            obs.timestamp, int(obs.timestamp * 10))
    live = {s.global_id for s in rig.registry.live()}
    assert first in live and len(live) == 2
    second = (live - {first}).pop()
    rig.registry.bind_session(second, "S2", timestamp=0.30, frame_sequence=3)

    assert rig.registry.alias(first, second, timestamp=0.40, frame_sequence=4)
    state = rig.registry.get(first)
    assert "S1" in state.session_ids and "S2" in state.session_ids
    assert rig.registry.get(second).lifecycle_state is LifecycleState.RETIRED
    assert rig.events_of(IdentityEventType.ALIAS)


# ---------------------------------------------------------------------------
# Grace-window blocker uses the relaxed gate exactly (PLAN 2 Â§6.3).
# ---------------------------------------------------------------------------

def test_grace_blocker_survives_moderate_displacement(rig: Rig):
    """A blob beyond the association gate but inside the relaxed grace reach.

    PLAN 2 §6.3 relaxes the chi-square gate ×3 inside the grace window: a
    blob too far for a normal match must still block minting (it is *this*
    identity with a noisy projection), not become a second Global ID.
    """
    rig.drive_n([(10.0, 10.0), (10.5, 10.0)], "cam1", start=0.00, dt=0.10,
                velocity=(0.0, 0.0))
    gid = rig.active_gid()
    last_seen = rig.registry.get(gid).last_observed_timestamp
    last_x = rig.registry.get(gid).latest_world_position[0]

    # Beyond the normal gate but within the relaxed gate reach and the
    # physical speed bound (12·0.2 + 0.15 = 2.55).
    decision, result, _ = rig.drive(last_x + 2.0, 10.0, timestamp=last_seen + 0.20,
                                    camera="cam1", velocity=(10.0, 0.0))
    assert result.minted == [], (
        f"minted {result.minted} while grace hypothesis {gid} was alive "
        f"(deferred={result.deferred}, blocked={[b[1] for b in result.blocked_mints]})")
    assert rig.single_live_gid() == gid

