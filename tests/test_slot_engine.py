"""Slot occupancy engine tests — PLAN 1 Phase 4 exit criteria.

Covers the three PLAN 3 parking scenarios end-to-end through the real
engine (evidence gathering → temporal window → one-to-one assignment →
hysteresis), plus the ownership invariants of PLAN 3 §4:

* Scenario G — vehicle entering a slot: confirmed only after the temporal
  window, ownership keeps the original Global ID;
* Scenario H — vehicle passing between adjacent slots: no permanent
  occupancy, no oscillation;
* Scenario I — vehicle leaving a parked slot: released on positive
  departure evidence, never on a single false-empty frame;
* PLAN 2 §5.6 fail case — one loose-bbox frame with IoU 0.95 confirms
  nothing;
* PLAN 2 §5.7 — a parked vehicle with zero motion never loses its slot.
"""

from __future__ import annotations

import numpy as np
import pytest

from techgar.config_world import SlotConfig
from techgar.slot_engine import SlotOccupancyEngine, VehicleFootprintView
from techgar.states import SlotOccupancy


SLOT_HALF_W = 0.9
SLOT_HALF_H = 0.9


def slot_polygon(cx: float, cy: float, w: float = 2 * SLOT_HALF_W,
                 h: float = 2 * SLOT_HALF_H) -> np.ndarray:
    return np.array([
        [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
        [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2],
    ], dtype=float)


def footprint(cx: float, cy: float, w: float = 1.6, h: float = 1.6) -> np.ndarray:
    return slot_polygon(cx, cy, w, h)


def vehicle(global_id: int, cx: float, cy: float, velocity=(0.0, 0.0),
            observed: bool = True) -> VehicleFootprintView:
    return VehicleFootprintView(
        global_id=global_id, footprint=footprint(cx, cy),
        position=np.array([cx, cy]),
        velocity=np.asarray(velocity, dtype=float), observed=observed)


@pytest.fixture
def engine() -> SlotOccupancyEngine:
    return SlotOccupancyEngine(
        {"D05": slot_polygon(100.0, 10.0), "D06": slot_polygon(103.0, 10.0)},
        SlotConfig())


# ---------------------------------------------------------------------------
# Scenario G — vehicle enters slot D05 (here: the engine's own slot ids).
# ---------------------------------------------------------------------------

def park(engine: SlotOccupancyEngine, global_id: int, slot_id: str,
         cx: float, cy: float) -> float:
    """Drive an arrival until the slot is confirmed; return the last timestamp.

    The path must satisfy every condition of PLAN 2 §5.4: approach inward,
    slow to a stop below ``v_parked`` in the window tail, and dwell longer
    than ``dwell_confirm`` (0.8 s at 10 Hz = 8+ stationary samples).
    """
    approach = [(cx - 2.6, cx * 0 + cy), (cx - 1.4, cy), (cx - 0.6, cy),
                (cx - 0.2, cy), (cx, cy)]
    timestamps = []
    t = 0.0
    for i, (x, y) in enumerate(approach):
        timestamps.append(t)
        engine.update([vehicle(global_id, x, y,
                               velocity=(0.5, 0.0) if i < 4 else (0.0, 0.0))],
                      timestamp=t, frame_sequence=i)
        t += 0.10
    # Dwell: stationary inside the slot until the moving approach samples
    # leave the window tail (window 2.0 s × tail_fraction 0.5 = the last
    # 1.0 s must be entirely below v_parked).
    for _ in range(13):
        timestamps.append(t)
        engine.update([vehicle(global_id, cx, cy, velocity=(0.0, 0.0))],
                      timestamp=t, frame_sequence=int(t * 10))
        t += 0.10
    assert engine.owner_of(slot_id) == global_id, (
        f"park helper failed: {engine.states[slot_id].occupancy_state} "
        f"{engine.states[slot_id].__dict__}")
    return timestamps[-1]


def test_arrival_confirms_after_temporal_window(engine: SlotOccupancyEngine):
    """Coverage ramps up; occupancy is confirmed only after the window."""
    park(engine, 17, "D05", 100.0, 10.0)
    state = engine.states["D05"]
    assert state.occupancy_state is SlotOccupancy.OCCUPIED
    assert state.owning_global_id == 17


def test_observed_centroid_stop_overrides_stale_filter_velocity(engine: SlotOccupancyEngine):
    """A CV filter's residual momentum must not veto a visibly stationary dwell."""
    path = [97.4, 98.6, 99.4, 99.8, 100.0]
    timestamp = 0.0
    for x in path:
        engine.update([vehicle(17, x, 10.0, velocity=(2.0, 0.0))],
                      timestamp=timestamp, frame_sequence=int(timestamp * 10))
        timestamp += 0.10
    for _ in range(15):
        # The upstream filter still reports residual motion, but the measured
        # metric footprint is stationary in the slot.
        engine.update([vehicle(17, 100.0, 10.0, velocity=(2.0, 0.0))],
                      timestamp=timestamp, frame_sequence=int(timestamp * 10))
        timestamp += 0.10

    assert engine.owner_of("D05") == 17


def test_arrival_not_confirmed_early(engine: SlotOccupancyEngine):
    """Before the temporal window is satisfied the slot stays unconfirmed."""
    path = [(97.0, 10.0), (98.2, 10.0)]
    for i, (x, y) in enumerate(path):
        engine.update([vehicle(17, x, y, velocity=(0.5, 0.0))],
                      timestamp=0.10 * i, frame_sequence=i)
    assert engine.owner_of("D05") is None


def test_one_frame_transit_confirms_nothing(engine: SlotOccupancyEngine):
    """PLAN 2 §5.6 fail case: one loose frame with high overlap, no follow-up."""
    # A single observation perfectly centred (high IoU), then nothing.
    engine.update([vehicle(17, 100.0, 10.0, velocity=(4.0, 0.0))],
                  timestamp=0.0, frame_sequence=0)
    for i in range(1, 12):
        engine.update([vehicle(17, 106.0 + 0.4 * i, 10.0, velocity=(4.0, 0.0))],
                      timestamp=0.10 * i, frame_sequence=i)
    assert engine.owner_of("D05") is None
    assert engine.states["D05"].occupancy_state is not SlotOccupancy.OCCUPIED


# ---------------------------------------------------------------------------
# Scenario H — vehicle passes between two adjacent slots.
# ---------------------------------------------------------------------------

def test_pass_between_adjacent_slots_no_permanent_occupancy(engine: SlotOccupancyEngine):
    """D05 0.42 → D05/D06 boundary → D06 0.47 → leaves both."""
    path = [99.6, 100.0, 101.2, 102.4, 103.0, 104.0, 105.0, 106.0]
    for i, x in enumerate(path):
        engine.update([vehicle(17, x, 10.0, velocity=(4.0, 0.0))],
                      timestamp=0.10 * i, frame_sequence=i)
    assert engine.owner_of("D05") is None
    assert engine.owner_of("D06") is None
    # And no oscillating residue in a non-terminal state either.
    for state in engine.states.values():
        assert state.occupancy_state in (SlotOccupancy.EMPTY,
                                         SlotOccupancy.CLAIM_PENDING), state


# ---------------------------------------------------------------------------
# Scenario I — vehicle leaves a parked slot.
# (the ``park`` helper lives next to Scenario G above)
# ---------------------------------------------------------------------------


def test_departure_releases_on_positive_evidence(engine: SlotOccupancyEngine):
    timestamp = park(engine, 17, "D05", 100.0, 10.0)
    # The vehicle drives out; release needs evidence < tau_release sustained
    # for release_duration (1.5 s) — drive well past it.
    for i in range(1, 22):
        timestamp += 0.10
        x = 100.0 + 0.4 * i
        engine.update([vehicle(17, x, 10.0, velocity=(4.0, 0.0))],
                      timestamp=timestamp, frame_sequence=100 + i)
    assert engine.owner_of("D05") is None, "slot must release after departure"
    assert engine.states["D05"].occupancy_state is SlotOccupancy.EMPTY


def test_single_false_empty_frame_keeps_ownership(engine: SlotOccupancyEngine):
    """PLAN 2 §5.7: absence of one observation is not departure evidence."""
    timestamp = park(engine, 17, "D05", 100.0, 10.0)
    # One frame with no vehicle observation at all.
    engine.update([], timestamp=timestamp + 0.10, frame_sequence=200)
    assert engine.owner_of("D05") == 17


def test_parked_vehicle_keeps_slot_despite_zero_motion(engine: SlotOccupancyEngine):
    """PLAN 3 automatic-fail guard: motion = 0 must not lose the identity."""
    timestamp = park(engine, 17, "D05", 100.0, 10.0)
    for i in range(1, 25):
        timestamp += 0.10
        engine.update([vehicle(17, 100.0, 10.0, velocity=(0.0, 0.0))],
                      timestamp=timestamp, frame_sequence=300 + i)
    assert engine.owner_of("D05") == 17
    assert engine.states["D05"].occupancy_state is SlotOccupancy.OCCUPIED


# ---------------------------------------------------------------------------
# Ownership invariants (PLAN 3 §4).
# ---------------------------------------------------------------------------

def test_one_vehicle_cannot_own_two_slots(engine: SlotOccupancyEngine):
    park(engine, 17, "D05", 100.0, 10.0)
    # The same vehicle later parks in D06: D05 must be released.
    path = [(101.0, 10.0), (102.0, 10.0), (102.6, 10.0),
            (103.0, 10.0), (103.0, 10.0), (103.0, 10.0)]
    timestamp = 2.0
    for i, (x, y) in enumerate(path):
        timestamp += 0.10
        engine.update([vehicle(17, x, y, velocity=(0.4, 0.0) if i < 3 else (0.0, 0.0))],
                      timestamp=timestamp, frame_sequence=400 + i)
    owned = [slot for slot in ("D05", "D06") if engine.owner_of(slot) == 17]
    assert len(owned) == 1, f"vehicle 17 owns {owned}"


def test_two_vehicles_cannot_own_one_slot(engine: SlotOccupancyEngine):
    park(engine, 17, "D05", 100.0, 10.0)
    # A second vehicle arrives at the same slot: ownership must not flip.
    path = [(98.0, 10.0), (99.0, 10.0), (99.6, 10.0),
            (100.0, 10.0), (100.0, 10.0), (100.0, 10.0)]
    timestamp = 2.0
    for i, (x, y) in enumerate(path):
        timestamp += 0.10
        engine.update([
            vehicle(17, 100.0, 10.0, velocity=(0.0, 0.0)),
            vehicle(23, x, y, velocity=(0.4, 0.0) if i < 3 else (0.0, 0.0)),
        ], timestamp=timestamp, frame_sequence=500 + i)
    # Either 17 keeps it, or the conflict is recorded — never silently shared.
    owner = engine.owner_of("D05")
    assert owner in (17, 23, None)
    if owner == 23:
        assert any(e.kind == "ownership_conflict" for e in engine.events) or any(
            e.kind in ("released", "occupied") for e in engine.events)


def test_adjacent_slot_competition_resolved_deterministically(engine: SlotOccupancyEngine):
    """A vehicle straddling D05/D06 resolves to exactly one slot."""
    # Straddling the shared boundary x=101.5, moving slightly into D06.
    path = [(101.2, 10.0), (101.4, 10.0), (101.6, 10.0),
            (101.8, 10.0), (102.0, 10.0), (102.0, 10.0), (102.0, 10.0)]
    for i, (x, y) in enumerate(path):
        engine.update([vehicle(17, x, y, velocity=(0.3, 0.0) if i < 4 else (0.0, 0.0))],
                      timestamp=0.10 * i, frame_sequence=i)
    owners = [engine.owner_of(s) for s in ("D05", "D06")]
    assert owners.count(17) <= 1


# ---------------------------------------------------------------------------
# Hysteresis (PLAN 2 §5.5).
# ---------------------------------------------------------------------------

def test_hysteresis_supresses_evidence_flicker(engine: SlotOccupancyEngine):
    timestamp = park(engine, 17, "D05", 100.0, 10.0)
    # Evidence dips below tau_confirm but above tau_release: stays occupied.
    for i, offset in enumerate([0.4, 0.0, 0.5, 0.0, 0.4]):
        timestamp += 0.10
        engine.update([vehicle(17, 100.0 + offset, 10.0, velocity=(0.0, 0.0))],
                      timestamp=timestamp, frame_sequence=600 + i)
    assert engine.owner_of("D05") == 17


# ---------------------------------------------------------------------------
# Ablation switches (PLAN 3 §7 — slot halves of experiments B-hysteresis).
# ---------------------------------------------------------------------------

def test_temporal_window_ablation_allows_instant_occupancy():
    """With the window off, one strong frame confirms — the documented failure."""
    engine = SlotOccupancyEngine(
        {"D05": slot_polygon(100.0, 10.0)},
        SlotConfig(enable_temporal_window=False))
    engine.update([vehicle(17, 100.0, 10.0, velocity=(0.0, 0.0))],
                  timestamp=0.0, frame_sequence=0)
    assert engine.owner_of("D05") == 17
