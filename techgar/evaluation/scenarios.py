"""Executable ground-truth scenarios A-I from PLAN 3 §2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..association import TopologyConstrainedAssociator
from ..config import TechgarConfig
from ..contracts import TopologyRegion
from ..registry import GlobalIdentityRegistry
from ..slot_engine import SlotOccupancyEngine, VehicleFootprintView
from ..states import LifecycleState, SlotOccupancy
from ..topology import CameraTopology, CameraZone, TopologyEdge
from ..world_contracts import FusedWorldDetection


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _topology() -> CameraTopology:
    cam1 = CameraZone(
        "cam1", _rect(0, 0, 45, 60),
        exit_polygons={"cam2": _rect(40, 10, 45, 20)},
        entry_polygons={"cam2": _rect(0, 10, 5, 20)},
    )
    cam2 = CameraZone(
        "cam2", _rect(44, 0, 100, 60),
        exit_polygons={"cam1": _rect(44, 10, 49, 20)},
        entry_polygons={"cam1": _rect(44, 10, 49, 20)},
    )
    return CameraTopology(
        zones={"cam1": cam1, "cam2": cam2},
        edges={
            ("cam1", "cam2"): TopologyEdge("cam1", "cam2", 0.0, 4.0, 0.3, 12.0),
            ("cam2", "cam1"): TopologyEdge("cam2", "cam1", 0.0, 4.0, 0.3, 12.0),
        },
        overlaps={("cam1", "cam2"): _rect(44, 0, 45, 60)},
    )


def _one_hot(index: int, size: int = 27) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector


@dataclass
class ScenarioResult:
    key: str
    name: str
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "passed": self.passed,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "detail": dict(self.detail),
        }


class _WorldRig:
    def __init__(self, config: TechgarConfig | None = None) -> None:
        self.config = config or TechgarConfig()
        self.config.identity.t_maturity = 0.0
        self.config.identity.n_maturity = 1
        self.topology = _topology()
        self.registry = GlobalIdentityRegistry(
            self.config.identity, self.config.world_kalman,
            self.config.association, self.topology, self.config.projection.rho_seam,
        )
        self.associator = TopologyConstrainedAssociator(
            self.topology, self.config.association, self.config.identity,
            self.config.projection.rho_seam,
        )
        self.observation_id = 0
        self.frame = 0

    def observation(self, x: float, y: float, timestamp: float, camera: str,
                    *, velocity=None, appearance=None, covariance: float = 0.25,
                    local_track_id: int | None = None, latent: bool = False
                    ) -> FusedWorldDetection:
        self.observation_id += 1
        local_id = self.observation_id if local_track_id is None else local_track_id
        return FusedWorldDetection(
            timestamp=timestamp,
            frame_sequence=self.frame,
            position=np.array([x, y], dtype=float),
            covariance=np.eye(2) * covariance,
            footprint=_rect(x - 1.0, y - 1.0, x + 1.0, y + 1.0),
            contributing_cameras=(camera,),
            contributing_observations=(self.observation_id,),
            fusion_confidence=1.0,
            topology_region=TopologyRegion.NORMAL,
            local_track_ids=((camera, local_id),),
            quality=1.0,
            velocity=None if velocity is None else np.asarray(velocity, dtype=float),
            footprint_area=4.0,
            footprint_aspect=1.0,
            appearance=appearance,
            latent=latent,
            observation_id=self.observation_id,
        )

    def step(self, timestamp: float, observations: list[FusedWorldDetection],
             overload: bool = False):
        self.frame += 1
        outcome = self.associator.associate(self.registry.views(timestamp), observations)
        ingest = self.registry.ingest(observations, outcome, timestamp, self.frame,
                                      overload=overload)
        return outcome, ingest

    def observe(self, x: float, y: float, timestamp: float, camera: str, **kwargs):
        observation = self.observation(x, y, timestamp, camera, **kwargs)
        outcome, ingest = self.step(timestamp, [observation])
        decision = next((item for item in outcome.decisions
                         if item.observation_id == observation.observation_id), None)
        global_id = decision.assigned_global_id if decision is not None else None
        if global_id is None and ingest.minted:
            global_id = ingest.minted[0]
        return global_id, decision, ingest, observation


def _switches(ids: list[int | None]) -> int:
    seen = [value for value in ids if value is not None]
    return sum(1 for a, b in zip(seen, seen[1:]) if a != b)


def _result(key: str, name: str, checks: dict[str, bool], **kwargs) -> ScenarioResult:
    return ScenarioResult(key, name, all(checks.values()), checks, **kwargs)


def _scenario_a(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    ids = [rig.observe(x, 20.0, index * 0.1, "cam1", velocity=(5.0, 0.0))[0]
           for index, x in enumerate((10.0, 10.5, 11.0))]
    checks = {"single_global_id": len(set(ids)) == 1, "idsw_zero": _switches(ids) == 0}
    return _result("A", "Chuyển động thường trong một camera", checks,
                   metrics={"id_switches": float(_switches(ids))}, detail={"ids": ids})


def _scenario_b(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    first = rig.observe(20.0, 30.0, 0.0, "cam1")[0]
    rig.step(0.1, [])
    held = rig.registry.get(first)
    last = rig.observe(21.0, 30.0, 0.2, "cam1")[0]
    checks = {
        "identity_held_during_gap": held is not None,
        "missing_state": held is not None and held.lifecycle_state in (
            LifecycleState.TEMPORARILY_MISSING, LifecycleState.OCCLUDED,
            LifecycleState.ACTIVE,
        ),
        "same_global_id": first == last,
    }
    return _result("B", "Gap detection một frame", checks,
                   metrics={"id_switches": float(first != last)},
                   detail={"first": first, "last": last})


def _scenario_c(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    ids = []
    ids.append(rig.observe(40.0, 15.0, 0.0, "cam1", velocity=(12.0, 0.0))[0])
    ids.append(rig.observe(41.2, 15.0, 0.1, "cam1", velocity=(12.0, 0.0))[0])
    ids.append(rig.observe(42.4, 15.0, 0.2, "cam1", velocity=(12.0, 0.0))[0])
    rig.step(0.3, [])
    ids.append(rig.observe(44.0, 15.0, 0.4, "cam2", velocity=(12.0, 0.0))[0])
    ids.append(rig.observe(45.2, 15.0, 0.5, "cam2", velocity=(12.0, 0.0))[0])
    invalid_rig = _WorldRig(config)
    # A visually/geometrically plausible camera change that is physically
    # impossible because the source never entered cam1's exit corridor and the
    # target is outside cam2's entry corridor.  Distance alone must not reject it:
    # otherwise the no-topology ablation would accidentally exercise the speed
    # gate instead of proving that the directed topology contributes evidence.
    source = invalid_rig.observe(20.0, 30.0, 0.0, "cam1", velocity=(5.0, 0.0))[0]
    target = invalid_rig.observe(20.5, 30.0, 0.1, "cam2", velocity=(5.0, 0.0))[0]
    checks = {
        "handoff_keeps_id": len(set(ids)) == 1,
        "invalid_transition_rejected": source != target,
    }
    return _result("C", "Camera handoff", checks,
                   metrics={"id_switches": float(_switches(ids)),
                            "invalid_handoff_rate": 0.0 if source != target else 1.0},
                   detail={"ids": ids, "invalid_pair": [source, target]})


def _scenario_d(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    a, b = _one_hot(1), _one_hot(2)
    physical = {"P01": [], "P02": []}
    for timestamp, x1, x2 in ((0.0, 30.0, 34.0), (0.2, 32.0, 32.0), (0.4, 34.0, 30.0)):
        o1 = rig.observation(x1, 20.0, timestamp, "cam1", velocity=(10.0, 0.0),
                             appearance=a, local_track_id=1)
        o2 = rig.observation(x2, 20.0, timestamp, "cam1", velocity=(-10.0, 0.0),
                             appearance=b, local_track_id=2)
        outcome, ingest = rig.step(timestamp, [o1, o2])
        decisions = {item.observation_id: item.assigned_global_id for item in outcome.decisions}
        for vehicle, observation in (("P01", o1), ("P02", o2)):
            value = decisions.get(observation.observation_id)
            if value is None:
                candidates = [gid for gid in ingest.minted if gid not in physical["P01"]
                              and gid not in physical["P02"]]
                value = candidates[0] if candidates else None
            physical[vehicle].append(value)
    switches = sum(_switches(values) for values in physical.values())
    checks = {
        "two_identities_survive": len(rig.registry.live()) == 2,
        "idsw_zero": switches == 0,
        "identities_distinct": physical["P01"][0] != physical["P02"][0],
    }
    return _result("D", "Hai xe giao nhau", checks,
                   metrics={"id_switches": float(switches)}, detail=physical)


def _scenario_e(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    a, b = _one_hot(3), _one_hot(4)
    o1 = rig.observation(30.0, 20.0, 0.0, "cam1", appearance=a, local_track_id=1)
    o2 = rig.observation(34.0, 20.0, 0.0, "cam1", appearance=b, local_track_id=2)
    _, initial = rig.step(0.0, [o1, o2])
    original = tuple(sorted(initial.minted))
    before = {gid: len(rig.registry.get(gid).appearance_gallery.samples) for gid in original}
    touched = rig.registry.note_occlusion("cam1", [1, 2], 0.1, rig.frame)
    rig.step(0.1, [])
    rig.step(0.2, [])
    during = tuple(sorted(state.global_id for state in rig.registry.live()))
    n1 = rig.observation(30.5, 20.0, 0.3, "cam1", appearance=a, local_track_id=1)
    n2 = rig.observation(33.5, 20.0, 0.3, "cam1", appearance=b, local_track_id=2)
    outcome, ingest = rig.step(0.3, [n1, n2])
    after = {gid: len(rig.registry.get(gid).appearance_gallery.samples) for gid in original}
    assigned = tuple(sorted(item.assigned_global_id for item in outcome.decisions
                            if item.assigned_global_id is not None))
    checks = {
        "two_latent_identities": tuple(sorted(touched)) == original and during == original,
        "gallery_frozen_during_merge": all(after[gid] <= before[gid] + 1 for gid in original),
        "split_recovers_original_ids": assigned == original and not ingest.minted,
    }
    return _result("E", "Detection hợp nhất", checks,
                   metrics={"identity_count": float(len(during))},
                   detail={"original": original, "assigned": assigned})


def _scenario_f(config=None) -> ScenarioResult:
    rig = _WorldRig(config)
    ids = []
    for index in range(6):
        ids.append(rig.observe(50.0 + 1.2 * index, 50.0, 200.0 + 0.1 * index,
                               "cam1", velocity=(12.0, 0.0))[0])
    state = rig.registry.get(ids[-1])
    last_x = float(state.latest_world_position[0])
    last_t = state.last_observed_timestamp
    recovered = rig.observe(last_x + 6.0, 50.0, last_t + 0.5,
                            "cam1", velocity=(12.0, 0.0))[0]
    checks = {"same_global_id": recovered == ids[-1], "no_fragmentation": len(set(ids)) == 1}
    return _result("F", "Displacement lớn do lag", checks,
                   metrics={"id_switches": float(recovered != ids[-1])},
                   detail={"before": ids[-1], "after": recovered})


def _slot_polygon(cx: float, cy: float, size: float = 1.8) -> np.ndarray:
    half = size / 2.0
    return _rect(cx - half, cy - half, cx + half, cy + half)


def _vehicle(global_id: int, x: float, y: float, velocity=(0.0, 0.0)) -> VehicleFootprintView:
    return VehicleFootprintView(global_id, _slot_polygon(x, y, 1.6), np.array([x, y]),
                                np.asarray(velocity, dtype=float))


def _park(engine: SlotOccupancyEngine, global_id: int, slot: str,
          x: float, y: float) -> tuple[float, list[str | None]]:
    timestamp = 0.0
    owners = []
    for index, px in enumerate((x - 2.6, x - 1.4, x - 0.6, x - 0.2, x)):
        engine.update([_vehicle(global_id, px, y,
                                (0.5, 0.0) if index < 4 else (0.0, 0.0))],
                      timestamp, index)
        owners.append(engine.owner_of(slot))
        timestamp += 0.1
    for index in range(13):
        engine.update([_vehicle(global_id, x, y)], timestamp, 10 + index)
        owners.append(engine.owner_of(slot))
        timestamp += 0.1
    return timestamp, owners


def _slot_engine(config=None) -> SlotOccupancyEngine:
    cfg = (config or TechgarConfig()).slot
    return SlotOccupancyEngine({"D05": _slot_polygon(100.0, 10.0),
                                "D06": _slot_polygon(103.0, 10.0)}, cfg)


def _scenario_g(config=None) -> ScenarioResult:
    engine = _slot_engine(config)
    _, owners = _park(engine, 17, "D05", 100.0, 10.0)
    first_occupied = next((index for index, owner in enumerate(owners) if owner == 17), None)
    checks = {
        "occupied_by_original_gid": engine.owner_of("D05") == 17,
        "not_first_frame": first_occupied is not None and first_occupied > 0,
    }
    return _result("G", "Xe vào slot", checks,
                   metrics={"first_occupied_frame": float(first_occupied or 0)})


def _scenario_h(config=None) -> ScenarioResult:
    engine = _slot_engine(config)
    for index, x in enumerate((99.6, 100.0, 101.2, 102.4, 103.0, 104.0, 105.0, 106.0)):
        engine.update([_vehicle(17, x, 10.0, (4.0, 0.0))], index * 0.1, index)
    checks = {
        "d05_not_occupied": engine.owner_of("D05") is None,
        "d06_not_occupied": engine.owner_of("D06") is None,
        "no_terminal_false_occupancy": all(
            state.occupancy_state is not SlotOccupancy.OCCUPIED
            for state in engine.states.values()
        ),
    }
    return _result("H", "Xe đi ngang giữa hai slot", checks)


def _scenario_i(config=None) -> ScenarioResult:
    engine = _slot_engine(config)
    timestamp, _ = _park(engine, 17, "D05", 100.0, 10.0)
    before = engine.owner_of("D05")
    engine.update([], timestamp, 100)
    after_false_empty = engine.owner_of("D05")
    for index in range(1, 23):
        timestamp += 0.1
        engine.update([_vehicle(17, 100.0 + 0.4 * index, 10.0, (4.0, 0.0))],
                      timestamp, 100 + index)
    checks = {
        "owned_before_departure": before == 17,
        "false_empty_does_not_release": after_false_empty == 17,
        "released_after_departure": engine.owner_of("D05") is None,
    }
    return _result("I", "Xe rời slot đã đỗ", checks)


SCENARIOS: dict[str, Callable[[TechgarConfig | None], ScenarioResult]] = {
    "A": _scenario_a,
    "B": _scenario_b,
    "C": _scenario_c,
    "D": _scenario_d,
    "E": _scenario_e,
    "F": _scenario_f,
    "G": _scenario_g,
    "H": _scenario_h,
    "I": _scenario_i,
}


def run_scenario(key: str, config: TechgarConfig | None = None) -> ScenarioResult:
    normalized = key.upper().removeprefix("SCENARIO_").removeprefix("SCENARIO ")
    if normalized not in SCENARIOS:
        raise KeyError(f"unknown scenario {key!r}; expected A-I")
    return SCENARIOS[normalized](config)


def run_all_scenarios(config: TechgarConfig | None = None) -> list[ScenarioResult]:
    return [runner(config) for runner in SCENARIOS.values()]
