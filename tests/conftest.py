"""Test harness for the world-side phases (PLAN 1 stages 5-8).

Shared fixtures:
* a two-camera topology whose exit/entry corridors reproduce the exact numbers
  of PLAN 3 Scenario C (C1 exit x∈[40,45], y∈[10,20]; C2 entry x∈[44,49]);
* ``Rig`` — drives the registry + associator exactly the way ``pipeline.py``
  does (views → associate → ingest), so scenario tests exercise the real
  production path, not a mock of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from techgar.association import TopologyConstrainedAssociator  # noqa: E402
from techgar.config_world import IdentityConfig, AssociationConfig  # noqa: E402
from techgar.contracts import TopologyRegion  # noqa: E402
from techgar.registry import GlobalIdentityRegistry  # noqa: E402
from techgar.topology import CameraTopology, CameraZone, TopologyEdge  # noqa: E402
from techgar.world_contracts import FusedWorldDetection  # noqa: E402


def _rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


@pytest.fixture
def topology() -> CameraTopology:
    """Two cameras with the PLAN 3 Scenario C geometry.

    C1 exit corridor towards C2: x ∈ [40, 45], y ∈ [10, 20]
    C2 entry corridor from C1:   x ∈ [44, 49], y ∈ [10, 20]
    Reverse corridors mirror them so the topology is symmetric.
    """
    cam1 = CameraZone(
        camera_id="cam1",
        fov_polygon=_rect(0, 0, 45, 60),
        exit_polygons={"cam2": _rect(40, 10, 45, 20)},
        entry_polygons={"cam2": _rect(0, 10, 5, 20)},
    )
    cam2 = CameraZone(
        camera_id="cam2",
        fov_polygon=_rect(44, 0, 100, 60),
        exit_polygons={"cam1": _rect(44, 10, 49, 20)},
        # Entry from cam1 sits on the shared seam (x∈[44,49]) — the region a
        # vehicle physically enters when crossing from cam1's exit corridor.
        entry_polygons={"cam1": _rect(44, 10, 49, 20)},
    )
    topology = CameraTopology(
        zones={"cam1": cam1, "cam2": cam2},
        edges={
            ("cam1", "cam2"): TopologyEdge("cam1", "cam2", dt_min=0.0, dt_max=4.0,
                                           dt_expected=0.3, v_max=12.0),
            ("cam2", "cam1"): TopologyEdge("cam2", "cam1", dt_min=0.0, dt_max=4.0,
                                           dt_expected=0.3, v_max=12.0),
        },
        overlaps={("cam1", "cam2"): _rect(44, 0, 45, 60)},
    )
    return topology


def make_observation(x: float, y: float, timestamp: float, camera: str,
                     observation_id: int, *, velocity=None, appearance=None,
                     covariance: float = 0.25, quality: float = 1.0,
                     latent: bool = False, partial: bool = False,
                     frame_sequence: int | None = None) -> FusedWorldDetection:
    """A fused world detection at ``(x, y)`` seen by exactly one camera."""
    return FusedWorldDetection(
        timestamp=float(timestamp),
        frame_sequence=int(frame_sequence if frame_sequence is not None
                           else round(timestamp * 10)),
        position=np.array([float(x), float(y)]),
        covariance=np.eye(2) * float(covariance),
        footprint=_rect(x - 1.0, y - 1.0, x + 1.0, y + 1.0),
        contributing_cameras=(camera,),
        contributing_observations=(observation_id,),
        fusion_confidence=1.0,
        topology_region=TopologyRegion.NORMAL,
        local_track_ids=((camera, observation_id),),
        quality=float(quality),
        velocity=(None if velocity is None else np.asarray(velocity, dtype=float)),
        footprint_area=4.0,
        footprint_aspect=1.0,
        appearance=appearance,
        latent=latent,
        partial=partial,
        observation_id=observation_id,
    )


class Rig:
    """Drives registry + associator the way the production pipeline does."""

    def __init__(self, topology: CameraTopology, registry: GlobalIdentityRegistry | None = None,
                 identity_config: IdentityConfig | None = None,
                 association_config: AssociationConfig | None = None) -> None:
        self.topology = topology
        identity_config = identity_config or IdentityConfig()
        self.registry = registry or GlobalIdentityRegistry(
            config=identity_config,
            topology=topology,
            association=association_config or AssociationConfig(),
            rho_seam=0.15)
        self.associator = TopologyConstrainedAssociator(
            topology, association_config or AssociationConfig(),
            identity_config, rho_seam=0.15)
        self._next_observation_id = 0

    def observation(self, x, y, timestamp, camera, **kwargs) -> FusedWorldDetection:
        self._next_observation_id += 1
        return make_observation(x, y, timestamp, camera, self._next_observation_id,
                                **kwargs)

    def step(self, *observations: FusedWorldDetection, timestamp: float,
             overload: bool = False):
        """One frame: associate then ingest, returning (decision map, IngestResult)."""
        frame_sequence = int(round(timestamp * 10))
        views = self.registry.views(timestamp)
        outcome = self.associator.associate(views, list(observations))
        result = self.registry.ingest(list(observations), outcome, timestamp,
                                      frame_sequence, overload=overload)
        decisions = {d.observation_id: d for d in outcome.decisions}
        return decisions, result

    def drive(self, x, y, timestamp, camera, **kwargs):
        """Convenience: one observation, one frame."""
        overload = kwargs.pop("overload", False)
        observation = self.observation(x, y, timestamp, camera, **kwargs)
        decisions, result = self.step(observation, timestamp=timestamp, overload=overload)
        return decisions.get(observation.observation_id), result, observation

    def drive_n(self, points, camera: str, *, dt: float = 0.10, start: float = 0.0,
                warm_until_active: bool = True, **kwargs):
        """Drive a vehicle through several observations.

        With ``warm_until_active`` (default) the sequence is extended by
        *extrapolating the last motion vector* until the minted identity
        leaves PROVISIONAL (PLAN 1 stage 8 logic 4d: ``n_maturity``
        observations *and* ``t_maturity`` seconds).  Repeating the last point
        instead would drag the learned velocity to zero, which silently
        changes the physics of every scenario that follows (Scenario F needs
        a *moving* vehicle behind its lag gap).
        """
        from techgar.states import LifecycleState
        last = None
        for index, (x, y) in enumerate(points):
            last = self.drive(x, y, start + index * dt, camera, **kwargs)
        if warm_until_active:
            # Extrapolate along the last motion vector, but clamp each warm
            # step to a fraction of the Kalman association gate so the warm-up
            # itself never oscillates past its own track (an overshooting
            # learned velocity would mint a new identity per frame — a test
            # artifact, not the behaviour under test).
            if len(points) >= 2:
                dx = points[-1][0] - points[-2][0]
                dy = points[-1][1] - points[-2][1]
            else:
                dx = dy = 0.0
            clamp = 0.5 * dt
            length = float(np.hypot(dx, dy))
            if length > clamp:
                dx, dy = dx * clamp / length, dy * clamp / length
            x, y = points[-1]
            guard = 0
            while guard < 40:
                live = self.registry.live()
                if len(live) == 1 and live[0].lifecycle_state is not LifecycleState.PROVISIONAL:
                    break
                x, y = x + dx, y + dy
                index = len(points) + guard
                last = self.drive(x, y, start + index * dt, camera, **kwargs)
                guard += 1
        return last

    # --- assertions ---------------------------------------------------------
    def single_live_gid(self) -> int:
        live = self.registry.live()
        assert len(live) == 1, f"expected exactly one live identity, got {[s.global_id for s in live]}"
        return live[0].global_id

    def active_gid(self) -> int:
        """The single live identity, asserted to have passed maturity."""
        gid = self.single_live_gid()
        state = self.registry.get(gid)
        assert state.lifecycle_state is not __import__(
            "techgar.states", fromlist=["LifecycleState"]).LifecycleState.PROVISIONAL, (
            "identity is still PROVISIONAL — use drive_n(...) to warm it up first")
        return gid

    def events_of(self, *types) -> list:
        return self.registry.events.of_type(*types)


@pytest.fixture
def rig(topology) -> Rig:
    return Rig(topology)


def make_fast_rig(topology: CameraTopology, **identity_overrides) -> Rig:
    """A Rig whose identities mature instantly (no PROVISIONAL warm-up).

    Scenario tests that depend on a *learned velocity* (e.g. Scenario F) must
    not have the warm-up loop clamp their motion; instant maturity lets the
    test control exactly how many observations the vehicle has.
    """
    config = IdentityConfig(t_maturity=0.0, n_maturity=1, **identity_overrides)
    return Rig(topology, identity_config=config)


def one_hot(index: int, dimension: int = 27) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector
