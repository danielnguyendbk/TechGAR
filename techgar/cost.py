"""PLAN 2 §4 — the composite association cost, and §6.1 — the identity score.

    C_ij = w_d C_distance + w_theta C_direction + w_g C_geometry
           + w_a C_appearance + C_topology + C_time

Every component is computed and kept, so an audit trail can say *why* an identity
matched.  Two failure modes the plan describes explicitly are structural here:

* §4.8 — distance must never dominate: it is one weighted term of six, and it is
  a Mahalanobis distance (only trustworthy *with* its covariance), never a raw
  Euclidean one;
* §4.9 — direction must never be removed, but must also never hard-reject: it is
  down-weighted by a reliability factor when either velocity is untrustworthy, so
  a legitimate U-turn stays matchable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config_world import AssociationConfig, IdentityConfig
from .linalg import CHI2_2DOF_99, mahalanobis_sq, positional_sigma
from .states import LifecycleState
from .topology import CameraTopology, HandoffCheck
from .world_contracts import FusedWorldDetection


@dataclass
class IdentityView:
    """An identity as the association layer is allowed to see it at time t_j."""

    global_id: int
    position: np.ndarray
    covariance: np.ndarray
    velocity: np.ndarray
    area: float
    aspect: float
    gallery: Any
    last_camera: str
    last_position: np.ndarray
    last_timestamp: float
    lifecycle: LifecycleState
    slot_id: str | None = None
    speed_reliability: float = 1.0
    #: successor camera -> last time this identity was *observed* inside the exit
    #: corridor towards it (PLAN 2 §3.4 "identity ĐÃ vào exit polygon").
    exit_corridor_at: dict[str, float] = field(default_factory=dict)


@dataclass
class CostComponents:
    distance: float = float("inf")
    direction: float = 0.0
    geometry: float = 0.0
    appearance: float = 0.0
    topology: float = 0.0
    time: float = 0.0
    total: float = float("inf")
    feasible: bool = False
    reason: str = ""
    direction_weight: float = 0.0
    handoff: HandoffCheck | None = None
    same_camera: bool = True
    identity_score: float = 0.0
    breakdown: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"distance": self.distance, "direction": self.direction,
                "geometry": self.geometry, "appearance": self.appearance,
                "topology": self.topology, "time": self.time, "total": self.total,
                "direction_weight": self.direction_weight, "reason": self.reason,
                "same_camera": self.same_camera, "identity_score": self.identity_score}


def direction_cost(v_i, v_j) -> float:
    """(1 - cos theta) / 2 in [0, 1] (PLAN 2 §4.2); 0.5 when either is unknown."""
    if v_i is None or v_j is None:
        return 0.5
    a = np.asarray(v_i, dtype=float).reshape(-1)
    b = np.asarray(v_j, dtype=float).reshape(-1)
    if a.size < 2 or b.size < 2:
        return 0.5
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.5
    return float((1.0 - float(a @ b) / (na * nb)) / 2.0)


def geometry_cost(area_j: float, area_i: float, aspect_j: float, aspect_i: float,
                  eta: float) -> float:
    """|log(A_j / A_i)| + eta |log(r_j / r_i)| (PLAN 2 §4.3)."""
    area = abs(np.log(max(area_j, 1e-6) / max(area_i, 1e-6)))
    aspect = abs(np.log(max(aspect_j, 1e-6) / max(aspect_i, 1e-6)))
    return float(area + eta * aspect)


def time_cost(dt: float, expected: float, dt_min: float, dt_max: float, soft: bool) -> float:
    """PLAN 2 §4.6 — hard window, plus the soft log penalty inside it."""
    if dt < dt_min - 1e-9 or dt > dt_max + 1e-9:
        return float("inf")
    if not soft or expected <= 0.0 or dt <= 0.0:
        return 0.0
    return float(abs(np.log(max(dt, 1e-3) / expected)))


def direction_reliability(identity: IdentityView, observation: FusedWorldDetection, dt: float,
                          config: AssociationConfig) -> float:
    """How much the direction term may be trusted (PLAN 2 §4.2 down-weighting)."""
    speed_i = float(np.linalg.norm(identity.velocity))
    speed_j = 0.0 if observation.velocity is None else float(np.linalg.norm(observation.velocity))
    if speed_i < config.direction_min_speed or speed_j < config.direction_min_speed:
        return 0.0
    if dt > config.direction_reliability_gap:
        return float(max(0.0, 1.0 - (dt - config.direction_reliability_gap)))
    return float(identity.speed_reliability)


def compute_cost(identity: IdentityView, observation: FusedWorldDetection,
                 topology: CameraTopology, config: AssociationConfig,
                 identity_config: IdentityConfig, rho_seam: float = 0.0) -> CostComponents:
    components = CostComponents()
    target_camera = observation.primary_camera
    dt = observation.timestamp - identity.last_timestamp
    same_camera = target_camera in {cam for cam in observation.contributing_cameras
                                    if cam == identity.last_camera}
    components.same_camera = bool(same_camera)

    # --- C_topology (PLAN 2 §4.5): infeasible candidates leave the matrix -----
    if not same_camera and config.enable_topology:
        tolerance = max(
            2.0 * positional_sigma(identity.covariance) + positional_sigma(observation.covariance) + rho_seam,
            0.25,
        )
        confirmed_at = identity.exit_corridor_at.get(target_camera, float("-inf"))
        source_confirmed = confirmed_at >= identity.last_timestamp - 1.0
        check = topology.check_handoff(identity.last_camera, identity.last_position,
                                      target_camera, observation.position, dt, rho_seam,
                                      tolerance=tolerance, source_confirmed=source_confirmed)
        components.handoff = check
        if not check.feasible:
            components.reason = f"topology:{check.reason}"
            components.topology = float("inf")
            return components
        expected = check.edge.dt_expected if check.edge is not None else config.handoff_dt_max / 2
        dt_min = check.edge.dt_min if check.edge is not None else config.handoff_dt_min
        dt_max = check.edge.dt_max if check.edge is not None else config.handoff_dt_max
    else:
        expected = max(dt, 1e-3)
        dt_min, dt_max = -1e-6, max(identity_config.t_max_missing, config.handoff_dt_max)

    # --- C_time (PLAN 2 §4.6) ------------------------------------------------
    components.time = time_cost(dt, expected, dt_min, dt_max, config.time_soft)
    if not np.isfinite(components.time):
        components.reason = "time_window"
        return components

    # --- physical speed bound (PLAN 2 §3.4 displacement constraint) ----------
    displacement = float(np.linalg.norm(observation.position - identity.last_position))
    if displacement > identity_config.v_max_world * max(dt, 1e-3) + rho_seam + 1e-9:
        components.reason = "speed_bound"
        components.total = float("inf")
        return components

    # --- C_distance (PLAN 2 §4.1) -------------------------------------------
    innovation = observation.position - identity.position
    s = np.asarray(identity.covariance, dtype=float) + np.asarray(observation.covariance,
                                                                  dtype=float)
    if not same_camera:
        seam_sigma = max(rho_seam, 0.12)
        s = s + (seam_sigma ** 2) * np.eye(2)
    components.distance = mahalanobis_sq(innovation, s)
    if components.distance >= config.gate:
        components.reason = f"gate:{components.distance:.2f}>={config.gate:.2f}"
        return components

    components.direction = direction_cost(identity.velocity, observation.velocity)
    components.direction_weight = direction_reliability(identity, observation, dt, config)
    components.geometry = geometry_cost(observation.footprint_area, identity.area,
                                        observation.footprint_aspect, identity.aspect,
                                        config.eta_aspect)
    components.appearance = (identity.gallery.cost(observation.appearance,
                                                   config.alpha_appearance)
                             if config.enable_appearance and identity.gallery is not None else 0.0)
    components.total = float(
        config.w_distance * components.distance
        + config.w_direction * components.direction_weight * components.direction
        + config.w_geometry * components.geometry
        + config.w_appearance * components.appearance
        + components.topology + components.time)
    components.feasible = True
    components.reason = "ok"
    components.identity_score = identity_score(components)
    return components


def identity_score(components: CostComponents) -> float:
    """PLAN 2 §6.1 composite identity score in [0, 1].

    S = w_p S_p + w_a S_a + w_t S_t + w_z S_z + w_c S_c, with each evidence channel
    converted from its cost: position from the chi-square gate, appearance from
    cosine distance, time from the log penalty, topology from validity, geometry
    from the log-ratio term.
    """
    s_position = float(np.exp(-components.distance / CHI2_2DOF_99))
    s_appearance = float(np.clip(1.0 - components.appearance, 0.0, 1.0))
    s_time = float(np.exp(-max(components.time, 0.0)))
    s_topology = 0.0 if not np.isfinite(components.topology) else 1.0
    s_geometry = float(np.exp(-max(components.geometry, 0.0)))
    weights = (0.35, 0.20, 0.15, 0.15, 0.15)
    values = (s_position, s_appearance, s_time, s_topology, s_geometry)
    return float(sum(w * v for w, v in zip(weights, values)))
