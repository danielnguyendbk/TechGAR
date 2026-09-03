"""Stage 8 — the Global Identity Registry: the *only* place a Global ID is born or dies.

Invariant (PLAN 1 §1.2): one physical vehicle <-> exactly one active Global ID.

A new Global ID requires all four conditions of PLAN 1 stage 8 logic 4:
 1. no identity satisfies the physical feasibility constraints;
 2. no identity sits inside a valid grace window with a plausible score (PLAN 2 §6.3);
 3. no unresolved occlusion group can explain the observation (PLAN 2 §7);
 4. the candidate survived a maturity period (it lives as PROVISIONAL until then and
    is never published to the frontend).

A Global ID retires only on a confirmed exit event, or after a long timeout with
*every* retention channel of PLAN 2 §6.4 exhausted.  Overload never mints
(PLAN 1 Phase 6) and conflicting evidence is quarantined, never merged (PLAN 2 §6.5).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np

from .appearance import AppearanceGallery
from .config_vision import KalmanConfig
from .config_world import AssociationConfig, IdentityConfig
from .cost import IdentityView, compute_cost
from .crossing import CrossingDetector
from .identity_events import IdentityEventLog
from .kalman import LagAwareKalman
from .persistence import PersistenceStore
from .states import GlobalVehicleState, IdentityEventType, LifecycleState
from .topology import CameraTopology
from .world_contracts import AssociationDecision, DecisionType, FusedWorldDetection


@dataclass
class IngestResult:
    matched: dict[int, int] = field(default_factory=dict)
    minted: list[int] = field(default_factory=list)
    promoted: list[int] = field(default_factory=list)
    deferred: list[int] = field(default_factory=list)
    blocked_mints: list[tuple[int, str]] = field(default_factory=list)
    quarantined: list[int] = field(default_factory=list)
    retired: list[int] = field(default_factory=list)
    collisions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def minted_count(self) -> int:
        return len(self.minted)


class GlobalIdentityRegistry:
    """The identity authority.  Camera-local trackers only ever *propose*."""

    def __init__(self, config: IdentityConfig | None = None, kalman: KalmanConfig | None = None,
                 association: AssociationConfig | None = None,
                 topology: CameraTopology | None = None, rho_seam: float = 0.0,
                 crossing_detector: CrossingDetector | None = None,
                 store: PersistenceStore | None = None,
                 site_id: str = "default_site") -> None:
        self.config = config or IdentityConfig()
        self.kalman_config = kalman or KalmanConfig(q=2.0, q_size=0.5, r0=0.04)
        self.association_config = association or AssociationConfig()
        self.topology = topology or CameraTopology()
        self.rho_seam = rho_seam
        self.crossing_detector = crossing_detector or CrossingDetector()
        self.store = store
        self.site_id = site_id
        self.identities: dict[int, GlobalVehicleState] = {}
        self.retired_identities: dict[int, GlobalVehicleState] = {}
        self.events = IdentityEventLog()
        self.quarantined_observations: list[dict] = []
        self._local_track_owner: dict[tuple[str, int], int] = {}
        self._corridor_seen: dict[int, dict[str, float]] = {}
        self._occlusion_pending: dict[int, float] = {}
        self._parked_position: dict[int, np.ndarray] = {}
        self._duplicate_watch: dict[tuple[int, int], float] = {}
        if self.store is not None:
            self._next_global_id = self.store.current_global_id(self.site_id)
            self.restore_from_store()
        else:
            self._next_global_id = 0

    # --- introspection ------------------------------------------------------
    def live(self) -> list[GlobalVehicleState]:
        return [s for s in self.identities.values() if s.lifecycle_state.is_live]

    def published(self) -> list[GlobalVehicleState]:
        return [s for s in self.identities.values() if s.lifecycle_state.is_published]

    def get(self, global_id: int) -> GlobalVehicleState | None:
        return self.identities.get(global_id) or self.retired_identities.get(global_id)

    def owner_of_local_track(self, camera_id: str, local_track_id: int) -> int | None:
        return self._local_track_owner.get((camera_id, local_track_id))

    def views(self, timestamp: float) -> list[IdentityView]:
        """Identities predicted to ``timestamp`` — the only view association gets."""
        views: list[IdentityView] = []
        for state in self.live():
            kinematics: LagAwareKalman = state.kinematics
            prediction = kinematics.predict(timestamp)
            if state.lifecycle_state in (LifecycleState.PARKED, LifecycleState.TEMPORARILY_MISSING, LifecycleState.OCCLUDED):
                last_pos = (np.asarray(state.last_observed_position, dtype=float)
                            if state.last_observed_position is not None
                            else np.asarray(state.latest_world_position, dtype=float))
                prediction.state[0:2] = last_pos
                prediction.state[2:4] = 0.0
            speed = float(np.linalg.norm(prediction.state[2:4]))
            reliability = min(1.0, state.observation_count / 3.0)
            if speed < self.association_config.direction_min_speed:
                reliability *= 0.3
            views.append(IdentityView(
                global_id=state.global_id, position=prediction.state[0:2].copy(),
                covariance=prediction.covariance[0:2, 0:2].copy(),
                velocity=prediction.state[2:4].copy(), area=state.footprint_area,
                aspect=state.footprint_aspect, gallery=state.appearance_gallery,
                last_camera=state.latest_camera,
                last_position=(np.asarray(state.last_observed_position, dtype=float)
                               if state.last_observed_position is not None
                               else np.asarray(state.latest_world_position, dtype=float)),
                last_timestamp=state.last_observed_timestamp,
                lifecycle=state.lifecycle_state, slot_id=state.slot_id,
                speed_reliability=float(reliability),
                exit_corridor_at=dict(self._corridor_seen.get(state.global_id, {}))))
        return views

    # --- main entry point ---------------------------------------------------
    def ingest(self, observations: list[FusedWorldDetection], outcome, timestamp: float,
               frame_sequence: int, overload: bool = False) -> IngestResult:
        result = IngestResult()
        by_id = {o.observation_id: o for o in observations}
        claimed: dict[int, int] = {}
        for decision in outcome.decisions:
            if decision.decision_type not in (DecisionType.CONTINUITY, DecisionType.HANDOFF,
                                              DecisionType.REACQUIRE):
                continue
            global_id = decision.assigned_global_id
            observation = by_id.get(decision.observation_id)
            if global_id is None or observation is None or global_id not in self.identities:
                continue
            if global_id in claimed:
                # PLAN 2 §6.5: one identity, two spatially separate observations.
                self._quarantine(observation, global_id, timestamp, frame_sequence,
                                 "identity_claimed_twice")
                result.quarantined.append(observation.observation_id)
                result.collisions.append((global_id, observation.observation_id))
                continue
            claimed[global_id] = observation.observation_id
            self._apply_match(global_id, observation, decision, timestamp, frame_sequence,
                              overload)
            result.matched[global_id] = observation.observation_id
        for decision in outcome.decisions:
            if decision.decision_type is DecisionType.DEFER:
                result.deferred.append(decision.observation_id)
                self.events.append(timestamp, frame_sequence, IdentityEventType.DEFER, None,
                                   detail=decision.defer_reason,
                                   evidence={"observation": decision.observation_id,
                                             "margin": decision.margin,
                                             "score": decision.identity_score,
                                             "competing": str(decision.competing_global_ids)})
        for decision in outcome.decisions:
            if decision.decision_type is not DecisionType.NEW_CANDIDATE:
                continue
            observation = by_id.get(decision.observation_id)
            if observation is None:
                continue
            global_id, reason = self._consider_mint(observation, outcome, timestamp,
                                                    frame_sequence, overload)
            if global_id is None:
                result.blocked_mints.append((observation.observation_id, reason))
                self.events.append(timestamp, frame_sequence, IdentityEventType.MINT_BLOCKED,
                                   None, detail=reason,
                                   evidence={"observation": observation.observation_id})
            else:
                result.minted.append(global_id)
        self._promote(timestamp, frame_sequence, result)
        self._sweep(timestamp, frame_sequence, set(result.matched), result)
        self._duplicate_scan(timestamp, frame_sequence, result)
        return result

    # --- matching -----------------------------------------------------------
    def _apply_match(self, global_id: int, observation: FusedWorldDetection,
                     decision: AssociationDecision, timestamp: float, frame_sequence: int,
                     overload: bool) -> None:
        state = self.identities[global_id]
        kinematics: LagAwareKalman = state.kinematics
        kinematics.advance(observation.timestamp)
        covariance = np.asarray(observation.covariance, dtype=float)
        if overload:
            covariance = covariance * self.config_overload_gain
        kinematics.update(observation.position, covariance, timestamp=observation.timestamp)
        state.latest_world_position = kinematics.position
        state.latest_world_covariance = kinematics.position_covariance
        state.velocity = kinematics.velocity
        state.last_observed_timestamp = observation.timestamp
        state.last_observed_position = np.asarray(observation.position, dtype=float).copy()
        if state.origin_world_position is not None:
            disp = float(np.linalg.norm(state.last_observed_position - state.origin_world_position))
            state.max_displacement = max(state.max_displacement, disp)
        state.latest_camera = observation.primary_camera
        state.latest_footprint = observation.footprint
        state.footprint_area = (0.7 * state.footprint_area + 0.3 * observation.footprint_area
                                if state.footprint_area > 0 else observation.footprint_area)
        state.footprint_aspect = (0.7 * state.footprint_aspect
                                  + 0.3 * observation.footprint_aspect
                                  if state.footprint_aspect > 0 else observation.footprint_aspect)
        state.observation_count += 1
        state.missing_since = None
        for camera_id in observation.contributing_cameras:
            state.last_camera_seen_at[camera_id] = observation.timestamp
        for camera_id, local_track_id in observation.local_track_ids:
            if local_track_id >= 0:
                self._local_track_owner[(camera_id, local_track_id)] = global_id
        if not observation.latent and observation.appearance is not None:
            state.appearance_gallery.unfreeze()
            state.appearance_gallery.add(observation.appearance, observation.timestamp,
                                         observation.quality)
        self._occlusion_pending.pop(global_id, None)
        self._note_corridor(global_id, observation)
        self._transition_on_match(state, observation, decision, timestamp, frame_sequence)

    def _note_corridor(self, global_id: int, observation: FusedWorldDetection) -> None:
        """Record that this identity was *observed* inside an exit corridor."""
        for camera_id in observation.contributing_cameras:
            zone = self.topology.zones.get(camera_id)
            if zone is None:
                continue
            for successor in zone.exit_polygons:
                if zone.in_exit_corridor(observation.position, successor):
                    self._corridor_seen.setdefault(global_id, {})[successor] = observation.timestamp

    @property
    def config_overload_gain(self) -> float:
        return 4.0

    def _transition_on_match(self, state: GlobalVehicleState, observation: FusedWorldDetection,
                             decision: AssociationDecision, timestamp: float,
                             frame_sequence: int) -> None:
        previous = state.lifecycle_state
        if previous is LifecycleState.PARKED:
            anchor = self._parked_position.get(state.global_id, state.latest_world_position)
            drift = float(np.linalg.norm(state.latest_world_position - anchor))
            speed = float(np.linalg.norm(state.velocity))
            unpark_min = getattr(self.config, "unpark_min_drift_m", 0.035)
            if drift > unpark_min and speed > self.config.v_max_world * 0.02:
                state.lifecycle_state = LifecycleState.ACTIVE
                state.appearance_gallery.unfreeze()
                self.events.append(timestamp, frame_sequence, IdentityEventType.UNPARK,
                                   state.global_id, detail=f"drift={drift:.2f}",
                                   camera_id=observation.primary_camera)
        elif previous is LifecycleState.PROVISIONAL:
            pass                                     # promotion happens in _promote
        else:
            state.lifecycle_state = LifecycleState.ACTIVE
        event_type = IdentityEventType.MATCH
        if decision.decision_type is DecisionType.HANDOFF:
            event_type = IdentityEventType.HANDOFF
        elif decision.decision_type is DecisionType.REACQUIRE or previous in (
                LifecycleState.TEMPORARILY_MISSING, LifecycleState.OCCLUDED):
            event_type = IdentityEventType.RECOVER
        self.events.append(timestamp, frame_sequence, event_type, state.global_id,
                           detail=f"{previous.value}->{state.lifecycle_state.value}",
                           camera_id=observation.primary_camera,
                           evidence={"observation": observation.observation_id,
                                     "score": decision.identity_score,
                                     "margin": decision.margin,
                                     "cameras": ",".join(observation.contributing_cameras)})

    # --- minting ------------------------------------------------------------
    def _consider_mint(self, observation: FusedWorldDetection, outcome, timestamp: float,
                       frame_sequence: int, overload: bool) -> tuple[int | None, str]:
        if overload:
            # PLAN 1 Phase 6 / PLAN 3 §6: overload may raise uncertainty, never mint.
            return None, "overload_mint_forbidden"
        if observation.latent:
            return None, "latent_observation"
        if observation.partial:
            # An incomplete blob (touching the image border, or far below the
            # expected vehicle footprint) is not evidence of a new vehicle.
            return None, "partial_observation"
        if observation.fusion_confidence < 0.45:
            return None, "low_confidence_candidate"
        if observation.footprint_area < 0.0010:
            return None, "sub_vehicle_footprint_area"
        feasible = outcome.feasible_identities(observation.observation_id)
        if feasible:
            return None, f"feasible_identity:{feasible}"
        blocker = self._grace_window_blocker(observation, timestamp)
        if blocker is not None:
            return None, f"grace_window_hypothesis:{blocker}"
        occluder = self._occlusion_blocker(observation, timestamp)
        if occluder is not None:
            return None, f"unresolved_occlusion:{occluder}"
        if self.config.require_entry_gate:
            if not self._has_entry_gate_evidence(observation):
                return None, "no_entry_gate_crossing"
        global_id = self._mint(observation, timestamp, frame_sequence)
        return global_id, "minted"

    def _has_entry_gate_evidence(self, observation: FusedWorldDetection) -> bool:
        """Check if this observation is near an entry gate zone."""
        from .geometry import point_in_polygon
        if not self.crossing_detector.gates:
            for camera_id in observation.contributing_cameras:
                zone = self.topology.zones.get(camera_id)
                if zone is None:
                    continue
                if zone.in_entry_corridor(observation.position, tolerance=0.5):
                    return True
            return False
        for gate in self.crossing_detector.gates:
            if gate.gate_type != "entry":
                continue
            if point_in_polygon(observation.position, gate.polygon):
                return True
        return False

    def _grace_window_blocker(self, observation: FusedWorldDetection,
                              timestamp: float) -> int | None:
        """PLAN 2 §6.3 — the central anti-fragmentation rule."""
        relaxed = copy.copy(self.association_config)
        relaxed.gate = self.association_config.gate * 3.0
        for view in self.views(observation.timestamp):
            state = self.identities[view.global_id]
            if state.missing_duration(timestamp) >= self.config.t_grace:
                continue
            components = compute_cost(view, observation, self.topology, relaxed, self.config,
                                      self.rho_seam)
            if components.feasible and components.identity_score >= self.config.tau_candidate:
                return view.global_id
        return None

    def _occlusion_blocker(self, observation: FusedWorldDetection,
                           timestamp: float) -> int | None:
        for global_id, pending_until in list(self._occlusion_pending.items()):
            state = self.identities.get(global_id)
            if state is None:
                self._occlusion_pending.pop(global_id, None)
                continue
            if timestamp > pending_until:
                continue
            elapsed = max(state.missing_duration(timestamp), 0.2)
            reach = self.config.v_max_world * elapsed + self.rho_seam + 2.0
            if float(np.linalg.norm(observation.position - state.latest_world_position)) <= reach:
                return global_id
        return None

    def _mint(self, observation: FusedWorldDetection, timestamp: float,
              frame_sequence: int) -> int:
        if self.store is not None:
            global_id = self.store.next_global_id(self.site_id)
            self._next_global_id = max(self._next_global_id, global_id)
        else:
            self._next_global_id += 1
            global_id = self._next_global_id
        kinematics = LagAwareKalman.create(
            self.kalman_config, observation.position, observation.timestamp,
            size=(1.0, 1.0), position_sigma=max(0.5, float(np.sqrt(
                np.trace(observation.covariance) / 2.0))), velocity_sigma=4.0)
        gallery = AppearanceGallery()
        gallery.add(observation.appearance, observation.timestamp, observation.quality)
        state = GlobalVehicleState(
            global_id=global_id, lifecycle_state=LifecycleState.PROVISIONAL,
            created_at=observation.timestamp, last_observed_timestamp=observation.timestamp,
            latest_world_position=kinematics.position,
            latest_world_covariance=kinematics.position_covariance,
            latest_camera=observation.primary_camera, latest_footprint=observation.footprint,
            footprint_area=observation.footprint_area,
            footprint_aspect=observation.footprint_aspect, observation_count=1,
            origin_world_position=np.asarray(observation.position, dtype=float).copy(),
            max_displacement=0.0,
            kinematics=kinematics, appearance_gallery=gallery)
        state.last_camera_seen_at[observation.primary_camera] = observation.timestamp
        self.identities[global_id] = state
        for camera_id, local_track_id in observation.local_track_ids:
            if local_track_id >= 0:
                self._local_track_owner[(camera_id, local_track_id)] = global_id
        self.events.append(timestamp, frame_sequence, IdentityEventType.MINT, global_id,
                           detail="provisional", camera_id=observation.primary_camera,
                           evidence={"observation": observation.observation_id,
                                     "position": float(observation.position[0])})
        if self.store is not None:
            self.store.save_identity(
                global_id=global_id,
                lifecycle_state=LifecycleState.PROVISIONAL.value,
                created_at=observation.timestamp,
                last_observed_at=observation.timestamp,
                primary_camera=observation.primary_camera,
                origin_pos=(float(observation.position[0]), float(observation.position[1])),
            )
        return global_id

    def _promote(self, timestamp: float, frame_sequence: int, result: IngestResult) -> None:
        for state in list(self.identities.values()):
            if state.lifecycle_state is not LifecycleState.PROVISIONAL:
                continue
            mature = (timestamp - state.created_at) >= self.config.t_maturity
            if not (mature and state.observation_count >= self.config.n_maturity):
                continue
            min_disp = getattr(self.config, "new_identity_min_displacement_m", 0.04)
            if state.max_displacement < min_disp:
                if (timestamp - state.created_at) > max(3.0 * self.config.t_maturity, 2.5):
                    self._retire(state, timestamp, frame_sequence, "static_noise_discarded", result, audit=False)
                continue
            state.lifecycle_state = LifecycleState.ACTIVE
            result.promoted.append(state.global_id)
            self.events.append(timestamp, frame_sequence, IdentityEventType.ACTIVATE,
                               state.global_id, detail="maturity_reached",
                               camera_id=state.latest_camera)
            if self.store is not None:
                self.store.save_identity(
                    global_id=state.global_id,
                    lifecycle_state=LifecycleState.ACTIVE.value,
                    created_at=state.created_at,
                    last_observed_at=timestamp,
                    primary_camera=state.latest_camera,
                    max_displacement=state.max_displacement,
                )

    # --- lifecycle sweep ----------------------------------------------------
    def _sweep(self, timestamp: float, frame_sequence: int, matched: set[int],
               result: IngestResult) -> None:
        for global_id, state in list(self.identities.items()):
            if global_id in matched:
                continue
            missing = state.missing_duration(timestamp)
            if state.lifecycle_state is LifecycleState.EXIT_CONFIRMED:
                self._retire(state, timestamp, frame_sequence, "exit_confirmed", result)
                continue
            if state.lifecycle_state is LifecycleState.PROVISIONAL:
                if missing > max(2.0 * self.config.t_maturity, 0.5):
                    self._retire(state, timestamp, frame_sequence, "provisional_expired", result,
                                 audit=False)
                continue
            if state.lifecycle_state is LifecycleState.PARKED:
                continue                     # a slot owner is never retired on time alone
            if missing <= 0.0:
                continue
            pending_occlusion = timestamp <= self._occlusion_pending.get(global_id, -1.0)
            target = (LifecycleState.OCCLUDED if pending_occlusion
                      else LifecycleState.TEMPORARILY_MISSING)
            if state.lifecycle_state is not target:
                state.lifecycle_state = target
                state.missing_since = state.last_observed_timestamp
                self.events.append(timestamp, frame_sequence,
                                   IdentityEventType.OCCLUDED if pending_occlusion
                                   else IdentityEventType.MISSING, global_id,
                                   detail=f"missing={missing:.3f}s")
            if not self.retention_ok(state, timestamp):
                self._retire(state, timestamp, frame_sequence,
                             f"retention_exhausted_after_{missing:.2f}s", result)

    def retention_ok(self, state: GlobalVehicleState, timestamp: float) -> bool:
        """PLAN 2 §6.4 — retention holds while *any* channel is still open."""
        missing = state.missing_duration(timestamp)
        if missing < self.config.t_max_missing:
            return True
        if state.slot_id is not None:
            return True
        if self._handoff_pending(state, timestamp):
            return True
        if state.appearance_gallery is not None and state.appearance_gallery.samples \
                and missing < self.config.t_retire_idle:
            return True
        return False

    def _handoff_pending(self, state: GlobalVehicleState, timestamp: float) -> bool:
        zone = self.topology.zones.get(state.latest_camera)
        if zone is None or not zone.exit_polygons:
            return False
        if not zone.in_exit_corridor(state.latest_world_position):
            return False
        horizon = max((edge.dt_max for (src, _), edge in self.topology.edges.items()
                       if src == state.latest_camera), default=self.config.t_grace)
        return state.missing_duration(timestamp) <= horizon

    def _retire(self, state: GlobalVehicleState, timestamp: float, frame_sequence: int,
                reason: str, result: IngestResult, audit: bool = True) -> None:
        state.lifecycle_state = LifecycleState.RETIRED
        self.identities.pop(state.global_id, None)
        self.retired_identities[state.global_id] = state
        self._occlusion_pending.pop(state.global_id, None)
        self._parked_position.pop(state.global_id, None)
        self._corridor_seen.pop(state.global_id, None)
        for key, owner in list(self._local_track_owner.items()):
            if owner == state.global_id:
                self._local_track_owner.pop(key, None)
        result.retired.append(state.global_id)
        if audit:
            self.events.append(timestamp, frame_sequence, IdentityEventType.RETIRE,
                               state.global_id, detail=reason)
        if self.store is not None:
            self.store.save_identity(
                global_id=state.global_id,
                lifecycle_state=state.lifecycle_state.value,
                created_at=state.created_at,
                last_observed_at=timestamp,
                primary_camera=state.latest_camera,
                slot_id=state.slot_id,
                max_displacement=state.max_displacement,
            )

    def _duplicate_scan(self, timestamp: float, frame_sequence: int,
                        result: IngestResult) -> None:
        """Two identities on one vehicle is a bug, but merging blindly is worse.

        Only a *provisional* duplicate is removed, and only after it has shadowed a
        mature identity for longer than the maturity period (PLAN 2 §6.5).
        """
        states = self.live()
        threshold = self.config.collision_separation / 3.0
        for i, a in enumerate(states):
            for b in states[i + 1:]:
                distance = float(np.linalg.norm(np.asarray(a.latest_world_position)
                                                - np.asarray(b.latest_world_position)))
                key = (min(a.global_id, b.global_id), max(a.global_id, b.global_id))
                if distance > threshold:
                    self._duplicate_watch.pop(key, None)
                    continue
                since = self._duplicate_watch.setdefault(key, timestamp)
                if timestamp - since < max(self.config.t_maturity * 2.0, 0.4):
                    continue
                provisional = [s for s in (a, b)
                               if s.lifecycle_state is LifecycleState.PROVISIONAL]
                other = b if provisional and provisional[0] is a else a
                if len(provisional) != 1:
                    self.events.append(timestamp, frame_sequence, IdentityEventType.COLLISION,
                                       a.global_id,
                                       detail=f"co_located_with_{b.global_id}_{distance:.2f}")
                    result.collisions.append((a.global_id, b.global_id))
                    continue
                self.events.append(timestamp, frame_sequence, IdentityEventType.ALIAS,
                                   provisional[0].global_id,
                                   detail=f"duplicate_of_{other.global_id}",
                                   evidence={"distance": distance})
                self._retire(provisional[0], timestamp, frame_sequence,
                             f"duplicate_of_{other.global_id}", result)
                self._duplicate_watch.pop(key, None)

    # --- external hooks -----------------------------------------------------
    def note_occlusion(self, camera_id: str, local_track_ids, timestamp: float,
                       frame_sequence: int, hold: float | None = None) -> list[int]:
        """A camera reports an unresolved occlusion group (PLAN 2 §7)."""
        hold = self.config.t_grace if hold is None else hold
        touched = []
        for local_track_id in local_track_ids:
            global_id = self._local_track_owner.get((camera_id, local_track_id))
            if global_id is None or global_id not in self.identities:
                continue
            state = self.identities[global_id]
            self._occlusion_pending[global_id] = timestamp + hold
            state.appearance_gallery.freeze("occlusion_group")
            if state.lifecycle_state in (LifecycleState.ACTIVE,
                                         LifecycleState.TEMPORARILY_MISSING):
                state.lifecycle_state = LifecycleState.OCCLUDED
                self.events.append(timestamp, frame_sequence, IdentityEventType.OCCLUDED,
                                   global_id, camera_id=camera_id, detail="occlusion_group")
            touched.append(global_id)
        return touched

    def _quarantine(self, observation: FusedWorldDetection, global_id: int, timestamp: float,
                    frame_sequence: int, reason: str) -> None:
        self.quarantined_observations.append({
            "observation_id": observation.observation_id, "global_id": global_id,
            "timestamp": timestamp, "reason": reason,
            "position": np.asarray(observation.position, dtype=float).tolist()})
        state = self.identities.get(global_id)
        if state is not None:
            state.quarantined = True
        self.events.append(timestamp, frame_sequence, IdentityEventType.QUARANTINE, global_id,
                           detail=reason, evidence={"observation": observation.observation_id})

    def mark_parked(self, global_id: int, slot_id: str, timestamp: float,
                    frame_sequence: int) -> None:
        state = self.identities.get(global_id)
        if state is None:
            return
        state.slot_id = slot_id
        state.appearance_gallery.freeze("parked")
        self._parked_position[global_id] = np.asarray(state.latest_world_position, dtype=float)
        if state.lifecycle_state is not LifecycleState.PARKED:
            state.lifecycle_state = LifecycleState.PARKED
            self.events.append(timestamp, frame_sequence, IdentityEventType.PARK, global_id,
                               detail=slot_id, camera_id=state.latest_camera)
        if self.store is not None:
            self.store.save_identity(
                global_id=global_id,
                lifecycle_state=LifecycleState.PARKED.value,
                created_at=state.created_at,
                last_observed_at=timestamp,
                slot_id=slot_id,
                primary_camera=state.latest_camera,
                max_displacement=state.max_displacement,
            )

    def release_slot(self, global_id: int, timestamp: float, frame_sequence: int) -> None:
        state = self.identities.get(global_id)
        if state is None:
            return
        slot_id, state.slot_id = state.slot_id, None
        self._parked_position.pop(global_id, None)
        if state.lifecycle_state is LifecycleState.PARKED:
            state.lifecycle_state = LifecycleState.ACTIVE
        self.events.append(timestamp, frame_sequence, IdentityEventType.UNPARK, global_id,
                           detail=str(slot_id))
        if self.store is not None:
            self.store.save_identity(
                global_id=global_id,
                lifecycle_state=LifecycleState.ACTIVE.value,
                created_at=state.created_at,
                last_observed_at=timestamp,
                slot_id=None,
                primary_camera=state.latest_camera,
                max_displacement=state.max_displacement,
            )

    def confirm_exit(self, global_id: int, timestamp: float, frame_sequence: int,
                     camera_id: str = "", detail: str = "exit_line") -> bool:
        state = self.identities.get(global_id)
        if state is None:
            return False
        state.lifecycle_state = LifecycleState.EXIT_CONFIRMED
        state.slot_id = None
        self.events.append(timestamp, frame_sequence, IdentityEventType.EXIT, global_id,
                           detail=detail, camera_id=camera_id)
        return True

    def alias(self, primary: int, secondary: int, timestamp: float, frame_sequence: int,
              evidence: dict | None = None) -> bool:
        """Explicit, audited identity stitch (PLAN 1 Phase 5 work item 2)."""
        keep = self.identities.get(primary)
        drop = self.identities.get(secondary)
        if keep is None or drop is None or primary == secondary:
            return False
        keep.session_ids = tuple(dict.fromkeys(keep.session_ids + drop.session_ids))
        for sample, moment in zip(drop.appearance_gallery.samples,
                                  drop.appearance_gallery.timestamps):
            keep.appearance_gallery.add(sample, moment)
        self.events.append(timestamp, frame_sequence, IdentityEventType.ALIAS, primary,
                           detail=f"absorbed_{secondary}", evidence=evidence or {})
        if self.store is not None:
            self.store.save_alias(secondary, primary, reason=f"absorbed_{secondary}")
        result = IngestResult()
        self._retire(drop, timestamp, frame_sequence, f"aliased_into_{primary}", result)
        return True

    def bind_session(self, global_id: int, session_id: str, timestamp: float,
                     frame_sequence: int) -> bool:
        state = self.identities.get(global_id)
        if state is None:
            return False
        if session_id not in state.session_ids:
            state.session_ids = state.session_ids + (session_id,)
        self.events.append(timestamp, frame_sequence, IdentityEventType.SESSION_BIND, global_id,
                           detail=session_id)
        return True

    def note_overload(self, timestamp: float, frame_sequence: int, detail: str) -> None:
        for state in self.live():
            state.latest_world_covariance = np.asarray(state.latest_world_covariance,
                                                       dtype=float) * 1.5
        self.events.append(timestamp, frame_sequence, IdentityEventType.OVERLOAD, None,
                           detail=detail)

    def reset(self, timestamp: float, frame_sequence: int) -> int:
        """Hard reset: retire all identities but PRESERVE the GID counter.

        The plan mandates "GID là số tăng đơn điệu theo site_id, không bao giờ
        tái sử dụng" — so the counter is *never* reset to zero.
        """
        count = len(self.identities)
        ids = tuple(sorted(self.identities))
        self.events.append(timestamp, frame_sequence, IdentityEventType.RESET, None,
                           detail=f"retired_identities={count}",
                           evidence={"global_ids": ",".join(str(value) for value in ids)})
        self.identities.clear()
        self.retired_identities.clear()
        self.quarantined_observations.clear()
        self._local_track_owner.clear()
        self._corridor_seen.clear()
        self._occlusion_pending.clear()
        self._parked_position.clear()
        self._duplicate_watch.clear()
        # IMPORTANT: do NOT reset self._next_global_id — GIDs are monotonic per site
        return count

    def soft_reset(self, timestamp: float, frame_sequence: int) -> dict:
        """Soft reset: keep parked identities and GID counter, recover moving identities.

        Per the plan:
        - Moving identity → RECOVERY_PENDING with increased covariance
        - Parked identity → unchanged (keeps slot ownership)
        - GID counter → preserved (never reset)
        - PROVISIONAL → retired immediately
        """
        retired_count = 0
        recovery_count = 0
        parked_kept = 0
        result = IngestResult()
        for state in list(self.identities.values()):
            if state.lifecycle_state is LifecycleState.PROVISIONAL:
                self._retire(state, timestamp, frame_sequence, "soft_reset_provisional", result,
                             audit=False)
                retired_count += 1
            elif state.lifecycle_state is LifecycleState.PARKED:
                parked_kept += 1
            elif state.lifecycle_state.is_live:
                state.lifecycle_state = LifecycleState.RECOVERY_PENDING
                # Increase covariance to reflect uncertainty after restart
                state.latest_world_covariance = np.asarray(
                    state.latest_world_covariance, dtype=float) * 4.0
                state.missing_since = timestamp
                recovery_count += 1
        self._local_track_owner.clear()
        self._corridor_seen.clear()
        self._occlusion_pending.clear()
        self._duplicate_watch.clear()
        self.events.append(timestamp, frame_sequence, IdentityEventType.RESET, None,
                           detail=f"soft_reset: parked_kept={parked_kept} "
                                  f"recovery={recovery_count} retired={retired_count}")
        return {"soft_reset": True, "parked_kept": parked_kept,
                "recovery_pending": recovery_count, "retired": retired_count}

    def restore_from_store(self) -> dict[str, int]:
        """Restore parked identities and recovery pending states from SQLite on startup."""
        if self.store is None:
            return {"parked": 0, "recovery_pending": 0}
        parked = self.store.load_parked_identities()
        active = self.store.load_active_identities()
        restored_parked = 0
        restored_recovery = 0
        for row in parked:
            gid = int(row["global_id"])
            slot_id = str(row["slot_id"])
            pos = np.array([float(row["origin_x"] or 0.0), float(row["origin_y"] or 0.0)])
            kinematics = LagAwareKalman.create(self.kalman_config, pos, float(row["last_observed_at"]))
            state = GlobalVehicleState(
                global_id=gid,
                lifecycle_state=LifecycleState.PARKED,
                created_at=float(row["created_at"]),
                last_observed_timestamp=float(row["last_observed_at"]),
                latest_world_position=pos,
                latest_world_covariance=np.eye(2) * 0.01,
                latest_camera=str(row["primary_camera"] or ""),
                slot_id=slot_id,
                kinematics=kinematics,
                appearance_gallery=AppearanceGallery(),
            )
            self.identities[gid] = state
            self._parked_position[gid] = pos
            restored_parked += 1
        for row in active:
            gid = int(row["global_id"])
            if gid in self.identities:
                continue
            pos = np.array([float(row["origin_x"] or 0.0), float(row["origin_y"] or 0.0)])
            kinematics = LagAwareKalman.create(self.kalman_config, pos, float(row["last_observed_at"]))
            state = GlobalVehicleState(
                global_id=gid,
                lifecycle_state=LifecycleState.RECOVERY_PENDING,
                created_at=float(row["created_at"]),
                last_observed_timestamp=float(row["last_observed_at"]),
                latest_world_position=pos,
                latest_world_covariance=np.eye(2) * 1.0,
                latest_camera=str(row["primary_camera"] or ""),
                missing_since=time.time(),
                kinematics=kinematics,
                appearance_gallery=AppearanceGallery(),
            )
            self.identities[gid] = state
            restored_recovery += 1
        return {"parked": restored_parked, "recovery_pending": restored_recovery}
