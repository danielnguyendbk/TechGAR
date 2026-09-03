"""The ten-stage pipeline, wired (PLAN 1 §2-§3).

    ingestion -> normalisation -> detection -> local tracking -> projection
      -> fusion -> association -> identity registry -> slots -> dispatch

Ordering rules the plan is emphatic about and that this file enforces:
fusion happens *before* association (two cameras seeing one vehicle become one
measurement first, §6.5), the registry is the only component that mints or retires
identities, and the snapshot publisher is a pure consumer at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import numpy as np

from .association import AssociationOutcome, TopologyConstrainedAssociator
from .config import TechgarConfig
from .contracts import FrameRecord, LocalDetection, LocalTrackObservation, MeasurementSource
from .detection import LocalDetector
from .fusion import WorldFusion
from .ingestion import DualStreamIngestion
from .local_tracker import LocalTracker
from .normalization import EnvironmentalNormalizer, NormalizedFrame
from .perf import LatencyTracker, OverloadMonitor, StageTimer, SubscriberGate, ThroughputMeter
from .profile import CameraProfile
from .projection import WorldProjector
from .registry import GlobalIdentityRegistry, IngestResult
from .sessions import SessionRegistry
from .slot_engine import SlotEvent, SlotOccupancyEngine, VehicleFootprintView, VisionSlotVote
from .snapshot import RuntimeSnapshot, SnapshotPublisher
from .states import LifecycleState
from .topology import CameraTopology
from .vision_occupancy import VisionOccupancyDetector, merge_camera_votes
from .world_contracts import FusedWorldDetection, WorldDetection


@dataclass
class StepResult:
    pair_sequence: int
    timestamp: float
    cameras: tuple[str, ...]
    detections: dict[str, list[LocalDetection]] = field(default_factory=dict)
    observations: list[LocalTrackObservation] = field(default_factory=list)
    world: list[WorldDetection] = field(default_factory=list)
    fused: list[FusedWorldDetection] = field(default_factory=list)
    outcome: AssociationOutcome | None = None
    ingest: IngestResult | None = None
    identity_bindings: dict[str, dict[int, int]] = field(default_factory=dict)
    slot_events: list[SlotEvent] = field(default_factory=list)
    snapshot: RuntimeSnapshot | None = None
    processing_seconds: float = 0.0
    overload: bool = False
    skew: float = 0.0


class TechgarPipeline:
    def __init__(self, profiles: dict[str, CameraProfile], topology: CameraTopology,
                 slots: dict[str, np.ndarray] | None = None,
                 config: TechgarConfig | None = None,
                 visualization_encoder: Callable[[StepResult], object] | None = None,
                 pixel_slots: dict[str, dict[str, np.ndarray]] | None = None) -> None:
        self.config = config or TechgarConfig()
        cfg = self.config
        self.profiles = profiles
        self.topology = topology
        cameras = tuple(sorted(profiles))
        self.ingestion = DualStreamIngestion(cameras, cfg.ingestion)
        self.normalizers = {cam: EnvironmentalNormalizer(cam, cfg.threshold, cfg.background,
                                                         cfg.motion, cfg.shadow)
                            for cam in cameras}
        self.detectors = {cam: LocalDetector(self._tuned(profiles[cam]), cfg.detection, cfg.motion)
                          for cam in cameras}
        self.trackers = {cam: LocalTracker(profiles[cam], cfg.local_track, cfg.pixel_kalman,
                                           self._camera_detection_config(profiles[cam]))
                         for cam in cameras}
        self.projectors = {cam: WorldProjector(profiles[cam], topology, cfg.projection)
                           for cam in cameras}
        self.fusion = WorldFusion(topology, cfg.fusion)
        self.associator = TopologyConstrainedAssociator(topology, cfg.association, cfg.identity,
                                                        cfg.projection.rho_seam)
        self.registry = GlobalIdentityRegistry(cfg.identity, cfg.world_kalman, cfg.association,
                                               topology, cfg.projection.rho_seam)
        self.slot_engine = SlotOccupancyEngine(slots or {}, cfg.slot)
        self.sessions = SessionRegistry(self.registry, cfg.session)
        self.publisher = SnapshotPublisher(cfg.identity.t_display_hold)
        self.timer = StageTimer()
        self.latency = LatencyTracker()
        self.overload = OverloadMonitor(cfg.perf)
        self.throughput = ThroughputMeter()
        self.subscribers = SubscriberGate()
        self.visualization_encoder = visualization_encoder
        self.latest_visualization: object | None = None
        self.history: list[StepResult] = []
        self.keep_history = True
        self._last_slot_run = float("-inf")
        self._frame_sequence = 0
        # Stage 9 vision fusion: pixel-content occupancy per camera, independent
        # of tracking.  A parked vehicle keeps the slot occupied on this alone.
        self.vision_detectors: dict[str, VisionOccupancyDetector] = {}
        self.pixel_slots: dict[str, dict[str, np.ndarray]] = pixel_slots or {}
        self.tracking_masks: dict[str, np.ndarray] = {}
        if pixel_slots:
            for camera_id, camera_slots in pixel_slots.items():
                if camera_slots:
                    self.vision_detectors[camera_id] = VisionOccupancyDetector(camera_slots)
        self._last_frames: dict[str, object] = {}

    def get_active_tracking_mask(self, camera_id: str) -> np.ndarray | None:
        """Return tracking mask for camera_id (constrained by camera ROI)."""
        return self.tracking_masks.get(camera_id)



    def _camera_detection_config(self, profile: CameraProfile):
        import copy
        cfg = copy.copy(self.config.detection)
        cfg.expected_vehicle_area = profile.expected_vehicle_area
        return cfg

    def _tuned(self, profile: CameraProfile):
        return profile

    def __post_init__(self) -> None:  # pragma: no cover - dataclass parity helper
        pass

    # --- driving ------------------------------------------------------------
    def submit(self, frame: FrameRecord) -> list[StepResult]:
        self.timer.start("s1_ingestion")
        self.ingestion.submit(frame)
        self.timer.stop("s1_ingestion")
        results = []
        while True:
            pair = self.ingestion.try_pair()
            if pair is None:
                break
            results.append(self.process_pair(pair))
        return results

    def run(self, frames: Iterable[FrameRecord]) -> Iterator[StepResult]:
        for frame in frames:
            for result in self.submit(frame):
                yield result
        for pair in self.ingestion.flush():
            yield self.process_pair(pair)

    # --- one synchronisation cycle -----------------------------------------
    def process_pair(self, pair) -> StepResult:
        import time
        started = time.perf_counter()
        timestamp = pair.timestamp
        self._frame_sequence += 1
        result = StepResult(pair.pair_sequence, timestamp, pair.camera_ids, skew=pair.timestamp_skew)
        world: list[WorldDetection] = []
        for camera_id, frame in sorted(pair.frames.items()):
            if self.vision_detectors:
                self._last_frames[camera_id] = frame
            with self.timer.measure("s2_normalization"):
                normalized = self.normalizers[camera_id].process(frame)
            with self.timer.measure("s3_detection"):
                active_mask = self.get_active_tracking_mask(camera_id)
                detections = self.detectors[camera_id].detect(
                    normalized, self.trackers[camera_id].predictions(),
                    tracking_mask=active_mask)
            with self.timer.measure("s4_local_tracking"):
                observations = self.trackers[camera_id].step(normalized, detections)
            result.detections[camera_id] = detections
            result.observations.extend(observations)
            with self.timer.measure("s5_projection"):
                world.extend(self._project(camera_id, observations, normalized))
            self._report_occlusions(camera_id, observations, timestamp)
        result.world = world
        with self.timer.measure("s6_fusion"):
            fused = self.fusion.fuse(world) if len(pair.frames) > 1 or world else []
        result.fused = fused
        # Declare an overload before identity association so the same slow frame
        # cannot mint a Global ID and only be marked overloaded afterwards.
        overload = self.overload.observe(time.perf_counter() - started)
        with self.timer.measure("s7_association"):
            self.registry.note_active_local_observations(fused)
            views = self.registry.views(timestamp)
            outcome = self.associator.associate(
                views, fused,
                owner_constraints=self.registry.owner_constraints(fused),
                blocked_observations=self.registry.suppressed_local_observations(fused),
                forbidden_pairs=self.registry.forbidden_binding_pairs(fused, timestamp),
            )
        result.outcome = outcome
        with self.timer.measure("s8_registry"):
            result.ingest = self.registry.ingest(fused, outcome, timestamp, self._frame_sequence,
                                                 overload=overload)
            for observation in result.observations:
                owner = self.registry.active_owner_of_local_track(
                    observation.camera_id, observation.local_track_id
                )
                if owner is not None:
                    result.identity_bindings.setdefault(observation.camera_id, {})[
                        observation.local_track_id
                    ] = owner
            self._confirm_exits(timestamp)
        with self.timer.measure("s9_slots"):
            result.slot_events = self._run_slots(timestamp)
        with self.timer.measure("s10_dispatch"):
            self.sessions.sweep(timestamp, self._frame_sequence)
            health = {cam: {"frames": h.frames_received, "dropped_stale": h.frames_dropped_stale,
                            "replaced": h.frames_replaced,
                            "last_timestamp": h.last_timestamp}
                      for cam, h in self.ingestion.health.items()}
            processing = time.perf_counter() - started
            published_at = timestamp + processing
            latency = self.latency.record(timestamp, published_at, processing)
            result.snapshot = self.publisher.publish(
                list(self.registry.identities.values()), self.slot_engine.states, timestamp,
                health, self.registry.events, {"e2e": latency, "processing": processing},
                overload=overload, published_at=published_at,
                slot_layout=self.slot_engine.slots)
            if self.subscribers.should_encode():
                self.latest_visualization = (
                    self.visualization_encoder(result)
                    if self.visualization_encoder is not None else result.snapshot.to_json()
                )
        result.processing_seconds = time.perf_counter() - started
        result.overload = self.overload.observe(result.processing_seconds)
        if result.overload:
            self.registry.note_overload(timestamp, self._frame_sequence,
                                        f"processing={result.processing_seconds * 1000:.0f}ms")
        self.throughput.tick(timestamp)
        if self.keep_history:
            self.history.append(result)
        return result

    # --- stage helpers ------------------------------------------------------
    def _project(self, camera_id: str, observations: list[LocalTrackObservation],
                 normalized: NormalizedFrame) -> list[WorldDetection]:
        projector = self.projectors[camera_id]
        quality = normalized.quality.quality
        gain = self.overload.uncertainty_gain
        world = []
        for observation in observations:
            if observation.latent or observation.source is MeasurementSource.COAST:
                continue                   # a coasting track proposes nothing
            detection = observation.extras.get("detection")
            if detection is None:
                detection = self._synthesise(observation)
            # Pixel velocity -> world velocity through the projection Jacobian, so the
            # direction cost of PLAN 2 §4.2 compares world-frame headings.
            jacobian = self.profiles[camera_id].calibration.jacobian(
                float(observation.ground_anchor[0]), float(observation.ground_anchor[1]))
            velocity = jacobian @ np.asarray(observation.motion_vector, dtype=float).reshape(2)
            world.append(projector.project_detection(
                detection, environment_quality=quality,
                local_track_id=observation.local_track_id, latent=observation.latent,
                occlusion_group_id=observation.occlusion_group_id,
                velocity=velocity, uncertainty_gain=gain))
        return world

    def _synthesise(self, observation: LocalTrackObservation) -> LocalDetection:
        """Wrap a template-recovery measurement as a detection for projection."""
        bbox = observation.extras.get("bbox", observation.predicted_bbox)
        return LocalDetection(
            camera_id=observation.camera_id, timestamp=observation.timestamp,
            frame_sequence=observation.frame_sequence, bbox=np.asarray(bbox, dtype=float),
            confidence=max(observation.confidence, self.config.local_track.template_confidence),
            local_center=np.asarray(observation.predicted_bbox[:2]) * 0.0
            + 0.5 * (np.asarray(bbox[:2]) + np.asarray(bbox[2:])),
            ground_anchor=observation.ground_anchor,
            footprint_pixels=observation.footprint_pixels, mask_area=0.0,
            quality_score=max(observation.quality, 0.2),
            occlusion_group_candidate=False,
            appearance=observation.appearance_reference, detection_id=-1)

    def _report_occlusions(self, camera_id: str, observations: list[LocalTrackObservation],
                           timestamp: float) -> None:
        latent = [o.local_track_id for o in observations if o.latent]
        if latent:
            self.registry.note_occlusion(camera_id, latent, timestamp, self._frame_sequence)

    def _run_slots(self, timestamp: float) -> list[SlotEvent]:
        """Slot analysis runs at a lower rate than tracking (PLAN 1 Phase 6.6)."""
        if not self.slot_engine.slots:
            return []
        if timestamp - self._last_slot_run < self.config.perf.slot_period:
            return []
        self._last_slot_run = timestamp
        # --- Vision fusion channel (stage 9): pixel content per slot ---
        if self.vision_detectors and self.config.slot.enable_vision_fusion:
            try:
                import cv2 as _cv2
                per_camera = []
                for camera_id, detector in sorted(self.vision_detectors.items()):
                    frame = self._last_frames.get(camera_id)
                    if frame is None:
                        continue
                    raw_img = frame.image if hasattr(frame, "image") else frame
                    if raw_img is None:
                        continue
                    frame_array = raw_img if isinstance(raw_img, np.ndarray) else np.asarray(raw_img)
                    if frame_array.dtype != np.uint8:
                        frame_array = _cv2.convertScaleAbs(frame_array)
                    per_camera.append(detector.detect(frame_array))
                merged = merge_camera_votes(per_camera)
                self.slot_engine.update_vision(
                    {slot_id: VisionSlotVote(slot_id, vote.occupied, vote.evidence, vote.ready)
                     for slot_id, vote in merged.items()},
                    timestamp)
            except Exception:  # pragma: no cover - vision must never break tracking
                pass
        vehicles = []
        for state in self.registry.identities.values():
            # A provisional hypothesis has not yet passed the identity maturity
            # gate and therefore cannot reserve or own physical infrastructure.
            # Letting startup/background transients claim slots leaves orphaned
            # occupancy after the hypothesis disappears (real M04 replay found
            # 27 occupied slots with only one published Global ID).
            if (not state.lifecycle_state.is_live
                    or state.lifecycle_state is LifecycleState.PROVISIONAL
                    or state.latest_footprint is None):
                continue
            observed = abs(state.last_observed_timestamp - timestamp) <= self.config.perf.slot_period
            # A coasting identity does not provide a current velocity measurement.
            # Keep its last positive footprint evidence during claim_grace, but do
            # not let a stale pre-occlusion speed veto an otherwise complete arrival
            # claim (Phase 4: interrupted tracking may still complete a claim).
            slot_velocity = (np.asarray(state.velocity, dtype=float)
                             if observed else np.zeros(2, dtype=float))
            vehicles.append(VehicleFootprintView(
                global_id=state.global_id,
                footprint=np.asarray(state.latest_footprint, dtype=float),
                position=np.asarray(state.latest_world_position, dtype=float),
                velocity=slot_velocity,
                quality=1.0 if observed else 0.6, observed=observed,
                parked_hint=state.lifecycle_state is LifecycleState.PARKED))
        events = self.slot_engine.update(vehicles, timestamp, self._frame_sequence)
        for event in events:
            if event.kind == "occupied" and event.global_id is not None:
                self.registry.mark_parked(event.global_id, event.slot_id, timestamp,
                                         self._frame_sequence)
            elif event.kind == "released" and event.global_id is not None:
                state = self.registry.identities.get(event.global_id)
                if state is not None and state.slot_id == event.slot_id:
                    self.registry.release_slot(event.global_id, timestamp, self._frame_sequence)
        return events

    def _confirm_exits(self, timestamp: float) -> None:
        """Retire only on evidence of physical exit (PLAN 1 stage 8 logic 5)."""
        for state in list(self.registry.identities.values()):
            if state.lifecycle_state not in (LifecycleState.TEMPORARILY_MISSING,
                                             LifecycleState.OCCLUDED):
                continue
            if state.slot_id is not None:
                continue
            missing = state.missing_duration(timestamp)
            if missing < self.config.identity.t_grace:
                continue
            gate = self.topology.in_exit_line(state.latest_world_position, state.latest_camera)
            if gate is None:
                continue
            self.registry.confirm_exit(state.global_id, timestamp, self._frame_sequence,
                                       camera_id=state.latest_camera,
                                       detail=f"exit_line:{gate}")

    # --- world-level driving (benchmarks with exact coordinates) -------------
    def step_from_detections(self, detections: dict[str, list[LocalDetection]], timestamp: float,
                            environment_quality: float = 1.0) -> StepResult:
        """Enter the pipeline at stage 4 with detections supplied directly.

        Used by the PLAN 3 scenario suite where ground truth is stated as exact
        world coordinates: stages 1-3 are bypassed, stages 4-10 run unchanged.
        """
        import time
        started = time.perf_counter()
        self._frame_sequence += 1
        result = StepResult(self._frame_sequence, timestamp, tuple(sorted(detections)))
        world: list[WorldDetection] = []
        for camera_id, camera_detections in sorted(detections.items()):
            frame = self._stub_frame(camera_id, timestamp, environment_quality)
            observations = self.trackers[camera_id].step(frame, camera_detections)
            result.detections[camera_id] = camera_detections
            result.observations.extend(observations)
            world.extend(self._project(camera_id, observations, frame))
            self._report_occlusions(camera_id, observations, timestamp)
        result.world = world
        fused = self.fusion.fuse(world)
        result.fused = fused
        self.registry.note_active_local_observations(fused)
        views = self.registry.views(timestamp)
        outcome = self.associator.associate(
            views, fused,
            owner_constraints=self.registry.owner_constraints(fused),
            blocked_observations=self.registry.suppressed_local_observations(fused),
            forbidden_pairs=self.registry.forbidden_binding_pairs(fused, timestamp),
        )
        result.outcome = outcome
        result.ingest = self.registry.ingest(fused, outcome, timestamp, self._frame_sequence,
                                             overload=self.overload.active)
        for observation in result.observations:
            owner = self.registry.active_owner_of_local_track(
                observation.camera_id, observation.local_track_id
            )
            if owner is not None:
                result.identity_bindings.setdefault(observation.camera_id, {})[
                    observation.local_track_id
                ] = owner
        self._confirm_exits(timestamp)
        result.slot_events = self._run_slots(timestamp)
        self.sessions.sweep(timestamp, self._frame_sequence)
        processing = time.perf_counter() - started
        self.latency.record(timestamp, timestamp + processing, processing)
        result.snapshot = self.publisher.publish(
            list(self.registry.identities.values()), self.slot_engine.states, timestamp,
            {}, self.registry.events, {"processing": processing},
            overload=self.overload.active, published_at=timestamp + processing,
            slot_layout=self.slot_engine.slots)
        result.processing_seconds = processing
        result.overload = self.overload.active
        self.throughput.tick(timestamp)
        if self.keep_history:
            self.history.append(result)
        return result

    def _stub_frame(self, camera_id: str, timestamp: float, quality: float) -> NormalizedFrame:
        from .contracts import EnvironmentQuality
        profile = self.profiles[camera_id]
        shape = (profile.height, profile.width)
        blank = np.zeros(shape, dtype=np.float32)
        return NormalizedFrame(
            camera_id=camera_id, timestamp=timestamp, frame_sequence=self._frame_sequence,
            gray=blank, color=None, foreground=np.zeros(shape, dtype=bool),
            shadow=np.zeros(shape, dtype=bool), candidates=np.zeros(shape, dtype=bool),
            background=blank, tau_map=blank,
            quality=EnvironmentQuality(camera_id, timestamp, 1.0, 100.0, 0.0, 0.0, 0.0, 10.0))

    # --- reporting ----------------------------------------------------------
    def performance_report(self) -> dict:
        return {"stages": self.timer.report(), "latency": self.latency.report(),
                "throughput": self.throughput.report(),
                "overload_episodes": self.overload.episodes,
                "ingestion": {"pairs": self.ingestion.stats.pairs,
                              "complete_pairs": self.ingestion.stats.complete_pairs,
                              "single_emissions": self.ingestion.stats.single_emissions,
                              "skew_rejections": self.ingestion.stats.skew_rejections,
                              "max_skew": self.ingestion.stats.max_skew},
                "video_encodes": self.subscribers.encodes,
                "video_skipped": self.subscribers.skipped}

    def reset_identities(self, include_sessions: bool = False,
                         timestamp: float | None = None) -> dict:
        """Atomically reset identity, local tracking and parking runtime state."""
        import time
        moment = time.time() if timestamp is None else float(timestamp)
        self._frame_sequence += 1
        retired = self.registry.reset(moment, self._frame_sequence)
        for tracker in self.trackers.values():
            tracker.tracks.clear()
            tracker.groups.clear()
            tracker._next_track_id = 0
            tracker._next_group_id = 0
        self.slot_engine = SlotOccupancyEngine(self.slot_engine.slots, self.config.slot)
        if include_sessions:
            self.sessions.sessions.clear()
        else:
            for session in self.sessions.sessions.values():
                if session.state != "closed":
                    session.state = "orphan"
                    session.orphaned_at = moment
                    session.global_id = None
        self.publisher.last = None
        return {"reset": True, "retired_identities": retired,
                "include_sessions": include_sessions}
