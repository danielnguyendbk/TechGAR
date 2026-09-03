"""Stage 4 — single-camera trajectory tracking, lag-resilient by construction.

PLAN 1 stage 4 logic, in order: predict to the current timestamp, associate
high-confidence detections first and low-confidence ones second, keep a
*time-based* missed counter, hold tracks through gaps with prediction plus
appearance plus size plus direction, keep both hypotheses alive when one detection
covers two tracks, and never create a new track while an existing one still has a
plausible re-acquisition hypothesis.
"""

from __future__ import annotations

import numpy as np

from .appearance import embed
from .assignment import solve_assignment
from .config_vision import DetectionConfig, KalmanConfig, LocalTrackConfig
from .contracts import LocalDetection, LocalTrackObservation, LocalTrackState, MeasurementSource
from .detection import TrackPrediction
from .geometry import bbox_to_polygon, polygon_iou
from .kalman import LagAwareKalman
from .linalg import CHI2_2DOF
from .local_track import LocalTrack, OcclusionGroup
from .normalization import NormalizedFrame
from .profile import CameraProfile
from .template import extract_template, match_template


class LocalTracker:
    def __init__(self, profile: CameraProfile, config: LocalTrackConfig | None = None,
                 kalman: KalmanConfig | None = None,
                 detection: DetectionConfig | None = None) -> None:
        self.profile = profile
        self.camera_id = profile.camera_id
        self.config = config or LocalTrackConfig()
        self.kalman_config = kalman or KalmanConfig()
        self.detection_config = detection or DetectionConfig()
        self.tracks: dict[int, LocalTrack] = {}
        self.groups: dict[int, OcclusionGroup] = {}
        self._next_track_id = 0
        self._next_group_id = 0
        self.decision_logs: list[dict] = []

    # --- helpers ------------------------------------------------------------
    def predictions(self) -> list[TrackPrediction]:
        return [TrackPrediction(track.local_track_id, track.predicted_bbox(), track.center)
                for track in self.tracks.values() if track.alive]

    def _new_track(self, detection: LocalDetection) -> LocalTrack:
        self._next_track_id += 1
        width = float(detection.bbox[2] - detection.bbox[0])
        height = float(detection.bbox[3] - detection.bbox[1])
        kalman = LagAwareKalman.create(self.kalman_config, detection.local_center,
                                      detection.timestamp, size=(width, height),
                                      position_sigma=3.0, velocity_sigma=60.0)
        track = LocalTrack(self._next_track_id, self.camera_id, kalman, detection.timestamp,
                           detection.timestamp, bbox=np.asarray(detection.bbox, dtype=float),
                           last_detection=detection)
        self.tracks[track.local_track_id] = track
        return track

    def _association_cost(self, track: LocalTrack, detection: LocalDetection) -> float:
        seam = 1.0 - self.profile.border_distance(detection.local_center) / max(
            self.profile.image_diagonal * 0.25, 1.0)
        r = track.filter.measurement_covariance(detection.confidence,
                                               1.0 if detection.occlusion_group_candidate else 0.0,
                                               max(0.0, seam))
        distance = track.filter.gate_distance(detection.local_center, r)
        if distance >= CHI2_2DOF[self.config.gate_confidence]:
            return float("inf")
        area = max((detection.bbox[2] - detection.bbox[0])
                   * (detection.bbox[3] - detection.bbox[1]), 1.0)
        geometry = abs(np.log(area / max(track.area, 1.0)))
        appearance = 0.0
        if detection.appearance is not None and track.gallery.samples:
            appearance = track.gallery.cost(detection.appearance)
        return float(distance + 2.0 * geometry + 4.0 * appearance)

    def _detect_groups(self, detections: list[LocalDetection], timestamp: float
                       ) -> dict[int, OcclusionGroup]:
        """Find detections that cover two or more live predictions."""
        groups: dict[int, OcclusionGroup] = {}
        live = [t for t in self.tracks.values() if t.alive]
        for detection in detections:
            covered = []
            box = bbox_to_polygon(detection.bbox)
            for track in live:
                inside = (detection.bbox[0] <= track.center[0] <= detection.bbox[2]
                          and detection.bbox[1] <= track.center[1] <= detection.bbox[3])
                overlap = polygon_iou(box, bbox_to_polygon(track.predicted_bbox()))
                if inside or overlap >= self.config.merged_coverage:
                    covered.append(track)
            if len(covered) < 2:
                continue
            self._next_group_id += 1
            group = OcclusionGroup(self._next_group_id, self.camera_id, detection.detection_id,
                                   tuple(t.local_track_id for t in covered),
                                   np.asarray(detection.bbox, dtype=float), timestamp, timestamp)
            groups[detection.detection_id] = group
            self.groups[group.group_id] = group
        return groups

    # --- main loop ----------------------------------------------------------
    def step(self, frame: NormalizedFrame, detections: list[LocalDetection]
             ) -> list[LocalTrackObservation]:
        cfg = self.config
        now = frame.timestamp
        for track in self.tracks.values():
            if track.alive:
                track.filter.advance(now)
        for detection in detections:
            if detection.appearance is None and frame.color is not None:
                detection.appearance = embed(frame.color, detection.bbox)

        groups = self._detect_groups(detections, now)
        grouped_tracks = {tid for group in groups.values() for tid in group.track_ids}
        for detection_id, group in groups.items():
            for track_id in group.track_ids:
                self.tracks[track_id].mark_merged(group.group_id, now)

        free_detections = [d for d in detections if d.detection_id not in groups]
        candidates = [t for t in self.tracks.values() if t.alive
                      and t.local_track_id not in grouped_tracks]
        matched_tracks: dict[int, LocalDetection] = {}
        for tier in (1, 2):
            tier_detections = [d for d in free_detections
                               if (d.confidence >= self.detection_config.high_confidence)
                               == (tier == 1)]
            open_tracks = [t for t in candidates if t.local_track_id not in matched_tracks]
            if not tier_detections or not open_tracks:
                continue
            cost = np.full((len(open_tracks), len(tier_detections)), np.inf)
            for i, track in enumerate(open_tracks):
                for j, detection in enumerate(tier_detections):
                    cost[i, j] = self._association_cost(track, detection)
            for i, j in solve_assignment(cost):
                matched_tracks[open_tracks[i].local_track_id] = tier_detections[j]

        assigned_detection_ids = {d.detection_id for d in matched_tracks.values()}
        for track_id, detection in matched_tracks.items():
            self._apply_measurement(self.tracks[track_id], detection, frame)

        # A relaxed lost-track recovery is still a frame-level bipartite
        # assignment.  The previous implementation performed this greedily in
        # _maybe_spawn, so two unassigned detections could update the same track
        # in one frame and the result depended on detection iteration order.
        remaining_detections = [d for d in free_detections
                                if d.detection_id not in assigned_detection_ids]
        recovery_tracks = [t for t in candidates
                           if t.local_track_id not in matched_tracks
                           and t.state in (LocalTrackState.TEMPORARILY_MISSED,
                                           LocalTrackState.OCCLUDED,
                                           LocalTrackState.RE_ACQUIRING)
                           and t.recoverable(now, cfg)]
        recovered = self._assign_lost_tracks(recovery_tracks, remaining_detections, now)
        for track_id, detection in recovered.items():
            self._apply_measurement(self.tracks[track_id], detection, frame)
            matched_tracks[track_id] = detection
            assigned_detection_ids.add(detection.detection_id)
            self.decision_logs.append({"timestamp": now, "action": "lost_track_reassociated",
                                       "track_id": track_id,
                                       "detection_id": detection.detection_id})

        for track in candidates:
            if track.local_track_id in matched_tracks:
                continue
            self._attempt_recovery(track, frame)

        for detection in free_detections:
            if detection.detection_id in assigned_detection_ids:
                continue
            self._maybe_spawn(detection, now, frame)

        observations = []
        min_vis = getattr(cfg, "min_visible_count", 1)
        for track in list(self.tracks.values()):
            track.at_border = self._at_border(track.center)
            if track.local_track_id not in matched_tracks:
                track.refresh_state(now, cfg)
            if not track.alive:
                del self.tracks[track.local_track_id]
                continue
            if track.observations >= min_vis or track.state != LocalTrackState.VISIBLE:
                observations.append(self._observation(track, now, frame))
        return observations

    def _assign_lost_tracks(self, tracks: list[LocalTrack], detections: list[LocalDetection],
                            now: float) -> dict[int, LocalDetection]:
        """Resolve relaxed recovery jointly, preserving one-to-one ownership."""
        if not tracks or not detections:
            return {}
        cost = np.full((len(tracks), len(detections)), np.inf)
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                dt = max(0.01, now - track.last_observed)
                speed = float(np.linalg.norm(track.filter.velocity))
                max_reach = max(speed * dt * 2.5, 80.0)
                distance = float(np.linalg.norm(detection.local_center - track.center))
                if distance > max_reach:
                    continue
                area = max(float((detection.bbox[2] - detection.bbox[0])
                                 * (detection.bbox[3] - detection.bbox[1])), 1.0)
                geometry = abs(np.log(area / max(track.area, 1.0)))
                appearance = (track.gallery.cost(detection.appearance)
                              if detection.appearance is not None and track.gallery.samples
                              else 0.0)
                cost[i, j] = distance / max_reach + geometry + 2.0 * appearance
        return {tracks[i].local_track_id: detections[j] for i, j in solve_assignment(cost)}

    def _apply_measurement(self, track: LocalTrack, detection: LocalDetection,
                           frame: NormalizedFrame) -> None:
        seam = max(0.0, 1.0 - self.profile.border_distance(detection.local_center)
                   / max(self.profile.image_diagonal * 0.25, 1.0))
        r = track.filter.measurement_covariance(
            detection.confidence, 1.0 if detection.occlusion_group_candidate else 0.0, seam)
        width = float(detection.bbox[2] - detection.bbox[0])
        height = float(detection.bbox[3] - detection.bbox[1])
        track.filter.update(detection.local_center, r, timestamp=detection.timestamp,
                            size=(width, height))
        track.mark_observed(detection.timestamp, detection.bbox, detection)
        if (not detection.occlusion_group_candidate
                and not detection.partial
                and detection.quality_score >= 0.40):
            # PLAN 1 stage 3 logic 6: never learn appearance from merged, partial, or low-quality regions.
            track.gallery.unfreeze()
            track.gallery.add(detection.appearance, detection.timestamp,
                              quality=detection.quality_score)
            template = extract_template(frame.gray, detection.bbox, self.config.template_size)
            if template is not None:
                track.template = template

    def _attempt_recovery(self, track: LocalTrack, frame: NormalizedFrame) -> None:
        cfg = self.config
        track.at_border = self._at_border(track.center)
        if not track.recovery_allowed(frame.timestamp, cfg):
            return
        match = match_template(frame.gray, track.template, track.center, cfg.template_search,
                               minimum_score=cfg.template_ncc_min)
        if not match.found:
            return
        r = track.filter.measurement_covariance(cfg.template_confidence, 0.5, 0.0)
        track.filter.update(match.center, r, timestamp=frame.timestamp)
        track.last_observed = frame.timestamp
        track.recoveries += 1
        track.blind_recoveries += 1
        track.state = LocalTrackState.RE_ACQUIRING
        track.last_measurement_source = MeasurementSource.TEMPLATE
        track.bbox = track.predicted_bbox()

    def _at_border(self, center) -> bool:
        margin = max(self.detection_config.border_margin * 2.0, 12.0)
        return self.profile.border_distance(center) <= margin

    def _maybe_spawn(self, detection: LocalDetection, now: float, frame: NormalizedFrame) -> None:
        cfg = self.config
        if len(self.tracks) >= cfg.max_tracks:
            self.decision_logs.append({"timestamp": now, "action": "spawn_rejected",
                                      "reason": "max_tracks_reached", "detection_id": detection.detection_id})
            return
        if detection.confidence < cfg.min_new_track_confidence:
            self.decision_logs.append({"timestamp": now, "action": "spawn_rejected",
                                      "reason": "low_confidence", "detection_id": detection.detection_id})
            return
        if detection.mask_area < 0.35 * self.detection_config.expected_vehicle_area:
            self.decision_logs.append({"timestamp": now, "action": "spawn_rejected",
                                      "reason": "sub_vehicle_area", "detection_id": detection.detection_id})
            return
        if detection.occlusion_group_candidate:
            return
        box = bbox_to_polygon(detection.bbox)
        # Check against existing tracks: a split fragment or nearby blob belonging to
        # an existing hypothesis must never spawn a competing active local track.
        for track in self.tracks.values():
            if not track.alive and not track.recoverable(now, cfg):
                continue
            pred = track.predicted_bbox()
            pred_box = bbox_to_polygon(pred)
            iou = polygon_iou(box, pred_box)
            if iou >= cfg.new_track_block_iou:
                self.decision_logs.append({"timestamp": now, "action": "spawn_blocked_by_iou",
                                          "track_id": track.local_track_id, "detection_id": detection.detection_id})
                return
            # Proximity split check: if center is inside expanded bounding box of an existing track
            w_margin = max(18.0, 0.30 * (pred[2] - pred[0]))
            h_margin = max(18.0, 0.30 * (pred[3] - pred[1]))
            if (pred[0] - w_margin <= detection.local_center[0] <= pred[2] + w_margin
                    and pred[1] - h_margin <= detection.local_center[1] <= pred[3] + h_margin):
                self.decision_logs.append({"timestamp": now, "action": "spawn_blocked_split_proximity",
                                          "track_id": track.local_track_id, "detection_id": detection.detection_id})
                return
        track = self._new_track(detection)
        if not detection.partial and detection.quality_score >= 0.40:
            track.gallery.add(detection.appearance, detection.timestamp, detection.quality_score)
        template = extract_template(frame.gray, detection.bbox, cfg.template_size)
        if template is not None:
            track.template = template
        self.decision_logs.append({"timestamp": now, "action": "track_spawned",
                                  "track_id": track.local_track_id, "detection_id": detection.detection_id})

    def _observation(self, track: LocalTrack, now: float,
                     frame: NormalizedFrame) -> LocalTrackObservation:
        observed = abs(track.last_observed - now) < 1e-9
        detection = track.last_detection
        source = track.last_measurement_source if observed else MeasurementSource.COAST
        # Carry the ground band forward with the filtered centre so that a recovered
        # or coasting track still projects to a plausible world footprint.
        if detection is not None:
            shift = track.center - np.asarray(detection.local_center, dtype=float)
            footprint = np.asarray(detection.footprint_pixels, dtype=float) + shift
            anchor = np.asarray(detection.ground_anchor, dtype=float) + shift
        else:
            footprint = bbox_to_polygon(track.predicted_bbox())
            anchor = track.center
        return LocalTrackObservation(
            local_track_id=track.local_track_id, camera_id=self.camera_id, timestamp=now,
            frame_sequence=frame.frame_sequence, predicted_bbox=track.predicted_bbox(),
            measured_bbox=np.asarray(track.bbox, dtype=float) if observed else None,
            state=track.state, missed_duration=track.missed_duration(now),
            motion_vector=track.filter.velocity,
            appearance_reference=track.gallery.centroid,
            footprint_pixels=footprint, ground_anchor=anchor,
            confidence=(detection.confidence if observed and detection is not None else 0.0),
            quality=(detection.quality_score if observed and detection is not None else 0.0),
            source=source, occlusion_group_id=track.occlusion_group_id, latent=track.latent,
            covariance=track.filter.position_covariance,
            extras={"observations": track.observations, "recoveries": track.recoveries,
                    "detection": (detection if observed and source is MeasurementSource.DETECTION
                                  else None),
                    "bbox": track.bbox if observed else track.predicted_bbox()})
