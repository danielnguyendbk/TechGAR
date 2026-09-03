"""Stage 6 — cross-camera world fusion under a one-to-one guarantee.

PLAN 1 stage 6 logic, in order: candidate pairs only inside the calibrated
overlap, reject temporally incompatible pairs, reject covariance-incompatible
pairs, reject pairs whose footprints describe two different vehicles, fuse the
survivors into one measurement, keep everything else separate — and make sure one
detection can never be fused into two vehicles in the same synchronisation cycle
(guaranteed structurally by solving an assignment rather than greedy pairing).

Fusion uses the information filter, so a confident observation dominates a vague
one; Euclidean averaging without covariance is the PLAN 1 Phase 2 Fail case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .appearance import cosine_distance
from .assignment import solve_assignment
from .config_world import FusionConfig
from .contracts import TopologyRegion
from .geometry import polygon_iou
from .linalg import information_fuse, mahalanobis_sq, positional_sigma
from .topology import CameraTopology
from .world_contracts import FusedWorldDetection, WorldDetection


@dataclass
class FusionRejection:
    camera_a: str
    camera_b: str
    reason: str
    value: float = 0.0


class WorldFusion:
    def __init__(self, topology: CameraTopology, config: FusionConfig | None = None) -> None:
        self.topology = topology
        self.config = config or FusionConfig()
        self.rejections: list[FusionRejection] = []
        self._next_id = 0

    def _pair_cost(self, a: WorldDetection, b: WorldDetection) -> tuple[float, str]:
        cfg = self.config
        if abs(a.timestamp - b.timestamp) > cfg.max_skew:
            return float("inf"), "time_incompatible"
        expansion = cfg.overlap_expansion_sigma * max(positional_sigma(a.world_covariance),
                                                      positional_sigma(b.world_covariance))
        if not (self.topology.in_overlap(a.camera_id, b.camera_id, a.world_position, expansion)
                and self.topology.in_overlap(a.camera_id, b.camera_id, b.world_position,
                                             expansion)):
            return float("inf"), "outside_overlap"
        residual = b.world_position - a.world_position
        distance = mahalanobis_sq(residual, a.world_covariance + b.world_covariance)
        if distance > cfg.gate:
            return float("inf"), "covariance_incompatible"
        area_ratio = abs(np.log(max(b.footprint_area, 1e-6) / max(a.footprint_area, 1e-6)))
        if area_ratio > cfg.max_area_log_ratio:
            return float("inf"), "footprint_area_mismatch"
        overlap = polygon_iou(a.world_footprint, b.world_footprint)
        if overlap < cfg.min_footprint_iou:
            return float("inf"), "footprint_disjoint"
        appearance = cosine_distance(a.appearance, b.appearance)
        if a.appearance is not None and b.appearance is not None and appearance > cfg.appearance_max:
            return float("inf"), "appearance_incompatible"
        return float(distance + cfg.appearance_weight * appearance + area_ratio), "ok"

    def _single(self, detection: WorldDetection) -> FusedWorldDetection:
        self._next_id += 1
        return FusedWorldDetection(
            timestamp=detection.timestamp, frame_sequence=detection.frame_sequence,
            position=detection.world_position, covariance=detection.world_covariance,
            footprint=detection.world_footprint,
            contributing_cameras=(detection.camera_id,),
            contributing_observations=(detection.observation_id,),
            fusion_confidence=float(detection.confidence),
            topology_region=detection.topology_region,
            local_track_ids=((detection.camera_id, detection.local_track_id),),
            quality=detection.quality, velocity=detection.world_velocity,
            footprint_area=detection.footprint_area, footprint_aspect=detection.footprint_aspect,
            appearance=detection.appearance, latent=detection.latent,
            partial=detection.partial,
            occlusion_group_id=detection.occlusion_group_id, observation_id=self._next_id)

    def _merge(self, a: WorldDetection, b: WorldDetection, distance: float
               ) -> FusedWorldDetection:
        position, covariance = information_fuse([a.world_position, b.world_position],
                                                [a.world_covariance, b.world_covariance])
        better = a if a.quality >= b.quality else b
        consistency = float(np.exp(-distance / max(self.config.gate, 1e-6)))
        self._next_id += 1
        return FusedWorldDetection(
            timestamp=max(a.timestamp, b.timestamp),
            frame_sequence=max(a.frame_sequence, b.frame_sequence),
            position=position, covariance=covariance, footprint=better.world_footprint,
            contributing_cameras=tuple(sorted((a.camera_id, b.camera_id))),
            contributing_observations=(a.observation_id, b.observation_id),
            fusion_confidence=float(min(0.99, 0.5 * (a.confidence + b.confidence)
                                        * (0.5 + 0.5 * consistency))),
            topology_region=TopologyRegion.OVERLAP,
            local_track_ids=((a.camera_id, a.local_track_id), (b.camera_id, b.local_track_id)),
            quality=float(max(a.quality, b.quality)),
            velocity=(a.world_velocity if a.world_velocity is not None else b.world_velocity),
            footprint_area=0.5 * (a.footprint_area + b.footprint_area),
            footprint_aspect=better.footprint_aspect,
            appearance=(better.appearance if better.appearance is not None else a.appearance),
            latent=a.latent or b.latent, partial=a.partial and b.partial,
            occlusion_group_id=a.occlusion_group_id or b.occlusion_group_id,
            observation_id=self._next_id)

    def fuse(self, detections: list[WorldDetection]) -> list[FusedWorldDetection]:
        self.rejections.clear()
        by_camera: dict[str, list[WorldDetection]] = {}
        for detection in detections:
            by_camera.setdefault(detection.camera_id, []).append(detection)
        cameras = sorted(by_camera)
        if len(cameras) < 2:
            return [self._single(d) for d in detections]
        left, right = by_camera[cameras[0]], by_camera[cameras[1]]
        cost = np.full((len(left), len(right)), np.inf)
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                value, reason = self._pair_cost(a, b)
                cost[i, j] = value
                if not np.isfinite(value):
                    self.rejections.append(FusionRejection(a.camera_id, b.camera_id, reason))
        pairs = solve_assignment(cost)
        fused: list[FusedWorldDetection] = []
        used_left = {i for i, _ in pairs}
        used_right = {j for _, j in pairs}
        for i, j in pairs:
            fused.append(self._merge(left[i], right[j], float(cost[i, j])))
        fused.extend(self._single(a) for i, a in enumerate(left) if i not in used_left)
        fused.extend(self._single(b) for j, b in enumerate(right) if j not in used_right)
        extra = [d for cam in cameras[2:] for d in by_camera[cam]]
        fused.extend(self._single(d) for d in extra)
        return sorted(fused, key=lambda f: f.observation_id)
