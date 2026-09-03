"""Stage 5 — planar projection of detections into the world frame.

Implements PLAN 2 §3.1-§3.2:

* the anchor is the ground-contact edge of the blob (the only part guaranteed to
  lie on the calibrated plane), corrected by the surveyed systematic bias;
* uncertainty is propagated analytically,
  ``Sigma_w = J_H Sigma_p J_H^T + Sigma_calib + Sigma_parallax``, and inflated by
  ``rho_seam^2 I`` near a seam, near the image border, and inside surveyed
  poor-calibration regions (PLAN 1 stage 5 logic 5).

Comparing raw pixel distances across cameras — the PLAN 1 stage 5 Fail case — is
impossible by construction here: nothing downstream of this module ever sees a
pixel coordinate as a metric quantity.
"""

from __future__ import annotations

import numpy as np

from .config_world import ProjectionConfig
from .contracts import LocalDetection, LocalTrackObservation, TopologyRegion
from .geometry import (inflate_polygon, oriented_rectangle, polygon_area, polygon_centroid,
                       polygon_extent)
from .linalg import inflate_isotropic, positional_sigma, symmetrize
from .profile import CameraProfile
from .topology import CameraTopology
from .world_contracts import WorldDetection


class WorldProjector:
    """One projector per camera (owns that camera's survey profile)."""

    def __init__(self, profile: CameraProfile, topology: CameraTopology,
                 config: ProjectionConfig | None = None) -> None:
        self.profile = profile
        self.topology = topology
        self.config = config or ProjectionConfig()
        self._next_observation_id = 0

    # --- uncertainty --------------------------------------------------------
    def pixel_covariance(self, confidence: float, occlusion: float, seam: float,
                         quality: float) -> np.ndarray:
        """Sigma_p with the PLAN 2 §2.4 quality expansion applied in pixel space."""
        cfg = self.config
        gain = (1.0 + cfg.quality_pixel_gain * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
                + 2.0 * occlusion + 1.0 * seam + 1.5 * (1.0 - float(np.clip(quality, 0.0, 1.0))))
        return np.diag([(cfg.sigma_pixel_u ** 2) * gain, (cfg.sigma_pixel_v ** 2) * gain])

    def _seam_score(self, camera_id: str, world_position, pixel) -> tuple[float, TopologyRegion]:
        region = self.topology.region_of(camera_id, world_position)
        score = 0.0
        if region in (TopologyRegion.OVERLAP, TopologyRegion.HANDOFF_EXIT,
                      TopologyRegion.HANDOFF_ENTRY):
            score += 1.0
        border = self.profile.border_distance(pixel)
        if border < self.config.border_margin_px:
            score += self.config.border_inflation * (1.0 - border / self.config.border_margin_px)
        zone = self.topology.zones.get(camera_id)
        if zone is not None:
            score += zone.extra_uncertainty(world_position)
        return score, region

    # --- projection ---------------------------------------------------------
    def project_detection(self, detection: LocalDetection, environment_quality: float = 1.0,
                          local_track_id: int = -1, latent: bool = False,
                          occlusion_group_id: int | None = None,
                          velocity: np.ndarray | None = None,
                          uncertainty_gain: float = 1.0) -> WorldDetection:
        profile = self.profile
        calib = profile.calibration
        cfg = self.config
        anchor_pixel = np.asarray(detection.ground_anchor, dtype=float)
        anchor_world = profile.correct_anchor(calib.project(anchor_pixel.reshape(1, 2))[0])
        occlusion = 1.0 if detection.occlusion_group_candidate else 0.0
        seam, region = self._seam_score(detection.camera_id, anchor_world, anchor_pixel)
        sigma_pixel = self.pixel_covariance(detection.confidence, occlusion, seam,
                                            environment_quality * detection.quality_score ** 0.25)
        parallax = cfg.sigma_parallax * (1.0 + profile.parallax_offset(anchor_world))
        parallax = parallax ** 2 + profile.anchor_bias_sigma ** 2
        covariance = calib.propagate(sigma_pixel, anchor_pixel[0], anchor_pixel[1],
                                     parallax * np.eye(2))
        if seam > 0.0:
            covariance = inflate_isotropic(covariance, cfg.rho_seam * min(seam, 3.0) ** 0.5)
        covariance = symmetrize(covariance * float(max(uncertainty_gain, 1.0)))

        footprint = self.world_footprint(detection, anchor_world, covariance)
        # PLAN 1 stage 5 logic 1: the *centre of the footprint* is the anchor the
        # world layers compare, because it is viewpoint independent — two cameras
        # observe opposite ground edges of the same vehicle, but one centre.
        centre = polygon_centroid(footprint)
        model_sigma = 0.5 * abs(profile.vehicle_dimensions[0] - profile.vehicle_dimensions[1]) * 0.18
        covariance = inflate_isotropic(covariance, model_sigma)
        width, height = polygon_extent(footprint)
        self._next_observation_id += 1
        return WorldDetection(
            camera_id=detection.camera_id, timestamp=detection.timestamp,
            frame_sequence=detection.frame_sequence, world_position=centre,
            world_covariance=covariance, world_footprint=footprint,
            source_pixel_position=anchor_pixel, topology_region=region,
            local_track_id=local_track_id, quality=detection.quality_score,
            confidence=detection.confidence, world_velocity=velocity,
            footprint_area=polygon_area(footprint),
            footprint_aspect=float(max(width, 1e-6) / max(height, 1e-6)),
            appearance=detection.appearance, latent=latent,
            partial=bool(detection.partial
                         or detection.mask_area < self.config_min_mint_area(profile)),
            occlusion_group_id=occlusion_group_id, observation_id=self._next_observation_id)

    @staticmethod
    def config_min_mint_area(profile: CameraProfile) -> float:
        """Below this blob area an observation is incomplete evidence, never a mint."""
        return 0.55 * profile.expected_vehicle_area

    def world_footprint(self, detection: LocalDetection, anchor_world: np.ndarray,
                        covariance: np.ndarray) -> np.ndarray:
        """Metric footprint from the ground edge plus the known vehicle model.

        ``footprint_pixels[0:2]`` is the ground-contact edge by contract (stage 3),
        so its projection is exact on the plane; the perpendicular extent comes
        from the surveyed vehicle dimensions (PLAN 1 stage 9 needs a *metric*
        footprint, not a bbox, or IoU/Coverage would be meaningless).
        """
        profile = self.profile
        edge = profile.calibration.project(np.asarray(detection.footprint_pixels)[:2])
        edge = np.array([profile.correct_anchor(edge[0]), profile.correct_anchor(edge[1])])
        observed = float(np.linalg.norm(edge[0] - edge[1]))
        # Temporal differencing often preserves only a thin leading/trailing rim
        # of a vehicle.  That rim is a reliable ground-contact *location*, but its
        # raw span is not a reliable metric vehicle width.  Reconstruct the edge
        # from the commissioned physical dimensions so a partial motion mask does
        # not collapse the world footprint and make a parked vehicle geometrically
        # incapable of satisfying the slot IoU threshold.
        length, width = sorted(profile.vehicle_dimensions, reverse=True)
        edge_span = float(np.clip(observed, width, length))
        vector = edge[1] - edge[0]
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            vector = np.array([1.0, 0.0])
        else:
            vector = vector / norm
        midpoint = 0.5 * (edge[0] + edge[1])
        edge = np.array([midpoint - 0.5 * edge_span * vector,
                         midpoint + 0.5 * edge_span * vector])
        observed = edge_span
        depth = profile.footprint_depth(observed)
        away = anchor_world + profile.away_direction(anchor_world)
        footprint = oriented_rectangle(edge[0], edge[1], depth, away)
        if self.config.calibration_scale > 0.0:
            margin = positional_sigma(covariance) * 0.0
            if margin > 0.0:
                footprint = inflate_polygon(footprint, margin)
        return footprint

    def project_observation(self, observation: LocalTrackObservation, detection: LocalDetection,
                            environment_quality: float = 1.0,
                            uncertainty_gain: float = 1.0) -> WorldDetection:
        """Project a stage-4 observation, carrying its latent/occlusion state."""
        return self.project_detection(
            detection, environment_quality=environment_quality,
            local_track_id=observation.local_track_id, latent=observation.latent,
            occlusion_group_id=observation.occlusion_group_id,
            uncertainty_gain=uncertainty_gain)
