"""Detection injection: enter the pipeline at stage 4 with exact world geometry.

PLAN 3 states scenarios A-I as exact world coordinates.  Rendering pixels and
hoping the detector recovers those exact numbers would test the renderer, not the
identity logic, so the scenario suite also drives the pipeline with detections
synthesised *from* the stated coordinates by inverting the calibrated homography.
Stages 4-10 then run completely unchanged.
"""

from __future__ import annotations

import numpy as np

from ..contracts import LocalDetection
from ..profile import CameraProfile


def vehicle_embedding(vehicle_id: str, dim: int = 27) -> np.ndarray:
    """Deterministic per-vehicle appearance descriptor."""
    rng = np.random.default_rng(abs(hash(vehicle_id)) % (2 ** 32))
    vector = rng.normal(0.6, 0.25, size=dim).astype(np.float32)
    vector = np.clip(vector, 0.05, None)
    return vector / float(np.linalg.norm(vector))


def footprint_rectangle(centre, heading: float = 0.0, dimensions=(4.6, 1.9)) -> np.ndarray:
    length, width = dimensions
    c = np.asarray(centre, dtype=float).reshape(2)
    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-np.sin(heading), np.cos(heading)])
    hl, hw = length / 2.0, width / 2.0
    return np.array([c + hl * forward + hw * left, c + hl * forward - hw * left,
                     c - hl * forward - hw * left, c - hl * forward + hw * left])


def synth_detection(profile: CameraProfile, world_centre, timestamp: float,
                    frame_sequence: int, heading: float = 0.0, confidence: float = 0.92,
                    quality: float = 0.9, vehicle_id: str = "P", detection_id: int = 1,
                    dimensions=None, occlusion_group_candidate: bool = False,
                    area_scale: float = 1.0) -> LocalDetection:
    """Synthesise the stage-3 record a perfect detector would emit."""
    dimensions = dimensions or profile.vehicle_dimensions
    world_footprint = footprint_rectangle(world_centre, heading, dimensions)
    pixels = profile.calibration.unproject(world_footprint)
    projection = pixels @ profile.ground_direction
    order = np.argsort(-projection)
    quad = np.array([pixels[order[0]], pixels[order[1]], pixels[order[2]], pixels[order[3]]])
    anchor = 0.5 * (pixels[order[0]] + pixels[order[1]])
    bbox = np.array([pixels[:, 0].min(), pixels[:, 1].min(), pixels[:, 0].max(),
                     pixels[:, 1].max()])
    return LocalDetection(
        camera_id=profile.camera_id, timestamp=float(timestamp), frame_sequence=frame_sequence,
        bbox=bbox, confidence=confidence, local_center=pixels.mean(axis=0), ground_anchor=anchor,
        footprint_pixels=quad, mask_area=profile.expected_vehicle_area * area_scale,
        quality_score=quality, occlusion_group_candidate=occlusion_group_candidate,
        partial=False, appearance=vehicle_embedding(vehicle_id), detection_id=detection_id)


def merged_detection(profile: CameraProfile, centres, timestamp: float, frame_sequence: int,
                     heading: float = 0.0, dimensions=None, detection_id: int = 1,
                     vehicle_id: str = "MERGED") -> LocalDetection:
    """One blob covering several vehicles (PLAN 3 scenario E, PLAN 2 §7)."""
    dimensions = dimensions or profile.vehicle_dimensions
    corners = np.vstack([footprint_rectangle(c, heading, dimensions) for c in centres])
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    hull = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    detection = synth_detection(profile, hull.mean(axis=0), timestamp, frame_sequence,
                                heading=heading, vehicle_id=vehicle_id,
                                detection_id=detection_id,
                                dimensions=(x1 - x0, y1 - y0),
                                occlusion_group_candidate=True, area_scale=2.0)
    return detection


def run_world_script(pipeline, profiles: dict[str, CameraProfile], script) -> list:
    """Drive ``pipeline.step_from_detections`` from a scripted world timeline.

    ``script`` is a sequence of ``(timestamp, {camera_id: [(vehicle_id, position,
    heading), ...]})``; ``heading`` may be omitted.
    """
    results = []
    for index, (timestamp, per_camera) in enumerate(script):
        detections: dict[str, list[LocalDetection]] = {}
        counter = 0
        for camera_id, entries in per_camera.items():
            camera_detections = []
            for entry in entries:
                counter += 1
                if isinstance(entry, LocalDetection):
                    camera_detections.append(entry)
                    continue
                vehicle_id, position = entry[0], entry[1]
                heading = entry[2] if len(entry) > 2 else 0.0
                camera_detections.append(synth_detection(
                    profiles[camera_id], position, timestamp, index, heading=heading,
                    vehicle_id=vehicle_id, detection_id=counter))
            detections[camera_id] = camera_detections
        results.append(pipeline.step_from_detections(detections, timestamp))
    return results
