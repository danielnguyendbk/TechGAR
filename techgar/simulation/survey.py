"""Commissioning survey measurements (PLAN 1 Phase 0, work items 3-5).

These functions play the part of a technician with a tape measure and a reference
vehicle.  They may use the simulator's exact geometry — a real survey uses real
measurements — but the *production* code only ever receives their numeric output
(anchor bias, seam budget, skew statistics), never the geometry itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..detection import LocalDetector
from ..normalization import EnvironmentalNormalizer
from ..projection import WorldProjector
from .layouts import CRUISE_SPEED, LANE_Y, Layout, cruise
from .recording import RecordingOptions, build_recording


@dataclass
class SurveyResult:
    camera_id: str
    samples: int
    anchor_bias: float
    anchor_bias_sigma: float
    residual_max: float


def _reference_run(layout: Layout, profiles, camera_id: str, x_from: float, x_to: float,
                   fps: float = 12.0):
    """Detect a single reference vehicle across one camera, frame by frame."""
    vehicle = cruise("REF", x_from, x_to, t0=1.0, speed=CRUISE_SPEED, y=LANE_Y)
    recording = build_recording(f"survey_{camera_id}", layout, [vehicle],
                                RecordingOptions(fps=fps, skew={c: 0.0 for c in layout.camera_ids}))
    normalizer = EnvironmentalNormalizer(camera_id)
    detector = LocalDetector(profiles[camera_id])
    for frame in recording.iter_frames():
        if frame.camera_id != camera_id:
            continue
        normalized = normalizer.process(frame)
        detections = detector.detect(normalized)
        truth = [g for g in recording.ground_truth
                 if g.camera_id == camera_id and abs(g.timestamp - frame.timestamp) < 1e-9]
        yield frame, normalized, detections, (truth[0] if truth else None)


def ground_edge_midpoint(footprint: np.ndarray, world_ground_direction) -> np.ndarray:
    """Midpoint of the footprint edge the detector anchors on.

    Stage 3 takes the extreme of the blob along the camera's pixel ground
    direction; in the world that is the extreme of the footprint along the
    corresponding world direction, so the reference point is built the same way.
    """
    corners = np.asarray(footprint, dtype=float)
    projection = corners @ np.asarray(world_ground_direction, dtype=float)
    order = np.argsort(-projection)
    return 0.5 * (corners[order[0]] + corners[order[1]])


def measure_anchor_bias(layout: Layout, profiles, camera_id: str, x_from: float = 6.0,
                        x_to: float = 42.0) -> SurveyResult:
    """Systematic offset between the detected ground anchor and the true contact edge."""
    profile = profiles[camera_id]
    calibration = profile.calibration
    residuals = []
    for _, _, detections, truth in _reference_run(layout, profiles, camera_id, x_from, x_to):
        if truth is None or truth.visible_fraction < 0.99 or not detections:
            continue
        detection = max(detections, key=lambda d: d.mask_area)
        if detection.mask_area < 0.75 * profile.expected_vehicle_area:
            continue
        anchor_pixel = np.asarray(detection.ground_anchor)
        measured = calibration.project(anchor_pixel.reshape(1, 2))[0]
        direction = profile.world_ground_direction(anchor_pixel)
        expected = ground_edge_midpoint(truth.footprint, direction)
        residuals.append(float((expected - measured) @ profile.away_direction(measured)))
    if not residuals:
        return SurveyResult(camera_id, 0, 0.0, 0.0, 0.0)
    array = np.asarray(residuals)
    return SurveyResult(camera_id, len(residuals), float(array.mean()), float(array.std()),
                        float(np.abs(array).max()))


def apply_anchor_survey(layout: Layout, profiles) -> dict[str, SurveyResult]:
    """Measure and install the anchor bias for every camera in a layout."""
    results = {}
    for camera_id in layout.camera_ids:
        span = layout.fov(camera_id)
        x0, x1 = float(span[:, 0].min()), float(span[:, 0].max())
        results[camera_id] = measure_anchor_bias(layout, profiles, camera_id,
                                                max(x0 + 6.0, 2.0), min(x1 - 6.0, 92.0))
        profiles[camera_id].anchor_bias = results[camera_id].anchor_bias
        profiles[camera_id].anchor_bias_sigma = results[camera_id].anchor_bias_sigma
    return results


def measure_seam_disagreement(layout: Layout, profiles, projection_config=None) -> dict:
    """World disagreement for the *same* vehicle seen simultaneously by both cameras.

    PLAN 2 §3.2 requires rho_seam to be measured with a real vehicle at vehicle
    height, not an artificial target, which is exactly what this does.
    """
    vehicle = cruise("REF", 30.0, 52.0, t0=1.0, speed=CRUISE_SPEED, y=LANE_Y)
    recording = build_recording("seam", layout, [vehicle],
                                RecordingOptions(fps=12.0,
                                                 skew={c: 0.0 for c in layout.camera_ids}))
    normalizers = {c: EnvironmentalNormalizer(c) for c in layout.camera_ids}
    detectors = {c: LocalDetector(profiles[c]) for c in layout.camera_ids}
    projectors = {c: WorldProjector(profiles[c], layout.topology, projection_config)
                  for c in layout.camera_ids}
    per_time: dict[float, dict[str, np.ndarray]] = {}
    for frame in recording.iter_frames():
        normalized = normalizers[frame.camera_id].process(frame)
        detections = detectors[frame.camera_id].detect(normalized)
        if not detections:
            continue
        detection = max(detections, key=lambda d: d.mask_area)
        if detection.mask_area < 0.75 * profiles[frame.camera_id].expected_vehicle_area:
            continue
        world = projectors[frame.camera_id].project_detection(detection)
        # Match the two streams by nearest capture time, never by an exact key:
        # jitter alone would otherwise discard most simultaneous observations.
        slot = None
        for key in per_time:
            if abs(key - frame.timestamp) <= 0.030:
                slot = key
                break
        if slot is None:
            slot = frame.timestamp
            per_time[slot] = {}
        per_time[slot][frame.camera_id] = world.world_position
    distances = [float(np.linalg.norm(entry["C1"] - entry["C2"]))
                 for entry in per_time.values() if {"C1", "C2"} <= set(entry)]
    if not distances:
        return {"samples": 0, "rho_seam": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(distances)
    return {"samples": len(distances), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max()),
            "rho_seam": float(np.percentile(array, 95))}


def measure_timestamp_skew(recording) -> dict:
    """Nearest-neighbour capture-time skew between the two camera streams."""
    by_camera: dict[str, list[float]] = {}
    for spec in recording.specs:
        by_camera.setdefault(spec.camera_id, []).append(spec.timestamp)
    cameras = sorted(by_camera)
    if len(cameras) < 2:
        return {"samples": 0}
    a = np.asarray(by_camera[cameras[0]])
    b = np.asarray(by_camera[cameras[1]])
    skews = [float(np.min(np.abs(b - t))) for t in a]
    array = np.asarray(skews)
    return {"samples": len(skews), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max()),
            "mean": float(array.mean())}
