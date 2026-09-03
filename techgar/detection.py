"""Stage 3 — local bounding-box detections with explicit merged-region handling.

The rule that matters here (PLAN 1 stage 3 logic 4-7): a connected component that
is too large, that covers several predicted vehicle positions, or that contains
several internal motion peaks is flagged ``occlusion_group_candidate`` and is
*not* allowed to update any identity's appearance, nor to become a new identity
while an existing one can explain it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config_vision import DetectionConfig, MotionConfig
from .contracts import LocalDetection
from .normalization import NormalizedFrame
from .profile import CameraProfile

STRUCTURE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class TrackPrediction:
    """What stage 4 predicts, fed back so stage 3 can spot merged regions."""

    local_track_id: int
    bbox: np.ndarray
    center: np.ndarray


def _band_quad(points: np.ndarray, direction: np.ndarray, band_ratio: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Ground band of a blob plus its anchor.

    ``direction`` points from the top of a vertical object towards its base, so
    the extreme of the blob along it lies on the floor; the band is the slice
    within ``band_ratio`` of that extreme.  Returns (quad, anchor).
    """
    perpendicular = np.array([-direction[1], direction[0]])
    s = points @ direction
    p = points @ perpendicular
    # 99th percentile rather than the maximum: a handful of stray pixels (a shadow
    # rim that survived suppression, compression ringing) must not move the anchor.
    s_hi = float(np.percentile(s, 99.0))
    thickness = max(band_ratio * float(s_hi - s.min()), 1.0)
    in_band = s >= s_hi - thickness
    p_band = p[in_band]
    p_lo, p_hi = float(p_band.min()), float(p_band.max())
    s_lo = s_hi - thickness
    quad = np.array([s_hi * direction + p_lo * perpendicular,
                     s_hi * direction + p_hi * perpendicular,
                     s_lo * direction + p_hi * perpendicular,
                     s_lo * direction + p_lo * perpendicular])
    anchor = s_hi * direction + 0.5 * (p_lo + p_hi) * perpendicular
    return quad, anchor


class LocalDetector:
    def __init__(self, profile: CameraProfile, config: DetectionConfig | None = None,
                 motion: MotionConfig | None = None) -> None:
        self.profile = profile
        self.config = config or DetectionConfig()
        self.motion = motion or MotionConfig()
        self._next_id = 0

    def _internal_peaks(self, mask: np.ndarray) -> int:
        eroded = ndi.binary_erosion(mask, structure=STRUCTURE,
                                    iterations=max(1, self.config.peak_erosion))
        if not eroded.any():
            return 1
        _, count = ndi.label(eroded, structure=STRUCTURE)
        return max(1, int(count))

    def detect(self, frame: NormalizedFrame,
               predictions: list[TrackPrediction] | None = None) -> list[LocalDetection]:
        cfg = self.config
        predictions = predictions or []
        labels, count = ndi.label(frame.foreground, structure=STRUCTURE)
        if count == 0:
            return []
        detections: list[LocalDetection] = []
        objects = ndi.find_objects(labels)
        min_area = self.motion.min_blob_area * (2 if frame.quality.unstable else 1)
        for index, window in enumerate(objects, start=1):
            if window is None:
                continue
            sub = labels[window] == index
            area = float(sub.sum())
            if area < min(min_area, self.motion.low_confidence_area):
                continue
            ys, xs = np.nonzero(sub)
            y0, x0 = window[0].start, window[1].start
            points = np.column_stack([xs + x0 + 0.5, ys + y0 + 0.5])
            bbox = np.array([points[:, 0].min(), points[:, 1].min(),
                             points[:, 0].max(), points[:, 1].max()])
            box_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            fill_ratio = area / box_area
            if fill_ratio < self.motion.min_fill_ratio:
                continue
            if frame.difference is not None and getattr(frame.difference, "mask", None) is not None:
                x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
                x2, y2 = min(frame.gray.shape[1], int(bbox[2]) + 1), min(frame.gray.shape[0], int(bbox[3]) + 1)
                diff_sub = frame.difference.mask[y1:y2, x1:x2]
                motion_pixels = int(diff_sub.sum()) if diff_sub.size > 0 else 0
                if motion_pixels < 20 or (motion_pixels / box_area) < 0.03:
                    continue
            size_score = float(np.exp(-abs(np.log(max(area, 1.0) / cfg.expected_vehicle_area))))
            env = 0.55 + 0.45 * frame.quality.quality
            confidence = float(np.clip((0.25 + 0.5 * size_score + 0.25 * fill_ratio) * env,
                                       0.0, 0.99))
            if area < min_area:
                confidence *= 0.6            # kept as secondary-association evidence
            if confidence < cfg.min_confidence:
                continue
            peaks = self._internal_peaks(sub)
            covered = tuple(p.local_track_id for p in predictions
                            if bbox[0] <= p.center[0] <= bbox[2] and bbox[1] <= p.center[1] <= bbox[3])
            merged = (area > cfg.merged_area_factor * cfg.expected_vehicle_area
                      or len(covered) >= 2 or peaks >= 2)
            quad, anchor = _band_quad(points, self.profile.ground_direction,
                                      cfg.footprint_band_ratio)
            margin = cfg.border_margin
            partial = bool(bbox[0] <= margin or bbox[1] <= margin
                           or bbox[2] >= self.profile.width - margin
                           or bbox[3] >= self.profile.height - margin)
            detections.append(LocalDetection(
                camera_id=frame.camera_id, timestamp=frame.timestamp,
                frame_sequence=frame.frame_sequence, bbox=bbox, confidence=confidence,
                local_center=points.mean(axis=0), ground_anchor=anchor, footprint_pixels=quad,
                mask_area=area, quality_score=float(size_score * fill_ratio
                                                    * frame.quality.quality),
                occlusion_group_candidate=bool(merged), partial=partial,
                internal_motion_peaks=peaks,
                covered_predictions=covered, detection_id=self._take_id()))
            if len(detections) >= cfg.max_detections:
                break
        return detections

    def _take_id(self) -> int:
        self._next_id += 1
        return self._next_id
