"""Classical multi-parameter consensus for stationary slot occupancy.

This is deliberately a corroborating sensor, not an identity source.  Twenty-five
gamma/CLAHE views vote on texture, edge and colour evidence inside each slot's
eroded core.  Global ID ownership still comes exclusively from the Stage 1-10
tracking pipeline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisionSlotEvidence:
    slot_id: str
    camera_id: str
    occupied: bool
    score: float
    votes: int
    total_votes: int
    stable: bool


class MultiParameterParkingConsensus:
    """25-view deterministic occupancy vote with temporal hysteresis."""

    def __init__(self, pixel_slots: dict[str, dict[str, np.ndarray]],
                 image_shapes: dict[str, tuple[int, int]], scale: float = 0.5,
                 history: int = 5, enter_score: float = 0.56,
                 release_score: float = 0.42) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("parking vision requires the video extra") from exc
        self.cv2 = cv2
        self.scale = float(scale)
        self.enter_score = float(enter_score)
        self.release_score = float(release_score)
        self.gamma_values = (0.72, 0.86, 1.0, 1.16, 1.32)
        self.clahe_values = (1.0, 1.5, 2.0, 2.5, 3.0)
        self._masks: dict[str, dict[str, np.ndarray]] = {}
        self._states: dict[str, bool] = {}
        self._history: dict[str, deque[float]] = {}
        for camera_id, slots in pixel_slots.items():
            source_height, source_width = image_shapes[camera_id]
            height, width = int(round(source_height * scale)), int(round(source_width * scale))
            camera_masks: dict[str, np.ndarray] = {}
            for slot_id, polygon in slots.items():
                points = np.rint(np.asarray(polygon) * scale).astype(np.int32)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                x, y, w, h = cv2.boundingRect(points)
                erosion = max(2, int(round(min(w, h) * 0.13)))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1)
                )
                core = cv2.erode(mask, kernel)
                if int(np.count_nonzero(core)) < 40:
                    core = mask
                camera_masks[slot_id] = core.astype(bool)
                self._states[slot_id] = False
                self._history[slot_id] = deque(maxlen=max(1, history))
            self._masks[camera_id] = camera_masks

    def process(self, frames: dict[str, np.ndarray]) -> dict[str, VisionSlotEvidence]:
        result: dict[str, VisionSlotEvidence] = {}
        for camera_id, frame in frames.items():
            result.update(self._process_camera(camera_id, frame))
        return result

    def _process_camera(self, camera_id: str,
                        frame: np.ndarray) -> dict[str, VisionSlotEvidence]:
        cv2 = self.cv2
        masks = self._masks[camera_id]
        height, width = next(iter(masks.values())).shape
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        base_l = lab[:, :, 0]
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        union = np.zeros_like(base_l, dtype=bool)
        for mask in masks.values():
            union |= mask
        votes = {slot_id: 0 for slot_id in masks}
        raw_scores = {slot_id: [] for slot_id in masks}
        for gamma in self.gamma_values:
            lookup = np.asarray([((value / 255.0) ** gamma) * 255.0
                                 for value in range(256)], dtype=np.uint8)
            corrected = cv2.LUT(base_l, lookup)
            for clip in self.clahe_values:
                enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(corrected)
                edges = cv2.Canny(enhanced, 45, 135) > 0
                floor_candidates = enhanced[union & (saturation < 0.20)]
                if floor_candidates.size:
                    # Across all slot cores the floor still contributes the
                    # majority of pixels even when half of the bays are occupied.
                    # A histogram mode is unsafe here: a large white/black toy car
                    # can create a sharper mode than the textured asphalt.
                    floor_level = int(np.median(floor_candidates))
                else:
                    floor_level = int(np.median(enhanced[union]))
                deviation = np.abs(enhanced.astype(np.int16) - floor_level)
                objects = ((deviation > 20) | (saturation > 0.24)) & union
                objects = cv2.morphologyEx(objects.astype(np.uint8), cv2.MORPH_OPEN,
                                           np.ones((3, 3), dtype=np.uint8)) > 0
                for slot_id, mask in masks.items():
                    if not mask.any():
                        continue
                    edge_density = float(np.mean(edges[mask]))
                    object_fraction = float(np.mean(objects[mask]))
                    # Empty asphalt remains close to the dominant low-saturation
                    # floor mode.  A vehicle must occupy a meaningful part of the
                    # core; a thin watermark or painted remnant cannot win alone.
                    area_score = float(np.clip((object_fraction - 0.08) / 0.30, 0.0, 1.0))
                    edge_score = float(np.clip((edge_density - 0.015) / 0.09, 0.0, 1.0))
                    score = 0.78 * area_score + 0.22 * edge_score
                    raw_scores[slot_id].append(score)
                    if score >= 0.50:
                        votes[slot_id] += 1
        camera_result: dict[str, VisionSlotEvidence] = {}
        total = len(self.gamma_values) * len(self.clahe_values)
        for slot_id in masks:
            vote_score = votes[slot_id] / total
            feature_score = float(np.median(raw_scores[slot_id]))
            score = 0.65 * vote_score + 0.35 * feature_score
            history = self._history[slot_id]
            history.append(score)
            temporal = float(np.median(history))
            occupied = self._states[slot_id]
            if occupied and temporal <= self.release_score:
                occupied = False
            elif not occupied and temporal >= self.enter_score:
                occupied = True
            self._states[slot_id] = occupied
            camera_result[slot_id] = VisionSlotEvidence(
                slot_id=slot_id, camera_id=camera_id, occupied=occupied,
                score=temporal, votes=votes[slot_id], total_votes=total,
                stable=len(history) == history.maxlen,
            )
        return camera_result
