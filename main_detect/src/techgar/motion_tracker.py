"""Fallback tracker cho camera bãi xe nhìn từ trên cao.

COCO YOLO không nhận được xe quá nhỏ/nhìn top-down trong video carPark mẫu.
Backend này dùng foreground motion để tạo detection, nhưng thay Particle Filter
cũ bằng Kalman constant-velocity + global assignment (LAPJV) + HSV appearance
histogram để hạn chế đổi ID và hỗ trợ Re-ID ngắn hạn.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from lap import lapjv

from .tracklet_descriptor import (
    AppearanceTracklet,
    compare_tracklets,
    histogram_distance,
    hsv_histogram,
)
from .vehicle_tracker import TrackStatus, TrackedVehicle


class MotionVehicleTracker:
    """Tracker không phụ thuộc model, tối ưu cho camera cố định có xe chuyển động."""

    def __init__(
        self,
        min_visible_count: int = 3,
        lost_track_ttl: int = 90,
        history_len: int = 10,
        min_area: int = 650,
        min_width: int = 22,
        min_height: int = 18,
        max_distance: float = 180.0,
        min_confirm_displacement: float = 12.0,
        motion_frame_gap: int = 3,
        motion_threshold: int = 25,
        motion_min_ratio: float = 0.08,
        motion_min_pixels: int = 160,
        reid_ttl: int = 720,
        homography: Optional[np.ndarray] = None,
        tracklet_max_samples: int = 12,
        tracklet_sample_interval: int = 3,
        slot_binder=None,  # SlotVehicleBinder instance (optional)
        temporal_short_seconds: float = 0.25,
        temporal_long_seconds: float = 0.80,
        motion_long_frame_gap: int = 9,
        priority_min_area: int = 350,
        priority_motion_min_ratio: float = 0.03,
        priority_motion_min_pixels: int = 50,
        priority_min_visible_count: int = 2,
        priority_min_confirm_displacement: float = 3.0,
        enable_multiscale_motion: bool = False,
        reject_cast_shadows: bool = False,
        shadow_attenuation_range: Tuple[float, float] = (0.32, 0.62),
        shadow_max_chromaticity_distance: float = 0.10,
        shadow_max_scaled_residual: float = 0.08,
        shadow_min_explained_fraction: float = 0.75,
        shadow_min_pixels: int = 120,
        reacquire_max_seconds: float = 0.75,
        lost_appearance_threshold: float = 0.30,
        min_reacquire_area_ratio: float = 0.35,
        max_reacquire_area_ratio: float = 2.80,
        merged_detection_area_ratio: float = 1.60,
        max_bbox_width_ratio: float = 0.30,
        max_bbox_height_ratio: float = 0.42,
        max_bbox_area_ratio: float = 0.12,
        motion_join_kernel_size: int = 5,
        motion_join_iterations: int = 1,
    ):
        self.min_visible_count = max(1, min_visible_count)
        self.lost_track_ttl = max(1, lost_track_ttl)
        self.history_len = max(2, history_len)
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.max_distance = max_distance
        self.min_confirm_displacement = float(min_confirm_displacement)
        self.motion_frame_gap = max(1, int(motion_frame_gap))
        self.motion_threshold = int(motion_threshold)
        self.motion_min_ratio = float(motion_min_ratio)
        self.motion_min_pixels = int(motion_min_pixels)
        self.temporal_short_seconds = max(0.01, float(temporal_short_seconds))
        self.temporal_long_seconds = max(self.temporal_short_seconds, float(temporal_long_seconds))
        self.motion_long_frame_gap = max(self.motion_frame_gap, int(motion_long_frame_gap))
        self.priority_min_area = max(1, int(priority_min_area))
        self.priority_motion_min_ratio = max(0.0, float(priority_motion_min_ratio))
        self.priority_motion_min_pixels = max(1, int(priority_motion_min_pixels))
        self.priority_min_visible_count = max(2, int(priority_min_visible_count))
        self.priority_min_confirm_displacement = max(0.0, float(priority_min_confirm_displacement))
        self.enable_multiscale_motion = bool(enable_multiscale_motion)
        self.reject_cast_shadows = bool(reject_cast_shadows)
        self.shadow_min_attenuation = min(
            float(shadow_attenuation_range[0]),
            float(shadow_attenuation_range[1]),
        )
        self.shadow_max_attenuation = max(
            float(shadow_attenuation_range[0]),
            float(shadow_attenuation_range[1]),
        )
        self.shadow_max_chromaticity_distance = max(
            0.0, float(shadow_max_chromaticity_distance)
        )
        self.shadow_max_scaled_residual = max(0.0, float(shadow_max_scaled_residual))
        self.shadow_min_explained_fraction = min(
            1.0, max(0.0, float(shadow_min_explained_fraction))
        )
        self.shadow_min_pixels = max(20, int(shadow_min_pixels))
        self.reacquire_max_seconds = max(0.05, float(reacquire_max_seconds))
        self.lost_appearance_threshold = max(0.0, float(lost_appearance_threshold))
        self.min_reacquire_area_ratio = max(0.01, float(min_reacquire_area_ratio))
        self.max_reacquire_area_ratio = max(
            self.min_reacquire_area_ratio, float(max_reacquire_area_ratio)
        )
        self.merged_detection_area_ratio = max(
            1.05, float(merged_detection_area_ratio)
        )
        self.max_bbox_width_ratio = min(1.0, max(0.05, float(max_bbox_width_ratio)))
        self.max_bbox_height_ratio = min(1.0, max(0.05, float(max_bbox_height_ratio)))
        self.max_bbox_area_ratio = min(1.0, max(0.01, float(max_bbox_area_ratio)))
        join_kernel_size = max(3, int(motion_join_kernel_size))
        self.motion_join_kernel_size = (
            join_kernel_size if join_kernel_size % 2 else join_kernel_size + 1
        )
        self.motion_join_iterations = max(1, int(motion_join_iterations))
        self.reid_ttl = max(reid_ttl, lost_track_ttl)
        self.homography = homography
        self.tracklet_max_samples = max(1, int(tracklet_max_samples))
        self.tracklet_sample_interval = max(1, int(tracklet_sample_interval))
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=32, detectShadows=True)
        self._tracks: Dict[int, TrackedVehicle] = {}
        self._exited_tracks: Dict[int, TrackedVehicle] = {}
        self._next_id = 1
        self._frame_idx = 0
        self.roi_mask: Optional[np.ndarray] = None
        # Keep enough samples for the 0.8 s slow-motion reference at common
        # DroidCam frame rates. Entries are (timestamp_s, frame_index, gray).
        self._gray_history = deque(maxlen=max(48, self.motion_long_frame_gap + 2))
        self._current_timestamp_s: Optional[float] = None
        self.slot_binder = slot_binder  # Tham chiếu tới SlotVehicleBinder
        self._newly_lost_tracks: List[Tuple[int, TrackedVehicle]] = []
        self._last_shadow_rejections: List[dict] = []
        self._last_detection_rejections: List[dict] = []
        self._suspended_tracks: Dict[int, TrackedVehicle] = {}
        self._ambiguous_detection_ids: set[int] = set()
        self._viable_pairs: set[Tuple[int, int]] = set()
        self._pair_metrics: Dict[Tuple[int, int], dict] = {}
        self._last_association_events: List[dict] = []

    @staticmethod
    def _bottom_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = box
        return x + w // 2, y + h

    @staticmethod
    def _iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = left
        bx, by, bw, bh = right
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _box_gap(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
        """Shortest edge-to-edge distance between two axis-aligned boxes."""
        ax, ay, aw, ah = left
        bx, by, bw, bh = right
        dx = max(ax - (bx + bw), bx - (ax + aw), 0)
        dy = max(ay - (by + bh), by - (ay + ah), 0)
        return float(np.hypot(dx, dy))

    @staticmethod
    def _histogram(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        return hsv_histogram(frame, box)

    def _ground_point(self, point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self.homography is None:
            return None
        source = np.array([[[point[0], point[1]]]], dtype=np.float32)
        target = cv2.perspectiveTransform(source, self.homography.astype(np.float32))[0, 0]
        return round(float(target[0]), 3), round(float(target[1]), 3)

    @staticmethod
    def _new_kalman(point: Tuple[int, int]) -> cv2.KalmanFilter:
        kalman = cv2.KalmanFilter(4, 2)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.processNoiseCov = np.diag([2.0, 2.0, 12.0, 12.0]).astype(np.float32)
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 8.0
        kalman.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        kalman.statePost = np.array([[point[0]], [point[1]], [0], [0]], dtype=np.float32)
        return kalman

    @staticmethod
    def _motion_between(current: np.ndarray, reference: np.ndarray, threshold: int) -> np.ndarray:
        """Return brightness-compensated change without treating a light shift as motion."""
        brightness_shift = float(np.median(current.astype(np.int16) - reference.astype(np.int16)))
        adjusted_reference = np.clip(
            reference.astype(np.float32) + brightness_shift,
            0,
            255,
        ).astype(np.uint8)
        difference = cv2.absdiff(current, adjusted_reference)
        _, return_mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        return return_mask

    def _timestamp_references(self, timestamp_s: float) -> List[np.ndarray]:
        history = list(self._gray_history)[:-1]
        references: List[np.ndarray] = []
        target_ages = (
            (self.temporal_short_seconds, self.temporal_long_seconds)
            if self.enable_multiscale_motion
            else (self.temporal_short_seconds,)
        )
        for target_age in target_ages:
            candidates = []
            for old_timestamp, _frame_idx, gray in history:
                if old_timestamp is None:
                    continue
                age = timestamp_s - old_timestamp
                # A reference that is far older than requested commonly means
                # a stream reconnect/pause. Comparing across it creates a
                # full-frame flash, so wait for fresh history instead.
                if target_age * 0.45 <= age <= target_age * 2.25:
                    candidates.append((abs(age - target_age), gray))
            if candidates:
                references.append(min(candidates, key=lambda item: item[0])[1])
        return references

    def _frame_gap_references(self) -> List[np.ndarray]:
        history = list(self._gray_history)
        references: List[np.ndarray] = []
        gaps = (
            (self.motion_frame_gap, self.motion_long_frame_gap)
            if self.enable_multiscale_motion
            else (self.motion_frame_gap,)
        )
        for gap in gaps:
            if len(history) > gap:
                references.append(history[-gap - 1][2])
        return references

    def _temporal_motion_mask(
        self,
        frame: np.ndarray,
        timestamp_s: Optional[float] = None,
    ) -> np.ndarray:
        """Chỉ giữ pixel thực sự thay đổi giữa hai thời điểm.

        MOG2 có thể đánh dấu xe đỗ khi ánh sáng/nén video thay đổi. Frame
        difference qua nhiều frame giúp bỏ các blob đứng yên. Median brightness
        shift được trừ trước để không coi thay đổi phơi sáng toàn khung là xe.
        """
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        valid_timestamp = None
        if timestamp_s is not None and np.isfinite(timestamp_s):
            valid_timestamp = float(timestamp_s)
            previous_timestamps = [item[0] for item in self._gray_history if item[0] is not None]
            if previous_timestamps and valid_timestamp < previous_timestamps[-1]:
                # Timestamp regression means a seek/reconnect. Do not compare
                # the new frame with an unrelated point in the old stream.
                self._gray_history.clear()
        self._gray_history.append((valid_timestamp, self._frame_idx, gray))
        references = (
            self._timestamp_references(valid_timestamp)
            if valid_timestamp is not None
            else self._frame_gap_references()
        )
        if not references:
            return np.zeros_like(gray)

        # Short-term motion catches fast vehicles; long-term motion accumulates
        # enough displacement for vehicles moving only a few pixels per frame.
        motion = np.zeros_like(gray)
        for reference in references:
            motion = cv2.bitwise_or(
                motion,
                self._motion_between(gray, reference, self.motion_threshold),
            )
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=2)
        # Large closing kernels used to join two nearby moving toy cars into
        # one foreground contour.  Keep enough closing to repair one car's
        # broken silhouette, but not enough to bridge the lane between cars.
        join_kernel = np.ones(
            (self.motion_join_kernel_size, self.motion_join_kernel_size),
            np.uint8,
        )
        motion = cv2.morphologyEx(
            motion,
            cv2.MORPH_CLOSE,
            join_kernel,
            iterations=self.motion_join_iterations,
        )
        return motion

    @staticmethod
    def _coerce_priority_polygon(region: Any) -> Optional[np.ndarray]:
        if isinstance(region, dict):
            region = region.get("polygon", region.get("points"))
        elif hasattr(region, "polygon"):
            region = getattr(region, "polygon")
        if region is None:
            return None
        try:
            polygon = np.asarray(region, dtype=np.int32).reshape(-1, 2)
        except (TypeError, ValueError):
            return None
        return polygon if len(polygon) >= 3 else None

    def _priority_mask(self, shape: Tuple[int, int], priority_regions: Optional[Sequence[Any]]) -> Optional[np.ndarray]:
        if priority_regions is None:
            return None
        if isinstance(priority_regions, np.ndarray) and priority_regions.shape == shape:
            return np.where(priority_regions > 0, 255, 0).astype(np.uint8)

        regions: Any = priority_regions
        try:
            numeric = np.asarray(regions)
            if numeric.ndim == 2 and numeric.shape[1] == 2 and numeric.dtype != object:
                regions = [regions]
        except (TypeError, ValueError):
            pass
        if isinstance(regions, dict) or hasattr(regions, "polygon"):
            regions = [regions]

        mask = np.zeros(shape, dtype=np.uint8)
        for region in regions:
            polygon = self._coerce_priority_polygon(region)
            if polygon is not None:
                cv2.fillPoly(mask, [polygon], 255)
        return mask if cv2.countNonZero(mask) else None

    @staticmethod
    def _box_is_priority(
        priority_mask: Optional[np.ndarray],
        box: Tuple[int, int, int, int],
        point: Tuple[int, int],
    ) -> bool:
        if priority_mask is None:
            return False
        x, y, w, h = box
        height, width = priority_mask.shape[:2]
        px, py = point
        if 0 <= px < width and 0 <= py < height and priority_mask[py, px] != 0:
            return True
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return False
        overlap = cv2.countNonZero(priority_mask[y1:y2, x1:x2])
        return overlap / float(max(1, w * h)) >= 0.10

    def _cast_shadow_metrics(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        box: Tuple[int, int, int, int],
        selection: np.ndarray,
    ) -> Optional[dict]:
        """Explain a foreground blob as a darker copy of the learned floor.

        A cast shadow approximately multiplies all BGR channels of the visible
        floor by one attenuation factor while preserving chromaticity and road
        texture. A physical object instead occludes that texture, changes
        colour, becomes brighter, or (for a truly dark object) falls below the
        deliberately narrow attenuation band. The test is intentionally
        fail-open: incomplete/young background evidence never rejects a blob.
        """
        if background is None or background.shape != frame.shape:
            return None
        x, y, width, height = box
        if selection.shape != (height, width):
            return None
        selected = selection.astype(bool)
        selected_count = int(np.count_nonzero(selected))
        if selected_count < self.shadow_min_pixels:
            return None

        current_bgr = frame[y:y + height, x:x + width].astype(np.float32)[selected]
        background_bgr = background[y:y + height, x:x + width].astype(np.float32)[selected]
        luminance_weights = np.asarray([0.114, 0.587, 0.299], dtype=np.float32)
        current_luminance = current_bgr @ luminance_weights
        background_luminance = background_bgr @ luminance_weights
        valid = (background_luminance >= 20.0) & (current_luminance >= 3.0)
        valid_count = int(np.count_nonzero(valid))
        if valid_count < self.shadow_min_pixels or valid_count < selected_count * 0.60:
            return None

        current_bgr = current_bgr[valid]
        background_bgr = background_bgr[valid]
        current_luminance = current_luminance[valid]
        background_luminance = background_luminance[valid]
        attenuation_values = current_luminance / np.maximum(background_luminance, 1e-3)
        attenuation = float(np.median(attenuation_values))

        current_chromaticity = current_bgr / np.maximum(
            np.sum(current_bgr, axis=1, keepdims=True), 1e-3
        )
        background_chromaticity = background_bgr / np.maximum(
            np.sum(background_bgr, axis=1, keepdims=True), 1e-3
        )
        chromaticity_distance = np.sum(
            np.abs(current_chromaticity - background_chromaticity), axis=1
        )
        scaled_residual = np.abs(
            current_luminance - attenuation * background_luminance
        ) / np.maximum(background_luminance, 1e-3)
        median_chromaticity_distance = float(np.median(chromaticity_distance))
        median_scaled_residual = float(np.median(scaled_residual))

        explained = (
            (np.abs(attenuation_values - attenuation) <= 0.18)
            & (chromaticity_distance <= self.shadow_max_chromaticity_distance * 1.20)
            & (scaled_residual <= self.shadow_max_scaled_residual * 1.50)
        )
        explained_fraction = float(np.mean(explained))
        return {
            "attenuation": attenuation,
            "chromaticity_distance": median_chromaticity_distance,
            "scaled_residual": median_scaled_residual,
            "explained_fraction": explained_fraction,
            "pixel_count": valid_count,
        }

    def _is_cast_shadow(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        box: Tuple[int, int, int, int],
        selection: np.ndarray,
    ) -> Tuple[bool, Optional[dict]]:
        if not self.reject_cast_shadows:
            return False, None
        metrics = self._cast_shadow_metrics(frame, background, box, selection)
        if metrics is None:
            return False, None
        is_shadow = (
            self.shadow_min_attenuation
            <= metrics["attenuation"]
            <= self.shadow_max_attenuation
            and metrics["chromaticity_distance"]
            <= self.shadow_max_chromaticity_distance
            and metrics["scaled_residual"] <= self.shadow_max_scaled_residual
            and metrics["explained_fraction"] >= self.shadow_min_explained_fraction
        )
        return bool(is_shadow), metrics

    def _detect(
        self,
        frame: np.ndarray,
        timestamp_s: Optional[float] = None,
        priority_regions: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[dict], np.ndarray]:
        self._last_shadow_rejections = []
        self._last_detection_rejections = []
        background_image = None
        background_getter = getattr(self.bg_sub, "getBackgroundImage", None)
        if self.reject_cast_shadows and callable(background_getter):
            try:
                background_image = background_getter()
            except cv2.error:
                background_image = None
        background_mask = self.bg_sub.apply(frame)
        _, background_mask = cv2.threshold(background_mask, 200, 255, cv2.THRESH_BINARY)
        temporal_motion = self._temporal_motion_mask(frame, timestamp_s=timestamp_s)
        # Motion mask là cổng bắt buộc: foreground đứng yên không được thành xe.
        support = cv2.dilate(temporal_motion, np.ones((17, 17), np.uint8), iterations=1)
        mask = cv2.bitwise_and(background_mask, support)
        if self.roi_mask is not None:
            mask = cv2.bitwise_and(mask, self.roi_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        join_kernel = np.ones(
            (self.motion_join_kernel_size, self.motion_join_kernel_size),
            np.uint8,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            join_kernel,
            iterations=self.motion_join_iterations,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = frame.shape[0] * frame.shape[1]
        image_height, image_width = frame.shape[:2]
        max_bbox_width = image_width * self.max_bbox_width_ratio
        max_bbox_height = image_height * self.max_bbox_height_ratio
        max_bbox_area = image_area * self.max_bbox_area_ratio
        priority_mask = self._priority_mask(frame.shape[:2], priority_regions)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image_area * 0.22:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width or h < self.min_height:
                continue
            bbox_area = float(w * h)
            if (
                w > max_bbox_width
                or h > max_bbox_height
                or bbox_area > max_bbox_area
            ):
                self._last_detection_rejections.append({
                    "type": "oversized_bbox",
                    "box": (int(x), int(y), int(w), int(h)),
                    "max_width": round(float(max_bbox_width), 1),
                    "max_height": round(float(max_bbox_height), 1),
                    "max_area": round(float(max_bbox_area), 1),
                })
                cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
                continue
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.25 or aspect_ratio > 5.0:
                continue
            box = (x, y, w, h)
            point = self._bottom_center(box)
            # A contour clipped by a polygon can still have a rectangular bbox
            # whose bottom centre lies outside the observation area.  Do not
            # create or update a vehicle beyond the camera's tracking ROI.
            if self.roi_mask is not None:
                px, py = point
                if (
                    px < 0
                    or py < 0
                    or py >= self.roi_mask.shape[0]
                    or px >= self.roi_mask.shape[1]
                    or self.roi_mask[py, px] == 0
                ):
                    self._last_detection_rejections.append({
                        "type": "anchor_outside_roi",
                        "box": (int(x), int(y), int(w), int(h)),
                        "anchor": (int(px), int(py)),
                    })
                    cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
                    continue
            is_priority = self._box_is_priority(priority_mask, box, point)
            min_area = self.priority_min_area if is_priority else self.min_area
            if area < min_area:
                continue
            motion_pixels = cv2.countNonZero(temporal_motion[y:y + h, x:x + w])
            min_pixels = self.priority_motion_min_pixels if is_priority else self.motion_min_pixels
            min_ratio = self.priority_motion_min_ratio if is_priority else self.motion_min_ratio
            if motion_pixels < min_pixels or motion_pixels / float(w * h) < min_ratio:
                continue
            if self.reject_cast_shadows:
                contour_selection = np.zeros((h, w), dtype=np.uint8)
                local_contour = contour.reshape((-1, 2)) - np.asarray(
                    [x, y], dtype=np.int32
                )
                cv2.fillPoly(
                    contour_selection,
                    [local_contour.astype(np.int32)],
                    255,
                )
                contour_selection = cv2.bitwise_and(
                    contour_selection,
                    mask[y:y + h, x:x + w],
                )
                is_shadow, shadow_metrics = self._is_cast_shadow(
                    frame,
                    background_image,
                    box,
                    contour_selection,
                )
                if is_shadow:
                    self._last_shadow_rejections.append({
                        "box": box,
                        "priority": bool(is_priority),
                        **(shadow_metrics or {}),
                    })
                    cv2.drawContours(mask, [contour], -1, 0, thickness=cv2.FILLED)
                    continue
            detections.append({
                "box": box,
                "point": point,
                "area": area,
                "bbox_area": bbox_area,
                "motion_fill_ratio": float(motion_pixels) / float(max(1, w * h)),
                "hist": self._histogram(frame, box),
                "priority": is_priority,
                "ambiguous_merged": False,
            })
        return self._suppress_duplicate_detections(detections), mask

    def _same_motion_echo(self, first: dict, second: dict) -> bool:
        """Detect two foreground blobs produced by one fast-moving vehicle.

        Frame differencing contains both the old and current silhouette of a
        fast car.  They have near-identical appearance/size and either overlap
        or almost touch, unlike two independent parked cars.
        """
        first_box, second_box = first["box"], second["box"]
        first_w, first_h = first_box[2], first_box[3]
        second_w, second_h = second_box[2], second_box[3]
        size_ratio = min(first_w * first_h, second_w * second_h) / max(first_w * first_h, second_w * second_h, 1)
        if size_ratio < 0.45:
            return False
        appearance = cv2.compareHist(first["hist"], second["hist"], cv2.HISTCMP_BHATTACHARYYA)
        if appearance > 0.22:
            return False
        distance = float(np.linalg.norm(np.subtract(first["point"], second["point"])))
        allowed_distance = max(24.0, 1.10 * max(min(first_w, first_h), min(second_w, second_h)))
        return distance <= allowed_distance and self._box_gap(first_box, second_box) <= 14.0

    def _suppress_duplicate_detections(self, detections: List[dict]) -> List[dict]:
        """Remove only near-identical blobs; defer uncertain echoes to assignment.

        Two real cars may touch in image space.  The former broad suppression
        discarded one of them before track predictions were available.
        """
        if len(detections) < 2:
            return detections
        kept = []
        for detection in sorted(detections, key=lambda item: item["area"], reverse=True):
            if any(
                self._same_motion_echo(detection, existing)
                and self._iou(detection["box"], existing["box"]) >= 0.25
                and cv2.compareHist(
                    detection["hist"], existing["hist"], cv2.HISTCMP_BHATTACHARYYA
                ) <= 0.12
                for existing in kept
            ):
                continue
            kept.append(detection)
        return kept

    def _is_echo_of_matched_track(self, detection: dict, track: TrackedVehicle) -> bool:
        """Reject a new detection that is the trailing silhouette of a track."""
        if len(track.history) < 4:
            return False
        older_point = track.history[-4]
        if np.linalg.norm(np.subtract(detection["point"], older_point)) > max(22.0, 0.70 * max(track.w, track.h)):
            return False
        reference = {
            "box": track.bbox,
            "point": (track.cx, track.cy),
            "hist": track.appearance,
            "area": track.area,
        }
        return self._same_motion_echo(detection, reference)

    def _predicted_box(self, track: TrackedVehicle, point: Tuple[int, int]) -> Tuple[int, int, int, int]:
        return point[0] - track.w // 2, point[1] - track.h, track.w, track.h

    def _assign(self, detections: List[dict]) -> Tuple[List[Tuple[int, int, Tuple[int, int]]], List[int], List[int]]:
        track_ids = list(self._tracks)
        predictions = {}
        self._ambiguous_detection_ids = set()
        self._viable_pairs = set()
        self._pair_metrics = {}
        for track_id in track_ids:
            predicted = self._tracks[track_id].kalman.predict()  # attached on create
            predictions[track_id] = int(predicted[0, 0]), int(predicted[1, 0])
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))

        costs = np.full((len(track_ids), len(detections)), 10.0, dtype=np.float64)
        reacquire_eligible_track_ids: set[int] = set()
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            predicted_point = predictions[track_id]
            predicted_box = self._predicted_box(track, predicted_point)
            invisible = int(track.consecutive_invisible_count)
            last_seen_timestamp = getattr(track, "last_seen_timestamp_s", None)
            lost_seconds = None
            if (
                invisible > 0
                and self._current_timestamp_s is not None
                and last_seen_timestamp is not None
            ):
                lost_seconds = max(0.0, self._current_timestamp_s - last_seen_timestamp)
                if lost_seconds > self.reacquire_max_seconds:
                    self._last_association_events.append({
                        "type": "association_rejected_stale_track",
                        "local_track_id": int(track_id),
                        "lost_seconds": round(lost_seconds, 3),
                    })
                    continue
            reacquire_eligible_track_ids.add(track_id)
            max_distance = self.max_distance * min(
                1.25,
                1.0 + min(invisible, 15) / 60.0,
            )
            for col, detection in enumerate(detections):
                distance = float(np.linalg.norm(np.subtract(predicted_point, detection["point"])))
                if distance > max_distance:
                    continue
                iou = self._iou(predicted_box, detection["box"])
                current_appearance_distance = (
                    histogram_distance(track.appearance, detection["hist"])
                    if track.appearance is not None
                    else 0.25
                )
                appearance_distance = min(
                    current_appearance_distance,
                    compare_tracklets(track, detection["hist"]).distance,
                )
                if invisible > 0 and appearance_distance > self.lost_appearance_threshold:
                    continue
                track_area = max(1.0, float(track.w * track.h))
                detection_area = max(
                    1.0,
                    float(
                        detection.get(
                            "bbox_area",
                            detection["box"][2] * detection["box"][3],
                        )
                    ),
                )
                area_ratio = detection_area / track_area
                if not (
                    self.min_reacquire_area_ratio
                    <= area_ratio
                    <= self.max_reacquire_area_ratio
                ):
                    continue
                size_error = min(
                    1.0,
                    abs(float(np.log(max(area_ratio, 1e-6))))
                    / abs(float(np.log(self.max_reacquire_area_ratio))),
                )
                cost = (
                    0.45 * (distance / max_distance)
                    + 0.25 * (1.0 - iou)
                    + 0.20 * appearance_distance
                    + 0.10 * size_error
                )
                costs[row, col] = cost
                self._viable_pairs.add((track_id, col))
                self._pair_metrics[(track_id, col)] = {
                    "position": round(distance / max_distance, 4),
                    "iou": round(1.0 - iou, 4),
                    "appearance": round(float(appearance_distance), 4),
                    "size": round(size_error, 4),
                    "total": round(float(cost), 4),
                    "lost_seconds": round(float(lost_seconds or 0.0), 3),
                }

        # One large contour that covers nearby cars is not a valid measurement.
        # Usually it encloses two predicted tracks; it can also enclose one
        # live track while the second car is still tentative/LOST.  In both
        # cases, coast rather than stretching one track over both cars.
        for col, detection in enumerate(detections):
            x, y, width, height = detection["box"]
            compatible = []
            for track_id in track_ids:
                # The ordinary association gate above already rejected this
                # LOST fragment because its reacquire window expired.  It
                # must not come back through the broader merged-contour
                # heuristic and freeze a detection belonging to live cars.
                if track_id not in reacquire_eligible_track_ids:
                    continue
                px, py = predictions[track_id]
                track = self._tracks[track_id]
                if not (x - 12 <= px <= x + width + 12 and y - 12 <= py <= y + height + 12):
                    continue
                distance = float(np.linalg.norm(np.subtract((px, py), detection["point"])))
                if distance > self.max_distance * 1.25:
                    continue
                appearance_distance = (
                    histogram_distance(track.appearance, detection["hist"])
                    if track.appearance is not None
                    else 0.25
                )
                if appearance_distance > max(0.55, self.lost_appearance_threshold):
                    continue
                compatible.append(track_id)
            if not compatible:
                continue
            reference_area = float(
                np.median([self._tracks[track_id].w * self._tracks[track_id].h for track_id in compatible])
            )
            detection_bbox_area = float(
                detection.get(
                    "bbox_area", detection["box"][2] * detection["box"][3]
                )
            )
            if detection_bbox_area < self.merged_detection_area_ratio * max(1.0, reference_area):
                continue
            event_type = "merged_detection_frozen"
            if len(compatible) == 1:
                stale_track_inside = any(
                    track_id not in reacquire_eligible_track_ids
                    and x - 12 <= predictions[track_id][0] <= x + width + 12
                    and y - 12 <= predictions[track_id][1] <= y + height + 12
                    for track_id in track_ids
                )
                # Preserve the stale-track rule: a LOST fragment past its
                # reacquire window cannot make a current vehicle disappear.
                if stale_track_inside:
                    continue
                # A wide/long jump relative to one *fresh* track is the same
                # unsafe measurement, even if the neighbouring car has not
                # yet become a compatible prediction.  Requiring an enlarged
                # area prevents ordinary perspective variation from freezing
                # a valid single-car observation.
                reference_track = self._tracks[compatible[0]]
                width_ratio = width / max(1.0, float(reference_track.w))
                height_ratio = height / max(1.0, float(reference_track.h))
                if width_ratio < 1.30 and height_ratio < 1.30:
                    continue
                event_type = "oversized_detection_frozen"
            detection["ambiguous_merged"] = True
            self._ambiguous_detection_ids.add(col)
            self._viable_pairs.update((track_id, col) for track_id in compatible)
            for row in range(len(track_ids)):
                costs[row, col] = 10.0
            self._last_association_events.append({
                "type": event_type,
                "detection_id": int(col),
                "local_track_ids": [int(value) for value in compatible],
                "bbox": [int(value) for value in detection["box"]],
            })

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.90)
        assignments, unmatched_tracks, unmatched_detections = [], [], set(range(len(detections)))
        for row, track_id in enumerate(track_ids):
            col = int(row_to_col[row])
            if col < 0:
                unmatched_tracks.append(track_id)
            else:
                assignments.append((track_id, col, predictions[track_id]))
                unmatched_detections.discard(col)
        unmatched_detections.difference_update(self._ambiguous_detection_ids)
        return assignments, unmatched_tracks, list(unmatched_detections)

    def _create_or_reid(self, detection: dict) -> None:
        point = detection["point"]

        # ── Bước 0: (Đã chuyển logic khôi phục ID sang two_camera.py để thống nhất quản lý Global ID) ──
        
        # ── Bước 1: Re-ID xe đã rời khung (appearance) ──
        candidate = None
        best_distance = 0.18
        for track_id, old in self._exited_tracks.items():
            if self._frame_idx - old.exited_frame > self.reid_ttl:
                continue
            current_appearance_distance = (
                histogram_distance(old.appearance, detection["hist"])
                if old.appearance is not None
                else 0.25
            )
            distance = min(
                current_appearance_distance,
                compare_tracklets(old, detection["hist"]).distance,
            )
            if distance < best_distance:
                candidate, best_distance = track_id, distance
        if candidate is not None:
            track = self._exited_tracks.pop(candidate)
            track.kalman = self._new_kalman(detection["point"])
            self._tracks[candidate] = track
            # A re-entering fragment gets its own origin. Keeping the historic
            # origin makes direction/displacement tests meaningless after the
            # bounded display history has been trimmed.
            track.first_observation_point = point
            track.first_observation_bbox = detection["box"]
            track.first_observation_frame = self._frame_idx
            track.first_observation_timestamp_s = self._current_timestamp_s
            # Count only detections belonging to this newly visible fragment.
            # ``total_visible_count`` includes the historic fragment after a
            # local Re-ID and therefore cannot safely gate creation of a new
            # Global ID.
            track.fragment_visible_count = 0
            track.fragment_area_history = []
            track.priority_track = bool(detection.get("priority", False))
            track.priority_observation_count = 0
            self._apply_detection(track, detection)
            track.status = TrackStatus.CONFIRMED
            return

        # ── Bước 2: Tạo track hoàn toàn mới ──
        track_id = self._next_id
        self._next_id += 1
        box, point = detection["box"], detection["point"]
        track = TrackedVehicle(
            track_id=track_id, cx=point[0], cy=point[1], bbox=box, area=float(detection["area"]),
            status=TrackStatus.CONFIRMED if self.min_visible_count == 1 else TrackStatus.TENTATIVE,
            history=[point], entered_frame=self._frame_idx, last_seen_frame=self._frame_idx,
            ground_point=self._ground_point(point),
        )
        track.kalman = self._new_kalman(point)
        track.appearance = detection["hist"]
        track.appearance_tracklet = AppearanceTracklet(
            max_samples=self.tracklet_max_samples,
            sample_interval=self.tracklet_sample_interval,
        )
        track.appearance_tracklet.update(detection["hist"], self._frame_idx)
        track.association_state = "new_tentative"
        track.assignment_cost = {}
        # Preserve fragment origin independently of ``history``. The latter is
        # intentionally bounded and must not silently move the origin used by
        # confirmation or slot-departure direction checks.
        track.first_observation_point = point
        track.first_observation_bbox = box
        track.first_observation_frame = self._frame_idx
        track.first_observation_timestamp_s = self._current_timestamp_s
        track.last_seen_timestamp_s = self._current_timestamp_s
        track.fragment_visible_count = 1
        track.fragment_area_history = [float(detection["area"])]
        track.priority_track = bool(detection.get("priority", False))
        track.priority_observation_count = 1 if track.priority_track else 0
        self._tracks[track_id] = track

    def _is_confirmable(self, track: TrackedVehicle) -> bool:
        priority_track = bool(getattr(track, "priority_track", False))
        min_visible_count = self.priority_min_visible_count if priority_track else self.min_visible_count
        min_displacement = (
            self.priority_min_confirm_displacement
            if priority_track
            else self.min_confirm_displacement
        )
        if track.total_visible_count < min_visible_count:
            return False
        if priority_track and getattr(track, "priority_observation_count", 0) < self.priority_min_visible_count:
            return False
        if min_displacement <= 0:
            return True
        # Stable fragment origins let the multiscale DroidCam mode accumulate
        # very slow travel beyond the bounded display history.  Keep the
        # legacy rolling-window confirmation semantics for older entry points
        # that did not opt into multiscale motion.
        origin = (
            getattr(track, "first_observation_point", track.history[0])
            if self.enable_multiscale_motion
            else track.history[0]
        )
        return np.linalg.norm(np.subtract((track.cx, track.cy), origin)) >= min_displacement

    def _apply_detection(self, track: TrackedVehicle, detection: dict) -> None:
        point, box = detection["point"], detection["box"]
        track.kalman.correct(np.array([[point[0]], [point[1]]], dtype=np.float32))
        track.cx, track.cy, track.bbox, track.area = point[0], point[1], box, float(detection["area"])
        track.age += 1
        track.total_visible_count += 1
        track.fragment_visible_count = int(
            getattr(track, "fragment_visible_count", 0)
        ) + 1
        fragment_areas = list(getattr(track, "fragment_area_history", ()))
        fragment_areas.append(float(detection["area"]))
        track.fragment_area_history = fragment_areas[-12:]
        track.consecutive_invisible_count = 0
        track.last_seen_frame = self._frame_idx
        track.last_seen_timestamp_s = self._current_timestamp_s
        track.ground_point = self._ground_point(point)
        if track.appearance is None:
            track.appearance = detection["hist"].copy()
        else:
            track.appearance = cv2.addWeighted(
                track.appearance, 0.75, detection["hist"], 0.25, 0
            )
        if track.appearance_tracklet is None:
            track.appearance_tracklet = AppearanceTracklet(
                max_samples=self.tracklet_max_samples,
                sample_interval=self.tracklet_sample_interval,
            )
        track.appearance_tracklet.update(detection["hist"], self._frame_idx)
        if detection.get("priority", False):
            track.priority_track = True
            track.priority_observation_count = getattr(track, "priority_observation_count", 0) + 1
        track.status = TrackStatus.CONFIRMED if self._is_confirmable(track) else TrackStatus.TENTATIVE
        track.history.append(point)
        if len(track.history) > self.history_len:
            track.history = track.history[-self.history_len:]

    def suspend_track(self, track_id: int, reason: str = "parked") -> Optional[TrackedVehicle]:
        """Remove a parked local fragment from ordinary motion association."""
        track = self._tracks.pop(int(track_id), None)
        if track is None:
            return None
        track.association_state = f"suspended_{reason}"
        self._suspended_tracks[int(track_id)] = track
        self._last_association_events.append({
            "type": "parked_local_track_suspended",
            "local_track_id": int(track_id),
            "reason": str(reason),
        })
        return track

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: Optional[float] = None,
        priority_regions: Optional[Sequence[Any]] = None,
    ):
        """Process one frame while preserving the legacy three-value return.

        ``timestamp_s`` makes slow/fast temporal references independent of
        effective FPS. ``priority_regions`` are temporary recovery polygons in
        which smaller slow-moving blobs may become *tentative* candidates. A
        priority candidate still needs two observations and displacement, so a
        one-frame shadow/noise blob cannot become a confirmed/local Global-ID
        candidate.
        """
        self._frame_idx += 1
        self._current_timestamp_s = (
            float(timestamp_s)
            if timestamp_s is not None and np.isfinite(timestamp_s)
            else None
        )
        self._last_association_events = []
        self._newly_lost_tracks = []
        detections, mask = self._detect(
            frame,
            timestamp_s=self._current_timestamp_s,
            priority_regions=priority_regions,
        )
        assignments, unmatched_tracks, unmatched_detections = self._assign(detections)
        for track_id, detection_id, _ in assignments:
            self._apply_detection(self._tracks[track_id], detections[detection_id])
            track = self._tracks[track_id]
            track.association_state = "matched"
            track.assignment_cost = dict(
                self._pair_metrics.get((track_id, detection_id), {})
            )
        expired = []
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.age += 1
            track.consecutive_invisible_count += 1
            frozen = any(
                detection_id in self._ambiguous_detection_ids
                and (track_id, detection_id) in self._viable_pairs
                for detection_id in self._ambiguous_detection_ids
            )
            track.association_state = "frozen_ambiguous" if frozen else "coasting"
            track.assignment_cost = {}
            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.LOST
                self._newly_lost_tracks.append((track_id, track))
            if track.consecutive_invisible_count > self.lost_track_ttl:
                expired.append(track_id)
        expired_tracks = []  # Danh sách tracks vừa expire frame này
        for track_id in expired:
            track = self._tracks.pop(track_id)
            # Blob nhiễu chưa từng đủ điều kiện xác nhận không được đưa vào output.
            if track.status != TrackStatus.TENTATIVE:
                track.exited_frame = self._frame_idx
                self._exited_tracks[track_id] = track
                expired_tracks.append((track_id, track))
        # Do not create a second local track from the old silhouette of a
        # vehicle that already received this frame's primary detection.
        matched_tracks = [self._tracks[track_id] for track_id, _, _ in assignments]
        unmatched_track_ids = set(unmatched_tracks)
        unmatched_detections = [
            detection_id for detection_id in unmatched_detections
            if not (
                any(
                    self._is_echo_of_matched_track(detections[detection_id], track)
                    for track in matched_tracks
                )
                and not any(
                    (track_id, detection_id) in self._viable_pairs
                    for track_id in unmatched_track_ids
                )
            )
        ]
        for detection_id in unmatched_detections:
            self._create_or_reid(detections[detection_id])
        return self._tracks, mask, expired_tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks=None,
        show_non_active: bool = False,
        id_overrides: Optional[Dict[int, int]] = None,
        confirmed_color: Tuple[int, int, int] = (0, 255, 0),
        confirmed_label: Optional[str] = None,
        point_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        out = frame.copy()
        tracks = self._tracks if tracks is None else tracks
        for track in tracks.values():
            if not show_non_active and track.status != TrackStatus.CONFIRMED:
                continue
            color = confirmed_color if track.status == TrackStatus.CONFIRMED else (0, 165, 255)
            cv2.rectangle(out, (track.x, track.y), (track.x + track.w, track.y + track.h), color, 2)
            shown_id = id_overrides.get(track.track_id, track.track_id) if id_overrides else track.track_id
            status_label = confirmed_label if track.status == TrackStatus.CONFIRMED and confirmed_label else track.status.value
            cv2.putText(out, f"G#{shown_id} {status_label}", (track.x, max(16, track.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(out, (track.cx, track.cy), 3, point_color, -1)
        return out

    @property
    def confirmed_tracks(self):
        return {track_id: track for track_id, track in self._tracks.items() if track.status in (TrackStatus.CONFIRMED, TrackStatus.LOST)}

    @property
    def active_tracks(self):
        return {track_id: track for track_id, track in self._tracks.items() if track.status == TrackStatus.CONFIRMED and track.consecutive_invisible_count == 0}

    @property
    def observable_tracks(self):
        """Tracks detected in the current frame, including tentative ones.

        Cross-camera handoff needs the first observation in the destination
        camera.  ``active_tracks`` deliberately hides tentative tracks from the
        UI and JSON, while this view lets the global-ID manager bind an old ID
        before local confirmation completes.
        """
        return {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.consecutive_invisible_count == 0
            and track.status in (TrackStatus.TENTATIVE, TrackStatus.CONFIRMED)
        }

    @property
    def newly_lost_tracks(self):
        """Confirmed tracks that disappeared in this exact frame."""
        return list(self._newly_lost_tracks)

    @property
    def all_tracks(self):
        return dict(self._tracks)

    @property
    def exited_tracks(self):
        return dict(self._exited_tracks)

    @property
    def frame_index(self):
        return self._frame_idx

    @property
    def last_shadow_rejections(self):
        """Cast-shadow blobs rejected in the latest frame, for diagnostics."""
        return [dict(item) for item in self._last_shadow_rejections]

    @property
    def last_detection_rejections(self):
        """Oversized/out-of-ROI foreground blobs rejected this frame."""
        return [dict(item) for item in self._last_detection_rejections]

    @property
    def association_events(self):
        return [dict(item) for item in self._last_association_events]

    def local_track_telemetry(self, global_ids: Optional[Dict[int, int]] = None) -> List[dict]:
        """Return JSON-safe association evidence for experiment recordings."""
        global_ids = global_ids or {}
        records = []
        for local_id, track in sorted(self._tracks.items()):
            records.append({
                "local_track_id": int(local_id),
                "global_id": (
                    int(global_ids[local_id]) if local_id in global_ids else None
                ),
                "bbox": [int(value) for value in track.bbox],
                "center": [int(track.cx), int(track.cy)],
                "state": track.status.value,
                "invisible_count": int(track.consecutive_invisible_count),
                "association_state": str(
                    getattr(track, "association_state", "unknown")
                ),
                "assignment_cost": dict(
                    getattr(track, "assignment_cost", {}) or {}
                ),
                "fragment_visible_count": int(
                    getattr(track, "fragment_visible_count", 0)
                ),
                "first_observation_frame": int(
                    getattr(track, "first_observation_frame", self._frame_idx)
                ),
            })
        return records
