"""Fallback tracker cho camera bãi xe nhìn từ trên cao.

COCO YOLO không nhận được xe quá nhỏ/nhìn top-down trong video carPark mẫu.
Backend này dùng foreground motion để tạo detection, nhưng thay Particle Filter
cũ bằng Kalman constant-velocity + global assignment (LAPJV) + HSV appearance
histogram để hạn chế đổi ID và hỗ trợ Re-ID ngắn hạn.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from lap import lapjv

from .vehicle_tracker import TrackStatus, TrackedVehicle


class MotionVehicleTracker:
    """Tracker không phụ thuộc model, tối ưu cho camera cố định có xe chuyển động."""

    def __init__(
        self,
        min_visible_count: int = 3,
        lost_track_ttl: int = 90,
        history_len: int = 90,
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
        slot_binder=None,  # SlotVehicleBinder instance (optional)
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
        self.reid_ttl = max(reid_ttl, lost_track_ttl)
        self.homography = homography
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=32, detectShadows=True)
        self._tracks: Dict[int, TrackedVehicle] = {}
        self._exited_tracks: Dict[int, TrackedVehicle] = {}
        self._next_id = 1
        self._frame_idx = 0
        self._gray_history = deque(maxlen=self.motion_frame_gap + 1)
        self.slot_binder = slot_binder  # Tham chiếu tới SlotVehicleBinder
        self._newly_lost_tracks: List[Tuple[int, TrackedVehicle]] = []

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
        x, y, w, h = box
        crop = frame[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        if crop.size == 0:
            return np.zeros((16, 16), dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).astype(np.float32)

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

    def _temporal_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Chỉ giữ pixel thực sự thay đổi giữa hai thời điểm.

        MOG2 có thể đánh dấu xe đỗ khi ánh sáng/nén video thay đổi. Frame
        difference qua nhiều frame giúp bỏ các blob đứng yên. Median brightness
        shift được trừ trước để không coi thay đổi phơi sáng toàn khung là xe.
        """
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self._gray_history.append(gray)
        if len(self._gray_history) <= self.motion_frame_gap:
            return np.zeros_like(gray)
        reference = self._gray_history[0]
        brightness_shift = float(np.median(gray.astype(np.int16) - reference.astype(np.int16)))
        adjusted_reference = cv2.convertScaleAbs(reference, alpha=1.0, beta=brightness_shift)
        difference = cv2.absdiff(gray, adjusted_reference)
        _, motion = cv2.threshold(difference, self.motion_threshold, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=2)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
        return motion

    def _detect(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        background_mask = self.bg_sub.apply(frame)
        _, background_mask = cv2.threshold(background_mask, 200, 255, cv2.THRESH_BINARY)
        temporal_motion = self._temporal_motion_mask(frame)
        # Motion mask là cổng bắt buộc: foreground đứng yên không được thành xe.
        support = cv2.dilate(temporal_motion, np.ones((17, 17), np.uint8), iterations=1)
        mask = cv2.bitwise_and(background_mask, support)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = frame.shape[0] * frame.shape[1]
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > image_area * 0.22:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width or h < self.min_height:
                continue
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.25 or aspect_ratio > 5.0:
                continue
            box = (x, y, w, h)
            motion_pixels = cv2.countNonZero(temporal_motion[y:y + h, x:x + w])
            if motion_pixels < self.motion_min_pixels or motion_pixels / float(w * h) < self.motion_min_ratio:
                continue
            detections.append({"box": box, "point": self._bottom_center(box), "area": area, "hist": self._histogram(frame, box)})
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
        """Keep one detection for a same-frame old/current motion echo pair."""
        if len(detections) < 2:
            return detections
        kept = []
        for detection in sorted(detections, key=lambda item: item["area"], reverse=True):
            if any(self._same_motion_echo(detection, existing) for existing in kept):
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
        for track_id in track_ids:
            predicted = self._tracks[track_id].kalman.predict()  # attached on create
            predictions[track_id] = int(predicted[0, 0]), int(predicted[1, 0])
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))

        costs = np.full((len(track_ids), len(detections)), 10.0, dtype=np.float64)
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            predicted_point = predictions[track_id]
            predicted_box = self._predicted_box(track, predicted_point)
            max_distance = self.max_distance * (1.0 + min(track.consecutive_invisible_count, 30) / 30.0)
            for col, detection in enumerate(detections):
                distance = float(np.linalg.norm(np.subtract(predicted_point, detection["point"])))
                if distance > max_distance:
                    continue
                iou = self._iou(predicted_box, detection["box"])
                appearance_distance = cv2.compareHist(track.appearance, detection["hist"], cv2.HISTCMP_BHATTACHARYYA)
                costs[row, col] = 0.50 * (distance / max_distance) + 0.30 * (1.0 - iou) + 0.20 * appearance_distance

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.90)
        assignments, unmatched_tracks, unmatched_detections = [], [], set(range(len(detections)))
        for row, track_id in enumerate(track_ids):
            col = int(row_to_col[row])
            if col < 0:
                unmatched_tracks.append(track_id)
            else:
                assignments.append((track_id, col, predictions[track_id]))
                unmatched_detections.discard(col)
        return assignments, unmatched_tracks, list(unmatched_detections)

    def _create_or_reid(self, detection: dict) -> None:
        point = detection["point"]

        # ── Bước 0: Kiểm tra Slot Binder (ưu tiên cao nhất) ──
        if self.slot_binder is not None:
            recovered_id = self.slot_binder.try_recover_id(point)
            if recovered_id is not None:
                # Khôi phục track với ID cũ từ ô đỗ
                box = detection["box"]
                track = TrackedVehicle(
                    track_id=recovered_id, cx=point[0], cy=point[1],
                    bbox=box, area=float(detection["area"]),
                    status=TrackStatus.CONFIRMED,
                    history=[point], entered_frame=self._frame_idx,
                    last_seen_frame=self._frame_idx,
                    ground_point=self._ground_point(point),
                )
                track.kalman = self._new_kalman(point)
                track.appearance = detection["hist"]
                self._tracks[recovered_id] = track
                # Xóa khỏi exited nếu có
                self._exited_tracks.pop(recovered_id, None)
                return

        # ── Bước 1: Re-ID xe đã rời khung (appearance) ──
        candidate = None
        best_distance = 0.18
        for track_id, old in self._exited_tracks.items():
            if self._frame_idx - old.exited_frame > self.reid_ttl:
                continue
            distance = cv2.compareHist(old.appearance, detection["hist"], cv2.HISTCMP_BHATTACHARYYA)
            if distance < best_distance:
                candidate, best_distance = track_id, distance
        if candidate is not None:
            track = self._exited_tracks.pop(candidate)
            track.kalman = self._new_kalman(detection["point"])
            self._tracks[candidate] = track
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
        self._tracks[track_id] = track

    def _is_confirmable(self, track: TrackedVehicle) -> bool:
        if track.total_visible_count < self.min_visible_count:
            return False
        if self.min_confirm_displacement <= 0:
            return True
        origin = track.history[0]
        return np.linalg.norm(np.subtract((track.cx, track.cy), origin)) >= self.min_confirm_displacement

    def _apply_detection(self, track: TrackedVehicle, detection: dict) -> None:
        point, box = detection["point"], detection["box"]
        track.kalman.correct(np.array([[point[0]], [point[1]]], dtype=np.float32))
        track.cx, track.cy, track.bbox, track.area = point[0], point[1], box, float(detection["area"])
        track.age += 1
        track.total_visible_count += 1
        track.consecutive_invisible_count = 0
        track.last_seen_frame = self._frame_idx
        track.ground_point = self._ground_point(point)
        track.appearance = cv2.addWeighted(track.appearance, 0.75, detection["hist"], 0.25, 0)
        track.status = TrackStatus.CONFIRMED if self._is_confirmable(track) else TrackStatus.TENTATIVE
        track.history.append(point)
        if len(track.history) > self.history_len:
            track.history = track.history[-self.history_len:]

    def process_frame(self, frame: np.ndarray):
        self._frame_idx += 1
        self._newly_lost_tracks = []
        detections, mask = self._detect(frame)
        assignments, unmatched_tracks, unmatched_detections = self._assign(detections)
        for track_id, detection_id, _ in assignments:
            self._apply_detection(self._tracks[track_id], detections[detection_id])
        expired = []
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.age += 1
            track.consecutive_invisible_count += 1
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
        unmatched_detections = [
            detection_id for detection_id in unmatched_detections
            if not any(self._is_echo_of_matched_track(detections[detection_id], track) for track in matched_tracks)
        ]
        for detection_id in unmatched_detections:
            self._create_or_reid(detections[detection_id])
        return self._tracks, mask, expired_tracks

    def draw_tracks(self, frame: np.ndarray, tracks=None, show_non_active: bool = False, id_overrides: Optional[Dict[int, int]] = None) -> np.ndarray:
        out = frame.copy()
        tracks = tracks or self._tracks
        for track in tracks.values():
            if not show_non_active and track.status != TrackStatus.CONFIRMED:
                continue
            color = (0, 255, 0) if track.status == TrackStatus.CONFIRMED else (0, 165, 255)
            cv2.rectangle(out, (track.x, track.y), (track.x + track.w, track.y + track.h), color, 2)
            shown_id = id_overrides.get(track.track_id, track.track_id) if id_overrides else track.track_id
            cv2.putText(out, f"G#{shown_id} {track.status.value}", (track.x, max(16, track.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(out, (track.cx, track.cy), 3, (0, 0, 255), -1)
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
