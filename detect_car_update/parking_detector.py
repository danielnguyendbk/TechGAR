"""Parking slot detector — tách logic ensemble detection từ ensemble_test.py.

Module này cung cấp class ParkingDetector dùng được như library,
không có UI/trackbar/main loop.

Thuật toán:
  1. Ensemble 25 biến thể (5 delta-gamma × 5 delta-CLAHE) vote trống/đầy
  2. Pass 1: Center Cluster Rescue — cứu xe trắng bị miss ở center
  3. Pass 2: Edge Recheck — cứu xe có edge ratio cao
  4. Temporal Smoothing — ổn định kết quả qua nhiều frame
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SlotResult:
    """Kết quả nhận diện cho 1 ô đỗ."""
    slot_id: str
    occupied: bool
    polygon: np.ndarray          # (N, 2) int32 — tọa độ đã scale
    center: Tuple[int, int]
    vehicle_id: Optional[int] = None   # Sẽ được binder gán sau


@dataclass
class _PrecomputedROI:
    slot_id: str
    polygon_pts: np.ndarray       # (N, 2) int32 đã scale
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    mask: np.ndarray              # local binary mask
    area: int
    center: Tuple[int, int]


class TemporalSmoother:
    """Smoothing trạng thái ô đỗ qua nhiều frame liên tiếp."""

    def __init__(self, num_slots: int, required_frames: int = 5):
        self.required = required_frames
        self.counters = [0] * num_slots
        self.pending: List[Optional[bool]] = [None] * num_slots
        self.confirmed = [False] * num_slots

    def update(self, slot_idx: int, is_occupied: bool) -> bool:
        if self.pending[slot_idx] == is_occupied:
            self.counters[slot_idx] += 1
        else:
            self.pending[slot_idx] = is_occupied
            self.counters[slot_idx] = 1
        if self.counters[slot_idx] >= self.required:
            self.confirmed[slot_idx] = is_occupied
        return self.confirmed[slot_idx]


class ParkingDetector:
    """Ensemble parking detection — dùng được như library.

    Params
    ------
    slots_file : str
        Đường dẫn tới file JSON chứa danh sách ô đỗ (polygon).
    base_gamma, base_clahe, clahe_grid, ratio_thr, edge_thr : float
        Tham số ensemble mặc định.  Có thể override mỗi frame.
    smoothing_frames : int
        Số frame liên tiếp cần đồng thuận trước khi đổi trạng thái.
        0 = tắt smoothing.
    """

    def __init__(
        self,
        slots_file: str,
        base_gamma: float = 2.8,
        base_clahe: float = 2.0,
        clahe_grid: int = 8,
        ratio_thr: float = 0.20,
        edge_thr: float = 0.25,
        smoothing_frames: int = 5,
    ):
        self.base_gamma = base_gamma
        self.base_clahe = base_clahe
        self.clahe_grid = max(2, clahe_grid)
        self.ratio_thr = ratio_thr
        self.edge_thr = edge_thr
        self.smoothing_frames = smoothing_frames

        # Load slots
        path = Path(slots_file)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy slots file: {slots_file}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self._raw_slots = data["slots"]
        self._img_w_ref = data["imageWidth"]
        self._img_h_ref = data["imageHeight"]

        # Parse polygon
        self._slot_polygons: List[dict] = []
        for s in self._raw_slots:
            poly = self._get_polygon(s)
            if poly:
                s["_polygon"] = poly
                self._slot_polygons.append(s)

        self._rois: List[Optional[_PrecomputedROI]] = []
        self._smoother: Optional[TemporalSmoother] = None
        self._lut_cache: Dict[float, np.ndarray] = {}
        self._initialized = False

    @property
    def slot_count(self) -> int:
        return len(self._slot_polygons)

    @property
    def slot_ids(self) -> List[str]:
        return [s["id"] for s in self._slot_polygons]

    @staticmethod
    def _get_polygon(slot: dict) -> Optional[list]:
        for key in ("polygon", "points", "coordinates", "vertices"):
            if key in slot and slot[key]:
                return slot[key]
        if "rect" in slot and slot["rect"]:
            r = slot["rect"]
            cx, cy, w, h = r["cx"], r["cy"], r["w"], r["h"]
            angle = r.get("angle", 0)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            hw, hh = w / 2, h / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            return [
                {"x": cx + dx * cos_a - dy * sin_a, "y": cy + dx * sin_a + dy * cos_a}
                for dx, dy in corners
            ]
        return None

    def _compute_rois(self, img_shape: Tuple[int, ...]) -> None:
        h, w = img_shape[:2]
        sx = w / self._img_w_ref
        sy = h / self._img_h_ref
        self._rois = []

        for slot in self._slot_polygons:
            poly = slot["_polygon"]
            try:
                if isinstance(poly[0], dict):
                    pts = np.array(
                        [[int(p["x"] * sx), int(p["y"] * sy)] for p in poly], np.int32
                    )
                else:
                    pts = np.array(
                        [[int(p[0] * sx), int(p[1] * sy)] for p in poly], np.int32
                    )
            except Exception:
                self._rois.append(None)
                continue

            x_min = max(0, int(np.min(pts[:, 0])) - 2)
            y_min = max(0, int(np.min(pts[:, 1])) - 2)
            x_max = min(w, int(np.max(pts[:, 0])) + 2)
            y_max = min(h, int(np.max(pts[:, 1])) + 2)

            roi_w, roi_h = x_max - x_min, y_max - y_min
            local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            local_pts = pts.copy()
            local_pts[:, 0] -= x_min
            local_pts[:, 1] -= y_min
            cv2.fillPoly(local_mask, [local_pts], 255)
            area = cv2.countNonZero(local_mask)
            center = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

            self._rois.append(
                _PrecomputedROI(
                    slot_id=slot["id"],
                    polygon_pts=pts,
                    bbox=(x_min, y_min, x_max, y_max),
                    mask=local_mask,
                    area=area,
                    center=center,
                )
            )

        self._smoother = TemporalSmoother(len(self._rois), self.smoothing_frames)
        self._initialized = True

    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        gamma = round(gamma, 1)
        if gamma not in self._lut_cache:
            self._lut_cache[gamma] = np.array(
                [np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255) for i in range(256)],
                dtype=np.uint8,
            )
        return self._lut_cache[gamma]

    def detect(
        self,
        frame: np.ndarray,
        apply_smoothing: bool = True,
    ) -> List[SlotResult]:
        """Phân tích 1 frame, trả về danh sách SlotResult."""
        if not self._initialized:
            self._compute_rois(frame.shape)

        base_gamma = self.base_gamma
        base_clahe = self.base_clahe
        clahe_grid = self.clahe_grid
        ratio_thr = self.ratio_thr
        edge_thr = self.edge_thr

        delta_gamma = [-0.2, -0.1, 0.0, 0.1, 0.2]
        delta_clahe = [-0.5, -0.2, 0.0, 0.2, 0.5]
        total_combinations = len(delta_clahe) * len(delta_gamma)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        kernel = np.ones((3, 3), np.uint8)
        empty_votes = [0] * len(self._rois)
        base_dilated = None

        for dc in delta_clahe:
            c_val = max(0.1, base_clahe + dc)
            clahe = cv2.createCLAHE(clipLimit=c_val, tileGridSize=(clahe_grid, clahe_grid))
            l_clahe = clahe.apply(l_channel)

            for dg in delta_gamma:
                g_val = max(0.1, base_gamma + dg)
                lut = self._get_gamma_lut(g_val)
                l_gamma = cv2.LUT(l_clahe, lut)

                blur = cv2.GaussianBlur(l_gamma, (3, 3), 1)
                thresh = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
                )
                median = cv2.medianBlur(thresh, 5)
                dilated = cv2.dilate(median, kernel, iterations=1)

                if dc == 0.0 and dg == 0.0:
                    base_dilated = dilated.copy()

                for i, roi in enumerate(self._rois):
                    if roi is None or roi.area == 0:
                        continue
                    x1, y1, x2, y2 = roi.bbox
                    roi_thresh = dilated[y1:y2, x1:x2]
                    masked = cv2.bitwise_and(roi_thresh, roi_thresh, mask=roi.mask)
                    count = cv2.countNonZero(masked)
                    if count / roi.area < ratio_thr:
                        empty_votes[i] += 1

        # ── Pass 1: Threshold + Center Cluster ──
        required_votes = total_combinations // 2
        is_free_list = [True] * len(self._rois)

        for i, roi in enumerate(self._rois):
            if roi is None:
                continue
            is_free = empty_votes[i] >= required_votes

            if is_free and base_dilated is not None and roi.area > 0:
                x1, y1, x2, y2 = roi.bbox
                center_mask = np.zeros_like(roi.mask)
                pts_local = roi.polygon_pts.copy()
                pts_local[:, 0] -= x1
                pts_local[:, 1] -= y1
                pts_f = pts_local.astype(np.float32)
                centroid_x = np.mean(pts_f[:, 0])
                centroid_y = np.mean(pts_f[:, 1])
                shrink = 0.4
                center_pts = np.array(
                    [
                        [
                            int(centroid_x + (px - centroid_x) * shrink),
                            int(centroid_y + (py - centroid_y) * shrink),
                        ]
                        for px, py in pts_f
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(center_mask, [center_pts], 255)
                center_area = cv2.countNonZero(center_mask)

                if center_area > 5:
                    roi_t = base_dilated[y1:y2, x1:x2]
                    full_masked = cv2.bitwise_and(roi_t, roi_t, mask=roi.mask)
                    full_ratio = cv2.countNonZero(full_masked) / roi.area
                    center_masked = cv2.bitwise_and(roi_t, roi_t, mask=center_mask)
                    center_ratio = cv2.countNonZero(center_masked) / center_area

                    if center_ratio >= 0.05 and (
                        full_ratio < 0.01 or center_ratio > full_ratio * 2
                    ):
                        is_free = False

            is_free_list[i] = is_free

        # ── Pass 2: Edge Recheck ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(gray_blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        for i, roi in enumerate(self._rois):
            if roi is None or roi.area == 0:
                continue
            x1, y1, x2, y2 = roi.bbox
            roi_edges = edges[y1:y2, x1:x2]
            masked_e = cv2.bitwise_and(roi_edges, roi_edges, mask=roi.mask)
            edge_ratio = cv2.countNonZero(masked_e) / roi.area
            if is_free_list[i] and edge_ratio >= edge_thr:
                is_free_list[i] = False

        # ── Build results ──
        results: List[SlotResult] = []
        for i, roi in enumerate(self._rois):
            if roi is None:
                continue

            is_free = is_free_list[i]
            if apply_smoothing and self._smoother is not None:
                is_free = not self._smoother.update(i, not is_free)

            results.append(
                SlotResult(
                    slot_id=roi.slot_id,
                    occupied=not is_free,
                    polygon=roi.polygon_pts,
                    center=roi.center,
                )
            )

        return results

    def get_roi_polygon(self, slot_id: str) -> Optional[np.ndarray]:
        """Trả polygon đã scale cho 1 slot ID."""
        for roi in self._rois:
            if roi is not None and roi.slot_id == slot_id:
                return roi.polygon_pts
        return None

    def draw_results(
        self,
        frame: np.ndarray,
        results: List[SlotResult],
    ) -> np.ndarray:
        """Vẽ overlay polygon lên frame."""
        out = frame.copy()
        free_count = sum(1 for r in results if not r.occupied)
        total = len(results)

        for result in results:
            color = (0, 0, 255) if result.occupied else (0, 255, 0)
            cv2.polylines(out, [result.polygon], True, color, 2)
            label = result.slot_id
            if result.vehicle_id is not None:
                label += f" #{result.vehicle_id}"
            cv2.putText(
                out, label, (result.center[0] - 12, result.center[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
            )

        cv2.putText(
            out,
            f"Parking: {free_count}/{total} free",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )
        return out
