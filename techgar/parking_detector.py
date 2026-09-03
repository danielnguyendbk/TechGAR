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
    evidence: float = 0.0


@dataclass
class _PrecomputedROI:
    slot_id: str
    polygon_pts: np.ndarray       # (N, 2) int32 đã scale
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    mask: np.ndarray              # local binary mask
    area: int
    analysis_mask: np.ndarray
    analysis_boundary: np.ndarray
    analysis_area: int
    analysis_size: Tuple[int, int]
    core_mask: np.ndarray
    core_area: int
    center: Tuple[int, int]


@dataclass
class _ROIEvidence:
    raw_ratio: float
    filtered_ratio: float
    core_ratio: float
    core_component_ratio: float
    core_component_count: int
    filtered_mask: np.ndarray = field(repr=False)


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
        use_edge_recheck: bool = True,
        border_ignore_ratio: float = 0.12,
        line_min_span_ratio: float = 0.45,
        line_max_thickness_ratio: float = 0.18,
        core_scale: float = 0.55,
        core_ratio_threshold: float = 0.18,
        core_component_threshold: float = 0.08,
        # VR5: Number of gamma/CLAHE variants (3×3=9 default, 5×5=25 original)
        gamma_variants: int = 3,
        clahe_variants: int = 3,
    ):
        self.base_gamma = base_gamma
        self.base_clahe = base_clahe
        self.clahe_grid = max(2, clahe_grid)
        self.ratio_thr = ratio_thr
        self.edge_thr = edge_thr
        self.smoothing_frames = smoothing_frames
        self.use_edge_recheck = bool(use_edge_recheck)
        self.border_ignore_ratio = min(0.40, max(0.0, float(border_ignore_ratio)))
        self.line_min_span_ratio = min(1.0, max(0.05, float(line_min_span_ratio)))
        self.line_max_thickness_ratio = min(1.0, max(0.01, float(line_max_thickness_ratio)))
        self.core_scale = min(0.95, max(0.10, float(core_scale)))
        self.core_ratio_threshold = min(1.0, max(0.0, float(core_ratio_threshold)))
        self.core_component_threshold = min(1.0, max(0.0, float(core_component_threshold)))
        
        # VR5: Store variant counts
        self.gamma_variants = max(1, min(5, int(gamma_variants)))
        self.clahe_variants = max(1, min(5, int(clahe_variants)))

        # Load slots
        self._slot_polygons: List[dict] = []
        if isinstance(slots_file, dict):
            if "slots" in slots_file:
                self._raw_slots = slots_file["slots"]
                self._img_w_ref = slots_file.get("imageWidth", 1280)
                self._img_h_ref = slots_file.get("imageHeight", 720)
                for s in self._raw_slots:
                    poly = self._get_polygon(s)
                    if poly:
                        s["_polygon"] = poly
                        self._slot_polygons.append(s)
            else:
                self._raw_slots = []
                self._img_w_ref = 1280
                self._img_h_ref = 720
                for slot_id, poly in slots_file.items():
                    poly_list = poly.tolist() if hasattr(poly, "tolist") else list(poly)
                    slot_entry = {"id": str(slot_id), "_polygon": poly_list}
                    self._raw_slots.append(slot_entry)
                    self._slot_polygons.append(slot_entry)
        else:
            path = Path(slots_file)
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy slots file: {slots_file}")
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self._raw_slots = data["slots"]
            self._img_w_ref = data.get("imageWidth", 1280)
            self._img_h_ref = data.get("imageHeight", 720)
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

    def get_global_mask(self, img_shape: Tuple[int, ...]) -> np.ndarray:
        if not self._initialized:
            self._compute_rois(img_shape)
        h, w = img_shape[:2]
        global_mask = np.zeros((h, w), dtype=np.uint8)
        for roi in self._rois:
            if roi is None:
                continue
            x1, y1, x2, y2 = roi.bbox
            global_mask[y1:y2, x1:x2] = cv2.bitwise_or(global_mask[y1:y2, x1:x2], roi.mask)
        return global_mask

    def configure_roi_filter(self, values: dict) -> None:
        """Apply ROI-filter parameters and rebuild geometric masks when needed."""
        old_geometry = (self.border_ignore_ratio, self.core_scale)
        self.border_ignore_ratio = min(
            0.40, max(0.0, float(values.get("border_ignore_ratio", self.border_ignore_ratio)))
        )
        self.line_min_span_ratio = min(
            1.0, max(0.05, float(values.get("line_min_span_ratio", self.line_min_span_ratio)))
        )
        self.line_max_thickness_ratio = min(
            1.0,
            max(0.01, float(values.get("line_max_thickness_ratio", self.line_max_thickness_ratio))),
        )
        self.core_scale = min(0.95, max(0.10, float(values.get("core_scale", self.core_scale))))
        self.core_ratio_threshold = min(
            1.0, max(0.0, float(values.get("core_ratio_threshold", self.core_ratio_threshold)))
        )
        self.core_component_threshold = min(
            1.0,
            max(0.0, float(values.get("core_component_threshold", self.core_component_threshold))),
        )
        if old_geometry != (self.border_ignore_ratio, self.core_scale):
            self._initialized = False

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
        all_x, all_y = [], []
        for s in self._slot_polygons:
            for p in s["_polygon"]:
                all_x.append(p["x"] if isinstance(p, dict) else p[0])
                all_y.append(p["y"] if isinstance(p, dict) else p[1])
        if all_x and max(all_x) <= w + 10 and max(all_y) <= h + 10:
            sx, sy = 1.0, 1.0
        else:
            sx = w / max(1, float(self._img_w_ref))
            sy = h / max(1, float(self._img_h_ref))
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

            min_dimension = max(1, min(roi_w, roi_h))
            inset_pixels = int(round(min_dimension * self.border_ignore_ratio))
            if inset_pixels > 0:
                inset_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (inset_pixels * 2 + 1, inset_pixels * 2 + 1),
                )
                analysis_mask = cv2.erode(local_mask, inset_kernel, iterations=1)
            else:
                analysis_mask = local_mask.copy()
            if cv2.countNonZero(analysis_mask) < 5:
                analysis_mask = local_mask.copy()

            inner_analysis = cv2.erode(
                analysis_mask, np.ones((3, 3), dtype=np.uint8), iterations=1
            )
            analysis_boundary = cv2.subtract(analysis_mask, inner_analysis)
            analysis_area = cv2.countNonZero(analysis_mask)
            analysis_points = cv2.findNonZero(analysis_mask)
            if analysis_points is None:
                analysis_size = (roi_w, roi_h)
            else:
                _, _, analysis_width, analysis_height = cv2.boundingRect(analysis_points)
                analysis_size = (max(1, analysis_width), max(1, analysis_height))

            centroid = np.mean(local_pts.astype(np.float32), axis=0)
            core_points = np.rint(
                centroid + (local_pts.astype(np.float32) - centroid) * self.core_scale
            ).astype(np.int32)
            core_mask = np.zeros_like(local_mask)
            cv2.fillPoly(core_mask, [core_points], 255)
            core_mask = cv2.bitwise_and(core_mask, analysis_mask)
            if cv2.countNonZero(core_mask) < 5:
                core_mask = analysis_mask.copy()
            core_area = cv2.countNonZero(core_mask)

            self._rois.append(
                _PrecomputedROI(
                    slot_id=slot["id"],
                    polygon_pts=pts,
                    bbox=(x_min, y_min, x_max, y_max),
                    mask=local_mask,
                    area=area,
                    analysis_mask=analysis_mask,
                    analysis_boundary=analysis_boundary,
                    analysis_area=analysis_area,
                    analysis_size=analysis_size,
                    core_mask=core_mask,
                    core_area=core_area,
                    center=center,
                )
            )

        self._smoother = TemporalSmoother(len(self._rois), self.smoothing_frames)
        self._initialized = True

    def _filter_roi_threshold(
        self,
        roi_threshold: np.ndarray,
        roi: _PrecomputedROI,
    ) -> _ROIEvidence:
        """Remove thin boundary-connected components while preserving central blobs."""
        raw_masked = cv2.bitwise_and(roi_threshold, roi_threshold, mask=roi.mask)
        raw_ratio = cv2.countNonZero(raw_masked) / max(1, roi.area)

        filtered = cv2.bitwise_and(
            roi_threshold, roi_threshold, mask=roi.analysis_mask
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        analysis_width, analysis_height = roi.analysis_size
        analysis_min_dimension = max(1, min(analysis_width, analysis_height))
        boundary_labels = labels[roi.analysis_boundary > 0]
        core_labels = labels[roi.core_mask > 0]

        for label in range(1, count):
            component_area = int(stats[label, cv2.CC_STAT_AREA])
            component_width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
            component_height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
            long_axis = max(component_width, component_height)
            span_ratio = max(
                component_width / analysis_width,
                component_height / analysis_height,
            )
            thickness_ratio = (component_area / long_axis) / analysis_min_dimension
            touches_boundary = bool(np.any(boundary_labels == label))
            component_core_ratio = (
                float(np.count_nonzero(core_labels == label)) / max(1, roi.core_area)
            )
            substantial_core = component_core_ratio >= self.core_component_threshold
            vehicle_like_core = (
                substantial_core
                and thickness_ratio >= self.line_max_thickness_ratio * 0.90
            )
            is_border_line = (
                touches_boundary
                and span_ratio >= self.line_min_span_ratio
                and thickness_ratio <= self.line_max_thickness_ratio
                and not vehicle_like_core
            )
            if is_border_line:
                filtered[labels == label] = 0

        filtered_count = cv2.countNonZero(filtered)
        core_pixels = cv2.bitwise_and(filtered, filtered, mask=roi.core_mask)
        core_count = cv2.countNonZero(core_pixels)
        core_component_ratio = 0.0
        core_component_count = 0
        if core_count:
            core_components, _, core_stats, _ = cv2.connectedComponentsWithStats(
                core_pixels, connectivity=8
            )
            if core_components > 1:
                component_areas = core_stats[1:, cv2.CC_STAT_AREA]
                core_component_ratio = float(np.max(component_areas)) / max(1, roi.core_area)
                minimum_component_area = max(3, int(round(roi.core_area * 0.005)))
                core_component_count = int(np.count_nonzero(component_areas >= minimum_component_area))

        return _ROIEvidence(
            raw_ratio=raw_ratio,
            filtered_ratio=filtered_count / max(1, roi.analysis_area),
            core_ratio=core_count / max(1, roi.core_area),
            core_component_ratio=core_component_ratio,
            core_component_count=core_component_count,
            filtered_mask=filtered,
        )

    def _has_core_rescue(self, evidence: _ROIEvidence) -> bool:
        central_blob = (
            evidence.core_ratio >= self.core_ratio_threshold
            and evidence.core_component_ratio >= self.core_component_threshold
        )
        textured_dense_blob = (
            evidence.raw_ratio >= min(1.0, self.ratio_thr * 1.5)
            and evidence.core_ratio >= self.core_component_threshold * 0.25
            and evidence.core_component_count >= 2
        )
        return central_blob or textured_dense_blob

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

        # VR5: Generate delta arrays dynamically based on variant counts
        # For 3 variants: [-0.2, 0.0, 0.2] gamma, [-0.5, 0.0, 0.5] CLAHE
        # For 5 variants: [-0.2, -0.1, 0.0, 0.1, 0.2] gamma, [-0.5, -0.2, 0.0, 0.2, 0.5] CLAHE
        delta_gamma = np.linspace(-0.2, 0.2, self.gamma_variants).tolist()
        delta_clahe = np.linspace(-0.5, 0.5, self.clahe_variants).tolist()
        total_combinations = len(delta_clahe) * len(delta_gamma)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        kernel = np.ones((3, 3), np.uint8)
        empty_votes = [0] * len(self._rois)
        base_evidence: List[Optional[_ROIEvidence]] = [None] * len(self._rois)
        vote_masks: List[Optional[np.ndarray]] = [None] * len(self._rois)

        base_clahe_filter = cv2.createCLAHE(
            clipLimit=max(0.1, base_clahe),
            tileGridSize=(clahe_grid, clahe_grid),
        )
        base_light = base_clahe_filter.apply(l_channel)
        base_light = cv2.LUT(base_light, self._get_gamma_lut(max(0.1, base_gamma)))
        base_threshold = cv2.adaptiveThreshold(
            cv2.GaussianBlur(base_light, (3, 3), 1),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            16,
        )
        base_threshold = cv2.dilate(
            cv2.medianBlur(base_threshold, 5), kernel, iterations=1
        )
        for i, roi in enumerate(self._rois):
            if roi is None or roi.analysis_area == 0:
                continue
            x1, y1, x2, y2 = roi.bbox
            roi_threshold = base_threshold[y1:y2, x1:x2]
            evidence = self._filter_roi_threshold(roi_threshold, roi)
            base_evidence[i] = evidence
            base_analysis = cv2.bitwise_and(
                roi_threshold, roi_threshold, mask=roi.analysis_mask
            )
            removed_pixels = cv2.subtract(base_analysis, evidence.filtered_mask)
            vote_masks[i] = cv2.bitwise_and(
                roi.analysis_mask, cv2.bitwise_not(removed_pixels)
            )

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

                for i, roi in enumerate(self._rois):
                    if roi is None or roi.analysis_area == 0:
                        continue
                    x1, y1, x2, y2 = roi.bbox
                    roi_thresh = dilated[y1:y2, x1:x2]
                    filtered = cv2.bitwise_and(
                        roi_thresh, roi_thresh, mask=vote_masks[i]
                    )
                    if cv2.countNonZero(filtered) / roi.analysis_area < ratio_thr:
                        empty_votes[i] += 1

        # ── Pass 1: Threshold + Center Cluster ──
        required_votes = total_combinations // 2
        is_free_list = [True] * len(self._rois)

        for i, roi in enumerate(self._rois):
            if roi is None:
                continue
            is_free = empty_votes[i] >= required_votes

            evidence = base_evidence[i]
            if (
                is_free
                and evidence is not None
                and self._has_core_rescue(evidence)
            ):
                is_free = False

            is_free_list[i] = is_free

        # ── Pass 2: Edge Recheck ──
        if self.use_edge_recheck:
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

            occupied = not is_free
            ev_score = 1.0 - (empty_votes[i] / max(1, total_combinations))
            if occupied:
                ev_score = max(0.65, ev_score)
            else:
                ev_score = min(0.35, ev_score)

            results.append(
                SlotResult(
                    slot_id=roi.slot_id,
                    occupied=occupied,
                    polygon=roi.polygon_pts,
                    center=roi.center,
                    evidence=float(ev_score),
                )
            )

        return results

    def get_roi_polygon(self, slot_id: str) -> Optional[np.ndarray]:
        """Trả polygon đã scale cho 1 slot ID."""
        for roi in self._rois:
            if roi is not None and roi.slot_id == slot_id:
                return roi.polygon_pts
        return None

    def build_debug_images(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Render raw and border-filtered black/white evidence without changing state."""
        if not self._initialized:
            self._compute_rois(frame.shape)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=max(0.1, self.base_clahe),
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )
        light = clahe.apply(lab[:, :, 0])
        light = cv2.LUT(light, self._get_gamma_lut(max(0.1, self.base_gamma)))
        threshold = cv2.adaptiveThreshold(
            cv2.GaussianBlur(light, (3, 3), 1), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16,
        )
        threshold = cv2.dilate(
            cv2.medianBlur(threshold, 5), np.ones((3, 3), np.uint8), iterations=1,
        )

        filtered_threshold = np.zeros_like(threshold)
        raw_view = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
        debug_rows = []
        for roi in self._rois:
            if roi is None or roi.analysis_area <= 0:
                continue
            x1, y1, x2, y2 = roi.bbox
            roi_threshold = threshold[y1:y2, x1:x2]
            evidence = self._filter_roi_threshold(roi_threshold, roi)
            target = filtered_threshold[y1:y2, x1:x2]
            cv2.bitwise_or(target, evidence.filtered_mask, dst=target)
            debug_rows.append((roi, evidence))

        filtered_view = cv2.cvtColor(filtered_threshold, cv2.COLOR_GRAY2BGR)
        for roi, evidence in debug_rows:
            occupied = evidence.filtered_ratio >= self.ratio_thr or self._has_core_rescue(evidence)
            raw_color = (0, 0, 255) if evidence.raw_ratio >= self.ratio_thr else (0, 255, 0)
            filtered_color = (0, 0, 255) if occupied else (0, 255, 0)
            cv2.polylines(raw_view, [roi.polygon_pts], True, raw_color, 2)
            cv2.polylines(filtered_view, [roi.polygon_pts], True, filtered_color, 2)

            x1, y1, _, _ = roi.bbox
            for mask, color in (
                (roi.analysis_mask, (255, 255, 0)),
                (roi.core_mask, (255, 0, 255)),
            ):
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour = contour + np.asarray([[[x1, y1]]], dtype=np.int32)
                    cv2.polylines(filtered_view, [contour], True, color, 1)

            cv2.putText(
                raw_view, roi.slot_id,
                (roi.center[0] - 10, roi.center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.25, (0, 255, 255), 1,
            )
            cv2.putText(
                raw_view, f"R{evidence.raw_ratio:.2f}",
                (roi.center[0] - 10, roi.center[1] + 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.23, (0, 255, 255), 1,
            )
            for text, y_offset in (
                (roi.slot_id, -9),
                (f"F{evidence.filtered_ratio:.2f}", 1),
                (f"C{evidence.core_ratio:.2f}", 11),
            ):
                cv2.putText(
                    filtered_view, text,
                    (roi.center[0] - 10, roi.center[1] + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.23, (0, 255, 255), 1,
                )

        cv2.putText(
            raw_view, f"RAW B/W | full ROI threshold={self.ratio_thr:.2f}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255), 2,
        )
        cv2.putText(
            filtered_view, "FILTERED B/W | cyan=analysis magenta=core",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 200, 255), 2,
        )
        return raw_view, filtered_view

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
