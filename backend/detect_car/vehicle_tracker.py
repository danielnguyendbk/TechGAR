"""
vehicle_tracker.py — Vehicle Tracker v2
Port từ Landzs/Tracking_Multiple_Objects_In_Surveillance_Cameras (MATLAB → Python)

Nâng cấp so với v1:
  1. Particle Filter (100 particles/track) — dự đoán vị trí khi xe bị che
  2. Hungarian Assignment (scipy) — matching tối ưu thay vì greedy nearest-neighbor
  3. Age/Visibility filtering — chỉ confirm track sau minVisibleCount frames
  4. Bounding box size + aspect ratio filter — loại nhiễu lá cây, bóng đổ
  5. Track lifecycle: age, totalVisibleCount, consecutiveInvisibleCount
  6. Re-ID: khi xe mất rồi xuất hiện lại gần vị trí cũ → giữ nguyên ID cũ
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class TrackStatus(Enum):
    TENTATIVE = "tentative"      # Track mới, chưa đủ tuổi để confirm
    CONFIRMED = "confirmed"      # Track đã xác nhận là xe thật
    LOST = "lost"                # Track mất dấu, đang dùng prediction


@dataclass
class TrackedVehicle:
    """Thông tin 1 xe đang được track — tương đương struct track trong MATLAB."""
    track_id: int
    cx: int                          # tâm x hiện tại (pixel)
    cy: int                          # tâm y hiện tại
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float

    # Particle Filter state (tương đương tracks(i).particles trong MATLAB)
    particles: np.ndarray = None     # shape (N_PARTICLES, 2) — mỗi row là (x, y)

    # Track lifecycle (từ repo tham khảo)
    age: int = 1
    total_visible_count: int = 1
    consecutive_invisible_count: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE

    # Lịch sử
    history: List[Tuple[int, int]] = field(default_factory=list)

    # Direction events (sẽ được DirectionDetector ghi vào)
    direction_events: List[dict] = field(default_factory=list)

    # Frame info
    entered_frame: int = 0
    exited_frame: int = 0

    @property
    def x(self):
        return self.bbox[0]

    @property
    def y(self):
        return self.bbox[1]

    @property
    def w(self):
        return self.bbox[2]

    @property
    def h(self):
        return self.bbox[3]

    @property
    def visibility(self):
        return self.total_visible_count / max(self.age, 1)


class VehicleTracker:
    """
    Vehicle Tracker v2 — Particle Filter + Hungarian Assignment.

    Pipeline mỗi frame (tương đương Tracking_Cars.m):
      1. detectObjects()              → foreground mask → contours → detections
      2. predictNewLocationsOfTracks() → Particle Filter diffusion + resample
      3. detectionToTrackAssignment()  → Hungarian matching
      4. updateAssignedTracks()        → pfCorrect + cập nhật bbox
      5. updateUnassignedTracks()      → tăng invisible count
      6. deleteLostTracks()            → xóa track mất dấu quá lâu
      7. createNewTracks()             → tạo track mới cho detection không match
    """

    # ── Tham số mặc định ──
    N_PARTICLES = 100              # Số particles mỗi track (từ MATLAB: ones(100,2))
    DIFFUSION_STD = 4.0            # Độ lệch chuẩn Gaussian cho diffusion (từ MATLAB: randn*4)
    COST_OF_NON_ASSIGNMENT = 50.0  # Ngưỡng chi phí để không gán (từ MATLAB: 20, tăng lên cho pixel lớn)

    def __init__(
        self,
        # Background Subtraction
        history: int = 500,
        var_threshold: int = 50,
        # Tiền xử lý ảnh (Gamma + CLAHE)
        gamma: float = 1.0,            # Gamma correction (< 1 sáng hơn, > 1 tối hơn)
        clahe_clip: float = 2.0,       # CLAHE clip limit (0 = tắt CLAHE)
        clahe_grid: int = 8,           # CLAHE grid size
        # Lọc nhiễu
        min_area: int = 800,
        min_width: int = 25,
        min_height: int = 20,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 5.0,
        # Gộp box gần nhau
        merge_distance: float = 60.0,  # Khoảng cách tâm tối đa để gộp 2 box
        merge_size_ratio: float = 0.5, # Box nhỏ so với median → ứng viên gộp
        # Track lifecycle (từ deleteLostTracks.m)
        age_threshold: int = 8,
        min_visibility: float = 0.4,
        min_visible_count: int = 8,
        invisible_for_too_long: int = 30,
        # Matching
        max_distance: float = 120.0,
        # Re-ID
        reid_distance: float = 100.0,
        reid_max_frames: int = 60,
        # History
        history_len: int = 50,
    ):
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.merge_distance = merge_distance
        self.merge_size_ratio = merge_size_ratio
        self.age_threshold = age_threshold
        self.min_visibility = min_visibility
        self.min_visible_count = min_visible_count
        self.invisible_for_too_long = invisible_for_too_long
        self.max_distance = max_distance
        self.reid_distance = reid_distance
        self.reid_max_frames = reid_max_frames
        self.history_len = history_len

        # Background Subtractor (MOG2)
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )

        # CLAHE object (tạo 1 lần, dùng lại)
        if self.clahe_clip > 0:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_grid, self.clahe_grid),
            )
        else:
            self._clahe = None

        # Gamma LUT (tạo 1 lần)
        # gamma < 1 → sáng hơn (pixel^0.7 tăng giá trị tối)
        # gamma > 1 → tối hơn (pixel^1.5 giảm giá trị sáng)
        if abs(self.gamma - 1.0) > 0.01:
            self._gamma_lut = np.array(
                [((i / 255.0) ** self.gamma) * 255 for i in range(256)]
            ).astype(np.uint8)
        else:
            self._gamma_lut = None

        # State
        self._tracks: Dict[int, TrackedVehicle] = {}
        self._exited_tracks: Dict[int, TrackedVehicle] = {}
        self._recently_exited: Dict[int, TrackedVehicle] = {}
        self._next_id: int = 1
        self._recycled_ids: List[int] = []  # Pool ID tái sử dụng
        self._frame_idx: int = 0

        # Thống kê kích thước box (để phát hiện box bất thường)
        self._area_history: List[float] = []  # Lưu area của confirmed tracks
        self._median_area: float = 0.0        # Median area hiện tại

    # ═══════════════════════════════════════════════
    # Bước 0: Tiền xử lý ảnh (Gamma + CLAHE)
    # ═══════════════════════════════════════════════
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Áp dụng Gamma correction + CLAHE trước khi đưa vào BGSub.
        Giúp cải thiện contrast trong điều kiện ánh sáng kém/không đều.
        """
        out = frame

        # Gamma correction
        if self._gamma_lut is not None:
            out = cv2.LUT(out, self._gamma_lut)

        # CLAHE trên kênh L (LAB color space) — cải thiện contrast cục bộ
        if self._clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            lab = cv2.merge([l_channel, a_channel, b_channel])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return out

    # ═══════════════════════════════════════════════
    # Bước 1: detectObjects (tương đương detectObjects.m)
    # ═══════════════════════════════════════════════
    def _create_fg_mask(self, frame: np.ndarray) -> np.ndarray:
        """Tiền xử lý → BGSub → morphology → foreground mask."""
        # Tiền xử lý (Gamma + CLAHE)
        processed = self._preprocess_frame(frame)

        fg_mask = self.bg_sub.apply(processed)

        # Loại shadow (pixel=127 trong MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphology
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # imfill holes
        fg_filled = fg_mask.copy()
        h, w = fg_filled.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(fg_filled, flood_mask, (0, 0), 255)
        fg_filled_inv = cv2.bitwise_not(fg_filled)
        fg_mask = fg_mask | fg_filled_inv

        return fg_mask

    def _detect_vehicles(self, fg_mask: np.ndarray) -> List[dict]:
        """
        Tìm contours → lọc theo area, kích thước tối thiểu, aspect ratio.
        Tương đương blobAnalyser trong MATLAB với MinimumBlobArea=600.
        """
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Lọc kích thước tối thiểu
            if w < self.min_width or h < self.min_height:
                continue

            # Lọc aspect ratio (loại vật thể dài mỏng bất thường)
            aspect = w / max(h, 1)
            if aspect < self.min_aspect_ratio or aspect > self.max_aspect_ratio:
                continue

            cx = x + w // 2
            cy = y + h // 2
            detections.append({
                "cx": cx, "cy": cy,
                "x": x, "y": y, "w": w, "h": h,
                "area": area,
            })

        return detections

    # ═══════════════════════════════════════════════
    # Bước 1b: Gộp box gần nhau + kiểm tra kích thước
    # ═══════════════════════════════════════════════
    def _merge_close_detections(self, detections: List[dict]) -> List[dict]:
        """
        Gộp 2 box gần nhau tạo thành 1 đường thẳng → xác định đó là 1 xe.
        Thuật toán:
          1. Tìm tất cả cặp box có tâm cách nhau < merge_distance
          2. Kiểm tra 2 box có roughly collinear (tạo thành 1 đường)
          3. Gộp thành 1 bounding box lớn
          4. Kiểm tra box sau gộp có kích thước hợp lệ
        """
        if len(detections) < 2:
            return detections

        n = len(detections)
        merged_flags = [False] * n
        result = []

        # Tính median area nếu có dữ liệu
        median_area = self._median_area if self._median_area > 0 else None

        # Tìm cặp box gần nhau
        for i in range(n):
            if merged_flags[i]:
                continue

            best_j = -1
            best_dist = self.merge_distance

            # Nếu box i đã đủ lớn so với median → không cần gộp
            if median_area and detections[i]["area"] >= median_area * self.merge_size_ratio:
                result.append(detections[i])
                continue

            for j in range(i + 1, n):
                if merged_flags[j]:
                    continue

                di, dj = detections[i], detections[j]
                dist = np.sqrt((di["cx"] - dj["cx"]) ** 2 + (di["cy"] - dj["cy"]) ** 2)

                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                di, dj = detections[i], detections[best_j]

                # Gộp bounding box
                x1 = min(di["x"], dj["x"])
                y1 = min(di["y"], dj["y"])
                x2 = max(di["x"] + di["w"], dj["x"] + dj["w"])
                y2 = max(di["y"] + di["h"], dj["y"] + dj["h"])
                mw, mh = x2 - x1, y2 - y1
                merged_area = mw * mh

                # Kiểm tra box sau gộp có hợp lệ không
                merged_ok = True
                if median_area:
                    # Sau gộp phải >= 50% median (không quá nhỏ)
                    # và <= 300% median (không quá lớn)
                    if merged_area < median_area * 0.3 or merged_area > median_area * 3.0:
                        merged_ok = False

                if merged_ok:
                    merged_flags[best_j] = True
                    merged_flags[i] = True
                    result.append({
                        "cx": x1 + mw // 2, "cy": y1 + mh // 2,
                        "x": x1, "y": y1, "w": mw, "h": mh,
                        "area": merged_area,
                    })
                else:
                    # Gộp không hợp lệ → giữ nguyên box i
                    result.append(detections[i])
            else:
                result.append(detections[i])

        # Thêm box chưa xét (chưa bị merge)
        for j in range(n):
            if not merged_flags[j] and detections[j] not in result:
                result.append(detections[j])

        return result

    def _validate_detections(self, detections: List[dict]) -> List[dict]:
        """
        Kiểm tra box bất thường so với kích thước đa số.
        - Box quá nhỏ so với median → loại
        - Các frame đầu chưa có median → tạm chấp nhận tất cả
        """
        if self._median_area <= 0:
            # Chưa có dữ liệu đa số → chấp nhận hết, thu thập dữ liệu
            return detections

        valid = []
        for det in detections:
            ratio = det["area"] / self._median_area
            if ratio >= self.merge_size_ratio:
                valid.append(det)
            # else: box quá nhỏ, đã thử merge ở bước trước mà vẫn nhỏ → bỏ

        return valid

    def _update_area_stats(self):
        """Cập nhật thống kê kích thước box từ confirmed tracks."""
        for tid, track in self._tracks.items():
            if track.status == TrackStatus.CONFIRMED:
                self._area_history.append(track.area)

        # Giữ tối đa 200 mẫu gần nhất
        if len(self._area_history) > 200:
            self._area_history = self._area_history[-200:]

        if len(self._area_history) >= 5:
            self._median_area = float(np.median(self._area_history))

    # ═══════════════════════════════════════════════
    # Bước 2: predictNewLocationsOfTracks
    #         (tương đương predictNewLocationsOfTracks.m)
    # ═══════════════════════════════════════════════
    def _pf_diffusion(self, particles: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        """
        Particle Filter Diffusion — thêm nhiễu Gaussian.
        Tương đương pfDiffusion.m: Particles = Particles + randn(N,2)*4
        """
        h, w = fg_mask.shape[:2]
        particles = particles + np.random.randn(*particles.shape) * self.DIFFUSION_STD

        # Clamp trong bounds (từ MATLAB)
        particles[:, 0] = np.clip(particles[:, 0], 1, w - 1)
        particles[:, 1] = np.clip(particles[:, 1], 1, h - 1)
        return particles

    def _pf_resample(self, particles: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        """
        Particle Filter Resample — systematic resampling dựa trên foreground mask.
        Tương đương pfResample.m.
        """
        n = particles.shape[0]
        weights = np.zeros(n)

        for i in range(n):
            px = int(round(particles[i, 0]))
            py = int(round(particles[i, 1]))
            px = np.clip(px, 0, fg_mask.shape[1] - 1)
            py = np.clip(py, 0, fg_mask.shape[0] - 1)
            weights[i] = fg_mask[py, px]

        total_w = np.sum(weights)
        if total_w == 0:
            return particles  # Không resample nếu không có foreground

        weights = weights / total_w
        cdf = np.cumsum(weights)

        # Systematic resampling (từ MATLAB)
        r0 = np.random.rand() / n
        new_particles = np.zeros_like(particles)
        for m in range(n):
            idx = np.searchsorted(cdf, r0)
            idx = min(idx, n - 1)
            new_particles[m] = particles[idx]
            r0 += 1.0 / n

        return new_particles

    def _pf_correct(self, particles: np.ndarray, centroid: Tuple[int, int]) -> np.ndarray:
        """
        Particle Filter Correct — reset tất cả particles về centroid mới.
        Tương đương pfCorrect.m: Particles = repmat(centroid, N, 1)
        """
        n = particles.shape[0]
        return np.tile(np.array([centroid[0], centroid[1]], dtype=np.float64), (n, 1))

    def _predict_new_locations(self, fg_mask: np.ndarray):
        """
        Dự đoán vị trí mới cho tất cả tracks dùng Particle Filter.
        Tương đương predictNewLocationsOfTracks.m.
        """
        for tid, track in self._tracks.items():
            if track.particles is None:
                continue

            # Diffusion + Resample
            track.particles = self._pf_diffusion(track.particles, fg_mask)
            track.particles = self._pf_resample(track.particles, fg_mask)

            # Predicted centroid = mean of particles
            predicted_cx = int(np.mean(track.particles[:, 0]))
            predicted_cy = int(np.mean(track.particles[:, 1]))

            # Shift bbox center to predicted location
            bw, bh = track.w, track.h
            new_x = predicted_cx - bw // 2
            new_y = predicted_cy - bh // 2
            track.bbox = (new_x, new_y, bw, bh)
            track.cx = predicted_cx
            track.cy = predicted_cy

    # ═══════════════════════════════════════════════
    # Bước 3: detectionToTrackAssignment
    #         (tương đương detectionToTrackAssignment.m)
    # ═══════════════════════════════════════════════
    def _compute_cost_matrix(self, detections: List[dict]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Tính cost matrix giữa tracks và detections.
        Cost = Euclidean distance giữa predicted centroid và detection centroid.
        """
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_detections = len(detections)

        if n_tracks == 0 or n_detections == 0:
            return np.empty((0, 0)), track_ids, list(range(n_detections))

        cost = np.zeros((n_tracks, n_detections))
        for i, tid in enumerate(track_ids):
            track = self._tracks[tid]
            pred_centroid = np.mean(track.particles, axis=0) if track.particles is not None else np.array([track.cx, track.cy])
            for j, det in enumerate(detections):
                dist = np.sqrt((pred_centroid[0] - det["cx"]) ** 2 + (pred_centroid[1] - det["cy"]) ** 2)
                cost[i, j] = dist

        return cost, track_ids, list(range(n_detections))

    def _assign_detections_to_tracks(self, detections: List[dict]):
        """
        Hungarian Assignment — tìm phép gán tối ưu (cost tổng nhỏ nhất).
        Tương đương detectionToTrackAssignment.m + assignDetectionsToTracks.
        """
        cost, track_ids, det_indices = self._compute_cost_matrix(detections)

        if cost.size == 0:
            return [], list(self._tracks.keys()), list(range(len(detections)))

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        unassigned_tracks = set(range(len(track_ids)))
        unassigned_detections = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.COST_OF_NON_ASSIGNMENT:
                assignments.append((track_ids[r], c))
                unassigned_tracks.discard(r)
                unassigned_detections.discard(c)

        unassigned_track_ids = [track_ids[i] for i in unassigned_tracks]
        unassigned_det_ids = list(unassigned_detections)

        return assignments, unassigned_track_ids, unassigned_det_ids

    # ═══════════════════════════════════════════════
    # Bước 4: updateAssignedTracks
    #         (tương đương updateAssignedTracks.m)
    # ═══════════════════════════════════════════════
    def _update_assigned_tracks(self, assignments: List[Tuple[int, int]], detections: List[dict]):
        """Cập nhật tracks đã match với detection."""
        for tid, det_idx in assignments:
            det = detections[det_idx]
            track = self._tracks[tid]

            # pfCorrect — reset particles về detection centroid
            track.particles = self._pf_correct(track.particles, (det["cx"], det["cy"]))

            # Cập nhật bbox
            track.bbox = (det["x"], det["y"], det["w"], det["h"])
            track.cx = det["cx"]
            track.cy = det["cy"]
            track.area = det["area"]

            # Cập nhật lifecycle (từ MATLAB)
            track.age += 1
            track.total_visible_count += 1
            track.consecutive_invisible_count = 0

            # Cập nhật status
            if track.status == TrackStatus.TENTATIVE and track.total_visible_count >= self.min_visible_count:
                track.status = TrackStatus.CONFIRMED

            # Lưu history
            track.history.append((det["cx"], det["cy"]))
            if len(track.history) > self.history_len:
                track.history = track.history[-self.history_len:]

    # ═══════════════════════════════════════════════
    # Bước 5: updateUnassignedTracks
    #         (tương đương updateUnassignedTracks.m)
    # ═══════════════════════════════════════════════
    def _update_unassigned_tracks(self, unassigned_track_ids: List[int]):
        """Tăng invisible count cho tracks không match."""
        for tid in unassigned_track_ids:
            if tid in self._tracks:
                track = self._tracks[tid]
                track.age += 1
                track.consecutive_invisible_count += 1
                if track.status == TrackStatus.CONFIRMED:
                    track.status = TrackStatus.LOST

    # ═══════════════════════════════════════════════
    # Bước 6: deleteLostTracks
    #         (tương đương deleteLostTracks.m)
    # ═══════════════════════════════════════════════
    def _delete_lost_tracks(self):
        """
        Xóa tracks mất dấu quá lâu hoặc visibility quá thấp.
        Tương đương deleteLostTracks.m logic:
          lost = (age < ageThreshold & visibility < 0.4) |
                 consecutiveInvisibleCount >= invisibleForTooLong
        """
        remove_ids = []
        for tid, track in self._tracks.items():
            is_young_and_invisible = (
                track.age < self.age_threshold and
                track.visibility < self.min_visibility
            )
            is_invisible_too_long = (
                track.consecutive_invisible_count >= self.invisible_for_too_long
            )

            if is_young_and_invisible or is_invisible_too_long:
                remove_ids.append(tid)

        for tid in remove_ids:
            track = self._tracks.pop(tid)
            # Chỉ lưu nếu track đã từng confirmed
            if track.total_visible_count >= self.min_visible_count:
                track.exited_frame = self._frame_idx
                track.status = TrackStatus.LOST
                # Đưa vào buffer Re-ID (chờ xe quay lại)
                self._recently_exited[tid] = track

    # ═══════════════════════════════════════════════
    # Bước 7: createNewTracks + Re-ID
    #         (tương đương createNewTracks.m + Re-identification)
    # ═══════════════════════════════════════════════
    def _try_reid(self, det: dict) -> Optional[int]:
        """
        Re-ID: kiểm tra detection mới có trùng với xe cũ vừa mất không.
        So sánh centroid detection với last_known_position của xe trong buffer.
        Trả về track_id cũ nếu match, None nếu không.
        """
        best_tid = None
        best_dist = self.reid_distance

        for tid, old_track in self._recently_exited.items():
            # Kiểm tra xe mất bao lâu rồi
            frames_since_exit = self._frame_idx - old_track.exited_frame
            if frames_since_exit > self.reid_max_frames:
                continue

            dist = np.sqrt(
                (old_track.cx - det["cx"]) ** 2 +
                (old_track.cy - det["cy"]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        return best_tid

    def _create_new_tracks(self, detections: List[dict], unassigned_det_ids: List[int]):
        """
        Tạo track mới cho detections chưa match.
        TRƯỚC KHI tạo ID mới → kiểm tra Re-ID với xe vừa mất.
        """
        for det_idx in unassigned_det_ids:
            det = detections[det_idx]
            centroid = np.array([det["cx"], det["cy"]], dtype=np.float64)
            particles = np.tile(centroid, (self.N_PARTICLES, 1))

            # ── Re-ID: thử nhận lại xe cũ ──
            reid_tid = self._try_reid(det)

            if reid_tid is not None:
                # Khôi phục track cũ với ID cũ
                old_track = self._recently_exited.pop(reid_tid)
                old_track.cx = det["cx"]
                old_track.cy = det["cy"]
                old_track.bbox = (det["x"], det["y"], det["w"], det["h"])
                old_track.area = det["area"]
                old_track.particles = particles
                old_track.consecutive_invisible_count = 0
                old_track.status = TrackStatus.CONFIRMED
                old_track.total_visible_count += 1
                old_track.age += 1
                old_track.history.append((det["cx"], det["cy"]))
                if len(old_track.history) > self.history_len:
                    old_track.history = old_track.history[-self.history_len:]

                self._tracks[reid_tid] = old_track
                print(f"  🔄 Re-ID: Xe #{reid_tid} quay lại tại ({det['cx']}, {det['cy']})")
            else:
                # Tạo track hoàn toàn mới - dùng ID tái sử dụng nếu có
                if self._recycled_ids:
                    new_id = self._recycled_ids.pop(0)
                else:
                    new_id = self._next_id
                    self._next_id += 1

                new_track = TrackedVehicle(
                    track_id=new_id,
                    cx=det["cx"],
                    cy=det["cy"],
                    bbox=(det["x"], det["y"], det["w"], det["h"]),
                    area=det["area"],
                    particles=particles,
                    age=1,
                    total_visible_count=1,
                    consecutive_invisible_count=0,
                    status=TrackStatus.TENTATIVE,
                    history=[(det["cx"], det["cy"])],
                    entered_frame=self._frame_idx,
                )
                self._tracks[new_id] = new_track

    # ═══════════════════════════════════════════════
    # Cleanup Re-ID buffer
    # ═══════════════════════════════════════════════
    def _cleanup_reid_buffer(self):
        """
        Chuyển xe quá hạn từ _recently_exited → _exited_tracks (vĩnh viễn).
        """
        expired = []
        for tid, track in self._recently_exited.items():
            frames_since = self._frame_idx - track.exited_frame
            if frames_since > self.reid_max_frames:
                expired.append(tid)

        for tid in expired:
            self._exited_tracks[tid] = self._recently_exited.pop(tid)

    # ═══════════════════════════════════════════════
    # Bước 10: Kiểm tra + hủy track bất thường
    # ═══════════════════════════════════════════════
    def _invalidate_small_tracks(self):
        """
        Kiểm tra confirmed tracks có area quá nhỏ so với median.
        Nếu có → hủy track + trả ID về pool tái sử dụng.
        Chỉ kiểm tra khi đã có đủ dữ liệu thống kê.
        """
        if self._median_area <= 0:
            return

        invalidate_ids = []
        for tid, track in self._tracks.items():
            # Chỉ kiểm tra tentative tracks (chưa confirm)
            if track.status != TrackStatus.TENTATIVE:
                continue

            ratio = track.area / self._median_area
            if ratio < self.merge_size_ratio:
                invalidate_ids.append(tid)

        for tid in invalidate_ids:
            self._tracks.pop(tid)
            # Trả lại ID vào pool tái sử dụng
            self._recycled_ids.append(tid)
            self._recycled_ids.sort()  # Giữ thứ tự tăng dần

    # ═══════════════════════════════════════════════
    # API chính: process_frame
    # ═══════════════════════════════════════════════
    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[int, TrackedVehicle], np.ndarray]:
        """
        Pipeline hoàn chỉnh cho 1 frame — tương đương vòng while trong Tracking_Cars.m:
          detectObjects → predict → assign → update → delete → create
        """
        self._frame_idx += 1

        # 1. Detect
        fg_mask = self._create_fg_mask(frame)
        detections = self._detect_vehicles(fg_mask)

        # 1b. Gộp box gần nhau + validate kích thước
        detections = self._merge_close_detections(detections)
        detections = self._validate_detections(detections)

        # 2. Predict (Particle Filter)
        self._predict_new_locations(fg_mask)

        # 3. Assign (Hungarian)
        assignments, unassigned_tracks, unassigned_dets = \
            self._assign_detections_to_tracks(detections)

        # 4-5. Update
        self._update_assigned_tracks(assignments, detections)
        self._update_unassigned_tracks(unassigned_tracks)

        # 6. Delete lost → chuyển vào buffer Re-ID
        self._delete_lost_tracks()

        # 7. Create new (có Re-ID)
        self._create_new_tracks(detections, unassigned_dets)

        # 8. Cleanup buffer Re-ID
        self._cleanup_reid_buffer()

        # 9. Cập nhật thống kê kích thước box
        self._update_area_stats()

        # 10. Kiểm tra + hủy tracks bất thường
        self._invalidate_small_tracks()

        return self._tracks, fg_mask

    # ═══════════════════════════════════════════════
    # Tiện ích: Vẽ lên frame
    # ═══════════════════════════════════════════════
    def draw_tracks(self, frame: np.ndarray, tracks: Optional[Dict[int, TrackedVehicle]] = None) -> np.ndarray:
        """
        Vẽ bounding box + ID + trail lên frame.
        Giống displayTrackingResults.m — chỉ hiển thị confirmed tracks.
        """
        out = frame.copy()
        if tracks is None:
            tracks = self._tracks

        confirmed_count = 0
        tentative_count = 0

        for tid, t in tracks.items():
            if t.status == TrackStatus.TENTATIVE:
                tentative_count += 1
                # Vẽ nhạt cho tentative
                cv2.rectangle(out, (t.x, t.y), (t.x + t.w, t.y + t.h), (100, 100, 100), 1)
                continue

            confirmed_count += 1

            # Màu theo status
            if t.status == TrackStatus.CONFIRMED:
                color = (0, 255, 0)
                label = f"#{tid}"
            else:  # LOST
                color = (0, 165, 255)
                label = f"#{tid} predicted"

            # Bounding box
            cv2.rectangle(out, (t.x, t.y), (t.x + t.w, t.y + t.h), color, 2)

            # Label
            cv2.putText(out, label, (t.x, t.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Tâm
            cv2.circle(out, (t.cx, t.cy), 4, (0, 0, 255), -1)

            # Trail
            if len(t.history) > 1:
                for j in range(1, len(t.history)):
                    alpha = j / len(t.history)
                    thickness = max(1, int(alpha * 3))
                    cv2.line(out, t.history[j - 1], t.history[j], (255, 200, 0), thickness)

            # Direction events
            for ev in t.direction_events:
                ev_text = f"{ev['decision']}"
                cv2.putText(out, ev_text, (t.x, t.y + t.h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Info overlay
        cv2.putText(out, f"Confirmed: {confirmed_count} | Tentative: {tentative_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return out

    # ═══════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════
    @property
    def confirmed_tracks(self) -> Dict[int, TrackedVehicle]:
        """Chỉ tracks đã confirmed (xe thật sự)."""
        return {
            tid: t for tid, t in self._tracks.items()
            if t.status in (TrackStatus.CONFIRMED, TrackStatus.LOST)
        }

    @property
    def active_tracks(self) -> Dict[int, TrackedVehicle]:
        """Tracks confirmed + đang visible (không phải predicted)."""
        return {
            tid: t for tid, t in self._tracks.items()
            if t.status == TrackStatus.CONFIRMED and t.consecutive_invisible_count == 0
        }

    @property
    def all_tracks(self) -> Dict[int, TrackedVehicle]:
        """Tất cả tracks hiện tại (gồm tentative)."""
        return dict(self._tracks)

    @property
    def exited_tracks(self) -> Dict[int, TrackedVehicle]:
        """Xe đã rời bãi."""
        return dict(self._exited_tracks)

    @property
    def frame_index(self) -> int:
        return self._frame_idx
