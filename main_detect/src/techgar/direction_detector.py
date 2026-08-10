"""
direction_detector.py — Xác định hướng đi của xe tại các ngã rẽ (ROI lines).

Thuật toán:
  1. Kiểm tra mỗi frame: xe nào cắt qua ROI line (junction)?
  2. Khi xe cắt vạch → ghi nhận "pending decision" và lưu vị trí trước vạch
  3. Trong K frame tiếp theo, theo dõi xe di chuyển
  4. So sánh hướng di chuyển SAU vạch so với hướng TRƯỚC vạch:
     - Lệch sang trái → TURN_LEFT
     - Lệch sang phải → TURN_RIGHT
     - Đi thẳng → STRAIGHT
  5. Ghi decision vào track.direction_events
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ROILine:
    """Một đường ROI tại ngã rẽ."""
    id: str
    name: str
    p1: Tuple[int, int]  # Điểm đầu (x1, y1)
    p2: Tuple[int, int]  # Điểm cuối (x2, y2)


@dataclass
class PendingDecision:
    """Xe đang chờ quyết định hướng sau khi cắt vạch."""
    track_id: int
    roi_id: str
    cross_frame: int                # Frame xe cắt vạch
    position_before: Tuple[int, int]  # Tọa độ trung bình TRƯỚC khi cắt
    positions_after: List[Tuple[int, int]] = field(default_factory=list)
    decided: bool = False
    decision: str = ""


class DirectionDetector:
    """
    Phát hiện hướng đi xe tại ngã rẽ dựa trên ROI lines.

    Tham số:
        decision_frames: Số frame sau khi cắt vạch để đưa ra quyết định
        angle_threshold: Góc (độ) để phân biệt rẽ trái/phải vs đi thẳng
    """

    def __init__(
        self,
        roi_lines: List[ROILine] = None,
        decision_frames: int = 10,
        angle_threshold: float = 20.0,
        history_before: int = 5,
        min_after_distance: float = 35.0,
        max_decision_frames: int = 90,
    ):
        self.roi_lines = roi_lines or []
        self.decision_frames = decision_frames
        self.angle_threshold = angle_threshold
        self.history_before = history_before
        self.min_after_distance = float(min_after_distance)
        self.max_decision_frames = int(max_decision_frames)

        # Track ID → danh sách ROI đã cắt (tránh detect lại cùng 1 vạch)
        self._crossed: Dict[int, set] = {}

        # Pending decisions
        self._pending: List[PendingDecision] = []

    @classmethod
    def from_json(cls, json_path: str, **kwargs) -> "DirectionDetector":
        """Load ROI lines từ file JSON."""
        path = Path(json_path)
        if not path.exists():
            print(f"[DirectionDetector] Không tìm thấy {json_path} → không có ROI lines")
            return cls(roi_lines=[], **kwargs)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []
        for item in data.get("lines", []):
            lines.append(ROILine(
                id=item["id"],
                name=item.get("name", item["id"]),
                p1=tuple(item["p1"]),
                p2=tuple(item["p2"]),
            ))

        print(f"[DirectionDetector] Loaded {len(lines)} ROI lines từ {json_path}")
        return cls(roi_lines=lines, **kwargs)

    # ──────────────────────────────────────────────
    # Kiểm tra xe cắt vạch
    # ──────────────────────────────────────────────
    @staticmethod
    def _ccw(A, B, C):
        """Counter-clockwise test cho 3 điểm."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(A, B, C, D) -> bool:
        """Kiểm tra 2 đoạn thẳng AB và CD có giao nhau không."""
        return (
            DirectionDetector._ccw(A, C, D) != DirectionDetector._ccw(B, C, D) and
            DirectionDetector._ccw(A, B, C) != DirectionDetector._ccw(A, B, D)
        )

    def _check_line_crossing(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> List[str]:
        """
        Kiểm tra xe có cắt qua ROI line nào giữa prev_pos → curr_pos.
        Trả về tất cả roi_id bị cắt. Một bước lớn có thể cắt nhiều vạch.
        """
        crossed = []
        for roi in self.roi_lines:
            if self._segments_intersect(prev_pos, curr_pos, roi.p1, roi.p2):
                crossed.append(roi.id)
        return crossed

    # ──────────────────────────────────────────────
    # Xác định hướng
    # ──────────────────────────────────────────────
    def _compute_direction(self, pos_before: Tuple[int, int], positions_after: List[Tuple[int, int]]) -> str:
        """
        Tính hướng di chuyển dựa trên vị trí trước và sau vạch.

        Logic:
        - Tính vector hướng TRƯỚC vạch (từ history)
        - Tính vector hướng SAU vạch (từ positions_after)
        - Tính góc giữa 2 vector → quyết định LEFT/RIGHT/STRAIGHT
        """
        if len(positions_after) < 3:
            return "UNKNOWN"

        # Vector hướng SAU vạch: từ điểm đầu đến điểm cuối
        after_start = np.array(positions_after[0], dtype=float)
        after_end = np.array(positions_after[-1], dtype=float)
        vec_after = after_end - after_start

        # Vector hướng TRƯỚC vạch: từ pos_before đến điểm cắt vạch
        before_pt = np.array(pos_before, dtype=float)
        vec_before = after_start - before_pt

        # Tính góc (dùng cross product để xác định chiều)
        len_before = np.linalg.norm(vec_before)
        len_after = np.linalg.norm(vec_after)

        if len_before < 1 or len_after < 1:
            return "STRAIGHT"

        # Cross product z-component: dương = rẽ trái, âm = rẽ phải
        cross = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]

        # Góc giữa 2 vector
        cos_angle = np.dot(vec_before, vec_after) / (len_before * len_after)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg < self.angle_threshold:
            return "STRAIGHT"
        elif cross > 0:
            return "TURN_LEFT"
        else:
            return "TURN_RIGHT"

    # ──────────────────────────────────────────────
    # API chính: update mỗi frame
    # ──────────────────────────────────────────────
    def update(self, tracks: dict, frame_idx: int) -> List[dict]:
        """
        Gọi mỗi frame sau khi tracker đã update.

        Returns:
            List of new direction events: [{"track_id": X, "roi": "...", "decision": "TURN_LEFT", "frame": N}]
        """
        if not self.roi_lines:
            return []

        new_events = []

        # Kiểm tra xe cắt vạch
        for tid, track in tracks.items():
            if len(track.history) < 2:
                continue

            # Chỉ xét confirmed tracks
            from .vehicle_tracker import TrackStatus
            if track.status == TrackStatus.TENTATIVE:
                continue

            prev_pos = track.history[-2]
            curr_pos = track.history[-1]

            # Tránh kết luận khi bbox rung vài pixel tại đúng vạch.
            if np.linalg.norm(np.subtract(curr_pos, prev_pos)) < 2:
                continue

            if tid not in self._crossed:
                self._crossed[tid] = set()
            for roi_id in self._check_line_crossing(prev_pos, curr_pos):
                # Đã cắt vạch này trước đó chưa?
                if roi_id in self._crossed[tid]:
                    continue
                self._crossed[tid].add(roi_id)

                # Tính vị trí trung bình TRƯỚC khi cắt
                n_before = min(self.history_before, len(track.history) - 1)
                before_positions = track.history[-(n_before + 1):-1]
                avg_before = (
                    int(np.mean([p[0] for p in before_positions])),
                    int(np.mean([p[1] for p in before_positions])),
                )
                self._pending.append(PendingDecision(
                    track_id=tid,
                    roi_id=roi_id,
                    cross_frame=frame_idx,
                    position_before=avg_before,
                ))

        # Cập nhật pending decisions
        for pd in self._pending:
            if pd.decided:
                continue

            if pd.track_id in tracks:
                track = tracks[pd.track_id]
                pd.positions_after.append((track.cx, track.cy))

                distance_after = np.linalg.norm(
                    np.subtract(pd.positions_after[-1], pd.positions_after[0])
                )
                # Chốt theo cả số quan sát và quãng đường: xe đi chậm vẫn chính xác,
                # xe đi nhanh không cần chờ đủ nhiều frame.
                if (
                    len(pd.positions_after) >= self.decision_frames
                    and distance_after >= self.min_after_distance
                ):
                    pd.decision = self._compute_direction(pd.position_before, pd.positions_after)
                    pd.decided = True

                    # Ghi event vào track
                    event = {
                        "roi": pd.roi_id,
                        "decision": pd.decision,
                        "frame": pd.cross_frame,
                    }
                    track.direction_events.append(event)
                    new_events.append({"track_id": pd.track_id, **event})
                    print(f"  🧭 Xe #{pd.track_id} tại {pd.roi_id}: {pd.decision} (frame {pd.cross_frame})")
            elif frame_idx - pd.cross_frame >= self.max_decision_frames:
                # Không còn observation đủ lâu: không dùng vị trí predicted để đoán hướng.
                pd.decided = True
                pd.decision = "UNKNOWN"

        # Cleanup pending đã xong
        self._pending = [pd for pd in self._pending if not pd.decided]

        # Cleanup crossed cho tracks đã bị xóa
        active_ids = set(tracks.keys())
        remove_tids = [tid for tid in self._crossed if tid not in active_ids]
        for tid in remove_tids:
            del self._crossed[tid]

        return new_events

    # ──────────────────────────────────────────────
    # Vẽ ROI lines lên frame
    # ──────────────────────────────────────────────
    def draw_roi_lines(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ tất cả ROI lines lên frame."""
        import cv2
        out = frame  # Modify in-place cho performance
        for roi in self.roi_lines:
            cv2.line(out, roi.p1, roi.p2, (255, 0, 255), 2)
            mid_x = (roi.p1[0] + roi.p2[0]) // 2
            mid_y = (roi.p1[1] + roi.p2[1]) // 2
            cv2.putText(out, roi.name, (mid_x, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return out
