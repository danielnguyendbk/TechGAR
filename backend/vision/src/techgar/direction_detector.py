"""
direction_detector.py ΓÇö X├íc ─æß╗ïnh h╞░ß╗¢ng ─æi cß╗ºa xe tß║íi c├íc ng├ú rß║╜ (ROI lines).

Thuß║¡t to├ín:
  1. Kiß╗âm tra mß╗ùi frame: xe n├áo cß║»t qua ROI line (junction)?
  2. Khi xe cß║»t vß║ích ΓåÆ ghi nhß║¡n "pending decision" v├á l╞░u vß╗ï tr├¡ tr╞░ß╗¢c vß║ích
  3. Trong K frame tiß║┐p theo, theo d├╡i xe di chuyß╗ân
  4. So s├ính h╞░ß╗¢ng di chuyß╗ân SAU vß║ích so vß╗¢i h╞░ß╗¢ng TR╞»ß╗ÜC vß║ích:
     - Lß╗çch sang tr├íi ΓåÆ TURN_LEFT
     - Lß╗çch sang phß║úi ΓåÆ TURN_RIGHT
     - ─Éi thß║│ng ΓåÆ STRAIGHT
  5. Ghi decision v├áo track.direction_events
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ROILine:
    """Mß╗Öt ─æ╞░ß╗¥ng ROI tß║íi ng├ú rß║╜."""
    id: str
    name: str
    p1: Tuple[int, int]  # ─Éiß╗âm ─æß║ºu (x1, y1)
    p2: Tuple[int, int]  # ─Éiß╗âm cuß╗æi (x2, y2)


@dataclass
class PendingDecision:
    """Xe ─æang chß╗¥ quyß║┐t ─æß╗ïnh h╞░ß╗¢ng sau khi cß║»t vß║ích."""
    track_id: int
    roi_id: str
    cross_frame: int                # Frame xe cß║»t vß║ích
    position_before: Tuple[int, int]  # Tß╗ìa ─æß╗Ö trung b├¼nh TR╞»ß╗ÜC khi cß║»t
    positions_after: List[Tuple[int, int]] = field(default_factory=list)
    decided: bool = False
    decision: str = ""


class DirectionDetector:
    """
    Ph├ít hiß╗çn h╞░ß╗¢ng ─æi xe tß║íi ng├ú rß║╜ dß╗▒a tr├¬n ROI lines.

    Tham sß╗æ:
        decision_frames: Sß╗æ frame sau khi cß║»t vß║ích ─æß╗â ─æ╞░a ra quyß║┐t ─æß╗ïnh
        angle_threshold: G├│c (─æß╗Ö) ─æß╗â ph├ón biß╗çt rß║╜ tr├íi/phß║úi vs ─æi thß║│ng
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

        # Track ID ΓåÆ danh s├ích ROI ─æ├ú cß║»t (tr├ính detect lß║íi c├╣ng 1 vß║ích)
        self._crossed: Dict[int, set] = {}

        # Pending decisions
        self._pending: List[PendingDecision] = []

    @classmethod
    def from_json(cls, json_path: str, **kwargs) -> "DirectionDetector":
        """Load ROI lines tß╗½ file JSON."""
        path = Path(json_path)
        if not path.exists():
            print(f"[DirectionDetector] Kh├┤ng t├¼m thß║Ñy {json_path} ΓåÆ kh├┤ng c├│ ROI lines")
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

        print(f"[DirectionDetector] Loaded {len(lines)} ROI lines tß╗½ {json_path}")
        return cls(roi_lines=lines, **kwargs)

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Kiß╗âm tra xe cß║»t vß║ích
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    @staticmethod
    def _ccw(A, B, C):
        """Counter-clockwise test cho 3 ─æiß╗âm."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(A, B, C, D) -> bool:
        """Kiß╗âm tra 2 ─æoß║ín thß║│ng AB v├á CD c├│ giao nhau kh├┤ng."""
        return (
            DirectionDetector._ccw(A, C, D) != DirectionDetector._ccw(B, C, D) and
            DirectionDetector._ccw(A, B, C) != DirectionDetector._ccw(A, B, D)
        )

    def _check_line_crossing(self, prev_pos: Tuple[int, int], curr_pos: Tuple[int, int]) -> List[str]:
        """
        Kiß╗âm tra xe c├│ cß║»t qua ROI line n├áo giß╗»a prev_pos ΓåÆ curr_pos.
        Trß║ú vß╗ü tß║Ñt cß║ú roi_id bß╗ï cß║»t. Mß╗Öt b╞░ß╗¢c lß╗¢n c├│ thß╗â cß║»t nhiß╗üu vß║ích.
        """
        crossed = []
        for roi in self.roi_lines:
            if self._segments_intersect(prev_pos, curr_pos, roi.p1, roi.p2):
                crossed.append(roi.id)
        return crossed

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # X├íc ─æß╗ïnh h╞░ß╗¢ng
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    def _compute_direction(self, pos_before: Tuple[int, int], positions_after: List[Tuple[int, int]]) -> str:
        """
        T├¡nh h╞░ß╗¢ng di chuyß╗ân dß╗▒a tr├¬n vß╗ï tr├¡ tr╞░ß╗¢c v├á sau vß║ích.

        Logic:
        - T├¡nh vector h╞░ß╗¢ng TR╞»ß╗ÜC vß║ích (tß╗½ history)
        - T├¡nh vector h╞░ß╗¢ng SAU vß║ích (tß╗½ positions_after)
        - T├¡nh g├│c giß╗»a 2 vector ΓåÆ quyß║┐t ─æß╗ïnh LEFT/RIGHT/STRAIGHT
        """
        if len(positions_after) < 3:
            return "UNKNOWN"

        # Vector h╞░ß╗¢ng SAU vß║ích: tß╗½ ─æiß╗âm ─æß║ºu ─æß║┐n ─æiß╗âm cuß╗æi
        after_start = np.array(positions_after[0], dtype=float)
        after_end = np.array(positions_after[-1], dtype=float)
        vec_after = after_end - after_start

        # Vector h╞░ß╗¢ng TR╞»ß╗ÜC vß║ích: tß╗½ pos_before ─æß║┐n ─æiß╗âm cß║»t vß║ích
        before_pt = np.array(pos_before, dtype=float)
        vec_before = after_start - before_pt

        # T├¡nh g├│c (d├╣ng cross product ─æß╗â x├íc ─æß╗ïnh chiß╗üu)
        len_before = np.linalg.norm(vec_before)
        len_after = np.linalg.norm(vec_after)

        if len_before < 1 or len_after < 1:
            return "STRAIGHT"

        # Cross product z-component: d╞░╞íng = rß║╜ tr├íi, ├óm = rß║╜ phß║úi
        cross = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]

        # G├│c giß╗»a 2 vector
        cos_angle = np.dot(vec_before, vec_after) / (len_before * len_after)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg < self.angle_threshold:
            return "STRAIGHT"
        elif cross > 0:
            return "TURN_LEFT"
        else:
            return "TURN_RIGHT"

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # API ch├¡nh: update mß╗ùi frame
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    def update(self, tracks: dict, frame_idx: int) -> List[dict]:
        """
        Gß╗ìi mß╗ùi frame sau khi tracker ─æ├ú update.

        Returns:
            List of new direction events: [{"track_id": X, "roi": "...", "decision": "TURN_LEFT", "frame": N}]
        """
        if not self.roi_lines:
            return []

        new_events = []

        # Kiß╗âm tra xe cß║»t vß║ích
        for tid, track in tracks.items():
            if len(track.history) < 2:
                continue

            # Chß╗ë x├⌐t confirmed tracks
            from .vehicle_tracker import TrackStatus
            if track.status == TrackStatus.TENTATIVE:
                continue

            prev_pos = track.history[-2]
            curr_pos = track.history[-1]

            # Tr├ính kß║┐t luß║¡n khi bbox rung v├ái pixel tß║íi ─æ├║ng vß║ích.
            if np.linalg.norm(np.subtract(curr_pos, prev_pos)) < 2:
                continue

            if tid not in self._crossed:
                self._crossed[tid] = set()
            for roi_id in self._check_line_crossing(prev_pos, curr_pos):
                # ─É├ú cß║»t vß║ích n├áy tr╞░ß╗¢c ─æ├│ ch╞░a?
                if roi_id in self._crossed[tid]:
                    continue
                self._crossed[tid].add(roi_id)

                # T├¡nh vß╗ï tr├¡ trung b├¼nh TR╞»ß╗ÜC khi cß║»t
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

        # Cß║¡p nhß║¡t pending decisions
        for pd in self._pending:
            if pd.decided:
                continue

            if pd.track_id in tracks:
                track = tracks[pd.track_id]
                pd.positions_after.append((track.cx, track.cy))

                distance_after = np.linalg.norm(
                    np.subtract(pd.positions_after[-1], pd.positions_after[0])
                )
                # Chß╗æt theo cß║ú sß╗æ quan s├ít v├á qu├úng ─æ╞░ß╗¥ng: xe ─æi chß║¡m vß║½n ch├¡nh x├íc,
                # xe ─æi nhanh kh├┤ng cß║ºn chß╗¥ ─æß╗º nhiß╗üu frame.
                if (
                    len(pd.positions_after) >= self.decision_frames
                    and distance_after >= self.min_after_distance
                ):
                    pd.decision = self._compute_direction(pd.position_before, pd.positions_after)
                    pd.decided = True

                    # Ghi event v├áo track
                    event = {
                        "roi": pd.roi_id,
                        "decision": pd.decision,
                        "frame": pd.cross_frame,
                    }
                    track.direction_events.append(event)
                    new_events.append({"track_id": pd.track_id, **event})
                    print(f"  ≡ƒº¡ Xe #{pd.track_id} tß║íi {pd.roi_id}: {pd.decision} (frame {pd.cross_frame})")
            elif frame_idx - pd.cross_frame >= self.max_decision_frames:
                # Kh├┤ng c├▓n observation ─æß╗º l├óu: kh├┤ng d├╣ng vß╗ï tr├¡ predicted ─æß╗â ─æo├ín h╞░ß╗¢ng.
                pd.decided = True
                pd.decision = "UNKNOWN"

        # Cleanup pending ─æ├ú xong
        self._pending = [pd for pd in self._pending if not pd.decided]

        # Cleanup crossed cho tracks ─æ├ú bß╗ï x├│a
        active_ids = set(tracks.keys())
        remove_tids = [tid for tid in self._crossed if tid not in active_ids]
        for tid in remove_tids:
            del self._crossed[tid]

        return new_events

    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Vß║╜ ROI lines l├¬n frame
    # ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    def draw_roi_lines(self, frame: np.ndarray) -> np.ndarray:
        """Vß║╜ tß║Ñt cß║ú ROI lines l├¬n frame."""
        import cv2
        out = frame  # Modify in-place cho performance
        for roi in self.roi_lines:
            cv2.line(out, roi.p1, roi.p2, (255, 0, 255), 2)
            mid_x = (roi.p1[0] + roi.p2[0]) // 2
            mid_y = (roi.p1[1] + roi.p2[1]) // 2
            cv2.putText(out, roi.name, (mid_x, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        return out
