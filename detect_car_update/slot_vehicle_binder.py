"""Slot ↔ Vehicle ID Binder — cầu nối giữa parking detection và vehicle tracking.

Khi xe đỗ vào ô:
  - Tracker mất track (xe đứng yên → không có chuyển động)
  - Nhưng Parking Detector vẫn thấy ô "occupied"
  - Binder ghi nhớ: ô D01 đang chứa xe #5

Khi xe rời ô:
  - Parking Detector thấy ô chuyển empty
  - Motion tracker phát hiện xe mới xuất phát từ ô D01
  - Binder trả ID cũ #5 cho xe mới → ID được giữ nguyên
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


@dataclass
class SlotBinding:
    """Trạng thái binding của 1 ô đỗ."""
    slot_id: str
    vehicle_id: Optional[int] = None
    occupied: bool = False
    polygon: Optional[np.ndarray] = None   # (N, 2) int32
    center: Tuple[int, int] = (0, 0)
    bound_at_frame: int = 0                # Frame mà xe được gán vào ô


class SlotVehicleBinder:
    """Quản lý mapping giữa ô đỗ (slot) và xe (vehicle ID).

    Workflow mỗi frame:
      1. tracker_main gọi ``update(active_tracks, slot_results, frame_idx)``
      2. Binder cập nhật trạng thái binding
      3. Khi tracker tạo detection mới, gọi ``try_recover_id(position)``
         để kiểm tra có slot nào vừa thả xe → trả ID cũ

    Parameters
    ----------
    margin : float
        Khoảng cách tối đa (pixel) giữa tâm xe và tâm ô đỗ
        để xét xe đang ở trong ô.  Dùng kết hợp với point-in-polygon.
    release_grace_frames : int
        Sau khi ô chuyển empty, giữ vehicle_id thêm N frame
        để tracker có thời gian phát hiện xe mới rời ô.
    """

    def __init__(
        self,
        margin: float = 50.0,
        release_grace_frames: int = 30,
    ):
        self.margin = margin
        self.release_grace_frames = release_grace_frames

        self._bindings: Dict[str, SlotBinding] = {}   # slot_id → SlotBinding
        self._vehicle_to_slot: Dict[int, str] = {}    # vehicle_id → slot_id
        # Ô vừa thả xe (chờ tracker nhận lại ID):
        # slot_id → (vehicle_id, released_at_frame)
        self._pending_release: Dict[str, Tuple[int, int]] = {}

    @property
    def bindings(self) -> Dict[str, SlotBinding]:
        return dict(self._bindings)

    def get_vehicle_id_for_slot(self, slot_id: str) -> Optional[int]:
        binding = self._bindings.get(slot_id)
        return binding.vehicle_id if binding else None

    def get_slot_for_vehicle(self, vehicle_id: int) -> Optional[str]:
        return self._vehicle_to_slot.get(vehicle_id)

    def get_all_parked_vehicle_ids(self) -> Set[int]:
        """Trả về set các vehicle_id đang đỗ trong ô."""
        return {
            b.vehicle_id
            for b in self._bindings.values()
            if b.vehicle_id is not None
        }

    @staticmethod
    def _point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
        """Kiểm tra point có nằm trong polygon không (dùng OpenCV)."""
        result = cv2.pointPolygonTest(
            polygon.reshape((-1, 1, 2)).astype(np.float32),
            (float(point[0]), float(point[1])),
            measureDist=False,
        )
        return result >= 0  # >= 0 nghĩa là bên trong hoặc trên biên

    def _find_vehicle_in_slot(
        self,
        slot_polygon: np.ndarray,
        slot_center: Tuple[int, int],
        active_tracks: dict,
        already_bound: Set[int],
    ) -> Optional[int]:
        """Tìm xe (từ active_tracks) đang nằm trong polygon của ô đỗ.

        Ưu tiên xe gần tâm nhất.  Bỏ qua xe đã bound vào ô khác.
        """
        best_tid = None
        best_dist = float("inf")

        for tid, track in active_tracks.items():
            if tid in already_bound:
                continue
            point = (track.cx, track.cy)
            if not self._point_in_polygon(point, slot_polygon):
                continue
            dist = np.sqrt(
                (point[0] - slot_center[0]) ** 2 + (point[1] - slot_center[1]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        return best_tid

    def update(
        self,
        active_tracks: dict,
        slot_results: list,
        frame_idx: int,
    ) -> None:
        """Cập nhật binding mỗi frame.

        Parameters
        ----------
        active_tracks : dict
            ``{track_id: TrackedVehicle}`` từ motion_tracker.active_tracks
        slot_results : list
            ``[SlotResult(...)]`` từ parking_detector.detect()
        frame_idx : int
            Số frame hiện tại
        """
        already_bound: Set[int] = set()

        for sr in slot_results:
            slot_id = sr.slot_id
            occupied = sr.occupied
            polygon = sr.polygon
            center = sr.center

            binding = self._bindings.get(slot_id)
            if binding is None:
                binding = SlotBinding(
                    slot_id=slot_id,
                    polygon=polygon,
                    center=center,
                )
                self._bindings[slot_id] = binding

            prev_occupied = binding.occupied
            binding.occupied = occupied
            binding.polygon = polygon
            binding.center = center

            if occupied:
                # ── Ô ĐANG CÓ XE ──
                # Xóa pending release nếu có (xe quay lại ô)
                self._pending_release.pop(slot_id, None)

                if binding.vehicle_id is None:
                    # Chưa có ID → tìm xe active trong polygon
                    vid = self._find_vehicle_in_slot(
                        polygon, center, active_tracks, already_bound
                    )
                    if vid is not None:
                        binding.vehicle_id = vid
                        binding.bound_at_frame = frame_idx
                        self._vehicle_to_slot[vid] = slot_id
                        already_bound.add(vid)
                        print(f"  🅿️ Xe #{vid} đỗ vào ô {slot_id}")
                else:
                    # Đã có ID → kiểm tra xe đó còn active không
                    vid = binding.vehicle_id
                    if vid in active_tracks:
                        already_bound.add(vid)
                    # Nếu xe không còn active → vẫn giữ ID (xe đứng yên)

                # Gán vehicle_id vào SlotResult để JSON/UI hiển thị
                sr.vehicle_id = binding.vehicle_id

            else:
                # ── Ô TRỐNG ──
                if prev_occupied and binding.vehicle_id is not None:
                    # Vừa chuyển occupied → empty: xe rời ô
                    vid = binding.vehicle_id
                    print(f"  🚗 Xe #{vid} rời ô {slot_id} (chờ tracker nhận lại)")
                    self._pending_release[slot_id] = (vid, frame_idx)
                    # Xóa binding
                    self._vehicle_to_slot.pop(vid, None)
                    binding.vehicle_id = None
                    sr.vehicle_id = None

        # Cleanup: xóa pending quá hạn
        expired = [
            sid
            for sid, (_, released_at) in self._pending_release.items()
            if frame_idx - released_at > self.release_grace_frames
        ]
        for sid in expired:
            vid, _ = self._pending_release.pop(sid)
            print(f"  ⏰ Hết hạn chờ nhận lại ID #{vid} từ ô {sid}")

    def try_recover_id(self, position: Tuple[int, int]) -> Optional[int]:
        """Kiểm tra xem position có nằm trong ô vừa thả xe → trả ID cũ.

        Gọi method này khi motion_tracker tạo detection mới (trước khi cấp ID).
        Nếu detection nằm trong ô pending release → trả vehicle_id cũ.
        """
        for slot_id, (vid, _) in list(self._pending_release.items()):
            binding = self._bindings.get(slot_id)
            if binding is None or binding.polygon is None:
                continue
            if self._point_in_polygon(position, binding.polygon):
                # Match! Trả ID cũ
                self._pending_release.pop(slot_id)
                print(f"  🔄 Recover: Xe mới từ ô {slot_id} → nhận lại ID #{vid}")
                return vid

        return None

    def to_json(self) -> Dict[str, dict]:
        """Export trạng thái ô đỗ có vehicle_id cho JSON output."""
        result = {}
        for slot_id, binding in self._bindings.items():
            result[slot_id] = {
                "occupied": binding.occupied,
                "status": "occupied" if binding.occupied else "empty",
                "vehicle_id": binding.vehicle_id,
            }
        return result
