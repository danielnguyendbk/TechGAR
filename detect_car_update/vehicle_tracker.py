"""Vehicle tracking cho bãi xe bằng YOLO + BoT-SORT/Re-ID.

Module này thay thế detector MOG2/contour cũ.  MOG2 chỉ nhìn thay đổi pixel nên
không thể theo xe đang đứng yên, dễ vỡ khi thay đổi ánh sáng và không có dữ liệu
để nhận lại ID sau che khuất.  YOLO phát hiện xe trực tiếp; BoT-SORT xử lý dự
đoán chuyển động, IoU, low-confidence association và Re-ID theo ngoại hình.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TrackStatus(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class TrackedVehicle:
    """Một xe với ID do BoT-SORT duy trì.

    ``cx, cy`` là điểm tiếp xúc mặt đường (giữa cạnh đáy bbox), ổn định hơn
    tâm bbox khi camera nhìn xiên.  Muốn có toạ độ mét/sơ đồ bãi, truyền
    homography vào :class:`VehicleTracker`.
    """

    track_id: int
    cx: int
    cy: int
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float = 0.0
    class_id: int = -1
    class_name: str = "vehicle"
    age: int = 1
    total_visible_count: int = 1
    consecutive_invisible_count: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE
    history: List[Tuple[int, int]] = field(default_factory=list)
    direction_events: List[dict] = field(default_factory=list)
    entered_frame: int = 0
    exited_frame: int = 0
    last_seen_frame: int = 0
    ground_point: Optional[Tuple[float, float]] = None

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def visibility(self) -> float:
        return self.total_visible_count / max(self.age, 1)


class VehicleTracker:
    """YOLO detector và tracker BoT-SORT stateful cho từng frame.

    Khi ``tracker_config`` là ``botsort_parking_reid.yaml``, tracker dùng Re-ID
    để cố giữ cùng ID khi xe bị che ngắn hạn.  Với camera cố định, cấu hình tắt
    camera-motion compensation để tiết kiệm chi phí.
    """

    DEFAULT_VEHICLE_CLASS_IDS = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck

    def __init__(
        self,
        model_path: str,
        tracker_config: str,
        confidence: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 960,
        device: Optional[str] = None,
        class_ids: Optional[List[int]] = None,
        min_visible_count: int = 2,
        lost_track_ttl: int = 90,
        history_len: int = 90,
        homography: Optional[np.ndarray] = None,
    ):
        # Không ghi settings vào AppData của hệ thống; giữ mọi state trong project.
        local_yolo_config = Path(__file__).resolve().parent / ".ultralytics"
        local_yolo_config.mkdir(exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(local_yolo_config))
        local_matplotlib_config = Path(__file__).resolve().parent / ".cache" / "matplotlib"
        local_matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(local_matplotlib_config))
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu ultralytics. Hãy cài dependencies bằng: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.model_path = str(model_path)
        self.tracker_config = str(tracker_config)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device or None
        self.class_ids = list(class_ids or self.DEFAULT_VEHICLE_CLASS_IDS)
        self.min_visible_count = max(1, int(min_visible_count))
        self.lost_track_ttl = max(1, int(lost_track_ttl))
        self.history_len = max(2, int(history_len))
        self.homography = homography

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy YOLO model: {self.model_path}")
        if not Path(self.tracker_config).exists():
            raise FileNotFoundError(f"Không tìm thấy tracker config: {self.tracker_config}")

        self.model = YOLO(self.model_path)
        self._tracks: Dict[int, TrackedVehicle] = {}
        self._exited_tracks: Dict[int, TrackedVehicle] = {}
        self._frame_idx = 0

    @staticmethod
    def _bottom_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[int, int]:
        return int(round((x1 + x2) / 2)), int(round(y2))

    def _project_ground_point(self, point: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self.homography is None:
            return None
        src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography.astype(np.float32))[0, 0]
        return round(float(dst[0]), 3), round(float(dst[1]), 3)

    def _update_track(
        self,
        track_id: int,
        xyxy: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str,
    ) -> None:
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        x, y = int(round(x1)), int(round(y1))
        w, h = max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
        point = self._bottom_center(x1, y1, x2, y2)
        ground_point = self._project_ground_point(point)

        track = self._tracks.get(track_id)
        if track is None:
            track = TrackedVehicle(
                track_id=track_id,
                cx=point[0], cy=point[1], bbox=(x, y, w, h), area=float(w * h),
                confidence=confidence, class_id=class_id, class_name=class_name,
                status=TrackStatus.CONFIRMED if self.min_visible_count == 1 else TrackStatus.TENTATIVE,
                history=[point], entered_frame=self._frame_idx, last_seen_frame=self._frame_idx,
                ground_point=ground_point,
            )
            self._tracks[track_id] = track
            return

        track.age += 1
        track.total_visible_count += 1
        track.consecutive_invisible_count = 0
        track.status = (
            TrackStatus.CONFIRMED
            if track.total_visible_count >= self.min_visible_count
            else TrackStatus.TENTATIVE
        )
        track.cx, track.cy = point
        track.bbox = (x, y, w, h)
        track.area = float(w * h)
        track.confidence = confidence
        track.class_id = class_id
        track.class_name = class_name
        track.last_seen_frame = self._frame_idx
        track.ground_point = ground_point
        track.history.append(point)
        if len(track.history) > self.history_len:
            track.history = track.history[-self.history_len:]

    def _age_missing_tracks(self, visible_ids: set[int]) -> List[Tuple[int, TrackedVehicle]]:
        expired: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in visible_ids:
                continue
            track.age += 1
            track.consecutive_invisible_count += 1
            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.LOST
            if track.consecutive_invisible_count > self.lost_track_ttl:
                expired.append(track_id)

        expired_tracks = []
        for track_id in expired:
            track = self._tracks.pop(track_id)
            track.exited_frame = self._frame_idx
            self._exited_tracks[track_id] = track
            expired_tracks.append((track_id, track))
        return expired_tracks

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict[int, TrackedVehicle], np.ndarray, List[Tuple[int, TrackedVehicle]]]:
        """Phát hiện + track một frame. Trả bbox mask để tương thích UI cũ."""
        self._frame_idx += 1
        kwargs = {
            "persist": True,
            "tracker": self.tracker_config,
            "classes": self.class_ids,
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.track(frame, **kwargs)
        result = results[0]
        boxes = result.boxes
        visible_ids: set[int] = set()

        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.int().cpu().tolist()
            names = result.names or {}
            for box, track_id, conf, class_id in zip(xyxy, ids, confs, classes):
                track_id = int(track_id)
                visible_ids.add(track_id)
                self._update_track(track_id, box, float(conf), int(class_id), str(names.get(int(class_id), class_id)))

        expired_tracks = self._age_missing_tracks(visible_ids)
        debug_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for track in self.active_tracks.values():
            cv2.rectangle(debug_mask, (track.x, track.y), (track.x + track.w, track.y + track.h), 255, -1)
        return self._tracks, debug_mask, expired_tracks

    def draw_tracks(self, frame: np.ndarray, tracks: Optional[Dict[int, TrackedVehicle]] = None, show_non_active: bool = False) -> np.ndarray:
        out = frame.copy()
        tracks = tracks if tracks is not None else self._tracks
        active = tentative = lost = 0
        for track in tracks.values():
            if not show_non_active and track.status != TrackStatus.CONFIRMED:
                continue
            if track.status == TrackStatus.TENTATIVE:
                tentative += 1
                color, thickness, suffix = (90, 90, 90), 1, " tentative"
            elif track.status == TrackStatus.LOST:
                lost += 1
                color, thickness, suffix = (0, 165, 255), 1, " lost"
            else:
                active += 1
                color, thickness, suffix = (0, 255, 0), 2, ""
            cv2.rectangle(out, (track.x, track.y), (track.x + track.w, track.y + track.h), color, thickness)
            cv2.putText(out, f"#{track.track_id} {track.confidence:.2f}{suffix}", (track.x, max(16, track.y - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(out, (track.cx, track.cy), 4, (0, 0, 255), -1)
            for i in range(1, len(track.history)):
                cv2.line(out, track.history[i - 1], track.history[i], (255, 200, 0), 2)
        cv2.putText(out, f"Active: {active} | Tentative: {tentative} | Lost: {lost}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return out

    @property
    def confirmed_tracks(self) -> Dict[int, TrackedVehicle]:
        return {tid: t for tid, t in self._tracks.items() if t.status in (TrackStatus.CONFIRMED, TrackStatus.LOST)}

    @property
    def active_tracks(self) -> Dict[int, TrackedVehicle]:
        return {tid: t for tid, t in self._tracks.items() if t.status == TrackStatus.CONFIRMED and t.consecutive_invisible_count == 0}

    @property
    def all_tracks(self) -> Dict[int, TrackedVehicle]:
        return dict(self._tracks)

    @property
    def exited_tracks(self) -> Dict[int, TrackedVehicle]:
        return dict(self._exited_tracks)

    @property
    def frame_index(self) -> int:
        return self._frame_idx
