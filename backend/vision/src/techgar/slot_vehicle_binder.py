"""Fuse parking-slot vision results with tracked global vehicle IDs.

``ParkingDetector`` remains the baseline source.  Tracking is deliberately a
one-way safety override: a stopped global vehicle can turn a visually-free slot
into occupied, but tracking never turns a visually-occupied slot into free.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from lap import lapjv


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class VehicleObservation:
    timestamp_s: float
    center: Point
    bbox: BBox
    slot_id: Optional[str]
    overlap: float = 0.0


@dataclass
class VehicleParkingState:
    global_id: int
    observations: Deque[VehicleObservation] = field(default_factory=deque)
    movement_state: str = "moving"
    candidate_slot_id: Optional[str] = None
    parked_slot_id: Optional[str] = None
    candidate_since: Optional[float] = None
    outside_since: Optional[float] = None
    last_appearance: Optional[np.ndarray] = None
    last_rejection_at: Optional[float] = None


@dataclass
class SlotBinding:
    slot_id: str
    camera_id: Optional[str] = None
    vehicle_id: Optional[int] = None
    occupied: bool = False
    vision_occupied: bool = False
    tracking_occupied: bool = False
    polygon: Optional[np.ndarray] = None
    center: Point = (0.0, 0.0)
    bound_at_frame: int = 0
    vehicle_overlap: float = 0.0
    stopped_for_ms: int = 0
    tracking_state: str = "moving"
    result_ref: object = field(default=None, repr=False, compare=False)

    @property
    def decision_source(self) -> str:
        if self.vision_occupied and self.tracking_occupied:
            return "vision_and_tracking"
        if self.tracking_occupied:
            return "tracking_override"
        if self.vision_occupied:
            return "vision"
        return "none"


class SlotVehicleBinder:
    """Maintain global-ID to parking-slot assignments.

    The class accepts either one camera or a shared, full-frame coordinate
    space.  Call :meth:`update_tracks` on every frame and :meth:`update_vision`
    whenever the slower ensemble parking detector runs.
    """

    def __init__(
        self,
        margin: float = 50.0,
        release_grace_frames: int = 30,
        bind_confirmations: int = 2,
        stop_seconds: float = 1.0,
        exit_seconds: float = 0.5,
        min_vehicle_overlap: float = 0.35,
        strong_vehicle_overlap: float = 0.60,
        stationary_radius_ratio: float = 0.06,
        stationary_drift_ratio: float = 0.10,
        recovery_expand_ratio: float = 0.15,
        min_stop_samples: int = 8,
        stop_commit_grace_seconds: float = 0.15,
    ):
        # Kept for backwards-compatible construction by older entry points.
        self.margin = float(margin)
        self.release_grace_frames = max(1, int(release_grace_frames))
        self.bind_confirmations = max(1, int(bind_confirmations))

        self.stop_seconds = max(0.1, float(stop_seconds))
        self.exit_seconds = max(0.05, float(exit_seconds))
        self.min_vehicle_overlap = float(min_vehicle_overlap)
        self.strong_vehicle_overlap = float(strong_vehicle_overlap)
        self.stationary_radius_ratio = float(stationary_radius_ratio)
        self.stationary_drift_ratio = float(stationary_drift_ratio)
        self.recovery_expand_ratio = max(0.0, float(recovery_expand_ratio))
        self.min_stop_samples = max(2, int(min_stop_samples))
        # A very short commit grace prevents a motion track that vanished at a
        # slot boundary from flashing "parked" one frame before it is observed
        # outside the ROI.  The measured stationary window is still 1 second.
        self.stop_commit_grace_seconds = max(0.0, float(stop_commit_grace_seconds))

        self._bindings: Dict[str, SlotBinding] = {}
        self._vehicle_to_slot: Dict[int, str] = {}
        self._vehicle_states: Dict[int, VehicleParkingState] = {}
        self._pending_release: Dict[str, Tuple[int, int]] = {}
        self._events: Deque[dict] = deque(maxlen=500)
        self._last_frame_idx = 0
        self._last_timestamp_s = 0.0

    @property
    def bindings(self) -> Dict[str, SlotBinding]:
        return dict(self._bindings)

    @property
    def events(self) -> List[dict]:
        return list(self._events)

    def get_vehicle_id_for_slot(self, slot_id: str) -> Optional[int]:
        binding = self._bindings.get(slot_id)
        return binding.vehicle_id if binding else None

    def get_slot_for_vehicle(self, vehicle_id: int) -> Optional[str]:
        return self._vehicle_to_slot.get(int(vehicle_id))

    def get_slot_state(self, slot_id: str) -> Optional[dict]:
        binding = self._bindings.get(slot_id)
        return self._binding_to_json(binding) if binding is not None else None

    def get_all_parked_vehicle_ids(self, camera_id: Optional[str] = None) -> Set[int]:
        return {
            int(binding.vehicle_id)
            for binding in self._bindings.values()
            if binding.vehicle_id is not None
            and (camera_id is None or binding.camera_id == camera_id)
        }

    def _event(self, event_type: str, **details) -> None:
        event = {
            "type": event_type,
            "frame": self._last_frame_idx,
            "timestamp_s": round(self._last_timestamp_s, 3),
            **details,
        }
        self._events.append(event)

    @staticmethod
    def _point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
        result = cv2.pointPolygonTest(
            np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2)),
            (float(point[0]), float(point[1])),
            measureDist=False,
        )
        return result >= 0

    @staticmethod
    def _offset_polygon(polygon: np.ndarray, offset: Point) -> np.ndarray:
        result = np.asarray(polygon, dtype=np.float32).reshape((-1, 2)).copy()
        result[:, 0] += float(offset[0])
        result[:, 1] += float(offset[1])
        return result

    @staticmethod
    def _track_value(track, key: str, default=None):
        if isinstance(track, dict):
            return track.get(key, default)
        return getattr(track, key, default)

    @classmethod
    def _track_bbox(cls, track, offset: Point = (0.0, 0.0)) -> BBox:
        bbox = cls._track_value(track, "bbox")
        if bbox is None:
            bbox = (
                cls._track_value(track, "x", 0),
                cls._track_value(track, "y", 0),
                cls._track_value(track, "w", 1),
                cls._track_value(track, "h", 1),
            )
        x, y, w, h = [float(value) for value in bbox]
        return x + float(offset[0]), y + float(offset[1]), max(1.0, w), max(1.0, h)

    @staticmethod
    def _bbox_polygon(bbox: BBox) -> np.ndarray:
        x, y, w, h = bbox
        return np.asarray(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32,
        )

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        return abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))

    def _overlap_geometry(self, bbox: BBox, binding: SlotBinding) -> Optional[dict]:
        if binding.polygon is None:
            return None
        vehicle_polygon = self._bbox_polygon(bbox)
        vehicle_area = max(1.0, bbox[2] * bbox[3])
        slot_area = max(1.0, self._polygon_area(binding.polygon))
        try:
            intersection_area, _ = cv2.intersectConvexConvex(
                vehicle_polygon.astype(np.float32),
                np.asarray(binding.polygon, dtype=np.float32),
            )
        except cv2.error:
            intersection_area = 0.0
        vehicle_overlap = float(intersection_area) / vehicle_area
        slot_overlap = float(intersection_area) / slot_area
        center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        center_inside = self._point_in_polygon(center, binding.polygon)
        qualifies = (
            center_inside and vehicle_overlap >= self.min_vehicle_overlap
        ) or vehicle_overlap >= self.strong_vehicle_overlap
        if not qualifies:
            return None

        x, y, w, h = cv2.boundingRect(np.asarray(binding.polygon, dtype=np.float32))
        slot_diagonal = max(1.0, float(np.hypot(w, h)))
        center_distance = float(np.hypot(center[0] - binding.center[0], center[1] - binding.center[1]))
        center_proximity = max(0.0, 1.0 - center_distance / slot_diagonal)
        score = 0.70 * vehicle_overlap + 0.20 * slot_overlap + 0.10 * center_proximity
        return {
            "vehicle_overlap": vehicle_overlap,
            "slot_overlap": slot_overlap,
            "center_proximity": center_proximity,
            "score": min(1.0, score),
        }

    def _batch_match(self, active_tracks: Dict[int, object]) -> Dict[int, Tuple[str, dict]]:
        track_ids = sorted(int(global_id) for global_id in active_tracks)
        bindings = [binding for binding in self._bindings.values() if binding.polygon is not None]
        if not track_ids or not bindings:
            return {}

        invalid_cost = 10.0
        costs = np.full((len(track_ids), len(bindings)), invalid_cost, dtype=np.float64)
        geometries: Dict[Tuple[int, int], dict] = {}
        for row, global_id in enumerate(track_ids):
            bbox = self._track_bbox(active_tracks[global_id])
            current_slot = self._vehicle_to_slot.get(global_id)
            for column, binding in enumerate(bindings):
                geometry = self._overlap_geometry(bbox, binding)
                if geometry is None:
                    continue
                score = geometry["score"]
                if current_slot == binding.slot_id:
                    score = min(1.0, score + 0.15)
                costs[row, column] = 1.0 - score
                geometries[(row, column)] = geometry

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.95)
        matches: Dict[int, Tuple[str, dict]] = {}
        for row, global_id in enumerate(track_ids):
            column = int(row_to_col[row])
            if column < 0 or (row, column) not in geometries:
                continue
            matches[global_id] = (bindings[column].slot_id, geometries[(row, column)])
        return matches

    def _is_stopped(self, state: VehicleParkingState, slot_id: str, now_s: float) -> Tuple[bool, int]:
        window_start = now_s - self.stop_seconds
        samples = [item for item in state.observations if item.timestamp_s >= window_start]
        if len(samples) < self.min_stop_samples:
            return False, 0
        duration = samples[-1].timestamp_s - samples[0].timestamp_s
        if duration < self.stop_seconds * 0.90:
            return False, int(max(0.0, duration) * 1000)
        same_slot_ratio = sum(item.slot_id == slot_id for item in samples) / len(samples)
        if same_slot_ratio < 0.80:
            return False, int(duration * 1000)

        centers = np.asarray([item.center for item in samples], dtype=np.float64)
        median_center = np.median(centers, axis=0)
        radius = np.linalg.norm(centers - median_center, axis=1)
        r95 = float(np.percentile(radius, 95))
        net_displacement = float(np.linalg.norm(centers[-1] - centers[0]))
        diagonals = [float(np.hypot(item.bbox[2], item.bbox[3])) for item in samples]
        bbox_diagonal = max(1.0, float(np.median(diagonals)))
        stopped = (
            r95 <= max(3.0, self.stationary_radius_ratio * bbox_diagonal)
            and net_displacement <= max(5.0, self.stationary_drift_ratio * bbox_diagonal)
        )
        return stopped, int(duration * 1000)

    def _bind_vehicle(self, global_id: int, slot_id: str, frame_idx: int, overlap: float, stopped_ms: int) -> None:
        old_slot = self._vehicle_to_slot.get(global_id)
        if old_slot is not None and old_slot != slot_id:
            self._release_vehicle(global_id, frame_idx, reason="reassigned")
        binding = self._bindings[slot_id]
        if binding.vehicle_id is not None and binding.vehicle_id != global_id:
            return
        binding.vehicle_id = global_id
        binding.tracking_occupied = True
        binding.bound_at_frame = frame_idx
        binding.vehicle_overlap = float(overlap)
        binding.stopped_for_ms = int(stopped_ms)
        binding.tracking_state = "parked"
        self._vehicle_to_slot[global_id] = slot_id
        state = self._vehicle_states[global_id]
        state.movement_state = "parked"
        state.parked_slot_id = slot_id
        state.candidate_slot_id = None
        state.candidate_since = None
        state.outside_since = None
        self._pending_release.pop(slot_id, None)
        self._event("vehicle_stopped_in_slot", global_id=global_id, slot_id=slot_id, overlap=round(overlap, 4))
        if not binding.vision_occupied:
            self._event("tracking_occupied_override", global_id=global_id, slot_id=slot_id)
        print(f"  [slot-park] Global #{global_id} dß╗½ng trong {slot_id} ({stopped_ms} ms)")
        self._sync_result(binding)

    def _release_vehicle(self, global_id: int, frame_idx: int, reason: str = "left_roi") -> None:
        slot_id = self._vehicle_to_slot.pop(global_id, None)
        if slot_id is None:
            return
        binding = self._bindings.get(slot_id)
        if binding is not None:
            binding.vehicle_id = None
            binding.tracking_occupied = False
            binding.vehicle_overlap = 0.0
            binding.stopped_for_ms = 0
            binding.tracking_state = "moving"
            self._sync_result(binding)
        self._pending_release[slot_id] = (global_id, frame_idx)
        state = self._vehicle_states.get(global_id)
        if state is not None:
            state.movement_state = "moving"
            state.parked_slot_id = None
            state.outside_since = None
        self._event("vehicle_left_slot", global_id=global_id, slot_id=slot_id, reason=reason)
        print(f"  ≡ƒÜù Global #{global_id} rß╗¥i {slot_id}; trß║ú trß║íng th├íi vß╗ü vision")

    def update_tracks(
        self,
        active_global_tracks: Dict[int, object],
        frame_idx: int,
        timestamp_s: float,
        camera_id: Optional[str] = None,
        coordinate_offset: Point = (0.0, 0.0),
    ) -> None:
        """Update motion/slot state from canonical global tracks every frame."""
        self._last_frame_idx = int(frame_idx)
        self._last_timestamp_s = float(timestamp_s)

        normalized: Dict[int, object] = {}
        for raw_id, track in active_global_tracks.items():
            global_id = int(raw_id)
            bbox = self._track_bbox(track, coordinate_offset)
            appearance = self._track_value(track, "appearance")
            normalized[global_id] = {"bbox": bbox, "appearance": appearance, "camera_id": camera_id}

        matches = self._batch_match(normalized)
        max_history_seconds = max(2.0, self.stop_seconds * 2.5)

        for global_id, track in normalized.items():
            state = self._vehicle_states.setdefault(global_id, VehicleParkingState(global_id=global_id))
            bbox = self._track_bbox(track)
            center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
            match = matches.get(global_id)
            slot_id = match[0] if match else None
            overlap = match[1]["vehicle_overlap"] if match else 0.0
            state.observations.append(VehicleObservation(float(timestamp_s), center, bbox, slot_id, overlap))
            while state.observations and timestamp_s - state.observations[0].timestamp_s > max_history_seconds:
                state.observations.popleft()
            if track.get("appearance") is not None:
                state.last_appearance = track["appearance"]

            if state.parked_slot_id is not None:
                parked_slot = state.parked_slot_id
                binding = self._bindings.get(parked_slot)
                if slot_id == parked_slot:
                    state.movement_state = "parked"
                    state.outside_since = None
                    if binding is not None:
                        binding.vehicle_overlap = overlap
                        binding.tracking_state = "parked"
                        self._sync_result(binding)
                else:
                    if state.outside_since is None:
                        state.outside_since = float(timestamp_s)
                        state.movement_state = "exit_pending"
                        if binding is not None:
                            binding.tracking_state = "exit_pending"
                    elif timestamp_s - state.outside_since >= self.exit_seconds:
                        self._release_vehicle(global_id, frame_idx)
                continue

            if slot_id is None:
                if state.candidate_slot_id is not None:
                    self._event(
                        "slot_candidate_cancelled",
                        global_id=global_id,
                        slot_id=state.candidate_slot_id,
                        reason="insufficient_overlap",
                    )
                state.movement_state = "moving"
                state.candidate_slot_id = None
                state.candidate_since = None
                continue

            binding = self._bindings[slot_id]
            if binding.vehicle_id is not None and binding.vehicle_id != global_id:
                self._event(
                    "slot_assignment_rejected",
                    global_id=global_id,
                    slot_id=slot_id,
                    reason="competing_vehicle",
                )
                continue

            if state.candidate_slot_id != slot_id:
                if state.candidate_slot_id is not None:
                    self._event(
                        "slot_candidate_cancelled",
                        global_id=global_id,
                        slot_id=state.candidate_slot_id,
                        reason="competing_slot",
                    )
                state.candidate_slot_id = slot_id
                state.candidate_since = float(timestamp_s)
                state.movement_state = "stop_candidate"
                self._event("slot_candidate_started", global_id=global_id, slot_id=slot_id)

            stopped, stopped_ms = self._is_stopped(state, slot_id, float(timestamp_s))
            binding.tracking_state = "stop_candidate"
            binding.vehicle_overlap = overlap
            binding.stopped_for_ms = stopped_ms
            self._sync_result(binding)
            candidate_age = (
                timestamp_s - state.candidate_since
                if state.candidate_since is not None
                else 0.0
            )
            if stopped and candidate_age >= self.stop_seconds + self.stop_commit_grace_seconds:
                self._bind_vehicle(global_id, slot_id, frame_idx, overlap, stopped_ms)
            elif stopped_ms >= int(self.stop_seconds * 900):
                if state.last_rejection_at is None or timestamp_s - state.last_rejection_at >= 1.0:
                    self._event(
                        "slot_assignment_rejected",
                        global_id=global_id,
                        slot_id=slot_id,
                        reason="unstable_position",
                    )
                    state.last_rejection_at = float(timestamp_s)

        self._cleanup_pending(frame_idx)

    def update_vision(
        self,
        slot_results: list,
        frame_idx: int,
        timestamp_s: float,
        camera_id: Optional[str] = None,
        coordinate_offset: Point = (0.0, 0.0),
    ) -> None:
        """Store unmodified detector evidence, then apply the tracking OR override."""
        self._last_frame_idx = int(frame_idx)
        self._last_timestamp_s = float(timestamp_s)
        for result in slot_results:
            slot_id = str(result.slot_id)
            polygon = self._offset_polygon(result.polygon, coordinate_offset)
            center = (
                float(result.center[0]) + float(coordinate_offset[0]),
                float(result.center[1]) + float(coordinate_offset[1]),
            )
            binding = self._bindings.get(slot_id)
            if binding is None:
                binding = SlotBinding(slot_id=slot_id)
                self._bindings[slot_id] = binding
            binding.camera_id = camera_id
            binding.vision_occupied = bool(result.occupied)
            binding.polygon = polygon
            binding.center = center
            binding.result_ref = result
            self._sync_result(binding)
        self._cleanup_pending(frame_idx)

    def update(self, active_tracks: dict, slot_results: list, frame_idx: int) -> None:
        """Compatibility wrapper for the former low-frequency API."""
        timestamp_s = float(frame_idx) / 30.0
        self.update_vision(slot_results, frame_idx, timestamp_s)
        self.update_tracks(active_tracks, frame_idx, timestamp_s)

    def _cleanup_pending(self, frame_idx: int) -> None:
        expired = [
            slot_id
            for slot_id, (_, released_at) in self._pending_release.items()
            if frame_idx - released_at > self.release_grace_frames
        ]
        for slot_id in expired:
            global_id, _ = self._pending_release.pop(slot_id)
            self._event("parked_id_recovery_expired", global_id=global_id, slot_id=slot_id)

    @staticmethod
    def _expanded_polygon(polygon: np.ndarray, ratio: float) -> np.ndarray:
        points = np.asarray(polygon, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        return centroid + (points - centroid) * (1.0 + ratio)

    def try_recover_id(
        self,
        position: Optional[Tuple[int, int]] = None,
        camera_id: Optional[str] = None,
        bbox: Optional[BBox] = None,
        appearance: Optional[np.ndarray] = None,
        coordinate_offset: Point = (0.0, 0.0),
    ) -> Optional[int]:
        """Recover the parked global ID before a new local/global ID is allocated."""
        if bbox is not None:
            x, y, w, h = bbox
            global_bbox = (x + coordinate_offset[0], y + coordinate_offset[1], w, h)
            point = (global_bbox[0] + global_bbox[2] / 2.0, global_bbox[1] + global_bbox[3] / 2.0)
        elif position is not None:
            global_bbox = None
            point = (position[0] + coordinate_offset[0], position[1] + coordinate_offset[1])
        else:
            return None

        candidates: List[Tuple[float, str, int]] = []
        for slot_id, binding in self._bindings.items():
            if binding.polygon is None or binding.vehicle_id is None:
                continue
            if camera_id is not None and binding.camera_id not in (None, camera_id):
                continue
            polygon = self._expanded_polygon(binding.polygon, self.recovery_expand_ratio)
            inside = self._point_in_polygon(point, polygon)
            overlap = 0.0
            if global_bbox is not None:
                temporary = SlotBinding(slot_id=slot_id, polygon=polygon, center=binding.center)
                geometry = self._overlap_geometry(global_bbox, temporary)
                overlap = geometry["vehicle_overlap"] if geometry else 0.0
            if not inside and overlap < self.min_vehicle_overlap:
                continue

            state = self._vehicle_states.get(binding.vehicle_id)
            if appearance is not None and state is not None and state.last_appearance is not None:
                appearance_distance = cv2.compareHist(
                    state.last_appearance.astype(np.float32),
                    appearance.astype(np.float32),
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                if appearance_distance > 0.50:
                    self._event(
                        "slot_assignment_rejected",
                        global_id=binding.vehicle_id,
                        slot_id=slot_id,
                        reason="appearance_mismatch",
                    )
                    continue
            distance = float(np.hypot(point[0] - binding.center[0], point[1] - binding.center[1]))
            candidates.append((distance, slot_id, int(binding.vehicle_id)))

        if not candidates:
            # Compatibility with a short post-release recovery window.
            for slot_id, (global_id, _) in self._pending_release.items():
                binding = self._bindings.get(slot_id)
                if binding is not None and binding.polygon is not None and self._point_in_polygon(point, binding.polygon):
                    candidates.append((0.0, slot_id, int(global_id)))
        if not candidates:
            return None

        _, slot_id, global_id = min(candidates)
        self._pending_release.pop(slot_id, None)
        state = self._vehicle_states.setdefault(global_id, VehicleParkingState(global_id=global_id))
        state.movement_state = "exit_pending"
        self._event("parked_id_recovered", global_id=global_id, slot_id=slot_id)
        print(f"  ≡ƒöä Track mß╗¢i tß╗½ {slot_id} nhß║¡n lß║íi Global #{global_id}")
        return global_id

    def resolve_pending_global_ids(self, active_global_ids: Set[int]) -> None:
        for slot_id, (vehicle_id, _) in list(self._pending_release.items()):
            if vehicle_id in active_global_ids:
                self._pending_release.pop(slot_id, None)

    def remap_vehicle_ids(self, canonicalize: Callable[[int], int]) -> None:
        """Move parked bindings/states to canonical IDs after a global-ID merge."""
        remapped_groups: Dict[int, List[SlotBinding]] = {}
        for binding in self._bindings.values():
            if binding.vehicle_id is None:
                continue
            old_id = int(binding.vehicle_id)
            new_id = int(canonicalize(old_id))
            if new_id != old_id:
                binding.vehicle_id = new_id
                old_state = self._vehicle_states.pop(old_id, None)
                if old_state is not None and new_id not in self._vehicle_states:
                    old_state.global_id = new_id
                    self._vehicle_states[new_id] = old_state
                self._event(
                    "parked_global_id_remapped",
                    old_global_id=old_id,
                    global_id=new_id,
                    slot_id=binding.slot_id,
                )
            remapped_groups.setdefault(new_id, []).append(binding)

        # A merge can reveal that two duplicate IDs had been assigned to two
        # neighbouring slots.  Keep exactly the strongest geometric binding.
        self._vehicle_to_slot.clear()
        for global_id, candidates in remapped_groups.items():
            winner = max(
                candidates,
                key=lambda item: (item.vehicle_overlap, item.stopped_for_ms, -item.bound_at_frame),
            )
            self._vehicle_to_slot[global_id] = winner.slot_id
            state = self._vehicle_states.get(global_id)
            if state is not None:
                state.parked_slot_id = winner.slot_id
            for binding in candidates:
                if binding is winner:
                    self._sync_result(binding)
                    continue
                binding.vehicle_id = None
                binding.tracking_occupied = False
                binding.tracking_state = "moving"
                binding.vehicle_overlap = 0.0
                binding.stopped_for_ms = 0
                self._event(
                    "slot_assignment_rejected",
                    global_id=global_id,
                    slot_id=binding.slot_id,
                    reason="global_id_merge_conflict",
                    kept_slot_id=winner.slot_id,
                )
                self._sync_result(binding)

    def _sync_result(self, binding: SlotBinding) -> None:
        binding.occupied = bool(binding.vision_occupied or binding.tracking_occupied)
        if binding.result_ref is not None:
            binding.result_ref.occupied = binding.occupied
            binding.result_ref.vehicle_id = binding.vehicle_id

    @staticmethod
    def _binding_to_json(binding: SlotBinding) -> dict:
        return {
            "occupied": bool(binding.occupied),
            "status": "occupied" if binding.occupied else "empty",
            "vehicle_id": binding.vehicle_id,
            "vision_occupied": bool(binding.vision_occupied),
            "tracking_occupied": bool(binding.tracking_occupied),
            "decision_source": binding.decision_source,
            "tracking_state": binding.tracking_state,
            "vehicle_overlap": round(float(binding.vehicle_overlap), 4),
            "stopped_for_ms": int(binding.stopped_for_ms),
        }

    def to_json(self, camera_id: Optional[str] = None) -> Dict[str, dict]:
        return {
            slot_id: self._binding_to_json(binding)
            for slot_id, binding in self._bindings.items()
            if camera_id is None or binding.camera_id == camera_id
        }
