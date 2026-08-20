"""Fuse parking-slot vision results with tracked global vehicle IDs.

``ParkingDetector`` remains the baseline source.  Tracking is deliberately a
one-way safety override: a stopped global vehicle can turn a visually-free slot
into occupied, but tracking never turns a visually-occupied slot into free.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Hashable, List, Optional, Set, Tuple

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
    fresh: bool = True


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
    last_bbox: Optional[BBox] = None
    last_rejection_at: Optional[float] = None
    parking_cooldown_until: float = 0.0
    strong_overlap_slot_id: Optional[str] = None
    strong_overlap_since: Optional[float] = None
    strong_overlap_last_seen: Optional[float] = None
    strong_overlap_last_frame: int = -1
    strong_overlap_samples: int = 0


@dataclass
class ArrivalClaim:
    """Short-lived evidence that one moving identity entered one slot.

    Motion-only trackers normally lose a vehicle as soon as it stops.  This
    claim preserves the last *fresh* observations until the slower parking
    detector has confirmed that the same slot became occupied.
    """

    slot_id: str
    global_id: int
    first_seen_s: float
    last_seen_s: float
    first_frame_idx: int
    last_frame_idx: int
    observations: int
    first_center: Point
    last_center: Point
    first_overlap: float
    max_overlap: float
    entered_from_outside: bool = False
    lost_at_s: Optional[float] = None
    last_bbox: Optional[BBox] = None
    last_appearance: Optional[np.ndarray] = None
    score: float = 0.0


@dataclass
class RecoveryCandidateEvidence:
    """Evidence accumulated for one local track against one departure token.

    A candidate is never allowed to consume a token from one noisy frame.  In
    particular, a shadow-shaped foreground blob can be protected from normal
    global-ID allocation while evidence is collected, but it cannot take the
    parked vehicle's ID without appearance, size and outward-motion support.
    """

    first_seen_s: float
    last_seen_s: float
    first_frame_idx: int
    last_frame_idx: int
    observations: int
    first_center: Point
    last_center: Point
    first_bbox: BBox
    last_bbox: BBox
    appearance_distances: Deque[float] = field(
        default_factory=lambda: deque(maxlen=5)
    )
    # All geometric/appearance gates passed while vision had not yet produced
    # two empty samples. Keep this evidence across a longer local-track gap so
    # a fast vehicle does not lose its parked identity before confirmation.
    qualified_predeparture: bool = False
    # The first fragment was physically inside/overlapping the owned slot.
    # Keep this weaker origin evidence longer than arbitrary noise: a fast
    # toy car can disappear after one frame and reappear well outside the ROI
    # before the 2 Hz vision detector confirms that the slot is empty.
    originated_in_slot: bool = False


@dataclass
class DepartureToken:
    """Time-bounded ownership of a Global ID after a vehicle leaves a slot."""

    slot_id: str
    global_id: int
    camera_id: Optional[str]
    created_at_s: float
    expires_at_s: float
    polygon: np.ndarray
    center: Point
    slot_diagonal: float
    last_bbox: Optional[BBox]
    last_appearance: Optional[np.ndarray]
    reason: str
    confirmed_empty: bool = False
    empty_observations: int = 0
    predeparture: bool = False
    candidates: Dict[Hashable, RecoveryCandidateEvidence] = field(default_factory=dict)


@dataclass
class RecoveryBatchResult:
    """Safe recovery decisions for one frame.

    ``protected_local_keys`` must be passed to the global-ID allocator so a
    plausible departure candidate is not assigned a new GID while the binder
    waits for a second observation.  Ambiguous/noisy candidates remain
    protected but are deliberately absent from ``recovered_ids``.
    """

    recovered_ids: Dict[Hashable, int] = field(default_factory=dict)
    protected_local_keys: Set[Hashable] = field(default_factory=set)
    ambiguous_local_keys: Set[Hashable] = field(default_factory=set)
    diagnostics: Dict[Hashable, dict] = field(default_factory=dict)


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
    decision_source: str = "none"
    vision_occupied_streak: int = 0
    vision_changed_at_s: float = 0.0


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
        policy: str = "legacy",
        recovery_retention_seconds: float = 5.0,
        recovery_initial_expand_ratio: float = 0.05,
        recovery_max_expand_ratio: float = 0.45,
        recovery_expand_seconds: float = 1.0,
        recovery_evidence_frames: int = 3,
        recovery_appearance_threshold: float = 0.55,
        recovery_relaxed_appearance_threshold: float = 0.70,
        recovery_ambiguity_margin: float = 0.15,
        recovery_size_ratio_range: Tuple[float, float] = (0.40, 2.50),
        recovery_relaxed_slot_overlap: float = 0.25,
        false_empty_grace_seconds: float = 1.25,
        predeparture_guard_seconds: float = 0.75,
        recovery_min_movement_px: float = 3.0,
        recovery_min_outward_px: float = 1.5,
        recovery_min_radial_gain_px: float = 0.75,
        arrival_lookback_seconds: float = 1.5,
        arrival_min_samples: int = 3,
        arrival_vision_confirmations: int = 2,
        arrival_competitor_margin: float = 0.15,
        arrival_absence_seconds: float = 0.75,
        arrival_lost_commit_delay_seconds: float = 0.35,
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

        normalized_policy = str(policy).strip().lower()
        if normalized_policy not in {"legacy", "vision_primary"}:
            raise ValueError("policy must be 'legacy' or 'vision_primary'")
        self.policy = normalized_policy
        self.recovery_retention_seconds = max(0.25, float(recovery_retention_seconds))
        self.recovery_initial_expand_ratio = max(0.0, float(recovery_initial_expand_ratio))
        self.recovery_max_expand_ratio = max(
            self.recovery_initial_expand_ratio,
            float(recovery_max_expand_ratio),
        )
        self.recovery_expand_seconds = max(0.05, float(recovery_expand_seconds))
        self.recovery_evidence_frames = max(2, int(recovery_evidence_frames))
        self.recovery_appearance_threshold = max(0.0, float(recovery_appearance_threshold))
        self.recovery_relaxed_appearance_threshold = max(
            self.recovery_appearance_threshold,
            float(recovery_relaxed_appearance_threshold),
        )
        self.recovery_ambiguity_margin = max(0.0, float(recovery_ambiguity_margin))
        self.recovery_size_ratio_min = max(0.01, float(recovery_size_ratio_range[0]))
        self.recovery_size_ratio_max = max(
            self.recovery_size_ratio_min,
            float(recovery_size_ratio_range[1]),
        )
        self.recovery_relaxed_slot_overlap = max(0.0, float(recovery_relaxed_slot_overlap))
        self.false_empty_grace_seconds = max(0.0, float(false_empty_grace_seconds))
        self.predeparture_guard_seconds = max(0.05, float(predeparture_guard_seconds))
        self.recovery_min_movement_px = max(0.0, float(recovery_min_movement_px))
        self.recovery_min_outward_px = max(0.0, float(recovery_min_outward_px))
        self.recovery_min_radial_gain_px = max(
            0.0,
            float(recovery_min_radial_gain_px),
        )
        self.arrival_lookback_seconds = max(0.25, float(arrival_lookback_seconds))
        self.arrival_min_samples = max(2, int(arrival_min_samples))
        self.arrival_vision_confirmations = max(
            1, int(arrival_vision_confirmations)
        )
        self.arrival_competitor_margin = max(
            0.0, float(arrival_competitor_margin)
        )
        self.arrival_absence_seconds = max(0.05, float(arrival_absence_seconds))
        self.arrival_lost_commit_delay_seconds = max(
            0.0, float(arrival_lost_commit_delay_seconds)
        )

        self._bindings: Dict[str, SlotBinding] = {}
        self._vehicle_to_slot: Dict[int, str] = {}
        self._vehicle_states: Dict[int, VehicleParkingState] = {}
        self._pending_release: Dict[str, Tuple[int, int]] = {}
        self._departure_tokens: Dict[str, DepartureToken] = {}
        self._arrival_claims: Dict[Tuple[str, int], ArrivalClaim] = {}
        self._events: Deque[dict] = deque(maxlen=500)
        self._last_frame_idx = 0
        self._last_timestamp_s = 0.0

    @property
    def bindings(self) -> Dict[str, SlotBinding]:
        return dict(self._bindings)

    @property
    def events(self) -> List[dict]:
        return list(self._events)

    @property
    def active_departure_tokens(self) -> Dict[str, DepartureToken]:
        """Return a shallow copy for diagnostics without exposing ownership."""
        return dict(self._departure_tokens)

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

    def get_identity_reservations(self) -> List[dict]:
        """Return durable parking ownership for the global ID coordinator."""
        reservations: Dict[int, dict] = {}
        for binding in self._bindings.values():
            if binding.vehicle_id is None:
                continue
            global_id = int(binding.vehicle_id)
            state = self._vehicle_states.get(global_id)
            reservations[global_id] = {
                "global_id": global_id,
                "slot_id": binding.slot_id,
                "camera_id": binding.camera_id,
                "state": "parked",
                "bbox": tuple(state.last_bbox) if state and state.last_bbox else None,
                "appearance": self._copy_appearance(
                    state.last_appearance if state is not None else None
                ),
            }
        for token in self._departure_tokens.values():
            global_id = int(token.global_id)
            reservations.setdefault(
                global_id,
                {
                    "global_id": global_id,
                    "slot_id": token.slot_id,
                    "camera_id": token.camera_id,
                    "state": "recovery_pending",
                    "bbox": tuple(token.last_bbox) if token.last_bbox else None,
                    "appearance": self._copy_appearance(token.last_appearance),
                },
            )
        return list(reservations.values())

    def prepare_predeparture_tokens(
        self,
        unbound_tracks: Dict[Hashable, object],
        timestamp_s: float,
        camera_id: Optional[str] = None,
    ) -> Set[Hashable]:
        """Open provisional tokens before occupied vision turns empty.

        A nearby unbound fragment is only protected here.  It still needs
        three outward observations, appearance/size agreement, and confirmed
        empty vision before it can consume the parked Global ID.
        """
        self._last_timestamp_s = float(timestamp_s)
        protected: Set[Hashable] = set()
        for key, track in unbound_tracks.items():
            bbox = self._optional_bbox(track, "recovery_bbox") or self._track_bbox(track)
            point = self._optional_point(track, "recovery_position") or self._bottom_center(bbox)
            choices = []
            for binding in self._bindings.values():
                if (
                    binding.vehicle_id is None
                    or not binding.vision_occupied
                    or binding.polygon is None
                    or binding.camera_id not in (None, camera_id)
                ):
                    continue
                diagonal = self._slot_diagonal(binding.polygon)
                signed = self._signed_polygon_distance(point, binding.polygon)
                overlap = self._raw_vehicle_overlap(bbox, binding.polygon)
                initial_radius = diagonal * self.recovery_initial_expand_ratio
                if signed < -initial_radius and overlap <= 0.0:
                    continue
                distance = float(np.linalg.norm(np.subtract(point, binding.center)))
                choices.append((distance / diagonal, binding))
            if not choices:
                continue
            choices.sort(key=lambda item: item[0])
            if len(choices) > 1 and choices[1][0] - choices[0][0] < 0.15:
                continue
            binding = choices[0][1]
            existing = self._departure_tokens.get(binding.slot_id)
            if existing is None:
                self._create_departure_token(
                    binding,
                    int(binding.vehicle_id),
                    float(timestamp_s),
                    reason="motion_started_in_parked_slot",
                    confirmed_empty=False,
                    predeparture=True,
                )
                self._event(
                    "predeparture_token_opened",
                    global_id=int(binding.vehicle_id),
                    slot_id=binding.slot_id,
                    local_key=repr(key),
                )
            protected.add(key)
        return protected

    def _event(self, event_type: str, **details) -> None:
        event = {
            "type": event_type,
            "frame": self._last_frame_idx,
            "timestamp_s": round(self._last_timestamp_s, 3),
            **details,
        }
        self._events.append(event)

    @staticmethod
    def _copy_appearance(appearance: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if appearance is None:
            return None
        value = np.asarray(appearance, dtype=np.float32)
        if value.size == 0 or not np.all(np.isfinite(value)):
            return None
        return value.copy()

    @staticmethod
    def _bottom_center(bbox: BBox) -> Point:
        return (float(bbox[0] + bbox[2] / 2.0), float(bbox[1] + bbox[3]))

    @staticmethod
    def _slot_diagonal(polygon: np.ndarray) -> float:
        _, _, width, height = cv2.boundingRect(np.asarray(polygon, dtype=np.float32))
        return max(1.0, float(np.hypot(width, height)))

    @staticmethod
    def _signed_polygon_distance(point: Point, polygon: np.ndarray) -> float:
        return float(
            cv2.pointPolygonTest(
                np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2)),
                (float(point[0]), float(point[1])),
                measureDist=True,
            )
        )

    @staticmethod
    def _raw_vehicle_overlap(bbox: BBox, polygon: np.ndarray) -> float:
        vehicle_polygon = SlotVehicleBinder._bbox_polygon(bbox)
        try:
            intersection_area, _ = cv2.intersectConvexConvex(
                vehicle_polygon.astype(np.float32),
                np.asarray(polygon, dtype=np.float32),
            )
        except cv2.error:
            return 0.0
        return float(intersection_area) / max(1.0, float(bbox[2] * bbox[3]))

    def _token_radius(self, token: DepartureToken, timestamp_s: float) -> float:
        age = max(0.0, float(timestamp_s) - token.created_at_s)
        progress = min(1.0, age / self.recovery_expand_seconds)
        ratio = self.recovery_initial_expand_ratio + progress * (
            self.recovery_max_expand_ratio - self.recovery_initial_expand_ratio
        )
        return float(token.slot_diagonal * ratio)

    def export_recovery_tokens(self, timestamp_s: Optional[float] = None) -> List[dict]:
        """Export token geometry for cross-camera/world-coordinate runners.

        The returned polygon is a copy.  A caller may transform candidate
        bboxes into this coordinate system and pass them back as
        ``recovery_bbox`` with ``allow_cross_camera=True``.
        """
        now_s = self._last_timestamp_s if timestamp_s is None else float(timestamp_s)
        self._cleanup_tokens(now_s)
        exported: List[dict] = []
        for token in sorted(self._departure_tokens.values(), key=lambda item: item.slot_id):
            exported.append(
                {
                    "slot_id": token.slot_id,
                    "global_id": int(token.global_id),
                    "camera_id": token.camera_id,
                    "created_at_s": float(token.created_at_s),
                    "expires_at_s": float(token.expires_at_s),
                    "age_ms": int(max(0.0, now_s - token.created_at_s) * 1000),
                    "remaining_ms": int(max(0.0, token.expires_at_s - now_s) * 1000),
                    "confirmed_empty": bool(token.confirmed_empty),
                    "predeparture": bool(token.predeparture),
                    "recovery_radius_px": round(self._token_radius(token, now_s), 3),
                    "polygon": token.polygon.copy(),
                    "center": tuple(token.center),
                    "last_bbox": tuple(token.last_bbox) if token.last_bbox is not None else None,
                    "continuation_evidence": [
                        {
                            "local_key": key,
                            "last_center": tuple(evidence.last_center),
                            "last_seen_s": float(evidence.last_seen_s),
                            "qualified_predeparture": bool(
                                evidence.qualified_predeparture
                            ),
                            "originated_in_slot": bool(
                                evidence.originated_in_slot
                            ),
                        }
                        for key, evidence in token.candidates.items()
                        if evidence.qualified_predeparture
                        or evidence.originated_in_slot
                    ],
                }
            )
        return exported

    def recovery_candidate_keys(self) -> Set[Hashable]:
        return {
            key
            for token in self._departure_tokens.values()
            for key in token.candidates
        }

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

    def _cleanup_arrival_claims(self, timestamp_s: float) -> None:
        cutoff = float(timestamp_s) - self.arrival_lookback_seconds
        for key, claim in list(self._arrival_claims.items()):
            reference_time = claim.lost_at_s or claim.last_seen_s
            if reference_time < cutoff:
                self._arrival_claims.pop(key, None)
                self._event(
                    "slot_arrival_claim_rejected",
                    global_id=claim.global_id,
                    slot_id=claim.slot_id,
                    reason="arrival_claim_expired",
                    observations=claim.observations,
                    max_overlap=round(claim.max_overlap, 4),
                )

    def _best_arrival_slot(self, bbox: BBox) -> Optional[Tuple[str, float]]:
        center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        candidates = []
        for binding in self._bindings.values():
            if binding.polygon is None or binding.vehicle_id is not None:
                continue
            overlap = self._raw_vehicle_overlap(bbox, binding.polygon)
            diagonal = self._slot_diagonal(binding.polygon)
            signed = self._signed_polygon_distance(center, binding.polygon)
            if overlap < 0.08 and signed < -0.15 * diagonal:
                continue
            center_distance = float(np.linalg.norm(np.subtract(center, binding.center)))
            score = overlap + 0.15 * max(0.0, 1.0 - center_distance / diagonal)
            candidates.append((score, binding.slot_id, overlap))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.08:
            return None
        return candidates[0][1], float(candidates[0][2])

    def _mark_arrival_absent(
        self,
        global_id: int,
        frame_idx: int,
        timestamp_s: float,
    ) -> None:
        for key, claim in list(self._arrival_claims.items()):
            if claim.global_id != int(global_id) or claim.lost_at_s is not None:
                continue
            if float(timestamp_s) - claim.last_seen_s < self.arrival_absence_seconds:
                continue
            # Leaving the ROI while the same track is still observable proves
            # a drive-through, not a parked vehicle. Only the explicit
            # `notify_track_lost()` transition may convert disappearance into
            # stopping evidence for a motion-only tracker. Treating ordinary
            # ROI absence as LOST caused a passing car to be reserved in C01,
            # suspended its real track, then forced a new GID downstream.
            self._arrival_claims.pop(key, None)
            self._event(
                "slot_arrival_claim_rejected",
                global_id=claim.global_id,
                slot_id=claim.slot_id,
                reason="vehicle_left_roi_before_track_lost",
                observations=claim.observations,
                max_overlap=round(claim.max_overlap, 4),
            )

    def _record_arrival_claim(
        self,
        global_id: int,
        slot_id: str,
        frame_idx: int,
        timestamp_s: float,
        bbox: BBox,
        overlap: float,
        appearance: Optional[np.ndarray],
        previous: Optional[VehicleObservation],
    ) -> None:
        binding = self._bindings.get(slot_id)
        if binding is None or binding.polygon is None:
            return
        key = (slot_id, int(global_id))
        center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        claim = self._arrival_claims.get(key)
        reset = (
            claim is None
            or float(timestamp_s) - claim.last_seen_s > 0.50
            or int(frame_idx) - claim.last_frame_idx > 6
        )
        entered_from_outside = bool(
            previous is not None
            and (previous.slot_id != slot_id or previous.overlap < self.min_vehicle_overlap)
        )
        if reset:
            claim = ArrivalClaim(
                slot_id=slot_id,
                global_id=int(global_id),
                first_seen_s=float(timestamp_s),
                last_seen_s=float(timestamp_s),
                first_frame_idx=int(frame_idx),
                last_frame_idx=int(frame_idx),
                observations=1,
                first_center=center,
                last_center=center,
                first_overlap=float(overlap),
                max_overlap=float(overlap),
                entered_from_outside=entered_from_outside,
                last_bbox=tuple(bbox),
                last_appearance=self._copy_appearance(appearance),
            )
            self._arrival_claims[key] = claim
            self._event(
                "slot_arrival_claim_started",
                global_id=int(global_id),
                slot_id=slot_id,
                overlap=round(float(overlap), 4),
            )
        elif claim.last_frame_idx != int(frame_idx):
            if claim.lost_at_s is not None:
                claim.lost_at_s = None
                self._event(
                    "slot_arrival_claim_resumed",
                    global_id=claim.global_id,
                    slot_id=claim.slot_id,
                    reason="same_identity_observed_after_lost",
                )
            claim.observations += 1
            claim.last_seen_s = float(timestamp_s)
            claim.last_frame_idx = int(frame_idx)
            claim.last_center = center
            claim.max_overlap = max(claim.max_overlap, float(overlap))
            claim.entered_from_outside |= entered_from_outside
            claim.last_bbox = tuple(bbox)
            if appearance is not None:
                claim.last_appearance = self._copy_appearance(appearance)

        diagonal = self._slot_diagonal(binding.polygon)
        first_distance = float(np.linalg.norm(np.subtract(claim.first_center, binding.center)))
        current_distance = float(np.linalg.norm(np.subtract(center, binding.center)))
        inward_progress = max(0.0, first_distance - current_distance) / diagonal
        overlap_progress = max(0.0, claim.max_overlap - claim.first_overlap)
        claim.score = (
            0.55 * min(1.0, claim.max_overlap)
            + 0.20 * min(1.0, claim.observations / self.arrival_min_samples)
            + 0.15 * min(1.0, inward_progress / 0.10)
            + 0.10 * min(1.0, overlap_progress / 0.15)
        )

    def _try_commit_arrival_claim(
        self,
        slot_id: str,
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[int]:
        binding = self._bindings.get(slot_id)
        if (
            binding is None
            or not binding.vision_occupied
            or binding.vision_occupied_streak < self.arrival_vision_confirmations
            or binding.vehicle_id is not None
        ):
            return None
        self._cleanup_arrival_claims(timestamp_s)
        claims = [
            claim
            for (candidate_slot, _), claim in self._arrival_claims.items()
            if candidate_slot == slot_id
            and claim.lost_at_s is not None
            and float(timestamp_s) - claim.lost_at_s <= self.arrival_lookback_seconds
            and float(timestamp_s) - claim.lost_at_s
            >= self.arrival_lost_commit_delay_seconds
            and claim.observations >= self.arrival_min_samples
            and claim.max_overlap >= self.min_vehicle_overlap
        ]
        if not claims:
            return None
        ranked = sorted(claims, key=lambda item: item.score, reverse=True)
        winner = ranked[0]
        if len(ranked) > 1 and winner.score - ranked[1].score < self.arrival_competitor_margin:
            self._event(
                "slot_arrival_claim_rejected",
                global_id=winner.global_id,
                slot_id=slot_id,
                reason="competing_vehicle",
                competitor_global_id=ranked[1].global_id,
            )
            return None
        diagonal = self._slot_diagonal(binding.polygon)
        first_distance = float(np.linalg.norm(np.subtract(winner.first_center, binding.center)))
        last_distance = float(np.linalg.norm(np.subtract(winner.last_center, binding.center)))
        demonstrated_entry = (
            winner.entered_from_outside
            or winner.max_overlap - winner.first_overlap >= 0.10
            or first_distance - last_distance >= max(2.0, diagonal * 0.05)
        )
        if not demonstrated_entry:
            self._event(
                "slot_arrival_claim_rejected",
                global_id=winner.global_id,
                slot_id=slot_id,
                reason="no_inward_trajectory",
            )
            return None
        state = self._vehicle_states.setdefault(
            winner.global_id,
            VehicleParkingState(global_id=winner.global_id),
        )
        if winner.last_bbox is not None:
            state.last_bbox = tuple(winner.last_bbox)
        if winner.last_appearance is not None:
            state.last_appearance = self._copy_appearance(winner.last_appearance)
        self._bind_vehicle(
            winner.global_id,
            slot_id,
            frame_idx,
            winner.max_overlap,
            int(max(0.0, timestamp_s - winner.first_seen_s) * 1000),
        )
        self._event(
            "slot_arrival_claim_confirmed",
            global_id=winner.global_id,
            slot_id=slot_id,
            observations=winner.observations,
            vision_confirmations=binding.vision_occupied_streak,
        )
        for key, claim in list(self._arrival_claims.items()):
            if claim.slot_id == slot_id or claim.global_id == winner.global_id:
                self._arrival_claims.pop(key, None)
        return winner.global_id

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
        samples = [
            item
            for item in state.observations
            if item.timestamp_s >= window_start
            and (self.policy != "vision_primary" or item.fresh)
        ]
        if self.policy == "vision_primary" and samples:
            # Only the final contiguous fresh suffix is real stationary
            # evidence.  Fresh samples on either side of a long LOST interval
            # must never combine into an artificial one-second dwell.
            suffix_start = 0
            max_gap_s = self._strong_overlap_max_gap_seconds()
            for index in range(1, len(samples)):
                if samples[index].timestamp_s - samples[index - 1].timestamp_s > max_gap_s:
                    suffix_start = index
            if suffix_start:
                samples = samples[suffix_start:]
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

    @staticmethod
    def _reset_strong_overlap(state: VehicleParkingState) -> None:
        state.strong_overlap_slot_id = None
        state.strong_overlap_since = None
        state.strong_overlap_last_seen = None
        state.strong_overlap_last_frame = -1
        state.strong_overlap_samples = 0

    def _strong_overlap_max_gap_seconds(self) -> float:
        return max(
            0.25,
            2.0 * self.stop_seconds / max(self.min_stop_samples, 2),
        )

    def _advance_strong_overlap(
        self,
        state: VehicleParkingState,
        slot_id: str,
        frame_idx: int,
        timestamp_s: float,
    ) -> Tuple[bool, int]:
        """Advance one continuous strong-overlap dwell observation."""
        max_gap_s = self._strong_overlap_max_gap_seconds()
        discontinuous = (
            state.strong_overlap_slot_id != slot_id
            or state.strong_overlap_since is None
            or state.strong_overlap_last_seen is None
            or float(timestamp_s) < state.strong_overlap_last_seen
            or float(timestamp_s) - state.strong_overlap_last_seen > max_gap_s
        )
        if discontinuous:
            self._reset_strong_overlap(state)
            state.strong_overlap_slot_id = slot_id
            state.strong_overlap_since = float(timestamp_s)
            state.strong_overlap_last_seen = float(timestamp_s)
            state.strong_overlap_last_frame = int(frame_idx)
            state.strong_overlap_samples = 1
        elif state.strong_overlap_last_frame != int(frame_idx):
            state.strong_overlap_last_seen = float(timestamp_s)
            state.strong_overlap_last_frame = int(frame_idx)
            state.strong_overlap_samples += 1

        duration = max(0.0, float(timestamp_s) - float(state.strong_overlap_since))
        required_duration = self.stop_seconds + self.stop_commit_grace_seconds
        ready = (
            duration >= required_duration
            and state.strong_overlap_samples >= self.min_stop_samples
            and self._strong_overlap_is_stable(state, slot_id)
        )
        return ready, int(duration * 1000)

    def _strong_overlap_is_stable(
        self,
        state: VehicleParkingState,
        slot_id: str,
    ) -> bool:
        """Reject oscillating shadows while tolerating sparse centroid outliers."""
        if state.strong_overlap_since is None:
            return False
        samples = [
            item
            for item in state.observations
            if item.fresh
            and item.timestamp_s >= state.strong_overlap_since
            and item.slot_id == slot_id
            and item.overlap >= self.strong_vehicle_overlap
        ]
        if len(samples) < self.min_stop_samples:
            return False

        centers = np.asarray([item.center for item in samples], dtype=np.float64)
        diagonals = np.asarray(
            [float(np.hypot(item.bbox[2], item.bbox[3])) for item in samples],
            dtype=np.float64,
        )
        bbox_diagonal = max(1.0, float(np.median(diagonals)))
        stationary_radius = max(3.0, self.stationary_radius_ratio * bbox_diagonal)
        robust_center = np.median(centers, axis=0)
        distances = np.linalg.norm(centers - robust_center, axis=1)
        inlier_ratio = float(np.mean(distances <= stationary_radius))
        if inlier_ratio < 0.60:
            return False

        # Compare robust endpoints, not the literal first/last sample: a
        # single detector outlier at either edge must not reject a real car.
        endpoint_count = max(2, min(len(samples) // 3, 5))
        first_center = np.median(centers[:endpoint_count], axis=0)
        last_center = np.median(centers[-endpoint_count:], axis=0)
        robust_drift = float(np.linalg.norm(last_center - first_center))
        stationary_drift = max(5.0, self.stationary_drift_ratio * bbox_diagonal)
        return robust_drift <= stationary_drift

    def _create_departure_token(
        self,
        binding: SlotBinding,
        global_id: int,
        timestamp_s: float,
        *,
        reason: str,
        confirmed_empty: bool,
        predeparture: bool = False,
    ) -> DepartureToken:
        """Create or advance a token without ever discarding its GID on noise."""
        global_id = int(global_id)
        existing = self._departure_tokens.get(binding.slot_id)
        if existing is not None and existing.global_id == global_id:
            if confirmed_empty:
                first_raw_empty = existing.empty_observations == 0
                existing.empty_observations += 1
                if existing.predeparture and first_raw_empty:
                    existing.created_at_s = float(timestamp_s)
                    existing.expires_at_s = (
                        float(timestamp_s) + self.recovery_retention_seconds
                    )
                    # Keep only fresh predeparture evidence. It was collected
                    # from a fragment that started inside this parked slot and
                    # is exactly what lets a fast vehicle retain its GID before
                    # the slower vision detector reports the slot empty.
                    # Stale or intermittent shadow evidence is still removed.
                    existing.candidates = {
                        key: evidence
                        for key, evidence in existing.candidates.items()
                        if evidence.qualified_predeparture
                        or evidence.originated_in_slot
                        or float(timestamp_s) - evidence.last_seen_s <= 0.50
                    }
                existing.predeparture = False
                existing.confirmed_empty = existing.empty_observations >= 2
            return existing

        state = self._vehicle_states.get(global_id)
        latest = state.observations[-1] if state is not None and state.observations else None
        last_bbox = (
            tuple(state.last_bbox)
            if state is not None and state.last_bbox is not None
            else (tuple(latest.bbox) if latest is not None else None)
        )
        last_appearance = self._copy_appearance(
            state.last_appearance if state is not None else None
        )
        polygon = np.asarray(binding.polygon, dtype=np.float32).reshape((-1, 2)).copy()
        token = DepartureToken(
            slot_id=binding.slot_id,
            global_id=global_id,
            camera_id=binding.camera_id,
            created_at_s=float(timestamp_s),
            expires_at_s=float(timestamp_s) + self.recovery_retention_seconds,
            polygon=polygon,
            center=tuple(binding.center),
            slot_diagonal=self._slot_diagonal(polygon),
            last_bbox=last_bbox,
            last_appearance=last_appearance,
            reason=str(reason),
            confirmed_empty=False,
            empty_observations=1 if confirmed_empty else 0,
            predeparture=bool(predeparture),
        )
        self._departure_tokens[binding.slot_id] = token
        self._event(
            "departure_token_created",
            global_id=global_id,
            slot_id=binding.slot_id,
            reason=reason,
            provisional=True,
            expires_in_ms=int(self.recovery_retention_seconds * 1000),
        )
        return token

    def _detach_binding_for_departure(
        self,
        binding: SlotBinding,
        token: DepartureToken,
        timestamp_s: float,
    ) -> None:
        global_id = int(token.global_id)
        if self._vehicle_to_slot.get(global_id) == binding.slot_id:
            self._vehicle_to_slot.pop(global_id, None)
        binding.vehicle_id = None
        binding.tracking_occupied = False
        binding.tracking_state = "recovery_pending"
        binding.vehicle_overlap = 0.0
        binding.stopped_for_ms = 0
        state = self._vehicle_states.get(global_id)
        if state is not None:
            state.movement_state = "exit_pending"
            state.parked_slot_id = None
            state.candidate_slot_id = None
            state.candidate_since = None
            state.outside_since = None
            state.parking_cooldown_until = float(timestamp_s) + self.predeparture_guard_seconds
            self._reset_strong_overlap(state)
            # Preserve the descriptor in the token, but do not allow the old
            # stationary samples to auto-park the departing identity again.
            state.observations.clear()

    def _restore_false_empty_token(self, token: DepartureToken) -> None:
        binding = self._bindings.get(token.slot_id)
        if binding is None or binding.vehicle_id not in (None, token.global_id):
            return
        binding.vehicle_id = int(token.global_id)
        binding.tracking_occupied = False
        binding.tracking_state = "parked"
        self._vehicle_to_slot[int(token.global_id)] = token.slot_id
        state = self._vehicle_states.setdefault(
            int(token.global_id),
            VehicleParkingState(global_id=int(token.global_id)),
        )
        state.movement_state = "parked"
        state.parked_slot_id = token.slot_id
        state.candidate_slot_id = None
        state.candidate_since = None
        state.outside_since = None
        self._reset_strong_overlap(state)
        recent_candidate = any(
            self._last_timestamp_s - evidence.last_seen_s
            <= self.false_empty_grace_seconds
            for evidence in token.candidates.values()
        )
        if recent_candidate:
            # A single occupied result in the middle of a real departure is
            # common when the moving vehicle still partly covers the ROI.
            # Restore the public slot owner but keep the token and its recent
            # trajectory; otherwise the next fragment receives a new GID.
            token.confirmed_empty = False
            token.empty_observations = 0
            token.predeparture = True
            token.created_at_s = float(self._last_timestamp_s)
            token.expires_at_s = (
                float(self._last_timestamp_s)
                + self.recovery_retention_seconds
            )
            self._event(
                "departure_token_rearmed_after_vision_rebound",
                global_id=token.global_id,
                slot_id=token.slot_id,
                candidate_count=len(token.candidates),
            )
            return
        self._departure_tokens.pop(token.slot_id, None)
        self._event(
            "departure_token_cancelled",
            global_id=token.global_id,
            slot_id=token.slot_id,
            reason="false_empty",
        )

    def _cleanup_tokens(self, timestamp_s: float) -> None:
        now_s = float(timestamp_s)
        for slot_id, token in list(self._departure_tokens.items()):
            if (
                token.predeparture
                and not token.confirmed_empty
                and token.empty_observations == 0
                and now_s - token.created_at_s > self.predeparture_guard_seconds
            ):
                self._departure_tokens.pop(slot_id, None)
                self._event(
                    "departure_token_cancelled",
                    global_id=token.global_id,
                    slot_id=slot_id,
                    reason="predeparture_guard_expired",
                )
                continue
            if now_s <= token.expires_at_s:
                # Forget candidates that disappeared; their old one-frame
                # evidence must never be reused by a later blob with the same
                # local ID.
                for key, evidence in list(token.candidates.items()):
                    evidence_ttl = (
                        self.recovery_retention_seconds
                        if (
                            evidence.qualified_predeparture
                            or evidence.originated_in_slot
                        )
                        else 0.75
                    )
                    if now_s - evidence.last_seen_s > evidence_ttl:
                        token.candidates.pop(key, None)
                continue
            self._departure_tokens.pop(slot_id, None)
            self._event(
                "parked_id_recovery_expired",
                global_id=token.global_id,
                slot_id=slot_id,
                retained_in_global_gallery=True,
            )

    def _bind_vehicle(self, global_id: int, slot_id: str, frame_idx: int, overlap: float, stopped_ms: int) -> None:
        old_slot = self._vehicle_to_slot.get(global_id)
        if old_slot is not None and old_slot != slot_id:
            self._release_vehicle(global_id, frame_idx, reason="reassigned")
        binding = self._bindings[slot_id]
        
        # CHỐNG CƯỚP SLOT (Anti ID-switch):
        # Nếu ô đã có ID (và đang bị khoá bởi Vision), không cho phép xe khác cướp!
        if binding.vehicle_id is not None and binding.vehicle_id != global_id:
            # Nếu xe cũ vẫn còn nằm đây (sticky ID), từ chối xe mới
            if binding.vision_occupied:
                self._event(
                    "slot_assignment_rejected",
                    global_id=global_id,
                    slot_id=slot_id,
                    reason="sticky_id_conflict",
                )
                return
            else:
                self._event(
                    "slot_assignment_rejected",
                    global_id=global_id,
                    slot_id=slot_id,
                    reason="competing_vehicle",
                )
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
        self._reset_strong_overlap(state)
        for key, claim in list(self._arrival_claims.items()):
            if claim.global_id == int(global_id) or claim.slot_id == slot_id:
                self._arrival_claims.pop(key, None)
        self._pending_release.pop(slot_id, None)
        token = self._departure_tokens.get(slot_id)
        if token is not None and token.global_id == int(global_id):
            self._departure_tokens.pop(slot_id, None)
            self._event(
                "departure_token_cancelled",
                global_id=global_id,
                slot_id=slot_id,
                reason="identity_returned_to_slot",
            )
        self._event("vehicle_stopped_in_slot", global_id=global_id, slot_id=slot_id, overlap=round(overlap, 4))
        if not binding.vision_occupied:
            self._event("tracking_occupied_override", global_id=global_id, slot_id=slot_id)
        print(f"  [slot-park] Global #{global_id} dừng trong {slot_id} ({stopped_ms} ms)")
        self._sync_result(binding)

    def _release_vehicle(self, global_id: int, frame_idx: int, reason: str = "left_roi") -> None:
        slot_id = self._vehicle_to_slot.pop(global_id, None)
        if slot_id is None:
            return
        binding = self._bindings.get(slot_id)
        if self.policy == "vision_primary" and binding is not None:
            if reason != "reassigned" and binding.vision_occupied and binding.polygon is not None:
                self._create_departure_token(
                    binding,
                    global_id,
                    self._last_timestamp_s,
                    reason="tracking_exit",
                    confirmed_empty=False,
                    predeparture=True,
                )
            # Vision owns the public colour.  Keep the sticky slot ID only
            # until the first raw empty result creates/confirms the departure.
            binding.tracking_occupied = False
            binding.tracking_state = "exit_pending"
            if not binding.vision_occupied:
                binding.vehicle_id = None
                binding.vehicle_overlap = 0.0
                binding.stopped_for_ms = 0
            self._sync_result(binding)
            state = self._vehicle_states.get(global_id)
            if state is not None:
                state.movement_state = "exit_pending"
                state.parked_slot_id = None
                state.candidate_slot_id = None
                state.candidate_since = None
                state.outside_since = None
                state.parking_cooldown_until = (
                    self._last_timestamp_s + self.predeparture_guard_seconds
                )
                self._reset_strong_overlap(state)
            self._event("vehicle_left_slot", global_id=global_id, slot_id=slot_id, reason=reason)
            return
        if binding is not None:
            if binding.vision_occupied:
                # STICKY ID: Giữ lại vehicle_id, chỉ gỡ tracking
                binding.tracking_occupied = False
                binding.tracking_state = "moving"
                # Không gỡ vehicle_id, vehicle_overlap, stopped_for_ms
                self._sync_result(binding)
                self._event("vehicle_tracking_lost_but_sticky", global_id=global_id, slot_id=slot_id)
            else:
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
            self._reset_strong_overlap(state)
        self._event("vehicle_left_slot", global_id=global_id, slot_id=slot_id, reason=reason)
        print(f"  🚗 Global #{global_id} rời {slot_id}; trả trạng thái về vision")

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
            try:
                invisible_count = int(
                    self._track_value(track, "consecutive_invisible_count", 0)
                )
            except (TypeError, ValueError):
                invisible_count = 1
            normalized[global_id] = {
                "bbox": bbox,
                "appearance": appearance,
                "camera_id": camera_id,
                "fresh": invisible_count == 0,
            }

        for global_id, state in self._vehicle_states.items():
            if global_id not in normalized and state.strong_overlap_since is not None:
                self._reset_strong_overlap(state)

        matches = self._batch_match(normalized)
        max_history_seconds = max(2.0, self.stop_seconds * 2.5)
        self._cleanup_arrival_claims(timestamp_s)
        for claim_global_id in {
            claim.global_id for claim in self._arrival_claims.values()
        } - set(normalized):
            self._mark_arrival_absent(
                claim_global_id,
                frame_idx,
                timestamp_s,
            )

        for global_id, track in normalized.items():
            state = self._vehicle_states.setdefault(global_id, VehicleParkingState(global_id=global_id))
            bbox = self._track_bbox(track)
            fresh_observation = (
                bool(track.get("fresh", True))
                if self.policy == "vision_primary"
                else True
            )
            if fresh_observation:
                state.last_bbox = tuple(bbox)
            center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
            match = matches.get(global_id)
            slot_id = match[0] if match else None
            overlap = match[1]["vehicle_overlap"] if match else 0.0
            previous_observation = state.observations[-1] if state.observations else None
            state.observations.append(
                VehicleObservation(
                    float(timestamp_s),
                    center,
                    bbox,
                    slot_id,
                    overlap,
                    fresh=fresh_observation,
                )
            )
            while state.observations and timestamp_s - state.observations[0].timestamp_s > max_history_seconds:
                state.observations.popleft()
            if fresh_observation and track.get("appearance") is not None:
                state.last_appearance = track["appearance"]
            if fresh_observation and state.parked_slot_id is None:
                arrival = (
                    (slot_id, overlap)
                    if slot_id is not None
                    else self._best_arrival_slot(bbox)
                )
                if arrival is not None:
                    self._record_arrival_claim(
                        global_id,
                        arrival[0],
                        frame_idx,
                        timestamp_s,
                        bbox,
                        arrival[1],
                        track.get("appearance"),
                        previous_observation,
                    )
                else:
                    self._mark_arrival_absent(
                        global_id,
                        frame_idx,
                        timestamp_s,
                    )

            if self.policy == "vision_primary" and not fresh_observation:
                # LOST/predicted bboxes are useful for a very short continuity
                # gap only.  They cannot move the dwell endpoint, add samples,
                # start a candidate, or trigger either parking commit path.
                if (
                    state.strong_overlap_last_seen is not None
                    and float(timestamp_s) - state.strong_overlap_last_seen
                    > self._strong_overlap_max_gap_seconds()
                ):
                    self._reset_strong_overlap(state)
                continue

            if state.parked_slot_id is not None:
                self._reset_strong_overlap(state)
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
                self._reset_strong_overlap(state)
                continue

            binding = self._bindings[slot_id]
            if (
                self.policy == "vision_primary"
                and (
                    not binding.vision_occupied
                    or float(timestamp_s) < state.parking_cooldown_until
                )
            ):
                state.movement_state = "moving"
                state.candidate_slot_id = None
                state.candidate_since = None
                self._reset_strong_overlap(state)
                binding.tracking_state = (
                    "recovery_pending"
                    if slot_id in self._departure_tokens
                    else "moving"
                )
                self._sync_result(binding)
                continue
            if binding.vehicle_id is not None and binding.vehicle_id != global_id:
                self._reset_strong_overlap(state)
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
                self._reset_strong_overlap(state)
                self._event("slot_candidate_started", global_id=global_id, slot_id=slot_id)

            strong_ready = False
            strong_dwell_ms = 0
            if self.policy == "vision_primary" and overlap >= self.strong_vehicle_overlap:
                strong_ready, strong_dwell_ms = self._advance_strong_overlap(
                    state,
                    slot_id,
                    frame_idx,
                    timestamp_s,
                )
            else:
                self._reset_strong_overlap(state)

            stopped, stopped_ms = self._is_stopped(state, slot_id, float(timestamp_s))
            binding.tracking_state = "stop_candidate"
            binding.vehicle_overlap = overlap
            binding.stopped_for_ms = max(stopped_ms, strong_dwell_ms)
            self._sync_result(binding)
            candidate_age = (
                timestamp_s - state.candidate_since
                if state.candidate_since is not None
                else 0.0
            )
            if strong_ready:
                self._event(
                    "strong_overlap_dwell_confirmed",
                    global_id=global_id,
                    slot_id=slot_id,
                    overlap=round(overlap, 4),
                    dwell_ms=strong_dwell_ms,
                )
                self._bind_vehicle(
                    global_id,
                    slot_id,
                    frame_idx,
                    overlap,
                    strong_dwell_ms,
                )
            elif stopped and candidate_age >= self.stop_seconds + self.stop_commit_grace_seconds:
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
        self._cleanup_tokens(timestamp_s)

    def notify_track_lost(
        self,
        global_id: int,
        frame_idx: int,
        timestamp_s: float,
    ) -> Optional[int]:
        """Use the first LOST transition as evidence that a car stopped.

        Expiry happens roughly ninety frames later and is too late to protect
        the identity.  A claim still needs independent occupied-vision and
        inward-trajectory evidence before it can own the slot.
        """
        self._last_frame_idx = int(frame_idx)
        self._last_timestamp_s = float(timestamp_s)
        global_id = int(global_id)
        candidates = [
            claim
            for (_slot_id, claim_gid), claim in self._arrival_claims.items()
            if claim_gid == global_id
            and float(timestamp_s) - claim.last_seen_s <= self.arrival_lookback_seconds
        ]
        if not candidates:
            return None
        winner = max(candidates, key=lambda item: (item.last_seen_s, item.score))
        winner.lost_at_s = float(timestamp_s)
        return self._try_commit_arrival_claim(
            winner.slot_id,
            int(frame_idx),
            float(timestamp_s),
        )

    def notify_track_expired(
        self,
        global_id: int,
        frame_idx: int,
        timestamp_s: float,
    ) -> None:
        """Called when the motion tracker permanently loses a track.

        If the vehicle was a stop_candidate with enough overlap when it
        disappeared, auto-commit the parking assignment.  This handles
        motion-based trackers which cannot observe stationary vehicles.
        """
        state = self._vehicle_states.get(global_id)
        if state is None:
            return

        # Đã park rồi — không cần làm gì
        if state.parked_slot_id is not None:
            return

        slot_id = state.candidate_slot_id
        if slot_id is None:
            return

        binding = self._bindings.get(slot_id)
        if binding is None:
            return

        if self.policy == "vision_primary":
            # Track expiry is weak evidence.  An unbound vision-primary track
            # necessarily failed the full stationary/strong-overlap dwell, so
            # expiry itself must not shortcut that requirement.  Legacy keeps
            # its historical auto-commit behavior below.
            state.movement_state = "moving"
            state.candidate_slot_id = None
            state.candidate_since = None
            self._reset_strong_overlap(state)
            binding.tracking_state = (
                "recovery_pending"
                if slot_id in self._departure_tokens
                else "moving"
            )
            self._event(
                "slot_assignment_rejected",
                global_id=global_id,
                slot_id=slot_id,
                reason=(
                    "vision_primary_track_expired_before_dwell"
                    if binding.vision_occupied
                    else "vision_primary_slot_empty"
                ),
            )
            self._sync_result(binding)
            return

        # Chỉ auto-commit nếu overlap đủ tốt
        if binding.vehicle_overlap < self.min_vehicle_overlap:
            return

        # Chỉ auto-commit nếu đã là stop_candidate đủ lâu (>= 50% stop_seconds)
        candidate_age = (
            timestamp_s - state.candidate_since
            if state.candidate_since is not None
            else 0.0
        )
        if candidate_age < self.stop_seconds * 0.50:
            return

        # Auto-bind khi track biến mất mà xe vẫn nằm trong ô
        stopped_ms = int(candidate_age * 1000)
        self._bind_vehicle(global_id, slot_id, frame_idx, binding.vehicle_overlap, stopped_ms)
        self._event(
            "auto_parked_on_track_expired",
            global_id=global_id,
            slot_id=slot_id,
            overlap=round(binding.vehicle_overlap, 4),
            candidate_age_ms=stopped_ms,
        )
        print(f"  🅿️ Auto-park: GID #{global_id} → {slot_id} (track expired, overlap={binding.vehicle_overlap:.2f})")

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
            previous_vision_occupied = bool(binding.vision_occupied)
            binding.camera_id = camera_id
            binding.vision_occupied = bool(result.occupied)
            if binding.vision_occupied:
                binding.vision_occupied_streak = (
                    binding.vision_occupied_streak + 1
                    if previous_vision_occupied
                    else 1
                )
            else:
                binding.vision_occupied_streak = 0
            if previous_vision_occupied != binding.vision_occupied:
                binding.vision_changed_at_s = float(timestamp_s)
            binding.polygon = polygon
            binding.center = center
            binding.result_ref = result

            if self.policy == "vision_primary" and not binding.vision_occupied:
                for state in self._vehicle_states.values():
                    if state.strong_overlap_slot_id == slot_id:
                        self._reset_strong_overlap(state)

            if self.policy == "vision_primary":
                token = self._departure_tokens.get(slot_id)
                if not binding.vision_occupied:
                    departure_id = (
                        int(binding.vehicle_id)
                        if binding.vehicle_id is not None
                        else (int(token.global_id) if token is not None else None)
                    )
                    if departure_id is not None and binding.polygon is not None:
                        token = self._create_departure_token(
                            binding,
                            departure_id,
                            timestamp_s,
                            reason=(
                                "vision_became_empty"
                                if previous_vision_occupied
                                else "vision_empty_confirmation"
                            ),
                            confirmed_empty=True,
                        )
                        self._detach_binding_for_departure(binding, token, timestamp_s)
                        if token.confirmed_empty:
                            self._event(
                                "departure_token_confirmed",
                                global_id=token.global_id,
                                slot_id=slot_id,
                                empty_observations=token.empty_observations,
                            )
                elif token is not None:
                    # If occupied vision returns before the token is consumed,
                    # the empty interval was a detector flicker.  Restore the
                    # parked owner regardless of token age; a real departure
                    # consumes/cancels the token through verified tracking.
                    self._restore_false_empty_token(token)

                if binding.vision_occupied and binding.vehicle_id is None:
                    self._try_commit_arrival_claim(
                        slot_id,
                        int(frame_idx),
                        float(timestamp_s),
                    )

                # Raw vision owns the public colour.  Unlike the legacy
                # policy, no stale observation is auto-bound here.
                self._sync_result(binding)
                continue
            
            # NGAY KHI CÓ VISION OCCUPIED, TÌM TRACK ĐỂ RÀNG BUỘC (Tránh lệch pha do tracker xoá ID)
            if binding.vision_occupied and binding.vehicle_id is None:
                best_id = None
                best_overlap = self.min_vehicle_overlap
                
                for state in self._vehicle_states.values():
                    if not state.observations:
                        continue
                        
                    # Tính overlap giữa xe (quan sát cuối cùng) và ô đỗ
                    latest = state.observations[-1]
                    geometry = self._overlap_geometry(latest.bbox, binding)
                    
                    if geometry is not None and geometry["vehicle_overlap"] >= best_overlap:
                        # Ưu tiên xe đang chưa đậu ô nào, hoặc đang đậu đúng ô này
                        if state.parked_slot_id is None or state.parked_slot_id == slot_id:
                            best_id = state.global_id
                            best_overlap = geometry["vehicle_overlap"]
                            
                if best_id is not None:
                    # Gán ID ngay lập tức
                    self._bind_vehicle(best_id, slot_id, frame_idx, best_overlap, 0)
                    print(f"  👁️ Vision Auto-bind: GID #{best_id} → {slot_id} (overlap={best_overlap:.2f})")

            # Xoá Sticky ID khi cả Vision và Tracker đều xác nhận trống
            if not binding.vision_occupied and not binding.tracking_occupied and binding.vehicle_id is not None:
                # Tracker đã gỡ (qua _release_vehicle) nhưng Vision mới đổi thành trống lúc này
                # Hoặc cả hai mất cùng lúc.
                old_id = binding.vehicle_id
                binding.vehicle_id = None
                binding.vehicle_overlap = 0.0
                binding.stopped_for_ms = 0
                self._pending_release[slot_id] = (old_id, frame_idx) # LƯU VÀO PENDING ĐỂ CHỜ XE RA
                print(f"  🧹 Xoá Sticky ID #{old_id} khỏi {slot_id} do ô đã trống hoàn toàn.")

            self._sync_result(binding)
        self._cleanup_pending(frame_idx)
        self._cleanup_tokens(timestamp_s)

    def retain_slot_ids(self, slot_ids: Set[str]) -> None:
        """Remove bindings for ROI IDs deleted by a live configuration reload."""
        valid_ids = {str(slot_id) for slot_id in slot_ids}
        for slot_id in list(self._bindings):
            if slot_id in valid_ids:
                continue
            binding = self._bindings.pop(slot_id)
            self._pending_release.pop(slot_id, None)
            removed_token = self._departure_tokens.pop(slot_id, None)
            if removed_token is not None:
                self._event(
                    "departure_token_cancelled",
                    global_id=removed_token.global_id,
                    slot_id=slot_id,
                    reason="parking_slot_removed",
                )
            if binding.vehicle_id is None:
                continue
            global_id = int(binding.vehicle_id)
            if self._vehicle_to_slot.get(global_id) == slot_id:
                self._vehicle_to_slot.pop(global_id, None)
            state = self._vehicle_states.get(global_id)
            if state is not None and state.parked_slot_id == slot_id:
                state.parked_slot_id = None
                state.candidate_slot_id = None
                state.movement_state = "moving"
            self._event("parking_slot_removed", global_id=global_id, slot_id=slot_id)

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

    def recovery_priority_regions(self, timestamp_s: Optional[float] = None) -> List[np.ndarray]:
        """Return current expanded token regions for the sensitive tracker path."""
        now_s = self._last_timestamp_s if timestamp_s is None else float(timestamp_s)
        self._cleanup_tokens(now_s)
        regions: List[np.ndarray] = []
        for token in self._departure_tokens.values():
            radius = self._token_radius(token, now_s)
            # Scaling around the centroid is only used to wake the sensitive
            # detector.  Recovery itself uses an exact signed pixel distance.
            scale_ratio = 2.0 * radius / max(token.slot_diagonal, 1.0)
            regions.append(self._expanded_polygon(token.polygon, scale_ratio).astype(np.float32))
        return regions

    @staticmethod
    def _appearance_distance(
        reference: Optional[np.ndarray],
        candidate: Optional[np.ndarray],
    ) -> Optional[float]:
        if reference is None or candidate is None:
            return None
        left = np.asarray(reference, dtype=np.float32)
        right = np.asarray(candidate, dtype=np.float32)
        if left.size == 0 or right.size == 0 or left.shape != right.shape:
            return None
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return None
        try:
            return float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))
        except cv2.error:
            return None

    @classmethod
    def _optional_bbox(cls, track, key: str) -> Optional[BBox]:
        raw = cls._track_value(track, key)
        if raw is None:
            return None
        try:
            x, y, width, height = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
        if width <= 0.0 or height <= 0.0:
            return None
        return (x, y, width, height)

    @classmethod
    def _optional_point(cls, track, key: str) -> Optional[Point]:
        raw = cls._track_value(track, key)
        if raw is None:
            return None
        try:
            x, y = [float(value) for value in raw]
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        return (x, y)

    @classmethod
    def _optional_float(cls, track, key: str) -> Optional[float]:
        raw = cls._track_value(track, key)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _consume_departure_token(
        self,
        token: DepartureToken,
        local_key: Hashable,
        timestamp_s: float,
        details: dict,
    ) -> int:
        self._departure_tokens.pop(token.slot_id, None)
        self._pending_release.pop(token.slot_id, None)
        state = self._vehicle_states.setdefault(
            int(token.global_id),
            VehicleParkingState(global_id=int(token.global_id)),
        )
        state.movement_state = "exit_pending"
        state.parked_slot_id = None
        state.candidate_slot_id = None
        state.candidate_since = None
        state.outside_since = None
        state.parking_cooldown_until = float(timestamp_s) + self.predeparture_guard_seconds
        self._event(
            "parked_id_recovered",
            global_id=token.global_id,
            slot_id=token.slot_id,
            local_key=repr(local_key),
            evidence_frames=details.get("evidence_frames"),
            appearance_distance=details.get("appearance_distance"),
            size_ratio=details.get("size_ratio"),
            outward_px=details.get("outward_px"),
        )
        return int(token.global_id)

    def batch_recover_ids(
        self,
        unbound_tracks: Dict[Hashable, object],
        frame_idx: int,
        timestamp_s: float,
        camera_id: Optional[str] = None,
        coordinate_offset: Point = (0.0, 0.0),
        allow_cross_camera: bool = False,
    ) -> RecoveryBatchResult:
        """Safely associate all unbound local tracks with departure tokens.

        For a cross-camera call, provide ``recovery_position`` and optionally
        ``recovery_first_position`` in the token binder's pixel coordinate
        system, or provide a transformed ``recovery_bbox``.  Set
        ``allow_cross_camera=True`` only after that transform.  Original bbox
        dimensions may still be supplied for size validation, or the caller
        can supply a perspective-correct ``recovery_size_ratio``.
        """
        result = RecoveryBatchResult()
        if self.policy != "vision_primary" or not unbound_tracks:
            return result

        self._last_frame_idx = int(frame_idx)
        self._last_timestamp_s = float(timestamp_s)
        self._cleanup_tokens(timestamp_s)
        if not self._departure_tokens:
            return result

        candidate_keys = list(unbound_tracks.keys())
        candidate_data: Dict[Hashable, dict] = {}
        for key, track in unbound_tracks.items():
            recovery_bbox = self._optional_bbox(track, "recovery_bbox")
            bbox_is_transformed = recovery_bbox is not None
            bbox = recovery_bbox or self._track_bbox(track, coordinate_offset)
            recovery_position = self._optional_point(track, "recovery_position")
            point = recovery_position or self._bottom_center(bbox)

            recovery_first_bbox = self._optional_bbox(track, "recovery_first_bbox")
            first_bbox = recovery_first_bbox
            if first_bbox is None and not allow_cross_camera:
                first_bbox = self._optional_bbox(track, "first_bbox")
                if first_bbox is not None and not bbox_is_transformed:
                    first_bbox = (
                        first_bbox[0] + float(coordinate_offset[0]),
                        first_bbox[1] + float(coordinate_offset[1]),
                        first_bbox[2],
                        first_bbox[3],
                    )
            recovery_first_position = self._optional_point(
                track,
                "recovery_first_position",
            )
            first_position = recovery_first_position
            if first_position is None and first_bbox is not None:
                first_position = self._bottom_center(first_bbox)
            first_timestamp_s = self._optional_float(
                track,
                "recovery_first_timestamp_s",
            )
            if first_timestamp_s is None:
                first_timestamp_s = self._optional_float(
                    track,
                    "first_observation_timestamp_s",
                )

            appearance = self._track_value(track, "recovery_appearance")
            if appearance is None:
                appearance = self._track_value(track, "appearance")
            size_ratio_override = self._track_value(track, "recovery_size_ratio")
            try:
                size_ratio_override = (
                    float(size_ratio_override) if size_ratio_override is not None else None
                )
            except (TypeError, ValueError):
                size_ratio_override = None
            source_camera = self._track_value(track, "camera_id", camera_id)
            candidate_data[key] = {
                "bbox": bbox,
                "point": point,
                "first_bbox": first_bbox,
                "first_position": first_position,
                "first_timestamp_s": first_timestamp_s,
                "explicit_recovery_origin": bool(
                    recovery_first_bbox is not None
                    or recovery_first_position is not None
                ),
                "appearance": appearance,
                "size_ratio_override": size_ratio_override,
                "source_camera": source_camera,
            }

        tokens = list(self._departure_tokens.values())
        invalid_cost = 10.0
        costs = np.full((len(tokens), len(candidate_keys)), invalid_cost, dtype=np.float64)
        pair_details: Dict[Tuple[int, int], dict] = {}
        protected_pairs: Set[Tuple[int, int]] = set()

        for row, token in enumerate(tokens):
            radius = self._token_radius(token, timestamp_s)
            max_radius = token.slot_diagonal * self.recovery_max_expand_ratio
            for column, key in enumerate(candidate_keys):
                data = candidate_data[key]
                source_camera = data["source_camera"]
                if (
                    not allow_cross_camera
                    and source_camera is not None
                    and token.camera_id not in (None, source_camera)
                ):
                    continue
                is_cross_camera = bool(
                    allow_cross_camera
                    and source_camera is not None
                    and token.camera_id not in (None, source_camera)
                )

                evidence = token.candidates.get(key)
                migrated_evidence = False
                if evidence is None:
                    # Motion-only tracking can replace a local ID every few
                    # frames while the toy car starts moving. Transfer the
                    # token evidence only when exactly one currently visible
                    # fragment continues one recently disappeared fragment.
                    # This check runs before the ordinary spatial gate because
                    # a qualified fast departure may already be outside it.
                    live_keys = set(candidate_keys)
                    continuation_limit = max(
                        20.0, 0.35 * token.slot_diagonal
                    )
                    possible_sources = []
                    for old_key, old_evidence in token.candidates.items():
                        if old_key in live_keys:
                            continue
                        continuation_seconds = (
                            self.recovery_retention_seconds
                            if (
                                old_evidence.qualified_predeparture
                                or old_evidence.originated_in_slot
                            )
                            else 0.75
                        )
                        if (
                            float(timestamp_s) - old_evidence.last_seen_s
                            > continuation_seconds
                        ):
                            continue
                        source_limit = (
                            max(continuation_limit, 1.50 * token.slot_diagonal)
                            if old_evidence.qualified_predeparture
                            else (
                                max(continuation_limit, 2.25 * token.slot_diagonal)
                                if old_evidence.originated_in_slot
                                else continuation_limit
                            )
                        )
                        nearby_current = [
                            candidate_key
                            for candidate_key, candidate in candidate_data.items()
                            if float(
                                np.linalg.norm(
                                    np.subtract(
                                        candidate["point"],
                                        old_evidence.last_center,
                                    )
                                )
                            )
                            <= source_limit
                        ]
                        if nearby_current == [key]:
                            possible_sources.append((old_key, old_evidence))
                    if len(possible_sources) == 1:
                        old_key, evidence = possible_sources[0]
                        token.candidates.pop(old_key, None)
                        token.candidates[key] = evidence
                        migrated_evidence = True
                        self._event(
                            "departure_candidate_fragment_continued",
                            global_id=token.global_id,
                            slot_id=token.slot_id,
                            previous_local_key=repr(old_key),
                            current_local_key=repr(key),
                            observations=evidence.observations,
                        )

                bbox = data["bbox"]
                point = data["point"]
                signed_distance = self._signed_polygon_distance(point, token.polygon)
                original_overlap = (
                    0.0
                    if is_cross_camera and data["first_bbox"] is None
                    else self._raw_vehicle_overlap(bbox, token.polygon)
                )
                continuing_candidate = evidence is not None
                if (
                    signed_distance < -max_radius
                    and original_overlap <= 0.0
                    and not continuing_candidate
                ):
                    continue
                protected_pairs.add((row, column))
                result.protected_local_keys.add(key)

                if float(timestamp_s) + 1e-6 < token.created_at_s:
                    result.diagnostics[key] = {
                        "reason": "waiting_for_post_token_observation"
                    }
                    continue

                explicit_first = data["first_position"]
                explicit_first_bbox = data["first_bbox"]
                first_timestamp_s = data["first_timestamp_s"]
                origin_is_post_token = (
                    first_timestamp_s is not None
                    and token.created_at_s - 1e-6
                    <= first_timestamp_s
                    <= float(timestamp_s) + 1e-6
                )
                if first_timestamp_s is not None and not origin_is_post_token:
                    explicit_first = None
                    explicit_first_bbox = None
                elif (
                    first_timestamp_s is None
                    and is_cross_camera
                    and data["explicit_recovery_origin"]
                ):
                    # Cross-camera histories are frequently older than this
                    # parking departure.  Without a timestamp proving the
                    # origin is post-token, fail closed and start from now.
                    explicit_first = None
                    explicit_first_bbox = None
                reset_evidence = False
                if evidence is not None:
                    time_gap = float(timestamp_s) - evidence.last_seen_s
                    frame_gap = int(frame_idx) - evidence.last_frame_idx
                    reset_evidence = (
                        time_gap < 0.0
                        or (
                            time_gap > 0.50
                            and not evidence.qualified_predeparture
                            and not migrated_evidence
                        )
                        or frame_gap < 0
                        or (
                            frame_gap > 6
                            and not evidence.qualified_predeparture
                            and not migrated_evidence
                        )
                    )
                new_observation = False
                if evidence is None or reset_evidence:
                    # After a gap, start at the current point.  Reusing an old
                    # first_position would let intermittent shadow fragments
                    # fake one continuous outward trajectory.
                    first_center = point if reset_evidence else (explicit_first or point)
                    first_bbox = explicit_first_bbox or bbox
                    if reset_evidence:
                        first_bbox = bbox
                    first_signed_distance = self._signed_polygon_distance(
                        first_center,
                        token.polygon,
                    )
                    first_vehicle_overlap = self._raw_vehicle_overlap(
                        first_bbox,
                        token.polygon,
                    )
                    evidence = RecoveryCandidateEvidence(
                        first_seen_s=float(timestamp_s),
                        last_seen_s=float(timestamp_s),
                        first_frame_idx=int(frame_idx),
                        last_frame_idx=int(frame_idx),
                        observations=1,
                        first_center=first_center,
                        last_center=point,
                        first_bbox=first_bbox,
                        last_bbox=bbox,
                        originated_in_slot=bool(
                            first_vehicle_overlap >= 0.10
                            or first_signed_distance
                            >= -0.20 * token.slot_diagonal
                        ),
                    )
                    token.candidates[key] = evidence
                    new_observation = True
                elif evidence.last_frame_idx != int(frame_idx):
                    evidence.observations += 1
                    evidence.last_seen_s = float(timestamp_s)
                    evidence.last_frame_idx = int(frame_idx)
                    evidence.last_center = point
                    evidence.last_bbox = bbox
                    new_observation = True

                current_appearance_distance = self._appearance_distance(
                    token.last_appearance,
                    data["appearance"],
                )
                if new_observation and current_appearance_distance is not None:
                    evidence.appearance_distances.append(current_appearance_distance)
                appearance_distance = (
                    float(np.median(np.asarray(evidence.appearance_distances, dtype=np.float64)))
                    if len(evidence.appearance_distances) >= 2
                    and current_appearance_distance is not None
                    else None
                )

                first_signed = self._signed_polygon_distance(
                    evidence.first_center,
                    token.polygon,
                )
                if first_signed < -max_radius:
                    result.diagnostics[key] = {"reason": "origin_outside_max_gate"}
                    continue
                if (
                    signed_distance < -radius
                    and original_overlap <= 0.0
                    and not continuing_candidate
                ):
                    result.diagnostics[key] = {
                        "reason": "waiting_for_expanding_gate",
                        "recovery_radius_px": round(radius, 3),
                    }
                    continue

                if data["size_ratio_override"] is not None:
                    size_ratio = data["size_ratio_override"]
                elif token.last_bbox is not None:
                    size_ratio = (bbox[2] * bbox[3]) / max(
                        1.0,
                        token.last_bbox[2] * token.last_bbox[3],
                    )
                else:
                    size_ratio = None
                if size_ratio is None or not (
                    self.recovery_size_ratio_min
                    <= size_ratio
                    <= self.recovery_size_ratio_max
                ):
                    result.diagnostics[key] = {
                        "reason": "size_missing_or_mismatch",
                        "size_ratio": size_ratio,
                    }
                    continue

                first_overlap = self._raw_vehicle_overlap(evidence.first_bbox, token.polygon)
                relaxed_appearance = (
                    not is_cross_camera
                    and max(first_overlap, original_overlap)
                    >= self.recovery_relaxed_slot_overlap
                )
                appearance_limit = (
                    self.recovery_relaxed_appearance_threshold
                    if relaxed_appearance
                    else self.recovery_appearance_threshold
                )
                if appearance_distance is None or appearance_distance > appearance_limit:
                    result.diagnostics[key] = {
                        "reason": "appearance_missing_or_mismatch",
                        "appearance_distance": appearance_distance,
                        "appearance_limit": appearance_limit,
                    }
                    continue

                first_vector = np.asarray(evidence.first_center, dtype=np.float64) - np.asarray(
                    token.center, dtype=np.float64
                )
                movement = np.asarray(point, dtype=np.float64) - np.asarray(
                    evidence.first_center, dtype=np.float64
                )
                radial_first = float(np.linalg.norm(first_vector))
                radial_now = float(
                    np.linalg.norm(np.asarray(point, dtype=np.float64) - np.asarray(token.center))
                )
                # Near the slot centre, a few pixels of detector jitter can
                # point ``first_vector`` toward the wrong side of the ROI.
                # Its dot product then reports inward motion even while the
                # vehicle's radial distance clearly grows by hundreds of
                # pixels. Use radial gain until the origin direction is
                # geometrically stable.
                outward_unit = (
                    first_vector / radial_first
                    if radial_first >= 0.35 * token.slot_diagonal
                    else None
                )
                outward_px = (
                    float(np.dot(movement, outward_unit))
                    if outward_unit is not None
                    else radial_now - radial_first
                )
                radial_gain = radial_now - radial_first
                moved_px = float(np.linalg.norm(movement))
                if (
                    evidence.observations < self.recovery_evidence_frames
                    or moved_px < self.recovery_min_movement_px
                    or outward_px < self.recovery_min_outward_px
                    or radial_gain < self.recovery_min_radial_gain_px
                ):
                    result.diagnostics[key] = {
                        "reason": "insufficient_outward_evidence",
                        "evidence_frames": evidence.observations,
                        "moved_px": round(moved_px, 3),
                        "outward_px": round(outward_px, 3),
                        "radial_gain_px": round(radial_gain, 3),
                    }
                    continue
                evidence.qualified_predeparture = True
                if not token.confirmed_empty:
                    result.diagnostics[key] = {
                        "reason": "departure_not_yet_confirmed",
                        "evidence_frames": evidence.observations,
                    }
                    continue

                spatial_cost = min(
                    1.0,
                    float(np.hypot(point[0] - token.center[0], point[1] - token.center[1]))
                    / max(1.0, token.slot_diagonal + radius),
                )
                size_cost = min(1.0, abs(float(np.log(max(size_ratio, 1e-6)))) / np.log(2.5))
                direction_cost = max(0.0, 1.0 - outward_px / max(3.0, token.slot_diagonal * 0.10))
                cost = (
                    0.45 * spatial_cost
                    + 0.35 * (appearance_distance / max(appearance_limit, 1e-6))
                    + 0.15 * size_cost
                    + 0.05 * direction_cost
                )
                details = {
                    "slot_id": token.slot_id,
                    "global_id": token.global_id,
                    "cost": float(cost),
                    "evidence_frames": evidence.observations,
                    "appearance_distance": round(float(appearance_distance), 4),
                    "size_ratio": round(float(size_ratio), 4),
                    "outward_px": round(float(outward_px), 3),
                    "recovery_radius_px": round(radius, 3),
                }
                costs[row, column] = float(cost)
                pair_details[(row, column)] = details
                result.diagnostics[key] = details

        if not pair_details:
            return result

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.999)
        for row, token in enumerate(tokens):
            column = int(row_to_col[row])
            if column < 0 or (row, column) not in pair_details:
                continue
            best_cost = float(costs[row, column])
            row_alternatives = sorted(
                float(costs[row, other])
                for other in range(len(candidate_keys))
                if other != column and costs[row, other] < invalid_cost
            )
            column_alternatives = sorted(
                float(costs[other, column])
                for other in range(len(tokens))
                if other != row and costs[other, column] < invalid_cost
            )
            row_gap = (row_alternatives[0] - best_cost) if row_alternatives else float("inf")
            column_gap = (
                column_alternatives[0] - best_cost
                if column_alternatives
                else float("inf")
            )
            if row_gap < self.recovery_ambiguity_margin or column_gap < self.recovery_ambiguity_margin:
                ambiguous_columns = {
                    other
                    for other in range(len(candidate_keys))
                    if costs[row, other] < invalid_cost
                    and float(costs[row, other]) - best_cost < self.recovery_ambiguity_margin
                }
                for other in ambiguous_columns:
                    result.ambiguous_local_keys.add(candidate_keys[other])
                self._event(
                    "parked_id_recovery_ambiguous",
                    global_id=token.global_id,
                    slot_id=token.slot_id,
                    candidate_count=len(ambiguous_columns),
                )
                continue

            key = candidate_keys[column]
            global_id = self._consume_departure_token(
                token,
                key,
                timestamp_s,
                pair_details[(row, column)],
            )
            result.recovered_ids[key] = global_id
            result.protected_local_keys.discard(key)

        return result

    def try_recover_id(
        self,
        position: Optional[Tuple[int, int]] = None,
        camera_id: Optional[str] = None,
        bbox: Optional[BBox] = None,
        appearance: Optional[np.ndarray] = None,
        coordinate_offset: Point = (0.0, 0.0),
        local_key: Optional[Hashable] = None,
        frame_idx: Optional[int] = None,
        timestamp_s: Optional[float] = None,
        allow_cross_camera: bool = False,
    ) -> Optional[int]:
        """Recover the parked global ID before a new local/global ID is allocated."""
        if self.policy == "vision_primary":
            # The compatibility wrapper is intentionally fail-closed.  A
            # stable local key is required to accumulate multi-frame evidence;
            # inventing one from a moving bbox could let unrelated noise
            # consume the token.
            if local_key is None:
                return None
            track_data = {
                "bbox": bbox,
                "appearance": appearance,
                "camera_id": camera_id,
            }
            if position is not None:
                track_data["recovery_position"] = position
            batch = self.batch_recover_ids(
                {local_key: track_data},
                self._last_frame_idx if frame_idx is None else int(frame_idx),
                self._last_timestamp_s if timestamp_s is None else float(timestamp_s),
                camera_id=camera_id,
                coordinate_offset=coordinate_offset,
                allow_cross_camera=allow_cross_camera,
            )
            return batch.recovered_ids.get(local_key)

        if bbox is not None:
            x, y, w, h = bbox
            global_bbox = (x + coordinate_offset[0], y + coordinate_offset[1], w, h)
            # SỬ DỤNG BOTTOM-CENTER THAY VÌ CENTER ĐỂ KHỚP VỚI LÚC DE XE RA
            point = (global_bbox[0] + global_bbox[2] / 2.0, global_bbox[1] + global_bbox[3])
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
                # Nới lỏng rào cản màu sắc (từ 0.50 lên 0.85) để không chặn nhầm xe đi ra do ngược sáng
                if appearance_distance > 0.85:
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
        print(f"  🔄 Track mới từ {slot_id} nhận lại Global #{global_id}")
        return global_id

    def resolve_pending_global_ids(self, active_global_ids: Set[int]) -> None:
        for slot_id, (vehicle_id, _) in list(self._pending_release.items()):
            if vehicle_id in active_global_ids:
                self._pending_release.pop(slot_id, None)

    def cancel_recovery_for_global_id(
        self,
        global_id: int,
        reason: str = "identity_observed_elsewhere",
    ) -> None:
        """Cancel tokens only after the runner validates the identity elsewhere."""
        target = int(global_id)
        for slot_id, token in list(self._departure_tokens.items()):
            if token.global_id != target:
                continue
            self._departure_tokens.pop(slot_id, None)
            self._event(
                "departure_token_cancelled",
                global_id=target,
                slot_id=slot_id,
                reason=reason,
            )

    def remap_vehicle_ids(self, canonicalize: Callable[[int], int]) -> None:
        """Move parked bindings/states to canonical IDs after a global-ID merge."""
        remapped_claims: Dict[Tuple[str, int], ArrivalClaim] = {}
        for claim in self._arrival_claims.values():
            new_id = int(canonicalize(int(claim.global_id)))
            claim.global_id = new_id
            key = (claim.slot_id, new_id)
            previous = remapped_claims.get(key)
            if previous is None or claim.score > previous.score:
                remapped_claims[key] = claim
        self._arrival_claims = remapped_claims

        token_groups: Dict[int, List[DepartureToken]] = {}
        for token in self._departure_tokens.values():
            old_id = int(token.global_id)
            new_id = int(canonicalize(old_id))
            token.global_id = new_id
            if new_id != old_id:
                old_state = self._vehicle_states.pop(old_id, None)
                if old_state is not None and new_id not in self._vehicle_states:
                    old_state.global_id = new_id
                    self._vehicle_states[new_id] = old_state
                self._event(
                    "departure_token_remapped",
                    old_global_id=old_id,
                    global_id=new_id,
                    slot_id=token.slot_id,
                )
            token_groups.setdefault(new_id, []).append(token)

        # A canonical identity can own at most one live departure.  Keep the
        # newest token; older duplicate bindings came from the IDs just merged.
        for global_id, candidates in token_groups.items():
            if len(candidates) <= 1:
                continue
            winner = max(candidates, key=lambda item: item.created_at_s)
            for token in candidates:
                if token is winner:
                    continue
                self._departure_tokens.pop(token.slot_id, None)
                self._event(
                    "departure_token_cancelled",
                    global_id=global_id,
                    slot_id=token.slot_id,
                    reason="global_id_merge_conflict",
                    kept_slot_id=winner.slot_id,
                )

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
        if self.policy == "vision_primary":
            binding.occupied = bool(binding.vision_occupied)
            binding.decision_source = "vision" if binding.vision_occupied else "none"
            if binding.result_ref is not None:
                binding.result_ref.occupied = binding.occupied
                binding.result_ref.vehicle_id = binding.vehicle_id
            return

        # Tracking chỉ được override vision khi xe ĐÃ THỰC SỰ DỪNG (≥ 500ms)
        # Nếu chỉ là nhiễu (track xuất hiện vài frame rồi mất), stopped_for_ms = 0
        # → không override → kết quả khớp với threshold pixel view
        tracking_confirmed = (
            binding.tracking_occupied
            and binding.stopped_for_ms >= 500
        )
        binding.occupied = bool(binding.vision_occupied or tracking_confirmed)

        # Cập nhật decision_source cho debug/JSON
        if binding.vision_occupied and tracking_confirmed:
            binding.decision_source = "vision_and_tracking"
        elif tracking_confirmed:
            binding.decision_source = "tracking_override"
        elif binding.vision_occupied:
            binding.decision_source = "vision"
        else:
            binding.decision_source = "none"

        if binding.result_ref is not None:
            binding.result_ref.occupied = binding.occupied
            binding.result_ref.vehicle_id = binding.vehicle_id

    def _binding_to_json(self, binding: SlotBinding) -> dict:
        token = self._departure_tokens.get(binding.slot_id)
        payload = {
            "occupied": bool(binding.occupied),
            "status": "occupied" if binding.occupied else "empty",
            "vehicle_id": binding.vehicle_id,
            "raw_occupied": bool(binding.vision_occupied),
            "vision_occupied": bool(binding.vision_occupied),
            "tracking_occupied": bool(binding.tracking_occupied),
            "decision_source": binding.decision_source,
            "tracking_state": binding.tracking_state,
            "vehicle_overlap": round(float(binding.vehicle_overlap), 4),
            "stopped_for_ms": int(binding.stopped_for_ms),
        }
        if token is None:
            payload.update(
                recovery_state="none",
                recovery_age_ms=0,
                recovery_radius_px=0.0,
            )
        else:
            payload.update(
                recovery_state="searching" if token.confirmed_empty else "provisional",
                recovery_global_id=int(token.global_id),
                recovery_age_ms=int(
                    max(0.0, self._last_timestamp_s - token.created_at_s) * 1000
                ),
                recovery_radius_px=round(
                    self._token_radius(token, self._last_timestamp_s),
                    3,
                ),
                recovery_candidate_count=len(token.candidates),
            )
        return payload

    def to_json(self, camera_id: Optional[str] = None) -> Dict[str, dict]:
        return {
            slot_id: self._binding_to_json(binding)
            for slot_id, binding in self._bindings.items()
            if camera_id is None or binding.camera_id == camera_id
        }
