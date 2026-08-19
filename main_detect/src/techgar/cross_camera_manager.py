"""Global vehicle IDs and predictive handoff between adjacent cameras.

Local tracker IDs belong to one camera only.  ``CrossCameraManager`` owns the
single global namespace and associates a new local observation with an
existing global vehicle before that observation is confirmed by its local
tracker.  This is important for fast vehicles: local confirmation can happen
several frames after the vehicle crossed a camera border.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from lap import lapjv

from .tracklet_descriptor import (
    aggregate_appearance,
    appearance_samples,
    compare_tracklets,
    merge_appearance_samples,
)


# (source camera, exit edge) -> target camera for the simulated 2x2 layout.
EDGE_ADJACENCY = {
    ("cam1", "right"): "cam2", ("cam1", "bottom"): "cam3",
    ("cam2", "left"): "cam1", ("cam2", "bottom"): "cam4",
    ("cam3", "top"): "cam1", ("cam3", "right"): "cam4",
    ("cam4", "top"): "cam2", ("cam4", "left"): "cam3",
}
OPPOSITE_EDGE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


@dataclass
class HandoffEntry:
    global_id: int
    source_cam: str
    source_local_track_id: int
    target_cam: str
    exit_edge: str
    last_world: Tuple[float, float]
    velocity_world: Tuple[float, float]
    bbox_size: Tuple[int, int]
    appearance: Optional[np.ndarray]
    appearance_samples: Tuple[np.ndarray, ...]
    created_at_frame: int
    updated_at_frame: int
    last_rejection_frame: int = -999999


@dataclass
class LostTrackEntry:
    global_id: int
    camera_id: str
    local_track_id: int
    last_world: Tuple[float, float]
    velocity_world: Tuple[float, float]
    bbox_size: Tuple[int, int]
    appearance: Optional[np.ndarray]
    appearance_samples: Tuple[np.ndarray, ...]
    lost_at_frame: int


@dataclass
class GlobalIdentityState:
    """Last reliable observation of one vehicle across local trackers."""

    global_id: int
    state: str
    last_camera: str
    last_local_track_id: int
    last_world: Tuple[float, float]
    velocity_world: Tuple[float, float]
    velocity_world_per_second: Tuple[float, float]
    bbox_size: Tuple[int, int]
    appearance: Optional[np.ndarray]
    appearance_samples: Tuple[np.ndarray, ...]
    last_seen_frame: int
    last_seen_time: Optional[float]
    dormant_since_frame: Optional[int] = None
    dormant_since_time: Optional[float] = None
    exited_at_frame: Optional[int] = None
    exited_at_time: Optional[float] = None


class CrossCameraManager:
    """Maintain unique global IDs and conservative predictive handoffs.

    A handoff is opened before the source vehicle reaches an exit edge.  Once
    the source is no longer visible, its last velocity extrapolates the
    expected destination position.  Candidate/entry matches are solved in one
    batch so an old global ID cannot be consumed by two new local tracks.
    """

    def __init__(
        self,
        camera_sizes: Dict[str, Tuple[int, int]],
        camera_crops: Dict[str, Tuple[int, int, int, int]],
        camera_transforms: Optional[Dict[str, np.ndarray]] = None,
        edge_adjacency: Optional[Dict[Tuple[str, str], str]] = None,
        overlap_regions: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
        custom_masks: Optional[Dict[str, dict]] = None,
        edge_margin: int = 40,
        handoff_ttl: int = 45,
        match_distance: float = 100.0,
        appearance_threshold: float = 0.45,
        relaxed_appearance_threshold: Optional[float] = None,
        cross_camera_duplicate_distance: Optional[float] = None,
        cross_camera_defer_frames: int = 8,
        lookahead_frames: int = 16,
        prediction_radius: float = 90.0,
        min_direction_cosine: float = 0.25,
        identity_retention_frames: int = 180,
        identity_retention_seconds: float = 8.0,
        dormant_match_distance: float = 160.0,
        dormant_appearance_threshold: Optional[float] = None,
        tracklet_gallery_size: int = 24,
        exit_zones: Optional[Dict[str, List[np.ndarray]]] = None,
        world_unit: str = "source_video_pixel",
        shared_map_anchor: str = "bottom_center",
    ):
        self.camera_sizes = camera_sizes
        self.camera_crops = camera_crops
        # Virtual cameras use crop offsets. Real cameras supply one homography
        # per camera into a shared ground-plane coordinate system instead.
        self.camera_transforms = {
            camera_id: np.asarray(transform, dtype=np.float64)
            for camera_id, transform in (camera_transforms or {}).items()
        }
        self.edge_adjacency = edge_adjacency or EDGE_ADJACENCY
        self.overlap_regions = {
            tuple(key): np.asarray(region, dtype=np.float32)
            for key, region in (overlap_regions or {}).items()
        }
        self.custom_masks = custom_masks or {}
        self.edge_margin = int(edge_margin)
        self.handoff_ttl = int(handoff_ttl)
        self.match_distance = float(match_distance)  # kept for overlap compatibility
        self.appearance_threshold = float(appearance_threshold)
        self.relaxed_appearance_threshold = float(
            relaxed_appearance_threshold
            if relaxed_appearance_threshold is not None
            else max(self.appearance_threshold, 0.82)
        )
        self.cross_camera_duplicate_distance = max(
            1.0,
            float(
                cross_camera_duplicate_distance
                if cross_camera_duplicate_distance is not None
                else self.match_distance * 0.60
            ),
        )
        self.strong_spatial_distance = min(
            self.cross_camera_duplicate_distance,
            max(1.0, self.match_distance * 0.40),
        )
        self.cross_camera_defer_frames = max(0, int(cross_camera_defer_frames))
        self.lookahead_frames = max(1, int(lookahead_frames))
        self.prediction_radius = max(1.0, float(prediction_radius))
        self.min_direction_cosine = float(np.clip(min_direction_cosine, -1.0, 1.0))
        self.identity_retention_frames = max(1, int(identity_retention_frames))
        self.identity_retention_seconds = max(0.1, float(identity_retention_seconds))
        self.dormant_match_distance = max(1.0, float(dormant_match_distance))
        self.dormant_appearance_threshold = float(
            dormant_appearance_threshold
            if dormant_appearance_threshold is not None
            else min(0.75, self.appearance_threshold + 0.15)
        )
        self.tracklet_gallery_size = max(1, int(tracklet_gallery_size))
        self.world_unit = str(world_unit or "source_video_pixel")
        self.shared_map_anchor = str(shared_map_anchor or "bottom_center")
        if self.shared_map_anchor not in {"bottom_center", "bbox_center"}:
            raise ValueError(
                "shared_map_anchor must be 'bottom_center' or 'bbox_center'"
            )
        self.exit_zones = {
            camera_id: [np.asarray(polygon, dtype=np.float32) for polygon in polygons]
            for camera_id, polygons in (exit_zones or {}).items()
        }
        self._next_global_id = 1
        self._local_to_global: Dict[Tuple[str, int], int] = {}
        self._gid_members: Dict[int, set[Tuple[str, int]]] = {}
        # Retired IDs are permanent aliases to the smaller canonical ID.  A
        # handoff/slot recovery that still references an old ID cannot revive it.
        self._global_aliases: Dict[int, int] = {}
        self._handoffs: List[HandoffEntry] = []
        self._recently_lost: List[LostTrackEntry] = []
        self._identities: Dict[int, GlobalIdentityState] = {}
        self._cross_camera_deferred_since: Dict[Tuple[str, int], int] = {}
        self._events: List[dict] = []

    def _allocate_global_id(self) -> int:
        global_id = self._next_global_id
        self._next_global_id += 1
        return global_id

    def _canonical_id(self, global_id: int) -> int:
        path = []
        while global_id in self._global_aliases:
            path.append(global_id)
            global_id = self._global_aliases[global_id]
        for old_id in path:
            self._global_aliases[old_id] = global_id
        return global_id

    def _bind(self, cam_id: str, local_track_id: int, global_id: int) -> int:
        global_id = self._canonical_id(global_id)
        key = (cam_id, local_track_id)
        previous = self._local_to_global.get(key)
        if previous is not None:
            previous = self._canonical_id(previous)
        if previous is not None and previous != global_id:
            self._gid_members.get(previous, set()).discard(key)
        self._local_to_global[key] = global_id
        self._gid_members.setdefault(global_id, set()).add(key)
        return global_id

    @staticmethod
    def _appearance_snapshot(appearance: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return aggregate_appearance(appearance)

    def _tracklet_snapshot(self, source) -> Tuple[np.ndarray, ...]:
        return tuple(
            sample.copy()
            for sample in appearance_samples(source)[-self.tracklet_gallery_size:]
        )

    def _are_adjacent(self, first_camera: str, second_camera: str) -> bool:
        if first_camera == second_camera:
            return False
        return any(
            (source == first_camera and target == second_camera)
            or (source == second_camera and target == first_camera)
            for (source, _edge), target in self.edge_adjacency.items()
        )

    def _in_exit_zone(self, cam_id: str, point: Tuple[int, int]) -> bool:
        return any(
            polygon.ndim == 2
            and polygon.shape[0] >= 3
            and polygon.shape[1] == 2
            and cv2.pointPolygonTest(polygon, point, False) >= 0
            for polygon in self.exit_zones.get(cam_id, [])
        )

    def _observe_identity(
        self,
        global_id: int,
        cam_id: str,
        local_track_id: int,
        track,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> GlobalIdentityState:
        global_id = self._canonical_id(global_id)
        world = self._track_world(cam_id, track)
        velocity_world = self._world_velocity(cam_id, track)
        previous = self._identities.get(global_id)
        gallery = merge_appearance_samples(
            previous.appearance_samples if previous is not None else (),
            track,
            self.tracklet_gallery_size,
        )
        appearance = aggregate_appearance(gallery)
        if appearance is None:
            appearance = self._appearance_snapshot(getattr(track, "appearance", None))
        velocity_per_second = previous.velocity_world_per_second if previous else (0.0, 0.0)
        if (
            previous is not None
            and timestamp_s is not None
            and previous.last_seen_time is not None
            and timestamp_s > previous.last_seen_time
        ):
            elapsed = timestamp_s - previous.last_seen_time
            measured = (
                (world[0] - previous.last_world[0]) / elapsed,
                (world[1] - previous.last_world[1]) / elapsed,
            )
            if np.hypot(*velocity_per_second) < 1e-6:
                velocity_per_second = measured
            else:
                velocity_per_second = (
                    0.65 * velocity_per_second[0] + 0.35 * measured[0],
                    0.65 * velocity_per_second[1] + 0.35 * measured[1],
                )
        state = GlobalIdentityState(
            global_id=global_id,
            state="active",
            last_camera=cam_id,
            last_local_track_id=local_track_id,
            last_world=world,
            velocity_world=velocity_world,
            velocity_world_per_second=velocity_per_second,
            bbox_size=(track.w, track.h),
            appearance=appearance,
            appearance_samples=gallery,
            last_seen_frame=frame_idx,
            last_seen_time=timestamp_s,
        )
        self._identities[global_id] = state
        return state

    def _set_identity_dormant(
        self,
        global_id: int,
        cam_id: str,
        local_track_id: int,
        track,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> None:
        identity = self._observe_identity(
            global_id, cam_id, local_track_id, track, frame_idx, timestamp_s
        )
        identity.state = "handoff" if any(
            self._canonical_id(entry.global_id) == identity.global_id
            for entry in self._handoffs
        ) else "dormant"
        identity.dormant_since_frame = frame_idx
        identity.dormant_since_time = timestamp_s

    def _mark_identity_exited(
        self,
        global_id: int,
        cam_id: str,
        local_track_id: int,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> None:
        global_id = self._canonical_id(global_id)
        identity = self._identities.get(global_id)
        if identity is None:
            return
        identity.state = "exited"
        identity.exited_at_frame = frame_idx
        identity.exited_at_time = timestamp_s
        identity.dormant_since_frame = None
        identity.dormant_since_time = None
        self._handoffs = [
            entry for entry in self._handoffs
            if self._canonical_id(entry.global_id) != global_id
        ]
        self._event(
            "global_identity_exited", frame_idx, global_id,
            camera=cam_id, local_track_id=local_track_id,
        )

    def bind_external_id(
        self,
        cam_id: str,
        local_track_id: int,
        global_id: int,
        frame_idx: int,
        source: str = "external",
    ) -> int:
        """Bind a verified global identity before normal ID allocation.

        Parking-slot recovery calls this for a first local observation leaving a
        just-vacated slot.  The binder stores global IDs, so this never revives
        a camera-local number in another camera.
        """
        global_id = self._canonical_id(global_id)
        bound = self._bind(cam_id, local_track_id, global_id)
        self._event("global_id_recovered", frame_idx, global_id, camera=cam_id,
                    local_track_id=local_track_id, source=source)
        return bound

    def notify_track_lost(
        self,
        cam_id: str,
        local_track_id: int,
        track,
        frame_idx: int,
        timestamp_s: Optional[float] = None,
    ) -> None:
        """Remember a just-lost global track for conservative same-camera Re-ID."""
        global_id = self._local_to_global.get((cam_id, local_track_id))
        if global_id is None:
            return
        global_id = self._canonical_id(global_id)
        self._recently_lost = [
            item for item in self._recently_lost
            if not (item.global_id == global_id and item.camera_id == cam_id)
        ]
        self._recently_lost.append(LostTrackEntry(
            global_id=global_id, camera_id=cam_id, local_track_id=local_track_id,
            last_world=self._track_world(cam_id, track),
            velocity_world=self._world_velocity(cam_id, track), bbox_size=(track.w, track.h),
            appearance=aggregate_appearance(track),
            appearance_samples=self._tracklet_snapshot(track),
            lost_at_frame=frame_idx,
        ))
        self._set_identity_dormant(
            global_id, cam_id, local_track_id, track, frame_idx, timestamp_s
        )
        self._event("local_track_lost", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)

    def _merge_global_ids(self, canonical_id: int, duplicate_id: int, frame_idx: int, reason: str) -> None:
        canonical_id = self._canonical_id(canonical_id)
        duplicate_id = self._canonical_id(duplicate_id)
        if canonical_id == duplicate_id:
            return
        # Product invariant: the smaller/older global ID always survives.
        canonical_id, duplicate_id = min(canonical_id, duplicate_id), max(canonical_id, duplicate_id)
        self._global_aliases[duplicate_id] = canonical_id
        for key, global_id in list(self._local_to_global.items()):
            if self._canonical_id(global_id) == canonical_id or global_id == duplicate_id:
                self._bind(key[0], key[1], canonical_id)
        self._gid_members.pop(duplicate_id, None)
        canonical_state = self._identities.get(canonical_id)
        duplicate_state = self._identities.pop(duplicate_id, None)
        merged_gallery = merge_appearance_samples(
            canonical_state.appearance_samples if canonical_state is not None else (),
            duplicate_state.appearance_samples if duplicate_state is not None else (),
            self.tracklet_gallery_size,
        )
        if duplicate_state is not None and (
            canonical_state is None
            or duplicate_state.last_seen_frame > canonical_state.last_seen_frame
        ):
            duplicate_state.global_id = canonical_id
            self._identities[canonical_id] = duplicate_state
            canonical_state = duplicate_state
        if canonical_state is not None:
            canonical_state.appearance_samples = merged_gallery
            canonical_state.appearance = aggregate_appearance(merged_gallery)
        for handoff in self._handoffs:
            handoff.global_id = self._canonical_id(handoff.global_id)
        for lost in self._recently_lost:
            lost.global_id = self._canonical_id(lost.global_id)
        self._handoffs = [
            item for index, item in enumerate(self._handoffs)
            if not any(index > earlier and item.global_id == prior.global_id and item.source_cam == prior.source_cam and item.target_cam == prior.target_cam for earlier, prior in enumerate(self._handoffs))
        ]
        self._event("global_id_merged", frame_idx, canonical_id, superseded_global_id=duplicate_id, reason=reason)
        print(f"  [merge] global #{duplicate_id} -> #{canonical_id} ({reason})")

    def _merge_recently_lost_duplicates(self, all_tracks: Dict[str, dict], frame_idx: int) -> None:
        """Prefer the pre-existing ID when its motion echo outlives it."""
        retained = []
        for entry in self._recently_lost:
            elapsed = frame_idx - entry.lost_at_frame
            if elapsed > 30:
                continue
            predicted = (
                entry.last_world[0] + entry.velocity_world[0] * elapsed,
                entry.last_world[1] + entry.velocity_world[1] * elapsed,
            )
            merged = False
            for local_id, track in all_tracks.get(entry.camera_id, {}).items():
                candidate_id = self._local_to_global.get((entry.camera_id, local_id))
                if candidate_id is None or candidate_id == entry.global_id:
                    continue
                world = self._track_world(entry.camera_id, track)
                if np.linalg.norm(np.subtract(world, predicted)) > 70.0:
                    continue
                if self._appearance_distance(
                    track, entry.appearance_samples or entry.appearance
                ) > 0.22:
                    continue
                if self._size_distance((track.w, track.h), entry.bbox_size) > 0.85:
                    continue
                direction = self._direction_cosine(entry.camera_id, track, entry.velocity_world)
                if direction is not None and direction < 0.50:
                    continue
                self._merge_global_ids(entry.global_id, candidate_id, frame_idx, "lost_track_continuation")
                merged = True
                break
            if not merged:
                retained.append(entry)
        self._recently_lost = retained

    def _event(self, kind: str, frame_idx: int, global_id: int, **details) -> None:
        self._events.append({"type": kind, "frame": frame_idx, "global_id": global_id, **details})
        self._events = self._events[-200:]

    def _world(self, cam_id: str, local_point: Tuple[int, int]) -> Tuple[float, float]:
        transform = self.camera_transforms.get(cam_id)
        if transform is not None:
            point = np.asarray([[[float(local_point[0]), float(local_point[1])]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(point, transform)[0, 0]
            return float(mapped[0]), float(mapped[1])
        x1, y1, _, _ = self.camera_crops[cam_id]
        return float(x1 + local_point[0]), float(y1 + local_point[1])

    def _bbox_anchor(
        self,
        cam_id: str,
        cx: float,
        cy: float,
        bbox_h: float,
    ) -> Tuple[float, float]:
        """Return the camera point used on the shared ground-plane map.

        The motion tracker deliberately follows the bbox bottom-centre because
        it is stable for local Kalman association.  With two opposing camera
        views, however, that point lands on opposite ends of the same vehicle.
        A bbox centre is substantially more view-invariant for this top-down
        opposing-camera setup, but a normal ground-plane homography may still
        require bottom-centre.  The calibration therefore selects the anchor
        explicitly. Virtual crop cameras keep their legacy tracker point so
        ``main.py`` retains exactly the old coordinate semantics.
        """
        if (
            cam_id in self.camera_transforms
            and self.shared_map_anchor == "bbox_center"
        ):
            return float(cx), float(cy) - float(bbox_h) * 0.5
        return float(cx), float(cy)

    def _track_local_anchor(self, cam_id: str, track) -> Tuple[float, float]:
        return self._bbox_anchor(cam_id, track.cx, track.cy, track.h)

    def _track_world(self, cam_id: str, track) -> Tuple[float, float]:
        return self._world(cam_id, self._track_local_anchor(cam_id, track))

    def _expired_track_world(
        self,
        cam_id: str,
        cx: float,
        cy: float,
        bbox_h: float,
    ) -> Tuple[float, float]:
        return self._world(cam_id, self._bbox_anchor(cam_id, cx, cy, bbox_h))

    def _local(self, cam_id: str, world_point: Tuple[float, float]) -> Tuple[float, float]:
        transform = self.camera_transforms.get(cam_id)
        if transform is not None:
            inverse = np.linalg.inv(transform)
            point = np.asarray([[[float(world_point[0]), float(world_point[1])]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(point, inverse)[0, 0]
            return float(mapped[0]), float(mapped[1])
        x1, y1, _, _ = self.camera_crops[cam_id]
        return world_point[0] - x1, world_point[1] - y1

    @staticmethod
    def _is_confirmed(track) -> bool:
        status = getattr(track, "status", None)
        return getattr(status, "value", status) == "confirmed"

    @classmethod
    def _dormant_reid_ready(cls, track, frame_idx: int) -> bool:
        """Require a mature *current fragment* before reviving a dormant GID.

        Motion-track objects can be recycled from the local exited-track
        gallery.  Their lifetime ``total_visible_count`` and display history
        then include the old fragment, so neither value alone proves that the
        newly appeared blob has survived more than one frame.  The tracker
        resets ``first_observation_frame`` for each fragment; prefer that
        evidence when available and retain history/count fallbacks for legacy
        trackers and tests.
        """
        if not cls._is_confirmed(track):
            return False

        fragment_start = getattr(track, "first_observation_frame", None)
        if fragment_start is not None:
            try:
                return int(frame_idx) - int(fragment_start) + 1 >= 3
            except (TypeError, ValueError):
                return False

        visible_count = getattr(track, "total_visible_count", None)
        if visible_count is not None:
            try:
                return int(visible_count) >= 3
            except (TypeError, ValueError):
                return False

        return len(getattr(track, "history", ())) >= 3

    @staticmethod
    def _velocity(track) -> Tuple[float, float]:
        """Robust per-frame local/world velocity from the recent history."""
        history = getattr(track, "history", [])
        if len(history) < 2:
            return 0.0, 0.0
        # Average over at most four frame-to-frame steps to suppress blob jitter.
        first = history[max(0, len(history) - 5)]
        steps = max(1, len(history) - max(0, len(history) - 5) - 1)
        return (float(track.cx - first[0]) / steps, float(track.cy - first[1]) / steps)

    def _world_velocity(self, cam_id: str, track) -> Tuple[float, float]:
        """Velocity in shared coordinates; preserves local behavior for crops."""
        if cam_id not in self.camera_transforms:
            return self._velocity(track)
        history = getattr(track, "history", [])
        if len(history) < 2:
            return 0.0, 0.0
        start_index = max(0, len(history) - 5)
        first = history[start_index]
        steps = max(1, len(history) - start_index - 1)
        current = self._track_world(cam_id, track)
        previous = self._world(
            cam_id,
            self._bbox_anchor(cam_id, first[0], first[1], track.h),
        )
        return (current[0] - previous[0]) / steps, (current[1] - previous[1]) / steps

    def _outward_edge(self, cam_id: str, track) -> Optional[Tuple[str, Tuple[float, float]]]:
        """Return the most likely exit edge and its outward velocity.

        The early zone scales with speed, allowing a fast car to open a handoff
        before it disappears from the source crop.
        """
        vx, vy = self._velocity(track)
        width, height = self.camera_sizes[cam_id]
        options = []
        
        if cam_id in self.custom_masks:
            mask_data = self.custom_masks[cam_id]
            polygon = mask_data["polygon"]
            handoff_edge_index = int(mask_data["handoff_edge"]) - 1  # 0-indexed
            
            # polygon is list of dicts {"x": x, "y": y}
            if 0 <= handoff_edge_index < len(polygon):
                p1 = polygon[handoff_edge_index]
                p2 = polygon[(handoff_edge_index + 1) % len(polygon)]
                
                # A point on the line
                a = np.array([p1["x"], p1["y"]], dtype=np.float32)
                b = np.array([p2["x"], p2["y"]], dtype=np.float32)
                
                edge_vec = b - a
                n1 = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32)
                n2 = np.array([edge_vec[1], -edge_vec[0]], dtype=np.float32)
                
                centroid = np.mean([[p["x"], p["y"]] for p in polygon], axis=0)
                midpoint = (a + b) / 2.0
                to_midpoint = midpoint - centroid
                
                # Choose the one that points AWAY from centroid
                if np.dot(n1, to_midpoint) > 0:
                    normal = n1
                else:
                    normal = n2
                    
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-5:
                    normal /= norm_len
                    
                v = np.array([vx, vy], dtype=np.float32)
                speed_towards = np.dot(v, normal)
                
                # if vehicle is moving towards the handoff edge
                if speed_towards > 0.1:
                    pos = np.array([track.cx, track.cy], dtype=np.float32)
                    v_cross_edge = v[0] * edge_vec[1] - v[1] * edge_vec[0]
                    if abs(v_cross_edge) > 1e-5:
                        t = ((a[0] - pos[0]) * edge_vec[1] - (a[1] - pos[1]) * edge_vec[0]) / v_cross_edge
                        # print(f"[DEBUG _outward] cam={cam_id} track={getattr(track, 'id', '?')} t={t:.2f} speed={speed_towards:.2f} lookahead={self.lookahead_frames}")
                        if t > -5.0: # Allow slight overshoot
                            options.append((max(0.0, t), str(handoff_edge_index + 1), (vx, vy)))
                # else:
                #     print(f"[DEBUG _outward] cam={cam_id} track={getattr(track, 'id', '?')} rejected speed_towards={speed_towards:.2f}")
        else:
            if vx < -1.0:
                options.append((max(0.0, track.cx) / abs(vx), "left", (vx, vy)))
            if vx > 1.0:
                options.append((max(0.0, width - track.cx) / abs(vx), "right", (vx, vy)))
            if vy < -1.0:
                options.append((max(0.0, track.cy) / abs(vy), "top", (vx, vy)))
            if vy > 1.0:
                options.append((max(0.0, height - track.cy) / abs(vy), "bottom", (vx, vy)))
                
        if not options:
            return None
        time_to_edge, edge, velocity = min(options, key=lambda item: item[0])
        if time_to_edge > self.lookahead_frames:
            return None
        return edge, velocity

    @staticmethod
    def _appearance_distance(left, right) -> float:
        return compare_tracklets(left, right).distance

    @staticmethod
    def _size_distance(first: Tuple[int, int], second: Tuple[int, int]) -> float:
        fw, fh = first
        sw, sh = second
        ratio_w = min(fw, sw) / max(fw, sw, 1)
        ratio_h = min(fh, sh) / max(fh, sh, 1)
        return 1.0 - ratio_w * ratio_h

    def _upsert_handoff(self, cam_id: str, local_track_id: int, track, frame_idx: int) -> None:
        global_id = self._local_to_global.get((cam_id, local_track_id))
        if global_id is None:
            return
        world = self._track_world(cam_id, track)
        velocity_world = self._world_velocity(cam_id, track)
        target_cam = None
        edge = None
        # In a partial-view setup, the overlap is the actual transfer zone.
        # Prefer it over an image edge, whose direction changes with camera
        # perspective and may not be "left/right" in pixel coordinates.
        if np.hypot(*velocity_world) >= 0.20:
            adjacent_targets = {
                target
                for (source, _source_edge), target in self.edge_adjacency.items()
                if source == cam_id
            }
            for candidate_target in adjacent_targets:
                region = self.overlap_regions.get((cam_id, candidate_target))
                if region is None:
                    region = self.overlap_regions.get((candidate_target, cam_id))
                if region is not None and cv2.pointPolygonTest(region, world, False) >= 0:
                    target_cam = candidate_target
                    edge = "overlap"
                    break
        if target_cam is None:
            exit_info = self._outward_edge(cam_id, track)
            if exit_info is None:
                return
            edge, _velocity = exit_info
            target_cam = self.edge_adjacency.get((cam_id, edge))
            if target_cam is None:
                return
        identity = self._identities.get(self._canonical_id(global_id))
        gallery = merge_appearance_samples(
            identity.appearance_samples if identity is not None else (),
            track,
            self.tracklet_gallery_size,
        )
        appearance = aggregate_appearance(gallery)
        for entry in self._handoffs:
            if entry.global_id == global_id and entry.source_cam == cam_id and entry.target_cam == target_cam:
                entry.last_world = world
                entry.velocity_world = velocity_world
                entry.bbox_size = (track.w, track.h)
                entry.appearance = appearance
                entry.appearance_samples = gallery
                entry.updated_at_frame = frame_idx
                return
        self._handoffs.append(HandoffEntry(
            global_id=global_id, source_cam=cam_id, source_local_track_id=local_track_id,
            target_cam=target_cam, exit_edge=edge, last_world=world,
            velocity_world=velocity_world, bbox_size=(track.w, track.h),
            appearance=appearance, appearance_samples=gallery,
            created_at_frame=frame_idx,
            updated_at_frame=frame_idx,
        ))
        print(
            f"\033[93m  [handoff opened] global #{global_id}: "
            f"{cam_id} -> {target_cam} (edge {edge})\033[0m"
        )
        self._event("handoff_opened", frame_idx, global_id, source_camera=cam_id,
                    target_camera=target_cam, edge=edge, velocity={"x": round(velocity_world[0], 2), "y": round(velocity_world[1], 2)})

    def _predicted_world(self, entry: HandoffEntry, frame_idx: int) -> Tuple[float, float]:
        elapsed = max(0, frame_idx - entry.updated_at_frame)
        return (
            entry.last_world[0] + entry.velocity_world[0] * elapsed,
            entry.last_world[1] + entry.velocity_world[1] * elapsed,
        )

    def _entry_depth(self, cam_id: str, track, edge: str) -> float:
        width, height = self.camera_sizes[cam_id]
        if cam_id in self.custom_masks:
            mask_data = self.custom_masks[cam_id]
            polygon = mask_data["polygon"]
            try:
                edge_index = int(edge) - 1
            except ValueError:
                return float("inf")
            if 0 <= edge_index < len(polygon):
                p1 = polygon[edge_index]
                p2 = polygon[(edge_index + 1) % len(polygon)]
                a = np.array([p1["x"], p1["y"]], dtype=np.float32)
                b = np.array([p2["x"], p2["y"]], dtype=np.float32)
                edge_vec = b - a
                # Two possible normals: (-dy, dx) and (dy, -dx)
                n1 = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float32)
                n2 = np.array([edge_vec[1], -edge_vec[0]], dtype=np.float32)
                
                centroid = np.mean([[p["x"], p["y"]] for p in polygon], axis=0)
                midpoint = (a + b) / 2.0
                to_centroid = centroid - midpoint
                
                # Choose the one that points towards centroid
                if np.dot(n1, to_centroid) > 0:
                    normal = n1
                else:
                    normal = n2
                    
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-5:
                    normal /= norm_len
                pos = np.array([track.cx, track.cy], dtype=np.float32)
                # entry depth is distance from edge into the polygon along the normal
                return float(np.dot(pos - a, normal))
                
        if edge == "left":
            return float(track.cx)
        if edge == "right":
            return float(width - track.cx)
        if edge == "top":
            return float(track.cy)
        return float(height - track.cy)

    def _direction_cosine(self, cam_id: str, track, expected_velocity: Tuple[float, float]) -> Optional[float]:
        vx, vy = self._world_velocity(cam_id, track)
        source_norm = float(np.hypot(*expected_velocity))
        target_norm = float(np.hypot(vx, vy))
        if source_norm < 1.0 or target_norm < 1.0:
            return None
        return float((vx * expected_velocity[0] + vy * expected_velocity[1]) / (source_norm * target_norm))

    def _candidate_cost(self, entry: HandoffEntry, cam_id: str, track, frame_idx: int) -> Tuple[Optional[float], str, dict]:
        """Return a conservative handoff cost, otherwise its rejection reason."""
        overlap_handoff = entry.exit_edge == "overlap"
        if overlap_handoff:
            target_edge = "overlap"
        elif cam_id in self.custom_masks:
            target_edge = str(self.custom_masks[cam_id]["handoff_edge"])
        else:
            target_edge = OPPOSITE_EDGE.get(entry.exit_edge, "unknown")
            if target_edge == "unknown":
                return None, "invalid_edge", {}
        predicted = self._predicted_world(entry, frame_idx)
        world = self._track_world(cam_id, track)
        residual = float(np.linalg.norm(np.subtract(world, predicted)))
        speed = float(np.hypot(*entry.velocity_world))
        if overlap_handoff:
            region = self.overlap_regions.get((entry.source_cam, cam_id))
            if region is None:
                region = self.overlap_regions.get((cam_id, entry.source_cam))
            signed_overlap_distance = (
                cv2.pointPolygonTest(region, world, True)
                if region is not None
                else float("-inf")
            )
            depth = max(0.0, -float(signed_overlap_distance))
            details = {
                "predicted_distance": round(residual, 2),
                "overlap_signed_distance": round(float(signed_overlap_distance), 2),
            }
            if signed_overlap_distance < -self.prediction_radius:
                return None, "outside_overlap", details
        else:
            depth = self._entry_depth(cam_id, track, target_edge)
            # A one-frame tentative observation has no target velocity yet. It
            # may still match only near the entry edge and predicted point.
            entry_limit = (
                self.edge_margin
                + self.prediction_radius
                + speed * min(4, self.lookahead_frames) * 0.25
            )
            details = {
                "predicted_distance": round(residual, 2),
                "entry_depth": round(depth, 2),
            }
            if depth > entry_limit:
                return None, "outside_entry_corridor", details
        if residual > self.prediction_radius:
            return None, "prediction_distance", details
        appearance_match = compare_tracklets(
            track, entry.appearance_samples or entry.appearance
        )
        appearance_distance = appearance_match.distance
        details["appearance_distance"] = round(appearance_distance, 3)
        details["tracklet_support"] = appearance_match.support
        details["tracklet_sample_pairs"] = appearance_match.sample_pairs
        if appearance_match.support <= 0:
            return None, "appearance_missing", details
        if appearance_distance > self.appearance_threshold:
            return None, "appearance", details
        size_distance = self._size_distance((track.w, track.h), entry.bbox_size)
        details["size_distance"] = round(size_distance, 3)
        # Foreground blobs can be clipped differently at each crop boundary;
        # size remains a scoring signal, but should only reject an implausibly
        # different candidate after position, direction and appearance agree.
        if size_distance > 0.90:
            return None, "size", details
        direction = self._direction_cosine(cam_id, track, entry.velocity_world)
        if direction is not None:
            details["direction_cosine"] = round(direction, 3)
            if direction < self.min_direction_cosine:
                return None, "direction", details
            direction_cost = (1.0 - direction) * 0.5
        else:
            direction_cost = 0.20
        cost = 0.55 * (residual / self.prediction_radius) + 0.30 * appearance_distance + 0.10 * size_distance + 0.05 * direction_cost
        return (float(cost) if cost <= 0.92 else None), ("score" if cost > 0.92 else "ok"), details

    def _record_rejection(self, entry: HandoffEntry, frame_idx: int, cam_id: str, local_track_id: int, reason: str, details: dict) -> None:
        # Diagnostics must be useful without filling the registry with the same
        # failed candidate every frame.
        if frame_idx - entry.last_rejection_frame < 8:
            return
        entry.last_rejection_frame = frame_idx
        self._event("handoff_rejected", frame_idx, entry.global_id, source_camera=entry.source_cam,
                    target_camera=cam_id, target_local_id=local_track_id, reason=reason, **details)

    def _match_pending_handoffs(self, all_tracks: Dict[str, dict], frame_idx: int) -> None:
        """Solve all unbound target observations against pending IDs at once."""
        entries = [entry for entry in self._handoffs if frame_idx - entry.updated_at_frame <= self.handoff_ttl]
        candidates = [
            (cam_id, local_id, track)
            for cam_id, tracks in all_tracks.items()
            for local_id, track in tracks.items()
            if (cam_id, local_id) not in self._local_to_global
        ]
        if not entries or not candidates:
            return
        invalid_cost = 10.0
        costs = np.full((len(entries), len(candidates)), invalid_cost, dtype=np.float64)
        details_by_pair = {}
        for row, entry in enumerate(entries):
            for col, (cam_id, local_id, track) in enumerate(candidates):
                if entry.target_cam != cam_id:
                    continue
                cost, reason, details = self._candidate_cost(entry, cam_id, track, frame_idx)
                if cost is None:
                    self._record_rejection(entry, frame_idx, cam_id, local_id, reason, details)
                    continue
                costs[row, col] = cost
                details_by_pair[(row, col)] = details
        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.92)
        accepted_entries = []
        for row, col in enumerate(row_to_col):
            if col < 0 or costs[row, col] >= invalid_cost:
                continue
            entry = entries[row]
            cam_id, local_id, _ = candidates[col]
            match_details = details_by_pair.get((row, col), {})
            self._bind(cam_id, local_id, entry.global_id)
            self._event("handoff_matched", frame_idx, entry.global_id, source_camera=entry.source_cam,
                        target_camera=cam_id, source_local_id=entry.source_local_track_id,
                        target_local_id=local_id, score=round(float(costs[row, col]), 3),
                        appearance_distance=match_details.get("appearance_distance"),
                        tracklet_support=match_details.get("tracklet_support", 0),
                        predicted_position={"x": round(self._predicted_world(entry, frame_idx)[0], 2), "y": round(self._predicted_world(entry, frame_idx)[1], 2)})
            print(f"  [handoff] global #{entry.global_id}: {entry.source_cam} -> {cam_id}")
            accepted_entries.append(entry)
        for entry in accepted_entries:
            self._handoffs.remove(entry)

    def _world_is_in_overlap(self, cam_id: str, other_cam: str, world: Tuple[float, float]) -> bool:
        region = self.overlap_regions.get((cam_id, other_cam))
        if region is None:
            region = self.overlap_regions.get((other_cam, cam_id))
        if region is not None:
            return cv2.pointPolygonTest(region, world, False) >= 0
        own_crop = self.camera_crops[cam_id]
        other_crop = self.camera_crops[other_cam]
        ix1, iy1 = max(own_crop[0], other_crop[0]), max(own_crop[1], other_crop[1])
        ix2, iy2 = min(own_crop[2], other_crop[2]), min(own_crop[3], other_crop[3])
        return (
            ix1 < ix2
            and iy1 < iy2
            and ix1 - self.edge_margin <= world[0] <= ix2 + self.edge_margin
            and iy1 - self.edge_margin <= world[1] <= iy2 + self.edge_margin
        )

    def _world_is_near_overlap(
        self,
        cam_id: str,
        other_cam: str,
        world: Tuple[float, float],
        margin: float,
    ) -> bool:
        """Return whether a shared-map point is in the transfer corridor.

        The active masks intentionally overlap only in a narrow strip.  A
        vehicle bbox centre can sit just outside that polygon while its body is
        still visible in both cameras, so duplicate reconciliation uses a
        small metric margin around the calibrated overlap.
        """
        region = self.overlap_regions.get((cam_id, other_cam))
        if region is None:
            region = self.overlap_regions.get((other_cam, cam_id))
        if region is None:
            return False
        signed_distance = cv2.pointPolygonTest(
            region,
            (float(world[0]), float(world[1])),
            True,
        )
        return float(signed_distance) >= -float(margin)

    def _cross_camera_pair_evidence(
        self,
        first_track,
        second_track,
        distance: float,
    ) -> Tuple[bool, object, float, bool, float]:
        """Evaluate appearance/size after a unique spatial pair is known."""
        appearance_match = compare_tracklets(first_track, second_track)
        appearance_threshold = self.appearance_threshold
        adaptive_appearance = False
        if (
            appearance_match.distance > self.appearance_threshold
            and distance <= self.strong_spatial_distance
        ):
            appearance_threshold = max(
                appearance_threshold,
                self.relaxed_appearance_threshold,
            )
            adaptive_appearance = True
        size_distance = self._size_distance(
            (first_track.w, first_track.h),
            (second_track.w, second_track.h),
        )
        accepted = (
            appearance_match.support > 0
            and appearance_match.distance <= appearance_threshold
            and size_distance <= 0.90
            and (not adaptive_appearance or appearance_match.support >= 2)
        )
        return (
            accepted,
            appearance_match,
            appearance_threshold,
            adaptive_appearance,
            size_distance,
        )

    def _match_unique_unbound_cross_camera_tracks(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
    ) -> set[Tuple[str, int]]:
        """Bind or briefly defer a unique overlap candidate before ID creation.

        A short graph-tracklet collection window is preferable to showing a
        wrong new ID for a few frames.  Only a confirmed, mutually unique and
        spatially close source/target pair can be deferred; unrelated tracks
        elsewhere receive an ID immediately as before.
        """
        if not self.camera_transforms or not self.overlap_regions:
            return set()

        bound: Dict[Tuple[str, int], Tuple[int, object, Tuple[float, float]]] = {}
        unbound: Dict[Tuple[str, int], Tuple[object, Tuple[float, float]]] = {}
        present_unbound_keys = set()
        for cam_id, tracks in all_tracks.items():
            for local_id, track in tracks.items():
                key = (cam_id, local_id)
                global_id = self._local_to_global.get(key)
                if global_id is None:
                    present_unbound_keys.add(key)
                    if self._is_confirmed(track):
                        unbound[key] = (track, self._track_world(cam_id, track))
                    continue
                if not self._is_confirmed(track):
                    continue
                global_id = self._canonical_id(global_id)
                graph_key = (cam_id, global_id)
                previous = bound.get(graph_key)
                area = float(getattr(track, "area", track.w * track.h))
                if previous is not None:
                    previous_area = float(
                        getattr(previous[1], "area", previous[1].w * previous[1].h)
                    )
                    if previous_area >= area:
                        continue
                bound[graph_key] = (
                    local_id,
                    track,
                    self._track_world(cam_id, track),
                )

        # Drop state for local tracks that disappeared or were bound elsewhere.
        self._cross_camera_deferred_since = {
            key: started
            for key, started in self._cross_camera_deferred_since.items()
            if key in present_unbound_keys
        }
        if not bound or not unbound:
            return set()

        bound_gid_cameras: Dict[int, set[str]] = {}
        for bound_cam, bound_global_id in bound:
            bound_gid_cameras.setdefault(bound_global_id, set()).add(bound_cam)

        unbound_neighbours: Dict[Tuple[str, int], set[Tuple[str, int]]] = {
            key: set() for key in unbound
        }
        bound_neighbours: Dict[Tuple[str, int], set[Tuple[str, int]]] = {
            key: set() for key in bound
        }
        distances = {}
        for unbound_key, (_track, world) in unbound.items():
            cam_id, _local_id = unbound_key
            for bound_key, (_other_local_id, _other_track, other_world) in bound.items():
                other_cam, bound_global_id = bound_key
                if len(bound_gid_cameras.get(bound_global_id, ())) != 1:
                    continue
                if not self._are_adjacent(cam_id, other_cam):
                    continue
                if not self._world_is_near_overlap(
                    cam_id,
                    other_cam,
                    world,
                    self.cross_camera_duplicate_distance,
                ):
                    continue
                if not self._world_is_near_overlap(
                    other_cam,
                    cam_id,
                    other_world,
                    self.cross_camera_duplicate_distance,
                ):
                    continue
                distance = float(np.linalg.norm(np.subtract(world, other_world)))
                if distance > self.cross_camera_duplicate_distance:
                    continue
                unbound_neighbours[unbound_key].add(bound_key)
                bound_neighbours[bound_key].add(unbound_key)
                distances[(unbound_key, bound_key)] = distance

        deferred = set()
        for unbound_key, linked in unbound_neighbours.items():
            if len(linked) != 1:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                continue
            bound_key = next(iter(linked))
            if bound_neighbours.get(bound_key) != {unbound_key}:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                continue
            distance = distances[(unbound_key, bound_key)]
            track, _world = unbound[unbound_key]
            other_local_id, other_track, _other_world = bound[bound_key]
            (
                accepted,
                appearance_match,
                appearance_threshold,
                adaptive_appearance,
                size_distance,
            ) = self._cross_camera_pair_evidence(track, other_track, distance)
            cam_id, local_id = unbound_key
            other_cam, global_id = bound_key
            if accepted:
                self._bind(cam_id, local_id, global_id)
                self._cross_camera_deferred_since.pop(unbound_key, None)
                self._event(
                    "cross_camera_unbound_matched",
                    frame_idx,
                    global_id,
                    source_camera=other_cam,
                    target_camera=cam_id,
                    source_local_id=other_local_id,
                    target_local_id=local_id,
                    world_distance=round(distance, 3),
                    appearance_distance=round(appearance_match.distance, 3),
                    appearance_threshold=round(appearance_threshold, 3),
                    adaptive_appearance=adaptive_appearance,
                    tracklet_support=appearance_match.support,
                    tracklet_sample_pairs=appearance_match.sample_pairs,
                    size_distance=round(size_distance, 3),
                )
                continue

            # Size incompatibility is evidence of a different object, not a
            # reason to delay its ID.  Appearance alone benefits from a few
            # additional tracklet samples before making an irreversible choice.
            if size_distance > 0.90 or self.cross_camera_defer_frames <= 0:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                continue
            started = self._cross_camera_deferred_since.setdefault(
                unbound_key,
                frame_idx,
            )
            if frame_idx - started >= self.cross_camera_defer_frames:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                continue
            deferred.add(unbound_key)
            if frame_idx == started:
                self._event(
                    "cross_camera_assignment_deferred",
                    frame_idx,
                    global_id,
                    source_camera=other_cam,
                    target_camera=cam_id,
                    target_local_id=local_id,
                    world_distance=round(distance, 3),
                    appearance_distance=round(appearance_match.distance, 3),
                    appearance_threshold=round(appearance_threshold, 3),
                    defer_frames=self.cross_camera_defer_frames,
                )
        return deferred

    def _identity_elapsed(
        self,
        identity: GlobalIdentityState,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> Tuple[float, bool]:
        if timestamp_s is not None and identity.last_seen_time is not None:
            return max(0.0, timestamp_s - identity.last_seen_time), True
        return float(max(0, frame_idx - identity.last_seen_frame)), False

    def _identity_is_recent(
        self,
        identity: GlobalIdentityState,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> bool:
        elapsed, uses_seconds = self._identity_elapsed(identity, frame_idx, timestamp_s)
        limit = self.identity_retention_seconds if uses_seconds else self.identity_retention_frames
        return elapsed <= limit

    def _predicted_identity_world(
        self,
        identity: GlobalIdentityState,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> Tuple[float, float]:
        elapsed, uses_seconds = self._identity_elapsed(identity, frame_idx, timestamp_s)
        velocity = (
            identity.velocity_world_per_second
            if uses_seconds and np.hypot(*identity.velocity_world_per_second) > 1e-6
            else identity.velocity_world
        )
        return (
            identity.last_world[0] + velocity[0] * elapsed,
            identity.last_world[1] + velocity[1] * elapsed,
        )

    def _match_dormant_identities(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
        camera_timestamps_s: Optional[Dict[str, float]],
    ) -> set[Tuple[str, int]]:
        """Recover mature fragments, deferring plausible one-frame matches.

        The returned keys look like the dormant identity but do not yet have
        enough current-fragment evidence.  Callers must keep them out of all
        downstream automatic assignment paths for this frame; otherwise a
        safety rejection here would merely turn into a duplicate new GID.
        """
        candidates = [
            (cam_id, local_id, track)
            for cam_id, tracks in all_tracks.items()
            for local_id, track in tracks.items()
            if (cam_id, local_id) not in self._local_to_global
        ]
        identities = [
            identity
            for identity in self._identities.values()
            if identity.state in {"dormant", "handoff"}
        ]
        if not candidates or not identities:
            return set()

        invalid_cost = 10.0
        costs = np.full((len(identities), len(candidates)), invalid_cost, dtype=np.float64)
        details_by_pair = {}
        deferred_keys: set[Tuple[str, int]] = set()
        for row, identity in enumerate(identities):
            for col, (cam_id, local_id, track) in enumerate(candidates):
                timestamp_s = (camera_timestamps_s or {}).get(cam_id)
                same_camera = identity.last_camera == cam_id
                if not same_camera and not self._are_adjacent(identity.last_camera, cam_id):
                    continue
                if not self._identity_is_recent(identity, frame_idx, timestamp_s):
                    continue
                # After a vehicle parks, its next local track starts close to
                # the last stationary position. Extrapolating the velocity
                # across that dormant interval would move the gate away from
                # the real car, especially for a motion-only tracker.
                predicted = (
                    identity.last_world
                    if same_camera
                    else self._predicted_identity_world(identity, frame_idx, timestamp_s)
                )
                world = self._track_world(cam_id, track)
                distance = float(np.linalg.norm(np.subtract(world, predicted)))
                elapsed, uses_seconds = self._identity_elapsed(identity, frame_idx, timestamp_s)
                velocity = (
                    identity.velocity_world_per_second
                    if uses_seconds and np.hypot(*identity.velocity_world_per_second) > 1e-6
                    else identity.velocity_world
                )
                distance_limit = (
                    self.dormant_match_distance
                    if same_camera
                    else min(
                        self.dormant_match_distance * 2.0,
                        self.dormant_match_distance
                        + np.hypot(*velocity) * min(elapsed, 2.0) * 0.25,
                    )
                )
                if distance > distance_limit:
                    continue
                appearance_match = compare_tracklets(
                    track, identity.appearance_samples or identity.appearance
                )
                appearance = appearance_match.distance
                if appearance_match.support <= 0:
                    continue
                appearance_threshold = (
                    min(0.55, self.dormant_appearance_threshold)
                    if same_camera
                    else self.dormant_appearance_threshold
                )
                if appearance > appearance_threshold:
                    continue
                size = self._size_distance((track.w, track.h), identity.bbox_size)
                if size > 0.92:
                    continue
                if not self._dormant_reid_ready(track, frame_idx):
                    deferred_keys.add((cam_id, local_id))
                    continue
                if same_camera:
                    direction_cost = 0.0
                    cost = (
                        0.55 * distance / max(distance_limit, 1.0)
                        + 0.35 * appearance / max(appearance_threshold, 1e-6)
                        + 0.10 * size
                    )
                else:
                    direction = self._direction_cosine(cam_id, track, velocity)
                    if direction is not None and direction < -0.35:
                        continue
                    direction_cost = 0.20 if direction is None else (1.0 - direction) * 0.5
                    cost = (
                        0.60 * distance / max(distance_limit, 1.0)
                        + 0.25 * appearance / max(appearance_threshold, 1e-6)
                        + 0.10 * size
                        + 0.05 * direction_cost
                    )
                costs[row, col] = cost
                details_by_pair[(row, col)] = (
                    distance, appearance, elapsed, appearance_match.support
                )

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.95)
        for row, col in enumerate(row_to_col):
            if col < 0 or costs[row, col] >= invalid_cost:
                continue
            identity = identities[row]
            cam_id, local_id, _track = candidates[col]
            self._bind(cam_id, local_id, identity.global_id)
            identity.state = "active"
            identity.dormant_since_frame = None
            identity.dormant_since_time = None
            distance, appearance, elapsed, tracklet_support = details_by_pair[(row, col)]
            self._event(
                "dormant_global_id_recovered", frame_idx, identity.global_id,
                source_camera=identity.last_camera, target_camera=cam_id,
                target_local_id=local_id, predicted_distance=round(distance, 2),
                appearance_distance=round(appearance, 3), elapsed=round(elapsed, 3),
                tracklet_support=tracklet_support,
            )
            self._handoffs = [
                entry for entry in self._handoffs
                if self._canonical_id(entry.global_id) != identity.global_id
            ]
            print(
                f"  [re-id] global #{identity.global_id}: "
                f"{identity.last_camera} -> {cam_id}"
            )
        return deferred_keys

    def _match_unbound_cross_camera_pairs(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
    ) -> None:
        """Allocate one ID when matching tracks first appear in two views together."""
        observations = [
            (cam_id, local_id, track)
            for cam_id, tracks in all_tracks.items()
            for local_id, track in tracks.items()
            if (cam_id, local_id) not in self._local_to_global
        ]
        pair_costs = []
        for left_index, (left_cam, left_local_id, left_track) in enumerate(observations):
            left_world = self._track_world(left_cam, left_track)
            for right_index in range(left_index + 1, len(observations)):
                right_cam, right_local_id, right_track = observations[right_index]
                if not self._are_adjacent(left_cam, right_cam):
                    continue
                if not (self._is_confirmed(left_track) or self._is_confirmed(right_track)):
                    continue
                right_world = self._track_world(right_cam, right_track)
                if not self._world_is_in_overlap(left_cam, right_cam, left_world):
                    continue
                if not self._world_is_in_overlap(right_cam, left_cam, right_world):
                    continue
                distance = float(np.linalg.norm(np.subtract(left_world, right_world)))
                if distance > self.match_distance:
                    continue
                appearance_match = compare_tracklets(left_track, right_track)
                appearance = appearance_match.distance
                if (
                    appearance_match.support <= 0
                    or appearance > self.appearance_threshold
                ):
                    continue
                cost = distance / max(self.match_distance, 1.0) + appearance
                pair_costs.append((cost, left_index, right_index))

        consumed = set()
        for _cost, left_index, right_index in sorted(pair_costs):
            if left_index in consumed or right_index in consumed:
                continue
            left_cam, left_local_id, _left_track = observations[left_index]
            right_cam, right_local_id, _right_track = observations[right_index]
            if (
                (left_cam, left_local_id) in self._local_to_global
                or (right_cam, right_local_id) in self._local_to_global
            ):
                continue
            global_id = self._allocate_global_id()
            self._bind(left_cam, left_local_id, global_id)
            self._bind(right_cam, right_local_id, global_id)
            consumed.update((left_index, right_index))
            self._event(
                "simultaneous_tracks_grouped", frame_idx, global_id,
                cameras=[left_cam, right_cam],
                local_track_ids=[left_local_id, right_local_id],
            )

    def _match_simultaneous_overlap(self, cam_id: str, local_track_id: int, track, all_tracks: Dict[str, dict]) -> Optional[int]:
        """Deduplicate one car seen in two overlapping views."""
        world = self._track_world(cam_id, track)
        for other_cam, other_tracks in all_tracks.items():
            if other_cam == cam_id:
                continue
            if not self._are_adjacent(cam_id, other_cam):
                continue
            if not self._world_is_in_overlap(cam_id, other_cam, world):
                continue
            for other_local_id, other_track in other_tracks.items():
                global_id = self._local_to_global.get((other_cam, other_local_id))
                if global_id is None:
                    continue
                other_world = self._track_world(other_cam, other_track)
                if np.linalg.norm(np.subtract(world, other_world)) > self.match_distance:
                    continue
                appearance_match = compare_tracklets(track, other_track)
                if (
                    appearance_match.support > 0
                    and appearance_match.distance <= self.appearance_threshold
                ):
                    self._bind(cam_id, local_track_id, global_id)
                    return global_id
        return None

    def _same_camera_motion_duplicate(self, first, second) -> bool:
        """Detect nearby boxes that are two observations of one slow/fast car."""
        first_box = (first.x, first.y, first.w, first.h)
        second_box = (second.x, second.y, second.w, second.h)
        first_area = max(1, first.w * first.h)
        second_area = max(1, second.w * second.h)
        if min(first_area, second_area) / max(first_area, second_area) < 0.30:
            return False
        appearance = self._appearance_distance(first, second)
        if appearance > self.appearance_threshold:
            return False
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
        gap_y = max(ay - (by + bh), by - (ay + ah), 0)
        # Slow vehicles often produce two slightly separated foreground boxes.
        # Merge them while both boxes are still visible, before either ID wins.
        if np.hypot(gap_x, gap_y) > 30.0:
            return False
        distance = float(np.linalg.norm(np.subtract((first.cx, first.cy), (second.cx, second.cy))))
        largest_dimension = max(first.w, first.h, second.w, second.h)
        if distance > max(40.0, 1.60 * largest_dimension):
            return False
        first_velocity = self._velocity(first)
        second_velocity = self._velocity(second)
        first_norm, second_norm = float(np.hypot(*first_velocity)), float(np.hypot(*second_velocity))
        # For a slow car one/both velocity estimates are often near zero.  In
        # that case proximity is the stronger signal.  Reject only clearly
        # opposing trajectories when both directions are measurable.
        if first_norm >= 1.0 and second_norm >= 1.0:
            cosine = float(np.dot(first_velocity, second_velocity) / (first_norm * second_norm))
            if cosine < 0.20:
                return False
        return True

    def _match_same_camera_duplicate(self, cam_id: str, local_track_id: int, track, all_tracks: Dict[str, dict], frame_idx: int) -> Optional[int]:
        """Attach a second motion-echo local track to the existing global ID."""
        for other_local_id, other_track in all_tracks.get(cam_id, {}).items():
            if other_local_id == local_track_id:
                continue
            global_id = self._local_to_global.get((cam_id, other_local_id))
            if global_id is None or not self._same_camera_motion_duplicate(track, other_track):
                continue
            self._bind(cam_id, local_track_id, global_id)
            self._event("same_camera_duplicate_merged", frame_idx, global_id, camera=cam_id,
                        kept_local_id=other_local_id, merged_local_id=local_track_id)
            return global_id
        return None

    def _merge_all_nearby_active_duplicates(self, all_tracks: Dict[str, dict], frame_idx: int) -> None:
        """Merge already-assigned nearby boxes immediately; smaller ID wins."""
        for cam_id, tracks in all_tracks.items():
            items = list(tracks.items())
            for index, (left_local_id, left_track) in enumerate(items):
                left_global_id = self._local_to_global.get((cam_id, left_local_id))
                if left_global_id is None:
                    continue
                for right_local_id, right_track in items[index + 1:]:
                    right_global_id = self._local_to_global.get((cam_id, right_local_id))
                    if right_global_id is None:
                        continue
                    left_global_id = self._canonical_id(left_global_id)
                    right_global_id = self._canonical_id(right_global_id)
                    if left_global_id == right_global_id:
                        continue
                    if not self._same_camera_motion_duplicate(left_track, right_track):
                        continue
                    kept_id, retired_id = min(left_global_id, right_global_id), max(left_global_id, right_global_id)
                    self._merge_global_ids(kept_id, retired_id, frame_idx, "nearby_boxes_same_camera")
                    left_global_id = kept_id

    def _merge_unique_cross_camera_duplicates(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
    ) -> None:
        """Reconcile two already-issued IDs for one cross-camera vehicle.

        A target local track can become confirmed just before the source opens
        its handoff.  Older code then allocated a new Global ID and never
        reconsidered it because handoff matching only examines unbound tracks.
        This pass treats active observations as graph nodes and merges only a
        mutually unique pair in the calibrated overlap corridor.

        Appearance is relaxed for a *very* tight spatial match.  That is
        necessary for opposing camera views, where colour/shape histograms are
        naturally less similar, while mutual uniqueness keeps the relaxed
        threshold from joining two nearby vehicles in a crowded transfer area.
        """
        if not self.camera_transforms or not self.overlap_regions:
            return

        # One strongest observation per (camera, canonical Global ID) avoids a
        # same-camera motion echo creating artificial graph ambiguity.
        observations: Dict[Tuple[str, int], Tuple[int, object, Tuple[float, float]]] = {}
        for cam_id, tracks in all_tracks.items():
            for local_id, track in tracks.items():
                if not self._is_confirmed(track):
                    continue
                global_id = self._local_to_global.get((cam_id, local_id))
                if global_id is None:
                    continue
                global_id = self._canonical_id(global_id)
                key = (cam_id, global_id)
                area = float(getattr(track, "area", track.w * track.h))
                previous = observations.get(key)
                if previous is not None:
                    previous_area = float(
                        getattr(previous[1], "area", previous[1].w * previous[1].h)
                    )
                    if previous_area >= area:
                        continue
                observations[key] = (
                    local_id,
                    track,
                    self._track_world(cam_id, track),
                )

        keys = list(observations)
        gid_cameras: Dict[int, set[str]] = {}
        for observation_cam, observation_global_id in keys:
            gid_cameras.setdefault(observation_global_id, set()).add(observation_cam)
        neighbours: Dict[Tuple[str, int], set[Tuple[str, int]]] = {
            key: set() for key in keys
        }
        pair_distance: Dict[frozenset, float] = {}
        for index, left_key in enumerate(keys):
            left_cam, left_global_id = left_key
            _left_local_id, _left_track, left_world = observations[left_key]
            for right_key in keys[index + 1:]:
                right_cam, right_global_id = right_key
                if left_cam == right_cam or left_global_id == right_global_id:
                    continue
                # Reconciliation is only for a transfer where each candidate
                # GID currently belongs to one camera. If either GID is already
                # represented in both views, cross-edges between two valid
                # multi-view vehicles would form a misleading unique pair.
                if (
                    len(gid_cameras.get(left_global_id, ())) != 1
                    or len(gid_cameras.get(right_global_id, ())) != 1
                ):
                    continue
                if not self._are_adjacent(left_cam, right_cam):
                    continue
                _right_local_id, _right_track, right_world = observations[right_key]
                if not self._world_is_near_overlap(
                    left_cam,
                    right_cam,
                    left_world,
                    self.cross_camera_duplicate_distance,
                ):
                    continue
                if not self._world_is_near_overlap(
                    right_cam,
                    left_cam,
                    right_world,
                    self.cross_camera_duplicate_distance,
                ):
                    continue
                distance = float(np.linalg.norm(np.subtract(left_world, right_world)))
                if distance > self.cross_camera_duplicate_distance:
                    continue
                neighbours[left_key].add(right_key)
                neighbours[right_key].add(left_key)
                pair_distance[frozenset((left_key, right_key))] = distance

        proposals = []
        visited_pairs = set()
        for left_key, linked in neighbours.items():
            if len(linked) != 1:
                continue
            right_key = next(iter(linked))
            if neighbours.get(right_key) != {left_key}:
                continue
            pair_key = frozenset((left_key, right_key))
            if pair_key in visited_pairs:
                continue
            visited_pairs.add(pair_key)
            distance = pair_distance[pair_key]
            left_local_id, left_track, _left_world = observations[left_key]
            right_local_id, right_track, _right_world = observations[right_key]
            (
                accepted,
                appearance_match,
                appearance_threshold,
                adaptive_appearance,
                size_distance,
            ) = self._cross_camera_pair_evidence(
                left_track,
                right_track,
                distance,
            )
            if not accepted:
                continue
            proposals.append((
                distance,
                left_key,
                right_key,
                left_local_id,
                right_local_id,
                appearance_match,
                appearance_threshold,
                adaptive_appearance,
                size_distance,
            ))

        for (
            distance,
            left_key,
            right_key,
            left_local_id,
            right_local_id,
            appearance_match,
            appearance_threshold,
            adaptive_appearance,
            size_distance,
        ) in sorted(proposals, key=lambda item: item[0]):
            left_cam, left_global_id = left_key
            right_cam, right_global_id = right_key
            left_global_id = self._canonical_id(left_global_id)
            right_global_id = self._canonical_id(right_global_id)
            if left_global_id == right_global_id:
                continue
            kept_id = min(left_global_id, right_global_id)
            retired_id = max(left_global_id, right_global_id)
            self._event(
                "cross_camera_duplicate_matched",
                frame_idx,
                kept_id,
                superseded_global_id=retired_id,
                source_camera=left_cam,
                target_camera=right_cam,
                source_local_id=left_local_id,
                target_local_id=right_local_id,
                world_distance=round(distance, 3),
                appearance_distance=round(appearance_match.distance, 3),
                appearance_threshold=round(appearance_threshold, 3),
                adaptive_appearance=adaptive_appearance,
                tracklet_support=appearance_match.support,
                tracklet_sample_pairs=appearance_match.sample_pairs,
                size_distance=round(size_distance, 3),
            )
            self._merge_global_ids(
                kept_id,
                retired_id,
                frame_idx,
                "unique_cross_camera_overlap",
            )

    def _refresh_identity_states(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
        camera_timestamps_s: Optional[Dict[str, float]],
    ) -> None:
        observations: Dict[int, List[Tuple[str, int, object]]] = {}
        for cam_id, tracks in all_tracks.items():
            for local_id, track in tracks.items():
                global_id = self._local_to_global.get((cam_id, local_id))
                if global_id is None:
                    continue
                global_id = self._canonical_id(global_id)
                observations.setdefault(global_id, []).append((cam_id, local_id, track))

        seen_ids = set()
        for global_id, candidates in observations.items():
            previous = self._identities.get(global_id)
            cam_id, local_id, track = max(
                candidates,
                key=lambda item: (
                    1 if previous is not None and item[0] == previous.last_camera else 0,
                    1 if self._is_confirmed(item[2]) else 0,
                    float(getattr(item[2], "area", item[2].w * item[2].h)),
                ),
            )
            timestamp_s = (camera_timestamps_s or {}).get(cam_id)
            identity = self._observe_identity(
                global_id, cam_id, local_id, track, frame_idx, timestamp_s
            )
            # Overlap can expose the same Global ID in multiple cameras. Keep
            # all views as graph-like appearance nodes even though only the
            # strongest observation supplies position and velocity.
            for other_cam_id, other_local_id, other_track in candidates:
                if other_cam_id == cam_id and other_local_id == local_id:
                    continue
                identity.appearance_samples = merge_appearance_samples(
                    identity.appearance_samples,
                    other_track,
                    self.tracklet_gallery_size,
                )
            identity.appearance = aggregate_appearance(identity.appearance_samples)
            seen_ids.add(global_id)

        pending_ids = {
            self._canonical_id(entry.global_id) for entry in self._handoffs
        }
        current_time_s = max((camera_timestamps_s or {}).values(), default=None)
        for global_id, identity in self._identities.items():
            if global_id in seen_ids or identity.state in {"exited", "expired"}:
                continue
            if identity.dormant_since_frame is None:
                identity.dormant_since_frame = frame_idx
                identity.dormant_since_time = current_time_s
            identity.state = "handoff" if global_id in pending_ids else "dormant"

    def update_all_tracks(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
        camera_timestamps_s: Optional[Dict[str, float]] = None,
        protected_local_keys: Optional[Iterable[Tuple[str, int]]] = None,
    ) -> Dict[str, Dict[int, int]]:
        """Assign global IDs for current observations from every camera.

        ``all_tracks`` may include tentative tracks.  They are allowed to
        receive an existing handoff ID, but only confirmed tracks may allocate
        a previously unseen global ID.

        ``protected_local_keys`` reserves currently-unbound local observations
        for an external, higher-confidence recovery step (for example, a
        vehicle leaving a known parking slot).  A reserved observation is
        excluded from every automatic assignment path for this call: handoff,
        dormant Re-ID, overlap grouping, duplicate matching, and new-ID
        allocation.  Reservations are deliberately call-scoped; omitting the
        key on a later call makes it eligible again.  A key already bound via
        :meth:`bind_external_id` remains active even when it is also listed as
        protected.
        """
        protected_keys = set(protected_local_keys or ())
        protected_unbound_keys = {
            (cam_id, local_track_id)
            for cam_id, tracks in all_tracks.items()
            for local_track_id in tracks
            if (
                (cam_id, local_track_id) in protected_keys
                and (cam_id, local_track_id) not in self._local_to_global
            )
        }
        assignable_tracks = {
            cam_id: {
                local_track_id: track
                for local_track_id, track in tracks.items()
                if (cam_id, local_track_id) not in protected_unbound_keys
            }
            for cam_id, tracks in all_tracks.items()
        }

        # Existing confirmed source tracks publish an early, velocity-aware handoff.
        for cam_id, tracks in all_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) in self._local_to_global and self._is_confirmed(track):
                    self._upsert_handoff(cam_id, local_track_id, track, frame_idx)

        self._match_pending_handoffs(assignable_tracks, frame_idx)

        # Resolve overlapping observations before allocating a new global ID.
        for cam_id, tracks in assignable_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) not in self._local_to_global:
                    self._match_simultaneous_overlap(
                        cam_id, local_track_id, track, assignable_tracks
                    )

        # Real streams can drop the exact boundary frame, so recover from the
        # durable identity registry before allocating any new ID.
        deferred_dormant = self._match_dormant_identities(
            assignable_tracks, frame_idx, camera_timestamps_s
        )
        post_dormant_tracks = {
            cam_id: {
                local_track_id: track
                for local_track_id, track in tracks.items()
                if (
                    (cam_id, local_track_id) not in deferred_dormant
                    or (cam_id, local_track_id) in self._local_to_global
                )
            }
            for cam_id, tracks in assignable_tracks.items()
        }

        # If matching observations first appear in both cameras together,
        # neither owns an ID yet. Group the pair before normal allocation.
        self._match_unbound_cross_camera_pairs(post_dormant_tracks, frame_idx)
        for cam_id, tracks in post_dormant_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) not in self._local_to_global:
                    self._match_simultaneous_overlap(
                        cam_id, local_track_id, track, post_dormant_tracks
                    )

        deferred_cross_camera = self._match_unique_unbound_cross_camera_tracks(
            post_dormant_tracks,
            frame_idx,
        )

        # Tentative tracks remain ID-less unless they consumed a handoff.
        for cam_id, tracks in post_dormant_tracks.items():
            for local_track_id, track in tracks.items():
                if (
                    (cam_id, local_track_id) in self._local_to_global
                    or (cam_id, local_track_id) in deferred_cross_camera
                    or not self._is_confirmed(track)
                ):
                    continue
                if self._match_same_camera_duplicate(
                    cam_id,
                    local_track_id,
                    track,
                    post_dormant_tracks,
                    frame_idx,
                ) is not None:
                    continue
                global_id = self._bind(cam_id, local_track_id, self._allocate_global_id())
                self._event("global_id_created", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)
                print(
                    f"  [new] global #{global_id} for {cam_id} "
                    f"(local #{local_track_id})"
                )

        self._merge_all_nearby_active_duplicates(all_tracks, frame_idx)
        self._merge_recently_lost_duplicates(all_tracks, frame_idx)
        self._merge_unique_cross_camera_duplicates(all_tracks, frame_idx)
        self._refresh_identity_states(
            all_tracks, frame_idx, camera_timestamps_s
        )

        current_time_s = max((camera_timestamps_s or {}).values(), default=None)
        self.cleanup(frame_idx, current_time_s)
        return {
            cam_id: {local_id: self._local_to_global[(cam_id, local_id)] for local_id in tracks if (cam_id, local_id) in self._local_to_global}
            for cam_id, tracks in all_tracks.items()
        }

    def notify_track_expired(
        self,
        cam_id: str,
        local_track_id: int,
        cx: int,
        cy: int,
        bbox_w: int,
        bbox_h: int,
        appearance: Optional[np.ndarray],
        frame_idx: int,
        timestamp_s: Optional[float] = None,
        appearance_tracklet=None,
    ) -> None:
        """Remove a local mapping while retaining its cross-camera identity."""
        key = (cam_id, local_track_id)
        global_id = self._local_to_global.pop(key, None)
        if global_id is not None:
            global_id = self._canonical_id(global_id)
            self._gid_members.get(global_id, set()).discard(key)
            identity = self._identities.get(global_id)
            appearance_source = (
                appearance_tracklet if appearance_tracklet is not None else appearance
            )
            if not appearance_samples(appearance_source):
                appearance_source = appearance
            if identity is None:
                gallery = self._tracklet_snapshot(appearance_source)
                identity = GlobalIdentityState(
                    global_id=global_id,
                    state="dormant",
                    last_camera=cam_id,
                    last_local_track_id=local_track_id,
                    last_world=self._expired_track_world(cam_id, cx, cy, bbox_h),
                    velocity_world=(0.0, 0.0),
                    velocity_world_per_second=(0.0, 0.0),
                    bbox_size=(bbox_w, bbox_h),
                    appearance=aggregate_appearance(gallery),
                    appearance_samples=gallery,
                    last_seen_frame=frame_idx,
                    last_seen_time=timestamp_s,
                    dormant_since_frame=frame_idx,
                    dormant_since_time=timestamp_s,
                )
                self._identities[global_id] = identity
            else:
                identity.appearance_samples = merge_appearance_samples(
                    identity.appearance_samples,
                    appearance_source,
                    self.tracklet_gallery_size,
                )
                identity.appearance = aggregate_appearance(identity.appearance_samples)
            if self._in_exit_zone(cam_id, (cx, cy)):
                self._mark_identity_exited(
                    global_id, cam_id, local_track_id, frame_idx, timestamp_s
                )
            elif identity.state != "exited":
                identity.state = "handoff" if any(
                    self._canonical_id(entry.global_id) == global_id
                    for entry in self._handoffs
                ) else "dormant"
                if identity.dormant_since_frame is None:
                    identity.dormant_since_frame = frame_idx
                    identity.dormant_since_time = timestamp_s
            self._event("local_track_expired", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)
            print(f"  [local lost] global #{global_id} at {cam_id}; identity retained")

    def cleanup(self, frame_idx: int, timestamp_s: Optional[float] = None) -> None:
        retained = []
        for entry in self._handoffs:
            if frame_idx - entry.updated_at_frame <= self.handoff_ttl:
                retained.append(entry)
                continue
            self._event("handoff_expired", frame_idx, entry.global_id, source_camera=entry.source_cam,
                        target_camera=entry.target_cam, source_local_id=entry.source_local_track_id)
        self._handoffs = retained
        for global_id, identity in self._identities.items():
            if identity.state not in {"dormant", "handoff"}:
                continue
            if self._identity_is_recent(identity, frame_idx, timestamp_s):
                continue
            identity.state = "expired"
            self._handoffs = [
                entry for entry in self._handoffs
                if self._canonical_id(entry.global_id) != global_id
            ]
            self._event(
                "global_identity_expired", frame_idx, global_id,
                last_camera=identity.last_camera,
            )

    def get_global_id(self, cam_id: str, local_track_id: int) -> Optional[int]:
        global_id = self._local_to_global.get((cam_id, local_track_id))
        return self._canonical_id(global_id) if global_id is not None else None

    def canonical_global_id(self, global_id: int) -> int:
        """Return the live canonical ID after any duplicate-ID merge."""
        return self._canonical_id(int(global_id))

    def to_json(self, all_tracks: Dict[str, dict]) -> dict:
        """Return active observations plus one deduplicated entry per global ID."""
        # Keep only the strongest local observation for one global ID in one
        # camera.  A merged motion echo must not shift the web-map position.
        per_camera_observation = {}
        for cam_id, tracks in all_tracks.items():
            for local_id, track in tracks.items():
                if not self._is_confirmed(track):
                    continue
                global_id = self._local_to_global.get((cam_id, local_id))
                if global_id is None:
                    continue
                global_id = self._canonical_id(global_id)
                key = (str(global_id), cam_id)
                local_anchor = self._track_local_anchor(cam_id, track)
                world_anchor = self._track_world(cam_id, track)
                observation = {
                    "camera_id": cam_id, "local_track_id": local_id,
                    "local_position": {"x": track.cx, "y": track.cy},
                    "shared_map_anchor": {
                        "x": round(local_anchor[0], 2),
                        "y": round(local_anchor[1], 2),
                        "reference": (
                            self.shared_map_anchor
                            if cam_id in self.camera_transforms
                            else "tracker_point"
                        ),
                    },
                    "global_position": {
                        "x": round(world_anchor[0], 2),
                        "y": round(world_anchor[1], 2),
                    },
                    "_area": float(getattr(track, "area", track.w * track.h)),
                }
                previous = per_camera_observation.get(key)
                if previous is None or observation["_area"] > previous["_area"]:
                    per_camera_observation[key] = observation
        active = {}
        for (global_id, _), observation in per_camera_observation.items():
            observation.pop("_area", None)
            entry = active.setdefault(global_id, {"global_id": int(global_id), "observations": []})
            entry["observations"].append(observation)
        map_vehicles = {}
        for global_id, entry in active.items():
            observations = entry["observations"]
            map_vehicles[global_id] = {
                "track_id": entry["global_id"],
                "position": {"x": round(sum(item["global_position"]["x"] for item in observations) / len(observations), 2), "y": round(sum(item["global_position"]["y"] for item in observations) / len(observations), 2), "reference": self.world_unit},
                "camera_ids": [item["camera_id"] for item in observations],
                "observation_count": len(observations),
            }
        identity_lifecycle = {
            str(global_id): {
                "global_id": global_id,
                "state": identity.state,
                "last_camera": identity.last_camera,
                "last_local_track_id": identity.last_local_track_id,
                "last_world": {
                    "x": round(identity.last_world[0], 2),
                    "y": round(identity.last_world[1], 2),
                },
                "velocity_world": {
                    "x": round(identity.velocity_world[0], 3),
                    "y": round(identity.velocity_world[1], 3),
                },
                "last_seen_frame": identity.last_seen_frame,
                "last_seen_time": identity.last_seen_time,
                "dormant_since_frame": identity.dormant_since_frame,
                "exited_at_frame": identity.exited_at_frame,
                "appearance_sample_count": len(identity.appearance_samples),
            }
            for global_id, identity in sorted(self._identities.items())
        }
        return {
            "world_unit": self.world_unit,
            "next_global_id": self._next_global_id,
            "retired_global_ids": {str(old_id): canonical_id for old_id, canonical_id in sorted(self._global_aliases.items())},
            "active_global_vehicles": active,
            "map_vehicles": map_vehicles,
            "identity_lifecycle": identity_lifecycle,
            "pending_handoffs": [{
                "global_id": item.global_id, "source_camera": item.source_cam,
                "source_local_track_id": item.source_local_track_id, "target_camera": item.target_cam,
                "exit_edge": item.exit_edge, "last_world": {"x": round(item.last_world[0], 2), "y": round(item.last_world[1], 2)},
                "velocity": {"x": round(item.velocity_world[0], 2), "y": round(item.velocity_world[1], 2)},
                "appearance_sample_count": len(item.appearance_samples),
                "created_at_frame": item.created_at_frame, "updated_at_frame": item.updated_at_frame,
            } for item in self._handoffs],
            "recent_events": self._events,
        }
