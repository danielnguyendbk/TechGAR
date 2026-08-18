"""Global vehicle IDs and predictive handoff between adjacent cameras.

Local tracker IDs belong to one camera only.  ``CrossCameraManager`` owns the
single global namespace and associates a new local observation with an
existing global vehicle before that observation is confirmed by its local
tracker.  This is important for fast vehicles: local confirmation can happen
several frames after the vehicle crossed a camera border.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from lap import lapjv


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
    lost_at_frame: int


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
        lookahead_frames: int = 16,
        prediction_radius: float = 90.0,
        min_direction_cosine: float = 0.25,
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
        self.lookahead_frames = max(1, int(lookahead_frames))
        self.prediction_radius = max(1.0, float(prediction_radius))
        self.min_direction_cosine = float(np.clip(min_direction_cosine, -1.0, 1.0))
        self._next_global_id = 1
        self._local_to_global: Dict[Tuple[str, int], int] = {}
        self._gid_members: Dict[int, set[Tuple[str, int]]] = {}
        # Retired IDs are permanent aliases to the smaller canonical ID.  A
        # handoff/slot recovery that still references an old ID cannot revive it.
        self._global_aliases: Dict[int, int] = {}
        self._handoffs: List[HandoffEntry] = []
        self._recently_lost: List[LostTrackEntry] = []
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

    def notify_track_lost(self, cam_id: str, local_track_id: int, track, frame_idx: int) -> None:
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
            last_world=self._world(cam_id, (track.cx, track.cy)),
            velocity_world=self._world_velocity(cam_id, track), bbox_size=(track.w, track.h),
            appearance=getattr(track, "appearance", None), lost_at_frame=frame_idx,
        ))
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
                world = self._world(entry.camera_id, (track.cx, track.cy))
                if np.linalg.norm(np.subtract(world, predicted)) > 70.0:
                    continue
                if self._appearance_distance(getattr(track, "appearance", None), entry.appearance) > 0.22:
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
        current = self._world(cam_id, (track.cx, track.cy))
        previous = self._world(cam_id, (first[0], first[1]))
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
    def _appearance_distance(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
        if left is None or right is None:
            return 0.25
        return float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def _size_distance(first: Tuple[int, int], second: Tuple[int, int]) -> float:
        fw, fh = first
        sw, sh = second
        ratio_w = min(fw, sw) / max(fw, sw, 1)
        ratio_h = min(fh, sh) / max(fh, sh, 1)
        return 1.0 - ratio_w * ratio_h

    def _upsert_handoff(self, cam_id: str, local_track_id: int, track, frame_idx: int) -> None:
        exit_info = self._outward_edge(cam_id, track)
        global_id = self._local_to_global.get((cam_id, local_track_id))
        if exit_info is None or global_id is None:
            return
        edge, velocity = exit_info
        target_cam = self.edge_adjacency.get((cam_id, edge))
        if target_cam is None:
            return
        world = self._world(cam_id, (track.cx, track.cy))
        for entry in self._handoffs:
            if entry.global_id == global_id and entry.source_cam == cam_id and entry.target_cam == target_cam:
                entry.last_world = world
                entry.velocity_world = self._world_velocity(cam_id, track)
                entry.bbox_size = (track.w, track.h)
                entry.appearance = getattr(track, "appearance", None)
                entry.updated_at_frame = frame_idx
                return
        self._handoffs.append(HandoffEntry(
            global_id=global_id, source_cam=cam_id, source_local_track_id=local_track_id,
            target_cam=target_cam, exit_edge=edge, last_world=world,
            velocity_world=self._world_velocity(cam_id, track), bbox_size=(track.w, track.h),
            appearance=getattr(track, "appearance", None), created_at_frame=frame_idx,
            updated_at_frame=frame_idx,
        ))
        print(f"\033[93m  [Handoff Opened] GID #{global_id} từ {cam_id} sang {target_cam} (cạnh {edge})\033[0m")
        self._event("handoff_opened", frame_idx, global_id, source_camera=cam_id,
                    target_camera=target_cam, edge=edge, velocity={"x": round(velocity[0], 2), "y": round(velocity[1], 2)})

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
        if cam_id in self.custom_masks:
            target_edge = str(self.custom_masks[cam_id]["handoff_edge"])
        else:
            target_edge = OPPOSITE_EDGE.get(entry.exit_edge, "unknown")
            if target_edge == "unknown":
                return None, "invalid_edge", {}
        predicted = self._predicted_world(entry, frame_idx)
        world = self._world(cam_id, (track.cx, track.cy))
        residual = float(np.linalg.norm(np.subtract(world, predicted)))
        depth = self._entry_depth(cam_id, track, target_edge)
        speed = float(np.hypot(*entry.velocity_world))
        # A one-frame tentative observation has no target velocity yet.  It may
        # still match only near the entry edge and near the predicted point.
        entry_limit = self.edge_margin + self.prediction_radius + speed * min(4, self.lookahead_frames) * 0.25
        details = {"predicted_distance": round(residual, 2), "entry_depth": round(depth, 2)}
        if depth > entry_limit:
            return None, "outside_entry_corridor", details
        if residual > self.prediction_radius:
            return None, "prediction_distance", details
        appearance_distance = self._appearance_distance(getattr(track, "appearance", None), entry.appearance)
        details["appearance_distance"] = round(appearance_distance, 3)
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
        for row, entry in enumerate(entries):
            for col, (cam_id, local_id, track) in enumerate(candidates):
                if entry.target_cam != cam_id:
                    continue
                cost, reason, details = self._candidate_cost(entry, cam_id, track, frame_idx)
                if cost is None:
                    self._record_rejection(entry, frame_idx, cam_id, local_id, reason, details)
                    continue
                costs[row, col] = cost
        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.92)
        accepted_entries = []
        for row, col in enumerate(row_to_col):
            if col < 0 or costs[row, col] >= invalid_cost:
                continue
            entry = entries[row]
            cam_id, local_id, _ = candidates[col]
            self._bind(cam_id, local_id, entry.global_id)
            self._event("handoff_matched", frame_idx, entry.global_id, source_camera=entry.source_cam,
                        target_camera=cam_id, source_local_id=entry.source_local_track_id,
                        target_local_id=local_id, score=round(float(costs[row, col]), 3),
                        predicted_position={"x": round(self._predicted_world(entry, frame_idx)[0], 2), "y": round(self._predicted_world(entry, frame_idx)[1], 2)})
            print(f"  [handoff] global #{entry.global_id}: {entry.source_cam} -> {cam_id}")
            accepted_entries.append(entry)
        for entry in accepted_entries:
            self._handoffs.remove(entry)

    def _match_simultaneous_overlap(self, cam_id: str, local_track_id: int, track, all_tracks: Dict[str, dict]) -> Optional[int]:
        """Deduplicate one car seen in two overlapping views."""
        world = self._world(cam_id, (track.cx, track.cy))
        for other_cam, other_tracks in all_tracks.items():
            if other_cam == cam_id:
                continue
            adjacent = any(source == cam_id and target == other_cam for (source, _), target in self.edge_adjacency.items()) or any(source == other_cam and target == cam_id for (source, _), target in self.edge_adjacency.items())
            if not adjacent:
                continue
            region = self.overlap_regions.get((cam_id, other_cam))
            if region is None:
                region = self.overlap_regions.get((other_cam, cam_id))
            if region is not None:
                if cv2.pointPolygonTest(region, world, False) < 0:
                    continue
            else:
                own_crop = self.camera_crops[cam_id]
                other_crop = self.camera_crops[other_cam]
                ix1, iy1 = max(own_crop[0], other_crop[0]), max(own_crop[1], other_crop[1])
                ix2, iy2 = min(own_crop[2], other_crop[2]), min(own_crop[3], other_crop[3])
                if ix1 >= ix2 or iy1 >= iy2:
                    continue
                if not (ix1 - self.edge_margin <= world[0] <= ix2 + self.edge_margin and iy1 - self.edge_margin <= world[1] <= iy2 + self.edge_margin):
                    continue
            for other_local_id, other_track in other_tracks.items():
                global_id = self._local_to_global.get((other_cam, other_local_id))
                if global_id is None:
                    continue
                other_world = self._world(other_cam, (other_track.cx, other_track.cy))
                if np.linalg.norm(np.subtract(world, other_world)) > self.match_distance * 0.5:
                    continue
                appearance_distance = self._appearance_distance(getattr(track, "appearance", None), getattr(other_track, "appearance", None))
                if appearance_distance <= self.appearance_threshold:
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
        appearance = self._appearance_distance(getattr(first, "appearance", None), getattr(second, "appearance", None))
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

    def update_all_tracks(self, all_tracks: Dict[str, dict], frame_idx: int) -> Dict[str, Dict[int, int]]:
        """Assign global IDs for current observations from every camera.

        ``all_tracks`` may include tentative tracks.  They are allowed to
        receive an existing handoff ID, but only confirmed tracks may allocate
        a previously unseen global ID.
        """
        # Existing confirmed source tracks publish an early, velocity-aware handoff.
        for cam_id, tracks in all_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) in self._local_to_global and self._is_confirmed(track):
                    self._upsert_handoff(cam_id, local_track_id, track, frame_idx)

        self._match_pending_handoffs(all_tracks, frame_idx)

        # Resolve overlapping observations before allocating a new global ID.
        for cam_id, tracks in all_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) not in self._local_to_global:
                    self._match_simultaneous_overlap(cam_id, local_track_id, track, all_tracks)

        # Tentative tracks remain ID-less unless they consumed a handoff.
        for cam_id, tracks in all_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) in self._local_to_global or not self._is_confirmed(track):
                    continue
                if self._match_same_camera_duplicate(cam_id, local_track_id, track, all_tracks, frame_idx) is not None:
                    continue
                global_id = self._bind(cam_id, local_track_id, self._allocate_global_id())
                self._event("global_id_created", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)
                print(f"  🆕 Cấp Global #{global_id} mới cho {cam_id} (local #{local_track_id})")

        self._merge_all_nearby_active_duplicates(all_tracks, frame_idx)
        self._merge_recently_lost_duplicates(all_tracks, frame_idx)

        self.cleanup(frame_idx)
        return {
            cam_id: {local_id: self._local_to_global[(cam_id, local_id)] for local_id in tracks if (cam_id, local_id) in self._local_to_global}
            for cam_id, tracks in all_tracks.items()
        }

    def notify_track_expired(self, cam_id: str, local_track_id: int, cx: int, cy: int, bbox_w: int, bbox_h: int, appearance: Optional[np.ndarray], frame_idx: int) -> None:
        """Remove a stale local mapping; its pre-opened global handoff remains."""
        key = (cam_id, local_track_id)
        global_id = self._local_to_global.pop(key, None)
        if global_id is not None:
            global_id = self._canonical_id(global_id)
            self._gid_members.get(global_id, set()).discard(key)
            self._event("local_track_expired", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)
            print(f"  ❌ Mất dấu Global #{global_id} ở {cam_id}")

    def cleanup(self, frame_idx: int) -> None:
        retained = []
        for entry in self._handoffs:
            if frame_idx - entry.updated_at_frame <= self.handoff_ttl:
                retained.append(entry)
                continue
            self._event("handoff_expired", frame_idx, entry.global_id, source_camera=entry.source_cam,
                        target_camera=entry.target_cam, source_local_id=entry.source_local_track_id)
        self._handoffs = retained

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
                observation = {
                    "camera_id": cam_id, "local_track_id": local_id,
                    "local_position": {"x": track.cx, "y": track.cy},
                    "global_position": {"x": round(self._world(cam_id, (track.cx, track.cy))[0], 2), "y": round(self._world(cam_id, (track.cx, track.cy))[1], 2)},
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
                "position": {"x": round(sum(item["global_position"]["x"] for item in observations) / len(observations), 2), "y": round(sum(item["global_position"]["y"] for item in observations) / len(observations), 2), "reference": "source_video_pixel"},
                "camera_ids": [item["camera_id"] for item in observations],
                "observation_count": len(observations),
            }
        return {
            "next_global_id": self._next_global_id,
            "retired_global_ids": {str(old_id): canonical_id for old_id, canonical_id in sorted(self._global_aliases.items())},
            "active_global_vehicles": active,
            "map_vehicles": map_vehicles,
            "pending_handoffs": [{
                "global_id": item.global_id, "source_camera": item.source_cam,
                "source_local_track_id": item.source_local_track_id, "target_camera": item.target_cam,
                "exit_edge": item.exit_edge, "last_world": {"x": round(item.last_world[0], 2), "y": round(item.last_world[1], 2)},
                "velocity": {"x": round(item.velocity_world[0], 2), "y": round(item.velocity_world[1], 2)},
                "created_at_frame": item.created_at_frame, "updated_at_frame": item.updated_at_frame,
            } for item in self._handoffs],
            "recent_events": self._events,
        }
