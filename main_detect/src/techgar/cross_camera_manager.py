"""Global vehicle IDs and predictive handoff between adjacent cameras.

Local tracker IDs belong to one camera only.  ``CrossCameraManager`` owns the
single global namespace and associates a new local observation with an
existing global vehicle before that observation is confirmed by its local
tracker.  This is important for fast vehicles: local confirmation can happen
several frames after the vehicle crossed a camera border.
"""

from __future__ import annotations

from collections import deque
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
    # Cross-camera HSV can differ strongly.  If this Global ID was observed in
    # the destination camera before, compare against that camera-specific
    # gallery instead of the mixed/source gallery.
    target_appearance_samples: Tuple[np.ndarray, ...] = ()
    target_bbox_size: Optional[Tuple[int, int]] = None


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
    camera_appearance_samples: Dict[str, Tuple[np.ndarray, ...]]
    camera_bbox_sizes: Dict[str, Tuple[int, int]]
    last_seen_frame: int
    last_seen_time: Optional[float]
    dormant_since_frame: Optional[int] = None
    dormant_since_time: Optional[float] = None
    exited_at_frame: Optional[int] = None
    exited_at_time: Optional[float] = None


@dataclass
class DirectionReIDClaim:
    """Bounded evidence for a spatial/appearance match with unstable motion."""

    global_id: int
    camera_id: str
    local_track_id: int
    first_frame: int
    last_frame: int
    first_time: Optional[float]
    last_time: Optional[float]
    observations: int = 1


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
        identity_retention_seconds: float = 60.0,
        dormant_match_distance: float = 160.0,
        dormant_appearance_threshold: Optional[float] = None,
        tracklet_gallery_size: int = 24,
        new_identity_min_observations: int = 5,
        new_identity_unusual_size_observations: int = 12,
        new_identity_min_displacement_ratio: float = 0.15,
        exit_zones: Optional[Dict[str, List[np.ndarray]]] = None,
        world_unit: str = "source_video_pixel",
        shared_map_anchor: str = "bottom_center",
        camera_fps: Optional[Dict[str, float]] = None,
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
        self.new_identity_min_observations = max(
            3, int(new_identity_min_observations)
        )
        self.new_identity_unusual_size_observations = max(
            self.new_identity_min_observations,
            int(new_identity_unusual_size_observations),
        )
        self.new_identity_min_displacement_ratio = max(
            0.0, float(new_identity_min_displacement_ratio)
        )
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
        self._global_created_frames: Dict[int, int] = {}
        self._local_to_global: Dict[Tuple[str, int], int] = {}
        # Frame at which a local fragment most recently acquired its current
        # Global ID.  This is deliberately different from the local track's
        # age: after a camera dropout an old local object may reappear while a
        # newer, correctly transferred fragment already owns the identity.
        self._local_binding_frames: Dict[Tuple[str, int], int] = {}
        self._processing_frame_idx: Optional[int] = None
        self._gid_members: Dict[int, set[Tuple[str, int]]] = {}
        # Retired IDs are permanent aliases to the smaller canonical ID.  A
        # handoff/slot recovery that still references an old ID cannot revive it.
        self._global_aliases: Dict[int, int] = {}
        self._handoffs: List[HandoffEntry] = []
        self._recently_lost: List[LostTrackEntry] = []
        self._identities: Dict[int, GlobalIdentityState] = {}
        self._cross_camera_deferred_since: Dict[Tuple[str, int], int] = {}
        self._cross_camera_deferred_claims: Dict[Tuple[str, int], dict] = {}
        self._handoff_candidate_evidence: Dict[
            Tuple[int, str, int], Tuple[int, int]
        ] = {}
        self._cross_camera_duplicate_evidence: Dict[
            Tuple[int, int], Tuple[int, int]
        ] = {}
        self._events: List[dict] = []
        self._parked_reservations: Dict[int, dict] = {}
        self._same_camera_duplicate_evidence: Dict[
            Tuple[str, int, int], Tuple[int, int]
        ] = {}
        self._lost_continuation_evidence: Dict[
            Tuple[str, int, int], Tuple[int, int]
        ] = {}
        self._dormant_rejection_last: Dict[
            Tuple[int, str, int, str], int
        ] = {}
        # A dormant LAPJV result can be geometrically plausible for two old
        # identities at once.  Preserve that warning for the next few frames
        # so a handoff belonging to another GID cannot immediately consume
        # the same uncertain local fragment.
        self._ambiguous_local_identities: Dict[
            Tuple[str, int], Tuple[int, set[int]]
        ] = {}
        self._new_identity_defer_last: Dict[Tuple[str, int, str], int] = {}
        # Robust per-camera size gallery. ``camera_bbox_sizes`` used to be
        # overwritten by the latest motion blob, so one lamp/edge fragment
        # could become the learned "vehicle size" and later validate more
        # false Global IDs. Keep a short median gallery and reject abrupt
        # one-frame size collapses once a stable reference exists.
        self._camera_bbox_samples: Dict[
            Tuple[int, str], List[Tuple[int, int]]
        ] = {}
        self._camera_fps_bootstrap = {
            str(camera_id): float(np.clip(fps, 1.0, 60.0))
            for camera_id, fps in (camera_fps or {}).items()
            if fps is not None and np.isfinite(float(fps)) and float(fps) > 0.0
        }
        self._camera_timestamp_deltas = {
            camera_id: deque(maxlen=30) for camera_id in self.camera_sizes
        }
        self._camera_last_timestamp: Dict[str, float] = {}
        self._direction_reid_claims: Dict[
            Tuple[int, str, int], DirectionReIDClaim
        ] = {}

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
        raw_previous = self._local_to_global.get(key)
        previous = (
            self._canonical_id(raw_previous)
            if raw_previous is not None
            else None
        )
        if previous is not None and previous != global_id:
            self._gid_members.get(previous, set()).discard(key)
        if previous != global_id or key not in self._local_binding_frames:
            if self._processing_frame_idx is not None:
                self._local_binding_frames[key] = int(
                    self._processing_frame_idx
                )
        self._local_to_global[key] = global_id
        self._gid_members.setdefault(global_id, set()).add(key)
        self._direction_reid_claims = {
            claim_key: claim
            for claim_key, claim in self._direction_reid_claims.items()
            if claim_key[1:] != (str(cam_id), int(local_track_id))
        }
        return global_id

    @staticmethod
    def _appearance_snapshot(appearance: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return aggregate_appearance(appearance)

    def _tracklet_snapshot(self, source) -> Tuple[np.ndarray, ...]:
        return tuple(
            sample.copy()
            for sample in appearance_samples(source)[-self.tracklet_gallery_size:]
        )

    @staticmethod
    def _copy_camera_galleries(
        identity: Optional[GlobalIdentityState],
    ) -> Dict[str, Tuple[np.ndarray, ...]]:
        if identity is None:
            return {}
        return {
            camera_id: tuple(sample.copy() for sample in samples)
            for camera_id, samples in identity.camera_appearance_samples.items()
        }

    def _target_camera_gallery(
        self,
        global_id: int,
        camera_id: str,
    ) -> Tuple[np.ndarray, ...]:
        identity = self._identities.get(self._canonical_id(global_id))
        if identity is None:
            return ()
        return identity.camera_appearance_samples.get(camera_id, ())

    def _are_adjacent(self, first_camera: str, second_camera: str) -> bool:
        if first_camera == second_camera:
            return False
        return any(
            (source == first_camera and target == second_camera)
            or (source == second_camera and target == first_camera)
            for (source, _edge), target in self.edge_adjacency.items()
        )

    def _has_confirmed_camera_member(
        self,
        global_id: int,
        cam_id: str,
        all_tracks: Dict[str, dict],
        *,
        exclude_local_id: Optional[int] = None,
    ) -> bool:
        """Return whether ``global_id`` already has a live owner in a camera.

        Overlapping physical cameras may observe one vehicle at the same time,
        but one Global ID must not own two independent, confirmed local tracks
        *inside the same camera*. This guard is checked before handoff/overlap
        binding so a reverse handoff cannot attach an already-active ID to a
        neighbouring vehicle.
        """
        global_id = self._canonical_id(global_id)
        for local_track_id, track in all_tracks.get(cam_id, {}).items():
            if exclude_local_id is not None and local_track_id == exclude_local_id:
                continue
            mapped = self._local_to_global.get((cam_id, local_track_id))
            if mapped is None or self._canonical_id(mapped) != global_id:
                continue
            if self._is_confirmed(track):
                return True
        return False

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
        camera_galleries = self._copy_camera_galleries(previous)
        camera_galleries[cam_id] = merge_appearance_samples(
            camera_galleries.get(cam_id, ()),
            track,
            self.tracklet_gallery_size,
        )
        camera_bbox_sizes = dict(
            previous.camera_bbox_sizes if previous is not None else {}
        )
        size_key = (global_id, cam_id)
        size_samples = self._camera_bbox_samples.setdefault(size_key, [])
        current_size = (int(track.w), int(track.h))
        if size_samples:
            reference_size = tuple(
                int(round(value))
                for value in np.median(
                    np.asarray(size_samples, dtype=np.float64), axis=0
                )
            )
            if (
                len(size_samples) < 5
                or self._size_distance(current_size, reference_size) <= 0.65
            ):
                size_samples.append(current_size)
        else:
            size_samples.append(current_size)
        if len(size_samples) > self.tracklet_gallery_size:
            del size_samples[:-self.tracklet_gallery_size]
        robust_size = tuple(
            int(round(value))
            for value in np.median(
                np.asarray(size_samples, dtype=np.float64), axis=0
            )
        )
        camera_bbox_sizes[cam_id] = robust_size
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
            bbox_size=robust_size,
            appearance=appearance,
            appearance_samples=gallery,
            camera_appearance_samples=camera_galleries,
            camera_bbox_sizes=camera_bbox_sizes,
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
        self._global_created_frames.setdefault(global_id, int(frame_idx))
        if global_id in self._parked_reservations:
            if source != "parking_departure_token":
                raise ValueError(
                    f"Global #{global_id} is parked and may only be recovered "
                    "from its departure token"
                )
            self._parked_reservations.pop(global_id, None)
        previous_processing_frame = self._processing_frame_idx
        self._processing_frame_idx = int(frame_idx)
        bound = self._bind(cam_id, local_track_id, global_id)
        self._processing_frame_idx = previous_processing_frame
        self._event("global_id_recovered", frame_idx, global_id, camera=cam_id,
                    local_track_id=local_track_id, source=source)
        return bound

    @property
    def parked_global_ids(self) -> set[int]:
        return set(self._parked_reservations)

    def sync_parked_reservations(
        self,
        reservations: Iterable[dict],
        frame_idx: int,
    ) -> Dict[int, dict]:
        """Synchronize one authoritative slot owner per canonical Global ID."""
        grouped: Dict[int, List[dict]] = {}
        for raw in reservations:
            if raw.get("global_id") is None:
                continue
            global_id = self._canonical_id(int(raw["global_id"]))
            value = dict(raw)
            value["global_id"] = global_id
            grouped.setdefault(global_id, []).append(value)

        selected: Dict[int, dict] = {}
        for global_id, claims in grouped.items():
            previous = self._parked_reservations.get(global_id)
            winner = None
            if previous is not None:
                winner = next(
                    (
                        claim
                        for claim in claims
                        if claim.get("slot_id") == previous.get("slot_id")
                    ),
                    None,
                )
            if winner is None:
                winner = max(
                    claims,
                    key=lambda item: (
                        item.get("state") == "parked",
                        str(item.get("camera_id") or ""),
                        str(item.get("slot_id") or ""),
                    ),
                )
            selected[global_id] = winner
            for rejected in claims:
                if rejected is winner:
                    continue
                self._event(
                    "parked_reservation_rejected",
                    frame_idx,
                    global_id,
                    kept_slot_id=winner.get("slot_id"),
                    rejected_slot_id=rejected.get("slot_id"),
                )

        slot_owners: Dict[Tuple[object, object], int] = {}
        for global_id in sorted(list(selected)):
            reservation = selected[global_id]
            slot_key = (reservation.get("camera_id"), reservation.get("slot_id"))
            current_owner = slot_owners.get(slot_key)
            if current_owner is None:
                slot_owners[slot_key] = global_id
                continue
            previous_owner = next(
                (
                    owner
                    for owner, old in self._parked_reservations.items()
                    if (old.get("camera_id"), old.get("slot_id")) == slot_key
                    and owner in {current_owner, global_id}
                ),
                None,
            )
            kept = previous_owner if previous_owner is not None else min(current_owner, global_id)
            rejected = global_id if kept == current_owner else current_owner
            selected.pop(rejected, None)
            slot_owners[slot_key] = kept
            self._event(
                "parked_reservation_rejected",
                frame_idx,
                rejected,
                kept_global_id=kept,
                rejected_slot_id=reservation.get("slot_id"),
                reason="slot_already_reserved",
            )

        for global_id, reservation in selected.items():
            previous = self._parked_reservations.get(global_id)
            if previous is None or previous.get("slot_id") != reservation.get("slot_id"):
                self._event(
                    "global_id_parked_reserved",
                    frame_idx,
                    global_id,
                    slot_id=reservation.get("slot_id"),
                    camera=reservation.get("camera_id"),
                )
            identity = self._identities.get(global_id)
            if identity is not None:
                identity.state = str(reservation.get("state") or "parked")
                identity.dormant_since_frame = None
                identity.dormant_since_time = None

        for global_id in set(self._parked_reservations) - set(selected):
            identity = self._identities.get(global_id)
            if identity is not None and identity.state in {"parked", "recovery_pending"}:
                identity.state = "dormant"
                identity.dormant_since_frame = int(frame_idx)
        self._parked_reservations = selected
        return {global_id: dict(value) for global_id, value in selected.items()}

    def detach_parked_local_tracks(self, frame_idx: int) -> List[Tuple[str, int, int]]:
        """Detach local fragments so a parked track cannot steal a new bbox."""
        detached = []
        for key, raw_global_id in list(self._local_to_global.items()):
            global_id = self._canonical_id(raw_global_id)
            if global_id not in self._parked_reservations:
                continue
            self._local_to_global.pop(key, None)
            self._local_binding_frames.pop(key, None)
            self._gid_members.get(global_id, set()).discard(key)
            detached.append((key[0], int(key[1]), global_id))
            self._event(
                "parked_local_track_suspended",
                frame_idx,
                global_id,
                camera=key[0],
                local_track_id=int(key[1]),
            )
        return detached

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
        canonical_birth = self._global_created_frames.get(canonical_id, frame_idx)
        duplicate_birth = self._global_created_frames.pop(duplicate_id, frame_idx)
        self._global_created_frames[canonical_id] = min(
            canonical_birth, duplicate_birth
        )
        for (sample_gid, sample_camera), samples in list(
            self._camera_bbox_samples.items()
        ):
            if sample_gid != duplicate_id:
                continue
            destination = self._camera_bbox_samples.setdefault(
                (canonical_id, sample_camera), []
            )
            destination.extend(samples)
            if len(destination) > self.tracklet_gallery_size:
                del destination[:-self.tracklet_gallery_size]
            self._camera_bbox_samples.pop(
                (duplicate_id, sample_camera), None
            )
        duplicate_reservation = self._parked_reservations.pop(duplicate_id, None)
        if duplicate_reservation is not None and canonical_id not in self._parked_reservations:
            duplicate_reservation["global_id"] = canonical_id
            self._parked_reservations[canonical_id] = duplicate_reservation
        for key, global_id in list(self._local_to_global.items()):
            if self._canonical_id(global_id) == canonical_id or global_id == duplicate_id:
                self._bind(key[0], key[1], canonical_id)
        self._gid_members.pop(duplicate_id, None)
        canonical_state = self._identities.get(canonical_id)
        duplicate_state = self._identities.pop(duplicate_id, None)
        merged_camera_galleries: Dict[str, Tuple[np.ndarray, ...]] = {}
        merged_camera_sizes: Dict[str, Tuple[int, int]] = {}
        camera_ids = set(
            canonical_state.camera_appearance_samples if canonical_state else {}
        ) | set(
            duplicate_state.camera_appearance_samples if duplicate_state else {}
        )
        for camera_id in camera_ids:
            merged_camera_galleries[camera_id] = merge_appearance_samples(
                (
                    canonical_state.camera_appearance_samples.get(camera_id, ())
                    if canonical_state is not None
                    else ()
                ),
                (
                    duplicate_state.camera_appearance_samples.get(camera_id, ())
                    if duplicate_state is not None
                    else ()
                ),
                self.tracklet_gallery_size,
            )
        if canonical_state is not None:
            merged_camera_sizes.update(canonical_state.camera_bbox_sizes)
        if duplicate_state is not None:
            for camera_id, bbox_size in duplicate_state.camera_bbox_sizes.items():
                if (
                    camera_id not in merged_camera_sizes
                    or canonical_state is None
                    or duplicate_state.last_seen_frame >= canonical_state.last_seen_frame
                ):
                    merged_camera_sizes[camera_id] = bbox_size
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
            canonical_state.camera_appearance_samples = merged_camera_galleries
            canonical_state.camera_bbox_sizes = merged_camera_sizes
        for handoff in self._handoffs:
            handoff.global_id = self._canonical_id(handoff.global_id)
        for lost in self._recently_lost:
            lost.global_id = self._canonical_id(lost.global_id)
        remapped_direction_claims: Dict[
            Tuple[int, str, int], DirectionReIDClaim
        ] = {}
        for (_claim_gid, claim_camera, claim_local_id), claim in (
            self._direction_reid_claims.items()
        ):
            claim.global_id = self._canonical_id(claim.global_id)
            key = (claim.global_id, claim_camera, claim_local_id)
            previous_claim = remapped_direction_claims.get(key)
            if previous_claim is None or claim.observations > previous_claim.observations:
                remapped_direction_claims[key] = claim
        self._direction_reid_claims = remapped_direction_claims
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
            if self._canonical_id(entry.global_id) in self._parked_reservations:
                retained.append(entry)
                continue
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
                if self._canonical_id(candidate_id) in self._parked_reservations:
                    continue
                world = self._track_world(entry.camera_id, track)
                if np.linalg.norm(np.subtract(world, predicted)) > 70.0:
                    continue
                if self._appearance_distance(
                    track, entry.appearance_samples or entry.appearance
                ) > 0.18:
                    continue
                if self._size_distance((track.w, track.h), entry.bbox_size) > 0.60:
                    continue
                direction = self._direction_cosine(entry.camera_id, track, entry.velocity_world)
                if direction is not None and direction < 0.70:
                    continue
                evidence_key = (
                    entry.camera_id,
                    int(entry.global_id),
                    int(candidate_id),
                )
                count, last_frame = self._lost_continuation_evidence.get(
                    evidence_key, (0, -999999)
                )
                if last_frame == int(frame_idx):
                    ready = count >= 3
                else:
                    count = count + 1 if int(frame_idx) - last_frame == 1 else 1
                    self._lost_continuation_evidence[evidence_key] = (
                        count, int(frame_idx)
                    )
                    ready = count >= 3
                if not ready:
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
    def _is_allocatable(cls, track) -> bool:
        if not cls._is_confirmed(track):
            return False
        return getattr(track, "association_state", "matched") not in {
            "frozen_ambiguous",
            "ambiguous_merged",
        }

    def _record_new_identity_deferred(
        self,
        cam_id: str,
        local_track_id: int,
        frame_idx: int,
        reason: str,
        **details,
    ) -> None:
        key = (str(cam_id), int(local_track_id), str(reason))
        if int(frame_idx) - self._new_identity_defer_last.get(
            key, -999999
        ) < 8:
            return
        self._new_identity_defer_last[key] = int(frame_idx)
        self._event(
            "new_global_id_deferred_insufficient_fragment_evidence",
            frame_idx,
            None,
            camera=cam_id,
            local_track_id=int(local_track_id),
            reason=str(reason),
            **details,
        )

    def _trusted_camera_size_references(
        self,
        cam_id: str,
        frame_idx: int,
    ) -> List[Tuple[int, int]]:
        """Return sizes learned from durable identities in this camera.

        Short motion blobs must not become the reference that later validates
        more blobs.  A reference therefore needs a meaningful lifetime or a
        parking reservation, but it need not already be a two-camera identity.
        """
        references: List[Tuple[int, int]] = []
        for identity in self._identities.values():
            global_id = self._canonical_id(identity.global_id)
            created_at = self._global_created_frames.get(global_id)
            lifetime = (
                int(identity.last_seen_frame) - int(created_at)
                if created_at is not None
                else 0
            )
            if (
                lifetime < 30
                and global_id not in self._parked_reservations
                and not self._identity_is_established(identity)
            ):
                continue
            size = identity.camera_bbox_sizes.get(cam_id)
            if size is not None and size[0] > 0 and size[1] > 0:
                references.append((int(size[0]), int(size[1])))
        return references

    def _new_global_id_ready(
        self,
        cam_id: str,
        local_track_id: int,
        track,
        frame_idx: int,
    ) -> bool:
        """Require coherent current-fragment evidence before minting a GID.

        Existing IDs may still bind tentative tracks through handoff/Re-ID.
        This gate applies only to creation of a brand-new identity, where a
        short lamp edge or motion tail must not pollute the global registry.
        """
        if not self._is_allocatable(track):
            return False
        observations = getattr(track, "fragment_visible_count", None)
        if observations is None:
            # Legacy/test trackers do not expose per-fragment evidence.
            return True
        try:
            observations = int(observations)
        except (TypeError, ValueError):
            return False
        if observations < self.new_identity_min_observations:
            self._record_new_identity_deferred(
                cam_id,
                local_track_id,
                frame_idx,
                "too_few_observations",
                observations=observations,
                required_observations=self.new_identity_min_observations,
            )
            return False

        origin = getattr(track, "first_observation_point", None)
        diagonal = max(1.0, float(np.hypot(track.w, track.h)))
        displacement = (
            float(np.linalg.norm(np.subtract((track.cx, track.cy), origin)))
            if origin is not None
            else diagonal
        )
        minimum_displacement = max(
            6.0, self.new_identity_min_displacement_ratio * diagonal
        )
        if displacement < minimum_displacement:
            self._record_new_identity_deferred(
                cam_id,
                local_track_id,
                frame_idx,
                "insufficient_displacement",
                observations=observations,
                displacement=round(displacement, 3),
                required_displacement=round(minimum_displacement, 3),
            )
            return False

        references = self._trusted_camera_size_references(cam_id, frame_idx)
        if references:
            best_size_distance = min(
                self._size_distance((track.w, track.h), reference)
                for reference in references
            )
            if best_size_distance > 0.65 and (
                observations < self.new_identity_unusual_size_observations
                or displacement < 0.60 * diagonal
            ):
                self._record_new_identity_deferred(
                    cam_id,
                    local_track_id,
                    frame_idx,
                    "unusual_size_without_long_trajectory",
                    observations=observations,
                    required_observations=(
                        self.new_identity_unusual_size_observations
                    ),
                    displacement=round(displacement, 3),
                    required_displacement=round(0.60 * diagonal, 3),
                    size_distance=round(float(best_size_distance), 3),
                )
                return False
        return True

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

    def _upsert_handoff(
        self,
        cam_id: str,
        local_track_id: int,
        track,
        frame_idx: int,
        all_tracks: Dict[str, dict],
    ) -> None:
        global_id = self._local_to_global.get((cam_id, local_track_id))
        if global_id is None:
            return
        world = self._track_world(cam_id, track)
        velocity_world = self._world_velocity(cam_id, track)
        # Once an early handoff has been opened, keep its motion evidence and
        # lifetime tied to the live source observation. Previously the entry
        # was refreshed only while `_outward_edge()` kept returning true. A
        # car could therefore remain visible and moving for dozens of frames,
        # yet its handoff expired before the first destination bbox appeared.
        existing_entries = [
            entry
            for entry in self._handoffs
            if (
                self._canonical_id(entry.global_id)
                == self._canonical_id(global_id)
                and entry.source_cam == cam_id
                and entry.source_local_track_id == local_track_id
            )
        ]
        if existing_entries:
            refreshed_any = False
            for entry in existing_entries:
                if self._has_confirmed_camera_member(
                    global_id, entry.target_cam, all_tracks
                ):
                    continue
                identity = self._identities.get(self._canonical_id(global_id))
                gallery = merge_appearance_samples(
                    (
                        identity.camera_appearance_samples.get(cam_id, ())
                        if identity is not None
                        else ()
                    ),
                    track,
                    self.tracklet_gallery_size,
                )
                entry.last_world = world
                entry.velocity_world = velocity_world
                entry.bbox_size = (track.w, track.h)
                entry.appearance = aggregate_appearance(gallery)
                entry.appearance_samples = gallery
                entry.target_appearance_samples = (
                    identity.camera_appearance_samples.get(entry.target_cam, ())
                    if identity is not None
                    else ()
                )
                entry.target_bbox_size = (
                    identity.camera_bbox_sizes.get(entry.target_cam)
                    if identity is not None
                    else None
                )
                entry.updated_at_frame = frame_idx
                refreshed_any = True
            self._handoffs = [
                entry
                for entry in self._handoffs
                if entry not in existing_entries
                or not self._has_confirmed_camera_member(
                    global_id, entry.target_cam, all_tracks
                )
            ]
            if refreshed_any:
                return
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
        # Do not open a reverse overlap handoff while this identity already
        # owns a confirmed track in the target camera. A pending duplicate
        # handoff can otherwise consume a neighbouring vehicle's fresh bbox
        # before dormant Re-ID has a chance to recover its real identity.
        if self._has_confirmed_camera_member(global_id, target_cam, all_tracks):
            self._handoffs = [
                entry
                for entry in self._handoffs
                if not (
                    self._canonical_id(entry.global_id)
                    == self._canonical_id(global_id)
                    and entry.source_cam == cam_id
                    and entry.target_cam == target_cam
                )
            ]
            return
        identity = self._identities.get(self._canonical_id(global_id))
        gallery = merge_appearance_samples(
            (
                identity.camera_appearance_samples.get(cam_id, ())
                if identity is not None
                else ()
            ),
            track,
            self.tracklet_gallery_size,
        )
        target_gallery = (
            identity.camera_appearance_samples.get(target_cam, ())
            if identity is not None
            else ()
        )
        target_bbox_size = (
            identity.camera_bbox_sizes.get(target_cam)
            if identity is not None
            else None
        )
        appearance = aggregate_appearance(gallery)
        for entry in self._handoffs:
            if entry.global_id == global_id and entry.source_cam == cam_id and entry.target_cam == target_cam:
                entry.last_world = world
                entry.velocity_world = velocity_world
                entry.bbox_size = (track.w, track.h)
                entry.appearance = appearance
                entry.appearance_samples = gallery
                entry.target_appearance_samples = target_gallery
                entry.target_bbox_size = target_bbox_size
                entry.updated_at_frame = frame_idx
                return
        self._handoffs.append(HandoffEntry(
            global_id=global_id, source_cam=cam_id, source_local_track_id=local_track_id,
            target_cam=target_cam, exit_edge=edge, last_world=world,
            velocity_world=velocity_world, bbox_size=(track.w, track.h),
            appearance=appearance, appearance_samples=gallery,
            created_at_frame=frame_idx,
            updated_at_frame=frame_idx,
            target_appearance_samples=target_gallery,
            target_bbox_size=target_bbox_size,
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
        predicted_residual = float(
            np.linalg.norm(np.subtract(world, predicted))
        )
        last_world_residual = float(
            np.linalg.norm(np.subtract(world, entry.last_world))
        )
        # Once the source disappears its last velocity can be dominated by a
        # clipped motion tail.  In the calibrated overlap the physically
        # plausible path is the segment from last observation to prediction,
        # so use the closer endpoint as a conservative corridor proxy.
        residual = (
            min(predicted_residual, last_world_residual)
            if overlap_handoff
            else predicted_residual
        )
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
                "extrapolated_distance": round(predicted_residual, 2),
                "last_position_distance": round(last_world_residual, 2),
                "overlap_signed_distance": round(float(signed_overlap_distance), 2),
            }
            if signed_overlap_distance < -self.prediction_radius:
                return None, "outside_overlap", details
        else:
            depth = self._entry_depth(cam_id, track, target_edge)
            # A one-frame tentative observation has no target velocity yet. It
            # may still match only near the entry edge and predicted point.
            if cam_id in self.custom_masks:
                # `_entry_depth()` is measured in destination-image pixels for
                # polygon masks, whereas `prediction_radius` and `speed` are
                # shared-map units (centimetres for calibrated cameras). Do
                # not add those incompatible units. A newly confirmed bbox may
                # already be roughly one vehicle length inside the polygon, so
                # scale this image-space corridor by its observed dimensions.
                entry_limit = max(
                    self.edge_margin * 1.5,
                    self.edge_margin + 1.25 * max(track.w, track.h),
                )
            else:
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
        target_camera_reference = bool(entry.target_appearance_samples)
        appearance_reference = (
            entry.target_appearance_samples
            if target_camera_reference
            else (entry.appearance_samples or entry.appearance)
        )
        appearance_match = compare_tracklets(track, appearance_reference)
        appearance_distance = appearance_match.distance
        details["appearance_distance"] = round(appearance_distance, 3)
        details["appearance_reference"] = (
            "target_camera" if target_camera_reference else "source_camera"
        )
        details["tracklet_support"] = appearance_match.support
        details["tracklet_sample_pairs"] = appearance_match.sample_pairs
        if appearance_match.support <= 0:
            return None, "appearance_missing", details
        appearance_limit = (
            min(0.45, self.appearance_threshold)
            if target_camera_reference
            else self.appearance_threshold
        )
        adaptive_appearance_radius = max(
            self.strong_spatial_distance * 1.25,
            max(1.0, self.match_distance * 0.50),
        )
        adaptive_appearance = (
            not target_camera_reference
            and appearance_distance > appearance_limit
            and residual <= adaptive_appearance_radius
            and appearance_distance
            <= self.relaxed_appearance_threshold
        )
        if adaptive_appearance:
            # Opposing camera viewpoints can move a vehicle's HSV histogram,
            # but an entry within only a few shared-map centimetres is strong
            # independent evidence. Keep a bounded relaxed gate; do not use
            # the full relaxed threshold for merely approximate predictions.
            appearance_limit = self.relaxed_appearance_threshold
            details["adaptive_appearance"] = True
            details["requires_temporal_evidence"] = True
            details["adaptive_appearance_radius"] = round(
                adaptive_appearance_radius, 3
            )
            details["appearance_limit"] = round(appearance_limit, 3)
        if appearance_distance > appearance_limit:
            return None, "appearance", details
        size_reference = entry.target_bbox_size or entry.bbox_size
        size_distance = self._size_distance((track.w, track.h), size_reference)
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

    def _record_dormant_rejection(
        self,
        identity: GlobalIdentityState,
        frame_idx: int,
        cam_id: str,
        local_track_id: int,
        reason: str,
        **details,
    ) -> None:
        key = (identity.global_id, cam_id, int(local_track_id), str(reason))
        if frame_idx - self._dormant_rejection_last.get(key, -999999) < 8:
            return
        self._dormant_rejection_last[key] = int(frame_idx)
        self._event(
            "dormant_reid_rejected",
            frame_idx,
            identity.global_id,
            source_camera=identity.last_camera,
            target_camera=cam_id,
            target_local_id=int(local_track_id),
            reason=str(reason),
            **details,
        )

    @staticmethod
    def _advance_consecutive_evidence(
        store: Dict[tuple, Tuple[int, int]],
        key: tuple,
        frame_idx: int,
    ) -> int:
        count, last_frame = store.get(key, (0, -999999))
        if last_frame == int(frame_idx):
            return count
        count = count + 1 if int(frame_idx) - last_frame == 1 else 1
        store[key] = (count, int(frame_idx))
        return count

    @staticmethod
    def _assignment_has_margin(
        costs: np.ndarray,
        row: int,
        col: int,
        invalid_cost: float,
        margin: float = 0.08,
    ) -> bool:
        selected = float(costs[row, col])
        row_alternatives = [
            float(value)
            for index, value in enumerate(costs[row])
            if index != col and value < invalid_cost
        ]
        column_alternatives = [
            float(costs[index, col])
            for index in range(costs.shape[0])
            if index != row and costs[index, col] < invalid_cost
        ]
        alternatives = row_alternatives + column_alternatives
        return not alternatives or min(alternatives) - selected >= margin

    def _cross_merge_would_conflict(
        self,
        first_global_id: int,
        second_global_id: int,
        all_tracks: Dict[str, dict],
    ) -> bool:
        """Reject a merge that would give one GID two real cars in a camera."""
        first_global_id = self._canonical_id(first_global_id)
        second_global_id = self._canonical_id(second_global_id)
        if first_global_id == second_global_id:
            return False
        for cam_id, tracks in all_tracks.items():
            first_tracks = []
            second_tracks = []
            for local_id, track in tracks.items():
                if not self._is_confirmed(track):
                    continue
                mapped = self._local_to_global.get((cam_id, local_id))
                if mapped is None:
                    continue
                mapped = self._canonical_id(mapped)
                if mapped == first_global_id:
                    first_tracks.append(track)
                elif mapped == second_global_id:
                    second_tracks.append(track)
            if not first_tracks or not second_tracks:
                continue
            # A true motion echo may temporarily own two local boxes. Only
            # that explicit same-camera evidence permits the cross-ID merge.
            if not any(
                self._same_camera_motion_duplicate(first, second)
                for first in first_tracks
                for second in second_tracks
            ):
                return True
        return False

    def _resolve_same_camera_global_conflicts(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
    ) -> None:
        """Enforce one physical observation per GID in each camera.

        A local track that disappeared may later become visible with its old
        mapping after a newer fragment has already recovered that same GID.
        If the two boxes are not explicit motion echoes, keeping both would
        display one identity on two real cars.  Retain the observation that is
        most consistent with the camera-conditioned identity gallery and
        detach the others.  Detached tracks stay ID-less for the rest of this
        frame and can be conservatively recovered on the next update.
        """
        groups: Dict[Tuple[str, int], List[Tuple[int, object]]] = {}
        for cam_id, tracks in all_tracks.items():
            for local_id, track in tracks.items():
                if not self._is_confirmed(track):
                    continue
                global_id = self._local_to_global.get((cam_id, local_id))
                if global_id is None:
                    continue
                groups.setdefault(
                    (cam_id, self._canonical_id(global_id)), []
                ).append((int(local_id), track))

        for (cam_id, global_id), members in groups.items():
            if len(members) < 2:
                continue
            if all(
                self._same_camera_motion_duplicate(left_track, right_track)
                for index, (_left_id, left_track) in enumerate(members)
                for _right_id, right_track in members[index + 1 :]
            ):
                continue

            identity = self._identities.get(global_id)
            gallery = (
                identity.camera_appearance_samples.get(cam_id, ())
                if identity is not None
                else ()
            )
            size_reference = (
                identity.camera_bbox_sizes.get(cam_id, identity.bbox_size)
                if identity is not None
                else None
            )

            def owner_rank(member: Tuple[int, object]) -> tuple:
                local_id, track = member
                appearance_match = compare_tracklets(track, gallery)
                appearance_distance = (
                    float(appearance_match.distance)
                    if appearance_match.support > 0
                    else 1.0
                )
                size_distance = (
                    self._size_distance((track.w, track.h), size_reference)
                    if size_reference is not None
                    else 1.0
                )
                binding_frame = self._local_binding_frames.get(
                    (cam_id, local_id), -1
                )
                visible_count = int(
                    getattr(track, "total_visible_count", 0)
                )
                return (
                    appearance_distance,
                    size_distance,
                    -binding_frame,
                    -visible_count,
                    local_id,
                )

            kept_local_id, kept_track = min(members, key=owner_rank)
            detached_local_ids = []
            for local_id, track in members:
                if local_id == kept_local_id:
                    continue
                if self._same_camera_motion_duplicate(kept_track, track):
                    continue
                key = (cam_id, local_id)
                removed = self._local_to_global.pop(key, None)
                if removed is None:
                    continue
                self._gid_members.get(global_id, set()).discard(key)
                self._local_binding_frames.pop(key, None)
                detached_local_ids.append(local_id)
            if detached_local_ids:
                self._event(
                    "same_camera_global_conflict_detached",
                    frame_idx,
                    global_id,
                    camera=cam_id,
                    kept_local_id=int(kept_local_id),
                    detached_local_ids=sorted(detached_local_ids),
                    reason="one_gid_multiple_non_echo_tracks",
                )

    def _bound_handoff_merge_allowed(
        self,
        entry: HandoffEntry,
        target_global_id: int,
        frame_idx: int,
        all_tracks: Dict[str, dict],
    ) -> Tuple[bool, str, dict]:
        """Allow reconciliation only when one GID is genuinely provisional."""
        source_global_id = self._canonical_id(entry.global_id)
        target_global_id = self._canonical_id(target_global_id)
        if source_global_id == target_global_id:
            return True, "already_same", {}
        if self._cross_merge_would_conflict(
            source_global_id, target_global_id, all_tracks
        ):
            return False, "same_camera_live_owner_conflict", {}

        source_birth = self._global_created_frames.get(
            source_global_id, -999999
        )
        target_birth = self._global_created_frames.get(
            target_global_id, -999999
        )
        source_age = int(frame_idx) - int(source_birth)
        target_age = int(frame_idx) - int(target_birth)
        proof = {
            "source_gid_age_frames": source_age,
            "target_gid_age_frames": target_age,
        }

        # Normal race: the destination confirmed and received a new GID one
        # or two frames before the pending handoff matured.
        if target_age <= 3:
            return True, "provisional_target_gid", proof

        # Reverse race: the source itself is a very new fragment. It may merge
        # into an established target only when the established identity has a
        # same-camera gallery proving that both source observations look alike.
        if source_age <= 3:
            target_identity = self._identities.get(target_global_id)
            same_camera_gallery = (
                target_identity.camera_appearance_samples.get(
                    entry.source_cam, ()
                )
                if target_identity is not None
                else ()
            )
            same_view_match = compare_tracklets(
                entry.appearance_samples or entry.appearance,
                same_camera_gallery,
            )
            proof.update(
                same_view_appearance_distance=round(
                    float(same_view_match.distance), 3
                ),
                same_view_tracklet_support=int(same_view_match.support),
            )
            if (
                same_view_match.support > 0
                and same_view_match.distance <= 0.30
            ):
                return True, "provisional_source_same_view", proof
            return False, "provisional_source_without_same_view_proof", proof

        return False, "both_global_ids_established", proof

    def _match_pending_handoffs(
        self,
        all_tracks: Dict[str, dict],
        frame_idx: int,
        camera_timestamps_s: Optional[Dict[str, float]] = None,
    ) -> set[Tuple[str, int]]:
        """Batch-match pending transfers and retain uncertain evidence.

        Bound candidates are included deliberately.  A target may have been
        confirmed and assigned a fresh GID one frame before the transfer had
        enough evidence; a later explicit handoff can safely reconcile that
        temporary identity instead of leaving two GIDs for one vehicle.
        """
        entries = []
        cancelled_entry_ids = set()
        for entry in self._handoffs:
            if frame_idx - entry.updated_at_frame > self.handoff_ttl:
                continue
            if self._has_confirmed_camera_member(
                entry.global_id, entry.target_cam, all_tracks
            ):
                cancelled_entry_ids.add(id(entry))
                self._event(
                    "handoff_cancelled_target_already_active",
                    frame_idx,
                    self._canonical_id(entry.global_id),
                    source_camera=entry.source_cam,
                    target_camera=entry.target_cam,
                    source_local_id=entry.source_local_track_id,
                )
                continue
            entries.append(entry)
        if cancelled_entry_ids:
            self._handoffs = [
                entry
                for entry in self._handoffs
                if id(entry) not in cancelled_entry_ids
            ]
        candidates = [
            (cam_id, local_id, track)
            for cam_id, tracks in all_tracks.items()
            for local_id, track in tracks.items()
            if not (
                (cam_id, local_id) in self._local_to_global
                and any(
                    entry.target_cam == cam_id
                    and self._canonical_id(entry.global_id)
                    == self._canonical_id(
                        self._local_to_global[(cam_id, local_id)]
                    )
                    for entry in entries
                )
            )
        ]
        if not entries or not candidates:
            return set()
        invalid_cost = 10.0
        costs = np.full((len(entries), len(candidates)), invalid_cost, dtype=np.float64)
        details_by_pair = {}
        deferred_keys: set[Tuple[str, int]] = set()
        for row, entry in enumerate(entries):
            for col, (cam_id, local_id, track) in enumerate(candidates):
                if entry.target_cam != cam_id:
                    continue
                cost, reason, details = self._candidate_cost(entry, cam_id, track, frame_idx)
                if cost is None:
                    self._record_rejection(entry, frame_idx, cam_id, local_id, reason, details)
                    continue
                target_global_id = self._local_to_global.get(
                    (cam_id, local_id)
                )
                if (
                    target_global_id is not None
                    and self._canonical_id(target_global_id)
                    != self._canonical_id(entry.global_id)
                ):
                    allowed, bound_reason, bound_proof = (
                        self._bound_handoff_merge_allowed(
                            entry,
                            target_global_id,
                            frame_idx,
                            all_tracks,
                        )
                    )
                    if not allowed:
                        self._event(
                            "handoff_bound_identity_rejected",
                            frame_idx,
                            self._canonical_id(entry.global_id),
                            target_global_id=self._canonical_id(
                                target_global_id
                            ),
                            source_camera=entry.source_cam,
                            target_camera=cam_id,
                            target_local_id=int(local_id),
                            reason=bound_reason,
                            **bound_proof,
                        )
                        continue
                    details["bound_merge_reason"] = bound_reason
                    details.update(bound_proof)
                ambiguity = self._ambiguous_local_identities.get(
                    (cam_id, int(local_id))
                )
                source_global_id = self._canonical_id(entry.global_id)
                if ambiguity is not None:
                    ambiguity_until, competing_ids = ambiguity
                    competing_ids = {
                        self._canonical_id(value)
                        for value in competing_ids
                    }
                    # An almost exact destination-camera appearance may
                    # override stale competition.  Otherwise keep the local
                    # fragment ID-less until the ambiguity expires or the
                    # correct handoff target becomes visible.
                    if (
                        int(frame_idx) <= int(ambiguity_until)
                        and source_global_id not in competing_ids
                        and float(details.get("appearance_distance", 1.0))
                        > 0.20
                    ):
                        if (cam_id, local_id) not in self._local_to_global:
                            deferred_keys.add((cam_id, local_id))
                        self._event(
                            "handoff_rejected_recent_identity_ambiguity",
                            frame_idx,
                            source_global_id,
                            source_camera=entry.source_cam,
                            target_camera=cam_id,
                            target_local_id=int(local_id),
                            competing_global_ids=sorted(competing_ids),
                            ambiguity_until_frame=int(ambiguity_until),
                            appearance_distance=details.get(
                                "appearance_distance"
                            ),
                        )
                        continue
                evidence_key = (
                    source_global_id,
                    cam_id,
                    int(local_id),
                )
                evidence_count = self._advance_consecutive_evidence(
                    self._handoff_candidate_evidence,
                    evidence_key,
                    frame_idx,
                )
                details["evidence_frames"] = evidence_count
                if (
                    details.get("requires_temporal_evidence")
                    and evidence_count < 2
                ):
                    if (cam_id, local_id) not in self._local_to_global:
                        deferred_keys.add((cam_id, local_id))
                    self._event(
                        "handoff_candidate_deferred",
                        frame_idx,
                        self._canonical_id(entry.global_id),
                        source_camera=entry.source_cam,
                        target_camera=cam_id,
                        target_local_id=int(local_id),
                        evidence_frames=evidence_count,
                        predicted_distance=details.get("predicted_distance"),
                        appearance_distance=details.get("appearance_distance"),
                        appearance_reference=details.get("appearance_reference"),
                    )
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
            if not self._assignment_has_margin(
                costs, row, col, invalid_cost, margin=0.08
            ):
                if (cam_id, local_id) not in self._local_to_global:
                    deferred_keys.add((cam_id, local_id))
                self._event(
                    "handoff_assignment_ambiguous",
                    frame_idx,
                    self._canonical_id(entry.global_id),
                    source_camera=entry.source_cam,
                    target_camera=cam_id,
                    target_local_id=int(local_id),
                    selected_cost=round(float(costs[row, col]), 3),
                )
                continue
            source_global_id = self._canonical_id(entry.global_id)
            target_global_id = self._local_to_global.get((cam_id, local_id))
            if target_global_id is not None:
                target_global_id = self._canonical_id(target_global_id)
                if target_global_id != source_global_id:
                    allowed, bound_reason, bound_proof = (
                        self._bound_handoff_merge_allowed(
                            entry,
                            target_global_id,
                            frame_idx,
                            all_tracks,
                        )
                    )
                    if not allowed:
                        self._event(
                            "handoff_bound_identity_rejected",
                            frame_idx,
                            source_global_id,
                            target_global_id=target_global_id,
                            source_camera=entry.source_cam,
                            target_camera=cam_id,
                            target_local_id=int(local_id),
                            reason=bound_reason,
                            phase="post_assignment_recheck",
                            **bound_proof,
                        )
                        continue
                    self._merge_global_ids(
                        source_global_id,
                        target_global_id,
                        frame_idx,
                        "explicit_predictive_handoff",
                    )
                    source_global_id = self._canonical_id(source_global_id)
            source_global_id = self._reconcile_handoff_dormant_alias(
                source_global_id,
                cam_id,
                local_id,
                candidates[col][2],
                frame_idx,
                (camera_timestamps_s or {}).get(cam_id),
                all_tracks,
            )
            self._bind(cam_id, local_id, source_global_id)
            event_type = (
                "handoff_matched_target_gallery"
                if match_details.get("appearance_reference") == "target_camera"
                else "handoff_matched"
            )
            self._event(event_type, frame_idx, source_global_id, source_camera=entry.source_cam,
                        target_camera=cam_id, source_local_id=entry.source_local_track_id,
                        target_local_id=local_id, score=round(float(costs[row, col]), 3),
                        appearance_distance=match_details.get("appearance_distance"),
                        tracklet_support=match_details.get("tracklet_support", 0),
                        evidence_frames=match_details.get("evidence_frames", 1),
                        appearance_reference=match_details.get("appearance_reference"),
                        predicted_position={"x": round(self._predicted_world(entry, frame_idx)[0], 2), "y": round(self._predicted_world(entry, frame_idx)[1], 2)})
            print(f"  [handoff] global #{source_global_id}: {entry.source_cam} -> {cam_id}")
            accepted_entries.append(entry)
        for entry in accepted_entries:
            self._handoffs.remove(entry)
        live_evidence_keys = {
            (
                self._canonical_id(entry.global_id),
                cam_id,
                int(local_id),
            )
            for entry in entries
            for cam_id, local_id, _track in candidates
            if entry.target_cam == cam_id
        }
        self._handoff_candidate_evidence = {
            key: value
            for key, value in self._handoff_candidate_evidence.items()
            if key in live_evidence_keys and frame_idx - value[1] <= 2
        }
        return deferred_keys

    def _reconcile_handoff_dormant_alias(
        self,
        source_global_id: int,
        cam_id: str,
        local_id: int,
        track,
        frame_idx: int,
        timestamp_s: Optional[float],
        all_tracks: Dict[str, dict],
    ) -> int:
        """Collapse an older same-vehicle identity revived by camera history.

        A previously failed transfer can leave G#old dormant in this camera
        while the same physical car currently carries G#new.  A successful
        predictive handoff is the safest moment to collapse those aliases,
        but only with a strong destination-camera gallery and a unique result.
        """
        source_global_id = self._canonical_id(source_global_id)
        world = self._track_world(cam_id, track)
        proposals = []
        for identity in self._identities.values():
            candidate_global_id = self._canonical_id(identity.global_id)
            if candidate_global_id == source_global_id:
                continue
            if identity.state not in {"dormant", "handoff"}:
                continue
            if candidate_global_id in self._parked_reservations:
                continue
            if not self._identity_is_recent(identity, frame_idx, timestamp_s):
                continue
            gallery = identity.camera_appearance_samples.get(cam_id, ())
            if not gallery:
                continue
            appearance_match = compare_tracklets(track, gallery)
            current_tracklet_samples = len(appearance_samples(track))
            if (
                appearance_match.support < 1
                or current_tracklet_samples < 2
                or appearance_match.distance > 0.28
            ):
                continue
            distance = float(
                np.linalg.norm(np.subtract(world, identity.last_world))
            )
            if distance > self.dormant_match_distance:
                continue
            size = self._size_distance(
                (track.w, track.h),
                identity.camera_bbox_sizes.get(cam_id, identity.bbox_size),
            )
            if size > 0.65:
                continue
            cost = (
                0.55 * distance / max(self.dormant_match_distance, 1.0)
                + 0.35 * appearance_match.distance / 0.28
                + 0.10 * size
            )
            proposals.append(
                (
                    cost,
                    candidate_global_id,
                    distance,
                    appearance_match.distance,
                    appearance_match.support,
                )
            )
        proposals.sort(key=lambda item: item[0])
        if not proposals:
            return source_global_id
        if len(proposals) > 1 and proposals[1][0] - proposals[0][0] < 0.12:
            self._event(
                "handoff_dormant_alias_ambiguous",
                frame_idx,
                source_global_id,
                target_camera=cam_id,
                target_local_id=int(local_id),
                competing_global_ids=[
                    int(item[1]) for item in proposals[:3]
                ],
            )
            return source_global_id
        _cost, dormant_global_id, distance, appearance, support = proposals[0]
        if self._cross_merge_would_conflict(
            source_global_id, dormant_global_id, all_tracks
        ):
            self._event(
                "handoff_dormant_alias_ambiguous",
                frame_idx,
                source_global_id,
                target_camera=cam_id,
                target_local_id=int(local_id),
                competing_global_ids=[int(dormant_global_id)],
                reason="same_camera_live_owner_conflict",
            )
            return source_global_id
        kept = min(source_global_id, dormant_global_id)
        retired = max(source_global_id, dormant_global_id)
        self._event(
            "handoff_reconciled_dormant_alias",
            frame_idx,
            kept,
            superseded_global_id=retired,
            target_camera=cam_id,
            target_local_id=int(local_id),
            world_distance=round(distance, 3),
            appearance_distance=round(float(appearance), 3),
            tracklet_support=int(support),
        )
        self._merge_global_ids(
            kept,
            retired,
            frame_idx,
            "handoff_destination_camera_history",
        )
        return self._canonical_id(kept)

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

        # Drop only truly disappeared targets.  The source observation may
        # vanish one frame before the destination tracklet becomes mature;
        # its claim must survive that normal transfer gap.
        self._cross_camera_deferred_since = {
            key: started
            for key, started in self._cross_camera_deferred_since.items()
            if key in present_unbound_keys
        }
        self._cross_camera_deferred_claims = {
            key: claim
            for key, claim in self._cross_camera_deferred_claims.items()
            if (
                key in present_unbound_keys
                and frame_idx - int(claim.get("last_frame", frame_idx))
                <= max(2, self.cross_camera_defer_frames)
            )
        }

        deferred = set()
        expired_deferred_keys: set[Tuple[str, int]] = set()
        for unbound_key, (track, world) in unbound.items():
            claim = self._cross_camera_deferred_claims.get(unbound_key)
            if claim is None:
                continue
            age = frame_idx - int(claim["started_frame"])
            distance = float(
                np.linalg.norm(np.subtract(world, claim["source_world"]))
            )
            appearance_match = compare_tracklets(
                track, claim["source_appearance_samples"]
            )
            size_distance = self._size_distance(
                (track.w, track.h), claim["source_bbox_size"]
            )
            accepted = (
                distance <= self.cross_camera_duplicate_distance * 1.35
                and appearance_match.support >= 2
                and appearance_match.distance <= self.relaxed_appearance_threshold
                and size_distance <= 0.90
            )
            if accepted:
                cam_id, local_id = unbound_key
                global_id = self._canonical_id(int(claim["global_id"]))
                if global_id not in self._parked_reservations:
                    self._bind(cam_id, local_id, global_id)
                    self._event(
                        "cross_camera_deferred_claim_matched",
                        frame_idx,
                        global_id,
                        source_camera=claim["source_camera"],
                        target_camera=cam_id,
                        target_local_id=int(local_id),
                        world_distance=round(distance, 3),
                        appearance_distance=round(
                            float(appearance_match.distance), 3
                        ),
                        tracklet_support=appearance_match.support,
                    )
                    self._cross_camera_deferred_since.pop(unbound_key, None)
                    self._cross_camera_deferred_claims.pop(unbound_key, None)
                    continue
            if age < self.cross_camera_defer_frames:
                claim["last_frame"] = frame_idx
                deferred.add(unbound_key)
            else:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                self._cross_camera_deferred_claims.pop(unbound_key, None)
                expired_deferred_keys.add(unbound_key)

        if not bound or not unbound:
            return deferred

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

        for unbound_key, linked in unbound_neighbours.items():
            if unbound_key not in unbound:
                continue
            if unbound_key in self._local_to_global:
                continue
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
            if accepted and adaptive_appearance:
                started = self._cross_camera_deferred_since.setdefault(
                    unbound_key, frame_idx
                )
                self._cross_camera_deferred_claims[unbound_key] = {
                    "global_id": self._canonical_id(global_id),
                    "source_camera": other_cam,
                    "source_local_id": int(other_local_id),
                    "source_world": tuple(
                        float(value) for value in _other_world
                    ),
                    "source_bbox_size": (other_track.w, other_track.h),
                    "source_appearance_samples": self._tracklet_snapshot(
                        other_track
                    ),
                    "started_frame": int(started),
                    "last_frame": int(frame_idx),
                }
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
                        appearance_distance=round(
                            appearance_match.distance, 3
                        ),
                        appearance_threshold=round(
                            appearance_threshold, 3
                        ),
                        defer_frames=self.cross_camera_defer_frames,
                        reason="relaxed_cross_view_requires_two_frames",
                    )
                continue
            if accepted:
                self._bind(cam_id, local_id, global_id)
                self._cross_camera_deferred_since.pop(unbound_key, None)
                self._cross_camera_deferred_claims.pop(unbound_key, None)
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
                self._cross_camera_deferred_claims.pop(unbound_key, None)
                continue
            if unbound_key in expired_deferred_keys:
                # The complete evidence window already elapsed. Do not create
                # the same claim again in this frame; downstream allocation
                # may now issue a separate GID for this different vehicle.
                continue
            started = self._cross_camera_deferred_since.setdefault(
                unbound_key,
                frame_idx,
            )
            if frame_idx - started >= self.cross_camera_defer_frames:
                self._cross_camera_deferred_since.pop(unbound_key, None)
                self._cross_camera_deferred_claims.pop(unbound_key, None)
                continue
            deferred.add(unbound_key)
            self._cross_camera_deferred_claims[unbound_key] = {
                "global_id": self._canonical_id(global_id),
                "source_camera": other_cam,
                "source_local_id": int(other_local_id),
                "source_world": tuple(float(value) for value in _other_world),
                "source_bbox_size": (other_track.w, other_track.h),
                "source_appearance_samples": self._tracklet_snapshot(other_track),
                "started_frame": int(started),
                "last_frame": int(frame_idx),
            }
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

    def _update_camera_timing(
        self,
        camera_timestamps_s: Optional[Dict[str, float]],
    ) -> None:
        """Learn processed FPS from capture timestamps, not encoded video FPS."""
        for camera_id, raw_timestamp in (camera_timestamps_s or {}).items():
            try:
                timestamp = float(raw_timestamp)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(timestamp):
                continue
            previous = self._camera_last_timestamp.get(camera_id)
            self._camera_last_timestamp[camera_id] = timestamp
            if previous is None:
                continue
            delta = timestamp - previous
            if delta <= 1e-6:
                continue
            self._camera_timestamp_deltas.setdefault(
                camera_id, deque(maxlen=30)
            ).append(delta)

    def effective_camera_fps(self, camera_id: str) -> float:
        deltas = self._camera_timestamp_deltas.get(camera_id, ())
        if len(deltas) >= 5:
            median_delta = float(np.median(np.asarray(deltas, dtype=np.float64)))
            if median_delta > 1e-6:
                return float(np.clip(1.0 / median_delta, 1.0, 60.0))
        return float(self._camera_fps_bootstrap.get(camera_id, 25.0))

    def _reid_window(
        self,
        camera_id: str,
        *,
        uses_seconds: bool,
        evidence: bool = False,
    ) -> float:
        frames = (
            max(self.cross_camera_defer_frames, self.new_identity_min_observations)
            if evidence
            else self.handoff_ttl
        )
        if not uses_seconds:
            return float(max(1, frames))
        return float(max(1, frames)) / self.effective_camera_fps(camera_id)

    def _direction_claim_action(
        self,
        identity: GlobalIdentityState,
        cam_id: str,
        local_id: int,
        track,
        frame_idx: int,
        timestamp_s: Optional[float],
        direction: float,
        appearance: float,
        distance: float,
        tracklet_support: int,
    ) -> str:
        """Return defer/resolve/reject for an otherwise valid ReID pair."""
        global_id = self._canonical_id(identity.global_id)
        key = (global_id, str(cam_id), int(local_id))
        claim = self._direction_reid_claims.get(key)
        created = claim is None
        if claim is None:
            claim = DirectionReIDClaim(
                global_id=global_id,
                camera_id=str(cam_id),
                local_track_id=int(local_id),
                first_frame=int(frame_idx),
                last_frame=int(frame_idx),
                first_time=(float(timestamp_s) if timestamp_s is not None else None),
                last_time=(float(timestamp_s) if timestamp_s is not None else None),
            )
            self._direction_reid_claims[key] = claim
        elif claim.last_frame != int(frame_idx):
            claim.observations += 1
            claim.last_frame = int(frame_idx)
            claim.last_time = float(timestamp_s) if timestamp_s is not None else None

        uses_seconds = timestamp_s is not None and claim.first_time is not None
        age = (
            max(0.0, float(timestamp_s) - float(claim.first_time))
            if uses_seconds
            else float(max(0, int(frame_idx) - claim.first_frame))
        )
        evidence_window = self._reid_window(
            cam_id, uses_seconds=uses_seconds, evidence=True
        )
        evidence_ready = claim.observations >= 3 and int(tracklet_support) >= 2

        if direction >= -0.35 and evidence_ready:
            self._direction_reid_claims.pop(key, None)
            self._event(
                "reid_direction_resolved",
                frame_idx,
                global_id,
                source_camera=identity.last_camera,
                target_camera=cam_id,
                target_local_id=int(local_id),
                observations=claim.observations,
                tracklet_support=int(tracklet_support),
                direction_cosine=round(float(direction), 3),
                effective_fps=round(self.effective_camera_fps(cam_id), 3),
            )
            return "resolve"

        if age <= evidence_window:
            if created:
                self._event(
                    "reid_direction_deferred",
                    frame_idx,
                    global_id,
                    source_camera=identity.last_camera,
                    target_camera=cam_id,
                    target_local_id=int(local_id),
                    direction_cosine=round(float(direction), 3),
                    appearance_distance=round(float(appearance), 3),
                    selected_distance=round(float(distance), 3),
                    evidence_window=round(float(evidence_window), 3),
                    effective_fps=round(self.effective_camera_fps(cam_id), 3),
                )
            return "defer"

        self._direction_reid_claims.pop(key, None)
        self._event(
            "reid_direction_rejected",
            frame_idx,
            global_id,
            source_camera=identity.last_camera,
            target_camera=cam_id,
            target_local_id=int(local_id),
            observations=claim.observations,
            tracklet_support=int(tracklet_support),
            direction_cosine=round(float(direction), 3),
            appearance_distance=round(float(appearance), 3),
            selected_distance=round(float(distance), 3),
            effective_fps=round(self.effective_camera_fps(cam_id), 3),
        )
        return "reject"

    def _cleanup_direction_reid_claims(
        self,
        all_tracks: Dict[str, dict],
    ) -> None:
        live_keys = {
            (str(camera_id), int(local_id))
            for camera_id, tracks in all_tracks.items()
            for local_id in tracks
        }
        self._direction_reid_claims = {
            (self._canonical_id(global_id), camera_id, local_id): claim
            for (global_id, camera_id, local_id), claim in self._direction_reid_claims.items()
            if (camera_id, local_id) in live_keys
            and self._canonical_id(global_id) in self._identities
        }

    def _identity_is_recent(
        self,
        identity: GlobalIdentityState,
        frame_idx: int,
        timestamp_s: Optional[float],
    ) -> bool:
        elapsed, uses_seconds = self._identity_elapsed(identity, frame_idx, timestamp_s)
        limit = self.identity_retention_seconds if uses_seconds else self.identity_retention_frames
        if self._identity_is_established(identity):
            # A mature identity that has been observed in both calibrated
            # cameras is very unlikely to be a one-frame motion artefact. Keep
            # it across longer blind regions; explicit exit zones still retire
            # it immediately. Short/noisy identities retain the normal TTL.
            limit *= 4.0
        return elapsed <= limit

    def _identity_is_established(
        self,
        identity: GlobalIdentityState,
    ) -> bool:
        created_at = self._global_created_frames.get(identity.global_id)
        if created_at is None:
            return False
        lifetime_frames = int(identity.last_seen_frame) - int(created_at)
        mature_lifetime = lifetime_frames >= max(
            30, self.identity_retention_frames
        )
        strong_camera_galleries = sum(
            1
            for samples in identity.camera_appearance_samples.values()
            if samples
        )
        return mature_lifetime and strong_camera_galleries >= 2

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
                predicted_distance = float(
                    np.linalg.norm(np.subtract(world, predicted))
                )
                last_position_distance = float(
                    np.linalg.norm(np.subtract(world, identity.last_world))
                )
                # During a short occlusion the final motion blob can be a
                # clipped tail/hand reflection, producing a bad velocity
                # vector. Treat last position -> extrapolated position as an
                # uncertainty corridor rather than trusting its far endpoint
                # absolutely. Appearance, size and topology remain hard gates.
                distance = (
                    predicted_distance
                    if same_camera
                    else min(predicted_distance, last_position_distance)
                )
                elapsed, uses_seconds = self._identity_elapsed(identity, frame_idx, timestamp_s)
                recent_reid_window = self._reid_window(
                    cam_id, uses_seconds=uses_seconds
                )
                long_cross_camera_gap = (
                    not same_camera and elapsed > 2.0 * recent_reid_window
                )
                target_camera_gallery = identity.camera_appearance_samples.get(
                    cam_id, ()
                )
                has_target_camera_history = bool(target_camera_gallery)
                velocity = (
                    identity.velocity_world_per_second
                    if uses_seconds and np.hypot(*identity.velocity_world_per_second) > 1e-6
                    else identity.velocity_world
                )
                distance_limit = (
                    self.dormant_match_distance
                    if same_camera
                    else (
                        self.dormant_match_distance * 1.60
                        if long_cross_camera_gap
                        and has_target_camera_history
                        and self._identity_is_established(identity)
                        else self.dormant_match_distance
                    )
                    if long_cross_camera_gap
                    else min(
                        self.dormant_match_distance * 2.0,
                        max(
                            (
                                self.dormant_match_distance * 1.60
                                if identity.camera_appearance_samples.get(
                                    cam_id, ()
                                )
                                else self.dormant_match_distance
                            ),
                            self.dormant_match_distance
                            + np.hypot(*velocity)
                            * min(elapsed, 2.0)
                            * 0.25,
                        ),
                    )
                )
                if distance > distance_limit:
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "position",
                        selected_distance=round(distance, 3),
                        extrapolated_distance=round(predicted_distance, 3),
                        last_position_distance=round(last_position_distance, 3),
                        distance_limit=round(float(distance_limit), 3),
                    )
                    continue
                if long_cross_camera_gap and not has_target_camera_history:
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "stale_without_target_camera_gallery",
                        elapsed=round(float(elapsed), 3),
                    )
                    self._event(
                        "dormant_reid_rejected_stale",
                        frame_idx,
                        identity.global_id,
                        source_camera=identity.last_camera,
                        target_camera=cam_id,
                        target_local_id=int(local_id),
                        elapsed=round(float(elapsed), 3),
                        reason="missing_target_camera_gallery",
                    )
                    continue
                appearance_reference = (
                    target_camera_gallery
                    if has_target_camera_history
                    else (identity.appearance_samples or identity.appearance)
                )
                appearance_match = compare_tracklets(track, appearance_reference)
                appearance = appearance_match.distance
                if appearance_match.support <= 0:
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "appearance_missing",
                    )
                    continue
                if same_camera:
                    appearance_threshold = min(
                        0.30, self.dormant_appearance_threshold
                    )
                elif has_target_camera_history:
                    appearance_threshold = min(
                        0.30 if long_cross_camera_gap else 0.45,
                        self.dormant_appearance_threshold,
                    )
                else:
                    # A source-camera histogram is only useful during a short
                    # transfer.  After that it can revive the wrong physical
                    # vehicle merely because it reaches the same location.
                    short_limit = recent_reid_window
                    if elapsed > short_limit:
                        self._record_dormant_rejection(
                            identity,
                            frame_idx,
                            cam_id,
                            local_id,
                            "stale_source_camera_appearance",
                            elapsed=round(float(elapsed), 3),
                        )
                        self._event(
                            "dormant_reid_rejected_stale",
                            frame_idx,
                            identity.global_id,
                            source_camera=identity.last_camera,
                            target_camera=cam_id,
                            target_local_id=int(local_id),
                            elapsed=round(float(elapsed), 3),
                            reason="source_camera_gallery_too_old",
                        )
                        continue
                    appearance_threshold = min(
                        0.45, self.dormant_appearance_threshold
                    )
                ambiguity = self._ambiguous_local_identities.get(
                    (cam_id, int(local_id))
                )
                identity_global_id = self._canonical_id(identity.global_id)
                ambiguity_claim_ids: set[int] = set()
                ambiguity_is_live = False
                if ambiguity is not None:
                    ambiguity_until, raw_claim_ids = ambiguity
                    ambiguity_is_live = int(frame_idx) <= int(
                        ambiguity_until
                    )
                    ambiguity_claim_ids = {
                        self._canonical_id(value)
                        for value in raw_claim_ids
                    }
                if (
                    ambiguity_is_live
                    and identity_global_id not in ambiguity_claim_ids
                ):
                    deferred_keys.add((cam_id, local_id))
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "recent_identity_ambiguity",
                        competing_global_ids=sorted(ambiguity_claim_ids),
                        appearance_distance=round(float(appearance), 3),
                    )
                    self._event(
                        "dormant_reid_rejected_recent_identity_ambiguity",
                        frame_idx,
                        identity.global_id,
                        source_camera=identity.last_camera,
                        target_camera=cam_id,
                        target_local_id=int(local_id),
                        competing_global_ids=sorted(ambiguity_claim_ids),
                        appearance_distance=round(float(appearance), 3),
                    )
                    continue
                if appearance > appearance_threshold:
                    # A short same-camera fragment can contain only a lamp or
                    # windscreen, so its HSV may be slightly worse than the
                    # strict recovery threshold.  Defer allocation instead of
                    # creating a duplicate GID; a persistent genuinely new car
                    # becomes eligible after this bounded grace period.
                    if (
                        same_camera
                        and appearance <= 0.45
                        and elapsed <= recent_reid_window
                    ):
                        deferred_keys.add((cam_id, local_id))
                        self._event(
                            "new_global_id_deferred_dormant_near_miss",
                            frame_idx,
                            identity.global_id,
                            camera=cam_id,
                            local_track_id=int(local_id),
                            appearance_distance=round(
                                float(appearance), 3
                            ),
                            strict_appearance_limit=round(
                                float(appearance_threshold), 3
                            ),
                            elapsed=round(float(elapsed), 3),
                        )
                        continue
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "appearance",
                        appearance_distance=round(float(appearance), 3),
                        appearance_limit=round(float(appearance_threshold), 3),
                        tracklet_support=appearance_match.support,
                    )
                    continue
                size_reference = identity.camera_bbox_sizes.get(
                    cam_id, identity.bbox_size
                )
                size = self._size_distance((track.w, track.h), size_reference)
                if size > 0.92:
                    if (
                        elapsed <= recent_reid_window
                        and has_target_camera_history
                        and self._identity_is_established(identity)
                    ):
                        deferred_keys.add((cam_id, local_id))
                        self._event(
                            "new_global_id_deferred_unstable_reid_size",
                            frame_idx,
                            identity.global_id,
                            source_camera=identity.last_camera,
                            target_camera=cam_id,
                            target_local_id=int(local_id),
                            size_distance=round(float(size), 3),
                            appearance_distance=round(float(appearance), 3),
                            elapsed=round(float(elapsed), 3),
                        )
                    self._record_dormant_rejection(
                        identity,
                        frame_idx,
                        cam_id,
                        local_id,
                        "size",
                        size_distance=round(float(size), 3),
                    )
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
                    direction_claim_key = (
                        identity_global_id,
                        str(cam_id),
                        int(local_id),
                    )
                    has_direction_claim = (
                        direction_claim_key in self._direction_reid_claims
                    )
                    if (
                        direction is not None and direction < -0.35
                    ) or has_direction_claim:
                        action = self._direction_claim_action(
                            identity,
                            cam_id,
                            local_id,
                            track,
                            frame_idx,
                            timestamp_s,
                            float(direction if direction is not None else -1.0),
                            appearance,
                            distance,
                            len(appearance_samples(track)),
                        )
                        if action == "defer":
                            deferred_keys.add((cam_id, local_id))
                            continue
                        if action == "reject":
                            self._record_dormant_rejection(
                                identity,
                                frame_idx,
                                cam_id,
                                local_id,
                                "direction",
                                direction_cosine=(
                                    round(float(direction), 3)
                                    if direction is not None
                                    else None
                                ),
                                selected_distance=round(distance, 3),
                                appearance_distance=round(float(appearance), 3),
                            )
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
                    distance,
                    appearance,
                    elapsed,
                    appearance_match.support,
                    predicted_distance,
                    last_position_distance,
                    (
                        "target_camera"
                        if has_target_camera_history
                        else "source_camera"
                    ),
                )

        _, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.95)
        for row, col in enumerate(row_to_col):
            if col < 0 or costs[row, col] >= invalid_cost:
                continue
            if not self._assignment_has_margin(
                costs, row, col, invalid_cost, margin=0.10
            ):
                identity = identities[row]
                cam_id, local_id, _track = candidates[col]
                deferred_keys.add((cam_id, local_id))
                ambiguity_key = (cam_id, int(local_id))
                ambiguity_until = int(frame_idx) + 3
                previous_ambiguity = self._ambiguous_local_identities.get(
                    ambiguity_key
                )
                competing_ids = (
                    set(previous_ambiguity[1])
                    if previous_ambiguity is not None
                    and previous_ambiguity[0] >= int(frame_idx)
                    else set()
                )
                competing_ids.add(self._canonical_id(identity.global_id))
                self._ambiguous_local_identities[ambiguity_key] = (
                    ambiguity_until,
                    competing_ids,
                )
                self._event(
                    "dormant_reid_ambiguous",
                    frame_idx,
                    identity.global_id,
                    source_camera=identity.last_camera,
                    target_camera=cam_id,
                    target_local_id=int(local_id),
                    selected_cost=round(float(costs[row, col]), 3),
                )
                continue
            identity = identities[row]
            cam_id, local_id, recovered_track = candidates[col]
            previous_camera = identity.last_camera
            self._bind(cam_id, local_id, identity.global_id)
            identity.state = "active"
            identity.dormant_since_frame = None
            identity.dormant_since_time = None
            (
                distance,
                appearance,
                elapsed,
                tracklet_support,
                predicted_distance,
                last_position_distance,
                appearance_reference,
            ) = details_by_pair[(row, col)]
            self._event(
                "dormant_global_id_recovered", frame_idx, identity.global_id,
                source_camera=identity.last_camera, target_camera=cam_id,
                target_local_id=local_id, predicted_distance=round(distance, 2),
                extrapolated_distance=round(predicted_distance, 2),
                last_position_distance=round(last_position_distance, 2),
                appearance_distance=round(appearance, 3), elapsed=round(elapsed, 3),
                tracklet_support=tracklet_support,
                appearance_reference=appearance_reference,
            )
            if cam_id == previous_camera:
                # Local fragmentation in the source camera does not mean the
                # planned cross-camera transfer was cancelled. Move that
                # pending handoff to the recovered local fragment and refresh
                # its prediction. The next frame can then continue updating it
                # normally through `_upsert_handoff()`.
                for entry in self._handoffs:
                    if (
                        self._canonical_id(entry.global_id)
                        != identity.global_id
                        or entry.source_cam != cam_id
                    ):
                        continue
                    entry.source_local_track_id = local_id
                    entry.last_world = self._track_world(cam_id, recovered_track)
                    entry.velocity_world = self._world_velocity(
                        cam_id, recovered_track
                    )
                    entry.bbox_size = (recovered_track.w, recovered_track.h)
                    entry.updated_at_frame = frame_idx
                    self._event(
                        "handoff_source_rebound",
                        frame_idx,
                        identity.global_id,
                        source_camera=cam_id,
                        target_camera=entry.target_cam,
                        source_local_id=local_id,
                    )
            else:
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
            self._global_created_frames.setdefault(global_id, int(frame_idx))
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
                if self._has_confirmed_camera_member(
                    global_id,
                    cam_id,
                    all_tracks,
                    exclude_local_id=local_track_id,
                ):
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
        """Return strong echo evidence without merging two merely touching cars."""
        first_box = (first.x, first.y, first.w, first.h)
        second_box = (second.x, second.y, second.w, second.h)
        first_area = max(1, first.w * first.h)
        second_area = max(1, second.w * second.h)
        if min(first_area, second_area) / max(first_area, second_area) < 0.55:
            return False
        appearance = self._appearance_distance(first, second)
        if appearance > min(self.appearance_threshold, 0.25):
            return False
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box
        gap_x = max(ax - (bx + bw), bx - (ax + aw), 0)
        gap_y = max(ay - (by + bh), by - (ay + ah), 0)
        if np.hypot(gap_x, gap_y) > 14.0:
            return False
        distance = float(np.linalg.norm(np.subtract((first.cx, first.cy), (second.cx, second.cy))))
        largest_dimension = max(first.w, first.h, second.w, second.h)
        if distance > max(28.0, 1.10 * largest_dimension):
            return False
        intersection_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        intersection_y = max(0, min(ay + ah, by + bh) - max(ay, by))
        intersection = intersection_x * intersection_y
        union = first_area + second_area - intersection
        iou = intersection / max(1.0, union)
        history_echo = any(
            float(np.linalg.norm(np.subtract(point, (second.cx, second.cy))))
            <= max(18.0, 0.55 * largest_dimension)
            for point in list(getattr(first, "history", []))[-6:-1]
        ) or any(
            float(np.linalg.norm(np.subtract(point, (first.cx, first.cy))))
            <= max(18.0, 0.55 * largest_dimension)
            for point in list(getattr(second, "history", []))[-6:-1]
        )
        if iou < 0.20 and not history_echo:
            return False
        first_velocity = self._velocity(first)
        second_velocity = self._velocity(second)
        first_norm, second_norm = float(np.hypot(*first_velocity)), float(np.hypot(*second_velocity))
        # For a slow car one/both velocity estimates are often near zero.  In
        # that case proximity is the stronger signal.  Reject only clearly
        # opposing trajectories when both directions are measurable.
        if first_norm >= 1.0 and second_norm >= 1.0:
            cosine = float(np.dot(first_velocity, second_velocity) / (first_norm * second_norm))
            if cosine < 0.60:
                return False
        return True

    def _same_camera_partial_echo(self, fragment, primary) -> bool:
        """Return whether a small touching bbox is part of a larger track.

        Motion differencing often leaves a short rear/front fragment beside a
        full vehicle bbox. It must stay ID-less while the full confirmed track
        exists; binding both would violate the one-track-per-camera invariant.
        """
        fragment_area = max(1, fragment.w * fragment.h)
        primary_area = max(1, primary.w * primary.h)
        area_ratio = fragment_area / primary_area
        if area_ratio > 0.45:
            return False
        if max(fragment.w, fragment.h) > 0.70 * max(primary.w, primary.h):
            return False
        primary_diagonal = max(
            1.0, float(np.hypot(primary.w, primary.h))
        )
        center_distance = float(
            np.linalg.norm(
                np.subtract(
                    (fragment.cx, fragment.cy),
                    (primary.cx, primary.cy),
                )
            )
        )
        if area_ratio <= 0.25 and center_distance <= 1.45 * primary_diagonal:
            # Very small lamp/edge fragments can trail farther than the full
            # bbox at toy-car speed. Defer them while they remain near a much
            # larger trajectory; a genuine separate vehicle may still obtain
            # an ID after it separates and builds its own coherent path.
            return True
        history_echo = any(
            float(
                np.linalg.norm(
                    np.subtract(point, (fragment.cx, fragment.cy))
                )
            )
            <= max(24.0, 0.85 * primary_diagonal)
            for point in list(getattr(primary, "history", ()))[:-1][-10:]
        )
        if history_echo:
            # A small detection on the exact recent path of a larger current
            # track is normally the old/new edge left by frame differencing.
            # Defer only; once a genuinely separate small vehicle leaves that
            # trail it can still accumulate normal new-identity evidence.
            return True
        appearance = self._appearance_distance(fragment, primary)
        # A very small fragment often contains only a lamp/edge, so its HSV is
        # expected to differ from the full vehicle crop. Geometry alone may
        # defer it (never merge it) while the full track remains confirmed.
        if (
            area_ratio > 0.25
            and appearance > min(self.relaxed_appearance_threshold, 0.60)
        ):
            return False
        px1, py1, pw, ph = primary.x, primary.y, primary.w, primary.h
        fx1, fy1, fw, fh = fragment.x, fragment.y, fragment.w, fragment.h
        gap_x = max(px1 - (fx1 + fw), fx1 - (px1 + pw), 0)
        gap_y = max(py1 - (fy1 + fh), fy1 - (py1 + ph), 0)
        if np.hypot(gap_x, gap_y) > max(12.0, 0.24 * primary_diagonal):
            return False
        # Motion echo trails sit just behind the full vehicle and can fall
        # slightly beyond its literal bbox. This path only defers the small
        # fragment; it never merges two Global IDs, so a wider guard is safer
        # than minting a second identity for the same car.
        margin_x = 0.45 * pw
        margin_y = 0.45 * ph
        fragment_center_x = fx1 + fw / 2.0
        fragment_center_y = fy1 + fh / 2.0
        return (
            px1 - margin_x <= fragment_center_x <= px1 + pw + margin_x
            and py1 - margin_y <= fragment_center_y <= py1 + ph + margin_y
        )

    def _same_camera_duplicate_ready(
        self,
        cam_id: str,
        first_local_id: int,
        second_local_id: int,
        frame_idx: int,
    ) -> bool:
        key = (cam_id, *sorted((int(first_local_id), int(second_local_id))))
        count, last_frame = self._same_camera_duplicate_evidence.get(key, (0, -999999))
        if int(frame_idx) == last_frame:
            return count >= 3
        count = count + 1 if int(frame_idx) - last_frame == 1 else 1
        self._same_camera_duplicate_evidence[key] = (count, int(frame_idx))
        return count >= 3

    def _match_same_camera_duplicate(self, cam_id: str, local_track_id: int, track, all_tracks: Dict[str, dict], frame_idx: int) -> Optional[int]:
        """Attach a second motion-echo local track to the existing global ID."""
        for other_local_id, other_track in all_tracks.get(cam_id, {}).items():
            if other_local_id == local_track_id:
                continue
            global_id = self._local_to_global.get((cam_id, other_local_id))
            if global_id is None:
                other_fragment_observations = int(
                    getattr(other_track, "fragment_visible_count", 0) or 0
                )
                if (
                    (
                        self._is_confirmed(other_track)
                        or other_fragment_observations
                        >= self.new_identity_min_observations
                    )
                    and other_track.w * other_track.h
                    > track.w * track.h
                    and self._same_camera_partial_echo(track, other_track)
                ):
                    self._event(
                        "new_global_id_deferred_partial_echo",
                        frame_idx,
                        None,
                        camera=cam_id,
                        local_track_id=int(local_track_id),
                        primary_local_id=int(other_local_id),
                        fragment_area=int(track.w * track.h),
                        primary_area=int(other_track.w * other_track.h),
                        primary_global_id_pending=True,
                    )
                    return 0
                continue
            if self._is_confirmed(other_track) and self._same_camera_partial_echo(
                track, other_track
            ):
                self._event(
                    "new_global_id_deferred_partial_echo",
                    frame_idx,
                    self._canonical_id(global_id),
                    camera=cam_id,
                    local_track_id=int(local_track_id),
                    primary_local_id=int(other_local_id),
                    fragment_area=int(track.w * track.h),
                    primary_area=int(other_track.w * other_track.h),
                )
                return 0
            if not self._same_camera_motion_duplicate(track, other_track):
                continue
            if not self._same_camera_duplicate_ready(
                cam_id, local_track_id, other_local_id, frame_idx
            ):
                self._event(
                    "new_global_id_deferred_ambiguous",
                    frame_idx,
                    self._canonical_id(global_id),
                    camera=cam_id,
                    local_track_id=int(local_track_id),
                    possible_echo_of_local_id=int(other_local_id),
                )
                return 0
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
                    if not self._same_camera_duplicate_ready(
                        cam_id, left_local_id, right_local_id, frame_idx
                    ):
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
                if len(linked) > 1:
                    self._event(
                        "cross_camera_merge_rejected_ambiguous",
                        frame_idx,
                        left_key[1],
                        camera=left_key[0],
                        competing_global_ids=sorted(
                            int(item[1]) for item in linked
                        ),
                    )
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
            evidence_key = tuple(sorted((left_key[1], right_key[1])))
            evidence_count = self._advance_consecutive_evidence(
                self._cross_camera_duplicate_evidence,
                evidence_key,
                frame_idx,
            )
            if evidence_count < 3:
                self._event(
                    "cross_camera_merge_deferred",
                    frame_idx,
                    min(evidence_key),
                    possible_duplicate_global_id=max(evidence_key),
                    source_camera=left_key[0],
                    target_camera=right_key[0],
                    evidence_frames=evidence_count,
                    world_distance=round(distance, 3),
                    appearance_distance=round(
                        float(appearance_match.distance), 3
                    ),
                )
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
            if self._cross_merge_would_conflict(
                left_global_id, right_global_id, all_tracks
            ):
                self._event(
                    "cross_camera_merge_rejected_ambiguous",
                    frame_idx,
                    min(left_global_id, right_global_id),
                    competing_global_ids=[
                        max(left_global_id, right_global_id)
                    ],
                    reason="same_camera_live_owner_conflict",
                )
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
        self._cross_camera_duplicate_evidence = {
            key: value
            for key, value in self._cross_camera_duplicate_evidence.items()
            if frame_idx - value[1] <= 1
        }

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
                identity.camera_appearance_samples[other_cam_id] = (
                    merge_appearance_samples(
                        identity.camera_appearance_samples.get(other_cam_id, ()),
                        other_track,
                        self.tracklet_gallery_size,
                    )
                )
                identity.camera_bbox_sizes[other_cam_id] = (
                    other_track.w,
                    other_track.h,
                )
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
            reservation = self._parked_reservations.get(global_id)
            if reservation is not None:
                identity.state = str(reservation.get("state") or "parked")
                identity.dormant_since_frame = None
                identity.dormant_since_time = None
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
        self._processing_frame_idx = int(frame_idx)
        self._update_camera_timing(camera_timestamps_s)
        self._cleanup_direction_reid_claims(all_tracks)
        self._ambiguous_local_identities = {
            key: value
            for key, value in self._ambiguous_local_identities.items()
            if int(value[0]) >= int(frame_idx)
        }
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
                    self._upsert_handoff(
                        cam_id,
                        local_track_id,
                        track,
                        frame_idx,
                        all_tracks,
                    )

        deferred_handoff = self._match_pending_handoffs(
            assignable_tracks, frame_idx, camera_timestamps_s
        )
        post_handoff_tracks = {
            cam_id: {
                local_track_id: track
                for local_track_id, track in tracks.items()
                if (
                    (cam_id, local_track_id) not in deferred_handoff
                    or (cam_id, local_track_id) in self._local_to_global
                )
            }
            for cam_id, tracks in assignable_tracks.items()
        }

        # Resolve overlapping observations before allocating a new global ID.
        for cam_id, tracks in post_handoff_tracks.items():
            for local_track_id, track in tracks.items():
                if (cam_id, local_track_id) not in self._local_to_global:
                    self._match_simultaneous_overlap(
                        cam_id, local_track_id, track, post_handoff_tracks
                    )

        # Real streams can drop the exact boundary frame, so recover from the
        # durable identity registry before allocating any new ID.
        deferred_dormant = self._match_dormant_identities(
            post_handoff_tracks, frame_idx, camera_timestamps_s
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
            for cam_id, tracks in post_handoff_tracks.items()
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
                    or not self._is_allocatable(track)
                ):
                    continue
                if self._match_same_camera_duplicate(
                    cam_id,
                    local_track_id,
                    track,
                    all_tracks,
                    frame_idx,
                ) is not None:
                    continue
                if not self._new_global_id_ready(
                    cam_id,
                    local_track_id,
                    track,
                    frame_idx,
                ):
                    continue
                allocated_global_id = self._allocate_global_id()
                self._global_created_frames.setdefault(
                    allocated_global_id, int(frame_idx)
                )
                global_id = self._bind(
                    cam_id, local_track_id, allocated_global_id
                )
                self._event("global_id_created", frame_idx, global_id, camera=cam_id, local_track_id=local_track_id)
                # A fast vehicle may receive its first GID while it is already
                # inside the transfer corridor and disappear before the next
                # frame. Publish its handoff immediately; waiting for the next
                # update would lose the only source-side observation.
                self._upsert_handoff(
                    cam_id,
                    local_track_id,
                    track,
                    frame_idx,
                    all_tracks,
                )
                print(
                    f"  [new] global #{global_id} for {cam_id} "
                    f"(local #{local_track_id})"
                )

        self._merge_all_nearby_active_duplicates(all_tracks, frame_idx)
        # Do not irreversibly merge an already-issued ID merely because it is
        # near a recently lost one. Two real vehicles commonly pass side by
        # side in a parking aisle. Unbound fragments are handled earlier by
        # conservative dormant Re-ID; confirmed IDs stay separate unless the
        # explicit same-camera echo or calibrated cross-camera proof succeeds.
        self._merge_unique_cross_camera_duplicates(all_tracks, frame_idx)
        self._resolve_same_camera_global_conflicts(all_tracks, frame_idx)
        self._refresh_identity_states(
            all_tracks, frame_idx, camera_timestamps_s
        )

        current_time_s = max((camera_timestamps_s or {}).values(), default=None)
        self.cleanup(frame_idx, current_time_s)
        result = {
            cam_id: {local_id: self._local_to_global[(cam_id, local_id)] for local_id in tracks if (cam_id, local_id) in self._local_to_global}
            for cam_id, tracks in all_tracks.items()
        }
        self._processing_frame_idx = None
        return result

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
        self._local_binding_frames.pop(key, None)
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
                    camera_appearance_samples={cam_id: gallery},
                    camera_bbox_sizes={cam_id: (bbox_w, bbox_h)},
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
                camera_gallery = merge_appearance_samples(
                    identity.camera_appearance_samples.get(cam_id, ()),
                    appearance_source,
                    self.tracklet_gallery_size,
                )
                identity.camera_appearance_samples[cam_id] = camera_gallery
                identity.camera_bbox_sizes[cam_id] = (bbox_w, bbox_h)
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
                "created_at_frame": self._global_created_frames.get(
                    global_id
                ),
                "last_seen_time": identity.last_seen_time,
                "dormant_since_frame": identity.dormant_since_frame,
                "exited_at_frame": identity.exited_at_frame,
                "appearance_sample_count": len(identity.appearance_samples),
                "camera_appearance_sample_counts": {
                    camera_id: len(samples)
                    for camera_id, samples in sorted(
                        identity.camera_appearance_samples.items()
                    )
                },
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
            "parked_identity_reservations": {
                str(global_id): {
                    "global_id": int(global_id),
                    "slot_id": reservation.get("slot_id"),
                    "camera_id": reservation.get("camera_id"),
                    "state": reservation.get("state", "parked"),
                    "bbox": list(reservation["bbox"])
                    if reservation.get("bbox") is not None
                    else None,
                }
                for global_id, reservation in sorted(
                    self._parked_reservations.items()
                )
            },
            "pending_handoffs": [{
                "global_id": item.global_id, "source_camera": item.source_cam,
                "source_local_track_id": item.source_local_track_id, "target_camera": item.target_cam,
                "exit_edge": item.exit_edge, "last_world": {"x": round(item.last_world[0], 2), "y": round(item.last_world[1], 2)},
                "velocity": {"x": round(item.velocity_world[0], 2), "y": round(item.velocity_world[1], 2)},
                "appearance_sample_count": len(item.appearance_samples),
                "target_appearance_sample_count": len(
                    item.target_appearance_samples
                ),
                "created_at_frame": item.created_at_frame, "updated_at_frame": item.updated_at_frame,
            } for item in self._handoffs],
            "recent_events": self._events,
        }
