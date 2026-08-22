"""Bounded shared-map trajectory memory for conservative vehicle ReID."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import numpy as np


Point = Tuple[float, float]


@dataclass(frozen=True)
class TrajectorySample:
    frame_idx: int
    timestamp_s: float
    camera_id: str
    local_track_id: int
    world: Point
    bbox_size: Tuple[int, int]


@dataclass(frozen=True)
class TrajectoryEstimate:
    position: Point
    velocity: Point
    speed: float
    curvature_rad: Optional[float]
    observations: int
    first_timestamp_s: float
    last_timestamp_s: float


@dataclass(frozen=True)
class TrajectoryMatchEvidence:
    score: float
    corridor_distance: float
    direction_cosine: Optional[float]
    speed_ratio: Optional[float]
    time_gap_s: float
    observations: int
    stable: bool
    hard_reject_reason: Optional[str] = None
    components: Optional[Dict[str, float]] = None


class WorldTrajectoryMemory:
    """Store numeric trails separately from any debug rendering.

    Global trails and provisional local fragments are kept independently.
    A local fragment is promoted only after another subsystem has proved its
    Global ID, so trajectory proximity alone can never consume an identity.
    """

    def __init__(
        self,
        history_seconds: float = 2.0,
        min_observations: int = 3,
        match_threshold: float = 0.78,
        ambiguity_margin: float = 0.12,
        max_samples: int = 240,
    ) -> None:
        self.history_seconds = max(0.25, float(history_seconds))
        self.min_observations = max(2, int(min_observations))
        self.match_threshold = float(np.clip(match_threshold, 0.0, 1.0))
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.max_samples = max(16, int(max_samples))
        self._global: Dict[int, List[TrajectorySample]] = {}
        self._provisional: Dict[Hashable, List[TrajectorySample]] = {}
        self._parked: set[int] = set()
        self._parked_origins: Dict[int, Point] = {}

    def _append(
        self,
        destination: Dict[Hashable, List[TrajectorySample]],
        key: Hashable,
        sample: TrajectorySample,
    ) -> None:
        samples = destination.setdefault(key, [])
        if samples and (
            samples[-1].frame_idx == sample.frame_idx
            and samples[-1].camera_id == sample.camera_id
            and samples[-1].local_track_id == sample.local_track_id
        ):
            samples[-1] = sample
        else:
            samples.append(sample)
        cutoff = float(sample.timestamp_s) - self.history_seconds
        samples[:] = [item for item in samples if item.timestamp_s >= cutoff]
        if len(samples) > self.max_samples:
            del samples[:-self.max_samples]

    def append_provisional(self, key: Hashable, sample: TrajectorySample) -> None:
        self._append(self._provisional, key, sample)

    def append_global(self, global_id: int, sample: TrajectorySample) -> None:
        global_id = int(global_id)
        if global_id in self._parked:
            return
        self._append(self._global, global_id, sample)

    def promote(self, key: Hashable, global_id: int) -> None:
        incoming = self._provisional.pop(key, [])
        if not incoming:
            return
        global_id = int(global_id)
        destination = self._global.setdefault(global_id, [])
        destination.extend(incoming)
        destination.sort(
            key=lambda item: (
                item.timestamp_s,
                item.frame_idx,
                item.camera_id,
                item.local_track_id,
            )
        )
        deduplicated: List[TrajectorySample] = []
        for sample in destination:
            if deduplicated and (
                deduplicated[-1].frame_idx == sample.frame_idx
                and deduplicated[-1].camera_id == sample.camera_id
                and deduplicated[-1].local_track_id == sample.local_track_id
            ):
                deduplicated[-1] = sample
            else:
                deduplicated.append(sample)
        self._global[global_id] = deduplicated[-self.max_samples :]

    def merge(self, canonical_id: int, duplicate_id: int) -> None:
        canonical_id, duplicate_id = int(canonical_id), int(duplicate_id)
        if canonical_id == duplicate_id:
            return
        incoming = self._global.pop(duplicate_id, [])
        if incoming:
            self._global.setdefault(canonical_id, []).extend(incoming)
            self._global[canonical_id].sort(
                key=lambda item: (item.timestamp_s, item.frame_idx, item.camera_id)
            )
            self._global[canonical_id] = self._global[canonical_id][
                -self.max_samples :
            ]
        if duplicate_id in self._parked:
            self._parked.discard(duplicate_id)
            self._parked.add(canonical_id)
        duplicate_origin = self._parked_origins.pop(duplicate_id, None)
        if duplicate_origin is not None:
            self._parked_origins.setdefault(canonical_id, duplicate_origin)

    def set_parked(
        self,
        global_id: int,
        parked: bool,
        origin: Optional[Point] = None,
    ) -> None:
        global_id = int(global_id)
        if parked:
            self._parked.add(global_id)
            if origin is not None:
                self._parked_origins[global_id] = (
                    float(origin[0]),
                    float(origin[1]),
                )
        else:
            self._parked.discard(global_id)

    def parked_origin(self, global_id: int) -> Optional[Point]:
        return self._parked_origins.get(int(global_id))

    def remove_missing_provisionals(self, live_keys: Iterable[Hashable]) -> None:
        live = set(live_keys)
        self._provisional = {
            key: samples for key, samples in self._provisional.items() if key in live
        }

    @staticmethod
    def _camera_tail(
        samples: List[TrajectorySample],
        *,
        require_local_id: bool = True,
    ) -> List[TrajectorySample]:
        if not samples:
            return []
        camera_id = samples[-1].camera_id
        local_id = samples[-1].local_track_id
        tail: List[TrajectorySample] = []
        for sample in reversed(samples):
            if sample.camera_id != camera_id or (
                require_local_id and sample.local_track_id != local_id
            ):
                if tail:
                    break
                continue
            tail.append(sample)
        tail.reverse()
        return tail

    @staticmethod
    def _estimate(
        samples: List[TrajectorySample],
        *,
        require_local_id: bool = True,
    ) -> Optional[TrajectoryEstimate]:
        tail = WorldTrajectoryMemory._camera_tail(
            samples, require_local_id=require_local_id
        )
        if not tail:
            return None
        steps: List[Tuple[float, float]] = []
        directions: List[np.ndarray] = []
        recent = tail[-6:]
        for first, second in zip(recent, recent[1:]):
            elapsed = float(second.timestamp_s - first.timestamp_s)
            if elapsed <= 1e-4:
                continue
            delta = np.subtract(second.world, first.world).astype(np.float64)
            steps.append((float(delta[0] / elapsed), float(delta[1] / elapsed)))
            norm = float(np.linalg.norm(delta))
            if norm > 1e-4:
                directions.append(delta / norm)
        velocity = (
            (
                float(np.median([item[0] for item in steps[-5:]])),
                float(np.median([item[1] for item in steps[-5:]])),
            )
            if steps
            else (0.0, 0.0)
        )
        curvature = None
        if len(tail) >= 6 and len(directions) >= 3:
            turns = []
            for first, second in zip(directions, directions[1:]):
                turns.append(
                    float(
                        np.arccos(
                            np.clip(float(np.dot(first, second)), -1.0, 1.0)
                        )
                    )
                )
            if turns:
                curvature = float(np.median(turns))
        return TrajectoryEstimate(
            position=tail[-1].world,
            velocity=velocity,
            speed=float(np.hypot(*velocity)),
            curvature_rad=curvature,
            observations=len(tail),
            first_timestamp_s=float(tail[0].timestamp_s),
            last_timestamp_s=float(tail[-1].timestamp_s),
        )

    @staticmethod
    def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
        point_value = np.asarray(point, dtype=np.float64)
        start_value = np.asarray(start, dtype=np.float64)
        segment = np.asarray(end, dtype=np.float64) - start_value
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-8:
            return float(np.linalg.norm(point_value - start_value))
        ratio = float(np.clip(np.dot(point_value - start_value, segment) / denominator, 0.0, 1.0))
        projection = start_value + ratio * segment
        return float(np.linalg.norm(point_value - projection))

    def match(
        self,
        global_id: int,
        provisional_key: Hashable,
        *,
        prediction_radius: float,
        recent_window_s: float,
        appearance_score: float,
        size_score: float,
        topology_score: float,
        min_direction_cosine: float,
        source_camera: Optional[str] = None,
    ) -> Optional[TrajectoryMatchEvidence]:
        global_samples = self._global.get(int(global_id), [])
        if source_camera is not None:
            camera_samples = [
                sample
                for sample in global_samples
                if sample.camera_id == source_camera
            ]
            if camera_samples:
                global_samples = camera_samples
        candidate_samples = self._provisional.get(provisional_key, [])
        source = self._estimate(global_samples, require_local_id=False)
        candidate = self._estimate(candidate_samples)
        if source is None or candidate is None:
            return None
        candidate_tail = self._camera_tail(candidate_samples)
        overlaps_in_time = bool(
            candidate.first_timestamp_s
            <= source.last_timestamp_s
            <= candidate.last_timestamp_s
        )
        if overlaps_in_time:
            candidate_origin = min(
                candidate_tail,
                key=lambda sample: abs(
                    sample.timestamp_s - source.last_timestamp_s
                ),
            ).world
            gap = 0.0
        else:
            candidate_origin = candidate_tail[0].world
            gap = max(
                0.0,
                candidate.first_timestamp_s - source.last_timestamp_s,
            )
        predicted = (
            source.position[0] + source.velocity[0] * gap,
            source.position[1] + source.velocity[1] * gap,
        )
        corridor_distance = self._point_segment_distance(
            candidate_origin, source.position, predicted
        )
        radius = max(1e-6, float(prediction_radius))
        recent_window = max(1e-6, float(recent_window_s))
        distance_score = float(np.clip(1.0 - corridor_distance / radius, 0.0, 1.0))
        time_score = float(np.clip(1.0 - gap / recent_window, 0.0, 1.0))

        direction_cosine = None
        direction_score = 0.5
        if source.speed >= 1e-3 and candidate.speed >= 1e-3:
            direction_cosine = float(
                np.dot(source.velocity, candidate.velocity)
                / (source.speed * candidate.speed)
            )
            direction_score = float(np.clip((direction_cosine + 1.0) * 0.5, 0.0, 1.0))
            if candidate.curvature_rad is not None:
                direction_score *= float(
                    np.clip(1.0 - candidate.curvature_rad / np.pi, 0.5, 1.0)
                )

        speed_ratio = None
        speed_score = 0.5
        if source.speed >= 1e-3 and candidate.speed >= 1e-3:
            speed_ratio = candidate.speed / source.speed
            speed_score = float(
                np.clip(1.0 - abs(np.log(max(speed_ratio, 1e-6))) / np.log(4.0), 0.0, 1.0)
            )

        hard_reject = None
        stable = candidate.observations >= self.min_observations
        if corridor_distance > radius:
            hard_reject = "teleport"
        elif topology_score <= 0.0:
            hard_reject = "camera_topology"
        elif stable and direction_cosine is not None and direction_cosine < min_direction_cosine:
            hard_reject = "stable_wrong_direction"

        score = (
            0.35 * distance_score
            + 0.25 * float(np.clip(appearance_score, 0.0, 1.0))
            + 0.15 * direction_score
            + 0.10 * speed_score
            + 0.10 * min(time_score, float(np.clip(topology_score, 0.0, 1.0)))
            + 0.05 * float(np.clip(size_score, 0.0, 1.0))
        )
        return TrajectoryMatchEvidence(
            score=float(np.clip(score, 0.0, 1.0)),
            corridor_distance=corridor_distance,
            direction_cosine=direction_cosine,
            speed_ratio=speed_ratio,
            time_gap_s=gap,
            observations=candidate.observations,
            stable=stable,
            hard_reject_reason=hard_reject,
            components={
                "corridor": distance_score,
                "appearance": float(np.clip(appearance_score, 0.0, 1.0)),
                "direction": direction_score,
                "speed": speed_score,
                "time_topology": min(
                    time_score, float(np.clip(topology_score, 0.0, 1.0))
                ),
                "size": float(np.clip(size_score, 0.0, 1.0)),
            },
        )

    def match_departure(
        self,
        global_id: int,
        provisional_key: Hashable,
        *,
        prediction_radius: float,
        recent_window_s: float,
        appearance_score: float,
        size_score: float,
        topology_score: float,
    ) -> Optional[TrajectoryMatchEvidence]:
        """Score an outward fragment against a frozen parking origin.

        The velocity before parking points *into* the slot, so ordinary
        trajectory matching would incorrectly reject the legitimate reverse
        direction.  Departure evidence instead requires a stable fragment to
        start near the parked origin and increase its radial distance.
        """
        global_id = int(global_id)
        candidate_samples = self._provisional.get(provisional_key, [])
        tail = self._camera_tail(candidate_samples)
        candidate = self._estimate(candidate_samples)
        origin = self._parked_origins.get(global_id)
        if origin is None:
            global_samples = self._global.get(global_id, [])
            origin = global_samples[-1].world if global_samples else None
        if origin is None or candidate is None or not tail:
            return None

        radius = max(1e-6, float(prediction_radius))
        recent_window = max(1e-6, float(recent_window_s))
        first_distance = float(
            np.linalg.norm(np.subtract(tail[0].world, origin))
        )
        last_distance = float(
            np.linalg.norm(np.subtract(tail[-1].world, origin))
        )
        radial_gain = last_distance - first_distance
        movement = np.subtract(tail[-1].world, tail[0].world).astype(np.float64)
        movement_norm = float(np.linalg.norm(movement))
        outward = np.subtract(tail[-1].world, origin).astype(np.float64)
        outward_norm = float(np.linalg.norm(outward))

        direction_cosine = None
        direction_score = 0.5
        if movement_norm > 1e-4 and outward_norm > 1e-4:
            direction_cosine = float(
                np.dot(movement, outward) / (movement_norm * outward_norm)
            )
            direction_score = float(
                np.clip((direction_cosine + 1.0) * 0.5, 0.0, 1.0)
            )

        distance_score = float(
            np.clip(1.0 - first_distance / radius, 0.0, 1.0)
        )
        duration = max(
            0.0,
            float(candidate.last_timestamp_s - candidate.first_timestamp_s),
        )
        time_score = float(np.clip(1.0 - duration / recent_window, 0.0, 1.0))
        # A departure has no trustworthy pre-parking speed reference: the car
        # may pause and then creep out. Use the robust median velocity as a
        # motion-vs-static signal, while the binder separately requires real
        # outward displacement. This avoids penalising a legitimate slow car.
        motion_floor = max(0.01, (radius / recent_window) * 0.002)
        speed_score = float(
            np.clip(candidate.speed / motion_floor, 0.0, 1.0)
        )
        stable = candidate.observations >= self.min_observations
        hard_reject = None
        if first_distance > radius:
            hard_reject = "teleport"
        elif topology_score <= 0.0:
            hard_reject = "camera_topology"
        elif stable and radial_gain <= 0.0:
            hard_reject = "stable_not_leaving_slot"
        elif stable and direction_cosine is not None and direction_cosine < 0.0:
            hard_reject = "stable_wrong_direction"

        score = (
            0.35 * distance_score
            + 0.25 * float(np.clip(appearance_score, 0.0, 1.0))
            + 0.15 * direction_score
            + 0.10 * speed_score
            + 0.10 * min(time_score, float(np.clip(topology_score, 0.0, 1.0)))
            + 0.05 * float(np.clip(size_score, 0.0, 1.0))
        )
        return TrajectoryMatchEvidence(
            score=float(np.clip(score, 0.0, 1.0)),
            corridor_distance=first_distance,
            direction_cosine=direction_cosine,
            speed_ratio=None,
            time_gap_s=duration,
            observations=candidate.observations,
            stable=stable,
            hard_reject_reason=hard_reject,
            components={
                "corridor": distance_score,
                "appearance": float(np.clip(appearance_score, 0.0, 1.0)),
                "direction": direction_score,
                "speed": speed_score,
                "time_topology": min(
                    time_score, float(np.clip(topology_score, 0.0, 1.0))
                ),
                "size": float(np.clip(size_score, 0.0, 1.0)),
            },
        )

    def global_samples(self, global_id: int) -> Tuple[TrajectorySample, ...]:
        return tuple(self._global.get(int(global_id), ()))

    def provisional_samples(self, key: Hashable) -> Tuple[TrajectorySample, ...]:
        return tuple(self._provisional.get(key, ()))
