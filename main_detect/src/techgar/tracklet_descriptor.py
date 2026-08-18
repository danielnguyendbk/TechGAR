"""Lightweight appearance tracklets for cross-camera vehicle association.

The graph-based MTMCT literature compares a sequence of observations instead
of reducing a vehicle to one last crop.  This module keeps that useful idea
without requiring a neural graph model: every local track owns a small gallery
of HSV histograms and matching uses all samples in both galleries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def hsv_histogram(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    """Return a normalized 2-D HSV histogram for a clipped vehicle crop."""
    x, y, width, height = box
    frame_height, frame_width = frame.shape[:2]
    left = max(0, min(frame_width, int(x)))
    top = max(0, min(frame_height, int(y)))
    right = max(left, min(frame_width, int(x + width)))
    bottom = max(top, min(frame_height, int(y + height)))
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros((16, 16), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).astype(np.float32)


def _normalized_copy(histogram: np.ndarray) -> Optional[np.ndarray]:
    value = np.asarray(histogram, dtype=np.float32)
    if value.size == 0 or not np.all(np.isfinite(value)):
        return None
    copied = value.copy()
    if float(np.linalg.norm(copied)) <= 1e-12:
        return None
    return cv2.normalize(copied, copied).astype(np.float32)


def histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Bhattacharyya distance normalized to the range used by OpenCV."""
    return float(
        cv2.compareHist(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )


@dataclass
class AppearanceTracklet:
    """Bounded, temporally sampled appearance gallery for one local track."""

    max_samples: int = 12
    sample_interval: int = 3
    samples: List[np.ndarray] = field(default_factory=list)
    sample_frames: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_samples = max(1, int(self.max_samples))
        self.sample_interval = max(1, int(self.sample_interval))

    def update(self, histogram: Optional[np.ndarray], frame_idx: int) -> bool:
        """Append one observation when enough frames passed since the last one."""
        if histogram is None:
            return False
        normalized = _normalized_copy(histogram)
        if normalized is None:
            return False
        if self.sample_frames and frame_idx - self.sample_frames[-1] < self.sample_interval:
            return False
        self.samples.append(normalized)
        self.sample_frames.append(int(frame_idx))
        if len(self.samples) > self.max_samples:
            overflow = len(self.samples) - self.max_samples
            del self.samples[:overflow]
            del self.sample_frames[:overflow]
        return True

    def snapshot(self) -> Tuple[np.ndarray, ...]:
        return tuple(sample.copy() for sample in self.samples)

    @property
    def aggregate(self) -> Optional[np.ndarray]:
        return aggregate_appearance(self.samples)


@dataclass(frozen=True)
class TrackletMatch:
    distance: float
    support: int
    sample_pairs: int
    aggregate_distance: float
    nearest_distance: float


def appearance_samples(source) -> Tuple[np.ndarray, ...]:
    """Extract a copied gallery from a track, descriptor, array or sequence."""
    if source is None:
        return ()
    if isinstance(source, AppearanceTracklet):
        return source.snapshot()

    descriptor = getattr(source, "appearance_tracklet", None)
    if isinstance(descriptor, AppearanceTracklet) and descriptor.samples:
        return descriptor.snapshot()
    if descriptor is not None and hasattr(descriptor, "snapshot"):
        samples = descriptor.snapshot()
        if samples:
            return tuple(
                normalized
                for item in samples
                if (normalized := _normalized_copy(item)) is not None
            )

    if not isinstance(source, (np.ndarray, list, tuple)):
        source = getattr(source, "appearance", None)
        if source is None:
            return ()
    values: Sequence[np.ndarray]
    if isinstance(source, np.ndarray):
        values = (source,)
    elif isinstance(source, (list, tuple)):
        values = source
    else:
        return ()
    return tuple(
        normalized
        for item in values
        if (normalized := _normalized_copy(item)) is not None
    )


def aggregate_appearance(source) -> Optional[np.ndarray]:
    """Median descriptor, used for compatibility with single-histogram code."""
    samples = appearance_samples(source)
    if not samples:
        return None
    aggregate = np.median(np.stack(samples, axis=0), axis=0).astype(np.float32)
    if float(np.linalg.norm(aggregate)) <= 1e-12:
        aggregate = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    return cv2.normalize(aggregate, aggregate).astype(np.float32)


def merge_appearance_samples(
    existing,
    incoming,
    max_samples: int = 24,
) -> Tuple[np.ndarray, ...]:
    """Merge local tracklets into a bounded cross-camera identity gallery."""
    maximum = max(1, int(max_samples))
    merged: List[np.ndarray] = []
    for sample in (*appearance_samples(existing), *appearance_samples(incoming)):
        # Repeated manager updates often carry the same local gallery. Avoid
        # filling the global identity with near-identical copies.
        duplicate_index = next(
            (
                index
                for index, current in enumerate(merged)
                if histogram_distance(current, sample) <= 0.015
            ),
            None,
        )
        if duplicate_index is not None:
            merged[duplicate_index] = sample.copy()
        else:
            merged.append(sample.copy())
    if len(merged) <= maximum:
        return tuple(merged)
    # Preserve observations across the complete route, including earlier
    # cameras, while keeping the gallery deterministic and bounded.
    indices = np.linspace(0, len(merged) - 1, maximum, dtype=int)
    return tuple(merged[int(index)].copy() for index in indices)


def compare_tracklets(left, right, missing_distance: float = 0.25) -> TrackletMatch:
    """Compare two galleries using robust symmetric nearest-neighbour evidence."""
    left_samples = appearance_samples(left)
    right_samples = appearance_samples(right)
    if not left_samples or not right_samples:
        fallback = float(missing_distance)
        return TrackletMatch(fallback, 0, 0, fallback, fallback)

    pairwise = np.empty((len(left_samples), len(right_samples)), dtype=np.float32)
    for row, left_sample in enumerate(left_samples):
        for column, right_sample in enumerate(right_samples):
            pairwise[row, column] = histogram_distance(left_sample, right_sample)

    left_nearest = pairwise.min(axis=1)
    right_nearest = pairwise.min(axis=0)
    symmetric_nearest = np.concatenate((left_nearest, right_nearest))
    # A destination tracklet is usually still short when handoff runs. Match
    # every node of the smaller graph to its best node in the larger graph;
    # old viewpoints in the longer source route should not veto a good entry.
    partial_nearest = (
        left_nearest
        if len(left_samples) <= len(right_samples)
        else right_nearest
    )
    nearest_distance = float(np.median(partial_nearest))
    coverage_distance = float(np.median(symmetric_nearest))
    left_aggregate = aggregate_appearance(left_samples)
    right_aggregate = aggregate_appearance(right_samples)
    aggregate_distance = histogram_distance(left_aggregate, right_aggregate)
    top_count = min(3, pairwise.size)
    best_pairs = float(np.mean(np.partition(pairwise.ravel(), top_count - 1)[:top_count]))
    distance = (
        0.50 * nearest_distance
        + 0.20 * coverage_distance
        + 0.20 * aggregate_distance
        + 0.10 * best_pairs
    )
    return TrackletMatch(
        distance=float(np.clip(distance, 0.0, 1.0)),
        support=min(len(left_samples), len(right_samples)),
        sample_pairs=int(pairwise.size),
        aggregate_distance=aggregate_distance,
        nearest_distance=nearest_distance,
    )
