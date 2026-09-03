"""Temporal-difference evidence and the dual-stage AND gate (PLAN 2 §1.2-§1.3).

    M^diff_{t,Delta} = 1( |I~_t - I~_{t-Delta}| > tau^diff )
    M_t             = M^bg_t  AND  ( OR over Delta in {short, long, lag} of M^diff )

The reference family exists because a single Delta cannot serve both a fast and a
near-stationary vehicle; the AND gate exists because background subtraction alone
marks a standing vehicle whenever exposure shifts, while frame differencing alone
misses slow ones (PLAN 2 §1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .config_vision import MotionConfig


@dataclass
class DifferenceResult:
    mask: np.ndarray
    deltas_used: dict[str, float] = field(default_factory=dict)
    brightness_shifts: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


class TemporalDifferenceEvidence:
    """One instance per camera; keeps a short timestamped frame history."""

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self._history: list[tuple[float, np.ndarray]] = []

    def reset(self) -> None:
        self._history.clear()

    @property
    def history_span(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return self._history[-1][0] - self._history[0][0]

    def push(self, timestamp: float, gray: np.ndarray) -> None:
        self._history.append((float(timestamp), np.asarray(gray, dtype=np.float32)))
        horizon = self.config.delta_long + self.config.delta_lag_max + 1.0
        cutoff = float(timestamp) - horizon
        while len(self._history) > 2 and self._history[0][0] < cutoff:
            self._history.pop(0)

    def _closest(self, target: float, tolerance: float):
        best, best_gap = None, float("inf")
        for entry in self._history:
            gap = abs(entry[0] - target)
            if gap < best_gap:
                best, best_gap = entry, gap
        if best is None or best_gap > tolerance:
            return None
        return best

    def _previous(self, now: float):
        for entry in reversed(self._history):
            if entry[0] < now - 1e-9:
                return entry
        return None

    def references(self, timestamp: float) -> tuple[dict[str, tuple[float, np.ndarray]], dict[str, str]]:
        cfg = self.config
        chosen: dict[str, tuple[float, np.ndarray]] = {}
        skipped: dict[str, str] = {}
        for name, delta in (("short", cfg.delta_short), ("long", cfg.delta_long)):
            ref = self._closest(timestamp - delta, max(0.6 * delta, 0.05))
            if ref is None:
                skipped[name] = "no_reference_in_tolerance"
            else:
                chosen[name] = ref
        previous = self._previous(timestamp)
        if previous is None:
            skipped["lag"] = "no_previous_frame"
        else:
            gap = timestamp - previous[0]
            if gap <= cfg.delta_short + 1e-9:
                skipped["lag"] = "no_lag"
            elif gap > cfg.delta_lag_max:
                # Bridging a stream pause would light up the whole frame.
                skipped["lag"] = f"gap_{gap:.3f}s_exceeds_delta_lag_max"
            else:
                chosen["lag"] = previous
        return chosen, skipped

    def evidence(self, gray: np.ndarray, timestamp: float, tau_map: np.ndarray) -> DifferenceResult:
        gray = np.asarray(gray, dtype=np.float32)
        tau_diff = np.asarray(tau_map, dtype=np.float32) * self.config.diff_threshold_scale
        mask = np.zeros(gray.shape, dtype=bool)
        result = DifferenceResult(mask=mask)
        references, skipped = self.references(timestamp)
        result.skipped = skipped
        for name, (ref_t, ref_gray) in references.items():
            # delta_t, PLAN 2 §1.2 — subsampled median: a global brightness offset
            # does not need every pixel to be estimated to 0.01 grey levels.
            shift = float(np.median(gray[::4, ::4] - ref_gray[::4, ::4]))
            compensated = np.clip(ref_gray + shift, 0.0, 255.0)
            mask |= np.abs(gray - compensated) > tau_diff
            result.deltas_used[name] = timestamp - ref_t
            result.brightness_shifts[name] = shift
        result.mask = mask
        return result


def dual_stage_gate(background_mask: np.ndarray, difference: DifferenceResult,
                    enable_frame_difference: bool = True) -> np.ndarray:
    """PLAN 2 §1.3.  With the difference channel ablated the gate degenerates to
    background subtraction alone — the configuration of PLAN 3 experiment B."""
    if not enable_frame_difference:
        return background_mask.copy()
    if not difference.deltas_used:
        # No usable reference yet: fall back to background evidence rather than
        # dropping every observation (an empty AND would blind the pipeline).
        return background_mask.copy()

    # Frame differencing is a good *motion seed*, but its raw intersection with
    # background subtraction contains only the leading/trailing rim of a moving
    # rigid body.  Projecting that rim as though it were the complete vehicle
    # biases the ground anchor and collapses the metric footprint.  Reconstruct
    # every connected background component touched by a difference seed.  The
    # result still requires both evidence channels (an unseeded background ghost
    # is rejected), while retaining the full current silhouette for geometry.
    background = np.asarray(background_mask, dtype=bool)
    seeds = background & np.asarray(difference.mask, dtype=bool)
    if not seeds.any():
        return np.zeros_like(background)
    labels, count = ndi.label(background, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(background)
    touched = np.unique(labels[seeds])
    touched = touched[touched != 0]
    if touched.size == 0:
        return np.zeros_like(background)
    return np.isin(labels, touched)
