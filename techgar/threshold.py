"""Adaptive threshold engine — PLAN 2 §1.4.

    tau_t = tau_0 + alpha * sigma^noise_t + beta * |L_t - L_{t-1}| + gamma * E^illum_t

with sigma^noise estimated per block by MAD (PLAN 2 §1.4: robust to the vehicle
motion that would drag a plain standard deviation).  §1.6 proves no fixed tau can
win in both the high-noise and the low-signal regime, which is why the threshold
is a *map*, recomputed every frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config_vision import ThresholdConfig
from .linalg import MAD_TO_SIGMA, clamp


def luminance(image: np.ndarray) -> np.ndarray:
    """BGR/RGB uint8 -> float32 luminance (order-independent weights)."""
    img = np.asarray(image)
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.shape[2] == 3:
        return (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1]
                + 0.299 * img[:, :, 2]).astype(np.float32)
    return img[..., 0].astype(np.float32)


def block_mad(diff: np.ndarray, block: int, stride: int = 2) -> np.ndarray:
    """Per-block MAD map, upsampled back to the frame shape.

    ``stride`` subsamples inside each block: a noise scale estimated from a
    quarter of the pixels is statistically indistinguishable here and costs a
    quarter of the time.
    """
    h, w = diff.shape
    pad_h = (-h) % block
    pad_w = (-w) % block
    padded = np.pad(diff, ((0, pad_h), (0, pad_w)), mode="edge")
    tiles = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block)
    if stride > 1:
        tiles = tiles[:, ::stride, :, ::stride]
    medians = np.median(tiles, axis=(1, 3), keepdims=True)
    mad = np.median(np.abs(tiles - medians), axis=(1, 3)) * MAD_TO_SIGMA
    upsampled = np.repeat(np.repeat(mad, block, axis=0), block, axis=1)
    return upsampled[:h, :w].astype(np.float32)


@dataclass
class ThresholdResult:
    tau_map: np.ndarray
    tau_global: float
    noise_sigma: float
    luminance: float
    luminance_delta: float
    illumination_event: float
    changed_fraction: float
    brightness_shift: float


class AdaptiveThresholdEngine:
    """One instance per camera; holds the previous frame for the noise estimate."""

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self.config = config or ThresholdConfig()
        self._previous: np.ndarray | None = None
        self._previous_luminance: float | None = None

    def reset(self) -> None:
        self._previous = None
        self._previous_luminance = None

    def update(self, gray: np.ndarray) -> ThresholdResult:
        cfg = self.config
        gray = np.asarray(gray, dtype=np.float32)
        mean_luma = float(gray.mean())
        if self._previous is None:
            diff = np.zeros_like(gray)
            shift = 0.0
        else:
            diff = gray - self._previous
            shift = float(np.median(diff[::4, ::4]))
        noise_map = block_mad(diff, cfg.noise_block)
        noise_sigma = float(np.median(noise_map))
        luminance_delta = 0.0 if self._previous_luminance is None else abs(
            mean_luma - self._previous_luminance)
        changed_fraction = float(np.mean(np.abs(diff) > cfg.illumination_change_px))
        illumination_event = clamp(changed_fraction / max(cfg.illumination_fraction_ref, 1e-6),
                                   0.0, 1.0)
        tau_map = (cfg.tau_0 + cfg.alpha * noise_map + cfg.beta * luminance_delta
                   + cfg.gamma * illumination_event)
        tau_map = np.clip(tau_map, cfg.tau_min, cfg.tau_max).astype(np.float32)
        self._previous = gray
        self._previous_luminance = mean_luma
        return ThresholdResult(
            tau_map=tau_map,
            tau_global=float(np.median(tau_map)),
            noise_sigma=noise_sigma,
            luminance=mean_luma,
            luminance_delta=luminance_delta,
            illumination_event=illumination_event,
            changed_fraction=changed_fraction,
            brightness_shift=shift,
        )


def scalar_threshold(config: ThresholdConfig, noise_sigma: float, luminance_delta: float,
                     illumination_event: float) -> float:
    """The PLAN 2 §1.4 formula in closed form (used by the unit tests)."""
    tau = (config.tau_0 + config.alpha * noise_sigma + config.beta * luminance_delta
           + config.gamma * illumination_event)
    return clamp(tau, config.tau_min, config.tau_max)
