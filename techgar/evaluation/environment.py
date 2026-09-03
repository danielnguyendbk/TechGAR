"""Deterministic environmental robustness probes from PLAN 3 section 6."""

from __future__ import annotations

import numpy as np

from ..contracts import FrameRecord
from ..normalization import EnvironmentalNormalizer
from ..shadow import shadow_mask


def _record(sequence: int, timestamp: float, image: np.ndarray) -> FrameRecord:
    height, width = image.shape[:2]
    return FrameRecord("C1", sequence, timestamp, width, height, True, image)


def run_environmental_checks() -> dict[str, bool]:
    """Exercise brightness compensation, shadow rejection and bounded noise."""
    shape = (96, 128)
    normalizer = EnvironmentalNormalizer("C1")
    base = np.full(shape, 100, dtype=np.uint8)
    for sequence in range(6):
        normalizer.process(_record(sequence, sequence * 0.1, base))
    bright = normalizer.process(_record(6, 0.6, np.full(shape, 160, dtype=np.uint8)))

    noise_normalizer = EnvironmentalNormalizer("C1")
    for sequence in range(6):
        noise_normalizer.process(_record(sequence, sequence * 0.1, base))
    rng = np.random.default_rng(1729)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-3, 4, shape), 0, 255).astype(np.uint8)
    noise_result = noise_normalizer.process(_record(6, 0.6, noisy))

    yy, xx = np.indices(shape)
    texture = (90 + ((xx + yy) % 12)).astype(np.float32)
    candidate = np.zeros(shape, dtype=bool)
    candidate[20:76, 24:104] = True
    current = texture.copy()
    current[candidate] *= 0.5
    background_color = np.repeat(texture[..., None], 3, axis=2)
    current_color = background_color.copy()
    current_color[candidate] *= 0.5
    shadow = shadow_mask(current, texture, current_color, background_color, candidate)
    shadow_fraction = float(shadow.shadow[candidate].mean())

    return {
        "brightness_transition": not bright.foreground.any() and bright.quality.unstable,
        "shadow_rejection": shadow_fraction >= 0.70,
        "compression_noise_bounded": not noise_result.foreground.any(),
    }
