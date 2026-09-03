"""Shadow rejection — PLAN 2 §1.5, all conditions simultaneously, fail-open.

A region is shadow only when the luminance is attenuated *inside a bounded band*,
the chromaticity is unchanged, the background texture still shows through, and no
independent boundary evidence supports a real object.  The fail-open clause is
what keeps a dark vehicle from being erased as its own shadow.

Everything is evaluated inside the bounding window of the candidate mask: the
test only has meaning where foreground evidence already exists, and a full-frame
evaluation would spend 25 ms per frame proving that empty floor is empty floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config_vision import ShadowConfig


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    gy = np.zeros_like(img)
    gx = np.zeros_like(img)
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    return np.hypot(gx, gy)


def chromaticity(color: np.ndarray) -> np.ndarray:
    """Unit-norm colour direction; shadow scales luminance but keeps direction."""
    c = np.asarray(color, dtype=np.float32)
    norm = np.sqrt((c * c).sum(axis=-1, keepdims=True))
    return c / np.maximum(norm, 1e-6)


@dataclass
class ShadowResult:
    shadow: np.ndarray
    attenuation_band: np.ndarray
    chroma_stable: np.ndarray
    texture_visible: np.ndarray
    object_support: np.ndarray


def _window(mask: np.ndarray, margin: int = 3):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, h)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, w)
    return slice(y0, y1), slice(x0, x1)


def shadow_mask(gray: np.ndarray, background_gray: np.ndarray, color: np.ndarray | None,
                background_color: np.ndarray | None, candidate: np.ndarray,
                config: ShadowConfig | None = None) -> ShadowResult:
    cfg = config or ShadowConfig()
    empty = np.zeros(np.shape(gray), dtype=bool)
    result = ShadowResult(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy())
    if not cfg.enable:
        return result
    box = _window(candidate)
    if box is None:
        return result
    rows, cols = box
    g = np.asarray(gray, dtype=np.float32)[rows, cols]
    b = np.asarray(background_gray, dtype=np.float32)[rows, cols]
    local_candidate = candidate[rows, cols]
    ratio = g / np.maximum(b, 1e-3)

    # Condition 1 — attenuation inside the shadow band.
    band = (ratio > cfg.a_min) & (ratio < cfg.a_max)

    # Condition 2 — chromaticity unchanged.
    if color is not None and background_color is not None:
        delta = np.abs(chromaticity(np.asarray(color)[rows, cols])
                       - chromaticity(np.asarray(background_color)[rows, cols])).sum(axis=2)
        chroma_stable = delta < cfg.epsilon_chroma
    else:
        chroma_stable = np.ones(g.shape, dtype=bool)

    # Condition 3 — background texture still visible: a shadow scales gradients by
    # the same factor as luminance, an object replaces them.
    grad_now = gradient_magnitude(g)
    grad_bg = gradient_magnitude(b)
    preserved = grad_now / np.maximum(grad_bg * ratio, 1e-3)
    low = cfg.texture_correlation_min
    texture_visible = (preserved > low) & (preserved < 1.0 / max(low, 1e-3))

    # Condition 4 (fail-open) — independent boundary support means "not shadow".
    # "Independent" is the operative word: a shadow's own rim is a luminance step
    # too, so only gradients that shadow physics cannot explain (i.e. where the
    # chromaticity or the attenuation band test fails) count as object support.
    shadow_like = band & chroma_stable
    new_edges = (grad_now > (grad_bg * ratio + 12.0)) & ~shadow_like
    # A 3x3 support window keeps the fail-open halo one pixel wide; a wider window
    # would protect a shadow rim several pixels deep and bias the ground anchor.
    strength = ndi.uniform_filter(new_edges.astype(np.float32), size=3, mode="nearest")
    object_support = strength > cfg.edge_support_ratio

    local_shadow = band & chroma_stable & texture_visible & ~object_support & local_candidate
    result.shadow[rows, cols] = local_shadow
    result.attenuation_band[rows, cols] = band
    result.chroma_stable[rows, cols] = chroma_stable
    result.texture_visible[rows, cols] = texture_visible
    result.object_support[rows, cols] = object_support
    return result
