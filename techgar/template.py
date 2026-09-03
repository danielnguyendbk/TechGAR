"""Template matching as an explicit *recovery measurement* (PLAN 1 Phase 1.6).

When a track has no detection to associate, direct block matching around the
prediction can still produce a measurement.  It is fed to the filter as a genuine
observation with a wider R (lower confidence), which is what lets a vehicle
survive a gap without any new identity being minted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemplateMatch:
    center: np.ndarray
    score: float
    found: bool


def extract_template(gray: np.ndarray, bbox, size: int = 24) -> np.ndarray | None:
    """Square patch centred on the bbox, subsampled to ``size``."""
    image = np.asarray(gray, dtype=np.float32)
    h, w = image.shape
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = max(6.0, 0.5 * max(x1 - x0, y1 - y0))
    a = int(max(0, round(cx - half)))
    b = int(min(w, round(cx + half)))
    c = int(max(0, round(cy - half)))
    d = int(min(h, round(cy + half)))
    if b - a < 6 or d - c < 6:
        return None
    patch = image[c:d, a:b]
    step_y = max(1, patch.shape[0] // size)
    step_x = max(1, patch.shape[1] // size)
    return patch[::step_y, ::step_x].copy()


def match_template(gray: np.ndarray, template: np.ndarray, center, search: int = 48,
                   stride: int = 2, minimum_score: float = 0.5) -> TemplateMatch:
    """Normalised cross-correlation over a bounded search window."""
    image = np.asarray(gray, dtype=np.float32)
    if template is None or template.size < 16:
        return TemplateMatch(np.asarray(center, dtype=float), 0.0, False)
    patch = np.asarray(template, dtype=np.float32)[::stride, ::stride]
    th, tw = patch.shape
    if th < 3 or tw < 3:
        return TemplateMatch(np.asarray(center, dtype=float), 0.0, False)
    cx, cy = float(center[0]), float(center[1])
    half_w = tw * stride / 2.0
    half_h = th * stride / 2.0
    x0 = int(max(0, np.floor(cx - half_w - search)))
    x1 = int(min(image.shape[1], np.ceil(cx + half_w + search)))
    y0 = int(max(0, np.floor(cy - half_h - search)))
    y1 = int(min(image.shape[0], np.ceil(cy + half_h + search)))
    region = image[y0:y1:stride, x0:x1:stride]
    if region.shape[0] < th or region.shape[1] < tw:
        return TemplateMatch(np.asarray(center, dtype=float), 0.0, False)
    windows = np.lib.stride_tricks.sliding_window_view(region, (th, tw))
    flat = windows.reshape(windows.shape[0], windows.shape[1], -1)
    template_flat = patch.reshape(-1)
    template_centered = template_flat - template_flat.mean()
    template_norm = float(np.linalg.norm(template_centered))
    if template_norm < 1e-6:
        return TemplateMatch(np.asarray(center, dtype=float), 0.0, False)
    means = flat.mean(axis=2, keepdims=True)
    centered = flat - means
    norms = np.linalg.norm(centered, axis=2)
    numerator = centered @ template_centered
    scores = numerator / np.maximum(norms * template_norm, 1e-6)
    index = int(np.argmax(scores))
    row, col = divmod(index, scores.shape[1])
    score = float(scores[row, col])
    best_x = x0 + (col + tw / 2.0) * stride
    best_y = y0 + (row + th / 2.0) * stride
    return TemplateMatch(np.array([best_x, best_y]), score, score >= minimum_score)
