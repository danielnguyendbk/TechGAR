"""Appearance descriptors and the multi-sample gallery of PLAN 2 §4.4.

    C_appearance(g, j) = min over the gallery of d(e_k, e_j)
    C_robust           = alpha * d_min + (1 - alpha) * median(d_nearest)

The gallery can be *frozen*: while a detection covers two tracks the blob is a
mixture of two vehicles, and PLAN 2 §7.2 forbids letting it contaminate either
identity's gallery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID = 3


def embed(image: np.ndarray, bbox, mask: np.ndarray | None = None) -> np.ndarray:
    """Colour-layout descriptor: per-cell mean colour, L2 normalised.

    Deterministic, cheap and viewpoint-tolerant enough for corridor-scoped Re-ID,
    which is the only Re-ID this system performs (PLAN 2 §3.5 forbids global
    appearance search, so the descriptor never has to separate a whole facility).
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    height, width = array.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0 = int(np.clip(np.floor(x0), 0, width - 1))
    x1 = int(np.clip(np.ceil(x1), x0 + 1, width))
    y0 = int(np.clip(np.floor(y0), 0, height - 1))
    y1 = int(np.clip(np.ceil(y1), y0 + 1, height))
    patch = array[y0:y1, x0:x1].astype(np.float32)
    if mask is not None:
        sub = mask[y0:y1, x0:x1]
        if sub.any():
            patch = patch * sub[:, :, None]
    cells = []
    rows = np.linspace(0, patch.shape[0], GRID + 1).astype(int)
    cols = np.linspace(0, patch.shape[1], GRID + 1).astype(int)
    for r in range(GRID):
        for c in range(GRID):
            block = patch[rows[r]:max(rows[r] + 1, rows[r + 1]),
                          cols[c]:max(cols[c] + 1, cols[c + 1])]
            cells.append(block.reshape(-1, patch.shape[2]).mean(axis=0))
    vector = np.concatenate(cells).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else vector


def cosine_distance(a, b) -> float:
    """1 - cosine similarity (PLAN 2 §4.4)."""
    if a is None or b is None:
        return 0.5
    va = np.asarray(a, dtype=float).reshape(-1)
    vb = np.asarray(b, dtype=float).reshape(-1)
    if va.size != vb.size:
        return 0.5
    denominator = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denominator < 1e-9:
        return 0.5
    return float(np.clip(1.0 - float(va @ vb) / denominator, 0.0, 2.0))


@dataclass
class AppearanceGallery:
    capacity: int = 8
    samples: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    frozen: bool = False
    frozen_reason: str = ""

    def freeze(self, reason: str = "occlusion_group") -> None:
        self.frozen = True
        self.frozen_reason = reason

    def unfreeze(self) -> None:
        self.frozen = False
        self.frozen_reason = ""

    def add(self, embedding, timestamp: float, quality: float = 1.0) -> bool:
        if embedding is None or self.frozen or quality <= 0.0:
            return False
        self.samples.append(np.asarray(embedding, dtype=np.float32))
        self.timestamps.append(float(timestamp))
        if len(self.samples) > self.capacity:
            self.samples.pop(0)
            self.timestamps.pop(0)
        return True

    def distances(self, embedding) -> list[float]:
        return [cosine_distance(sample, embedding) for sample in self.samples]

    def cost(self, embedding, alpha: float = 0.6) -> float:
        """PLAN 2 §4.4 robust appearance cost; 0.5 when nothing is known."""
        if embedding is None or not self.samples:
            return 0.5
        distances = sorted(self.distances(embedding))
        d_min = distances[0]
        nearest = distances[:max(1, len(distances) // 2)]
        return float(alpha * d_min + (1.0 - alpha) * float(np.median(nearest)))

    @property
    def centroid(self) -> np.ndarray | None:
        if not self.samples:
            return None
        mean = np.mean(np.stack(self.samples), axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-6 else mean
