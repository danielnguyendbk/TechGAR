"""Background model and background-subtraction evidence (PLAN 2 §1.1).

    M^bg_t(x,y) = 1( |I_t(x,y) - B_t(x,y)| > tau^bg_t(x,y) )

Foreground pixels are absorbed at a much slower rate than background pixels
rather than being frozen: a vehicle that parks and stops moving *does* eventually
dissolve into the background, which is precisely why slot ownership may not
depend on continued motion evidence (PLAN 2 §5.7) and why ablation B (no frame
difference) is expected to lose slow vehicles (PLAN 3 §7).
"""

from __future__ import annotations

import numpy as np

from .config_vision import BackgroundConfig


class BackgroundModel:
    def __init__(self, config: BackgroundConfig | None = None) -> None:
        self.config = config or BackgroundConfig()
        self.gray: np.ndarray | None = None
        self.color: np.ndarray | None = None
        self.frames_seen = 0

    @property
    def ready(self) -> bool:
        return self.gray is not None and self.frames_seen >= self.config.init_frames

    def reset(self) -> None:
        self.gray = None
        self.color = None
        self.frames_seen = 0

    def initialise(self, gray: np.ndarray, color: np.ndarray | None = None) -> None:
        self.gray = np.asarray(gray, dtype=np.float32).copy()
        if color is not None:
            self.color = np.asarray(color, dtype=np.float32).copy()
        self.frames_seen = 1

    def evidence(self, gray: np.ndarray, tau_map: np.ndarray) -> np.ndarray:
        """M^bg_t — everything is foreground until the model has warmed up."""
        if self.gray is None:
            return np.zeros(np.shape(gray), dtype=bool)
        return np.abs(np.asarray(gray, dtype=np.float32) - self.gray) > tau_map

    def update(self, gray: np.ndarray, foreground: np.ndarray | None = None,
               color: np.ndarray | None = None) -> None:
        gray = np.asarray(gray, dtype=np.float32)
        if self.gray is None:
            self.initialise(gray, color)
            return
        cfg = self.config
        # Learn everywhere at the background rate, then claw back the difference on
        # foreground pixels, so no full-frame rate array has to be materialised.
        correction = cfg.learn_rate - cfg.absorb_rate
        delta = gray - self.gray
        self.gray += cfg.learn_rate * delta
        if foreground is not None and correction != 0.0:
            self.gray[foreground] -= correction * delta[foreground]
        if color is not None:
            color = np.asarray(color, dtype=np.float32)
            if self.color is None or self.color.shape != color.shape:
                self.color = color.copy()
            else:
                delta_c = color - self.color
                self.color += cfg.learn_rate * delta_c
                if foreground is not None and correction != 0.0:
                    self.color[foreground] -= correction * delta_c[foreground]
        self.frames_seen += 1
