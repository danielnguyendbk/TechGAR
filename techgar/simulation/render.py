"""Rasteriser for the synthetic rig — static floor, depth-sorted boxes, shadows,
structural blind regions, lighting flicker and sensor noise.

Static structure (floor markings, pillars) is deliberately part of the *static*
scene so that background subtraction removes it, exactly as on site; only the
vehicles and the lighting change between frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import VirtualCamera


@dataclass
class RenderOptions:
    noise_sigma: float = 2.0
    shadow: bool = True
    shadow_offset: tuple[float, float] = (1.1, -0.9)
    shadow_gain: float = 0.55
    brightness_gain: float = 1.0
    quantize: int = 0            # >0 emulates coarse compression banding


def fill_polygon(image: np.ndarray, poly: np.ndarray, color, gain: float | None = None) -> None:
    """Even-odd scanline fill of one convex/concave polygon (in place)."""
    h, w = image.shape[:2]
    x0 = max(int(np.floor(poly[:, 0].min())), 0)
    x1 = min(int(np.ceil(poly[:, 0].max())) + 1, w)
    y0 = max(int(np.floor(poly[:, 1].min())), 0)
    y1 = min(int(np.ceil(poly[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = np.zeros(grid_x.shape, dtype=bool)
    n = len(poly)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            straddles = (ay > grid_y) != (by > grid_y)
            t = (grid_y - ay) / (by - ay)
            crossing = ax + t * (bx - ax)
            inside ^= straddles & (grid_x < crossing)
    if not inside.any():
        return
    patch = image[y0:y1, x0:x1]
    if gain is not None:
        patch[inside] = np.clip(patch[inside].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    else:
        patch[inside] = np.asarray(color, dtype=np.uint8)


class Renderer:
    """One renderer per camera; owns that camera's static floor image."""

    def __init__(self, camera: VirtualCamera, slots: dict[str, np.ndarray] | None = None,
                 seed: int = 5) -> None:
        self.camera = camera
        self.rng = np.random.default_rng(seed)
        self.floor = self._build_floor(slots or {})

    def _build_floor(self, slots: dict[str, np.ndarray]) -> np.ndarray:
        h, w = self.camera.height, self.camera.width
        yy, xx = np.mgrid[0:h, 0:w]
        base = 104.0 + 9.0 * (((xx // 48) + (yy // 48)) % 2)
        texture = 5.0 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
        grain = np.random.default_rng(3).normal(0.0, 2.0, size=(h, w))
        floor = np.clip(base + texture + grain, 0, 255)
        image = np.repeat(floor[:, :, None], 3, axis=2).astype(np.uint8)
        for poly in slots.values():          # painted slot markings
            pix = self.camera.project_floor(poly)
            for i in range(len(pix)):
                self._draw_line(image, pix[i], pix[(i + 1) % len(pix)], (176, 176, 170))
        return image

    @staticmethod
    def _draw_line(image, p0, p1, color, thickness: int = 1) -> None:
        h, w = image.shape[:2]
        steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) * 2) + 2
        for u in np.linspace(0.0, 1.0, steps):
            x = int(round(p0[0] + u * (p1[0] - p0[0])))
            y = int(round(p0[1] + u * (p1[1] - p0[1])))
            if 0 <= x < w and 0 <= y < h:
                image[max(0, y - thickness + 1):y + 1, max(0, x - thickness + 1):x + 1] = color

    def render(self, vehicles, t: float, options: RenderOptions | None = None,
               blind_regions=None) -> np.ndarray:
        options = options or RenderOptions()
        image = self.floor.copy()
        cam = self.camera
        if options.shadow:
            for vehicle in vehicles:
                if not vehicle.present(t):
                    continue
                offset = np.asarray(options.shadow_offset, dtype=float)
                fill_polygon(image, cam.project_floor(vehicle.footprint(t) + offset),
                             None, gain=options.shadow_gain)
        faces = []
        for vehicle in vehicles:
            if not vehicle.present(t):
                continue
            for index, face in enumerate(vehicle.box_faces(t)):
                depth = float(np.mean(cam.depth_3d(face)))
                shade = (1.18 if index == 1 else 0.82 if index >= 2 else 0.7)
                if np.min(cam.depth_3d(face)) <= 0.05:
                    continue
                faces.append((depth, cam.project_3d(face),
                              np.clip(np.asarray(vehicle.color, dtype=float) * shade, 0, 255)))
        for _, poly, color in sorted(faces, key=lambda item: -item[0]):
            fill_polygon(image, poly, color)
        for poly in (blind_regions or []):    # static structure hides what is behind it
            pix = cam.project_floor(poly)
            mask_region = np.zeros(image.shape[:2], dtype=bool)
            self._mask_polygon(mask_region, pix)
            image[mask_region] = self.floor[mask_region]
        if options.brightness_gain != 1.0:
            image = np.clip(image.astype(np.float32) * options.brightness_gain, 0, 255).astype(np.uint8)
        if options.noise_sigma > 0:
            noise = self.rng.normal(0.0, options.noise_sigma, size=image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if options.quantize > 1:
            step = options.quantize
            image = ((image // step) * step).astype(np.uint8)
        return image

    @staticmethod
    def _mask_polygon(mask: np.ndarray, poly: np.ndarray) -> None:
        stub = np.zeros((*mask.shape, 3), dtype=np.uint8)
        fill_polygon(stub, poly, (255, 255, 255))
        mask |= stub[:, :, 0] > 0

    def silhouette_polygon(self, vehicle, t: float) -> np.ndarray:
        """Pixel polygon of the vehicle's ground footprint (used for ground truth)."""
        return self.camera.project_floor(vehicle.footprint(t))
