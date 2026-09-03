"""Planar polygon geometry — the substrate for PLAN 2 §3.3/§3.4 zones and
§5.1 slot overlap metrics.

Polygons are ``(n, 2)`` float arrays.  Clipping is Sutherland-Hodgman, which is
exact when the *clip* polygon is convex; every clip polygon in TechGAR (slot
rectangles, overlap/exit/entry zones, projected vehicle footprints) is convex.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def as_polygon(points) -> np.ndarray:
    poly = np.asarray(points, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"polygon must have shape (n>=3, 2); got {poly.shape}")
    return poly


def signed_area(poly) -> float:
    p = as_polygon(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(poly) -> float:
    return abs(signed_area(poly))


def ensure_ccw(poly) -> np.ndarray:
    p = as_polygon(poly)
    return p if signed_area(p) >= 0.0 else np.ascontiguousarray(p[::-1])


def polygon_centroid(poly) -> np.ndarray:
    p = ensure_ccw(poly)
    a = signed_area(p)
    if abs(a) < EPS:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    return np.array([float(np.dot(x + xn, cross) / (6.0 * a)),
                     float(np.dot(y + yn, cross) / (6.0 * a))])


def polygon_bounds(poly) -> tuple[float, float, float, float]:
    p = as_polygon(poly)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def polygon_extent(poly) -> tuple[float, float]:
    x0, y0, x1, y1 = polygon_bounds(poly)
    return x1 - x0, y1 - y0


def bbox_to_polygon(bbox) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _left_of(p, a, b) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-11


def _edge_intersection(p, q, a, b) -> np.ndarray:
    d1 = q - p
    d2 = b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < EPS:
        return q.copy()
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    return p + t * d1


def clip_polygon(subject, clip) -> np.ndarray | None:
    """Intersection of ``subject`` with the *convex* polygon ``clip``."""
    out = [row for row in ensure_ccw(subject)]
    cp = ensure_ccw(clip)
    for i in range(len(cp)):
        if not out:
            return None
        a, b = cp[i], cp[(i + 1) % len(cp)]
        inp, out = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            cur_in, prev_in = _left_of(cur, a, b), _left_of(prev, a, b)
            if cur_in:
                if not prev_in:
                    out.append(_edge_intersection(prev, cur, a, b))
                out.append(cur)
            elif prev_in:
                out.append(_edge_intersection(prev, cur, a, b))
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=float)


def intersection_area(a, b) -> float:
    inter = clip_polygon(a, b)
    return 0.0 if inter is None else polygon_area(inter)


def polygon_iou(a, b) -> float:
    """PLAN 2 §5.1: |V ∩ S| / |V ∪ S|."""
    inter = intersection_area(a, b)
    union = polygon_area(a) + polygon_area(b) - inter
    return float(inter / union) if union > EPS else 0.0


def polygon_coverage(a, b) -> float:
    """PLAN 2 §5.1: |V ∩ S| / |V| — how much of *a* lies inside *b*."""
    area_a = polygon_area(a)
    return float(intersection_area(a, b) / area_a) if area_a > EPS else 0.0


def point_in_polygon(point, poly) -> bool:
    p = as_polygon(poly)
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(p)
    for i in range(n):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x < x0 + t * (x1 - x0):
                inside = not inside
    return inside


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < EPS else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def point_polygon_distance(point, poly) -> float:
    """0 inside, else the shortest distance to the boundary."""
    p = np.asarray(point, dtype=float)
    if point_in_polygon(p, poly):
        return 0.0
    q = as_polygon(poly)
    return min(_point_segment_distance(p, q[i], q[(i + 1) % len(q)]) for i in range(len(q)))


def scale_polygon(poly, factor: float, about=None) -> np.ndarray:
    p = as_polygon(poly)
    c = polygon_centroid(p) if about is None else np.asarray(about, dtype=float)
    return c + (p - c) * float(factor)


def inflate_polygon(poly, margin: float) -> np.ndarray:
    """Push every vertex radially outwards by ``margin`` world units.

    Used to widen a footprint by its own positional uncertainty (PLAN 1 stage 9
    "footprint mở rộng uncertainty") and to widen the overlap zone by the seam
    budget (PLAN 2 §3.3, expansion driven by uncertainty rather than a global
    arbitrary radius).
    """
    p = as_polygon(poly)
    if margin == 0.0:
        return p.copy()
    c = polygon_centroid(p)
    radial = p - c
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return p + radial / norms * float(margin)


def convex_hull(points) -> np.ndarray:
    """Monotone-chain hull, counter-clockwise."""
    pts = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def half(sequence):
        chain: list[np.ndarray] = []
        for point in sequence:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) <= 0:
                    chain.pop()
                else:
                    break
            chain.append(point)
        return chain

    lower = half(pts)
    upper = half(pts[::-1])
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def oriented_rectangle(far_edge_a, far_edge_b, depth: float, towards) -> np.ndarray:
    """Rectangle built from one edge, extended ``depth`` towards a reference point.

    This is how a world footprint is reconstructed from the observed ground edge
    of a blob: the edge is metric (it lies on the floor) and the missing extent is
    the known vehicle dimension perpendicular to it.
    """
    a = np.asarray(far_edge_a, dtype=float)
    b = np.asarray(far_edge_b, dtype=float)
    edge = b - a
    length = np.linalg.norm(edge)
    if length < EPS:
        edge = np.array([1.0, 0.0])
        length = 1.0
    normal = np.array([-edge[1], edge[0]]) / length
    midpoint = 0.5 * (a + b)
    reference = np.asarray(towards, dtype=float)
    if normal @ (reference - midpoint) < 0:
        normal = -normal
    return np.array([a, b, b + depth * normal, a + depth * normal])
    p = as_polygon(poly)
    if margin == 0.0:
        return p.copy()
    c = polygon_centroid(p)
    radial = p - c
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    return p + radial / norms * float(margin)
