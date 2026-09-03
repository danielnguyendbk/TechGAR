"""Global one-to-one assignment (PLAN 2 §4: "assignment toàn cục, không greedy").

Jonker-Volgenant style shortest-augmenting-path Hungarian solver, O(n^2 m),
written from first principles.  ``+inf`` entries mean *forbidden* (PLAN 2 §4.5:
a topology-invalid candidate "must be removed from the matrix before solving")
and are guaranteed never to appear in the returned matching.
"""

from __future__ import annotations

import numpy as np


def _hungarian_wide(cost: np.ndarray) -> np.ndarray:
    """Optimal row->column assignment for a finite ``(n, m)`` cost, ``n <= m``."""
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)   # p[j] = 1-based row matched to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = np.flatnonzero(~used[1:]) + 1
            cur = cost[i0 - 1, free - 1] - u[i0] - v[free]
            better = cur < minv[free]
            minv[free[better]] = cur[better]
            way[free[better]] = j0
            k = int(np.argmin(minv[free]))
            delta = float(minv[free[k]])
            j1 = int(free[k])
            used_cols = np.flatnonzero(used)
            u[p[used_cols]] += delta
            v[used_cols] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


def solve_assignment(cost) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one matching; forbidden (non-finite) pairs excluded.

    Returns ``[(row, col), ...]`` sorted by row.
    """
    c = np.asarray(cost, dtype=float)
    if c.size == 0:
        return []
    if c.ndim != 2:
        raise ValueError("cost must be 2-D")
    finite = np.isfinite(c)
    if not finite.any():
        return []
    # Any matching that uses a forbidden pair must cost more than every matching
    # that does not, so the solver never prefers one; survivors are filtered out.
    span = float(np.abs(c[finite]).sum()) + 1.0
    big = span * (c.shape[0] + c.shape[1] + 1.0)
    dense = np.where(finite, c, big)
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        dense = dense.T
    matching = _hungarian_wide(dense)
    pairs = []
    for i, j in enumerate(matching):
        if j < 0:
            continue
        r, col = (j, i) if transposed else (i, j)
        if finite[r, col]:
            pairs.append((int(r), int(col)))
    return sorted(pairs)


def assignment_cost(cost, pairs) -> float:
    c = np.asarray(cost, dtype=float)
    return float(sum(c[i, j] for i, j in pairs))


def greedy_assignment(cost) -> list[tuple[int, int]]:
    """Sequential nearest-first matching.

    Present only as the *counter-example* referenced by PLAN 2 §4: greedy
    matching is demonstrably sub-optimal and is never used by the pipeline.
    """
    c = np.asarray(cost, dtype=float)
    order = sorted(
        ((c[i, j], i, j) for i in range(c.shape[0]) for j in range(c.shape[1]) if np.isfinite(c[i, j]))
    )
    taken_rows: set[int] = set()
    taken_cols: set[int] = set()
    pairs = []
    for _, i, j in order:
        if i in taken_rows or j in taken_cols:
            continue
        taken_rows.add(i)
        taken_cols.add(j)
        pairs.append((i, j))
    return sorted(pairs)
