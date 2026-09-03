import type { WorldPoint } from '@/lib/domain/types';

function segmentDistance(point: WorldPoint, a: WorldPoint, b: WorldPoint): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(point[0] - a[0], point[1] - a[1]);
  const projection = Math.max(0, Math.min(1,
    ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / lengthSquared,
  ));
  return Math.hypot(point[0] - (a[0] + projection * dx), point[1] - (a[1] + projection * dy));
}

export function distanceFromRoute(point: WorldPoint, route: readonly WorldPoint[]): number {
  if (route.length === 0) return Number.POSITIVE_INFINITY;
  if (route.length === 1) return Math.hypot(point[0] - route[0][0], point[1] - route[0][1]);
  return Math.min(...route.slice(1).map((end, index) => segmentDistance(point, route[index], end)));
}

export function isOffRoute(
  point: WorldPoint,
  route: readonly WorldPoint[],
  threshold = 8,
): boolean {
  return distanceFromRoute(point, route) > threshold;
}

