import type { WorldPoint } from '@/lib/domain/types';

export function shouldAnimateMarker(
  previous: WorldPoint | undefined,
  current: WorldPoint,
  teleportThreshold = 18,
): boolean {
  if (!previous) return false;
  return Math.hypot(current[0] - previous[0], current[1] - previous[1]) <= teleportThreshold;
}

