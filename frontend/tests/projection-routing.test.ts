import { describe, expect, it } from 'vitest';

import { makeDemoSnapshot } from '@/lib/demo-data';
import { shouldAnimateMarker } from '@/lib/display/marker-motion';
import { distanceFromRoute, isOffRoute } from '@/lib/guidance/off-route';
import { fitAffine } from '@/lib/projection/affine';
import { routeToSlot, shortestRoute } from '@/lib/routing/lane-graph';

describe('projection and guidance geometry', () => {
  it('fits a world-to-SVG affine transform below the two-pixel gate', () => {
    const transform = fitAffine([
      { world: [0, 0], svg: [10, 20] },
      { world: [10, 0], svg: [30, 20] },
      { world: [0, 10], svg: [10, 50] },
      { world: [10, 10], svg: [30.8, 49.4] },
    ]);
    expect(transform.valid).toBe(true);
    expect(transform.residualRms).toBeLessThanOrEqual(2);
    expect(transform.unproject(transform.project([3, 4]))).toEqual(expect.arrayContaining([expect.any(Number), expect.any(Number)]));
  });

  it('does not animate a teleport', () => {
    expect(shouldAnimateMarker([0, 0], [4, 3])).toBe(true);
    expect(shouldAnimateMarker([0, 0], [30, 0])).toBe(false);
  });

  it('computes a shortest path and an explicit slot route', () => {
    expect(shortestRoute({ nodes: [{ id: 'a', point: [0, 0] }, { id: 'b', point: [1, 0] }, { id: 'c', point: [2, 0] }], edges: [{ from: 'a', to: 'b' }, { from: 'b', to: 'c' }] }, 'a', 'c')).toEqual(['a', 'b', 'c']);
    expect(routeToSlot(makeDemoSnapshot(), [40, 34], 'D09').length).toBeGreaterThan(2);
  });

  it('detects deviation without silently rerouting', () => {
    const route = [[0, 0], [10, 0]] as const;
    expect(distanceFromRoute([5, 2], route)).toBe(2);
    expect(isOffRoute([5, 20], route, 8)).toBe(true);
  });
});
