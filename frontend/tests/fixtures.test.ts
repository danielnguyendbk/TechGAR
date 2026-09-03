import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { validateRuntimeSnapshot } from '@/lib/domain/schema';

const names = [
  'normal', 'flicker-gap', 'ghost', 'parked-long', 'parked-fallback',
  'driver-isolation', 'off-route', 'post-reset', 'offline',
];

describe('deterministic replay fixtures', () => {
  it.each(names)('%s stays schema-valid and monotonic', (name) => {
    const raw: unknown = JSON.parse(readFileSync(resolve('public', 'fixtures', `${name}.json`), 'utf8'));
    expect(Array.isArray(raw)).toBe(true);
    let previous: number | null = null;
    for (const item of raw as unknown[]) {
      const snapshot = validateRuntimeSnapshot(item, previous);
      previous = snapshot.frame_index;
    }
  });
});
