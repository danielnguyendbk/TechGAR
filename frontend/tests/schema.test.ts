import { describe, expect, it } from 'vitest';

import { makeDemoSnapshot } from '@/lib/demo-data';
import { SnapshotValidationError, validateRuntimeSnapshot } from '@/lib/domain/schema';

describe('snapshot boundary', () => {
  it('accepts the canonical 1.0 contract', () => {
    const snapshot = makeDemoSnapshot(1_000);
    expect(validateRuntimeSnapshot(snapshot).frame_index).toBe(1264);
  });

  it.each([
    ['unknown schema', { schema_version: '2.0' }],
    ['GPS use', { gps_used: true }],
    ['non-finite time', { timestamp: Number.NaN }],
  ])('rejects %s', (_name, change) => {
    expect(() => validateRuntimeSnapshot({ ...makeDemoSnapshot(), ...change })).toThrow(SnapshotValidationError);
  });

  it('rejects regressing frames and duplicate Global IDs', () => {
    const snapshot = makeDemoSnapshot();
    expect(() => validateRuntimeSnapshot(snapshot, snapshot.frame_index + 1)).toThrow(/regressed/);
    expect(() => validateRuntimeSnapshot({ ...snapshot, vehicles: [snapshot.vehicles[0], snapshot.vehicles[0]] })).toThrow(/unique/);
  });
});
