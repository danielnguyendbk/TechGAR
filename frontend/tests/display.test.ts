import { describe, expect, it } from 'vitest';

import { makeDemoSnapshot } from '@/lib/demo-data';
import { resolveDisplayVehicle, selectSessionVehicle, selectVisibleVehicles } from '@/lib/display/resolve';
import type { DriverSession, RuntimeVehicle } from '@/lib/domain/types';

describe('marker truth table', () => {
  const snapshot = makeDemoSnapshot();
  const vehicle = snapshot.vehicles[0];

  it('renders an observed vehicle at the runtime position', () => {
    expect(resolveDisplayVehicle(vehicle, snapshot)).toMatchObject({ visibleReason: 'observed', position: vehicle.position });
  });

  it('holds a one-frame gap without creating a new marker', () => {
    const missing: RuntimeVehicle = { ...vehicle, observed: false, state: 'temporarily_missing', stale_seconds: 0.1 };
    expect(resolveDisplayVehicle(missing, snapshot)).toMatchObject({ globalId: vehicle.global_id, visibleReason: 'hold' });
  });

  it('hides a stale ghost without a parking slot', () => {
    const ghost: RuntimeVehicle = { ...vehicle, observed: false, state: 'hidden', stale_seconds: 4 };
    expect(resolveDisplayVehicle(ghost, snapshot)).toBeNull();
  });

  it('anchors a parked vehicle to the slot center for 90 snapshots', () => {
    const parked: RuntimeVehicle = { ...vehicle, observed: false, state: 'parked', parked_slot_id: 'D03', stale_seconds: 900 };
    for (let frame = 0; frame < 90; frame += 1) {
      expect(resolveDisplayVehicle({ ...parked, position: [frame, frame] }, snapshot)?.position).toEqual([29.1, 15]);
    }
  });

  it('filters the driver view to exactly its own Global ID', () => {
    const session: DriverSession = { sessionId: 'S42', state: 'WAITING', globalVehicleId: 17, targetSpotId: null, parkedSpotId: null, claimedAt: null, updatedAt: 0 };
    expect(selectSessionVehicle(snapshot, session)?.globalId).toBe(17);
    expect(selectVisibleVehicles(snapshot).length).toBeGreaterThan(1);
  });
});
