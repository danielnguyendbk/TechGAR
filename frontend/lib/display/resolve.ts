import type {
  DisplayVehicle,
  DriverSession,
  RuntimeSnapshot,
  RuntimeVehicle,
  WorldPoint,
} from '@/lib/domain/types';

export function polygonCenter(points: readonly WorldPoint[]): WorldPoint {
  if (points.length === 0) return [0, 0];
  const total = points.reduce<WorldPoint>(
    (sum, point) => [sum[0] + point[0], sum[1] + point[1]],
    [0, 0],
  );
  return [total[0] / points.length, total[1] / points.length];
}

function slotCenter(snapshot: RuntimeSnapshot, slotId: string | null): WorldPoint | null {
  if (!slotId) return null;
  const layout = snapshot.slot_layout.find((slot) => slot.slot_id === slotId);
  return layout ? polygonCenter(layout.polygon) : null;
}

export function resolveDisplayVehicle(
  vehicle: RuntimeVehicle,
  snapshot: RuntimeSnapshot,
): DisplayVehicle | null {
  if (vehicle.observed) {
    return {
      globalId: vehicle.global_id,
      position: vehicle.position,
      state: 'observed',
      parkedSlotId: vehicle.parked_slot_id,
      visibleReason: 'observed',
      uncertainty: vehicle.uncertainty,
    };
  }
  if (vehicle.parked_slot_id !== null) {
    return {
      globalId: vehicle.global_id,
      position: slotCenter(snapshot, vehicle.parked_slot_id) ?? vehicle.position,
      state: 'parked',
      parkedSlotId: vehicle.parked_slot_id,
      visibleReason: 'parked',
      uncertainty: vehicle.uncertainty,
    };
  }
  if (vehicle.stale_seconds <= vehicle.display_hold_seconds) {
    return {
      globalId: vehicle.global_id,
      position: vehicle.position,
      state: 'temporarily_missing',
      parkedSlotId: null,
      visibleReason: 'hold',
      uncertainty: vehicle.uncertainty,
    };
  }
  return null;
}

export function selectVisibleVehicles(snapshot: RuntimeSnapshot): DisplayVehicle[] {
  return snapshot.vehicles
    .map((vehicle) => resolveDisplayVehicle(vehicle, snapshot))
    .filter((vehicle): vehicle is DisplayVehicle => vehicle !== null);
}

export function selectSessionVehicle(
  snapshot: RuntimeSnapshot,
  session: DriverSession,
): DisplayVehicle | null {
  const own = snapshot.vehicles.find(
    (vehicle) => vehicle.global_id === session.globalVehicleId,
  );
  if (own) return resolveDisplayVehicle(own, snapshot);
  const fallbackSlot = session.parkedSpotId ?? session.targetSpotId;
  const center = slotCenter(snapshot, fallbackSlot);
  if (session.globalVehicleId !== null && fallbackSlot && center) {
    return {
      globalId: session.globalVehicleId,
      position: center,
      state: session.parkedSpotId ? 'parked' : 'temporarily_missing',
      parkedSlotId: session.parkedSpotId,
      visibleReason: 'slot-fallback',
      uncertainty: 0,
    };
  }
  return null;
}

export function selectCounts(snapshot: RuntimeSnapshot) {
  const visibleVehicles = selectVisibleVehicles(snapshot).length;
  const occupied = snapshot.parking_slots.filter((slot) => slot.occupied).length;
  return {
    vehicles: visibleVehicles,
    occupied,
    empty: snapshot.parking_slots.length - occupied,
    capacity: snapshot.parking_slots.length,
  };
}

