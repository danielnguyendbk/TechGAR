import type {
  DriverSession,
  RuntimeParkingSlot,
  RuntimeSnapshot,
  SlotLayout,
  WorldPoint,
} from '@/lib/domain/types';

function slotPolygon(index: number): readonly WorldPoint[] {
  const row = index < 12 ? 0 : 1;
  const column = index % 12;
  const x = 12 + column * 7.2;
  const y = row === 0 ? 10 : 48;
  return [[x, y], [x + 5.4, y], [x + 5.4, y + 10], [x, y + 10]];
}

// Keep the server and first client render byte-identical. Live/fixture polling
// replaces this snapshot immediately; a wall-clock default causes React
// hydration mismatches in the monitor event timestamps.
export function makeDemoSnapshot(now = 1_700_000_000): RuntimeSnapshot {
  const occupied = new Map<string, number>([
    ['D03', 5], ['D06', 9], ['D09', 17], ['D10', 22], ['D15', 31], ['D19', 38],
  ]);
  const slot_layout: SlotLayout[] = Array.from({ length: 24 }, (_, index) => ({
    slot_id: `D${String(index + 1).padStart(2, '0')}`,
    camera_id: index < 12 ? 'C1' : 'C2',
    polygon: slotPolygon(index),
  }));
  const parking_slots: RuntimeParkingSlot[] = slot_layout.map((slot) => ({
    slot_id: slot.slot_id,
    occupied: occupied.has(slot.slot_id),
    status: occupied.has(slot.slot_id) ? 'occupied' : 'empty',
    owning_global_id: occupied.get(slot.slot_id) ?? null,
    overlap_score: occupied.has(slot.slot_id) ? 0.91 : 0,
    dwell_duration: occupied.has(slot.slot_id) ? 24.5 : 0,
    confirmation_confidence: occupied.has(slot.slot_id) ? 0.96 : 0,
  }));
  return {
    schema_version: '1.0',
    frame_index: 1264,
    timestamp: now - 0.3,
    published_at: now - 0.2,
    vehicles: [
      {
        global_id: 5, state: 'observed', observed: true, parked_slot_id: null,
        stale_seconds: 0, display_hold_seconds: 2.5, position: [31, 34],
        velocity: [2.1, 0], camera_id: 'C1', session_ids: [], uncertainty: 0.17,
        footprint: [],
      },
      {
        global_id: 9, state: 'parked', observed: false, parked_slot_id: 'D06',
        stale_seconds: 93, display_hold_seconds: 2.5, position: [51.8, 15],
        velocity: [0, 0], camera_id: 'C1', session_ids: [], uncertainty: 0.12,
        footprint: [],
      },
      {
        global_id: 17, state: 'observed', observed: true, parked_slot_id: null,
        stale_seconds: 0, display_hold_seconds: 2.5, position: [56, 34],
        velocity: [2.8, 0], camera_id: 'C2', session_ids: ['S42'], uncertainty: 0.19,
        footprint: [],
      },
      {
        global_id: 22, state: 'temporarily_missing', observed: false, parked_slot_id: null,
        stale_seconds: 1.1, display_hold_seconds: 2.5, position: [70, 34],
        velocity: [1.2, 0], camera_id: 'C2', session_ids: [], uncertainty: 0.55,
        footprint: [],
      },
    ],
    parking_slots,
    slot_layout,
    cameras: {
      C1: { online: true, frames: 15342, dropped_stale: 12, replaced: 34, last_timestamp: now - 0.3 },
      C2: { online: true, frames: 15318, dropped_stale: 9, replaced: 28, last_timestamp: now - 0.3 },
    },
    identity_events: [
      { event_id: 91, timestamp: now - 2, frame_sequence: 1262, type: 'handoff', global_id: 17, detail: 'C1→C2 · score 0.96' },
      { event_id: 90, timestamp: now - 8, frame_sequence: 1254, type: 'park', global_id: 9, detail: 'D06' },
      { event_id: 89, timestamp: now - 14, frame_sequence: 1248, type: 'recover', global_id: 5, detail: 'temporarily_missing→active' },
    ],
    latency: { e2e: 0.184, processing: 0.071 },
    overload: false,
    gps_used: false,
  };
}

export const demoSession: DriverSession = {
  sessionId: 'S42',
  state: 'WAITING',
  globalVehicleId: 17,
  targetSpotId: null,
  parkedSpotId: null,
  claimedAt: Date.now() / 1_000,
  updatedAt: Date.now() / 1_000,
};
