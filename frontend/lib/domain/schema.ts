import type {
  CameraHealth,
  IdentityEvent,
  RuntimeParkingSlot,
  RuntimeSnapshot,
  RuntimeVehicle,
  SlotLayout,
  VehicleDisplayState,
  WorldPoint,
} from './types';

export class SnapshotValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotValidationError';
  }
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new SnapshotValidationError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new SnapshotValidationError(`${path} must be an array`);
  }
  return value;
}

function number(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new SnapshotValidationError(`${path} must be a finite number`);
  }
  return value;
}

function integer(value: unknown, path: string): number {
  const parsed = number(value, path);
  if (!Number.isInteger(parsed)) {
    throw new SnapshotValidationError(`${path} must be an integer`);
  }
  return parsed;
}

function string(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new SnapshotValidationError(`${path} must be a non-empty string`);
  }
  return value;
}

function boolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new SnapshotValidationError(`${path} must be a boolean`);
  }
  return value;
}

function nullableString(value: unknown, path: string): string | null {
  return value === null ? null : string(value, path);
}

function nullableInteger(value: unknown, path: string): number | null {
  return value === null ? null : integer(value, path);
}

function point(value: unknown, path: string): WorldPoint {
  const values = array(value, path);
  if (values.length !== 2) {
    throw new SnapshotValidationError(`${path} must contain exactly two coordinates`);
  }
  return [number(values[0], `${path}[0]`), number(values[1], `${path}[1]`)];
}

function displayState(value: unknown, path: string): VehicleDisplayState {
  const parsed = string(value, path);
  if (!['observed', 'temporarily_missing', 'parked', 'hidden'].includes(parsed)) {
    throw new SnapshotValidationError(`${path} has an unsupported display state`);
  }
  return parsed as VehicleDisplayState;
}

function parseVehicle(value: unknown, index: number): RuntimeVehicle {
  const path = `vehicles[${index}]`;
  const item = record(value, path);
  return {
    global_id: integer(item.global_id, `${path}.global_id`),
    state: displayState(item.state, `${path}.state`),
    observed: boolean(item.observed, `${path}.observed`),
    parked_slot_id: nullableString(item.parked_slot_id, `${path}.parked_slot_id`),
    stale_seconds: number(item.stale_seconds, `${path}.stale_seconds`),
    display_hold_seconds: number(item.display_hold_seconds, `${path}.display_hold_seconds`),
    position: point(item.position, `${path}.position`),
    velocity: item.velocity === undefined ? [0, 0] : point(item.velocity, `${path}.velocity`),
    camera_id: typeof item.camera_id === 'string' ? item.camera_id : '',
    session_ids: item.session_ids === undefined
      ? []
      : array(item.session_ids, `${path}.session_ids`).map((entry, child) =>
          string(entry, `${path}.session_ids[${child}]`),
        ),
    uncertainty: item.uncertainty === undefined
      ? 0
      : number(item.uncertainty, `${path}.uncertainty`),
    footprint: item.footprint === undefined
      ? []
      : array(item.footprint, `${path}.footprint`).map((entry, child) =>
          point(entry, `${path}.footprint[${child}]`),
        ),
  };
}

function parseSlot(value: unknown, index: number): RuntimeParkingSlot {
  const path = `parking_slots[${index}]`;
  const item = record(value, path);
  const status = string(item.status, `${path}.status`);
  if (!['empty', 'claim_pending', 'occupied', 'releasing'].includes(status)) {
    throw new SnapshotValidationError(`${path}.status is unsupported`);
  }
  return {
    slot_id: string(item.slot_id, `${path}.slot_id`),
    occupied: boolean(item.occupied, `${path}.occupied`),
    status: status as RuntimeParkingSlot['status'],
    owning_global_id: item.owning_global_id === undefined
      ? null
      : nullableInteger(item.owning_global_id, `${path}.owning_global_id`),
    overlap_score: item.overlap_score === undefined ? 0 : number(item.overlap_score, `${path}.overlap_score`),
    dwell_duration: item.dwell_duration === undefined ? 0 : number(item.dwell_duration, `${path}.dwell_duration`),
    confirmation_confidence: item.confirmation_confidence === undefined
      ? 0
      : number(item.confirmation_confidence, `${path}.confirmation_confidence`),
  };
}

function parseLayout(value: unknown, index: number): SlotLayout {
  const path = `slot_layout[${index}]`;
  const item = record(value, path);
  const polygon = array(item.polygon, `${path}.polygon`).map((entry, child) =>
    point(entry, `${path}.polygon[${child}]`),
  );
  if (polygon.length < 3) {
    throw new SnapshotValidationError(`${path}.polygon needs at least three points`);
  }
  return {
    slot_id: string(item.slot_id, `${path}.slot_id`),
    camera_id: typeof item.camera_id === 'string' ? item.camera_id : 'shared',
    polygon,
  };
}

function parseCameras(value: unknown): Record<string, CameraHealth> {
  const cameras = record(value, 'cameras');
  return Object.fromEntries(Object.entries(cameras).map(([cameraId, raw]) => {
    const camera = record(raw, `cameras.${cameraId}`);
    return [cameraId, {
      online: boolean(camera.online, `cameras.${cameraId}.online`),
      frames: camera.frames === undefined ? undefined : integer(camera.frames, `cameras.${cameraId}.frames`),
      dropped_stale: camera.dropped_stale === undefined ? undefined : integer(camera.dropped_stale, `cameras.${cameraId}.dropped_stale`),
      replaced: camera.replaced === undefined ? undefined : integer(camera.replaced, `cameras.${cameraId}.replaced`),
      last_timestamp: camera.last_timestamp === undefined || camera.last_timestamp === null
        ? null
        : number(camera.last_timestamp, `cameras.${cameraId}.last_timestamp`),
    } satisfies CameraHealth];
  }));
}

function parseEvents(value: unknown): IdentityEvent[] {
  return array(value, 'identity_events').map((raw, index) => {
    const path = `identity_events[${index}]`;
    const event = record(raw, path);
    return {
      event_id: integer(event.event_id, `${path}.event_id`),
      timestamp: number(event.timestamp, `${path}.timestamp`),
      frame_sequence: integer(event.frame_sequence, `${path}.frame_sequence`),
      type: string(event.type, `${path}.type`),
      global_id: nullableInteger(event.global_id, `${path}.global_id`),
      detail: typeof event.detail === 'string' ? event.detail : '',
    };
  });
}

export function validateRuntimeSnapshot(
  input: unknown,
  previousFrameIndex: number | null = null,
): RuntimeSnapshot {
  const source = record(input, 'snapshot');
  if (source.schema_version !== '1.0') {
    throw new SnapshotValidationError('schema_version must equal 1.0');
  }
  const frameIndex = integer(source.frame_index, 'frame_index');
  if (previousFrameIndex !== null && frameIndex < previousFrameIndex) {
    throw new SnapshotValidationError(
      `frame_index regressed from ${previousFrameIndex} to ${frameIndex}`,
    );
  }
  const vehicles = array(source.vehicles, 'vehicles').map(parseVehicle);
  const ids = new Set(vehicles.map((vehicle) => vehicle.global_id));
  if (ids.size !== vehicles.length) {
    throw new SnapshotValidationError('vehicles.global_id values must be unique');
  }
  const slots = array(source.parking_slots, 'parking_slots').map(parseSlot);
  const slotIds = new Set(slots.map((slot) => slot.slot_id));
  if (slotIds.size !== slots.length) {
    throw new SnapshotValidationError('parking_slots.slot_id values must be unique');
  }
  const latencyRecord = source.latency === undefined ? {} : record(source.latency, 'latency');
  const latency = Object.fromEntries(Object.entries(latencyRecord).map(([key, value]) => [
    key,
    number(value, `latency.${key}`),
  ]));
  return {
    schema_version: '1.0',
    frame_index: frameIndex,
    timestamp: number(source.timestamp, 'timestamp'),
    published_at: number(source.published_at, 'published_at'),
    vehicles,
    parking_slots: slots,
    slot_layout: array(source.slot_layout, 'slot_layout').map(parseLayout),
    cameras: parseCameras(source.cameras),
    identity_events: parseEvents(source.identity_events),
    latency,
    overload: boolean(source.overload, 'overload'),
    gps_used: source.gps_used === false
      ? false
      : (() => { throw new SnapshotValidationError('gps_used must be false'); })(),
  };
}

