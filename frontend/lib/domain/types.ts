export type WorldPoint = readonly [number, number];

export type ConnectionState = 'connecting' | 'live' | 'stale' | 'error';
export type TrackingSource = 'live' | 'demo' | 'replay';
export type VehicleDisplayState =
  | 'observed'
  | 'temporarily_missing'
  | 'parked'
  | 'hidden';

export interface RuntimeVehicle {
  readonly global_id: number;
  readonly state: VehicleDisplayState;
  readonly observed: boolean;
  readonly parked_slot_id: string | null;
  readonly stale_seconds: number;
  readonly display_hold_seconds: number;
  readonly position: WorldPoint;
  readonly velocity: WorldPoint;
  readonly camera_id: string;
  readonly session_ids: readonly string[];
  readonly uncertainty: number;
  readonly footprint: readonly WorldPoint[];
}

export interface RuntimeParkingSlot {
  readonly slot_id: string;
  readonly occupied: boolean;
  readonly status: 'empty' | 'claim_pending' | 'occupied' | 'releasing';
  readonly owning_global_id: number | null;
  readonly overlap_score: number;
  readonly dwell_duration: number;
  readonly confirmation_confidence: number;
}

export interface SlotLayout {
  readonly slot_id: string;
  readonly camera_id: string;
  readonly polygon: readonly WorldPoint[];
}

export interface CameraHealth {
  readonly online: boolean;
  readonly frames?: number;
  readonly dropped_stale?: number;
  readonly replaced?: number;
  readonly last_timestamp?: number | null;
}

export interface IdentityEvent {
  readonly event_id: number;
  readonly timestamp: number;
  readonly frame_sequence: number;
  readonly type: string;
  readonly global_id: number | null;
  readonly detail: string;
}

export interface RuntimeSnapshot {
  readonly schema_version: '1.0';
  readonly frame_index: number;
  readonly timestamp: number;
  readonly published_at: number;
  readonly vehicles: readonly RuntimeVehicle[];
  readonly parking_slots: readonly RuntimeParkingSlot[];
  readonly slot_layout: readonly SlotLayout[];
  readonly cameras: Readonly<Record<string, CameraHealth>>;
  readonly identity_events: readonly IdentityEvent[];
  readonly latency: Readonly<Record<string, number>>;
  readonly overload: boolean;
  readonly gps_used: false;
}

export type SessionState =
  | 'WAITING'
  | 'NAVIGATING'
  | 'PARKED'
  | 'EXIT_NAVIGATION';

export interface DriverSession {
  readonly sessionId: string;
  readonly state: SessionState;
  readonly globalVehicleId: number | null;
  readonly targetSpotId: string | null;
  readonly parkedSpotId: string | null;
  readonly claimedAt: number | null;
  readonly updatedAt: number;
}

export interface DisplayVehicle {
  readonly globalId: number;
  readonly position: WorldPoint;
  readonly state: VehicleDisplayState;
  readonly parkedSlotId: string | null;
  readonly visibleReason: 'observed' | 'parked' | 'hold' | 'slot-fallback';
  readonly uncertainty: number;
}

export interface ConnectionStatus {
  readonly state: ConnectionState;
  readonly lastPublishedAt: number | null;
  readonly fetchError: string | null;
  readonly failures: number;
}

