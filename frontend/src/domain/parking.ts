/**
 * Domain types for TechGAR Parking System.
 *
 * Review fixes:
 *   #14 — Removed hardcoded cam-left/cam-right camera ownership
 *   #22 — Frontend does not own camera IDs; backend/vision manages coverage
 *   #12 — Spot IDs are consistent across all layers (A01..F10)
 *   #30 — Frontend derives mode from backend session state
 */

export const ALL_ZONE_IDS = ["A", "B", "C", "D", "E", "F"] as const;
export const MAIN_ZONE_ORDER = ["E", "D", "C", "B", "A"] as const;
export const PARKING_STATUSES = ["empty", "occupied", "transitioning", "unknown"] as const;

export type MainZoneId = (typeof MAIN_ZONE_ORDER)[number];
export type ZoneId = (typeof ALL_ZONE_IDS)[number];
export type ParkingStatus = (typeof PARKING_STATUSES)[number];
export type CameraHealth = "online" | "offline";
export type ParkingRow = "top" | "bottom" | "vertical";
export type DriverMode = "entry" | "browse" | "recommendation" | "navigation";
export type DestinationNeed = "shopping" | "services" | "entertainment";
export type BrowseFilter = "all" | "empty";
export type SpotId = `${ZoneId}${number}`;

/**
 * Session states from backend (the authority).
 * Frontend derives its UI mode from this.
 */
export type SessionState =
  | "WAITING_FOR_SCAN"
  | "SELECTING_SPOT"
  | "NAVIGATING_TO_SPOT"
  | "PARKED"
  | "EXIT_NAVIGATION"
  | "CLOSED";

export interface ParkingSpotState {
  id: SpotId;
  zone: ZoneId;
  number: number;
  row: ParkingRow;
  status: ParkingStatus;
  confidence: number;
  revision: number;
  updatedAt: string;
  /** vehicleId bound to this spot (from vision Binder) */
  vehicleId?: number | null;
}

export interface CameraState {
  cameraId: string;
  health: CameraHealth;
  updatedAt: string;
}

export interface SpotStatusEvent {
  type: "spot.status.changed";
  spotId: SpotId;
  status: ParkingStatus;
  confidence: number;
  revision: number;
  updatedAt: string;
}

export interface CameraHealthEvent {
  type: "camera.health.changed";
  cameraId: string;
  health: CameraHealth;
  updatedAt: string;
}

export type ParkingEvent = SpotStatusEvent | CameraHealthEvent;

export interface ParkingSnapshot {
  spots: ParkingSpotState[];
  cameras: Record<string, CameraState>;
  capturedAt: string;
}

export interface ParkingDataSource {
  getSnapshot(): Promise<ParkingSnapshot>;
  subscribe(listener: (event: ParkingEvent) => void): () => void;
  start(): void;
  stop(): void;
}

export interface ParkingCounts {
  total: number;
  empty: number;
  occupied: number;
  transitioning: number;
  unknown: number;
}

export interface RankedSpot {
  spotId: SpotId;
  zone: ZoneId;
  totalScore: number;
  drivingDistance: number;
  walkingDistance: number;
  estimatedWalkingMinutes: number;
  reason: string;
}

export interface RecommendationResult {
  need: DestinationNeed;
  best: RankedSpot;
  alternatives: RankedSpot[];
  calculatedAt: string;
}

export interface InvalidSpotWarning {
  spotId: SpotId;
  status: Exclude<ParkingStatus, "empty">;
  alternativeSpotId?: SpotId;
}

/** Backend session data */
export interface BackendSession {
  sessionId: string;
  globalVehicleId: number | null;
  state: SessionState;
  targetSpotId: string | null;
  parkedSpotId: string | null;
  entryGateId: string | null;
  createdAt: string;
  claimedAt: string | null;
  parkedAt: string | null;
  exitStartedAt: string | null;
  closedAt: string | null;
  vehicle?: {
    globalVehicleId: number;
    trackingState: string;
    position: { x: number; y: number } | null;
    parkedSpotId: string | null;
  };
}

export const DESTINATION_LABELS: Record<DestinationNeed, string> = {
  shopping: "Shopping",
  services: "Dịch vụ",
  entertainment: "Giải trí",
};

export const STATUS_LABELS: Record<ParkingStatus, string> = {
  empty: "Trống",
  occupied: "Đã có xe",
  transitioning: "Đang chuyển tiếp",
  unknown: "Không xác định",
};

export function formatSpotId(zone: ZoneId, number: number): SpotId {
  return `${zone}${String(number).padStart(2, "0")}` as SpotId;
}

export function parseSpotId(spotId: SpotId): { zone: ZoneId; number: number } {
  return {
    zone: spotId.slice(0, 1) as ZoneId,
    number: Number(spotId.slice(1)),
  };
}

export function isSelectableStatus(status: ParkingStatus): boolean {
  return status === "empty";
}

export function getInvalidSpotWarningText(spotId: SpotId, status: Exclude<ParkingStatus, "empty">): string {
  if (status === "transitioning") {
    return `Ô ${spotId} đang có phương tiện di chuyển vào hoặc ra.`;
  }
  if (status === "occupied") {
    return `Ô ${spotId} hiện không còn trống.`;
  }
  return `Trạng thái ô ${spotId} hiện chưa xác định.`;
}

/**
 * Derive frontend DriverMode from backend SessionState.
 * Review fix #22: Backend session is the authority; frontend mode is derived.
 */
export function deriveDriverMode(sessionState: SessionState | null): DriverMode {
  switch (sessionState) {
    case "WAITING_FOR_SCAN":
      return "entry";
    case "SELECTING_SPOT":
      return "browse";
    case "NAVIGATING_TO_SPOT":
      return "navigation";
    case "PARKED":
      return "browse"; // show parked UI overlay
    case "EXIT_NAVIGATION":
      return "navigation"; // exit navigation
    default:
      return "entry";
  }
}
