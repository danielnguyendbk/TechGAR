export const MAIN_ZONE_ORDER = ["E", "D", "C", "B", "A"] as const;
export const ALL_ZONE_IDS = ["A", "B", "C", "D", "E", "F"] as const;
export const CAMERA_IDS = ["cam-left", "cam-right"] as const;
export const PARKING_STATUSES = ["empty", "occupied", "transitioning", "unknown"] as const;

export type MainZoneId = (typeof MAIN_ZONE_ORDER)[number];
export type ZoneId = (typeof ALL_ZONE_IDS)[number];
export type CameraId = (typeof CAMERA_IDS)[number];
export type ParkingStatus = (typeof PARKING_STATUSES)[number];
export type CameraHealth = "online" | "offline";
export type ParkingRow = "top" | "bottom" | "vertical";
export type DriverMode = "entry" | "browse" | "recommendation" | "navigation";
export type DestinationNeed = "shopping" | "services" | "entertainment";
export type BrowseFilter = "all" | "empty";
export type SpotId = `${ZoneId}${number}`;

export interface ParkingSpotState {
  id: SpotId;
  zone: ZoneId;
  number: number;
  row: ParkingRow;
  owner: CameraId;
  status: ParkingStatus;
  confidence: number;
  revision: number;
  updatedAt: string;
}

export interface CameraState {
  cameraId: CameraId;
  health: CameraHealth;
  updatedAt: string;
}

export interface SpotStatusEvent {
  type: "spot.status.changed";
  cameraId: CameraId;
  spotId: SpotId;
  status: ParkingStatus;
  confidence: number;
  revision: number;
  updatedAt: string;
}

export interface CameraHealthEvent {
  type: "camera.health.changed";
  cameraId: CameraId;
  health: CameraHealth;
  updatedAt: string;
}

export type ParkingEvent = SpotStatusEvent | CameraHealthEvent;

export interface ParkingSnapshot {
  spots: ParkingSpotState[];
  cameras: Record<CameraId, CameraState>;
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

export function getSpotOwner(spotId: SpotId): CameraId {
  const { zone, number } = parseSpotId(spotId);
  if (zone === "F") return "cam-right";
  return (number >= 1 && number <= 8) || (number >= 16 && number <= 23)
    ? "cam-left"
    : "cam-right";
}

export function cameraOwnsSpot(cameraId: CameraId, spotId: SpotId): boolean {
  return getSpotOwner(spotId) === cameraId;
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
