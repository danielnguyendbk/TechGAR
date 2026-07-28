import {
  CAMERA_IDS,
  cameraOwnsSpot,
  getSpotOwner,
  type CameraHealth,
  type CameraHealthEvent,
  type CameraId,
  type ParkingDataSource,
  type ParkingEvent,
  type ParkingSnapshot,
  type ParkingSpotState,
  type ParkingStatus,
  type SpotId,
} from "../domain/parking";
import { PARKING_GEOMETRY } from "../geometry/parkingGeometry";

const BASE_TIME = Date.parse("2026-07-25T08:00:00.000Z");

export const MOCK_SCENARIOS = [
  { id: "normal-independent", label: "Cập nhật trái / phải độc lập" },
  { id: "recommendation-transitioning", label: "Ô đề xuất chuyển vàng" },
  { id: "selected-transitioning", label: "Ô đã chọn chuyển vàng" },
  { id: "recommendation-occupied", label: "Ô đề xuất chuyển đỏ" },
  { id: "amber-to-empty", label: "Ô vàng trở lại trống" },
  { id: "cam-left-offline-recovery", label: "Camera trái mất / phục hồi" },
  { id: "cam-right-offline-recovery", label: "Camera phải mất / phục hồi" },
  { id: "stale-revision", label: "Sự kiện revision cũ" },
  { id: "unauthorized-ownership", label: "Sự kiện sai quyền camera" },
] as const;

export type MockScenarioId = (typeof MOCK_SCENARIOS)[number]["id"];

export interface MockScenarioContext {
  recommendedSpotId?: SpotId;
  selectedSpotId?: SpotId;
}

type PendingAction =
  | { kind: "status"; spotId: SpotId; status: ParkingStatus; cameraId?: CameraId }
  | { kind: "health"; cameraId: CameraId; health: CameraHealth }
  | { kind: "stale"; spotId: SpotId }
  | { kind: "unauthorized"; spotId: SpotId; cameraId: CameraId };

function initialStatus(number: number): ParkingStatus {
  if (number % 7 === 0) return "occupied";
  if (number % 11 === 0) return "transitioning";
  if (number % 13 === 0) return "unknown";
  return "empty";
}

export function createInitialSnapshot(): ParkingSnapshot {
  const capturedAt = new Date(BASE_TIME).toISOString();
  const spots: ParkingSpotState[] = PARKING_GEOMETRY.spots.map((geometry) => ({
    id: geometry.id,
    zone: geometry.zone,
    number: geometry.number,
    row: geometry.row,
    owner: getSpotOwner(geometry.id),
    status: initialStatus(geometry.number),
    confidence: 0.96,
    revision: 1,
    updatedAt: capturedAt,
  }));

  return {
    spots,
    cameras: {
      "cam-left": { cameraId: "cam-left", health: "online", updatedAt: capturedAt },
      "cam-right": { cameraId: "cam-right", health: "online", updatedAt: capturedAt },
    },
    capturedAt,
  };
}

export class MockParkingDataSource implements ParkingDataSource {
  private readonly listeners = new Set<(event: ParkingEvent) => void>();
  private readonly revisions = new Map<SpotId, number>();
  private readonly pendingActions: PendingAction[] = [];
  private leftTimer?: ReturnType<typeof globalThis.setInterval>;
  private rightTimer?: ReturnType<typeof globalThis.setInterval>;
  private clockTick = 0;
  private leftCycleIndex = 0;
  private rightCycleIndex = 0;
  private rejectedCount = 0;
  private lastDiagnostic = "Chưa có sự kiện kiểm tra";

  public constructor() {
    createInitialSnapshot().spots.forEach((spot) => this.revisions.set(spot.id, spot.revision));
  }

  public async getSnapshot(): Promise<ParkingSnapshot> {
    return createInitialSnapshot();
  }

  public subscribe(listener: (event: ParkingEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public start(): void {
    if (this.leftTimer || this.rightTimer) return;
    const leftCycle: Array<[SpotId, ParkingStatus]> = [
      ["A01", "transitioning"],
      ["A01", "occupied"],
      ["A02", "transitioning"],
      ["A02", "empty"],
    ];
    const rightCycle: Array<[SpotId, ParkingStatus]> = [
      ["A28", "transitioning"],
      ["A28", "empty"],
      ["F01", "occupied"],
      ["F01", "empty"],
    ];

    this.leftTimer = globalThis.setInterval(() => {
      const action = leftCycle[this.leftCycleIndex % leftCycle.length];
      this.leftCycleIndex += 1;
      if (action) this.emitSpotStatus(action[0], action[1]);
    }, 60_000);
    this.rightTimer = globalThis.setInterval(() => {
      const action = rightCycle[this.rightCycleIndex % rightCycle.length];
      this.rightCycleIndex += 1;
      if (action) this.emitSpotStatus(action[0], action[1]);
    }, 75_000);
  }

  public stop(): void {
    if (this.leftTimer) globalThis.clearInterval(this.leftTimer);
    if (this.rightTimer) globalThis.clearInterval(this.rightTimer);
    this.leftTimer = undefined;
    this.rightTimer = undefined;
  }

  public emitSpotStatus(
    spotId: SpotId,
    status: ParkingStatus,
    cameraId: CameraId = getSpotOwner(spotId),
    revision?: number,
  ): boolean {
    const currentRevision = this.revisions.get(spotId) ?? 0;
    const eventRevision = revision ?? currentRevision + 1;
    const event = {
      type: "spot.status.changed",
      cameraId,
      spotId,
      status,
      confidence: status === "unknown" ? 0.42 : 0.95,
      revision: eventRevision,
      updatedAt: this.nextTime(),
    } satisfies ParkingEvent;

    if (!cameraOwnsSpot(cameraId, spotId)) {
      this.reject(`Từ chối ${cameraId} cập nhật ${spotId}: camera không sở hữu ô này.`);
      return false;
    }
    if (eventRevision > currentRevision) this.revisions.set(spotId, eventRevision);
    this.dispatch(event);
    this.lastDiagnostic = `${cameraId} cập nhật ${spotId} thành ${status}, revision ${eventRevision}`;
    return true;
  }

  public emitCameraHealth(cameraId: CameraId, health: CameraHealth): void {
    const event: CameraHealthEvent = {
      type: "camera.health.changed",
      cameraId,
      health,
      updatedAt: this.nextTime(),
    };
    this.dispatch(event);
    this.lastDiagnostic = `${cameraId} chuyển sang ${health}`;
  }

  public queueScenario(id: MockScenarioId, context: MockScenarioContext = {}): number {
    const recommended = context.recommendedSpotId ?? "E12";
    const selected = context.selectedSpotId ?? "A01";
    const actions: Record<MockScenarioId, PendingAction[]> = {
      "normal-independent": [
        { kind: "status", spotId: "A01", status: "transitioning" },
        { kind: "status", spotId: "A28", status: "transitioning" },
        { kind: "status", spotId: "A01", status: "occupied" },
        { kind: "status", spotId: "A28", status: "empty" },
      ],
      "recommendation-transitioning": [{ kind: "status", spotId: recommended, status: "transitioning" }],
      "selected-transitioning": [{ kind: "status", spotId: selected, status: "transitioning" }],
      "recommendation-occupied": [{ kind: "status", spotId: recommended, status: "occupied" }],
      "amber-to-empty": [{ kind: "status", spotId: "A11", status: "empty" }],
      "cam-left-offline-recovery": [
        { kind: "health", cameraId: "cam-left", health: "offline" },
        { kind: "health", cameraId: "cam-left", health: "online" },
      ],
      "cam-right-offline-recovery": [
        { kind: "health", cameraId: "cam-right", health: "offline" },
        { kind: "health", cameraId: "cam-right", health: "online" },
      ],
      "stale-revision": [{ kind: "stale", spotId: "A02" }],
      "unauthorized-ownership": [{ kind: "unauthorized", spotId: "F01", cameraId: "cam-left" }],
    };
    this.pendingActions.push(...actions[id]);
    this.lastDiagnostic = `Đã xếp kịch bản “${MOCK_SCENARIOS.find((scenario) => scenario.id === id)?.label ?? id}”`;
    return this.pendingActions.length;
  }

  public stepScenario(): boolean {
    const action = this.pendingActions.shift();
    if (!action) {
      this.lastDiagnostic = "Không còn bước kịch bản đang chờ";
      return false;
    }
    if (action.kind === "status") {
      return this.emitSpotStatus(action.spotId, action.status, action.cameraId);
    }
    if (action.kind === "health") {
      this.emitCameraHealth(action.cameraId, action.health);
      return true;
    }
    if (action.kind === "stale") {
      const revision = this.revisions.get(action.spotId) ?? 1;
      return this.emitSpotStatus(action.spotId, "occupied", getSpotOwner(action.spotId), revision);
    }
    return this.emitSpotStatus(
      action.spotId,
      "occupied",
      action.cameraId,
      (this.revisions.get(action.spotId) ?? 1) + 1,
    );
  }

  public getPendingCount(): number {
    return this.pendingActions.length;
  }

  public getRejectedCount(): number {
    return this.rejectedCount;
  }

  public getLastDiagnostic(): string {
    return this.lastDiagnostic;
  }

  private dispatch(event: ParkingEvent): void {
    this.listeners.forEach((listener) => listener(event));
  }

  private nextTime(): string {
    this.clockTick += 1;
    return new Date(BASE_TIME + this.clockTick * 1_000).toISOString();
  }

  private reject(message: string): void {
    this.rejectedCount += 1;
    this.lastDiagnostic = message;
    if (import.meta.env.DEV) console.warn(`[MockParkingDataSource] ${message}`);
  }
}

export const mockParkingDataSource = new MockParkingDataSource();

export function validateOwnershipIsDisjoint(): boolean {
  return PARKING_GEOMETRY.spots.every((spot) => {
    const owners = CAMERA_IDS.filter((cameraId) => cameraOwnsSpot(cameraId, spot.id));
    return owners.length === 1;
  });
}
