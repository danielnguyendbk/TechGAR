import { create } from "zustand";
import {
  CAMERA_IDS,
  cameraOwnsSpot,
  type CameraId,
  type CameraState,
  type ParkingCounts,
  type ParkingEvent,
  type ParkingSnapshot,
  type ParkingSpotState,
  type SpotId,
} from "../domain/parking";

export type EventApplyResult = "applied" | "stale" | "unauthorized" | "unknown-spot";

export interface ParkingStoreState {
  spots: Partial<Record<SpotId, ParkingSpotState>>;
  cameras: Record<CameraId, CameraState>;
  lastEventTime?: string;
  rejectedEvents: number;
  staleEvents: number;
  applySnapshot: (snapshot: ParkingSnapshot) => void;
  applyEvent: (event: ParkingEvent) => EventApplyResult;
  reset: () => void;
}

const initialCameras: Record<CameraId, CameraState> = {
  "cam-left": { cameraId: "cam-left", health: "offline", updatedAt: "" },
  "cam-right": { cameraId: "cam-right", health: "offline", updatedAt: "" },
};

const initialState = {
  spots: {} as Partial<Record<SpotId, ParkingSpotState>>,
  cameras: initialCameras,
  lastEventTime: undefined,
  rejectedEvents: 0,
  staleEvents: 0,
};

export function deriveParkingCounts(spots: readonly ParkingSpotState[]): ParkingCounts {
  return spots.reduce<ParkingCounts>(
    (counts, spot) => ({
      ...counts,
      total: counts.total + 1,
      [spot.status]: counts[spot.status] + 1,
    }),
    { total: 0, empty: 0, occupied: 0, transitioning: 0, unknown: 0 },
  );
}

export const useParkingStore = create<ParkingStoreState>((set) => ({
  ...initialState,
  applySnapshot: (snapshot) => {
    const spots = Object.fromEntries(snapshot.spots.map((spot) => [spot.id, spot])) as Record<SpotId, ParkingSpotState>;
    set({
      spots,
      cameras: snapshot.cameras,
      lastEventTime: snapshot.capturedAt,
      rejectedEvents: 0,
      staleEvents: 0,
    });
  },
  applyEvent: (event) => {
    let result: EventApplyResult = "applied";
    set((state) => {
      if (event.type === "camera.health.changed") {
        return {
          cameras: {
            ...state.cameras,
            [event.cameraId]: {
              cameraId: event.cameraId,
              health: event.health,
              updatedAt: event.updatedAt,
            },
          },
          lastEventTime: event.updatedAt,
        };
      }

      const current = state.spots[event.spotId];
      if (!current) {
        result = "unknown-spot";
        return { rejectedEvents: state.rejectedEvents + 1 };
      }
      if (!cameraOwnsSpot(event.cameraId, event.spotId) || current.owner !== event.cameraId) {
        result = "unauthorized";
        if (import.meta.env.DEV) {
          console.warn(`[parkingStore] Từ chối ${event.cameraId} cập nhật ${event.spotId}.`);
        }
        return { rejectedEvents: state.rejectedEvents + 1 };
      }
      if (event.revision <= current.revision) {
        result = "stale";
        return { staleEvents: state.staleEvents + 1 };
      }
      return {
        spots: {
          ...state.spots,
          [event.spotId]: {
            ...current,
            status: event.status,
            confidence: event.confidence,
            revision: event.revision,
            updatedAt: event.updatedAt,
          },
        },
        lastEventTime: event.updatedAt,
      };
    });
    return result;
  },
  reset: () => set({ ...initialState, cameras: { ...initialCameras } }),
}));

export function getParkingSpots(): ParkingSpotState[] {
  return Object.values(useParkingStore.getState().spots).filter(
    (spot): spot is ParkingSpotState => spot !== undefined,
  );
}

export function areAllCamerasOnline(cameras: Record<CameraId, CameraState>): boolean {
  return CAMERA_IDS.every((cameraId) => cameras[cameraId].health === "online");
}
