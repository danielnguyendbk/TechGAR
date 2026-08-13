/**
 * Parking Store — manages parking spot states.
 *
 * Review fixes:
 *   #14 — Removed hardcoded cam-left/cam-right
 *   #22 — Camera coverage managed by backend/vision, not frontend
 *   #17 — Data comes from backend API, not JSON files
 */

import { create } from "zustand";
import type {
  CameraState,
  ParkingCounts,
  ParkingSnapshot,
  ParkingSpotState,
  SpotId,
} from "../domain/parking";
import { PARKING_GEOMETRY } from "../geometry/parkingGeometry";

export type EventApplyResult = "applied" | "stale" | "unknown-spot";

export interface ParkingStoreState {
  spots: Partial<Record<SpotId, ParkingSpotState>>;
  cameras: Record<string, CameraState>;
  lastEventTime?: string;
  staleEvents: number;
  applySnapshot: (snapshot: ParkingSnapshot) => void;
  applySpotUpdate: (spotId: SpotId, status: string, vehicleId?: number | null) => void;
  applyBulkSpots: (spotsData: Record<string, any>) => void;
  reset: () => void;
}

const initialSpots = Object.fromEntries(
  PARKING_GEOMETRY.spots.map((geom) => [
    geom.id,
    {
      id: geom.id,
      zone: geom.zone,
      number: geom.number,
      row: geom.row,
      status: "empty",
      confidence: 1,
      revision: 0,
      updatedAt: new Date().toISOString(),
    } as ParkingSpotState
  ])
) as Record<SpotId, ParkingSpotState>;

const initialState = {
  spots: initialSpots,
  cameras: {} as Record<string, CameraState>,
  lastEventTime: undefined,
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
    const spots = Object.fromEntries(
      snapshot.spots.map((spot) => [spot.id, spot]),
    ) as Record<SpotId, ParkingSpotState>;
    set({
      spots,
      cameras: snapshot.cameras,
      lastEventTime: snapshot.capturedAt,
      staleEvents: 0,
    });
  },

  applySpotUpdate: (spotId, status, vehicleId) => {
    set((state) => {
      const current = state.spots[spotId];
      const now = new Date().toISOString();
      return {
        spots: {
          ...state.spots,
          [spotId]: {
            ...(current ?? {
              id: spotId,
              zone: spotId.slice(0, 1),
              number: Number(spotId.slice(1)),
              row: "top" as const,
              confidence: 0.99,
              revision: 0,
            }),
            status: status as any,
            vehicleId: vehicleId ?? null,
            updatedAt: now,
            revision: Date.now(),
          },
        },
        lastEventTime: now,
      };
    });
  },

  applyBulkSpots: (spotsData) => {
    set((state) => {
      const newSpots = { ...state.spots };
      const now = new Date().toISOString();

      for (const [spotId, spotData] of Object.entries(spotsData)) {
        const status = spotData.occupied ? "occupied" : "empty";
        const current = newSpots[spotId as SpotId];
        newSpots[spotId as SpotId] = {
          ...(current ?? {
            id: spotId as SpotId,
            zone: spotId.slice(0, 1) as any,
            number: Number(spotId.slice(1)),
            row: "top" as const,
            confidence: 0.99,
            revision: 0,
          }),
          status,
          vehicleId: spotData.vehicleId ?? null,
          updatedAt: now,
          revision: Date.now(),
        } as ParkingSpotState;
      }

      return { spots: newSpots, lastEventTime: now };
    });
  },

  reset: () => set({ ...initialState, cameras: {} }),
}));

export function getParkingSpots(): ParkingSpotState[] {
  return Object.values(useParkingStore.getState().spots).filter(
    (spot): spot is ParkingSpotState => spot !== undefined,
  );
}
