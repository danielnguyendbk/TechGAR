import { beforeEach, describe, expect, it, vi } from "vitest";
import { cameraOwnsSpot, type SpotStatusEvent } from "../domain/parking";
import {
  MockParkingDataSource,
  createInitialSnapshot,
  validateOwnershipIsDisjoint,
} from "../mocks/MockParkingDataSource";
import { deriveParkingCounts, getParkingSpots, useParkingStore } from "../stores/parkingStore";

describe("camera ownership", () => {
  it("is disjoint and matches all specified boundaries", () => {
    expect(validateOwnershipIsDisjoint()).toBe(true);
    expect(cameraOwnsSpot("cam-left", "A01")).toBe(true);
    expect(cameraOwnsSpot("cam-left", "A08")).toBe(true);
    expect(cameraOwnsSpot("cam-right", "A09")).toBe(true);
    expect(cameraOwnsSpot("cam-left", "A16")).toBe(true);
    expect(cameraOwnsSpot("cam-left", "A23")).toBe(true);
    expect(cameraOwnsSpot("cam-right", "A24")).toBe(true);
    expect(cameraOwnsSpot("cam-right", "A30")).toBe(true);
    expect(cameraOwnsSpot("cam-right", "F01")).toBe(true);
    expect(cameraOwnsSpot("cam-left", "F10")).toBe(false);
  });

  it("rejects an unauthorized source event before notifying subscribers", () => {
    const source = new MockParkingDataSource();
    const listener = vi.fn();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    source.subscribe(listener);
    expect(source.emitSpotStatus("F01", "occupied", "cam-left", 2)).toBe(false);
    expect(listener).not.toHaveBeenCalled();
    expect(source.getRejectedCount()).toBe(1);
    warn.mockRestore();
  });
});

describe("canonical parking store", () => {
  beforeEach(() => {
    useParkingStore.getState().reset();
    useParkingStore.getState().applySnapshot(createInitialSnapshot());
  });

  it("ignores stale revisions", () => {
    const before = useParkingStore.getState().spots.A02;
    const event: SpotStatusEvent = {
      type: "spot.status.changed",
      cameraId: "cam-left",
      spotId: "A02",
      status: "occupied",
      confidence: 0.99,
      revision: 1,
      updatedAt: "2026-07-25T08:01:00.000Z",
    };
    expect(useParkingStore.getState().applyEvent(event)).toBe("stale");
    expect(useParkingStore.getState().spots.A02).toEqual(before);
    expect(useParkingStore.getState().staleEvents).toBe(1);
  });

  it("rejects wrong camera ownership in the canonical store", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    expect(
      useParkingStore.getState().applyEvent({
        type: "spot.status.changed",
        cameraId: "cam-right",
        spotId: "A01",
        status: "occupied",
        confidence: 0.9,
        revision: 2,
        updatedAt: "2026-07-25T08:01:01.000Z",
      }),
    ).toBe("unauthorized");
    expect(useParkingStore.getState().spots.A01?.status).toBe("empty");
    warn.mockRestore();
  });

  it("derives counts from canonical state after every accepted event", () => {
    const before = deriveParkingCounts(getParkingSpots());
    expect(before.total).toBe(160);
    useParkingStore.getState().applyEvent({
      type: "spot.status.changed",
      cameraId: "cam-left",
      spotId: "A01",
      status: "occupied",
      confidence: 0.97,
      revision: 2,
      updatedAt: "2026-07-25T08:01:02.000Z",
    });
    const after = deriveParkingCounts(getParkingSpots());
    expect(after.total).toBe(160);
    expect(after.empty).toBe(before.empty - 1);
    expect(after.occupied).toBe(before.occupied + 1);
    expect(after.total).toBe(after.empty + after.occupied + after.transitioning + after.unknown);
  });

  it("preserves spot states while a camera is offline", () => {
    const before = useParkingStore.getState().spots.A01;
    useParkingStore.getState().applyEvent({
      type: "camera.health.changed",
      cameraId: "cam-left",
      health: "offline",
      updatedAt: "2026-07-25T08:02:00.000Z",
    });
    expect(useParkingStore.getState().cameras["cam-left"].health).toBe("offline");
    expect(useParkingStore.getState().spots.A01).toEqual(before);
  });
});
