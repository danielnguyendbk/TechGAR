import { describe, expect, it } from "vitest";
import { createInitialSnapshot } from "../mocks/MockParkingDataSource";
import { getEligibleSpots, recommendParkingSpots } from "../recommendation/recommendationEngine";

describe("recommendation engine", () => {
  const spots = createInitialSnapshot().spots;

  it("makes only stable empty spots eligible", () => {
    const eligible = getEligibleSpots(spots);
    expect(eligible.length).toBeGreaterThan(0);
    expect(eligible.every((spot) => spot.status === "empty")).toBe(true);
    expect(eligible.some((spot) => spot.status === "transitioning")).toBe(false);
  });

  it("never recommends an amber spot", () => {
    const result = recommendParkingSpots(spots, "shopping", { calculatedAt: "2026-07-25T08:00:00.000Z" });
    expect(result).not.toBeNull();
    const ids = result ? [result.best.spotId, ...result.alternatives.map((spot) => spot.spotId)] : [];
    ids.forEach((spotId) => expect(spots.find((spot) => spot.id === spotId)?.status).toBe("empty"));
    expect(ids).not.toContain("A11");
  });

  it("returns deterministic best plus two alternatives for each supported need", () => {
    const needs = ["shopping", "services", "entertainment"] as const;
    const bestIds = needs.map((need) => {
      const first = recommendParkingSpots(spots, need, { calculatedAt: "2026-07-25T08:00:00.000Z" });
      const second = recommendParkingSpots(spots, need, { calculatedAt: "2026-07-25T08:00:00.000Z" });
      expect(first).toEqual(second);
      expect(first?.alternatives).toHaveLength(2);
      return first?.best.spotId;
    });
    expect(new Set(bestIds).size).toBeGreaterThan(1);
  });

  it("removes an invalid unconfirmed best and recalculates", () => {
    const original = recommendParkingSpots(spots, "entertainment", { calculatedAt: "2026-07-25T08:00:00.000Z" });
    expect(original).not.toBeNull();
    if (!original) return;
    const changed = spots.map((spot) =>
      spot.id === original.best.spotId ? { ...spot, status: "transitioning" as const, revision: spot.revision + 1 } : spot,
    );
    const next = recommendParkingSpots(changed, "entertainment", { calculatedAt: "2026-07-25T08:00:01.000Z" });
    expect(next?.best.spotId).not.toBe(original.best.spotId);
    expect([next?.best, ...(next?.alternatives ?? [])].some((spot) => spot?.spotId === original.best.spotId)).toBe(false);
  });
});
