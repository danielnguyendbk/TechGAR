import { describe, expect, it } from "vitest";
import { MAIN_ZONE_ORDER } from "../domain/parking";
import { PARKING_GEOMETRY } from "../geometry/parkingGeometry";

describe("parking geometry", () => {
  it("generates exactly 160 unique spot IDs", () => {
    const ids = PARKING_GEOMETRY.spots.map((spot) => spot.id);
    expect(ids).toHaveLength(160);
    expect(new Set(ids).size).toBe(160);
  });

  it("keeps E, D, C, B, A in visual order from top to bottom", () => {
    const ordered = [...PARKING_GEOMETRY.zones]
      .sort((a, b) => a.bounds.y - b.bounds.y)
      .map((zone) => zone.id);
    expect(ordered).toEqual(MAIN_ZONE_ORDER);
    expect(PARKING_GEOMETRY.zones.find((zone) => zone.id === "A")?.bounds.y).toBeGreaterThan(
      PARKING_GEOMETRY.zones.find((zone) => zone.id === "B")?.bounds.y ?? 0,
    );
  });

  it("creates two opposing rows of 15 spots for every main zone", () => {
    PARKING_GEOMETRY.zones.forEach((zone) => {
      const zoneSpots = PARKING_GEOMETRY.spots.filter((spot) => spot.zone === zone.id);
      expect(zoneSpots).toHaveLength(30);
      expect(zoneSpots.filter((spot) => spot.row === "top")).toHaveLength(15);
      expect(zoneSpots.filter((spot) => spot.row === "bottom")).toHaveLength(15);
      expect(zone.upperSpotIds).toHaveLength(15);
      expect(zone.lowerSpotIds).toHaveLength(15);
    });
  });

  it("creates zone F as a ten-spot vertical strip on the right", () => {
    const fSpots = PARKING_GEOMETRY.spots.filter((spot) => spot.zone === "F");
    expect(fSpots).toHaveLength(10);
    expect(fSpots.every((spot) => spot.row === "vertical")).toBe(true);
    expect(new Set(fSpots.map((spot) => spot.x)).size).toBe(1);
    expect(fSpots.every((spot) => spot.x > PARKING_GEOMETRY.zones[0]!.bounds.x + PARKING_GEOMETRY.zones[0]!.bounds.width)).toBe(true);
    expect(fSpots.map((spot) => spot.id)).toEqual([
      "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
    ]);
    expect(fSpots.map((spot) => spot.y)).toEqual([...fSpots.map((spot) => spot.y)].sort((a, b) => a - b));
  });

  it("places the main road between zones A-E and zone F", () => {
    const { accessRoad, fStripBounds, layout } = PARKING_GEOMETRY;
    expect(layout.zoneRightX).toBeLessThan(accessRoad.x);
    expect(accessRoad.x + accessRoad.width).toBeLessThan(fStripBounds.x);
    expect(layout.mainRoadCenterX).toBe(accessRoad.x + accessRoad.width / 2);
    expect(PARKING_GEOMETRY.entrance.x).toBe(layout.mainRoadCenterX);
    expect(PARKING_GEOMETRY.exit.x).toBe(layout.mainRoadCenterX);
  });

  it("aligns all five zone connectors to the shared main-road centerline", () => {
    expect(PARKING_GEOMETRY.zoneConnectors).toHaveLength(5);
    const connectorEndX = new Set(PARKING_GEOMETRY.zoneConnectors.map((connector) => connector.centerline.end.x));
    expect([...connectorEndX]).toEqual([PARKING_GEOMETRY.layout.mainRoadCenterX]);

    PARKING_GEOMETRY.zones.forEach((zone) => {
      const connector = PARKING_GEOMETRY.zoneConnectors.find((candidate) => candidate.id === `connector-${zone.id}`);
      expect(connector?.centerline.start.y).toBe(zone.laneY);
      expect(connector?.centerline.end.y).toBe(zone.laneY);
      expect(PARKING_GEOMETRY.layout.zoneLaneY[zone.id]).toBe(zone.laneY);
    });
  });

  it("gives each F spot a right-turn access connector from the main road", () => {
    expect(PARKING_GEOMETRY.fAccessConnectors).toHaveLength(10);
    PARKING_GEOMETRY.fAccessConnectors.forEach((connector) => {
      expect(connector.centerline.start.x).toBe(PARKING_GEOMETRY.layout.mainRoadCenterX);
      expect(connector.centerline.end.x).toBeGreaterThan(connector.centerline.start.x);
      expect(connector.bounds.x + connector.bounds.width).toBe(connector.centerline.end.x);
    });
  });
});
