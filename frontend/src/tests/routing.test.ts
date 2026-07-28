import { describe, expect, it } from "vitest";
import { PARKING_GEOMETRY, pointInsideRect } from "../geometry/parkingGeometry";
import { LANE_GRAPH } from "../routing/laneGraph";
import { findVehicleRoute, routeUsesOnlyValidEdges } from "../routing/routeEngine";

function expectMainZoneRoute(spotId: "C10" | "A05" | "E12", zoneId: "C" | "A" | "E"): void {
  const route = findVehicleRoute(LANE_GRAPH, spotId);
  const zone = PARKING_GEOMETRY.zones.find((candidate) => candidate.id === zoneId);
  const spot = PARKING_GEOMETRY.spots.find((candidate) => candidate.id === spotId);
  expect(route).not.toBeNull();
  expect(zone).toBeDefined();
  expect(spot).toBeDefined();
  if (!route || !zone || !spot) return;

  const accessNodeIndex = route.nodeIds.indexOf(`access-${String(zone.laneY).replace(".", "-")}`);
  expect(accessNodeIndex).toBeGreaterThan(0);
  expect(route.points.slice(0, accessNodeIndex + 1).every((point) => point.x === PARKING_GEOMETRY.layout.mainRoadCenterX)).toBe(true);
  expect(route.points[accessNodeIndex]).toEqual({ x: PARKING_GEOMETRY.layout.mainRoadCenterX, y: zone.laneY });

  const lanePoints = route.points.slice(accessNodeIndex + 1, -1);
  expect(lanePoints.length).toBeGreaterThan(0);
  expect(lanePoints.every((point) => point.y === zone.laneY)).toBe(true);
  expect(lanePoints.every((point) => point.x < PARKING_GEOMETRY.layout.mainRoadCenterX)).toBe(true);

  const penultimate = route.points.at(-2);
  const terminal = route.points.at(-1);
  expect(penultimate?.x).toBe(terminal?.x);
  expect(terminal).toEqual(spot.entryPoint);
  expect(Math.abs((penultimate?.y ?? 0) - (terminal?.y ?? 0))).toBe(15);
}

describe("route engine", () => {
  it("connects the entrance to every spot using only valid directed edges", () => {
    PARKING_GEOMETRY.spots.forEach((spot) => {
      const route = findVehicleRoute(LANE_GRAPH, spot.id);
      expect(route, spot.id).not.toBeNull();
      if (!route) return;
      expect(route.nodeIds[0]).toBe(LANE_GRAPH.entranceNodeId);
      expect(route.nodeIds.at(-1)).toBe(LANE_GRAPH.spotEntryNodeIds[spot.id]);
      expect(routeUsesOnlyValidEdges(LANE_GRAPH, route)).toBe(true);
    });
  });

  it("keeps all route points outside parking rectangles", () => {
    PARKING_GEOMETRY.spots.forEach((target) => {
      const route = findVehicleRoute(LANE_GRAPH, target.id);
      expect(route).not.toBeNull();
      route?.points.forEach((point) => {
        expect(PARKING_GEOMETRY.spots.some((spot) => pointInsideRect(point, spot)), `${target.id} at ${point.x},${point.y}`).toBe(false);
      });
    });
  });

  it("routes C10 up the main road, left through C, and stops beside C10", () => {
    expectMainZoneRoute("C10", "C");
  });

  it("routes A05 only to the A junction before turning left", () => {
    expectMainZoneRoute("A05", "A");
  });

  it("routes E12 along the full main road before turning left", () => {
    expectMainZoneRoute("E12", "E");
  });

  it("routes F04 right from the main road without entering zones A-E", () => {
    const route = findVehicleRoute(LANE_GRAPH, "F04");
    const spot = PARKING_GEOMETRY.spots.find((candidate) => candidate.id === "F04");
    expect(route).not.toBeNull();
    expect(spot).toBeDefined();
    if (!route || !spot) return;

    expect(route.points.slice(0, -1).every((point) => point.x === PARKING_GEOMETRY.layout.mainRoadCenterX)).toBe(true);
    expect(route.points.at(-1)).toEqual(spot.entryPoint);
    expect(route.points.at(-1)!.x).toBeGreaterThan(PARKING_GEOMETRY.layout.mainRoadCenterX);
    expect(route.nodeIds.some((nodeId) => nodeId.startsWith("zone-"))).toBe(false);
  });
});
