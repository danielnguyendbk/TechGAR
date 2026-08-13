import type { SpotId } from "../domain/parking";
import {
  PARKING_GEOMETRY,
  distanceBetween,
  type ParkingGeometry,
  type Point,
} from "../geometry/parkingGeometry";

export interface LaneNode extends Point {
  id: string;
  kind: "entrance" | "exit" | "junction" | "lane" | "spot-entry" | "access-anchor";
}

export interface LaneEdge {
  id: string;
  from: string;
  to: string;
  distance: number;
  direction: "one-way" | "two-way";
}

export interface LaneGraph {
  nodes: LaneNode[];
  edges: LaneEdge[];
  entranceNodeId: string;
  exitNodeId: string;
  spotEntryNodeIds: Record<SpotId, string>;
}

function accessNodeId(y: number): string {
  return `access-${String(y).replace(".", "-")}`;
}

export function buildLaneGraph(geometry: ParkingGeometry = PARKING_GEOMETRY): LaneGraph {
  const nodes: LaneNode[] = [];
  const edges: LaneEdge[] = [];
  const spotEntryNodeIds = {} as Record<SpotId, string>;
  const nodeById = new Map<string, LaneNode>();

  const addNode = (node: LaneNode): void => {
    if (!nodeById.has(node.id)) {
      nodes.push(node);
      nodeById.set(node.id, node);
    }
  };

  const addEdge = (from: string, to: string, direction: LaneEdge["direction"] = "one-way"): void => {
    const start = nodeById.get(from);
    const end = nodeById.get(to);
    if (!start || !end) throw new Error(`Không tìm thấy nút làn đường cho cạnh ${from} -> ${to}`);
    edges.push({
      id: `${from}=>${to}`,
      from,
      to,
      distance: distanceBetween(start, end),
      direction,
    });
  };

  const entranceNodeId = "entrance";
  const exitNodeId = "exit";
  addNode({ id: entranceNodeId, kind: "entrance", ...geometry.entrance });
  addNode({ id: exitNodeId, kind: "exit", ...geometry.exit });

  const accessYValues = new Set<number>();
  geometry.zones.forEach((zone) => accessYValues.add(zone.laneY));
  geometry.spots.filter((spot) => spot.zone === "F").forEach((spot) => accessYValues.add(spot.entryPoint.y));

  const orderedAccessY = [...accessYValues].sort((a, b) => b - a);
  orderedAccessY.forEach((y) => {
    addNode({ id: accessNodeId(y), kind: "junction", x: geometry.entrance.x, y });
  });

  const accessPath = [
    entranceNodeId,
    ...orderedAccessY.map(accessNodeId),
    exitNodeId,
  ];
  for (let index = 0; index < accessPath.length - 1; index += 1) {
    const from = accessPath[index];
    const to = accessPath[index + 1];
    if (from && to) addEdge(from, to);
  }

  geometry.zones.forEach((zone) => {
    const connector = geometry.zoneConnectors.find((candidate) => candidate.id === `connector-${zone.id}`);
    if (!connector) throw new Error(`Thiếu đường nối cho khu ${zone.id}`);
    const zoneStartId = `zone-${zone.id}-start`;
    addNode({ id: zoneStartId, kind: "junction", ...connector.centerline.start });
    addEdge(accessNodeId(zone.laneY), zoneStartId);

    const zoneSpots = geometry.spots.filter((spot) => spot.zone === zone.id && spot.row === "top");
    const orderedColumns = [...zoneSpots].sort((a, b) => b.x - a.x);
    let previousNodeId = zoneStartId;

    orderedColumns.forEach((topSpot) => {
      const columnNodeId = `zone-${zone.id}-column-${topSpot.number}`;
      addNode({
        id: columnNodeId,
        kind: "lane",
        x: topSpot.x + topSpot.width / 2,
        y: zone.laneY,
      });
      addEdge(previousNodeId, columnNodeId);
      previousNodeId = columnNodeId;

      const bottomSpot = geometry.spots.find(
        (candidate) => candidate.zone === zone.id && candidate.number === topSpot.number + 15,
      );
      if (!bottomSpot) throw new Error(`Thiếu hình học ô đối diện ${topSpot.id}`);

      [topSpot, bottomSpot].forEach((spot) => {
        const entryId = `spot-${spot.id}`;
        spotEntryNodeIds[spot.id] = entryId;
        addNode({ id: entryId, kind: "spot-entry", ...spot.entryPoint });
        addEdge(columnNodeId, entryId);
      });
    });
  });

  geometry.spots.filter((spot) => spot.zone === "F").forEach((spot) => {
    const entryId = `spot-${spot.id}`;
    spotEntryNodeIds[spot.id] = entryId;
    addNode({ id: entryId, kind: "spot-entry", ...spot.entryPoint });
    addEdge(accessNodeId(spot.entryPoint.y), entryId);
  });

  Object.values(geometry.anchors).forEach((anchor) => {
    addNode({ id: `anchor-${anchor.id}`, kind: "access-anchor", x: anchor.x, y: anchor.y });
  });

  return { nodes, edges, entranceNodeId, exitNodeId, spotEntryNodeIds };
}

export const LANE_GRAPH = buildLaneGraph();
