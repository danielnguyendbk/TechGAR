import type { SpotId } from "../domain/parking";
import type { LaneEdge, LaneGraph, LaneNode } from "./laneGraph";

interface DirectedStep {
  edgeId: string;
  from: string;
  to: string;
  distance: number;
}

export interface RouteResult {
  nodeIds: string[];
  edgeIds: string[];
  points: Array<Pick<LaneNode, "x" | "y">>;
  distance: number;
}

function buildAdjacency(graph: LaneGraph): Map<string, DirectedStep[]> {
  const adjacency = new Map<string, DirectedStep[]>();
  graph.nodes.forEach((node) => adjacency.set(node.id, []));

  graph.edges.forEach((edge) => {
    adjacency.get(edge.from)?.push({ edgeId: edge.id, from: edge.from, to: edge.to, distance: edge.distance });
    if (edge.direction === "two-way") {
      adjacency.get(edge.to)?.push({ edgeId: edge.id, from: edge.to, to: edge.from, distance: edge.distance });
    }
  });
  return adjacency;
}

export function findRoute(graph: LaneGraph, startNodeId: string, endNodeId: string): RouteResult | null {
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  if (!nodeById.has(startNodeId) || !nodeById.has(endNodeId)) return null;

  const adjacency = buildAdjacency(graph);
  const unvisited = new Set(graph.nodes.map((node) => node.id));
  const distances = new Map<string, number>(graph.nodes.map((node) => [node.id, Number.POSITIVE_INFINITY]));
  const previous = new Map<string, { nodeId: string; edgeId: string }>();
  distances.set(startNodeId, 0);

  while (unvisited.size > 0) {
    let current: string | undefined;
    let currentDistance = Number.POSITIVE_INFINITY;
    unvisited.forEach((nodeId) => {
      const candidateDistance = distances.get(nodeId) ?? Number.POSITIVE_INFINITY;
      if (candidateDistance < currentDistance) {
        current = nodeId;
        currentDistance = candidateDistance;
      }
    });

    if (!current || !Number.isFinite(currentDistance)) break;
    unvisited.delete(current);
    if (current === endNodeId) break;

    (adjacency.get(current) ?? []).forEach((step) => {
      if (!unvisited.has(step.to)) return;
      const candidate = currentDistance + step.distance;
      if (candidate < (distances.get(step.to) ?? Number.POSITIVE_INFINITY)) {
        distances.set(step.to, candidate);
        previous.set(step.to, { nodeId: step.from, edgeId: step.edgeId });
      }
    });
  }

  const finalDistance = distances.get(endNodeId) ?? Number.POSITIVE_INFINITY;
  if (!Number.isFinite(finalDistance)) return null;

  const reversedNodes = [endNodeId];
  const reversedEdges: string[] = [];
  let cursor = endNodeId;
  while (cursor !== startNodeId) {
    const prior = previous.get(cursor);
    if (!prior) return null;
    reversedEdges.push(prior.edgeId);
    cursor = prior.nodeId;
    reversedNodes.push(cursor);
  }

  const nodeIds = reversedNodes.reverse();
  const edgeIds = reversedEdges.reverse();
  return {
    nodeIds,
    edgeIds,
    points: nodeIds.map((nodeId) => {
      const node = nodeById.get(nodeId);
      if (!node) throw new Error(`Thiếu nút ${nodeId} trong tuyến đường`);
      return { x: node.x, y: node.y };
    }),
    distance: finalDistance,
  };
}

export function findVehicleRoute(graph: LaneGraph, spotId: SpotId): RouteResult | null {
  const targetNodeId = graph.spotEntryNodeIds[spotId];
  return targetNodeId ? findRoute(graph, graph.entranceNodeId, targetNodeId) : null;
}

export function routeUsesOnlyValidEdges(graph: LaneGraph, route: RouteResult): boolean {
  const edgeById = new Map<string, LaneEdge>(graph.edges.map((edge) => [edge.id, edge]));
  return route.edgeIds.every((edgeId, index) => {
    const edge = edgeById.get(edgeId);
    const from = route.nodeIds[index];
    const to = route.nodeIds[index + 1];
    if (!edge || !from || !to) return false;
    return (
      (edge.from === from && edge.to === to) ||
      (edge.direction === "two-way" && edge.from === to && edge.to === from)
    );
  });
}
