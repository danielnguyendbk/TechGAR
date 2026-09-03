import type { RuntimeSnapshot, WorldPoint } from '@/lib/domain/types';
import { polygonCenter } from '@/lib/display/resolve';

export interface LaneNode {
  readonly id: string;
  readonly point: WorldPoint;
}

export interface LaneEdge {
  readonly from: string;
  readonly to: string;
  readonly cost?: number;
  readonly blocked?: boolean;
}

export interface LaneGraph {
  readonly nodes: readonly LaneNode[];
  readonly edges: readonly LaneEdge[];
}

function distance(a: WorldPoint, b: WorldPoint): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function shortestRoute(graph: LaneGraph, startId: string, targetId: string): string[] {
  const nodes = new Set(graph.nodes.map((node) => node.id));
  if (!nodes.has(startId) || !nodes.has(targetId)) return [];
  const distances = new Map([...nodes].map((id) => [id, Number.POSITIVE_INFINITY]));
  const previous = new Map<string, string>();
  const queue = new Set(nodes);
  distances.set(startId, 0);
  while (queue.size > 0) {
    const current = [...queue].reduce((best, id) =>
      (distances.get(id) ?? Infinity) < (distances.get(best) ?? Infinity) ? id : best,
    );
    queue.delete(current);
    if (current === targetId) break;
    const currentNode = graph.nodes.find((node) => node.id === current);
    if (!currentNode || !Number.isFinite(distances.get(current))) break;
    for (const edge of graph.edges) {
      if (edge.blocked || (edge.from !== current && edge.to !== current)) continue;
      const neighbor = edge.from === current ? edge.to : edge.from;
      if (!queue.has(neighbor)) continue;
      const neighborNode = graph.nodes.find((node) => node.id === neighbor);
      if (!neighborNode) continue;
      const weight = edge.cost ?? distance(currentNode.point, neighborNode.point);
      const candidate = (distances.get(current) ?? Infinity) + weight;
      if (candidate < (distances.get(neighbor) ?? Infinity)) {
        distances.set(neighbor, candidate);
        previous.set(neighbor, current);
      }
    }
  }
  if (!Number.isFinite(distances.get(targetId))) return [];
  const route = [targetId];
  while (route[0] !== startId) {
    const parent = previous.get(route[0]);
    if (!parent) return [];
    route.unshift(parent);
  }
  return route;
}

export function routeToSlot(
  snapshot: RuntimeSnapshot,
  start: WorldPoint,
  slotId: string,
): WorldPoint[] {
  const slots = snapshot.slot_layout;
  const target = slots.find((slot) => slot.slot_id === slotId);
  if (!target) return [];
  const centers = slots.map((slot) => ({ slot, center: polygonCenter(slot.polygon) }));
  const laneY = centers.reduce((sum, entry) => sum + entry.center[1], 0) / Math.max(centers.length, 1);
  const sortedX = [...new Set(centers.map((entry) => Math.round(entry.center[0] * 100) / 100))]
    .sort((a, b) => a - b);
  const nodes: LaneNode[] = sortedX.map((x, index) => ({ id: `lane-${index}`, point: [x, laneY] }));
  const edges: LaneEdge[] = nodes.slice(1).map((node, index) => ({ from: nodes[index].id, to: node.id }));
  const nearestStart = nodes.reduce((best, node) =>
    distance(node.point, start) < distance(best.point, start) ? node : best,
  );
  const targetCenter = polygonCenter(target.polygon);
  const nearestTarget = nodes.reduce((best, node) =>
    distance(node.point, targetCenter) < distance(best.point, targetCenter) ? node : best,
  );
  const ids = shortestRoute({ nodes, edges }, nearestStart.id, nearestTarget.id);
  return [start, ...ids.map((id) => nodes.find((node) => node.id === id)?.point)
    .filter((point): point is WorldPoint => point !== undefined), targetCenter];
}

