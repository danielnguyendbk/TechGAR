# Routing Specification

## Lane graph
Use typed nodes and edges:
```ts
export interface LaneNode {
  id: string;
  x: number;
  y: number;
  kind: "entrance" | "exit" | "junction" | "lane" | "spot-entry" | "access-anchor";
}

export interface LaneEdge {
  from: string;
  to: string;
  distance: number;
  direction: "one-way" | "two-way";
}
```

## Vehicle route
- Start: bottom-right entrance node.
- End: selected spot-entry node.
- Use Dijkstra or A*.
- Respect edge direction.
- Route geometry must use graph edges only.
- Render as blue line with white under-stroke for contrast.
- Use arrowheads to indicate direction.

## Walking route
For recommendations, optionally show a dotted blue path from selected spot to the broad need anchor.

## Invalid selected spot
When selected spot leaves `empty`:
- clear or visually pause route;
- show warning sheet;
- calculate next best alternative;
- wait for explicit user action.

## Route validity tests
- Every route segment corresponds to a graph edge.
- No point enters a parking-spot polygon.
- No route crosses landscaping/island polygons.
- Route begins at entrance and ends at selected spot-entry node.
