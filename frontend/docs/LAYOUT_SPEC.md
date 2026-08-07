# Parking Layout Specification

## Global map
Use one SVG `viewBox`, recommended approximately `0 0 1200 900`.

### Fixed vertical order
Top to bottom:
1. Zone E
2. Zone D
3. Zone C
4. Zone B
5. Zone A

Zone F is a vertical strip on the right.

### Entry and exit
- Entrance: bottom-right, connected to the right-side vertical access road.
- Exit: top-right, continuing from the same access road.

## Zones A–E
Each zone contains:
- upper parking row facing the center lane;
- one drivable center lane;
- lower parking row facing the center lane;
- rounded outer boundary/island treatment.

Each zone contains 30 spots:
- upper row: 01–15
- lower row: 16–30

The row orientation should visually face the center lane.

## Zone F
- Vertical strip on the right.
- Ten spots: F01–F10.
- May be stacked bottom-to-top or top-to-bottom visually, but labels and geometry must be deterministic and match route nodes.
- Keep enough roadway between F and the outer entrance/exit road.

## Map layers
Render in this order:
1. background and landscaping;
2. roads and islands;
3. lane centerlines and direction arrows;
4. parking spots;
5. zone labels;
6. destination-need access markers;
7. route overlays;
8. recommendation/selection pins;
9. interactive hit areas.

## Status visuals
- empty: green fill
- occupied: red fill
- transitioning: amber fill
- unknown: gray fill
- hover/focus: subtle brighter border
- recommended: 3 px blue outline + soft blue glow
- confirmed: 4 px blue outline + map pin

## Direction arrows
Direction arrows must be decorative but consistent with graph direction. Never show arrows that contradict route direction.

## Mobile
- Preserve one SVG geometry; do not create a second hardcoded layout.
- Provide pan, pinch zoom, and reset-view.
- On initial load, fit entire parking lot width or a useful overview.
