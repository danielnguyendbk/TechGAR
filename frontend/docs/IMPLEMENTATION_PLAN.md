# Smart Parking Frontend Implementation Plan

## Architecture

The application is a single Vite-powered React page with no backend dependencies. Strict TypeScript domain modules own parking state, geometry, recommendation, and routing. React components consume those modules through two Zustand stores:

- the parking store is the canonical source for spot status, per-spot revisions, camera health, and derived counts;
- the driver-flow store owns entry, browse, recommendation, and navigation state without duplicating spot status;
- a `ParkingDataSource` adapter supplies the initial snapshot and deterministic camera events;
- pure recommendation and routing functions operate on explicit typed inputs and remain independent of React and Zustand;
- one SVG coordinate system contains parking geometry, road/lane geometry, access anchors, route overlays, labels, and interaction targets.

Components never import mock event sequences. The application bootstrap connects the data source to the parking store and tears the subscription down on unmount.

## File Plan

```text
src/
  app/App.tsx
  components/
    AlternativeSpotList.tsx
    BrowseToolbar.tsx
    CameraHealthIndicator.tsx
    DestinationNeedSelector.tsx
    EntryChoiceSheet.tsx
    InvalidSpotWarningSheet.tsx
    MapControls.tsx
    MockControlPanel.tsx
    NavigationStatusBar.tsx
    ParkingLegend.tsx
    ParkingMap.tsx
    ParkingSpotShape.tsx
    RecommendationPanel.tsx
    SmartParkingHeader.tsx
    SpotDetailSheet.tsx
    SummaryCards.tsx
  domain/parking.ts
  geometry/parkingGeometry.ts
  mocks/MockParkingDataSource.ts
  mocks/scenarios.ts
  recommendation/recommendationEngine.ts
  routing/laneGraph.ts
  routing/routeEngine.ts
  stores/driverFlowStore.ts
  stores/parkingStore.ts
  styles/index.css
  tests/
  main.tsx
tests/e2e/
artifacts/screenshots/
```

Small shared UI helpers may be added under `src/components` or `src/lib` when they remove real duplication. No server, database, authentication, camera-video, or reservation code will be created.

## State Model

`SpotState` contains ID, zone, row, camera owner, status, confidence, revision, and updated time. `ParkingSnapshot` includes all 160 spot records and both camera health records. The parking store accepts snapshots and events, rejects ownership violations, ignores revisions that are not newer than the canonical per-spot revision, and derives counts from the spot map on every read.

The driver-flow store uses the required `DriverMode` union and owns the active need, current recommendation result, inspected spot, confirmed spot, warning status, browse filter, and paused-route state. The app reacts to canonical spot changes:

- an unconfirmed recommendation is recalculated when one of its candidates stops being empty;
- a confirmed spot leaving `empty` pauses the route and opens the exact status-specific warning;
- switching to an alternative is always an explicit driver action.

## Geometry Strategy

The fixed SVG viewBox is `0 0 1200 900`. Zones E, D, C, B, and A use generated data with monotonically increasing Y positions. Each zone has 15 upper-row and 15 lower-row spot rectangles facing a center lane. Zone F is a generated ten-spot vertical strip on the right. The entrance is at the bottom-right and exit at the top-right.

### Road Topology Correction

The corrected horizontal order is `zones A-E -> main access road -> zone F`. Shared layout constants own the zone right edge, main-road bounds and centerline, zone-F bounds, entrance/exit coordinates, and every zone-lane Y coordinate. Each A-E lane has a typed connector rectangle from the common zone edge to the same `mainRoadCenterX`; the SVG road surface, dashed centerline, and graph junction all consume those coordinates.

Zone F remains a separate vertical block to the right of the main road, with F01 at the top and F10 at the bottom. Its spot-entry anchors face left toward the road. The road and F block have a visible landscaping gap except at graph-valid F access segments.

The lane graph keeps the entrance, all ordered main-road junctions, and exit on `mainRoadCenterX`. A-E routes turn left through the appropriate connector, continue on the zone center lane, and use only the final short perpendicular edge to reach the spot entry. F routes turn right from the main road to a left-facing F spot entry. The rendered route is derived only from the Dijkstra result and remains in the same SVG coordinate system during resize, pan, and zoom.

Acceptance coverage will assert exact topology for C10, A05, E12, and F04, including the correct turn direction, zone lane, terminal anchor, aligned A-E junctions, and the absence of route points inside unrelated parking rectangles.

Geometry is generated once as typed data containing spot rectangles, zone bounds, lanes, landscaping islands, access markers, and each spot's graph-entry coordinate. Rendered spots are mapped from this data rather than repeated JSX. Mobile pan and zoom transform this same coordinate system; the map is never re-authored for a second breakpoint.

## Mock Event Strategy

`MockParkingDataSource` implements `getSnapshot`, `subscribe`, `start`, and `stop`, plus development-only deterministic stepping controls. The snapshot status pattern is based only on spot index, so it is reproducible. Two independent scenario queues cover normal updates, recommendation invalidation, selected-spot invalidation, recovery from amber, camera offline/recovery, stale revisions, and unauthorized ownership.

Events include camera ID, spot ID, status, confidence, revision, and a deterministic ISO timestamp. Ownership is checked both in the source and the canonical store. Invalid events are rejected and logged only in development. Offline camera events update health only and preserve all owned spot states.

## Recommendation Logic

The pure recommendation engine accepts canonical spots, geometry, the lane graph, a destination need, and a calculation timestamp. It first filters strictly to `status === "empty"`. It then computes:

```text
totalScore = drivingDistance * 0.35 + walkingDistance * 0.65 + congestionPenalty
```

Driving distance comes from the lane route between the entrance and the spot-entry node. Walking distance is Euclidean distance from the spot to the configured broad-need access anchor. Congestion penalty is deterministic and initially zero. Results sort by score then spot ID, returning one best result and up to two alternatives with an explainable Vietnamese reason and estimated walking time. Only Shopping, Dịch vụ, and Giải trí exist as selectable needs.

## Routing Logic

The typed lane graph follows zone center lanes and the right-side access road. Directed edges connect the entrance upward along the access road, connect each zone lane in the displayed direction, and connect every spot-entry node to its lane junction without crossing a spot. Dijkstra computes the vehicle route from the entrance to the confirmed spot. Route results contain node IDs, graph-edge IDs, coordinates, and total distance so tests can prove every rendered segment is a valid directed edge.

The active route is rendered with a white under-stroke, blue foreground stroke, and directional marker only after explicit confirmation. If the confirmed spot is no longer empty, active route styling is hidden until the driver explicitly changes destination or returns to the map.

## Responsive Behavior

The page is mobile-first. A compact header and summary strip leave most of the viewport to the map. Entry, recommendation, spot details, and warning content use accessible bottom sheets on mobile and restrained side panels on desktop. Browse and navigation actions remain reachable without covering critical map controls. Every action has at least a 44-by-44 CSS-pixel target.

The visual system follows the approved references: white application chrome, navy text, blue primary actions, dark asphalt roads, green islands, compact metric cards, and clear status colors with text/icon reinforcement. The QR panel in the desktop reference is intentionally omitted because the written source of truth prohibits showing a QR code in the app.

## Validation Plan

Vitest covers geometry totals/order/rows, camera ownership and authorization, stale revisions, derived counts, recommendation eligibility and determinism, and graph-valid routing. React Testing Library covers entry actions, browse filtering and selection eligibility, explicit recommendation confirmation, alternatives/disclaimer, and invalid selected-spot warnings.

Playwright covers browse-to-navigation, all three recommendation needs, recommendation invalidation and recalculation, selected-spot warning and alternative switching, camera degradation, mobile controls and sheets, desktop layout, and screenshot capture. Final validation runs:

```text
pnpm lint
pnpm typecheck
pnpm test
pnpm playwright test
pnpm build
```

Screenshots will be saved under `artifacts/screenshots` at 390x844 and 1440x900, and `artifacts/VALIDATION.md` will record command results, behavioral checks, spot/count evidence, and known limitations.
