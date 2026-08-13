# Prompt to paste into Codex

```text
Use $smart-parking-ui, $smart-parking-mock, $smart-parking-recommendation, $smart-parking-routing, and $smart-parking-qa.

Build the frontend-only Smart Parking driver application. Treat AGENTS.md, every file under docs/, and every image under assets/reference/ as the source of truth.

Do not create Python, FastAPI, Flask, OpenCV, a database, a REST server, a WebSocket server, or placeholder backend code.

Before writing code:
1. Inspect the current repository and all source files.
2. Write docs/IMPLEMENTATION_PLAN.md containing architecture, file plan, state model, geometry strategy, mock-event strategy, recommendation logic, routing logic, responsive behavior, and validation plan.
3. Then implement the complete application without stopping after scaffolding.

Main product behavior:
- The user opens the page after scanning a QR code at the parking entrance.
- The first mobile bottom sheet asks:
  1. "Nhận đề xuất vị trí đỗ xe"
  2. "Chỉ xem các ô đang trống"
  3. secondary text action "Bỏ qua"
- Recommendations are optional.
- If the user chooses recommendation, ask only for one of three high-level needs:
  - Shopping
  - Dịch vụ
  - Giải trí
- Do not ask for specific cinema, ATM, supermarket, restaurant, or other sub-destinations.
- Show one best recommendation and two alternatives.
- Only stable green `empty` spots may be recommended.
- Amber `transitioning`, red `occupied`, and gray `unknown` spots must never be recommended or selected for navigation.
- Recommendation styling is a blue outline/glow and map pin while preserving the original fill color.
- Driver must explicitly press "Chọn {spotId} và chỉ đường" before a route appears.
- The driver can press "Bỏ gợi ý và xem toàn bộ bãi" at any time.
- In browse mode, the user can inspect all spots and manually choose any green spot for navigation.
- Add a persistent "Tìm chỗ phù hợp" action in browse mode.

Parking layout:
- Zones from top to bottom: E, D, C, B, A.
- A is at the bottom.
- Every A–E zone has two opposing rows with one drivable center lane.
- F is a vertical parking strip on the right.
- Entrance is bottom-right and exit is top-right.
- IDs: A01–A30, B01–B30, C01–C30, D01–D30, E01–E30, F01–F10.
- Render 160 unique spots.
- Routes must remain inside valid lanes and never cross spots, medians, dividers, or walls.

Status rules:
- empty = green
- occupied = red
- transitioning = amber/yellow
- unknown = gray
- recommended/selected = preserve status fill and add blue selection treatment

Two-camera deterministic mock:
- For each A–E zone, cam-left owns top 01–08 and bottom 16–23.
- For each A–E zone, cam-right owns top 09–15 and bottom 24–30.
- cam-right owns F01–F10.
- Ownership must be validated and disjoint.
- Events include cameraId, spotId, status, confidence, revision, and updatedAt.
- Reject unauthorized camera-to-spot events.
- Ignore stale revisions.
- Camera offline preserves the last known spot states and shows degraded health; never mark its spots empty automatically.

Deterministic scenarios must include:
1. normal independent left/right updates;
2. an unconfirmed recommendation becoming transitioning;
3. a confirmed selected spot becoming transitioning;
4. a recommended spot becoming occupied;
5. an amber spot returning to empty;
6. cam-left offline and recovery;
7. cam-right offline and recovery;
8. stale revision event;
9. unauthorized ownership event.

When an unconfirmed recommended spot becomes non-empty:
- remove it from candidates;
- recalculate top three;
- update recommendation card without page reload.

When a confirmed selected spot becomes transitioning, occupied, or unknown:
- pause/hide active route styling;
- do not silently redirect;
- show a Vietnamese warning;
- show the next alternative;
- offer:
  - "Chuyển sang {alternativeSpotId}"
  - "Tiếp tục xem bản đồ"

Use these exact warnings:
- transitioning: "Ô {spotId} đang có phương tiện di chuyển vào hoặc ra."
- occupied: "Ô {spotId} hiện không còn trống."
- unknown: "Trạng thái ô {spotId} hiện chưa xác định."

Frontend stack:
- React + TypeScript + Vite
- Tailwind CSS
- shadcn/ui where useful
- Lucide icons
- Zustand
- SVG geometry and routes
- Vitest + React Testing Library
- Playwright

Required implementation modules:
- domain parking types
- typed parking geometry generator
- two-camera ownership validator
- deterministic MockParkingDataSource
- canonical parking Zustand store
- driver-flow Zustand store
- recommendation engine
- lane graph and route engine
- responsive SVG ParkingMap
- EntryChoiceSheet
- BrowseToolbar
- DestinationNeedSelector
- RecommendationPanel
- AlternativeSpotList
- SpotDetailSheet
- NavigationStatusBar
- CameraHealthIndicator
- ParkingLegend
- SummaryCards
- development-only MockControlPanel

Required tests:
- 160 unique IDs
- E/D/C/B/A visual order
- two rows per A–E zone
- F vertical strip
- ownership disjointness and authorization
- stale revisions ignored
- counts always match state
- only empty spots are eligible
- amber spot never recommended
- user can skip recommendations
- recommendation requires confirmation
- manual green-spot navigation works
- selected spot status-change warning flow
- routes use only valid lane edges
- desktop and mobile flows

Validation:
- run lint
- run TypeScript typecheck
- run unit/integration tests
- run Playwright
- run production build
- capture screenshots at 390×844 and 1440×900 under artifacts/screenshots
- write artifacts/VALIDATION.md

Do not ask questions unless a required local file is missing or unreadable. Make reasonable implementation decisions consistent with the specifications and continue until the complete frontend works.
```
