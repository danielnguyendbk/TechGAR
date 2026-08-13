# Smart Parking Frontend — Codex repository instructions

## Phase and objective
Build the **frontend only** for a Smart Parking web app used by drivers after scanning a QR code at the parking entrance. The page must run without any backend by consuming deterministic mock events from two independent cameras.

The app has three driver experiences:
1. **Browse:** view the current parking map and optionally select any stable empty spot.
2. **Recommendation:** choose one high-level destination need — `Shopping`, `Dịch vụ`, or `Giải trí` — and receive the best three currently empty spots.
3. **Navigation:** after explicit confirmation, display a valid driving route from the bottom-right entrance to the selected spot.

Read before coding:
- `docs/PRODUCT_REQUIREMENTS.md`
- `docs/LAYOUT_SPEC.md`
- `docs/MOCK_DATA_SPEC.md`
- `docs/RECOMMENDATION_SPEC.md`
- `docs/ROUTING_SPEC.md`
- `docs/COMPONENT_SPEC.md`
- `docs/TEST_PLAN.md`
- `assets/reference/approved-desktop.png`
- `assets/reference/approved-mobile-entry.png`
- `assets/reference/approved-mobile-recommendation.png`

## Required stack
- React 19 or current stable React supported by Vite
- TypeScript with `strict: true`
- Vite
- Tailwind CSS
- shadcn/ui primitives only where they reduce implementation time
- Lucide React icons
- Zustand
- SVG for parking geometry, markers, and routes
- Vitest + React Testing Library
- Playwright

Use `pnpm` unless the repository already uses another package manager. Do not use `any`.

## Out of scope
Do not create or scaffold:
- Python, FastAPI, Flask, OpenCV
- REST or WebSocket servers
- databases
- authentication
- booking/reservation/payment
- license-plate recognition
- raw camera video
- ROI calibration/admin pages

## Non-negotiable parking layout
- One driver-facing page.
- Do not show a QR code inside the app.
- Vertical order from top to bottom: `E`, `D`, `C`, `B`, `A`.
- Zone `A` must be at the bottom.
- Every zone A–E has exactly two opposing parking rows with one drivable center lane.
- Zone `F` is a vertical strip on the right.
- Entrance is bottom-right; exit is top-right.
- Spot IDs:
  - A01–A30
  - B01–B30
  - C01–C30
  - D01–D30
  - E01–E30
  - F01–F10
- Total: 160 spots.
- Status colors:
  - `empty`: green
  - `occupied`: red
  - `transitioning`: amber/yellow
  - `unknown`: gray
- Recommendation or selection never changes the status fill; use a blue outline, glow, and map pin.
- Routes must follow the lane graph and never cross spots, islands, medians, or walls.

## Driver-mode rules
Use:
```ts
export type DriverMode = "entry" | "browse" | "recommendation" | "navigation";
```

### Entry mode
Show a bottom sheet with:
- `Nhận đề xuất vị trí đỗ xe`
- `Chỉ xem các ô đang trống`
- secondary link `Bỏ qua`

### Browse mode
- Do not ask for destination.
- Show all spots by default.
- Filters: `Chỉ hiện ô trống`, `Hiện tất cả trạng thái`.
- An empty spot may be selected manually.
- Amber, red, and gray spots may be inspected but cannot be selected for navigation.
- Persistent action: `Tìm chỗ phù hợp`.

### Recommendation mode
- Ask only for one high-level need:
  - `Shopping`
  - `Dịch vụ`
  - `Giải trí`
- Do not drill down to cinema, ATM, supermarket, etc.
- Return best recommendation plus two alternatives.
- Driver must explicitly confirm with `Chọn {spotId} và chỉ đường`.
- Show `Bỏ gợi ý và xem toàn bộ bãi`.
- Show disclaimer: `Vị trí không được giữ trước và có thể thay đổi theo tình trạng thực tế.`

### Navigation mode
- Draw route only after user confirmation.
- Manual browse selection can also start navigation.
- If selected spot becomes amber, red, or gray, pause the route and show a warning; do not silently redirect.

## Parking-status eligibility
Only `empty` spots are selectable and recommendable.

Never recommend or navigate to:
- `transitioning`
- `occupied`
- `unknown`

Amber/yellow is exclusively the transitional camera state. It must never represent a recommendation.

## Two-camera ownership
Ownership is explicit and disjoint.

For every zone A–E:
- `cam-left`: top row 01–08 and bottom row 16–23
- `cam-right`: top row 09–15 and bottom row 24–30

Zone F:
- `cam-right`: F01–F10

Validate ownership in code and tests. A camera event that targets a non-owned spot must be rejected and logged in development.

## Architecture rules
- Geometry is typed data, not repeated JSX.
- Domain state is separate from visual geometry.
- Components consume stores/services, never raw mock sequences.
- Use one SVG coordinate system for layout, lane graph, spots, labels, access markers, and routes.
- Put camera mock behind an interface:

```ts
export interface ParkingDataSource {
  getSnapshot(): Promise<ParkingSnapshot>;
  subscribe(listener: (event: ParkingEvent) => void): () => void;
  start(): void;
  stop(): void;
}
```

- Mock scenarios must be deterministic; do not use unseeded randomness.
- Revisions are monotonic per spot; stale events are ignored.
- Counts derive from canonical spot state.
- Recommendation logic is a pure function.
- Routing logic is a pure function over a typed lane graph.

## UX rules
- Mobile-first; drivers use the page on a phone.
- Desktop should visually follow the approved reference.
- Use Vietnamese copy exactly where specified.
- Minimum touch target: 44×44 CSS pixels.
- Map supports pan/zoom on mobile and has reset-view.
- Use color plus label/icon; do not rely on color alone.
- Header remains compact while map gets maximum space.
- Bottom sheets must have clear close/back affordances.

## Workflow
1. Inspect repository and all source files.
2. Write `docs/IMPLEMENTATION_PLAN.md` before coding.
3. Scaffold only missing frontend files.
4. Implement domain types, geometry, and ownership validation.
5. Implement deterministic two-camera mock and Zustand stores.
6. Implement entry, browse, recommendation, and navigation modes.
7. Implement recommendation and routing as testable pure modules.
8. Implement responsive desktop/mobile UI.
9. Add unit, integration, and Playwright tests.
10. Run lint, typecheck, tests, build, and capture screenshots.
11. Write `artifacts/VALIDATION.md` with checks and known limitations.

## Definition of done
- Runs with one command and no backend.
- 160 unique spots render.
- E→A order and F strip are correct.
- Every A–E zone has two opposing rows and a center lane.
- Two camera mocks update independently.
- Amber spots are never recommendable/selectable.
- User can skip recommendation and browse only.
- User explicitly confirms before route rendering.
- Recommendation supports only Shopping, Dịch vụ, Giải trí.
- Selected spot changing status pauses route and asks user what to do.
- Desktop and mobile screenshots exist.
- Lint, typecheck, unit tests, Playwright, and build pass.
