# PROMPT — TechGAR Production Rewrite: Read 6 Plans, Implement All 6 (Backend + Frontend)

You are a Principal Computer Vision, Distributed Systems & Frontend Engineer. Your mission is to **read six specification documents and implement the complete TechGAR smart-parking system from scratch** — backend multi-camera vehicle tracking AND web frontend — a fresh rewrite with zero legacy code.

---

## 1. READ THESE SIX DOCUMENTS FIRST (in order)

**Backend (tracking, identity, parking):**
```
1. "D:\techgar_new\PLAN_1_SYSTEM_WORKFLOW_ARCHITECTURE_ROADMAP.md"
2. "D:\techgar_new\PLAN_2_ALGORITHMIC_FORMULATIONS_MATHEMATICAL_MODELS.md"
3. "D:\techgar_new\PLAN_3_GROUND_TRUTH_BENCHMARKS_VALIDATION_RUBRIC.md"
```

**Frontend (driver / monitor / kiosk web application):**
```
4. "D:\techgar_new\PLAN_4_FRONTEND_SYSTEM_WORKFLOW_ARCHITECTURE_ROADMAP.md"
5. "D:\techgar_new\PLAN_5_FRONTEND_ALGORITHMIC_FORMULATIONS_MATHEMATICAL_MODELS.md"
6. "D:\techgar_new\PLAN_6_FRONTEND_GROUND_TRUTH_BENCHMARKS_VALIDATION_RUBRIC.md"
```

- **PLAN 1** defines the 10-stage backend pipeline (dual-stream ingestion → environmental normalization → local detection → local tracking → homography projection → world fusion → topology-constrained association → Global Identity Registry → slot occupancy engine → event dispatch) and the Phase 0–6 gated roadmap with entry/exit criteria.
- **PLAN 2** defines every backend mathematical mechanism (adaptive motion segmentation, lag-aware Kalman state model, homography covariance propagation, composite cost association, slot IoU/Coverage/hysteresis, identity retention rules) in LaTeX — implement exactly as specified.
- **PLAN 3** defines the 9 backend benchmark scenarios (A–I), the metric suite (IDSW, MOTA, IDF1, slot P/R/F1, handoff accuracy, latency), the 4-experiment ablation framework, and the 100-point backend rubric.
- **PLAN 4** defines the 8-stage frontend pipeline (snapshot ingestion & validation → snapshot store → world→SVG projection → display-state resolution → session & navigation → SVG rendering → camera panels → operator actions), three pages (Driver `/?session=`, Monitor `/monitor`, Kiosk `/kiosk/entry`), and the Phase FE-0→FE-5 roadmap with entry/exit criteria.
- **PLAN 5** defines every frontend mathematical mechanism (least-squares affine world→SVG fit + inverse, marker smoothing with teleport guard, display-hold state machine, lane-graph Dijkstra routing, off-route detection, session-parking resolution, polling backoff) in LaTeX.
- **PLAN 6** defines the 9 frontend benchmark scenarios (F-A…F-I), frontend metrics (persistence, flicker count, projection accuracy, render latency, route validity, a11y), the 3-experiment frontend ablation, and the 100-point frontend rubric.

**Do not begin writing code until you have read all six documents completely.**

---

## 2. CORE INVARIANTS (memorize before coding)

**Backend:**
```
ONE physical vehicle ⟷ exactly ONE active Global ID — for its entire lifetime in the facility.
```
Uncertainty (lag, occlusion, camera handoff, parking dwell) may only change **confidence and lifecycle state**, never arbitrarily change a vehicle's identity. Session survival must be 100% while a vehicle remains in the facility.

**Frontend:**
```
The frontend NEVER infers identity — it renders only what the backend decides.
```
Display state (observed / missing-hold / parked / ghost-hidden) comes from backend snapshot flags (`stale_seconds`, `display_hold_seconds`, `parked_slot_id`) per the truth table in PLAN 4 Stage F4. A parked vehicle stays visible with zero motion; a marker never flickers on one empty poll; a route renders only after explicit user confirmation.

---

## 3. IMPLEMENTATION RULES

1. **ZERO LEGACY CODE** — do not copy from, import, or reuse any existing module in this repository. Formulate fresh implementations from first principles per the six plans. Create NEW packages; do not modify existing pipeline files.
2. **Follow phase gates strictly** — backend Phase 0→6 (PLAN 1) then frontend Phase FE-0→FE-5 (PLAN 4), in order. Each phase's exit criteria (Pass/Fail tests) must pass before starting the next phase. Write the tests defined in the plans as you go.
3. **Implement the math faithfully** — every formula in PLAN 2 (adaptive threshold τ_t, covariance propagation Σ_w = J_H·Σ_p·J_Hᵀ + ..., cost matrix C_ij with all six components, slot temporal window, hysteresis) and PLAN 5 (affine fit A*, teleport guard d_snap, display-hold visible(v,t), Dijkstra lane graph, off-route d_route) must appear in code with the exact semantics specified.
4. **Validate from day one** — backend: 9 scenarios (A–I) as automated tests + metric suite + ablation (PLAN 3). Frontend: 9 scenarios (F-A…F-I) as component/e2e tests + frontend metrics + ablation (PLAN 6). Both 100-point rubrics are your acceptance tests.
5. **GPS is forbidden.** All positions live in the calibrated local world coordinate system (cm/m on the floor plane); the frontend works in the SVG projection of that same system.
6. **Time is the primary axis** — every TTL, grace window, evidence window, and display hold is measured in real seconds from timestamps, never in frame counts or frontend-invented timers.
7. **Frontend is a pure consumer** — it may filter, interpolate position (within hold limits), and change visual state; it must NEVER assign, change, or merge Global IDs, and never cache a vehicle beyond the backend's display-hold to "keep it alive".

---

## 4. TARGET ARCHITECTURE

### 4a. Backend (from PLAN 1 §2)

```
Camera 1 ─► Frame Ingestion ─┐
                             ├─► Timestamp Alignment
Camera 2 ─► Frame Ingestion ─┘
          │
          ▼
Environmental Normalization ─► Local Detection ─► Local Camera Tracking
          │
          ▼
Pixel-to-World Projection ─► World Detection Fusion
          │
          ▼
Topology-Constrained Association ─► GLOBAL IDENTITY REGISTRY (sole authority)
          │
          ├─► Vehicle State
          ├─► Parking Slot State Engine
          ├─► Session Binding
          └─► Runtime API (JSON snapshot + MJPEG) ── consumed by frontend
```

Constraints:
- The **Global Identity Registry is the single authority** that mints/retires Global IDs. Camera-local trackers only *propose* observations.
- Topology before appearance: Camera 2 may only match a Camera 1 identity if it exited through the calibrated exit polygon into a valid successor's entry polygon, within feasible time and distance bounds.

### 4b. Frontend (from PLAN 4 §2)

```
Runtime API (snapshot ~1 Hz polling + MJPEG <img>)
   │
   ▼
F1 Snapshot Ingestion & Validation (schema, monotonic frame_index, backoff, coalesce)
   ▼
F2 Snapshot Store (single UI truth — no identity inference on commit)
   ▼
F3 World→SVG Projection (one affine for slots, markers, routes, gates)
   ├──► F4 Display-State Resolution (truth table: observed | parked | hold | ghost)
   │        ▼
   │     F6 SVG Rendering (layers, marker transition + teleport guard, pan/zoom)
   ├──► F5 Session & Navigation (claim, poll, route after confirm, voice, off-route)
   └──► F7 Camera Panels (MJPEG + health from snapshot)
F8 Operator Actions (Reset ID with confirm, gate editor SVG→world)
```

Runtime API contract the backend MUST expose and the frontend MUST consume:
- `GET /api/runtime/snapshot` — immutable JSON: vehicles (with `global_id, state, observed, parked_slot_id, stale_seconds, display_hold_seconds, position`), slot occupancy, slot_layout, cameras, events, timestamp/sequence
- `GET /api/runtime/status` — liveness + frame index
- `GET /api/runtime/cameras/{cam}.mjpg` — annotated MJPEG
- `POST /api/runtime/reset-identities` — operator reset (returns retired count)
- Session API: claim / get / select-spot / exit / waiting list

Three pages: Driver `/?session=<id>` (only the session's vehicle + route + Vietnamese voice), Monitor `/monitor` (all vehicles, dual cameras, event trace, Reset ID), Kiosk `/kiosk/entry` (QR deep-links).

Suggested frontend stack (document your choice): React + TypeScript strict (no `any`) + Vite + Tailwind, Zustand store, SVG map, Vitest + React Testing Library + Playwright. Pure functions for projection/display/routing (testable without DOM); components only consume stores.

---

## 5. MANDATORY BENCHMARK TARGETS

**Backend (from PLAN 3):**
```
IDSW (mandatory scenarios)          = 0
One vehicle → one ID                = always true
Invalid handoff rate                = 0
Session survival                    = 100%
Slot occupancy F1                   ≥ 0.97
Slot ownership accuracy             ≥ 0.98
Handoff median latency              ≤ 500 ms, p95 ≤ 1.5 s
Mean throughput                     ≥ 10 FPS, sustained min ≥ 6 FPS
Median e2e latency                  ≤ 250 ms, p95 ≤ 750 ms
Backend rubric score                ≥ 90/100
No GPS dependency                   = verified
```

**Frontend (from PLAN 6):**
```
Flicker                             = 0 (standard fixtures)
Persistence (should-display)        = 1.00
Projection accuracy ε_proj          ≤ 2 px
Driver page shows only session GID  = always true
Route only after confirmation       = always true
Reset ID confirmed + idempotent     = correct
Offline never clears the map        = correct (F-I)
Render p95 (160 slots)              ≤ 100 ms
Bundle gzip                         ≤ 350 KB
E2E 9 scenarios (F-A…F-I)           = all pass
Frontend rubric score               ≥ 90/100
```

Automatic rejection — backend (any one = fail): ID change during valid handoff · ID change after short occlusion · one GID on two physical vehicles · session inaccessible while vehicle inside · one-frame transit causing permanent occupancy · parked vehicle disappearing because motion = 0 · overload causing ID minting.

Automatic rejection — frontend (any one = fail): parked marker vanishing without observation (F-A) · marker flickering on missing snapshots (F-B) · ghost persisting forever (F-C) · driver page showing another vehicle (F-D) · route drawn before user confirmation (F-F) · silent off-route redirect (F-G) · reset double-fire or without confirm (F-H) · network loss blanking the map (F-I) · frontend assigning/changing any Global ID anywhere.

---

## 6. EXECUTION CHECKLIST

Backend phases (in order, report before moving on):

- [ ] **Phase 0**: data contracts (timestamp/world/slot/identity schemas), measure camera skew + homography residuals (use > 4 calibration points — 4-point homography gives zero residual and proves nothing), measure seam disagreement empirically, baseline metrics.
- [ ] **Phase 1**: latest-frame buffering, timestamp pairing, adaptive background + dual-stage motion evidence, time-based track persistence, occlusion-group handling, template-matching recovery, no new Global ID during recovery grace window.
- [ ] **Phase 2**: world projection with covariance, overlap/handoff polygons, cross-camera detection fusion (information filter), one-to-one fusion guarantee.
- [ ] **Phase 3**: Global Identity Registry with full lifecycle (PROVISIONAL → ACTIVE → TEMPORARILY_MISSING/OCCLUDED → PARKED → EXIT_CONFIRMED → RETIRED), append-only audit events, new-ID prohibition window, collision quarantine.
- [ ] **Phase 4**: slot occupancy engine — world footprint projection, IoU + Coverage + centroid + inward-motion, temporal arrival claims, dwell confirmation, hysteresis, parked identity persistence.
- [ ] **Phase 5**: session binding via Global ID + persistent vehicle fingerprint, audited identity remapping, exit-only session deletion.
- [ ] **Phase 6**: performance hardening — stage timing, bounded queues, batched expensive ops, subscriber-gated video encoding, overload behavior that raises uncertainty but never mints IDs.

Frontend phases (after backend Phase 3 exposes a stable snapshot schema — mock fixtures may substitute earlier):

- [ ] **Phase FE-0**: TypeScript types from backend JSON contracts (strict, no `any`); 6 standard fixtures (normal, flicker-gap, ghost, parked-long, post-reset, offline); Vitest + Playwright skeleton.
- [ ] **Phase FE-1**: ingestion & store — coalesce (1 in-flight), exponential backoff (cap 5 s), schema validation (invalid = keep old state, never clear), monotonic frame_index, connection states.
- [ ] **Phase FE-2**: projection & display-state — least-squares affine both directions (ε ≤ 2 px), truth-table rows each unit-tested, teleport guard; ghosts hidden by backend flags, NOT frontend timers.
- [ ] **Phase FE-3**: map rendering & camera panels — SVG layers, marker transform transition, 160-slot render ≤ 100 ms p95, MJPEG panels with health + offline placeholder.
- [ ] **Phase FE-4**: driver navigation & session — idempotent claim, session polling, only-session filter, slot-center fallback, route after explicit confirmation, Vietnamese voice, off-route warning (no silent redirect), parked/exit flows, session-404 end page.
- [ ] **Phase FE-5**: monitor, kiosk & operator — Reset ID (confirm + pending + result), gate editor (6 points SVG→world, pan locked), waiting-session kiosk QR, full e2e suite green.

Then run:
- [ ] Backend: 9 scenarios (A–I) automated + 4-ablation + 100-point rubric self-assessment.
- [ ] Frontend: 9 scenarios (F-A…F-I) component/e2e + 3-ablation + 100-point rubric self-assessment.
- [ ] Integrated: live end-to-end run (backend runtime server + frontend dev server) exercising QR → driver navigation → parking → exit.

---

## 7. DELIVERABLES

1. A new backend implementation package.
2. A new frontend application (separate directory).
3. Unit tests for every mathematical mechanism in PLAN 2 and PLAN 5.
4. Automated benchmark tests: backend 9 scenarios (A–I), frontend 9 scenarios (F-A…F-I).
5. Evaluation harnesses: backend metric suite + 4-ablation; frontend metric suite + 3-ablation.
6. A `README` (root) explaining how to run both applications, all tests, benchmarks, and ablations.
7. Final rubric scorecards — one backend, one frontend (each out of 100) with per-section breakdown and pass/fail evidence.

Begin by reading the six plan documents now. Then present: (a) your reading summary in ≤ 12 bullets, (b) your backend module breakdown mapped to the 10 pipeline stages, (c) your frontend module breakdown mapped to the 8 frontend stages, (d) your Phase 0 + FE-0 plan. Wait for confirmation only if something in the plans is contradictory — otherwise proceed autonomously phase by phase.
