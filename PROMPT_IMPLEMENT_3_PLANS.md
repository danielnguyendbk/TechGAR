# PROMPT — TechGAR Production Rewrite: Read 3 Plans, Implement All 3

You are a Principal Computer Vision & Distributed Systems Engineer. Your mission is to **read three specification documents and implement the complete TechGAR smart-parking multi-camera vehicle tracking system from scratch** — a fresh rewrite with zero legacy code.

---

## 1. READ THESE THREE DOCUMENTS FIRST (in order)

```
1. "D:\techgar_new\PLAN_1_SYSTEM_WORKFLOW_ARCHITECTURE_ROADMAP.md"
2. "D:\techgar_new\PLAN_2_ALGORITHMIC_FORMULATIONS_MATHEMATICAL_MODELS.md"
3. "D:\techgar_new\PLAN_3_GROUND_TRUTH_BENCHMARKS_VALIDATION_RUBRIC.md"
```

- **PLAN 1** defines the 10-stage pipeline (dual-stream ingestion → environmental normalization → local detection → local tracking → homography projection → world fusion → topology-constrained association → Global Identity Registry → slot occupancy engine → event dispatch) and the Phase 0–6 gated roadmap with entry/exit criteria.
- **PLAN 2** defines every mathematical mechanism (adaptive motion segmentation, lag-aware Kalman state model, homography covariance propagation, composite cost association, slot IoU/Coverage/hysteresis, identity retention rules) in LaTeX — implement them exactly as specified.
- **PLAN 3** defines the 9 numeric benchmark scenarios (A–I), the metric suite (IDSW, MOTA, IDF1, slot P/R/F1, handoff accuracy, latency), the 4-experiment ablation framework, and the 100-point acceptance rubric.

**Do not begin writing code until you have read all three documents completely.**

---

## 2. CORE INVARIANT (memorize before coding)

```
ONE physical vehicle ⟷ exactly ONE active Global ID — for its entire lifetime in the facility.
```

Every design decision must honor this. Uncertainty (lag, occlusion, camera handoff, parking dwell) may only change **confidence and lifecycle state**, never arbitrarily change a vehicle's identity. Session survival must be 100% while a vehicle remains in the parking facility.

---

## 3. IMPLEMENTATION RULES

1. **ZERO LEGACY CODE** — do not copy from, import, or reuse any existing module in this repository. Formulate fresh implementations from first principles per the three plans. Create a NEW package; do not modify existing pipeline files.
2. **Follow PLAN 1's phase gates strictly** — Phase 0 → 6 in order. Each phase's exit criteria (Pass/Fail tests) must pass before starting the next phase. Write the tests defined in the plans as you go.
3. **Implement PLAN 2's math faithfully** — every formula (adaptive threshold τ_t, covariance propagation Σ_w = J_H·Σ_p·J_Hᵀ + ..., cost matrix C_ij with all six components, slot temporal window, hysteresis thresholds) must appear in code with the exact semantics specified.
4. **Validate against PLAN 3 from day one** — implement the 9 benchmark scenarios as automated tests, the metric suite as an evaluation harness, and the ablation framework as runnable experiments. The 100-point rubric is your acceptance test.
5. **GPS is forbidden.** All positions live in the calibrated local world coordinate system (cm/m on the floor plane).
6. **Time is the primary axis** — every TTL, grace window, and evidence window is measured in real seconds from timestamps, never in frame counts.

---

## 4. TARGET ARCHITECTURE (from PLAN 1 §2)

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
          └─► Frontend Runtime State (JSON snapshot + MJPEG)
```

Constraints:
- The **Global Identity Registry is the single authority** that mints/retires Global IDs. Camera-local trackers only *propose* observations.
- Topology before appearance: Camera 2 may only match a Camera 1 identity if it exited through the calibrated exit polygon into a valid successor's entry polygon, within feasible time and distance bounds.
- The frontend is a visualization consumer — never a source of identity truth.

---

## 5. MANDATORY BENCHMARK TARGETS (from PLAN 3)

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
Overall rubric score                ≥ 90/100
No GPS dependency                   = verified
```

Automatic rejection (any one occurrence = fail): ID change during valid handoff · ID change after short occlusion · one GID representing two physical vehicles · session inaccessible while vehicle still inside · one-frame transit causing permanent occupancy · parked vehicle disappearing because motion = 0 · overload causing ID minting.

---

## 6. EXECUTION CHECKLIST

For each phase, complete in this order and report before moving on:

- [ ] **Phase 0**: data contracts (timestamp/world/slot/identity schemas), measure camera skew + homography residuals (use > 4 calibration points — 4-point homography gives zero residual and proves nothing), measure seam disagreement empirically, baseline metrics.
- [ ] **Phase 1**: latest-frame buffering, timestamp pairing, adaptive background + dual-stage motion evidence, time-based track persistence, occlusion-group handling, template-matching recovery, no new Global ID during recovery grace window.
- [ ] **Phase 2**: world projection with covariance, overlap/handoff polygons, cross-camera detection fusion (information filter), one-to-one fusion guarantee.
- [ ] **Phase 3**: Global Identity Registry with full lifecycle (PROVISIONAL → ACTIVE → TEMPORARILY_MISSING/OCCLUDED → PARKED → EXIT_CONFIRMED → RETIRED), append-only audit events, new-ID prohibition window, collision quarantine.
- [ ] **Phase 4**: slot occupancy engine — world footprint projection, IoU + Coverage + centroid + inward-motion, temporal arrival claims, dwell confirmation, hysteresis, parked identity persistence.
- [ ] **Phase 5**: session binding via Global ID + persistent vehicle fingerprint, audited identity remapping, exit-only session deletion.
- [ ] **Phase 6**: performance hardening — stage timing, bounded queues, batched expensive ops, subscriber-gated video encoding, overload behavior that raises uncertainty but never mints IDs.

Then run:
- [ ] The 9 benchmark scenarios (A–I) as automated pass/fail tests.
- [ ] The ablation suite (Full vs No-Frame-Difference vs No-Prediction vs No-Topology) — full pipeline must win on every metric.
- [ ] The 100-point rubric self-assessment.

---

## 7. DELIVERABLES

1. A new implementation package 
2. Unit tests for every mathematical mechanism in PLAN 2.
3. Automated benchmark tests for the 9 scenarios in PLAN 3.
4. An evaluation harness that computes the full metric suite and ablation experiments.
5. A `README` explaining how to run the pipeline, the tests, the benchmarks, and the ablations.
6. A final rubric scorecard (out of 100) with per-section breakdown and pass/fail evidence.

Begin by reading the three plan documents now. Then present: (a) your reading summary in ≤ 10 bullets, (b) your module breakdown mapped to the 10 pipeline stages, (c) your Phase 0 plan. Wait for confirmation only if something in the plans is contradictory — otherwise proceed autonomously phase by phase.
