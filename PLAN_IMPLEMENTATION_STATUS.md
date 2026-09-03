# TECHGAR — IMPLEMENTATION STATUS & REMAINING ACCEPTANCE GATES

Ngày đối chiếu: 2026-09-03  
Phạm vi: `PLAN_1` đến `PLAN_6`, source hiện tại, test tự động và benchmark deterministic.

## 1. Kết luận điều hành

- Backend đã có pipeline Stage 1–10, Registry, Slot Engine, Session API, commissioning,
  evaluator A–I, environmental checks, ablation và performance instrumentation.
- Frontend FE-0–FE-5 đã triển khai, build production và vượt toàn bộ fixture/E2E tự động.
- Toàn hệ thống **chưa được ký production acceptance**. Hai gate ngoại cảnh còn mở là:
  1. đã import ba recording hai camera cùng ROI/layout/calibration cũ, nhưng calibration
     chỉ có 4 điểm/camera (0 bậc tự do dư), `vd_17/vd_18` chưa có nhãn và calibration
     của `m_04` chưa xác định;
  2. chưa chạy lại bộ benchmark/handoff/parking trên recording thật sau khi chốt
     calibration, nên các chỉ số hiện tại chỉ chứng minh deterministic synthetic acceptance.
- Positive pixel parking synthetic hiện đạt `Slot F1 = 1.00`, ownership `= 1.00`,
  `IDSW = 0`; hard gate mô phỏng trước đây đã được đóng mà không nới threshold.
- Frontend đã publish private, nhưng chưa nối tới backend thực qua tunnel/public endpoint;
  vì vậy kết quả frontend hiện là acceptance trên fixture deterministic, chưa phải live-site
  acceptance end-to-end.

Ký hiệu trạng thái:

- **PASS**: code đã có và exit criteria tương ứng đã có test deterministic pass.
- **PARTIAL**: code chính đã có, nhưng production/field gate còn thiếu bằng chứng hoặc fail.
- **OPEN**: chưa thể chấp nhận theo rubric bắt buộc.

## 2. PLAN 1 — Backend phase map

| Phase | Phần đã thực hiện | Bằng chứng hiện tại | Trạng thái |
|---|---|---|---|
| 0 — Measurement & Data Contract | timestamp/camera contracts; pixel/world/covariance/slot/identity schema; calibration report; skew/seam survey; event taxonomy; commissioning; portable site manifest + legacy asset audit; MP4 replay theo monotonic timestamp | 3 recording hai camera đã có; ROI và 48 slot pixel hợp lệ; `vd_17/vd_18` nối được `shared_m_01`; replay smoke 220/220 cặp không reject skew và tạo Global ID; audit chỉ ra cả hai calibration cũ có 4 điểm/camera | **PARTIAL** — đã có recording/replay thật nhưng chưa có calibration dư bậc tự do, ground-contact/parallax survey và ground truth đầy đủ |
| 1 — Lag-Resilient Ingestion & Local Tracking | latest-frame replacement; timestamp pairing/skew reject; adaptive background + temporal difference; time-based recovery; occlusion group; template recovery; mint guard | lag 500 ms không làm chết track; recovery 1 s giữ local ID; merged blob giữ 2 latent tracks; bounded latest-frame test pass | **PASS** trong deterministic suite |
| 2 — World Projection & Fusion | anchor pixel→world; covariance propagation; seam/topology regions; covariance-weighted one-to-one fusion; metric footprint reconstruction | positive-definite covariance và dual-camera fusion tests pass; thin motion rim được clamp về kích thước xe đã survey | **PASS** trong deterministic suite; calibration thực địa còn phụ thuộc Phase 0 |
| 3 — Global Identity Registry | một Registry duy nhất; provisional/missing/occluded/parked/retired; recovery-before-mint; quarantine; append-only events; topology/time/direction/appearance association | scenarios A–F pass; `IDSW = 0` trong các scenario bắt buộc; no-prediction giảm còn 6/9 | **PASS** trong scenario suite; handoff thực địa còn cần đo |
| 4 — Parking Slot State Engine | IoU + coverage + inward + temporal claim + dwell + one-to-one ownership + hysteresis + parked retention | unit/world scenarios G–I pass; transit không chiếm slot; false-empty không release; positive pixel benchmark đạt `Slot F1 = 1.00`, ownership `= 1.00` | **PASS** trong deterministic suite; cần xác nhận lại trên recording thực |
| 5 — Session Protection | bind session↔GID/fingerprint; audited alias/reset; reconnect; chỉ xóa sau confirmed exit; session claim/select/parked/exit API | API/session/registry tests pass; snapshot và reset contract pass | **PARTIAL** — thiếu live physical-exit/reconnect E2E trên recording thực |
| 6 — Performance & Deployment Hardening | stage timers; ingestion tách processing; bounded queues; lower-rate slot analysis; subscriber-gated encoding; overload uncertainty + no-mint guard | pixel parking: mean `12.04 FPS`, min sustained `12.00 FPS`, p95 processing khoảng `35 ms`; overload/queue/video-gate tests pass | **PARTIAL** — thiếu browser-to-backend displayed latency, hardware acceleration và soak test thực |

## 3. PLAN 2 — Algorithm/model coverage

| Mục | Hiện trạng |
|---|---|
| §1 Motion segmentation & adaptive noise rejection | Đã triển khai background evidence, multi-delta frame difference, adaptive threshold, illumination verdict, shadow rejection. Difference mask được dùng làm seed để tái dựng trọn connected background component, tránh chiếu một motion rim mỏng như cả xe. Brightness/shadow/compression checks đều pass. |
| §2 Single-camera kinematic tracking | Đã triển khai Kalman theo timestamp, lag sub-step, covariance, time-based missing states và template recovery. |
| §3 Homography & topology | Đã triển khai homography, Jacobian covariance propagation, seam inflation, directed transition zones và cấm blind global search. |
| §4 Cross-camera cost matrix | Đã triển khai Mahalanobis, direction, geometry, appearance, topology, time feasibility, margin/defer và one-to-one assignment. |
| §5 Slot state engine | Đã triển khai đầy đủ cơ chế công thức và hysteresis. Inward evidence lấy trên toàn cửa sổ; stability và parked speed lấy trên tail window để không tạo điều kiện tự mâu thuẫn. Pixel benchmark đạt gate F1/ownership. |
| §6 Identity lifecycle | Đã triển khai score/gates, anti-fragmentation grace, retention, collision quarantine và audited reset/remap. |
| §7 Occlusion group | Đã triển khai group/latent tracks; merged observation không được dùng để ghi đè appearance hoặc mint identity. |

Các thay đổi đóng gate pixel→slot:

1. motion difference chỉ làm seed; connected component hiện tại từ background
   evidence cung cấp silhouette đầy đủ;
2. benchmark có empty-scene pre-roll đúng contract commissioning, không học chiếc
   xe đầu tiên thành background rồi tạo ghost khi xe rời vị trí;
3. Stage 5 vẫn clamp cạnh footprint về kích thước xe đã survey;
4. parked speed dùng displacement của footprint quan sát liên tiếp, không dùng
   residual momentum của Kalman như một phép đo mới;
5. inward motion được đo trong toàn window, còn ổn định/vận tốc được đo ở tail.

Các regression test khóa cả component reconstruction, metric footprint và positive
pixel parking acceptance; không threshold IoU/coverage nào bị hạ.

## 4. PLAN 3 — Validation/rubric evidence

### Bộ kết quả hiện tại

| Kiểm tra | Kết quả |
|---|---:|
| Backend pytest | **89 pass** |
| Scenario A–I | **9/9 pass** |
| Environmental checks | **3/3 pass** |
| Ablation `full` | **9/9 pass** |
| Ablation `no_prediction` | **6/9 pass** — có degradation như kỳ vọng |
| Ablation `no_frame_difference` | **9/9 pass** — suite world-level chưa bộc lộ degradation |
| Ablation `no_topology` | **8/9 pass** — invalid handoff xuất hiện đúng cơ chế suy hao |

Positive pixel parking benchmark (`P01` đi vào `D05`, dừng 6 giây):

| Metric | Giá trị | Gate PLAN 3 | Kết quả |
|---|---:|---:|---|
| IDSW | 0 | 0 | PASS |
| Global-ID fragmentation | 1 | = 1 | PASS |
| IDF1 | 0.9547 | ≥ 0.95 cảnh thường | PASS |
| MOTA | 0.9083 | ≥ 0.90 stress/occlusion | PASS |
| Mean throughput | 12.04 FPS | ≥ 10 FPS | PASS |
| Min sustained throughput | 12.00 FPS | ≥ 6 FPS | PASS |
| p95 processing latency | ~35 ms | ≤ 750 ms | PASS |
| Slot F1 (transition-tolerant) | 1.00 | ≥ 0.97 | PASS |
| Ownership accuracy (transition-tolerant) | 1.00 | ≥ 0.98 | PASS |
| False vacancy rate (transition-tolerant) | 0.00 | thấp | PASS |

Rubric deterministic trả `accepted = true`: 100/100 và mọi hard gate synthetic đều
đạt. Metric strict vẫn được báo cáo riêng (`Slot F1 = 0.8382`) vì nó cố ý tính cả
độ trễ temporal-confirm/release quanh thời điểm ground-truth đổi trạng thái; gate
PLAN 3 dùng metric transition-tolerant để không tự phạt hysteresis bắt buộc.

Kết quả này **không phải production acceptance**: evaluator đang chạy trên raster
synthetic, chưa phải camera/calibration/ground-truth thực địa.

Lệnh tái lập:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m techgar.site_assets config\site_manifest.json
.venv\Scripts\python.exe -m techgar.evaluation.cli --ablation --pixel-parking
```

## 5. PLAN 4 — Frontend phase map

| Phase | Phần đã thực hiện | Trạng thái |
|---|---|---|
| FE-0 — Contract & Fixture | strict domain types + manual boundary validation; 9 fixtures gồm normal, flicker-gap, ghost, parked-long, parked-fallback, driver-isolation, off-route, post-reset và offline; Vitest + Playwright | **PASS** |
| FE-1 — Ingestion & Store | one in-flight/coalesce; backoff 1/2/4/5 s; monotonic frame index; invalid/network error giữ snapshot cũ | **PASS** |
| FE-2 — Projection & Display-State | affine least-squares hai chiều; fallback; truth table; ghost/hold/parked rules; teleport guard | **PASS** |
| FE-3 — Map & Camera | SVG layers; 160 slots; keyboard/touch/ARIA; transform animation; MJPEG/offline placeholder | **PASS** |
| FE-4 — Driver Navigation & Session | idempotent deep-link claim; chỉ hiển thị GID của session; explicit route confirm; Dijkstra; off-route warning không silent reroute; voice-once tiếng Việt; parked/exit/session-ended | **PASS** |
| FE-5 — Monitor, Kiosk & Operator | monitor counts/cameras/trace; reset confirm/pending/result/map clear; gate editor đúng 6 điểm; kiosk waiting sessions + QR | **PASS** trên deterministic E2E |

## 6. PLAN 5 & PLAN 6 — Frontend algorithms and acceptance

Đã triển khai các mô hình PLAN 5: affine `world↔SVG`, smoothing/teleport guard,
display-hold state machine, lane-graph Dijkstra, off-route geometry, session parking
fallback, coalesced polling/backoff và QR deep-link.

Kết quả PLAN 6:

| Gate | Kết quả |
|---|---:|
| Vitest | **27/27 pass** |
| Playwright F-A…F-I desktop | **9/9 pass** |
| Playwright F-A…F-I mobile | **9/9 pass** |
| TypeScript strict typecheck | **pass** |
| Lint | **pass** |
| Production build (`/`, `/monitor`, `/kiosk/entry`) | **pass** |
| 160-slot render p95 | **≤ 100 ms, pass** |
| Client JS + CSS gzip | **221.9 KiB / 350 KiB, pass** |

Frontend đạt rubric trên fixture deterministic. Live acceptance vẫn cần backend thực,
camera stream và session/exit thật; deployment hiện chưa có tunnel binding tới backend.

Private production deployment:

- URL: <https://techgar-control.scli-xa2330670.chatgpt.site>
- Sites version: `3`
- Source commit: `4f07554fb1c5a1d00e163ec8afb19ce328863081`
- OpenGraph/Twitter image: `frontend/public/og.png`

## 7. Công việc còn lại theo thứ tự bắt buộc

1. **Đóng Phase 0 thực địa**: dùng các recording đã import, đo lại ít nhất 6 điểm
   calibration độc lập/camera, xác nhận đơn vị và kích thước slot, ground-contact/parallax,
   seam/topology; sau đó gán ground-truth slot/event/identity. Không dùng residual gần 0
   của calibration 4 điểm làm bằng chứng acceptance.
2. **Chạy positive pixel parking + handoff benchmark trên dữ liệu thật** và xác nhận
   `Slot F1 ≥ 0.97`, `ownership ≥ 0.98`, `IDSW = 0`, invalid handoff `= 0`.
3. **Nâng ablation frame-difference lên recording có nuisance thực** để
   `no_frame_difference` tạo degradation đo được; topology đã có degradation 8/9.
4. **Nối deployment frontend tới backend** bằng endpoint/tunnel được cấp quyền, rồi
   chạy live E2E cho snapshot, MJPEG, reset, gate save, session claim, parked và exit.
5. **Hardening cuối**: đo displayed-state latency browser `< 1 s`, soak test, CPU spike,
   camera stall/reconnect, hardware acceleration và lập production sign-off report.

Cho đến khi mục 1–2 hoàn tất, trạng thái release toàn hệ thống phải giữ là
**NOT PRODUCTION ACCEPTED**.
