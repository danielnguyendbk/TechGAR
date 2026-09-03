# TECHGAR — PLAN 4: FRONTEND SYSTEM WORKFLOW, ARCHITECTURE PIPELINE & IMPLEMENTATION ROADMAP

> **Loại tài liệu**: Production-grade Frontend Redesign Specification — từ first principles, zero legacy code
> **Phạm vi**: Ứng dụng web frontend của TechGAR — ba trải nghiệm (tài xế / giám sát / kiosk QR) tiêu thụ Runtime API + Session API
> **Ràng buộc nền tảng**: Frontend là **consumer thuần túy** — không bao giờ là nguồn chân lý danh tính; mọi quyết định hiển thị lấy từ snapshot backend
> **Liên kết**: PLAN 1 (backend pipeline, Stage 10) · PLAN 5 (toán frontend) · PLAN 6 (benchmark frontend)

---

## 1. Core Problem Formulation

Frontend TechGAR phải giải quyết năm bài toán riêng, không trùng với backend:

| # | Bài toán | Định lượng |
|---|---|---|
| F1 | **Độ trễ dữ liệu vs ổn định hiển thị** | Snapshot cập nhật ~1 Hz; camera MJPEG ~8 FPS; track thực ~10 FPS. Một lần polling trống KHÔNG được làm marker biến mất — nhấp nháy ID là lỗi nghiêm trọng (PLAN 1 Stage 10) |
| F2 | **Ba trải nghiệm, một nguồn dữ liệu** | Tài xế (chỉ xe của mình + dẫn đường), giám sát (toàn bộ + camera + operator action), kiosk (QR) — cùng snapshot, ba chế độ lọc khác nhau |
| F3 | **Bản đồ SVG một hệ tọa độ** | Slot polygon (world cm) → SVG px; marker xe, tuyến đường, nhãn, cổng — tất cả trong một phép biến đổi duy nhất, không trộn hệ tọa độ |
| F4 | **Session survival hiển thị** | Xe trong bãi → session phải hiển thị được 100% thời gian; xe đỗ (motion = 0, không observation) vẫn hiện đúng ID tại đúng slot |
| F5 | **Mobile-first trong môi trường hầm** | Tài xế dùng điện thoại khi lái: touch target ≥ 44×44 px, giọng nói tiếng Việt, map pan/zoom, header gọn — không được che map |

### 1.1. Bất biến hiển thị nền tảng

\[
\boxed{\text{Frontend KHÔNG BAO GIỜ tự suy luận danh tính} — \text{chỉ vẽ lại những gì backend quyết định}}
\]

Frontend được phép: lọc (xe nào hiển thị theo cờ backend), nội suy vị trí (trong giới hạn hold window), đổi trạng thái hiển thị.

Frontend KHÔNG được phép: gán ID, đổi ID, merge xe, quyết định xe nào "thật", cache snapshot cũ hơn `display_hold_seconds` để "giữ" một xe mà backend đã ẩn.

### 1.2. Trạng thái ĐÚNG / SAI

**Pass:**
```text
t=0.0s  snapshot: GID 17 observed     → marker đỏ + halo, vị trí P1
t=0.4s  snapshot: GID 17 missing      → marker giữ nguyên vị trí + trạng thái "tạm mất dấu"
t=1.2s  snapshot: GID 17 observed, P2  → marker chuyển động mượt tới P2, vẫn GID 17
t=5.0s  snapshot: GID 17 không có     → marker ẨN (đã quá display-hold, không parked)
```

**Fail (flickering):**
```text
t=0.4s  snapshot trống → marker biến mất
t=1.2s  snapshot có lại → marker xuất hiện với GID khác (mới vẽ từ stale cache sai logic)
```

---

## 2. Kiến trúc đích

```text
┌────────────────────────── Runtime API (backend, cổng 8001) ──────────────────────────┐
│  /api/runtime/snapshot   /api/runtime/status   /api/runtime/cameras/camN.mjpg        │
└───────────────┬──────────────────────────────────────────────────────┬──────────────┘
                │ polling ~1 Hz                                          │ MJPEG <img>
                ▼                                                        ▼
┌─────────────────────────── Frontend Application ─────────────────────────────────────┐
│                                                                                      │
│  Stage F1: Snapshot Ingestion & Validation (schema + staleness + backoff)            │
│           │                                                                          │
│           ▼                                                                          │
│  Stage F2: Snapshot Store (zustand —单一 nguồn state UI)                             │
│           │                                                                          │
│           ▼                                                                          │
│  Stage F3: World→SVG Projection (affine fit từ slot_layout, PLAN 5 §1)               │
│           │                                                                          │
│           ├──► Stage F4: Display-State Resolution (cờ backend + hold window)         │
│           │        │                                                                 │
│           │        ▼                                                                 │
│           │   Stage F6: SVG Map Rendering (slots / markers / routes / labels)        │
│           │                                                                          │
│           ├──► Stage F5: Session Resolution (driver view — QR deep-link)             │
│           │        ├──► Route & Guidance Engine (PLAN 5 §4: Dijkstra + voice)        │
│           │        └──► Voice Guidance (Web Speech API, tiếng Việt)                  │
│           │                                                                          │
│           └──► Stage F7: Camera Panels (MJPEG subscriber + health)                   │
│                                                                                      │
│  Stage F8: Interaction & Operator Actions (Reset ID confirm, gate config, filter)    │
│                                                                                      │
└────────────────────────── Session API (backend, cổng 8000) ──────────────────────────┘
     claim / select-spot / exit  ← chỉ driver view + operator dùng
```

**Nguyên tắc phân lớp:**
- **Unidirectional data flow**: Runtime API → snapshot store → derived selectors → render. Không bao giờ ngược lại.
- **Domain logic tách visual**: mọi phép tính (projection, route, display-state) là pure function trên dữ liệu typed — test được độc lập với React.
- **Một hệ tọa độ SVG duy nhất** cho slots, markers, routes, labels, cổng (PLAN 5 §1).
- **Giao thức video tách giao thức dữ liệu**: MJPEG là `<img>` stream; snapshot là JSON polling — hai kênh độc lập, sai lệch không phá nhau.

---

## 3. Sequential Execution Pipeline

Mỗi stage: Inputs → Operational Logic → Outputs → Pass/Fail.

---

### Stage F1: Snapshot Ingestion & Validation

**Inputs**
```text
- Runtime API base URL (env VITE_RUNTIME_URL, fallback qua proxy dev-server)
- AbortController cho mỗi request
- Cấu hình polling interval (1 s) + max retry backoff (5 s)
```

**Operational Logic**
1. Poll `GET /api/runtime/snapshot` theo chu kỳ, cache disabled (`no-store`).
2. **Coalesce request trùng lặp**: một request đang bay thì request mới không được tạo — trả về promise đang chờ (tránh bão request khi React re-render).
3. Gặp lỗi → exponential backoff: 1s → 2s → 4s → 5s (cap); thành công → reset chu kỳ.
4. **Validate schema trước khi commit vào store**: mọi bản ghi phải có `schema_version`, `frame_index` đơn điệu tăng, `timestamp`/`published_at` parse được, `vehicles[].global_id` là số nguyên, `parking_slots[].slot_id` duy nhất. Vi phạm → coi như lỗi mạng (bỏ snapshot, giữ state cũ, tăng retry).
5. Ghi `lastPublishedAt` — mốc đo staleness cho toàn app.
6. Trạng thái kết nối: `connecting | live | stale | error` (stale = `published_at` cũ hơn 5 s dù vẫn nhận JSON).

**Outputs**
```text
ValidatedRuntimeSnapshot:
    snapshot (typed) | connectionState | lastPublishedAt | fetchError
```

**Pass**
```text
snapshot.frame_index: 100 → 105 → 110 (đơn điệu);
một response thiếu global_id → bị loại, store giữ snapshot 105, connectionState=error.
```

**Fail**
```text
snapshot frame_index đi lùi (105 → 103) được commit → render dùng dữ liệu cũ hơn
hoặc response lỗi 500 làm store bị clear (map trắng giật).
```

---

### Stage F2: Snapshot Store (single source of UI truth)

**Inputs**: ValidatedRuntimeSnapshot từ F1.

**Operational Logic**
1. Store giữ đúng cấu trúc snapshot — **không transform, không suy luận danh tính** khi commit.
2. Derived state (danh sách xe hiển thị, xe của session, số ô trống/đầy) tính bằng **selector pure function**, không lưu vào store.
3. Store phân biệt ba khoá state độc lập: `trackingSource` (chế độ dữ liệu), `snapshot`, `connection`.
4. Operator "Reset ID" gọi `POST /api/runtime/reset-identities` (xác nhận qua dialog — không bấm nhầm) rồi **clear local snapshot về trạng thái rỗng**, để polling tự tái dựng (không cố "sửa" map cũ).

**Outputs**
```text
ParkingStore: { snapshot, connection, trackingSource }
Selector: selectVisibleVehicles, selectSlotById, selectCounts, selectSessionVehicle
```

**Pass**: sau Reset ID, map trống đúng 1-2 chu kỳ polling rồi xe xuất hiện lại từ GID 1 (theo backend).

**Fail**: sau Reset ID, marker cũ từ snapshot cached vẫn hiển thị (ghost tận mạng frontend).

---

### Stage F3: World→SVG Projection

**Inputs**
```text
- snapshot.slot_layout: [{slot_id, camera_id, polygon: [[x,y]×N] world-cm}]
- PARKING_GEOMETRY tĩnh: vị trí/ kích thước slot chuẩn trên canvas SVG (px)
```

**Operational Logic**
1. Ghép cặp: slot_id của slot_layout (world) ↔ slot_id của geometry chuẩn (SVG) — N cặp điểm tương ứng.
2. Fit **affine 2D tối thiểu bình phương** (PLAN 5 §1: $A^\*=\arg\min\|MA-B\|^2_F$) từ N≥10 cặp tâm slot.
3. Xây hai chiều: `project(world→svg)` cho marker/route; `unproject(svg→world)` cho click map (cấu hình cổng).
4. **Fallback khi fit không đủ cặp** (N<3 hoặc singular): identity mapping + cảnh báo — map vẫn render slot tĩnh, marker dùng world coords clamp vào canvas.
5. Chiếu mọi polygon slot lên SVG MỘT LẦN mỗi khi layout đổi (per camera layout giống nhau → cache).

**Outputs**
```text
worldToSvg: (worldPoint) => svgPoint
svgToWorld: (svgPoint) => worldPoint | null
projectedSlotPolygons: Map<slotId, svgPolygon>
```

**Pass**: tâm slot D08 world (100, 200)cm chiếu vào đúng tâm hình chữ nhật D08 trên SVG (sai số ≤ 2 px sau fit).

**Fail**: marker xe hiển thị lệch hẳn sang ô khác vì fit dùng sai chiều (world→SVG bị đảo SVG→world).

---

### Stage F4: Display-State Resolution (frontend lọc, không suy luận)

**Inputs**
```text
- snapshot.vehicles[]: mỗi xe có {global_id, state, observed, parked_slot_id,
  stale_seconds, display_hold_seconds, position}
```

**Operational Logic — bảng chân lý hiển thị (nguồn: PLAN 1 Stage 10):**

| Điều kiện backend | Hiển thị frontend | Visual |
|---|---|---|
| `observed = true` | HIỂN THỊ | marker đỏ + halo nhấp nháy (đang chạy) |
| `observed = false` ∧ `parked_slot_id ≠ null` | HIỂN THỊ | marker 🔒 xanh tại tâm slot, tĩnh |
| `observed = false` ∧ `stale_seconds ≤ display_hold_seconds` | HIỂN THỊ | marker mờ + nhãn "tạm mất dấu", giữ vị trí last-seen |
| `observed = false` ∧ `stale > hold` ∧ không slot | **ẨN** | (ghost — backend đã tính, frontend chỉ vâng) |

1. Frontend **chỉ áp bảng trên** — không tự đếm giây, không tự quyết "xe này chắc còn".
2. Driver view: thêm filter `global_id === sessionVehicleId` (xe của mình) + fallback vị trí: không observation ∧ có parked_slot → vẽ tại tâm slot polygon đó.
3. Đếm "Xe hiện tại" trên monitor = số xe HIỂN THỊ theo bảng trên (không đếm ghost).

**Outputs**
```text
DisplayVehicle[]: {trackId, x, y, state, parkedSlotId, visibleReason}
```

**Pass** (Scenario F-A, PLAN 6): xe đỗ D07 90 giây không observation → marker 🔒 xanh tại D07 toàn thời gian.

**Fail**: xe đỗ biến mất sau 2 giây không observation → tài xế mở app thấy map trống.

---

### Stage F5: Session Resolution & Driver Navigation

**Inputs**
```text
- URL ?session=<id> (QR deep-link)
- Session API: claim / get / select-spot / exit
- snapshot.vehicles (để tìm xe của session)
```

**Operational Logic**
1. Lần đầu mở `?session=` → tự động `POST claim` một lần duy nhất (ref-guard chống double-claim).
2. Poll `GET session` chu kỳ 500 ms: state machine session `WAITING → NAVIGATING → PARKED → EXIT_NAVIGATION → (deleted)`.
3. `session 404` → trạng thái "Phiên đã kết thúc" (xe đã qua exit-line thật) — KHÔNG phải lỗi.
4. Xe của phiên = `session.globalVehicleId`; nếu GID đó merge thành GID khác, backend remap trước khi frontend biết (session luôn trỏ đúng xe).
5. **Tuyến đường chỉ vẽ sau xác nhận của người dùng** (chọn slot × "Chỉ đường") hoặc đã có `targetSpotId` từ phiên cũ (khôi phục điều hướng).
6. Xe chạy nhưng đi sai tuyến > ngưỡng lệch (PLAN 5 §4.4) → cảnh báo thoại + banner đỏ — không âm thầm vẽ lại tuyến.
7. Session `PARKED` → thông báo "Đã đỗ thành công tại ô X" + chế độ "Lấy xe ra".

**Outputs**
```text
DriverFlowState: mode ∈ {entry, browse, recommendation, navigation, parked, ended}
selectedRoute: polyline SVG + bước chỉ dẫn thoại
```

**Pass**: tài xế vào từ QR, xe hiện đúng một xe duy nhất, chọn D08 → tuyến xanh từ vị trí xe tới D08, giọng nói "Phía trước đi thẳng...".

**Fail**: trang tài xế hiển thị cả xe của người khác (không lọc theo GID session).

---

### Stage F6: SVG Map Rendering

**Inputs**: projectedSlotPolygons + DisplayVehicle[] + selectedRoute + gateOverlay.

**Operational Logic**
1. Render theo lớp đúng thứ tự z: nền bãi → slot polygons (màu trạng thái) → route polyline → vehicle markers → nhãn → gate overlay.
2. Màu slot theo trạng thái backend: `empty` xanh / `occupied` đỏ — **đề xuất/ lựa chọn KHÔNG đổi màu fill**, chỉ outline xanh dương + pin (PLAN backend quy ước).
3. Marker xe: dịch chuyển bằng CSS transform `transition: transform 350ms linear` — mượt mà không re-render cả SVG tree.
4. **Teleport guard** (PLAN 5 §2): nhảy vị trí > ngưỡng → snap tức thì (bỏ transition), tránh marker bay xuyên bãi.
5. Pan/zoom: transform trên `<g>` chứa map, có nút reset-view; khóa pan khi đang chọn điểm cấu hình cổng.
6. Click slot → bottom-sheet chi tiết; click xe (monitor) → panel chọn xe.
7. Accessibility: mọi marker/spot là element có `role` + `aria-label` ("Xe Global ID 17", "Ô D08, trống"); màu đi kèm nhãn — không phụ thuộc màu đơn thuần.

**Outputs**: Virtual SVG DOM.

**Pass**: 160 slot render đúng thứ tự zone; chọn xe → aria "Xe Global ID 17" đọc được bởi screen reader.

**Fail**: transition làm marker bay từ cổng vào tới ô F khi GID tái xuất hiện ở vị trí xa (không teleport guard).

---

### Stage F7: Camera Panels (MJPEG)

**Inputs**: `/api/runtime/cameras/cam1.mjpg`, `cam2.mjpg`.

**Operational Logic**
1. `<img src={mjpegUrl}>` — MJPEG multipart tự play, không JS decode.
2. **Health theo snapshot**: `cameras.camN.online` + staleness tổng — không infer từ việc img onload.
3. Offline → placeholder "Mất tín hiệu" (không giữ frame cũ tô mờ giả sống).
4. Chỉ monitor hiển thị camera; driver view KHÔNG stream (tiết kiệm băng thông + tập trung dẫn đường).

**Outputs**: hai panel `<section aria-label="Video trực tuyến camN">`.

**Pass**: ngắt backend → panel chuyển placeholder trong ≤ 2 giây; backend sống lại → stream tự tiếp tục.

**Fail**: panel treo frame cũ 30 giây không báo trạng thái.

---

### Stage F8: Interaction & Operator Actions

**Inputs**: user events + API endpoints.

**Operational Logic**
1. **Reset ID** (monitor): nút → confirm dialog (mô tả hậu quả: xóa toàn bộ Global ID; checkbox "xóa cả phiên QR" → include_sessions) → POST reset → hiển thị kết quả số ID đã reset. Pending state disable nút; 409 (đang reset khác) hiện cảnh báo.
2. **Cấu hình cổng** (monitor): chế độ vẽ — click 6 điểm (2 đầu vạch × 2 cổng + 2 điểm hướng), unproject SVG→world, POST gate config; pan/zoom khóa trong lúc chọn.
3. **Bộ lọc browse** (driver): "Chỉ hiện ô trống" / "Tất cả" — chỉ ảnh hưởng render slot, không đổi dữ liệu.
4. Mọi hành động nguy hiểm (reset) đều qua confirmation; mọi hành động POST đều có pending + error state.

**Outputs**: gọi API + cập nhật store/hiển thị.

**Pass**: bấm Reset ID khi tracking đang chạy → dialog hiện, hủy không gọi API; xác nhận → map sạch, GID bắt đầu từ 1.

**Fail**: double-click Reset tạo 2 request → 409 hiện lỗi khó hiểu thay vì disable nút khi pending.

---

## 4. Cấu trúc dự án chuẩn

```text
frontend/
├── src/
│   ├── api/            # runtime client (snapshot/status/reset), session client
│   ├── domain/         # types: RuntimeSnapshot, Vehicle, Slot, Session (typed, strict)
│   ├── projection/     # worldToSvg / svgToWorld / affine fit (pure functions)
│   ├── display/        # display-state resolution table (pure)
│   ├── routing/        # lane graph, Dijkstra, instruction generation (pure)
│   ├── guidance/       # voice synthesis, off-route detector
│   ├── stores/         # snapshot store, driver-flow store
│   ├── components/     # ParkingMap, CameraPanel, RouteLayer, VehicleMarker,
│   │                   # BottomSheets, ResetIdDialog, GateEditor, Header...
│   ├── app/            # main.tsx router: / | /monitor | /kiosk/entry
│   └── styles/
└── tests/              # unit (pure modules) + component (RTL) + e2e (Playwright)
```

**Quy ước bắt buộc:**
- TypeScript `strict: true`, **cấm `any`**.
- Mọi module tính toán (projection/display/routing) là pure function — phải test được không cần DOM.
- Component chỉ consume store/selector — không fetch trực tiếp trong component.
- Dev proxy: `/api/runtime → :8001`, `/api → :8000` (tách rõ hai backend).

---

## 5. Phase-Gated Implementation Roadmap

### Phase FE-0: Contract & Fixture

**Work**: định nghĩa TypeScript types từ JSON schema backend; fixture snapshot mẫu (live/replay/ghost/parked/reset); cấu trúc test vitest + Playwright.
**Entry**: PLAN 4+5+6 được duyệt; backend contract JSON có ví dụ thực.
**Exit (Pass)**:
```text
✓ Types strict biên dịch, không any
✓ Fixture 5 snapshot (normal, ghost, parked, stale, post-reset) load được vào test
✓ Vitest + Playwright chạy skeleton
```
**Fail**: types dùng `any`/`unknown` escape; fixture chỉ có trường hợp happy-path.

---

### Phase FE-1: Ingestion & Store

**Work**: Stage F1 + F2 (coalesce, backoff, schema validation, connection state).
**Exit (Pass)**:
```text
✓ Response lỗi/không hợp lệ KHÔNG clear store
✓ frame_index đơn điệu được kiểm tra (hạ cấp = bỏ snapshot)
✓ 100 request/s không xảy ra khi component re-render (coalesce chứng minh bằng test)
```
**Fail**: store bị null khi 1 request lỗi → map trắng.

---

### Phase FE-2: Projection & Display-State

**Work**: Stage F3 + F4 (affine fit hai chiều + bảng hiển thị + teleport guard).
**Exit (Pass)**:
```text
✓ Chiếu tâm slot khớp geometry chuẩn (≤ 2 px)
✓ Bảng display-state: 4 hàng của bảng chân lý có unit test riêng từng hàng
✓ Ghost ẩn, parked giữ, hold giữ — theo cờ backend, KHÔNG có timer tự chế
```
**Fail**: frontend tự đếm giây để ẩn ghost (trái nguyên tắc consumer).

---

### Phase FE-3: Map Rendering & Camera Panels

**Work**: Stage F6 + F7 (SVG layers, marker transition + teleport guard, MJPEG panel + health).
**Exit (Pass)**:
```text
✓ 160 slot render < 300 ms (perf test)
✓ Marker di chuyển mượt qua 5 snapshot liên tiếp; teleport snap khi nhảy xa
✓ Camera offline → placeholder ≤ 2 s
✓ aria-label đầy đủ cho slot + marker
```
**Fail**: re-render toàn SVG mỗi snapshot (không transform transition) → giật trên điện thoại.

---

### Phase FE-4: Driver Navigation & Session

**Work**: Stage F5 (claim, poll session, route sau xác nhận, voice, off-route warning, parked flow, exit flow).
**Exit (Pass)**:
```text
✓ Deep-link ?session= tự claim đúng MỘT lần
✓ Trang tài xế chỉ hiển thị xe của session (+ fallback tâm slot khi đỗ)
✓ Tuyến chỉ vẽ sau confirm; off-route phát giọng nói + banner
✓ Session 404 → màn "Phiên kết thúc" đúng ngữ nghĩa
```
**Fail**: route vẽ ngay khi chọn slot chưa confirm (vi phạm xác minh tường minh).

---

### Phase FE-5: Monitor, Kiosk & Operator

**Work**: Stage F8 + trang kiosk QR + gate editor + bộ đếm/thống kê monitor + e2e toàn luồng.
**Exit (Pass)**:
```text
✓ Monitor: đủ xe hiển thị + 2 camera + event trace + Reset ID (có confirm, có kết quả)
✓ Gate editor: 6 điểm → world coords → lưu được; pan khóa khi chọn
✓ Kiosk: liệt kê phiên waiting, QR link ?session=
✓ E2E Playwright pass 6 kịch bản PLAN 6 §2
```
**Fail**: Reset ID không qua confirm hoặc không hiển thị số ID đã reset.

---

## 6. Tóm tắt nguyên tắc thiết kế frontend

1. **Consumer thuần túy** — mọi quyết định danh tính/hiển thị nằm ở cờ backend; frontend chỉ áp bảng.
2. **Hold-window là thuộc tính dữ liệu** — `stale_seconds` + `display_hold_seconds` từ snapshot, không phải timer tự chế.
3. **Một phép chiếu duy nhất** — affine world→SVG dùng cho mọi layer.
4. **Pure-function core** — projection/display/routing test được không DOM; component chỉ lắp ráp.
5. **Xe đỗ không bao giờ biến mất** — hiển thị qua slot ownership, độc lập với observation.
6. **Xác nhận tường minh trước hành động nguy hiểm** — reset ID, vẽ route.
