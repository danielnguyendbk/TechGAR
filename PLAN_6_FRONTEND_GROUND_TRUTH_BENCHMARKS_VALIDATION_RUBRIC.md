# TECHGAR — PLAN 6: FRONTEND GROUND TRUTH BENCHMARKS, VALIDATION CRITERIA & METRIC SCORING RUBRIC

> **Loại tài liệu**: Chuẩn kiểm định frontend, kịch bản số cụ thể, bộ chỉ số, khung ablation, rubric 100 điểm
> **Liên kết**: PLAN 4 (workflow frontend) · PLAN 5 (toán frontend) · PLAN 3 (benchmark backend — nguồn snapshot)
> **Nguyên tắc vàng**: fixture là **snapshot JSON cố định theo thời gian** — mọi kỳ vọng hiển thị suy ra được từ fixture bằng bảng chân lý PLAN 4 §F4, không phụ thuộc cảm tính

---

## 1. Chuẩn Fixture Ground Truth

Mỗi fixture là chuỗi snapshot tuần tự, mỗi snapshot mô tả:

```text
frame_index, published_at
vehicles[]: { global_id, state, observed, parked_slot_id,
              stale_seconds, display_hold_seconds, position }
parking_slots[]: { slot_id, occupied, status }
cameras: { cam1: online, cam2: online }
```

### Bộ fixture chuẩn (bắt buộc có đủ)

| Fixture | Mô tả | Mục đích |
|---|---|---|
| `normal` | 2 xe observed, di chuyển đều qua 10 snapshot | baseline render |
| `flicker-gap` | 1 xe: observed→missing→observed (hold đập liên tục 10 lần) | anti-flicker |
| `ghost` | 1 xe stale 40 s, không slot | ghost ẩn |
| `parked-long` | 1 xe parked 90 s không observation | parked persist |
| `post-reset` | snapshot rỗng → xe từ GID 1 | reset flow |
| `offline` | 0 snapshot (network fail × 10) | connection state |

---

## 2. Kịch Bản Benchmark Số Cụ Thể

### Scenario F-A: Xe đỗ tồn tại không observation

**Ground truth (fixture `parked-long`)**
```text
Snapshot 1..90 (mỗi 1 s):
  GID 9, state=parked, observed=false,
  parked_slot_id="D07", stale_seconds=0→89
```

**PASS**
```text
Snapshot 1 → 90: marker 🔒 xanh hiển thị liên tục tại tâm slot D07
N_flip = 0 (không lần nào biến mất)
Bộ đếm monitor "xe hiện tại" ≥ 1 toàn thời gian
```

**FAIL**
```text
Snapshot 3: marker biến mất (đã bỏ qua parked_slot_id)
→ tài xế mở app thấy map trống dù xe đang đỗ thật
```

---

### Scenario F-B: Hold window qua mất dấu ngắn

**Ground truth (fixture `flicker-gap`)**
```text
t=0 s:  GID 17 observed, w=(100, 100)
t=1 s:  GID 17 observed=false, stale=0.9, hold=2.5
t=2 s:  GID 17 observed, w=(110, 100)
t=3 s:  GID 17 observed=false, stale=0.8
t=4 s:  GID 17 observed, w=(120, 100)
... (lặp 5 chu kỳ)
```

**PASS**
```text
Marker KHÔNG biến mất lần nào (stale ≤ hold toàn bộ)
Trạng thái visual đổi moving ↔ missing (mờ + nhãn), không đổi ID
N_flip(moving↔hidden) = 0
N_style_change được phép (đây là theo thiết kế)
Vị trí marker chuyển động mượt 100→110→120
```

**FAIL**
```text
t=1 s: marker ẩn rồi t=2 s hiện lại → N_flip = 5 chu kỳ = flickering
```

---

### Scenario F-C: Ghost bị ẩn đúng

**Ground truth (fixture `ghost`)**
```text
t=0 s:  GID 3, observed=false, parked=null, stale=1.0, hold=2.5
t=3 s:  GID 3, observed=false, parked=null, stale=4.0
t=10 s: GID 3, observed=false, parked=null, stale=11.0
```

**PASS**
```text
t=0:  hiển thị (missing style)
t=3:  ẨN (stale > hold, không slot)
t=10: vẫn ẨN
N_flip ≤ 1 trong cả chuỗi
Bộ đếm xe = 0 từ t=3
```

**FAIL**
```text
t=10: marker vẫn đứng tại vị trí cũ (ghost tận mạng frontend)
hoặc t=0 đã ẩn (hold bị bỏ — marker nhấp nháy ở ranh giới)
```

---

### Scenario F-D: Trang tài xế — chỉ xe của mình

**Ground truth**
```text
Snapshot: 3 xe hiển thị GID {5, 17, 22}
Session: sessionId="S42", globalVehicleId=17, state=NAVIGATING, targetSpotId="D08"
```

**PASS**
```text
Trang /?session=S42 render đúng 1 marker: GID 17
GID 5, GID 22 KHÔNG xuất hiện (kể cả trong layer ẩn — không render)
Route polyline vẽ từ vị trí GID 17 → slot D08 theo lane graph
QR kiosk widget KHÔNG xuất hiện trên trang cá nhân
```

**FAIL**
```text
Map tài xế hiện 3 xe (filter GID bị bỏ)
hoặc route vẽ từ vị trí GID 5 (lấy nhầm xe)
```

---

### Scenario F-E: Fallback vị trí khi xe đỗ không observation

**Ground truth**
```text
Session S42: parkedSpotId="B04", globalVehicleId=17
Snapshot: KHÔNG có vehicle 17 (không observation),
slot B04 polygon tâm world (50, 120)
```

**PASS**
```text
Marker GID 17 hiển thị tại tâm slot B04 (chiếu SVG từ polygon)
Thông báo "Đã đỗ tại ô B04" + nút "Lấy xe ra"
```

**FAIL**
```text
Trang trắng / "mất kết nối" (dùng observation thay vì fallback slot)
hoặc marker đứng tại last-seen cũ cách slot 3 mét
```

---

### Scenario F-F: Route chỉ vẽ sau xác nhận

**Ground truth (tương tác)**
```text
Bước 1: user chọn slot D08 (tap)
Bước 2: bottom-sheet hiện nút "Chọn D08 và chỉ đường"
Bước 3: user ĐÓNG sheet chưa bấm
Bước 4: user mở lại sheet, bấm nút
```

**PASS**
```text
Sau bước 3: KHÔNG có route polyline (chỉ outline xanh chọn slot)
Sau bước 4: route polyline xuất hiện + voice bắt đầu
```

**FAIL**
```text
Route vẽ ngay bước 1 (chọn slot đã tự vẽ đường)
→ vi phạm "xác nhận tường minh trước khi render route" (PLAN 4 F5 logic 5)
```

---

### Scenario F-G: Off-route warning

**Ground truth**
```text
Session đang navigate → D08, route P* qua (100,100)→(100,150)
Snapshot t: xe session observed tại w=(140, 100)
d_off = 40 cm (vượt ngưỡng), observed=true
```

**PASS**
```text
Banner đỏ "ĐANG ĐI SAI TUYẾN" + voice cảnh báo phát đúng 1 lần
Route polyline KHÔNG tự vẽ lại (người lái quyết định)
```

**FAIL**
```text
Tuyến tự vẽ lại từ vị trí mới (silent redirect — cấm theo PLAN 4 F5 logic 6)
hoặc warning không phát vì d_route tính trên vị trí hold cũ (missing state)
```

---

### Scenario F-H: Reset ID end-to-end

**Ground truth (tương tác + API)**
```text
Map đang hiển thị GID {3, 9, 17}
User bấm "Reset ID" → dialog hiện → bấm "Hủy" → bấm lại → "Xác nhận"
API POST /api/runtime/reset-identities → { reset: true, retired_identities: 3 }
Snapshot kế tiếp: rỗng → rồi GID 1 xuất hiện
```

**PASS**
```text
Lần hủy: KHÔNG có request POST nào (network spy = 0 call)
Lần xác nhận: đúng 1 POST; nút disable khi pending
Kết quả "Đã reset 3 Global ID" hiển thị
Marker cũ biến mất ngay khi snapshot rỗng về; GID mới render từ 1
```

**FAIL**
```text
Double-click tạo 2 POST → 409 hiện lỗi thô
hoặc marker GID 3 cũ vẫn hiện sau reset (cache stale)
```

---

### Scenario F-I: Kết nối backend đứt — state không trắng

**Ground truth (fixture `offline`)**
```text
10 lần poll liên tiếp fail (network error)
Trước đó app đang hiển thị snapshot ổn định
```

**PASS**
```text
Map GIỮ NGUYÊN dữ liệu cuối (không clear store)
Banner lỗi "Mất kết nối Runtime API" hiện
Backoff tăng: poll cách 1s→2s→4s→5s
Backend sống lại → state live, dữ liệu mới commit
```

**FAIL**
```text
Store bị clear → map trắng (vi phạm PLAN 4 F1 logic "lỗi không clear store")
hoặc app retry 10 Hz (spam request khi lỗi)
```

---

## 3. Bộ Chỉ Số Chuẩn Hóa

### 3.1. Marker persistence (chống nhấp nháy)

Trên chuỗi $K$ snapshot, với xe $v$ được phép hiển thị theo bảng chân lý:

\[
Persistence(v)
=
\frac{
\left|\{k:\;marker(v)\text{ hiển thị đúng}\}\right|
}{
K
}
\]

**Target:**
```text
Persistence = 1.00 cho mọi xe "nên hiển thị" (normal + parked + hold)
N_flip ≤ 1 mỗi xe mỗi cửa sổ 3 s
```

### 3.2. Flicker count

\[
Flicker
=
\sum_v
\mathbf{1}
\left[
N_{flip}(v,W_{flicker})>1
\right]
\]

**Target:** `Flicker = 0` trên toàn fixture chuẩn.

### 3.3. Projection accuracy

\[
\varepsilon_{proj}
=
\max_k
\left\|
\hat{\mathbf{s}}_k-\mathbf{s}_k
\right\|
\]

(tâm slot chiếu so geometry chuẩn, sau fit $A^\*$).

**Target:** `ε_proj ≤ 2 px` (khớp chuẩn $r_{fit}$ PLAN 5 §1.5).

### 3.4. Render performance

\[
T_{render}
=
\text{thời gian commit React từ snapshot mới → DOM cập nhật}
\]

**Target:**
```text
160 slot + 10 xe: T_render ≤ 100 ms (p95)
Map pan/zoom: 60 FPS trên desktop, ≥ 30 FPS mobile (transform layer)
Bundle ≤ 350 KB gzip
```

### 3.5. Update latency hiển thị

\[
L_{display}
=
t_{marker\_vị\_trí\_mới\_commit}
-
t_{snapshot\_published}
\]

**Target:** `median ≤ 150 ms` (phần frontend — sau khi snapshot tới tay).

### 3.6. Route validity

\[
Validity_{route}
=
\mathbf{1}
\left[
P^\*\text{ không cắt slot polygon nào}
\;\land\;
P^\*[0]=\text{gần vị trí xe}
\;\land\;
P^\*[-1]=\text{đầu slot đích}
\right]
\]

**Target:** `= 1` cho mọi cặp (vị trí, slot) trong bộ test lane graph.

### 3.7. Accessibility

**Target:**
```text
100% marker/slot có aria-label mô tả (id + trạng thái)
100% nút ≥ 44×44 px (mobile audit)
Màu trạng thái kèm nhãn/icon (không chỉ màu — WCAG contrast ≥ 4.5:1)
```

### 3.8. Session correctness

**Target:**
```text
Claim idempotent: 1 claim / phiên dù StrictMode double-invoke
Session 404 → trang kết thúc (không crash, không loop)
Trang tài xế KHÔNG bao giờ render xe GID ≠ session.gid
```

---

## 4. Khung Ablation Study (3 thí nghiệm)

Cùng fixture + cùng máy cho mọi thí nghiệm.

### Ablation F-1: Bỏ Display-Hold (marker theo observed thuần)

**Tắt**: bảng chân lý hàng hold (render chỉ khi `observed=true`).

**Cơ chế suy hao** (PLAN 5 §3.2 bị vi phạm — frontend tự quyết thay backend):
- Mỗi snapshot missing → marker biến mất
- Fixture `flicker-gap`: 5 lần biến mất / 10 s

**Kỳ vọng định lượng:**
```text
Persistence: 1.00 → ~0.5
Flicker: 0 → ≥ 5
```

### Ablation F-2: Bỏ Teleport Guard (luôn animate)

**Tắt**: điều kiện snap (PLAN 5 §2.2).

**Cơ chế suy hao**: xe re-acquire ở xa → marker animate bay xuyên map 350 ms.

**Kỳ vọng định lượng:**
```text
Số lần marker di chuyển > v_max^display: 0 → bằng số re-acquire
Visual glitch score (đánh giá tay): tăng rõ rệt
```

### Ablation F-3: Bỏ Schema Validation (commit mọi JSON)

**Tắt**: kiểm tra frame_index/field bắt buộc (PLAN 4 F1 logic 4).

**Cơ chế suy hao**: response lỗi/truncated → store rác → render crash hoặc marker ID undefined.

**Kỳ vọng định lượng:**
```text
Fixture injection 5 JSON hỏng: 5/5 gây lỗi hiển thị
(cấu hình đầy đủ: 0/5)
```

### Quy tắc chấp nhận ablation

Cấu hình đầy đủ phải thắng cả 3 ablation trên: `Persistence`, `Flicker`, `ε_proj`, `T_render`, crash count. Không có penalty đo được → module redundant hoặc test chưa đủ.

---

## 5. Rubric Chấp Nhận 100 Điểm

### A. Data Discipline — 15 điểm

| Tiêu chí | Điểm |
|---|---:|
| Coalesce + backoff đúng (1 in-flight, cap 5 s) | 3 |
| Schema validation trước commit (hỏng = giữ state cũ) | 4 |
| frame_index đơn điệu kiểm tra | 2 |
| Lỗi mạng KHÔNG clear store | 3 |
| Không suy luận danh tính ở frontend (chỉ áp cờ backend) | 3 |

**Pass:** `≥ 14/15`

---

### B. Display Correctness — 25 điểm

| Tiêu chí | Điểm |
|---|---:|
| Bảng chân lý 4 hàng implement đúng (unit test từng hàng) | 8 |
| Parked hiển thị qua slot ownership, không cần observation | 5 |
| Hold window theo stale_seconds backend (không timer tự chế) | 4 |
| Ghost ẩn đúng ngưỡng, không sớm không muộn | 4 |
| Flicker = 0 trên fixture chuẩn | 4 |

**Pass:** `≥ 23/25` VÀ `Flicker = 0` (bắt buộc).

**Automatic FAIL:** marker xe đỗ biến mất khi motion = 0 (Scenario F-A fail).

---

### C. Projection & Rendering — 20 điểm

| Tiêu chí | Điểm |
|---|---:|
| Affine fit least-squares hai chiều, ε_proj ≤ 2 px | 5 |
| Một hệ tọa độ duy nhất cho mọi layer (slot/marker/route/gate) | 4 |
| Teleport guard (snap khi nhảy xa) | 3 |
| Marker transition mượt (transform, không re-render tree) | 3 |
| 160 slot render ≤ 100 ms p95 | 3 |
| Camera panel health + placeholder offline | 2 |

**Pass:** `≥ 18/20`

---

### D. Driver Experience — 25 điểm

| Tiêu chí | Điểm |
|---|---:|
| Deep-link ?session= claim idempotent đúng 1 lần | 3 |
| Trang tài xế chỉ hiện xe của session (filter GID) | 4 |
| Fallback vị trí: tâm slot khi đỗ / slot đích khi mất dấu | 4 |
| Route chỉ vẽ sau xác nhận tường minh | 4 |
| Off-route: warning + KHÔNG silent redirect | 4 |
| Voice guidance tiếng Việt trigger theo vị trí, không lặp | 3 |
| Session 404 → trang kết thúc đúng ngữ nghĩa | 3 |

**Pass:** `≥ 23/25`

**Automatic FAIL:** route vẽ ngay khi chọn slot chưa xác nhận (F-F); trang tài xế hiện xe người khác (F-D).

---

### E. Monitor, Kiosk & Operator — 10 điểm

| Tiêu chí | Điểm |
|---|---:|
| Reset ID: confirm + pending + kết quả hiển thị + map sạch | 4 |
| Gate editor: 6 điểm SVG→world, pan khóa khi chọn | 2 |
| Kiosk: danh sách phiên chờ + QR deep-link | 2 |
| Event trace + thống kê monitor (đếm theo visible) | 2 |

**Pass:** `≥ 9/10`

---

### F. Robustness & A11y — 5 điểm

| Tiêu chí | Điểm |
|---|---:|
| Offline fixture: map giữ + banner + backoff (F-I) | 2 |
| aria-label đầy đủ + màu kèm nhãn | 1 |
| Touch target ≥ 44 px | 1 |
| Mobile e2e pass (Playwright viewport điện thoại) | 1 |

**Pass:** `≥ 4/5`

---

## 6. Điều Kiện Chấp Nhận Cuối Cùng

Frontend production được chấp nhận CHỈ KHI:

```text
Overall score                       ≥ 90/100
Flicker                             = 0 (fixture chuẩn)
Persistence (xe nên hiển thị)       = 1.00
ε_proj                              ≤ 2 px
Trang tài xế chỉ hiện xe session    = luôn đúng
Route chỉ sau xác nhận              = luôn đúng
Reset ID có confirm + idempotent    = đúng
Offline không clear map             = đúng (F-I)
T_render p95 (160 slot)             ≤ 100 ms
Bundle gzip                         ≤ 350 KB
E2E Playwright 9 scenario (F-A..F-I)= pass toàn bộ
```

### Điều kiện TỪ CHỐI tuyệt đối (một điều = reject)

```text
✗ Marker xe đỗ biến mất khi không có observation (F-A)
✗ Marker nhấp nháy theo snapshot missing (F-B, Flicker > 0)
✗ Ghost đứng vĩnh viễn trên map (F-C)
✗ Trang tài xế hiển thị xe của người khác (F-D)
✗ Route vẽ trước khi người dùng xác nhận (F-F)
✗ Off-route tự redirect âm thầm (F-G)
✗ Reset ID double-fire hoặc không qua confirm (F-H)
✗ Mất mạng làm map trắng (F-I)
✗ Frontend tự gán/đổi Global ID bất kỳ nơi nào (vi phạm consumer thuần túy)
```

---

## 7. Nguyên tắc kết

Frontend đúng chỉ khi **mọi biến động của dữ liệu — mất dấu, đỗ xe, ghost, reset, mất mạng — đều thay đổi TRẠNG THÁI HIỂN THỊ theo bảng chân lý backend, nhưng không bao giờ làm người dùng thấy xe biến mất, đổi ID, hoặc tuyến tự ý thay đổi.**
