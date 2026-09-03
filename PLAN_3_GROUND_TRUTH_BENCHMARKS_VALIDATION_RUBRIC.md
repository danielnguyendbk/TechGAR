# TECHGAR — PLAN 3: GROUND TRUTH BENCHMARKS, VALIDATION CRITERIA & METRIC SCORING RUBRIC

> **Loại tài liệu**: Chuẩn kiểm định, kịch bản số cụ thể, bộ chỉ số MOT/Parking, khung ablation, rubric 100 điểm
> **Liên kết**: PLAN 1 (pipeline) · PLAN 2 (toán học)
> **Nguyên tắc vàng**: evaluator KHÔNG BAO GIỜ dùng Global ID dự đoán làm ground truth

---

## 1. Chuẩn Ground Truth Annotation

Mọi bản ghi validation phải chứa annotation đồng bộ:

```text
frame index
timestamp
physical vehicle ID
camera ID
bounding box hoặc footprint
world anchor
vehicle phase
slot ID (nếu đỗ)
handoff source camera
handoff target camera
visibility state
occlusion state
```

Nhãn xe vật lý ổn định suốt một bản ghi:

```text
Physical vehicle: P01, P02, P03, ...
```

### Nhãn trạng thái ground truth

```text
VISIBLE
OCCLUDED
MERGED_WITH_OTHER_VEHICLE
OUTSIDE_CAMERA
IN_HANDOFF_ZONE
PARKING_APPROACH
PARKED
DEPARTING
EXITED
```

---

## 2. Kịch bản Benchmark Số Cụ Thể

### Scenario A: Chuyển động thường trong một camera

**Ground truth**
```text
Vehicle: P01, Camera: C1
Frame 100: world (10.0, 20.0)
Frame 101: world (10.5, 20.0)
Frame 102: world (11.0, 20.0)
```

Vận tốc kỳ vọng: $\mathbf{v}=(5.0,\;0.0)$ world-units/s.

**PASS**
```text
Frame 100: GID 17
Frame 101: GID 17
Frame 102: GID 17
+ quỹ đạo chiếu nằm trong dung sai world error cấu hình
```

**FAIL**
```text
Frame 100: GID 17
Frame 101: không identity
Frame 102: GID 18
```
→ ID loss + ID switch.

---

### Scenario B: Gap detection 1 frame

**Ground truth**
```text
P01
Frame 200: world (20.0, 30.0)
Frame 201: occluded
Frame 202: world (21.0, 30.0)
```

**PASS**
```text
Frame 200: GID 17
Frame 201: GID 17, state = OCCLUDED / TEMPORARILY_MISSING
Frame 202: GID 17
```

**FAIL**
```text
Frame 200: GID 17
Frame 201: GID 17 retired
Frame 202: GID 23
```

---

### Scenario C: Camera handoff (chuyển camera)

**Ground truth**
```text
Vehicle P01
Frame 300, C1: world (40.0, 15.0)
Frame 301, C1: world (42.0, 15.0)  — trong C1 exit polygon
Frame 302: không camera nào quan sát
Frame 303, C2: world (44.0, 15.0)  — trong C2 entry polygon
Frame 304, C2: world (46.0, 15.0)
```

**PASS**
```text
Frame 300–304: toàn bộ GID 17
(handoff latency có thể biểu diễn bằng uncertainty state,
 nhưng identity KHÔNG đổi)
```

**FAIL**
```text
Frame 300–302: GID 17
Frame 303–304: GID 22
```
→ handoff ID switch rõ ràng.

**Fail topology cụ thể**
```text
C1 exit polygon: x ∈ [40,45], y ∈ [10,20]
C2 entry polygon: x ∈ [44,49], y ∈ [10,20]
Điểm C2 quan sát: (80, 80)
```
→ KHÔNG match identity C1 — điểm ngoài entry region đích.

---

### Scenario D: Hai xe giao nhau (crossing)

**Ground truth**
```text
P01: F400 (30,20) → F401 (32,20) → F402 (34,20)
P02: F400 (34,20) → F401 (32,20) → F402 (30,20)
```
Hai xe trùng vùng ở frame 401.

**PASS**
```text
P01: GID 17 suốt frames 400–402
P02: GID 18 suốt frames 400–402
(state occlusion/ambiguity chấp nhận được ở F401
 nếu cả hai identity được giữ)
```

**FAIL**
```text
P01: GID 17 → GID 18
P02: GID 18 → GID 17
```
→ hoán đổi identity hai vật.

---

### Scenario E: Detection hợp nhất (merged)

**Ground truth**
```text
Frame 500: P01 + P02 tách biệt
Frame 501: 1 merged detection phủ cả hai
Frame 502: 1 merged detection
Frame 503: 2 detection tách trở lại
```

**PASS**
- 2 latent identity track sống trong frames 501–502
- KHÔNG appearance gallery nào update từ vùng merged
- Frame 503: P01, P02 về đúng ID gốc

**FAIL**
- 1 identity bị xóa trong giai đoạn merge
- 1 Global ID gán cho cả 2 xe
- Global ID thứ 3 xuất hiện ngay sau khi tách

---

### Scenario F: Displacement lớn do lag

**Ground truth**
```text
Frame 600 timestamp 200.000s: vị trí (50, 50)
Frame 601 timestamp 200.500s: vị trí (90, 50)
```
Xe đi 40 world-units trong 500 ms.

**PASS**
- Covariance prediction mở rộng
- Quan sát nằm trong physical speed bound
- Global ID giữ nguyên

**FAIL**
- Hệ thống coi displacement là teleport bất khả thi (dù model tốc độ cho phép)
- Identity cũ retire → identity mới mint ở frame 601

---

### Scenario G: Xe vào slot

**Ground truth**
```text
Slot D08 tâm: (100, 200)
Vehicle P01:
Frame 700: footprint coverage 0.20
Frame 701: coverage 0.55
Frame 702: coverage 0.78
Frame 703: coverage 0.84
Frame 704: coverage 0.83
Frame 705: coverage 0.84
```

**PASS**
- D08 thành occupied bởi Global ID hiện có của P01
- Global ID gốc giữ nguyên
- Confirm chỉ sau temporal window thỏa

**FAIL**
- D08 occupied ngay frame 700
- D08 assigned Global ID khác
- Ownership D08 dao động vì centroid jitter

---

### Scenario H: Xe đi ngang giữa hai slot kề

**Ground truth**
```text
Frame 800: overlap D05 = 0.42
Frame 801: overlap D05 = 0.45, D06 = 0.43
Frame 802: overlap D06 = 0.47
Frame 803: rời cả hai slot
```

**PASS**
- KHÔNG slot nào occupied vĩnh viễn
- Chỉ ghi nhận transit evidence

**FAIL**
- D05 hoặc D06 bị reserve vĩnh viễn
- Ownership giao động D05↔D06, kết thúc bằng trạng thái parking giả

---

### Scenario I: Xe rời slot đã đỗ

**Ground truth**
```text
Slot B04 owned by GID 17
Frame 900–920: xe đứng yên trong B04
Frame 921–925: xe di chuyển ra ngoài
Frame 926: xe ngoài B04
```

**PASS**
- B04 giữ ownership GID 17 đến khi departure evidence confirm
- Slot release khi xe đã rời rõ ràng
- GID 17 vẫn là identity của xe đang rời

**FAIL**
- B04 thành empty sau 1 false-negative frame
- Xe khác kế thừa GID 17
- GID 17 bị xóa trước khi xe qua exit topology

---

## 3. Bộ Chỉ Số Chuẩn Hóa (MOT & Parking)

### 3.1. Identity Switches

Với xe ground-truth $P_i$:

\[
IDSW
=
\sum_t
\mathbf 1
\left(
\widehat{g}_i(t)
\ne
\widehat{g}_i(t-1)
\;\land\;
P_i\text{ vẫn hiện diện}
\right)
\]

**Target:**
```text
IDSW = 0 cho chuyển động thường
IDSW = 0 qua handoff camera hợp lệ
IDSW = 0 qua occlusion ngắn
```

### 3.2. Global ID Fragmentation

Với mỗi xe vật lý:

\[
F_i
=
\left|
\left\{
\widehat g:\;
\widehat g\text{ từng gán cho }P_i
\right\}
\right|
\]

**Target:**
\[
F_i = 1
\quad\text{(hoạt động)},\qquad
F_i \le 1.05 \;\text{(khẩn cấp, bản ghi khó)}
\]

### 3.3. IDF1

\[
IDF1
=
\frac{2\,IDTP}
{2\,IDTP+IDFP+IDFN}
\]

**Target:**
```text
IDF1 ≥ 0.95  — cảnh thường
IDF1 ≥ 0.90  — cảnh occlusion nặng
IDF1 ≥ 0.95  — qua camera handoff
```

### 3.4. MOTA

\[
MOTA
=
1-
\frac{FN+FP+IDSW}{GT}
\]

**Target:**
```text
MOTA ≥ 0.95 — chuyển động thường
MOTA ≥ 0.90 — stress tắc nghẽn + occlusion
```

### 3.5. FP / FN phân tầng

\[
Precision
=
\frac{TP}{TP+FP}
\qquad
Recall
=
\frac{TP}{TP+FN}
\]

Báo cáo TÁCH BIEETT theo: xe visible thường · occlusion · seam · xe đỗ · nhiễu môi trường (mỗi tầng có ngưỡng riêng — FP ở seam do parallax là lỗi khác với FP do nhiễu đèn).

---

## 4. Chỉ Số Parking Slot

So sánh từng slot-frame với ground truth:

\[
P_{slot}
=
\frac{TP_{occupied}}
{TP_{occupied}+FP_{occupied}}
\qquad
R_{slot}
=
\frac{TP_{occupied}}
{TP_{occupied}+FN_{occupied}}
\qquad
F1_{slot}
=
\frac{2P_{slot}R_{slot}}
{P_{slot}+R_{slot}}
\]

**Target:**
```text
Slot occupancy precision ≥ 0.98
Slot occupancy recall    ≥ 0.95
Slot occupancy F1        ≥ 0.97
False vacancy rate       ≤ 0.02
False occupied rate      ≤ 0.01
```

### Slot ownership accuracy

\[
Accuracy_{owner}
=
\frac{
\text{slot occupied gán đúng identity}
}{
\text{tổng slot occupied theo ground truth}
}
\]

**Target:**
```text
Slot ownership accuracy ≥ 0.98
Số ca 1 Global ID sở hữu 2 slot vật lý đồng thời = 0
Số ca 2 Global ID sở hữu 1 slot đồng thời         = 0
```

---

## 5. Chỉ Số Handoff

### Handoff identity accuracy

\[
Accuracy_{handoff}
=
\frac{\text{handoff giữ đúng ID}}
{\text{tổng handoff hợp lệ}}
\]

**Target:** `≥ 0.98` trên đường handoff đã hiệu chuẩn.

### Handoff latency

\[
T_{handoff}
=
t_{first\_valid\_target\_association}
-
t_{last\_valid\_source\_observation}
\]

**Target:**
```text
Median ≤ 500 ms
p95    ≤ 1.5 s
```

### Invalid handoff rate

\[
R_{invalid\_handoff}
=
\frac{\text{handoff vi phạm topology/thời gian}}
{\text{tổng handoff chấp nhận}}
\]

**Target:** `= 0` (bất biến tuyệt đối).

---

## 6. Chỉ Số Hiệu Năng Tính Toán

### End-to-end latency

\[
L_{e2e}
=
t_{frontend\_render}
-
t_{camera\_capture}
\]

Báo cáo: median · p95 · max · khi tắc nghẽn · khi slot analysis chạy · khi handoff.

**Target:**
```text
Median ≤ 250 ms
p95    ≤ 750 ms
Max sustained ≤ 1 s
```

### Throughput

\[
FPS_{effective}
=
\frac{\text{frame pairs đã xử lý}}
{\text{thời gian xử lý trôi}}
\]

**Target:**
```text
Mean ≥ 10 FPS
Min sustained ≥ 6 FPS
Không có tăng trưởng queue vô hạn
```

### Overload behavior test (bắt buộc)

Ép CPU overload có chủ đích, verify:

```text
FPS có thể giảm
uncertainty có thể tăng
temporarily-missing có thể tăng
Global ID KHÔNG đổi chỉ vì overload  ← bất biến
```

---

## 7. Khung Ablation Study (4 kịch bản)

Cùng một bản ghi + calibration + phần cứng + evaluator cho mọi thí nghiệm.

### Experiment A: Full Proposed Pipeline (Baseline)

Bật đủ: background subtraction · temporal frame difference · timestamp state prediction · homography projection · topology constraints · appearance association · temporal slot confirmation · identity retention policy.

**Kỳ vọng:**
```text
IDSW:           tối thiểu
IDF1 / MOTA:    cao nhất
Slot F1:        cao nhất
Handoff acc:    cao nhất
```

### Experiment B: No Frame Difference

Tắt bằng chứng frame-difference, giữ background modeling.

**Cơ chế suy hao** (từ PLAN 2 §1.3 — mất cổng AND):
- Xe chậm khó phát hiện (background model dần nuốt xe gần đứng yên)
- Xe dừng gần slot biến mất sớm → arrival confirmation giảm
- FN tăng trong chuyển động low-motion

**Phạm vi định lượng kỳ vọng:**
```text
Recall penalty:    5–15 điểm phần trăm
Slot F1 penalty:   5–12 điểm phần trăm
IDSW increase:     vừa phải
```

### Experiment C: No State-Space Prediction

Tắt velocity + covariance prediction.

**Cơ chế suy hao** (từ PLAN 2 §2 — mất $F(\Delta t)$, $Q(\Delta t)$):
- Gap 1 frame thành identity break
- Lag gây association error lớn (không có gate nở theo $\Delta t$)
- Timing handoff mong manh
- Displacement lớn bị coi là vật mới

**Phạm vi định lượng kỳ vọng:**
```text
IDSW increase:               2–5×
IDF1 penalty:                10–25 điểm phần trăm
Handoff accuracy penalty:    15–35 điểm phần trăm
```

### Experiment D: No Topological Constraint

Cho phép identity lịch sử BẤT KỲ match quan sát camera mới theo similarity chung.

**Cơ chế suy hao** (từ PLAN 2 §3.5, §4.5):
- Xe giống nhau ở vùng khác nhau thành ứng viên lẫn nhau
- Lỗi handoff tăng
- Xe song song/ngược chiều trao đổi identity
- False identity merge phổ biến

**Phạm vi định lượng kỳ vọng:**
```text
Invalid handoff rate:  khác 0 (vi phạm bất biến = 0)
IDSW (cảnh overlap):   tăng 2–10×
IDF1 penalty:          15–35 điểm phần trăm
```

### Quy tắc chấp nhận ablation

Full pipeline phải thắng MỌI cấu hình suy hao trên:

```text
IDSW · IDF1 · MOTA · handoff accuracy
· slot ownership accuracy · false vacancy rate
```

Loại một component mà không gây penalty đo được → component đó thừa, tích hợp sai, hoặc test chưa đủ.

---

## 8. Rubric Chấp Nhận 100 Điểm

### A. Architectural Integrity — 20 điểm

| Tiêu chí | Điểm |
|---|---:|
| Ingestion camera tách khỏi processing | 3 |
| Tồn tại đúng một Global Identity Registry có thẩm quyền | 4 |
| Lifecycle local track tách khỏi lifecycle Global ID | 3 |
| Hệ tọa độ world tường minh + tài liệu hóa | 3 |
| Topology camera là transition có hướng | 3 |
| Quyết định danh tính auditable + timestamp | 2 |
| Session state tách khỏi transient local track | 2 |

**Pass:** `≥ 18/20`

**Fail ví dụ:**
- Tracker cục bộ camera trực tiếp sở hữu user session
- Một camera mint ID vĩnh viễn không qua Registry

---

### B. Identity Continuity — 30 điểm

| Tiêu chí | Điểm |
|---|---:|
| IDSW = 0 chuyển động thường | 6 |
| IDSW = 0 handoff camera hợp lệ | 6 |
| Occlusion ngắn khôi phục đúng identity | 5 |
| Merged detection giữ mọi latent identity | 4 |
| Không có va chạm 1-GID/2-xe | 4 |
| Không tạo ID mới không cần thiết trong grace window | 3 |
| Identity retention hoạt động qua lag đo được | 2 |

**Pass:** `≥ 28/30` VÀ `IDSW = 0` trong kịch bản bắt buộc.

**Automatic FAIL:**
- 1 xe vật lý nhận 2 ID vĩnh viễn
- 1 Global ID gán đồng thời 2 xe vật lý
- Session hợp lệ không truy cập được vì ID đổi nội bộ

---

### C. Re-ID & Handoff Correctness — 15 điểm

| Tiêu chí | Điểm |
|---|---:|
| Topology hợp lệ bắt buộc trước handoff | 4 |
| Từ chối transition camera không hợp lệ | 2 |
| Thực thi time feasibility | 2 |
| Thực thi world-distance feasibility | 2 |
| Kết hợp direction + appearance | 2 |
| Ứng viên ambiguous bị defer thay vì ép | 2 |
| Handoff event log kèm bằng chứng | 1 |

**Pass:** `≥ 14/15` VÀ `invalid handoff rate = 0`

---

### D. Parking Slot Correctness — 20 điểm

| Tiêu chí | Điểm |
|---|---:|
| Footprint xe chiếu sang world coords | 3 |
| Cả IoU + Coverage được dùng | 3 |
| Centroid làm bằng chứng hỗ trợ | 2 |
| Verify inward movement | 2 |
| Temporal confirmation window tồn tại | 3 |
| cạnh tranh slot kề giải quyết deterministic | 2 |
| Hysteresis dập false vacancy | 2 |
| Parked Global ID tồn tại sau khi xe dừng | 3 |

**Pass:** `≥ 18/20` VÀ `Slot F1 ≥ 0.97` VÀ `ownership accuracy ≥ 0.98`

**Automatic FAIL:**
- Transit 1 frame gây occupied vĩnh viễn
- Xe đỗ mất ownership vì 1 false-empty frame
- 2 Global ID cùng sở hữu 1 slot

---

### E. Environmental Robustness — 5 điểm

| Tiêu chí | Điểm |
|---|---:|
| Chuyển tiếp brightness không tạo detection xe | 1 |
| Bóng không tạo detection xe | 1 |
| Nhiễu nén bounded | 1 |
| Uncertainty phối cảnh + seam được biểu diễn | 1 |
| Không cần GPS để vận hành | 1 |

**Pass:** `≥ 4/5`

---

### F. Computational Efficiency — 10 điểm

| Tiêu chí | Điểm |
|---|---:|
| Mean throughput ≥ 10 FPS | 3 |
| Min sustained ≥ 6 FPS | 2 |
| Median e2e latency ≤ 250 ms | 2 |
| p95 latency ≤ 750 ms | 1 |
| Không buffering vô hạn | 1 |
| Overload không gây ID minting | 1 |

**Pass:** `≥ 9/10`

---

## 9. Điều Kiện Chấp Nhận Cuối Cùng

Rewrite production được chấp nhận CHỈ KHI mọi điều kiện bắt buộc đúng:

```text
Overall score                          ≥ 90/100
IDSW (kịch bản bắt buộc)              = 0
Một xe vật lý → một ID                = luôn đúng
Invalid handoff rate                  = 0
Session survival                      = 100%
Slot occupancy F1                     ≥ 0.97
Slot ownership accuracy               ≥ 0.98
Mean throughput                       ≥ 10 FPS
p95 latency                           ≤ 750 ms
Không phụ thuộc GPS                   = verified
```

### Điều kiện TỪ CHỐI tuyệt đối (một điều xảy ra = reject)

```text
✗ Xe đổi Global ID trong handoff hợp lệ
✗ Xe đổi Global ID sau occlusion ngắn
✗ Một Global ID đại diện hai xe vật lý
✗ Session không truy cập được khi xe còn trong bãi
✗ Slot đánh dấu occupied từ transit 1 frame
✗ Xe đỗ biến mất khỏi identity map vì motion = 0
✗ System overload gây minting danh tính
```

---

## 10. Nguyên tắc kết

Hệ thống đúng chỉ khi **mọi nguồn bất định — detection, chuyển camera, lag, occlusion, trạng thái đỗ — đều thay đổi CONFIDENCE và LIFECYCLE STATE, nhưng không bao giờ tùy tiện thay đổi identity của xe vật lý.**
