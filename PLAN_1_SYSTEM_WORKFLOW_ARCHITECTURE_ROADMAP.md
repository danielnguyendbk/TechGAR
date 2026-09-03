# TECHGAR — PLAN 1: SYSTEM WORKFLOW, ARCHITECTURE PIPELINE & IMPLEMENTATION ROADMAP

> **Loại tài liệu**: Production-grade System Redesign Specification — Redesign từ first principles, zero legacy code
> **Phạm vi**: Bãi đỗ xe thông minh trong nhà / hầm bê tông cốt thép, KHÔNG dùng GPS
> **Ràng buộc cốt lõi**: Mỗi xe vật lý ↔ đúng MỘT Global ID trong suốt vòng đời trong bãi
> **Liên kết**: PLAN 2 (toán học) · PLAN 3 (benchmark & rubric)

---

## 1. Core Problem Formulation

TechGAR là hệ thống quản lý bãi đỗ + dẫn dẫn xe trong môi trường ngầm/nội thất nơi GPS không khả dụng. Hệ thống phải suy luận danh tính xe, chuyển động, chuyển camera và trạng thái ô đỗ hoàn toàn từ quan sát camera + hệ tọa độ cục bộ đã hiệu chỉnh.

### 1.1. Năm bài toán ghép耦合

| # | Bài toán | Định lượng |
|---|---|---|
| 1 | **Giới hạn FOV một camera** | Xe rời vùng nhìn C1 trước khi vào C2; phải coi 2 quan sát là một xe khi bằng chứng không-thời gian hợp lệ |
| 2 | **Điểm mù kết cấu** | Trụ, dốc, tường, xe đỗ sẵn, đèn — che xe tạm thời; biến mất tạm KHÔNG được tạo danh tính mới |
| 3 | **Che khuất khi tắc nghẽn** | Hai xe chạm nhau → bbox hợp nhất; tách ra phải giữ 2 danh tính độc lập |
| 4 | **Skew phối cảnh góc chéo** | Khoảng cách pixel C1 ≭ pixel C2; phải diễn giải trong hệ tọa độ mặt phẳng sàn chung |
| 5 | **GPS denial hoàn toàn** | Không lat/long; mọi vị trí trong hệ tọa độ bãi cục bộ (cm/m theo mặt sàn hiệu chuẩn) |

### 1.2. Bất biến danh tính nền tảng

\[
\boxed{\text{Một xe vật lý} \mapsto \text{đúng một Global ID đang hoạt động}}
\]

Mất bằng chứng thị giác tạm thời → đánh dấu identity là *không chắc chắn*, nhưng **KHÔNG** đúc ngay identity thay thế.

**Trạng thái ĐÚNG (Pass):**
```text
Xe vào C1 = GID 17 → biến mất 1.2s sau trụ → xuất hiện ở C2 → vẫn GID 17
```

**Trạng thái SAI (Fail):**
```text
Xe vào = GID 17 → biến mất 1.2s → xuất hiện = GID 23
trong khi GID 17 vẫn active / vẫn gắn session
```

---

## 2. Kiến trúc đích

Kiến trúc phân lớp độc lập với **một dịch vụ danh tính duy nhất có thẩm quyền**:

```text
Camera 1
   │
   ▼
Frame Ingestion ──┐
                  ├──► Timestamp Alignment
Camera 2          │
   │              │
   ▼              │
Frame Ingestion ──┘
          │
          ▼
Environmental Normalization
          │
          ▼
Local Detection
          │
          ▼
Local Camera Tracking
          │
          ▼
Pixel-to-World Projection
          │
          ▼
World Detection Fusion
          │
          ▼
Topology-Constrained Association
          │
          ▼
Global Identity Registry          ← THẨM QUYỀN DUY NHẤT
          │
          ├──► Vehicle State
          ├──► Parking Slot State Engine
          ├──► Session Binding
          └──► Frontend Runtime State
```

**Nguyên tắc phân quyền:**
- Frontend map là **consumer hiển thị** — không phải hệ tọa độ tracking, không phải nguồn quyết định danh tính.
- Quyết định danh tính có thẩm quyền nằm ở **Global Identity Registry phía PC**. Tracker cục bộ theo camera chỉ *đề xuất* quan sát — không thể tự tạo danh tính vĩnh viễn hướng người dùng.

---

## 3. Sequential Execution Pipeline

Mỗi stage định nghĩa: Inputs → Operational Logic → Outputs → Pass/Fail cụ thể.

---

### Stage 1: Dual-Stream Frame Ingestion & Timestamp Synchronization

**Inputs**
```text
- Camera 1 image stream
- Camera 2 image stream
- Camera identifiers
- Camera frame timestamps
- Frame dimensions
- Stream-health metadata
```

**Operational Logic**
1. Đọc frame độc lập từng camera.
2. Gắn timestamp đơn điệu cho MỖI frame sau giải mã.
3. Ghi lại: camera ID · sequence number · decode timestamp · width/height · decode success/failure.
4. Duy trì latest-frame buffer cho từng camera.
5. **Drop frame cũ thay vì xử lý hàng đợi tích lũy.**
6. KHÔNG ghép cặp frame theo thứ tự đến.
7. Ghép cặp frame theo **độ gần timestamp**.

**Outputs**
```text
SynchronizedFramePair:
    frame_cam1
    frame_cam2
    timestamp_cam1
    timestamp_cam2
    timestamp_skew
    sequence_cam1
    sequence_cam2
```

**Pass**
```text
cam1 timestamp = 100.120 s
cam2 timestamp = 100.155 s
skew = 35 ms  →  cặp được chấp nhận (trong chính sách đồng bộ)
```

**Fail**
```text
cam1 timestamp = 100.120 s
cam2 timestamp = 102.400 s
skew = 2.280 s  →  KHÔNG được coi là cặp quan sát đồng thời
```

---

### Stage 2: Background Modeling & Environmental Noise Suppression

**Inputs**
```text
- Current frame mỗi camera
- Historical background model
- Previous normalized frames
- Camera-specific environmental configuration
```

**Operational Logic**
1. Chuyển frame sang biểu diễn luminance ổn định.
2. Áp normalization brightness/contrast theo từng camera.
3. Ước lượng nền.
4. Tính bằng chứng foreground từ:
   - background subtraction
   - temporal frame difference
   - edge consistency
   - local texture change
5. Ước lượng shadow likelihood.
6. Chỉ dập bóng khi CẢ BA điều kiện:
   - chromaticity gần nền
   - luminance thay đổi trong dải bóng
   - biên vật KHÔNG được hỗ trợ độc lập
7. Dập noise rời rạc bằng ràng buộc diện tích + hình dạng tối thiểu.
8. **Giữ vùng foreground low-confidence làm bằng chứng ứng viên** thay vì xóa ngay.
9. Đánh dấu **environmental instability** khi phần lớn frame đổi đồng loạt.

**Outputs**
```text
NormalizedFrame
ForegroundMask
ShadowMask
CandidateMotionRegions
EnvironmentQuality
```

**Pass**
```text
Xe di chuyển, sàn + ánh sáng ổn định → vùng xe giữ trong foreground mask;
global brightness shift chậm bị loại.
```

**Fail**
```text
Đèn huỳnh quang nhấp nháy → toàn sàn thành vùng foreground hình xe.
Hệ thống PHẢI classify là environmental instability, không phải detection.
```

---

### Stage 3: Local Bounding-Box Detections

**Inputs**
```text
- Normalized frame
- Foreground mask
- Environmental quality
- Camera-specific detection zones
```

**Operational Logic**
1. Sinh object candidates.
2. Ước lượng bbox, confidence, class, footprint cục bộ.
3. Giữ ứng viên low-confidence cho association thứ cấp.
4. Phát **merged region** khi một connected component:
   - lớn vượt footprint xe kỳ vọng
   - phủ nhiều vị trí xe dự đoán
   - có nhiều đỉnh motion nội bộ
5. Đánh dấu `occlusion_group_candidate`.
6. **KHÔNG update appearance danh tính từ merged region.**
7. KHÔNG tạo danh tính mới từ merged region nếu bất kỳ danh tính hiện hữu nào giải thích được quan sát sau grace period đầy đủ.

**Outputs**
```text
LocalDetection:
    camera_id
    bbox
    confidence
    local_center
    footprint_mask
    quality_score
    occlusion_group_candidate
```

**Pass**
```text
Hai xe chạm → 1 vùng lớn → đánh dấu occlusion_group_candidate (không gán vào 1 xe).
```

**Fail**
```text
Merged region gán thẳng vào Xe A; Xe B bị xóa vĩnh viễn khỏi identity set.
```

---

### Stage 4: Single-Camera Trajectory Tracking

**Inputs**
```text
- Current local detections
- Existing camera-local track states
- Previous timestamps
- Detection quality
- Occlusion-group candidates
```

**Operational Logic**
1. Predict từng local track tới timestamp hiện tại.
2. Associate detection conf-cao trước (tier 1).
3. Associate detection conf-thấp sau (tier 2).
4. Dùng bộ đếm missed-observation **theo thời gian** (không phải đếm frame).
5. Giữ track qua gap ngắn bằng: position prediction · appearance memory · bbox-size consistency · motion direction.
6. Phân biệt trạng thái:

```text
visible → temporarily_missed → occluded → merged → re-acquiring → retired
```

7. KHÔNG terminate track vì 1 frame missed.
8. KHÔNG tạo track mới khi track hiện hữu còn hypothesis re-acquisition hợp lý.
9. Khi 2 track bị 1 detection phủ: **giữ cả 2 track hypothesis sống (latent)**.

**Outputs**
```text
LocalTrackObservation:
    local_track_id
    camera_id
    predicted_bbox
    measured_bbox
    state
    missed_duration
    motion_vector
    appearance_reference
```

**Pass**
```text
Xe không được phát hiện trong 400 ms → track ở re-acquiring, đủ điều kiện khôi phục.
```

**Fail**
```text
1 frame missed → track retire ngay → xe xuất hiện lại nhận track mới.
```

---

### Stage 5: Planar Homography Projection (Pixel → Bird's-Eye-View)

**Inputs**
```text
- Local detection anchor point
- Camera-specific homography matrix H
- Pixel uncertainty
- Calibration uncertainty
- Seam-transition configuration
```

**Operational Logic**
1. Chọn anchor vật lý nhất quán:
   - ground-contact point cho motion mặt sàn
   - center-of-footprint point cho hình học top-down
2. Biến đổi anchor sang hệ tọa độ bãi chung.
3. Biến đổi uncertainty cục bộ → world uncertainty (lan truyền covariance).
4. Đánh dấu vùng topology: normal · overlap · handoff-exit-corridor · handoff-entry-corridor.
5. Tăng uncertainty gần: biên camera · seam overlap · vùng hiệu chuẩn kém · parallax đã biết.

**Outputs**
```text
WorldDetection:
    camera_id
    world_position
    world_covariance
    world_footprint
    source_pixel_position
    topology_region
```

**Pass**
```text
Cùng một xe ở pixel khác nhau từ 2 camera → project ra world positions
nằm trong seam uncertainty đã hiệu chuẩn.
```

**Fail**
```text
So sánh bằng pixel thô → coi là xa nhau vì 2 camera góc nhìn khác nhau.
```

---

### Stage 6: World Detection Fusion

**Inputs**
```text
- World detections từ Camera 1 & 2
- World covariance matrices
- Appearance descriptors
- Overlap polygon
```

**Operational Logic**
1. Sinh cặp ứng viên cross-camera **CHỈ trong overlap topology đã hiệu chuẩn**.
2. Loại cặp có khoảng thời gian không tương thích.
3. Loại cặp có world covariance không tương thích.
4. Loại cặp footprint đại diện 2 xe khác biệt.
5. Fusion quan sát trùng hợp lệ → 1 world detection.
6. Giữ detection không fusion riêng biệt.
7. Đảm bảo 1 detection KHÔNG fusion vào 2 xe trong cùng chu kỳ đồng bộ.

**Outputs**
```text
FusedWorldDetection:
    fused_position
    fused_covariance
    contributing_cameras
    contributing_observations
    fusion_confidence
```

**Pass**
```text
C1 + C2 quan sát cùng xe trong overlap → 1 fused world detection.
```

**Fail**
```text
2 xe khác nhau gần seam bị fuse thành 1 measurement vì tâm trùng nhau.
```

---

### Stage 7: Topology-Constrained Cross-Camera Association (Handoff Zone)

**Inputs**
```text
- Existing global identities
- Fused world detections
- Camera topology graph (directed)
- Camera exit polygons / entry polygons
- Predicted trajectories
- Appearance descriptors
```

**Operational Logic**
1. **KHÔNG Re-ID toàn cục mù quáng** qua mọi camera.
2. C2 được match identity C1 CHỈ KHI:
   - identity cuối cùng được C1 quan sát
   - identity ĐÃ vào exit polygon hiệu chuẩn của C1
   - C2 là successor topology hợp lệ
   - time gap vật lý khả thi
   - world displacement vật lý khả thi
3. Áp one-to-one assignment trên mọi ứng viên.
4. Yêu cầu **margin** giữa selected và competing identity.
5. Defer assignment khi ambiguity chưa giải.
6. Giữ Global ID cũ trong suốt defer period.
7. KHÔNG thay identity established bằng ID mới chỉ vì best-candidate tạm thời không chắc chắn.

**Outputs**
```text
AssociationDecision:
    observation_id
    assigned_global_id
    decision_type
    confidence
    competing_global_ids
    defer_reason
```

**Pass**
```text
Xe thoát C1 qua handoff corridor bên phải hợp lệ → xuất hiện C2
trong khoảng travel-time hiệu chuẩn → giữ nguyên Global ID.
```

**Fail**
```text
Xe ở vùng C2 không liên quan nhận Global ID của xe cuối cùng ở C1
chỉ vì màu giống nhau.
```

---

### Stage 8: Global Identity Registry

**Inputs**
```text
- Association decisions
- Re-identification evidence
- Track continuity
- Parking-slot ownership
- Exit-line events
- Session bindings
```

**Operational Logic**
1. Duy trì đúng MỘT identity record có thẩm quyền / xe vật lý.
2. Tách lifecycle identity khỏi lifecycle local track.
3. Lifecycle states:

```text
PROVISIONAL → ACTIVE → TEMPORARILY_MISSING → OCCLUDED
                            │
                            ▼
                    (recovery path)
                            │
             PARKED → EXIT_CONFIRMED → RETIRED
```

4. Global ID mới CHỈ được tạo khi CẢ BỐN điều kiện:
   - không identity nào thỏa ràng buộc khả thi vật lý
   - không identity nào trong grace window hợp lệ
   - không occlusion group chưa giải nào giải thích được quan sát
   - ứng viên sống sót maturity period
5. Global ID CHỈ retire khi:
   - exit event hợp lệ, HOẶC
   - timeout dài + không bằng chứng spatial/temporal/appearance/slot nào
6. Lưu mọi identity transition là **event có audit**.
7. Chặn trạng thái mâu thuẫn đồng thời.

**Outputs**
```text
GlobalVehicleState:
    global_id
    lifecycle_state
    latest_world_position
    last_observed_timestamp
    latest_camera
    slot_id
    appearance_memory
    uncertainty
    session_ids
```

**Pass**
```text
GID 17 missing 1.2s → state=TEMPORARILY_MISSING → quan sát hợp lệ kế tiếp gán ID 17.
```

**Fail**
```text
GID 17 retire sau gap 1.2s → cùng xe được mint là ID 23.
```

---

### Stage 9: Spatial-Temporal Parking Slot Occupancy Engine

**Inputs**
```text
- World vehicle footprint
- Slot polygon (world coords)
- Global vehicle state
- Vision occupancy evidence
- Motion state
- Temporal evidence window
```

**Operational Logic**
1. Mỗi ô đỗ = polygon world; mỗi xe = footprint polygon (hoặc footprint mở rộng uncertainty).
2. Tính overlap footprint/slot (IoU + Coverage — xem PLAN 2 §5).
3. Chọn slot candidate chỉ khi:
   - overlap trên ngưỡng tối thiểu
   - xe đang di chuyển inward HOẶC đã ổn định
   - không có slot cạnh tranh với điểm tương đương
4. Tích lũy **arrival evidence theo thời gian**.
5. Hysteresis: entry threshold > confirmation threshold > release threshold.
6. Overlap 1 frame = transit, KHÔNG phải parking.
7. Tracking mất tạm thời → **giữ arrival claim** trong grace period.
8. Commit parking chỉ khi: đủ quan sát thời gian + đủ occupancy evidence + vị trí ổn định + không competing ownership.
9. Global ID gắn slot trong suốt thời gian đỗ.

**Outputs**
```text
ParkingSlotState:
    slot_id
    occupancy_state
    owning_global_id
    overlap_score
    dwell_duration
    confirmation_confidence
    last_update_timestamp
```

**Pass**
```text
Xe overlap B04 = 0.78, di chuyển inward, giữ trong 5 cm tâm slot 2 giây,
không competing → B04 = occupied bởi GID 17.
```

**Fail**
```text
Xe đi QUA B04 1 frame với 0.35 overlap → B04 bị reserve vĩnh viễn.
```

---

### Stage 10: Upstream Event Dispatch & Frontend Synchronization

**Inputs**
```text
- Global vehicle state
- Slot state
- Camera health
- Session state
- Identity lifecycle events
```

**Operational Logic**
1. Publish **immutable runtime snapshot**: active vehicles · temporarily-missing (trong display hold) · parked · slot occupancy · identity events · timestamp + sequence.
2. Phân biệt rõ: `observed` / `temporarily_missing` / `parked` / `retired`.
3. Frontend KHÔNG xóa marker khi thiếu 1 update.
4. Xe đỗ luôn được đại diện bởi slot + Global ID.
5. Identity stale-dài không có observation/ownership → ẩn khỏi hiển thị.
6. Session lookup dùng persistent session identifier + identity mapping hiện tại.

**Pass**
```text
Frontend không nhận quan sát 300 ms → giữ nguyên marker + Global ID.
Sau 2s → đổi visual state "temporarily missing" (không tạo/hiển thị ID khác).
```

**Fail**
```text
Marker biến mất 1 polling interval → xuất hiện lại với Global ID khác.
```

---

## 4. Phase-Gated Implementation Roadmap

### Phase 0: Measurement & Data Contract

**Inputs**: bản ghi camera hiện hữu · điểm hiệu chuẩn · layout bãi · kịch bản đã annotate.

**Work**
1. Định nghĩa đơn vị timestamp chuẩn.
2. Định nghĩa schema pixel / world / slot / identity.
3. Đo camera timestamp skew.
4. Đo homography residuals bằng **nhiều hơn** số điểm hiệu chuẩn tối thiểu.
5. Đo seam disagreement bằng quan sát đồng thời.
6. Định nghĩa taxonomy event + lifecycle.
7. Thiết lập baseline metrics.

**Entry criteria**: có ≥ 2 bản ghi đồng bộ 2 camera + layout bãi.
**Exit (Pass) criteria**
```text
✓ Mọi detection có timestamp + camera ID
✓ Mọi world observation chứa covariance
✓ Mọi identity event có frame + timestamp
✓ Báo cáo hiệu chuẩn có bậc tự do dư > 0 (non-zero DOF)
```
**Fail criteria**
```text
✗ Quyết định danh tính phụ thuộc wall-clock không tracked
✗ Quan sát camera không match được timestamp
✗ Calibration quality = 0 error chỉ vì dùng đúng 4 điểm
  (homography 8-DOF = nghiệm chính xác — overfit, không phản ánh sai số thật)
```

---

### Phase 1: Lag-Resilient Ingestion & Local Tracking

**Dependencies**: Phase 0 data contract.

**Work**
1. Latest-frame buffering.
2. Timestamp-based frame pairing.
3. Adaptive background + motion evidence.
4. Time-based local track persistence.
5. Occlusion-group representation.
6. Template/block matching trực tiếp làm **phép đo khôi phục**.
7. Không Global ID mới trong local recovery grace window.

**Exit (Pass) criteria**
```text
✓ Delay input 500 ms không terminate local track tự động
✓ Xe missing 1 giây khôi phục được không cần identity mới
✓ 2 xe chạm giữ 2 latent track
```
**Fail criteria**
```text
✗ 1 frame missed → identity mới
✗ Blob hợp nhất lớn gán vào 1 xe duy nhất
✗ Queue stale frame tạo latency tích lũy đa giây
```

---

### Phase 2: World Projection & Detection Fusion

**Dependencies**: Phase 1 local detections + calibration hợp lệ Phase 0.

**Work**
1. Project anchors sang world.
2. Lan truyền covariance measurement.
3. Định nghĩa overlap + handoff polygons.
4. Fuse duplicate cross-camera detections.
5. Giữ detection tách cho 2 xe tách.

**Exit (Pass) criteria**
```text
✓ Cùng xe 2 camera → 1 fused observation
✓ 2 xe gần nhau giữ tách biệt khi covariance ellipse + appearance không tương thích
```
**Fail criteria**
```text
✗ Dùng pixel thô cho khoảng cách cross-camera
✗ 1 observation gán vào 2 xe
✗ Fusion chỉ dùng Euclidean, bỏ covariance
```

---

### Phase 3: Global Identity Registry

**Dependencies**: Phase 2 world detections + topology graph + lifecycle rules.

**Work**
1. Tập trung quyền tạo identity.
2. Tách local-track ID khỏi Global ID.
3. Thêm states: provisional/missing/occluded/parked/retired.
4. Thực thi identity invariants.
5. Append-only identity event records.
6. Identity recovery TRƯỚC khi tạo ID mới.

**Exit (Pass) criteria**
```text
✓ 1 xe vật lý = 1 Global ID qua mọi camera transition
✓ Tạo ID mới bị chặn khi identity nào còn plausible về vật lý
✓ Quan sát mâu thuẫn bị quarantine thay vì merge mù
```
**Fail criteria**
```text
✗ 2 identity cùng đại diện 1 xe đồng thời
✗ 1 identity đại diện 2 xe ở 2 vị trí
✗ Tạo identity mới khi dormant identity hợp lệ tồn tại
```

---

### Phase 4: Parking Slot State Engine

**Dependencies**: Global ID ổn định + world footprints + slot polygons.

**Work**
1. World-footprint/slot overlap.
2. Centroid + inward-motion checks.
3. Temporal arrival claims.
4. Dwell confirmation.
5. Slot ownership + release hysteresis.
6. Parked identity tồn tại độc lập với visibility track đang chạy.

**Exit (Pass) criteria**
```text
✓ Xe đi ngang không reserve slot
✓ Xe tracking gián đoạn vẫn hoàn tất arrival claim
✓ Xe đỗ đã confirm giữ Global ID gốc
```
**Fail criteria**
```text
✗ Overlap 1 frame tạo parking assignment
✗ Slot release vì 1 false-empty observation
✗ Xe đỗ mất identity khi motion = 0
```

---

### Phase 5: Session Protection

**Dependencies**: Global Identity Registry + Slot State Engine.

**Work**
1. Bind session vào Global ID + **persistent vehicle fingerprint**.
2. Identity remapping qua explicit audited events.
3. Reconnect session vào recovered Global ID.
4. Chặn xóa session trước exit vật lý confirmed.
5. Dùng bằng chứng slot + topology khi khôi phục session.

**Exit (Pass) criteria**
```text
✓ Xe biến mất tạm + vào lại cùng session
✓ Identity alias/stitch event cập nhật session không cần user action
✓ Session xóa chỉ sau exit event hợp lệ
```
**Fail criteria**
```text
✗ Gap ngắn xóa session
✗ User phải quét QR mới sau re-ID hợp lệ
✗ Stale identity làm đóng sai session
```

---

### Phase 6: Performance & Deployment Hardening

**Dependencies**: mọi phase trước.

**Work**
1. Đo processing time từng stage.
2. Tách ingestion khỏi processing.
3. Bound mọi queue.
4. Batch thao tác đắt.
5. Hardware acceleration khi có.
6. Slot analysis tần số thấp chạy tách khỏi tracking tần số cao.
7. Visualization encoding chỉ khi có subscriber active.
8. Overload behavior:
   - skip stale frame
   - preserve state
   - tăng uncertainty
   - **KHÔNG BAO GIỜ mint ID mới chỉ vì overload**

**Exit (Pass) criteria**
```text
✓ Mean tracking latency < 100 ms dưới tải kỳ vọng
✓ End-to-end displayed-state latency < 1 giây
✓ CPU overload tạm tăng uncertainty nhưng KHÔNG đổi ID
✓ Frame drop không tích lũy backlog
```
**Fail criteria**
```text
✗ Processing delay tăng liên tục vô hạn
✗ CPU spike → Global ID mới
✗ Frontend hiển thị frame cũ sau khi frame mới có sẵn
```

---

## 5. Ma trận trách nhiệm stage × tính năng

| Tính năng | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Đồng bộ 2 stream | ● | | | | | | | | | |
| Chống nhiễu môi trường | | ● | ● | | | | | | | |
| Track cục bộ lag-robust | | | | ● | | | | | | |
| BEV / homography | | | | | ● | | | | | |
| Fusion overlap | | | | | | ● | | | | |
| Handoff topology | | | | | | | ● | ● | | |
| 1 xe = 1 ID | | | | | | | | ● | | |
| Slot occupancy đúng | | | | | | | | | ● | |
| Session QR sống | | | | | | | | ● | ● | ● |
| Hiển thị không nhấp nháy | | | | | | | | | | ● |

---

## 6. Tóm tắt nguyên tắc thiết kế

1. **Một thẩm quyền danh tính** — Global Identity Registry là nơi duy nhất mint/retire.
2. **Thời gian là chiều chính** — mọi TTL/grace/window tính bằng giây thực, không phải đếm frame.
3. **Uncertainty là trạng thái, không phải lỗi** — lag/occlusion làm tăng covariance và đổi lifecycle state, không đổi identity.
4. **Topology trước, ngoại hình sau** — chỉ được Re-ID trong hành lang camera hợp lệ.
5. **Fusion trước association** — 2 camera nhìn cùng xe phải thành 1 measurement trước khi so với identity.
6. **Frontend là consumer** — không bao giờ là nguồn chân lý danh tính.
