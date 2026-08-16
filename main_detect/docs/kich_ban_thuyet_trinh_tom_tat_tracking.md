# KỊCH BẢN THUYẾT TRÌNH TÓM TẮT LUỒNG TRACKING TECHGAR

> Thời lượng đề xuất: 10–12 phút. Tài liệu bắt đầu trực tiếp từ thuật toán, không lặp phần giới thiệu đề tài. Ví dụ xuyên suốt là xe `G#2` đi từ `cam4` sang `cam3`, vào ô `P056`, dừng rồi rời ô.

## 1. Sơ đồ cần trình bày đầu tiên — khoảng 45 giây

```text
Frame video
    ↓
Chia thành bốn crop mô phỏng
    ↓
Tìm vùng chuyển động
    ↓
Theo dõi và cấp Local ID trong từng crop
    ↓
Dự đoán xe sang crop kề và bàn giao Global ID
    ↓
Gộp ID trùng về Canonical Global ID
    ↓
Kiểm tra xe giao với ROI ô đỗ
    ↓
Kiểm tra xe đã dừng hay chỉ đi qua
    ↓
Kết hợp với OpenCV nhận diện trạng thái ô
    ↓
Ghi một Global ID duy nhất lên JSON/web map
```

### Câu nói mẫu

“Luồng tracking của hệ thống không chỉ là vẽ một bbox. Mỗi frame phải đi qua bốn tầng: phát hiện chuyển động, tracking trong từng góc nhìn, duy trì Global ID giữa các góc nhìn và cuối cùng ghép Global ID với ô đỗ. Tôi sẽ dùng một xe `G#2` làm ví dụ xuyên suốt.”

---

## 2. Ba loại ID phải phân biệt — khoảng 45 giây

| Khái niệm | Phạm vi | Ví dụ |
|---|---|---|
| Local ID | Chỉ trong một tracker/crop | `cam4 local #7` |
| Global ID | Dùng chung toàn hệ thống | `G#2` |
| Canonical Global ID | ID cuối sau khi gộp trùng | `G#4 → G#2` |

Ví dụ:

```text
(cam4, local #7)  → G#2
(cam3, local #12) → G#2
```

Hai local track trên là hai quan sát của cùng một xe thật.

### Câu nói mẫu

“Local ID không thể đưa thẳng lên web vì bốn tracker đều có thể tạo local #1. Vì vậy khóa thật của bản đồ là Global ID. Nếu lỗi tạm thời tạo thêm `G#4`, manager gộp nó về `G#2`; từ đó 4 chỉ là alias và không được xuất hiện lại.”

---

## 3. Từ frame đến detection chuyển động — khoảng 1 phút

**File:** `src/techgar/motion_tracker.py`  
**Hàm:** `_temporal_motion_mask()`, `_detect()`

Hệ thống dùng hai nguồn bằng chứng:

1. MOG2 kiểm tra pixel khác nền dài hạn.
2. Frame difference kiểm tra pixel có thực sự thay đổi trong vài frame gần đây.

```python
background_mask = self.bg_sub.apply(frame)
temporal_motion = self._temporal_motion_mask(frame)
support = cv2.dilate(
    temporal_motion, np.ones((17, 17), np.uint8), 1
)
mask = cv2.bitwise_and(background_mask, support)
```

Trước khi trừ frame, code bù thay đổi sáng toàn ảnh bằng median:

\[
\Delta L=\operatorname{median}(I_t-I_{ref})
\]

Mask sau đó qua opening, dilation, closing, contour và các điều kiện về diện tích, width/height, aspect ratio và mật độ chuyển động.

Mỗi detection chứa:

```python
{
    "box": (x, y, w, h),
    "point": bbox_bottom_center,
    "area": contour_area,
    "hist": hsv_histogram,
}
```

### Điểm nổi bật

`_suppress_duplicate_detections()` loại hiện tượng một xe tạo hai silhouette cũ–mới khi frame difference. Nó so kích thước, HSV, khoảng cách tâm và khoảng hở bbox.

### Câu nói mẫu

“MOG2 cho bằng chứng nền dài hạn, frame difference cho bằng chứng chuyển động ngắn hạn. Tôi lấy giao hai mask, vì nếu chỉ tin MOG2 thì thay đổi ánh sáng hoặc xe đỗ lâu có thể trở thành foreground giả.”

---

## 4. Tracking trong một crop — khoảng 1 phút 30 giây

**Hàm:** `_new_kalman()`, `_assign()`, `_apply_detection()`, `process_frame()`

Kalman giữ state:

\[
\mathbf{x}=[x,y,v_x,v_y]^T
\]

- `predict()` ước lượng vị trí tiếp theo.
- `correct()` cập nhật bằng tọa độ detection thật.

Một detection được đối chiếu với track bằng ba sai số:

\[
E=0.50E_{distance}+0.30E_{IoU}+0.20E_{HSV}
\]

```python
costs[row, col] = (
    0.50 * (distance / max_distance)
    + 0.30 * (1.0 - iou)
    + 0.20 * appearance_distance
)
_, row_to_col, _ = lapjv(
    costs, extend_cost=True, cost_limit=0.90
)
```

LAPJV ghép toàn bộ track với toàn bộ detection theo quan hệ một-một, thay vì detection nào được duyệt trước thì thắng trước.

Vòng đời:

```text
tentative → confirmed → lost → expired
```

- `tentative`: quan sát mới, chưa đủ bằng chứng.
- `confirmed`: đủ số lần nhìn thấy và quãng đường.
- `lost`: đã confirmed nhưng tạm mất detection.
- `expired`: mất quá `lost_track_ttl`.

Nếu detection mới giống histogram của track đã expired và còn trong `reid_ttl`, `_create_or_reid()` khôi phục Local ID cũ.

### Điều không được nói sai

Kalman không nhận diện xe và không xác định xe dừng. Nó chỉ dự đoán vị trí để hỗ trợ nối track.

---

## 5. Từ Local ID sang Global ID — khoảng 2 phút

**File:** `src/techgar/cross_camera_manager.py`  
**Hàm chính:** `update_all_tracks()`

Hàm này điều phối nhiều hàm con theo thứ tự:

```text
_upsert_handoff()
    ↓
_match_pending_handoffs()
    ↓
_match_simultaneous_overlap()
    ↓
chỉ confirmed chưa có ID mới được cấp G# mới
    ↓
gộp các bbox/Global ID trùng
    ↓
cleanup handoff hết hạn
```

### Mở handoff sớm

Manager tính vận tốc từ tối đa năm điểm gần nhất:

\[
\mathbf v=\frac{\mathbf p_{last}-\mathbf p_{first}}{N-1}
\]

Thời gian tới biên:

\[
t_{edge}=\frac{d_{edge}}{|v_{axis}|}
\]

Nếu `t_edge ≤ 16 frame`, `_upsert_handoff()` lưu:

- Global ID;
- camera nguồn và đích;
- cạnh ra;
- vị trí toàn cục cuối;
- vận tốc;
- kích thước bbox;
- histogram HSV.

### Ba tầng kiểm tra xe ở camera đích

1. Đúng camera kề và đúng hành lang cạnh đi vào.
2. Gần vị trí dự đoán và không đi sai hướng.
3. HSV và kích thước bbox phù hợp.

Vị trí dự đoán:

\[
\hat{p}=p_{last}+v\Delta frame
\]

Chi phí:

\[
E_h=0.55E_{position}+0.30E_{HSV}
+0.10E_{size}+0.05E_{direction}
\]

LAPJV bảo đảm một Global ID chỉ được một local track nhận.

### Quy tắc tentative

```text
Tentative + nhận được ID cũ → dùng ID cũ ngay
Tentative + không nhận được ID cũ → tiếp tục chờ
Confirmed + không có giải thích cũ → mới tạo Global ID mới
```

Đây là cơ chế cứu xe chạy nhanh nhưng vẫn chặn detection nhiễu tạo xe mới trên map.

---

## 6. Ví dụ bàn giao `G#2`: cam4 sang cam3 — khoảng 1 phút

### Frame 100

```text
cam4 local #7 → G#2
vị trí gần cạnh trái
vx = -8 px/frame
cách biên 24 px
```

Thời gian tới biên:

\[
24/8=3\ frame
\]

Ba frame nhỏ hơn lookahead 16 nên manager mở:

```text
handoff G#2: cam4 → cam3
```

### Frame 102

Cam3 tạo `local #12 tentative`. Nó chưa được phép tạo Global ID mới.

Manager kiểm tra:

```text
Đúng cam3 và cạnh vào bên phải
Sai số vị trí dự đoán = 18 px < 90 px
HSV phù hợp
Kích thước phù hợp
```

LAPJV chọn:

```text
(cam3, local #12) → G#2
```

Kết quả manager trả:

```python
{
    "cam3": {12: 2},
    "cam4": {7: 2},
}
```

### Câu nói mẫu

“Tôi không đợi xe biến mất mới bàn giao. Khi xe sắp tới biên, hồ sơ Global ID đã được mở. Vì vậy local track đầu tiên ở cam3, dù còn tentative, vẫn có thể nhận ngay `G#2`.”

---

## 7. Gộp ID khi một xe bị tạo hai bbox — khoảng 45 giây

Nếu một xe chậm tạo hai bbox gần nhau và đã có `G#2`, `G#4`, manager kiểm tra:

- tỉ lệ diện tích;
- khoảng cách/gap bbox;
- HSV;
- hướng vận tốc nếu đo được.

Nếu chứng minh là cùng xe:

```text
G#4 → G#2
```

`_global_aliases[4]=2` được giữ vĩnh viễn. `canonical_global_id(4)` luôn trả 2. Binder cũng chạy `remap_vehicle_ids()` để slot không còn giữ ID 4.

ID nhỏ hơn được giữ vì thường được tạo trước và đã có lịch sử trên web; bbox hiển thị tốt nhất lại được chọn riêng theo visibility và area.

---

## 8. Xe đi vào ROI ô đỗ — khoảng 1 phút 30 giây

**File:** `src/techgar/slot_vehicle_binder.py`  
**Hàm:** `_overlap_geometry()`, `_batch_match()`

Binder chuyển bbox thành polygon và dùng:

```python
intersection_area, _ = cv2.intersectConvexConvex(
    vehicle_polygon,
    slot_polygon,
)
```

Hai tỉ lệ:

\[
O_v=\frac{A_{giao}}{A_{xe}},\qquad
O_s=\frac{A_{giao}}{A_{ô}}
\]

Xe được xét nằm trong ô nếu:

\[
(tâm\ trong\ ROI\land O_v\ge0.35)\lor O_v\ge0.60
\]

Điểm xe–ô:

\[
S=0.70O_v+0.20O_s+0.10C
\]

`_batch_match()` dùng LAPJV để bảo đảm:

- một Global ID chỉ thuộc một ô;
- một ô chỉ chứa một Global ID.

Xe mới giao ROI chỉ chuyển sang `stop_candidate`; chưa được xem là parked.

---

## 9. Chứng minh xe đã dừng — khoảng 1 phút

**Hàm:** `_is_stopped()`

Điều kiện:

- cửa sổ khoảng một giây;
- tối thiểu 8 mẫu;
- ít nhất 80% mẫu cùng thuộc một slot;
- vị trí không phân tán và không trôi liên tục.

Lấy median center, tính bán kính phân vị 95% và độ dịch chuyển đầu–cuối:

\[
r_{95}\le\max(3,0.06D)
\]

\[
d_{net}\le\max(5,0.10D)
\]

`D` là đường chéo bbox trung vị. Chuẩn hóa theo D giúp ngưỡng phù hợp hơn khi độ phân giải hoặc kích thước xe thay đổi.

Nếu đạt điều kiện và qua commit grace 0.15 giây, `_bind_vehicle()` ghi:

```python
binding.vehicle_id = 2
binding.tracking_occupied = True
binding.tracking_state = "parked"
```

---

## 10. Kết hợp với OpenCV nhận diện ô — khoảng 1 phút

**File:** `src/techgar/parking_detector.py`

OpenCV phân tích polygon ô bằng:

- 25 tổ hợp `5 gamma × 5 CLAHE`;
- adaptive threshold;
- center-cluster rescue;
- Canny edge recheck;
- temporal smoothing.

Code dùng ngưỡng trống `12/25`; đây là voting nhiều tham số, không phải nhiều mô hình AI và không phải majority tuyệt đối.

Hai nguồn được hợp nhất:

\[
occupied_{final}=vision_{occupied}\lor tracking_{occupied}
\]

| Vision | Tracking | Kết quả |
|---|---|---|
| Thấy xe | Chưa có ID | Occupied, `vehicle_id=null` |
| Báo trống sai | G#2 đã dừng | Occupied, `vehicle_id=2` |
| Thấy xe | G#2 đã dừng | Occupied, `vehicle_id=2` |

Tracking chỉ sửa “trống sai” thành “có xe”; không tự biến ô vision đang đỏ thành xanh.

---

## 11. Xe mất motion, chạy lại và rời P056 — khoảng 1 phút

### Khi xe đã parked và motion biến mất

Binder vẫn giữ:

```text
P056 → G#2
tracking_state = parked
```

Không có motion không đồng nghĩa xe đã rời ô.

### Khi xe bắt đầu chạy lại

Trước khi cấp Global ID mới, `try_recover_id()`:

1. mở polygon P056 thêm 15%;
2. kiểm tra bbox mới nằm trong/giao ROI;
3. kiểm tra camera và HSV nếu có;
4. trả lại ID 2.

Sau đó:

```python
manager.bind_external_id(
    cam_id, local_id, recovered_global_id, frame_idx
)
```

Local track mới nhận `G#2` trước `update_all_tracks()`.

### Khi xe ra khỏi ô

State chuyển:

```text
parked → exit_pending
```

Nếu cùng G#2 nằm ngoài ROI liên tục 0.5 giây, tracking override được gỡ. Trạng thái cuối quay lại theo vision, không tự kết luận trống.

---

## 12. JSON gửi web — khoảng 45 giây

Trước khi ghi:

```python
parking_binder.remap_vehicle_ids(
    manager.canonical_global_id
)
```

Nếu cùng Global ID có nhiều bbox, code ưu tiên:

1. bbox đang visible hơn bbox lost;
2. nếu cùng trạng thái, bbox có area mạnh hơn.

JSON một slot:

```json
{
  "P056": {
    "occupied": true,
    "vehicle_id": 2,
    "vision_occupied": false,
    "tracking_occupied": true,
    "decision_source": "tracking_override",
    "tracking_state": "parked",
    "stopped_for_ms": 1260
  }
}
```

JSON được ghi vào `.tmp` rồi `replace()` để web không đọc file đang ghi dở. `map_vehicles` chỉ có một entry cho mỗi Canonical Global ID.

---

## 13. Toàn bộ ví dụ `G#2` trong một bảng

| Bước | Trạng thái |
|---|---|
| 1 | Cam4 local #7 đã được map thành G#2. |
| 2 | Xe tiến gần cạnh trái cam4; manager dự đoán ba frame nữa tới biên. |
| 3 | Handoff G#2 cam4→cam3 được mở sớm. |
| 4 | Cam3 xuất hiện local #12 tentative. |
| 5 | Qua ba tầng kiểm tra và LAPJV, local #12 nhận G#2. |
| 6 | Bbox G#2 giao đủ mạnh với polygon P056. |
| 7 | Xe chuyển `moving → stop_candidate`. |
| 8 | Tọa độ ổn định khoảng một giây; state thành `parked`. |
| 9 | P056 lưu `vehicle_id=2`, kể cả khi motion mất. |
| 10 | Xe chạy lại; bbox mới trong ROI mở rộng nhận lại G#2. |
| 11 | Xe ra ngoài ROI: `exit_pending`. |
| 12 | Ngoài liên tục 0.5 giây: gỡ tracking override, trả trạng thái về vision. |
| 13 | JSON/web vẫn chỉ có một xe G#2. |

---

## 14. Kết luận nói trong 30–45 giây

“Điểm chính của TechGAR không nằm ở một detector riêng lẻ. Hệ thống xây chuỗi bằng chứng: MOG2 và frame difference tạo detection; Kalman, IoU, HSV và LAPJV tạo local track; predictive handoff duy trì Global ID; canonical merge loại ID trùng; polygon intersection và thống kê một giây chứng minh xe đã đỗ. Cuối cùng, vision và tracking được hợp nhất theo công thức OR để giảm trường hợp báo trống sai, và web chỉ nhận một Canonical Global ID cho mỗi xe.”

---

## 15. Năm câu phản biện ngắn

### “Đây có phải AI không?”

“Pipeline mặc định của `main.py` là computer vision cổ điển. YOLO/BoT-SORT chỉ là backend tùy chọn trong `single_camera.py`.”

### “Kalman có xác định xe dừng không?”

“Không. Kalman hỗ trợ dự đoán track; dừng được xác định bằng tọa độ detection thật, r95 và displacement.”

### “Làm sao tránh một xe có hai ID?”

“Loại motion echo, handoff tentative trước cấp ID mới, LAPJV một-một và canonical merge giữ ID nhỏ hơn.”

### “Xe đứng sẵn trước khi bật hệ thống thì sao?”

“Parking vision vẫn báo occupied nhưng có thể `vehicle_id=null` vì motion tracker chưa từng thấy xe đi vào.”

### “Bốn camera đã là camera vật lý chưa?”

“Chưa. Hiện là bốn crop của cùng video. Camera vật lý cần homography, đồng bộ timestamp và vùng bàn giao đã hiệu chỉnh.”

---

## Checklist trước khi nói

- [ ] Không gọi Local ID là ID duy nhất toàn hệ thống.
- [ ] Không nói tentative tự tạo Global ID mới.
- [ ] Không nói Kalman xác định xe dừng.
- [ ] Không gọi voting 12/25 là majority tuyệt đối.
- [ ] Không nói bốn crop là bốn camera vật lý.
- [ ] Dùng cùng ví dụ G#2 từ đầu đến cuối.
- [ ] Khi nói `update_all_tracks()`, nhắc các hàm con chứ không chỉ đọc một dòng gọi hàm.
