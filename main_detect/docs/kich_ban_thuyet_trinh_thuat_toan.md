# Kịch bản thuyết trình code và thuật toán TechGAR

> Phạm vi: đi thẳng vào cấu trúc code và thuật toán, không lặp lại phần giới thiệu đề tài.  
> Thời lượng nói chính: khoảng 20 phút. Các mục “Phản biện” và “Phụ lục” dùng khi hội đồng hỏi sâu.  
> Mọi tọa độ “toàn cục” trong bản demo bốn crop là tọa độ pixel của video gốc, chưa phải tọa độ mét trong thế giới thật.

## Phân bổ thời gian

| Thời gian | Nội dung |
|---:|---|
| 0:00–2:00 | Luồng `main.py`, ROI và tần suất xử lý |
| 2:00–6:00 | Motion detection, Kalman và tracking một camera |
| 6:00–10:00 | Handoff, Global ID, chống trùng ID |
| 10:00–14:00 | Nhận diện ô đỗ bằng multi-parameter voting |
| 14:00–18:00 | Ghép xe–ô, xác định dừng, state machine |
| 18:00–20:00 | Hợp nhất hai nguồn, hướng xe, JSON và giới hạn |

---

## 1. Luồng xử lý trung tâm trong `main.py`

`main.py` không chứa toàn bộ thuật toán nhận diện. Vai trò của nó là **orchestrator**: mở video, tạo các mô-đun, gọi chúng đúng thứ tự, đổi hệ tọa độ và xuất kết quả. Thuật toán chi tiết nằm trong các file thuộc `src/techgar/`.

> Cách sử dụng khi thuyết trình: các mục 1.1–1.13 là phần học và trả lời phản biện; khi nói trên sân khấu, dùng bản rút gọn ở mục 1.14 trong khoảng 2–3 phút rồi chuyển sang thuật toán motion tracking.

### 1.1. Bản đồ các hàm quan trọng

| Thành phần | Nhiệm vụ thật | Có quyết định thuật toán không? |
|---|---|---|
| `CameraSimulator.__init__()` | Đọc metadata video, xác định bốn crop và gọi chia ROI | Có, vì tạo hệ tọa độ của bốn góc nhìn |
| `_load_and_split_slots()` | Scale 69 polygon ROI, phân vào cam1–cam4 và đổi sang tọa độ local | Có |
| `_classify_point()` | Chọn camera dựa trên tâm ROI và bốn quadrant | Có |
| `get_camera_frame()` | Cắt một crop từ frame gốc | Có |
| `get_all_camera_frames()` | Gọi cắt cho cả bốn camera | Có |
| `run_detection()` | Điều phối tracker, Global ID, binder, parking detector và JSON | Quan trọng nhất |
| `_save_json_atomic()` | Ghi file tạm rồi replace để web không đọc JSON dở | Quan trọng cho độ ổn định |
| `run_preview()` | Chỉ xem crop/ROI, không chạy nhận diện | Không |
| `draw_split_overlay()` | Vẽ đường chia bốn vùng để debug | Không |
| `draw_camera_with_slots()` | Vẽ polygon lên ảnh preview | Không |
| `parse_args()` | Khai báo tham số CLI và đường dẫn mặc định | Cấu hình, không phải thuật toán |

Điểm cần nhớ: `CameraSimulator` không phải camera driver và cũng không nhận diện xe. Nó chỉ giả lập bốn góc nhìn bằng cách cắt cùng một frame.

### 1.2. `CameraSimulator.__init__()`: tạo không gian của hệ thống

**Code thật rút gọn:**

```python
cap = cv2.VideoCapture(source_path)
if not cap.isOpened():
    raise RuntimeError(f"Không mở được video: {source_path}")
self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

self.mid_x = self.frame_w // 2
self.mid_y = self.frame_h // 2
```

Các biến:

- `frame_w`, `frame_h`: kích thước frame gốc. Video mẫu là 1100×720.
- `fps`: FPS metadata của video; nếu bằng 0 hoặc không hợp lệ thì dùng 30.
- `mid_x=550`, `mid_y=360`: đường chia giữa video.
- `total_frames`: số frame, dùng để biết chiều dài nguồn.

Sau đó class tạo bốn crop:

```text
cam1: (0,   0)   → (550, 360)  top-left
cam2: (550, 0)   → (1100,360)  top-right
cam3: (0,   360) → (550, 720)  bottom-left
cam4: (550, 360) → (1100,720)  bottom-right
```

Một crop được lưu theo dạng `(x1, y1, x2, y2)`. Với `overlap=0`, các crop không chồng pixel. Điều này giúp cùng một xe không bị hai motion tracker nhìn thấy quá lâu tại vùng biên.

**Câu nói mẫu:** “CameraSimulator xây hệ tọa độ cho demo. Video 1100×720 được chia tại 550 và 360; đây là bốn crop ảo chứ chưa phải bốn camera vật lý.”

### 1.3. `_load_and_split_slots()`: đưa ROI về đúng camera

File `config/parking_slots.json` lưu polygon theo độ phân giải tham chiếu. Nếu độ phân giải video thay đổi, ROI phải scale:

```python
ref_w = data["imageWidth"]
ref_h = data["imageHeight"]
sx = self.frame_w / ref_w
sy = self.frame_h / ref_h

center_x = slot["center"]["x"] * sx
center_y = slot["center"]["y"] * sy
cam_id = self._classify_point(center_x, center_y)

lx = point["x"] * sx - crop[0]
ly = point["y"] * sy - crop[1]
```

Có hai phép biến đổi khác nhau:

```text
Scale ROI:
x_scaled = x_config × frame_width  / config_width
y_scaled = y_config × frame_height / config_height

Global → local crop:
x_local = x_scaled - crop_x1
y_local = y_scaled - crop_y1
```

`_classify_point()` dùng tâm ROI:

- trái và trên → cam1;
- phải và trên → cam2;
- trái và dưới → cam3;
- phải và dưới → cam4.

Mỗi camera nhận một file `parking_slots_camX.json` trong `runtime_output`. Parking detector của camera đó chỉ đọc các polygon local của chính nó.

**Ví dụ:** một đỉnh ROI có tọa độ global `(620, 200)` thuộc cam2. Offset cam2 là `(550, 0)`, nên tọa độ local là:

```text
x_local = 620 - 550 = 70
y_local = 200 - 0   = 200
```

**Lỗi hàm này xử lý:** nếu không trừ offset, polygon Pxxx của cam2 sẽ bị vẽ lệch sang phải 550 px và nằm ngoài crop.

### 1.4. `get_camera_frame()` và `get_all_camera_frames()`

```python
def get_camera_frame(self, frame, cam_id):
    x1, y1, x2, y2 = self.cameras[cam_id]["crop"]
    return frame[y1:y2, x1:x2].copy()

def get_all_camera_frames(self, frame):
    return {
        cam_id: self.get_camera_frame(frame, cam_id)
        for cam_id in self.cameras
    }
```

OpenCV/NumPy truy cập ảnh theo `frame[y1:y2, x1:x2]`: hàng trước, cột sau. Hàm trả `.copy()` để việc vẽ bbox lên crop không làm thay đổi frame gốc.

Kết quả là dictionary:

```python
cam_frames = {
    "cam1": frame_top_left,
    "cam2": frame_top_right,
    "cam3": frame_bottom_left,
    "cam4": frame_bottom_right,
}
```

Mỗi `MotionVehicleTracker` chỉ nhìn một phần tử trong dictionary này. Vì vậy local ID số 1 ở cam1 và local ID số 1 ở cam2 là hai namespace khác nhau; không được gửi local ID trực tiếp lên web.

### 1.5. `run_detection()`: khởi tạo bốn tracker nhưng chỉ một manager và một binder

```python
manager = CrossCameraManager(
    camera_sizes=camera_sizes,
    camera_crops={cid: tuple(cam["crop"]) for cid, cam in sim.cameras.items()},
    lookahead_frames=args.handoff_lookahead_frames,
    prediction_radius=args.handoff_prediction_radius,
)
parking_binder = SlotVehicleBinder(
    stop_seconds=args.slot_stop_seconds,
    exit_seconds=args.slot_exit_seconds,
)
```

Kiến trúc đối tượng:

```text
cam1 ─ MotionVehicleTracker ┐
cam2 ─ MotionVehicleTracker ├─→ một CrossCameraManager
cam3 ─ MotionVehicleTracker ┤          │
cam4 ─ MotionVehicleTracker ┘          ↓
                                  một SlotVehicleBinder
```

- Bốn `MotionVehicleTracker`: mỗi crop có nền MOG2, Kalman và local track riêng.
- Một `CrossCameraManager`: sở hữu namespace Global ID duy nhất.
- Một `SlotVehicleBinder`: sở hữu ánh xạ Global ID ↔ ô đỗ trên toàn bản đồ.
- Bốn `ParkingDetector`: mỗi camera xử lý ROI local của nó.

Tại sao không tạo bốn binder? Nếu mỗi camera có một binder riêng, cùng G#2 có thể bị gắn vào một ô ở cam3 và một ô khác ở cam4. Binder dùng chung mới thực thi được ràng buộc một xe–một ô trên toàn hệ thống.

### 1.6. Bước 1 mỗi frame: local tracking

```python
ok, frame = cap.read()
frame_idx += 1
now = time.monotonic()
cam_frames = sim.get_all_camera_frames(frame)

for cam_id, tracker in trackers.items():
    _, _, expired_tracks = tracker.process_frame(cam_frames[cam_id])
    for tid, track in tracker.newly_lost_tracks:
        manager.notify_track_lost(cam_id, tid, track, frame_idx)
```

`tracker.process_frame()` thực hiện detection chuyển động, Kalman, LAPJV và cập nhật trạng thái tentative/confirmed/lost/expired. `main.py` không tự làm các phép này.

Khi một track vừa lost, manager lưu vị trí, vận tốc, bbox và appearance để chuẩn bị merge/handoff. Khi expired, mapping local cũ được dọn nhưng hồ sơ handoff đã mở vẫn được giữ tới `handoff_ttl`.

### 1.7. Ba tập track khác nhau — điểm dễ nhầm nhất

| Biến/API | Chứa gì | Dùng để làm gì? |
|---|---|---|
| `observable_tracks` | Tentative + confirmed + lost còn quan sát được | Handoff/recovery sớm |
| `active_tracks` | Các track local còn sống | JSON/debug và registry |
| `confirmed_tracks` | Track đã đủ bằng chứng xác nhận | Ghép với ô và hiển thị ổn định |

Đoạn sau gom cả tentative:

```python
all_observable_tracks = {
    cam_id: tracker.observable_tracks
    for cam_id, tracker in trackers.items()
}
```

Tại sao cần tentative? Xe chạy nhanh có thể chỉ xuất hiện vài frame ở camera đích. Nếu chờ confirmed rồi mới handoff, hệ thống có thể đã cấp ID mới hoặc xe đã đi sâu khỏi entry edge.

Tentative được quyền **nhận lại ID có sẵn**, nhưng nếu không match thì chưa được tự tạo Global ID mới. Đây là hàng rào chống nhiễu.

### 1.8. Phục hồi ID từ ô đỗ phải chạy trước handoff/cấp ID mới

```python
recovered_global_id = parking_binder.try_recover_id(
    position=(track.cx, track.cy),
    camera_id=cam_id,
    bbox=(track.x, track.y, track.w, track.h),
    appearance=track.appearance,
    coordinate_offset=(crop[0], crop[1]),
)
if recovered_global_id is not None:
    manager.bind_external_id(
        cam_id, local_id, recovered_global_id, frame_idx
    )
```

Giải thích từng biến:

- `cam_id`: camera đang nhìn thấy local track mới.
- `local_id`: ID do tracker camera đó cấp.
- `track.cx, track.cy`: bottom-center của track dùng cho quỹ đạo.
- `track.x, y, w, h`: bbox local theo dạng tọa độ góc trái + kích thước.
- `coordinate_offset=(crop[0], crop[1])`: cộng offset để binder kiểm tra trong hệ video gốc.
- `appearance`: histogram HSV hỗ trợ loại candidate khác xe.
- `recovered_global_id`: ID đã được slot parked giữ trước đó.

Khi có `bbox`, `try_recover_id()` dùng tâm bbox và polygon ô mở rộng 15% để thử nhận lại xe đang rời ô. `bind_external_id()` không sinh ID mới; nó chỉ gắn cặp `(cam_id, local_id)` với G# cũ đã được xác minh.

**Thứ tự không được đảo:**

```text
recovery từ parked slot
        ↓
predictive handoff giữa camera
        ↓
chỉ confirmed track còn lại mới được tạo Global ID mới
```

Nếu cấp ID mới trước recovery, cùng một xe có thể đang giữ G#2 trong P056 nhưng local track mới lại nhận G#4.

### 1.9. `update_all_tracks()`: từ Local ID sang Global ID

```python
global_ids_per_cam = manager.update_all_tracks(
    all_observable_tracks, frame_idx
)
```

Kết quả có dạng:

```python
global_ids_per_cam = {
    "cam3": {12: 2},   # local #12 của cam3 là Global #2
    "cam4": {7: 2},    # local #7 của cam4 cũng là Global #2
}
```

Hai local track có thể cùng trỏ G#2 trong vùng overlap. Đây không phải hai xe; đó là hai quan sát của cùng một xe. Bên trong `update_all_tracks()`, manager lần lượt:

1. mở handoff dự đoán từ track nguồn;
2. ghép batch handoff với track đích bằng LAPJV;
3. xử lý cùng xe xuất hiện đồng thời ở overlap;
4. chỉ cấp Global ID mới cho confirmed track chưa match;
5. merge duplicate và canonicalize alias;
6. dọn handoff hết hạn.

Chi tiết toán học của bước này được trình bày ở Mục 5–7, không nên nhét toàn bộ vào `main.py`.

### 1.10. Canonicalization và tạo một quan sát duy nhất cho binder

```python
parking_binder.remap_vehicle_ids(manager.canonical_global_id)
global_active_tracks = {}
for cam_id, tracker in trackers.items():
    crop = sim.cameras[cam_id]["crop"]
    for local_id, track in tracker.confirmed_tracks.items():
        global_id = manager.get_global_id(cam_id, local_id)
        candidate = {
            "bbox": (track.x + crop[0], track.y + crop[1], track.w, track.h),
            "camera_id": cam_id,
            "visible": track.consecutive_invisible_count == 0,
        }
```

`remap_vehicle_ids()` xử lý trường hợp merge G#4 → G#2. Mọi state và slot từng giữ số 4 phải đổi về canonical ID 2.

`global_active_tracks` có khóa là Global ID, không phải local ID. Bbox được đổi local → global:

```text
x_global = x_local + crop_x1
y_global = y_local + crop_y1
w_global = w_local
h_global = h_local
```

Đây là phép **cộng**, không phải trừ. Trừ offset chỉ dùng khi đổi từ global về local trong `_load_and_split_slots()`.

Nếu hai camera cùng thấy G#2, code chỉ giữ candidate mạnh hơn:

1. ưu tiên track đang visible hơn track lost;
2. nếu cùng trạng thái, ưu tiên bbox có area lớn hơn.

Nhờ vậy binder và map không nhận hai bbox đồng thời cho một canonical ID.

**Ví dụ đúng với video hiện tại:** cam4 có offset `(550,360)`. Một bbox local bắt đầu tại `(20,120)` sẽ có góc trái global:

```text
x_global = 20  + 550 = 570
y_global = 120 + 360 = 480
```

Nếu nói về tâm/bottom-center local `(20,120)`, vị trí global tương ứng cũng là `(570,480)`.

### 1.11. Binder chạy mỗi frame, ParkingDetector chạy 2 Hz

```python
tracking_timestamp_s = frame_idx / max(sim.fps, 1.0)
parking_binder.update_tracks(
    global_active_tracks, frame_idx, tracking_timestamp_s
)

if now - last_parking_at >= parking_interval:
    results = detector.detect(cam_frames[cam_id], apply_smoothing=True)
    parking_binder.update_vision(
        results, frame_idx, tracking_timestamp_s,
        camera_id=cam_id, coordinate_offset=(crop[0], crop[1]),
    )
```

Có ba nhịp khác nhau:

| Nhịp | Mặc định | Lý do |
|---|---:|---|
| Local tracking + binder | mọi frame | Cần đủ điểm tính vận tốc, handoff và cửa sổ dừng |
| Parking vision | 2 Hz | Ô đỗ đổi chậm, pipeline 25 cấu hình nặng hơn |
| Ghi JSON | 5 Hz | Web đủ mượt, tránh ghi file ở mọi frame |

`tracking_timestamp_s = frame_idx / fps` biểu diễn thời gian của nội dung video. `time.monotonic()` dùng để điều khiển “đã đến lúc chạy parking/ghi JSON chưa” theo thời gian chạy thực tế và không bị ảnh hưởng nếu đồng hồ hệ thống đổi.

Nếu binder chỉ chạy 2 Hz, cửa sổ dừng một giây chỉ có khoảng hai mẫu, không đạt yêu cầu tối thiểu tám mẫu. Vì vậy binder bắt buộc chạy theo tracker, không chạy theo ParkingDetector.

### 1.12. `_save_json_atomic()`: vì sao không `json.dump` trực tiếp?

```python
def _save_json_atomic(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except PermissionError:
        pass
```

Nếu ghi thẳng vào `global_vehicle_registry.json`, web có thể đọc đúng lúc Python mới ghi một nửa file và gặp lỗi JSON. Atomic write làm hai bước:

1. ghi hoàn chỉnh vào `global_vehicle_registry.json.tmp`;
2. dùng `replace()` đổi file tạm thành file chính.

Web hoặc đọc bản cũ hoàn chỉnh, hoặc bản mới hoàn chỉnh; không đọc bản đang viết dở. `PermissionError` được bỏ qua để một lần web giữ file không làm dừng toàn bộ video, nhưng production nên log lỗi này thay vì im lặng hoàn toàn.

Các output:

- `parking_status_cam1..4.json`: trạng thái ô theo camera;
- `vehicle_positions_cam1..4.json`: local/global ID và bbox từng camera;
- `global_vehicle_registry.json`: canonical Global ID và tọa độ map dùng chung.

### 1.13. Toàn bộ luồng của một xe cụ thể

Giả sử G#2 đang đỗ ở P056 thuộc cam4:

1. Xe đứng lâu nên MOG2 không còn tạo motion detection, nhưng binder vẫn giữ `P056.vehicle_id=2`.
2. Xe bắt đầu chạy; cam4 tạo tentative local #18.
3. `try_recover_id()` thấy bbox #18 nằm trong P056 mở rộng và appearance phù hợp.
4. `bind_external_id(cam4, 18, 2)` gắn local #18 với G#2 trước khi có ID mới.
5. Xe đi về biên trái cam4; manager mở handoff cam4 → cam3.
6. Cam3 tạo tentative local #7; `update_all_tracks()` ghép #7 với handoff của G#2.
7. Trong overlap có thể đồng thời tồn tại `(cam4,#18)→G#2` và `(cam3,#7)→G#2`.
8. `global_active_tracks` chỉ chọn một bbox mạnh nhất cho G#2.
9. Binder thấy G#2 rời P056 đủ 0.5 giây thì gỡ tracking override.
10. JSON/web vẫn chỉ có một xe mang ID 2 trong toàn bộ hành trình.

### 1.14. Đoạn nói mẫu trước hội đồng

“`main.py` là bộ điều phối chứ không phải một detector duy nhất. Mỗi frame được cắt thành bốn crop; bốn motion tracker sinh local track trong namespace riêng. Tentative track được thử phục hồi ID từ ô đỗ trước, sau đó CrossCameraManager mới handoff hoặc cấp Global ID. Tất cả ID được canonicalize, bbox được cộng crop offset về hệ pixel video gốc, rồi một binder dùng chung ghép xe với 69 ô và kiểm tra dừng mỗi frame. Parking vision chạy 2 Hz, JSON chạy 5 Hz. Cuối cùng atomic replace bảo đảm web không đọc file đang ghi dở.”

### 1.15. Phản biện thường gặp

**Hỏi:** “Tại sao vừa có local ID vừa có Global ID?”  
**Trả lời:** “Local ID chỉ duy nhất trong một tracker. Bốn camera đều có thể sinh local #1. Global ID do manager cấp trong một namespace chung và mới là ID đưa lên map.”

**Hỏi:** “Tại sao tentative chưa confirmed mà đã được dùng?”  
**Trả lời:** “Tentative chỉ được thử nhận ID cũ từ slot hoặc handoff; nó chưa được tự tạo Global ID mới. Điều này cứu xe chạy nhanh nhưng vẫn chặn detection nhiễu.”

**Hỏi:** “Tại sao lúc chia ROI thì trừ offset, lúc ghép map lại cộng?”  
**Trả lời:** “Global → local phải trừ gốc crop. Local → global phải cộng lại gốc crop. Đây là hai phép biến đổi ngược nhau.”

**Hỏi:** “Tại sao không chạy mọi thuật toán cùng FPS?”  
**Trả lời:** “Tracking cần chuỗi dày để tính chuyển động; trạng thái ô thay đổi chậm; web không cần ghi file mỗi frame. Tách ba nhịp giảm tải mà không mất bằng chứng động học.”

**Điểm mạnh:** luồng có thứ tự rõ ràng, một namespace Global ID, một binder toàn hệ thống và output an toàn.  
**Giới hạn:** bốn crop đang dùng chung hệ pixel của một video. Với camera vật lý, offset crop phải được thay bằng homography sang mặt phẳng bản đồ và cần đồng bộ timestamp giữa các camera.

---

## 2. Vẽ, lưu và biến đổi ROI

**File và thành phần:** `tools/ParkingSpacePicker_ve_js.py`; hàm `order_points()`, callback chuột và hàm lưu JSON. `tools/draw_direction_lines.py`; class `ROIDrawer`.

### Code trọng tâm

```python
def order_points(points):
    values = np.asarray(points, dtype=np.float32)
    center = np.mean(values, axis=0)
    angles = np.arctan2(values[:, 1] - center[1],
                        values[:, 0] - center[0])
    ordered = values[np.argsort(angles)]
    top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -top_left, axis=0)
    return [tuple(int(value) for value in point) for point in ordered]

mask = np.zeros((height, width), dtype=np.uint8)
cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
```

Người dùng chọn bốn đỉnh. `order_points()` lấy tâm trung bình, tính góc cực bằng `atan2`, sắp các điểm quanh tâm rồi xoay mảng để điểm gần góc trên-trái đứng đầu. Việc này tránh polygon tự cắt khi người vẽ bấm không đúng thứ tự.

File ROI lưu `imageWidth`, `imageHeight`. Khi video có độ phân giải khác:

\[
x'=x\frac{W_{frame}}{W_{ref}},\qquad
y'=y\frac{H_{frame}}{H_{ref}}
\]

`fillPoly` tạo mask đúng theo hình tứ giác, không dùng hình chữ nhật bao ngoài. Vì chỗ đỗ nhìn xiên thường là hình thang, rectangle sẽ lấy dư mặt đường và xe bên cạnh.

`draw_direction_lines.py` lại nhận đúng hai điểm `p1,p2` và lưu `junction_n`. Parking polygon trả lời “xe có nằm trong vùng hay không”; direction line trả lời “quỹ đạo có cắt qua vạch hay không”. Hai loại ROI không thể dùng thay nhau.

**Ví dụ:** P056 có bốn đỉnh tạo hình thang. Bbox xe lấn rectangle bao ngoài nhưng không lấn polygon thật sẽ không được tính vào P056.

**Lỗi được xử lý:** điểm ROI sai thứ tự; ROI lệch khi đổi độ phân giải; xe ở ô sát bên bị đếm nhầm do rectangle.

**Điểm mạnh:** JSON dễ chỉnh và web có thể dùng chung polygon.  
**Giới hạn:** scale tuyến tính chỉ đúng khi ảnh cùng góc nhìn; đổi vị trí camera phải hiệu chỉnh lại ROI.

**Câu nói mẫu:** “Em không xem ô đỗ là một rectangle. Em giữ polygon bốn đỉnh, scale theo độ phân giải và dùng mask thật, vì sai số ở ranh giữa hai ô chính là nơi rectangle gây lỗi nhiều nhất.”

**Phản biện:** “Sao không tự phát hiện vạch ô?”  
**Trả lời:** “Trong phiên bản này ROI được hiệu chỉnh một lần để ưu tiên tính ổn định. Tự phát hiện vạch là mô-đun có thể bổ sung, nhưng không thay đổi thuật toán ghép xe–ô phía sau.”

---

## 3. Phát hiện xe chuyển động: MOG2 kết hợp frame difference

**File và thành phần:** `src/techgar/motion_tracker.py`; class `MotionVehicleTracker`; hàm `_temporal_motion_mask()`, `_detect()`, `_suppress_duplicate_detections()`.

### Code trọng tâm

```python
reference = self._gray_history[0]
brightness_shift = float(np.median(gray.astype(np.float32)
                                   - reference.astype(np.float32)))
reference = cv2.convertScaleAbs(reference, alpha=1.0, beta=brightness_shift)
diff = cv2.absdiff(gray, reference)
_, temporal = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)

background = self.bg_sub.apply(frame)
_, background = cv2.threshold(background, 200, 255, cv2.THRESH_BINARY)
support = cv2.dilate(temporal, np.ones((17, 17), np.uint8))
motion = cv2.bitwise_and(background, support)
```

MOG2 học mô hình nền dài hạn với `history=700`, `varThreshold=32`. Nhược điểm của MOG2 là bóng, nhiễu nền và xe đỗ lâu có thể ảnh hưởng mô hình. Frame difference trả lời câu hỏi ngắn hạn: vùng này có thực sự thay đổi so với vài frame trước hay không.

Trước khi trừ ảnh, code lấy median chênh lệch sáng toàn frame và bù bằng `beta`. Median bền hơn mean trước một vùng xe sáng hoặc tối:

\[
\Delta L=\operatorname{median}(I_t-I_{ref}),\qquad
D=|I_t-(I_{ref}+\Delta L)|
\]

Mask cuối là giao giữa foreground MOG2 và vùng hỗ trợ temporal đã giãn. Sau đó code thực hiện opening, dilation, closing; tìm contour và lọc:

- diện tích quá nhỏ hoặc lớn hơn 22% frame;
- width/height không đủ;
- aspect ratio ngoài khoảng 0.25–5;
- số pixel chuyển động và `motion_ratio` không đạt.

### Chống “motion echo”

Một xe chậm có thể tạo hai silhouette: mép cũ và mép mới. `_same_motion_echo()` chỉ coi hai detection là một khi kích thước tương đương, histogram gần, khoảng cách/gap nhỏ; `_suppress_duplicate_detections()` giữ contour có bằng chứng diện tích mạnh hơn.

**Ví dụ:** ánh sáng toàn cảnh tăng 12 mức xám. Nếu trừ trực tiếp, gần như toàn frame thành foreground. Sau bù median, chỉ phần thay đổi cục bộ do xe còn lại.

**Lỗi được xử lý:** camera cố định nhưng ánh sáng dao động; bóng; nhiễu nhỏ; một xe tạo hai bbox.

**Điểm mạnh:** nhẹ, không cần GPU hay dữ liệu huấn luyện; phù hợp camera cố định.  
**Giới hạn:** không biết semantic “đây chắc chắn là ô tô”; camera rung hoặc nền chuyển động mạnh sẽ làm giảm độ tin cậy.

**Câu nói mẫu:** “MOG2 cho nền dài hạn, frame difference cho bằng chứng chuyển động ngắn hạn. Em lấy giao của hai nguồn thay vì tin riêng một mask, sau đó mới lọc hình học.”

**Phản biện:** “Đây có phải AI nhận diện xe không?”  
**Trả lời:** “Backend mặc định của `main.py` là computer vision cổ điển dựa trên chuyển động, không phải YOLO. YOLO/BoT-SORT là backend tùy chọn ở `single_camera.py`.”

---

## 4. Theo dõi xe trong một camera

**File và thành phần:** `src/techgar/motion_tracker.py`; `MotionVehicleTracker._new_kalman()`, `_assign()`, `_apply_detection()`, `process_frame()`. Dữ liệu track dùng `TrackedVehicle` và `TrackStatus` từ `src/techgar/vehicle_tracker.py`.

### 4.1 Kalman state và dự đoán

```python
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
```

Vector trạng thái:

\[
\mathbf{x}=[x,\ y,\ v_x,\ v_y]^T
\]

`predict()` dùng mô hình vận tốc gần như không đổi:

\[
\hat{\mathbf{x}}_t=F\mathbf{x}_{t-1}
\]

`correct()` nhận measurement tâm detection \([x_m,y_m]\) và kéo dự đoán về quan sát. Process noise cho vận tốc lớn hơn vị trí để xe có thể đổi hướng; measurement noise giảm rung bbox.

### 4.2 Đặc trưng appearance và ma trận chi phí

```python
distance = float(np.linalg.norm(np.subtract(
    predicted_point, detection["point"]
)))
if distance > max_distance:
    continue
iou = self._iou(predicted_box, detection["box"])
appearance_distance = cv2.compareHist(
    track.appearance, detection["hist"], cv2.HISTCMP_BHATTACHARYYA
)
costs[row, col] = (0.50 * (distance / max_distance)
                   + 0.30 * (1.0 - iou) + 0.20 * appearance_distance)
_, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.90)
```

Histogram HSV lấy hai kênh H và S, lưới \(16\times16\), sau đó normalize. Màu sắc ít nhạy với thay đổi độ sáng hơn histogram BGR. Ba lỗi được cân:

\[
E=0.50E_{distance}+0.30E_{IoU}+0.20E_{HSV}
\]

LAPJV giải bài toán gán toàn cục một-một. Nếu hai xe cùng gần một detection, thuật toán chọn cấu hình tổng chi phí nhỏ nhất thay vì ai duyệt trước được nhận trước.

### 4.3 Vòng đời track

- `tentative`: detection mới, chưa đủ bằng chứng chuyển động liên tục.
- `confirmed`: đủ `min_visible_count` và displacement tối thiểu.
- `lost`: track đã confirmed nhưng tạm mất detection.
- `expired`: vượt `lost_track_ttl`; được đưa vào bộ nhớ re-ID trong thời hạn.

Appearance của track được cập nhật EMA: 75% lịch sử + 25% detection mới. Khi tạo track, code ưu tiên re-ID từ track đã rời nếu histogram đủ gần và còn trong `reid_ttl`; nếu không mới tạo local ID.

**Ví dụ:** hai xe đi sát nhau. Detection A gần track 1 hơn nhưng IoU/màu giống track 2; detection B ngược lại. LAPJV xét cả ma trận, giảm tình trạng đổi ID chéo.

**Lỗi được xử lý:** bbox rung; mất detection ngắn; greedy assignment; ID switch khi hai xe gần nhau.

**Điểm mạnh:** kết hợp động học, hình học và appearance; không dựa vào một ngưỡng duy nhất.  
**Giới hạn:** HSV yếu với hai xe cùng màu; Kalman vận tốc hằng không mô hình hóa cua gấp hoàn hảo.

**Câu nói mẫu:** “Kalman không phát hiện xe; nó dự đoán vị trí track. Detection mới được ghép bằng tổng sai số khoảng cách, IoU và HSV, rồi LAPJV đảm bảo gán một-một.”

**Phản biện:** “Kalman có xác định xe đã dừng không?”  
**Trả lời:** “Không. Kalman chỉ hỗ trợ tracking. Kết luận dừng dùng tọa độ detection thực trong binder vì Kalman có thể còn quán tính vận tốc giả.”

---

## 5. Dự đoán xe chuyển camera

**File và thành phần:** `src/techgar/cross_camera_manager.py`; class `CrossCameraManager`; hàm `_velocity()`, `_outward_edge()`, `_upsert_handoff()`, `_candidate_cost()`.

### Code trọng tâm

```python
history = getattr(track, "history", [])
if len(history) < 2:
    return 0.0, 0.0
first_index = max(0, len(history) - 5)
first = history[first_index]
steps = max(1, len(history) - first_index - 1)
vx = float(track.cx - first[0]) / steps
vy = float(track.cy - first[1]) / steps

elapsed = max(0, frame_idx - entry.updated_at_frame)
predicted = (entry.last_world[0] + entry.velocity_world[0] * elapsed,
             entry.last_world[1] + entry.velocity_world[1] * elapsed)
```

Vận tốc được ước lượng từ tối đa năm điểm gần nhất để bớt nhạy với một frame rung:

\[
\mathbf v=\frac{\mathbf p_{last}-\mathbf p_{first}}
{frame_{last}-frame_{first}}
\]

`_outward_edge()` chỉ xét biên phù hợp với dấu vận tốc. Thời gian ước lượng đến biên:

\[
t_{edge}=\frac{d_{edge}}{|v_{axis}|}
\]

Nếu \(t_{edge}\le\text{lookahead\_frames}\), manager mở handoff sớm, không đợi bbox biến mất. Vị trí ở camera đích được dự đoán:

\[
\hat{\mathbf p}=\mathbf p_{last}+\mathbf v\Delta t
\]

Topology hiện tại:

| Camera nguồn | Biên ra | Camera đích |
|---|---|---|
| cam1 | phải / dưới | cam2 / cam3 |
| cam2 | trái / dưới | cam1 / cam4 |
| cam3 | trên / phải | cam1 / cam4 |
| cam4 | trên / trái | cam2 / cam3 |

Biên vào ở camera đích phải là biên đối diện biên ra. Ví dụ cam4 sang cam3: xe rời cạnh trái cam4 và vào cạnh phải cam3.

**Ví dụ:** G#7 ở cam4 còn cách biên trái 25 px, \(v_x=-8\) px/frame. \(t_{edge}\approx3.1\) frame, nằm trong lookahead nên hồ sơ handoff đã mở trước khi track cam4 mất.

**Lỗi được xử lý:** xe chạy nhanh nhảy sâu vào crop đích trước khi track tentative đủ confirmed; so với vị trí cuối cố định gây mất ID.

**Điểm mạnh:** dự đoán theo tốc độ và hướng; mở cửa sổ handoff sớm.  
**Giới hạn:** topology đang khai báo thủ công; camera vật lý thật cần vùng chuyển giao và phép biến đổi tọa độ được hiệu chỉnh.

**Câu nói mẫu:** “Em không đợi xe biến mất mới tìm ở camera kế. Khi thời gian dự đoán đến biên đủ nhỏ, hệ thống mở handoff, mang theo vận tốc, kích thước và HSV sang camera đích.”

**Phản biện:** “Xe dừng sát biên có bị handoff không?”  
**Trả lời:** “Không chỉ dựa vào khoảng cách biên. Vận tốc phải hướng ra và thời gian đến biên phải nằm trong lookahead; candidate đích còn phải qua các gate khác.”  

---

## 6. Cấp và phục hồi Global ID giữa các camera

**File và thành phần:** `src/techgar/cross_camera_manager.py`; `CrossCameraManager.update_all_tracks()`, `_match_pending_handoffs()`, `_candidate_cost()`, `canonical_global_id()`.

### Code trọng tâm

```python
if residual > self.prediction_radius:
    return None, "prediction_distance", details
appearance_distance = self._appearance_distance(
    getattr(track, "appearance", None), entry.appearance
)
size_distance = self._size_distance((track.w, track.h), entry.bbox_size)
direction = self._direction_cosine(track, entry.velocity_world)
direction_cost = 0.20 if direction is None else (1.0 - direction) * 0.5
cost = (0.55 * (residual / self.prediction_radius)
        + 0.30 * appearance_distance + 0.10 * size_distance
        + 0.05 * direction_cost)
_, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.92)
```

Manager lưu ánh xạ:

\[
(\text{camera\_id},\text{local\_track\_id})
\longrightarrow \text{global\_id}
\]

Local ID chỉ có nghĩa trong một tracker. Global ID mới là khóa được gửi lên bản đồ. Một track tentative ở camera đích được phép thử nhận Global ID từ handoff; nhưng nếu không match thì **chưa được cấp Global ID mới**. Chỉ track confirmed không còn khả năng phục hồi mới tạo ID mới.

Candidate phải qua các gate bảo thủ trước khi vào ma trận:

- camera đích đúng topology;
- xuất hiện ở đúng entry edge;
- sai số so với vị trí dự đoán không vượt `prediction_radius` (được truyền từ CLI `--handoff-prediction-radius`);
- cosine hướng không thấp hơn ngưỡng;
- histogram HSV và tỉ lệ kích thước không lệch quá mức.

Với candidate hợp lệ, chi phí là:

\[
E_h=0.55E_{position}+0.30E_{HSV}
+0.10E_{size}+0.05E_{direction}
\]

LAPJV ghép tất cả handoff với tất cả track mới theo batch. Vì vậy một G# không thể được cấp cho hai candidate, và một candidate không thể chiếm hai G#.

### Vùng overlap

Khi cùng một xe đồng thời xuất hiện trong hai crop kề nhau, manager so vị trí toàn cục, appearance và vùng giao thực của hai crop. Nếu đủ giống, hai local track cùng trỏ vào một Global ID thay vì tạo hai xe trên map.

**Ví dụ:** tentative L#12 ở cam3 xuất hiện sau khi G#7 mở handoff từ cam4. Dù L#12 chưa confirmed, nếu residual dự đoán 18 px, đúng hướng vào, HSV và size phù hợp, nó nhận ngay G#7. Nếu một candidate khác cạnh đó có tổng cost cao hơn, LAPJV không cho candidate đó chiếm G#7.

**Lỗi được xử lý:** một xe qua camera bị cấp ID mới; hai xe cạnh tranh một ID; cùng xe ở overlap xuất hiện hai lần.

**Điểm mạnh:** cấp lại ID trước confirmation nhưng chỉ sau nhiều gate; matching một-một toàn batch.  
**Giới hạn:** HSV không đủ phân biệt hai xe giống màu và kích thước; camera thật cần embedding Re-ID hoặc biển số nếu yêu cầu nhận dạng mạnh hơn.

**Câu nói mẫu:** “Tentative không đồng nghĩa với không dùng được. Em cho tentative tham gia nhận lại ID cũ, nhưng không cho nó tự sinh ID mới. Nhờ vậy xe nhanh không phải chờ confirmation ở camera đích.”

**Phản biện:** “Tại sao vị trí chiếm 0.55, HSV chỉ 0.30?”  
**Trả lời:** “Trong mô phỏng crop chung, tọa độ và hướng chuyển biên là bằng chứng mạnh nhất. HSV hỗ trợ chống nhầm nhưng hai xe cùng màu có thể giống nhau, nên không được chi phối toàn bộ quyết định.”

---

## 7. Gộp ID và chống một xe có hai ID

**File và thành phần:** `src/techgar/motion_tracker.py` và `src/techgar/cross_camera_manager.py`; các hàm `_same_motion_echo()`, `_suppress_duplicate_detections()`, `_same_camera_motion_duplicate()`, `_merge_global_ids()`. Đồng bộ ô ở `SlotVehicleBinder.remap_vehicle_ids()`.

### Code trọng tâm

```python
canonical_id = self._canonical_id(canonical_id)
duplicate_id = self._canonical_id(duplicate_id)
if canonical_id == duplicate_id:
    return
canonical_id, duplicate_id = min(canonical_id, duplicate_id), \
                             max(canonical_id, duplicate_id)
self._global_aliases[duplicate_id] = canonical_id
for key, global_id in list(self._local_to_global.items()):
    if self._canonical_id(global_id) == canonical_id \
            or global_id == duplicate_id:
        self._bind(key[0], key[1], canonical_id)
```

Hệ thống chống duplicate ở ba tầng:

1. **Detection:** hai contour là motion echo được gộp trước tracking.
2. **Local track:** hai track cùng camera, bbox gap nhỏ, appearance/kích thước/hướng phù hợp được xem xét gộp.
3. **Global ID:** nếu cùng một vật thể đã mang hai G#, ID nhỏ hơn được giữ làm canonical; ID lớn hơn trở thành alias.

Quy tắc ID nhỏ hơn không có nghĩa “nhỏ hơn luôn chính xác hơn về thị giác”. Nó là quy tắc xác định và ổn định: ID xuất hiện sớm hơn thường là lịch sử gốc, và chọn `min` giúp kết quả merge không phụ thuộc thứ tự duyệt.

\[
\operatorname{canonical}(G_4)=G_2,\qquad
G_4\rightarrow G_2\ \text{vĩnh viễn}
\]

`_canonical_id()` thực hiện path compression. Mọi lần tra G#4 sau đó đều trả G#2. Binder cũng remap vehicle state và slot binding; nếu merge tạo tình huống cùng canonical ID nằm ở hai ô, chỉ binding có bằng chứng overlap/thời gian dừng mạnh hơn được giữ, binding còn lại bị xóa.

Khi xuất map, manager chọn bbox có quan sát mạnh hơn: ưu tiên visible, sau đó area lớn hơn. Nó không vẽ cả hai bbox của cùng canonical G#.

**Ví dụ:** cùng xe chậm tạo G#2 và G#4. Sau merge, registry có alias 4→2; P056 đang giữ 4 được đổi sang 2; web chỉ nhận một entry G#2. Không có bước nào sau đó được “hồi sinh” G#4 thành xe mới.

**Lỗi được xử lý:** hai bbox của một xe; ID mới thắng ID cũ; một Global ID bị giữ ở hai ô sau merge.

**Điểm mạnh:** merge nhất quán xuyên suốt tracker, registry, binder và JSON.  
**Giới hạn:** merge quá mạnh có nguy cơ gộp hai xe thật đi quá sát; vì vậy code còn gate appearance, kích thước, gap và cosine vận tốc.

**Câu nói mẫu:** “Em không chỉ đổi nhãn hiển thị từ 4 thành 2. ID 4 bị retire thành alias, toàn bộ member local và binding ô được chuyển sang canonical ID 2; vì thế ID cũ không tái xuất hiện.”

**Phản biện:** “Tại sao không xóa hẳn record ID lớn?”  
**Trả lời:** “Alias phải được giữ để dữ liệu trễ hoặc binding cũ vẫn canonicalize đúng. Xóa record mà không giữ alias có thể làm G#4 được cấp lại hoặc tồn tại trong JSON cũ.”

---

## 8. Nhận diện trạng thái ô đỗ bằng multi-parameter voting

**File và thành phần:** `src/techgar/parking_detector.py`; class `ParkingDetector`, `TemporalSmoother`; hàm tạo LUT gamma và `detect()`.

### Code trọng tâm

```python
for dc in delta_clahe:
    c_val = max(0.1, base_clahe + dc)
    clahe = cv2.createCLAHE(
        clipLimit=c_val, tileGridSize=(clahe_grid, clahe_grid)
    )
    l_clahe = clahe.apply(l_channel)
    for dg in delta_gamma:
        g_val = max(0.1, base_gamma + dg)
        l_gamma = cv2.LUT(l_clahe, self._get_gamma_lut(g_val))
        blur = cv2.GaussianBlur(l_gamma, (3, 3), 1)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 16,
        )
```

Ảnh được đổi sang LAB và chỉ dùng kênh L. CLAHE tăng tương phản cục bộ, gamma làm thay đổi phân bố sáng. Code tạo 25 tổ hợp:

\[
5\ \gamma \times 5\ \text{CLAHE}=25\ \text{phiên bản}
\]

Mỗi phiên bản đi qua Gaussian blur, adaptive threshold, median blur và dilation. Với mỗi polygon:

\[
r_{fg}=\frac{\#\text{pixel foreground trong mask}}
{\#\text{pixel của polygon}}
\]

Nếu \(r_{fg}<0.20\), phiên bản đó bỏ phiếu “trống”. Code hiện dùng:

\[
\text{required\_votes}=25//2=12
\]

Phải nói đúng là **ngưỡng đồng thuận 12/25**, không gọi là “majority tuyệt đối”, vì majority nghiêm ngặt của 25 phải là ít nhất 13.

### Hai tầng recheck

- **Center-cluster rescue:** thu polygon về 40% quanh tâm. Nếu foreground tập trung rõ ở giữa, ô đang bị vote trống có thể bị lật lại thành occupied.
- **Canny edge recheck:** nếu mật độ cạnh trong ROI ít nhất khoảng 0.25, cũng có bằng chứng vật thể.
- **TemporalSmoother:** trạng thái mới phải lặp đủ số lần trước khi commit, tránh nhấp nháy đỏ–xanh.

**Ví dụ:** xe màu gần giống nền làm full foreground ratio thấp. Nhưng vùng trung tâm có cụm threshold và Canny edge cao; rescue sửa kết quả trống sai thành occupied.

**Lỗi được xử lý:** điều kiện sáng khác nhau; xe sáng/tối; ROI đổi trạng thái do một frame nhiễu; vật thể tập trung ở tâm nhưng tỷ lệ toàn ô thấp.

**Điểm mạnh:** nhẹ, giải thích được từng bước, không cần gán nhãn huấn luyện.  
**Giới hạn:** 25 biến thể không phải 25 mô hình độc lập; các vote tương quan với nhau và threshold phải kiểm định theo dữ liệu.

**Câu nói mẫu:** “Đây không phải ensemble nhiều AI model. Đây là multi-parameter consensus: cùng một pipeline được khảo sát dưới 25 cấu hình gamma–CLAHE, sau đó có center và edge recheck.”

**Phản biện:** “12/25 có phải majority không?”  
**Trả lời:** “Không phải majority tuyệt đối. Em gọi chính xác là ngưỡng đồng thuận của implementation. Đây cũng là tham số cần đưa vào ablation 11, 12, 13 hoặc tối ưu trên validation set.”

---

## 9. Ghép Global ID với ô đỗ

**File và thành phần:** `src/techgar/slot_vehicle_binder.py`; `SlotVehicleBinder._overlap_geometry()`, `_batch_match()`, `_bind_vehicle()`.

### Code trọng tâm

```python
x, y, w, h = bbox
vehicle_polygon = np.array(
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
    dtype=np.float32,
)
intersection_area, _ = cv2.intersectConvexConvex(
    vehicle_polygon, binding.polygon.astype(np.float32)
)
vehicle_overlap = float(intersection_area) / max(1.0, w * h)
slot_area = max(1.0, self._polygon_area(binding.polygon))
slot_overlap = float(intersection_area) / slot_area
score = 0.70 * vehicle_overlap + 0.20 * slot_overlap + 0.10 * center_proximity
_, row_to_col, _ = lapjv(costs, extend_cost=True, cost_limit=0.95)
```

Binder không chỉ hỏi tâm xe có nằm trong ROI. Bbox được đổi thành polygon và lấy diện tích giao chính xác bằng `intersectConvexConvex`.

\[
O_v=\frac{A_{intersection}}{A_{vehicle}},\qquad
O_s=\frac{A_{intersection}}{A_{slot}}
\]

Candidate hợp lệ khi:

\[
(\text{center trong ROI}\land O_v\ge0.35)
\quad\lor\quad O_v\ge0.60
\]

Điểm ghép:

\[
S=0.70O_v+0.20O_s+0.10C
\]

\(C\in[0,1]\) giảm theo khoảng cách từ tâm bbox tới tâm ô, chuẩn hóa bằng đường chéo slot. Vehicle overlap chiếm 0.70 vì mục tiêu chính là phần thân xe nằm trong ô; slot overlap hỗ trợ khi ô lớn hơn bbox; proximity phá hòa.

`_batch_match()` đổi score thành cost \(1-S\), có loyalty bonus 0.15 cho xe đang ở candidate cũ, rồi dùng LAPJV. Như vậy một Global ID chỉ được chọn một ô và một ô chỉ được chọn một Global ID.

**Ví dụ:** bbox G#2 lấn P056 72% và P058 31%. P056 qua gate và có score cao; P058 có thể không qua ngưỡng 0.35. G#2 chỉ được gán P056.

**Lỗi được xử lý:** bbox lấn hai ô; hai xe cạnh tranh một ô; thứ tự dictionary làm kết quả thay đổi.

**Điểm mạnh:** giao polygon thật và assignment một-một; score giải thích được.  
**Giới hạn:** bbox từ ảnh không phải footprint thật của xe trên mặt đất; góc camera xiên mạnh nên dùng ground polygon/homography sẽ chính xác hơn.

**Câu nói mẫu:** “Em phân biệt overlap theo xe và overlap theo ô. 70% trọng số dành cho phần thân xe nằm trong polygon, sau đó LAPJV giải toàn batch để không có một xe ở hai ô.”

**Phản biện:** “Sao không chỉ dùng bottom-center?”  
**Trả lời:** “Bottom-center tốt cho định vị mặt đất nhưng ở ranh ô nó bỏ qua kích thước xe. Area intersection cho biết toàn bbox thực sự phủ ô nào; center chỉ là một phần của gate.”

---

## 10. Xác định xe dừng bằng quan sát thực

**File và thành phần:** `src/techgar/slot_vehicle_binder.py`; `VehicleObservation`, `VehicleParkingState`, `SlotVehicleBinder._is_stopped()`.

### Code trọng tâm

```python
centers = np.asarray([item.center for item in samples], dtype=np.float64)
median_center = np.median(centers, axis=0)
radius = np.linalg.norm(centers - median_center, axis=1)
r95 = float(np.percentile(radius, 95))
net_displacement = float(np.linalg.norm(centers[-1] - centers[0]))
diagonals = [
    float(np.hypot(item.bbox[2], item.bbox[3]))
    for item in samples
]
bbox_diagonal = max(1.0, float(np.median(diagonals)))
stopped = (r95 <= max(3.0, self.stationary_radius_ratio * bbox_diagonal)
           and net_displacement <= max(5.0, self.stationary_drift_ratio * bbox_diagonal))
```

Cửa sổ dùng timestamp thật, dài khoảng một giây, tối thiểu tám mẫu và ít nhất 80% mẫu match cùng một slot. Đây là điểm quan trọng: không giả định video luôn đúng FPS khai báo.

Mặc định `stop_seconds=1.0`, nhưng binder chỉ commit `parked` khi tuổi candidate đạt `stop_seconds + stop_commit_grace_seconds`, tức khoảng 1.15 giây với grace 0.15 giây. Cửa sổ thống kê vẫn là một giây; 0.15 giây bổ sung chống việc vừa chạm ngưỡng đã chốt ngay.

Median center bền trước một bbox outlier. Bán kính rung ở percentile 95:

\[
r_{95}=P_{95}\left(\left\|\mathbf p_i-
\operatorname{median}(\mathbf p)\right\|\right)
\]

Độ trôi đầu–cuối:

\[
d_{net}=\|\mathbf p_{last}-\mathbf p_{first}\|
\]

Với \(D\) là median đường chéo bbox, xe dừng khi:

\[
r_{95}\le\max(3,\ 0.06D)
\quad\land\quad
d_{net}\le\max(5,\ 0.10D)
\]

\(r_{95}\) bắt rung cục bộ; \(d_{net}\) bắt trường hợp xe bò chậm theo một hướng. Chuẩn hóa theo đường chéo bbox giúp ngưỡng thích nghi khi xe gần hoặc xa camera.

Code dùng tâm bbox của **detection thực**, không dùng Kalman prediction. Kalman làm mượt nhưng có thể tiếp tục sinh vận tốc sau khi xe đã dừng hoặc tự dự đoán khi mất measurement; dùng nó để kết luận dừng sẽ tạo bằng chứng giả.

**Ví dụ:** xe dừng trong P056, bbox rung 2–3 px. Với \(D=100\), ngưỡng \(r_{95}\) là 6 px và drift là 10 px nên vẫn parked. Xe bò 14 px trong một giây sẽ trượt điều kiện net displacement.

**Lỗi được xử lý:** bbox rung; dừng giả dưới một giây; xe bò chậm; FPS video không ổn định.

**Điểm mạnh:** robust statistics và scale normalization; giải thích được.  
**Giới hạn:** ngưỡng một giây phù hợp demo nhưng phải hiệu chỉnh theo tốc độ bãi thật; detection quá thưa sẽ không đủ tám mẫu.

**Câu nói mẫu:** “Em không định nghĩa dừng bằng một frame có vận tốc bằng không. Em yêu cầu một cửa sổ thời gian, 80% quan sát cùng ô, rung 95% nhỏ và độ trôi đầu–cuối cũng nhỏ.”

**Phản biện:** “Xe kẹt lại một giây khi đang chạy qua ô có bị coi là đỗ?”  
**Trả lời:** “Có thể trở thành candidate nếu đứng trong ROI đủ lâu; đây là giới hạn ngữ nghĩa. Có thể tăng dwell time hoặc bổ sung điều kiện hướng/độ nằm gọn trong ô, và cần đánh giá trade-off bằng thực nghiệm.”

---

## 11. State machine đỗ xe

**File và thành phần:** `src/techgar/slot_vehicle_binder.py`; `VehicleParkingState`; `SlotVehicleBinder.update_tracks()`, `_bind_vehicle()`, `_release_vehicle()`, `try_recover_id()`.

### Code trọng tâm

```python
if state.parked_slot_id is not None:
    parked_slot = state.parked_slot_id
    binding = self._bindings.get(parked_slot)
    if slot_id == parked_slot:
        state.movement_state = "parked"
        state.outside_since = None
        if binding is not None:
            binding.tracking_state = "parked"
    else:
        if state.outside_since is None:
            state.outside_since = float(timestamp_s)
            state.movement_state = "exit_pending"
        elif timestamp_s - state.outside_since >= self.exit_seconds:
            self._release_vehicle(global_id, frame_idx)
```

State machine:

```text
moving → stop_candidate → parked
   ↑          │
   └──────────┘  nếu rời ROI hoặc còn di chuyển

parked → exit_pending → moving
   ↑          │
   └──────────┘  nếu xe quay lại ô
```

- `moving → stop_candidate`: xe match hợp lệ với một slot.
- `stop_candidate → parked`: cùng slot, `_is_stopped()` đúng và candidate mặc định đủ khoảng 1.15 giây.
- `stop_candidate → moving`: rời slot hoặc vị trí không ổn định.
- `parked → exit_pending`: cùng Global ID bắt đầu nằm ngoài ROI.
- `exit_pending → moving`: ngoài ROI liên tục mặc định 0.5 giây.
- `exit_pending → parked`: xe rung hoặc quay lại trước timeout.

Điểm đặc biệt là parked binding không bị xóa chỉ vì motion tracker không còn detection. Xe đứng lâu thường được MOG2 hấp thụ vào nền; nếu xóa theo TTL của tracker thì đúng lúc xe đỗ ổn định nhất, ID lại mất.

### Recovery khi xe rời ô

`try_recover_id()` chạy trước khi manager cấp Global ID mới. Nó mở rộng polygon đang giữ xe parked thêm 15%, kiểm tra tentative bbox/position, camera, kích thước và HSV nếu có. Candidate hợp lệ nhận lại Global ID đang lưu trong slot và chuyển sang `exit_pending`.

**Ví dụ:** G#2 đỗ P056 rồi biến mất khỏi motion tracker. Năm phút sau xe chạy ra, local tracker tạo tentative L#18. Bbox L#18 nằm trong vùng P056 mở rộng; binder trả G#2 trước khi manager tạo G#4.

**Lỗi được xử lý:** mất track khi xe đứng; xe rời ô bị cấp ID mới; ROI rung làm giải phóng binding tức thời.

**Điểm mạnh:** trạng thái nghiệp vụ “đang đỗ” tồn tại độc lập với vòng đời motion track.  
**Giới hạn:** nếu hai xe đổi chỗ sát nhau trong cùng vùng recovery và appearance giống nhau, cần Re-ID mạnh hơn hoặc biển số.

**Câu nói mẫu:** “Track biến mất không đồng nghĩa xe biến mất. Khi trạng thái đã parked, ô giữ ID vô thời hạn; chỉ cùng ID được quan sát rời ROI đủ 0.5 giây mới gỡ binding.”

**Phản biện:** “Giữ vô thời hạn có làm ô bị kẹt occupied không?”  
**Trả lời:** “Tracking override có thể kẹt nếu không bao giờ quan sát được xe rời. Đây là đánh đổi ưu tiên không báo trống sai. Hệ thống thực tế nên có sự kiện camera health, thao tác quản trị hoặc detector tĩnh mạnh để hòa giải dài hạn.”

---

## 12. Hợp nhất vision và tracking

**File và thành phần:** `src/techgar/slot_vehicle_binder.py`; `SlotBinding.decision_source`, `_sync_result()`, `update_vision()`, `to_json()`.

### Code trọng tâm

```python
@property
def decision_source(self) -> str:
    if self.vision_occupied and self.tracking_occupied:
        return "vision_and_tracking"
    if self.tracking_occupied:
        return "tracking_override"
    if self.vision_occupied:
        return "vision"
    return "none"

def _sync_result(self, binding):
    binding.occupied = bool(binding.vision_occupied or binding.tracking_occupied)
```

Công thức quyết định cuối:

\[
occupied_{final}=vision_{occupied}\lor tracking_{occupied}
\]

Hai nguồn có vai trò không đối xứng:

- Vision nhận cả xe đã đỗ trước khi bật hệ thống, khi chưa có Global ID.
- Tracking chứng minh một xe có ID đã đi vào và dừng, nên được phép sửa lỗi “trống giả”.
- Tracking không có quyền tự biến một ô vision đang báo occupied thành trống.
- Khi xe rời, tracking chỉ gỡ override; kết quả quay về vision.

| Vision | Tracking | Final | ID | `decision_source` |
|---|---|---|---|---|
| 0 | 0 | 0 | null | none |
| 1 | 0 | 1 | null | vision |
| 0 | 1 | 1 | G# | tracking_override |
| 1 | 1 | 1 | G# | vision_and_tracking |

**Ví dụ:** thuật toán threshold báo P056 trống do xe màu giống nền. G#2 dừng ổn định trong polygon, nên `tracking_occupied=True`; final vẫn occupied và web nhận `vehicle_id=2`.

**Lỗi được xử lý:** xe có ID đỗ nhưng vision báo xanh; xe tồn tại từ trước không có track; tracking mất tạm thời.

**Điểm mạnh:** giải thích được nguồn quyết định và ưu tiên an toàn, giảm false free.  
**Giới hạn:** phép OR có thể giữ false occupied của vision; để sửa cả hai chiều cần nguồn bằng chứng rời ô mạnh và chính sách conflict có kiểm định.

**Câu nói mẫu:** “Em không thay ParkingDetector bằng tracker. Em hợp nhất hai nguồn: vision cho trạng thái nền, tracking chỉ tạo occupied override khi có bằng chứng xe đi vào và dừng.”

**Phản biện:** “Tại sao tracking không được sửa đỏ sai thành xanh?”  
**Trả lời:** “Vì không thấy track không chứng minh ô trống: xe có thể đã đỗ trước khi hệ thống chạy hoặc bị hấp thụ vào nền. Trong bài toán bãi xe, báo trống sai thường nguy hiểm hơn báo bận sai.”

---

## 13. Xác định hướng xe qua junction

**File và thành phần:** `src/techgar/direction_detector.py`; `ROILine`, `PendingDecision`, `DirectionDetector`; hàm `_segments_intersect()`, `_compute_direction()`, `update()`.

Mô-đun này hiện được gọi trong `single_camera.py`. `main.py` bốn crop chưa xuất event rẽ trái/phải; nó chỉ dùng cosine hướng trong gate handoff. Khi thuyết trình phải phân biệt hai loại “hướng” này.

### Code trọng tâm

```python
vec_after = after_end - after_start
vec_before = after_start - before_pt
len_before = np.linalg.norm(vec_before)
len_after = np.linalg.norm(vec_after)
cross = vec_before[0] * vec_after[1] - vec_before[1] * vec_after[0]
cos_angle = np.dot(vec_before, vec_after) / (len_before * len_after)
angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
if angle_deg < self.angle_threshold:
    return "STRAIGHT"
return "TURN_LEFT" if cross > 0 else "TURN_RIGHT"
```

Đầu tiên, hai đoạn thẳng được kiểm tra giao bằng quy tắc CCW. Khi quỹ đạo cắt junction, detector chưa kết luận ngay mà giữ một `PendingDecision`: điểm trước vạch và chuỗi điểm sau vạch.

Góc đổi hướng lấy từ dot product:

\[
\theta=\cos^{-1}
\left(\frac{\mathbf a\cdot\mathbf b}
{\|\mathbf a\|\|\mathbf b\|}\right)
\]

Nếu góc nhỏ hơn ngưỡng, xe đi thẳng. Nếu không, dấu cross product 2D:

\[
z=a_xb_y-a_yb_x
\]

xác định trái/phải theo quy ước trục ảnh. Code chỉ kết luận sau đủ `decision_frames` và quãng đường `min_after_distance`, đồng thời bỏ dịch chuyển rung dưới khoảng 2 px.

**Ví dụ:** xe cắt `junction_2`; ba điểm sau vạch tạo vector lệch 38°. Cross \(z>0\), detector trả `TURN_LEFT`. Nếu mới chỉ có một điểm sau vạch, trạng thái còn pending.

**Lỗi được xử lý:** kết luận hướng từ hai điểm quá gần; jitter quanh vạch tạo crossing giả; một track cắt vạch nhiều lần.

**Điểm mạnh:** hình học rõ ràng, không cần học máy.  
**Giới hạn:** trái/phải phụ thuộc quy ước trục ảnh và hướng đặt vạch; camera phối cảnh mạnh nên vector pixel có thể méo.

**Câu nói mẫu:** “Vạch junction chỉ kích hoạt sự kiện. Hướng không lấy tại đúng frame cắt vạch mà chờ đủ quãng đường phía sau, rồi dùng dot product cho góc và cross product cho trái/phải.”

**Phản biện:** “Tại sao không lấy dấu \(v_x\) hoặc \(v_y\)?”  
**Trả lời:** “Dấu một trục chỉ đúng với đường thẳng ngang/dọc. Dot và cross product làm việc với junction ở góc bất kỳ và đo trực tiếp độ đổi hướng.”

---

## 14. Backend YOLO/BoT-SORT tùy chọn

**File và thành phần:** `src/techgar/vehicle_tracker.py`; class `VehicleTracker`; `single_camera.py`; cấu hình `config/botsort_parking_reid.yaml`.

### Code trọng tâm

```python
kwargs = {
    "persist": True,
    "tracker": self.tracker_config,
    "classes": self.class_ids,
    "conf": self.confidence,
    "iou": self.iou,
    "imgsz": self.imgsz,
    "verbose": False,
}
if self.device:
    kwargs["device"] = self.device
results = self.model.track(frame, **kwargs)
```

`VehicleTracker` gọi `Ultralytics.YOLO.track`. `persist=True` giữ trạng thái tracker giữa các frame; `classes` lọc nhóm phương tiện; `conf` và `iou` điều khiển detection/NMS. BoT-SORT trong YAML bật Re-ID, với các threshold high/low/new track, buffer và appearance matching.

`single_camera.py` cho chọn `--backend motion|yolo`, mặc định vẫn là motion. **`main.py` hiện không chạy YOLO mặc định.** Đây phải được nói rõ để không tạo ấn tượng sai rằng kết quả demo bốn crop đến từ deep learning.

Nếu có homography, `cv2.perspectiveTransform` có thể đổi ground point từ pixel ảnh sang mặt phẳng bản đồ. Điều đó là bước cần thiết khi thay bốn crop bằng camera vật lý thật.

**Ví dụ:** camera có xe đứng yên sẵn. Motion backend không phát hiện vì không chuyển động; YOLO vẫn nhận class car. Tuy nhiên Global ID xuyên camera vẫn cần manager hoặc Re-ID cấp hệ thống, không tự có chỉ vì dùng YOLO.

**Lỗi được xử lý:** semantic detection cho xe đứng; camera nền động tốt hơn motion-only; Re-ID appearance giàu hơn HSV.

**Điểm mạnh:** nhận diện phương tiện theo class và có BoT-SORT/Re-ID.  
**Giới hạn:** cần model/dependency, tài nguyên tính toán và benchmark trên góc top-down; ID BoT-SORT vẫn thường local theo từng stream.

**Câu nói mẫu:** “Dự án có adapter YOLO/BoT-SORT để mở rộng, nhưng em phân biệt rõ backend tùy chọn với pipeline đang demo. Global ID xuyên camera là tầng riêng, không được giải quyết tự động chỉ bằng `persist=True`.”

**Phản biện:** “Vậy tại sao không dùng YOLO luôn?”  
**Trả lời:** “Motion backend chứng minh pipeline chạy nhẹ trên camera cố định. YOLO là hướng nâng cấp cho camera thật và xe tĩnh; quyết định chọn backend phải dựa trên benchmark accuracy, latency và phần cứng, không phải tên mô hình.”

---

## 15. JSON và web map

**File và thành phần:** `main.py`; `_save_json_atomic()`; `CrossCameraManager.to_json()`; `SlotVehicleBinder.to_json()`.

### Code trọng tâm

```python
def _save_json_atomic(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except PermissionError:
        pass
```

Ghi trực tiếp một file JSON đang được web đọc có thể làm web bắt đúng lúc file mới ghi một nửa và parse lỗi. Cơ chế atomic write ghi hoàn chỉnh vào `.tmp`, sau đó `replace` trên cùng filesystem.

Các nhóm output:

- `parking_status_cam*.json`: trạng thái polygon, `occupied`, `vehicle_id`, nguồn quyết định.
- `vehicle_positions_cam*.json`: quan sát xe theo camera.
- `global_vehicle_registry.json`: canonical G#, member local, handoff/merge và telemetry.

Trước khi xuất, mọi ID đi qua `canonical_global_id()`. Nếu cùng G# có nhiều observation do overlap, code chọn observation mạnh nhất để vẽ hoặc hợp nhất vị trí; web chỉ nhận một entry cho mỗi canonical G#.

JSON vị trí hiện là pixel video gốc. Web demo cùng layout có thể scale trực tiếp. Khi dùng camera thật, output nên thêm `world_x`, `world_y` sau homography và metadata đơn vị.

**Ví dụ:** cam3 và cam4 cùng thấy G#2 trong overlap. Registry giữ hai member local phục vụ debug, nhưng payload map chỉ có khóa `"2"`, không có hai xe độc lập.

**Lỗi được xử lý:** JSON hỏng giữa lúc ghi; alias ID xuất hiện lại; web vẽ trùng xe ở overlap.

**Điểm mạnh:** tương thích web, atomic, có telemetry giải thích.  
**Giới hạn:** file polling phù hợp demo; production nên dùng WebSocket/message broker, schema version và timestamp/sequence để xử lý dữ liệu trễ.

**Câu nói mẫu:** “Điểm cuối của pipeline là invariant: một canonical Global ID chỉ có một entry trên map. Atomic replace đảm bảo web hoặc đọc bản cũ hoàn chỉnh, hoặc bản mới hoàn chỉnh.”

**Phản biện:** “JSON file có phải realtime không?”  
**Trả lời:** “Ở mức demo 5 Hz là near-real-time. Với triển khai thật, thuật toán giữ nguyên nhưng transport nên đổi sang WebSocket hoặc broker; JSON schema vẫn tái sử dụng.”

---

# Kịch bản nói liền mạch khoảng 20 phút

Phần dưới đây có thể đọc và tập gần như nguyên văn; khi trình chiếu code chỉ mở đúng file được nhắc đến.

## Mở thẳng vào code — 0:00 đến 2:00

“Em bắt đầu trực tiếp từ `main.py`. Một frame video được chia thành bốn crop để mô phỏng bốn góc nhìn; em không xem đây là bốn camera vật lý. Mỗi crop có tracker local riêng, nhưng Global ID manager và parking binder là dùng chung.

Tracker chạy mọi frame. Parking detector chỉ chạy 2 Hz vì trạng thái ô thay đổi chậm, JSON xuất 5 Hz cho web. Thứ tự xử lý là recovery ID từ ô đỗ, handoff giữa camera, canonicalize ID, ghép ID với slot, rồi mới xuất JSON. Tọa độ local được cộng offset crop để quay về hệ pixel video gốc.”

## Motion và local tracking — 2:00 đến 6:00

“Trong `motion_tracker.py`, em dùng MOG2 cho foreground dài hạn và frame difference cho chuyển động ngắn hạn. Trước khi trừ frame, em bù thay đổi sáng toàn cục bằng median. Mask cuối là giao giữa hai bằng chứng, sau đó morphology, contour và lọc hình học.

Một detection có tâm, bbox và histogram HSV 16×16. Kalman giữ trạng thái \([x,y,v_x,v_y]\), predict trước khi ghép và correct khi có measurement. Chi phí ghép gồm 50% khoảng cách, 30% IoU và 20% HSV. LAPJV giải toàn bộ ma trận một-một, tránh greedy làm đổi ID.

Track đi qua tentative, confirmed, lost và expired. Một lớp chống motion echo gộp hai contour gần nhau của cùng xe trước khi chúng sinh hai track.”

## Handoff và Global ID — 6:00 đến 10:00

“Local ID chỉ duy nhất trong một camera. Manager lưu ánh xạ cặp camera–local ID sang Global ID. Từ tối đa năm điểm gần nhất, em tính vector vận tốc và thời gian đến biên. Nếu xe đang hướng ra và sắp chạm biên trong lookahead, handoff được mở sớm.

Vị trí dự đoán ở camera đích bằng vị trí cuối cộng vận tốc nhân số frame đã trôi. Candidate phải đúng camera kề, đúng entry edge, đúng hướng, đủ gần vị trí dự đoán, tương đồng HSV và kích thước. Chi phí handoff là 55% vị trí, 30% HSV, 10% kích thước, 5% hướng; LAPJV tiếp tục đảm bảo một-một.

Tentative track được thử lấy ID cũ ngay, nhưng chưa được tự tạo Global ID mới. Nếu cùng xe bị sinh hai G#, ID nhỏ hơn làm canonical, ID lớn hơn thành alias vĩnh viễn; binding ô và JSON cũng remap theo.”

## Nhận diện ô và ghép xe–ô — 10:00 đến 14:00

“Trong `parking_detector.py`, ảnh sang LAB, kênh L qua 25 tổ hợp gamma–CLAHE. Mỗi cấu hình dùng adaptive threshold và tính tỷ lệ foreground trong polygon. Code dùng ngưỡng đồng thuận 12/25; em không gọi đây là majority tuyệt đối và cũng không gọi là ensemble nhiều AI model.

Nếu kết quả trống chưa chắc, center-cluster và Canny edge kiểm tra lại; temporal smoother chống nhấp nháy. Song song, binder đổi bbox xe thành polygon, tính giao thật với ô. Điểm ghép là 70% vehicle overlap, 20% slot overlap, 10% độ gần tâm, và LAPJV đảm bảo một xe–một ô.”

## Dừng, state machine và hợp nhất — 14:00 đến 18:00

“Xe không được coi là đỗ chỉ vì nằm trong ROI. Binder giữ cửa sổ timestamp một giây, tối thiểu tám mẫu, 80% thuộc cùng slot. Em dùng median center, bán kính rung percentile 95 và độ trôi đầu–cuối, chuẩn hóa bằng đường chéo bbox. Đây là detection thật, không phải Kalman prediction.

State đi từ moving sang stop candidate rồi parked. Khi parked, mất motion track không làm mất binding. Chỉ khi cùng ID được quan sát ngoài ROI 0.5 giây mới chuyển qua exit pending và release. Khi xe bắt đầu rời mà local tracker tạo track mới, ROI mở rộng 15% giúp lấy lại ID parked trước khi cấp ID mới.

Kết quả cuối là vision occupied OR tracking occupied. Vision giữ khả năng nhận xe đã đỗ trước lúc bật máy; tracking chỉ sửa trường hợp vision báo trống sai, không tự sửa ô đỏ thành xanh.”

## Hướng, backend tùy chọn và output — 18:00 đến 20:00

“Direction detector dùng vạch junction. Giao đoạn thẳng kích hoạt pending; sau đủ frame và quãng đường, dot product cho góc, cross product cho trái hoặc phải.

Dự án có `VehicleTracker` dùng YOLO/BoT-SORT trong `single_camera.py`, nhưng `main.py` hiện dùng motion tracker. Em tách bạch điều này vì YOLO không tự giải quyết Global ID xuyên camera.

Cuối cùng, mọi ID được canonicalize trước khi ghi. JSON được ghi vào file tạm rồi atomic replace. Invariant phía web là mỗi canonical Global ID chỉ có một entry trên map. Với camera thật, bước tiếp theo là homography để đổi pixel từng camera sang cùng mặt phẳng tọa độ thực.”

---

# Phụ lục phản biện sâu

## A. Invariant hệ thống phải luôn giữ

1. Một `(camera_id, local_track_id)` chỉ trỏ tới một canonical Global ID.
2. Một handoff không thể ghép cho hai track; một track không thể nhận hai handoff.
3. Một canonical Global ID không nằm trong hai ô cùng lúc.
4. Một ô không chứa hai canonical Global ID.
5. Tentative không match ID cũ thì chưa được tạo Global ID mới.
6. Global ID đã retire không được xuất trở lại; mọi lookup phải qua canonicalization.
7. Parked binding không bị xóa chỉ vì motion track mất.
8. Tracking không được biến `vision_occupied=True` thành `occupied=False`.
9. Web chỉ nhận một entry cho mỗi canonical Global ID.

## B. Phân biệt bốn khái niệm thường bị hỏi

| Khái niệm | Phạm vi | Ai tạo | Có gửi web làm ID xe không? |
|---|---|---|---|
| Detection | Một frame | MOG2/frame diff hoặc YOLO | Không |
| Local track ID | Một camera/stream | Local tracker | Không dùng làm ID cuối |
| Global ID | Toàn hệ thống camera | CrossCameraManager | Có |
| Canonical Global ID | Sau merge/alias | CrossCameraManager | Có, đây là khóa cuối |

## C. Vì sao dùng ba lần LAPJV?

- Local tracking: track ↔ detection.
- Cross-camera: pending handoff ↔ tentative/confirmed track đích.
- Parking binding: Global ID ↔ parking slot.

Ba bài toán khác dữ liệu và cost, nhưng cùng ràng buộc một-một. Không thể dùng một ma trận chung vì chúng xảy ra ở các tầng ngữ nghĩa và thời điểm khác nhau.

## D. Các thư viện chính

| Thư viện | Hàm/thành phần dùng | Vai trò |
|---|---|---|
| OpenCV | MOG2, KalmanFilter, morphology, contour, HSV/LAB, CLAHE, adaptiveThreshold, Canny, intersectConvexConvex, pointPolygonTest | Xử lý ảnh và hình học |
| NumPy | vector, median, percentile, norm, histogram arrays | Tính toán số |
| `lap` | `lapjv` | Assignment một-một tối ưu |
| `json`, `pathlib` | dump, temp/replace | Giao tiếp dữ liệu |
| Ultralytics, tùy chọn | `YOLO.track` | Detection/BoT-SORT backend |

## E. Những điều tuyệt đối không nói sai

- Không nói bốn crop là bốn camera vật lý.
- Không nói tọa độ global hiện tại là mét; nó là pixel video gốc.
- Không nói `main.py` đang chạy YOLO mặc định.
- Không nói Kalman xác định xe dừng.
- Không nói 25 cấu hình là 25 AI model.
- Không nói 12/25 là majority tuyệt đối.
- Không nói HSV bảo đảm nhận dạng duy nhất một phương tiện.
- Không nói Global ID xuyên camera đã giải quyết hoàn chỉnh camera vật lý chưa calibrate.

## F. Các câu hỏi phản biện khó và trả lời thẳng

### 1. “Độ chính xác của hệ thống là bao nhiêu?”

Không trả lời bằng cảm giác từ video demo. Câu trả lời đúng:

“Phiên bản code đã định nghĩa rõ event và invariant, nhưng độ chính xác phải báo trên bộ ground truth. Em cần đo precision/recall/F1 cho occupied, IDF1/HOTA và số ID switch cho tracking, cùng latency/FPS. Nếu chưa có số liệu đủ lớn, em không khẳng định một phần trăm tùy ý.”

### 2. “Vì sao đây là nghiên cứu chứ không chỉ ghép OpenCV?”

“Đóng góp cần được chứng minh ở thiết kế hợp nhất: motion+appearance+assignment, predictive handoff, canonical merge, và OR fusion giữa vision với tracking-stationary evidence. Giá trị nghiên cứu chỉ mạnh khi em có baseline và ablation cho từng tầng, không phải vì dùng nhiều hàm OpenCV.”

### 3. “Baseline cần so gì?”

- Parking vision đơn lẻ.
- Center-point ROI thay cho polygon intersection.
- Greedy matching thay LAPJV.
- Handoff vị trí cuối thay predictive handoff.
- Không appearance.
- Không duplicate merge.
- Không tracking override.
- YOLO/BoT-SORT tùy chọn nếu phần cứng cho phép.

### 4. “Ablation cần bỏ lần lượt gì?”

| Ablation | Kỳ vọng đo |
|---|---|
| Bỏ median brightness compensation | false motion khi đổi sáng tăng |
| Chỉ MOG2, bỏ frame difference | foreground nhiễu/xe tĩnh khác |
| Bỏ HSV | ID switch khi xe gần nhau tăng |
| Greedy thay LAPJV | conflict một-một tăng |
| Bỏ lookahead/prediction | fast handoff miss tăng |
| Bỏ duplicate merge | một xe hai G# tăng |
| Chỉ center-in-ROI | gán nhầm ô biên tăng |
| Bỏ \(r_{95}\) hoặc net displacement | rung hoặc xe bò bị kết luận dừng |
| Bỏ tracking override | false-free của parking vision tăng |
| Bỏ recovery từ parked slot | xe rời ô nhận ID mới tăng |

### 5. “Metrics cụ thể là gì?”

- Ô đỗ: precision, recall, F1 cho class occupied; đặc biệt báo false-free rate.
- Tracking: IDF1, ID precision/recall, HOTA, số ID switches, track fragmentation.
- Cross-camera: handoff recall, handoff precision, thời gian nhận lại ID.
- Binder: slot assignment accuracy, false park event, park confirmation delay, false release.
- Hệ thống: FPS, end-to-end latency, CPU/RAM/GPU.

### 6. “Tại sao ưu tiên không ghép nhầm?”

Ghép nhầm làm lịch sử của hai xe bị trộn và rất khó khôi phục. Bỏ lỡ handoff chỉ tạo candidate/new ID có thể merge sau. Vì vậy các gate handoff bảo thủ là lựa chọn thiết kế, nhưng trade-off phải được báo bằng precision và recall.

### 7. “Nếu hai xe cùng màu và cùng kích thước?”

HSV không đủ. Hệ thống hiện còn vị trí, hướng, topology và thời gian. Với camera thật đông xe, cần deep Re-ID embedding, biển số hoặc multi-view calibration; đây là giới hạn đã xác định, không nên che giấu.

### 8. “Camera rung thì sao?”

Motion backend giả định camera cố định. Camera rung làm cả MOG2 và frame difference phát foreground rộng. Giải pháp thực tế: gá camera chắc, video stabilization/GMC, vùng mask loại cây/cờ, hoặc chuyển YOLO backend.

### 9. “Tại sao OR fusion mà không học trọng số?”

OR thể hiện chính sách ưu tiên an toàn và dễ audit: bằng chứng tracking đủ mạnh có quyền sửa false free. Fusion học trọng số có thể tốt hơn nhưng cần dataset conflict được gán nhãn; hiện tại OR là baseline rõ ràng để so sánh.

### 10. “Khi camera vật lý thật, phần nào giữ lại?”

Giữ local tracker interface, predictive handoff, batch LAPJV, Global ID canonicalization, binder, state machine và JSON schema. Thay crop offset bằng homography/ground-plane calibration, định nghĩa topology/transition zone thật và bổ sung đồng bộ timestamp.

## G. Checklist mở code khi thuyết trình

1. `main.py`: mở vòng lặp `run_detection()`, chỉ đúng thứ tự recovery → manager → binder.
2. `motion_tracker.py`: mở `_temporal_motion_mask()`, `_assign()`, `_new_kalman()`.
3. `cross_camera_manager.py`: mở `EDGE_ADJACENCY`, `_candidate_cost()`, `_match_pending_handoffs()`.
4. `parking_detector.py`: mở vòng 5×5 và `required_votes`.
5. `slot_vehicle_binder.py`: mở `_overlap_geometry()`, `_is_stopped()`, `_sync_result()`.
6. `direction_detector.py`: mở dot/cross product.
7. `vehicle_tracker.py`: chỉ mở nếu hội đồng hỏi YOLO; nói rõ là backend tùy chọn.

## H. Câu kết kỹ thuật

“Điểm nổi bật của code không nằm ở một detector duy nhất. Hệ thống xây chuỗi bằng chứng có kiểm soát: chuyển động tạo detection, Kalman và LAPJV tạo local track, handoff dự đoán tạo Global ID, canonical merge bảo vệ tính duy nhất, polygon intersection và thống kê một giây chứng minh xe đã dừng, sau đó mới tạo tracking override cho trạng thái ô. Mỗi tầng đều có gate, telemetry và giới hạn rõ ràng để có thể đo bằng baseline và ablation.”

## I. Bảng tham số mặc định trọng yếu của `main.py`

| Nhóm | Tham số CLI | Mặc định | Được dùng ở đâu |
|---|---|---:|---|
| Nhịp | `--parking-fps` | 2.0 Hz | Chu kỳ gọi `ParkingDetector.detect()` |
| Nhịp | `--json-fps` | 5.0 Hz | Chu kỳ ghi các file JSON |
| Vision | `--base-gamma` | 2.4 | Tâm của năm gamma trong `main.py` |
| Vision | `--base-clahe` | 2.0 | Tâm của năm CLAHE clip limit |
| Vision | `--ratio-thr` / `--edge-thr` | 0.20 / 0.25 | Foreground vote / Canny recheck |
| Local track | `--min-visible-count` | 4 | Số quan sát để xét confirm |
| Local track | `--lost-track-ttl` | 90 frame | Giữ track lost |
| Local track | `--motion-min-area` | 900 px² | Lọc contour trong `main.py` |
| Local track | `--motion-max-distance` | 180 px | Gate khoảng cách Kalman |
| Handoff | `--handoff-ttl` | 45 frame | Tuổi hồ sơ handoff |
| Handoff | `--handoff-lookahead-frames` | 16 | Mở handoff sớm |
| Handoff | `--handoff-prediction-radius` | 90 px | Gate residual dự đoán |
| Handoff | `--handoff-appearance-threshold` | 0.45 | Gate Bhattacharyya HSV |
| Handoff | `--handoff-min-direction-cosine` | 0.25 | Gate hướng |
| Slot | `--slot-stop-seconds` / `--slot-exit-seconds` | 1.0 / 0.5 s | Cửa sổ dừng / rời |
| Slot | `--slot-min-vehicle-overlap` | 0.35 | Gate khi tâm nằm trong ROI |
| Slot | `--slot-strong-vehicle-overlap` | 0.60 | Gate mạnh không cần tâm |
| Slot | `--slot-stationary-radius-ratio` | 0.06 | Ngưỡng (r_{95}/D) |
| Slot | `--slot-stationary-drift-ratio` | 0.10 | Ngưỡng (d_{net}/D) |
| Slot | `--slot-recovery-expand-ratio` | 0.15 | Mở rộng ROI recovery |
| Slot | `--slot-release-grace` | 90 frame | Giữ pending ID sau release |

**Lưu ý audit code:** `--slot-bind-confirmations` được truyền vào binder để tương thích API cũ, nhưng state machine hiện chốt bằng timestamp, `_is_stopped()` và `stop_commit_grace_seconds`; biến `bind_confirmations` chưa tham gia nhánh quyết định hiện tại. Khi thuyết trình không được nói “hệ thống đang dùng hai confirmation của detector để bind xe”, trừ khi code được sửa để thật sự sử dụng biến này.
