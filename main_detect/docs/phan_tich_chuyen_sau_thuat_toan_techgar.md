# PHÂN TÍCH CHUYÊN SÂU THUẬT TOÁN TECHGAR

> Tài liệu này tập trung vào logic thuật toán và code đang có trong `main_detect`. Phần chính mô tả đúng pipeline mặc định của `main.py`: bốn vùng ảnh được cắt từ cùng một video và mỗi vùng dùng motion tracker. Đây chưa phải bốn camera vật lý độc lập. Backend YOLO/BoT-SORT chỉ được trình bày ở phụ lục.

## Cách sử dụng tài liệu

- **Phần 1 — 0 đến 10 phút:** hiểu toàn bộ khái niệm và vai trò từng thuật toán.
- **Phần 2 — 10 đến 32 phút:** mở code, lần theo lời gọi hàm và giải thích công thức.
- **Phần 3 — 32 đến 40 phút:** kết luận kỹ thuật, giới hạn và trả lời phản biện.
- **Phụ lục:** dùng khi hội đồng hỏi sâu về YOLO/BoT-SORT hoặc tham số.

Tài liệu dùng một tình huống xuyên suốt: xe thật đang mang **Global ID `G#2`**, đi từ `cam4` sang `cam3`, vào ô `P056`, dừng, mất detection chuyển động, rồi chạy ra và nhận lại đúng `G#2`.

---

# PHẦN 1 — KHÁI NIỆM TỔNG QUAN VÀ TOÀN BỘ THUẬT TOÁN

## 1. Bản đồ toàn hệ thống

```text
Frame video
    ↓
Chia thành bốn góc nhìn mô phỏng
    ↓
Phát hiện vùng đang chuyển động trong từng góc nhìn
    ↓
Theo dõi xe và tạo Local ID riêng cho từng góc nhìn
    ↓
Dự đoán xe sắp sang góc nhìn kề và bàn giao Global ID
    ↓
Gộp các ID trùng về một Canonical Global ID
    ↓
Nhận diện trạng thái từng ô đỗ bằng xử lý ảnh
    ↓
Đối chiếu bbox xe với polygon ô đỗ
    ↓
Kiểm tra xe có dừng ổn định hay chỉ chạy ngang qua
    ↓
Hợp nhất bằng chứng vision và tracking
    ↓
Ghi JSON cho web map
```

Không có một “siêu thuật toán” làm tất cả. TechGAR là một chuỗi thuật toán nhỏ, mỗi tầng trả lời một câu hỏi:

| Tầng | Câu hỏi được trả lời |
|---|---|
| ROI | Khu vực nào là ô đỗ, khu vực nào là vạch ngã rẽ? |
| Phát hiện chuyển động | Pixel nào đang thay đổi giống một vật thể di chuyển? |
| Local tracking | Vùng chuyển động ở frame này có phải track đã có ở frame trước không? |
| Global ID | Track ở hai góc nhìn có cùng đại diện cho một xe thật không? |
| Parking vision | Bên trong polygon ô đỗ có dấu hiệu của xe không? |
| Vehicle–slot binding | Xe Global ID nào thực sự nằm trong ô nào? |
| Stop detection | Xe đang đỗ hay chỉ đi ngang qua ô? |
| Fusion | Kết luận cuối cùng của ô là trống hay có xe? |
| Output | Làm sao web chỉ nhận một xe cho mỗi Canonical Global ID? |

## 2. Từ điển thuật ngữ bắt buộc hiểu

| Thuật ngữ | Cách hiểu trong dự án |
|---|---|
| Frame | Một ảnh đơn lấy từ video/camera tại một thời điểm. |
| ROI — Region of Interest | Vùng ảnh mà hệ thống quan tâm. ROI ô đỗ là polygon; ROI junction là đoạn thẳng. |
| Polygon | Đa giác nhiều điểm. Ô đỗ hiện dùng bốn đỉnh nhưng detector chấp nhận polygon nói chung. |
| Mask | Ảnh nhị phân chỉ cho phép tính toán bên trong vùng ROI. |
| Detection | Một quan sát bbox vừa được tìm thấy trong đúng một frame; chưa có lịch sử. |
| Track | Chuỗi detection qua nhiều frame được xem là cùng một vật thể. |
| Bbox — bounding box | Hình chữ nhật `(x, y, width, height)` bao quanh vùng chuyển động/xe. |
| Local ID | ID do một tracker riêng lẻ cấp; chỉ duy nhất trong góc nhìn đó. |
| Global ID | ID dùng chung cho toàn hệ thống góc nhìn. Đây là ID web cần dùng. |
| Canonical Global ID | ID cuối cùng còn hiệu lực sau khi gộp ID trùng. |
| Tentative | Track tạm thời: mới xuất hiện, chưa đủ bằng chứng để tạo xe mới trên map. |
| Confirmed | Track đã đủ số lần nhìn thấy và quãng đường di chuyển để được công nhận. |
| Lost | Track đã confirmed nhưng frame hiện tại không còn detection. |
| Expired | Track mất lâu hơn thời hạn giữ; bị đưa khỏi danh sách đang theo dõi. |
| Prediction — dự đoán | Ước lượng vị trí tiếp theo từ vị trí và vận tốc trước đó. Đây là thuật ngữ dùng trong Kalman và handoff. |
| Predicate — điều kiện đúng/sai | Biểu thức kiểm tra trả `True/False`, ví dụ “residual có nhỏ hơn prediction radius không?”. Predicate không phải tên thuật toán dự đoán. |
| Handoff — bàn giao ID | Chuyển Global ID đang có ở góc nhìn nguồn sang local track mới ở góc nhìn đích. |
| Re-ID — nhận lại danh tính | Đối chiếu một quan sát mới với thông tin cũ để tái sử dụng ID, thay vì cấp ID mới. |
| Appearance — đặc trưng ngoại hình | Trong motion backend hiện tại là histogram màu HSV của bbox. |
| IoU | Tỉ lệ phần giao trên phần hợp của hai bbox. |
| Overlap — độ chồng lấn | Tỉ lệ diện tích giao giữa bbox xe và polygon ô. |
| Gate — điều kiện loại | Điều kiện bắt buộc. Không đạt thì cặp đối chiếu bị loại trước khi tính ghép tối ưu. |
| Đối tượng đang được xét | Track chưa có ID hoặc chưa có ô, đang được kiểm tra với hồ sơ cũ. Trong tài liệu tiếng Anh thường gọi là candidate. |
| LAPJV | Thuật toán ghép một-một toàn cục với tổng chi phí nhỏ; tránh ai được duyệt trước thì thắng trước. |
| State machine — máy trạng thái | Mô hình chỉ cho phép đối tượng chuyển giữa các trạng thái hợp lệ. |
| Telemetry | Các sự kiện chẩn đoán như `handoff_opened`, `handoff_rejected`, `vehicle_stopped_in_slot`. |

### Bốn khái niệm “nhận lại ID” không được trộn lẫn

1. **Local Re-ID:** `_create_or_reid()` so detection mới với `_exited_tracks` trong cùng một motion tracker.
2. **Handoff:** `CrossCameraManager` mang Global ID từ góc nhìn nguồn sang góc nhìn đích.
3. **Parking recovery:** `SlotVehicleBinder.try_recover_id()` lấy lại Global ID đang lưu trong ô khi xe bắt đầu chạy.
4. **BoT-SORT Re-ID:** backend YOLO tùy chọn dùng đặc trưng ngoại hình của tracker; không tự thay thế Global ID manager xuyên camera.

## 3. ROI: cách hệ thống biết “đâu là ô đỗ” và “đâu là ngã rẽ”

### 3.1. ROI ô đỗ dạng polygon

**Mục tiêu:** mô tả đúng hình dạng ô đỗ thay vì dùng một hình chữ nhật cố định.

**Đầu vào:** bốn lần click của người cấu hình trên một frame chuẩn.

**Thuật toán:** `order_points()` lấy tâm trung bình của bốn điểm, tính góc của từng điểm quanh tâm bằng `arctan2`, sắp xếp theo góc, rồi xoay danh sách để điểm trên-trái đứng đầu. Nhờ vậy người dùng không bắt buộc click đúng thứ tự.

**Đầu ra:** JSON chứa `imageWidth`, `imageHeight`, danh sách `slots`, polygon, center và ID `P001`, `P002`, ...

**Khi chạy:** `ParkingDetector._compute_rois()` scale polygon từ độ phân giải lúc vẽ sang độ phân giải frame hiện tại, tạo bounding rectangle nhỏ quanh polygon và dùng `cv2.fillPoly()` tạo mask nhị phân.

**Ví dụ:** P056 được vẽ ở video chuẩn 1100×720. Nếu nguồn mới là 2200×1440, mọi tọa độ nhân đôi trước khi tạo mask.

**Lỗi được xử lý:** ô bị xiên theo phối cảnh; người dùng click sai thứ tự; video chạy ở độ phân giải khác lúc vẽ.

**Điểm mạnh:** trực quan, không cần huấn luyện dữ liệu; polygon khớp ô nghiêng tốt hơn rectangle.

**Giới hạn:** ROI là cấu hình thủ công. Camera đổi vị trí thì phải hiệu chỉnh lại.

### 3.2. ROI junction dạng đoạn thẳng

ROI junction không đại diện một vùng diện tích. Nó là đoạn thẳng `p1 → p2` đặt tại ngã rẽ. Khi đoạn chuyển động của xe từ frame trước đến frame hiện tại cắt đoạn junction, hệ thống bắt đầu theo dõi hướng sau khi qua vạch.

| ROI ô đỗ | ROI junction |
|---|---|
| Polygon bốn đỉnh | Đoạn thẳng hai điểm |
| Dùng mask và diện tích giao | Dùng kiểm tra hai đoạn thẳng giao nhau |
| Trả lời xe có nằm trong ô không | Trả lời xe đã đi qua mốc quyết định hướng chưa |

### 3.3. Bốn góc nhìn và hai hệ tọa độ

`CameraSimulator` cắt frame tại `mid_x`, `mid_y`:

```text
cam1 (trên-trái)  | cam2 (trên-phải)
------------------+------------------
cam3 (dưới-trái)  | cam4 (dưới-phải)
```

Đây là **bốn crop của cùng một video**, không phải bốn camera vật lý. Mặc định `overlap=0`, tức không cố ý lặp pixel ở biên.

Một điểm local trong crop được đưa về hệ pixel video gốc:

\[
x_{global}=x_{local}+x_{crop},\qquad
y_{global}=y_{local}+y_{crop}
\]

Phép cộng offset này hoạt động vì bốn crop cùng lấy từ một frame. Với camera vật lý, phải thay bằng homography hoặc phép hiệu chỉnh mặt phẳng.

## 4. Phát hiện xe chuyển động bằng hai nguồn bằng chứng

### 4.1. MOG2: mô hình nền dài hạn

`cv2.createBackgroundSubtractorMOG2(history=700, varThreshold=32, detectShadows=True)` học phân bố màu của nền qua thời gian. Pixel khác nền đủ lớn trở thành foreground — tiền cảnh.

MOG2 trả lời: “Pixel này có khác mô hình nền dài hạn không?”. Nó nhẹ và phù hợp camera cố định, nhưng có thể nhầm bóng, thay đổi sáng và nhiễu nén.

### 4.2. Frame difference: thay đổi ngắn hạn

Frame difference lấy ảnh xám hiện tại trừ ảnh tham chiếu cách vài frame. Trước khi trừ, code bù thay đổi sáng toàn ảnh:

\[
\Delta L=\operatorname{median}(I_t-I_{ref})
\]

\[
D=\left|I_t-(I_{ref}+\Delta L)\right|
\]

Median được dùng thay mean vì một vùng xe rất sáng hoặc tối ít kéo lệch median toàn frame.

Frame difference trả lời: “Pixel này có thực sự thay đổi trong vài frame gần đây không?”. Xe đã đứng yên không còn thỏa điều kiện này.

### 4.3. Giao hai mask và morphology

Mask cuối chỉ giữ pixel vừa khác nền MOG2, vừa có hỗ trợ chuyển động ngắn hạn:

```text
mask cuối = MOG2 foreground AND vùng hỗ trợ temporal motion
```

Sau đó dùng:

- **Opening:** xóa các chấm nhiễu nhỏ.
- **Dilation:** nới vùng chuyển động để nối các phần bị đứt.
- **Closing:** lấp lỗ và nối các vùng gần nhau.
- **Contour:** tìm đường biên từng vùng trắng.
- **Bounding rectangle:** chuyển contour thành bbox.

Các bbox bị loại nếu diện tích quá nhỏ, lớn hơn 22% frame, width/height không đủ, aspect ratio ngoài 0.25–5, số pixel chuyển động hoặc tỉ lệ chuyển động không đạt.

**Đầu ra:** mỗi detection gồm bbox, điểm giữa cạnh đáy, diện tích contour và histogram HSV.

**Ví dụ:** ánh sáng toàn bãi tăng 12 mức xám. MOG2 có thể đánh dấu nhiều vùng, nhưng bù median làm frame difference giữ chủ yếu thay đổi cục bộ do xe.

**Điểm mạnh:** không cần GPU hoặc dữ liệu huấn luyện.

**Giới hạn:** đây là phát hiện chuyển động, không phải nhận diện ngữ nghĩa “đây chắc chắn là ô tô”. Camera rung hoặc nền chuyển động mạnh sẽ gây nhiễu.

### 4.4. Chống một xe tạo hai silhouette cũ–mới

Khi trừ hai thời điểm, một xe có thể tạo hai hình: vị trí cũ vừa biến mất và vị trí mới vừa xuất hiện. `_same_motion_echo()` chỉ coi hai detection là cùng một xe khi:

- tỉ lệ diện tích tối thiểu 0.45;
- khoảng cách histogram Bhattacharyya không quá 0.22;
- tâm đủ gần theo kích thước bbox;
- khoảng hở giữa hai bbox không quá 14 px.

`_suppress_duplicate_detections()` duyệt bbox lớn trước và bỏ silhouette yếu hơn. `_is_echo_of_matched_track()` còn kiểm tra detection chưa được ghép có nằm gần vị trí lịch sử cách bốn quan sát của một track vừa nhận detection hay không.

**Lỗi được xử lý:** một xe sinh hai local track và sau đó hai Global ID.

**Giới hạn:** hai xe thật cùng màu, gần nhau và tương đương kích thước vẫn là trường hợp khó; vì vậy còn tầng gộp ID phía manager.

## 5. Tracking trong một góc nhìn

### 5.1. Detection khác track như thế nào?

- Detection chỉ tồn tại ở một frame.
- Track chứa ID, lịch sử tọa độ, bbox, tuổi, số lần nhìn thấy, số frame mất, histogram và Kalman filter.

Tracker phải quyết định detection mới thuộc track cũ nào. Nếu quyết định sai, hệ thống đổi ID dù detector vẫn tìm đúng xe.

### 5.2. Kalman dự đoán vị trí

State — vector trạng thái:

\[
\mathbf{x}=[x,y,v_x,v_y]^T
\]

Measurement — quan sát thực tế chỉ có:

\[
\mathbf{z}=[x_m,y_m]^T
\]

Ma trận chuyển trạng thái giả định vận tốc gần như không đổi:

\[
\hat{x}_t=x_{t-1}+v_{x,t-1},\qquad
\hat{y}_t=y_{t-1}+v_{y,t-1}
\]

- `predict()` dự đoán trước khi thấy detection mới.
- `correct()` dùng điểm detection thật để kéo state về quan sát.

Kalman không phát hiện xe và cũng không kết luận xe dừng. Nó chỉ giúp dự đoán track nên xuất hiện ở đâu.

### 5.3. Ba bằng chứng để nối detection với track

1. **Khoảng cách:** detection gần điểm Kalman dự đoán đến đâu.
2. **IoU:** bbox detection chồng lên bbox dự đoán bao nhiêu.
3. **HSV:** màu sắc detection khác histogram của track bao nhiêu.

Chi phí:

\[
E_{local}=0.50E_{distance}+0.30E_{IoU}+0.20E_{HSV}
\]

Chi phí nhỏ là giống nhau hơn. Nếu khoảng cách vượt `max_distance`, cặp bị loại. `lapjv(..., cost_limit=0.90)` ghép toàn bộ track với toàn bộ detection theo quan hệ một-một.

**Vì sao không duyệt tuần tự?** Nếu track 1 và track 2 cùng gần detection A, cách greedy có thể cho track 1 lấy A trước, khiến track 2 buộc nhận detection sai. LAPJV tìm cấu hình có tổng chi phí nhỏ nhất cho cả ma trận.

### 5.4. Vòng đời track

```text
detection mới → tentative
tentative đủ bằng chứng → confirmed
confirmed mất detection → lost
lost quá lost_track_ttl → expired → lưu vào _exited_tracks
```

Trong `main.py`, mặc định cần `min_visible_count=4` và displacement tối thiểu 12 px để confirmed. Tentative được phép nhận ID cũ từ handoff nhưng không được tự tạo Global ID mới.

### 5.5. Local Re-ID

Khi detection không nối được với track đang sống, `_create_or_reid()` so histogram của nó với `_exited_tracks`. Hồ sơ phải còn trong `reid_ttl` và khoảng cách màu phải nhỏ hơn 0.18. Hồ sơ gần nhất được khôi phục, Kalman được tạo lại và ID local cũ được dùng tiếp.

**Điểm hay:** giảm tạo local ID mới sau mất detection dài trong cùng một tracker.

**Giới hạn:** chỉ dựa HSV nên có thể nhầm hai xe cùng màu. Đây không phải bằng chứng định danh mạnh như biển số hoặc embedding Re-ID đã huấn luyện.

## 6. Local ID, Global ID và Canonical Global ID

### 6.1. Vì sao Local ID không đủ?

Mỗi `MotionVehicleTracker` có `_next_id` riêng. Bốn tracker đều có thể tạo local #1:

```text
cam1 local #1 → xe A
cam2 local #1 → xe B
```

Nếu đưa local ID thẳng lên web, hai xe khác nhau sẽ trùng khóa. `CrossCameraManager` tạo ánh xạ:

\[
(camera\_id,local\_track\_id)\rightarrow global\_id
\]

Ví dụ:

```text
(cam4, local #7) → G#2
(cam3, local #12) → G#2
```

Hai dòng trên không có nghĩa hai xe khác nhau dùng chung ID. Chúng nói hai quan sát local cùng đại diện một xe thật.

### 6.2. Canonical Global ID

Nếu lỗi trước đó đã tạo `G#4` cho xe vốn là `G#2`, `_merge_global_ids()` giữ ID nhỏ hơn:

```text
G#4 → alias của G#2
```

`_canonical_id(4)` từ đó luôn trả 2. Hàm có path compression: nếu alias tạo thành chuỗi `8 → 4 → 2`, nó rút gọn thành `8 → 2` và `4 → 2`.

**Invariant:** ID đã nghỉ không được sống lại. Mọi lookup phải qua canonicalization.

## 7. Handoff dự đoán: bàn giao ID trước khi xe biến mất

### 7.1. Vì sao phải mở sớm?

Nếu đợi track cam4 biến mất rồi mới tìm ở cam3, xe chạy nhanh có thể đã đi sâu vào cam3 và local track mới đã được confirmed/cấp ID mới. Hệ thống hiện tại ước lượng vận tốc từ tối đa năm điểm gần nhất:

\[
\mathbf{v}=\frac{\mathbf{p}_{last}-\mathbf{p}_{first}}{N-1}
\]

Với mỗi hướng vận tốc, `_outward_edge()` tính thời gian đến cạnh:

\[
t_{edge}=\frac{d_{edge}}{|v_{axis}|}
\]

Nếu `t_edge ≤ lookahead_frames` — mặc định 16 frame — một `HandoffEntry` được tạo hoặc cập nhật.

### 7.2. Topology hiện tại

| Nguồn | Cạnh ra | Đích | Cạnh vào tương ứng |
|---|---|---|---|
| cam1 | right | cam2 | left |
| cam1 | bottom | cam3 | top |
| cam2 | left | cam1 | right |
| cam2 | bottom | cam4 | top |
| cam3 | top | cam1 | bottom |
| cam3 | right | cam4 | left |
| cam4 | top | cam2 | bottom |
| cam4 | left | cam3 | right |

### 7.3. Ba tầng kiểm tra handoff

Tài liệu gọi “đối tượng đang được xét” là local track chưa có Global ID ở góc nhìn đích.

#### Tầng 1 — Đúng đường chuyển camera

- `target_cam` phải đúng bảng topology.
- Track phải xuất hiện trong hành lang gần cạnh vào đối diện cạnh ra.
- Track quá sâu ngoài hành lang bị loại với lý do `outside_entry_corridor`.

#### Tầng 2 — Đúng chuyển động dự đoán

Vị trí dự đoán:

\[
\hat{\mathbf p}=\mathbf p_{last}+\mathbf v\Delta frame
\]

Sai số vị trí:

\[
r=\|\mathbf p_{observed}-\hat{\mathbf p}\|
\]

`r` phải không vượt `prediction_radius=90 px`. Nếu track đích đã có đủ lịch sử để đo vận tốc, cosine hướng phải không nhỏ hơn 0.25:

\[
\cos\theta=\frac{\mathbf v_{source}\cdot\mathbf v_{target}}
{\|\mathbf v_{source}\|\|\mathbf v_{target}\|}
\]

Nếu track mới chỉ có một điểm, chưa tính được hướng; code không loại ngay mà gán một mức phạt nhỏ 0.20.

#### Tầng 3 — Đúng ngoại hình và kích thước

- Khoảng cách histogram HSV không vượt 0.45.
- Sai khác kích thước không vượt 0.90.
- Kích thước chỉ là bằng chứng hỗ trợ vì bbox có thể bị cắt ở biên crop.

Sau các điều kiện loại, chi phí handoff:

\[
E_h=0.55E_{position}+0.30E_{HSV}+0.10E_{size}+0.05E_{direction}
\]

LAPJV ghép toàn bộ hồ sơ bàn giao với toàn bộ track chưa có Global ID cùng lúc. Một G# không thể bị hai track lấy; một track không thể nhận hai G#.

### 7.4. Ví dụ cam4 sang cam3

G#2 còn cách cạnh trái cam4 24 px, vận tốc `vx=-8 px/frame`. Thời gian tới cạnh xấp xỉ 3 frame, nhỏ hơn lookahead 16 nên hồ sơ được mở. Hai frame sau, cam3 có tentative local #12:

- đúng `cam3` và đúng cạnh vào phải;
- cách vị trí dự đoán 18 px, nhỏ hơn 90;
- HSV và kích thước phù hợp;
- chưa đủ điểm tính hướng.

Local #12 nhận ngay `G#2` dù chưa confirmed. Nếu không đủ bằng chứng, nó tiếp tục tentative chứ chưa được tự tạo G# mới.

**Điểm mạnh:** cứu ID xe chạy nhanh mà vẫn ưu tiên không ghép nhầm.

**Giới hạn:** topology hiện viết tay và tọa độ chung dựa crop offset. Camera vật lý cần vùng chuyển giao và hiệu chỉnh riêng.

## 8. Vùng chồng lấn và gộp ID trùng

### 8.1. Nhìn thấy đồng thời ở hai crop

`_match_simultaneous_overlap()` chỉ hoạt động khi hai crop thật sự có phần diện tích giao. Nó kiểm tra camera kề, điểm toàn cục nằm gần phần giao, khoảng cách vị trí và HSV. Nếu phù hợp, local track mới trỏ vào Global ID đã có ở crop kia.

Mặc định `main.py --overlap 0`, nên hai crop không có phần giao diện tích; cơ chế này chủ yếu dành cho cấu hình chẩn đoán có overlap. Handoff dự đoán là cơ chế chính ở mặc định.

### 8.2. Hai bbox trong cùng góc nhìn

`_same_camera_motion_duplicate()` kiểm tra:

- tỉ lệ diện tích ít nhất 0.30;
- HSV không vượt ngưỡng appearance;
- khoảng hở bbox không quá 30 px;
- khoảng cách tâm theo kích thước;
- nếu cả hai có vận tốc đo được thì không được đi ngược hướng rõ ràng.

Nếu hai bbox đã có hai Global ID, `_merge_all_nearby_active_duplicates()` gọi `_merge_global_ids()` và giữ ID nhỏ hơn. `_merge_recently_lost_duplicates()` còn xử lý trường hợp bbox thứ hai sống lâu hơn bbox cũ.

## 9. Nhận diện trạng thái ô đỗ bằng xử lý ảnh nhiều tham số

### 9.1. Đây không phải ensemble nhiều mô hình AI

`ParkingDetector.detect()` chạy cùng một chuỗi xử lý ảnh với 25 cấu hình:

```text
5 mức gamma × 5 mức CLAHE = 25 phiếu
```

- LAB tách kênh sáng L khỏi màu.
- CLAHE tăng tương phản cục bộ.
- Gamma điều chỉnh vùng sáng/tối phi tuyến.
- Gaussian blur giảm nhiễu cao tần.
- Adaptive threshold tạo mask theo vùng lân cận.
- Median blur bỏ chấm muối tiêu.
- Dilation nối các nét foreground.

Trong mỗi polygon:

\[
foreground\_ratio=\frac{\text{số pixel trắng trong mask ROI}}
{\text{diện tích mask ROI}}
\]

Nếu ratio nhỏ hơn `ratio_thr=0.20`, cấu hình đó bỏ một phiếu “trống”.

Code dùng:

```python
required_votes = total_combinations // 2
is_free = empty_votes[i] >= required_votes
```

Với 25 tổ hợp, `25 // 2 = 12`. Vì đa số tuyệt đối của 25 phải là 13, phải gọi đúng đây là **ngưỡng voting 12/25**, không phải majority tuyệt đối.

Trong `main.py`, `base_gamma` mặc định là 2.4. Constructor `ParkingDetector` có mặc định 2.8, nhưng giá trị đó bị `main.py` truyền đè khi chạy pipeline bốn crop.

### 9.2. Center-cluster rescue

Nếu voting cho rằng ô trống, code thu polygon về 40% quanh tâm và đo foreground tập trung ở giữa. Nếu `center_ratio ≥ 0.05` và mạnh bất thường so với toàn ô, kết quả được cứu thành có xe. Tầng này nhắm tới xe sáng hoặc xe có foreground tập trung ở thân giữa.

### 9.3. Canny edge recheck

Nếu ô vẫn bị xem là trống, hệ thống chạy Canny `50–150`, dilation và đo mật độ cạnh trong ROI. Nếu `edge_ratio ≥ edge_thr=0.25`, ô được đổi thành có xe.

### 9.4. Temporal smoothing

`TemporalSmoother` chỉ đổi trạng thái confirmed sau khi cùng kết quả lặp đủ `smoothing_frames`, mặc định `main.py` là 5 lần ParkingDetector chạy. Vì parking mặc định 2 Hz, độ trễ lý thuyết có thể khoảng 2.5 giây trước khi vision đổi trạng thái ổn định.

**Điểm mạnh:** không phụ thuộc model; chịu được một khoảng thay đổi sáng/độ tương phản; nhận được xe đã đỗ trước khi bật hệ thống.

**Giới hạn:** ngưỡng phụ thuộc góc camera, mặt nền và ROI; xử lý 25 biến thể tốn hơn một cấu hình; không biết vehicle ID.

## 10. Ghép Global ID với ô đỗ bằng hình học

### 10.1. Vì sao không chỉ kiểm tra một điểm?

Một điểm tâm có thể lọt vào ô trong khi phần lớn bbox còn ở lối đi. Binder chuyển bbox thành polygon bốn đỉnh và dùng `cv2.intersectConvexConvex()` tính diện tích giao.

\[
O_v=\frac{A_{intersection}}{A_{vehicle}},\qquad
O_s=\frac{A_{intersection}}{A_{slot}}
\]

Xe đủ điều kiện xét với ô nếu:

\[
(center\_inside\land O_v\ge0.35)\lor O_v\ge0.60
\]

Điểm gần tâm ô:

\[
C=\max\left(0,1-\frac{d(center_{vehicle},center_{slot})}
{diagonal_{slot}}\right)
\]

Điểm tổng:

\[
S=0.70O_v+0.20O_s+0.10C
\]

Binder đổi thành chi phí `1-S` và dùng LAPJV. Nếu xe đang gắn với cùng ô, score được thưởng 0.15 để giảm nhảy qua lại giữa hai ô sát nhau.

**Invariant:** một Global ID chỉ nhận một ô; một ô chỉ nhận một Global ID trong một lượt ghép.

## 11. Xác định xe dừng bằng tọa độ detection thật

### 11.1. Cửa sổ quan sát

Mỗi Global ID có deque `VehicleObservation(timestamp, center, bbox, slot_id, overlap)`. Điều kiện đầu:

- tối thiểu 8 mẫu;
- thời lượng quan sát ít nhất `0.90 × stop_seconds`;
- ít nhất 80% mẫu cùng thuộc một ô;
- mặc định `stop_seconds=1.0`.

### 11.2. Hai thước đo ổn định

Lấy median center của các quan sát. Với mỗi điểm, tính khoảng cách tới median; `r95` là phân vị 95%:

\[
r_{95}=P_{95}(\|p_i-\operatorname{median}(p)\|)
\]

Độ trôi đầu–cuối:

\[
d_{net}=\|p_{last}-p_{first}\|
\]

Chuẩn hóa theo đường chéo bbox trung vị `D`:

\[
r_{95}\le\max(3,0.06D)
\]

\[
d_{net}\le\max(5,0.10D)
\]

Binder còn đợi tuổi trạng thái chờ đạt `stop_seconds + 0.15 giây` trước khi cam kết parked, tránh chớp đỗ đúng lúc track mất ở biên ROI.

### 11.3. Vì sao không dùng vị trí Kalman để kết luận dừng?

Kalman mang vận tốc nội tại và vẫn có thể dự đoán vật thể tiếp tục trôi khi detection đã dừng/mất. Stop detector dùng bbox đo thật; với LOST track, `main.py` giữ bbox đo thật cuối cùng, không lấy vị trí Kalman dự đoán mới.

**Điểm mạnh:** chịu được bbox rung 2–3 px và ít phụ thuộc độ phân giải.

**Giới hạn:** motion tracker có thể mất xe quá sớm; code khắc phục một phần bằng lost TTL và giữ binding parked.

## 12. Máy trạng thái đỗ xe

```text
moving
  │ xe giao hợp lệ với một ROI
  ▼
stop_candidate
  │ dừng ổn định đủ thời gian
  ▼
parked
  │ cùng ID được thấy ngoài ROI
  ▼
exit_pending
  ├─ quay lại ROI → parked
  └─ ngoài ROI ≥ 0.5 s → moving và gỡ tracking override
```

- `moving`: chưa có bằng chứng xe đang đỗ.
- `stop_candidate`: xe đang nằm trong ô nhưng chưa chứng minh dừng.
- `parked`: binder giữ `tracking_occupied=True` và `vehicle_id`.
- `exit_pending`: xe có dấu hiệu rời ô nhưng chưa gỡ ngay để chống rung biên.

Track motion biến mất khi xe đã parked không làm binding biến mất. Đây là chủ đích: xe đứng yên không còn motion nhưng ô vẫn phải giữ ID.

## 13. Phục hồi Global ID từ ô đỗ

Khi xe parked bắt đầu chạy, motion tracker có thể tạo local track mới. Trước khi manager cấp Global ID mới, `main.py` gọi `try_recover_id()`:

1. Đổi bbox local về tọa độ toàn video bằng crop offset.
2. Mở rộng polygon của ô đang giữ ID thêm 15% quanh centroid.
3. Kiểm tra tâm hoặc overlap bbox với polygon mở rộng.
4. Nếu có appearance cũ, yêu cầu khoảng cách HSV không quá 0.50.
5. Nếu nhiều ô phù hợp, lấy ô có tâm gần nhất.
6. Trả Global ID đang lưu trong ô và chuyển state sang `exit_pending`.
7. `manager.bind_external_id()` gắn ID đó vào local track trước handoff/cấp ID.

Trong chế độ bốn crop, `MotionVehicleTracker` được tạo với `slot_binder=None`. Recovery ô đỗ được làm ở tầng Global ID như trên; không tái sử dụng nhánh local `slot_binder` của `_create_or_reid()`.

## 14. Hợp nhất vision và tracking

Vision và tracking không có quyền ngang nhau:

\[
occupied_{final}=vision_{occupied}\lor tracking_{occupied}
\]

| Vision | Tracking | Cuối cùng | Ý nghĩa |
|---|---|---|---|
| false | false | false | Không có bằng chứng xe. |
| true | false | true | Vision thấy xe, có thể chưa có ID. |
| false | true | true | Tracker chứng minh xe có ID đã dừng; sửa lỗi vision báo trống. |
| true | true | true | Hai nguồn đồng thuận. |

Tracking chỉ có quyền sửa **trống sai → có xe**, không tự sửa **có xe → trống**. Khi xe rời ô, tracking gỡ override; kết quả quay lại phụ thuộc vision.

`decision_source` giải thích nguồn kết luận:

- `none`;
- `vision`;
- `tracking_override`;
- `vision_and_tracking`.

## 15. Xác định hướng tại junction

Giữa hai frame, đường đi ngắn của xe là đoạn `prev_pos → curr_pos`. Junction là đoạn `p1 → p2`. `_segments_intersect()` dùng kiểm tra CCW để biết hai đoạn có cắt nhau.

Sau khi cắt, hệ thống không kết luận ngay. Nó lưu vị trí trung bình trước vạch và chờ đủ:

- mặc định 10 quan sát sau vạch;
- quãng đường sau vạch ít nhất 35 px.

Vector trước và sau:

\[
\mathbf a=p_{cross}-p_{before},\qquad
\mathbf b=p_{after,last}-p_{after,first}
\]

Dot product cho góc:

\[
\theta=\arccos\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|}
\]

Cross product 2D cho chiều:

\[
cross=a_xb_y-a_yb_x
\]

- góc dưới 20°: `STRAIGHT`;
- góc đủ lớn và cross dương: `TURN_LEFT`;
- góc đủ lớn và cross âm: `TURN_RIGHT`.

Nếu track biến mất quá 90 frame trước khi đủ bằng chứng, kết quả là `UNKNOWN`; code không dùng vị trí dự đoán để đoán bừa.

Lưu ý: `DirectionDetector` đang được nối trong `single_camera.py`; `main.py` bốn crop hiện chưa gọi detector hướng.

## 16. Canonical hóa và ghi JSON an toàn

Trước khi binder và web dùng ID, `parking_binder.remap_vehicle_ids(manager.canonical_global_id)` chuyển mọi binding ID cũ sang ID canonical. Nếu merge làm một Global ID xuất hiện ở hai ô, binder giữ binding có overlap mạnh hơn, thời gian dừng dài hơn và thời điểm gắn sớm hơn; binding còn lại bị gỡ.

`CrossCameraManager.to_json()`:

1. chỉ lấy confirmed track;
2. canonical hóa ID;
3. nếu cùng Global ID có nhiều bbox trong một camera, giữ bbox có area mạnh nhất;
4. gom quan sát theo Global ID;
5. lấy trung bình vị trí toàn cục nếu xe được thấy đồng thời ở nhiều góc nhìn;
6. chỉ tạo một phần tử `map_vehicles` cho mỗi canonical G#.

`_save_json_atomic()` ghi vào file `.tmp`, sau đó `replace()` file đích. Web hoặc đọc bản cũ hoàn chỉnh, hoặc bản mới hoàn chỉnh; không đọc trúng JSON đang ghi dở.

---

# PHẦN 2 — BÓC TÁCH CODE VÀ LUỒNG HÀM

Trong tên hàm của code có từ `_match`. Ở đây `match` nghĩa là **đối chiếu rồi ghép hai quan sát được cho là cùng đối tượng**. Tài liệu sẽ luôn nói rõ đang ghép track–detection, handoff–local track hay xe–ô đỗ.

## 17. Bộ điều phối `main.py`

### 17.1. Thành phần và chuỗi lời gọi

**File:** `main.py`  
**Class:** `CameraSimulator`  
**Hàm trung tâm:** `run_detection()`

```text
parse_args()
  └─ CameraSimulator.__init__()
       └─ _load_and_split_slots()

run_detection()
  ├─ tạo CrossCameraManager dùng chung
  ├─ tạo SlotVehicleBinder dùng chung
  ├─ tạo 4 ParkingDetector
  ├─ tạo 4 MotionVehicleTracker
  └─ vòng lặp mỗi frame
       ├─ get_all_camera_frames()
       ├─ tracker.process_frame() cho từng crop
       ├─ parking_binder.try_recover_id()
       ├─ manager.update_all_tracks()
       ├─ parking_binder.remap_vehicle_ids()
       ├─ parking_binder.update_tracks()
       ├─ detector.detect() ở nhịp parking
       ├─ parking_binder.update_vision()
       └─ _save_json_atomic() ở nhịp JSON
```

**Đầu vào:** video, file slot gốc, tham số CLI.  
**Đầu ra:** bốn cửa sổ crop/overview nếu bật display; JSON trạng thái ô, vị trí xe và registry Global ID.

### 17.2. Cắt frame thành bốn góc nhìn

```python
self.mid_x = self.frame_w // 2
self.mid_y = self.frame_h // 2
self.cameras = {
    "cam1": {"crop": (0, 0, self.mid_x + overlap,
                       self.mid_y + overlap)},
    "cam2": {"crop": (self.mid_x - overlap, 0,
                       self.frame_w, self.mid_y + overlap)},
    "cam3": {"crop": (0, self.mid_y - overlap,
                       self.mid_x + overlap, self.frame_h)},
    "cam4": {"crop": (self.mid_x - overlap, self.mid_y - overlap,
                       self.frame_w, self.frame_h)},
}
```

Giải thích:

1. `frame_w // 2`, `frame_h // 2` lấy tâm nguyên của frame.
2. Mỗi crop lưu `(x1, y1, x2, y2)` trong hệ tọa độ video gốc.
3. Nếu `overlap > 0`, biên crop được mở sang crop kề; mặc định CLI là 0.
4. `get_camera_frame()` thực hiện `frame[y1:y2, x1:x2].copy()` để mỗi tracker nhận ảnh riêng.
5. `.copy()` tránh các thao tác vẽ/sửa crop vô tình thay frame gốc.

### 17.3. Chia ROI gốc về từng crop

```python
sx = self.frame_w / ref_w
sy = self.frame_h / ref_h
center_x = slot["center"]["x"] * sx
center_y = slot["center"]["y"] * sy
cam_id = self._classify_point(center_x, center_y)
crop = self.cameras[cam_id]["crop"]
for p in slot["polygon"]:
    lx = p["x"] * sx - crop[0]
    ly = p["y"] * sy - crop[1]
    local_polygon.append({"x": round(lx), "y": round(ly)})
```

Giải thích:

1. `sx`, `sy` scale từ kích thước tham chiếu trong JSON sang video thật.
2. Tâm slot sau scale quyết định slot thuộc quadrant nào.
3. `_classify_point()` dùng so sánh với `mid_x`, `mid_y` để trả `cam1..cam4`.
4. Mỗi đỉnh polygon được scale rồi trừ offset crop để thành tọa độ local.
5. Mỗi camera có file `parking_slots_camX.json` riêng; nhưng binder vẫn dùng chung và sau đó cộng offset trở lại tọa độ toàn video.

**Ví dụ:** P056 có tâm toàn cục `(780, 520)`. Nếu `mid=(550,360)`, nó thuộc cam4. Tọa độ local là `(780-550, 520-360)=(230,160)`.

### 17.4. Khởi tạo: bốn tracker nhưng một namespace ID và một binder

```python
manager = CrossCameraManager(
    camera_sizes=camera_sizes,
    camera_crops={cam_id: tuple(cam["crop"])
                  for cam_id, cam in sim.cameras.items()},
    lookahead_frames=args.handoff_lookahead_frames,
    prediction_radius=args.handoff_prediction_radius,
    min_direction_cosine=args.handoff_min_direction_cosine,
)
parking_binder = SlotVehicleBinder(
    stop_seconds=args.slot_stop_seconds,
    exit_seconds=args.slot_exit_seconds,
    recovery_expand_ratio=args.slot_recovery_expand_ratio,
)
```

- `manager` là nơi duy nhất được cấp Global ID.
- `parking_binder` là nơi duy nhất giữ ánh xạ Global ID ↔ slot toàn bản đồ.
- `detectors[cam_id]` và `trackers[cam_id]` là riêng cho từng crop vì pixel local và mô hình nền khác nhau.
- `binders[cam_id] = parking_binder` chỉ tạo nhiều tham chiếu tới cùng một object, không tạo bốn binder.
- Motion tracker trong `main.py` nhận `slot_binder=None`; recovery ô đỗ được làm sau local tracking, ở tầng Global ID.

### 17.5. Thứ tự quan trọng nhất mỗi frame

```python
all_observable_tracks = {
    cam_id: tracker.observable_tracks
    for cam_id, tracker in trackers.items()
}
recovered_global_id = parking_binder.try_recover_id(...)
if recovered_global_id is not None:
    manager.bind_external_id(cam_id, local_id,
                             recovered_global_id, frame_idx)
global_ids_per_cam = manager.update_all_tracks(
    all_observable_tracks, frame_idx
)
parking_binder.remap_vehicle_ids(manager.canonical_global_id)
parking_binder.update_tracks(global_active_tracks, frame_idx,
                             tracking_timestamp_s)
```

Giải thích từng lệnh:

1. `observable_tracks` lấy track đang thật sự được thấy ở frame này, gồm tentative và confirmed. Lost không được dùng làm quan sát mới ở camera đích.
2. `try_recover_id()` ưu tiên lấy ID từ ô parked trước khi bất kỳ ID mới nào được tạo.
3. `bind_external_id()` ghi thẳng ánh xạ đã được binder xác minh.
4. `update_all_tracks()` mở/ghép handoff, xử lý trùng và chỉ cuối cùng mới tạo G# mới.
5. `remap_vehicle_ids()` cập nhật slot nếu manager vừa merge `G#4 → G#2`.
6. `update_tracks()` ghép canonical Global ID với slot và cập nhật trạng thái dừng.

Nếu đảo thứ tự cấp ID mới lên trước recovery, xe G#2 chạy khỏi P056 có thể bị tạo G#4; vì vậy thứ tự là một phần của thuật toán, không chỉ là tổ chức code.

### 17.6. Ba tập track và lý do tồn tại

| Property | Chứa gì | Dùng ở đâu |
|---|---|---|
| `observable_tracks` | tentative/confirmed được thấy ở frame hiện tại | Handoff và parking recovery sớm |
| `active_tracks` | confirmed và hiện đang thấy | Hiển thị, JSON xe hoạt động |
| `confirmed_tracks` | confirmed và lost | Binder hoàn tất cửa sổ dừng từ bbox đo cuối |

Không thể thay tất cả bằng một dictionary:

- chỉ dùng active sẽ bỏ lỡ tentative xe chạy nhanh;
- đưa tentative lên web sẽ hiển thị nhiễu;
- bỏ lost khỏi binder có thể làm mất bằng chứng dừng ngay khi motion biến mất.

## 18. Công cụ vẽ và lưu ROI

### 18.1. `order_points()` — sửa thứ tự click

**File:** `tools/ParkingSpacePicker_ve_js.py`

```python
values = np.asarray(points, dtype=np.float32)
center = np.mean(values, axis=0)
angles = np.arctan2(values[:, 1] - center[1],
                    values[:, 0] - center[0])
ordered = values[np.argsort(angles)]
top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
ordered = np.roll(ordered, -top_left, axis=0)
```

1. Chuyển list điểm sang mảng số thực để tính vector.
2. `center` là trung bình bốn đỉnh.
3. `arctan2(dy, dx)` cho góc từng điểm quanh tâm.
4. `argsort` sắp điểm theo vòng quanh polygon, tránh đường chéo tự cắt.
5. Điểm có `x+y` nhỏ nhất thường là trên-trái.
6. `roll` xoay danh sách để định dạng JSON ổn định.

### 18.2. `_mouse()`, `_contains()` và `save()`

```python
if event == cv2.EVENT_LBUTTONDOWN and len(self.pending) < 4:
    self.pending.append((x, y))
elif event == cv2.EVENT_RBUTTONDOWN:
    for index, polygon in enumerate(self.polygons):
        if self._contains((x, y), polygon):
            self.polygons.pop(index)
```

- Click trái gom tối đa bốn điểm chờ.
- Phím `A` mới gọi `order_points()` và chấp nhận polygon.
- Click phải dùng `cv2.pointPolygonTest()` tìm slot chứa con trỏ để xóa.
- `save()` tự đánh ID `P001...`, tính center bằng mean và ghi atomic qua file `.tmp` rồi replace.

JSON tối thiểu:

```json
{
  "imageWidth": 1100,
  "imageHeight": 720,
  "slots": [{
    "id": "P056",
    "type": "polygon",
    "polygon": [{"x": 1, "y": 2}],
    "center": {"x": 40, "y": 60},
    "status": "empty"
  }]
}
```

Trong file thật, `polygon` có bốn điểm đầy đủ.

### 18.3. ROI junction

**File:** `tools/draw_direction_lines.py`  
**Class:** `ROIDrawer`

`_mouse_callback()` dùng click đầu làm `current_point`, click thứ hai tạo một line `junction_N`. `save()` ghi `id`, `name`, `p1`, `p2`. Khác tool slot, file này hiện ghi JSON trực tiếp, không dùng file tạm.

## 19. `MotionVehicleTracker`: từ frame đến detection

### 19.1. Chuỗi lời gọi

```text
process_frame(frame)
  ├─ _detect(frame)
  │    ├─ bg_sub.apply(frame)
  │    ├─ _temporal_motion_mask(frame)
  │    ├─ findContours()
  │    ├─ _histogram(frame, box)
  │    └─ _suppress_duplicate_detections()
  ├─ _assign(detections)
  ├─ _apply_detection() cho cặp được ghép
  ├─ chuyển unmatched track thành lost/expired
  ├─ _is_echo_of_matched_track()
  └─ _create_or_reid() cho detection còn lại
```

### 19.2. `_temporal_motion_mask()` — bù sáng và đổi frame

```python
gray = cv2.GaussianBlur(
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0
)
self._gray_history.append(gray)
reference = self._gray_history[0]
brightness_shift = float(np.median(
    gray.astype(np.int16) - reference.astype(np.int16)
))
adjusted_reference = cv2.convertScaleAbs(
    reference, alpha=1.0, beta=brightness_shift
)
difference = cv2.absdiff(gray, adjusted_reference)
_, motion = cv2.threshold(
    difference, self.motion_threshold, 255, cv2.THRESH_BINARY
)
```

1. Đổi BGR sang gray vì mục tiêu là thay đổi sáng, không phải màu.
2. Gaussian blur giảm nhiễu pixel/nén.
3. Deque giữ `motion_frame_gap+1` ảnh; mặc định constructor là gap 3.
4. Ép sang `int16` tránh tràn số khi trừ hai ảnh `uint8`.
5. Median chênh lệch là lượng bù phơi sáng toàn frame.
6. `absdiff` lấy độ lớn thay đổi, không quan tâm sáng lên hay tối xuống.
7. Threshold mặc định 25 đổi ảnh chênh lệch thành nhị phân.
8. Sau đoạn trích, opening 3×3, dilation 5×5 hai lần và closing 11×11 hai lần làm sạch mask.

### 19.3. `_detect()` — MOG2 là cổng thứ nhất, motion là cổng thứ hai

```python
background_mask = self.bg_sub.apply(frame)
_, background_mask = cv2.threshold(
    background_mask, 200, 255, cv2.THRESH_BINARY
)
temporal_motion = self._temporal_motion_mask(frame)
support = cv2.dilate(
    temporal_motion, np.ones((17, 17), np.uint8), iterations=1
)
mask = cv2.bitwise_and(background_mask, support)
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
```

- Threshold 200 loại nhãn shadow mức xám của MOG2, chủ yếu giữ foreground trắng.
- Motion được giãn 17×17 để một pixel chuyển động hỗ trợ vùng foreground lân cận.
- `bitwise_and` buộc pixel thỏa cả hai nguồn.
- `RETR_EXTERNAL` chỉ lấy contour ngoài, tránh mỗi lỗ trong xe thành contour riêng.
- Sau đó từng contour qua gate diện tích, kích thước, aspect ratio và motion ratio.

Detection được tạo:

```python
{
    "box": (x, y, w, h),
    "point": self._bottom_center(box),
    "area": area,
    "hist": self._histogram(frame, box),
}
```

Điểm tracking dùng giữa cạnh đáy bbox, không dùng tâm hình học. Với camera nhìn xiên, điểm này gần vị trí tiếp xúc mặt đường hơn.

### 19.4. `_histogram()` — đặc trưng màu 16×16

```python
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
histogram = cv2.calcHist(
    [hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]
)
return cv2.normalize(histogram, histogram).astype(np.float32)
```

- Kênh 0 là Hue — sắc màu; kênh 1 là Saturation — độ bão hòa.
- Không dùng Value để giảm nhạy với độ sáng.
- 16×16 tạo 256 ô histogram, cân bằng độ chi tiết và chi phí.
- Normalize cho phép so bbox lớn và nhỏ trên cùng thang.
- `cv2.compareHist(..., HISTCMP_BHATTACHARYYA)` trả khoảng cách: 0 rất giống, lớn hơn là khác.

## 20. Kalman và ghép track–detection

### 20.1. `_new_kalman()`

```python
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], np.float32)
kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], np.float32)
```

- `KalmanFilter(4,2)` nghĩa là bốn biến state và hai biến đo được.
- Hàng một: `x_new = x_old + vx_old`.
- Hàng hai: `y_new = y_old + vy_old`.
- Hai hàng cuối giữ vận tốc.
- Measurement matrix nói camera chỉ đo x,y; vx,vy được filter suy ra.
- Process noise code đặt `[2,2,12,12]`: cho vận tốc linh hoạt hơn vị trí.
- Measurement noise là `8I`: thừa nhận bbox đo có rung.

### 20.2. `_assign()` — dựng ma trận chi phí

```python
predicted = track.kalman.predict()
distance = np.linalg.norm(
    np.subtract(predicted_point, detection["point"])
)
if distance > max_distance:
    continue
iou = self._iou(predicted_box, detection["box"])
appearance_distance = cv2.compareHist(
    track.appearance, detection["hist"],
    cv2.HISTCMP_BHATTACHARYYA
)
costs[row, col] = (
    0.50 * (distance / max_distance)
    + 0.30 * (1.0 - iou)
    + 0.20 * appearance_distance
)
```

1. Mỗi track gọi `predict()` đúng một lần trước khi dựng ma trận.
2. Nếu track đã invisible, `max_distance` tăng dần tối đa gấp đôi trong 30 frame để có cơ hội nối lại.
3. Vượt khoảng cách thì ô ma trận giữ giá trị vô hiệu 10.
4. `1-IoU` biến độ giống thành độ sai.
5. Ba độ sai đã chuẩn hóa được cộng theo trọng số 50/30/20.
6. `lapjv(costs, extend_cost=True, cost_limit=0.90)` trả cột detection được chọn cho mỗi hàng track.
7. `col < 0` nghĩa là track không được ghép; detection không bị cột nào lấy cũng được trả riêng.

### 20.3. `_apply_detection()` — sửa track bằng measurement thật

```python
track.kalman.correct(np.array(
    [[point[0]], [point[1]]], dtype=np.float32
))
track.cx, track.cy = point
track.bbox, track.area = box, float(detection["area"])
track.consecutive_invisible_count = 0
track.appearance = cv2.addWeighted(
    track.appearance, 0.75, detection["hist"], 0.25, 0
)
track.status = (
    TrackStatus.CONFIRMED
    if self._is_confirmable(track)
    else TrackStatus.TENTATIVE
)
```

- `correct()` cập nhật Kalman bằng điểm thật.
- Bbox và area luôn lấy detection thật, không lấy bbox dự đoán.
- Invisible count về 0 vì frame này đã thấy.
- Histogram dùng EMA 75% lịch sử + 25% mới để bớt đổi màu đột ngột.
- `_is_confirmable()` yêu cầu đủ visible count và displacement từ điểm đầu.

## 21. `process_frame()` và Local Re-ID

### 21.1. Trình tự cập nhật vòng đời

```python
detections, mask = self._detect(frame)
assignments, unmatched_tracks, unmatched_detections = \
    self._assign(detections)
for track_id, detection_id, _ in assignments:
    self._apply_detection(self._tracks[track_id],
                          detections[detection_id])
for track_id in unmatched_tracks:
    track.consecutive_invisible_count += 1
    if track.status == TrackStatus.CONFIRMED:
        track.status = TrackStatus.LOST
    if track.consecutive_invisible_count > self.lost_track_ttl:
        expired.append(track_id)
```

Track được ghép nhận measurement. Track không được ghép tăng số frame mất; confirmed chuyển lost. Vượt TTL thì bị xóa khỏi `_tracks`; nếu nó từng là track thật, được lưu `_exited_tracks` để Re-ID.

### 21.2. `_create_or_reid()`

```python
best_track_id = None
best_distance = 0.18
for track_id, old in self._exited_tracks.items():
    if self._frame_idx - old.exited_frame > self.reid_ttl:
        continue
    distance = cv2.compareHist(
        old.appearance, detection["hist"],
        cv2.HISTCMP_BHATTACHARYYA
    )
    if distance < best_distance:
        best_track_id, best_distance = track_id, distance
```

Tên biến thật trong code là `candidate`; ở đây nó chỉ có nghĩa “ID cũ đang có khoảng cách màu tốt nhất”. Nếu tìm được:

1. pop track khỏi `_exited_tracks`;
2. tạo Kalman mới tại detection hiện tại;
3. đưa track trở lại `_tracks`;
4. gọi `_apply_detection()`;
5. đặt trạng thái confirmed.

Nếu không tìm được, `_next_id` tạo local ID mới. Trong pipeline bốn crop, nhánh kiểm tra `self.slot_binder` ở đầu hàm không chạy vì `main.py` truyền `None`; Global parking recovery diễn ra ngoài tracker.

## 22. `CrossCameraManager`: nơi cấp và bảo vệ Global ID

### 22.1. Cấu trúc dữ liệu

```python
self._next_global_id = 1
self._local_to_global = {}
self._gid_members = {}
self._global_aliases = {}
self._handoffs = []
self._recently_lost = []
```

| Biến | Ý nghĩa |
|---|---|
| `_next_global_id` | Số G# chưa dùng tiếp theo. |
| `_local_to_global` | Ánh xạ `(cam_id, local_id) → global_id`. |
| `_gid_members` | Các local member đang thuộc cùng G#. |
| `_global_aliases` | ID đã nghỉ → ID canonical. |
| `_handoffs` | Hồ sơ bàn giao đang chờ góc nhìn đích. |
| `_recently_lost` | Hồ sơ track vừa mất để gộp continuation. |

### 22.2. `_canonical_id()` và `_bind()`

```python
def _canonical_id(self, global_id):
    path = []
    while global_id in self._global_aliases:
        path.append(global_id)
        global_id = self._global_aliases[global_id]
    for old_id in path:
        self._global_aliases[old_id] = global_id
    return global_id

def _bind(self, cam_id, local_track_id, global_id):
    global_id = self._canonical_id(global_id)
    key = (cam_id, local_track_id)
    self._local_to_global[key] = global_id
    self._gid_members.setdefault(global_id, set()).add(key)
    return global_id
```

- `_canonical_id()` đi theo chuỗi alias đến ID sống.
- `path` nhớ những ID trung gian để nén đường dẫn.
- `_bind()` luôn canonical hóa trước khi ghi; vì vậy G# đã retired không được gắn lại.
- Nếu local key từng thuộc G# khác, code thật còn xóa key khỏi member set cũ.

### 22.3. `bind_external_id()`

Đây là API cho một nguồn bên ngoài manager nhưng đã xác minh ID, hiện là parking recovery:

```python
global_id = self._canonical_id(global_id)
bound = self._bind(cam_id, local_track_id, global_id)
self._event(
    "global_id_recovered", frame_idx, global_id,
    camera=cam_id, local_track_id=local_track_id,
    source=source,
)
return bound
```

Nó không “tin local ID”. Nó nhận Global ID mà binder đang giữ, canonical hóa rồi gắn vào local observation mới.

### 22.4. `_velocity()` và `_outward_edge()`

```python
history = getattr(track, "history", [])
first_index = max(0, len(history) - 5)
first = history[first_index]
steps = max(1, len(history) - first_index - 1)
vx = float(track.cx - first[0]) / steps
vy = float(track.cy - first[1]) / steps
```

- Tối đa năm điểm nghĩa là tối đa bốn bước frame-to-frame.
- Trung bình nhiều bước giảm rung hơn hiệu hai frame cuối.
- Nếu dưới hai điểm, vận tốc `(0,0)`.

`_outward_edge()` chỉ thêm cạnh có dấu vận tốc hướng ra. Ví dụ `vx < -1` mới xét cạnh trái. Với mỗi cạnh, tính `distance/speed`; chọn cạnh có thời gian nhỏ nhất và chỉ trả về nếu thời gian không quá lookahead.

### 22.5. `_upsert_handoff()`

```python
exit_info = self._outward_edge(cam_id, track)
global_id = self._local_to_global.get((cam_id, local_track_id))
if exit_info is None or global_id is None:
    return
edge, velocity = exit_info
target_cam = EDGE_ADJACENCY.get((cam_id, edge))
if target_cam is None:
    return
world = self._world(cam_id, (track.cx, track.cy))
```

1. Xe phải thật sự hướng ra một cạnh đủ sớm.
2. Local track phải đã có Global ID.
3. Cạnh ra phải có camera kề trong topology.
4. `_world()` cộng crop offset để lưu vị trí chung.
5. Nếu hồ sơ cùng G#, nguồn và đích đã có, code cập nhật vị trí, vận tốc, size, appearance và frame cuối.
6. Nếu chưa có, tạo `HandoffEntry` và event `handoff_opened`.

### 22.6. `_predicted_world()`

```python
elapsed = max(0, frame_idx - entry.updated_at_frame)
return (
    entry.last_world[0] + entry.velocity_world[0] * elapsed,
    entry.last_world[1] + entry.velocity_world[1] * elapsed,
)
```

`updated_at_frame`, không phải `created_at_frame`, là mốc dự đoán. Khi xe nguồn còn thấy, handoff liên tục cập nhật measurement mới; sau khi nguồn mất, vị trí mới được ngoại suy từ measurement cuối.

### 22.7. `_candidate_cost()` — ba tầng kiểm tra trong code

```python
target_edge = OPPOSITE_EDGE[entry.exit_edge]
predicted = self._predicted_world(entry, frame_idx)
world = self._world(cam_id, (track.cx, track.cy))
residual = float(np.linalg.norm(np.subtract(world, predicted)))
depth = self._entry_depth(cam_id, track, target_edge)
speed = float(np.hypot(*entry.velocity_world))
entry_limit = (
    self.edge_margin + self.prediction_radius
    + speed * min(4, self.lookahead_frames) * 0.25
)
```

- `target_edge` là cạnh đối diện cạnh ra.
- `residual` là sai số vị trí toàn cục.
- `depth` là độ sâu từ cạnh vào: vào bên phải thì `width-track.cx`; vào bên trái thì `track.cx`.
- `entry_limit` gồm margin cố định, bán kính dự đoán và một phần mở theo tốc độ.

Các điều kiện loại tiếp theo:

```python
if depth > entry_limit:
    return None, "outside_entry_corridor", details
if residual > self.prediction_radius:
    return None, "prediction_distance", details
if appearance_distance > self.appearance_threshold:
    return None, "appearance", details
if size_distance > 0.90:
    return None, "size", details
if direction is not None and direction < self.min_direction_cosine:
    return None, "direction", details
```

`return None` nghĩa là cặp handoff–local track không được phép đi vào ma trận ghép. Chuỗi `reason` được ghi telemetry để biết lý do loại.

Nếu qua hết:

```python
cost = (
    0.55 * (residual / self.prediction_radius)
    + 0.30 * appearance_distance
    + 0.10 * size_distance
    + 0.05 * direction_cost
)
```

Chi phí vẫn phải ≤ 0.92. Gate bảo đảm hợp lệ tối thiểu; cost dùng để chọn cặp tốt nhất trong các cặp hợp lệ.

### 22.8. `_match_pending_handoffs()` — ghép toàn batch

```python
entries = [
    entry for entry in self._handoffs
    if frame_idx - entry.updated_at_frame <= self.handoff_ttl
]
candidates = [
    (cam_id, local_id, track)
    for cam_id, tracks in all_tracks.items()
    for local_id, track in tracks.items()
    if (cam_id, local_id) not in self._local_to_global
]
costs = np.full(
    (len(entries), len(candidates)), 10.0, dtype=np.float64
)
```

Trong code, biến `candidates` nghĩa là danh sách local track chưa có Global ID đang được xét. Các bước:

1. Bỏ handoff đã quá TTL.
2. Chỉ lấy local track chưa có mapping.
3. Tạo ma trận handoff × local track, mặc định 10 là vô hiệu.
4. Nếu `entry.target_cam != cam_id`, không đối chiếu.
5. Gọi `_candidate_cost()` cho từng cặp còn lại.
6. Cặp bị loại sinh `handoff_rejected`, nhưng `_record_rejection()` giới hạn mỗi hồ sơ tối đa một log trong tám frame để tránh spam.
7. `lapjv(..., cost_limit=0.92)` chọn cặp một-một.
8. Cặp thắng gọi `_bind()`, sinh `handoff_matched` và xóa hồ sơ đã dùng.

### 22.9. `update_all_tracks()` — hàm ngắn nhưng điều phối nhiều thuật toán

```python
for cam_id, tracks in all_tracks.items():
    for local_id, track in tracks.items():
        if self.get_global_id(cam_id, local_id) is not None \
                and self._is_confirmed(track):
            self._upsert_handoff(cam_id, local_id, track, frame_idx)

self._match_pending_handoffs(all_tracks, frame_idx)
# xử lý simultaneous overlap
# chỉ confirmed chưa có ID mới được allocate
self._merge_all_nearby_active_duplicates(all_tracks, frame_idx)
self._merge_recently_lost_duplicates(all_tracks, frame_idx)
self.cleanup(frame_idx)
```

Hành vi đầy đủ theo thứ tự:

1. Track nguồn confirmed đã có G# mở/cập nhật handoff sớm.
2. Track đích, kể cả tentative, thử nhận G# từ handoff.
3. Quan sát đồng thời trong phần crop giao thử dùng cùng G#.
4. Tentative còn lại tiếp tục không có Global ID.
5. Confirmed chưa có ID thử gộp với bbox trùng trong cùng camera.
6. Chỉ confirmed thật sự chưa được giải thích mới `_allocate_global_id()`.
7. Gộp hai G# của bbox gần nhau đang cùng hoạt động.
8. Gộp G# mới với hồ sơ vừa lost nếu chứng minh là continuation.
9. Handoff hết hạn sinh `handoff_expired`.
10. Trả dictionary local→global cho từng camera.

Kết quả ví dụ:

```python
{
    "cam3": {12: 2},
    "cam4": {7: 2},
}
```

### 22.10. Gộp Global ID

```python
canonical_id = self._canonical_id(canonical_id)
duplicate_id = self._canonical_id(duplicate_id)
canonical_id, duplicate_id = (
    min(canonical_id, duplicate_id),
    max(canonical_id, duplicate_id),
)
self._global_aliases[duplicate_id] = canonical_id
```

Sau đó hàm:

- chuyển mọi local member sang ID nhỏ hơn;
- xóa member set của ID nghỉ;
- canonical hóa ID trong handoff và recently-lost;
- loại hồ sơ handoff trùng;
- ghi `global_id_merged`.

Giữ ID nhỏ hơn là chính sách ổn định và dễ giải thích, không phải bằng chứng rằng ID nhỏ hơn luôn có bbox tốt hơn. Chất lượng bbox hiển thị được chọn riêng bằng area/visibility.

---

## 23. `ParkingDetector`: từ polygon đến trạng thái vision

### 23.1. Chuỗi lời gọi

```text
ParkingDetector.__init__(slots_file, tham số)
  ├─ đọc JSON
  └─ _get_polygon() chuẩn hóa schema

detect(frame)
  ├─ _compute_rois() ở lần đầu/độ phân giải đầu tiên
  ├─ _get_gamma_lut() và CLAHE cho 25 biến thể
  ├─ adaptiveThreshold → medianBlur → dilate
  ├─ voting theo foreground ratio
  ├─ center-cluster rescue
  ├─ Canny edge recheck
  ├─ TemporalSmoother.update()
  └─ trả List[SlotResult]
```

**Đầu vào:** một crop frame và file slot local của camera tương ứng.  
**Đầu ra:** mỗi `SlotResult` có `slot_id`, `occupied`, polygon đã scale, center và `vehicle_id=None` ban đầu.

### 23.2. `_compute_rois()` — chỉ tính geometry một lần

```python
h, w = img_shape[:2]
sx = w / self._img_w_ref
sy = h / self._img_h_ref
pts = np.array([
    [int(p["x"] * sx), int(p["y"] * sy)] for p in poly
], np.int32)
x_min = max(0, int(np.min(pts[:, 0])) - 2)
y_min = max(0, int(np.min(pts[:, 1])) - 2)
x_max = min(w, int(np.max(pts[:, 0])) + 2)
y_max = min(h, int(np.max(pts[:, 1])) + 2)
```

1. Scale độc lập x/y để hỗ trợ độ phân giải khác JSON.
2. Bounding rectangle được nới hai pixel và chặn trong frame.
3. Mask chỉ có kích thước bounding rectangle, không phải cả frame; giảm chi phí cho từng slot.
4. Polygon được trừ `(x_min,y_min)` để thành tọa độ local trong mask.
5. `cv2.fillPoly()` tô 255 bên trong polygon.
6. `cv2.countNonZero(mask)` là diện tích ROI theo pixel.
7. `_PrecomputedROI` giữ sẵn tất cả dữ liệu này cho các lần `detect()` sau.

### 23.3. Vòng lặp 25 cấu hình

```python
delta_gamma = [-0.2, -0.1, 0.0, 0.1, 0.2]
delta_clahe = [-0.5, -0.2, 0.0, 0.2, 0.5]
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
l_channel = lab[:, :, 0]
for dc in delta_clahe:
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, base_clahe + dc),
        tileGridSize=(clahe_grid, clahe_grid),
    )
    l_clahe = clahe.apply(l_channel)
    for dg in delta_gamma:
        lut = self._get_gamma_lut(max(0.1, base_gamma + dg))
        l_gamma = cv2.LUT(l_clahe, lut)
```

- LAB được tính một lần; chỉ kênh L đi qua CLAHE/gamma.
- CLAHE được tạo năm lần, mỗi kết quả được dùng cho năm gamma.
- `_get_gamma_lut()` cache bảng 256 giá trị theo gamma làm tròn một chữ số, tránh tính lũy thừa cho từng pixel ở mọi frame.
- `max(0.1, ...)` tránh gamma/clip không hợp lệ.

Chuỗi nhị phân:

```python
blur = cv2.GaussianBlur(l_gamma, (3, 3), 1)
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    25, 16,
)
median = cv2.medianBlur(thresh, 5)
dilated = cv2.dilate(median, np.ones((3, 3), np.uint8),
                     iterations=1)
```

- Block size 25 lấy ngưỡng theo lân cận 25×25.
- Hằng số C=16 được trừ khỏi trung bình Gaussian cục bộ.
- `THRESH_BINARY_INV` biến nét tối tương đối thành trắng.
- Median 5 bỏ nhiễu đơn lẻ; dilation 3×3 nối nét.

### 23.4. Phiếu trống

```python
roi_thresh = dilated[y1:y2, x1:x2]
masked = cv2.bitwise_and(roi_thresh, roi_thresh, mask=roi.mask)
count = cv2.countNonZero(masked)
if count / roi.area < ratio_thr:
    empty_votes[i] += 1

required_votes = total_combinations // 2
is_free = empty_votes[i] >= required_votes
```

- `masked` xóa toàn bộ pixel ngoài polygon.
- `count/roi.area` là mật độ foreground.
- Ít foreground hơn 0.20 thì cấu hình bỏ phiếu trống.
- `required_votes=12`, không phải 13.

### 23.5. Center rescue

```python
centroid_x = np.mean(pts_f[:, 0])
centroid_y = np.mean(pts_f[:, 1])
shrink = 0.4
center_pts = np.array([
    [
        int(centroid_x + (px - centroid_x) * shrink),
        int(centroid_y + (py - centroid_y) * shrink),
    ]
    for px, py in pts_f
], dtype=np.int32)
```

Mỗi vector từ centroid tới đỉnh được nhân 0.4, tạo polygon đồng dạng nằm giữa ô. Code đo `center_ratio` bằng mask của cấu hình gốc `dc=0,dg=0`. Nếu foreground ở tâm ít nhất 5% và tập trung bất thường so với toàn ô, ô bị đổi từ free sang occupied.

### 23.6. Edge recheck và smoothing

```python
edges = cv2.Canny(gray_blur, 50, 150)
edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), 1)
edge_ratio = cv2.countNonZero(masked_e) / roi.area
if is_free_list[i] and edge_ratio >= edge_thr:
    is_free_list[i] = False
```

Tầng cạnh chỉ có quyền cứu “trống nghi ngờ” thành “có xe”. Sau đó:

```python
if apply_smoothing and self._smoother is not None:
    is_free = not self._smoother.update(i, not is_free)
```

`TemporalSmoother` lưu trạng thái pending, bộ đếm liên tiếp và trạng thái confirmed. Tham số truyền vào smoother là `is_occupied = not is_free`; kết quả được đảo lại để trả `is_free`.

## 24. `SlotVehicleBinder`: hình học xe–ô và ghép một-một

### 24.1. Ba cấu trúc trạng thái

| Dataclass | Giữ gì? |
|---|---|
| `VehicleObservation` | timestamp, center, bbox, slot đang giao, overlap. |
| `VehicleParkingState` | lịch sử một Global ID và trạng thái moving/candidate/parked. |
| `SlotBinding` | bằng chứng vision, tracking, vehicle ID và polygon của một slot. |

Hai index quan trọng:

```text
_vehicle_states[global_id] → lịch sử và state của xe
_vehicle_to_slot[global_id] → slot đang giữ xe
_bindings[slot_id] → trạng thái và vehicle_id của slot
```

### 24.2. `_overlap_geometry()`

```python
vehicle_polygon = self._bbox_polygon(bbox)
vehicle_area = max(1.0, bbox[2] * bbox[3])
slot_area = max(1.0, self._polygon_area(binding.polygon))
intersection_area, _ = cv2.intersectConvexConvex(
    vehicle_polygon.astype(np.float32),
    np.asarray(binding.polygon, dtype=np.float32),
)
vehicle_overlap = intersection_area / vehicle_area
slot_overlap = intersection_area / slot_area
```

1. Bbox trở thành polygon bốn góc theo tâm hình học, khác điểm bottom-center dùng cho tracking.
2. `max(1.0, area)` tránh chia 0 khi dữ liệu lỗi.
3. `intersectConvexConvex()` trả diện tích giao hai polygon lồi.
4. `vehicle_overlap` hỏi bao nhiêu phần xe nằm trong ô.
5. `slot_overlap` hỏi bao nhiêu phần ô bị xe che.

Điều kiện nhận cặp:

```python
qualifies = (
    center_inside and vehicle_overlap >= self.min_vehicle_overlap
) or vehicle_overlap >= self.strong_vehicle_overlap
```

- Nếu tâm bbox đã trong ROI, yêu cầu ít nhất 35% bbox giao.
- Nếu tâm ngoài ROI, phải có overlap mạnh ít nhất 60%.
- Nếu không đạt, hàm trả `None`; cặp xe–ô không được vào ma trận.

Điểm:

```python
center_proximity = max(
    0.0, 1.0 - center_distance / slot_diagonal
)
score = (
    0.70 * vehicle_overlap
    + 0.20 * slot_overlap
    + 0.10 * center_proximity
)
```

### 24.3. `_batch_match()`

```python
for row, global_id in enumerate(track_ids):
    bbox = self._track_bbox(active_tracks[global_id])
    current_slot = self._vehicle_to_slot.get(global_id)
    for column, binding in enumerate(bindings):
        geometry = self._overlap_geometry(bbox, binding)
        if geometry is None:
            continue
        score = geometry["score"]
        if current_slot == binding.slot_id:
            score = min(1.0, score + 0.15)
        costs[row, column] = 1.0 - score
```

- Hàng là Global ID, cột là slot.
- Cặp không hợp lệ giữ cost 10.
- Cùng slot hiện tại được loyalty bonus 0.15.
- LAPJV với `cost_limit=0.95` bảo đảm một hàng–một cột.
- Dictionary trả về có dạng `{global_id: (slot_id, geometry)}`.

Tên biến `matches` trong code là kết quả ghép xe–ô, không liên quan handoff.

## 25. Binder: kiểm tra dừng và máy trạng thái

### 25.1. `_is_stopped()`

```python
window_start = now_s - self.stop_seconds
samples = [item for item in state.observations
           if item.timestamp_s >= window_start]
if len(samples) < self.min_stop_samples:
    return False, 0
duration = samples[-1].timestamp_s - samples[0].timestamp_s
if duration < self.stop_seconds * 0.90:
    return False, int(max(0.0, duration) * 1000)
same_slot_ratio = sum(
    item.slot_id == slot_id for item in samples
) / len(samples)
if same_slot_ratio < 0.80:
    return False, int(duration * 1000)
```

Ba gate đầu bảo đảm đủ mẫu, đủ thời gian và không dao động qua nhiều slot.

```python
centers = np.asarray([item.center for item in samples])
median_center = np.median(centers, axis=0)
radius = np.linalg.norm(centers - median_center, axis=1)
r95 = float(np.percentile(radius, 95))
net_displacement = float(np.linalg.norm(
    centers[-1] - centers[0]
))
bbox_diagonal = float(np.median([
    np.hypot(item.bbox[2], item.bbox[3]) for item in samples
]))
```

- Median center bền trước một điểm bbox nhảy.
- `radius` đo độ phân tán quanh vị trí điển hình.
- `r95` bỏ ảnh hưởng cực đoan của khoảng 5% mẫu ngoài cùng, nhưng vẫn chặt hơn mean.
- `net_displacement` bắt xe bò đều: nó có thể có r95 nhỏ trong cửa sổ ngắn nhưng đầu–cuối vẫn dịch chuyển.
- Đường chéo bbox chuẩn hóa cho độ phân giải/kích thước xe.

Kết luận:

```python
stopped = (
    r95 <= max(3.0, 0.06 * bbox_diagonal)
    and net_displacement <= max(5.0, 0.10 * bbox_diagonal)
)
```

### 25.2. `update_tracks()` — từ geometry sang state

```text
update_tracks(global_tracks, frame, timestamp)
  ├─ chuẩn hóa bbox/appearance
  ├─ _batch_match() xe–ô
  ├─ thêm VehicleObservation
  ├─ nếu đang parked: kiểm tra còn trong ô hay exit_pending
  ├─ nếu không giao ô: moving, hủy stop candidate
  ├─ nếu ô bị xe khác giữ: từ chối
  ├─ nếu mới vào ô: tạo stop_candidate
  ├─ _is_stopped()
  └─ đủ dừng + commit grace → _bind_vehicle()
```

Đoạn parked:

```python
if state.parked_slot_id is not None:
    if slot_id == parked_slot:
        state.movement_state = "parked"
        state.outside_since = None
    else:
        if state.outside_since is None:
            state.outside_since = timestamp_s
            state.movement_state = "exit_pending"
        elif timestamp_s - state.outside_since >= self.exit_seconds:
            self._release_vehicle(global_id, frame_idx)
    continue
```

- Xe vẫn trong ô: xóa mốc outside và giữ parked.
- Lần đầu ngoài ô: chỉ chuyển exit_pending.
- Ngoài liên tục 0.5 giây: release.
- `continue` ngăn xe đang parked chạy lại logic tạo candidate mới trong cùng frame.

Đoạn cam kết parked:

```python
stopped, stopped_ms = self._is_stopped(
    state, slot_id, timestamp_s
)
candidate_age = timestamp_s - state.candidate_since
if stopped and candidate_age >= (
    self.stop_seconds + self.stop_commit_grace_seconds
):
    self._bind_vehicle(global_id, slot_id, frame_idx,
                       overlap, stopped_ms)
```

`stop_commit_grace_seconds` mặc định 0.15. Tham số `bind_confirmations` vẫn được lưu để tương thích constructor cũ nhưng code hiện tại không dùng nó để cam kết parked; quyết định thật dựa timestamp và thống kê vị trí.

### 25.3. `_bind_vehicle()`

Hàm đảm bảo:

1. nếu xe đã ở slot khác, release binding cũ;
2. nếu slot đang giữ xe khác, không ghi đè;
3. đặt `vehicle_id`, `tracking_occupied=True`;
4. lưu overlap, stopped time, frame bind;
5. cập nhật cả `_vehicle_to_slot` và `VehicleParkingState`;
6. xóa pending release của slot;
7. sinh `vehicle_stopped_in_slot`;
8. nếu vision đang báo trống, sinh `tracking_occupied_override`;
9. gọi `_sync_result()`.

### 25.4. `_release_vehicle()`

Hàm không kết luận slot trống. Nó chỉ:

- xóa ánh xạ Global ID–slot;
- đặt `tracking_occupied=False` và `vehicle_id=None`;
- lưu `_pending_release[slot]=(global_id, frame)` cho cửa sổ nhận lại;
- chuyển vehicle state về moving;
- gọi `_sync_result()` để kết quả quay lại vision.

## 26. Parking recovery và đồng bộ ID sau merge

### 26.1. `try_recover_id()`

```python
x, y, w, h = bbox
global_bbox = (
    x + coordinate_offset[0],
    y + coordinate_offset[1],
    w, h,
)
point = (
    global_bbox[0] + global_bbox[2] / 2.0,
    global_bbox[1] + global_bbox[3] / 2.0,
)
```

Recovery dùng tâm hình học bbox trong hệ toàn video. Với mỗi slot đang có `vehicle_id`:

```python
polygon = self._expanded_polygon(
    binding.polygon, self.recovery_expand_ratio
)
inside = self._point_in_polygon(point, polygon)
geometry = self._overlap_geometry(global_bbox, temporary_binding)
if not inside and overlap < self.min_vehicle_overlap:
    continue
```

- `_expanded_polygon()` lấy centroid và nhân vector centroid→đỉnh với `1+0.15`.
- Tâm nằm trong hoặc overlap đủ mới xét tiếp.
- Camera phải đúng binding camera nếu caller truyền `camera_id`.
- Nếu có appearance hai phía, khoảng cách >0.50 bị loại.
- Danh sách hợp lệ được sắp theo khoảng cách tâm; gần nhất thắng.

Khi trả ID, hàm đặt state `exit_pending` và event `parked_id_recovered`. `main.py` ngay lập tức gọi `bind_external_id()`.

### 26.2. `remap_vehicle_ids()`

Sau merge `G#4 → G#2`, binder có thể vẫn giữ vehicle 4. Hàm nhận callback `manager.canonical_global_id` và:

1. thay ID ở mọi binding;
2. chuyển `VehicleParkingState` sang key mới;
3. gom các binding cùng canonical ID;
4. nếu một G# lộ ra ở hai slot, chọn winner bằng tuple:

   ```text
   vehicle_overlap lớn hơn
   → stopped_for_ms lớn hơn
   → bound_at_frame sớm hơn
   ```

5. gỡ binding thua và sinh `global_id_merge_conflict`.

Nhờ vậy ID đã merge không còn nằm ngầm trong dữ liệu parking.

## 27. Hợp nhất vision–tracking và JSON slot

### 27.1. `update_vision()`

Mỗi lần ParkingDetector chạy, binder:

- cộng `coordinate_offset` vào polygon/center local;
- tạo `SlotBinding` nếu chưa có;
- lưu `camera_id`;
- cập nhật `vision_occupied`;
- giữ tham chiếu tới `SlotResult` để overlay được đồng bộ;
- gọi `_sync_result()`.

### 27.2. `_sync_result()`

```python
binding.occupied = bool(
    binding.vision_occupied or binding.tracking_occupied
)
if binding.result_ref is not None:
    binding.result_ref.occupied = binding.occupied
    binding.result_ref.vehicle_id = binding.vehicle_id
```

Hai dòng cuối giải thích vì sao `ParkingDetector.draw_results()` có thể vẽ cả vehicle ID dù detector nguyên thủy không biết ID: binder đã sửa object `SlotResult` được giữ trong `result_ref`.

### 27.3. `_binding_to_json()`

```python
return {
    "occupied": bool(binding.occupied),
    "status": "occupied" if binding.occupied else "empty",
    "vehicle_id": binding.vehicle_id,
    "vision_occupied": bool(binding.vision_occupied),
    "tracking_occupied": bool(binding.tracking_occupied),
    "decision_source": binding.decision_source,
    "tracking_state": binding.tracking_state,
    "vehicle_overlap": round(binding.vehicle_overlap, 4),
    "stopped_for_ms": int(binding.stopped_for_ms),
}
```

`occupied` giữ tương thích web. Các trường còn lại giải thích vì sao hệ thống kết luận như vậy, rất hữu ích khi đánh giá sai số.

## 28. `DirectionDetector`: phát hiện cắt vạch rồi mới kết luận hướng

### 28.1. Chuỗi lời gọi

```text
single_camera.run()
  ├─ DirectionDetector.from_json()
  └─ mỗi frame: direction_detector.update(active_tracks, frame_idx)
       ├─ _check_line_crossing()
       │    └─ _segments_intersect()
       │         └─ _ccw()
       └─ đủ dữ liệu sau vạch → _compute_direction()
```

### 28.2. Hai đoạn thẳng giao nhau

```python
return (
    self._ccw(A, C, D) != self._ccw(B, C, D)
    and self._ccw(A, B, C) != self._ccw(A, B, D)
)
```

`A,B` là hai vị trí liên tiếp của xe; `C,D` là hai đầu junction. Hai đoạn giao nếu C,D nằm khác phía AB và A,B nằm khác phía CD theo kiểm tra counter-clockwise.

### 28.3. Tạo quyết định chờ

Khi track confirmed cắt vạch lần đầu:

```python
n_before = min(self.history_before, len(track.history) - 1)
before_positions = track.history[-(n_before + 1):-1]
avg_before = (
    int(np.mean([p[0] for p in before_positions])),
    int(np.mean([p[1] for p in before_positions])),
)
self._pending.append(PendingDecision(
    track_id=tid,
    roi_id=roi_id,
    cross_frame=frame_idx,
    position_before=avg_before,
))
```

Trung bình tối đa năm điểm trước giảm rung. `_crossed[tid]` ngăn cùng xe tạo nhiều event ở cùng junction.

Mỗi frame sau, `positions_after` được thêm điểm. Chỉ khi số điểm ≥10 và quãng đường ≥35 px mới gọi `_compute_direction()`.

### 28.4. `_compute_direction()`

```python
vec_after = after_end - after_start
vec_before = after_start - before_pt
cross = (
    vec_before[0] * vec_after[1]
    - vec_before[1] * vec_after[0]
)
cos_angle = np.dot(vec_before, vec_after) / (
    np.linalg.norm(vec_before) * np.linalg.norm(vec_after)
)
angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
```

- `clip` tránh sai số floating-point làm `acos` nhận giá trị ngoài [-1,1].
- Dưới angle threshold 20° là thẳng.
- Nếu đủ góc, dấu cross quyết định trái/phải theo hệ tọa độ ảnh hiện tại.
- Nếu vector quá ngắn, code trả `STRAIGHT`; nếu dưới ba điểm sau, trả `UNKNOWN`.

## 29. Output và atomic JSON

### 29.1. Chọn quan sát mạnh nhất cho một Global ID

Trong `main.py`, khi dựng `global_active_tracks`, nếu cùng G# có nhiều track:

```python
if (
    previous is None
    or (candidate["visible"] and not previous["visible"])
    or (
        candidate["visible"] == previous["visible"]
        and candidate["area"] > previous["area"]
    )
):
    global_active_tracks[global_id] = candidate
```

Thứ tự ưu tiên:

1. chưa có quan sát;
2. quan sát đang thấy thật thắng LOST bbox;
3. cùng trạng thái visible thì area mạnh hơn thắng.

Điều này ngăn bbox motion echo yếu làm lệch binder/map.

### 29.2. `CrossCameraManager.to_json()`

Hàm tiếp tục canonical hóa và chọn area mạnh nhất cho cùng `(G#, camera)`. Nếu cùng G# được thấy ở nhiều crop, `map_vehicles` lấy trung bình vị trí toàn cục của các quan sát. Kết quả có:

- `next_global_id`;
- `retired_global_ids`;
- `active_global_vehicles` cùng các observation;
- `map_vehicles` một phần tử mỗi G#;
- `pending_handoffs`;
- `events` telemetry.

### 29.3. `_save_json_atomic()`

```python
def _save_json_atomic(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    tmp.replace(path)
```

1. Đảm bảo thư mục output tồn tại.
2. Ghi toàn bộ JSON vào file tạm.
3. Đóng file sau `with` để flush dữ liệu.
4. Replace đích bằng một thao tác filesystem.

Đây là atomic ở mức file replace trong cùng filesystem; nó không phải transaction phân tán, nhưng đủ tránh web đọc nửa chuỗi JSON khi producer đang ghi.

## 30. Ví dụ xuyên suốt G#2 theo từng giai đoạn

| Giai đoạn | Hàm chính | Trạng thái và kết quả |
|---|---|---|
| 1. Xe đang ở cam4 | `process_frame()`, `_assign()` | cam4 local #7 confirmed; manager map `(cam4,7)→G#2`. |
| 2. Gần cạnh trái | `_velocity()`, `_outward_edge()` | `vx<0`, thời gian tới cạnh ≤16 frame. |
| 3. Mở bàn giao | `_upsert_handoff()` | Lưu G#2, cam4→cam3, position, velocity, HSV, size. |
| 4. Xuất hiện cam3 | `observable_tracks` | cam3 local #12 còn tentative, chưa có G#. |
| 5. Đối chiếu | `_candidate_cost()` | Qua topology/position/direction/appearance/size. |
| 6. Nhận ID | `_match_pending_handoffs()` | LAPJV chọn hồ sơ G#2 ↔ local #12; `_bind()`. |
| 7. Đi vào P056 | `_overlap_geometry()`, `_batch_match()` | Bbox đủ overlap, state `stop_candidate`. |
| 8. Dừng | `_is_stopped()` | ≥8 mẫu, cùng slot ≥80%, r95/drift đạt ngưỡng. |
| 9. Cam kết đỗ | `_bind_vehicle()` | P056 `tracking_occupied=true`, `vehicle_id=2`. |
| 10. Motion mất | `confirmed_tracks`, binder | Track lost giữ bbox đo cuối; binding parked không bị xóa. |
| 11. Xe chạy lại | `try_recover_id()` | Tentative bbox nằm trong P056 mở rộng, nhận lại G#2. |
| 12. Gắn trước cấp mới | `bind_external_id()` | Local track mới trỏ G#2 trước `update_all_tracks()`. |
| 13. Ra khỏi ô | `update_tracks()` | `exit_pending`; ngoài liên tục 0.5 s thì release override. |
| 14. Xuất JSON | `to_json()`, `_save_json_atomic()` | Web chỉ nhận một canonical G#2 và trạng thái P056 cuối. |

---

# PHẦN 3 — KẾT LUẬN KỸ THUẬT, GIỚI HẠN VÀ PHẢN BIỆN

## 31. Dự án này thực sự có những điểm kỹ thuật nào đáng nói?

Điểm mạnh không nằm ở việc dùng nhiều tên thuật toán, mà nằm ở cách các tầng giới hạn lỗi của nhau.

### 31.1. Chuỗi bằng chứng thay vì một ngưỡng duy nhất

- MOG2 nói pixel khác nền dài hạn.
- Frame difference xác nhận pixel thật sự thay đổi ngắn hạn.
- Kalman nói track có khả năng xuất hiện ở đâu.
- IoU và HSV giúp chọn detection thuộc track nào.
- Handoff dùng topology, vị trí dự đoán, hướng, màu và kích thước.
- Binder dùng giao polygon và thống kê dừng.
- Vision giữ khả năng nhận xe đã đỗ trước khi hệ thống bật.

Không tầng nào một mình chứng minh hoàn toàn “đây là đúng xe”. Độ tin cậy đến từ nhiều điều kiện độc lập cùng phù hợp.

### 31.2. Tách Local ID khỏi Global ID

Đây là yêu cầu kiến trúc bắt buộc cho nhiều camera. Nếu chỉ thay nhãn hiển thị mà không có namespace chung, web vẫn nhận ID trùng. `CrossCameraManager` không chỉ đổi số; nó giữ mapping, handoff, alias, member và telemetry của vòng đời Global ID.

### 31.3. Cho tentative nhận ID cũ nhưng không tự tạo ID mới

Đây là cân bằng quan trọng:

- nếu bắt tentative chờ confirmed, xe nhanh có thể mất ID;
- nếu cho mọi tentative tạo G# mới, nhiễu trở thành xe trên map;
- thiết kế hiện tại chỉ cho tentative **nhận lại ID đã có sau kiểm tra**, còn tạo ID hoàn toàn mới phải chờ confirmed.

### 31.4. Parking tracking là override một chiều

Tracking chỉ bổ sung bằng chứng “có xe” khi Global ID đã nằm trong ROI và dừng. Nó không có quyền biến ô vision đang báo occupied thành empty. Chính sách này ưu tiên an toàn: false free — báo trống sai khi thật sự có xe — nguy hiểm hơn false occupied trong hướng dẫn xe đỗ.

### 31.5. State parked sống lâu hơn motion detection

Motion detector tự nhiên mất xe khi xe dừng. Binder không xem đó là lỗi phải xóa ID; nó chuyển sự thật từ “đang thấy chuyển động” sang “đã chứng minh parked”. Đây là lý do hệ thống vẫn giữ xe sau khi foreground biến mất.

### 31.6. Telemetry làm thuật toán có thể kiểm chứng

Các event như `handoff_opened`, `handoff_matched`, `handoff_rejected`, `global_id_merged`, `vehicle_stopped_in_slot` giúp truy ngược quyết định. Một hệ thống nghiên cứu tốt không chỉ trả kết quả; nó phải cho biết kết quả được tạo qua bằng chứng nào và bị từ chối vì lý do gì.

## 32. Những điều chưa được phép tuyên bố quá mức

### 32.1. Chưa phải multi-camera vật lý hoàn chỉnh

Bốn input hiện là bốn crop đồng bộ tuyệt đối từ cùng một video. Phép cộng crop offset đưa tất cả về cùng hệ pixel một cách chính xác. Camera vật lý có phối cảnh, độ trễ, FPS, màu và timestamp khác nhau; không thể chỉ cộng offset.

### 32.2. Motion backend không phải nhận diện ngữ nghĩa xe

Nó tìm vùng chuyển động có hình học hợp lý. Người đi bộ, xe máy hoặc bóng lớn có thể thỏa điều kiện nếu không có semantic classifier. Vì vậy không được nói pipeline mặc định “dùng AI để nhận diện ô tô”.

### 32.3. Global ID là suy luận, không phải danh tính tuyệt đối

Global ID hiện dựa vị trí, hướng, topology, HSV và kích thước. Hai xe cùng màu/kích thước đi gần nhau vẫn có khả năng nhầm. Biển số hoặc embedding Re-ID mạnh sẽ cần thiết nếu yêu cầu định danh mức cao.

### 32.4. Chưa có con số độ chính xác nếu chưa gán ground truth

Không được dùng cảm giác xem video để nói “95%”. Phải có dữ liệu đúng do người gán nhãn, evaluator và báo cáo trên nhiều điều kiện. Demo tốt chỉ chứng minh hệ thống chạy được, không tự chứng minh độ chính xác tổng quát.

### 32.5. ROI và tham số còn phụ thuộc bối cảnh

ROI, topology, gamma, CLAHE, threshold, diện tích motion và handoff radius hiện được cấu hình thủ công. Đổi camera, độ cao, bề mặt hoặc ánh sáng có thể cần hiệu chỉnh.

## 33. Các invariant — điều luôn phải đúng trong thiết kế

Invariant là quy tắc hệ thống phải giữ ở mọi frame, không phải chỉ ở ví dụ đẹp.

| Mã | Quy tắc |
|---|---|
| I1 | Một `(camera_id, local_track_id)` chỉ trỏ tới một canonical Global ID. |
| I2 | Một Global ID đã retired luôn phân giải về ID canonical, không được sống lại. |
| I3 | Tentative không nhận ID cũ thì chưa được tạo Global ID mới. |
| I4 | Một hồ sơ handoff chỉ được một local track lấy. |
| I5 | Một local track chỉ nhận một hồ sơ handoff. |
| I6 | Một Global ID chỉ được gắn với một slot. |
| I7 | Một slot chỉ giữ một Global ID. |
| I8 | Tracking không tự đổi ô vision occupied thành empty. |
| I9 | Xe parked mất motion detection không tự mất binding. |
| I10 | Web chỉ nhận một entry cho mỗi canonical Global ID. |

Khi debug, thay vì chỉ hỏi “ảnh nhìn có đúng không”, hãy tìm invariant đầu tiên bị phá.

## 34. Phản biện thường gặp và câu trả lời thẳng

### 34.1. “Dự án này có phải AI không?”

**Trả lời:** “Pipeline mặc định của `main.py` là computer vision cổ điển: MOG2, frame difference, morphology, Kalman, histogram HSV, LAPJV và hình học polygon. Dự án có backend YOLO/BoT-SORT tùy chọn trong `single_camera.py`, nhưng em không gọi phần mặc định là AI.”

### 34.2. “Vì sao không dùng YOLO luôn?”

**Trả lời:** “Video mẫu nhìn từ trên cao, xe nhỏ và khác góc nhìn COCO thông thường; model YOLO chưa fine-tune có thể bỏ sót. Motion backend nhẹ, không cần GPU và phù hợp camera cố định. YOLO sẽ hữu ích hơn khi có dữ liệu top-down đã gán nhãn và cần nhận xe đứng yên theo semantic.”

### 34.3. “Kalman có phải thuật toán xác định xe dừng không?”

**Trả lời:** “Không. Kalman dự đoán vị trí để nối track. Xe dừng được kết luận từ bbox measurement thật trong cửa sổ timestamp: median center, r95 và displacement. Em tránh dùng Kalman vì state vận tốc có thể còn quán tính sau khi xe dừng.”

### 34.4. “Tại sao gọi ensemble nếu chỉ có một detector?”

**Trả lời:** “Đây là multi-parameter voting: cùng một pipeline xử lý ảnh được chạy với 25 cặp gamma–CLAHE. Nó không phải ensemble nhiều mạng AI. Code dùng ngưỡng 12/25, em không gọi đó là majority tuyệt đối.”

### 34.5. “Làm sao chắc chắn một xe không có hai ID?”

**Trả lời:** “Hệ thống phòng vệ ở nhiều tầng: bỏ motion echo trước khi tạo track, ghép một-một bằng LAPJV, handoff tentative trước cấp ID mới, phát hiện bbox trùng trong cùng camera và canonical merge giữ ID nhỏ hơn. Tuy nhiên đây là invariant phần mềm dưới các gate hiện tại, không phải chứng minh danh tính tuyệt đối trong mọi cảnh.”

### 34.6. “Vì sao sau merge lại giữ ID nhỏ hơn?”

**Trả lời:** “ID nhỏ hơn thường được tạo trước và đã xuất hiện trong lịch sử/web/slot. Giữ nó làm canonical giúp output ổn định. Bbox tốt nhất được chọn riêng theo visibility và area, nên giữ ID nhỏ không có nghĩa giữ bbox yếu.”

### 34.7. “Hai xe cùng màu và cùng kích thước thì sao?”

**Trả lời:** “HSV và size mất sức phân biệt; lúc đó hệ thống dựa nhiều hơn vào topology, vị trí dự đoán và hướng. Nếu hai xe còn đi sát nhau qua cùng một biên, rủi ro vẫn tồn tại. Hướng nâng cấp là embedding Re-ID được fine-tune, biển số, nhiều keypoint hoặc vùng chuyển giao không cho hai xe nhập nhằng.”

### 34.8. “Xe đã đỗ trước khi bật hệ thống thì có ID không?”

**Trả lời:** “Motion tracker không thấy xe đứng yên sẵn nên có thể không có ID. Parking vision vẫn nhận ô occupied và `vehicle_id=null`. Đây là hành vi đúng: biết ô có xe khác với biết danh tính xe.”

### 34.9. “Nếu vision báo có xe sai thì tracking có sửa thành trống không?”

**Trả lời:** “Không. Fusion hiện là OR một chiều. Thiết kế ưu tiên không báo trống sai. Muốn tracking phủ nhận vision phải có bằng chứng rời ô và một chính sách confidence khác; code hiện chưa làm điều đó.”

### 34.10. “Camera rung hoặc đổi sáng mạnh thì sao?”

**Trả lời:** “Bù median xử lý thay đổi sáng toàn frame ở mức vừa; MOG2 kết hợp temporal mask giảm foreground đứng yên. Camera rung làm gần như toàn ảnh thay đổi nên vẫn là giới hạn. Cần cố định camera hoặc thêm global motion compensation/stabilization.”

### 34.11. “Tại sao cần LAPJV ba lần?”

**Trả lời:** “Ba bài toán khác nhau nhưng cùng có ràng buộc một-một: local track–detection, handoff–track ở camera đích và Global ID–slot. Dùng giải pháp toàn cục tránh kết quả phụ thuộc thứ tự dictionary.”

### 34.12. “Đưa sang camera vật lý cần thay gì?”

**Trả lời:** “Giữ interface local tracker, Global ID manager, batch assignment, binder, state machine và JSON. Thay crop offset bằng homography/ground-plane calibration, đồng bộ timestamp, định nghĩa transition zone thật, hiệu chỉnh màu và đo vận tốc theo thời gian thực thay vì frame index.”

### 34.13. “Vì sao một hàm `update_all_tracks()` ngắn mà nói nó làm nhiều việc?”

**Trả lời:** “Nó là hàm điều phối. Mỗi dòng gọi một thuật toán con: mở handoff, ghép batch, xử lý overlap, cấp ID, gộp duplicate và cleanup. Độ phức tạp nằm trong các hàm con; em trình bày call graph thay vì chỉ đếm dòng của hàm ngoài.”

## 35. Nếu trình bày theo hướng nghiên cứu, phải đo gì?

### 35.1. Ground truth

Ground truth là dữ liệu đúng do người xem video và gán nhãn:

- mỗi frame có những xe thật nào;
- bbox hoặc tâm xe;
- ID thật xuyên frame/camera;
- slot nào occupied;
- Global ID nào vào/ra slot ở frame nào.

Không cần nhìn mắt và nhớ frame. Video debug phải in `frame_idx`; file timestamp ánh xạ frame với thời gian. Người gán nhãn dùng hai thông tin đó để ghi khoảng frame.

### 35.2. Chỉ số tracking và Global ID

| Chỉ số | Ý nghĩa dễ hiểu |
|---|---|
| ID Switches | Số lần cùng một xe thật bị đổi sang ID khác. Càng thấp càng tốt. |
| Fragmentation | Một hành trình xe bị vỡ thành bao nhiêu đoạn ID. |
| False merge | Hai xe thật khác nhau bị gộp chung một G#. Đây là lỗi ưu tiên tránh. |
| Handoff accuracy | Tỉ lệ bàn giao camera giữ đúng ID. |
| IDF1 | Mức đúng của danh tính theo toàn bộ chuỗi thời gian, cân bằng precision/recall ID. |
| HOTA | Đánh giá đồng thời detection và association; dùng khi có evaluator tracking chuẩn. |

### 35.3. Chỉ số trạng thái ô

| Chỉ số | Ý nghĩa |
|---|---|
| Occupancy precision | Trong các ô hệ thống báo có xe, bao nhiêu ô đúng. |
| Occupancy recall | Trong các ô thật sự có xe, hệ thống tìm được bao nhiêu. |
| F1 | Trung bình điều hòa precision và recall. |
| False-free rate | Tỉ lệ báo trống trong khi thật có xe; quan trọng với hệ thống hướng dẫn đỗ. |
| Transition latency | Độ trễ từ xe thật dừng/rời đến khi trạng thái JSON đổi. |
| Vehicle-slot ID accuracy | Tỉ lệ ô có `vehicle_id` đúng xe thật. |

### 35.4. Chỉ số hiệu năng

- FPS xử lý thực tế;
- thời gian trung bình và p95 mỗi frame;
- CPU/GPU/RAM;
- độ trễ JSON;
- hiệu năng theo số camera/độ phân giải.

## 36. Baseline và ablation nên làm như thế nào?

### 36.1. Baseline — mốc so sánh đơn giản

1. **Parking vision only:** chỉ `ParkingDetector`, không tracking override.
2. **Centroid tracking:** nối detection bằng khoảng cách gần nhất, không Kalman/IoU/HSV/LAPJV.
3. **Reactive handoff:** chỉ tìm xe sau khi track nguồn mất, không lookahead/prediction.
4. **Point-in-ROI parking:** chỉ kiểm tra một điểm tâm, không polygon intersection và stop statistics.

TechGAR chỉ có ý nghĩa nghiên cứu nếu chứng minh được pipeline đề xuất tốt hơn các mốc đơn giản này trên cùng ground truth.

### 36.2. Ablation — tháo từng thành phần để đo đóng góp

Giữ nguyên tập dữ liệu và tham số còn lại, lần lượt bỏ:

- bù sáng median;
- temporal motion gate;
- motion echo suppression;
- HSV trong local assignment;
- Kalman prediction;
- LAPJV, thay bằng greedy;
- predictive handoff;
- direction/appearance/size gate của handoff;
- duplicate Global ID merge;
- stop statistics, chỉ dùng overlap;
- tracking override, chỉ dùng vision;
- center rescue hoặc edge recheck.

Mỗi ablation phải báo cùng nhóm metrics. Không được thay đồng thời nhiều thành phần rồi kết luận thành phần nào có tác dụng.

## 37. Tổng quan kỹ thuật trong hai phút

> “Hệ thống bắt đầu từ frame video và chia thành bốn crop mô phỏng. Mỗi crop có motion tracker riêng. Detection được tạo bằng giao giữa MOG2 dài hạn và frame difference ngắn hạn đã bù sáng; morphology và contour tạo bbox, sau đó loại motion echo.
>
> Trong mỗi crop, Kalman dự đoán vị trí; khoảng cách, IoU và histogram HSV tạo ma trận chi phí; LAPJV ghép detection với track theo quan hệ một-một. Local ID chỉ có nghĩa trong một tracker, nên CrossCameraManager tạo Global ID dùng chung.
>
> Khi xe sắp đến biên, manager ước lượng vận tốc và mở handoff sớm. Track mới ở camera đích, kể cả tentative, được kiểm tra qua đúng camera/cạnh vào, vị trí–hướng dự đoán và HSV–kích thước. LAPJV bàn giao ID một-một. Nếu trùng ID đã xảy ra, ID nhỏ hơn làm canonical và mọi slot/output được remap.
>
> Trạng thái ô có hai nguồn. Parking vision chạy 25 cấu hình gamma–CLAHE, adaptive threshold, center rescue, edge recheck và smoothing. Binder tính giao bbox–polygon, ghép Global ID với slot và chỉ gán parked nếu tọa độ detection ổn định trong cửa sổ thời gian. Kết quả cuối là vision OR tracking; tracking chỉ sửa trống sai thành có xe. JSON được canonical hóa và atomic replace để web luôn thấy một entry hoàn chỉnh cho mỗi Global ID.”

## 38. Câu kết luận trước hội đồng

> “Đóng góp chính của TechGAR không phải một detector riêng lẻ, mà là kiến trúc duy trì bằng chứng theo thời gian. Chuyển động tạo quan sát; Kalman, HSV và LAPJV tạo local track; handoff dự đoán và canonical merge duy trì Global ID; polygon intersection và thống kê dừng chứng minh xe thuộc ô; fusion một chiều bảo vệ trạng thái occupied. Em phân biệt rõ phần đã chạy trong mô phỏng với phần cần bổ sung cho camera vật lý, và em đánh giá hệ thống bằng ground truth, baseline, ablation thay vì chỉ dựa vào video demo.”

---

# PHỤ LỤC A — BACKEND YOLO/BOT-SORT TÙY CHỌN

## A1. Phạm vi chính xác

**File:** `src/techgar/vehicle_tracker.py`  
**Entry point sử dụng:** `single_camera.py --backend yolo`  
**Không phải backend mặc định:** `main.py` bốn crop khởi tạo `MotionVehicleTracker`, không khởi tạo `VehicleTracker`.

YOLO và BoT-SORT giải quyết detection/tracking trong một nguồn camera. Global ID xuyên camera vẫn là tầng khác.

## A2. Khởi tạo `VehicleTracker`

```python
tracker = VehicleTracker(
    model_path=args.model,
    tracker_config=args.tracker,
    confidence=args.conf,
    iou=args.iou,
    imgsz=args.imgsz,
    device=args.device,
    min_visible_count=args.min_visible_count,
    lost_track_ttl=args.lost_track_ttl,
    history_len=args.history_len,
    homography=load_homography(args.homography),
)
```

- `model_path`: trọng số YOLO `.pt`; file phải tồn tại.
- `tracker_config`: YAML BoT-SORT.
- `confidence`: ngưỡng detection, mặc định CLI 0.25.
- `iou`: IoU dùng trong NMS, mặc định 0.5.
- `imgsz`: kích thước inference, mặc định 960.
- `device`: CPU hoặc GPU.
- `homography`: tùy chọn chiếu bottom-center sang tọa độ mặt đất.

## A3. Lời gọi `YOLO.track()`

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
results = self.model.track(frame, **kwargs)
```

- `persist=True`: giữ state tracker giữa các lần gọi frame; nếu false, mỗi frame có thể thành phiên mới.
- `tracker`: chọn cấu hình BoT-SORT trong `config/botsort_parking_reid.yaml`.
- `classes`: mặc định COCO `(2,3,5,7)` — car, motorcycle, bus, truck.
- `conf`: bỏ detection có confidence thấp hơn ngưỡng.
- `iou`: điều khiển loại bbox YOLO trùng trong NMS.
- `imgsz`: ảnh được resize/pad cho inference.

Kết quả `boxes.id` là ID BoT-SORT. `_update_track()` chuyển `xyxy` thành `(x,y,w,h)`, lấy bottom-center, cập nhật `TrackedVehicle`; `_age_missing_tracks()` chuyển track không thấy sang lost rồi expired.

## A4. BoT-SORT và Re-ID trong config hiện tại

Các giá trị chính:

```yaml
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.35
track_buffer: 90
match_thresh: 0.80
fuse_score: true
gmc_method: none
proximity_thresh: 0.50
appearance_thresh: 0.25
with_reid: true
model: auto
```

- Detection confidence cao được ưu tiên nối trước; confidence thấp hỗ trợ cứu track.
- `track_buffer=90` giữ track mất ngắn hạn.
- `fuse_score=true` kết hợp score detection trong association.
- `gmc_method=none` vì giả định camera cố định, không bù chuyển động camera.
- `with_reid=true` bật đặc trưng ngoại hình trong tracker.
- `proximity_thresh` buộc bbox đủ gần trước khi tin appearance.

Tên và ý nghĩa chi tiết của threshold phụ thuộc phiên bản Ultralytics đang cài; khi báo cáo thực nghiệm phải ghi phiên bản thư viện.

## A5. Ba loại ID không được gọi chung

| ID | Ai cấp? | Phạm vi |
|---|---|---|
| YOLO detection | Không có ID bền; chỉ bbox/class/confidence | Một frame |
| BoT-SORT track ID | BoT-SORT | Một tracker/một nguồn |
| TechGAR Global ID | `CrossCameraManager` | Toàn hệ thống camera |

`persist=True` không tự làm BoT-SORT ID trở thành Global ID xuyên camera. Hai instance BoT-SORT khác nhau vẫn có namespace riêng.

---

# PHỤ LỤC B — BẢNG THAM SỐ MẶC ĐỊNH CỦA PIPELINE `main.py`

| Nhóm | Tham số | Mặc định | Vai trò |
|---|---|---:|---|
| Crop | `overlap` | 0 px | Không lặp pixel ở biên mặc định. |
| Parking | `parking_fps` | 2 Hz | Nhịp chạy ParkingDetector. |
| Parking | `parking_smoothing` | 5 lần | Số kết quả liên tiếp trước đổi vision state. |
| Parking | `base_gamma` | 2.4 | Gamma trung tâm của 25 biến thể. |
| Parking | `base_clahe` | 2.0 | CLAHE clip trung tâm. |
| Parking | `ratio_thr` | 0.20 | Ngưỡng foreground ratio cho phiếu trống. |
| Parking | `edge_thr` | 0.25 | Ngưỡng Canny edge rescue. |
| Local track | `min_visible_count` | 4 | Số observation trước confirmed. |
| Local track | `lost_track_ttl` | 90 frame | Thời gian giữ lost track. |
| Local track | `motion_min_area` | 900 px² | Contour tối thiểu do `main.py` truyền. |
| Local track | `motion_max_distance` | 180 px | Gate khoảng cách Kalman. |
| Local track | `motion_min_displacement` | 12 px | Chuyển động tối thiểu để confirmed. |
| Handoff | `handoff_ttl` | 45 frame | Thời gian giữ hồ sơ bàn giao. |
| Handoff | `lookahead_frames` | 16 | Dự đoán sớm tới biên. |
| Handoff | `prediction_radius` | 90 px | Sai số vị trí tối đa. |
| Handoff | `appearance_threshold` | 0.45 | Sai khác HSV tối đa. |
| Handoff | `min_direction_cosine` | 0.25 | Mức cùng hướng tối thiểu. |
| Slot | `stop_seconds` | 1.0 s | Cửa sổ chứng minh dừng. |
| Slot | `exit_seconds` | 0.5 s | Thời gian ngoài ROI trước release. |
| Slot | `min_vehicle_overlap` | 0.35 | Overlap khi tâm trong ROI. |
| Slot | `strong_vehicle_overlap` | 0.60 | Overlap đủ mạnh dù tâm ngoài. |
| Slot | `stationary_radius_ratio` | 0.06 | Ngưỡng r95 theo đường chéo bbox. |
| Slot | `stationary_drift_ratio` | 0.10 | Ngưỡng drift theo đường chéo bbox. |
| Slot | `recovery_expand_ratio` | 0.15 | Mở ROI để nhận ID khi xe chạy lại. |
| Output | `json_fps` | 5 Hz | Nhịp ghi JSON cho web. |

Lưu ý: `handoff_match_distance=100` hiện chủ yếu còn dùng cho đối chiếu trong vùng crop overlap; handoff dự đoán chính dùng `prediction_radius=90`. `slot_bind_confirmations` được giữ vì tương thích API cũ, còn quyết định parked hiện dựa cửa sổ timestamp và thống kê dừng.

---

# CHECKLIST TRƯỚC KHI THUYẾT TRÌNH

- [ ] Nói rõ bốn góc nhìn hiện là bốn crop mô phỏng.
- [ ] Phân biệt detection, local track, Local ID, Global ID và canonical ID.
- [ ] Giải thích từ tiếng Anh ngay lần đầu dùng.
- [ ] Không gọi motion backend là AI nhận diện xe.
- [ ] Không nói Kalman kết luận xe dừng.
- [ ] Không gọi voting 12/25 là majority tuyệt đối.
- [ ] Không nói YOLO chạy mặc định trong `main.py`.
- [ ] Không tuyên bố phần trăm chính xác khi chưa có ground truth/evaluator.
- [ ] Khi mở `update_all_tracks()`, trình bày cả call graph hàm con.
- [ ] Dùng ví dụ G#2 cam4→cam3→P056 xuyên suốt để tránh giải thích rời rạc.
- [ ] Kết luận bằng đóng góp, giới hạn và kế hoạch đo, không chỉ kể tên thư viện.
