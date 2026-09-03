# Phan tich loi tracking ID va ke hoach khac phuc

## 1. Pham vi du lieu da kiem tra

Thu muc ket qua:

```text
D:\techgar\TechGAR_Parking - Copy\backend\main_detect\experiment_test\output\droidcam_shared_m_04\droidcam_shared_m_04
```

Các tệp chính:

- `raw_cam1.mp4`
- `raw_cam2.mp4`
- `debug_cam1.mp4`
- `debug_cam2.mp4`
- `predictions.schema2.jsonl`
- `evaluation_report_v3.md`
- `evaluation_results_v3.json`
- `frame_timestamps.csv`
- `performance.csv`
- `ground_truth_events.legacy.csv`
- `ground_truth_slots.legacy.csv`
- `session_info.schema2.json`

Phiên có:

- 976 frame
- FPS đánh giá: 25 FPS
- 2 camera: `cam1`, `cam2`
- Thời lượng khoảng 39 giây
- 4 xe vật lý: `M04_V1`, `M04_V2`, `M04_V3`, `M04_V4`

Kết quả tổng thể:

```text
Classification: FAIL
Practical System Score: 40.85/100
Critical errors: 1
```

## 2. Các loại lỗi đang xảy ra

### 2.1. GID bị dùng chung cho hai xe vật lý

Đây là lỗi nghiêm trọng nhất:

```text
Canonical GID 3 is observed for multiple physical vehicles
```

Cụ thể:

- `M04_V1` được ánh xạ vào GID `3`.
- `M04_V4` cũng bị ánh xạ vào GID `3`.
- Phạm vi lỗi: frame `300-650`.
- Xuất hiện ở cả `cam1` và `cam2`.

Đây là **identity collision** hoặc **ID merge sai**. Hai xe vật lý khác nhau bị hệ thống coi là cùng một xe.

Nó nghiêm trọng hơn ID loss thông thường vì:

- Một xe có thể bị mất ID.
- Một GID dùng cho hai xe khiến dữ liệu lịch sử của hai xe bị trộn lẫn.
- Parking binding, camera handover, departure recovery và session management sau đó đều có thể bị sai theo.

Báo cáo chỉ ra bằng chứng:

```text
M04_V1 -> GID 3 tại frame 300, cam2
M04_V1 -> GID 3 tại frame 650, cam1
M04_V4 -> GID 3 tại frame 601, cam1
```

### 2.2. ID loss hoặc không có GID tại checkpoint

Nhiều checkpoint bắt buộc không có Global ID hợp lệ:

- `M04_V1`: thiếu ID ở frame `40`, `80`, `200`, `350`, `450`, `520`, `584`.
- `M04_V2`: thiếu ID ở frame `100`, `200`, `280`, `300`, `350`, `700`.
- `M04_V3`: thiếu ID ở frame `100`, `545`, `560`.
- `M04_V4`: thiếu ID ở frame `100`, `610`, `650`, `700`.

Đây là **ID loss**, **unassigned observation** hoặc **identity continuity failure**.

Một số trường hợp không phải ID bị đổi ngay lập tức, mà là xe tồn tại trong tracking nhưng local track chưa được gắn vào Global ID.

### 2.3. Xe bị tạo Global ID mới thay vì khôi phục ID cũ

Báo cáo ghi nhận các GID tồn tại lâu nhưng không gắn với slot hoặc không chứng minh được danh tính:

```text
GID 3: visible 302 frames, không sở hữu/reserve slot
GID 4: visible 88 frames, không sở hữu/reserve slot
GID 7: visible 226 frames, không sở hữu/reserve slot
GID 8: visible 46 frames, không sở hữu/reserve slot
GID 9: visible 44 frames, không sở hữu/reserve slot
```

Điều này cho thấy pipeline có xu hướng:

1. Local tracker tạo fragment mới.
2. Fragment được duy trì đủ lâu.
3. Fragment chưa được Re-ID chính xác về identity cũ.
4. Nó tồn tại như một GID mới hoặc GID không hoàn chỉnh.

Đây là lỗi **fragmentation**, **delayed identity recovery** và **new identity allocation quá sớm**.

### 2.4. Xe `M04_V4` không được nhận dạng độc lập

Báo cáo:

```text
M04_V4 has no required independent checkpoint matched to a valid GID
```

Mapping cuối:

```json
{
  "M04_V1": 3,
  "M04_V2": 4,
  "M04_V3": 7,
  "M04_V4": null
}
```

Đây là dấu hiệu trực tiếp cho thấy xe `M04_V4` bị bỏ sót ở giai đoạn khởi tạo, bị gộp vào GID của xe khác hoặc bị giữ ở trạng thái local fragment không được cấp Global ID.

### 2.5. Slot binding thất bại

Các lỗi binding:

- `cam1/F01` không bind được GID `7`.
- `cam1/F02` không bind được GID `3`.
- `cam1/F03` không bind được GID của `M04_V4`.
- `cam2/B04` không bind được GID `4`.
- `cam2/B05` không bind được GID `3`.

Các lỗi này không nhất thiết bắt nguồn từ bộ nhận diện ô đỗ. Occupancy detector đang hoạt động khá tốt:

```text
Occupied F1: 0.9729
Balanced accuracy: 0.9735
False occupied rate: 0.0003
```

Vấn đề chính nằm ở phần **gắn danh tính xe vào trạng thái ô đỗ**, không phải xác định ô có xe hay không.

## 3. Kiến trúc tracking hiện tại

### 3.1. Tracking cục bộ theo từng camera

Pipeline trong `two_camera.py` tạo một `MotionVehicleTracker` cho từng camera. Mỗi camera có namespace Local ID riêng:

```text
cam1/local_track_id
cam2/local_track_id
```

`MotionVehicleTracker` sử dụng:

- Background subtraction MOG2.
- Temporal frame difference.
- Morphological operations.
- Kalman filter constant velocity.
- LAPJV assignment.
- HSV/LAB histogram appearance.
- Reacquisition trong thời gian ngắn.
- Logic xử lý merged contour.
- Cơ chế trì hoãn assignment khi có ambiguity.

Cấu hình phiên:

```text
min_visible_count: 3
lost_track_ttl: 90
motion_min_area: 650
motion_max_distance: 180
motion_reacquire_seconds: 0.75
motion_lost_appearance_threshold: 0.3
motion_merged_area_ratio: 1.6
```

### 3.2. Hợp nhất Global ID liên camera

`CrossCameraManager` chịu trách nhiệm chuyển Local ID thành Global ID. Nó sử dụng:

- Homography đưa tọa độ hai camera về shared world map.
- Camera adjacency.
- Overlap polygon.
- Dự đoán vị trí bằng vận tốc.
- Hướng chuyển động.
- Kích thước bounding box.
- Appearance gallery.
- Tracklet gallery.
- LAPJV batch assignment.
- Handoff TTL.
- Dormant Re-ID.
- Merge alias giữa các Global ID.
- Slot recovery token.

Cấu hình calibration:

```json
{
  "shared_map_anchor": "bbox_center",
  "handoff_match_distance": 15.0,
  "handoff_prediction_radius": 25.0,
  "dormant_match_distance": 35.0,
  "cross_camera_duplicate_distance": 9.0
}
```

### 3.3. Appearance descriptor hiện tại

`tracklet_descriptor.py` tạo descriptor bằng:

- HSV histogram.
- LAB chromaticity histogram.
- LAB luminance histogram.

Giới hạn:

- Không phải embedding Re-ID học sâu.
- Nhạy với ánh sáng giữa hai camera.
- Nhạy với góc nhìn khác nhau.
- Dễ bị ảnh hưởng bởi nền đường trong bounding box.
- Xe nhìn từ trên cao có ít đặc trưng phân biệt.
- Hai xe cùng màu hoặc hình dáng tương tự có thể có histogram gần nhau.

Tracklet gallery lưu tối đa 12 hoặc 24 mẫu, nhưng nếu đầu vào đã bị trộn nền hoặc crop sai thì gallery sẽ học sai identity.

## 4. Phân tích nguyên nhân

### 4.1. Nguyên nhân trực tiếp của lỗi GID `3` dùng chung

Chuỗi lỗi có khả năng cao:

1. Hai xe ở gần nhau hoặc che khuất nhau.
2. Motion detector tạo contour hợp nhất hoặc không ổn định.
3. Một local track cũ bị mất measurement.
4. Một fragment mới được tạo.
5. Fragment mới được gán vào GID cũ dựa trên khoảng cách world, hướng chuyển động, appearance gallery hoặc handoff candidate.
6. Cơ chế merge/canonical alias coi hai GID hoặc hai local fragment là cùng identity.
7. GID `3` tiếp tục được sử dụng cho cả hai xe.

Trong code có nhiều cơ chế merge:

- `_merge_global_ids`
- `_merge_recently_lost_duplicates`
- `_merge_unique_cross_camera_duplicates`
- `_merge_all_nearby_active_duplicates`
- `_reconcile_handoff_dormant_alias`

Các cơ chế này sửa trường hợp một xe bị tạo hai ID. Tuy nhiên, nếu detector hoặc association đã nhầm hai xe thành cùng một lineage, merge sẽ làm lỗi trở nên gần như không thể đảo ngược.

### 4.2. Lỗi do motion detector khi hai xe chạm nhau

Khi hai xe gần nhau:

- Hai vùng chuyển động có thể nối thành một contour.
- Morphological dilation và closing có thể làm cầu nối giữa hai xe.
- Một contour lớn được xem là merged detection.
- Khi xe tách ra, contour có thể chia thành hai fragment mới.
- Local tracker phải quyết định fragment nào là xe cũ nào là xe mới.

Code đã có bảo vệ như `merged_detection_area_ratio`, `merged_detection_frozen`, `oversized_detection_frozen`, `split_assignment_deferred` và `split_assignment_margin`, nhưng thông tin identity vẫn có thể bị mất trong đoạn xe chạm nhau.

### 4.3. Lỗi do che khuất

Khi một xe che xe khác:

- Bounding box của xe bị che có thể co lại.
- Một phần xe có thể bị xem là xe nhỏ mới.
- Điểm anchor thay đổi mạnh.
- Histogram chứa chủ yếu nền hoặc xe đang che.
- Kalman prediction đi xa measurement thực tế.
- Xe gốc bị chuyển sang `LOST`.
- Fragment mới xuất hiện với Local ID khác.

`lost_track_ttl=90` giữ track khoảng 3,6 giây ở 25 FPS, nhưng giữ track lâu không đồng nghĩa với prediction vẫn chính xác.

### 4.4. Lỗi khi qua ranh giới camera

Camera handover dùng homography, overlap polygon, adjacency, dự đoán vị trí, entry corridor, appearance, direction, size và temporal evidence.

Các ngưỡng hiện tại khá chặt trong khi dữ liệu có:

- Camera skew tới khoảng 47-55 ms ở một số frame.
- Bounding box motion không ổn định.
- Anchor giữa hai góc nhìn có thể không nằm trên cùng một điểm vật lý.
- Hai camera khác nhau về perspective và ánh sáng.

Khi source camera mất xe trước khi target camera có track ổn định:

1. Handoff được mở.
2. Target fragment xuất hiện.
3. Fragment còn tentative hoặc chưa đủ appearance samples.
4. Handoff không đạt đủ evidence.
5. Target fragment bị trì hoãn hoặc tạo GID mới.
6. Handoff cũ hết TTL hoặc bị fragment khác cạnh tranh.
7. Xe bị ID loss hoặc duplicate GID.

### 4.5. Camera synchronization chưa loại bỏ hoàn toàn rủi ro

Skew ảnh hưởng đến:

- Vị trí tương ứng của cùng xe giữa hai camera.
- So sánh vận tốc.
- Direction cosine.
- Dự đoán thời điểm xe xuất hiện ở camera kế tiếp.
- Batch association khi nhiều xe cùng nằm trong overlap.

Ghép hai frame theo `frame_idx` không luôn có nghĩa hai camera quan sát cùng một thời điểm vật lý.

### 4.6. Nhiều xe cùng nằm trong overlap

LAPJV tìm nghiệm tối ưu theo cost matrix, nhưng không đảm bảo đúng về vật lý nếu:

- Candidate matrix có cost sai.
- Appearance của hai xe quá giống nhau.
- World position bị lệch do calibration.
- Một xe bị detector mất còn xe kia vẫn được nhìn thấy.
- Fragment của xe A nằm gần trajectory của xe B.

### 4.7. Merge alias quá mạnh khi evidence chưa đủ

Nguyên tắc an toàn nên là:

```text
Không merge nếu chưa đủ bằng chứng chắc chắn.
ID duplicate tạm thời ít nguy hiểm hơn ID collision.
```

Cần đặc biệt kiểm soát:

- Merge giữa hai GID đều đã tồn tại lâu.
- Merge khi cả hai GID đều đang active.
- Merge khi trajectory không liên tục.
- Merge khi hai GID cùng có slot reservation.
- Merge khi hai GID đồng thời xuất hiện ở hai vị trí khác nhau.
- Merge chỉ dựa trên khoảng cách và histogram.

## 5. Phân tích cấu hình phiên

### 5.1. Motion tracking

```text
min_visible_count: 3
lost_track_ttl: 90
motion_min_area: 650
motion_max_distance: 180
motion_min_displacement: 6
motion_threshold: 20
motion_min_pixels: 100
motion_min_ratio: 0.05
motion_reacquire_seconds: 0.75
motion_lost_appearance_threshold: 0.3
motion_merged_area_ratio: 1.6
```

Điểm đáng chú ý:

- `motion_max_distance=180` có thể tạo nhiều candidate khi xe gần nhau.
- `motion_reacquire_seconds=0.75` tương đương khoảng 19 frame.
- `lost_track_ttl=90` lớn hơn nhiều cửa sổ reacquire.
- `motion_min_displacement=6` thấp, dễ xác nhận fragment ít chuyển động.
- MOG2 vẫn nhạy với ánh sáng, bóng và nén video.

### 5.2. Handoff

```text
handoff_ttl: 45 frames
handoff_match_distance: 15 cm
handoff_prediction_radius: 25 cm
handoff_min_direction_cosine: 0.25
cross_camera_defer_frames: 8
handoff_lookahead_frames: 16
```

Điểm đáng chú ý:

- Handoff TTL khoảng 1,8 giây ở 25 FPS.
- Tăng TTL mà không cải thiện candidate gating sẽ tăng nguy cơ gán nhầm.
- `min_direction_cosine=0.25` khá permissive.
- `cross_camera_defer_frames=8` khoảng 320 ms, có thể chưa đủ cho fragment bị che.

### 5.3. Re-ID

```text
tracklet_max_samples: 12
tracklet_sample_interval: 3
global_gallery_max_samples: 24
dormant_match_distance: 35 cm
dormant_appearance_threshold: 0.60
```

Gallery dài không tự động nâng chất lượng nếu sample bị crop sai, chứa nền, đến từ góc nhìn khác hoặc được thu thập trong lúc xe bị che.

## 6. Kết luận kỹ thuật

Đây là lỗi liên chuỗi:

```text
Detection không ổn định
    -> Merged/fragmented local tracks
    -> Local ID mất hoặc đổi
    -> Handoff thiếu hoặc sai candidate
    -> Appearance/world association không đủ phân biệt
    -> Merge GID sai
    -> Một canonical GID dùng cho nhiều xe
```

Ưu tiên xử lý:

1. Ngăn detector tạo measurement hợp nhất sai.
2. Ngăn local tracker assignment tùy tiện khi xe tách ra.
3. Ngăn Global ID merge nếu evidence chưa đủ chắc chắn.
4. Cải thiện camera handover bằng topology và trajectory vật lý.
5. Chỉ cấp GID mới sau khi loại trừ fragment của identity cũ.
6. Phát hiện identity collision sớm trong runtime.

## 7. Hướng giải quyết đề xuất

### 7.1. Cải thiện detector

#### Ưu tiên detector object-level

Nên đánh giá theo thứ tự:

1. YOLO fine-tune cho góc nhìn top-down của dữ liệu DroidCam.
2. YOLO segmentation hoặc oriented bounding box nếu bbox thường bị dính.
3. Detector object-level chạy mỗi frame hoặc mỗi vài frame.
4. Motion mask chỉ hỗ trợ temporal tracking, không là nguồn detection duy nhất.

#### Nếu tiếp tục dùng motion detector

- Adaptive morphology theo kích thước xe.
- Giảm closing kernel ở vùng mật độ cao.
- Distance transform để tách merged blob.
- Watershed hoặc contour splitting.
- Kiểm tra số peak trên foreground mask.
- So sánh với predicted track centers.
- Tạo trạng thái `merged_observation` thay vì cố gán contour cho một xe.

#### Multi-hypothesis trong merged interval

- Không gán contour hợp nhất cho một xe duy nhất.
- Giữ cả hai Kalman tracks ở trạng thái coasting.
- Lưu predicted states của từng xe.
- Khi contour tách, đánh giá toàn bộ chuỗi fragment.
- Không cấp ID mới nếu assignment margin chưa rõ.

### 7.2. Cải thiện local tracker

#### State machine rõ hơn

```text
TENTATIVE
CONFIRMED
OCCLUDED
MERGED
LOST
REACQUIRING
RETIRED
```

#### Không cập nhật từ merged detection

Khi detection hợp nhất:

- Freeze position measurement.
- Chỉ chạy Kalman prediction.
- Không update appearance.
- Không update kích thước chuẩn.
- Không dùng merged crop làm gallery sample.

#### Association tốt hơn

Cost nên gồm:

```text
Mahalanobis motion distance
IoU hoặc generalized IoU
appearance embedding distance
bbox size/ratio
velocity consistency
direction consistency
occlusion state
camera-zone prior
```

#### Xử lý split bằng track lineage

- Dùng lịch sử trước merged interval.
- Tính continuity, appearance, predicted position, direction và area score.
- Yêu cầu nghiệm one-to-one có margin đủ lớn.
- Nếu ambiguous, giữ track ở `REACQUIRING` và không cấp GID mới.

### 7.3. Cải thiện Re-ID

#### Mask background tốt hơn

- Dùng segmentation mask hoặc inner bbox crop.
- Ưu tiên vùng thân xe.
- Không lấy crop khi merged hoặc occluded.
- Lưu quality metadata cho mỗi appearance sample.

Metadata nên gồm:

```text
camera_id
frame_idx
visibility_score
occlusion_score
bbox_area
foreground_ratio
blur_score
brightness_score
```

#### Hiệu chỉnh màu giữa camera

- White balance calibration.
- Histogram normalization.
- CLAHE hoặc gamma thống nhất.
- Color transfer giữa camera.
- Camera-specific normalization.

#### Deep Re-ID embedding

Đánh giá:

- OSNet.
- FastReID.
- Torchreid.
- Vehicle Re-ID model phù hợp.

Kết hợp feature:

```text
appearance_cost =
    0.60 * deep_embedding_distance
  + 0.25 * HSV/LAB_distance
  + 0.15 * shape/size_distance
```

#### Camera-specific gallery

```text
identity.gallery[cam1]
identity.gallery[cam2]
identity.gallery[global]
```

- So với target-camera gallery trước.
- Nếu chưa có target gallery, dùng global gallery với confidence thấp hơn.
- Không trộn sample chất lượng thấp.
- Không cập nhật gallery sau assignment chưa chắc chắn.

### 7.4. Cải thiện camera handover

#### Hiệu chỉnh overlap

Kiểm tra lại:

- Homography của hai camera.
- `overlap_world_polygon`.
- Camera coverage polygon.
- Hướng đi thực tế.
- `bbox_center`, `bottom_center` hoặc anchor từ segmentation.

#### Transfer corridor

Mỗi hướng handover nên có:

```text
source exit corridor
expected world path
minimum/maximum travel time
allowed direction
```

#### Dùng thời gian thay vì frame

Dùng:

- `timestamp_s`
- `last_seen_time`
- `handoff_created_time`
- `expected_arrival_time`

Các ngưỡng:

```text
min_transfer_time_s
max_transfer_time_s
handoff_ttl_s
```

#### Batch handover

1. Thu thập tất cả source exits.
2. Thu thập tất cả target entries.
3. Tạo candidate matrix.
4. Gating theo topology.
5. Gating theo corridor.
6. Gating theo thời gian.
7. Gating theo hướng.
8. Gating theo appearance.
9. Chạy one-to-one assignment.
10. Kiểm tra assignment margin.
11. Chấp nhận, trì hoãn hoặc từ chối toàn bộ batch.

#### Không merge hai identity trưởng thành chỉ vì overlap

```text
Nếu cả hai GID đã mature và đều active:
    không merge chỉ bằng proximity + appearance.
```

### 7.5. Cải thiện Global ID policy

#### Phân biệt unknown và new identity

Khi chưa đủ bằng chứng:

```text
global_id = null
identity_state = provisional
```

#### Cấm một GID xuất hiện ở hai vị trí không tương thích

Nếu vi phạm:

- Không merge tự động.
- Đánh dấu `identity_collision_suspected`.
- Giữ hai lineage riêng.
- Chọn observation đáng tin hơn cho GID hiện tại.
- Fragment còn lại trở về `provisional`.
- Ghi diagnostics đầy đủ.

#### Merge hai pha

```text
candidate duplicate
    -> merge pending
    -> merge committed
```

Yêu cầu evidence liên tục 3-5 frame và kiểm tra không có physical conflict trước khi tạo canonical alias.

#### Merge có lineage record

Lưu:

```text
merge_id
canonical_id
retired_id
reason
evidence_frames
spatial_residual
appearance_distance
trajectory_score
conflict_check
```

## 8. Kế hoạch triển khai chi tiết

### Giai đoạn 1: Xây dựng bộ chẩn đoán lỗi

1. Chuẩn hóa identity event log cho từng local track.
2. Log mọi quyết định merge với đầy đủ evidence.
3. Xuất visual diagnostic cho Local ID, Global ID, predictions, overlap và merge.
4. Tạo timeline theo từng physical vehicle.

Mục tiêu là phân biệt rõ ID switch, ID loss, duplicate GID, fragmentation, handoff delay và slot binding failure.

### Giai đoạn 2: Sửa local motion tracker

1. Thêm trạng thái `MERGED`, `OCCLUDED`, `REACQUIRING`.
2. Freeze track và appearance khi contour bị merged.
3. Thử tách blob bằng distance transform và watershed.
4. Dùng Mahalanobis distance và uncertainty-aware gating.
5. Quarantine fragment sau split cho tới khi assignment đủ rõ.

### Giai đoạn 3: Cải thiện appearance Re-ID

1. Lọc sample kém chất lượng.
2. Chuẩn hóa màu riêng từng camera.
3. Benchmark descriptor hiện tại bằng same-ID và different-ID pairs.
4. Thử deep embedding và so sánh A/B với HSV/LAB.

### Giai đoạn 4: Sửa camera handover

1. Xác minh calibration và anchor.
2. Đo transfer time từ ground truth.
3. Thay frame-based TTL bằng time-based TTL.
4. Tạo handover corridor theo hướng.
5. Batch assignment với row/column uniqueness margin.
6. Kiểm tra active owner và trajectory trước khi bind.

Ground truth handover hiện có:

```text
M04_V2: cam2 -> cam1, frame 240-320
M04_V1: cam2 -> cam1, frame 315-360
```

### Giai đoạn 5: Sửa Global ID và merge policy

1. Dùng provisional identity trước khi cấp GID.
2. Thêm collision guard trước mọi bind/merge.
3. Áp dụng merge probation nhiều frame.
4. Hạn chế merge hai GID trưởng thành.
5. Tách xử lý identity duplicate và identity collision.

### Giai đoạn 6: Slot binding và parking recovery

1. Không để slot binding quyết định GID quá sớm.
2. Dùng slot làm prior về origin, direction, corridor và appearance.
3. Khóa reservation tạm thời nếu GID bị nghi identity collision.

### Giai đoạn 7: Kiểm thử hồi quy

Unit test cần có:

- Hai xe chạm nhau rồi tách.
- Một xe bị che hoàn toàn.
- Hai xe cùng màu.
- Một contour bao phủ hai predicted tracks.
- Một xe qua camera boundary khi target detection chậm.
- Hai xe cùng xuất hiện trong overlap.
- Handoff candidate ambiguity.
- Merge hai GID provisional.
- Từ chối merge hai GID mature.
- Một GID xuất hiện ở hai vị trí.
- Camera timestamp skew.
- Source biến mất trước target.
- Target fragment chỉ có 1-2 frame.

Replay acceptance cho `droidcam_shared_m_04`:

- Không còn `gid_shared_between_vehicles`.
- `M04_V4` có identity độc lập.
- Các checkpoint bắt buộc đều có GID.
- `M04_V1` giữ cùng GID từ B05 qua F02.
- `M04_V2` giữ cùng GID từ B04 qua handoff `cam2 -> cam1`.
- `M04_V3` giữ cùng GID khi rời F01.
- `M04_V4` giữ identity khi rời F03.
- Slot binding không bị mất do identity collision.

Tiêu chí mục tiêu:

```text
Critical identity collisions: 0
GID shared between physical vehicles: 0
Required checkpoint coverage: 100%
Camera handoff continuity: >= 95%
Departure recovery: >= 95%
Slot identity ownership: >= 95%
Practical System Score: >= 85
```

## 9. Thứ tự ưu tiên thực hiện

1. Thêm diagnostics để xác định chính xác frame bắt đầu merge sai.
2. Cấm merge khi có xung đột physical observation.
3. Freeze appearance và bbox size trong merged/occluded interval.
4. Chuyển fragment mới sang provisional thay vì cấp GID ngay.
5. Cải thiện split assignment sau merged contour.
6. Hiệu chỉnh handoff corridor và time window bằng ground truth.
7. Cải thiện camera synchronization và time-based handoff.
8. Benchmark HSV/LAB hiện tại.
9. Thử deep vehicle Re-ID.
10. Chạy replay regression và điều chỉnh threshold.

## 10. Kết luận

Occupancy detection đang tốt, nhưng identity management chưa an toàn. Lỗi trung tâm là **GID collision**, trong đó GID `3` bị dùng cho cả `M04_V1` và `M04_V4` trong frame `300-650`.

Nguyên nhân có khả năng là sự kết hợp của merged motion contour, fragment local track, handoff thiếu chắc chắn và chính sách merge Global ID quá mạnh khi evidence chưa đủ.

Giải pháp không nên chỉ là tăng `lost_track_ttl` hoặc nới `appearance_threshold`. Chính sách an toàn cần là:

```text
Khi không chắc chắn, giữ identity ở trạng thái provisional/ambiguous.
Không tạo GID mới quá sớm.
Không merge hai lineage nếu chưa chứng minh được chúng là cùng một xe.
Không cho một GID đồng thời đại diện cho hai vị trí vật lý khác nhau.
```
