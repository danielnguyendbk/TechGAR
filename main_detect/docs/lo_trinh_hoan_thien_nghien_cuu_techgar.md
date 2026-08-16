# Lộ trình cô đọng để hoàn thiện TechGAR theo hướng nghiên cứu khoa học

## 1. Khóa đúng trọng tâm

Không mở rộng thêm chức năng trong giai đoạn này. Trọng tâm nên là:

> **Đánh giá việc kết hợp nhận diện ROI với Global Vehicle ID và trạng thái dừng theo thời gian có làm giảm lỗi nhận diện ô đỗ và mất ID so với phương pháp vision-only hay không.**

Tên hướng nghiên cứu đề xuất:

> **Identity-Aware Temporal Fusion for Robust Parking-Slot Occupancy and Cross-Camera Vehicle Tracking**

Phạm vi tuyên bố:

- Bãi xe cố định, camera trên cao và đã hiệu chỉnh ROI.
- Phiên bản bốn crop hiện tại chỉ là mô phỏng thuật toán.
- Kết quả chính phải được kiểm tra lại bằng ít nhất hai camera/điện thoại thật.
- Không tuyên bố hoạt động tổng quát cho camera đường phố, góc thấp hoặc camera chưa calibration.

## 2. Ba câu hỏi nghiên cứu phải trả lời bằng số

1. **RQ1 – Parking:** Tracking-aware fusion giảm bao nhiêu lỗi ô có xe nhưng báo trống so với ROI vision-only?
2. **RQ2 – Global ID:** Predictive handoff giảm bao nhiêu ID switch so với cấp ID riêng từng camera và nearest-neighbor?
3. **RQ3 – Hiệu năng:** Các cải tiến làm thay đổi FPS, độ trễ, CPU và RAM bao nhiêu?

Nếu báo cáo chưa trả lời được ba câu này bằng bảng số liệu thì đề tài vẫn mới dừng ở prototype.

## 3. Việc cần làm theo đúng thứ tự

### Bước 1 – Đóng băng phiên bản baseline

- Giữ một commit cố định cho code hiện tại.
- Lưu toàn bộ tham số mặc định vào một file cấu hình.
- Không tiếp tục chỉnh ngưỡng bằng mắt trên video demo.
- Ghi rõ video bốn crop chỉ dùng để debug và minh họa.

**Kết quả cần có:** một phiên bản có thể chạy lại và cho cùng kết quả với cùng dữ liệu.

### Bước 2 – Thu dữ liệu camera thật

Thiết lập tối thiểu:

- 2–4 điện thoại/camera ở góc nhìn khác nhau.
- Có vùng chuyển giao thật giữa các camera.
- Ít nhất hai cách bố trí hoặc góc camera.
- Thu ở ba điều kiện sáng: sáng ổn định, bóng đổ/thay đổi sáng và ánh sáng yếu.

Số tình huống tối thiểu:

- 120 lần xe chuyển camera.
- 120 lần xe vào hoặc rời ô.
- Có xe nhanh, xe chậm, dừng tạm, hai xe gần nhau, che khuất và mất camera.
- Ít nhất 20.000 slot-frame được gán nhãn.

Mỗi frame phải lưu timestamp nhận được để đo độ trễ và xử lý camera không đồng bộ.

**Kết quả cần có:** bộ video thực nghiệm có danh sách kịch bản và điều kiện quay.

### Bước 3 – Tạo ground truth

Ground truth là dữ liệu nhãn chuẩn để so với kết quả thuật toán. Schema tối thiểu:

```text
frame_index
timestamp
camera_id
vehicle_global_id
bbox_x, bbox_y, bbox_w, bbox_h
slot_id
occupied
event: enter / park / leave / handoff
```

Quy tắc:

- Một xe giữ cùng `vehicle_global_id` ở mọi camera.
- Ghi rõ frame xe bắt đầu vào ô, dừng, rời ô và chuyển camera.
- Tách video theo buổi quay hoặc camera setup; không chia ngẫu nhiên các frame liền nhau vào train/test.
- Kiểm tra chéo một phần nhãn bởi thành viên thứ hai.

**Kết quả cần có:** file nhãn chuẩn và tài liệu hướng dẫn gán nhãn.

### Bước 4 – Viết evaluator tự động

Evaluator phải đọc ground truth và JSON đầu ra để tính:

**Nhận diện ô đỗ**

- Precision, Recall và F1 cho lớp occupied.
- Precision, Recall và F1 cho lớp free.
- Macro-F1 hoặc Balanced Accuracy.
- False-free rate: ô có xe nhưng báo trống.
- False-occupied rate: ô trống nhưng báo có xe.
- Transition delay: thời gian từ sự kiện thật đến khi JSON cập nhật.
- Flicker rate: số lần trạng thái đỏ/xanh đổi sai trong một phút.

**Tracking và Global ID**

- IDF1.
- ID switch.
- Track fragmentation.
- Handoff precision và handoff recall.
- Thời gian nhận lại ID ở camera mới.
- Số lần một Global ID xuất hiện thành hai xe trên map.

**Hiệu năng**

- FPS với 1, 2 và 4 camera.
- Độ trễ frame → JSON ở p50 và p95.
- CPU, RAM và GPU nếu có.

**Kết quả cần có:** một lệnh chạy evaluator và sinh bảng CSV/Markdown tự động.

### Bước 5 – Chạy baseline và ablation

#### Nhóm nhận diện ô đỗ

| Mã | Phiên bản |
|---|---|
| P0 | Vision ROI đơn |
| P1 | P0 + temporal smoothing |
| P2 | P1 + bbox–polygon overlap |
| P3 | P2 + stop detection |
| P4 | P3 + Global ID binding |
| P5 | Full: ID recovery + merge + vision/tracking fusion |

#### Nhóm Global ID

| Mã | Phiên bản |
|---|---|
| T0 | Mỗi camera cấp ID độc lập |
| T1 | Ghép theo khoảng cách gần nhất |
| T2 | T1 + dự đoán vận tốc |
| T3 | T2 + HSV, kích thước và hướng |
| T4 | T3 + LAPJV |
| T5 | Full: slot recovery + duplicate merge |

Mỗi phiên bản phải chạy trên cùng dữ liệu và cùng ground truth. Kết quả phải chỉ ra thành phần nào tạo cải thiện, không chỉ chứng minh full system chạy tốt.

**Kết quả cần có:** hai bảng ablation, kèm FPS/độ trễ của từng phiên bản.

### Bước 6 – Kiểm tra độ nhạy tham số

Không chỉ báo một bộ tham số tốt nhất. Cần thử quanh các giá trị:

- Trọng số ghép ô: `0.70 / 0.20 / 0.10`.
- Handoff prediction radius.
- HSV appearance threshold.
- Stop seconds và exit seconds.
- Stationary radius/drift ratio.
- Ngưỡng voting 11/25, 12/25 và 13/25.

Mục tiêu là chứng minh kết quả không sụp đổ khi tham số thay đổi nhẹ.

**Kết quả cần có:** bảng hoặc biểu đồ sensitivity analysis.

### Bước 7 – Hoàn thiện mô hình camera thật

- Hiệu chỉnh homography cho từng camera:

```text
(x_image, y_image) → (X_parking_map, Y_parking_map)
```

- Khai báo topology camera và vùng exit/entry hợp lệ.
- Đồng bộ timestamp hoặc có buffer chịu được camera trễ.
- Thêm kiểm tra camera bị rung/lệch ROI.
- Dùng tọa độ map chung thay cho pixel của crop khi tính handoff.

**Kết quả cần có:** demo ít nhất hai camera thật giữ đúng một Global ID trên web map.

### Bước 8 – Phân loại và báo cáo lỗi

Tạo failure taxonomy gồm:

- Motion echo tạo hai bbox.
- Xe chạy nhanh mất handoff.
- Xe bò chậm bị nhận là dừng.
- Xe đứng mất motion detection.
- Hai xe có appearance giống nhau.
- Một xe xuất hiện ở vùng overlap.
- Bóng đổ làm sai parking vision.
- Xe chiếm hai ô.
- Camera rung hoặc lệch ROI.

Mỗi nhóm phải ghi:

- Tổng số trường hợp.
- Số phát hiện đúng/sai.
- Tỷ lệ thuật toán sửa thành công.
- Một hình minh họa thành công và một failure case.

**Kết quả cần có:** bảng failure analysis thay cho mô tả cảm tính.

## 4. Bảng kết quả tối thiểu trong báo cáo

### Parking

| Phiên bản | Occupied F1 | Free F1 | False-free | Delay p95 | Flicker/min | FPS |
|---|---:|---:|---:|---:|---:|---:|
| P0 |  |  |  |  |  |  |
| P1 |  |  |  |  |  |  |
| P2 |  |  |  |  |  |  |
| P3 |  |  |  |  |  |  |
| P4 |  |  |  |  |  |  |
| P5 |  |  |  |  |  |  |

### Global ID

| Phiên bản | IDF1 | ID switch | Handoff precision | Handoff recall | Recovery delay | FPS |
|---|---:|---:|---:|---:|---:|---:|
| T0 |  |  |  |  |  |  |
| T1 |  |  |  |  |  |  |
| T2 |  |  |  |  |  |  |
| T3 |  |  |  |  |  |  |
| T4 |  |  |  |  |  |  |
| T5 |  |  |  |  |  |  |

## 5. Cấu trúc báo cáo nên dùng

1. Vấn đề: ROI vision dao động, báo trống sai và mất xe đứng yên.
2. Khoảng trống: occupancy theo từng frame chưa tận dụng Global ID và lịch sử dừng.
3. Giả thuyết nghiên cứu và ba câu hỏi RQ1–RQ3.
4. Ba đóng góp:
   - predictive Global ID handoff nhẹ;
   - ghép bbox–polygon và stop detection;
   - identity-aware temporal fusion.
5. Công thức, state machine và kiến trúc.
6. Dataset, camera setup và quy trình gán nhãn.
7. Baseline, ablation và sensitivity analysis.
8. Kết quả định lượng.
9. Failure cases và giới hạn.
10. Demo realtime camera → thuật toán → JSON → web map.

## 6. Những việc chưa nên làm

- Không thêm chức năng mới khi chưa có evaluator.
- Không tiếp tục chỉnh tham số chỉ bằng cách xem video.
- Không gọi 25 cấu hình gamma–CLAHE là ensemble nhiều AI model; gọi là multi-parameter consensus voting.
- Không gọi bốn crop là bốn camera vật lý.
- Không tuyên bố “tracking xuyên camera thực tế” trước khi có thí nghiệm camera thật.
- Không chỉ báo Accuracy hoặc một video demo đẹp.
- Không giấu failure case hoặc chi phí giảm FPS.

## 7. Definition of Done

TechGAR có thể được xem là một nghiên cứu hoàn chỉnh hơn khi đạt đủ:

- [ ] Có câu hỏi nghiên cứu RQ1–RQ3 rõ ràng.
- [ ] Có ít nhất hai camera thật và hai setup/góc nhìn.
- [ ] Có ground truth cho occupancy, Global ID và sự kiện chuyển trạng thái.
- [ ] Có evaluator chạy tự động.
- [ ] Có baseline P0/T0.
- [ ] Có bảng ablation P0–P5 và T0–T5.
- [ ] Có sensitivity analysis cho các ngưỡng chính.
- [ ] Có metric accuracy, ID, latency và tài nguyên.
- [ ] Có failure taxonomy và giới hạn tuyên bố.
- [ ] Mọi kết luận trong báo cáo đều đi kèm số liệu hoặc bảng kết quả.

## 8. Năm việc cần bắt đầu ngay

1. Khóa commit hiện tại làm baseline.
2. Viết schema và công cụ gán ground truth.
3. Thu video từ ít nhất hai điện thoại ở hai góc thật.
4. Viết evaluator tính false-free, F1, ID switch, handoff và latency.
5. Chạy P0–P5/T0–T5 trước khi sửa thêm thuật toán.

Tóm lại: **giai đoạn tiếp theo không phải thêm nhiều thuật toán hơn, mà là biến các tuyên bố hiện có thành số liệu có thể lặp lại, so sánh và phản biện.**
