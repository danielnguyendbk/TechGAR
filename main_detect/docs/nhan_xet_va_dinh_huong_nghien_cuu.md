# Nhận xét thẳng về TechGAR và định hướng nghiên cứu

TechGAR hiện tại là một prototype kỹ thuật khá tốt, nhưng chưa phải một công trình nghiên cứu khoa học mạnh.

Nói hơi phũ: bạn đã trả lời được câu hỏi “hệ thống có chạy không?”, nhưng chưa trả lời được câu hỏi quan trọng hơn:

> “Thuật toán mới của tôi cải thiện cái gì, cải thiện bao nhiêu, so với phương pháp nào, trên dữ liệu nào, và có tổng quát sang bãi xe khác không?”

Nếu chấm dưới góc độ sản phẩm demo, tôi cho khoảng **7/10**. Nếu chấm như một nghiên cứu khoa học ở trạng thái hiện tại, chỉ khoảng **3.5–4/10**.

Vấn đề không nằm ở việc ý tưởng yếu. Vấn đề là bạn đang có rất nhiều thuật toán và chức năng, nhưng chưa tổ chức chúng thành một giả thuyết nghiên cứu có thể kiểm chứng.

---

## 1. Các dự án CNTT mạnh trong kỷ yếu tập trung vào gì?

Sau khi xem riêng phần Công nghệ thông tin, tôi thấy các dự án nổi bật đều có một cấu trúc khá giống nhau:

| Dự án | Điểm họ tập trung | Bằng chứng họ đưa ra |
|---|---|---|
| Multi-camera warehouse tracking | Giữ ID xuyên camera, định vị 3D | HOTA, DetA, AssA, LocA; ablation; xếp hạng leaderboard |
| Phát hiện DGA botnet | Hai mô hình bổ trợ nhau có tốt hơn một mô hình không | 80.000 mẫu train, 8.000 test, 36 đặc trưng, F1/FPR/FNR, kết quả từng họ botnet |
| HemoGAT | CNN/Transformer và graph khắc phục nhược điểm của nhau | Hai benchmark, nhiều baseline, accuracy, weighted F1 và ablation |
| ResEViT-Road | Mô hình vừa chính xác vừa nhẹ để chạy edge | Hai dataset, Precision/Recall/F1, tham số, FLOPs, FPS và điện năng trên Raspberry Pi/Jetson |
| Kính hỗ trợ người khiếm thị | Chứng minh sản phẩm có hoạt động trong các môi trường khác nhau | 6.965 ảnh, 16.172 annotation, độ chính xác trong nhà/ngoài trời, sai số cảm biến, FPS |

Nguồn đối chiếu: tài liệu `kyyeu2026.pdf`.

Điểm quan trọng: họ không nhất thiết có sản phẩm phức tạp hơn TechGAR. Họ mạnh hơn vì biết cô lập câu hỏi nghiên cứu và tạo số liệu chứng minh.

Ví dụ, bài multi-camera warehouse gần nhất với TechGAR:

- Có baseline ban đầu.
- Thêm trajectory và 3D IoU.
- Sau đó thêm VGCR.
- Đo HOTA của từng phiên bản.
- Công bố chi phí đánh đổi: VGCR làm inference chậm hơn 3–4 lần.
- Đánh giá trên một benchmark có dữ liệu ẩn và leaderboard.

AI City Challenge quy định rõ một `object_id` phải giữ nguyên xuyên tất cả camera và dùng 3D HOTA để đo đồng thời detection, association và localization. Đây là điểm khiến tuyên bố “giữ ID xuyên camera” trở thành một kết quả đo được, không chỉ là ảnh demo. [AI City Challenge Track 1](https://www.aicitychallenge.org/2025-track1/)

Tuy nhiên, bài đó cũng không hoàn hảo:

- Abstract ghi HOTA 25,4%, nhưng phần khác ghi 28,03%.
- Bảng có baseline 13,84 và full 28,03, nhưng phần thảo luận lại ghi tăng 11,55%, không khớp trực tiếp với phép trừ trong bảng.

Tức là bạn không cần thần tượng hóa họ. Nhưng dù có lỗi báo cáo, họ vẫn có benchmark, metric và ablation — những thứ TechGAR đang thiếu.

---

## 2. TechGAR hiện đang mạnh ở đâu?

Sau khi đối chiếu mã nguồn, phần đáng giá nhất của TechGAR không phải giao diện web hay ô ROI. Nó nằm ở ba cơ chế sau.

### 2.1. Handoff Global ID nhẹ, không cần deep Re-ID

[cross_camera_manager.py](/D:/techgar/main_detect/src/techgar/cross_camera_manager.py:58) hiện đã sử dụng:

- Vị trí và vận tốc dự đoán.
- Quan hệ camera kề nhau.
- Hướng chuyển động.
- Kích thước bbox.
- HSV appearance.
- Ghép một-một bằng LAPJV.
- Canonical hóa và merge ID trùng.
- Phục hồi ID từ handoff hoặc ô đỗ.

Đây là một hướng nghiên cứu hợp lý nếu bạn định vị nó là:

> Một phương pháp Global ID nhẹ cho bãi xe camera cố định, không cần huấn luyện deep Re-ID.

### 2.2. Hợp nhất nhận diện ảnh và lịch sử chuyển động

[slot_vehicle_binder.py](/D:/techgar/main_detect/src/techgar/slot_vehicle_binder.py:72) có logic khá tốt:

- Tính phần giao giữa bbox và polygon ô đỗ.
- Không dùng mỗi tâm bbox.
- Đo `r95` và độ trôi vị trí để xác định xe dừng.
- Chỉ gán xe vào ô nếu xe thực sự dừng.
- Giữ binding khi motion detector mất xe đứng yên.
- Phục hồi Global ID khi xe bắt đầu rời ô.
- Không cho một ID thuộc hai ô hoặc một ô chứa hai ID.

Đây mới là điểm khác biệt có giá trị hơn một bộ phân loại ô trống thông thường.

### 2.3. Vision và tracking có vai trò khác nhau

Bạn đang dùng:

```text
final_occupied = vision_occupied OR tracking_occupied
```

Ý tưởng này hợp lý:

- Vision nhận được xe đã đỗ từ trước.
- Tracking sửa trường hợp vision báo trống sai.
- Tracking không tự ý biến một ô đang đỏ thành xanh.
- Khi xe rời ô, trạng thái quay lại theo vision.

Đây có thể được trình bày như một cơ chế “identity-aware temporal evidence fusion”.

---

## 3. Những điểm đang làm TechGAR yếu về nghiên cứu

### 3.1. Bốn camera hiện tại thực chất là bốn crop

Đây là nhược điểm nghiêm trọng nhất.

Bốn camera ảo hiện tại:

- Cùng một video.
- Cùng timestamp.
- Cùng hệ tọa độ pixel gốc.
- Có vùng ảnh trùng nhau.
- Không có khác biệt thật về góc nhìn, màu sắc, độ trễ và camera calibration.

Do đó bạn chưa thể tuyên bố:

> “Thuật toán đã giải quyết bài toán tracking xuyên nhiều camera thực tế.”

Bạn mới chứng minh được:

> “Thuật toán giữ ID giữa các vùng quan sát được cắt từ cùng một camera.”

Đây vẫn là một bước thử nghiệm hợp lệ, nhưng phải gọi đúng tên. Nếu trình bày quá tay, giảng viên chuyên computer vision sẽ phát hiện ngay.

### 3.2. Chỉ có một video 28 giây

Video hiện tại có:

- 679 frame.
- 24 FPS.
- Độ phân giải 1100×720.
- 69 ô đỗ.
- Khoảng 28,3 giây.

Một video duy nhất không thể chứng minh độ tổng quát. Các tham số như:

- Gamma.
- CLAHE.
- Ngưỡng overlap.
- Handoff radius.
- HSV threshold.
- Stationary ratio.
- TTL.

đều có nguy cơ được chỉnh cho đúng chính video đó.

Đây là overfitting bằng tay, dù bạn không huấn luyện mạng neural.

### 3.3. Chưa có ground truth

Hiện bạn chưa có bộ nhãn chuẩn gồm:

```text
frame, vehicle_global_id, bbox, camera_id, slot_id, occupied
```

Không có ground truth thì không thể biết:

- Có bao nhiêu ID switch.
- Bao nhiêu xe bị mất.
- Bao nhiêu ô đỏ/xanh sai.
- Xe vào ô mất bao lâu mới cập nhật.
- Merge ID có ghép đúng hay ghép nhầm.
- Tham số mới thực sự tốt hơn hay chỉ nhìn có vẻ tốt.

18 unit test hiện tại rất hữu ích để chống regression, nhưng chúng chủ yếu kiểm tra các tình huống nhân tạo. Chúng không thay thế được evaluation trên video được gán nhãn.

### 3.4. “Ensemble” hiện tại dễ bị trình bày quá mức

[parking_detector.py](/D:/techgar/main_detect/src/techgar/parking_detector.py:1) chạy 25 tổ hợp từ 5 biến thể gamma × 5 biến thể CLAHE.

Về kỹ thuật, đây là voting nhiều cấu hình của cùng một pipeline. Các thành viên rất tương quan vì:

- Cùng ảnh.
- Cùng đặc trưng.
- Cùng cách threshold.
- Chỉ khác một ít tham số tiền xử lý.

Tôi khuyên gọi nó là:

> Multi-parameter consensus voting hoặc robust parameter voting.

Không nên mô tả như một “ensemble AI model” vì dễ bị phản biện rằng đây không phải các mô hình độc lập.

### 3.5. Không có baseline và ablation

Bạn đã thêm rất nhiều cơ chế, nhưng chưa chứng minh cơ chế nào thực sự có ích:

- Kalman.
- HSV.
- Motion prediction.
- Direction gate.
- Size gate.
- LAPJV.
- Duplicate merge.
- Slot recovery.
- Stop detection.
- Vision OR tracking.

Nếu full system tốt, hội đồng vẫn có thể hỏi:

> “Tốt là nhờ phần nào? Nếu bỏ HSV có giảm không? LAPJV có hơn greedy matching không? Stop detection có giảm false occupied không?”

Hiện bạn chưa trả lời được.

---

## 4. Trọng tâm nghiên cứu tôi đề xuất

Đừng trình bày đề tài như một “hệ thống bãi đỗ xe thông minh” chung chung. Phạm vi đó quá rộng và không có điểm khoa học sắc nét.

Tên hướng nghiên cứu phù hợp hơn:

> **Phương pháp hợp nhất nhận diện ROI và Global Vehicle ID theo thời gian nhằm nâng cao độ ổn định nhận diện trạng thái ô đỗ trong hệ thống camera cố định.**

Tên tiếng Anh:

> **Identity-Aware Temporal Fusion for Robust Parking-Slot Occupancy and Cross-Camera Vehicle Tracking**

Câu hỏi nghiên cứu trung tâm:

> Liệu việc kết hợp bằng chứng thị giác theo ROI với lịch sử Global ID và trạng thái dừng của xe có giảm lỗi nhận diện ô trống và dao động trạng thái so với phương pháp vision-only hay không?

Ba câu hỏi phụ:

- RQ1: Tracking-aware fusion giảm bao nhiêu lỗi “ô trống giả” so với ROI vision-only?
- RQ2: Motion-predicted handoff giảm bao nhiêu ID switch so với cấp ID độc lập hoặc nearest-neighbor?
- RQ3: Chi phí của các cải tiến này là bao nhiêu FPS, độ trễ, CPU và RAM?

Đây là câu chuyện nghiên cứu gọn hơn, đúng với code hiện có và không bắt buộc bạn phải cạnh tranh trực tiếp với các hệ thống deep Re-ID rất lớn.

---

## 5. Trình bày thuật toán như thế nào?

### 5.1. Matching xe với ô đỗ

Định nghĩa:

\[
O_v = \frac{|B_i \cap R_j|}{|B_i|}
\]

\[
O_s = \frac{|B_i \cap R_j|}{|R_j|}
\]

Trong đó:

- \(B_i\): bbox xe có Global ID \(i\).
- \(R_j\): polygon ô đỗ \(j\).
- \(O_v\): bao nhiêu phần xe nằm trong ô.
- \(O_s\): bao nhiêu phần ô bị xe che phủ.

Điểm ghép hiện tại:

\[
S_{ij}=0.70O_v+0.20O_s+0.10C_{ij}
\]

Sau đó dùng LAPJV để tìm phép gán một-một giữa xe và ô.

Điểm cần nhấn mạnh: các trọng số `0.70, 0.20, 0.10` hiện là heuristic. Bạn phải có sensitivity analysis, ví dụ thử các bộ trọng số khác nhau và cho thấy kết quả không sụp đổ khi thay đổi nhỏ.

### 5.2. Xác định xe dừng

Trong cửa sổ một giây:

\[
r_{95} = P_{95}\left(\lVert p_t-\tilde{p}\rVert\right)
\]

\[
d_{net} = \lVert p_{last}-p_{first}\rVert
\]

Chuẩn hóa bằng đường chéo bbox \(D\):

\[
\frac{r_{95}}{D}\leq \tau_r,\qquad
\frac{d_{net}}{D}\leq \tau_d
\]

Đây là điểm hay của thuật toán: ngưỡng chuyển động được chuẩn hóa theo kích thước xe, thay vì dùng một ngưỡng pixel cứng.

### 5.3. Handoff xuyên camera

Có thể trình bày hàm chi phí hiện tại:

\[
C_{ij} =
0.55E_{position}
+0.30E_{HSV}
+0.10E_{size}
+0.05E_{direction}
\]

Trước khi tính chi phí, ứng viên phải vượt qua các gate:

- Camera đích phải kề camera nguồn.
- Vị trí phải gần tọa độ dự đoán.
- Hướng không được đối nghịch.
- Appearance và kích thước không được khác bất hợp lý.

Sau đó dùng LAPJV để bảo đảm:

- Một ID cũ không ghép cho hai xe.
- Một xe mới không nhận hai ID.

### 5.4. Quyết định trạng thái ô

\[
Occupied_{final} =
Occupied_{vision}\lor ParkedByTracking
\]

Nhưng trong báo cáo phải giải thích đây không phải phép OR tùy tiện:

- Vision cung cấp bằng chứng tức thời.
- Tracking cung cấp bằng chứng theo lịch sử.
- State machine quyết định khi nào tracking đủ đáng tin.
- Global ID cung cấp tính liên tục khi xe mất detection tạm thời.

---

## 6. Bộ thực nghiệm tối thiểu để đề tài trở nên thuyết phục

### Dữ liệu tự thu

Tôi đề xuất tối thiểu:

- 3 ngày hoặc 3 điều kiện chiếu sáng.
- Ít nhất 2 cách bố trí camera khác nhau.
- 3–4 điện thoại hoặc camera thật.
- 120 sự kiện handoff xuyên camera.
- 120 sự kiện xe vào/rời ô.
- Ít nhất 20.000 slot-frame được gán nhãn.
- Có tình huống xe nhanh, chậm, dừng tạm, hai xe gần nhau, che khuất và mất camera.

Đây chưa phải dataset lớn, nhưng đã đủ để một đề tài sinh viên có bằng chứng đáng tin hơn nhiều so với một video demo.

### Dữ liệu công khai

Bạn có thể dùng:

- PKLot có 12.417 ảnh toàn cảnh, khoảng 695.900 patch ô đỗ, gồm ngày nắng, nhiều mây và mưa. [PKLot – UFPR](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/)
- CNRPark+EXT có khoảng 150.000 patch, 164 ô, 9 camera, nhiều góc nhìn, thời tiết, bóng đổ và che khuất. [CNRPark+EXT](https://cnrpark.it/)

Các bộ này phù hợp để đánh giá phần `ParkingDetector`, nhưng không đủ cho Global ID xuyên camera vì chủ yếu là dữ liệu occupancy.

Quan trọng: không chia ngẫu nhiên các frame liền kề vào train/test. Các frame gần nhau gần như giống nhau và làm kết quả cao giả tạo. Nghiên cứu ACPDS chỉ ra rằng chia theo bãi xe chưa từng thấy mới phản ánh khả năng tổng quát; họ còn chạy mỗi cấu hình năm lần và báo sai số chuẩn. [ACPDS paper](https://arxiv.org/abs/2107.12207)

---

## 7. Baseline và ablation bắt buộc

### Thí nghiệm nhận diện ô đỗ

| Phiên bản | Thành phần |
|---|---|
| P0 | Vision ROI đơn |
| P1 | Vision + temporal smoothing |
| P2 | P1 + bbox/polygon overlap |
| P3 | P2 + stop detection |
| P4 | P3 + Global ID binding |
| P5 | Full: ID recovery + merge + vision/tracking fusion |

### Thí nghiệm Global ID

| Phiên bản | Thành phần |
|---|---|
| T0 | Mỗi camera cấp ID riêng |
| T1 | Ghép theo khoảng cách gần nhất |
| T2 | T1 + dự đoán vận tốc |
| T3 | T2 + HSV, kích thước, hướng |
| T4 | T3 + LAPJV |
| T5 | Full: slot recovery và duplicate merge |

Nếu T2 giảm mạnh ID switch thì motion prediction là đóng góp chính. Nếu T5 chỉ cải thiện rất ít thì không nên dành nửa bài báo để nói về merge.

---

## 8. Các chỉ số phải đo

### Parking occupancy

Đừng chỉ báo Accuracy vì số ô có xe/trống có thể mất cân bằng.

Nên dùng:

- Precision, Recall và F1 cho lớp occupied.
- Precision, Recall và F1 cho lớp free.
- Macro-F1 hoặc Balanced Accuracy.
- False-free rate: ô có xe nhưng hệ thống báo trống.
- False-occupied rate: ô trống nhưng báo có xe.
- Transition delay: thời gian từ lúc xe đỗ/rời đến khi JSON cập nhật.
- Flicker rate: số lần một ô đổi đỏ/xanh sai trong một phút.

False-free rate nên là metric quan trọng nhất, vì hướng dẫn xe vào một ô thực tế đang có xe nguy hiểm hơn việc tạm báo thiếu một ô trống.

### Tracking

Dùng:

- HOTA.
- DetA, AssA.
- IDF1.
- ID switches.
- Fragmentation.
- Handoff success rate.
- Global-ID uniqueness violation.
- Thời gian nhận lại ID sau khi xe xuất hiện ở camera mới.

TrackEval là bộ đánh giá chuẩn hỗ trợ HOTA, DetA, AssA, LocA, IDF1, fragmentation và nhiều benchmark tracking. [TrackEval](https://github.com/JonathonLuiten/TrackEval)

### Hiệu năng hệ thống

Phải báo:

- FPS tổng với 1, 2 và 4 camera.
- Độ trễ p50 và p95 từ frame đến JSON/web.
- CPU, RAM, GPU nếu có.
- Kích thước mô hình.
- Thời gian khôi phục khi camera mất kết nối.

Bài ResEViT-Road mạnh chính vì không dừng ở F1: họ còn đưa số tham số, FLOPs, FPS và điện năng trên thiết bị biên.

---

## 9. Tính thực tế: bạn còn thiếu gì?

Muốn nói “triển khai thực tế”, TechGAR cần thêm:

1. **Camera thật và đồng bộ thời gian**

   Dùng điện thoại làm RTSP/virtual webcam được, nhưng cần ghi timestamp lúc nhận frame và có buffer cho camera trễ.

2. **Hệ tọa độ bãi xe chung**

   Mỗi camera phải có homography từ ảnh camera sang mặt phẳng map:

   \[
   (x_{image},y_{image})\rightarrow(X_{parking},Y_{parking})
   \]

   Khi đó khoảng cách giữa cam3 và cam4 được tính theo mét hoặc tọa độ map, không phải pixel crop.

3. **Topology camera**

   Cần khai báo cạnh chuyển hợp lệ:

   ```text
   cam1 ↔ cam2
   cam1 ↔ cam3
   cam2 ↔ cam4
   cam3 ↔ cam4
   ```

   Kèm vùng exit/entry tương ứng, không chỉ biết hai camera “kề nhau”.

4. **Chống lệch camera**

   ROI cố định sẽ sai nếu camera bị rung hoặc bị xoay. Nên có bước kiểm tra camera alignment bằng feature matching/homography và cảnh báo hiệu chỉnh lại ROI.

5. **Giới hạn tuyên bố**

   Phiên bản hiện tại phù hợp với:

   > Bãi xe cố định, camera trên cao, vùng quan sát đã hiệu chỉnh, số lượng xe vừa phải.

   Chưa nên tuyên bố hoạt động tốt với:

   - Camera đường phố tự do.
   - Góc nhìn thấp.
   - Ban đêm cực tối.
   - Mưa lớn.
   - Bãi không có vạch rõ.
   - Camera không đồng bộ và không calibration.

---

## 10. Cách trình bày trước hội đồng

Đừng mở đầu bằng giao diện hoặc video demo. Hãy trình bày theo thứ tự:

1. **Vấn đề thực tế:** vision ROI dao động, báo trống sai và mất xe đứng yên.
2. **Khoảng trống nghiên cứu:** các phương pháp occupancy thường xử lý từng ô hoặc từng frame; tracking thường không tận dụng trạng thái ô để phục hồi ID.
3. **Giả thuyết:** Global ID + stop detection + temporal fusion sẽ giảm false-free và ID switch.
4. **Ba đóng góp chính.**
5. **Công thức và state machine.**
6. **Dataset và cách gán nhãn.**
7. **Baseline và ablation.**
8. **Bảng kết quả có metric.**
9. **Demo realtime điện thoại → thuật toán → JSON → web map.**
10. **Failure cases và giới hạn.**

Một câu giới thiệu tốt:

> TechGAR không chỉ đếm ô trống. Hệ thống duy trì danh tính phương tiện theo thời gian, sử dụng quá trình xe đi vào, dừng và rời ô làm bằng chứng để sửa các lỗi nhận diện trạng thái từ ảnh đơn lẻ.

---

## 11. Việc cần làm theo thứ tự ưu tiên

### Ưu tiên 1 — Dừng chỉnh tham số bằng mắt

Tạo ground truth và script đánh giá tự động trước. Mỗi lần đổi tham số phải sinh ra bảng metric.

### Ưu tiên 2 — Thu video camera thật

Dùng ít nhất hai điện thoại ở góc nhìn khác nhau. Không crop chung một video nếu dùng làm kết quả chính.

### Ưu tiên 3 — Viết evaluator

Evaluator phải tính được:

- Parking F1.
- False-free.
- Transition delay.
- IDF1 hoặc handoff accuracy.
- ID switches.
- FPS và latency.

### Ưu tiên 4 — Chạy ablation

Đây là phần biến code hiện tại thành nghiên cứu khoa học.

### Ưu tiên 5 — Viết failure taxonomy

Phân loại lỗi thành:

- Motion echo tạo hai bbox.
- Xe chạy nhanh.
- Xe bò chậm.
- Xe dừng mất detection.
- Appearance giống nhau.
- Camera overlap.
- Vision bị bóng đổ.
- Xe chiếm hai ô.
- Camera lệch ROI.

Mỗi loại phải có số trường hợp và tỷ lệ sửa thành công.

---

Kết luận chủ quan của tôi: **TechGAR không thua vì thuật toán ít hay sản phẩm đơn giản. Nó thua vì bạn đang đầu tư 80% công sức vào làm hệ thống và sửa case lỗi, nhưng chỉ khoảng 20% vào thiết kế thực nghiệm và chứng minh.** Muốn dự án “xịn” lên, đừng thêm chức năng nữa. Hãy khóa phạm vi, thu dữ liệu thật, tạo ground truth, chạy baseline–ablation và biến từng lời khẳng định thành một con số.
