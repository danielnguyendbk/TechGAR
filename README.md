# 🚗 Hệ Thống Nhận Diện Bãi Đỗ Xe (TechGAR) - Hybrid Detection Pipeline

Dự án này là hệ thống phân tích và phát hiện trạng thái bãi đỗ xe (Trống / Có xe) tự động bằng cách sử dụng phương pháp **Hybrid (kết hợp OpenCV và mạng nơ-ron CNN)**. Hệ thống tối ưu hóa tốc độ xử lý bằng cách dùng các bộ lọc ảnh truyền thống (OpenCV) xử lý nhanh để đánh giá ban đầu, và chỉ dùng mô hình học sâu (CNN) cho những khu vực phức tạp/nghi ngờ, giúp cân bằng hoàn hảo giữa độ chính xác và hiệu suất khung hình trên giây (FPS).

---

## 📁 Cấu trúc Pipeline

Dự án được chia thành các bước rõ ràng từ xử lý dữ liệu thô, chuẩn bị dữ liệu huấn luyện (Crop Images), đánh giá Baseline, đến đường ống dự đoán kết hợp (Hybrid Pipeline).

### 1. Phân tích và Tiền Xử lý Dữ liệu (Dataset Parsers)
Các script dùng để quét và chuẩn hóa các dataset phổ biến về định dạng chung `.csv`:
*   `xulypk1.py`: Parse dataset **PKLot** định dạng **COCO JSON** (đọc `image_id`, `category_id`, thông tin bounding box và thống kê phân bổ rỗng/đầy).
*   `xulypk2.py`: Parse dataset **PKLot** định dạng **XML** đặc thù (trích xuất thông tin vẽ contour từ thẻ `<space>` thay vì VOC `<object>`).
*   `xulycnrpark.py`: Parse dataset ảnh đã cắt sẵn **CNRPark-Patches** (lấy danh sách tập huấn luyện từ các thư mục phân cấp `busy` / `free`).

### 2. Gộp Dữ Liệu (Merger)
*   `merge.py`: Gom file CSV xuất ra từ 3 script trên lại thành một metadata duy nhất là `master_parking_dataset.csv`. File này đóng vai trò mỏ neo lưu trữ toàn bộ thông tin thống nhất (`dataset`, `status`, `status_num` cho machine learning, `bbox`).

### 3. Tạo Dữ Liệu Huấn Luyện CNN (Data Generators)
*   `create_crops.py`: Đọc master CSV, tiến hành cắt (crop) ảnh từng ô đỗ và resize chuẩn hóa về `128x128`. Sau đó tự động chia thành tập huấn luyện train (80%) và tập đánh giá val (20%). Dữ liệu được lưu trữ phân cấp vào `crops/train/0`, `crops/train/1`, v.v... phù hợp cho Pytorch/TensorFlow ImageGenerator.

### 4. Xây dựng Baseline với OpenCV
*   `baselime_opencv.py`: Bước chạy thử nghiệm thuật toán phát hiện cổ điển bằng OpenCV (Giai đoạn 1 của TechGAR: `GaussianBlur` -> `Adaptive Threshold` -> `FindContours`) trên ảnh grayscale cực kỳ nhẹ để đo lường giới hạn của F1-Score và tốc độ.
*   `baseline_tuned.py`: Phiên bản tinh chỉnh các tham số độ nhạy (threshold, kernel size) của OpenCV để tăng độ chính xác lên tối đa có thể với phương pháp thị giác máy tính truyền thống.

### 5. Phương pháp Hybrid (Hệ thống chính)
*   `test_hybrid.py`: Hybrid Pipeline cuối cùng. 
    *   **Phase 1**: Dùng **OpenCV** để lọc siêu nhanh các góc/quang cảnh/ô đỗ chắc chắn trống (chiếm xấp xỉ tỉ lệ cao thời gian thực).
    *   **Phase 2**: Dùng mạng **CNN** (`cnn_parking.h5`) trên ảnh màu RGB cắt `128x128` để nhận diện những trường hợp khó (rìa xe, đổ bóng cây, thời tiết) lọt qua Phase 1.
    *   Đo lường thời gian trễ giữa 2 thao tác và tính điểm số F1-Score, sau đó lưu kết quả vào `hybrid_results.csv`.

---

## 🚀 Hướng Dẫn Chạy & Cài Đặt

### Cài đặt môi trường
Phiên bản Python khuyến nghị: `3.9+`. Mở Terminal, trỏ dòng lệnh vào thư mục dự án và chạy:
```bash
pip install pandas numpy opencv-python matplotlib scikit-learn tensorflow tqdm
```

### Quy trình khởi chạy:
1.  **Extract Data**: Chạy lần lượt `python xulypk1.py`, `python xulypk2.py`, `python xulycnrpark.py` để parse dữ liệu và in biểu đồ (tuỳ chọn).
2.  **Merge Data**: Chạy `python merge.py` để biên dịch danh sách Master CSV.
3.  **Create CNN Crops**: Chạy `python create_crops.py` để chuẩn bị data training model.
4.  *(Tuỳ chọn) Train Model*: Các bạn có thể viết script xây dựng model cho Deep Learning từ thư mục `crops/train` để tạo ra `cnn_parking.h5`.
5.  **Test Pipeline**: Kiểm tra bằng `python baseline_tuned.py` trước, rồi chạy `python test_hybrid.py` để tận mắt thấy FPS tăng vượt bậc so với việc chỉ xài CNN cho toàn bộ quy trình.

---

## 📊 Bộ Dữ Liệu Tương Thích
Hệ thống viết sẵn parser, hỗ trợ nhận diện tự động cấu trúc cho các bộ dữ liệu parking space lớn nhất thế giới:
*   [PKLot (PUCPR/UFPR)](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
*   [CNRPark + EXT](http://cnrpark.it/)
