# 🅿️ TechGAR - Smart Parking & Navigation System

Hệ thống Quản lý & Dẫn đường Bãi đỗ xe Thông minh kết hợp Camera AI Tracking thời gian thực (YOLOv8 + CNN), thuật toán tìm đường Dijkstra, Giao diện Cá nhân hóa QR Code, Hướng dẫn bằng Giọng nói Tiếng Việt và Cảnh báo Đi sai đường.

---

## 🏗 1. Kiến trúc Hệ thống & 2 Nguồn Dữ liệu

Hệ thống hỗ trợ **2 chế độ vận hành**:
1. **Chế độ Dữ liệu Mẫu (Sample Mode)**: Dùng `sample_tracking_simulator.py` để giả lập di chuyển xe mượt mà mà không cần GPU/Camera.
2. **Chế độ Camera OpenCV Thực tế (Real OpenCV Mode)**: Dùng `opencv_test_js_2.py` chạy mô hình **YOLOv8** (`yolov8n.pt`) kết hợp **CNN** (`cnn_parking.h5`) xử lý video/camera thực tế (`output3_video.mp4`), ghi tọa độ xe thật vào `vehicle_positions.json`.

---

## 🛠 2. Yêu cầu Môi trường & Thư viện

### Frontend:
* **Node.js**: v18.0 trở lên

### Backend:
* **Python**: v3.9 trở lên
* **Cài đặt thư viện Python**:
```bash
pip install opencv-python ultralytics tensorflow numpy requests pillow
```

---

## 🚀 3. Hướng dẫn Khởi chạy Hệ thống

---

### 🟢 CHẾ ĐỘ 1: Dữ liệu Mẫu (Sample Data Mode - Khuyên dùng khi Demo nhanh)

Mở 3 Terminal độc lập:

* **Terminal 1 (Frontend Web App)**:
  ```bash
  cd frontend
  npm run dev
  ```
  *(Truy cập `http://localhost:4173/`)*

* **Terminal 2 (Simulator di chuyển xe)**:
  ```bash
  python backend/sample_tracking_simulator.py
  ```

* **Terminal 3 (Gate Session Controller & API)**:
  ```bash
  python backend/gate_session_controller.py --source vehicle_positions_sample.json
  ```

---

### 🔵 CHẾ ĐỘ 2: Camera AI Tracking Thực tế (Real OpenCV Mode - Nhận diện qua Video/Camera)

Mở 3 Terminal độc lập:

* **Terminal 1 (Frontend Web App)**:
  ```bash
  cd frontend
  npm run dev
  ```

* **Terminal 2 (Chạy OpenCV YOLO + CNN Tracking từ Video)**:
  ```bash
  python backend/opencv_test_js_2.py
  ```
  * Chương trình sẽ mở cửa sổ OpenCV xử lý video `output3_video.mp4`, nhận diện xe và ghi vị trí thời gian thực vào `public/vehicle_positions.json`.

* **Terminal 3 (Gate Session Controller cho nguồn OpenCV)**:
  ```bash
  python backend/gate_session_controller.py --source vehicle_positions.json
  ```

* **Thao tác trên Web**: Trên thanh Header Web, đổi công tắc nguồn từ **"Dữ liệu mẫu"** $\rightarrow$ **"Camera OpenCV"**.

---

## 🧪 4. Kịch bản Kiểm thử Chi tiết Toàn bộ Hệ thống (Full Test Suite)

---

### 📌 PHẦN A: Test Các Chức Năng Dẫn Đường & Giao Diện (Sample & Real Mode)

#### 🔹 Test Case 1: Giám sát Bãi xe Chung & QR Kiosk Cổng vào
1. Truy cập `http://localhost:4173/` (Trang chung).
2. **Kỳ vọng**:
   * Bản đồ hiển thị tất cả các xe đang di chuyển trong bãi.
   * Khi xe tiến vào cổng, **QR Kiosk** xuất hiện ở góc dưới bên phải hiển thị mã QR `/?session=ID`.
   * QR Kiosk tự đổi xe hoặc tự ẩn khi hết xe ở cổng.

#### 🔹 Test Case 2: Giao diện Cá nhân & Giọng nói Tiếng Việt (Web Speech API)
1. Mở trang cá nhân của xe: `http://localhost:4173/?session=3`.
2. **Kỳ vọng**:
   * Bản đồ **chỉ hiển thị duy nhất Xe #3**, ẩn các xe khác.
   * Không xuất hiện QR Kiosk.
   * Bảng *"Bạn muốn tìm chỗ đỗ theo cách nào?"* tự động mở.
3. Chọn ô đỗ (ví dụ ô **D08**):
   * Đường mũi tên chỉ dẫn màu xanh xuất hiện từ vị trí xe tới ô **D08**.
   * Trình duyệt phát **giọng nói Tiếng Việt**: *"Phía trước đi thẳng"*, *"Phía trước rẽ phải vào làn đỗ"*,...

#### 🔹 Test Case 3: Nút Thoát / Mở lại Chỉ dẫn (`❌ Thoát chỉ dẫn` / `🧭 Mở lại chỉ dẫn`)
1. Đang có đường mũi tên xanh chỉ dẫn trên trang cá nhân.
2. Bấm nút **`❌ Thoát chỉ dẫn`**: Đường xanh chỉ hướng trên bản đồ biến mất ngay lập tức.
3. Bấm nút **`🧭 Mở lại chỉ dẫn`**: Đường xanh xuất hiện trở lại.

#### 🔹 Test Case 4: Cảnh báo Đi sai Tuyến đường (Off-Route Warning)
1. Khi xe di chuyển chệch khỏi tuyến đường chỉ dẫn $>75px$:
2. **Kỳ vọng**:
   * Phát âm thanh khẩn cấp: *"Cảnh báo: Bạn đang đi sai tuyến đường chỉ dẫn!"*.
   * Xuất hiện Bảng thông báo đỏ nhấp nháy: `⚠️ CẢNH BÁO: BẠN ĐANG ĐI SAI TUYẾN ĐƯỜNG CHỈ DẪN!`.

#### 🔹 Test Case 5: Cập nhật Trạng thái Ô đỗ Thực tế
1. Khi xe đỗ hoàn tất vào ô đỗ thực tế (ô **D08**):
2. **Kỳ vọng**:
   * Hệ thống đổi trạng thái ô **D08** sang màu **Đỏ (Occupied)**.
   * Ô nhấp chọn ban đầu (ví dụ A01) giữ nguyên màu **Xanh (Trống)** nếu xe không đỗ vào đó.
   * Trạng thái phiên chuyển sang **`PARKED`**.

#### 🔹 Test Case 6: Dẫn đường Lấy xe ra Cổng (`🚗 Lấy xe ra`)
1. Khi phiên ở trạng thái `PARKED`, nhấp nút **`🚗 Lấy xe ra`**.
2. **Kỳ vọng**:
   * Thuật toán tự động vẽ tuyến đường từ **ô đỗ thực tế (D08)** ra **CỔNG EXIT**.
   * Giọng nói phát chỉ dẫn: *"Tiếp tục đi theo đường dẫn ra cổng xuất bãi"*.
   * Khi xe ra khỏi bãi, ô **D08** chuyển lại màu **Xanh (Trống)**.

---

### 📌 PHẦN B: Test Nhận Diện Xe & AI Tracking Thực Tế (`opencv_test_js_2.py`)

#### 🔹 Test Case 7: Kiểm thử Module OpenCV + YOLOv8 + CNN Detection
1. Chạy lệnh:
   ```bash
   python backend/opencv_test_js_2.py
   ```
2. **Kỳ vọng**:
   * Màn hình OpenCV hiển thị frame video từ `carPark.mp4` / `output3_video.mp4`.
   * Các ô đỗ xe được vẽ khung Polygon màu **Xanh (Trống)** hoặc **Đỏ (Đã đỗ)** dựa trên mô hình CNN `cnn_parking.h5`.
   * Các xe đang di chuyển được YOLOv8 đóng khung Bounding Box và gán `track_id`.
   * File `public/vehicle_positions.json` được cập nhật tọa độ liên tục.

#### 🔹 Test Case 8: Chuyển đổi Nguồn Dữ liệu Real-time trên Web
1. Mở trang Web `http://localhost:4173/`.
2. Nhấp nút chuyển nguồn ở Header từ **"Dữ liệu mẫu"** $\rightarrow$ **"Camera OpenCV"**.
3. **Kỳ vọng**:
   * Bản đồ Web hiển thị chính xác tọa độ các xe đang chạy được trích xuất trực tiếp từ video qua script `opencv_test_js_2.py`.

---

### 📌 PHẦN C: Công cụ Định vị & Căn chỉnh Ô đỗ (Parking Slot Picker)

#### 🔹 Test Case 9: Chạy Công cụ Vẽ và Điều chỉnh Ô đỗ (ROI Calibration)
1. Chạy lệnh:
   ```bash
   python backend/ParkingSpacePicker_ve_js.py
   ```
2. **Sử dụng**:
   * Click chuột trái vào hình ảnh bãi xe để thêm ô đỗ mới.
   * Click chuột phải để xóa ô đỗ.
   * Tọa độ các ô đỗ sẽ tự động lưu vào `CarParkPos` / `parking_slots.json` để phục vụ cho OpenCV và Bản đồ Web.

---

## 📁 5. Tổng hợp Cấu trúc File Dự án

```text
TechGAR/
├── backend/
│   ├── opencv_test_js_2.py             # Script AI Tracking chính (OpenCV + YOLOv8 + CNN)
│   ├── sample_tracking_simulator.py    # Simulator dữ liệu giả lập di chuyển xe
│   ├── gate_session_controller.py      # HTTP API Server (port 8000) & Gate Controller
│   ├── ParkingSpacePicker_ve_js.py     # Công cụ UI vẽ & căn chỉnh tọa độ ô đỗ (ROI)
│   ├── yolov8n.pt / cnn_parking.h5     # Các mô hình AI nhận diện xe & đỗ xe
│   └── output3_video.mp4 / carPark.mp4 # Video đầu vào giả lập camera
└── frontend/
    ├── public/
    │   ├── vehicle_positions.json      # Tọa độ xe realtime từ OpenCV
    │   ├── vehicle_positions_sample.json # Tọa độ xe realtime từ Simulator
    │   └── parking_status_sample.json  # Trạng thái ô đỗ bãi xe
    └── src/
        ├── app/App.tsx                 # Web Controller chính
        ├── components/EntryQRKiosk.tsx # Widget QR Kiosk tại cổng vào
        └── routing/
            ├── laneGraph.ts            # Đồ thị làn đường giao thông bãi đỗ
            ├── routeEngine.ts          # Thuật toán tìm đường Dijkstra (Inbound/Exit)
            └── voiceGuidance.ts        # Web Speech API Giọng nói Tiếng Việt & Off-route Warning
```