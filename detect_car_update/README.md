# Vehicle tracking nâng cấp

Hệ thống có hai backend:

- `motion` (mặc định): cho camera bãi xe top-down hiện tại. Detector foreground
  được ghép với temporal-motion gate, Kalman constant-velocity, global
  assignment và HSV appearance Re-ID; nó thay Particle Filter/random-walk cũ.
- `yolo`: YOLO + BoT-SORT + Re-ID. Chỉ dùng sau khi có model đã fine-tune ảnh
  top-down của bãi xe; YOLO COCO mặc định không nhận được xe trong video mẫu.

Tọa độ `x/y` xuất ra JSON là **giữa cạnh đáy bbox**, tức điểm xe tiếp xúc mặt
đường, không phải tâm bbox.

## Cài và chạy

Từ thư mục `detect_car_update`:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe tracker_main.py --video ..\dataset\carPark.mp4 --no-display --verbose
```

Để xem trực tiếp, bỏ `--no-display`. Video mặc định chạy nhanh nhất có thể;
dùng `--realtime` để phát đúng FPS gốc, hoặc ví dụ `--playback-fps 5` để xem
chậm 5 frame/giây:

```powershell
..\.venv\Scripts\python.exe tracker_main.py --video ..\dataset\carPark.mp4 --playback-fps 5
```

Giao diện mặc định chỉ vẽ xe đang có chuyển động thật. Nếu cần chẩn đoán, thêm
`--show-debug-tracks` để hiện cả các track `tentative` và `lost`.

Camera thật (ưu tiên frame mới nhất, không cố xử lý backlog):

```powershell
..\.venv\Scripts\python.exe tracker_main.py --camera 0 --device 0 --verbose
```

Lệnh camera ở trên dùng backend `motion` nên không cần `--device`; có thể bỏ nó.
Khi đã có model fine-tune, dùng YOLO như sau:

```powershell
..\.venv\Scripts\python.exe tracker_main.py --backend yolo --model parking_topdown.pt --camera 0 --device 0
```

`--device 0` dùng CUDA GPU đầu tiên nếu PyTorch/CUDA có sẵn; bỏ tham số này để
Ultralytics tự chọn. Dùng `--device cpu` khi không có GPU.

## File output

`vehicle_positions.json` có ba nhóm rõ ràng:

- `active_vehicles`: detection thật tại frame mới nhất, dùng cho dashboard.
- `lost_vehicles`: tạm mất detection nhưng BoT-SORT vẫn có cơ hội gán lại ID.
- `exited_vehicles`: hết `lost-track-ttl` frame không được nhìn thấy.

Không hiển thị xe `lost` là xe đang hiện hữu để tránh đưa tọa độ dự đoán cũ vào
bài toán ô đỗ.

## Hiệu chuẩn toạ độ mặt bằng (khuyên dùng)

Nếu cần toạ độ theo sơ đồ bãi/met thay vì pixel, đo tối thiểu 4 điểm tương ứng
giữa ảnh và mặt bằng rồi tạo `homography.json`:

```json
{"homography": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]}
```

Sau đó thêm `--homography homography.json`. `position.ground_x/ground_y` sẽ có
trong JSON. Đơn vị đúng bằng đơn vị bạn dùng khi tạo 4 điểm đích (mét hoặc pixel
trên sơ đồ).

## Chỉnh khi cần

- Backend `motion`: tăng `--lost-track-ttl` khi xe bị che; giảm
  `--motion-min-area` (ví dụ `450`) nếu mất xe ở xa; chỉ tăng
  `--motion-max-distance` khi xe chạy nhanh. Nếu nhiễu ánh sáng vẫn sinh xe
  giả, tăng `--motion-min-ratio` từ `0.08` lên `0.12`.
- Backend `yolo`: tăng `track_buffer` trong `botsort_parking_reid.yaml` và
  `--lost-track-ttl` khi che khuất. Điều chỉnh `--conf` và `--imgsz` sau khi
  kiểm tra model trên chính video bãi xe.
