# Kịch bản dựng bản đồ chung từ hai DroidCam

Hai camera chỉ quan sát hai phần của bãi. Không cần camera thứ ba quay toàn bộ
mô hình. Mỗi camera được hiệu chỉnh độc lập vào cùng một hệ tọa độ centimet:

```text
pixel cam1 --H1--> (X, Y) cm trên bãi
pixel cam2 --H2--> (X, Y) cm trên bãi
```

## 1. Chuẩn bị và đo ngoài thực tế

1. Cố định hai điện thoại đúng vị trí sẽ dùng khi chạy chương trình.
2. Chọn một góc cố định của mô hình làm gốc `O=(0,0)`.
3. Chọn trục X dọc theo cạnh ngang của bãi và trục Y dọc theo cạnh vuông góc.
4. Đơn vị của toàn bộ phép đo là centimet.
5. Đặt tối thiểu bốn điểm không thẳng hàng trong vùng cam1: `A,B,C,D`.
6. Đặt tối thiểu bốn điểm không thẳng hàng trong vùng cam2: `E,F,G,H`.
7. Nên dùng 6-10 điểm mỗi camera, trải rộng gần các góc vùng quan sát.
8. Nếu có thể, đặt thêm `O1,O2,O3,O4` trong overlap để cả hai camera cùng thấy.

Với mỗi điểm, đo trực tiếp tọa độ X và Y so với hai trục. Không chỉ cộng dồn
khoảng cách A-B, B-C vì sai số sẽ tích lũy.

Ví dụ minh họa, không dùng các số này cho mô hình thật:

| Điểm | X (cm) | Y (cm) |
|---|---:|---:|
| A | 10.0 | 10.0 |
| B | 70.0 | 10.0 |
| C | 70.0 | 50.0 |
| D | 10.0 | 50.0 |
| E | 60.0 | 10.0 |
| F | 130.0 | 10.0 |
| G | 130.0 | 50.0 |
| H | 60.0 | 50.0 |

Bốn điểm của một camera phải tạo thành một tứ giác có diện tích rõ ràng. Không
đặt cả bốn điểm dọc theo một vạch hoặc một cạnh ô đỗ.

## 2. Dừng runner cũ

Nhấn `Q` hoặc `Esc` trong `two_camera.py`. Không capture calibration trong khi
runner cũ vẫn giữ hai stream DroidCam.

Từ PowerShell:

```powershell
cd D:\NCKH\TechGAR\main_detect
```

## 3. Cách chạy đơn giản (khuyên dùng)

Toàn bộ việc chụp ảnh, chọn điểm, nhập số đo và dựng map đã được gom vào
`calibrate_map.py`. Với đúng hai địa chỉ DroidCam hiện tại, chỉ chạy một lệnh:

```powershell
cd D:\NCKH\TechGAR\main_detect
& ..\.venv\Scripts\python.exe .\calibrate_map.py
```

Trước khi chạy, đặt một hình chữ nhật phẳng trong đoạn giao nhau để cả hai cam
cùng nhìn thấy và đo hai cạnh `AB`, `AD` bằng centimet. Sau đó:

1. Trên ảnh cam1, click bốn góc của hình theo đúng vòng `A → B → C → D`.
2. Trên ảnh cam2, click lại đúng bốn góc vật lý đó theo cùng thứ tự.
3. Terminal chỉ hỏi `AB_cm` và `AD_cm`. Nhập hai chiều dài rồi script tự đặt
   `A=(0,0)`, tính H1/H2 và dựng bản đồ.

Click phải xóa điểm gần nhất, `R` xóa toàn bộ điểm camera hiện tại, `Q` hoặc
`Esc` hủy. Khi hoàn thành cam 2, script tự tạo:

- `config\two_camera.shared_cm.json`: calibration để chạy `two_camera.py`.
- `config\shared_map_01\shared_map_full_view.png`: ghép 50/50 toàn bộ hai ảnh,
  chưa áp ROI; dùng kiểm tra các vạch sơn có chồng khít hay không.
- `config\shared_map_01\shared_map_active_roi.png`: sau khi áp
  `roi_mask_cam1/2` và vẽ `parking_slots_cam1/2`; đây là vùng tracking sử dụng.
- `config\shared_map_01\shared_map_preview.png`: bản đồ ROI dạng polygon cũ,
  được giữ để tương thích.
- Ảnh gốc, ảnh đã đánh dấu và CSV được giữ lại để đối chiếu.

Script mở `shared_map_full_view.png` trước. Bấm phím bất kỳ để chuyển sang
`shared_map_active_roi.png`, rồi bấm phím bất kỳ lần nữa để đóng. Parking slots
chỉ là lớp vẽ kiểm tra; overlap runtime vẫn được tính từ `roi_mask_cam1/2`.

Nếu IP DroidCam thay đổi, vẫn chỉ chạy một lệnh nhưng truyền URL mới:

```powershell
& ..\.venv\Scripts\python.exe .\calibrate_map.py --cam1-url "http://IP_CAM1:4747/video/force/1280x720" --cam2-url "http://IP_CAM2:4747/video/force/1280x720"
```

Các mục từ đây trở xuống là quy trình nâng cao/thủ công để sửa lại dữ liệu đã
chụp mà không cần chụp lại.

## 4. Chụp hai ảnh calibration (nâng cao)

```powershell
& ..\.venv\Scripts\python.exe tools\calibrate_shared_map.py capture --cam1-url "http://192.168.100.53:4747/video/force/1280x720" --cam2-url "http://192.168.100.198:4747/video/force/1280x720" --workspace "config\shared_map_01"
```

Kết quả:

```text
config/shared_map_01/capture_cam1.png
config/shared_map_01/capture_cam2.png
config/shared_map_01/capture_manifest.json
```

Không di chuyển camera hoặc đổi độ phân giải sau bước này.

## 5. Click các điểm trên ảnh

Nếu chỉ có A-D và E-H:

```powershell
& ..\.venv\Scripts\python.exe tools\calibrate_shared_map.py mark --workspace "config\shared_map_01" --cam1-labels "A,B,C,D" --cam2-labels "E,F,G,H"
```

Khuyến nghị thêm các điểm overlap chung:

```powershell
& ..\.venv\Scripts\python.exe tools\calibrate_shared_map.py mark --workspace "config\shared_map_01" --cam1-labels "A,B,C,D,O1,O2,O3,O4" --cam2-labels "E,F,G,H,O1,O2,O3,O4"
```

Điều khiển cửa sổ:

- Chuột trái: thêm điểm đang hiển thị ở góc trên.
- Chuột phải: bỏ điểm vừa chọn.
- `R`: chọn lại toàn bộ camera hiện tại.
- `Enter`: xác nhận sau khi chọn đủ.
- `Q` hoặc `Esc`: hủy.

Công cụ tạo:

```text
marked_cam1.png
marked_cam2.png
calibration_points.csv
```

## 6. Điền số đo centimet

Mở `config/shared_map_01/calibration_points.csv`. Không sửa bốn cột đầu. Chỉ
điền `world_x_cm` và `world_y_cm`:

```csv
camera,label,pixel_x,pixel_y,world_x_cm,world_y_cm
cam1,A,243,518,10.0,10.0
cam1,B,911,506,70.0,10.0
cam1,C,875,241,70.0,50.0
cam1,D,276,252,10.0,50.0
cam2,E,188,560,60.0,10.0
cam2,F,1033,548,130.0,10.0
cam2,G,982,220,130.0,50.0
cam2,H,221,231,60.0,50.0
```

Nếu `O1` xuất hiện ở cả hai camera thì hai dòng `O1` phải có cùng tọa độ cm.

## 7. Tính H1/H2 và dựng map

Hai file `roi_mask_cam1.json` và `roi_mask_cam2.json` phải bao đúng phần mặt đất
hợp lệ của từng camera. Công cụ biến đổi hai polygon này sang centimet rồi tự
tính phần giao.

```powershell
& ..\.venv\Scripts\python.exe tools\calibrate_shared_map.py build --workspace "config\shared_map_01" --coverage-cam1 "config\roi_mask_cam1.json" --coverage-cam2 "config\roi_mask_cam2.json" --output "config\two_camera.shared_cm.json"
```

Kết quả:

```text
config/two_camera.shared_cm.json
config/shared_map_01/shared_map_preview.png
```

File preview hiển thị:

- Viền đen: vùng cam1 trên bản đồ chung.
- Viền đỏ: vùng cam2.
- Màu tím: overlap.
- Lưới: mỗi 10 cm.
- Các điểm calibration và nhãn.

Công cụ từ chối kết quả nếu RMS lớn hơn 3 cm hoặc hai coverage không giao nhau.
Với đúng bốn điểm, homography luôn đi qua bốn điểm nên chưa đủ dữ liệu đánh giá;
do đó nên dùng nhiều hơn bốn điểm và có điểm chung trong overlap.

## 8. Tận dụng ROI ô đỗ

ROI pixel hiện tại vẫn dùng để nhận diện chỗ đỗ. Sau khi có H1/H2, từng đỉnh ROI
có thể được đổi sang `(X,Y)` cm để hiển thị trên map chung. Tên A01, B01... chỉ
cho biết topology; muốn có bản đồ đúng tỷ lệ vẫn cần chiều rộng ô, chiều dài ô,
khoảng cách giữa dãy và vị trí của ít nhất một góc chuẩn.

Các marker đo thực tế là nguồn calibration chính. Kích thước ô đỗ và ROI dùng
để dựng/kiểm tra bản đồ, không thay thế hoàn toàn marker.

## 9. Chạy thử với calibration cm

Đổi tên session mỗi lần chạy:

```powershell
& ..\.venv\Scripts\python.exe .\two_camera.py --cam1-url "http://192.168.100.53:4747/video/force/1280x720" --cam2-url "http://192.168.100.198:4747/video/force/1280x720" --slots-cam1 "config\parking_slots_cam1.json" --slots-cam2 "config\parking_slots_cam2.json" --calibration "config\two_camera.shared_cm.json" --mask-cam1 "config\roi_mask_cam1.json" --mask-cam2 "config\roi_mask_cam2.json" --output-dir "experiment_test\output\runtime_shared_cm_01" --session-dir "experiment_test\output\droidcam_shared_cm_01" --identity-retention-seconds 30 --tracklet-max-samples 12 --tracklet-sample-interval 3 --global-gallery-max-samples 24
```

`two_camera.py` tự đọc các ngưỡng cm trong `matching_defaults`. Không truyền lại
ngưỡng pixel cũ như 100/160 khi dùng calibration cm.

## 10. Kịch bản kiểm tra

1. Đặt một marker hoặc xe trong overlap và di chuyển rất chậm.
2. Hai quan sát phải có vị trí world gần nhau và dùng cùng Global ID.
3. Chạy cam1-only -> overlap -> cam2-only.
4. Chạy chiều ngược lại.
5. Đỗ xe trong cam1, chờ local motion mất rồi chạy lại: ID phải giữ nguyên.
6. Lặp lại trong cam2.
7. Sau khi một xe ổn định mới thử hai xe đồng thời.

Không đặt `exit_zones` tại biên cam1/cam2. Exit zone chỉ nằm ở cổng ra khỏi hợp
của cả hai vùng quan sát.
