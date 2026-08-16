# Chay hai camera DroidCam

`main.py` van la demo cat mot video thanh bon camera. Dung `two_camera.py` khi
can hai dien thoai that. Tracker, parking detector va HSV ReID dung nguyen ban.

## Chuan bi

1. Co dinh hai dien thoai, cung do phan giai DroidCam va cung mang Wi-Fi.
2. Tao `config/parking_slots_cam1.json` va `config/parking_slots_cam2.json`
   bang cong cu ROI. Moi ID slot phai duy nhat, vi du `C1_P001` va `C2_P001`.
3. Chon bon goc cua overlap, nhin thay o ca hai camera. Khong doi vi tri, zoom
   hay do phan giai sau khi hieu chinh. Tao JSON bang lenh:

```powershell
.\.venv\Scripts\python.exe tools\calibrate_two_cameras.py `
  --cam1-url "http://<IP_CAM1>:4747/video" `
  --cam2-url "http://<IP_CAM2>:4747/video" `
  --output config\two_camera.local.json
```

Click bon goc theo chieu kim dong ho tren cam1, sau do click dung bon diem do
tren cam2 theo cung thu tu. `cam1` la he toa do world tham chieu.

## Chay

```powershell
.\.venv\Scripts\python.exe two_camera.py `
  --cam1-url "http://<IP_CAM1>:4747/video/force/1280x720" `
  --cam2-url "http://<IP_CAM2>:4747/video/force/1280x720" `
  --slots-cam1 config\parking_slots_cam1.json `
  --slots-cam2 config\parking_slots_cam2.json `
  --calibration C:\duong-dan\two_camera_calibration.json `
  --session-dir experiment_test\output\two_camera_01
```

Nhan `Q`, `Esc`, hoac `Ctrl+C` de dung. Sau do kiem tra:

```powershell
.\.venv\Scripts\python.exe experiment_test\validate_session.py `
  --session experiment_test\output\two_camera_01
```

Session hai camera gom `raw_cam1.mp4`, `raw_cam2.mp4`, debug video, JSONL
Global ID, timestamp/performance va ground-truth templates.

## Chinh threshold truc tiep

Khi chay co giao dien, `two_camera.py` chi dung threshold pixel trang/den va mo
them hai cua so:

- `Parking detector settings`: Gamma, CLAHE, Grid va Ratio rieng cho cam1/cam2.
- `B/W threshold pixels`: pixel trang la foreground ma threshold dang dem.

Mau do tren debug threshold nghia la ty le cua ROI da vuot nguong hien tai.
Pass Canny/Edge bi tat trong `two_camera.py`; gia tri `edge_thr` cu trong profile
khong tham gia ket qua nhan dien.
Nhan `S` de luu cac thanh vao `config/two_camera_detector.local.json`; thoat
bang `Q`/`Esc` cung tu dong luu. Lan chay sau profile nay duoc nap lai. Them
`--no-parking-debug` neu chi can quay va muon giam tai hien thi.
