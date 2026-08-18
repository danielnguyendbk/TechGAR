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

Hai MJPEG stream duoc doc boi hai background thread. Moi camera chi giu frame
moi nhat; neu detector dang ban thi frame cu bi bo thay vi xep hang. Cach nay
giu do tre live thap, du FPS hien thi co the giam khi CPU dang xu ly parking.

Trong cua so `2 Cameras - Tracking + Parking`, xe dang duoc nhin thay va chua
duoc xac nhan do se co bbox, diem va nhan `G#... moving` mau xanh duong. Khi
`SlotVehicleBinder` da gan Global ID vao mot ROI do, lop moving mau xanh bien
mat va ROI tiep tuc hien thi trang thai do. Xe roi o se duoc theo doi lai, uu
tien khoi phuc Global ID cu.

## Chinh threshold truc tiep

Khi chay co giao dien, `two_camera.py` chi dung threshold pixel trang/den va mo
them hai cua so:

- `Parking detector settings`: Gamma, CLAHE, Grid va Ratio rieng cho cam1/cam2.
- `B/W pixels - raw | filtered`: moi hang la mot camera; anh trai la threshold
  goc, anh phai la pixel con lai sau khi bo vien.

Trong anh filtered, duong cyan la `analysis_mask`, duong magenta la `core_mask`.
Nhan hien thi `R` (raw), `F` (filtered) va `C` (core). Mau do nghia la bang
chung filtered/core da vuot nguong hien tai.
Pass Canny/Edge bi tat trong `two_camera.py`; gia tri `edge_thr` cu trong profile
khong tham gia ket qua nhan dien.
Nhan `S` de luu cac thanh vao `config/two_camera_detector.local.json`; thoat
bang `Q`/`Esc` cung tu dong luu. Lan chay sau profile nay duoc nap lai. Them
`--no-parking-debug` neu chi can quay va muon giam tai hien thi.

Moi camera co sau tham so loc vien rieng trong detector profile:

- `border_ignore_ratio`: do sau co ROI de tao analysis mask.
- `line_min_span_ratio`: chieu dai toi thieu de mot component duoc xem la vach.
- `line_max_thickness_ratio`: do day toi da cua vach.
- `core_scale`: kich thuoc vung trung tam so voi ROI.
- `core_ratio_threshold`: mat do pixel trang de core rescue.
- `core_component_threshold`: kich thuoc component trung tam toi thieu.

Voting van dung du 25 bien the gamma/CLAHE. Component vach duoc xac dinh tu
threshold base, sau do spatial ignore-mask duoc ap vao tat ca 25 phieu.

## Demo mot camera DroidCam

Dung `single_camera.py` de kiem tra rieng detector, tracking va ROI cua mot dien
thoai. Vi du cam1:

```powershell
.\.venv\Scripts\python.exe single_camera.py `
  --stream-url "http://192.168.100.53:4747/video/force/1280x720" `
  --slots-file config\parking_slots_cam1.json `
  --detector-profile config\two_camera_detector.local.json `
  --profile-camera cam1
```

Cua so `Parking B/W - raw | filtered` hien raw ben trai va filtered ben phai.
Nhan `Q` hoac `Esc` de dung. Them `--no-parking-debug` neu chi can xem overlay
ROI. Demo mot camera chi co local track ID; Global ID chuyen giao cam1 -> cam2
chi co trong `two_camera.py`.

## Chinh ROI truc tiep tren hai camera

Cua so chinh co mot panel huong dan rieng ben phai hai camera. Panel khong che
video va khong thay doi cua so `B/W threshold pixels`. Tracking xe, ReID va
Global ID van chay trong luc dang chinh ROI.

- `E`: bat/tat che do chinh ROI.
- `1`/`2`: chon cam1 hoac cam2. Click truc tiep vao mot camera cung se chon cam do.
- Chuot trai gan mot dinh mau vang: keo dinh ROI.
- `N`: them ROI; click bon diem roi nhan `A`. `D` huy bon diem dang ve.
- `X`: xoa ROI dang chon. `Z`: hoan tac thao tac gan nhat.
- `O`/`P`: chuyen den day ke tiep/tru, vi du A -> B -> C.
- `Space`: dong bang hinh cua camera dang chon de dat diem. Xu ly tracking nen
  van tiep tuc chay.
- `S`: luu hai file ROI co thay doi, nap lai detector va luu ca detector profile.
- `Q`: luu ROI roi thoat. `Esc`: thoat ma khong luu thay doi ROI chua bam `S`.

Detector chi dung ROI cu trong luc keo. ROI moi chi tham gia tinh trang thai o
dau sau khi bam `S`, do do polygon dang ve do khong lam ket qua nhay sai.
