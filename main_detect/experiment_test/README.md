# Thực nghiệm TechGAR schema v3

Schema v3 đánh giá đúng vòng đời xe và Global ID, không chỉ đếm trạng thái
occupied/free theo từng frame. Thuật toán runtime và giao diện OpenCV không bị
thay đổi bởi bộ đánh giá này.

## File của một session hai camera

- `predictions.jsonl`: prediction schema v3, một record cho mỗi cặp frame.
- `ground_truth_slots.csv`: trạng thái thật và chủ xe thật của ô.
- `ground_truth_events.csv`: cửa sổ sự kiện trong vòng đời xe.
- `ground_truth_identity.csv`: checkpoint danh tính do người xem raw video đặt.
- `frame_timestamps.csv`, `performance.csv`, `session_info.json`.
- `raw_cam1.mp4`, `raw_cam2.mp4`, `debug_cam1.mp4`, `debug_cam2.mp4` nếu session
  không dùng `--no-session-video`.

`two_camera.py` tự tạo đủ ba CSV ground truth với đúng header schema v3. Nhãn
`physical_vehicle_id` như `M02_V1` thuộc namespace của người gán nhãn; nó không
cần bằng GID số do hệ thống sinh.

Xe được thao tác trong kịch bản đặt `identity_required=true`. Xe nền đứng yên
chỉ dùng để chấm occupied/free, đặt `identity_required=false` và để trống
`physical_vehicle_id`.

Checkpoint xe đang đỗ dùng `slot_id` và để trống `anchor_x,anchor_y`. Checkpoint
xe đang chạy để trống `slot_id` và nhập tâm xe theo pixel của đúng raw frame.
Không đọc prediction hoặc màu đỏ/xanh trên debug video để tạo ground truth.

## Chuyển session schema 2 sang schema 3

Luôn chạy dry-run trước:

```powershell
cd D:\NCKH\TechGAR\main_detect
..\.venv\Scripts\python.exe experiment_test\migrate_schema_v3.py --dry-run `
  experiment_test\output\droidcam_shared_m_01 `
  experiment_test\output\droidcam_shared_m_02 `
  experiment_test\output\droidcam_shared_m_03 `
  experiment_test\output\droidcam_shared_m_04
```

Nếu dry-run thành công, bỏ `--dry-run` để ghi. Mỗi session được staging và có
rollback; prediction cũ được giữ tại `predictions.schema2.jsonl`. Migration chỉ
đổi định dạng, tuyệt đối không tạo checkpoint danh tính từ prediction.

## Kiểm tra và chấm điểm

Kiểm tra một session:

```powershell
..\.venv\Scripts\python.exe experiment_test\validate_session.py `
  --session experiment_test\output\droidcam_shared_m_01
```

Chấm đồng thời bốn session và tạo báo cáo tổng hợp:

```powershell
..\.venv\Scripts\python.exe evaluate.py `
  experiment_test\output\droidcam_shared_m_01 `
  experiment_test\output\droidcam_shared_m_02 `
  experiment_test\output\droidcam_shared_m_03 `
  experiment_test\output\droidcam_shared_m_04 `
  --fps 25
```

Mỗi session sinh `evaluation_results_v3.json` và `evaluation_report_v3.md`.
Thư mục cha sinh `evaluation_summary_v3.json` và
`evaluation_summary_v3.md`. CLI trả exit code `1` nếu có session `FAIL`; đây là
kết quả đánh giá, không phải lỗi chạy Python.

Evaluator chỉ nhận schema v3. Một lỗi gán nhầm GID, dùng chung GID cho hai xe,
hoặc lưu sai chủ ô đủ lâu sẽ làm session `FAIL` và giới hạn điểm tối đa 49/100.
