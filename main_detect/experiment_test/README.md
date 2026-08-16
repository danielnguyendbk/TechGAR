# Ghi dữ liệu thực nghiệm một camera

## Việc cần làm

1. Cài thư viện một lần:

   ```powershell
   cd D:\techgar\main_detect
   .\.venv\Scripts\python.exe -m pip install -r experiment_test\requirements.txt
   ```

2. Cắm điện thoại ở chế độ webcam USB, kiểm tra camera xuất hiện trong Windows, rồi chạy:

   ```powershell
   .\.venv\Scripts\python.exe experiment_test\record_experiment.py --camera 0 --session-name test_01
   ```

   Nếu camera điện thoại không phải số `0`, thử `--camera 1` hoặc `--camera 2`.

3. Nếu dùng ứng dụng điện thoại phát MJPEG/RTSP qua Wi-Fi, thay bằng URL ứng dụng hiển thị:

   ```powershell
   .\.venv\Scripts\python.exe experiment_test\record_experiment.py --stream-url "http://192.168.1.10:8080/video" --session-name test_01
   ```

4. Thực hiện kịch bản thử. Nhấn `Q` hoặc `Esc` để kết thúc. Không đóng cửa sổ terminal đột ngột.

5. Kiểm tra bộ file vừa ghi:

   ```powershell
   .\.venv\Scripts\python.exe experiment_test\validate_session.py --session experiment_test\output\test_01
   ```

6. Mở `debug_video.mp4`, dùng số `frame` in trên video để điền:

   - `ground_truth_slots.csv`: khoảng frame một ô thực sự trống/có xe.
   - `ground_truth_events.csv`: frame xe vào ô, dừng, rời ô hoặc ra khỏi khung.

Mỗi session tự sinh `raw_video.mp4`, `debug_video.mp4`, `predictions.jsonl`, `frame_timestamps.csv`, `performance.csv`, `session_info.json` và hai file ground truth.
