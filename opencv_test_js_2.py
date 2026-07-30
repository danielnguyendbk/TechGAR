"""
opencv_test_js_2.py – Nhận diện chỗ đỗ xe (phiên bản cải tiến)
Hỗ trợ: Threshold / YOLO / CNN / Hybrid
Cải tiến: LAB CLAHE, auto-gamma, đa format JSON, temporal smoothing
"""
import cv2
import json
import math
import numpy as np
import time
import os

# ══════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════
SLOTS_FILE    = 'parking_slots_polygon.json'
RATIO_THRESH  = 0.25
SKIP_FRAME    = 5
OUTPUT_JSON   = 'parking_status.json'
UPDATE_EVERY  = 1.0

# ══════════════════════════════════════════════
#  CHỌN CHẾ ĐỘ NHẬN DIỆN
# ══════════════════════════════════════════════
print("Chọn chế độ nhận diện:")
print("  1 - Threshold (OpenCV thuần)")
print("  2 - YOLO (detect xe bằng AI)")
print("  3 - Hybrid (Threshold + YOLO cho vùng xám)")
mode_choice = input("Nhập (1/2/3, Enter=1): ").strip()
DETECT_MODE = {'1': 'threshold', '2': 'yolo', '3': 'hybrid'}.get(mode_choice, 'threshold')
print(f"✅ Chế độ: {DETECT_MODE.upper()}")

# Load YOLO nếu cần
yolo_model = None
if DETECT_MODE in ('yolo', 'hybrid'):
    try:
        from parking_yolo_utils import VehicleDetectorYOLO, overlap_ratio_roi
        yolo_model = VehicleDetectorYOLO("yolov8n.pt", conf=0.20, iou=0.45)
        print("✅ YOLO model loaded")
    except Exception as e:
        print(f"⚠️ Không load được YOLO: {e}")
        print("   Cài bằng: pip install ultralytics")
        if DETECT_MODE == 'yolo':
            exit()
        DETECT_MODE = 'threshold'
        print("   → Chuyển về mode threshold")

# ══════════════════════════════════════════════
#  LOAD TỌA ĐỘ Ô ĐỖ XE (hỗ trợ đa format)
# ══════════════════════════════════════════════
try:
    with open(SLOTS_FILE, 'r') as f:
        data = json.load(f)
    slots     = data['slots']
    IMG_W_REF = data['imageWidth']
    IMG_H_REF = data['imageHeight']
    print(f"✅ Load {len(slots)} ô từ {SLOTS_FILE}")
except FileNotFoundError:
    print(f"❌ Không tìm thấy {SLOTS_FILE}")
    exit()


def get_polygon(slot):
    """Lấy polygon từ slot, hỗ trợ cả format polygon lẫn OBB rect."""
    # Format 1: polygon trực tiếp [{x,y}, ...]
    for key in ['polygon', 'points', 'coordinates', 'vertices']:
        if key in slot and slot[key]:
            return slot[key]

    # Format 2: OBB rect {cx, cy, w, h, angle} → chuyển thành polygon
    if 'rect' in slot and slot['rect']:
        r = slot['rect']
        cx, cy = r['cx'], r['cy']
        w, h = r['w'], r['h']
        angle = r.get('angle', 0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw, hh = w / 2, h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        polygon = []
        for dx, dy in corners:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            polygon.append({'x': round(rx, 2), 'y': round(ry, 2)})
        return polygon

    return None


# Lọc slot hợp lệ
valid_slots = []
for slot in slots:
    poly = get_polygon(slot)
    if poly is None:
        continue
    slot['_polygon'] = poly
    valid_slots.append(slot)

print(f"✅ Hợp lệ: {len(valid_slots)}/{len(slots)} ô")
slots = valid_slots

# ══════════════════════════════════════════════
#  CHỌN NGUỒN ĐẦU VÀO
# ══════════════════════════════════════════════
print("\nChọn nguồn:")
print("1 - Video")
print("2 - Ảnh tĩnh")
print("3 - Camera trực tiếp")
choice = input("Nhập (1/2/3): ").strip()

use_image = False
if choice == '1':
    source = input("Đường dẫn video: ").strip()
    cap = cv2.VideoCapture(source)
elif choice == '2':
    source = input("Đường dẫn ảnh: ").strip()
    use_image = True
elif choice == '3':
    cam_id = input("Camera ID (Enter=0): ").strip()
    cap = cv2.VideoCapture(int(cam_id) if cam_id else 0)
else:
    cap = cv2.VideoCapture('carPark.mp4')

# ══════════════════════════════════════════════
#  TRACKBAR
# ══════════════════════════════════════════════
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Settings', 400, 200)
cv2.createTrackbar('Gamma x10',  'Settings', 16, 30,  lambda x: None)
cv2.createTrackbar('CLAHE Clip', 'Settings', 30, 60,  lambda x: None)
cv2.createTrackbar('CLAHE Grid', 'Settings', 14, 16,  lambda x: None)
cv2.createTrackbar('Threshold',  'Settings', 20, 100, lambda x: None)
cv2.createTrackbar('Denoise',    'Settings', 1,  1,   lambda x: None)


def get_params():
    try:
        if cv2.getWindowProperty('Settings', cv2.WND_PROP_VISIBLE) < 1:
            return 1.6, 3.0, 14, 0.20, 1
        gamma      = max(cv2.getTrackbarPos('Gamma x10',  'Settings') / 10.0, 0.1)
        clahe_clip = max(cv2.getTrackbarPos('CLAHE Clip', 'Settings') / 10.0, 0.1)
        clahe_grid = max(cv2.getTrackbarPos('CLAHE Grid', 'Settings'), 2)
        ratio_thr  = max(cv2.getTrackbarPos('Threshold',  'Settings') / 100.0, 0.01)
        denoise    = cv2.getTrackbarPos('Denoise', 'Settings')
        return gamma, clahe_clip, clahe_grid, ratio_thr, denoise
    except cv2.error:
        return 1.6, 3.0, 14, 0.20, 1

# ══════════════════════════════════════════════
#  GAMMA LUT
# ══════════════════════════════════════════════
_last_gamma = None
_lut        = None


def get_gamma_lut(gamma):
    global _last_gamma, _lut
    if gamma != _last_gamma:
        _lut = np.array([np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
                         for i in range(256)], dtype=np.uint8)
        _last_gamma = gamma
    return _lut

# ══════════════════════════════════════════════
#  PREPROCESS – CẢI TIẾN: dùng LAB color space
# ══════════════════════════════════════════════
def preprocess(img, gamma, clahe_clip, clahe_grid, denoise):
    # === CẢI TIẾN: Dùng kênh L của LAB thay vì Grayscale ===
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    if denoise:
        l_channel = cv2.bilateralFilter(l_channel, 5, 50, 50)

    # CLAHE trên kênh L (cân bằng sáng cục bộ – xử lý bóng râm)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,
                             tileGridSize=(clahe_grid, clahe_grid))
    l_clahe = clahe.apply(l_channel)

    # Auto-gamma: tự tăng gamma nếu ảnh quá tối
    mean_brightness = np.mean(l_clahe)
    effective_gamma = gamma
    if mean_brightness < 60:
        effective_gamma = max(gamma, 2.0)
    elif mean_brightness < 100:
        effective_gamma = max(gamma, 1.5)

    lut = get_gamma_lut(effective_gamma)
    l_gamma = cv2.LUT(l_clahe, lut)

    blur    = cv2.GaussianBlur(l_gamma, (3, 3), 1)
    thresh  = cv2.adaptiveThreshold(blur, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY_INV, 25, 16)
    median  = cv2.medianBlur(thresh, 5)
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated

# ══════════════════════════════════════════════
#  TEMPORAL SMOOTHING – chống nhấp nháy cho video
# ══════════════════════════════════════════════
class TemporalSmoother:
    def __init__(self, num_slots, required_frames=10):
        self.required = required_frames
        self.counters = [0] * num_slots
        self.pending  = [None] * num_slots
        self.confirmed = [False] * num_slots

    def update(self, slot_idx, is_occupied):
        if self.pending[slot_idx] == is_occupied:
            self.counters[slot_idx] += 1
        else:
            self.pending[slot_idx] = is_occupied
            self.counters[slot_idx] = 1
        if self.counters[slot_idx] >= self.required:
            self.confirmed[slot_idx] = is_occupied
        return self.confirmed[slot_idx]

smoother = TemporalSmoother(len(slots), required_frames=10)

# ══════════════════════════════════════════════
#  XUẤT JSON TRẠNG THÁI
# ══════════════════════════════════════════════
def save_status_json(slot_results):
    free_count     = sum(1 for s in slot_results if not s['occupied'])
    occupied_count = sum(1 for s in slot_results if s['occupied'])
    output = {
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode":       DETECT_MODE,
        "total":      len(slot_results),
        "free":       free_count,
        "occupied":   occupied_count,
        "slots": {
            s['id']: {"occupied": s['occupied'],
                      "status": "occupied" if s['occupied'] else "empty"}
            for s in slot_results
        }
    }
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except PermissionError:
        pass  # Bỏ qua nếu file đang bị lock bởi tiến trình khác (web server)

# ══════════════════════════════════════════════
#  DETECT
# ══════════════════════════════════════════════
def detect(img, ratio_thr, apply_smoothing=False):
    h, w = img.shape[:2]
    sx = w / IMG_W_REF
    sy = h / IMG_H_REF

    gamma, clahe_clip, clahe_grid, ratio_thr_bar, denoise = get_params()
    ratio_thr = ratio_thr_bar

    imgPro = preprocess(img, gamma, clahe_clip, clahe_grid, denoise)
    free = 0
    slot_results = []

    # YOLO detection (chạy 1 lần trên toàn frame)
    yolo_boxes = []
    if DETECT_MODE in ('yolo', 'hybrid') and yolo_model:
        yolo_boxes = yolo_model.detect(img)

    for idx, slot in enumerate(slots):
        poly = slot['_polygon']
        try:
            if isinstance(poly[0], dict):
                pts = np.array([[int(p['x'] * sx), int(p['y'] * sy)]
                                for p in poly], np.int32)
            else:
                pts = np.array([[int(p[0] * sx), int(p[1] * sy)]
                                for p in poly], np.int32)
        except Exception:
            continue

        # Tính ratio threshold
        mask   = np.zeros(imgPro.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(imgPro, imgPro, mask=mask)
        count  = cv2.countNonZero(masked)
        area   = cv2.contourArea(pts)
        ratio  = count / area if area > 0 else 0

        # === Quyết định trạng thái theo mode ===
        if DETECT_MODE == 'threshold':
            is_free = ratio < ratio_thr

        elif DETECT_MODE == 'yolo':
            # Tính bounding rect của polygon → overlap với YOLO boxes
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            x_max = int(np.max(pts[:, 0]))
            y_max = int(np.max(pts[:, 1]))
            roi_xyxy = (x_min, y_min, x_max, y_max)
            max_overlap = 0.0
            for d in yolo_boxes:
                box_xyxy = (d["x1"], d["y1"], d["x2"], d["y2"])
                ov = overlap_ratio_roi(roi_xyxy, box_xyxy)
                if ov > max_overlap:
                    max_overlap = ov
            is_free = max_overlap < 0.05  # overlap < 5% → trống (aerial view cần thấp)

        elif DETECT_MODE == 'hybrid':
            # Hybrid: dùng threshold nhanh cho trường hợp rõ ràng
            if ratio < 0.10:
                is_free = True   # chắc chắn trống
            elif ratio > 0.45:
                is_free = False  # chắc chắn có xe
            else:
                # Vùng xám → hỏi YOLO
                x_min = int(np.min(pts[:, 0]))
                y_min = int(np.min(pts[:, 1]))
                x_max = int(np.max(pts[:, 0]))
                y_max = int(np.max(pts[:, 1]))
                roi_xyxy = (x_min, y_min, x_max, y_max)
                max_overlap = 0.0
                for d in yolo_boxes:
                    box_xyxy = (d["x1"], d["y1"], d["x2"], d["y2"])
                    ov = overlap_ratio_roi(roi_xyxy, box_xyxy)
                    if ov > max_overlap:
                        max_overlap = ov
                is_free = max_overlap < 0.05
        else:
            is_free = ratio < ratio_thr

        # Temporal smoothing cho video
        if apply_smoothing:
            is_free = not smoother.update(idx, not is_free)

        color = (0, 255, 0) if is_free else (0, 0, 255)
        if is_free:
            free += 1

        slot_results.append({
            "id":       slot['id'],
            "occupied": not is_free
        })

        cv2.polylines(img, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, slot['id'], (cx - 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Vẽ YOLO boxes nếu có
    for d in yolo_boxes:
        cv2.rectangle(img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (255, 200, 0), 1)

    total = len(slots)
    cv2.putText(img, f'Trong: {free}/{total}  [{DETECT_MODE.upper()}]',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(img,
                f'G={gamma:.1f} CLAHE={clahe_clip:.1f} Thr={ratio_thr:.0%}',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    return img, slot_results

# ══════════════════════════════════════════════
#  CHẠY
# ══════════════════════════════════════════════
if use_image:
    img_src = cv2.imread(source)
    if img_src is None:
        print(f"❌ Không đọc được ảnh: {source}")
        exit()

    print("Nhấn phím bất kỳ để đóng | Chỉnh trackbar để thay đổi tham số")
    while True:
        img_copy = img_src.copy()
        result, slot_results = detect(img_copy, RATIO_THRESH, apply_smoothing=False)
        save_status_json(slot_results)
        cv2.imshow('Parking', result)
        key = cv2.waitKey(100) & 0xFF
        if key != 255:
            break
        try:
            if cv2.getWindowProperty('Parking', cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

else:
    print("Đang chạy... Nhấn Q để thoát")
    frame_count    = 0
    last_result    = None
    last_save_time = time.time()
    cv2.setTrackbarPos('Denoise', 'Settings', 0)

    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            continue

        img = cv2.resize(img, (IMG_W_REF, IMG_H_REF))
        frame_count += 1

        if frame_count % SKIP_FRAME == 0:
            last_result, slot_results = detect(
                img.copy(), RATIO_THRESH, apply_smoothing=True
            )
            now = time.time()
            if now - last_save_time >= UPDATE_EVERY:
                save_status_json(slot_results)
                last_save_time = now

        if last_result is not None:
            cv2.imshow('Parking', last_result)
        else:
            cv2.imshow('Parking', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()