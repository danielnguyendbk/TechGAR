import cv2
import json
import numpy as np
import time
import os

# ══════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════
SLOTS_FILE    = 'parking_slots_1.json'
RATIO_THRESH  = 0.25
SKIP_FRAME    = 5
OUTPUT_JSON   = 'parking_status.json'   # ← file web đọc
UPDATE_EVERY  = 1.0                     # ← cập nhật mỗi 1 giây (video)

# ══════════════════════════════════════════════
#  LOAD TỌA ĐỘ Ô ĐỖ XE
# ══════════════════════════════════════════════
try:
    with open(SLOTS_FILE, 'r') as f:
        data = json.load(f)
    slots     = data['slots']
    IMG_W_REF = data['imageWidth']
    IMG_H_REF = data['imageHeight']
    print(f"✅ load {len(slots)} ô từ {SLOTS_FILE}")
except FileNotFoundError:
    print(f"❌ Không tìm thấy {SLOTS_FILE}")
    exit()

# ── Lấy polygon data tự động ──
def get_polygon(slot):
    for key in ['polygon', 'points', 'coordinates', 'vertices', 'bbox']:
        if key in slot and slot[key]:
            return slot[key]
    return None

# ── Lọc slot hợp lệ ──
valid_slots = []
for slot in slots:
    poly = get_polygon(slot)
    if poly is None:
        print(f"⚠️ Bỏ qua ô {slot.get('id','?')} – không tìm thấy polygon")
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
    gamma      = max(cv2.getTrackbarPos('Gamma x10',  'Settings') / 10.0, 0.1)
    clahe_clip = max(cv2.getTrackbarPos('CLAHE Clip', 'Settings') / 10.0, 0.1)
    clahe_grid = max(cv2.getTrackbarPos('CLAHE Grid', 'Settings'), 2)
    ratio_thr  = max(cv2.getTrackbarPos('Threshold',  'Settings') / 100.0, 0.01)
    denoise    = cv2.getTrackbarPos('Denoise', 'Settings')
    return gamma, clahe_clip, clahe_grid, ratio_thr, denoise

# ══════════════════════════════════════════════
#  GAMMA LUT
# ══════════════════════════════════════════════
_last_gamma = None
_lut        = None

def get_gamma_lut(gamma):
    global _last_gamma, _lut
    if gamma != _last_gamma:
        _lut        = np.array([np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
                                 for i in range(256)], dtype=np.uint8)
        _last_gamma = gamma
    return _lut

# ══════════════════════════════════════════════
#  PREPROCESS
# ══════════════════════════════════════════════
def preprocess(img, gamma, clahe_clip, clahe_grid, denoise):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if denoise:
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

    clahe      = cv2.createCLAHE(clipLimit=clahe_clip,
                                  tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray)

    lut      = get_gamma_lut(gamma)
    gray_gam = cv2.LUT(gray_clahe, lut)

    blur    = cv2.GaussianBlur(gray_gam, (3, 3), 1)
    thresh  = cv2.adaptiveThreshold(blur, 255,
                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                  cv2.THRESH_BINARY_INV, 25, 16)
    median  = cv2.medianBlur(thresh, 5)
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated

# ══════════════════════════════════════════════
#  XUẤT JSON TRẠNG THÁI
# ══════════════════════════════════════════════
def save_status_json(slot_results):
    """
    slot_results: list of {"id": "A01", "occupied": True/False}
    Xuất ra parking_status.json cho web đọc
    """
    free_count     = sum(1 for s in slot_results if not s['occupied'])
    occupied_count = sum(1 for s in slot_results if s['occupied'])

    output = {
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "total":          len(slot_results),
        "free":           free_count,
        "occupied":       occupied_count,
        "slots": {
            s['id']: {
                "occupied": s['occupied'],
                "status":   "occupied" if s['occupied'] else "empty"
            }
            for s in slot_results
        }
    }

    # Ghi file tạm rồi rename → tránh web đọc file đang ghi dở
    tmp_file = OUTPUT_JSON + '.tmp'
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    os.replace(tmp_file, OUTPUT_JSON)

# ══════════════════════════════════════════════
#  DETECT
# ══════════════════════════════════════════════
def detect(img, ratio_thr):
    h, w = img.shape[:2]
    sx   = w / IMG_W_REF
    sy   = h / IMG_H_REF

    gamma, clahe_clip, clahe_grid, ratio_thr_bar, denoise = get_params()
    ratio_thr = ratio_thr_bar

    imgPro = preprocess(img, gamma, clahe_clip, clahe_grid, denoise)
    free   = 0
    slot_results = []   # ← thu thập kết quả để xuất JSON

    for slot in slots:
        poly = slot['_polygon']

        try:
            if isinstance(poly[0], dict):
                pts = np.array([[int(p['x'] * sx), int(p['y'] * sy)]
                                for p in poly], np.int32)
            else:
                pts = np.array([[int(p[0] * sx), int(p[1] * sy)]
                                for p in poly], np.int32)
        except Exception as e:
            print(f"⚠️ Lỗi parse polygon ô {slot.get('id','?')}: {e}")
            continue

        mask   = np.zeros(imgPro.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(imgPro, imgPro, mask=mask)
        count  = cv2.countNonZero(masked)
        area   = cv2.contourArea(pts)
        ratio  = count / area if area > 0 else 0

        is_free = ratio < ratio_thr
        color   = (0, 255, 0) if is_free else (0, 0, 255)
        if is_free:
            free += 1

        # Thu thập kết quả
        slot_results.append({
            "id":       slot['id'],
            "occupied": not is_free
        })

        cv2.polylines(img, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, slot['id'], (cx - 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    total = len(slots)
    cv2.putText(img, f'Trong: {free}/{total}',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.putText(img,
                f'G={gamma:.1f} CLAHE={clahe_clip:.1f} Thr={ratio_thr:.0%}',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    return img, slot_results   # ← trả thêm slot_results

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
        img_copy           = img_src.copy()
        result, slot_results = detect(img_copy, RATIO_THRESH)

        # ── Ảnh tĩnh: xuất JSON 1 lần (cập nhật liên tục khi chỉnh trackbar) ──
        save_status_json(slot_results)

        cv2.imshow('Parking', result)
        key = cv2.waitKey(100) & 0xFF
        if key != 255:
            break

else:
    print("Đang chạy... Nhấn Q để thoát")
    frame_count   = 0
    last_result   = None
    last_save_time = time.time()

    # Tắt Denoise khi chạy video
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
            last_result, slot_results = detect(img.copy(), RATIO_THRESH)

            # ── Video: xuất JSON mỗi 1 giây ──
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