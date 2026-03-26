import cv2
import json
import numpy as np

# ══════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════
SLOTS_FILE   = 'parking_slots.json'
RATIO_THRESH = 0.25   # tỷ lệ pixel trắng để coi là "có xe"

# ══════════════════════════════════════════════
#  LOAD TỌA ĐỘ Ô ĐỖ XE
# ══════════════════════════════════════════════
try:
    with open(SLOTS_FILE, 'r') as f:
        data = json.load(f)
    slots      = data['slots']
    IMG_W_REF  = data['imageWidth']
    IMG_H_REF  = data['imageHeight']
    print(f"✅ Đã load {len(slots)} ô từ {SLOTS_FILE}")
except FileNotFoundError:
    print(f"❌ Không tìm thấy {SLOTS_FILE}")
    exit()

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
#  TRACKBAR – CHỈNH LIVE
# ══════════════════════════════════════════════
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Settings', 400, 200)
cv2.createTrackbar('Gamma x10',  'Settings', 18,  30,  lambda x: None)  # default 1.5
cv2.createTrackbar('CLAHE Clip', 'Settings', 60,  60,  lambda x: None)  # default 2.0
cv2.createTrackbar('CLAHE Grid', 'Settings', 15,   16,  lambda x: None)  # default 8
cv2.createTrackbar('Threshold%', 'Settings', 19,  100, lambda x: None)  # default 25%
cv2.createTrackbar('Denoise',    'Settings', 1,   1,   lambda x: None)  # 0/1

def get_params():
    gamma      = max(cv2.getTrackbarPos('Gamma x10',  'Settings') / 10.0, 0.1)
    clahe_clip = max(cv2.getTrackbarPos('CLAHE Clip', 'Settings') / 10.0, 0.1)
    clahe_grid = max(cv2.getTrackbarPos('CLAHE Grid', 'Settings'), 2)
    ratio_thr  = max(cv2.getTrackbarPos('Threshold%', 'Settings') / 100.0, 0.01)
    denoise    = cv2.getTrackbarPos('Denoise', 'Settings')
    return gamma, clahe_clip, clahe_grid, ratio_thr, denoise

# ══════════════════════════════════════════════
#  GAMMA LUT – tính trước để tái sử dụng
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
#  PREPROCESS – xử lý bóng râm / sương mù / mưa
# ══════════════════════════════════════════════
def preprocess(img, gamma, clahe_clip, clahe_grid, denoise):
    # 1. Chuyển sang Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Denoise nhẹ nếu bật (mưa / nhiễu camera)
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. CLAHE – cân bằng sáng cục bộ (xử lý bóng râm, sương mù)
    clahe      = cv2.createCLAHE(clipLimit=clahe_clip,
                                  tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray)

    # 4. Gamma correction – tăng sáng vùng tối
    lut       = get_gamma_lut(gamma)
    gray_gam  = cv2.LUT(gray_clahe, lut)

    # 5. Adaptive Threshold
    blur      = cv2.GaussianBlur(gray_gam, (3, 3), 1)
    thresh    = cv2.adaptiveThreshold(blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 25, 16)

    # 6. Làm sạch nhiễu
    median    = cv2.medianBlur(thresh, 5)
    kernel    = np.ones((3, 3), np.uint8)
    dilated   = cv2.dilate(median, kernel, iterations=1)

    return dilated

# ══════════════════════════════════════════════
#  DETECT – vẽ ô và đếm chỗ trống
# ══════════════════════════════════════════════
def detect(img, ratio_thr):
    h, w   = img.shape[:2]
    sx     = w / IMG_W_REF
    sy     = h / IMG_H_REF

    gamma, clahe_clip, clahe_grid, ratio_thr_bar, denoise = get_params()
    # ưu tiên tham số từ trackbar nếu đang dùng video/camera
    ratio_thr = ratio_thr_bar

    imgPro = preprocess(img, gamma, clahe_clip, clahe_grid, denoise)
    free   = 0

    for slot in slots:
        pts = np.array(
            [[int(p['x'] * sx), int(p['y'] * sy)] for p in slot['polygon']],
            np.int32
        )

        # Tính tỷ lệ pixel trắng trong ô
        mask   = np.zeros(imgPro.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(imgPro, imgPro, mask=mask)
        count  = cv2.countNonZero(masked)
        area   = cv2.contourArea(pts)
        ratio  = count / area if area > 0 else 0

        # Màu và đếm
        is_free = ratio < ratio_thr
        color   = (0, 255, 0) if is_free else (0, 0, 255)
        if is_free:
            free += 1

        cv2.polylines(img, [pts], True, color, 2)

        # Tên ô
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(img, slot['id'], (cx - 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Hiển thị thống kê + thông số đang dùng
    total = len(slots)
    cv2.putText(img, f'Trong: {free}/{total}',
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    cv2.putText(img,
                f'G={gamma:.1f} CLAHE={clahe_clip:.1f} Thr={ratio_thr:.0%}',
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    return img

# ══════════════════════════════════════════════
#  CHẠY
# ══════════════════════════════════════════════
if use_image:
    img_src = cv2.imread(source)
    if img_src is None:
        print(f"❌ Không đọc được ảnh: {source}")
        exit()

    print("Nhấn Q hoặc phím bất kỳ để đóng | Chỉnh trackbar để thay đổi tham số")
    while True:
        img_copy = img_src.copy()
        result   = detect(img_copy, RATIO_THRESH)
        cv2.imshow('Parking', result)
        key = cv2.waitKey(100) & 0xFF
        if key != 255:   # nhấn bất kỳ phím nào → thoát
            break

else:
    print("Đang chạy... Nhấn Q để thoát")
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            continue

        result = detect(img, RATIO_THRESH)
        cv2.imshow('Parking', result)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()

#18 60 15 19 1