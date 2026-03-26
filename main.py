import cv2
import pickle
import cvzone
import numpy as np

# ── Cấu hình ──
width, height = 20, 39      # kích thước ô chữ nhật (dùng cho mode cũ)
THRESHOLD = 234               # chỉnh nếu kết quả sai
USE_POLYGON = False           # True = dùng polygon, False = dùng chữ nhật cũ

# ── Load tọa độ ô ──
try:
    if USE_POLYGON:
        with open('CarParkPos_polygon', 'rb') as f:
            posList = pickle.load(f)
        print(f"✅ Polygon mode: {len(posList)} ô")
    else:
        with open('CarParkPos', 'rb') as f:
            posList = pickle.load(f)
        print(f"✅ Rectangle mode: {len(posList)} ô")
except FileNotFoundError as e:
    print(f"❌ Không tìm thấy file tọa độ: {e}")
    exit()

# ── Chọn nguồn đầu vào ──
print("\nChon nguon:")
print("1 - Video")
print("2 - Anh tinh")
print("3 - Camera truc tiep")
choice = input("Nhap (1/2/3): ").strip()

use_image = False

if choice == '1':
    source = input("Duong dan video: ").strip()
    cap = cv2.VideoCapture(source)
elif choice == '2':
    source = input("Duong dan anh: ").strip()
    use_image = True
elif choice == '3':
    cam_id = input("Camera ID (Enter=0): ").strip()
    cap = cv2.VideoCapture(int(cam_id) if cam_id else 0)
else:
    cap = cv2.VideoCapture('carPark.mp4')

# ── Xử lý ảnh (giữ nguyên pipeline tốt của bạn) ──
def preprocess(img):
    imgGray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur      = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255,
                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian    = cv2.medianBlur(imgThreshold, 5)
    kernel       = np.ones((3, 3), np.uint8)
    imgDilate    = cv2.dilate(imgMedian, kernel, iterations=1)
    return imgDilate

# ── Check ô chữ nhật (mode cũ) ──
def checkParkingSpace_rect(imgPro, img):
    spaceCounter = 0
    for pos in posList:
        x, y = pos
        imgCrop = imgPro[y:y+height, x:x+width]
        count = cv2.countNonZero(imgCrop)

        cvzone.putTextRect(img, str(count), (x, y+height-3),
                           scale=1, thickness=2, offset=0,
                           colorT=(255,255,255), colorR=(0,0,255))

        if count < THRESHOLD:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)

        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, 2)

    cvzone.putTextRect(img, f'Available: {spaceCounter}/{len(posList)}',
                       (50, 50), scale=2, thickness=2, offset=5,
                       colorR=(0, 200, 0))

# ── Check ô polygon (mode mới) ──
def checkParkingSpace_polygon(imgPro, img):
    spaceCounter = 0
    for i, spot in enumerate(posList):
        pts  = np.array(spot, np.int32)
        mask = np.zeros(imgPro.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(imgPro, imgPro, mask=mask)
        count  = cv2.countNonZero(masked)
        area   = cv2.contourArea(pts)
        ratio  = count / area if area > 0 else 0

        # Tâm ô để hiện số
        cx = int(np.mean([p[0] for p in spot]))
        cy = int(np.mean([p[1] for p in spot]))

        if ratio < 0.25:
            color = (0, 255, 0)
            spaceCounter += 1
        else:
            color = (0, 0, 255)

        cv2.polylines(img, [pts], True, color, 2)
        cvzone.putTextRect(img, str(i+1), (cx-10, cy+5),
                           scale=0.8, thickness=1, offset=2)

    cvzone.putTextRect(img, f'Available: {spaceCounter}/{len(posList)}',
                       (50, 50), scale=2, thickness=2, offset=5,
                       colorR=(0, 200, 0))

# ── Chạy ──
def run_frame(img):
    imgPro = preprocess(img)
    if USE_POLYGON:
        checkParkingSpace_polygon(imgPro, img)
    else:
        checkParkingSpace_rect(imgPro, img)
    return img

if use_image:
    img = cv2.imread(source)
    if img is None:
        print(f"❌ Không đọc được ảnh: {source}")
    else:
        result = run_frame(img)
        cv2.imshow('Parking Detection', result)
        print("Nhan phim bat ky de dong...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("Dang chay... Nhan Q de thoat")
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        if not success:
            continue

        result = run_frame(img)
        cv2.imshow('Parking Detection', result)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()