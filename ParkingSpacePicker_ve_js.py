# ParkingSpacePicker.py - Version mới có xuất JSON
import cv2
import pickle
import json
import numpy as np

# ── Cấu hình ──
IMG_PATH = "D:\opencv\parking-space-detection\parkingimg1.png"   # ảnh của bạn
WIDTH    = 115
HEIGHT   = 50

# ── Load tọa độ cũ nếu có ──
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
    print(f"✅ Load {len(posList)} ô cũ")
except:
    posList = []

# ── Load ảnh gốc 1 lần ──
img_src = cv2.imread(IMG_PATH)
if img_src is None:
    print(f"❌ Không đọc được {IMG_PATH}")
    exit()

H_IMG, W_IMG = img_src.shape[:2]
print(f"✅ Ảnh: {W_IMG}x{H_IMG}")
print("="*50)
print("HƯỚNG DẪN:")
print("  Click TRÁI   → thêm ô đỗ xe")
print("  Click PHẢI   → xóa ô đỗ xe")
print("  Phím S       → lưu pickle + JSON")
print("  Phím Z       → hoàn tác ô cuối")
print("  Phím C       → xóa tất cả")
print("  Phím Q       → lưu và thoát")
print("="*50)

def save_all():
    # Lưu pickle
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

    # Xuất JSON
    slots = []
    for i, (x, y) in enumerate(posList):
        slots.append({
            "id": f"P{i+1:03d}",
            "type": "polygon",
            "polygon": [
                {"x": x,         "y": y},
                {"x": x+WIDTH,   "y": y},
                {"x": x+WIDTH,   "y": y+HEIGHT},
                {"x": x,         "y": y+HEIGHT}
            ],
            "center": {"x": x + WIDTH//2, "y": y + HEIGHT//2},
            "status": "empty"
        })

    data = {
        "imageWidth":  W_IMG,
        "imageHeight": H_IMG,
        "slots": slots
    }

    with open('parking_slots_1.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu {len(posList)} ô → CarParkPos + parking_slots.json")

def mouseCLick(events, x, y, flags, param):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))

    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + WIDTH and y1 < y < y1 + HEIGHT:
                posList.pop(i)
                break

    # Tự động lưu sau mỗi click
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

# ── Vòng lặp chính ──
cv2.namedWindow("ParkingSpacePicker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ParkingSpacePicker", mouseCLick)

while True:
    img = img_src.copy()

    # Vẽ tất cả ô
    for i, pos in enumerate(posList):
        cv2.rectangle(img, pos,
                      (pos[0] + WIDTH, pos[1] + HEIGHT),
                      (255, 0, 255), 2)
        # Hiện số thứ tự
        cv2.putText(img, str(i+1),
                    (pos[0]+2, pos[1]+12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255, 255, 255), 1)

    # Thông tin góc trên
    cv2.rectangle(img, (0, 0), (400, 35), (0, 0, 0), -1)
    cv2.putText(img,
                f"So o: {len(posList)} | S=Luu JSON | Z=Hoan tac | C=Xoa het | Q=Thoat",
                (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1)

    cv2.imshow("ParkingSpacePicker", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        save_all()

    elif key == ord('z'):
        if posList:
            posList.pop()
            print(f"Hoàn tác, còn {len(posList)} ô")

    elif key == ord('c'):
        posList.clear()
        print("Đã xóa tất cả!")

    elif key == ord('q'):
        save_all()
        break

cv2.destroyAllWindows()
print("Đã thoát.")