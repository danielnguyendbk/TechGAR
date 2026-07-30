# ParkingSpacePicker.py - Version chấm 4 góc polygon + xuất JSON
import cv2
import pickle
import json
import numpy as np
import math

# ── Cấu hình ──
IMG_PATH = "dataset/parkingimg.jpg"   # ảnh của bạn

# ── Load tọa độ cũ nếu có ──
# Mỗi phần tử: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
    # Kiểm tra format cũ
    if posList:
        sample = posList[0]
        if isinstance(sample, tuple) and len(sample) == 2:
            # Format cũ (x, y) → rect polygon
            print("⚠️  Phát hiện format cũ (x,y), chuyển sang polygon 4 góc")
            old = posList[:]
            posList = []
            for (x, y) in old:
                w, h = 115, 50
                posList.append([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])
        elif isinstance(sample, tuple) and len(sample) == 4:
            # Format rect (x1,y1,x2,y2) → polygon
            print("⚠️  Phát hiện format rect, chuyển sang polygon 4 góc")
            old = posList[:]
            posList = []
            for (x1,y1,x2,y2) in old:
                posList.append([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
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
print("="*60)
print("HƯỚNG DẪN:")
print("  Click TRÁI   → chấm 4 góc của ô đỗ xe")
print("  Phím A       → xác nhận ô (sau khi chấm đủ 4 điểm)")
print("  Phím D       → hủy các điểm đang chấm")
print("  Click PHẢI   → xóa ô đã lưu (click vào ô cần xóa)")
print("  Phím S       → lưu pickle + JSON")
print("  Phím Z       → hoàn tác ô cuối")
print("  Phím C       → xóa tất cả")
print("  Phím Q       → lưu và thoát")
print("="*60)

# ── Trạng thái chấm điểm ──
temp_points = []  # điểm đang chấm (chưa xác nhận)

def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: trên-trái, trên-phải, dưới-phải, dưới-trái."""
    pts = np.array(pts, dtype=np.float32)
    # Tính centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    # Tính góc từ centroid
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    # Sắp xếp theo góc (bắt đầu từ trên-trái, đi theo chiều kim đồng hồ)
    order = np.argsort(angles)
    ordered = pts[order]
    # Tìm điểm trên-trái (y nhỏ nhất, x nhỏ nhất nếu cùng y)
    # Xoay lại sao cho điểm đầu tiên là top-left
    top_idx = np.argmin(ordered[:, 1] + ordered[:, 0])
    ordered = np.roll(ordered, -top_idx, axis=0)
    return [tuple(p.astype(int)) for p in ordered]

def point_in_polygon(px, py, polygon):
    """Kiểm tra điểm (px, py) có nằm trong polygon không."""
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(px), float(py)), False)
    return result >= 0

def save_all():
    # Lưu pickle
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)

    # Xuất JSON (tương thích ensemble_test.py)
    slots = []
    for i, poly in enumerate(posList):
        pts = np.array(poly, dtype=np.float32)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        slots.append({
            "id": f"P{i+1:03d}",
            "type": "polygon",
            "polygon": [
                {"x": int(p[0]), "y": int(p[1])} for p in poly
            ],
            "center": {"x": cx, "y": cy},
            "status": "empty"
        })

    data = {
        "imageWidth":  W_IMG,
        "imageHeight": H_IMG,
        "slots": slots
    }

    with open('parking_slots_2.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu {len(posList)} ô → CarParkPos + parking_slots_1.json")

def mouseCallback(event, x, y, flags, param):
    global temp_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(temp_points) < 4:
            temp_points.append((x, y))
            n = len(temp_points)
            print(f"  📌 Điểm {n}/4: ({x}, {y})"
                  + (" → Bấm A để xác nhận!" if n == 4 else ""))

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Xóa ô chứa điểm click
        for i, poly in enumerate(posList):
            if point_in_polygon(x, y, poly):
                posList.pop(i)
                print(f"  ✖ Xóa ô {i+1}, còn {len(posList)} ô")
                with open('CarParkPos', 'wb') as f:
                    pickle.dump(posList, f)
                break

# ── Màu sắc ──
COLOR_SAVED     = (255, 0, 255)   # Hồng — ô đã lưu
COLOR_TEMP_PT   = (0, 255, 255)   # Vàng — điểm đang chấm
COLOR_TEMP_LINE = (0, 255, 0)     # Xanh lá — đường nối tạm
COLOR_READY     = (0, 200, 0)     # Xanh đậm — sẵn sàng xác nhận (4 điểm)

# ── Vòng lặp chính ──
cv2.namedWindow("ParkingSpacePicker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("ParkingSpacePicker", mouseCallback)

while True:
    img = img_src.copy()

    # ── Vẽ các ô đã lưu ──
    for i, poly in enumerate(posList):
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=COLOR_SAVED, thickness=2)
        # Số thứ tự tại centroid
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        cv2.putText(img, str(i+1), (cx-8, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── Vẽ các điểm đang chấm ──
    n_temp = len(temp_points)
    if n_temp > 0:
        for j, pt in enumerate(temp_points):
            # Vẽ chấm tròn
            cv2.circle(img, pt, 5, COLOR_TEMP_PT, -1)
            cv2.putText(img, str(j+1), (pt[0]+8, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEMP_PT, 1)

        # Nối các điểm với nhau
        line_color = COLOR_READY if n_temp == 4 else COLOR_TEMP_LINE
        for j in range(n_temp - 1):
            cv2.line(img, temp_points[j], temp_points[j+1], line_color, 2)
        # Nối điểm cuối → đầu nếu đủ 4
        if n_temp == 4:
            cv2.line(img, temp_points[3], temp_points[0], COLOR_READY, 2)

    # ── Thanh trạng thái ──
    bar_w = 580
    cv2.rectangle(img, (0, 0), (bar_w, 40), (0, 0, 0), -1)
    if n_temp > 0 and n_temp < 4:
        status = f"Dang cham: {n_temp}/4 diem | Click tiep... | D=Huy"
    elif n_temp == 4:
        status = f"DU 4 DIEM! Bam A=Xac nhan | D=Huy"
    else:
        status = f"So o: {len(posList)} | Click 4 goc | S=Luu | Z=Undo | C=Xoa | Q=Thoat"
    cv2.putText(img, status, (5, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    cv2.imshow("ParkingSpacePicker", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        # Xác nhận polygon (cần đủ 4 điểm)
        if len(temp_points) == 4:
            ordered = order_points(temp_points)
            posList.append(ordered)
            print(f"  ✅ Xác nhận ô {len(posList)}: {ordered}")
            temp_points = []
            with open('CarParkPos', 'wb') as f:
                pickle.dump(posList, f)
        else:
            print(f"  ⚠️  Chưa đủ 4 điểm (hiện có {len(temp_points)})")

    elif key == ord('d'):
        # Hủy các điểm đang chấm
        if temp_points:
            temp_points = []
            print("  🔄 Đã hủy các điểm đang chấm")

    elif key == ord('s'):
        save_all()

    elif key == ord('z'):
        if temp_points:
            removed = temp_points.pop()
            print(f"  ↩ Bỏ điểm cuối, còn {len(temp_points)} điểm")
        elif posList:
            posList.pop()
            print(f"  ↩ Hoàn tác ô cuối, còn {len(posList)} ô")
            with open('CarParkPos', 'wb') as f:
                pickle.dump(posList, f)

    elif key == ord('c'):
        posList.clear()
        temp_points = []
        print("  🗑 Đã xóa tất cả!")
        with open('CarParkPos', 'wb') as f:
            pickle.dump(posList, f)

    elif key == ord('q'):
        save_all()
        break

cv2.destroyAllWindows()
print("Đã thoát.")