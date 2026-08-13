import os
import json
import argparse
from pathlib import Path
import numpy as np
import cv2

# Khởi tạo biến toàn cục để tối ưu tính toán Gamma LUT
_last_gamma = None
_lut = None

def get_gamma_lut(gamma):
    global _last_gamma, _lut
    if gamma != _last_gamma:
        _lut = np.array([np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255)
                         for i in range(256)], dtype=np.uint8)
        _last_gamma = gamma
    return _lut

def preprocess(img, gamma, clahe_clip, clahe_grid, denoise):
    # 1. Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Khử nhiễu nếu bật
    if denoise:
        gray = cv2.bilateralFilter(gray, 5, 50, 50)

    # 3. CLAHE - Cân bằng sáng cục bộ
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray)

    # 4. Gamma Correction - Bù sáng vùng tối
    lut = get_gamma_lut(gamma)
    gray_gam = cv2.LUT(gray_clahe, lut)

    # 5. Phân ngưỡng thích nghi (Adaptive Threshold)
    blur = cv2.GaussianBlur(gray_gam, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 16)
    
    # 6. Làm sạch nhiễu hạt bằng Median Blur và Dilation
    median = cv2.medianBlur(thresh, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(median, kernel, iterations=1)
    return dilated

def get_polygon(slot):
    for key in ['polygon', 'points', 'coordinates', 'vertices', 'bbox']:
        if key in slot and slot[key]:
            return slot[key]
    return None

def main():
    print("=" * 60)
    print("      HỆ THỐNG XỬ LÝ ẢNH BÃI XE HÀNG LOẠT (KHÔNG DÙNG AI)      ")
    print("=" * 60)

    # 1. Nhập đường dẫn thư mục đầu vào từ người dùng
    while True:
        input_dir = input("👉 Nhập (hoặc kéo thả) đường dẫn thư mục chứa ảnh đầu vào: ").strip()
        # Loại bỏ dấu nháy kép/đơn ở đầu và cuối đường dẫn (khi người dùng kéo thả thư mục vào Terminal)
        input_dir = input_dir.strip('"').strip("'")
        
        if not input_dir:
            print("❌ Đường dẫn không được để trống. Vui lòng nhập lại!")
            continue
            
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ Thư mục không tồn tại: {input_path.resolve()}. Vui lòng kiểm tra lại!")
            continue
        break

    # 2. Cố định thư mục đầu ra tại TechGar/Output
    script_dir = Path(__file__).parent
    output_path = script_dir / "Output"

    # 3. Chọn file tọa độ ô đỗ
    print("\n📁 CHỌN FILE TỌA ĐỘ Ô ĐỖ:")
    print("   1 - parking_slots_polygon.json (Mặc định - 283 ô PKLot)")
    print("   2 - parking_slots_1.json       (File vẽ tay của bạn)")
    print("   3 - Nhập đường dẫn file JSON khác")
    
    choice_slots = input("👉 Nhập lựa chọn (Mặc định = 1): ").strip()
    if choice_slots == '2':
        slots_file = "parking_slots_1.json"
    elif choice_slots == '3':
        while True:
            slots_file = input("👉 Nhập đường dẫn file tọa độ JSON: ").strip().strip('"').strip("'")
            if Path(slots_file).exists():
                break
            print("❌ File không tồn tại! Vui lòng nhập lại.")
    else:
        slots_file = "parking_slots_polygon.json"

    # 4. Cấu hình tham số xử lý ảnh
    print("\n⚙️ CẤU HÌNH THAM SỐ:")
    print("   1 - Sử dụng tham số mặc định (Khuyên dùng: Threshold=20%, Gamma=1.6, Denoise=Bật, Max=1000)")
    print("   2 - Tùy chỉnh tham số thủ công")
    choice_params = input("👉 Nhập lựa chọn (Mặc định = 1): ").strip()

    gamma = 1.6
    clahe_clip = 3.0
    clahe_grid = 14
    threshold = 0.20
    denoise = 1
    max_images = 1000

    if choice_params == '2':
        try:
            val = input("👉 Hệ số Gamma (Enter = 1.6): ").strip()
            if val: gamma = float(val)

            val = input("👉 CLAHE Clip (Enter = 3.0): ").strip()
            if val: clahe_clip = float(val)

            val = input("👉 CLAHE Grid (Enter = 14): ").strip()
            if val: clahe_grid = int(val)

            val = input("👉 Ngưỡng Threshold (0.01 -> 1.0, Enter = 0.20): ").strip()
            if val: threshold = float(val)

            val = input("👉 Khử nhiễu (1: Bật, 0: Tắt, Enter = 1): ").strip()
            if val: denoise = int(val)

            val = input("👉 Số ảnh xử lý tối đa (Enter = 1000): ").strip()
            if val: max_images = int(val)
        except ValueError:
            print("⚠️ Nhập sai định dạng, tự động dùng tham số mặc định!")

    # Đọc dữ liệu tọa độ các ô đỗ xe
    slots_file_path = Path(slots_file)
    if not slots_file_path.exists():
        print(f"❌ Không tìm thấy file tọa độ: {slots_file_path.resolve()}")
        return

    try:
        with open(slots_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        raw_slots = data['slots']
        img_w_ref = data['imageWidth']
        img_h_ref = data['imageHeight']
    except Exception as e:
        print(f"❌ Lỗi đọc file tọa độ JSON: {e}")
        return

    # Lọc các ô đỗ xe có tọa độ đa giác hợp lệ
    slots = []
    for slot in raw_slots:
        poly = get_polygon(slot)
        if poly is not None:
            slot['_polygon'] = poly
            slots.append(slot)
    
    print(f"\n✅ Đã tải thành công {len(slots)} ô đỗ từ file: {slots_file_path.name}")
    print(f"   Độ phân giải ảnh tham chiếu gốc: {img_w_ref}x{img_h_ref}")

    # Lấy danh sách toàn bộ các file ảnh trong thư mục đầu vào
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = sorted([
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    total_images = len(all_images)
    if total_images == 0:
        print(f"❌ Không tìm thấy ảnh hợp lệ nào trong thư mục: {input_path.resolve()}")
        return

    # Giới hạn số lượng ảnh xử lý
    images_to_process = all_images[:max_images]
    num_to_process = len(images_to_process)
    print(f"📸 Tìm thấy tổng cộng {total_images} ảnh. Sẽ xử lý {num_to_process} ảnh đầu tiên.")

    # Tạo thư mục đầu ra chứa ảnh đã vẽ (TechGar/Output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Ảnh kết quả vẽ ô sẽ cố định lưu tại: {output_path.resolve()}\n")

    # Bắt đầu vòng lặp xử lý
    for idx, img_file in enumerate(images_to_process, 1):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠️ Bỏ qua file {img_file.name} (Không đọc được dữ liệu ảnh).")
            continue

        h, w = img.shape[:2]
        # Tính tỉ lệ co giãn (scale) nếu ảnh đầu vào khác kích thước ảnh lúc vẽ tọa độ
        sx = w / img_w_ref
        sy = h / img_h_ref

        # Chạy tiền xử lý ảnh
        img_pro = preprocess(img, gamma, clahe_clip, clahe_grid, denoise)
        
        free_count = 0
        img_draw = img.copy()

        for slot in slots:
            poly = slot['_polygon']
            try:
                # Đọc tọa độ điểm của ô đỗ
                if isinstance(poly[0], dict):
                    pts = np.array([[int(p['x'] * sx), int(p['y'] * sy)] for p in poly], np.int32)
                else:
                    pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in poly], np.int32)
            except Exception:
                continue

            # Tính toán tỉ lệ pixel trắng trong đa giác
            mask = np.zeros(img_pro.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            masked = cv2.bitwise_and(img_pro, img_pro, mask=mask)
            count = cv2.countNonZero(masked)
            area = cv2.contourArea(pts)
            ratio = count / area if area > 0 else 0

            # Phân loại trạng thái dựa trên ngưỡng threshold
            is_free = ratio < threshold
            color = (0, 255, 0) if is_free else (0, 0, 255) # Xanh lá = TRỐNG, Đỏ = CÓ XE
            if is_free:
                free_count += 1

            # Vẽ đa giác ô đỗ
            cv2.polylines(img_draw, [pts], True, color, 2)
            
            # Vẽ ID của ô đỗ lên tâm ô
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(img_draw, slot['id'], (cx - 12, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Ghi các thông tin tổng kết lên góc trái ảnh
        total_slots = len(slots)
        cv2.putText(img_draw, f'Trong: {free_count}/{total_slots}', (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(img_draw, f'Thr={threshold:.0%}', (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        # Lưu ảnh kết quả ra thư mục mới với tên đặt theo số thứ tự (ví dụ: 1.jpg, 2.jpg,...)
        out_file_name = f"{idx}{img_file.suffix}"
        out_file_path = output_path / out_file_name
        cv2.imwrite(str(out_file_path), img_draw)

        # Log tiến độ xử lý
        print(f"[{idx}/{num_to_process}] Đã vẽ xong {img_file.name} -> Lưu thành: {out_file_name} -> Số ô trống: {free_count}/{total_slots}")

    print(f"\n🎉 HOÀN THÀNH! Đã lưu {num_to_process} ảnh kết quả vào thư mục: {output_path.resolve()}")

if __name__ == "__main__":
    main()
