import json
import os
import cv2
import matplotlib.pyplot as plt
import random

# === ĐƯỜNG DẪN ===
dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive"
json_path = os.path.join(dataset_path, "train", "_annotations.coco.json")

print("="*60)
print("BƯỚC 3.5: VẼ BOUNDING BOX LÊN ẢNH")
print("="*60)

# Đọc file JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Tạo mapping category
cat_names = {cat['id']: cat['name'] for cat in data['categories']}
print(f"Categories: {cat_names}")

# Nhập đường dẫn ảnh từ người dùng
while True:
    image_path = input("Nhập đường dẫn tuyệt đối đến ảnh (trong dataset/train): ").strip()
    if not os.path.exists(image_path):
        print("Đường dẫn không tồn tại. Vui lòng thử lại.")
        continue
    image_file = os.path.basename(image_path)
    # Tìm image trong data['images']
    image_info = next((img for img in data['images'] if img['file_name'] == image_file), None)
    if image_info is None:
        print("Ảnh không có trong dataset. Vui lòng chọn ảnh khác.")
        continue
    image_id = image_info['id']
    break

print(f"\nChọn ảnh:")
print(f"  - ID: {image_id}")
print(f"  - File: {image_file}")
print(f"  - Path: {image_path}")

# Lấy annotations của ảnh này
image_anns = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
print(f"  - Số ô đỗ trong ảnh: {len(image_anns)}")

# Đếm số ô trống và có xe
empty_count = sum(1 for ann in image_anns if ann['category_id'] == 1)
occupied_count = sum(1 for ann in image_anns if ann['category_id'] == 2)
print(f"  - Ô trống (empty): {empty_count}")
print(f"  - Ô có xe (occupied): {occupied_count}")

# Đọc ảnh
img = cv2.imread(image_path)
if img is None:
    print("LỖI: Không thể đọc ảnh! Kiểm tra đường dẫn.")
    exit()

# Chuyển từ BGR sang RGB để hiển thị đúng màu
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Vẽ từng ô đỗ
for ann in image_anns:
    cat_id = ann['category_id']
    x, y, w, h = [int(v) for v in ann['bbox']]
    
    # Chọn màu: Xanh cho empty (id=1), Đỏ cho occupied (id=2)
    if cat_id == 1:  # empty
        color = (0, 255, 0)  # Xanh lá
        label = "empty"
    else:  # occupied
        color = (255, 0, 0)  # Đỏ
        label = "occupied"
    
    # Vẽ hình chữ nhật
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Hiển thị ảnh
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.title(f"Ảnh: {image_file}\nXanh: trống, Đỏ: có xe", fontsize=14)
plt.axis('off')
plt.show()

print("\nĐã hiển thị ảnh! Đóng cửa sổ ảnh để kết thúc.")