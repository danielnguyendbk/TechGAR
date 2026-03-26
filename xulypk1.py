import json
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd

dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive"

json_path = os.path.join(dataset_path, "train", "_annotations.coco.json")

print("="*60)
print("BƯỚC 1: KIỂM TRA ĐƯỜNG DẪN")
print("="*60)
print(f"Đường dẫn file JSON: {json_path}")
print(f"File tồn tại? {os.path.exists(json_path)}")

# Nếu file không tồn tại, dừng lại
if not os.path.exists(json_path):
    print("LỖI: Không tìm thấy file JSON! Kiểm tra lại đường dẫn.")
    exit()
input("\nNhấn Enter để tiếp tục...")
import json
import os
from collections import Counter

# === ĐƯỜNG DẪN ===
dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive"
json_path = os.path.join(dataset_path, "train", "_annotations.coco.json")

print("="*60)
print("BƯỚC 2: ĐỌC FILE JSON")
print("="*60)

# Đọc file JSON
with open(json_path, 'r') as f:
    data = json.load(f)

print("ĐỌC FILE THÀNH CÔNG!")
print(f"\n1. CẤU TRÚC CƠ BẢN:")
print(f"   - Số keys trong file: {list(data.keys())}")
print(f"   - Số images: {len(data['images'])}")
print(f"   - Số annotations: {len(data['annotations'])}")
print(f"   - Số categories: {len(data['categories'])}")

# Xem categories
print(f"\n2. CATEGORIES:")
for cat in data['categories']:
    print(f"   - id={cat['id']}: {cat['name']}")

input("\nNhấn Enter để tiếp tục...")
print(f"\n3. PHÂN TÍCH ẢNH (IMAGES):")

# Lấy ảnh đầu tiên
first_img = data['images'][0]
print(f"   - Ảnh đầu tiên:")
print(f"     * ID: {first_img['id']}")
print(f"     * Tên file: {first_img['file_name']}")
print(f"     * Kích thước: {first_img['width']}x{first_img['height']}")

# Phân tích kích thước ảnh (kiểm tra xem tất cả có cùng kích thước không)
widths = [img['width'] for img in data['images']]
heights = [img['height'] for img in data['images']]

print(f"\n   - Thống kê kích thước:")
print(f"     * Chiều rộng: min={min(widths)}, max={max(widths)}, unique={len(set(widths))}")
print(f"     * Chiều cao: min={min(heights)}, max={max(heights)}, unique={len(set(heights))}")

# Phân tích thông tin thời tiết từ tên file
weather_count = {'sunny': 0, 'cloudy': 0, 'rainy': 0, 'other': 0}
for img in data['images'][:100]:  # Chỉ kiểm tra 100 ảnh đầu
    filename = img['file_name'].lower()
    if 'sunny' in filename:
        weather_count['sunny'] += 1
    elif 'cloudy' in filename:
        weather_count['cloudy'] += 1
    elif 'rainy' in filename:
        weather_count['rainy'] += 1
    else:
        weather_count['other'] += 1

print(f"\n   - Thời tiết (100 ảnh đầu):")
for weather, count in weather_count.items():
    print(f"     * {weather}: {count} ảnh")
    
input("\nNhấn Enter để tiếp tục...")
# === TIẾP TỤC THÊM VÀO FILE ===
print(f"\n4. PHÂN TÍCH ANNOTATIONS (Ô ĐỖ):")

# Đếm số lượng theo category
cat_counts = Counter()
for ann in data['annotations']:
    cat_counts[ann['category_id']] += 1

print(f"\n   - Phân bố theo category:")
total = len(data['annotations'])
for cat_id, count in cat_counts.items():
    cat_name = next((c['name'] for c in data['categories'] if c['id'] == cat_id), 'unknown')
    print(f"     * {cat_name} (id={cat_id}): {count} ô ({count/total*100:.2f}%)")

# Phân tích bounding box
print(f"\n   - Thống kê bounding box (1000 ô đầu):")
bbox_widths = []
bbox_heights = []
bbox_areas = []

for ann in data['annotations'][:1000]:  # Chỉ lấy 1000 ô để phân tích nhanh
    x, y, w, h = ann['bbox']
    bbox_widths.append(w)
    bbox_heights.append(h)
    bbox_areas.append(w * h)

print(f"     * Chiều rộng: min={min(bbox_widths):.1f}, max={max(bbox_widths):.1f}, mean={sum(bbox_widths)/len(bbox_widths):.1f}")
print(f"     * Chiều cao: min={min(bbox_heights):.1f}, max={max(bbox_heights):.1f}, mean={sum(bbox_heights)/len(bbox_heights):.1f}")
print(f"     * Diện tích: min={min(bbox_areas):.1f}, max={max(bbox_areas):.1f}, mean={sum(bbox_areas)/len(bbox_areas):.1f}")

# Phân tích số lượng ô đỗ trên mỗi ảnh
anns_per_image = Counter()
for ann in data['annotations']:
    anns_per_image[ann['image_id']] += 1

counts = list(anns_per_image.values())
print(f"\n   - Số ô đỗ trên mỗi ảnh:")
print(f"     * Min: {min(counts)} ô/ảnh")
print(f"     * Max: {max(counts)} ô/ảnh")
print(f"     * Trung bình: {sum(counts)/len(counts):.2f} ô/ảnh")

# === ĐƯỜNG DẪN ===
dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive"
json_path = os.path.join(dataset_path, "train", "_annotations.coco.json")

print("="*60)
print("BƯỚC 4: VẼ BOUNDING BOX LÊN ẢNH")
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

df = pd.DataFrame([{
    'image_id': ann['image_id'],
    'image_file': next(img['file_name'] for img in data['images'] if img['id']==ann['image_id']),
    'status': cat_names[ann['category_id']],
    'status_num': 0 if ann['category_id']==1 else 1,  # 0=empty, 1=occupied
    'bbox_x': ann['bbox'][0], 'bbox_y': ann['bbox'][1],
    'bbox_w': ann['bbox'][2], 'bbox_h': ann['bbox'][3],
    'area': ann['area']
} for ann in data['annotations']])
df.to_csv('pklot_coco_train.csv', index=False)
print(f"\n TẠO CSV: pklot_coco_train.csv ({len(df)} spots)")
print(df['status'].value_counts(normalize=True))
