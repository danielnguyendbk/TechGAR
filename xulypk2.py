import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np

dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive1\PKLot\PKLot"  # Thay đường dẫn

print("="*60)
print("BƯỚC 1: KIỂM TRA XML FOLDER")
print("="*60)

# Ví dụ folder (thay bằng folder cụ thể)
xml_folder = os.path.join(dataset_path, "PUCPR/Cloudy/2012-09-12")  # Input folder
print(f"Folder: {xml_folder}")
print(f"Tồn tại: {os.path.exists(xml_folder)}")
print(f"Số XML: {len([f for f in os.listdir(xml_folder) if f.endswith('.xml')])}")

input("\nNhấn Enter...")

print("\nBƯỚC 2: PARSE XML")
data = []
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        img_name = xml_file.replace('.xml', '.jpg')
        
        img_path = os.path.join(xml_folder, img_name)
        if os.path.exists(img_path):
            # Read shape only (can be cached if we want, but doing it per image is fine for a single folder)
            img = cv2.imread(img_path)
            width, height = (img.shape[1], img.shape[0]) if img is not None else (1280, 720)
        else:
            width, height = 1280, 720
            
        for space in root.findall('space'):
            occupied = space.get('occupied')
            label = 'Occupied' if occupied == '1' else 'Empty'
            
            contour = space.find('contour')
            if contour is not None:
                xs = [int(p.get('x')) for p in contour.findall('point')]
                ys = [int(p.get('y')) for p in contour.findall('point')]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                bbox = [0, 0, 0, 0]
                
            data.append({'img': img_name, 'label': label, 'bbox': bbox, 'w': width, 'h': height})

df = pd.DataFrame(data)
print(f"Parse OK: {len(df)} spots")
print("Label dist:", df['label'].value_counts(normalize=True))

input("\nNhấn Enter...")

print("\nBƯỚC 3: STATS")
print(f"Kích thước ảnh: {df['w'].iloc[0]}x{df['h'].iloc[0]} (uniform)")
bbox_stats = np.array([d['bbox'] for d in data])
print(f"Bbox W: {bbox_stats[:,2].min():.0f}-{bbox_stats[:,2].max():.0f} mean={bbox_stats[:,2].mean():.0f}")
print(f"Spots/ảnh: min={df.groupby('img').size().min()}, max={df.groupby('img').size().max()}")

input("\nNhấn Enter...")

print("\nBƯỚC 4: VISUALIZE")
img_path = input("Nhập đường dẫn ảnh tương ứng (cùng folder XML): ")
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
img_anns = df[df['img']==os.path.basename(img_path)]

colors = {'Empty': (0,255,0), 'Occupied': (0,0,255)}
for _, ann in img_anns.iterrows():
    x1,y1,x2,y2 = ann['bbox']
    cv2.rectangle(img, (x1,y1), (x2,y2), colors[ann['label']], 2)
    cv2.putText(img, ann['label'], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[ann['label']], 2)

plt.figure(figsize=(12,8))
plt.imshow(img)
plt.title(f"PKLot XML: {len(img_anns)} spots")
plt.axis('off')
plt.show()

df.to_csv('pklot_xml_stats.csv', index=False)
print("Lưu CSV OK!")

