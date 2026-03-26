import cv2, pandas as pd, os, numpy as np
from sklearn.model_selection import train_test_split

master = pd.read_csv('master_parking_dataset.csv', low_memory=False)
coco_df = master[master.dataset=='pklot_coco'].dropna(subset=['bbox_x'])

# Resize uniform 128x128 cho CNN
for path in ['crops/train/0', 'crops/train/1', 'crops/val/0', 'crops/val/1']:
    os.makedirs(path, exist_ok=True)

train_df, val_df = train_test_split(coco_df, test_size=0.2, stratify=coco_df.status_num, random_state=42)
for split, df in [('train', train_df), ('val', val_df)]:
    
    count = 0
    for _, row in df.iterrows():
        img_path = os.path.join(r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive\train", row.image_file)
        if os.path.exists(img_path) and count < 5000:  # 5k/split
            img = cv2.imread(img_path)
            crop = img[int(row.bbox_y):int(row.bbox_y+row.bbox_h), int(row.bbox_x):int(row.bbox_x+row.bbox_w)]
            crop = cv2.resize(crop, (128,128))  # CNN input
            cv2.imwrite(f"crops/{split}/{int(row.status_num)}/crop_{count}.jpg", crop)
            count += 1
    
    print(f"{split}: {count} crops ({df.status_num.value_counts(normalize=True).to_dict()})")

print("✅ 10k crops ready train CNN!")
