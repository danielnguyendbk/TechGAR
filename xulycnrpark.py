import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

dataset_path = r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\cnrpark\CNRPark-Patches-150x150"

print("="*60)
print("BƯỚC 1: SCAN FOLDERS")
print("="*60)

data = []
for cam in ['A', 'B']:
    busy_dir = os.path.join(dataset_path, cam, 'busy')
    free_dir = os.path.join(dataset_path, cam, 'free')
    busy_files = len(os.listdir(busy_dir)) if os.path.exists(busy_dir) else 0
    free_files = len(os.listdir(free_dir)) if os.path.exists(free_dir) else 0
    print(f"Cam {cam}: busy={busy_files}, free={free_files}")
    # List files for DF
    for f in os.listdir(busy_dir): data.append({'cam': cam, 'status': 'busy', 'file': f})
    for f in os.listdir(free_dir): data.append({'cam': cam, 'status': 'free', 'file': f})

df = pd.DataFrame(data)
print(f"Tổng: {len(df)} patches")
print("Dist:", df['status'].value_counts(normalize=True))  # ~33% free[file:1]

input("\nNhấn Enter...")

print("\nBƯỚC 2: STATS")
print("Kích thước: 150x150 uniform")
print("File naming: YYYYMMDD_HHMM_SLOTID.jpg → thời gian + slot")
df['time'] = df['file'].str.extract(r'(\d{4}\d{2}\d{2}_\d{4})')
print("Thời gian mẫu:", df['time'].unique()[:5])

input("\nNhấn Enter...")

print("\nBƯỚC 3: VISUALIZE SAMPLES")
fig, axs = plt.subplots(2, 4, figsize=(15,8))
samples = df.sample(frac=1, random_state=42).groupby(['cam', 'status']).head(2).reset_index(drop=True)
for i, (_, row) in enumerate(samples.iterrows()):
    ax = axs[(i//4), (i%4)]
    img_path = os.path.join(dataset_path, row['cam'], row['status'], row['file'])
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(f"{row['cam']}-{row['status']}")
    ax.axis('off')
plt.suptitle("CNRPark Samples (33% free)")
plt.tight_layout()
plt.show()

df.to_csv('cnrpark_stats.csv', index=False)
print("Lưu CSV OK!")

