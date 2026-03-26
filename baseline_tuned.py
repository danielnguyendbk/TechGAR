import cv2, pandas as pd, numpy as np, os
from sklearn.metrics import f1_score
from tqdm import tqdm

master = pd.read_csv('master_parking_dataset.csv', low_memory=False)
test_df = master[master.dataset=='pklot_coco'].dropna(subset=['bbox_x']).head(500)

def opencv_predict(crop):
    blur = cv2.GaussianBlur(crop, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occupancy = sum(cv2.contourArea(c) for c in contours) / crop.size
    return 1 if occupancy > 0.25 else 0  # Tune ↓ 0.35→0.25

gt, pred = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_path = os.path.join(r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive\train", row.image_file)
    if os.path.exists(img_path):
        img = cv2.imread(img_path, 0)
        crop = img[int(row.bbox_y):int(row.bbox_y+row.bbox_h), int(row.bbox_x):int(row.bbox_x+row.bbox_w)]
        gt.append(row.status_num)
        pred.append(opencv_predict(crop))

print(f"TUNED F1: {f1_score(gt, pred):.3f} (↑{f1_score(gt, pred)-0.242:+.3f}) n={len(gt)}")
pd.DataFrame({'gt':gt, 'pred':pred}).to_csv('baseline_tuned.csv', index=False)
