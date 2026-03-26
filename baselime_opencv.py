import cv2, pandas as pd, numpy as np, os
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

master = pd.read_csv('master_parking_dataset.csv')
# Test PKLot COCO (có bbox đầy đủ)
test_df = master[master.dataset=='pklot_coco'].dropna(subset=['bbox_x']).head(200).reset_index(drop=True)

gt, pred, times = [], [], []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    img_path = os.path.join(r"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive\train", row.image_file)
    if os.path.exists(img_path):
        img = cv2.imread(img_path, 0)  # Grayscale
        y1,x1 = int(row.bbox_y), int(row.bbox_x)
        y2,x2 = y1+int(row.bbox_h), x1+int(row.bbox_w)
        crop = img[y1:y2, x1:x2]
        
        t0 = cv2.getTickCount()
        # **OpenCV Pipeline** (Giai đoạn 1 TechGAR)
        blur = cv2.GaussianBlur(crop, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        occupancy = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c)>crop.size*0.1) / crop.size
        time_ms = (cv2.getTickCount() - t0)/cv2.getTickFrequency()*1000
        
        gt.append(row.status_num)
        pred.append(1 if occupancy > 0.35 else 0)  # Tune threshold
        times.append(time_ms)

print(f"✅ BASELINE OpenCV (n={len(gt)}):")
print(f"F1: {f1_score(gt, pred):.3f} | Accuracy: {accuracy_score(gt, pred):.3f}")
print(f"Time: {np.mean(times):.1f}ms/frame (FPS: {1000/np.mean(times):.0f})")
print("Lưu kết quả:", pd.DataFrame({'gt':gt, 'pred':pred}).to_csv('baseline_results.csv'))
