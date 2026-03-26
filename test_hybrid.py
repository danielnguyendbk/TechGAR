import cv2, tensorflow as tf, numpy as np, pandas as pd, time, os
from sklearn.metrics import f1_score
print("✅ Load CNN:", tf.keras.models.load_model('cnn_parking.h5'))

master = pd.read_csv('master_parking_dataset.csv', low_memory=False)
test_df = master[master.dataset=='pklot_coco'].dropna(subset=['bbox_x']).head(100)

def hybrid_predict(model, row):
    img_path = rf"D:\N23DCCN155- Dang Van Hiep\HK6\NCKH\dataset\archive\train\{row.image_file}"
    if not os.path.exists(img_path): return None
    
    img = cv2.imread(img_path, 0)
    crop = img[int(row.bbox_y):int(row.bbox_y+row.bbox_h), int(row.bbox_x):int(row.bbox_x+row.bbox_w)]
    
    # OpenCV filter (99% fast empty)
    thresh = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    occupancy = cv2.countNonZero(thresh) / crop.size
    if occupancy < 0.15: return 0  # Empty fast
    
    # CNN heavy (1% motion)
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (128,128)), cv2.COLOR_GRAY2RGB) / 255.0
    pred = model.predict(np.expand_dims(crop_rgb, 0), verbose=0)[0][0]
    return 1 if pred > 0.5 else 0

model = tf.keras.models.load_model('cnn_parking.h5')
gt, pred, times = [], [], []
for _, row in test_df.iterrows():
    t0 = time.time()
    p = hybrid_predict(model, row)
    if p is not None:
        gt.append(row.status_num); pred.append(p); times.append((time.time()-t0)*1000)

print(f"HYBRID F1: {f1_score(gt, pred):.3f} | Time: {np.mean(times):.1f}ms | n={len(gt)}")
pd.DataFrame({'gt':gt,'pred':pred}).to_csv('hybrid_results.csv')
