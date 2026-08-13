"""
ensemble_test.py – Multi-Parameter Ensemble Thresholding
Thuật toán bỏ phiếu 36 tổ hợp (6 Gamma x 6 CLAHE) cho độ chính xác cực cao.
"""
import cv2
import json
import math
import numpy as np
import time
import os

# ══════════════════════════════════════════════
#  CẤU HÌNH
# ══════════════════════════════════════════════
SLOTS_FILE    = 'parking_slots_2.json'
OUTPUT_JSON   = 'parking_status.json'
UPDATE_EVERY  = 1.0

# ══════════════════════════════════════════════
#  LOAD TỌA ĐỘ VÀ TIỀN XỬ LÝ ROI ĐỂ TỐI ƯU TỐC ĐỘ
# ══════════════════════════════════════════════
try:
    with open(SLOTS_FILE, 'r') as f:
        data = json.load(f)
    slots_data = data['slots']
    IMG_W_REF  = data['imageWidth']
    IMG_H_REF  = data['imageHeight']
except FileNotFoundError:
    print(f"❌ Không tìm thấy {SLOTS_FILE}")
    exit()

def get_polygon(slot):
    for key in ['polygon', 'points', 'coordinates', 'vertices']:
        if key in slot and slot[key]:
            return slot[key]
    if 'rect' in slot and slot['rect']:
        r = slot['rect']
        cx, cy, w, h = r['cx'], r['cy'], r['w'], r['h']
        angle = r.get('angle', 0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        hw, hh = w / 2, h / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [{'x': cx + dx * cos_a - dy * sin_a, 'y': cy + dx * sin_a + dy * cos_a} for dx, dy in corners]
    return None

slots = []
for s in slots_data:
    poly = get_polygon(s)
    if poly:
        s['_polygon'] = poly
        slots.append(s)

print(f"✅ Load thành công {len(slots)} ô đỗ xe")

# ══════════════════════════════════════════════
#  PRE-COMPUTE ROIs (Tính trước bounding box và mask cho từng ô)
# ══════════════════════════════════════════════
# Sẽ được tính lại trong lần detect đầu tiên khi biết kích thước ảnh thực tế
precomputed_rois = []

def compute_rois(img_shape):
    global precomputed_rois
    precomputed_rois = []
    h, w = img_shape[:2]
    sx = w / IMG_W_REF
    sy = h / IMG_H_REF

    for slot in slots:
        poly = slot['_polygon']
        try:
            if isinstance(poly[0], dict):
                pts = np.array([[int(p['x'] * sx), int(p['y'] * sy)] for p in poly], np.int32)
            else:
                pts = np.array([[int(p[0] * sx), int(p[1] * sy)] for p in poly], np.int32)
        except Exception:
            precomputed_rois.append(None)
            continue

        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
        
        # Thêm padding nhẹ để tránh lỗi out of bounds
        x_min = max(0, x_min - 2)
        y_min = max(0, y_min - 2)
        x_max = min(w, x_max + 2)
        y_max = min(h, y_max + 2)

        # Tạo local mask cho ROI này
        roi_w = x_max - x_min
        roi_h = y_max - y_min
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        
        # Dịch chuyển tọa độ polygon về hệ tọa độ của local mask
        local_pts = pts.copy()
        local_pts[:, 0] -= x_min
        local_pts[:, 1] -= y_min
        cv2.fillPoly(local_mask, [local_pts], 255)
        
        area = cv2.countNonZero(local_mask)
        
        precomputed_rois.append({
            'pts': pts,
            'bbox': (x_min, y_min, x_max, y_max),
            'mask': local_mask,
            'area': area,
            'id': slot['id']
        })

# ══════════════════════════════════════════════
#  GAMMA LUT CACHE
# ══════════════════════════════════════════════
_lut_cache = {}
def get_gamma_lut(gamma):
    gamma = round(gamma, 1)
    if gamma not in _lut_cache:
        _lut_cache[gamma] = np.array([np.clip(pow(i / 255.0, 1.0 / gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    return _lut_cache[gamma]

# ══════════════════════════════════════════════
#  TRACKBAR
# ══════════════════════════════════════════════
cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Settings', 400, 250)
cv2.createTrackbar('Base Gamma x10',  'Settings', 28, 40,  lambda x: None)
cv2.createTrackbar('Base CLAHE x10',  'Settings', 20, 60,  lambda x: None)
cv2.createTrackbar('CLAHE Grid',      'Settings', 8, 16,  lambda x: None)
cv2.createTrackbar('Threshold',       'Settings', 20, 100, lambda x: None)
cv2.createTrackbar('Pass2 Edge Thr', 'Settings', 25, 100, lambda x: None)

def get_params():
    try:
        if cv2.getWindowProperty('Settings', cv2.WND_PROP_VISIBLE) < 1:
            return 1.6, 3.0, 14, 0.20, 0.25
        gamma = max(cv2.getTrackbarPos('Base Gamma x10', 'Settings') / 10.0, 0.1)
        clahe = max(cv2.getTrackbarPos('Base CLAHE x10', 'Settings') / 10.0, 0.1)
        grid  = max(cv2.getTrackbarPos('CLAHE Grid', 'Settings'), 2)
        thr   = max(cv2.getTrackbarPos('Threshold', 'Settings') / 100.0, 0.01)
        edge_thr = max(cv2.getTrackbarPos('Pass2 Edge Thr', 'Settings') / 100.0, 0.01)
        return gamma, clahe, grid, thr, edge_thr
    except cv2.error:
        return 1.6, 3.0, 14, 0.20, 0.25

# ══════════════════════════════════════════════
#  TEMPORAL SMOOTHING & JSON SAVER
# ══════════════════════════════════════════════
class TemporalSmoother:
    def __init__(self, num_slots, required_frames=5):
        self.required = required_frames
        self.counters = [0] * num_slots
        self.pending = [None] * num_slots
        self.confirmed = [False] * num_slots

    def update(self, slot_idx, is_occupied):
        if self.pending[slot_idx] == is_occupied:
            self.counters[slot_idx] += 1
        else:
            self.pending[slot_idx] = is_occupied
            self.counters[slot_idx] = 1
        if self.counters[slot_idx] >= self.required:
            self.confirmed[slot_idx] = is_occupied
        return self.confirmed[slot_idx]

def save_status_json(slot_results):
    free_count = sum(1 for s in slot_results if not s['occupied'])
    occupied_count = sum(1 for s in slot_results if s['occupied'])
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "ensemble",
        "total": len(slot_results),
        "free": free_count,
        "occupied": occupied_count,
        "slots": {
            s['id']: {"occupied": s['occupied'], "status": "occupied" if s['occupied'] else "empty"}
            for s in slot_results
        }
    }
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except PermissionError:
        pass

smoother = None

# ══════════════════════════════════════════════
#  CORE ENSEMBLE DETECT
# ══════════════════════════════════════════════
def detect_ensemble(img, apply_smoothing=False):
    global smoother
    if not precomputed_rois:
        compute_rois(img.shape)
        smoother = TemporalSmoother(len(precomputed_rois), required_frames=5)

    base_gamma, base_clahe, clahe_grid, ratio_thr, edge_thr = get_params()
    
    delta_gamma = [-0.2, -0.1, 0.0, 0.1, 0.2]
    delta_clahe = [-0.5, -0.2, 0.0, 0.2, 0.5]
    
    # 1. Chuyển LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    kernel = np.ones((3, 3), np.uint8)
    
    # Khởi tạo mảng đếm phiếu (vote TRỐNG) cho mỗi ô
    empty_votes = [0] * len(precomputed_rois)
    total_combinations = len(delta_clahe) * len(delta_gamma)
    
    # Lưu ảnh threshold của bộ Base (delta=0,0) để debug
    base_dilated = None
    
    # 2. Sinh 25 biến thể và chấm điểm ngay
    for dc in delta_clahe:
        c_val = max(0.1, base_clahe + dc)
        clahe = cv2.createCLAHE(clipLimit=c_val, tileGridSize=(clahe_grid, clahe_grid))
        l_clahe = clahe.apply(l_channel)
        
        for dg in delta_gamma:
            g_val = max(0.1, base_gamma + dg)
            lut = get_gamma_lut(g_val)
            l_gamma = cv2.LUT(l_clahe, lut)
            
            blur = cv2.GaussianBlur(l_gamma, (3, 3), 1)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            median = cv2.medianBlur(thresh, 5)
            dilated = cv2.dilate(median, kernel, iterations=1)
            
            if dc == 0.0 and dg == 0.0:
                base_dilated = dilated.copy()
            
            for i, roi in enumerate(precomputed_rois):
                if roi is None or roi['area'] == 0:
                    continue
                x1, y1, x2, y2 = roi['bbox']
                roi_thresh = dilated[y1:y2, x1:x2]
                masked = cv2.bitwise_and(roi_thresh, roi_thresh, mask=roi['mask'])
                count = cv2.countNonZero(masked)
                ratio = count / roi['area']
                if ratio < ratio_thr:
                    empty_votes[i] += 1

    # ══════════════════════════════════════════════
    #  PASS 1: Threshold + Center Cluster
    # ══════════════════════════════════════════════
    required_votes = total_combinations // 2
    is_free_list = [True] * len(precomputed_rois)
    rescued_center = 0
    
    for i, roi in enumerate(precomputed_rois):
        if roi is None:
            continue
        
        is_free = empty_votes[i] >= required_votes
        
        # ── CENTER CLUSTER RESCUE ──
        if is_free and base_dilated is not None and roi['area'] > 0:
            x1, y1, x2, y2 = roi['bbox']
            
            center_mask = np.zeros_like(roi['mask'])
            pts_local = roi['pts'].copy()
            pts_local[:, 0] -= x1
            pts_local[:, 1] -= y1
            pts_f = pts_local.astype(np.float32)
            centroid_x = np.mean(pts_f[:, 0])
            centroid_y = np.mean(pts_f[:, 1])
            shrink = 0.4
            center_pts = np.array([
                [int(centroid_x + (px - centroid_x) * shrink),
                 int(centroid_y + (py - centroid_y) * shrink)]
                for px, py in pts_f
            ], dtype=np.int32)
            cv2.fillPoly(center_mask, [center_pts], 255)
            center_area = cv2.countNonZero(center_mask)
            
            if center_area > 5:
                roi_t = base_dilated[y1:y2, x1:x2]
                full_masked = cv2.bitwise_and(roi_t, roi_t, mask=roi['mask'])
                full_ratio = cv2.countNonZero(full_masked) / roi['area']
                center_masked = cv2.bitwise_and(roi_t, roi_t, mask=center_mask)
                center_ratio = cv2.countNonZero(center_masked) / center_area
                
                if (center_ratio >= 0.05 and 
                    (full_ratio < 0.01 or center_ratio > full_ratio * 2)):
                    is_free = False
                    rescued_center += 1
        
        is_free_list[i] = is_free

    # ══════════════════════════════════════════════
    #  PASS 2: Edge Recheck (chỉ kiểm tra ô vẫn xanh)
    # ══════════════════════════════════════════════
    EDGE_THR = edge_thr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 1)
    edges = cv2.Canny(gray_blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    edge_ratios = [0.0] * len(precomputed_rois)
    rescued_edge = 0
    for i, roi in enumerate(precomputed_rois):
        if roi is None or roi['area'] == 0:
            continue
        
        x1, y1, x2, y2 = roi['bbox']
        roi_edges = edges[y1:y2, x1:x2]
        masked_e = cv2.bitwise_and(roi_edges, roi_edges, mask=roi['mask'])
        edge_ratio = cv2.countNonZero(masked_e) / roi['area']
        edge_ratios[i] = edge_ratio
        
        # Chỉ cứu ô đang là free ở Pass 1
        if is_free_list[i] and edge_ratio >= EDGE_THR:
            is_free_list[i] = False
            rescued_edge += 1

    # ══════════════════════════════════════════════
    #  VẼ KẾT QUẢ
    # ══════════════════════════════════════════════
    free_count = 0
    slot_results = []
    img_draw = img.copy()
    
    if base_dilated is not None:
        debug_img = cv2.cvtColor(base_dilated, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = np.zeros_like(img)
        
    debug_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    for i, roi in enumerate(precomputed_rois):
        if roi is None:
            continue
        
        is_free = is_free_list[i]
        
        if apply_smoothing and smoother is not None:
            is_free = not smoother.update(i, not is_free)

        if is_free:
            free_count += 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            
        slot_results.append({
            "id": roi['id'],
            "occupied": not is_free
        })
        
        pts = roi['pts']
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        
        cv2.polylines(img_draw, [pts], True, color, 2)
        cv2.putText(img_draw, roi['id'], (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Debug threshold
        vote_pct = empty_votes[i] / total_combinations
        cv2.polylines(debug_img, [pts], True, color, 2)
        if base_dilated is not None and roi['area'] > 0:
            x1, y1, x2, y2 = roi['bbox']
            roi_t = base_dilated[y1:y2, x1:x2]
            m = cv2.bitwise_and(roi_t, roi_t, mask=roi['mask'])
            pix_ratio = cv2.countNonZero(m) / roi['area']
            info = f"{vote_pct:.0%}|{pix_ratio:.0%}"
            cv2.putText(debug_img, info, (cx - 18, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

        # Debug edge
        cv2.polylines(debug_edge, [pts], True, color, 2)
        edge_info = f"{edge_ratios[i]:.0%}"
        cv2.putText(debug_edge, edge_info, (cx - 10, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)

    total = len([r for r in precomputed_rois if r is not None])
    cv2.putText(img_draw, f'Trong: {free_count}/{total} [ENSEMBLE 25]', 
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 0), 3)
    cv2.putText(img_draw, f'G={base_gamma:.1f} CLAHE={base_clahe:.1f} Thr={ratio_thr:.0%} | Center:{rescued_center} EdgePass2({EDGE_THR:.0%}):{rescued_edge}', 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
    
    cv2.putText(debug_img, f'THRESHOLD (G={base_gamma:.1f} CLAHE={base_clahe:.1f}) | Vote%|Pixel%',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    cv2.putText(debug_edge, f'DEBUG EDGE (Pass 2 Thr >= {EDGE_THR:.0%}) | Edge%',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    return img_draw, slot_results, debug_img, debug_edge

# ══════════════════════════════════════════════
#  CHẠY CHÍNH
# ══════════════════════════════════════════════
print("Chọn nguồn:")
print("1 - Video")
print("2 - Ảnh tĩnh")
print("3 - Camera trực tiếp")
choice = input("Nhập (1/2/3, Enter=2): ").strip() or '2'

if choice == '1':
    source = input("Đường dẫn video (carPark.mp4): ").strip() or 'carPark.mp4'
    cap = cv2.VideoCapture(source)
    use_image = False
elif choice == '2':
    source = input("Đường dẫn ảnh (parkingimg.jpg): ").strip() or 'parkingimg.jpg'
    use_image = True
else:
    cam_id = input("Camera ID (Enter=0): ").strip() or '0'
    cap = cv2.VideoCapture(int(cam_id))
    use_image = False

if use_image:
    img_src = cv2.imread(source)
    if img_src is None:
        print(f"❌ Không đọc được ảnh: {source}")
        exit()
    
    img_src = cv2.resize(img_src, (IMG_W_REF, IMG_H_REF))
    print("⏳ Đang xử lý Ensemble (25 biến thể)...")
    while True:
        start_time = time.time()
        result, slot_results, debug_img, debug_edge = detect_ensemble(img_src, apply_smoothing=False)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(result, f'FPS: {fps:.1f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        
        cv2.imshow('Parking', result)
        cv2.imshow('Debug Threshold', debug_img)
        cv2.imshow('Debug Edge', debug_edge)
        key = cv2.waitKey(100) & 0xFF
        if key != 255: break
        try:
            if cv2.getWindowProperty('Parking', cv2.WND_PROP_VISIBLE) < 1: break
        except cv2.error: break
else:
    print("Đang chạy Video/Cam... Nhấn Q để thoát")
    last_save_time = time.time()
    
    while True:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        img = cv2.resize(img, (IMG_W_REF, IMG_H_REF))
        start_time = time.time()
        result, slot_results, debug_img, debug_edge = detect_ensemble(img, apply_smoothing=True)
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(result, f'FPS: {fps:.1f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        now = time.time()
        if now - last_save_time >= UPDATE_EVERY:
            save_status_json(slot_results)
            last_save_time = now

        cv2.imshow('Parking', result)
        cv2.imshow('Debug Threshold', debug_img)
        cv2.imshow('Debug Edge', debug_edge)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        try:
            if cv2.getWindowProperty('Parking', cv2.WND_PROP_VISIBLE) < 1: break
        except cv2.error: break

    cap.release()

cv2.destroyAllWindows()
