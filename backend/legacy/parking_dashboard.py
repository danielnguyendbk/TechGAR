import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from parking_cnn_utils import SlotCNNClassifier
from parking_yolo_utils import VehicleDetectorYOLO, overlap_ratio_roi

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def choose_image_file() -> Path | None:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    p = filedialog.askopenfilename(
        title="Chọn ảnh để detect",
        filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")],
    )
    root.destroy()
    return Path(p) if p else None


def load_rois(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rois = data.get("rois", [])
    if not rois:
        raise ValueError("rois.json không có dữ liệu ROI")
    return rois


def save_rois(path: Path, rois):
    payload = {"rois": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in rois]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def select_rois(img):
    print("[ROI] Kéo chuột chọn nhiều ô -> ENTER để lưu, ESC để hủy")
    boxes = cv2.selectROIs("Chon ROI", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Chon ROI")
    rois = []
    for b in boxes:
        x, y, w, h = [int(v) for v in b]
        if w > 0 and h > 0:
            rois.append((x, y, w, h))
    return rois


def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 15)
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary


def detect_slots(frame, rois, mode="threshold", threshold=0.18, cnn_model=None, yolo_detector=None, occ_overlap=0.15):
    h, w = frame.shape[:2]
    binary = preprocess(frame)
    results = []

    yolo_boxes = yolo_detector.detect(frame) if mode == "yolo" else []

    for i, r in enumerate(rois, start=1):
        x = max(0, int(r["x"]))
        y = max(0, int(r["y"]))
        rw = max(1, min(int(r["w"]), w - x))
        rh = max(1, min(int(r["h"]), h - y))

        roi_bgr = frame[y: y + rh, x: x + rw]
        roi_bin = binary[y: y + rh, x: x + rw]
        ratio = cv2.countNonZero(roi_bin) / float(rw * rh)

        if mode == "cnn":
            occupied, confidence, p_occ = cnn_model.predict_slot(roi_bgr)
            metric = p_occ
            metric_name = "p_occ"
        elif mode == "yolo":
            roi_xyxy = (x, y, x + rw, y + rh)
            max_overlap = 0.0
            max_conf = 0.0
            for d in yolo_boxes:
                box_xyxy = (d["x1"], d["y1"], d["x2"], d["y2"])
                ov = overlap_ratio_roi(roi_xyxy, box_xyxy)
                if ov > max_overlap:
                    max_overlap = ov
                if ov > 0 and d["conf"] > max_conf:
                    max_conf = d["conf"]
            occupied = max_overlap >= occ_overlap
            confidence = max_conf if occupied else (1.0 - min(1.0, max_overlap / max(occ_overlap, 1e-6)))
            metric = max_overlap
            metric_name = "ovlp"
        else:
            occupied = ratio > threshold
            confidence = ratio if occupied else (1.0 - ratio)
            metric = ratio
            metric_name = "ratio"

        results.append(
            {
                "slot": i,
                "occupied": occupied,
                "ratio": ratio,
                "metric": metric,
                "metric_name": metric_name,
                "confidence": confidence,
                "x": x,
                "y": y,
                "w": rw,
                "h": rh,
            }
        )

    return binary, results, yolo_boxes


def draw_frame_with_rois(frame, rois, results=None, yolo_boxes=None):
    out = frame.copy()
    if yolo_boxes:
        for d in yolo_boxes:
            cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (255, 200, 0), 2)
            cv2.putText(out, f"{d['cls_name']} {d['conf']:.2f}", (d["x1"], max(16, d["y1"] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    if results is None:
        for i, r in enumerate(rois, start=1):
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 255), 2)
            cv2.putText(out, f"S{i} CHUA_DETECT", (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        return out

    for r in results:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        occupied = r["occupied"]
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"S{r['slot']} {'CO_XE' if occupied else 'TRONG'} {r['metric_name']}={r['metric']:.2f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
    return out


def draw_dashboard(total_slots, results=None, cols=4):
    n = total_slots
    rows = int(np.ceil(n / cols))
    cell_w, cell_h = 190, 100
    pad = 18
    width = cols * cell_w + (cols + 1) * pad
    height = rows * cell_h + (rows + 1) * pad + 80

    canvas = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.putText(canvas, "SMART PARKING DASHBOARD", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

    occ_count, empty_count = 0, 0

    for idx in range(n):
        rr, cc = idx // cols, idx % cols
        x1 = pad + cc * (cell_w + pad)
        y1 = 55 + pad + rr * (cell_h + pad)
        x2, y2 = x1 + cell_w, y1 + cell_h

        if results is None:
            color, text = (120, 120, 120), "CHUA_DETECT"
            sub = ""
        else:
            occupied = results[idx]["occupied"]
            if occupied:
                color, text = (0, 0, 230), "CO XE"
                occ_count += 1
            else:
                color, text = (0, 170, 0), "TRONG"
                empty_count += 1
            sub = f"{results[idx]['metric_name']}={results[idx]['metric']:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 2)
        cv2.putText(canvas, f"S{idx+1}", (x1 + 12, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(canvas, text, (x1 + 12, y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
        if sub:
            cv2.putText(canvas, sub, (x1 + 12, y1 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)

    summary = f"Tong: {n} | Dang cho detect" if results is None else f"Tong: {n} | Trong: {empty_count} | Co xe: {occ_count}"
    cv2.putText(canvas, summary, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    return canvas


def run_webcam(rois, camera_id, interval, mode, threshold, cols, cnn_model, yolo_detector, occ_overlap):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[LỖI] Không mở được camera.")
        return

    print("Webcam mode: Q/ESC để thoát")
    last_detect = 0.0
    cached_frame, cached_binary, cached_results, cached_yolo = None, None, None, []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Mất frame camera")
            break

        now = time.time()
        if (now - last_detect) >= interval:
            cached_binary, cached_results, cached_yolo = detect_slots(
                frame,
                rois,
                mode=mode,
                threshold=threshold,
                cnn_model=cnn_model,
                yolo_detector=yolo_detector,
                occ_overlap=occ_overlap,
            )
            cached_frame = draw_frame_with_rois(frame, rois, cached_results, yolo_boxes=cached_yolo)
            last_detect = now
        elif cached_frame is None:
            cached_frame = draw_frame_with_rois(frame, rois, results=None)
            cached_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        dashboard = draw_dashboard(len(rois), cached_results, cols=cols)
        cv2.imshow("Camera + ROI", cached_frame)
        cv2.imshow("Dashboard Slot", dashboard)
        cv2.imshow("Binary Debug", cached_binary)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(rois, rois_path, image_path, mode, threshold, cols, cnn_model, yolo_detector, occ_overlap):
    print("Image mode: L tải ảnh | D detect | M đánh dấu ROI mới | Q/ESC thoát")

    current_img = None
    current_view = np.full((540, 960, 3), 40, dtype=np.uint8)
    cv2.putText(current_view, "Nhan L de tai anh", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 2)
    dashboard = draw_dashboard(len(rois), results=None, cols=cols)
    binary = np.zeros((540, 960), dtype=np.uint8)

    if image_path:
        p = Path(image_path)
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                current_img = img
                current_view = draw_frame_with_rois(current_img, rois, results=None)

    while True:
        cv2.imshow("Image + ROI", current_view)
        cv2.imshow("Dashboard Slot", dashboard)
        cv2.imshow("Binary Debug", binary)

        k = cv2.waitKey(50) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k in (ord("l"), ord("L")):
            picked = choose_image_file()
            if picked and picked.exists():
                img = cv2.imread(str(picked))
                if img is not None:
                    current_img = img
                    current_view = draw_frame_with_rois(current_img, rois, results=None)
                    dashboard = draw_dashboard(len(rois), results=None, cols=cols)
                    binary = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    print(f"[OK] Da tai anh: {picked}")
                else:
                    print("[LỖI] Không đọc được ảnh")
        elif k in (ord("d"), ord("D")):
            if current_img is None:
                print("[WARN] Chưa có ảnh. Nhấn L để tải ảnh trước.")
                continue
            binary, results, yolo_boxes = detect_slots(
                current_img,
                rois,
                mode=mode,
                threshold=threshold,
                cnn_model=cnn_model,
                yolo_detector=yolo_detector,
                occ_overlap=occ_overlap,
            )
            current_view = draw_frame_with_rois(current_img, rois, results, yolo_boxes=yolo_boxes)
            dashboard = draw_dashboard(len(rois), results=results, cols=cols)
            print("[OK] Detect xong, dashboard đã tô màu.")
        elif k in (ord("m"), ord("M")):
            if current_img is None:
                print("[WARN] Chưa có ảnh. Nhấn L để tải ảnh trước khi mark ROI.")
                continue
            new_rois = select_rois(current_img)
            if not new_rois:
                print("[WARN] Không có ROI nào được chọn.")
                continue
            save_rois(rois_path, new_rois)
            rois = load_rois(rois_path)
            current_view = draw_frame_with_rois(current_img, rois, results=None)
            dashboard = draw_dashboard(len(rois), results=None, cols=cols)
            binary = np.zeros((current_img.shape[0], current_img.shape[1]), dtype=np.uint8)
            print(f"[OK] Đã lưu {len(rois)} ROI vào: {rois_path.resolve()}")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Dashboard ô đỗ: threshold / CNN / YOLO")
    parser.add_argument("--rois", default="rois.json", help="File ROI json")
    parser.add_argument("--source", choices=["webcam", "image"], default="webcam", help="Nguồn dữ liệu")
    parser.add_argument("--camera-id", type=int, default=0, help="ID webcam")
    parser.add_argument("--interval", type=float, default=0.5, help="Webcam: giây giữa mỗi lần detect")
    parser.add_argument("--cols", type=int, default=4, help="Số cột dashboard")
    parser.add_argument("--image", type=str, help="Image mode: đường dẫn ảnh ban đầu (optional)")

    parser.add_argument("--mode", choices=["threshold", "cnn", "yolo"], default="threshold", help="Chế độ phân loại")
    parser.add_argument("--threshold", type=float, default=0.18, help="Ngưỡng ratio nếu mode=threshold")

    parser.add_argument("--cnn-model", type=str, default="slot_cnn.keras", help="Model Keras nếu mode=cnn")
    parser.add_argument("--patch-size", type=int, default=64, help="Input patch size cho CNN")
    parser.add_argument("--cnn-threshold", type=float, default=0.5, help="Ngưỡng occupied cho CNN")

    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="Model YOLO, vd yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold YOLO")
    parser.add_argument("--yolo-iou", type=float, default=0.5, help="IoU NMS YOLO")
    parser.add_argument("--occ-overlap", type=float, default=0.15, help="Tỷ lệ overlap(box,roi)/roi để coi là CÓ XE")

    args = parser.parse_args()

    rois_path = Path(args.rois)
    rois = load_rois(rois_path)
    print(f"[ROI] Đang dùng file: {rois_path.resolve()} | Số ô: {len(rois)}")

    cnn_model = None
    yolo_detector = None

    if args.mode == "cnn":
        try:
            cnn_model = SlotCNNClassifier(args.cnn_model, patch_size=args.patch_size, decision_threshold=args.cnn_threshold)
            print(f"[OK] Đã nạp CNN model: {Path(args.cnn_model).resolve()}")
        except Exception as e:
            print(f"[LỖI] Không dùng được mode=cnn: {e}")
            print("[GỢI Ý] Cài TensorFlow + đặt đúng đường dẫn model .keras/.h5")
            return

    if args.mode == "yolo":
        try:
            yolo_detector = VehicleDetectorYOLO(args.yolo_model, conf=args.yolo_conf, iou=args.yolo_iou)
            print(f"[OK] Đã nạp YOLO model: {args.yolo_model}")
        except Exception as e:
            print(f"[LỖI] Không dùng được mode=yolo: {e}")
            print("[GỢI Ý] Cài ultralytics: pip install ultralytics")
            return

    if args.source == "webcam":
        run_webcam(rois, args.camera_id, args.interval, args.mode, args.threshold, args.cols, cnn_model, yolo_detector, args.occ_overlap)
    else:
        run_image(rois, rois_path, args.image, args.mode, args.threshold, args.cols, cnn_model, yolo_detector, args.occ_overlap)


if __name__ == "__main__":
    main()
