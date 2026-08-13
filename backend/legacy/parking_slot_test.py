import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

from parking_cnn_utils import SlotCNNClassifier
from parking_yolo_utils import VehicleDetectorYOLO, overlap_ratio_roi

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def choose_image_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh bãi xe để test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def load_rois(json_path: Path):
    if not json_path.exists():
        return []
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("rois", [])


def save_rois(json_path: Path, rois):
    payload = {"rois": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in rois]}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def select_rois(img, scale=1.0):
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()
    print("[ROI] Chọn các ô đỗ -> ENTER để xác nhận, ESC để kết thúc.")
    boxes = cv2.selectROIs("Chon ROI (ENTER xong)", view, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Chon ROI (ENTER xong)")

    rois = []
    for b in boxes:
        x, y, w, h = b
        if w <= 0 or h <= 0:
            continue
        if scale != 1.0:
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
        rois.append((int(x), int(y), int(w), int(h)))
    return rois


def preprocess_for_detection(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 15)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


def classify_slots(
    img_bgr,
    rois,
    mode="threshold",
    occ_threshold=0.18,
    cnn_model=None,
    yolo_detector=None,
    occ_overlap=0.15,
):
    h_img, w_img = img_bgr.shape[:2]
    binary = preprocess_for_detection(img_bgr)
    out = img_bgr.copy()

    yolo_boxes = yolo_detector.detect(img_bgr) if mode == "yolo" else []

    results = []
    for i, roi in enumerate(rois, start=1):
        x = max(0, int(roi["x"] if isinstance(roi, dict) else roi[0]))
        y = max(0, int(roi["y"] if isinstance(roi, dict) else roi[1]))
        w = int(roi["w"] if isinstance(roi, dict) else roi[2])
        h = int(roi["h"] if isinstance(roi, dict) else roi[3])
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        roi_bgr = img_bgr[y:y + h, x:x + w]
        roi_bin = binary[y:y + h, x:x + w]
        occ_ratio = cv2.countNonZero(roi_bin) / float(w * h)

        if mode == "cnn":
            occupied, confidence, p_occ = cnn_model.predict_slot(roi_bgr)
            metric = p_occ
            metric_name = "p_occ"
        elif mode == "yolo":
            roi_xyxy = (x, y, x + w, y + h)
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
            occupied = occ_ratio > occ_threshold
            confidence = occ_ratio if occupied else (1.0 - occ_ratio)
            metric = occ_ratio
            metric_name = "ratio"

        color = (0, 0, 255) if occupied else (0, 255, 0)
        text = f"S{i}: {'CO_XE' if occupied else 'TRONG'} {metric_name}={metric:.2f}"
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, text, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        results.append(
            {
                "slot": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "occ_ratio": float(occ_ratio),
                "confidence": float(confidence),
                "metric": float(metric),
                "status": "occupied" if occupied else "empty",
                "mode": mode,
            }
        )

    if mode == "yolo":
        for d in yolo_boxes:
            cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (255, 200, 0), 2)
            cv2.putText(out, f"{d['cls_name']} {d['conf']:.2f}", (d["x1"], max(16, d["y1"] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    return out, binary, results


def main():
    parser = argparse.ArgumentParser(description="Test phát hiện ô trống: threshold / CNN / YOLO")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh cần test")
    parser.add_argument("--rois", type=str, default="rois.json", help="File JSON chứa ROI")
    parser.add_argument("--mark", action="store_true", help="Bật chế độ đánh dấu ROI mới")

    parser.add_argument("--mode", choices=["threshold", "cnn", "yolo"], default="threshold", help="Chế độ phân loại")
    parser.add_argument("--threshold", type=float, default=0.18, help="Ngưỡng occupied ratio (mode=threshold)")

    parser.add_argument("--cnn-model", type=str, default="slot_cnn.keras", help="Model Keras cho mode=cnn")
    parser.add_argument("--patch-size", type=int, default=64, help="Kích thước input patch cho CNN")
    parser.add_argument("--cnn-threshold", type=float, default=0.5, help="Ngưỡng quyết định occupied cho CNN")

    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="Model YOLO, vd yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold YOLO")
    parser.add_argument("--yolo-iou", type=float, default=0.5, help="IoU NMS YOLO")
    parser.add_argument("--occ-overlap", type=float, default=0.15, help="Tỷ lệ overlap(box,roi)/roi để coi là CÓ XE")

    parser.add_argument("--save", type=str, default="result_annotated.jpg", help="Tên file ảnh kết quả")
    parser.add_argument("--save-bin", type=str, default="result_binary.jpg", help="Tên file binary kết quả")
    parser.add_argument("--mark-only", action="store_true", help="Chỉ đánh dấu ROI rồi thoát")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image_file()
    if not image_path or not image_path.exists():
        print("[LỖI] Chưa chọn ảnh hoặc ảnh không tồn tại.")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print("[LỖI] Không đọc được ảnh.")
        return

    rois_path = Path(args.rois)

    if args.mark:
        max_w = 1400
        scale = min(1.0, max_w / img.shape[1])
        new_rois = select_rois(img, scale=scale)
        if not new_rois:
            print("[ROI] Không có ROI nào được chọn.")
            return
        save_rois(rois_path, new_rois)
        print(f"[OK] Đã lưu {len(new_rois)} ROI vào {rois_path.resolve()}")
        cv2.destroyAllWindows()
        if args.mark_only:
            print("[OK] Mark-only hoàn tất.")
            return

    rois = load_rois(rois_path)
    if not rois:
        print(f"[LỖI] Không có ROI trong {rois_path.resolve()}. Chạy lại với --mark để chọn ô.")
        return

    cnn_model = None
    yolo_detector = None
    if args.mode == "cnn":
        try:
            cnn_model = SlotCNNClassifier(args.cnn_model, patch_size=args.patch_size, decision_threshold=args.cnn_threshold)
            print(f"[OK] Đã nạp CNN model: {Path(args.cnn_model).resolve()}")
        except Exception as e:
            print(f"[LỖI] Không dùng được mode=cnn: {e}")
            print("[GỢI Ý] Cài TensorFlow + kiểm tra file model .keras/.h5")
            return
    elif args.mode == "yolo":
        try:
            yolo_detector = VehicleDetectorYOLO(args.yolo_model, conf=args.yolo_conf, iou=args.yolo_iou)
            print(f"[OK] Đã nạp YOLO model: {args.yolo_model}")
        except Exception as e:
            print(f"[LỖI] Không dùng được mode=yolo: {e}")
            print("[GỢI Ý] pip install ultralytics")
            return

    annotated, binary, results = classify_slots(
        img,
        rois,
        mode=args.mode,
        occ_threshold=args.threshold,
        cnn_model=cnn_model,
        yolo_detector=yolo_detector,
        occ_overlap=args.occ_overlap,
    )

    print("\n=== KẾT QUẢ ===")
    empty_count = sum(1 for r in results if r["status"] == "empty")
    occ_count = len(results) - empty_count
    for r in results:
        st = "CÓ XE" if r["status"] == "occupied" else "TRỐNG"
        print(f"Slot {r['slot']:02d}: {st:6s} | metric={r['metric']:.3f} | conf={r['confidence']:.3f}")

    print(f"Tổng: {len(results)} ô | Trống: {empty_count} | Có xe: {occ_count}")

    save_path = Path(args.save)
    save_bin_path = Path(args.save_bin)
    cv2.imwrite(str(save_path), annotated)
    cv2.imwrite(str(save_bin_path), binary)
    print(f"[OK] Đã lưu ảnh kết quả: {save_path.resolve()}")
    print(f"[OK] Đã lưu ảnh binary : {save_bin_path.resolve()}")

    cv2.imshow("Ket qua nhan dien", annotated)
    cv2.imshow("Binary", binary)
    print("Nhấn phím bất kỳ để đóng cửa sổ...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
