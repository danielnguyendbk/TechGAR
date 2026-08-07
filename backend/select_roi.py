import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import cv2


def choose_image_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để đánh dấu ROI",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
    )
    root.destroy()
    return Path(file_path) if file_path else None


def save_rois(json_path: Path, rois):
    payload = {"rois": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in rois]}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def select_rois(img, scale=1.0):
    view = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img.copy()

    print("[ROI] Kéo thả nhiều ô -> SPACE/ENTER để chốt từng ô -> ESC để kết thúc")
    boxes = cv2.selectROIs("Select ROI", view, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    rois = []
    for b in boxes:
        x, y, w, h = b
        if w <= 0 or h <= 0:
            continue
        if scale != 1.0:
            x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
        rois.append((int(x), int(y), int(w), int(h)))
    return rois


def main():
    parser = argparse.ArgumentParser(description="Chọn ROI và lưu vào rois.json")
    parser.add_argument("--image", type=str, help="Ảnh đầu vào (nếu bỏ qua sẽ mở hộp thoại chọn ảnh)")
    parser.add_argument("--out", type=str, default="rois.json", help="File JSON đầu ra")
    parser.add_argument("--max-width", type=int, default=1400, help="Resize preview để dễ chọn ROI")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else choose_image_file()
    if not image_path or not image_path.exists():
        print("[LỖI] Không có ảnh hợp lệ.")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print("[LỖI] Không đọc được ảnh.")
        return

    scale = min(1.0, args.max_width / img.shape[1]) if args.max_width > 0 else 1.0
    rois = select_rois(img, scale=scale)

    if not rois:
        print("[WARN] Không có ROI nào được chọn.")
        return

    out_path = Path(args.out)
    save_rois(out_path, rois)
    print(f"[OK] Đã lưu {len(rois)} ROI vào: {out_path.resolve()}")


if __name__ == "__main__":
    main()
