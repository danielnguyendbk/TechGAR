from pathlib import Path

import numpy as np


class VehicleDetectorYOLO:
    """YOLO vehicle detector wrapper.

    Output mỗi detection:
    {
      x1,y1,x2,y2, conf, cls_id, cls_name
    }
    """

    # COCO vehicle classes
    DEFAULT_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.25, iou: float = 0.5, class_ids=None):
        self.model_path = model_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.class_ids = set(class_ids) if class_ids is not None else set(self.DEFAULT_VEHICLE_CLASS_IDS)
        self.model = None

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("Chưa cài ultralytics. Cài bằng: pip install ultralytics") from e

        # ultralytics tự tải model nếu là yolov8n.pt
        self.model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray):
        results = self.model.predict(source=frame_bgr, conf=self.conf, iou=self.iou, verbose=False)
        detections = []

        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return detections

        names = getattr(r, "names", {}) or {}

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

        for b, c, k in zip(xyxy, confs, clss):
            if self.class_ids and int(k) not in self.class_ids:
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(c),
                    "cls_id": int(k),
                    "cls_name": str(names.get(int(k), k)),
                }
            )

        return detections


def intersection_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


def overlap_ratio_roi(roi_xyxy, box_xyxy):
    rx1, ry1, rx2, ry2 = roi_xyxy
    bx1, by1, bx2, by2 = box_xyxy
    inter = intersection_area(rx1, ry1, rx2, ry2, bx1, by1, bx2, by2)
    roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
    return inter / float(roi_area)
