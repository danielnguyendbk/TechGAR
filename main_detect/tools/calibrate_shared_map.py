"""Calibrate two partial camera views into one measured ground-plane map.

Workflow::

    capture -> mark -> fill calibration_points.csv -> build

Each camera gets its own pixel-to-world homography.  Calibration points do not
have to be visible in both cameras; they only need measured ``(x, y)`` values
in the same centimetre coordinate system on the physical parking model.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


MANIFEST_NAME = "capture_manifest.json"
POINTS_NAME = "calibration_points.csv"
CAMERA_IDS = ("cam1", "cam2")
CSV_FIELDS = (
    "camera",
    "label",
    "pixel_x",
    "pixel_y",
    "world_x_cm",
    "world_y_cm",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"File da ton tai: {path}. Dung --overwrite neu muon ghi de."
        )


def capture_frame(source: str, warmup_frames: int = 15) -> np.ndarray:
    """Read a recent frame instead of the first buffered network frame."""
    capture = cv2.VideoCapture(source)
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Khong mo duoc stream: {source}")
        latest = None
        for _ in range(max(1, int(warmup_frames))):
            ok, frame = capture.read()
            if ok and frame is not None:
                latest = frame
        if latest is None:
            raise RuntimeError(f"Khong doc duoc frame: {source}")
        return latest
    finally:
        capture.release()


def _parse_labels(value: str) -> List[str]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if len(labels) < 4:
        raise ValueError("Moi camera can it nhat 4 nhan diem.")
    if len(set(labels)) != len(labels):
        raise ValueError("Nhan diem trong mot camera phai duy nhat.")
    return labels


def select_labeled_points(
    window_name: str,
    image: np.ndarray,
    labels: Sequence[str],
) -> List[Tuple[int, int]]:
    """Collect one pixel point per label with undo/reset controls."""
    points: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = image.copy()
        panel_height = 72
        cv2.rectangle(preview, (0, 0), (preview.shape[1], panel_height), (0, 0, 0), -1)
        current = labels[len(points)] if len(points) < len(labels) else "DONE"
        cv2.putText(
            preview,
            f"Click point: {current}   ({len(points)}/{len(labels)})",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            "Left: add | Right: undo | R: reset | Enter: confirm | Q/Esc: cancel",
            (12, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
        )
        for index, point in enumerate(points):
            cv2.circle(preview, point, 7, (0, 255, 255), -1)
            cv2.circle(preview, point, 10, (0, 0, 0), 2)
            cv2.putText(
                preview,
                labels[index],
                (point[0] + 10, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == len(labels):
            return points


def draw_labeled_points(
    image: np.ndarray,
    labels: Sequence[str],
    points: Sequence[Tuple[int, int]],
) -> np.ndarray:
    output = image.copy()
    for label, point in zip(labels, points):
        cv2.circle(output, point, 8, (0, 255, 255), -1)
        cv2.circle(output, point, 12, (0, 0, 0), 2)
        cv2.putText(
            output,
            label,
            (point[0] + 12, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
        )
    return output


def _load_manifest(workspace: Path) -> dict:
    path = workspace / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Thieu {path}. Hay chay lenh capture truoc."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def capture_command(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace / MANIFEST_NAME
    _ensure_writable(manifest_path, args.overwrite)
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(),
        "cameras": {},
    }
    for camera_id, source in (("cam1", args.cam1_url), ("cam2", args.cam2_url)):
        frame = capture_frame(source, args.warmup_frames)
        image_path = workspace / f"capture_{camera_id}.png"
        _ensure_writable(image_path, args.overwrite)
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Khong ghi duoc anh: {image_path}")
        payload["cameras"][camera_id] = {
            "source": source,
            "image": image_path.name,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
        print(f"Da chup {camera_id}: {image_path}")
    _write_json(manifest_path, payload)
    print(f"Da ghi manifest: {manifest_path}")


def mark_command(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    manifest = _load_manifest(workspace)
    points_path = workspace / POINTS_NAME
    _ensure_writable(points_path, args.overwrite)
    labels_by_camera = {
        "cam1": _parse_labels(args.cam1_labels),
        "cam2": _parse_labels(args.cam2_labels),
    }
    rows = []
    try:
        for camera_id in CAMERA_IDS:
            image_path = workspace / manifest["cameras"][camera_id]["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
            labels = labels_by_camera[camera_id]
            points = select_labeled_points(
                f"Shared-map calibration - {camera_id}", image, labels
            )
            annotated = draw_labeled_points(image, labels, points)
            annotated_path = workspace / f"marked_{camera_id}.png"
            _ensure_writable(annotated_path, args.overwrite)
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"Khong ghi duoc anh: {annotated_path}")
            for label, (pixel_x, pixel_y) in zip(labels, points):
                rows.append({
                    "camera": camera_id,
                    "label": label,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y,
                    "world_x_cm": "",
                    "world_y_cm": "",
                })
            print(f"Da danh dau {camera_id}: {annotated_path}")
    finally:
        cv2.destroyAllWindows()

    with points_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Da tao file do: {points_path}")
    print("Hay dien world_x_cm va world_y_cm, sau do chay lenh build.")


def load_measurements(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {camera_id: [] for camera_id in CAMERA_IDS}
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing_columns = set(CSV_FIELDS) - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"CSV thieu cot: {sorted(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            camera_id = str(row["camera"]).strip()
            if camera_id not in grouped:
                raise ValueError(f"Dong {line_number}: camera khong hop le")
            try:
                grouped[camera_id].append({
                    "camera": camera_id,
                    "label": str(row["label"]).strip(),
                    "pixel": (float(row["pixel_x"]), float(row["pixel_y"])),
                    "world": (
                        float(row["world_x_cm"]),
                        float(row["world_y_cm"]),
                    ),
                })
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dong {line_number}: hay dien day du toa do pixel/world"
                ) from exc
    for camera_id, rows in grouped.items():
        if len(rows) < 4:
            raise ValueError(f"{camera_id} can it nhat 4 diem, hien co {len(rows)}")
        labels = [row["label"] for row in rows]
        if len(labels) != len(set(labels)):
            raise ValueError(f"{camera_id} co label trung nhau")
    return grouped


def compute_homography(
    measurements: Sequence[dict],
    ransac_threshold_cm: float = 2.0,
) -> Tuple[np.ndarray, dict]:
    image_points = np.asarray([row["pixel"] for row in measurements], dtype=np.float32)
    world_points = np.asarray([row["world"] for row in measurements], dtype=np.float32)
    if np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2:
        raise ValueError("Cac diem pixel gan thang hang; can trai rong thanh tu giac.")
    if np.linalg.matrix_rank(world_points - world_points.mean(axis=0)) < 2:
        raise ValueError("Cac diem world gan thang hang; can toa do 2 chieu.")
    method = cv2.RANSAC if len(measurements) > 4 else 0
    homography, inlier_mask = cv2.findHomography(
        image_points,
        world_points,
        method=method,
        ransacReprojThreshold=float(ransac_threshold_cm),
    )
    if homography is None or abs(float(np.linalg.det(homography))) < 1e-12:
        raise RuntimeError("Khong tinh duoc homography hop le")
    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_points, axis=1)
    inliers = (
        inlier_mask.reshape(-1).astype(bool)
        if inlier_mask is not None
        else np.ones(len(measurements), dtype=bool)
    )
    diagnostics = {
        "point_count": len(measurements),
        "inlier_count": int(inliers.sum()),
        "rms_error_cm": float(np.sqrt(np.mean(np.square(errors)))),
        "max_error_cm": float(errors.max()),
        "points": [
            {
                "label": row["label"],
                "pixel": [round(row["pixel"][0], 3), round(row["pixel"][1], 3)],
                "world_cm": [round(row["world"][0], 3), round(row["world"][1], 3)],
                "projected_world_cm": [round(float(value), 3) for value in point],
                "error_cm": round(float(error), 3),
                "inlier": bool(inlier),
            }
            for row, point, error, inlier in zip(
                measurements, projected, errors, inliers
            )
        ],
    }
    return homography.astype(np.float64), diagnostics


def transform_polygon(points: Iterable[Sequence[float]], homography: np.ndarray) -> np.ndarray:
    polygon = np.asarray(list(points), dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("Coverage polygon phai co it nhat 3 diem [x, y]")
    transformed = cv2.perspectiveTransform(
        polygon.reshape(-1, 1, 2), homography.astype(np.float64)
    ).reshape(-1, 2)
    return transformed.astype(np.float32)


def load_coverage_pixels(path: Path | None, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if path is None:
        return np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("polygon", [])
    points = [
        [float(item["x"]), float(item["y"])]
        if isinstance(item, dict)
        else [float(item[0]), float(item[1])]
        for item in raw
    ]
    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError(f"Coverage polygon khong hop le: {path}")
    return polygon


def full_frame_polygon(image_size: Tuple[int, int]) -> np.ndarray:
    """Return the complete camera frame, independent from the active ROI."""
    width, height = image_size
    return np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def load_parking_slots_world(
    path: Path | None,
    image_size: Tuple[int, int],
    homography: np.ndarray,
) -> List[dict]:
    """Load parking-space polygons and transform them into shared cm space."""
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = image_size
    reference_width = max(1.0, float(data.get("imageWidth", width)))
    reference_height = max(1.0, float(data.get("imageHeight", height)))
    scale_x = width / reference_width
    scale_y = height / reference_height
    slots = []
    for index, item in enumerate(data.get("slots", []), start=1):
        raw = item.get("polygon") or item.get("points") or []
        try:
            pixels = [
                [float(point["x"]) * scale_x, float(point["y"]) * scale_y]
                if isinstance(point, dict)
                else [float(point[0]) * scale_x, float(point[1]) * scale_y]
                for point in raw
            ]
        except (KeyError, TypeError, ValueError):
            continue
        if len(pixels) < 3:
            continue
        slots.append({
            "id": str(item.get("id", f"slot_{index}")),
            "polygon": transform_polygon(pixels, homography),
        })
    return slots


def convex_intersection(first: np.ndarray, second: np.ndarray) -> Tuple[float, np.ndarray]:
    first_hull = cv2.convexHull(np.asarray(first, dtype=np.float32))
    second_hull = cv2.convexHull(np.asarray(second, dtype=np.float32))
    area, intersection = cv2.intersectConvexConvex(first_hull, second_hull)
    if intersection is None or float(area) <= 1e-6:
        return 0.0, np.empty((0, 2), dtype=np.float32)
    return float(area), intersection.reshape(-1, 2).astype(np.float32)


def _world_to_canvas(
    point: Sequence[float],
    bounds: Tuple[float, float, float, float],
    size: Tuple[int, int],
    padding: int,
) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    width, height = size
    scale_x = (width - 2 * padding) / max(max_x - min_x, 1e-6)
    scale_y = (height - 2 * padding) / max(max_y - min_y, 1e-6)
    scale = min(scale_x, scale_y)
    x = padding + (float(point[0]) - min_x) * scale
    y = height - padding - (float(point[1]) - min_y) * scale
    return int(round(x)), int(round(y))


def _preview_bounds(polygons: Iterable[np.ndarray]) -> Tuple[float, float, float, float]:
    valid = [
        np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        for polygon in polygons
        if polygon is not None and np.asarray(polygon).size >= 6
    ]
    if not valid:
        raise ValueError("Khong co polygon world hop le de ve preview")
    all_points = np.vstack(valid)
    if not np.isfinite(all_points).all():
        raise ValueError("Full-view projection co toa do vo han; kiem tra diem calibration")
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    margin_x = max(5.0, float(max_x - min_x) * 0.08)
    margin_y = max(5.0, float(max_y - min_y) * 0.08)
    return (
        float(min_x - margin_x),
        float(min_y - margin_y),
        float(max_x + margin_x),
        float(max_y + margin_y),
    )


def _world_to_canvas_matrix(
    bounds: Tuple[float, float, float, float],
    size: Tuple[int, int],
    padding: int,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = bounds
    width, height = size
    scale_x = (width - 2 * padding) / max(max_x - min_x, 1e-6)
    scale_y = (height - 2 * padding) / max(max_y - min_y, 1e-6)
    scale = min(scale_x, scale_y)
    return np.asarray([
        [scale, 0.0, padding - scale * min_x],
        [0.0, -scale, height - padding + scale * min_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def draw_full_view_preview(
    images: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    full_coverages: Dict[str, np.ndarray],
    full_overlap: np.ndarray,
    diagnostics: Dict[str, dict],
    output_path: Path,
    bounds: Tuple[float, float, float, float] | None = None,
) -> dict:
    """Warp and blend both complete camera frames to validate H1/H2 visually."""
    canvas_size = (1400, 900)
    padding = 60
    bounds = bounds or _preview_bounds([*full_coverages.values(), full_overlap])
    world_to_canvas = _world_to_canvas_matrix(bounds, canvas_size, padding)
    warped_images = {}
    warped_masks = {}
    for camera_id in CAMERA_IDS:
        image = images[camera_id]
        image_to_canvas = world_to_canvas @ transforms[camera_id]
        warped_images[camera_id] = cv2.warpPerspective(
            image,
            image_to_canvas,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_masks[camera_id] = cv2.warpPerspective(
            source_mask,
            image_to_canvas,
            canvas_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    canvas = np.full((canvas_size[1], canvas_size[0], 3), 245, dtype=np.uint8)
    first_valid = warped_masks["cam1"] > 0
    second_valid = warped_masks["cam2"] > 0
    only_first = first_valid & ~second_valid
    only_second = second_valid & ~first_valid
    both = first_valid & second_valid
    canvas[only_first] = warped_images["cam1"][only_first]
    canvas[only_second] = warped_images["cam2"][only_second]
    blend = cv2.addWeighted(
        warped_images["cam1"], 0.5, warped_images["cam2"], 0.5, 0
    )
    canvas[both] = blend[both]

    colors = {"cam1": (20, 20, 20), "cam2": (20, 20, 235)}
    for camera_id, polygon in full_coverages.items():
        canvas_polygon = np.asarray([
            _world_to_canvas(point, bounds, canvas_size, padding)
            for point in polygon
        ], dtype=np.int32)
        cv2.polylines(canvas, [canvas_polygon], True, colors[camera_id], 3)
    overlap_polygon = np.asarray([
        _world_to_canvas(point, bounds, canvas_size, padding)
        for point in full_overlap
    ], dtype=np.int32)
    cv2.polylines(canvas, [overlap_polygon], True, (210, 30, 210), 3)

    for camera_id, diagnostic in diagnostics.items():
        for point in diagnostic["points"]:
            canvas_point = _world_to_canvas(
                point["world_cm"], bounds, canvas_size, padding
            )
            cv2.circle(canvas, canvas_point, 6, colors[camera_id], -1)
            cv2.putText(
                canvas,
                f"{camera_id}:{point['label']}",
                (canvas_point[0] + 7, canvas_point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                colors[camera_id],
                1,
            )

    cv2.rectangle(canvas, (15, 12), (890, 94), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "FULL CAMERA VIEW - 50/50 blend in shared world map",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (30, 30, 30),
        2,
    )
    cv2.putText(
        canvas,
        "Check painted ground lines: aligned = H1/H2 good; double/ghosted = recalibrate",
        (30, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (70, 70, 70),
        1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Khong ghi duoc full-view preview: {output_path}")
    return {
        "min_x_cm": round(bounds[0], 3),
        "min_y_cm": round(bounds[1], 3),
        "max_x_cm": round(bounds[2], 3),
        "max_y_cm": round(bounds[3], 3),
    }


def draw_active_roi_preview(
    images: Dict[str, np.ndarray],
    transforms: Dict[str, np.ndarray],
    active_pixels: Dict[str, np.ndarray],
    active_coverages: Dict[str, np.ndarray],
    active_overlap: np.ndarray,
    full_coverages: Dict[str, np.ndarray],
    parking_slots: Dict[str, List[dict]],
    output_path: Path,
    bounds: Tuple[float, float, float, float],
) -> None:
    """Warp captured images clipped by active ROI, then overlay parking slots."""
    canvas_size = (1400, 900)
    padding = 60
    world_to_canvas = _world_to_canvas_matrix(bounds, canvas_size, padding)
    warped_images = {}
    warped_masks = {}
    for camera_id in CAMERA_IDS:
        image = images[camera_id]
        image_to_canvas = world_to_canvas @ transforms[camera_id]
        source_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            source_mask,
            [np.rint(active_pixels[camera_id]).astype(np.int32)],
            255,
        )
        warped_images[camera_id] = cv2.warpPerspective(
            image,
            image_to_canvas,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_masks[camera_id] = cv2.warpPerspective(
            source_mask,
            image_to_canvas,
            canvas_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    canvas = np.full((canvas_size[1], canvas_size[0], 3), 245, dtype=np.uint8)
    first_valid = warped_masks["cam1"] > 0
    second_valid = warped_masks["cam2"] > 0
    only_first = first_valid & ~second_valid
    only_second = second_valid & ~first_valid
    both = first_valid & second_valid
    canvas[only_first] = warped_images["cam1"][only_first]
    canvas[only_second] = warped_images["cam2"][only_second]
    blend = cv2.addWeighted(
        warped_images["cam1"], 0.5, warped_images["cam2"], 0.5, 0
    )
    canvas[both] = blend[both]

    colors = {"cam1": (20, 20, 20), "cam2": (20, 20, 235)}
    # Faint complete-frame outlines make the difference from active ROI clear.
    for camera_id, polygon in full_coverages.items():
        full_polygon = np.asarray([
            _world_to_canvas(point, bounds, canvas_size, padding)
            for point in polygon
        ], dtype=np.int32)
        cv2.polylines(canvas, [full_polygon], True, (175, 175, 175), 1)
    for camera_id, polygon in active_coverages.items():
        active_polygon = np.asarray([
            _world_to_canvas(point, bounds, canvas_size, padding)
            for point in polygon
        ], dtype=np.int32)
        cv2.polylines(canvas, [active_polygon], True, colors[camera_id], 4)
    overlap_polygon = np.asarray([
        _world_to_canvas(point, bounds, canvas_size, padding)
        for point in active_overlap
    ], dtype=np.int32)
    if len(overlap_polygon) >= 3:
        cv2.polylines(canvas, [overlap_polygon], True, (210, 30, 210), 4)

    for camera_id, slots in parking_slots.items():
        for slot in slots:
            slot_polygon = np.asarray([
                _world_to_canvas(point, bounds, canvas_size, padding)
                for point in slot["polygon"]
            ], dtype=np.int32)
            cv2.polylines(canvas, [slot_polygon], True, colors[camera_id], 2)
            center = tuple(np.rint(slot_polygon.mean(axis=0)).astype(int))
            cv2.putText(
                canvas,
                slot["id"],
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                colors[camera_id],
                1,
            )

    cv2.rectangle(canvas, (15, 12), (930, 94), (255, 255, 255), -1)
    cv2.putText(
        canvas,
        "ACTIVE ROI + PARKING SLOTS - used by tracking",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (30, 30, 30),
        2,
    )
    cv2.putText(
        canvas,
        "Black=cam1 ROI | Red=cam2 ROI | Purple=operational overlap | Gray=full frame",
        (30, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (70, 70, 70),
        1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Khong ghi duoc active-ROI preview: {output_path}")


def draw_map_preview(
    coverages: Dict[str, np.ndarray],
    overlap: np.ndarray,
    diagnostics: Dict[str, dict],
    output_path: Path,
) -> dict:
    all_points = np.vstack([*coverages.values(), overlap])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    margin_x = max(5.0, float(max_x - min_x) * 0.08)
    margin_y = max(5.0, float(max_y - min_y) * 0.08)
    bounds = (
        float(min_x - margin_x),
        float(min_y - margin_y),
        float(max_x + margin_x),
        float(max_y + margin_y),
    )
    canvas_size = (1400, 900)
    padding = 55
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 248, dtype=np.uint8)

    # Ten-centimetre grid in the shared measured coordinate system.
    grid_start_x = int(np.floor(bounds[0] / 10.0) * 10)
    grid_end_x = int(np.ceil(bounds[2] / 10.0) * 10)
    grid_start_y = int(np.floor(bounds[1] / 10.0) * 10)
    grid_end_y = int(np.ceil(bounds[3] / 10.0) * 10)
    for world_x in range(grid_start_x, grid_end_x + 1, 10):
        top = _world_to_canvas((world_x, bounds[3]), bounds, canvas_size, padding)
        bottom = _world_to_canvas((world_x, bounds[1]), bounds, canvas_size, padding)
        cv2.line(canvas, top, bottom, (225, 225, 225), 1)
    for world_y in range(grid_start_y, grid_end_y + 1, 10):
        left = _world_to_canvas((bounds[0], world_y), bounds, canvas_size, padding)
        right = _world_to_canvas((bounds[2], world_y), bounds, canvas_size, padding)
        cv2.line(canvas, left, right, (225, 225, 225), 1)

    colors = {"cam1": (40, 40, 40), "cam2": (40, 40, 230)}
    overlay = canvas.copy()
    for camera_id, polygon in coverages.items():
        canvas_polygon = np.asarray([
            _world_to_canvas(point, bounds, canvas_size, padding)
            for point in polygon
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [canvas_polygon], colors[camera_id])
    cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)

    overlap_polygon = np.asarray([
        _world_to_canvas(point, bounds, canvas_size, padding)
        for point in overlap
    ], dtype=np.int32)
    overlap_layer = canvas.copy()
    cv2.fillPoly(overlap_layer, [overlap_polygon], (180, 70, 180))
    cv2.addWeighted(overlap_layer, 0.35, canvas, 0.65, 0, canvas)

    for camera_id, polygon in coverages.items():
        canvas_polygon = np.asarray([
            _world_to_canvas(point, bounds, canvas_size, padding)
            for point in polygon
        ], dtype=np.int32)
        cv2.polylines(canvas, [canvas_polygon], True, colors[camera_id], 3)
    cv2.polylines(canvas, [overlap_polygon], True, (180, 30, 180), 3)

    for camera_id, diagnostic in diagnostics.items():
        for point in diagnostic["points"]:
            canvas_point = _world_to_canvas(
                point["world_cm"], bounds, canvas_size, padding
            )
            cv2.circle(canvas, canvas_point, 5, colors[camera_id], -1)
            cv2.putText(
                canvas,
                f"{camera_id}:{point['label']}",
                (canvas_point[0] + 7, canvas_point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                colors[camera_id],
                1,
            )

    cv2.rectangle(canvas, (15, 12), (510, 92), (255, 255, 255), -1)
    cv2.putText(canvas, "CAM1 coverage", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["cam1"], 2)
    cv2.putText(canvas, "CAM2 coverage", (190, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["cam2"], 2)
    cv2.putText(canvas, "OVERLAP", (365, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 30, 180), 2)
    cv2.putText(canvas, "Grid: 10 cm", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Khong ghi duoc preview: {output_path}")
    return {
        "min_x_cm": round(bounds[0], 3),
        "min_y_cm": round(bounds[1], 3),
        "max_x_cm": round(bounds[2], 3),
        "max_y_cm": round(bounds[3], 3),
    }


def build_command(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    manifest = _load_manifest(workspace)
    measurements = load_measurements(workspace / POINTS_NAME)
    transforms = {}
    diagnostics = {}
    coverages = {}
    coverage_pixels_by_camera = {}
    full_coverages = {}
    images = {}
    parking_slots = {}
    coverage_paths = {
        "cam1": args.coverage_cam1.resolve() if args.coverage_cam1 else None,
        "cam2": args.coverage_cam2.resolve() if args.coverage_cam2 else None,
    }
    raw_slot_paths = {
        "cam1": getattr(args, "slots_cam1", None),
        "cam2": getattr(args, "slots_cam2", None),
    }
    slot_paths = {
        camera_id: path.resolve() if path else None
        for camera_id, path in raw_slot_paths.items()
    }
    for camera_id in CAMERA_IDS:
        homography, diagnostic = compute_homography(
            measurements[camera_id], args.ransac_threshold_cm
        )
        transforms[camera_id] = homography
        diagnostics[camera_id] = diagnostic
        camera_manifest = manifest["cameras"][camera_id]
        image_size = (
            int(camera_manifest["width"]),
            int(camera_manifest["height"]),
        )
        image_path = workspace / camera_manifest["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Khong doc duoc anh capture: {image_path}")
        images[camera_id] = image
        coverage_pixels = load_coverage_pixels(
            coverage_paths[camera_id],
            image_size,
        )
        coverage_pixels_by_camera[camera_id] = coverage_pixels
        coverages[camera_id] = transform_polygon(coverage_pixels, homography)
        full_coverages[camera_id] = transform_polygon(
            full_frame_polygon(image_size), homography
        )
        parking_slots[camera_id] = load_parking_slots_world(
            slot_paths[camera_id], image_size, homography
        )
        if diagnostic["rms_error_cm"] > args.max_rms_error_cm and not args.allow_high_error:
            raise ValueError(
                f"{camera_id} RMS={diagnostic['rms_error_cm']:.2f} cm vuot "
                f"{args.max_rms_error_cm:.2f} cm. Kiem tra diem do hoac dung "
                "--allow-high-error de chap nhan co chu y."
            )

    output_path = args.output.resolve()
    _ensure_writable(output_path, args.overwrite)
    preview_path = workspace / "shared_map_preview.png"
    full_preview_path = workspace / "shared_map_full_view.png"
    active_preview_path = workspace / "shared_map_active_roi.png"
    _ensure_writable(preview_path, args.overwrite)
    _ensure_writable(full_preview_path, args.overwrite)
    _ensure_writable(active_preview_path, args.overwrite)

    full_overlap_area, full_overlap = convex_intersection(
        full_coverages["cam1"], full_coverages["cam2"]
    )
    if full_overlap_area <= 0 or len(full_overlap) < 3:
        raise ValueError(
            "Hai khung hinh day du khong giao nhau tren world map. "
            "Kiem tra thu tu A-B-C-D cua hai camera."
        )
    full_bounds_tuple = _preview_bounds([*full_coverages.values(), full_overlap])
    full_world_bounds = draw_full_view_preview(
        images,
        transforms,
        full_coverages,
        full_overlap,
        diagnostics,
        full_preview_path,
        bounds=full_bounds_tuple,
    )

    overlap_area, overlap = convex_intersection(
        coverages["cam1"], coverages["cam2"]
    )
    draw_active_roi_preview(
        images,
        transforms,
        coverage_pixels_by_camera,
        coverages,
        overlap,
        full_coverages,
        parking_slots,
        active_preview_path,
        full_bounds_tuple,
    )
    if overlap_area <= 0 or len(overlap) < 3:
        raise ValueError(
            "Full-view preview da tao, nhung hai ROI hoat dong khong giao nhau. "
            "Kiem tra roi_mask_cam1/2 trong shared_map_active_roi.png."
        )

    world_bounds = draw_map_preview(coverages, overlap, diagnostics, preview_path)

    common_labels = set(row["label"] for row in measurements["cam1"]) & set(
        row["label"] for row in measurements["cam2"]
    )
    shared_point_checks = []
    for label in sorted(common_labels):
        first = next(
            point for point in diagnostics["cam1"]["points"] if point["label"] == label
        )
        second = next(
            point for point in diagnostics["cam2"]["points"] if point["label"] == label
        )
        distance = float(np.linalg.norm(
            np.subtract(first["projected_world_cm"], second["projected_world_cm"])
        ))
        shared_point_checks.append({
            "label": label,
            "cross_camera_error_cm": round(distance, 3),
        })

    payload = {
        "schema_version": 4,
        "world": {
            "unit": "cm",
            "coordinate_system": "measured_parking_ground_plane",
            "bounds": world_bounds,
            "full_view_bounds": full_world_bounds,
        },
        "camera_transforms": {
            camera_id: transforms[camera_id].tolist()
            for camera_id in CAMERA_IDS
        },
        "camera_coverage_world": {
            camera_id: [[round(float(x), 4), round(float(y), 4)] for x, y in coverages[camera_id]]
            for camera_id in CAMERA_IDS
        },
        "camera_full_view_world": {
            camera_id: [[round(float(x), 4), round(float(y), 4)] for x, y in full_coverages[camera_id]]
            for camera_id in CAMERA_IDS
        },
        "parking_slots_world": {
            camera_id: [
                {
                    "id": slot["id"],
                    "polygon": [
                        [round(float(x), 4), round(float(y), 4)]
                        for x, y in slot["polygon"]
                    ],
                }
                for slot in parking_slots[camera_id]
            ]
            for camera_id in CAMERA_IDS
        },
        "edge_adjacency": [
            {"source_camera": "cam1", "exit_edge": "right", "target_camera": "cam2"},
            {"source_camera": "cam2", "exit_edge": "left", "target_camera": "cam1"},
        ],
        "overlap_world_polygon": [
            [round(float(x), 4), round(float(y), 4)] for x, y in overlap
        ],
        "full_view_overlap_world_polygon": [
            [round(float(x), 4), round(float(y), 4)] for x, y in full_overlap
        ],
        "tracking_defaults": {
            # These cameras view the toy vehicle from opposing top-down
            # angles, so bbox centre is more camera-invariant than the local
            # tracker's ground/contact point.
            "shared_map_anchor": "bbox_center",
        },
        "matching_defaults": {
            "unit": "cm",
            "handoff_match_distance": float(args.handoff_match_distance_cm),
            "handoff_prediction_radius": float(args.handoff_prediction_radius_cm),
            "dormant_match_distance": float(args.dormant_match_distance_cm),
            # Tight post-allocation reconciliation gate.  It is intentionally
            # smaller than the normal handoff radius and uses bbox-centre map
            # anchors plus mutual uniqueness before any ID can be merged.
            "cross_camera_duplicate_distance": float(
                args.handoff_match_distance_cm * 0.60
            ),
        },
        "calibration_quality": {
            "cameras": diagnostics,
            "shared_point_checks": shared_point_checks,
            "overlap_area_cm2": round(overlap_area, 3),
            "active_roi_overlap_area_cm2": round(overlap_area, 3),
            "full_view_overlap_area_cm2": round(full_overlap_area, 3),
        },
        "source": {
            "workspace": str(workspace),
            "measurements": POINTS_NAME,
            "coverage_masks": {
                camera_id: str(path) if path else None
                for camera_id, path in coverage_paths.items()
            },
            "parking_slots": {
                camera_id: str(path) if path else None
                for camera_id, path in slot_paths.items()
            },
            "preview": str(preview_path),
            "previews": {
                "full_view": str(full_preview_path),
                "active_roi": str(active_preview_path),
                "legacy_active_map": str(preview_path),
            },
        },
    }
    _write_json(output_path, payload)
    print(f"Da tao calibration: {output_path}")
    print(f"Da tao full-view preview: {full_preview_path}")
    print(f"Da tao active-ROI preview: {active_preview_path}")
    print(f"Da tao legacy map preview: {preview_path}")
    print(f"Full-view overlap: {full_overlap_area:.2f} cm^2")
    print(f"Active-ROI overlap: {overlap_area:.2f} cm^2")
    for camera_id in CAMERA_IDS:
        quality = diagnostics[camera_id]
        suffix = " (4 diem: chua co du lieu du de cross-check)" if quality["point_count"] == 4 else ""
        print(
            f"{camera_id}: RMS={quality['rms_error_cm']:.3f} cm, "
            f"max={quality['max_error_cm']:.3f} cm, "
            f"inlier={quality['inlier_count']}/{quality['point_count']}{suffix}"
        )
    for item in shared_point_checks:
        print(
            f"Shared {item['label']}: cross-camera error "
            f"{item['cross_camera_error_cm']:.3f} cm"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate two partial DroidCam views into one centimetre map"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Chup frame cam1/cam2")
    capture.add_argument("--cam1-url", required=True)
    capture.add_argument("--cam2-url", required=True)
    capture.add_argument("--workspace", required=True, type=Path)
    capture.add_argument("--warmup-frames", type=int, default=15)
    capture.add_argument("--overwrite", action="store_true")
    capture.set_defaults(handler=capture_command)

    mark = subparsers.add_parser("mark", help="Click cac diem pixel co nhan")
    mark.add_argument("--workspace", required=True, type=Path)
    mark.add_argument("--cam1-labels", default="A,B,C,D")
    mark.add_argument("--cam2-labels", default="E,F,G,H")
    mark.add_argument("--overwrite", action="store_true")
    mark.set_defaults(handler=mark_command)

    build = subparsers.add_parser("build", help="Tinh H1/H2 va dung world map")
    build.add_argument("--workspace", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--coverage-cam1", type=Path)
    build.add_argument("--coverage-cam2", type=Path)
    build.add_argument("--slots-cam1", type=Path)
    build.add_argument("--slots-cam2", type=Path)
    build.add_argument("--ransac-threshold-cm", type=float, default=2.0)
    build.add_argument("--max-rms-error-cm", type=float, default=3.0)
    build.add_argument("--allow-high-error", action="store_true")
    build.add_argument("--handoff-match-distance-cm", type=float, default=15.0)
    build.add_argument("--handoff-prediction-radius-cm", type=float, default=25.0)
    build.add_argument("--dormant-match-distance-cm", type=float, default=35.0)
    build.add_argument("--overwrite", action="store_true")
    build.set_defaults(handler=build_command)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        args.handler(args)
    except KeyboardInterrupt:
        print("Da huy thao tac.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
