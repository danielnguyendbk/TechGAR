r"""Calibrate two DroidCam views without manually entering world X/Y.

Place one measured rectangle on the ground inside the area visible to both
cameras. Run::

    & ..\.venv\Scripts\python.exe .\calibrate_map.py

Click the same physical corners A, B, C, D on cam1 and cam2. The script asks
only for AB and AD in centimetres, chooses A=(0,0) automatically, and builds
both pixel-to-world homographies plus the shared map.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from tools.calibrate_shared_map import (
    CSV_FIELDS,
    MANIFEST_NAME,
    POINTS_NAME,
    build_command,
    capture_frame,
    draw_labeled_points,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CAM1_URL = "http://192.168.100.53:4747/video/force/1280x720"
DEFAULT_CAM2_URL = "http://192.168.100.198:4747/video/force/1280x720"
POINT_LABELS = ("A", "B", "C", "D")


def _read_positive_length(prompt: str) -> float:
    while True:
        try:
            raw = input(prompt).strip().replace(",", ".")
        except EOFError as exc:
            raise KeyboardInterrupt from exc
        try:
            value = float(raw)
        except ValueError:
            print("Hay nhap mot so, vi du: 25 hoac 25.5")
            continue
        if value <= 0:
            print("Chieu dai phai lon hon 0 cm.")
            continue
        return value


def _render_selection(
    image: np.ndarray,
    camera_id: str,
    points: Sequence[Tuple[int, int]],
) -> np.ndarray:
    preview = image.copy()
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 76), (0, 0, 0), -1)
    next_label = POINT_LABELS[len(points)] if len(points) < 4 else "DONE"
    cv2.putText(
        preview,
        f"{camera_id.upper()}: click SAME overlap rectangle - next point {next_label}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        preview,
        f"Points: {len(points)}/4 | Order A-B-C-D around rectangle | Right: undo | R: reset | Q: cancel",
        (12, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (235, 235, 235),
        1,
    )
    for label, point in zip(POINT_LABELS, points):
        cv2.circle(preview, point, 7, (0, 255, 255), -1)
        cv2.circle(preview, point, 10, (0, 0, 0), 2)
        cv2.putText(
            preview,
            label,
            (point[0] + 11, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 255),
            2,
        )
    if len(points) >= 2:
        cv2.polylines(
            preview,
            [np.asarray(points, dtype=np.int32)],
            len(points) == 4,
            (0, 220, 255),
            2,
        )
    return preview


def select_rectangle_points(camera_id: str, image: np.ndarray) -> List[Tuple[int, int]]:
    """Select the same four physical rectangle corners in one camera."""
    points: List[Tuple[int, int]] = []
    window_name = f"Shared rectangle - {camera_id}"

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            removed_label = POINT_LABELS[len(points) - 1]
            points.pop()
            print(f"Da xoa diem {camera_id}:{removed_label}")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        cv2.imshow(window_name, _render_selection(image, camera_id, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        if key == ord("r"):
            points.clear()
            print(f"Da xoa tat ca diem cua {camera_id}.")
        if len(points) == 4:
            cv2.imshow(window_name, _render_selection(image, camera_id, points))
            cv2.waitKey(250)
            cv2.destroyWindow(window_name)
            return points.copy()


def _capture_both(args: argparse.Namespace, workspace: Path) -> dict:
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now().astimezone().isoformat(),
        "cameras": {},
    }
    for camera_id, source in (("cam1", args.cam1_url), ("cam2", args.cam2_url)):
        print(f"Dang chup {camera_id}: {source}")
        frame = capture_frame(source, args.warmup_frames)
        image_path = workspace / f"capture_{camera_id}.png"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Khong ghi duoc anh: {image_path}")
        manifest["cameras"][camera_id] = {
            "source": source,
            "image": image_path.name,
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
        print(f"Da chup {camera_id}: {image_path}")
    (workspace / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _pixel_length(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.subtract(first, second)))


def _make_rows(
    points_by_camera: Dict[str, Sequence[Tuple[int, int]]],
    ab_cm: float,
    ad_cm: float,
) -> List[dict]:
    # The origin and axes are generated by the script, not entered by the user.
    world_by_label = {
        "A": (0.0, 0.0),
        "B": (ab_cm, 0.0),
        "C": (ab_cm, ad_cm),
        "D": (0.0, ad_cm),
    }
    rows = []
    for camera_id in ("cam1", "cam2"):
        for label, (pixel_x, pixel_y) in zip(
            POINT_LABELS, points_by_camera[camera_id]
        ):
            world_x, world_y = world_by_label[label]
            rows.append({
                "camera": camera_id,
                "label": label,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "world_x_cm": world_x,
                "world_y_cm": world_y,
            })
    return rows


def _save_points_and_images(
    workspace: Path,
    manifest: dict,
    points_by_camera: Dict[str, Sequence[Tuple[int, int]]],
    rows: List[dict],
) -> None:
    points_path = workspace / POINTS_NAME
    with points_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for camera_id in ("cam1", "cam2"):
        image_path = workspace / manifest["cameras"][camera_id]["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Khong doc duoc anh: {image_path}")
        marked = draw_labeled_points(
            image,
            POINT_LABELS,
            points_by_camera[camera_id],
        )
        marked_path = workspace / f"marked_{camera_id}.png"
        if not cv2.imwrite(str(marked_path), marked):
            raise RuntimeError(f"Khong ghi duoc anh: {marked_path}")
    print(f"Da tu dong luu diem va toa do do script tao: {points_path}")


def _print_scale_diagnostics(
    points_by_camera: Dict[str, Sequence[Tuple[int, int]]],
    ab_cm: float,
    ad_cm: float,
) -> None:
    print("\nTy le tai cac canh tham chieu (chi de kiem tra):")
    for camera_id in ("cam1", "cam2"):
        points = points_by_camera[camera_id]
        p_ab = _pixel_length(points[0], points[1])
        p_ad = _pixel_length(points[0], points[3])
        print(
            f"{camera_id}: pAB={p_ab:.2f}px, pAD={p_ad:.2f}px, "
            f"AB={p_ab / ab_cm:.3f}px/cm, AD={p_ad / ad_cm:.3f}px/cm"
        )
    print(
        "Script dung homography 4 diem, khong dung mot ty le px/cm co dinh, "
        "vi anh co phoi canh."
    )


def run(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    print("\nDat MOT hinh chu nhat tren mat bai, nam tron trong vung CA HAI cam cung thay.")
    print("Danh dau 4 goc A-B-C-D theo vong quanh hinh. Khong can tu chon goc toa do.\n")

    manifest = _capture_both(args, workspace)
    points_by_camera: Dict[str, List[Tuple[int, int]]] = {}
    try:
        for camera_id in ("cam1", "cam2"):
            image_path = workspace / manifest["cameras"][camera_id]["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Khong doc duoc anh: {image_path}")
            print(f"\nClick A-B-C-D cua CUNG hinh chu nhat tren {camera_id}.")
            points_by_camera[camera_id] = select_rectangle_points(camera_id, image)
    finally:
        cv2.destroyAllWindows()

    print("\nChi can nhap hai chieu dai that cua hinh chu nhat:")
    ab_cm = _read_positive_length("Nhap chieu dai AB (cm): ")
    ad_cm = _read_positive_length("Nhap chieu dai AD (cm): ")
    _print_scale_diagnostics(points_by_camera, ab_cm, ad_cm)

    rows = _make_rows(points_by_camera, ab_cm, ad_cm)
    _save_points_and_images(workspace, manifest, points_by_camera, rows)
    build_command(argparse.Namespace(
        workspace=workspace,
        output=args.output.resolve(),
        coverage_cam1=args.coverage_cam1.resolve() if args.coverage_cam1 else None,
        coverage_cam2=args.coverage_cam2.resolve() if args.coverage_cam2 else None,
        slots_cam1=args.slots_cam1.resolve() if args.slots_cam1 else None,
        slots_cam2=args.slots_cam2.resolve() if args.slots_cam2 else None,
        ransac_threshold_cm=2.0,
        max_rms_error_cm=3.0,
        allow_high_error=False,
        handoff_match_distance_cm=15.0,
        handoff_prediction_radius_cm=25.0,
        dormant_match_distance_cm=35.0,
        overwrite=True,
    ))

    previews = (
        (
            workspace / "shared_map_full_view.png",
            "1/2 FULL CAMERA VIEW - press any key for active ROI",
        ),
        (
            workspace / "shared_map_active_roi.png",
            "2/2 ACTIVE ROI + PARKING SLOTS - press any key to close",
        ),
    )
    for preview_path, window_name in previews:
        preview = cv2.imread(str(preview_path))
        if preview is None:
            continue
        print(f"Dang hien thi: {preview_path}")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, preview)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a two-camera cm map from one shared measured rectangle"
    )
    parser.add_argument("--cam1-url", default=DEFAULT_CAM1_URL)
    parser.add_argument("--cam2-url", default=DEFAULT_CAM2_URL)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PROJECT_ROOT / "config" / "shared_map_01",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "config" / "two_camera.shared_cm.json",
    )
    parser.add_argument(
        "--coverage-cam1",
        type=Path,
        default=PROJECT_ROOT / "config" / "roi_mask_cam1.json",
    )
    parser.add_argument(
        "--coverage-cam2",
        type=Path,
        default=PROJECT_ROOT / "config" / "roi_mask_cam2.json",
    )
    parser.add_argument(
        "--slots-cam1",
        type=Path,
        default=PROJECT_ROOT / "config" / "parking_slots_cam1.json",
    )
    parser.add_argument(
        "--slots-cam2",
        type=Path,
        default=PROJECT_ROOT / "config" / "parking_slots_cam2.json",
    )
    parser.add_argument("--warmup-frames", type=int, default=15)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("Da huy hieu chuan; chua dung map moi.")
        return 130
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"Khong the dung map: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
