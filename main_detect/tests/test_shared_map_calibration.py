import argparse
import csv
import json

import cv2
import numpy as np

from calibrate_map import _make_rows, _pixel_length
from tools.calibrate_shared_map import (
    CSV_FIELDS,
    build_command,
    compute_homography,
    convex_intersection,
    load_parking_slots_world,
)


def test_easy_calibration_generates_shared_origin_from_rectangle_lengths():
    points = {
        "cam1": [(10, 20), (110, 20), (110, 70), (10, 70)],
        "cam2": [(30, 40), (230, 50), (210, 150), (20, 130)],
    }

    rows = _make_rows(points, ab_cm=40.0, ad_cm=25.0)
    worlds = {
        camera: {
            row["label"]: (row["world_x_cm"], row["world_y_cm"])
            for row in rows
            if row["camera"] == camera
        }
        for camera in ("cam1", "cam2")
    }

    expected = {
        "A": (0.0, 0.0),
        "B": (40.0, 0.0),
        "C": (40.0, 25.0),
        "D": (0.0, 25.0),
    }
    assert worlds["cam1"] == expected
    assert worlds["cam2"] == expected
    assert _pixel_length((0, 0), (3, 4)) == 5.0


def test_compute_homography_maps_camera_points_to_centimetres():
    measurements = [
        {"label": "A", "pixel": (0, 0), "world": (10, 20)},
        {"label": "B", "pixel": (100, 0), "world": (60, 20)},
        {"label": "C", "pixel": (100, 100), "world": (60, 70)},
        {"label": "D", "pixel": (0, 100), "world": (10, 70)},
        {"label": "M", "pixel": (50, 50), "world": (35, 45)},
    ]

    homography, diagnostics = compute_homography(measurements)
    mapped = homography @ np.array([20.0, 40.0, 1.0])
    mapped = mapped[:2] / mapped[2]

    assert np.allclose(mapped, [20.0, 40.0], atol=1e-3)
    assert diagnostics["rms_error_cm"] < 1e-3
    assert diagnostics["inlier_count"] == 5


def test_convex_intersection_returns_the_shared_camera_strip():
    cam1 = np.array([[0, 0], [60, 0], [60, 50], [0, 50]], np.float32)
    cam2 = np.array([[40, 0], [100, 0], [100, 50], [40, 50]], np.float32)

    area, overlap = convex_intersection(cam1, cam2)

    assert area == 1000.0
    assert np.isclose(overlap[:, 0].min(), 40.0)
    assert np.isclose(overlap[:, 0].max(), 60.0)


def test_parking_slots_are_scaled_and_projected_to_world(tmp_path):
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(json.dumps({
        "imageWidth": 200,
        "imageHeight": 200,
        "slots": [{
            "id": "A01",
            "polygon": [
                {"x": 20, "y": 40},
                {"x": 60, "y": 40},
                {"x": 60, "y": 80},
                {"x": 20, "y": 80},
            ],
        }],
    }), encoding="utf-8")

    slots = load_parking_slots_world(slots_path, (100, 100), np.eye(3))

    assert slots[0]["id"] == "A01"
    assert np.allclose(
        slots[0]["polygon"],
        [[10, 20], [30, 20], [30, 40], [10, 40]],
    )


def test_build_command_creates_two_transforms_overlap_and_preview(tmp_path):
    workspace = tmp_path / "calibration"
    workspace.mkdir()
    (workspace / "capture_manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "cameras": {
            "cam1": {"image": "capture_cam1.png", "width": 100, "height": 100},
            "cam2": {"image": "capture_cam2.png", "width": 100, "height": 100},
        },
    }), encoding="utf-8")
    assert cv2.imwrite(
        str(workspace / "capture_cam1.png"),
        np.full((100, 100, 3), (40, 120, 200), dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(workspace / "capture_cam2.png"),
        np.full((100, 100, 3), (180, 100, 30), dtype=np.uint8),
    )
    rows = []
    for camera, labels, offset in (
        ("cam1", "ABCD", 0.0),
        ("cam2", "EFGH", 5.0),
    ):
        for label, (pixel_x, pixel_y) in zip(
            labels, ((0, 0), (100, 0), (100, 100), (0, 100))
        ):
            rows.append({
                "camera": camera,
                "label": label,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "world_x_cm": offset + pixel_x / 10.0,
                "world_y_cm": pixel_y / 10.0,
            })
    with (workspace / "calibration_points.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    coverage_cam1 = tmp_path / "coverage_cam1.json"
    coverage_cam2 = tmp_path / "coverage_cam2.json"
    coverage_cam1.write_text(json.dumps({
        "polygon": [[60, 0], [99, 0], [99, 99], [60, 99]],
    }), encoding="utf-8")
    coverage_cam2.write_text(json.dumps({
        "polygon": [[20, 0], [60, 0], [60, 99], [20, 99]],
    }), encoding="utf-8")

    destination = tmp_path / "shared_map.json"
    build_command(argparse.Namespace(
        workspace=workspace,
        output=destination,
        coverage_cam1=coverage_cam1,
        coverage_cam2=coverage_cam2,
        ransac_threshold_cm=2.0,
        max_rms_error_cm=3.0,
        allow_high_error=False,
        handoff_match_distance_cm=15.0,
        handoff_prediction_radius_cm=25.0,
        dormant_match_distance_cm=35.0,
        overwrite=False,
    ))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    overlap = np.asarray(payload["overlap_world_polygon"])
    assert payload["world"]["unit"] == "cm"
    assert payload["tracking_defaults"]["shared_map_anchor"] == "bbox_center"
    assert set(payload["camera_transforms"]) == {"cam1", "cam2"}
    assert set(payload["camera_full_view_world"]) == {"cam1", "cam2"}
    assert np.isclose(overlap[:, 0].min(), 7.0, atol=0.1)
    assert np.isclose(overlap[:, 0].max(), 9.9, atol=0.1)
    quality = payload["calibration_quality"]
    assert quality["full_view_overlap_area_cm2"] > quality["active_roi_overlap_area_cm2"]
    assert (workspace / "shared_map_full_view.png").is_file()
    assert (workspace / "shared_map_active_roi.png").is_file()
    assert (workspace / "shared_map_preview.png").is_file()
