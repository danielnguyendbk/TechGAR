import json

import cv2
import numpy as np

from techgar.parking_detector import ParkingDetector
from two_camera import apply_detector_parameters


def test_debug_images_match_frame_and_do_not_change_slot_state(tmp_path):
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(json.dumps({
        "imageWidth": 100,
        "imageHeight": 80,
        "slots": [{
            "id": "A01",
            "polygon": [{"x": 10, "y": 10}, {"x": 45, "y": 10}, {"x": 45, "y": 60}, {"x": 10, "y": 60}],
            "center": {"x": 27, "y": 35},
        }],
    }), encoding="utf-8")
    detector = ParkingDetector(str(slots_path), smoothing_frames=2)
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    frame[20:45, 20:35] = 240

    threshold, edges = detector.build_debug_images(frame)

    assert threshold.shape == frame.shape
    assert edges.shape == frame.shape
    assert detector._smoother.confirmed == [False]


def test_disabled_edge_recheck_cannot_override_pixel_result(tmp_path, monkeypatch):
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(json.dumps({
        "imageWidth": 100,
        "imageHeight": 80,
        "slots": [{
            "id": "A01",
            "polygon": [
                {"x": 10, "y": 10},
                {"x": 45, "y": 10},
                {"x": 45, "y": 60},
                {"x": 10, "y": 60},
            ],
        }],
    }), encoding="utf-8")
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    monkeypatch.setattr(cv2, "Canny", lambda image, _low, _high: np.full_like(image, 255))

    pixel_only = ParkingDetector(str(slots_path), use_edge_recheck=False)
    legacy = ParkingDetector(str(slots_path), use_edge_recheck=True)

    assert pixel_only.use_edge_recheck is False
    assert pixel_only.detect(frame, apply_smoothing=False)[0].occupied is False
    assert legacy.detect(frame, apply_smoothing=False)[0].occupied is True


def test_two_camera_profile_cannot_enable_edge_recheck(tmp_path):
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(json.dumps({
        "imageWidth": 20,
        "imageHeight": 20,
        "slots": [],
    }), encoding="utf-8")
    detector = ParkingDetector(str(slots_path), use_edge_recheck=False)

    apply_detector_parameters(detector, {"use_edge_recheck": True, "edge_thr": 0.01})

    assert detector.use_edge_recheck is False
