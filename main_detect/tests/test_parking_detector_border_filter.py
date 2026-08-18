import json

import cv2
import numpy as np

from techgar.parking_detector import ParkingDetector
from two_camera import apply_detector_parameters, detector_parameters


def make_detector(tmp_path, **overrides):
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(json.dumps({
        "imageWidth": 100,
        "imageHeight": 100,
        "slots": [{
            "id": "B01",
            "polygon": [
                {"x": 10, "y": 10},
                {"x": 90, "y": 10},
                {"x": 90, "y": 90},
                {"x": 10, "y": 90},
            ],
        }],
    }), encoding="utf-8")
    detector = ParkingDetector(
        str(slots_path),
        use_edge_recheck=False,
        border_ignore_ratio=0.10,
        line_min_span_ratio=0.45,
        line_max_thickness_ratio=0.18,
        core_scale=0.55,
        core_ratio_threshold=0.18,
        core_component_threshold=0.08,
        **overrides,
    )
    detector._compute_rois((100, 100, 3))
    return detector, detector._rois[0]


def evidence_for(detector, roi, threshold):
    x1, y1, x2, y2 = roi.bbox
    return detector._filter_roi_threshold(threshold[y1:y2, x1:x2], roi)


def test_full_roi_border_is_ignored(tmp_path):
    detector, roi = make_detector(tmp_path)
    threshold = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(threshold, (10, 10), (90, 90), 255, 5)

    evidence = evidence_for(detector, roi, threshold)

    assert evidence.raw_ratio > 0.15
    assert evidence.filtered_ratio == 0.0


def test_thin_vertical_and_diagonal_lines_are_removed(tmp_path):
    detector, roi = make_detector(tmp_path)
    for start, end in (((50, 10), (50, 90)), ((10, 10), (90, 90))):
        threshold = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(threshold, start, end, 255, 3)

        evidence = evidence_for(detector, roi, threshold)

        assert evidence.filtered_ratio == 0.0
        assert evidence.core_ratio == 0.0


def test_center_vehicle_blob_is_kept(tmp_path):
    detector, roi = make_detector(tmp_path)
    threshold = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(threshold, (34, 30), (66, 72), 255, -1)

    evidence = evidence_for(detector, roi, threshold)

    assert evidence.filtered_ratio > detector.ratio_thr
    assert evidence.core_ratio > detector.core_ratio_threshold
    assert evidence.core_component_ratio > detector.core_component_threshold
    assert detector._has_core_rescue(evidence) is True


def test_vehicle_blob_connected_to_border_line_is_kept(tmp_path):
    detector, roi = make_detector(tmp_path)
    threshold = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(threshold, (50, 10), (50, 42), 255, 3)
    cv2.rectangle(threshold, (34, 35), (66, 72), 255, -1)

    evidence = evidence_for(detector, roi, threshold)

    assert evidence.filtered_ratio > detector.ratio_thr
    assert evidence.core_component_ratio > detector.core_component_threshold


def test_dense_textured_core_rescue_keeps_multiple_vehicle_fragments(tmp_path):
    detector, roi = make_detector(tmp_path)
    threshold = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(threshold, (10, 10), (90, 90), 255, 9)
    cv2.rectangle(threshold, (34, 40), (43, 54), 255, -1)
    cv2.rectangle(threshold, (57, 40), (66, 54), 255, -1)

    evidence = evidence_for(detector, roi, threshold)

    assert evidence.raw_ratio >= detector.ratio_thr * 1.5
    assert evidence.core_component_count == 2
    assert detector._has_core_rescue(evidence) is True


def test_two_camera_profile_round_trips_roi_filter_parameters(tmp_path):
    detector, _ = make_detector(tmp_path)
    values = {
        "border_ignore_ratio": 0.14,
        "line_min_span_ratio": 0.50,
        "line_max_thickness_ratio": 0.16,
        "core_scale": 0.60,
        "core_ratio_threshold": 0.21,
        "core_component_threshold": 0.09,
    }

    apply_detector_parameters(detector, values)
    snapshot = detector_parameters(detector)

    for key, value in values.items():
        assert snapshot[key] == value
    assert detector._initialized is False
