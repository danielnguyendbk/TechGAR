import json
from types import SimpleNamespace

import cv2
import numpy as np

from techgar.live_roi_editor import GUIDE_WIDTH, LiveROIEditor, VIEW_HEIGHT, VIEW_WIDTH
from techgar.slot_vehicle_binder import SlotVehicleBinder


def write_slots(path, slot_id="A01"):
    path.write_text(json.dumps({
        "imageWidth": 100,
        "imageHeight": 80,
        "slots": [{
            "id": slot_id,
            "type": "polygon",
            "polygon": [
                {"x": 10, "y": 10},
                {"x": 40, "y": 10},
                {"x": 40, "y": 60},
                {"x": 10, "y": 60},
            ],
            "center": {"x": 25, "y": 35},
        }],
    }), encoding="utf-8")


def make_editor(tmp_path):
    cam1 = tmp_path / "cam1.json"
    cam2 = tmp_path / "cam2.json"
    write_slots(cam1)
    write_slots(cam2)
    editor = LiveROIEditor(
        {"cam1": cam1, "cam2": cam2},
        {"cam1": (100, 80), "cam2": (100, 80)},
    )
    return editor, cam1, cam2


def test_drag_vertex_saves_source_coordinates(tmp_path):
    editor, cam1, _ = make_editor(tmp_path)
    editor.handle_key(ord("e"))

    start_x = round(10 * VIEW_WIDTH / 100)
    start_y = round(10 * VIEW_HEIGHT / 80)
    target_x = round(20 * VIEW_WIDTH / 100)
    target_y = round(20 * VIEW_HEIGHT / 80)
    editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, start_x, start_y, 0)
    editor.handle_mouse(cv2.EVENT_MOUSEMOVE, target_x, target_y, 0)
    editor.handle_mouse(cv2.EVENT_LBUTTONUP, target_x, target_y, 0)

    assert editor.save() == {"cam1"}
    saved = json.loads(cam1.read_text(encoding="utf-8"))
    saved_points = {(point["x"], point["y"]) for point in saved["slots"][0]["polygon"]}
    assert (20, 20) in saved_points
    assert saved["imageWidth"] == 100
    assert saved["imageHeight"] == 80


def test_add_roi_to_selected_camera_and_row(tmp_path):
    editor, cam1, cam2 = make_editor(tmp_path)
    editor.handle_key(ord("e"))
    editor.handle_key(ord("2"))
    editor.handle_key(ord("o"))
    editor.handle_key(ord("n"))
    for source_x, source_y in ((55, 10), (90, 10), (90, 60), (55, 60)):
        display_x = VIEW_WIDTH + round(source_x * VIEW_WIDTH / 100)
        display_y = round(source_y * VIEW_HEIGHT / 80)
        editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, display_x, display_y, 0)
    editor.handle_key(ord("a"))

    assert editor.save() == {"cam2"}
    assert len(json.loads(cam1.read_text(encoding="utf-8"))["slots"]) == 1
    saved_cam2 = json.loads(cam2.read_text(encoding="utf-8"))
    assert [slot["id"] for slot in saved_cam2["slots"]] == ["A01", "B01"]


def test_guide_panel_is_outside_both_camera_views(tmp_path):
    editor, _, _ = make_editor(tmp_path)
    cam1 = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), (10, 20, 30), dtype=np.uint8)
    cam2 = np.full((VIEW_HEIGHT, VIEW_WIDTH, 3), (40, 50, 60), dtype=np.uint8)

    output = editor.compose_main_view({"cam1": cam1, "cam2": cam2})

    assert output.shape == (VIEW_HEIGHT, VIEW_WIDTH * 2 + GUIDE_WIDTH, 3)
    assert tuple(output[200, 300]) == (10, 20, 30)
    assert tuple(output[200, VIEW_WIDTH + 300]) == (40, 50, 60)
    assert not np.array_equal(output[:, VIEW_WIDTH * 2:], cam1[:, :GUIDE_WIDTH])


def test_retain_slot_ids_keeps_existing_binding_and_removes_deleted_slot():
    binder = SlotVehicleBinder()
    polygon = np.asarray([[0, 0], [20, 0], [20, 30], [0, 30]], dtype=np.int32)
    results = [
        SimpleNamespace(slot_id="A01", polygon=polygon, center=(10, 15), occupied=False, vehicle_id=None),
        SimpleNamespace(slot_id="A02", polygon=polygon + 30, center=(40, 45), occupied=False, vehicle_id=None),
    ]
    binder.update_vision(results, frame_idx=1, timestamp_s=0.1, camera_id="cam1")
    existing = binder._bindings["A01"]

    binder.retain_slot_ids({"A01"})

    assert binder._bindings == {"A01": existing}
