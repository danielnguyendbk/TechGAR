"""Live two-camera parking ROI editor used by ``two_camera.py``."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


VIEW_WIDTH = 640
VIEW_HEIGHT = 360
GUIDE_WIDTH = 380
MAIN_WINDOW = "2 Cameras - Tracking + Parking"


def _order_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape((-1, 2))
    center = np.mean(values, axis=0)
    angles = np.arctan2(values[:, 1] - center[1], values[:, 0] - center[0])
    ordered = values[np.argsort(angles)]
    top_left = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -top_left, axis=0).astype(np.int32)


@dataclass
class EditableSlot:
    slot_id: str
    polygon: np.ndarray
    payload: dict = field(default_factory=dict)

    def clone(self) -> "EditableSlot":
        return EditableSlot(self.slot_id, self.polygon.copy(), copy.deepcopy(self.payload))


@dataclass
class CameraROIState:
    camera_id: str
    path: Path
    frame_size: tuple[int, int]
    root_payload: dict
    slots: list[EditableSlot]
    row: str = "A"
    selected_index: Optional[int] = None
    adding: bool = False
    pending: list[tuple[int, int]] = field(default_factory=list)
    dirty: bool = False
    undo_stack: list[list[EditableSlot]] = field(default_factory=list)


class LiveROIEditor:
    """Edit ROI polygons while the camera processing loop keeps running."""

    def __init__(
        self,
        slot_files: dict[str, str | Path],
        frame_sizes: dict[str, tuple[int, int]],
    ) -> None:
        self.states = {
            camera_id: self._load_state(camera_id, Path(slot_files[camera_id]), frame_sizes[camera_id])
            for camera_id in ("cam1", "cam2")
        }
        self.enabled = False
        self.active_camera = "cam1"
        self.frozen = {"cam1": False, "cam2": False}
        self._frozen_views: dict[str, Optional[np.ndarray]] = {"cam1": None, "cam2": None}
        self._dragging: Optional[tuple[str, int, int]] = None
        self._drag_moved = False

    @staticmethod
    def _load_state(
        camera_id: str,
        path: Path,
        frame_size: tuple[int, int],
    ) -> CameraROIState:
        data = json.loads(path.read_text(encoding="utf-8"))
        frame_width, frame_height = frame_size
        ref_width = max(1, int(data.get("imageWidth", frame_width)))
        ref_height = max(1, int(data.get("imageHeight", frame_height)))
        scale_x = frame_width / ref_width
        scale_y = frame_height / ref_height
        slots = []
        for item in data.get("slots", []):
            polygon = item.get("polygon") or item.get("points") or []
            if len(polygon) < 3:
                continue
            try:
                points = np.asarray([
                    (
                        round(float(point["x"]) * scale_x),
                        round(float(point["y"]) * scale_y),
                    )
                    if isinstance(point, dict)
                    else (
                        round(float(point[0]) * scale_x),
                        round(float(point[1]) * scale_y),
                    )
                    for point in polygon
                ], dtype=np.int32)
            except (KeyError, TypeError, ValueError):
                continue
            slots.append(EditableSlot(str(item.get("id", f"A{len(slots) + 1:02d}")), points, copy.deepcopy(item)))

        rows = [match.group(1) for slot in slots if (match := re.fullmatch(r"([A-Z])(\d+)", slot.slot_id.upper()))]
        row = rows[-1] if rows else "A"
        return CameraROIState(camera_id, path, frame_size, copy.deepcopy(data), slots, row=row)

    @property
    def state(self) -> CameraROIState:
        return self.states[self.active_camera]

    @property
    def dirty_camera_ids(self) -> set[str]:
        return {camera_id for camera_id, state in self.states.items() if state.dirty}

    def _push_undo(self, state: CameraROIState) -> None:
        state.undo_stack.append([slot.clone() for slot in state.slots])
        if len(state.undo_stack) > 30:
            state.undo_stack.pop(0)

    @staticmethod
    def _source_point(state: CameraROIState, display_x: int, display_y: int) -> tuple[int, int]:
        frame_width, frame_height = state.frame_size
        x = int(round(display_x * frame_width / VIEW_WIDTH))
        y = int(round(display_y * frame_height / VIEW_HEIGHT))
        return min(frame_width - 1, max(0, x)), min(frame_height - 1, max(0, y))

    @staticmethod
    def _display_polygon(state: CameraROIState, polygon: np.ndarray) -> np.ndarray:
        frame_width, frame_height = state.frame_size
        points = polygon.astype(np.float32).copy()
        points[:, 0] *= VIEW_WIDTH / frame_width
        points[:, 1] *= VIEW_HEIGHT / frame_height
        return np.rint(points).astype(np.int32)

    def _camera_at(self, x: int, y: int) -> Optional[tuple[str, int, int]]:
        if y < 0 or y >= VIEW_HEIGHT:
            return None
        if 0 <= x < VIEW_WIDTH:
            return "cam1", x, y
        if VIEW_WIDTH <= x < VIEW_WIDTH * 2:
            return "cam2", x - VIEW_WIDTH, y
        return None

    def _nearest_vertex(self, state: CameraROIState, x: int, y: int) -> Optional[tuple[int, int]]:
        best = None
        best_distance = 13.0
        for slot_index, slot in enumerate(state.slots):
            points = self._display_polygon(state, slot.polygon)
            for vertex_index, point in enumerate(points):
                distance = float(np.hypot(float(point[0] - x), float(point[1] - y)))
                if distance < best_distance:
                    best = (slot_index, vertex_index)
                    best_distance = distance
        return best

    @staticmethod
    def _slot_at(state: CameraROIState, point: tuple[int, int]) -> Optional[int]:
        for index in range(len(state.slots) - 1, -1, -1):
            contour = state.slots[index].polygon.astype(np.float32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return index
        return None

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _parameter=None) -> None:
        if not self.enabled:
            return
        target = self._camera_at(x, y)
        if target is None:
            return
        camera_id, local_x, local_y = target
        self.active_camera = camera_id
        state = self.states[camera_id]
        source_point = self._source_point(state, local_x, local_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if state.adding:
                if len(state.pending) < 4:
                    state.pending.append(source_point)
                return
            nearest = self._nearest_vertex(state, local_x, local_y)
            if nearest is not None:
                slot_index, vertex_index = nearest
                self._push_undo(state)
                state.selected_index = slot_index
                self._dragging = (camera_id, slot_index, vertex_index)
                self._drag_moved = False
                return
            state.selected_index = self._slot_at(state, source_point)

        elif event == cv2.EVENT_MOUSEMOVE and self._dragging is not None:
            drag_camera, slot_index, vertex_index = self._dragging
            if drag_camera != camera_id:
                return
            state.slots[slot_index].polygon[vertex_index] = source_point
            state.dirty = True
            self._drag_moved = True

        elif event == cv2.EVENT_LBUTTONUP and self._dragging is not None:
            drag_camera, slot_index, _ = self._dragging
            if drag_camera == camera_id and self._drag_moved:
                state.slots[slot_index].polygon = _order_points(state.slots[slot_index].polygon)
            elif drag_camera == camera_id and state.undo_stack:
                state.undo_stack.pop()
            self._dragging = None
            self._drag_moved = False

    def _next_slot_id(self, state: CameraROIState) -> str:
        pattern = re.compile(rf"^{re.escape(state.row)}(\d+)$", re.IGNORECASE)
        highest = 0
        for slot in state.slots:
            match = pattern.fullmatch(slot.slot_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"{state.row}{highest + 1:02d}"

    def handle_key(self, key: int) -> bool:
        """Handle an editor key. Return True when the key was consumed."""
        if key in (ord("e"), ord("E")):
            self.enabled = not self.enabled
            self._dragging = None
            return True
        if not self.enabled:
            return False
        if key == ord("1"):
            self.active_camera = "cam1"
        elif key == ord("2"):
            self.active_camera = "cam2"
        elif key == ord(" "):
            camera_id = self.active_camera
            self.frozen[camera_id] = not self.frozen[camera_id]
            self._frozen_views[camera_id] = None
        elif key in (ord("n"), ord("N")):
            self.state.adding = True
            self.state.pending.clear()
            self.state.selected_index = None
        elif key in (ord("a"), ord("A")):
            state = self.state
            if state.adding and len(state.pending) == 4:
                self._push_undo(state)
                slot_id = self._next_slot_id(state)
                polygon = _order_points(np.asarray(state.pending, dtype=np.int32))
                payload = {"id": slot_id, "type": "polygon", "status": "empty"}
                state.slots.append(EditableSlot(slot_id, polygon, payload))
                state.selected_index = len(state.slots) - 1
                state.pending.clear()
                state.adding = False
                state.dirty = True
        elif key in (ord("d"), ord("D")):
            self.state.pending.clear()
            self.state.adding = False
        elif key in (ord("x"), ord("X")):
            state = self.state
            if state.selected_index is not None:
                self._push_undo(state)
                state.slots.pop(state.selected_index)
                state.selected_index = None
                state.dirty = True
        elif key in (ord("z"), ord("Z")):
            state = self.state
            if state.undo_stack:
                state.slots = state.undo_stack.pop()
                state.selected_index = None
                state.pending.clear()
                state.adding = False
                state.dirty = True
        elif key in (ord("o"), ord("O"), ord("p"), ord("P")):
            state = self.state
            direction = 1 if key in (ord("o"), ord("O")) else -1
            state.row = chr((ord(state.row) - ord("A") + direction) % 26 + ord("A"))
        else:
            return False
        return True

    def save(self) -> set[str]:
        saved = set()
        for camera_id, state in self.states.items():
            if not state.dirty:
                continue
            frame_width, frame_height = state.frame_size
            payload = copy.deepcopy(state.root_payload)
            serialized_slots = []
            for slot in state.slots:
                item = copy.deepcopy(slot.payload)
                points = _order_points(slot.polygon)
                center = np.mean(points, axis=0)
                item.update({
                    "id": slot.slot_id,
                    "type": "polygon",
                    "polygon": [{"x": int(x), "y": int(y)} for x, y in points],
                    "center": {"x": int(round(center[0])), "y": int(round(center[1]))},
                })
                serialized_slots.append(item)
                slot.polygon = points
            payload["imageWidth"] = frame_width
            payload["imageHeight"] = frame_height
            payload["slots"] = serialized_slots
            state.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = state.path.with_suffix(state.path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(state.path)
            state.root_payload = copy.deepcopy(payload)
            state.dirty = False
            saved.add(camera_id)
        return saved

    def render_camera(self, camera_id: str, base_view: np.ndarray, slot_results: list) -> np.ndarray:
        state = self.states[camera_id]
        if self.frozen[camera_id]:
            if self._frozen_views[camera_id] is None:
                self._frozen_views[camera_id] = base_view.copy()
            view = self._frozen_views[camera_id].copy()
        else:
            self._frozen_views[camera_id] = None
            view = base_view.copy()

        results = {str(result.slot_id): result for result in slot_results}
        for index, slot in enumerate(state.slots):
            points = self._display_polygon(state, slot.polygon)
            result = results.get(slot.slot_id)
            color = (0, 0, 255) if result is not None and result.occupied else (0, 255, 0)
            cv2.polylines(view, [points], True, color, 2)
            center = tuple(np.mean(points, axis=0).astype(int))
            label = slot.slot_id
            if result is not None and result.vehicle_id is not None:
                label += f" #{result.vehicle_id}"
            cv2.putText(view, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            if self.enabled and camera_id == self.active_camera and index == state.selected_index:
                cv2.polylines(view, [points], True, (0, 255, 255), 3)
                for vertex in points:
                    cv2.circle(view, tuple(vertex), 5, (0, 255, 255), -1)

        if self.enabled and camera_id == self.active_camera and state.pending:
            pending = self._display_polygon(state, np.asarray(state.pending, dtype=np.int32))
            for index, point in enumerate(pending):
                cv2.circle(view, tuple(point), 5, (255, 255, 0), -1)
                if index:
                    cv2.line(view, tuple(pending[index - 1]), tuple(point), (255, 255, 0), 2)
            if len(pending) == 4:
                cv2.line(view, tuple(pending[-1]), tuple(pending[0]), (255, 255, 0), 2)
        return view

    def render_guide(self) -> np.ndarray:
        panel = np.full((VIEW_HEIGHT, GUIDE_WIDTH, 3), (28, 30, 32), dtype=np.uint8)
        state = self.state
        mode = "EDIT ROI" if self.enabled else "LIVE TRACKING"
        stream = "FROZEN VIEW" if self.frozen[self.active_camera] else "LIVE VIEW"
        dirty = ", ".join(camera_id.upper() for camera_id in sorted(self.dirty_camera_ids)) or "none"
        lines = [
            ("TWO-CAMERA ROI CONTROLS", (0, 220, 255), 0.52),
            (f"Mode: {mode}", (255, 255, 255), 0.47),
            (f"Active: {self.active_camera.upper()} | {stream}", (255, 255, 255), 0.47),
            (f"Row: {state.row} | Next: {self._next_slot_id(state)}", (255, 255, 255), 0.47),
            (f"Unsaved: {dirty}", (0, 200, 255), 0.47),
            ("", (255, 255, 255), 0.44),
            ("E       Edit on/off", (210, 210, 210), 0.44),
            ("1 / 2   Select camera", (210, 210, 210), 0.44),
            ("Mouse   Select or drag yellow point", (210, 210, 210), 0.44),
            ("N       New ROI, then click 4 points", (210, 210, 210), 0.44),
            ("A / D   Accept / cancel new ROI", (210, 210, 210), 0.44),
            ("X / Z   Delete selected / undo", (210, 210, 210), 0.44),
            ("O / P   Next / previous row", (210, 210, 210), 0.44),
            ("Space   Freeze selected view", (210, 210, 210), 0.44),
            ("S       Save ROI + detector settings", (210, 210, 210), 0.44),
            ("Q       Save ROI and quit", (210, 210, 210), 0.44),
            ("Esc     Quit without ROI save", (210, 210, 210), 0.44),
            ("Tracking + Global ID keep running", (80, 220, 120), 0.44),
        ]
        y = 22
        for text, color, scale in lines:
            if text:
                cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 19
        return panel

    def compose_main_view(self, camera_views: dict[str, np.ndarray]) -> np.ndarray:
        views = [cv2.resize(camera_views[camera_id], (VIEW_WIDTH, VIEW_HEIGHT)) for camera_id in ("cam1", "cam2")]
        return np.hstack([views[0], views[1], self.render_guide()])
