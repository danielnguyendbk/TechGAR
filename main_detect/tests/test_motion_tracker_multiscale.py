import cv2
import numpy as np

from techgar.motion_tracker import MotionVehicleTracker
from techgar.vehicle_tracker import TrackStatus


def _frame_with_soft_blob(center_x: int, brightness: int = 30) -> np.ndarray:
    height, width = 96, 144
    yy, xx = np.mgrid[:height, :width]
    blob = 58.0 * np.exp(-(((xx - center_x) ** 2) + ((yy - 48) ** 2)) / (2.0 * 10.0 ** 2))
    gray = np.clip(brightness + blob, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _detection(tracker: MotionVehicleTracker, frame: np.ndarray, x: int, *, priority: bool) -> dict:
    box = (x, 20, 28, 24)
    return {
        "box": box,
        "point": tracker._bottom_center(box),
        "area": float(box[2] * box[3]),
        "hist": tracker._histogram(frame, box),
        "priority": priority,
    }


def test_constructor_preserves_requested_track_history_length():
    tracker = MotionVehicleTracker(history_len=37)

    assert tracker.history_len == 37


def test_constructor_keeps_legacy_detection_defaults_opt_in_only():
    tracker = MotionVehicleTracker()

    assert tracker.history_len == 10
    assert tracker.min_confirm_displacement == 12.0
    assert tracker.motion_threshold == 25
    assert tracker.motion_min_ratio == 0.08
    assert tracker.motion_min_pixels == 160
    assert tracker.enable_multiscale_motion is False
    assert tracker.reject_cast_shadows is False


def test_timestamp_history_selects_short_and_long_references(monkeypatch):
    tracker = MotionVehicleTracker(enable_multiscale_motion=True)
    calls = []

    def record_reference(current, reference, threshold):
        calls.append(int(reference[0, 0]))
        return np.zeros_like(current)

    monkeypatch.setattr(tracker, "_motion_between", record_reference)
    for index in range(9):
        value = index * 10
        frame = np.full((32, 32, 3), value, dtype=np.uint8)
        tracker._temporal_motion_mask(frame, timestamp_s=index * 0.1)

    calls.clear()
    tracker._temporal_motion_mask(np.full((32, 32, 3), 90, dtype=np.uint8), timestamp_s=0.9)

    assert len(calls) == 2
    assert any(value <= 20 for value in calls)  # approximately 0.8 s old
    assert any(50 <= value <= 70 for value in calls)  # approximately 0.25 s old


def test_timestamp_gap_does_not_compare_across_stream_pause(monkeypatch):
    tracker = MotionVehicleTracker()
    calls = []

    def record_reference(current, reference, threshold):
        calls.append(reference)
        return np.full_like(current, 255)

    monkeypatch.setattr(tracker, "_motion_between", record_reference)
    tracker._temporal_motion_mask(np.zeros((32, 32, 3), dtype=np.uint8), timestamp_s=0.0)
    tracker._temporal_motion_mask(np.zeros((32, 32, 3), dtype=np.uint8), timestamp_s=0.2)
    calls.clear()

    mask = tracker._temporal_motion_mask(
        np.full((32, 32, 3), 255, dtype=np.uint8),
        timestamp_s=7.0,
    )

    assert not calls
    assert cv2.countNonZero(mask) == 0


def test_multiscale_mask_detects_a_slow_soft_object():
    tracker = MotionVehicleTracker(
        motion_threshold=20,
        enable_multiscale_motion=True,
    )
    last_mask = None

    for index in range(10):
        last_mask = tracker._temporal_motion_mask(
            _frame_with_soft_blob(40 + index),
            timestamp_s=index * 0.1,
        )

    assert last_mask is not None
    assert cv2.countNonZero(last_mask) > 0


def test_uniform_brightness_change_does_not_create_temporal_motion():
    tracker = MotionVehicleTracker(
        motion_threshold=20,
        enable_multiscale_motion=True,
    )

    masks = [
        tracker._temporal_motion_mask(
            np.full((72, 96, 3), 40 + index * 6, dtype=np.uint8),
            timestamp_s=index * 0.1,
        )
        for index in range(10)
    ]

    assert all(cv2.countNonZero(mask) == 0 for mask in masks)


def test_priority_region_allows_small_blob_but_normal_region_rejects_it(monkeypatch):
    tracker = MotionVehicleTracker(min_area=650, priority_min_area=350)
    foreground = np.zeros((100, 120), dtype=np.uint8)
    foreground[40:60, 30:55] = 255

    class FixedBackground:
        def apply(self, _frame):
            return foreground.copy()

    tracker.bg_sub = FixedBackground()
    monkeypatch.setattr(
        tracker,
        "_temporal_motion_mask",
        lambda _frame, timestamp_s=None: foreground.copy(),
    )
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    normal_detections, _ = tracker._detect(frame)
    priority_detections, _ = tracker._detect(
        frame,
        priority_regions=[[(20, 30), (70, 30), (70, 80), (20, 80)]],
    )

    assert normal_detections == []
    assert len(priority_detections) == 1
    assert priority_detections[0]["priority"] is True


def test_mog_shadow_value_cannot_become_priority_candidate(monkeypatch):
    tracker = MotionVehicleTracker(
        priority_min_area=350,
        reject_cast_shadows=True,
    )
    mog_shadow = np.zeros((100, 120), dtype=np.uint8)
    mog_shadow[40:65, 30:60] = 127  # OpenCV MOG2 detectShadows marker
    temporal_motion = np.zeros_like(mog_shadow)
    temporal_motion[40:65, 30:60] = 255

    class FixedBackground:
        def apply(self, _frame):
            return mog_shadow.copy()

    tracker.bg_sub = FixedBackground()
    monkeypatch.setattr(
        tracker,
        "_temporal_motion_mask",
        lambda _frame, timestamp_s=None: temporal_motion.copy(),
    )

    detections, _ = tracker._detect(
        np.zeros((100, 120, 3), dtype=np.uint8),
        priority_regions=[[(20, 30), (70, 30), (70, 80), (20, 80)]],
    )

    assert detections == []


def _synthetic_background_and_motion(box=(50, 40, 40, 40)):
    yy, xx = np.mgrid[:120, :160]
    floor = (95 + (xx % 13) * 2 + (yy % 9)).astype(np.uint8)
    background = np.stack(
        [
            np.clip(floor.astype(np.int16) - 5, 0, 255),
            floor,
            np.clip(floor.astype(np.int16) + 4, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    foreground = np.zeros(background.shape[:2], dtype=np.uint8)
    x, y, width, height = box
    foreground[y:y + height, x:x + width] = 255
    return background, foreground


class _FixedBackgroundModel:
    def __init__(self, background, foreground):
        self.background = background
        self.foreground = foreground

    def getBackgroundImage(self):
        return self.background.copy()

    def apply(self, _frame):
        return self.foreground.copy()


def test_achromatic_scaled_cast_shadow_is_rejected_even_in_priority_region(monkeypatch):
    box = (50, 40, 40, 40)
    background, foreground = _synthetic_background_and_motion(box)
    frame = background.copy()
    x, y, width, height = box
    frame[y:y + height, x:x + width] = np.clip(
        background[y:y + height, x:x + width].astype(np.float32) * 0.43,
        0,
        255,
    ).astype(np.uint8)
    tracker = MotionVehicleTracker(
        priority_min_area=350,
        reject_cast_shadows=True,
    )
    tracker.bg_sub = _FixedBackgroundModel(background, foreground)
    monkeypatch.setattr(
        tracker,
        "_temporal_motion_mask",
        lambda _frame, timestamp_s=None: foreground.copy(),
    )

    detections, _ = tracker._detect(
        frame,
        priority_regions=[[(40, 30), (105, 30), (105, 95), (40, 95)]],
    )

    assert detections == []
    assert len(tracker.last_shadow_rejections) == 1
    rejection = tracker.last_shadow_rejections[0]
    assert 0.40 <= rejection["attenuation"] <= 0.46
    assert rejection["explained_fraction"] >= 0.90


def test_dark_neutral_physical_object_is_not_rejected_as_scaled_shadow(monkeypatch):
    box = (50, 40, 40, 40)
    background, foreground = _synthetic_background_and_motion(box)
    frame = background.copy()
    x, y, width, height = box
    frame[y:y + height, x:x + width] = 10
    tracker = MotionVehicleTracker(reject_cast_shadows=True)
    tracker.bg_sub = _FixedBackgroundModel(background, foreground)
    monkeypatch.setattr(
        tracker,
        "_temporal_motion_mask",
        lambda _frame, timestamp_s=None: foreground.copy(),
    )

    detections, _ = tracker._detect(frame)

    assert len(detections) == 1
    assert tracker.last_shadow_rejections == []


def test_colored_slow_vehicle_keeps_priority_detection_with_shadow_filter(monkeypatch):
    # Area is below the normal threshold, so this also guards the relaxed
    # slow-car path against accidentally being disabled by shadow rejection.
    box = (50, 40, 25, 20)
    background, foreground = _synthetic_background_and_motion(box)
    frame = background.copy()
    x, y, width, height = box
    frame[y:y + height, x:x + width] = (12, 55, 18)
    tracker = MotionVehicleTracker(
        min_area=650,
        priority_min_area=350,
        reject_cast_shadows=True,
    )
    tracker.bg_sub = _FixedBackgroundModel(background, foreground)
    monkeypatch.setattr(
        tracker,
        "_temporal_motion_mask",
        lambda _frame, timestamp_s=None: foreground.copy(),
    )

    normal_detections, _ = tracker._detect(frame)
    priority_detections, _ = tracker._detect(
        frame,
        priority_regions=[[(40, 30), (90, 30), (90, 80), (40, 80)]],
    )

    assert normal_detections == []
    assert len(priority_detections) == 1
    assert priority_detections[0]["priority"] is True
    assert tracker.last_shadow_rejections == []


def test_priority_noise_needs_two_displaced_observations_to_confirm(monkeypatch):
    tracker = MotionVehicleTracker(
        min_visible_count=3,
        min_confirm_displacement=6,
        priority_min_visible_count=2,
        priority_min_confirm_displacement=3,
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detections = iter([
        [_detection(tracker, frame, 20, priority=True)],
        [_detection(tracker, frame, 24, priority=True)],
    ])
    monkeypatch.setattr(
        tracker,
        "_detect",
        lambda _frame, timestamp_s=None, priority_regions=None: (
            next(detections),
            np.zeros(_frame.shape[:2], dtype=np.uint8),
        ),
    )

    tracks, _mask, _expired = tracker.process_frame(frame, timestamp_s=0.0)
    assert tracks[1].status == TrackStatus.TENTATIVE

    tracks, _mask, _expired = tracker.process_frame(frame, timestamp_s=0.1)
    assert tracks[1].status == TrackStatus.CONFIRMED


def test_non_priority_track_keeps_stricter_confirmation_and_legacy_call(monkeypatch):
    tracker = MotionVehicleTracker(min_visible_count=3, min_confirm_displacement=6)
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    detections = iter([
        [_detection(tracker, frame, 20, priority=False)],
        [_detection(tracker, frame, 24, priority=False)],
    ])
    monkeypatch.setattr(
        tracker,
        "_detect",
        lambda _frame, timestamp_s=None, priority_regions=None: (
            next(detections),
            np.zeros(_frame.shape[:2], dtype=np.uint8),
        ),
    )

    tracker.process_frame(frame)
    tracks, _mask, _expired = tracker.process_frame(frame)

    assert tracks[1].status == TrackStatus.TENTATIVE


def test_stale_lost_track_cannot_capture_a_new_detection():
    tracker = MotionVehicleTracker(reacquire_max_seconds=0.75)
    frame = np.zeros((100, 180, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._current_timestamp_s = 0.0
    tracker._create_or_reid(_detection(tracker, frame, 30, priority=False))
    old = tracker._tracks[1]
    old.status = TrackStatus.LOST
    old.consecutive_invisible_count = 1
    old.last_seen_timestamp_s = 0.0

    tracker._current_timestamp_s = 1.0
    assignments, unmatched_tracks, unmatched_detections = tracker._assign(
        [_detection(tracker, frame, 32, priority=False)]
    )

    assert assignments == []
    assert unmatched_tracks == [1]
    assert unmatched_detections == [0]
    assert any(
        event["type"] == "association_rejected_stale_track"
        for event in tracker.association_events
    )


def test_large_detection_covering_two_tracks_is_frozen():
    tracker = MotionVehicleTracker(merged_detection_area_ratio=1.6)
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._current_timestamp_s = 0.0
    tracker._create_or_reid(_detection(tracker, frame, 50, priority=False))
    tracker._create_or_reid(_detection(tracker, frame, 90, priority=False))
    for item in tracker._tracks.values():
        item.status = TrackStatus.CONFIRMED

    merged = _detection(tracker, frame, 42, priority=False)
    merged["box"] = (42, 16, 90, 35)
    merged["point"] = (87, 51)
    merged["area"] = 2500.0
    merged["bbox_area"] = 3150.0
    tracker._current_timestamp_s = 0.1
    assignments, unmatched_tracks, unmatched_detections = tracker._assign([merged])

    assert assignments == []
    assert set(unmatched_tracks) == {1, 2}
    assert unmatched_detections == []
    assert any(
        event["type"] == "merged_detection_frozen"
        for event in tracker.association_events
    )


def test_stale_track_does_not_freeze_merged_detection_with_live_track():
    tracker = MotionVehicleTracker(
        merged_detection_area_ratio=1.6,
        reacquire_max_seconds=0.75,
    )
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._current_timestamp_s = 0.0
    tracker._create_or_reid(_detection(tracker, frame, 50, priority=False))
    tracker._create_or_reid(_detection(tracker, frame, 90, priority=False))
    stale = tracker._tracks[1]
    live = tracker._tracks[2]
    stale.status = TrackStatus.LOST
    stale.consecutive_invisible_count = 1
    stale.last_seen_timestamp_s = 0.0
    live.status = TrackStatus.CONFIRMED
    live.last_seen_timestamp_s = 1.0

    merged = _detection(tracker, frame, 42, priority=False)
    merged["box"] = (42, 16, 90, 35)
    merged["point"] = (87, 51)
    merged["area"] = 2500.0
    merged["bbox_area"] = 3150.0
    tracker._current_timestamp_s = 1.0
    _assignments, _unmatched_tracks, unmatched_detections = tracker._assign([merged])

    assert unmatched_detections == [0]
    assert not any(
        event["type"] == "merged_detection_frozen"
        for event in tracker.association_events
    )


def test_suspended_parked_track_is_removed_from_assignment():
    tracker = MotionVehicleTracker()
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._create_or_reid(_detection(tracker, frame, 20, priority=False))

    suspended = tracker.suspend_track(1)

    assert suspended is not None
    assert tracker.all_tracks == {}


def test_first_observation_survives_bounded_display_history():
    tracker = MotionVehicleTracker(
        history_len=3,
        min_visible_count=2,
        min_confirm_displacement=20,
        enable_multiscale_motion=True,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._create_or_reid(_detection(tracker, frame, 10, priority=False))
    track = tracker._tracks[1]

    for frame_index, x in enumerate((14, 18, 22, 26), start=2):
        tracker._frame_idx = frame_index
        tracker._apply_detection(track, _detection(tracker, frame, x, priority=False))

    assert len(track.history) == 3
    assert track.first_observation_point == (24, 44)
    assert track.history[0] != track.first_observation_point
    assert track.status == TrackStatus.TENTATIVE

    tracker._frame_idx += 1
    tracker._apply_detection(track, _detection(tracker, frame, 31, priority=False))
    assert track.status == TrackStatus.CONFIRMED


def test_legacy_confirmation_keeps_rolling_history_origin():
    tracker = MotionVehicleTracker(
        history_len=3,
        min_visible_count=2,
        min_confirm_displacement=20,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    tracker._frame_idx = 1
    tracker._create_or_reid(_detection(tracker, frame, 10, priority=False))
    track = tracker._tracks[1]

    for frame_index, x in enumerate((14, 18, 22, 26, 31), start=2):
        tracker._frame_idx = frame_index
        tracker._apply_detection(track, _detection(tracker, frame, x, priority=False))

    assert track.first_observation_point == (24, 44)
    assert track.status == TrackStatus.TENTATIVE
