"""Vision occupancy detector + slot-engine fusion tests.

The regression this guards against: a parked (motionless) vehicle keeps its
slot OCCUPIED on pixel-content evidence alone, even when tracking has lost
the identity — the exact web complaint "không đổi màu ở các ô có xe khi xe
đứng yên".
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from techgar.config_world import SlotConfig
from techgar.slot_engine import (SlotOccupancyEngine, VehicleFootprintView,
                                 VisionSlotVote)
from techgar.states import SlotOccupancy
from techgar.vision_occupancy import (VisionOccupancyDetector, VisionSlotConfig,
                                      merge_camera_votes)


FRAME_H, FRAME_W = 240, 320
SLOT_W, SLOT_H = 80, 80


def slot_poly(cx: float, cy: float) -> np.ndarray:
    return np.array([
        [cx - SLOT_W / 2, cy - SLOT_H / 2], [cx + SLOT_W / 2, cy - SLOT_H / 2],
        [cx + SLOT_W / 2, cy + SLOT_H / 2], [cx - SLOT_W / 2, cy + SLOT_H / 2],
    ], dtype=np.int32)


def empty_floor(seed: int = 7) -> np.ndarray:
    """A flat, slightly textured floor with no vehicles."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 3.0, (FRAME_H, FRAME_W, 3)).astype(np.float32)
    return np.clip(70.0 + noise, 0, 255).astype(np.uint8)


def floor_with_vehicle(cx: float, cy: float, seed: int = 7) -> np.ndarray:
    """The same floor with a high-contrast vehicle rectangle in one slot."""
    frame = empty_floor(seed)
    x0 = int(cx - SLOT_W * 0.38)
    x1 = int(cx + SLOT_W * 0.38)
    y0 = int(cy - SLOT_H * 0.38)
    y1 = int(cy + SLOT_H * 0.38)
    frame[y0:y1, x0:x1] = (35, 150, 60)          # saturated green vehicle
    return frame


@pytest.fixture
def detector() -> VisionOccupancyDetector:
    return VisionOccupancyDetector(
        {"A01": slot_poly(100, 120), "A02": slot_poly(200, 120)},
        VisionSlotConfig(warmup_frames=3, confirm_frames=2))


class TestVisionOccupancyDetector:

    def test_empty_floor_votes_empty(self, detector: VisionOccupancyDetector):
        for _ in range(10):
            votes = detector.detect(empty_floor())
        assert votes["A01"].ready
        assert not votes["A01"].occupied
        assert not votes["A02"].occupied

    def test_vehicle_in_slot_votes_occupied(self, detector: VisionOccupancyDetector):
        # Warm up on the empty floor first.
        for _ in range(6):
            detector.detect(empty_floor())
        # A vehicle appears in A01.
        for _ in range(5):
            votes = detector.detect(floor_with_vehicle(100, 120))
        assert votes["A01"].occupied, (
            f"evidence={votes['A01'].evidence:.3f} — a motionless vehicle in the "
            "polygon must be vision-occupied")
        assert not votes["A02"].occupied

    def test_vehicle_at_startup_is_detected(self):
        """A vehicle parked before the system even starts (no motion ever seen)."""
        detector = VisionOccupancyDetector(
            {"A01": slot_poly(100, 120)},
            VisionSlotConfig(warmup_frames=3, confirm_frames=2))
        # Frames from the very beginning already contain the parked vehicle.
        for _ in range(8):
            votes = detector.detect(floor_with_vehicle(100, 120))
        assert votes["A01"].occupied, "startup-parked vehicle must be detected"

    def test_vehicle_departure_votes_empty_again(self, detector: VisionOccupancyDetector):
        for _ in range(6):
            detector.detect(empty_floor())
        for _ in range(5):
            detector.detect(floor_with_vehicle(100, 120))
        assert detector.detect(empty_floor())["A01"].occupied  # hysteresis hold
        for _ in range(6):
            votes = detector.detect(empty_floor())
        assert not votes["A01"].occupied

    def test_warmup_not_ready_before_reference(self):
        detector = VisionOccupancyDetector(
            {"A01": slot_poly(100, 120)}, VisionSlotConfig(warmup_frames=5))
        votes = detector.detect(empty_floor())
        assert not votes["A01"].ready


class TestMergeCameraVotes:

    def test_or_merge_across_cameras(self):
        per_camera = [
            {"A01": VisionOccupancyDetector.__dict__},  # placeholder, use manual
        ]
        # Build manual vote dicts instead.
        from techgar.vision_occupancy import VisionOccupancyResult
        cam1 = {"A01": VisionOccupancyResult("A01", False, 0.05, True)}
        cam2 = {"A01": VisionOccupancyResult("A01", True, 0.55, True)}
        merged = merge_camera_votes([cam1, cam2])
        assert merged["A01"].occupied


class TestSlotEngineVisionFusion:

    def make_engine(self) -> SlotOccupancyEngine:
        return SlotOccupancyEngine(
            {"A01": np.array([[60, 80], [140, 80], [140, 160], [60, 160]], dtype=float),
             "A02": np.array([[160, 80], [240, 80], [240, 160], [160, 160]], dtype=float)},
            SlotConfig(vision_confirm_frames=2, vision_release_frames=3))

    def feed_vision(self, engine: SlotOccupancyEngine, a01: bool, a02: bool,
                    timestamp: float) -> None:
        engine.update_vision({
            "A01": VisionSlotVote("A01", a01, 0.6 if a01 else 0.05, True),
            "A02": VisionSlotVote("A02", a02, 0.6 if a02 else 0.05, True),
        }, timestamp)

    def test_vision_marks_slot_occupied_without_identity(self):
        """The web bug: slot colour must change even when no Global ID owns it."""
        engine = self.make_engine()
        for frame in range(5):
            self.feed_vision(engine, a01=True, a02=False, timestamp=0.1 * frame)
            engine.update([], timestamp=0.1 * frame, frame_sequence=frame)
        state = engine.states["A01"]
        assert state.occupancy_state is SlotOccupancy.OCCUPIED
        assert state.owning_global_id is None     # anonymous vision occupancy
        assert state.vision_occupied

    def test_vision_keeps_parked_slot_after_tracking_lost(self):
        """Tracking confirms, then disappears; vision holds the colour."""
        engine = self.make_engine()
        # Tracking-driven confirmation path first (footprint evidence).
        def vehicle(gid, cx, cy, velocity=(0.0, 0.0)):
            return VehicleFootprintView(
                global_id=gid, footprint=np.array(
                    [[cx - 20, cy - 20], [cx + 20, cy - 20],
                     [cx + 20, cy + 20], [cx - 20, cy + 20]], dtype=float),
                position=np.array([cx, cy]), velocity=np.asarray(velocity, dtype=float))
        timestamp = 0.0
        approach = [(70, 120), (85, 120), (95, 120), (100, 120)]
        for i, (x, y) in enumerate(approach):
            timestamp = 0.1 * i
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([vehicle(17, x, y, velocity=(0.5, 0.0))],
                          timestamp=timestamp, frame_sequence=i)
        for i in range(12):
            timestamp += 0.1
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([vehicle(17, 100, 120)], timestamp=timestamp,
                          frame_sequence=int(timestamp * 10))
        assert engine.owner_of("A01") == 17
        # Tracking now disappears completely — vision alone must hold the slot.
        for i in range(20):
            timestamp += 0.1
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([], timestamp=timestamp, frame_sequence=int(timestamp * 10))
        state = engine.states["A01"]
        assert state.occupancy_state is SlotOccupancy.OCCUPIED, (
            "the parked vehicle lost tracking; vision must keep the slot red")

    def test_vision_vetoes_release_of_confirmed_owner(self):
        """Owner's footprint evidence decays, but vision still sees the vehicle."""
        engine = self.make_engine()
        def vehicle(gid, cx, cy):
            return VehicleFootprintView(
                global_id=gid, footprint=np.array(
                    [[cx - 20, cy - 20], [cx + 20, cy - 20],
                     [cx + 20, cy + 20], [cx - 20, cy + 20]], dtype=float),
                position=np.array([cx, cy]), velocity=np.zeros(2))
        timestamp = 0.0
        approach = [(70, 120), (85, 120), (95, 120), (100, 120)]
        for i, (x, y) in enumerate(approach):
            timestamp = 0.1 * i
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([vehicle(17, x, y)], timestamp=timestamp, frame_sequence=i)
        for i in range(12):
            timestamp += 0.1
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([vehicle(17, 100, 120)], timestamp=timestamp,
                          frame_sequence=int(timestamp * 10))
        assert engine.owner_of("A01") == 17
        # The vehicle is *observed* but its footprint drifted off-slot
        # (tracking noisy) while vision still sees it inside A01.
        for i in range(8):
            timestamp += 0.1
            self.feed_vision(engine, a01=True, a02=False, timestamp=timestamp)
            engine.update([vehicle(17, 160, 120)], timestamp=timestamp,
                          frame_sequence=int(timestamp * 10))
        state = engine.states["A01"]
        assert state.occupancy_state is SlotOccupancy.OCCUPIED

    def test_vision_release_of_anonymous_slot(self):
        """An anonymous vision-occupied slot empties when vision says empty."""
        engine = self.make_engine()
        for frame in range(5):
            self.feed_vision(engine, a01=True, a02=False, timestamp=0.1 * frame)
            engine.update([], timestamp=0.1 * frame, frame_sequence=frame)
        assert engine.states["A01"].occupancy_state is SlotOccupancy.OCCUPIED
        # Vehicle leaves before tracking ever saw it.
        timestamp = 0.5
        for frame in range(6):
            timestamp += 0.1
            self.feed_vision(engine, a01=False, a02=False, timestamp=timestamp)
            engine.update([], timestamp=timestamp, frame_sequence=100 + frame)
        assert engine.states["A01"].occupancy_state is SlotOccupancy.EMPTY

    def test_vision_never_assigns_identity(self):
        engine = self.make_engine()
        for frame in range(5):
            self.feed_vision(engine, a01=True, a02=False, timestamp=0.1 * frame)
            engine.update([], timestamp=0.1 * frame, frame_sequence=frame)
        assert engine.states["A01"].owning_global_id is None

    def test_vision_disabled_keeps_tracking_only(self):
        engine = SlotOccupancyEngine(
            {"A01": np.array([[60, 80], [140, 80], [140, 160], [60, 160]], dtype=float)},
            SlotConfig(enable_vision_fusion=False, vision_confirm_frames=1))
        engine.update_vision({"A01": VisionSlotVote("A01", True, 0.9, True)}, 0.1)
        engine.update([], timestamp=0.1, frame_sequence=1)
        assert engine.states["A01"].occupancy_state is SlotOccupancy.EMPTY
