"""Tests for Phase C ONNX ReID extractor, fallback contract, and appearance gallery invariants."""

import numpy as np
import pytest

from techgar.appearance import AppearanceGallery, cosine_distance
from techgar.reid_onnx import ONNXReIDExtractor, get_default_reid_extractor
from techgar.states import LifecycleState
from conftest import Rig, _rect


class TestONNXReIDFallback:
    """Test fallback contract when model is not provided or missing."""

    def test_fallback_when_no_model_provided(self):
        extractor = ONNXReIDExtractor(None)
        assert extractor.status == "fallback_color"
        assert extractor.embedding_dim == 27

        # Extract on dummy 64x64 RGB image
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        feat = extractor.extract(img, (10, 10, 50, 50))
        assert feat.shape == (27,)
        assert abs(np.linalg.norm(feat) - 1.0) < 1e-4

    def test_fallback_when_model_file_missing(self, tmp_path):
        fake_model = tmp_path / "non_existent.onnx"
        extractor = ONNXReIDExtractor(fake_model)
        assert extractor.status == "fallback_color"
        assert extractor.embedding_dim == 27

    def test_cosine_distance_bounds(self):
        # Identical vectors -> 0.0
        v1 = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_distance(v1, v1) - 0.0) < 1e-5

        # Orthogonal vectors -> 1.0
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_distance(v1, v2) - 1.0) < 1e-5

        # Opposite vectors -> 2.0
        v3 = np.array([-1.0, 0.0, 0.0])
        assert abs(cosine_distance(v1, v3) - 2.0) < 1e-5


class TestGalleryFreezeInvariants:
    """Gallery updates and freeze invariants during park and unpark."""

    def test_gallery_does_not_update_when_frozen(self):
        gallery = AppearanceGallery()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Before freeze: add succeeds
        assert gallery.add(v, timestamp=1.0) is True
        assert len(gallery.samples) == 1

        # Freeze as parked
        gallery.freeze("parked")
        assert gallery.frozen is True
        assert gallery.frozen_reason == "parked"

        # While parked: adding sample is rejected
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert gallery.add(v2, timestamp=2.0) is False
        assert len(gallery.samples) == 1

        # Unfreeze
        gallery.unfreeze()
        assert gallery.frozen is False
        assert gallery.add(v2, timestamp=3.0) is True
        assert len(gallery.samples) == 2

    def test_registry_freezes_gallery_on_park_and_unfreezes_on_unpark(self, topology):
        rig = Rig(topology)
        rig.drive_n([(10, 30), (10.8, 30), (11.6, 30)], "cam1")
        gid = rig.single_live_gid()
        state = rig.registry.get(gid)
        assert state.appearance_gallery.frozen is False

        # Park vehicle
        rig.registry.mark_parked(gid, "A01", timestamp=5.0, frame_sequence=50)
        assert state.lifecycle_state == LifecycleState.PARKED
        assert state.appearance_gallery.frozen is True
        assert state.appearance_gallery.frozen_reason == "parked"
