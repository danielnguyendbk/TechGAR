from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from techgar.replay import (build_replay_pipeline, iter_decoded_pairs, load_replay_site,
                            process_pair, roi_mask)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "site_manifest.json"


def test_vd18_replay_uses_portable_metric_bootstrap() -> None:
    site = load_replay_site(MANIFEST, "droidcam_shared_vd_18")

    assert site.calibration_profile == "replay_m_01"
    assert site.source_world_unit == "cm"
    assert site.runtime_world_unit == "m"
    assert site.processing_scale == 0.5
    assert len(site.timestamps) == 1321
    assert len(site.world_slots) == 48
    assert all(path.is_file() for path in site.video_paths.values())
    assert site.profiles["cam1"].vehicle_dimensions == (0.08, 0.04)
    assert np.max(np.abs(np.vstack(list(site.world_slots.values())))) < 1.0


def test_m04_uses_calibration_confirmed_by_its_session_metadata() -> None:
    site = load_replay_site(MANIFEST, "droidcam_shared_m_04")

    assert site.calibration_profile == "m04_cm_02"
    assert len(site.timestamps) == 976
    assert all(path.is_file() for path in site.video_paths.values())


def test_real_mp4_pair_reaches_current_pipeline() -> None:
    pytest.importorskip("cv2")
    site = load_replay_site(MANIFEST, "droidcam_shared_vd_18")
    pipeline = build_replay_pipeline(site)
    masks = {camera_id: roi_mask(site, camera_id) for camera_id in site.camera_ids}
    last = None
    for timestamp, frames in iter_decoded_pairs(site, limit=12):
        last = process_pair(site, pipeline, timestamp, frames, masks)

    assert last is not None
    assert last.cameras == ("cam1", "cam2")
    assert pipeline.ingestion.stats.complete_pairs == 12
    assert pipeline.ingestion.stats.skew_rejections == 0
    assert last.snapshot is not None
