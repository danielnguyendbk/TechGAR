from __future__ import annotations

import json
from pathlib import Path

import pytest

from techgar.replay import (build_replay_pipeline, iter_decoded_pairs, load_replay_site,
                            process_pair, roi_mask)
from techgar.replay_output import ReplayOutputWriter


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "site_manifest.json"


def test_saved_replay_run_is_separate_complete_and_auditable(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    site = load_replay_site(MANIFEST, "droidcam_shared_m_04")
    pipeline = build_replay_pipeline(site)
    masks = {camera_id: roi_mask(site, camera_id) for camera_id in site.camera_ids}
    writer = ReplayOutputWriter(site, tmp_path, playback_speed=20.0,
                                runtime_id="test-runtime")

    for timestamp, frames in iter_decoded_pairs(site, limit=3):
        result = process_pair(site, pipeline, timestamp, frames, masks)
        writer.write(timestamp, frames, result, result.processing_seconds, pipeline)
    writer.finish(pipeline)

    output = writer.directory
    session = json.loads((output / "session_info.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation_status.json").read_text(encoding="utf-8"))

    assert session["status"] == "completed"
    assert session["processed_frames"] == 3
    assert session["source_mode"] == "replay"
    assert session["calibration_profile"] == "m04_cm_02"
    assert len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    first_prediction = json.loads(
        (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    trace = first_prediction["identity_trace"]
    assert {"detections", "local_observations", "world_observations",
            "fused_observations", "association", "ingest"} <= set(trace)
    assert len((output / "performance.csv").read_text(encoding="utf-8").splitlines()) == 4
    assert (output / "debug_cam1.mp4").stat().st_size > 0
    assert (output / "debug_cam2.mp4").stat().st_size > 0
    assert (output / "ground_truth_slots.csv").is_file()
    assert (output / "ground_truth_events.csv").is_file()
    assert (output / "ground_truth_identity.csv").is_file()
    assert (output / "identity_audit.json").is_file()
    assert evaluation["status"] == "not_run"
    assert any(asset["role"] == "calibration" and len(asset["sha256"]) == 64
               for asset in manifest["assets"])
