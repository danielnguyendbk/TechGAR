from __future__ import annotations

import json

from techgar.identity_audit import audit_predictions


def test_audit_detects_local_owner_reassignment(tmp_path):
    rows = []
    for frame, owner, assigned in ((1, 7, 7), (2, 9, 9)):
        rows.append({
            "frame_index": frame,
            "timestamp": frame / 10,
            "vehicles": [],
            "slots": [],
            "identity_trace": {
                "fused_observations": [{"observation_id": frame,
                                          "local_track_ids": [["cam1", 3]]}],
                "association": [{"observation_id": frame,
                                   "assigned_global_id": assigned}],
                "local_observations": [{"camera_id": "cam1", "local_track_id": 3,
                                         "owner_global_id": owner, "observed": True,
                                         "latent": False}],
                "local_tracker_decisions": {},
                "ingest": {"minted": [], "retired": [], "deferred": [],
                           "quarantined": [], "blocked_mints": []},
            },
        })
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = audit_predictions(path)
    assert report["status"] == "fail"
    assert report["issue_counts"]["owner_reassignment"] == 1
    assert report["issue_counts"]["owner_map_changed"] == 1
