from __future__ import annotations

import json
from pathlib import Path

from techgar.site_assets import audit_site_manifest, main


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "site_manifest.json"


def test_imported_droidcam_assets_are_structurally_valid() -> None:
    report = audit_site_manifest(MANIFEST)

    assert report["assets_valid"] is True
    assert report["cameras"]["cam1"]["pixel_slot_count"] == 24
    assert report["cameras"]["cam2"]["pixel_slot_count"] == 24
    assert report["ready_datasets"] == [
        "droidcam_shared_vd_17",
        "droidcam_shared_vd_18",
        "droidcam_shared_m_04",
    ]


def test_four_point_legacy_calibrations_are_not_commissioned() -> None:
    report = audit_site_manifest(MANIFEST)

    assert report["commissioned"] is False
    for profile in report["calibrations"].values():
        assert profile["accepted"] is False
        for camera in profile["cameras"].values():
            assert camera["point_count"] == 4
            assert camera["dof_redundancy"] == 0
            assert camera["overfit_exact_solution"] is True
            assert camera["matrix_valid"] is True
            assert camera["slot_ids_match"] is True


def test_manifest_does_not_adopt_legacy_absolute_paths_or_anchor() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest)

    assert "D:\\\\" not in serialized
    assert manifest["detector_reference"]["usage"] == (
        "reference_only_do_not_apply_automatically"
    )
    assert "tracking_defaults.shared_map_anchor=bbox_center" in (
        manifest["legacy_fields_not_adopted"]
    )


def test_cli_can_require_real_commissioning() -> None:
    assert main([str(MANIFEST)]) == 0
    assert main([str(MANIFEST), "--require-commissioned"]) == 3
