"""Portable audit for imported real-site assets.

This module deliberately treats legacy files as untrusted data.  It validates
their shape and provenance but does not import legacy tracking behaviour into
the current Stage 1-10 pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def _inside_root(root: Path, raw_path: str) -> Path:
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes asset root: {raw_path}") from exc
    return candidate


def _polygon_valid(points: Any, width: int | None = None,
                   height: int | None = None) -> bool:
    if not isinstance(points, list) or len(points) < 3:
        return False
    for point in points:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            return False
        x, y = point["x"], point["y"]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
        if width is not None and not 0 <= x < width:
            return False
        if height is not None and not 0 <= y < height:
            return False
    return True


def audit_site_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate portable paths, camera assets, calibrations and replay data."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    root = (manifest_path.parent / manifest.get("asset_root", ".")).resolve()
    issues: list[dict[str, str]] = []

    def issue(severity: str, code: str, detail: str) -> None:
        issues.append({"severity": severity, "code": code, "detail": detail})

    camera_reports: dict[str, dict[str, Any]] = {}
    pixel_slot_ids: dict[str, set[str]] = {}
    for camera_id, camera in manifest.get("cameras", {}).items():
        expected = camera.get("resolution_px", [])
        width, height = (expected + [None, None])[:2]
        report: dict[str, Any] = {"resolution_px": expected}
        try:
            roi_path = _inside_root(root, camera["roi_file"])
            slot_path = _inside_root(root, camera["pixel_slots_file"])
            roi = _read_json(roi_path)
            slots = _read_json(slot_path)
            roi_size = roi.get("image_size")
            slot_size = [slots.get("imageWidth"), slots.get("imageHeight")]
            roi_ok = roi_size == expected and _polygon_valid(
                roi.get("polygon"), width, height
            )
            raw_slots = slots.get("slots", [])
            ids = [slot.get("id") for slot in raw_slots if isinstance(slot, dict)]
            slot_shapes_ok = all(
                isinstance(slot, dict)
                and isinstance(slot.get("id"), str)
                and _polygon_valid(slot.get("polygon"), width, height)
                for slot in raw_slots
            )
            slots_ok = (slot_size == expected and len(ids) == len(set(ids))
                        and bool(ids) and slot_shapes_ok)
            if not roi_ok:
                issue("error", "invalid_roi", camera_id)
            if not slots_ok:
                issue("error", "invalid_pixel_slots", camera_id)
            pixel_slot_ids[camera_id] = set(ids)
            report.update({"roi_valid": roi_ok, "pixel_slots_valid": slots_ok,
                           "pixel_slot_count": len(ids)})
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            issue("error", "camera_asset_unreadable", f"{camera_id}: {exc}")
            report.update({"roi_valid": False, "pixel_slots_valid": False,
                           "pixel_slot_count": 0})
        camera_reports[camera_id] = report

    minimum_points = int(manifest.get("minimum_calibration_points_per_camera", 6))
    calibration_reports: dict[str, dict[str, Any]] = {}
    for profile_id, profile in manifest.get("calibration_profiles", {}).items():
        report = {"accepted": False, "cameras": {}}
        try:
            calibration_path = _inside_root(root, profile["file"])
            calibration = _read_json(calibration_path)
            unit = calibration.get("world", {}).get("unit")
            if unit != profile.get("declared_world_unit"):
                issue("error", "world_unit_mismatch", profile_id)
            transforms = calibration.get("camera_transforms", {})
            quality = calibration.get("calibration_quality", {}).get("cameras", {})
            world_slots = calibration.get("parking_slots_world", {})
            all_valid = True
            for camera_id in manifest.get("cameras", {}):
                matrix = np.asarray(transforms.get(camera_id, []), dtype=float)
                matrix_valid = (matrix.shape == (3, 3)
                                and np.isfinite(matrix).all()
                                and abs(float(np.linalg.det(matrix))) > 1e-12)
                point_count = int(quality.get(camera_id, {}).get("point_count", 0))
                redundancy = 2 * point_count - 8
                enough_points = point_count >= minimum_points and redundancy > 0
                shared_ids = {
                    slot.get("id") for slot in world_slots.get(camera_id, [])
                    if isinstance(slot, dict) and isinstance(slot.get("id"), str)
                }
                slots_match = shared_ids == pixel_slot_ids.get(camera_id, set())
                camera_ok = matrix_valid and enough_points and slots_match
                all_valid = all_valid and camera_ok
                report["cameras"][camera_id] = {
                    "matrix_valid": matrix_valid,
                    "point_count": point_count,
                    "dof_redundancy": redundancy,
                    "overfit_exact_solution": redundancy <= 0,
                    "slot_ids_match": slots_match,
                    "accepted": camera_ok,
                }
                if not enough_points:
                    issue("warning", "calibration_points_insufficient",
                          f"{profile_id}/{camera_id}: {point_count} < {minimum_points}")
                if not matrix_valid:
                    issue("error", "invalid_homography", f"{profile_id}/{camera_id}")
                if not slots_match:
                    issue("error", "slot_id_mismatch", f"{profile_id}/{camera_id}")
            report.update({"declared_world_unit": unit, "accepted": all_valid})
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            issue("error", "calibration_unreadable", f"{profile_id}: {exc}")
        calibration_reports[profile_id] = report

    dataset_reports: list[dict[str, Any]] = []
    profiles = manifest.get("calibration_profiles", {})
    for dataset in manifest.get("datasets", []):
        dataset_id = str(dataset.get("id", "unknown"))
        report: dict[str, Any] = {"id": dataset_id}
        try:
            directory = _inside_root(root, dataset["directory"])
            raw = dataset.get("raw_videos", {})
            raw_ok = bool(raw) and all((directory / name).is_file()
                                       for name in raw.values())
            timestamp_path = directory / dataset["timestamps"]
            timestamp_rows = _data_rows(timestamp_path) if timestamp_path.is_file() else 0
            profile_id = dataset.get("calibration_profile")
            profile_known = isinstance(profile_id, str) and profile_id in profiles
            labels: dict[str, int] = {}
            for label_type, name in dataset.get("ground_truth", {}).items():
                label_path = directory / name
                labels[label_type] = _data_rows(label_path) if label_path.is_file() else 0
            ready = raw_ok and timestamp_rows > 0 and profile_known
            if not raw_ok or timestamp_rows <= 0:
                issue("error", "replay_asset_missing", dataset_id)
            if not profile_known:
                issue("warning", "dataset_calibration_unconfirmed", dataset_id)
            if labels and not any(labels.values()):
                issue("warning", "ground_truth_empty", dataset_id)
            report.update({"raw_videos_present": raw_ok,
                           "timestamp_rows": timestamp_rows,
                           "calibration_profile": profile_id,
                           "calibration_profile_known": profile_known,
                           "ground_truth_rows": labels,
                           "ready_for_replay": ready})
        except (KeyError, OSError, ValueError) as exc:
            issue("error", "dataset_unreadable", f"{dataset_id}: {exc}")
            report["ready_for_replay"] = False
        dataset_reports.append(report)

    active_profile = manifest.get("active_calibration_profile")
    active_accepted = calibration_reports.get(active_profile, {}).get("accepted", False)
    assets_valid = not any(item["severity"] == "error" for item in issues)
    annotated_identity_dataset = any(
        report.get("ground_truth_rows", {}).get("identity", 0) > 0
        for report in dataset_reports
    )
    return {
        "site_id": manifest.get("site_id"),
        "manifest": str(manifest_path),
        "assets_valid": assets_valid,
        "ready_datasets": [report["id"] for report in dataset_reports
                           if report.get("ready_for_replay")],
        "commissioned": bool(assets_valid and active_accepted
                             and annotated_identity_dataset),
        "cameras": camera_reports,
        "calibrations": calibration_reports,
        "datasets": dataset_reports,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit TechGAR site/replay assets")
    parser.add_argument("manifest", nargs="?", default="config/site_manifest.json")
    parser.add_argument("--require-commissioned", action="store_true")
    args = parser.parse_args(argv)
    report = audit_site_manifest(args.manifest)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["assets_valid"]:
        return 2
    if args.require_commissioned and not report["commissioned"]:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
