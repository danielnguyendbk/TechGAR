"""Tests for Phase A data root resolution, manifest availability, and commissioning gates."""

import os
from pathlib import Path
import pytest

from techgar.commissioning import commission
from techgar.data_root import check_dataset_available, get_data_root, resolve_data_path
from techgar.simulation.layouts import cruise, gap_layout
from techgar.simulation.recording import RecordingOptions, build_recording
from tools.import_data import compute_sha256, import_asset, import_directory


class TestDataRoot:
    """Data root and path resolution."""

    def test_default_data_root_points_to_repo(self):
        root = get_data_root()
        assert (root / "techgar").is_dir()
        assert (root / "config").is_dir()

    def test_custom_env_data_root(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom_data"
        custom_dir.mkdir()
        monkeypatch.setenv("TECHGAR_DATA_ROOT", str(custom_dir))
        assert get_data_root() == custom_dir.resolve()

    def test_check_dataset_availability(self, tmp_path):
        # Empty folder -> unavailable
        dataset = {
            "directory": "fake_dataset",
            "timestamps": "ts.csv",
            "raw_videos": {"cam1": "c1.mp4", "cam2": "c2.mp4"},
        }
        assert check_dataset_available(dataset, base=tmp_path) is False

        # Create dummy files -> available
        ds_dir = tmp_path / "fake_dataset"
        ds_dir.mkdir()
        (ds_dir / "ts.csv").write_text("dummy", encoding="utf-8")
        (ds_dir / "c1.mp4").write_text("dummy", encoding="utf-8")
        (ds_dir / "c2.mp4").write_text("dummy", encoding="utf-8")
        assert check_dataset_available(dataset, base=tmp_path) is True


class TestDataImportTool:
    """Clean-room asset importer tool."""

    def test_import_allowed_asset_with_checksum(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        sample_video = src_dir / "test_video.mp4"
        sample_video.write_bytes(b"dummy video content 12345")

        dst_dir = tmp_path / "dst"
        record = import_asset(sample_video, dst_dir)
        assert record["filename"] == "test_video.mp4"
        assert record["size_bytes"] == len(b"dummy video content 12345")
        assert len(record["sha256"]) == 64
        assert (dst_dir / "test_video.mp4").is_file()

    def test_import_forbids_python_code_files(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        py_file = src_dir / "exploit.py"
        py_file.write_text("import os; print('bad')", encoding="utf-8")

        dst_dir = tmp_path / "dst"
        with pytest.raises(ValueError, match="only data assets can be imported"):
            import_asset(py_file, dst_dir)

    def test_import_directory_creates_provenance(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "data1.csv").write_text("a,b,c", encoding="utf-8")
        (src_dir / "data2.json").write_text("{}", encoding="utf-8")

        dst_dir = tmp_path / "dst"
        records = import_directory(src_dir, dst_dir)
        assert len(records) == 2
        assert (dst_dir / "provenance.json").is_file()


class TestCommissioningGates:
    """Commissioning gate invariants."""

    def test_gate_rejects_fewer_than_6_points(self):
        layout = gap_layout(calib_points=8, noise_px=0.4)
        vehicle = cruise("P01", 8.0, 24.0, speed=8.0, y=14.0)
        recording = build_recording(
            "phase0", layout, [vehicle],
            RecordingOptions(fps=10.0, skew={"C1": 0.0, "C2": 0.03}, jitter=0.0),
        )
        # Require 10 points, but layout only has 8 -> gate fails!
        report = commission(layout, recording, require_seam_samples=False, require_min_points=10)
        gate = next(g for g in report.gates if g.name == "minimum_6_calibration_points")
        assert gate.passed is False, "Gate must fail when calibration points < required"

    def test_gate_passes_with_sufficient_points(self):
        layout = gap_layout(calib_points=8, noise_px=0.4)
        vehicle = cruise("P01", 8.0, 24.0, speed=8.0, y=14.0)
        recording = build_recording(
            "phase0", layout, [vehicle],
            RecordingOptions(fps=10.0, skew={"C1": 0.0, "C2": 0.03}, jitter=0.0),
        )
        report = commission(layout, recording, require_seam_samples=False, require_min_points=6)
        gate = next(g for g in report.gates if g.name == "minimum_6_calibration_points")
        assert gate.passed is True
