import json

from experiment_test.validate_session import validate


def test_schema_two_uses_two_camera_required_files(tmp_path):
    (tmp_path / "session_info.json").write_text(
        json.dumps({"schema_version": 2, "processed_frames": 0}),
        encoding="utf-8",
    )

    errors, _ = validate(tmp_path)

    assert "Thieu file raw_cam1.mp4" in errors
    assert "Thieu file raw_video.mp4" not in errors
