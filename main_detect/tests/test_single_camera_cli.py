import sys

from single_camera import parse_args


def test_single_camera_accepts_droidcam_stream_url(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "single_camera.py",
        "--stream-url",
        "http://192.168.100.53:4747/video/force/1280x720",
        "--profile-camera",
        "cam1",
    ])

    args = parse_args()

    assert args.stream_url == "http://192.168.100.53:4747/video/force/1280x720"
    assert args.video is None
    assert args.camera is None
    assert args.profile_camera == "cam1"
