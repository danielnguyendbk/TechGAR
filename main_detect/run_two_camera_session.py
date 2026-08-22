"""Run TechGAR from two DroidCam URLs or replay one recorded raw session.

Examples (run from ``main_detect``):

Live DroidCam::

    python run_two_camera_session.py --cam1-video http://... \
        --cam2-video http://... --slots-cam1 ... --slots-cam2 ... \
        --calibration ... --session-dir ...

Recorded session::

    python run_two_camera_session.py \
        --cam1-video experiment_test/output/old/raw_cam1.mp4 \
        --cam2-video experiment_test/output/old/raw_cam2.mp4 \
        --slots-cam1 ... --slots-cam2 ... --calibration ... \
        --session-dir experiment_test/output/new

For recorded input, the two raw videos and ``frame_timestamps.csv`` must be in
the same directory.  Keeping the recorded timestamps is important because the
ReID windows are calculated from the effective camera FPS.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import two_camera


NETWORK_SCHEMES = {"http", "https", "rtsp", "rtmp"}


def _is_network_source(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme.lower() in NETWORK_SCHEMES and bool(parsed.netloc)


def _resolve_raw_session(
    parser,
    cam1_value: str,
    cam2_value: str,
) -> Path:
    cam1_path = Path(cam1_value).expanduser().resolve()
    cam2_path = Path(cam2_value).expanduser().resolve()

    missing = [str(path) for path in (cam1_path, cam2_path) if not path.is_file()]
    if missing:
        parser.error("Khong tim thay raw video: " + ", ".join(missing))

    if cam1_path.parent != cam2_path.parent:
        parser.error(
            "raw_cam1.mp4 va raw_cam2.mp4 phai nam trong cung mot session"
        )
    if cam1_path.name.lower() != "raw_cam1.mp4":
        parser.error("--cam1-video phai tro den file raw_cam1.mp4")
    if cam2_path.name.lower() != "raw_cam2.mp4":
        parser.error("--cam2-video phai tro den file raw_cam2.mp4")

    timestamps_path = cam1_path.parent / "frame_timestamps.csv"
    if not timestamps_path.is_file():
        parser.error(
            "Session raw thieu frame_timestamps.csv; khong the replay dung FPS thuc"
        )
    return cam1_path.parent


def make_parser():
    parser = two_camera.make_parser()
    parser.description = (
        "Chay hai DroidCam hoac replay raw video cua mot session da ghi"
    )
    parser.add_argument(
        "--cam1-video",
        required=True,
        help="URL DroidCam hoac duong dan den raw_cam1.mp4",
    )
    parser.add_argument(
        "--cam2-video",
        required=True,
        help="URL DroidCam hoac duong dan den raw_cam2.mp4",
    )
    return parser


def main() -> None:
    two_camera.configure_console_utf8()
    parser = make_parser()
    args = parser.parse_args()

    if args.cam1_url or args.cam2_url or args.replay_session:
        parser.error(
            "File nay chi nhan --cam1-video/--cam2-video; "
            "khong truyen them --cam1-url, --cam2-url hoac --replay-session"
        )

    cam1_network = _is_network_source(args.cam1_video)
    cam2_network = _is_network_source(args.cam2_video)
    if cam1_network != cam2_network:
        parser.error("Hai dau vao phai cung la URL hoac cung la raw video")

    if cam1_network:
        args.cam1_url = args.cam1_video
        args.cam2_url = args.cam2_video
        print("Che do: LIVE DROIDCAM")
    else:
        replay_session = _resolve_raw_session(
            parser,
            args.cam1_video,
            args.cam2_video,
        )
        args.replay_session = str(replay_session)
        print(f"Che do: REPLAY RAW SESSION ({replay_session})")

    two_camera.run(args)


if __name__ == "__main__":
    main()
