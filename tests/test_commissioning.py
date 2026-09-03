"""PLAN 1 Phase 0 measurement gates."""

from techgar.commissioning import commission
from techgar.simulation.layouts import cruise, gap_layout
from techgar.simulation.recording import RecordingOptions, build_recording


def test_synthetic_commissioning_has_real_dof_and_measured_skew():
    layout = gap_layout(calib_points=12, noise_px=0.4)
    vehicle = cruise("P01", 8.0, 24.0, speed=8.0, y=14.0)
    recording = build_recording(
        "phase0",
        layout,
        [vehicle],
        RecordingOptions(fps=10.0, skew={"C1": 0.0, "C2": 0.03}, jitter=0.0),
    )
    report = commission(layout, recording, require_seam_samples=False)
    assert report.passed, report.as_dict()
    assert all(row["dof_redundancy"] > 0 for row in report.calibration)
    assert report.timestamp_skew["samples"] > 0

