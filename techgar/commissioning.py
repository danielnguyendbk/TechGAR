"""Phase 0 commissioning gate and reproducible measurement report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .simulation.layouts import build_profiles
from .simulation.survey import measure_seam_disagreement, measure_timestamp_skew


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class CommissioningReport:
    calibration: list[dict[str, Any]]
    timestamp_skew: dict[str, float]
    seam: dict[str, float]
    gates: list[GateResult]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "calibration": self.calibration,
            "timestamp_skew": self.timestamp_skew,
            "seam": self.seam,
            "gates": [asdict(gate) for gate in self.gates],
        }


def commission(layout, recording, profiles=None, max_skew: float = 0.120,
               max_calibration_rms: float = 0.50,
               require_seam_samples: bool | None = None) -> CommissioningReport:
    """Evaluate Phase 0 exit gates from a site layout and dual-camera recording."""
    profiles = build_profiles(layout) if profiles is None else profiles
    calibration = layout.calibration_report()
    skew = measure_timestamp_skew(recording)
    if require_seam_samples is None:
        require_seam_samples = bool(layout.topology.overlaps)
    seam = (measure_seam_disagreement(layout, profiles)
            if require_seam_samples else
            {"samples": 0, "rho_seam": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0})
    gates = [
        GateResult(
            "dual_camera_recording",
            len(recording.camera_ids) >= 2 and all(
                any(spec.camera_id == camera for spec in recording.specs)
                for camera in recording.camera_ids
            ),
            f"cameras={','.join(recording.camera_ids)} frames={len(recording.specs)}",
        ),
        GateResult(
            "calibration_nonzero_dof",
            bool(calibration) and all(row["dof_redundancy"] > 0 for row in calibration),
            "; ".join(f"{row['camera_id']}:dof={row['dof_redundancy']}"
                      for row in calibration),
        ),
        GateResult(
            "calibration_residual",
            bool(calibration) and all(row["rms_residual_world"] <= max_calibration_rms
                                      for row in calibration),
            "; ".join(f"{row['camera_id']}:rms={row['rms_residual_world']:.4f}"
                      for row in calibration),
        ),
        GateResult(
            "timestamp_skew",
            skew.get("samples", 0) > 0 and skew.get("p95", float("inf")) <= max_skew,
            f"samples={skew.get('samples', 0)} p95={skew.get('p95', float('inf')):.4f}s",
        ),
        GateResult(
            "seam_measurement",
            (not require_seam_samples) or seam.get("samples", 0) > 0,
            f"samples={seam.get('samples', 0)} rho={seam.get('rho_seam', 0.0):.4f}",
        ),
    ]
    return CommissioningReport(calibration, skew, seam, gates)
