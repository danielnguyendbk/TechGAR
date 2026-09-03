"""Command-line entry point for deterministic scenario and ablation reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..commissioning import commission
from ..simulation import RecordingOptions, build_recording, overlap_layout, parking_layout
from ..simulation.layouts import cruise, slot_centre
from ..simulation.vehicles import SimVehicle, Waypoint
from .ablation import run_ablation_suite
from .environment import run_environmental_checks
from .harness import run_recording
from .rubric import score_rubric
from .scenarios import run_all_scenarios


def _pixel_smoke_report():
    layout = overlap_layout()
    recording = build_recording(
        "pixel_smoke",
        layout,
        [cruise("P01", 5.0, 35.0, speed=8.0)],
        RecordingOptions(fps=12.0, jitter=0.0, tail=0.2),
    )
    return recording, commission(layout, recording), run_recording(recording)


def _pixel_parking_report():
    """Positive slot case: approach D05, stop, and remain for the dwell window."""
    layout = parking_layout(blind_band=(90.0, 91.0))
    centre = slot_centre("D05", layout.slots)
    # Give the adaptive background model an empty-scene pre-roll.  Initialising a
    # background-subtraction benchmark with the vehicle already in frame teaches
    # the reference model that vehicle as static background and creates a false
    # negative "ghost" when it moves.  Field recordings follow the same empty
    # warm-up contract during commissioning.
    vehicle = SimVehicle("P01", [
        Waypoint(1.0, float(centre[0]), 12.0),
        Waypoint(5.0, float(centre[0]), float(centre[1])),
        Waypoint(11.0, float(centre[0]), float(centre[1])),
    ])
    recording = build_recording(
        "pixel_parking",
        layout,
        [vehicle],
        RecordingOptions(fps=12.0, jitter=0.0, tail=0.5),
    )
    return recording, commission(layout, recording), run_recording(recording)


def build_report(include_ablation: bool = False, include_pixel_smoke: bool = False,
                 include_pixel_parking: bool = False) -> dict:
    scenarios = run_all_scenarios()
    environmental = run_environmental_checks()
    run = None
    report = {
        "scenarios": [scenario.as_dict() for scenario in scenarios],
        "environmental": environmental,
    }
    if include_ablation:
        report["ablations"] = [outcome.as_dict() for outcome in run_ablation_suite()]
    if include_pixel_smoke:
        recording, commissioning, run = _pixel_smoke_report()
        report["commissioning"] = commissioning.as_dict()
        report["pixel_smoke"] = run.as_dict()
        report["pixel_smoke"]["deterministic_seed"] = recording.options.seed
    if include_pixel_parking:
        recording, commissioning, run = _pixel_parking_report()
        report["pixel_parking_commissioning"] = commissioning.as_dict()
        report["pixel_parking"] = run.as_dict()
        report["pixel_parking"]["deterministic_seed"] = recording.options.seed
    report["rubric"] = score_rubric(scenarios, run=run, environmental=environmental).as_dict()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run TechGAR PLAN 3 validation")
    parser.add_argument("--ablation", action="store_true", help="run all ablation variants")
    parser.add_argument("--pixel-smoke", action="store_true",
                        help="run deterministic pixels through stages 1-10")
    parser.add_argument("--pixel-parking", action="store_true",
                        help="run a positive pixel-to-slot parking benchmark")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    text = json.dumps(build_report(args.ablation, args.pixel_smoke, args.pixel_parking),
                      indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
