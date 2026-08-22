r"""CLI for the strict TechGAR schema-v3 practical-system evaluator.

Examples (PowerShell):

    ..\.venv\Scripts\python.exe evaluate.py `
      experiment_test\output\droidcam_shared_m_01 --fps 25

    ..\.venv\Scripts\python.exe evaluate.py `
      experiment_test\output\droidcam_shared_m_01 `
      experiment_test\output\droidcam_shared_m_02 `
      experiment_test\output\droidcam_shared_m_03 `
      experiment_test\output\droidcam_shared_m_04 --fps 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from techgar.evaluation_v3 import (  # noqa: E402
    EvaluationValidationError,
    _markdown_cell,
    _markdown_error_table,
    aggregate_results,
    evaluate_session,
)


def _print_result(result: dict) -> None:
    print("\n" + "=" * 68)
    print(f"  {result['session']}: {result['classification']}")
    print(f"  Practical System Score: {result['practical_system_score']:.2f}/100")
    if result["score_cap_applied"]:
        print(
            "  CRITICAL CAP: "
            f"{result['uncapped_practical_system_score']:.2f} -> "
            f"{result['practical_system_score']:.2f}"
        )
    print("-" * 68)
    labels = {
        "identity_continuity_handoff": "Identity continuity + handoff",
        "slot_identity_ownership": "Correct slot ownership",
        "departure_recovery": "Departure recovery / ReID",
        "occupancy": "Occupied / free",
        "delay_stability": "Delay + stability",
    }
    for key, label in labels.items():
        value = result["scores"].get(key)
        display = "N/A" if value is None else f"{value:.2f}"
        print(f"  {label:<34} {display:>8}")
    print(f"  {'Critical errors':<34} {result['critical_error_count']:>8}")
    for item in result["critical_errors"][:10]:
        location = "/".join(
            str(value) for value in (item.get("camera_id"), item.get("slot_id")) if value
        )
        print(
            f"    - frame {item.get('frame_idx', '?')} {location}: "
            f"{item['code']} — {item['message']}"
        )
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate strict schema-v3 TechGAR predictions against lifecycle "
            "ground truth. Schema v1/v2 is intentionally rejected."
        )
    )
    parser.add_argument(
        "session_dirs",
        nargs="+",
        type=Path,
        help="One or more session directories containing schema-v3 files",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help=(
            "Directory for aggregate JSON/Markdown. Default for multiple sessions: "
            "their common parent directory"
        ),
    )
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be > 0")

    results = []
    try:
        for session_dir in args.session_dirs:
            result = evaluate_session(session_dir, fps=args.fps, write_outputs=True)
            results.append(result)
            _print_result(result)
    except EvaluationValidationError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"\nERROR: cannot read/write evaluation files: {exc}", file=sys.stderr)
        return 2

    aggregate = None
    if len(results) > 1:
        aggregate = aggregate_results(results)
        output_dir = args.aggregate_output
        if output_dir is None:
            resolved_parents = [str(path.resolve().parent) for path in args.session_dirs]
            output_dir = Path(os.path.commonpath(resolved_parents))
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "evaluation_summary_v3.json"
        report_path = output_dir / "evaluation_summary_v3.md"
        json_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            _render_aggregate_report(aggregate), encoding="utf-8"
        )
        _print_result(aggregate)
        print(f"Aggregate JSON: {json_path}")
        print(f"Aggregate report: {report_path}")

    failed = any(result["classification"] == "FAIL" for result in results)
    failed = failed or bool(aggregate and aggregate["classification"] == "FAIL")
    return 1 if failed else 0


def _render_aggregate_report(result: dict) -> str:
    lines = [
        "# TechGAR Practical System Report — aggregate",
        "",
        f"- Kết luận: **{result['classification']}**",
        f"- Practical System Score: **{result['practical_system_score']:.2f}/100**",
        f"- Critical errors: **{result['critical_error_count']}**",
        f"- Sessions: {', '.join(result['sessions'])}",
        "",
        "## Điểm thành phần",
        "",
        "| Thành phần | Điểm | Số đơn vị tổng hợp |",
        "|---|---:|---:|",
    ]
    units = result["aggregation_units"]
    rows = (
        ("identity_continuity_handoff", "Identity continuity + handoff", units["identity_vehicle_lifecycles"]),
        ("slot_identity_ownership", "Đúng chủ sở hữu ô", units["slot_lifecycles"]),
        ("departure_recovery", "Departure recovery / ReID", units["departure_events"]),
        ("occupancy", "Occupied / free", units["occupancy_sessions"]),
        ("delay_stability", "Delay + stability", units["delay_sessions"]),
    )
    for key, label, count in rows:
        value = result["scores"].get(key)
        lines.append(f"| {label} | {'N/A' if value is None else f'{value:.2f}'} | {count} |")
    lines.extend(["", "## Critical errors", ""])
    if not result["critical_errors"]:
        lines.append("Không có critical error.")
    else:
        detailed_table = _markdown_error_table(result["critical_errors"])
        lines.append("| Session " + detailed_table[0])
        lines.append("|---" + detailed_table[1])
        for item, row in zip(result["critical_errors"], detailed_table[2:]):
            lines.append(f"| {_markdown_cell(item.get('session'))} {row}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
