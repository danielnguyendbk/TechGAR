"""Offline structural audit for replay identity traces.

This tool does not pretend to calculate IDF1/IDSW without dense physical-ID
ground truth.  It proves implementation invariants and reports high-signal
fragmentation/switch proxies that can be inspected before annotation exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def audit_predictions(path: str | Path, *, max_world_speed: float = 15.0) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "predictions.jsonl"
    issues: list[dict[str, Any]] = []
    metrics: Counter[str] = Counter()
    previous_owner: dict[tuple[str, int], int] = {}
    previous_position: dict[int, tuple[float, np.ndarray, int]] = {}
    minted_at: dict[int, int] = {}
    retired_at: dict[int, int] = {}
    promoted_at: dict[int, int] = {}
    last_seen_local: dict[tuple[str, int], int] = {}
    open_slot_issues: set[tuple[str, str]] = set()
    open_superseded: set[tuple[str, int]] = set()
    previous_active_local: dict[tuple[str, int], int] = {}

    def issue(kind: str, severity: str, frame: int, detail: str, **evidence: Any) -> None:
        issues.append({"kind": kind, "severity": severity, "frame": frame,
                       "detail": detail, "evidence": evidence})
        metrics[f"issue_{kind}"] += 1

    for row in _rows(source):
        frame = int(row.get("frame_index", 0))
        timestamp = float(row.get("timestamp", 0.0))
        trace = row.get("identity_trace") or {}
        ingest = trace.get("ingest") or {}
        fused_by_id = {int(obs["observation_id"]): obs
                       for obs in trace.get("fused_observations", [])}
        decisions = trace.get("association", [])
        minted = [int(gid) for gid in ingest.get("minted", [])]
        retired = [int(gid) for gid in ingest.get("retired", [])]
        promoted = [int(gid) for gid in ingest.get("promoted", [])]
        metrics["frames"] += 1
        metrics["minted"] += len(minted)
        metrics["retired"] += len(retired)
        metrics["promoted"] += len(promoted)
        metrics["deferred"] += len(ingest.get("deferred", []))
        metrics["quarantined"] += len(ingest.get("quarantined", []))
        metrics["blocked_mints"] += len(ingest.get("blocked_mints", []))
        if len(minted) > 1:
            issue("mint_burst", "warning", frame,
                  "More than one Global ID was minted in one synchronized frame.", gids=minted)
        for gid in minted:
            minted_at[gid] = frame
        for gid in retired:
            retired_at[gid] = frame
        for gid in promoted:
            promoted_at[gid] = frame

        # Compare a decision with ownership as it existed before this frame.
        for decision in decisions:
            metrics[f"association_{decision.get('decision', 'unknown')}"] += 1
            assigned = decision.get("assigned_global_id")
            obs = fused_by_id.get(int(decision["observation_id"]))
            if assigned is None or obs is None:
                continue
            prior = {previous_owner[(str(camera), int(local_id))]
                     for camera, local_id in obs.get("local_track_ids", [])
                     if (str(camera), int(local_id)) in previous_owner}
            if prior and prior != {int(assigned)}:
                issue("owner_reassignment", "error", frame,
                      "A previously-owned Local ID was assigned to a different Global ID.",
                      assigned_global_id=int(assigned), prior_global_ids=sorted(prior),
                      local_track_ids=obs.get("local_track_ids", []))

        observed_by_owner: dict[tuple[str, int], list[int]] = defaultdict(list)
        current_superseded: set[tuple[str, int]] = set()
        for obs in trace.get("local_observations", []):
            key = (str(obs["camera_id"]), int(obs["local_track_id"]))
            owner = obs.get("owner_global_id")
            historical_owner = obs.get("historical_owner_global_id")
            if (owner is None and historical_owner is not None and obs.get("observed")
                    and not obs.get("latent")):
                current_superseded.add(key)
                if key not in open_superseded:
                    issue("superseded_local_reappearance", "warning", frame,
                          "A superseded Local ID became observed again; global binding stayed blocked.",
                          camera_id=key[0], local_track_id=key[1],
                          historical_global_id=int(historical_owner))
            last_seen_local[key] = frame
            if owner is not None:
                owner = int(owner)
                active_key = (key[0], owner)
                old_active = previous_active_local.get(active_key)
                if old_active is not None and old_active != key[1]:
                    issue("active_local_binding_switch", "warning", frame,
                          "A Global ID switched its active Local ID in the same camera.",
                          camera_id=key[0], global_id=owner,
                          old_local_track_id=old_active, new_local_track_id=key[1])
                previous_active_local[active_key] = key[1]
                old = previous_owner.get(key)
                if old is not None and old != owner:
                    issue("owner_map_changed", "error", frame,
                          "The registry owner map changed for a live Local ID.",
                          camera_id=key[0], local_track_id=key[1], old=old, new=owner)
                previous_owner[key] = owner
                if obs.get("observed") and not obs.get("latent"):
                    observed_by_owner[(key[0], owner)].append(key[1])
        for (camera, gid), local_ids in observed_by_owner.items():
            unique = sorted(set(local_ids))
            if len(unique) > 1:
                issue("same_camera_duplicate_owner", "error", frame,
                      "One Global ID owns multiple independently observed Local IDs in one camera.",
                      camera_id=camera, global_id=gid, local_track_ids=unique)
        open_superseded = current_superseded

        for camera, entries in (trace.get("local_tracker_decisions") or {}).items():
            for entry in entries:
                action = str(entry.get("action", "unknown"))
                metrics[f"local_{action}"] += 1

        vehicles = row.get("vehicles", [])
        live_gids = {int(vehicle["global_id"]) for vehicle in vehicles}
        for vehicle in vehicles:
            gid = int(vehicle["global_id"])
            position = np.asarray(vehicle.get("world_position", vehicle.get("position")), dtype=float)
            prior = previous_position.get(gid)
            if prior is not None:
                prior_time, prior_pos, prior_frame = prior
                dt = timestamp - prior_time
                if dt > 1e-6:
                    speed = float(np.linalg.norm(position - prior_pos) / dt)
                    if speed > max_world_speed:
                        issue("teleport_proxy", "warning", frame,
                              "Published identity exceeded the audit world-speed bound.",
                              global_id=gid, speed=speed, dt=dt, previous_frame=prior_frame)
            previous_position[gid] = (timestamp, position, frame)

        current_slot_issues: set[tuple[str, str]] = set()
        for slot in row.get("slots", []):
            if slot.get("occupancy_state") != "occupied":
                continue
            owner = slot.get("owning_global_id")
            if owner is None:
                key = (str(slot.get("slot_id")), "ownerless_occupied_slot")
                current_slot_issues.add(key)
                if key not in open_slot_issues:
                    issue("ownerless_occupied_slot", "warning", frame,
                          "Occupied slot has no Global ID owner (episode start).",
                          slot_id=slot.get("slot_id"))
            elif int(owner) not in live_gids:
                key = (str(slot.get("slot_id")), "slot_owner_not_published")
                current_slot_issues.add(key)
                if key not in open_slot_issues:
                    issue("slot_owner_not_published", "error", frame,
                          "Occupied slot points at a Global ID absent from the runtime snapshot "
                          "(episode start).", slot_id=slot.get("slot_id"),
                          global_id=int(owner))
        open_slot_issues = current_slot_issues

    for gid, start in minted_at.items():
        end = retired_at.get(gid)
        if end is not None and end == start:
            issue("same_frame_mint_retire", "warning", end,
                  "A provisional Global ID was minted and retired in the same frame.",
                  global_id=gid, minted_frame=start, retired_frame=end)
        elif gid in promoted_at and end is not None and end - promoted_at[gid] <= 15:
            issue("short_lived_published_gid", "warning", end,
                  "A published Global ID retired within 15 frames of promotion.",
                  global_id=gid, promoted_frame=promoted_at[gid], retired_frame=end)

    frame_count = max(metrics["frames"], 1)
    if metrics["quarantined"] > 0.20 * frame_count:
        issue("quarantine_storm", "warning", 0,
              "Quarantined observations exceeded 20% of replay frames; inspect local split/merge churn.",
              quarantined=metrics["quarantined"], frames=frame_count,
              ratio=metrics["quarantined"] / frame_count)

    by_kind = Counter(item["kind"] for item in issues)
    errors = sum(item["severity"] == "error" for item in issues)
    warnings = sum(item["severity"] == "warning" for item in issues)
    return {
        "schema_version": 1,
        "source": str(source),
        "scope": "structural_and_proxy_only",
        "ground_truth_note": (
            "IDF1, ID switches, and physical identity continuity require dense identity ground truth; "
            "this report does not fabricate those metrics."
        ),
        "status": "fail" if errors else ("review" if warnings else "pass"),
        "error_count": errors,
        "warning_count": warnings,
        "issue_counts": dict(sorted(by_kind.items())),
        "metrics": dict(sorted(metrics.items())),
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a saved TechGAR identity trace")
    parser.add_argument("run", help="Run directory or predictions.jsonl")
    parser.add_argument("--max-world-speed", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_predictions(args.run, max_world_speed=args.max_world_speed)
    output = args.output
    if output is None:
        run = Path(args.run).resolve()
        output = (run if run.is_dir() else run.parent) / "identity_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "error_count", "warning_count", "issue_counts", "metrics")},
                     indent=2, ensure_ascii=False))
    print(f"report: {output}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
