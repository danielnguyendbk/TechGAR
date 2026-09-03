"""Reproducible, non-destructive output bundle for real-video replay runs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .replay import ReplaySite, ReplayTimestamp


IDENTITY_GT_HEADER = [
    "schema_version", "observation_id", "physical_vehicle_id", "frame_idx",
    "camera_id", "anchor_x", "anchor_y", "slot_id", "phase", "required", "notes",
]


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class ReplayOutputWriter:
    """Write one immutable run directory without modifying replay input files."""

    def __init__(self, site: ReplaySite, output_root: str | Path, playback_speed: float,
                 runtime_id: str | None = None) -> None:
        self.site = site
        self.runtime_id = runtime_id or uuid4().hex
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{stamp}_{self.runtime_id[:8]}"
        self.directory = Path(output_root).resolve() / site.dataset_id / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.playback_speed = float(playback_speed)
        self.started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.processed_frames = 0
        self._prediction = (self.directory / "predictions.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._events = (self.directory / "identity_events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._last_event_id = 0
        self._videos: dict[str, Any] = {}
        self._timestamp_handle = (self.directory / "frame_timestamps.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._timestamp_csv = csv.writer(self._timestamp_handle)
        self._timestamp_csv.writerow([
            "frame_idx", "capture_seconds", "cam1_timestamp_seconds",
            "cam2_timestamp_seconds", "camera_skew_ms",
        ])
        self._performance_handle = (self.directory / "performance.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._performance_csv = csv.writer(self._performance_handle)
        self._performance_csv.writerow([
            "frame_idx", "total_processing_ms", "detections_cam1", "detections_cam2",
            "global_vehicles", "occupied_slots", "overload",
        ])
        self._slot_handle = (self.directory / "slot_events.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._slot_csv = csv.writer(self._slot_handle)
        self._slot_csv.writerow([
            "frame_idx", "timestamp", "slot_id", "global_id", "event_type", "detail",
        ])
        self._copy_ground_truth()
        self._write_manifest()
        self._write_session("running")

    def _copy_ground_truth(self) -> None:
        standard_names = {
            "slots": "ground_truth_slots.csv",
            "events": "ground_truth_events.csv",
            "identity": "ground_truth_identity.csv",
        }
        for label_type, destination_name in standard_names.items():
            source = self.site.ground_truth_paths.get(label_type)
            destination = self.directory / destination_name
            if source is not None and source.is_file():
                shutil.copy2(source, destination)
            elif label_type == "identity":
                with destination.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle).writerow(IDENTITY_GT_HEADER)

    def _asset_manifest(self) -> list[dict[str, Any]]:
        paths = {
            "site_manifest": self.site.manifest_path,
            "calibration": self.site.calibration_path,
            "timestamps": self.site.timestamp_path,
            **{f"video_{camera}": path for camera, path in self.site.video_paths.items()},
            **{f"ground_truth_{kind}": path
               for kind, path in self.site.ground_truth_paths.items()},
        }
        for camera_id in self.site.camera_ids:
            camera = _read_manifest_camera(self.site.manifest_path, camera_id)
            paths[f"roi_{camera_id}"] = camera["roi"]
            paths[f"pixel_slots_{camera_id}"] = camera["slots"]
        return [{"role": role, "path": str(path), "bytes": path.stat().st_size,
                 "sha256": _sha256(path)}
                for role, path in sorted(paths.items()) if path.is_file()]

    def _write_manifest(self) -> None:
        try:
            import cv2
            opencv_version = cv2.__version__
        except ImportError:  # pragma: no cover
            opencv_version = None
        _atomic_json(self.directory / "run_manifest.json", {
            "schema_version": 1,
            "run_id": self.run_id,
            "runtime_id": self.runtime_id,
            "source_mode": "replay",
            "dataset": self.site.dataset_id,
            "calibration_profile": self.site.calibration_profile,
            "processing_scale": self.site.processing_scale,
            "playback_speed": self.playback_speed,
            "world_unit": self.site.runtime_world_unit,
            "bootstrap_warning": self.site.bootstrap_warning,
            "created_at": self.started_at,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "techgar": _package_version("techgar"),
                "numpy": np.__version__,
                "opencv": opencv_version,
            },
            "assets": self._asset_manifest(),
        })

    def _write_session(self, status: str, error: str | None = None) -> None:
        _atomic_json(self.directory / "session_info.json", {
            "schema_version": 4,
            "runtime_id": self.runtime_id,
            "run_id": self.run_id,
            "source_mode": "replay",
            "status": status,
            "started_at": self.started_at,
            "ended_at": (datetime.now().astimezone().isoformat(timespec="milliseconds")
                         if status != "running" else None),
            "processed_frames": self.processed_frames,
            "camera_ids": list(self.site.camera_ids),
            "dataset": self.site.dataset_id,
            "calibration_profile": self.site.calibration_profile,
            "calibration_accepted": False,
            "processing_scale": self.site.processing_scale,
            "playback_speed": self.playback_speed,
            "error": error,
            "files": {
                "predictions": "predictions.jsonl",
                "timestamps": "frame_timestamps.csv",
                "performance": "performance.csv",
                "identity_events": "identity_events.jsonl",
                "slot_events": "slot_events.csv",
                "debug_cam1": "debug_cam1.mp4",
                "debug_cam2": "debug_cam2.mp4",
                "manifest": "run_manifest.json",
                "performance_summary": "performance_summary.json",
                "evaluation_status": "evaluation_status.json",
            },
        })

    def _video_writer(self, camera_id: str, frame: np.ndarray):
        if camera_id in self._videos:
            return self._videos[camera_id]
        import cv2
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(self.directory / f"debug_{camera_id}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create debug video for {camera_id}")
        self._videos[camera_id] = writer
        return writer

    def write(self, timestamp: ReplayTimestamp, annotated_frames: dict[str, np.ndarray],
              result, total_processing_seconds: float, pipeline) -> None:
        snapshot = result.snapshot if result is not None else None
        camera_times = timestamp.camera_times
        skew = ((max(camera_times.values()) - min(camera_times.values())) * 1000.0
                if camera_times else 0.0)
        for camera_id, frame in annotated_frames.items():
            self._video_writer(camera_id, frame).write(frame)
        payload = (snapshot.to_dict() if snapshot is not None
                   else {"sequence": 0, "vehicles": [], "slots": []})
        payload.update({
            "schema_version": 4,
            "runtime_id": self.runtime_id,
            "source_mode": "replay",
            "dataset": self.site.dataset_id,
            "frame_index": timestamp.frame_index,
            "camera_timestamps": camera_times,
            "camera_skew_ms": skew,
            "detections": ({camera: len(result.detections.get(camera, []))
                            for camera in self.site.camera_ids}
                           if result is not None else {}),
        })
        self._prediction.write(json.dumps(payload, ensure_ascii=False,
                                          default=_json_default) + "\n")
        self._timestamp_csv.writerow([
            timestamp.frame_index, f"{timestamp.capture_time:.9f}",
            f"{camera_times.get('cam1', float('nan')):.9f}",
            f"{camera_times.get('cam2', float('nan')):.9f}", f"{skew:.3f}",
        ])
        detections = payload["detections"]
        self._performance_csv.writerow([
            timestamp.frame_index, f"{total_processing_seconds * 1000.0:.3f}",
            detections.get("cam1", 0), detections.get("cam2", 0),
            len(snapshot.vehicles) if snapshot else 0,
            sum(slot.occupancy_state == "occupied" for slot in snapshot.slots)
            if snapshot else 0,
            bool(result.overload) if result is not None else False,
        ])
        if result is not None:
            for event in result.slot_events:
                self._slot_csv.writerow([
                    timestamp.frame_index, f"{event.timestamp:.9f}", event.slot_id,
                    "" if event.global_id is None else event.global_id,
                    event.kind, event.detail,
                ])
        for event in pipeline.registry.events:
            if event.event_id <= self._last_event_id:
                continue
            self._events.write(json.dumps({
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "frame_sequence": event.frame_sequence,
                "type": event.event_type.value,
                "global_id": event.global_id,
                "camera_id": event.camera_id,
                "detail": event.detail,
                "evidence": event.evidence,
            }, ensure_ascii=False, default=_json_default) + "\n")
            self._last_event_id = event.event_id
        self.processed_frames += 1
        if self.processed_frames % 25 == 0:
            self._write_session("running")

    def finish(self, pipeline, status: str = "completed", error: str | None = None) -> None:
        for writer in self._videos.values():
            writer.release()
        self._videos.clear()
        for handle in (self._prediction, self._events, self._timestamp_handle,
                       self._performance_handle, self._slot_handle):
            if not handle.closed:
                handle.close()
        _atomic_json(self.directory / "performance_summary.json", pipeline.performance_report())
        _atomic_json(self.directory / "evaluation_status.json", {
            "schema_version": 1,
            "status": "not_run",
            "reason": (
                "Dense identity ground truth is not available for this new run; "
                "legacy evaluator output is intentionally not reused."
            ),
            "required_next": "Annotate ground_truth_identity.csv then run the current evaluator.",
        })
        self._write_session(status, error)


def _read_manifest_camera(manifest_path: Path, camera_id: str) -> dict[str, Path]:
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    root = (manifest_path.parent / manifest.get("asset_root", ".")).resolve()
    camera = manifest["cameras"][camera_id]
    return {
        "roi": (root / camera["roi_file"]).resolve(),
        "slots": (root / camera["pixel_slots_file"]).resolve(),
    }
