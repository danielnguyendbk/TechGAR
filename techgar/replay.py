"""Data-only adapter for synchronized DroidCam MP4 replay.

Legacy JSON is converted into the current metric contracts here.  The adapter
does not import legacy tracking code or tune the current algorithms from old
runtime thresholds.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import TechgarConfig
from .contracts import FrameRecord
from .homography import HomographyCalibration, project_points
from .pipeline import StepResult, TechgarPipeline
from .profile import CameraProfile
from .topology import CameraTopology, CameraZone, TopologyEdge


@dataclass(frozen=True)
class ReplayTimestamp:
    frame_index: int
    capture_time: float
    camera_times: dict[str, float]


@dataclass
class ReplaySite:
    manifest_path: Path
    dataset_id: str
    dataset_directory: Path
    video_paths: dict[str, Path]
    timestamp_path: Path
    ground_truth_paths: dict[str, Path]
    calibration_path: Path
    timestamps: list[ReplayTimestamp]
    profiles: dict[str, CameraProfile]
    topology: CameraTopology
    world_slots: dict[str, np.ndarray]
    roi_polygons: dict[str, np.ndarray]
    pixel_slots: dict[str, dict[str, np.ndarray]]
    processing_scale: float
    source_world_unit: str
    runtime_world_unit: str
    calibration_profile: str
    bootstrap_warning: str

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.profiles))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {value}") from exc
    return path


def _pixel_polygon(raw: list[dict[str, float]]) -> np.ndarray:
    return np.asarray([[point["x"], point["y"]] for point in raw], dtype=float)


def _polygon_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) * 0.5


def load_timestamps(path: Path, camera_ids: tuple[str, ...]) -> list[ReplayTimestamp]:
    """Load per-camera monotonic timestamps and normalize them to replay t=0."""
    raw_rows: list[tuple[int, int, dict[str, int]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera_ns = {camera_id: int(row[f"{camera_id}_monotonic_ns"])
                         for camera_id in camera_ids}
            raw_rows.append((int(row["frame_idx"]), int(row["capture_unix_ns"]), camera_ns))
    if not raw_rows:
        raise ValueError(f"{path}: no timestamp rows")
    base_capture = raw_rows[0][1]
    base_camera = min(raw_rows[0][2].values())
    result = [ReplayTimestamp(
        frame_index=frame_index,
        capture_time=(capture_ns - base_capture) / 1e9,
        camera_times={camera_id: (value - base_camera) / 1e9
                      for camera_id, value in camera_ns.items()},
    ) for frame_index, capture_ns, camera_ns in raw_rows]
    for previous, current in zip(result, result[1:]):
        if current.capture_time < previous.capture_time:
            raise ValueError(f"{path}: capture timestamps are not monotonic")
    return result


def load_replay_site(manifest_path: str | Path, dataset_id: str,
                     calibration_override: str | None = None,
                     processing_scale: float | None = None) -> ReplaySite:
    """Build current-world profiles from imported configuration data."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    root = (manifest_path.parent / manifest.get("asset_root", ".")).resolve()
    datasets = {item["id"]: item for item in manifest.get("datasets", [])}
    if dataset_id not in datasets:
        raise KeyError(f"unknown dataset {dataset_id!r}; choices={','.join(sorted(datasets))}")
    dataset = datasets[dataset_id]
    profile_id = calibration_override or dataset.get("calibration_profile")
    if not profile_id:
        raise ValueError(
            f"{dataset_id}: calibration is unconfirmed; pass --calibration only after checking it"
        )
    profile_specs = manifest.get("calibration_profiles", {})
    if profile_id not in profile_specs:
        raise KeyError(f"unknown calibration profile {profile_id!r}")

    bootstrap = manifest.get("demo_bootstrap", {})
    scale = float(processing_scale if processing_scale is not None
                  else bootstrap.get("processing_scale", 1.0))
    if not 0.1 <= scale <= 1.0:
        raise ValueError("processing_scale must be in [0.1, 1.0]")
    world_scale = float(bootstrap.get("source_to_runtime_scale", 1.0))
    calibration_path = _resolve(root, profile_specs[profile_id]["file"])
    calibration_data = _read_json(calibration_path)
    declared_unit = calibration_data.get("world", {}).get("unit")
    if declared_unit != bootstrap.get("source_world_unit"):
        raise ValueError(f"calibration declares {declared_unit!r}, bootstrap expects "
                         f"{bootstrap.get('source_world_unit')!r}")

    cameras = manifest.get("cameras", {})
    camera_ids = tuple(sorted(cameras))
    roi_polygons: dict[str, np.ndarray] = {}
    pixel_slots: dict[str, dict[str, np.ndarray]] = {}
    expected_area: dict[str, float] = {}
    for camera_id, spec in cameras.items():
        roi = _read_json(_resolve(root, spec["roi_file"]))
        slots = _read_json(_resolve(root, spec["pixel_slots_file"]))
        roi_polygons[camera_id] = _pixel_polygon(roi["polygon"])
        pixel_slots[camera_id] = {
            slot["id"]: _pixel_polygon(slot["polygon"]) * scale for slot in slots["slots"]
        }
        areas = [_polygon_area(polygon) for polygon in pixel_slots[camera_id].values()]
        expected_area[camera_id] = (
            float(np.median(areas))
            * float(bootstrap.get("expected_vehicle_to_slot_area_ratio", 0.55))
        )

    quality = calibration_data.get("calibration_quality", {}).get("cameras", {})
    transforms = calibration_data["camera_transforms"]
    profiles: dict[str, CameraProfile] = {}
    sigma_floor = float(bootstrap.get("calibration_sigma_floor_m", 0.005))
    runtime_dimensions = tuple(float(value)
                               for value in bootstrap.get("vehicle_dimensions_m", [0.08, 0.04]))
    pixel_rescale = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    for camera_id, spec in cameras.items():
        points = quality.get(camera_id, {}).get("points", [])
        pixel_points = np.asarray([point["pixel"] for point in points], dtype=float) * scale
        world_points = (np.asarray([point["world_cm"] for point in points], dtype=float)
                        * world_scale)
        residuals = (np.asarray([point.get("error_cm", 0.0) for point in points], dtype=float)
                     * world_scale)
        h = np.asarray(transforms[camera_id], dtype=float)
        h[:2, :] *= world_scale
        h = h @ pixel_rescale
        calibration = HomographyCalibration(
            camera_id=camera_id,
            h=h,
            pixel_points=pixel_points,
            world_points=world_points,
            residuals=residuals,
            sigma_calib=(sigma_floor ** 2) * np.eye(2),
            sigma_floor=sigma_floor,
        )
        source_width, source_height = spec["resolution_px"]
        width, height = int(round(source_width * scale)), int(round(source_height * scale))
        camera_ground_point = project_points(h, np.asarray([[width / 2.0, height]], dtype=float))[0]
        profiles[camera_id] = CameraProfile(
            camera_id=camera_id,
            calibration=calibration,
            width=width,
            height=height,
            ground_direction=np.asarray(
                bootstrap.get("ground_direction_px", {}).get(camera_id, [0.0, 1.0]),
                dtype=float,
            ),
            camera_ground_point=camera_ground_point,
            vehicle_dimensions=runtime_dimensions,
            vehicle_height=float(bootstrap.get("vehicle_height_m", 0.035)),
            expected_vehicle_area=expected_area[camera_id],
            parallax_gain=0.0,
            anchor_bias=0.0,
            anchor_bias_sigma=sigma_floor,
        )

    coverage = calibration_data["camera_coverage_world"]
    overlap = np.asarray(calibration_data["overlap_world_polygon"], dtype=float) * world_scale
    zones = {
        camera_id: CameraZone(
            camera_id=camera_id,
            fov_polygon=np.asarray(coverage[camera_id], dtype=float) * world_scale,
        ) for camera_id in camera_ids
    }
    edges: dict[tuple[str, str], TopologyEdge] = {}
    for item in calibration_data.get("edge_adjacency", []):
        source, target = item["source_camera"], item["target_camera"]
        zones[source].exit_polygons[target] = overlap
        zones[target].entry_polygons[source] = overlap
        edges[(source, target)] = TopologyEdge(source, target, dt_max=60.0, v_max=1.5)
    topology = CameraTopology(
        zones=zones,
        edges=edges,
        overlaps={(camera_ids[0], camera_ids[1]): overlap},
    )

    world_slots: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        for slot in calibration_data["parking_slots_world"][camera_id]:
            slot_id = slot["id"]
            if slot_id in world_slots:
                raise ValueError(f"duplicate physical slot id {slot_id}")
            world_slots[slot_id] = np.asarray(slot["polygon"], dtype=float) * world_scale

    dataset_directory = _resolve(root, dataset["directory"])
    video_paths = {camera_id: (dataset_directory / name).resolve()
                   for camera_id, name in dataset["raw_videos"].items()}
    for path in video_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    timestamp_path = dataset_directory / dataset["timestamps"]
    timestamps = load_timestamps(timestamp_path, camera_ids)
    ground_truth_paths = {
        label_type: (dataset_directory / name).resolve()
        for label_type, name in dataset.get("ground_truth", {}).items()
    }
    return ReplaySite(
        manifest_path=manifest_path,
        dataset_id=dataset_id,
        dataset_directory=dataset_directory,
        video_paths=video_paths,
        timestamp_path=timestamp_path,
        ground_truth_paths=ground_truth_paths,
        calibration_path=calibration_path,
        timestamps=timestamps,
        profiles=profiles,
        topology=topology,
        world_slots=world_slots,
        roi_polygons=roi_polygons,
        pixel_slots=pixel_slots,
        processing_scale=scale,
        source_world_unit=str(declared_unit),
        runtime_world_unit=str(bootstrap.get("runtime_world_unit", "m")),
        calibration_profile=profile_id,
        bootstrap_warning=(
            "BOOTSTRAP ONLY: 4 calibration points/camera; ground direction and "
            "miniature vehicle dimensions still require field confirmation"
        ),
    )


def build_replay_pipeline(site: ReplaySite) -> TechgarPipeline:
    """Create a fresh Stage 1-10 pipeline with scale-aware miniature-site units."""
    config = TechgarConfig()
    config.ingestion.max_pair_skew = 0.120
    config.background.init_frames = 8
    config.projection.sigma_pixel_u *= site.processing_scale
    config.projection.sigma_pixel_v *= site.processing_scale
    config.projection.sigma_parallax = 0.005
    config.projection.rho_seam = 0.015
    config.world_kalman.q = 0.02
    config.world_kalman.q_size = 0.001
    config.world_kalman.r0 = 0.000025
    config.association.direction_min_speed = 0.01
    config.association.handoff_dt_max = 60.0
    config.association.margin_min = 0.10
    config.identity.v_max_world = 1.0
    config.identity.collision_separation = 0.03
    config.identity.new_identity_min_displacement_m = 0.015
    config.identity.t_maturity = 0.20
    config.identity.tau_margin = 0.0
    config.identity.t_max_missing = 60.0
    config.identity.t_retire_idle = 120.0
    config.identity.t_grace = 5.0
    config.identity.t_display_hold = 15.0
    config.local_track.min_visible_count = 3
    config.slot.tau_center = 0.025
    config.slot.tau_inward = 0.008
    config.slot.sigma2_stable = 0.0001
    config.slot.v_parked = 0.025
    config.slot.vision_confirm_frames = 1
    # The imported datasets do not contain commissioned empty-slot reference
    # images.  Learning "empty" from frame 1 is circular when vehicles are
    # already parked and produced 20+ false anonymous occupied slots.  Keep the
    # independent vision channel fail-closed until reference assets are supplied.
    config.slot.enable_vision_fusion = False
    config.perf.overload_stage_budget = 0.500
    pipeline = TechgarPipeline(site.profiles, site.topology, site.world_slots, config,
                               pixel_slots=site.pixel_slots)
    pipeline.tracking_masks = {c: roi_mask(site, c) for c in site.camera_ids}
    for det in pipeline.vision_detectors.values():
        det.config.warmup_frames = 0
        det.config.confirm_frames = 1
        if det._ensemble_detector:
            det._ensemble_detector.smoothing_frames = 1
    pipeline.keep_history = False
    return pipeline


def roi_mask(site: ReplaySite, camera_id: str) -> np.ndarray:
    """Rasterize one imported ROI at the configured processing resolution."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install video support: pip install -e .[video]") from exc
    profile = site.profiles[camera_id]
    polygon = np.rint(site.roi_polygons[camera_id] * site.processing_scale).astype(np.int32)
    mask = np.zeros((profile.height, profile.width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask.astype(bool)


def iter_decoded_pairs(site: ReplaySite, limit: int | None = None
                       ) -> Iterator[tuple[ReplayTimestamp, dict[str, np.ndarray]]]:
    """Decode matching frame indices from both MP4 files."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Install video support: pip install -e .[video]") from exc
    captures = {camera_id: cv2.VideoCapture(str(path))
                for camera_id, path in site.video_paths.items()}
    try:
        failed = [camera_id for camera_id, capture in captures.items() if not capture.isOpened()]
        if failed:
            raise RuntimeError(f"cannot open video for {','.join(failed)}")
        rows = site.timestamps if limit is None else site.timestamps[:limit]
        for timestamp in rows:
            frames: dict[str, np.ndarray] = {}
            for camera_id, capture in captures.items():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"{camera_id}: decode failed at frame {timestamp.frame_index}"
                    )
                frames[camera_id] = frame
            yield timestamp, frames
    finally:
        for capture in captures.values():
            capture.release()


def process_pair(site: ReplaySite, pipeline: TechgarPipeline, timestamp: ReplayTimestamp,
                 frames: dict[str, np.ndarray], masks: dict[str, np.ndarray]
                 ) -> StepResult | None:
    """Resize/mask a decoded pair and submit it using recorded camera timestamps."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install video support: pip install -e .[video]") from exc
    results: list[StepResult] = []
    for camera_id in site.camera_ids:
        profile = site.profiles[camera_id]
        image = cv2.resize(frames[camera_id], (profile.width, profile.height),
                           interpolation=cv2.INTER_AREA)
        record = FrameRecord(
            camera_id=camera_id,
            sequence=timestamp.frame_index,
            timestamp=timestamp.camera_times[camera_id],
            width=profile.width,
            height=profile.height,
            decode_ok=True,
            image=image,
        )
        results.extend(pipeline.submit(record))
    return results[-1] if results else None
