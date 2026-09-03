from __future__ import annotations

import numpy as np

from techgar.config_vision import IngestionConfig
from techgar.contracts import EnvironmentQuality, FrameRecord, LocalDetection, LocalTrackState, TopologyRegion
from techgar.fusion import WorldFusion
from techgar.geometry import polygon_area
from techgar.ingestion import DualStreamIngestion
from techgar.local_tracker import LocalTracker
from techgar.normalization import NormalizedFrame
from techgar.perf import BoundedQueue, SubscriberGate, ThroughputMeter
from techgar.pipeline import TechgarPipeline
from techgar.projection import WorldProjector
from techgar.simulation.layouts import build_profiles, overlap_layout
from techgar.world_contracts import WorldDetection
from techgar.contracts import MeasurementSource
from techgar.template import TemplateMatch


def _frame(camera_id: str, sequence: int, timestamp: float) -> FrameRecord:
    return FrameRecord(camera_id, sequence, timestamp, 8, 8, True, np.zeros((8, 8, 3), dtype=np.uint8))


def test_latest_frame_buffer_replaces_backlog_and_pairs_by_timestamp():
    ingestion = DualStreamIngestion(("C1", "C2"), IngestionConfig(max_pair_skew=0.12))
    assert ingestion.submit(_frame("C1", 1, 0.00))
    assert ingestion.submit(_frame("C1", 2, 0.10))
    assert ingestion.health["C1"].frames_replaced == 1
    assert ingestion.submit(_frame("C2", 1, 0.11))
    pair = ingestion.try_pair()
    assert pair is not None and pair.camera_ids == ("C1", "C2")
    assert pair.frames["C1"].sequence == 2

    late = DualStreamIngestion(("C1", "C2"), IngestionConfig(max_pair_skew=0.12))
    late.submit(_frame("C1", 1, 0.00))
    late.submit(_frame("C2", 1, 2.28))
    rejected = late.try_pair()
    assert rejected is not None and len(rejected.frames) == 1
    assert rejected.reject_reason.startswith("skew_")


def _normalized(profile, timestamp: float, sequence: int) -> NormalizedFrame:
    shape = (profile.height, profile.width)
    gray = np.zeros(shape, dtype=np.float32)
    return NormalizedFrame(
        camera_id=profile.camera_id,
        timestamp=timestamp,
        frame_sequence=sequence,
        gray=gray,
        color=None,
        foreground=np.zeros(shape, dtype=bool),
        shadow=np.zeros(shape, dtype=bool),
        candidates=np.zeros(shape, dtype=bool),
        background=gray,
        tau_map=gray,
        quality=EnvironmentQuality(profile.camera_id, timestamp, 1.0, 100.0, 0.0, 0.0, 0.0, 10.0),
    )


def _detection(profile, detection_id: int, timestamp: float, x: float, *, merged=False) -> LocalDetection:
    width = 120.0 if merged else 42.0
    bbox = np.array([x - width / 2, 150.0, x + width / 2, 205.0])
    return LocalDetection(
        camera_id=profile.camera_id,
        timestamp=timestamp,
        frame_sequence=round(timestamp * 10),
        bbox=bbox,
        confidence=0.95,
        local_center=np.array([x, 177.5]),
        ground_anchor=np.array([x, 205.0]),
        footprint_pixels=np.array([[x - 20, 205], [x + 20, 205], [x + 20, 190], [x - 20, 190]]),
        mask_area=profile.expected_vehicle_area,
        quality_score=1.0,
        occlusion_group_candidate=merged,
        detection_id=detection_id,
    )


def test_local_tracks_survive_measured_lag_and_merged_blob_keeps_latent_tracks():
    profile = build_profiles(overlap_layout())["C1"]
    tracker = LocalTracker(profile)
    first = tracker.step(_normalized(profile, 0.0, 0), [_detection(profile, 1, 0.0, 180.0)])
    track_id = first[0].local_track_id
    held = tracker.step(_normalized(profile, 0.5, 5), [])
    assert held[0].local_track_id == track_id
    assert held[0].state is not LocalTrackState.RETIRED
    recovered = tracker.step(_normalized(profile, 1.0, 10), [_detection(profile, 2, 1.0, 180.0)])
    assert recovered[0].local_track_id == track_id
    assert len(tracker.tracks) == 1

    grouped = LocalTracker(profile)
    grouped.step(_normalized(profile, 0.0, 0), [
        _detection(profile, 10, 0.0, 160.0),
        _detection(profile, 11, 0.0, 210.0),
    ])
    latent = grouped.step(_normalized(profile, 0.1, 1), [_detection(profile, 12, 0.1, 185.0, merged=True)])
    assert len(latent) == 2
    assert all(observation.latent and observation.state is LocalTrackState.MERGED for observation in latent)


def test_relaxed_local_recovery_is_one_to_one_per_frame():
    profile = build_profiles(overlap_layout())["C1"]
    tracker = LocalTracker(profile)
    tracker.step(_normalized(profile, 0.0, 0), [_detection(profile, 1, 0.0, 180.0)])
    tracker.step(_normalized(profile, 0.5, 5), [])
    track = next(iter(tracker.tracks.values()))
    detections = [
        _detection(profile, 2, 0.6, 178.0),
        _detection(profile, 3, 0.6, 182.0),
    ]
    recovered = tracker._assign_lost_tracks([track], detections, 0.6)
    assert list(recovered) == [track.local_track_id]
    assert len({det.detection_id for det in recovered.values()}) == 1


def test_template_recovery_is_not_relabelled_as_stale_detection(monkeypatch):
    profile = build_profiles(overlap_layout())["C1"]
    tracker = LocalTracker(profile)
    tracker.step(_normalized(profile, 0.0, 0), [_detection(profile, 1, 0.0, 180.0)])
    track = next(iter(tracker.tracks.values()))
    track.template = np.ones((12, 12), dtype=np.float32)
    monkeypatch.setattr(
        "techgar.local_tracker.match_template",
        lambda *args, **kwargs: TemplateMatch(np.array([181.0, 177.5]), 0.9, True),
    )
    observations = tracker.step(_normalized(profile, 0.2, 2), [])
    recovered = observations[0]
    assert recovered.source is MeasurementSource.TEMPLATE
    assert recovered.extras["detection"] is None
    assert recovered.timestamp == 0.2


def _world(camera_id: str, position, covariance, observation_id: int, footprint) -> WorldDetection:
    return WorldDetection(
        camera_id=camera_id,
        timestamp=1.0,
        frame_sequence=1,
        world_position=np.asarray(position),
        world_covariance=np.asarray(covariance),
        world_footprint=np.asarray(footprint),
        source_pixel_position=np.zeros(2),
        topology_region=TopologyRegion.OVERLAP,
        local_track_id=observation_id,
        quality=1.0,
        confidence=0.9,
        footprint_area=8.0,
        footprint_aspect=2.0,
        observation_id=observation_id,
    )


def test_projection_propagates_covariance_and_fusion_is_one_to_one():
    layout = overlap_layout()
    profiles = build_profiles(layout)
    profile = profiles["C1"]
    world_point = np.array([44.0, 14.0])
    pixel = layout.cameras["C1"].project_floor(world_point.reshape(1, 2))[0]
    detection = LocalDetection(
        camera_id="C1", timestamp=1.0, frame_sequence=1,
        bbox=np.array([pixel[0] - 20, pixel[1] - 50, pixel[0] + 20, pixel[1]]),
        confidence=0.9, local_center=pixel - np.array([0, 25]), ground_anchor=pixel,
        footprint_pixels=np.array([pixel + [-20, 0], pixel + [20, 0], pixel + [20, -10], pixel + [-20, -10]]),
        mask_area=profile.expected_vehicle_area, quality_score=0.9, detection_id=1,
    )
    projected = WorldProjector(profile, layout.topology).project_detection(detection)
    assert projected.world_covariance.shape == (2, 2)
    assert np.linalg.eigvalsh(projected.world_covariance).min() > 0

    overlap = next(iter(layout.topology.overlaps.values()))
    centre = overlap.mean(axis=0)
    footprint = np.array([centre + [-2, -1], centre + [2, -1], centre + [2, 1], centre + [-2, 1]])
    fused = WorldFusion(layout.topology).fuse([
        _world("C1", centre + [-0.05, 0], np.eye(2) * 0.3, 1, footprint),
        _world("C2", centre + [0.05, 0], np.eye(2) * 1.2, 2, footprint),
    ])
    assert len(fused) == 1
    assert fused[0].contributing_cameras == ("C1", "C2")
    assert np.trace(fused[0].covariance) < np.trace(np.eye(2) * 0.3)


def test_projection_reconstructs_metric_footprint_from_a_thin_motion_rim():
    layout = overlap_layout()
    profile = build_profiles(layout)["C1"]
    centre = np.array([22.0, 20.0])
    narrow_edge_world = np.array([centre + [-0.2, -2.3], centre + [0.2, -2.3]])
    narrow_edge_pixels = layout.cameras["C1"].project_floor(narrow_edge_world)
    detection = LocalDetection(
        camera_id="C1", timestamp=1.0, frame_sequence=1,
        bbox=np.array([100.0, 100.0, 120.0, 140.0]), confidence=0.9,
        local_center=narrow_edge_pixels.mean(axis=0),
        ground_anchor=narrow_edge_pixels.mean(axis=0),
        footprint_pixels=np.vstack([narrow_edge_pixels, narrow_edge_pixels]),
        mask_area=profile.expected_vehicle_area, quality_score=1.0, detection_id=1,
    )
    footprint = WorldProjector(profile, layout.topology).world_footprint(
        detection, narrow_edge_world.mean(axis=0), np.eye(2) * 0.01)
    length, width = profile.vehicle_dimensions
    assert np.isclose(polygon_area(footprint), length * width, rtol=0.05)


def test_performance_primitives_bound_backlog_measure_intervals_and_gate_encoding():
    queue = BoundedQueue(2)
    assert queue.push(1) and queue.push(2)
    assert not queue.push(3)
    assert len(queue) == 2 and queue.pop() == 2 and queue.dropped == 1

    meter = ThroughputMeter()
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        meter.tick(timestamp)
    assert meter.fps == 10.0

    subscribers = SubscriberGate()
    assert not subscribers.should_encode()
    subscribers.subscribe()
    assert subscribers.should_encode()
    assert subscribers.skipped == 1 and subscribers.encodes == 1


def test_pipeline_encodes_visualization_only_with_an_active_subscriber():
    layout = overlap_layout()
    encoded = []
    pipeline = TechgarPipeline(
        build_profiles(layout), layout.topology, {},
        visualization_encoder=lambda result: encoded.append(result.pair_sequence) or b"frame",
    )
    width, height = layout.cameras["C1"].width, layout.cameras["C1"].height
    image = np.zeros((height, width, 3), dtype=np.uint8)
    pipeline.submit(FrameRecord("C1", 1, 0.0, width, height, True, image))
    first = pipeline.submit(FrameRecord("C2", 1, 0.01, width, height, True, image))
    assert first and not encoded and pipeline.subscribers.skipped == 1

    pipeline.subscribers.subscribe()
    pipeline.submit(FrameRecord("C1", 2, 0.1, width, height, True, image))
    second = pipeline.submit(FrameRecord("C2", 2, 0.11, width, height, True, image))
    assert second and encoded == [second[0].pair_sequence]
    assert pipeline.latest_visualization == b"frame"
