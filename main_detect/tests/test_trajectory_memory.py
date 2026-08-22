import pytest

from techgar.trajectory_memory import TrajectorySample, WorldTrajectoryMemory


def sample(frame, timestamp, x, y=0.0, camera="cam1", local_id=1):
    return TrajectorySample(
        frame_idx=frame,
        timestamp_s=timestamp,
        camera_id=camera,
        local_track_id=local_id,
        world=(float(x), float(y)),
        bbox_size=(20, 30),
    )


def test_history_is_time_bounded_and_velocity_uses_recent_median():
    memory = WorldTrajectoryMemory(history_seconds=2.0)
    for frame, timestamp, x in (
        (1, 0.0, 0),
        (2, 1.0, 10),
        (3, 2.0, 20),
        (4, 3.1, 31),
    ):
        memory.append_global(1, sample(frame, timestamp, x))

    samples = memory.global_samples(1)
    assert [item.frame_idx for item in samples] == [3, 4]
    estimate = memory._estimate(list(samples))
    assert estimate.speed == pytest.approx(10.0)


def test_curvature_requires_six_stable_samples():
    memory = WorldTrajectoryMemory()
    five = [sample(index, index * 0.1, index) for index in range(5)]
    assert memory._estimate(five).curvature_rad is None

    six = five + [sample(5, 0.5, 5)]
    assert memory._estimate(six).curvature_rad == pytest.approx(0.0)


def test_promote_alias_merge_and_parked_freeze():
    memory = WorldTrajectoryMemory()
    key = ("cam1", 7)
    memory.append_provisional(key, sample(1, 0.0, 0, local_id=7))
    memory.promote(key, 2)
    assert not memory.provisional_samples(key)
    assert len(memory.global_samples(2)) == 1

    memory.append_global(3, sample(2, 0.1, 1, local_id=8))
    memory.merge(2, 3)
    assert len(memory.global_samples(2)) == 2
    assert not memory.global_samples(3)

    memory.set_parked(2, True, origin=(2.0, 0.0))
    memory.append_global(2, sample(3, 0.2, 99, local_id=9))
    assert len(memory.global_samples(2)) == 2
    assert memory.parked_origin(2) == (2.0, 0.0)


def test_world_match_accepts_clear_corridor_and_rejects_opposite_or_teleport():
    memory = WorldTrajectoryMemory(history_seconds=5.0)
    for index, x in enumerate((0, 10, 20)):
        memory.append_global(2, sample(index, index, x))

    good_key = ("cam2", 4)
    for index, x in enumerate((25, 35, 45), start=3):
        memory.append_provisional(
            good_key, sample(index, 2.5 + (index - 3), x, camera="cam2", local_id=4)
        )
    good = memory.match(
        2,
        good_key,
        prediction_radius=30,
        recent_window_s=3,
        appearance_score=1.0,
        size_score=1.0,
        topology_score=1.0,
        min_direction_cosine=-0.35,
        source_camera="cam1",
    )
    assert good.hard_reject_reason is None
    assert good.score >= 0.78

    reverse_key = ("cam2", 5)
    for index, x in enumerate((25, 15, 5), start=3):
        memory.append_provisional(
            reverse_key, sample(index, 2.5 + (index - 3), x, camera="cam2", local_id=5)
        )
    reverse = memory.match(
        2,
        reverse_key,
        prediction_radius=30,
        recent_window_s=3,
        appearance_score=1.0,
        size_score=1.0,
        topology_score=1.0,
        min_direction_cosine=-0.35,
        source_camera="cam1",
    )
    assert reverse.hard_reject_reason == "stable_wrong_direction"

    teleport_key = ("cam2", 6)
    for index, x in enumerate((100, 110, 120), start=3):
        memory.append_provisional(
            teleport_key, sample(index, 2.5 + (index - 3), x, camera="cam2", local_id=6)
        )
    teleport = memory.match(
        2,
        teleport_key,
        prediction_radius=30,
        recent_window_s=3,
        appearance_score=1.0,
        size_score=1.0,
        topology_score=1.0,
        min_direction_cosine=-0.35,
        source_camera="cam1",
    )
    assert teleport.hard_reject_reason == "teleport"


def test_departure_uses_frozen_slot_origin_and_continuous_outward_motion():
    memory = WorldTrajectoryMemory(history_seconds=5.0)
    memory.set_parked(2, True, origin=(0.0, 0.0))
    key = ("cam2", 9)
    for index, x in enumerate((1, 3, 6), start=1):
        memory.append_provisional(
            key, sample(index, index * 0.1, x, camera="cam2", local_id=9)
        )
    evidence = memory.match_departure(
        2,
        key,
        prediction_radius=20,
        recent_window_s=2,
        appearance_score=1.0,
        size_score=1.0,
        topology_score=1.0,
    )
    assert evidence.stable
    assert evidence.hard_reject_reason is None
    assert evidence.score >= 0.78

    inward_key = ("cam2", 10)
    for index, x in enumerate((6, 3, 1), start=1):
        memory.append_provisional(
            inward_key, sample(index, index * 0.1, x, camera="cam2", local_id=10)
        )
    inward = memory.match_departure(
        2,
        inward_key,
        prediction_radius=20,
        recent_window_s=2,
        appearance_score=1.0,
        size_score=1.0,
        topology_score=1.0,
    )
    assert inward.hard_reject_reason in {
        "stable_not_leaving_slot",
        "stable_wrong_direction",
    }
