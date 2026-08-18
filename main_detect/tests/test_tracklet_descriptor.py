import numpy as np

from techgar.tracklet_descriptor import AppearanceTracklet, compare_tracklets


def histogram(bin_index: int) -> np.ndarray:
    value = np.zeros((16, 16), dtype=np.float32)
    value.flat[bin_index] = 1.0
    return value


def test_tracklet_samples_over_time_and_keeps_a_bounded_recent_gallery():
    tracklet = AppearanceTracklet(max_samples=2, sample_interval=3)

    assert tracklet.update(histogram(1), 1)
    assert not tracklet.update(histogram(2), 2)
    assert tracklet.update(histogram(2), 4)
    assert tracklet.update(histogram(3), 7)

    assert tracklet.sample_frames == [4, 7]
    assert len(tracklet.samples) == 2


def test_tracklet_match_uses_route_samples_instead_of_only_last_appearance():
    source = AppearanceTracklet(max_samples=4, sample_interval=1)
    source.update(histogram(1), 1)
    source.update(histogram(2), 2)
    target = AppearanceTracklet(max_samples=4, sample_interval=1)
    target.update(histogram(2), 3)

    match = compare_tracklets(source, target)

    assert match.support == 1
    assert match.sample_pairs == 2
    assert match.distance < 0.45
    assert compare_tracklets(histogram(1), histogram(2)).distance > 0.90
