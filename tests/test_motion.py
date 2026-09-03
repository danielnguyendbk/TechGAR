"""Dual-evidence motion-mask reconstruction tests."""

import numpy as np

from techgar.motion import DifferenceResult, dual_stage_gate


def test_difference_seed_recovers_complete_background_component():
    background = np.zeros((12, 18), dtype=bool)
    background[3:9, 2:8] = True
    background[2:6, 12:16] = True  # unrelated, stationary foreground evidence
    difference = np.zeros_like(background)
    difference[3:9, 7] = True      # only the moving object's thin rim changed

    gated = dual_stage_gate(
        background,
        DifferenceResult(mask=difference, deltas_used={"long": 0.5}),
    )

    assert np.array_equal(gated[3:9, 2:8], np.ones((6, 6), dtype=bool))
    assert not gated[2:6, 12:16].any()


def test_difference_seed_does_not_bridge_disconnected_background_components():
    background = np.zeros((10, 15), dtype=bool)
    background[2:8, 1:5] = True
    background[2:8, 7:11] = True
    difference = np.zeros_like(background)
    difference[4, 3] = True

    gated = dual_stage_gate(
        background,
        DifferenceResult(mask=difference, deltas_used={"short": 0.1}),
    )

    assert gated[2:8, 1:5].all()
    assert not gated[2:8, 7:11].any()
