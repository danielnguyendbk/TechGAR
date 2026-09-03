from techgar.evaluation.environment import run_environmental_checks


def test_environmental_robustness_probes_pass():
    checks = run_environmental_checks()
    assert checks == {
        "brightness_transition": True,
        "shadow_rejection": True,
        "compression_noise_bounded": True,
    }
