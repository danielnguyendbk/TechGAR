"""Camera-side tuning surface (stages 1-4).  Every constant that PLAN 2 names
appears here exactly once, so an ablation only has to swap a config object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestionConfig:
    """PLAN 1 stage 1."""

    max_pair_skew: float = 0.120        # accept 35 ms (Pass), reject 2.28 s (Fail)
    buffer_depth: int = 1               # latest-frame buffer, never a queue
    stale_frame_age: float = 0.500      # drop rather than accumulate backlog
    require_all_cameras: bool = False   # a single-camera pair is still processable


@dataclass
class ThresholdConfig:
    """PLAN 2 §1.4:  tau_t = tau_0 + alpha*sigma_noise + beta*|L_t - L_{t-1}| + gamma*E_illum."""

    tau_0: float = 10.0
    alpha: float = 3.0
    beta: float = 1.5
    gamma: float = 30.0
    tau_min: float = 6.0
    tau_max: float = 90.0
    noise_block: int = 32               # block size for the local MAD estimate
    illumination_change_px: float = 8.0  # |dI| above which a pixel "changed"
    illumination_fraction_ref: float = 0.25  # E_illum = 1 at this changed fraction


@dataclass
class BackgroundConfig:
    """Adaptive background estimate B_t (PLAN 2 §1.1)."""

    learn_rate: float = 0.020          # background pixels
    absorb_rate: float = 0.0015        # foreground pixels: slow absorption
    init_frames: int = 5
    variance_learn_rate: float = 0.02


@dataclass
class MotionConfig:
    """Dual-stage foreground evidence (PLAN 2 §1.2, §1.3)."""

    delta_short: float = 0.10
    delta_long: float = 0.50
    delta_lag_max: float = 1.50        # Delta_lag only inside this gap bound
    diff_threshold_scale: float = 0.75  # tau_diff = scale * tau_t
    min_blob_area: int = 120
    min_fill_ratio: float = 0.20
    open_iterations: int = 1
    instability_fraction: float = 0.35  # PLAN 1 stage 2 logic 9
    low_confidence_area: int = 80       # keep as candidate evidence, do not delete
    #: Ablation experiment B (PLAN 3 §7): drop the frame-difference channel.
    enable_frame_difference: bool = True


@dataclass
class ShadowConfig:
    """PLAN 2 §1.5 — all conditions must hold, with a fail-open escape."""

    a_min: float = 0.32
    a_max: float = 0.62
    epsilon_chroma: float = 0.12
    texture_correlation_min: float = 0.35
    edge_support_ratio: float = 0.18    # condition 4: independent boundary support
    enable: bool = True


@dataclass
class DetectionConfig:
    """PLAN 1 stage 3."""

    expected_vehicle_area: float = 2400.0   # px^2, per camera scale
    merged_area_factor: float = 1.55        # "lớn vượt footprint xe kỳ vọng"
    min_confidence: float = 0.20            # keep low-confidence candidates
    high_confidence: float = 0.55           # tier-1 threshold
    footprint_band_ratio: float = 0.35      # bottom band = ground contact patch
    max_detections: int = 32
    peak_erosion: int = 2
    border_margin: float = 6.0               # blob within this of the edge = partial
    min_mint_area_ratio: float = 0.55        # a partial sliver may never mint an identity                   # internal motion peak counting


@dataclass
class KalmanConfig:
    """PLAN 2 §2.  Instantiated twice: pixel space (local tracks) and world
    space (global identities)."""

    q: float = 900.0                  # process noise density for Q(dt)
    q_size: float = 200.0             # W, H, R random walk
    r0: float = 9.0                   # R_0 in PLAN 2 §2.4
    lam: float = 1.5                  # lambda: (1 - confidence) weight
    mu: float = 2.0                   # occlusion weight
    nu: float = 1.0                   # seam / perspective weight
    dt_threshold: float = 0.25         # VR1 trigger
    substep: float = 0.10              # h ~ 100 ms
    velocity_damping: float = 0.50     # lambda_d
    max_dt: float = 5.0
    #: Ablation experiment C (PLAN 3 §7): no velocity / covariance prediction.
    enable_prediction: bool = True


@dataclass
class LocalTrackConfig:
    """PLAN 1 stage 4 / PLAN 2 §2.5 — all thresholds are *durations*."""

    t_missed: float = 0.12
    t_occluded: float = 0.30
    t_reacquire: float = 0.60
    t_retire: float = 2.50
    gate_confidence: float = 0.99
    template_size: int = 24
    template_ncc_min: float = 0.50
    template_search: int = 48
    template_confidence: float = 0.40
    min_new_track_confidence: float = 0.30
    new_track_block_iou: float = 0.30   # an existing hypothesis explains it
    max_tracks: int = 64
    merged_coverage: float = 0.55
    t_retire_border: float = 0.40   # a track at the image edge has left the camera
    max_blind_recoveries: int = 3   # template matching may not resurrect forever        # a detection covering >=2 predictions
