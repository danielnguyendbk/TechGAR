"""World-side tuning surface (stages 5-10) plus the composed root config."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config_vision import (BackgroundConfig, DetectionConfig, IngestionConfig, KalmanConfig,
                            LocalTrackConfig, MotionConfig, ShadowConfig, ThresholdConfig)


@dataclass
class ProjectionConfig:
    """PLAN 2 §3.2 — Sigma_w = J Sigma_p J^T + Sigma_calib + Sigma_parallax."""

    sigma_pixel_u: float = 2.0
    sigma_pixel_v: float = 3.0          # ground-contact row is the noisier axis
    quality_pixel_gain: float = 3.0     # low quality inflates Sigma_p
    sigma_parallax: float = 0.05        # world units, vehicle-height induced
    rho_seam: float = 0.15              # measured in Phase 0, injected here
    border_margin_px: float = 24.0
    border_inflation: float = 2.0
    calibration_scale: float = 1.0


@dataclass
class FusionConfig:
    """PLAN 1 stage 6."""

    max_skew: float = 0.120
    gate: float = 9.210                 # chi2(2, 0.99)
    appearance_weight: float = 2.0
    appearance_max: float = 0.65
    min_footprint_iou: float = 0.05
    max_area_log_ratio: float = 0.80
    overlap_expansion_sigma: float = 3.0  # uncertainty-driven zone expansion


@dataclass
class AssociationConfig:
    """PLAN 2 §4 cost weights and §4.7 margin."""

    w_distance: float = 1.0
    w_direction: float = 2.0
    w_geometry: float = 1.0
    w_appearance: float = 3.0
    eta_aspect: float = 0.5
    alpha_appearance: float = 0.60       # robust gallery mix (PLAN 2 §4.4)
    gate: float = 9.210
    margin_min: float = 0.75             # M_min
    direction_min_speed: float = 0.30
    direction_reliability_gap: float = 0.80
    handoff_dt_min: float = 0.0
    handoff_dt_max: float = 4.0
    time_soft: bool = True
    #: Ablation experiment D (PLAN 3 §7): drop topology gating entirely.
    enable_topology: bool = True
    enable_appearance: bool = True


@dataclass
class IdentityConfig:
    """PLAN 2 §6 retention / anti-fragmentation constants."""

    tau_accept: float = 0.45
    tau_candidate: float = 0.30
    tau_margin: float = 0.06
    t_grace: float = 2.00                # no new Global ID while a hypothesis lives
    t_max_missing: float = 8.00
    t_maturity: float = 0.25
    n_maturity: int = 2
    new_identity_min_displacement_m: float = 0.0  # 0.0 for unit tests, set to >= 0.04 for real video replays
    active_grace_period_s: float = 0.0  # 0.0 for unit tests, set to >= 0.35 for real video replays
    max_identities: int | None = None   # Hard cap on active Global IDs if known for site/dataset
    t_retire_idle: float = 20.0
    t_display_hold: float = 6.0
    v_max_world: float = 12.0            # physical speed bound, world units / s
    collision_separation: float = 1.5    # two observations this far apart = 2 vehicles
    #: When True, GID can only be minted at an entry gate crossing in valid
    #: direction.  Mid-lot detections are tagged UNKNOWN and never minted.
    #: Default False for backward compatibility with existing synthetic tests.
    require_entry_gate: bool = False
    #: Recovery window after restart (seconds).  Identities in RECOVERY_PENDING
    #: state have this long to be re-observed before retirement.
    recovery_window: float = 15.0


@dataclass
class SlotConfig:
    """PLAN 2 §5 (numbers from §5.6 and PLAN 3 scenarios G/H/I)."""

    tau_iou: float = 0.50
    tau_coverage: float = 0.75
    tau_center: float = 1.20
    tau_inward: float = 0.40
    n_iou: int = 3
    n_coverage: int = 3
    window: float = 2.00                 # sliding *time* window
    sigma2_stable: float = 0.09
    v_parked: float = 0.40
    tail_fraction: float = 0.50          # "phần cuối cửa sổ" for the speed test
    tau_enter: float = 0.55              # hysteresis: enter > confirm > release
    tau_confirm: float = 0.45
    tau_release: float = 0.30
    release_duration: float = 1.50
    claim_grace: float = 3.00
    score_margin: float = 0.08           # deterministic adjacent-slot resolution
    dwell_confirm: float = 0.80
    footprint_uncertainty_inflation: bool = True
    enable_temporal_window: bool = True
    enable_hysteresis: bool = True
    #: Vision fusion (PLAN 1 stage 9): pixel-content occupancy votes must
    #: agree for this many frames before they flip / release a slot.
    vision_confirm_frames: int = 2
    vision_release_frames: int = 3
    enable_vision_fusion: bool = True


@dataclass
class SessionConfig:
    require_exit_for_delete: bool = True
    fingerprint_gallery: int = 8
    t_orphan_hold: float = 60.0


@dataclass
class PerfConfig:
    """PLAN 1 Phase 6."""

    queue_max: int = 2
    slot_period: float = 0.20            # slot analysis runs at a lower rate
    overload_stage_budget: float = 0.100
    overload_uncertainty_gain: float = 2.0
    drop_stale_frames: bool = True
    encode_video_when_subscribed: bool = True


@dataclass
class AblationFlags:
    """PLAN 3 §7 experiment switches, surfaced in one place."""

    name: str = "full"
    frame_difference: bool = True
    prediction: bool = True
    topology: bool = True
    appearance: bool = True
    temporal_slot_window: bool = True
    hysteresis: bool = True


@dataclass
class TechgarConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    pixel_kalman: KalmanConfig = field(default_factory=KalmanConfig)
    world_kalman: KalmanConfig = field(
        default_factory=lambda: KalmanConfig(q=2.0, q_size=0.5, r0=0.04, dt_threshold=0.25))
    local_track: LocalTrackConfig = field(default_factory=LocalTrackConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    slot: SlotConfig = field(default_factory=SlotConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
    ablation: AblationFlags = field(default_factory=AblationFlags)
    strict_contracts: bool = True

    def apply_ablation(self, flags: AblationFlags) -> "TechgarConfig":
        """Return a copy with the ablation switches pushed into every component."""
        import copy

        cfg = copy.deepcopy(self)
        cfg.ablation = flags
        cfg.motion.enable_frame_difference = flags.frame_difference
        cfg.pixel_kalman.enable_prediction = flags.prediction
        cfg.world_kalman.enable_prediction = flags.prediction
        cfg.association.enable_topology = flags.topology
        cfg.association.enable_appearance = flags.appearance
        cfg.slot.enable_temporal_window = flags.temporal_slot_window
        cfg.slot.enable_hysteresis = flags.hysteresis
        return cfg
