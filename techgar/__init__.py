"""TechGAR — indoor multi-camera vehicle tracking with a single identity authority.

Fresh implementation of PLAN 1 (pipeline & roadmap), PLAN 2 (mathematics) and
PLAN 3 (benchmarks & rubric).  No GPS anywhere: every position lives in the
calibrated local world frame (see :mod:`techgar.units`).

Core invariant enforced by :mod:`techgar.registry`::

    one physical vehicle  <->  exactly one active Global ID
"""

__version__ = "1.0.0"

CORE_INVARIANT = "one physical vehicle <-> exactly one active Global ID"

# The pipeline stages of PLAN 1 §3, in execution order.  Used by the perf
# instrumentation (PLAN 1 Phase 6) and by the rubric evidence collector.
PIPELINE_STAGES = (
    "s1_ingestion",
    "s2_normalization",
    "s3_detection",
    "s4_local_tracking",
    "s5_projection",
    "s6_fusion",
    "s7_association",
    "s8_registry",
    "s9_slots",
    "s10_dispatch",
)
