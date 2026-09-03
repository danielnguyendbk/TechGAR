"""Synthetic validation rig: virtual cameras, renderer, layouts, recordings.

There is no field footage in the repository, so PLAN 3's ground truth is
produced here instead: a fully known world (positions, identities, slot
ownership, visibility, occlusion) rendered into two oblique camera streams with
configurable skew, lag, lighting flicker, shadows and structural occluders.

The pipeline receives *only* pixels + timestamps, exactly as it would on site;
the ground truth stays inside this package and is handed to the evaluator, never
to the tracker.
"""

from .camera import VirtualCamera
from .layouts import Layout, gap_layout, overlap_layout, parking_layout
from .recording import Recording, RecordingOptions, build_recording
from .vehicles import SimVehicle, Waypoint

__all__ = [
    "Layout", "Recording", "RecordingOptions", "SimVehicle", "VirtualCamera", "Waypoint",
    "build_recording", "gap_layout", "overlap_layout", "parking_layout",
]
