"""Site layouts for the synthetic rig: camera poses, calibration surveys, slot
polygons, blind regions and the directed topology graph.

A layout plays the role of the *site survey*: it hands the production code only
what a real commissioning pass would produce — a noisy homography estimate per
camera plus hand-drawn world polygons — never the simulator's exact geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import clip_polygon, convex_hull, polygon_area, polygon_bounds
from ..homography import HomographyCalibration, calibrate
from ..profile import CameraProfile
from ..topology import CameraTopology, CameraZone, TopologyEdge
from .camera import VirtualCamera
from .vehicles import SimVehicle

FOCAL = 285.0
RESOLUTION = (480, 360)
FLOOR = (0.0, -2.0, 96.0, 30.0)
LANE_Y = 14.0
LANE_BAND = (9.0, 27.0)
SLOT_WIDTH, SLOT_DEPTH, SLOT_CY = 2.6, 5.4, 22.2
SLOT_PITCH = 2.9
BLOCKS = {"D": 8.0, "B": 56.0}
V_MAX = 25.0                      # world units / s, layout scale
CRUISE_SPEED = 8.0


def slot_polygon(centre_x: float, centre_y: float = SLOT_CY) -> np.ndarray:
    hw, hd = SLOT_WIDTH / 2.0, SLOT_DEPTH / 2.0
    return np.array([[centre_x - hw, centre_y - hd], [centre_x + hw, centre_y - hd],
                     [centre_x + hw, centre_y + hd], [centre_x - hw, centre_y + hd]])


def build_slots() -> dict[str, np.ndarray]:
    slots: dict[str, np.ndarray] = {}
    for block, x0 in BLOCKS.items():
        for k in range(10):
            slots[f"{block}{k + 1:02d}"] = slot_polygon(x0 + k * SLOT_PITCH)
    return slots


def slot_centre(slot_id: str, slots: dict[str, np.ndarray] | None = None) -> np.ndarray:
    slots = build_slots() if slots is None else slots
    return slots[slot_id].mean(axis=0)


def _band_rect(band=LANE_BAND, x0: float = -1e4, x1: float = 1e4) -> np.ndarray:
    return np.array([[x0, band[0]], [x1, band[0]], [x1, band[1]], [x0, band[1]]])


def _corridor(fov: np.ndarray, side: str, width: float, band=LANE_BAND,
              lane_y: float = LANE_Y) -> np.ndarray:
    """Strip of a camera's FOV against the given side of the drive lane.

    The extent is measured on the lane centre line (not over the whole band):
    the FOV is a trapezoid, so a band-wide measurement would put the corridor
    where no vehicle ever drives and the last pre-handoff observation would fall
    outside it.
    """
    lane = clip_polygon(fov, _band_rect((lane_y - 1.5, lane_y + 1.5)))
    if lane is None:
        raise ValueError("FOV does not reach the drive lane")
    lx0, _, lx1, _ = polygon_bounds(lane)
    if side == "right":
        x_lo, x_hi = lx1 - width, lx1
    else:
        x_lo, x_hi = lx0, lx0 + width
    return np.array([[x_lo, band[0]], [x_hi, band[0]], [x_hi, band[1]], [x_lo, band[1]]])


@dataclass
class Layout:
    name: str
    cameras: dict[str, VirtualCamera]
    calibrations: dict[str, HomographyCalibration]
    topology: CameraTopology
    slots: dict[str, np.ndarray] = field(default_factory=build_slots)
    blind_regions: dict[str, list[np.ndarray]] = field(default_factory=dict)
    floor_bounds: tuple = FLOOR
    lane_y: float = LANE_Y
    v_max: float = V_MAX

    @property
    def camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cameras))

    def calibration_report(self) -> list[dict]:
        return [self.calibrations[c].report() for c in self.camera_ids]

    def fov(self, camera_id: str) -> np.ndarray:
        return self.topology.zones[camera_id].fov_polygon


def _make_camera(camera_id: str, x: float) -> VirtualCamera:
    return VirtualCamera(camera_id, (x, -4.0, 20.0), (x, 6.0, 0.0), focal=FOCAL,
                         width=RESOLUTION[0], height=RESOLUTION[1])


def _estimated_fov(camera: VirtualCamera, calib: HomographyCalibration) -> np.ndarray:
    """FOV as the *system* knows it: image corners through the estimated H."""
    corners = np.array([[0.0, 0.0], [camera.width, 0.0],
                        [camera.width, camera.height], [0.0, camera.height]])
    quad = calib.project(corners)
    x0, y0, x1, y1 = FLOOR
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    clipped = clip_polygon(quad, rect)
    return quad if clipped is None else clipped


def _build(name: str, x_cam1: float, x_cam2: float, corridor: float, seed: int,
           calib_points: int, noise_px: float, blind: dict | None = None) -> Layout:
    rng = np.random.default_rng(seed)
    cameras = {"C1": _make_camera("C1", x_cam1), "C2": _make_camera("C2", x_cam2)}
    calibrations, zones = {}, {}
    for cam_id, cam in cameras.items():
        pixel_pts, world_pts = cam.calibration_points(calib_points, noise_px, rng, FLOOR)
        calibrations[cam_id] = calibrate(cam_id, pixel_pts, world_pts)
        zones[cam_id] = CameraZone(cam_id, _estimated_fov(cam, calibrations[cam_id]))
    zones["C1"].exit_polygons["C2"] = _corridor(zones["C1"].fov_polygon, "right", corridor)
    zones["C1"].entry_polygons["C2"] = _corridor(zones["C1"].fov_polygon, "right", corridor)
    zones["C2"].entry_polygons["C1"] = _corridor(zones["C2"].fov_polygon, "left", corridor)
    zones["C2"].exit_polygons["C1"] = _corridor(zones["C2"].fov_polygon, "left", corridor)
    overlap = clip_polygon(zones["C1"].fov_polygon, zones["C2"].fov_polygon)
    if overlap is not None:
        drivable = clip_polygon(overlap, _band_rect(LANE_BAND))
        # A sliver at the far corner of two trapezoids is not a usable seam: the
        # site survey only declares an overlap where both cameras see drivable floor.
        overlap = drivable if drivable is not None and polygon_area(drivable) >= 6.0 else None
    edges = {
        ("C1", "C2"): TopologyEdge("C1", "C2", 0.0, 4.0, 0.6, V_MAX),
        ("C2", "C1"): TopologyEdge("C2", "C1", 0.0, 4.0, 0.6, V_MAX),
    }
    overlaps = {} if overlap is None else {("C1", "C2"): overlap}
    exit_lines = {}
    far_right = _corridor(zones["C2"].fov_polygon, "right", 3.0)
    far_left = _corridor(zones["C1"].fov_polygon, "left", 3.0)
    exit_lines["C2"] = far_right          # facility exit ramp
    exit_lines["C1"] = far_left           # facility entrance / exit gate
    topology = CameraTopology(zones=zones, edges=edges, overlaps=overlaps, exit_lines=exit_lines)
    return Layout(name, cameras, calibrations, topology, blind_regions=blind or {})


def gap_layout(seed: int = 11, calib_points: int = 12, noise_px: float = 0.6) -> Layout:
    """Disjoint fields of view with a structural blind gap between them."""
    return _build("gap", 23.0, 75.0, 6.0, seed, calib_points, noise_px)


def overlap_layout(seed: int = 12, calib_points: int = 14, noise_px: float = 0.6) -> Layout:
    """Fields of view sharing a seam — the fusion / seam-measurement layout."""
    return _build("overlap", 23.0, 60.0, 6.0, seed, calib_points, noise_px)


def parking_layout(seed: int = 13, calib_points: int = 12, noise_px: float = 0.6,
                   blind_band=(29.0, 37.0)) -> Layout:
    """Gap layout plus an overhead structure that hides part of the drive lane."""
    x0, x1 = blind_band
    blind = {"C1": [np.array([[x0, 9.0], [x1, 9.0], [x1, 27.0], [x0, 27.0]])]}
    return _build("parking", 23.0, 75.0, 6.0, seed, calib_points, noise_px, blind)


def cruise(vehicle_id: str, x_from: float, x_to: float, t0: float = 0.0,
           speed: float = CRUISE_SPEED, y: float = LANE_Y, **kw) -> SimVehicle:
    from .vehicles import Waypoint
    duration = abs(x_to - x_from) / speed
    return SimVehicle(vehicle_id, [Waypoint(t0, x_from, y), Waypoint(t0 + duration, x_to, y)], **kw)


def build_profiles(layout: Layout, vehicle_dimensions=(4.6, 1.9), vehicle_height: float = 1.5
                   ) -> dict[str, CameraProfile]:
    """The commissioning survey: measure per-camera vision parameters.

    ``ground_direction`` comes from a plumb-line marker (base and top of a
    vertical rod of vehicle height), ``expected_vehicle_area`` from one reference
    vehicle parked at the centre of the field of view.
    """
    profiles: dict[str, CameraProfile] = {}
    for cam_id, cam in layout.cameras.items():
        fov = layout.fov(cam_id)
        centre = fov.mean(axis=0)
        directions = []
        for probe in (centre, centre + np.array([4.0, 0.0]), centre - np.array([4.0, 0.0])):
            base = cam.project_floor(probe)[0]
            top = cam.project_3d(np.array([[probe[0], probe[1], vehicle_height]]))[0]
            directions.append(base - top)
        direction = np.mean(directions, axis=0)
        length, width = vehicle_dimensions
        footprint = np.array([[centre[0] - length / 2, centre[1] - width / 2],
                              [centre[0] + length / 2, centre[1] - width / 2],
                              [centre[0] + length / 2, centre[1] + width / 2],
                              [centre[0] - length / 2, centre[1] + width / 2]])
        corners = np.vstack([np.column_stack([footprint, np.zeros(4)]),
                             np.column_stack([footprint, np.full(4, vehicle_height)])])
        silhouette = polygon_area(convex_hull(cam.project_3d(corners)))
        profiles[cam_id] = CameraProfile(
            camera_id=cam_id, calibration=layout.calibrations[cam_id],
            width=cam.width, height=cam.height, ground_direction=direction,
            camera_ground_point=np.array([cam.position[0], cam.position[1]]),
            vehicle_dimensions=vehicle_dimensions, vehicle_height=vehicle_height,
            expected_vehicle_area=float(silhouette * 0.85),
            parallax_gain=float(vehicle_height / cam.position[2]))
    return profiles
