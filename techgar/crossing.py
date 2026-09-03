"""Gate crossing detection — the only place a vehicle earns the right to be minted.

Implements the plan's invariant: "Track mới chỉ được cấp GID sau khi cắt entry_gate
đúng hướng" and "Detection xuất hiện giữa bãi chỉ mang trạng thái UNKNOWN".

A crossing is a segment (prev_position → curr_position) that intersects a gate
line/polygon in the valid direction.  Anti-duplicate protection uses an event key
so the same physical crossing is never counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import point_in_polygon


@dataclass(frozen=True)
class CrossingEvent:
    """A confirmed gate crossing."""

    gate_id: str
    direction: str          # "entry" or "exit"
    camera_id: str
    position: np.ndarray
    timestamp: float
    event_key: str          # anti-duplicate key


@dataclass
class GateDefinition:
    """A gate polygon with a valid crossing direction (inward normal).

    ``inward_direction`` is a unit vector pointing *into* the facility.
    A vehicle crossing the gate polygon is valid when
    ``dot(movement_vector, inward_direction) > 0`` (entering)
    or ``dot(movement_vector, inward_direction) < 0`` (exiting).
    """

    gate_id: str
    polygon: np.ndarray                # (N, 2) world-coordinate polygon
    inward_direction: np.ndarray       # (2,) unit vector pointing into facility
    gate_type: str = "entry"           # "entry" or "exit"
    camera_id: str = ""                # which camera observes this gate

    def crosses(self, prev_pos, curr_pos, min_displacement: float = 0.01) -> bool:
        """Check if movement from prev_pos to curr_pos crosses this gate polygon."""
        prev_inside = point_in_polygon(prev_pos, self.polygon)
        curr_inside = point_in_polygon(curr_pos, self.polygon)
        # A crossing occurs when one point is inside the gate zone and the other is not,
        # OR both are inside (traversing through the gate zone).
        if prev_inside == curr_inside and not curr_inside:
            return False
        displacement = float(np.linalg.norm(
            np.asarray(curr_pos, dtype=float) - np.asarray(prev_pos, dtype=float)))
        return displacement >= min_displacement

    def valid_direction(self, prev_pos, curr_pos) -> bool:
        """Check if the movement direction is consistent with this gate's expected direction."""
        movement = np.asarray(curr_pos, dtype=float) - np.asarray(prev_pos, dtype=float)
        norm = float(np.linalg.norm(movement))
        if norm < 1e-9:
            return False
        movement = movement / norm
        dot = float(movement @ np.asarray(self.inward_direction, dtype=float))
        if self.gate_type == "entry":
            return dot > 0.0      # moving inward
        else:
            return dot < 0.0      # moving outward


@dataclass
class CrossingDetector:
    """Tracks gate crossings and prevents duplicate events.

    The detector maintains a set of recent event keys to prevent the same
    physical crossing from being reported multiple times.
    """

    gates: list[GateDefinition] = field(default_factory=list)
    _seen_keys: set[str] = field(default_factory=set)
    _key_timestamps: dict[str, float] = field(default_factory=dict)
    key_ttl: float = 10.0          # seconds before an event key expires

    def add_gate(self, gate: GateDefinition) -> None:
        self.gates.append(gate)

    def check(self, prev_pos, curr_pos, camera_id: str,
              timestamp: float, track_id: int = 0,
              gate_type: str | None = None) -> CrossingEvent | None:
        """Check whether the movement crosses any gate.

        Returns the first valid crossing event, or None.
        ``gate_type`` filters to only check "entry" or "exit" gates.
        """
        self._expire_keys(timestamp)
        for gate in self.gates:
            if gate_type is not None and gate.gate_type != gate_type:
                continue
            if gate.camera_id and gate.camera_id != camera_id:
                continue
            if not gate.crosses(prev_pos, curr_pos):
                continue
            if not gate.valid_direction(prev_pos, curr_pos):
                continue
            event_key = f"{gate.gate_id}:{camera_id}:{track_id}:{int(timestamp * 10)}"
            if event_key in self._seen_keys:
                continue
            self._seen_keys.add(event_key)
            self._key_timestamps[event_key] = timestamp
            return CrossingEvent(
                gate_id=gate.gate_id,
                direction=gate.gate_type,
                camera_id=camera_id,
                position=np.asarray(curr_pos, dtype=float),
                timestamp=timestamp,
                event_key=event_key,
            )
        return None

    def has_entry_crossing(self, prev_pos, curr_pos, camera_id: str,
                           timestamp: float, track_id: int = 0) -> CrossingEvent | None:
        """Convenience: check only entry gates."""
        return self.check(prev_pos, curr_pos, camera_id, timestamp, track_id, "entry")

    def has_exit_crossing(self, prev_pos, curr_pos, camera_id: str,
                          timestamp: float, track_id: int = 0) -> CrossingEvent | None:
        """Convenience: check only exit gates."""
        return self.check(prev_pos, curr_pos, camera_id, timestamp, track_id, "exit")

    def _expire_keys(self, now: float) -> None:
        expired = [k for k, t in self._key_timestamps.items() if now - t > self.key_ttl]
        for k in expired:
            self._seen_keys.discard(k)
            del self._key_timestamps[k]
