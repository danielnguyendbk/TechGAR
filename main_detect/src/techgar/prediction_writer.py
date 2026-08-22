"""Build the experiment-only ``predictions.jsonl`` schema version 3.

The live JSON files and the OpenCV UI intentionally do not use this module.
It flattens tracker/binder state for offline evaluation and converts the
rolling event histories exposed by the runtime into true per-frame deltas.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional


PREDICTION_SCHEMA_VERSION = 3


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-safe representation for event hashing."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_value(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _event_fingerprint(event: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_value(event),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


class _RollingEventDelta:
    """Extract appended entries from a bounded rolling event history.

    Comparing suffix/prefix overlap is robust when a deque drops old entries.
    It also preserves two genuinely distinct but byte-identical consecutive
    events, unlike a global content-hash set.
    """

    def __init__(self) -> None:
        self._previous: tuple[tuple[int, str], ...] = ()
        # Retaining the objects prevents CPython from recycling an id while it
        # still participates in the next rolling-window overlap comparison.
        self._previous_events: tuple[Mapping[str, Any], ...] = ()

    def take(self, events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        current_events = list(events)
        current = tuple(
            (id(event), _event_fingerprint(event)) for event in current_events
        )
        max_overlap = min(len(self._previous), len(current))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if self._previous[-size:] == current[:size]:
                overlap = size
                break
        self._previous = current
        self._previous_events = tuple(current_events)
        return current_events[overlap:]


class PredictionV3Builder:
    """Stateful builder for one schema-v3 JSONL stream."""

    def __init__(self) -> None:
        self._manager_events = _RollingEventDelta()
        self._parking_events: dict[str, _RollingEventDelta] = {}
        self._next_event_serial = 1

    @staticmethod
    def _canonical_gid(
        value: Any,
        canonicalize: Callable[[int], int],
    ) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(canonicalize(int(value)))
        except (TypeError, ValueError, KeyError):
            return None

    def _event_uid(self, category: str) -> str:
        uid = f"{category}-{self._next_event_serial:09d}"
        self._next_event_serial += 1
        return uid

    def _normalise_event(
        self,
        event: Mapping[str, Any],
        *,
        category: str,
        source: str,
        fallback_frame_idx: int,
        canonicalize: Callable[[int], int],
        camera_id: Optional[str] = None,
        fallback_raw_gid: Any = None,
    ) -> dict[str, Any]:
        event_type = str(event.get("type") or event.get("event_type") or "unknown")
        raw_gid = event.get(
            "global_id",
            event.get("vehicle_id", fallback_raw_gid),
        )
        canonical_gid = self._canonical_gid(raw_gid, canonicalize)
        frame_idx = event.get("frame", event.get("frame_idx", fallback_frame_idx))
        try:
            frame_idx = int(frame_idx)
        except (TypeError, ValueError):
            frame_idx = int(fallback_frame_idx)
        reserved = {
            "type",
            "event_type",
            "frame",
            "frame_idx",
            "global_id",
            "vehicle_id",
        }
        payload = {
            "event_uid": self._event_uid(category),
            "source": source,
            "event_type": event_type,
            "frame_idx": frame_idx,
            "canonical_gid": canonical_gid,
            "raw_gid": int(raw_gid) if raw_gid is not None else None,
            "details": {
                str(key): _json_value(value)
                for key, value in event.items()
                if key not in reserved
            },
        }
        if camera_id is not None:
            payload["camera_id"] = str(camera_id)
        return payload

    @staticmethod
    def _aliases(
        registry: Mapping[str, Any],
        canonicalize: Callable[[int], int],
    ) -> tuple[list[dict[str, int]], dict[int, list[int]]]:
        rows = []
        by_canonical: dict[int, set[int]] = {}
        for alias, target in registry.get("retired_global_ids", {}).items():
            try:
                alias_gid = int(alias)
                canonical_gid = int(canonicalize(int(target)))
            except (TypeError, ValueError, KeyError):
                continue
            rows.append({
                "alias_gid": alias_gid,
                "canonical_gid": canonical_gid,
            })
            by_canonical.setdefault(canonical_gid, {canonical_gid}).add(alias_gid)
        rows.sort(key=lambda item: (item["canonical_gid"], item["alias_gid"]))
        return rows, {
            canonical_gid: sorted(values)
            for canonical_gid, values in by_canonical.items()
        }

    @staticmethod
    def _world_observations(registry: Mapping[str, Any]) -> dict[tuple[str, int], dict]:
        lookup = {}
        for identity in registry.get("active_global_vehicles", {}).values():
            for observation in identity.get("observations", []):
                try:
                    key = (
                        str(observation["camera_id"]),
                        int(observation["local_track_id"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                lookup[key] = observation
        return lookup

    def _slots(
        self,
        binders: Mapping[str, Any],
        canonicalize: Callable[[int], int],
    ) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        rows = []
        ownerships: dict[int, list[dict[str, Any]]] = {}
        for camera_id, binder in sorted(binders.items()):
            for slot_id, slot in sorted(binder.to_json(camera_id=camera_id).items()):
                raw_gid = slot.get("vehicle_id")
                canonical_gid = self._canonical_gid(raw_gid, canonicalize)
                recovery_gid = self._canonical_gid(
                    slot.get("recovery_global_id"), canonicalize
                )
                row = {
                    "camera_id": str(camera_id),
                    "slot_id": str(slot_id),
                    "occupied": bool(slot.get("occupied", False)),
                    "raw_vehicle_gid": int(raw_gid) if raw_gid is not None else None,
                    "canonical_vehicle_gid": canonical_gid,
                    "vision_occupied": bool(
                        slot.get("vision_occupied", slot.get("raw_occupied", False))
                    ),
                    "tracking_occupied": bool(slot.get("tracking_occupied", False)),
                    "decision_source": str(slot.get("decision_source", "unknown")),
                    "tracking_state": str(slot.get("tracking_state", "unknown")),
                    "vehicle_overlap": float(slot.get("vehicle_overlap", 0.0)),
                    "stopped_for_ms": int(slot.get("stopped_for_ms", 0)),
                    "recovery_state": str(slot.get("recovery_state", "none")),
                    "recovery_global_id": recovery_gid,
                    "recovery_age_ms": int(slot.get("recovery_age_ms", 0)),
                    "recovery_radius_px": float(slot.get("recovery_radius_px", 0.0)),
                    "recovery_candidate_count": int(
                        slot.get("recovery_candidate_count", 0)
                    ),
                }
                rows.append(row)
                if canonical_gid is not None:
                    ownerships.setdefault(canonical_gid, []).append({
                        "camera_id": str(camera_id),
                        "slot_id": str(slot_id),
                        "state": "occupied" if row["occupied"] else "reserved",
                    })
        return rows, ownerships

    def _observations(
        self,
        *,
        frame_idx: int,
        trackers: Mapping[str, Any],
        global_ids: Mapping[str, Mapping[int, int]],
        registry: Mapping[str, Any],
        ownerships: Mapping[int, Sequence[Mapping[str, Any]]],
        aliases_by_canonical: Mapping[int, Sequence[int]],
        canonicalize: Callable[[int], int],
    ) -> list[dict[str, Any]]:
        rows = []
        world_lookup = self._world_observations(registry)
        lifecycle = registry.get("identity_lifecycle", {})
        world_unit = str(registry.get("world_unit", "unknown"))
        for camera_id, tracker in sorted(trackers.items()):
            telemetry = tracker.local_track_telemetry(global_ids.get(camera_id, {}))
            for track in telemetry:
                local_id = int(track["local_track_id"])
                raw_gid = track.get("global_id")
                canonical_gid = self._canonical_gid(raw_gid, canonicalize)
                world_observation = world_lookup.get((str(camera_id), local_id), {})
                global_position = world_observation.get("global_position")
                shared_anchor = world_observation.get("shared_map_anchor", {})
                center = track.get("center") or [None, None]
                identity = (
                    lifecycle.get(str(canonical_gid), {})
                    if canonical_gid is not None
                    else {}
                )
                gid_aliases = []
                if canonical_gid is not None:
                    gid_aliases = list(
                        aliases_by_canonical.get(canonical_gid, [canonical_gid])
                    )
                    if canonical_gid not in gid_aliases:
                        gid_aliases.append(canonical_gid)
                        gid_aliases.sort()
                owners = list(ownerships.get(canonical_gid, ()))
                rows.append({
                    "observation_uid": (
                        f"frame-{int(frame_idx):09d}:{camera_id}:{local_id}"
                    ),
                    "camera_id": str(camera_id),
                    "local_track_id": local_id,
                    "raw_gid": int(raw_gid) if raw_gid is not None else None,
                    "canonical_gid": canonical_gid,
                    "gid_aliases": gid_aliases,
                    "bbox": [int(value) for value in track.get("bbox", [])],
                    "anchor_pixel": {
                        "x": int(center[0]) if center[0] is not None else None,
                        "y": int(center[1]) if center[1] is not None else None,
                        "reference": "tracker_center",
                    },
                    "anchor_world": (
                        {
                            "x": float(global_position["x"]),
                            "y": float(global_position["y"]),
                            "unit": world_unit,
                            "reference": str(
                                shared_anchor.get("reference", "tracker_point")
                            ),
                        }
                        if global_position is not None
                        else None
                    ),
                    "track_state": str(track.get("state", "unknown")),
                    "association_state": str(
                        track.get("association_state", "unknown")
                    ),
                    "invisible_count": int(track.get("invisible_count", 0)),
                    "assignment_cost": _json_value(
                        track.get("assignment_cost", {})
                    ),
                    "fragment_visible_count": int(
                        track.get("fragment_visible_count", 0)
                    ),
                    "first_observation_frame": int(
                        track.get("first_observation_frame", frame_idx)
                    ),
                    "identity_state": (
                        str(identity.get("state", "unknown"))
                        if canonical_gid is not None
                        else "unassigned"
                    ),
                    "slot_ownership": dict(owners[0]) if owners else None,
                })
        return rows

    def _identity_events(
        self,
        *,
        frame_idx: int,
        trackers: Mapping[str, Any],
        global_ids: Mapping[str, Mapping[int, int]],
        registry: Mapping[str, Any],
        canonicalize: Callable[[int], int],
    ) -> list[dict[str, Any]]:
        rows = [
            self._normalise_event(
                event,
                category="identity",
                source="global_manager",
                fallback_frame_idx=frame_idx,
                canonicalize=canonicalize,
            )
            for event in self._manager_events.take(
                registry.get("recent_events", [])
            )
        ]
        for camera_id, tracker in sorted(trackers.items()):
            camera_global_ids = global_ids.get(camera_id, {})
            for event in tracker.association_events:
                local_id = event.get("local_track_id")
                fallback_gid = None
                if local_id is not None:
                    try:
                        fallback_gid = camera_global_ids.get(int(local_id))
                    except (TypeError, ValueError):
                        pass
                rows.append(self._normalise_event(
                    event,
                    category="identity",
                    source=f"motion_tracker:{camera_id}",
                    fallback_frame_idx=frame_idx,
                    canonicalize=canonicalize,
                    camera_id=str(camera_id),
                    fallback_raw_gid=fallback_gid,
                ))
        return rows

    def _parking_event_deltas(
        self,
        *,
        frame_idx: int,
        binders: Mapping[str, Any],
        canonicalize: Callable[[int], int],
    ) -> list[dict[str, Any]]:
        rows = []
        for camera_id, binder in sorted(binders.items()):
            stream = self._parking_events.setdefault(
                str(camera_id), _RollingEventDelta()
            )
            for event in stream.take(binder.events):
                rows.append(self._normalise_event(
                    event,
                    category="parking",
                    source=f"slot_binder:{camera_id}",
                    fallback_frame_idx=frame_idx,
                    canonicalize=canonicalize,
                    camera_id=str(camera_id),
                ))
        return rows

    def build_frame(
        self,
        *,
        frame_idx: int,
        capture_unix_ns: int,
        wall_time_iso: str,
        camera_timestamps_ns: Mapping[str, int],
        camera_skew_ms: float,
        trackers: Mapping[str, Any],
        global_ids: Mapping[str, Mapping[int, int]],
        binders: Mapping[str, Any],
        manager: Any,
        registry: Mapping[str, Any],
        parking_recovery: Sequence[Mapping[str, Any]],
        parked_identity_reservations: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build one JSON-safe frame without changing live runtime state."""
        canonicalize = getattr(manager, "canonical_global_id", int)
        gid_aliases, aliases_by_canonical = self._aliases(
            registry, canonicalize
        )
        slots, ownerships = self._slots(binders, canonicalize)
        observations = self._observations(
            frame_idx=frame_idx,
            trackers=trackers,
            global_ids=global_ids,
            registry=registry,
            ownerships=ownerships,
            aliases_by_canonical=aliases_by_canonical,
            canonicalize=canonicalize,
        )
        reservations = []
        for raw_gid, reservation in sorted(
            parked_identity_reservations.items(), key=lambda item: int(item[0])
        ):
            canonical_gid = self._canonical_gid(raw_gid, canonicalize)
            reservations.append({
                "canonical_gid": canonical_gid,
                "camera_id": reservation.get("camera_id"),
                "slot_id": reservation.get("slot_id"),
                "state": str(reservation.get("state", "parked")),
                "bbox": (
                    [int(value) for value in reservation["bbox"]]
                    if reservation.get("bbox") is not None
                    else None
                ),
            })
        return {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "frame_idx": int(frame_idx),
            "capture_unix_ns": int(capture_unix_ns),
            "wall_time_iso": str(wall_time_iso),
            "camera_timestamps_ns": {
                str(key): int(value)
                for key, value in camera_timestamps_ns.items()
            },
            "camera_skew_ms": round(float(camera_skew_ms), 3),
            "observations": observations,
            "slots": slots,
            "gid_aliases": gid_aliases,
            "identity_events": self._identity_events(
                frame_idx=frame_idx,
                trackers=trackers,
                global_ids=global_ids,
                registry=registry,
                canonicalize=canonicalize,
            ),
            "parking_events": self._parking_event_deltas(
                frame_idx=frame_idx,
                binders=binders,
                canonicalize=canonicalize,
            ),
            "parking_recovery": _json_value(parking_recovery),
            "parked_identity_reservations": reservations,
        }
