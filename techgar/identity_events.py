"""Append-only identity audit log (PLAN 1 stage 8 logic 6, rubric A & C).

There is no update or delete API.  Every identity transition, every blocked mint,
every quarantine and every session action lands here with a timestamp and a frame
sequence number, so any identity decision can be replayed after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .states import IdentityEvent, IdentityEventType


@dataclass
class IdentityEventLog:
    _events: list[IdentityEvent] = field(default_factory=list)
    _next_id: int = 0

    def append(self, timestamp: float, frame_sequence: int, event_type: IdentityEventType,
               global_id: int | None, detail: str = "", camera_id: str = "",
               evidence: dict | None = None) -> IdentityEvent:
        if frame_sequence is None:
            # The dataclass would catch this, but the int() coercion below
            # crashes first with an opaque TypeError — fail with the contract
            # name instead (rubric A: every event carries frame + timestamp).
            from .contracts import ContractViolation
            raise ContractViolation("identity event without a frame sequence")
        self._next_id += 1
        event = IdentityEvent(event_id=self._next_id, timestamp=float(timestamp),
                              frame_sequence=int(frame_sequence), event_type=event_type,
                              global_id=global_id, detail=detail, camera_id=camera_id,
                              evidence=dict(evidence or {}))
        self._events.append(event)
        return event

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(tuple(self._events))

    @property
    def events(self) -> tuple[IdentityEvent, ...]:
        return tuple(self._events)

    def for_global_id(self, global_id: int) -> list[IdentityEvent]:
        return [e for e in self._events if e.global_id == global_id]

    def of_type(self, *event_types: IdentityEventType) -> list[IdentityEvent]:
        wanted = set(event_types)
        return [e for e in self._events if e.event_type in wanted]

    def since(self, timestamp: float) -> list[IdentityEvent]:
        return [e for e in self._events if e.timestamp >= timestamp]

    def tail(self, count: int = 20) -> list[IdentityEvent]:
        return list(self._events[-count:])

    def to_json(self, limit: int | None = None) -> str:
        events = self._events if limit is None else self._events[-limit:]
        return json.dumps([{
            "event_id": e.event_id, "timestamp": e.timestamp,
            "frame_sequence": e.frame_sequence, "type": e.event_type.value,
            "global_id": e.global_id, "camera_id": e.camera_id, "detail": e.detail,
            "evidence": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                         for k, v in e.evidence.items()},
        } for e in events], indent=1)

    def count_of(self, event_type: IdentityEventType) -> int:
        return sum(1 for e in self._events if e.event_type == event_type)
