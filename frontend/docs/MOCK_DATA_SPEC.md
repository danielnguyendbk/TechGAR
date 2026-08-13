# Deterministic Two-Camera Mock Specification

## Camera IDs
```ts
export type CameraId = "cam-left" | "cam-right";
```

## Ownership
For every zone A–E:
- cam-left: 01–08, 16–23
- cam-right: 09–15, 24–30

Zone F:
- cam-right: F01–F10

## Event shape
```ts
export type ParkingStatus = "empty" | "occupied" | "transitioning" | "unknown";

export interface SpotStatusEvent {
  type: "spot.status.changed";
  cameraId: CameraId;
  spotId: string;
  status: ParkingStatus;
  confidence: number;
  revision: number;
  updatedAt: string;
}

export interface CameraHealthEvent {
  type: "camera.health.changed";
  cameraId: CameraId;
  health: "online" | "offline";
  updatedAt: string;
}
```

## Snapshot
The initial snapshot contains:
- all 160 spots;
- ownership;
- status;
- confidence;
- revision;
- last updated time;
- both camera health states.

## Determinism
- Use fixed arrays of events.
- Use a deterministic mock clock abstraction.
- No unseeded random data.
- Tests can manually advance events.

## Required scenarios
1. cam-left changes one empty spot to transitioning, then occupied.
2. cam-right changes one occupied spot to transitioning, then empty.
3. an active recommendation becomes transitioning.
4. a confirmed spot becomes occupied.
5. one amber spot returns to empty.
6. cam-left offline then online.
7. cam-right offline then online.
8. stale revision event.
9. camera attempts unauthorized spot update.

## Offline semantics
- Preserve last known spot states.
- Mark camera health degraded.
- Do not convert owned spots to empty.
- Optional: visually mark owned spots with a subtle stale-data indicator, without changing their status color.
