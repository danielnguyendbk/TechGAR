# Skill: smart-parking-mock

Use this skill to implement deterministic frontend-only data from two virtual cameras.

## Read first
- `AGENTS.md`
- `docs/MOCK_DATA_SPEC.md`

## Responsibilities
- Define strict domain types.
- Implement `ParkingDataSource`.
- Generate complete initial snapshot for 160 spots.
- Implement independent cam-left and cam-right event producers.
- Enforce ownership authorization.
- Ignore stale revisions.
- Preserve last-known states during camera offline.
- Expose development-only controls for stepping scenarios.
- Keep all scenarios deterministic and testable.

## Prohibitions
- No `Math.random()` without a fixed seed abstraction.
- No direct mock-sequence imports in UI components.
- No backend folders or network server.
