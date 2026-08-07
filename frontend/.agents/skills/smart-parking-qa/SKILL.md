# Skill: smart-parking-qa

Use this skill after implementation and during every major change.

## Read first
- `AGENTS.md`
- `docs/TEST_PLAN.md`

## Responsibilities
- Add unit, component, and Playwright tests.
- Run lint, typecheck, test, Playwright, and build.
- Verify 160 unique spots and exact layout order.
- Verify camera ownership and stale-event behavior.
- Verify amber spots are never recommended/selectable.
- Verify optional recommendation and browse-only flows.
- Verify confirmation is required before routing.
- Verify invalid selected-spot warning behavior.
- Verify route graph validity.
- Capture required screenshots.
- Write `artifacts/VALIDATION.md` with command results and known limitations.

## Do not finish with failing checks
Fix failures before reporting completion unless blocked by an external dependency, and document that block precisely.
