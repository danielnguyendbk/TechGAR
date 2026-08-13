# Test Plan

## Unit tests
- generate exactly 160 unique spots;
- zone order E,D,C,B,A;
- 30 spots per A–E and 10 in F;
- two rows per A–E;
- ownership is disjoint;
- event authorization rejects wrong camera;
- stale revision ignored;
- counts derived correctly;
- recommendation only uses empty spots;
- amber never recommended;
- need changes produce deterministic ranking;
- route edges are valid and directed correctly.

## Component tests
- entry sheet actions change mode correctly;
- `Bỏ qua` opens browse mode;
- browse empty-only filter works;
- non-empty spot disables navigation button;
- recommendation requires explicit confirmation;
- alternatives render;
- disclaimer renders;
- invalid confirmed spot opens warning sheet.

## Playwright flows
1. Entry → browse → select empty spot → route.
2. Entry → recommendation → Shopping → confirm best → route.
3. Entry → recommendation → Dịch vụ → abandon → browse.
4. Entry → recommendation → Giải trí → recommended spot turns amber → recalculation.
5. Confirm spot → spot becomes occupied → warning → switch alternative.
6. Camera goes offline → health state visible, statuses preserved.
7. Mobile pan/zoom and bottom sheets.
8. Desktop visual smoke test.

## Screenshot targets
- `390x844-entry.png`
- `390x844-recommendation.png`
- `390x844-navigation.png`
- `1440x900-desktop.png`
