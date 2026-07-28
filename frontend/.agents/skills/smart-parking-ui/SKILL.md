# Skill: smart-parking-ui

Use this skill when implementing or refining the Smart Parking driver interface.

## Read first
- `AGENTS.md`
- `docs/PRODUCT_REQUIREMENTS.md`
- `docs/LAYOUT_SPEC.md`
- `docs/COMPONENT_SPEC.md`
- approved images under `assets/reference/`

## Responsibilities
- Build the one-page React/TypeScript UI.
- Implement responsive desktop/mobile behavior.
- Render the parking lot from typed SVG geometry.
- Preserve E→D→C→B→A vertical order and right-side F strip.
- Render two opposing rows and one center lane for every A–E zone.
- Implement entry, browse, recommendation, and navigation visual states.
- Use exact Vietnamese copy from requirements.
- Apply status colors correctly.
- Keep recommendation/selection as blue outlines without changing fill.
- Implement accessible interaction and minimum 44×44 touch targets.

## Visual constraints
- Clean white app chrome with navy text and blue primary actions.
- Dark asphalt map area with green landscaping/islands.
- Compact summary cards.
- Mobile uses bottom sheets.
- Do not show QR code.
- Do not add admin controls to production UI.

## Completion checks
- No repeated hardcoded spot JSX.
- No layout drift from source images.
- All modes are reachable without page reload.
