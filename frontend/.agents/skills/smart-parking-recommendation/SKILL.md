# Skill: smart-parking-recommendation

Use this skill for optional driver recommendations.

## Read first
- `AGENTS.md`
- `docs/RECOMMENDATION_SPEC.md`
- `docs/PRODUCT_REQUIREMENTS.md`

## Responsibilities
- Support exactly three high-level needs: Shopping, Dịch vụ, Giải trí.
- Do not create sub-destination selection.
- Rank only stable empty spots.
- Return best plus two alternatives.
- Keep scoring deterministic and explainable.
- Require explicit user confirmation before navigation.
- Recalculate unconfirmed results when candidate state changes.
- Do not silently redirect after confirmed spot becomes invalid.

## UI copy
Use the exact labels and disclaimer in `PRODUCT_REQUIREMENTS.md`.
