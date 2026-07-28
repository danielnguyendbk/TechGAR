# Skill: smart-parking-routing

Use this skill to construct and validate driving and walking routes.

## Read first
- `AGENTS.md`
- `docs/LAYOUT_SPEC.md`
- `docs/ROUTING_SPEC.md`

## Responsibilities
- Build a typed lane graph matching the SVG geometry.
- Use A* or Dijkstra.
- Respect one-way edges.
- Connect entrance to every selectable spot-entry node.
- Render route from graph edges only.
- Never cross parking spots, islands, walls, or medians.
- Pause route and surface warning when selected spot is no longer empty.
- Add pure-function tests for route validity.
