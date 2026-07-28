# How to use this Codex kit

1. Copy the complete contents of this folder into the root of the Smart Parking frontend repository.
2. Keep the `.agents` folder hidden files included.
3. Start Codex from the repository root:

```bash
codex
```

4. Check skills:

```text
/skills
```

Expected skills:
- `$smart-parking-ui`
- `$smart-parking-mock`
- `$smart-parking-recommendation`
- `$smart-parking-routing`
- `$smart-parking-qa`

5. Paste the complete content of `PROMPT.md` into Codex.
6. Let Codex inspect, plan, implement, test, and capture screenshots.

## Recommended review sequence
- Review `docs/IMPLEMENTATION_PLAN.md` first.
- Check desktop and mobile screenshots.
- Run the app and use the development mock panel.
- Verify the two camera streams update separate spot ownership sets.
- Verify a yellow spot cannot be selected or recommended.
