# SLAB — project notes

## Documentation style: ASD-STE100 as guiding principles

Prose in `README.md` and `docs/` follows the guiding principles of ASD-STE100
(Simplified Technical English). Apply them as a writing style, not as the
controlled vocabulary:

- Prefer the active voice. Name the actor: SLAB, the CLI, `relax`, you.
- Write instructions in the imperative ("Set `command` in `slab.toml`").
- Keep sentences short. One instruction or one idea per sentence, roughly
  25 words or fewer. But keep a natural cadence: join closely related short
  sentences with "and", "so", or "because" rather than leaving a staccato
  run of fragments.
- Do not attach a parenthetical clause with an em dash or a colon. Either
  cut it, or give it its own sentence. A colon may introduce a list or a
  code block.
- Keep one topic per paragraph, and keep paragraphs short.
- Use one name per concept, and always the same name (engine, registry,
  workspace, run, artifact, cache identity).
- State facts literally. Do not use idioms or metaphor in procedural text.
  Keep the opening description of a page plain and undramatic.
- Use lists and tables for sequences, conditions, and enumerations.
- Put a warning before the step it guards, in command form.
- Keep the articles ("the", "a"). Do not stack more than three nouns.

Exception: theoretical and research material may break these rules where
the argument needs it. That covers `ARCHITECTURE.md` and the
design-provenance parts of `docs/tutorials/mason.md`. Strict enforcement is
not the goal; clarity is.

Fenced code blocks and recorded outputs in the docs are exact captures from
real executions. Do not edit them for style.
