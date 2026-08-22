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

## Package layering

The repository ships one distribution, `slab-stack`, containing three
import packages. Each has a console script of the same name.

| Package | Owns | May import |
|---|---|---|
| `slab` | Access to computational software: engines and calculators, the engine registry, QE protocols, pseudopotential families, the SLURM layer, the config loader | nothing else here |
| `foundation` | Workflows and state: runs, lifecycle, the artifact store, retention, tracing and caching, verification, the ready-made tasks, the MCP server | `slab` |
| `mason` | The resident research agent: LLM clients, the ReAct loop, tools, session, prompts, model serving | `foundation`, `slab` |

The dependency direction is a hard rule, and `tests/test_layering.py`
enforces it by reading the AST. That catches imports which only execute on
some paths, inside a function, under `if TYPE_CHECKING:`, or in a
`try`/`except ImportError` fallback.

Put new code in Foundation only if it sits between the calculator and the
agent: state, provenance, workflows, or a surface that exposes them.
Anything specific to one agent implementation belongs in `mason`. Anything
that talks to a computational code, a registry, or the scheduler belongs in
`slab`.

Configuration is one `slab.toml` with per-table owners. `[paths]`,
`[engines]`, and `[hpc]` belong to `slab`; `[workspace]` to `foundation`;
`[agent]` and `[agent.serve]` to `mason`. The loader lives in
`slab.config`, and each package validates only its own tables. Add a new
top-level table to `KNOWN_TOP_LEVEL_KEYS` in the same change that adds its
model, or the loader refuses the table.

On-disk names stay under the SLAB umbrella and do not follow the package
split: `slab.toml`, `$SLAB_*`, `.slab/`, and `~/.config/slab/`.
