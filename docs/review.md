# The science review

This page describes how a scored campaign becomes a list of defects, how
each defect names the file a revision edits, and how a revision is
validated against the benchmark before release.

The benchmark scorer says whether a campaign passed and why it did not.
That is an outcome, not a defect. "a0 outside the band" names nothing a
revision can edit. The review closes that gap. Two evaluators read the
same evidence and raise flags. A flag is one attributable defect. The
flags travel in the campaign record beside `passed` and `reason`, so the
records file is also the defect list.

## Flags

A flag has five fields.

| Field | Content |
| --- | --- |
| `rule` | What went wrong, as a short kebab-case name (`script-not-used`). |
| `target` | What a revision edits to prevent it. One of the targets below. |
| `evidence` | Where in the campaign: a step number, a run id, or `finish`. |
| `note` | One sentence a maintainer can act on. |
| `raised_by` | `rules` or `referee`. |

A target names a file, or a part of one.

| Target | What a revision edits |
| --- | --- |
| `skill:<name>` | The skill's description, its body, or a bundled script. |
| `card:<agent>` | The entry card's role prompt. |
| `tool:<name>` | A tool's description or schema. |
| `prompt` | The system prompt outside any card. |

A flag on a skill is raised against one revision of that skill. See
"Skill revisions" below.

## The evaluators

### The rules

The rules are code. They run on every scored campaign, and they cost
nothing. Each rule blames one target.

| Rule | Fires when | Target |
| --- | --- | --- |
| `skill-not-loaded` | A skill the question lists never loaded. The description did not trigger. | `skill:<name>` |
| `script-not-used` | A loaded skill bundles scripts, and no shell call after the load named one. | `skill:<name>` |
| `script-failed` | A shell call that ran a bundled script failed. | `skill:<name>` |
| `run-not-verified` | A run the session made never reached `verified`. | The skill in force when the run started, else `card:<agent>` |
| `unit-mismatch` | A reported unit is not the question's unit. | `tool:finish` |
| `finish-incomplete` | No finish, no structured results, or no run ids. | `card:<agent>` |
| `no-progress-loop` | Fifteen or more consecutive steps only looked: shell, reads, listings, with no run launched, no plan change, no note, no brief, and no finish. | `card:<agent>` |
| `reasoning-heavy` | A step was billed 8,000 or more completion tokens and wrote no plan, file, note, or report. The note names the recorded effort. | `prompt` |

The skill in force is the skill loaded most recently before the run
started. The rule reads the load time from the transcript and the start
time from the run record.

### The referee

The referee is a model. It reads an evidence pack and argues with the
procedure. Ask for it with `--referee` on `score` or `run`:

```bash
slab benchmark score --referee
```

The referee uses the `[agent]` model and endpoint, so on a cluster it
reaches the served model the campaign used. `--referee-model`,
`--referee-endpoint`, and `--referee-provider` override that. One model
call is made per campaign.

The evidence pack contains:

- the instruction and the scorer's verdict;
- the targets the referee may blame, as a list;
- the body of every skill the question lists or the campaign loaded, so
  the procedure is judged against what the agent was told;
- one line per tool call, with its arguments and its result, truncated;
- every run the session made, with its tasks, engines, and checks;
- the finish: the results, the run ids, and the full report.

The rubric asks for defects in the procedure, not in the number:
convergence declared or assumed, structure relaxed at the engine that
produced the number, the scan bracketing its minimum, slab thickness and
vacuum, supercell size, equilibration and sampling, a held-out set that
was held out, units and conversions stated, uncertainty estimated, the
bundled script used, every number tied to a verified run.

The referee answers with JSON. A target outside the allowed list is
re-attributed to the card, and the note keeps what the referee named. A
reply the parser cannot read leaves the rules' flags in place and records
the failure under `referee_error`, so a dark endpoint never blanks a
review.

## Skill revisions

Every skill has a digest: a short hash of every file under its root. One
changed byte in `SKILL.md` or in a script is a new revision. The `skill`
tool records the digest when the skill loads, and the campaign record
carries it under `skills`. A flag on a skill is therefore raised against
one revision, and a later campaign under a different digest is evidence
about a different skill.

`slab benchmark flags` lists the flags on the latest record per model,
machine, and question, with a status:

| Status | Meaning |
| --- | --- |
| `open` | The target is unchanged since the flag was raised. |
| `pending` | The skill has a newer revision, and no campaign has run under it. |
| `unknown` | The skill is not in the catalog, or the record carried no digest. |

```bash
slab benchmark flags --status open
```

## The gate

`slab benchmark gate <skill>` decides whether the benchmark validates the
catalog's revision of a skill. For every model, machine, and question
that exercised the skill, it compares the newest record under the current
digest with the newest record under any earlier one.

| Verdict | Meaning |
| --- | --- |
| `validated` | A campaign ran under this revision, it passes if the earlier one passed, and it raises no flag against the skill. |
| `not validated` | No scored campaign loaded this revision. |
| `regressed` | The cell passed under an earlier revision and fails now. |
| `still flagged` | A campaign under this revision still raises a flag against the skill. |

The command exits 1 unless every cell is validated. No cell at all is not
validation either: a skill nobody exercised has no evidence.

Recorded against the two laptop campaigns on the benchmark page, which
were scored before the `skill` tool recorded digests:

```bash
slab benchmark gate equation-of-state
```

```text
equation-of-state revision 0f2314b350ba
  Q1 a0        llama3.1:8b              laptop       not validated: no scored campaign loaded revision 0f2314b350ba
  Q1 a0        llama3.1:8b-32k          laptop       not validated: no scored campaign loaded revision 0f2314b350ba
not validated
```

Both cells refuse, and the exit code is 1. That is the truthful state:
no campaign has run under this revision of the skill.

Do not merge a skill revision that the gate refuses.

## The loop

1. Score the campaigns. The rules run on each one. Add `--referee` for
   the model's review.

    ```bash
    slab benchmark score --machine <label> --referee
    ```

2. Read the defect list, and pick a target.

    ```bash
    slab benchmark flags --status open
    ```

3. Revise the target. For a skill, edit its description, its body, or
   its script, and run the skill's script test.

4. Run the campaigns for the questions that list the skill, under the new
   revision, and score them.

5. Check the gate before release.

    ```bash
    slab benchmark gate equation-of-state
    ```

6. Rewrite the tables, and commit the records file with the revision.

    ```bash
    slab benchmark tables
    ```

The results and flags tables on [the benchmark page](benchmark.md) are
rendered from the records, so the defect list cannot drift from the
evidence.
