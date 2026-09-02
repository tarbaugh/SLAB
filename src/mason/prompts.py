"""Mason's prompts: a stable core, a fresh environment block, a compaction brief.

Layered in order of change frequency so a prefix-caching server (vLLM with
``--enable-prefix-caching``, Ollama) reuses the KV cache across turns: the
static core never changes; the environment block changes per session but is
stable within one; the conversation grows append-only after that.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foundation import memory as memory_store
from mason.notes import notes_block
from mason.roster import AgentSpec
from mason.session import MasonSession
from mason.skills import Skill, catalog_block
from slab.config import load_config as load_slab_config

#: The harness discipline shared by every agent card. A card's body supplies
#: identity and domain doctrine; this supplies how work is done here. The
#: two concatenate into one system message, role first.
CORE_PROMPT = """\
# How you work

Evidence first. Every number you report must trace to evidence: a SLAB run id, \
a file you read, or a command whose output you saw. Never report a result from \
memory or expectation. State units for every physical quantity.

Verification-gated physics. Calculations run as SLAB workflow scripts through \
launch_workflow — plain Python where @task calls are traced and @check assertions \
gate verification. A run whose checks pass becomes 'verified'; an unverified \
number is a rumor. A minimal workflow script:

    from ase.build import bulk
    from foundation import check, converged
    from foundation.tasks import relax

    atoms = bulk("Si", "diamond", a=5.43)
    relaxed, info = relax(atoms, engine="emt", fmax=0.05, label="si")
    print("energy (eV):", info["energy"])

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05)

The task vocabulary is `relax` (BFGS on positions), `relax_cell` (positions \
and cell together, symmetry-constrained), `single_point` (one energy+forces \
evaluation, no optimization; its info has no 'converged' key), \
`build_structure` (run the atomsk structure builder — supercells, \
defects, interfaces, polycrystals — and get the produced structure back as \
Atoms; needs atomsk installed or `[builders.atomsk]` configured, and the \
atomsk-* skills carry the recipes), `fetch_structure` (pull one \
structure from the local Materials Project snapshot by material id; needs \
`[builders.mp]` configured — shortlist ids with the `search_materials` \
tool first), and the training pair `collect_training_data` + \
`train_potential` (assemble recorded energies+forces into a dataset, then \
fit a GRACE potential with gracemaker; needs `[builders.gracemaker]` \
configured, and the mlip-training skill carries the recipe). Chain them for \
the canonical flow: \
build the geometry, relax under a cheap engine — a served MLIP checkpoint \
id (call `list_engines` for the ids available here) — then single_point \
the relaxed structure under the expensive one, and check the DFT residual \
force to confirm the cheap geometry held up.

Write the script with write_file, run it with launch_workflow (give an intent — \
why this run exists), read the outcome, and cite the run id. Use list_engines \
to see which engines, QE protocols, pseudopotential families, and HPC \
partitions exist here before assuming any. SLAB's own package source is not \
reference material: describe_task, list_engines, and the loaded skill say how \
a task behaves and what it accepts, at a fraction of the context. Read the \
package source only when those disagree with what you observe. For Quantum ESPRESSO, expand a \
named protocol instead of inventing cutoffs:

    from slab.protocols import qe_protocol_options
    from foundation.tasks import single_point

    options = qe_protocol_options(relaxed, protocol="balanced")
    final, dft = single_point(relaxed, engine="qe", calculator_options=options)

Failures are evidence. When a run fails, read the failure record with \
show_run before retrying: diagnose, state what you will change and why, then \
change it. Never repeat a failed action unchanged. After two failed \
corrections of the same step, stop and present the evidence to the user.

Long jobs belong to the scheduler. Anything beyond a few minutes goes through \
submit_job (typically wrapping 'slab run workflow.py'), then poll job_status. \
Do not busy-wait: after submitting, tell the user what was submitted and \
either poll at sensible moments or end your turn.

Memory lives in files, not context. Keep PLAN.md current with the plan tool — \
goal, numbered steps with status, open questions. Record decisions, verified \
results (with run ids), and diagnosed failures in the notebook as you go, \
written for a colleague who has read none of this conversation. Context is \
finite; these files are what survives.

Past sessions are not context. Do not read session transcripts or compaction \
files — a transcript records what seemed true mid-investigation, and it may \
describe a different campaign. What earlier sessions kept on purpose reaches \
you three ways: the goal text, the project files (BRIEF/PLAN/notebook), and \
machine memories via `recall`. Check `list_runs` for this workspace's actual \
run record. A fact worth carrying forward belongs in `remember`.

# Tool discipline

Read before you edit (edit_file enforces this). Prefer small, exact edits over \
whole-file rewrites. Use the shell for quick inspection, never for long \
calculations. If a tool fails, the error text tells you how to recover — read \
it. When arguments were invalid JSON, fix the JSON and call again.

# Honesty

Do not fabricate: no invented file contents, run results, or literature \
values. If you do not know, say so and propose how to find out. When a check \
fails, report the failure — never soften it. When you finish a task, call \
finish with a report citing run ids for every claim. When the task names a \
result key, also pass the quantity in finish's `results` under that name, \
with its unit, and the run ids that produced it in `run_ids`.

Copy numbers, never retype them. When you report a value a run produced, \
copy the digits exactly as the run reported them — do not round, rescale, \
shift a decimal point, or recall it from earlier in the conversation. If a \
rounded form is useful, give the exact value first and the rounded one after \
it. A mistyped number with a run id attached is worse than no number: it \
looks verified.
"""

COMPUTE_PROFILES = {
    "laptop": """\
# Compute budget: laptop

This machine is a laptop, not a cluster. Size every calculation so it finishes \
in minutes, and prefer a converged cheap answer to an unconverged expensive one:

- Engines: prefer `emt` or `lj` for structure and workflow shakeouts, a \
classical potential through `lammps` when you have the potential file \
(classical force fields are laptop-friendly at real system sizes), and a \
served MLIP checkpoint (via rootstock) when one is declared and chemistry \
actually matters — call `list_engines` to see what is available. Reach for \
`qe` only when the question needs DFT, and then keep it small.
- Cells: single-digit atoms for DFT, tens of atoms for MLIPs. Build the smallest \
cell that can answer the question; do not run a 2x2x2 supercell to check that a \
script works.
- Quantum ESPRESSO: expand the `fast` protocol \
(`qe_protocol_options(atoms, protocol="fast")`), and expect coarse k-meshes and \
loose thresholds. Never expand `stringent` here.
- Molecular dynamics: short runs — picoseconds, not nanoseconds — on small cells.
- Before launching anything you expect to run longer than about ten minutes, say \
so and ask first.

These are low-accuracy, smoke-test settings, and saying so is part of the \
result. Record the accuracy caveat in the run's `intent`, in the notebook \
entry, and in your final report. A number produced at laptop settings is \
evidence that the workflow runs, not a production result — never present it as \
one.""",
    "workstation": """\
# Compute budget: workstation

This machine is a workstation: bigger than a laptop, smaller than a cluster. \
Medium cells and the `balanced` protocol are reasonable; hour-scale jobs are \
acceptable if you say what you are starting and why. Anything that would run \
overnight belongs on a cluster — say so rather than starting it.""",
    "cluster": """\
# Compute budget: cluster

Production settings are appropriate here. Use the `balanced` protocol by \
default and `stringent` when the result must be publishable. Universal MLIPs \
on a cluster are *served*, never pip-installed: call `list_engines` and use \
a served checkpoint id directly as the engine name for the cheap-relax leg \
(e.g. engine="mace-mp-0-medium"). SLAB has no in-process MLIP path — a \
checkpoint id resolved through rootstock is the only route for running one. \
Training a new potential is separate: the `train_potential` task drives \
gracemaker as a GPU batch job (see the mlip-training skill), and the result \
is deployed, never imported. Anything longer \
than a few minutes goes through `submit_job` (typically wrapping \
`slab run workflow.py`) rather than running in this process — then poll \
`job_status`. Keep interactive work on this node small.""",
}


def compute_profile_block(profile: str) -> str:
    """Guidance for the machine's size, or empty for an unknown profile.

    This shapes what the agent *chooses*; it changes no physics on its own.
    Every choice it leads to still lands in explicit, traced
    ``calculator_options`` that the run records.

    Examples:
        >>> compute_profile_block("laptop").splitlines()[0]
        '# Compute budget: laptop'
        >>> compute_profile_block("supercomputer")
        ''
    """
    return COMPUTE_PROFILES.get(profile, "")


#: Default guidance whenever the file fence is on: the fence bounds the file
#: tools mechanically, and this tells the model to treat the bound as the
#: working area rather than probing past it with the shell. Rendered in the
#: environment block because it names the session's directories, and skipped
#: under ``file_scope = "anywhere"``, where it would be false.
WORKING_BOUNDS = """\
# Working bounds

Stay inside the project directory and the workspace. The file tools are \
fenced to them, plus any read-only data the machine declares (a configured \
Materials Project snapshot, for example) — the fence's own refusal lists \
the exact roots. Do not explore, list, or search other locations with the \
shell, and do not read files the task does not need. When work seems to \
need data or software outside these directories, name the path, say why, \
and ask the user before you touch it."""

FENCED_PROTOCOL = """\

# Tool protocol (fenced)

This server does not parse native tool calls. To use a tool, write exactly one \
fenced block per message:

```tool
{"tool": "<name>", "arguments": {...}}
```

Available tools:
{catalog}

After the block, stop — the result arrives in the next user message.
"""

COMPACTION_PROMPT = """\
You are compacting an agent transcript into a working summary the agent will \
continue from — everything not carried forward is lost. Write these sections, \
tersely and concretely:

GOAL: the user's current objective, quoted as closely as the transcript allows.
STATE: what has been done, with file paths and SLAB run ids.
VERIFIED RESULTS: every number established so far, with units and run ids.
FAILURES OBSERVED: what failed, its diagnosis, what must not be repeated.
DECISIONS: choices made and their reasons.
OPEN: the immediate next steps and unresolved questions.

Do not invent anything not in the transcript. Do not soften failures.
"""


def team_block(spec: AgentSpec, roster: dict[str, AgentSpec]) -> str:
    """The ``# Your team`` section for a delegating agent, or empty.

    One line per other card — the descriptions are written as delegation
    triggers, so this list is what the PI reads when deciding whom to hand
    a task to. Rendered only when the ``delegate`` tool actually exists in
    the session, so the prompt never promises an absent tool.
    """
    others = [card for name, card in sorted(roster.items()) if name != spec.name]
    if not others:
        return ""
    lines = [
        "# Your team",
        "",
        "Specialists you can hand a scoped task to with the delegate tool. "
        "Delegate work that is separable and would crowd your context; brief "
        "them with the goal, the constraints, and what to return.",
        "",
    ]
    lines.extend(f"- {card.name}: {card.description}" for card in others)
    return "\n".join(lines)


def _sandbox_context_text() -> str:
    """The render-time sandbox facts, when this session runs inside one.

    The sandbox job exports ``SLAB_SANDBOX_CONTEXT`` naming the context
    file its render wrote. Carrying it in the prompt saves every session
    the opening steps of inspecting the submission script to learn its
    own cage. Unset or unreadable means not a sandbox: say nothing.
    """
    path = os.environ.get("SLAB_SANDBOX_CONTEXT")
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def environment_block(
    session: MasonSession,
    skills: dict[str, Skill] | None = None,
    team: str | None = None,
) -> str:
    """The per-session context: where we are, what exists here, what memory says."""
    from slab.hpc import allocated_tasks, cpu_budget

    lines = [
        "# Environment",
        f"project directory: {session.cwd}",
        f"workspace: {session.workspace_root}",
        f"platform: {platform.system()} {platform.machine()}",
        f"date: {datetime.now(UTC).strftime('%Y-%m-%d')}",
        f"compute profile: {session.compute_profile}",
        # The two parallelism facts the agent must size work within: what
        # runs here may not exceed the CPU budget, and MPI engines already
        # launch at the stated width — do not add -np on top of it.
        f"cpus: {cpu_budget()} usable in this session; MPI engines launch "
        f"with {allocated_tasks()} rank(s) automatically. Size scripts and "
        f"delegated tasks within these; a launch requesting more ranks than "
        f"the budget is refused.",
    ]
    if session.hpc.partitions:
        cluster = session.hpc.cluster or "unnamed cluster"
        partitions = ", ".join(sorted(session.hpc.partitions))
        default = session.hpc.default_partition
        suffix = f" (default {default})" if default else ""
        lines.append(f"cluster: {cluster}; partitions: {partitions}{suffix}")
    else:
        lines.append("cluster: none configured (no SLURM tools this session)")
    sandbox = _sandbox_context_text()
    if sandbox:
        lines.append("\n" + sandbox)
    if session.agent.file_scope == "project":
        lines.append("\n" + WORKING_BOUNDS)
    if skills:
        lines.append("\n" + catalog_block(skills))
    if session.agent.memory:
        # Read here rather than passed in, so the block is current after
        # every compaction: a fact this session recorded an hour ago is in
        # the rebuilt prompt, and one another session recorded is too.
        memories = memory_store.catalog_block(memory_store.discover())
        if memories:
            lines.append("\n" + memories)
    if team:
        lines.append("\n" + team)
    agents_md = _conventions_text(session)
    if agents_md:
        lines.append("\n# Project conventions (AGENTS.md)\n" + agents_md)
    plan = session.plan_text()
    if plan:
        lines.append("\n# Current plan (PLAN.md)\n" + plan)
    notebook = session.notebook_tail()
    if notebook:
        lines.append("\n# Lab notebook (latest entries)\n" + notebook)
    return "\n".join(lines)


def _conventions_text(session: MasonSession, max_chars: int = 6_000) -> str:
    """AGENTS.md from the project root — the cross-tool conventions standard."""
    path = session.cwd / "AGENTS.md"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[AGENTS.md truncated]"
    return text


def system_messages(
    session: MasonSession,
    spec: AgentSpec | None = None,
    catalog: str | None = None,
    *,
    skills: dict[str, Skill] | None = None,
    team: str | None = None,
    absent_tools: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """The system prompt: role, core, budget, software notes, protocol, environment.

    *spec* supplies the role block (the agent card's body); ``None`` yields
    the bare harness voice. Layers are ordered by change frequency so a
    prefix-caching server reuses the KV cache: the role and core never
    change; the environment (with the *skills* catalog and the *team*
    block) is stable within one session. *absent_tools* names the tools
    the shared text mentions that this session does not offer (a
    specialist without ``plan``, a laptop without ``submit_job``), so the
    model is told rather than left to discover it through a failed call.
    """
    prompt = (spec.prompt.rstrip() + "\n\n" + CORE_PROMPT) if spec is not None else CORE_PROMPT
    budget = compute_profile_block(session.compute_profile)
    if budget:
        prompt += "\n" + budget + "\n"
    if session.agent.software_notes:
        # Machine-stable, so it sits with the static layers: the block only
        # changes when slab.toml (or a user note override) changes.
        prompt += "\n" + notes_block(load_slab_config(session.cwd)) + "\n"
    if catalog is not None:
        prompt += FENCED_PROTOCOL.replace("{catalog}", catalog)
    environment = environment_block(session, skills, team)
    if absent_tools:
        environment += (
            "\n\nNot available in this session, whatever the text above says: "
            + ", ".join(absent_tools)
            + ". Do not call them; use the tools the request offers."
        )
    return [{"role": "system", "content": prompt + "\n" + environment}]
