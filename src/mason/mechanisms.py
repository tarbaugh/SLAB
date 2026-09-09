"""The harness mechanisms as switches, and the conditions the benchmark runs.

Each distinctive thing the harness does beyond calling the model in a loop
is a *mechanism* with a name. The name is a switch: ``[agent] mechanisms``
lists the ones a session runs with, and the loop, the toolbox, and the
prompt each consult :func:`enabled` before doing what the mechanism does.
A mechanism that cannot be switched off cannot be measured, so nothing
here is decorative: the benchmark's ablation grid (``slab benchmark
matrix``) turns one off at a time, and the record says which.

A *condition* is a card plus a mechanism set. Three ship, one per arm of
the reliability question: ``slab`` is Mason as it is; ``protocol`` is the
skill collection with a file protocol and no runtime gates, the shape of
the AICC control plane; ``bare`` is the model with a shell and a
one-paragraph prompt. Plumbing (JSON repair, truncation detection, sending
the effort field) has no switch, because it is necessary and is not a
finding.

This module imports nothing from the rest of ``mason`` so the config
loader can validate against it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Mechanism:
    """One harness mechanism: its switch name, what it does, its evidence."""

    name: str
    does: str
    evidence: str


MECHANISMS: tuple[Mechanism, ...] = (
    Mechanism(
        "check-gating",
        "Calculations run as traced workflow scripts through launch_workflow, "
        "wait_for_run, list_runs, show_run, and read_artifact; a run whose "
        "checks pass is verified, and the prompt teaches that path. Off, the "
        "shell is the only way to run a script.",
        "SLAB's verification gate (ARCHITECTURE.md); a number without a verified run is a rumor.",
    ),
    Mechanism(
        "failure-records",
        "A failed run's structured failure record (trimmed traceback and "
        "diagnostic notes) is returned with the run, and the prompt asks for a "
        "diagnosis before a retry. Off, a failed run reports its status only.",
        "Reflexion (arXiv:2303.11366); Manus, keep failures in context.",
    ),
    Mechanism(
        "critic-gate",
        "The review tool reaches a read-only critic, and a card that reviews "
        "first spends no compute before the critic approves the plan.",
        "Agent Laboratory (arXiv:2501.04227): fixed tool libraries with "
        "checkpoints beat full autonomy.",
    ),
    Mechanism(
        "machine-memory",
        "Facts a session learned about this machine persist through remember "
        "and enter later prompts through recall and the memory catalog.",
        "MemGPT (arXiv:2310.08560); an overnight job that trips on a quirk "
        "at 03:00 should not trip on it twice.",
    ),
    Mechanism(
        "context-hygiene",
        "Old tool results are cleared to placeholders once the prompt is "
        "large, and superseded plan echoes are folded, before compaction.",
        "SWE-agent and OpenHands: masking old observations matches "
        "summarization at half the cost; Anthropic's clear_tool_uses.",
    ),
    Mechanism(
        "identical-result-annotation",
        "A tool result identical to the same call's previous result carries a "
        "note the model reads as evidence, escalating with the repeat count.",
        "A real 240-call session spent 82 calls on one byte-identical readelf pipeline.",
    ),
    Mechanism(
        "budget-hint",
        "An ephemeral step-of-budget line follows every request, stricter "
        "near the ceiling, with a note when the last steps only looked.",
        "Transcripts stopped at the call budget mid-inquiry with nothing "
        "written down; the hint moved the finish earlier.",
    ),
    Mechanism(
        "skills",
        "The skill tool loads procedures and tested scripts from the catalog "
        "in the Agent Skills format, one line per skill until loaded.",
        "Anthropic, Agent Skills; the skills audit of 2026-09-03.",
    ),
    Mechanism(
        "delegation",
        "A lead hands a separable task to a specialist card that runs its own "
        "loop one level down and returns a report.",
        "Anthropic's multi-agent research system: context isolation pays for "
        "separable subtasks only.",
    ),
    Mechanism(
        "adaptive-effort",
        "A reply cut at the token budget is retried once at lower effort with "
        "a request for brevity before the turn ends.",
        "Transcripts where a high-effort reply was cut twice and the turn ended with no report.",
    ),
)

MECHANISM_NAMES: frozenset[str] = frozenset(m.name for m in MECHANISMS)
#: Every mechanism: the set Mason runs with unless configured otherwise.
ALL_MECHANISMS: frozenset[str] = MECHANISM_NAMES


@dataclass(frozen=True)
class Condition:
    """One harness arm: the entry card and the mechanisms it runs with."""

    name: str
    card: str
    mechanisms: frozenset[str]
    summary: str


#: The mechanisms a general coding harness supplies on its own: the
#: protocol arm keeps these, because the AICC shape runs inside one.
_LOOP_MECHANISMS = frozenset(
    {"context-hygiene", "identical-result-annotation", "budget-hint", "adaptive-effort"}
)

CONDITIONS: dict[str, Condition] = {
    "slab": Condition(
        "slab",
        "pi",
        ALL_MECHANISMS,
        "Mason as it is: the PI card, every mechanism on, verification gated in code.",
    ),
    "protocol": Condition(
        "protocol",
        "protocol",
        _LOOP_MECHANISMS | {"skills"},
        "The skill collection with a file protocol: scripts run with the shell, "
        "an append-only provenance log in the project, verification is what the "
        "agent writes down. No run tools, no failure records, no critic, no memory.",
    ),
    "bare": Condition(
        "bare",
        "bare",
        frozenset(),
        "The model with read, write, shell, and finish, a one-paragraph prompt, "
        "and no mechanism at all.",
    ),
}


class ConditionError(ValueError):
    """A condition or mechanism name nothing here defines."""


def check_mechanisms(names: Iterable[str]) -> tuple[str, ...]:
    """The names, sorted and de-duplicated, or a :class:`ConditionError`.

    Examples:
        >>> check_mechanisms(["skills", "budget-hint", "skills"])
        ('budget-hint', 'skills')
        >>> check_mechanisms(["budget"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        mason.mechanisms.ConditionError: no mechanism named 'budget'; the switches: ...
    """
    wanted = sorted(set(names))
    unknown = [n for n in wanted if n not in MECHANISM_NAMES]
    if unknown:
        raise ConditionError(
            f"no mechanism named {', '.join(repr(n) for n in unknown)}; the switches: "
            + ", ".join(sorted(MECHANISM_NAMES))
        )
    return tuple(wanted)


def resolve(condition: str | None, without: Iterable[str] = ()) -> tuple[Condition, frozenset[str]]:
    """The condition (``slab`` when unnamed) and its mechanism set minus *without*.

    Examples:
        >>> cond, on = resolve("protocol")
        >>> cond.card, "check-gating" in on, "skills" in on
        ('protocol', False, True)
        >>> _, on = resolve(None, ["budget-hint"])
        >>> "budget-hint" in on, len(on) == len(MECHANISMS) - 1
        (False, True)
        >>> resolve("aicc")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        mason.mechanisms.ConditionError: no condition named 'aicc'; the conditions: ...
    """
    chosen = CONDITIONS.get(condition or "slab")
    if chosen is None:
        raise ConditionError(
            f"no condition named {condition!r}; the conditions: " + ", ".join(sorted(CONDITIONS))
        )
    ablated = check_mechanisms(without)
    absent = [n for n in ablated if n not in chosen.mechanisms]
    if absent:
        raise ConditionError(
            f"condition {chosen.name!r} does not run {', '.join(absent)}, so there is "
            f"nothing to switch off"
        )
    return chosen, chosen.mechanisms - set(ablated)


class _Switches(Protocol):
    """The slice of ``[agent]`` the switches read (an ``AgentConfig`` fits)."""

    @property
    def mechanisms(self) -> tuple[str, ...] | None: ...

    @property
    def memory(self) -> bool: ...

    @property
    def delegation(self) -> bool: ...

    @property
    def clear_tool_results(self) -> bool: ...


#: The older per-mechanism flags in ``[agent]``, each still honored: a
#: mechanism is on when the set allows it *and* its own flag does.
_LEGACY_FLAGS = {
    "machine-memory": "memory",
    "delegation": "delegation",
    "context-hygiene": "clear_tool_results",
}


def enabled(agent: _Switches, name: str) -> bool:
    """Whether the session runs mechanism *name*.

    ``None`` for ``mechanisms`` means every one. An unknown name is a
    programming error and raises, so a misspelled switch in code cannot
    read as "off".
    """
    if name not in MECHANISM_NAMES:
        raise ConditionError(f"no mechanism named {name!r}")
    if agent.mechanisms is not None and name not in agent.mechanisms:
        return False
    flag = _LEGACY_FLAGS.get(name)
    return True if flag is None else bool(getattr(agent, flag))


def effective(agent: _Switches) -> tuple[str, ...]:
    """The mechanisms a session actually runs with, in name order."""
    return tuple(m.name for m in MECHANISMS if enabled(agent, m.name))


def harness_label(condition: str | None, ablated: Iterable[str] = ()) -> str:
    """The one-word-or-so name of an arm for a table: ``slab``, ``slab -budget-hint``.

    Examples:
        >>> harness_label(None)
        'slab'
        >>> harness_label("slab", ["budget-hint"])
        'slab -budget-hint'
    """
    label = condition or "slab"
    return label + "".join(f" -{n}" for n in sorted(set(ablated)))


def entry_card(condition: str | None, agent: str | None = None) -> str | None:
    """The card a session runs as: an explicit *agent* wins, else the condition's.

    Examples:
        >>> entry_card("protocol"), entry_card("protocol", "planner"), entry_card(None)
        ('protocol', 'planner', None)
    """
    if agent is not None:
        return agent
    if condition is None:
        return None
    return resolve(condition)[0].card


def mechanisms_table() -> str:
    """The mechanism ledger as a markdown table: switch, what it does, evidence."""
    lines = ["| Switch | What it does | Evidence |", "| --- | --- | --- |"]
    for m in MECHANISMS:
        lines.append(f"| `{m.name}` | {m.does} | {m.evidence} |")
    return "\n".join(lines)


def conditions_table() -> str:
    """The three conditions as a markdown table: name, card, mechanisms on."""
    lines = ["| Condition | Card | Mechanisms on | What it is |", "| --- | --- | --- | --- |"]
    for cond in CONDITIONS.values():
        on = ", ".join(f"`{n}`" for n in sorted(cond.mechanisms)) or "none"
        lines.append(f"| `{cond.name}` | `{cond.card}` | {on} | {cond.summary} |")
    return "\n".join(lines)


__all__ = [
    "ALL_MECHANISMS",
    "CONDITIONS",
    "MECHANISMS",
    "MECHANISM_NAMES",
    "Condition",
    "ConditionError",
    "Mechanism",
    "check_mechanisms",
    "conditions_table",
    "effective",
    "enabled",
    "entry_card",
    "harness_label",
    "mechanisms_table",
    "resolve",
]
