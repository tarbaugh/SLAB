"""The benchmark campaigns: fixed questions, one scorer, rendered tables.

Five copper questions with known answers (``docs/benchmark.md``). Mason
runs each as a campaign, per model and per machine, and the campaign
passes when the agent's structured finish carries the value inside the
tolerance band *and* every run it cites reached ``verified`` — the
verification condition is what makes "correct" mean *computed*, not
guessed. A campaign that fails records why: no finish report, no
structured result, a cited run that never verified, or a value outside
the band. The score is the answer to one question: was a correct answer
achieved?

Records are JSON lines in ``benchmarks/results.jsonl`` in the project
directory, so they travel from the cluster into the repository by an
ordinary commit. ``render`` rewrites marker regions in the docs page and
the README from those records. Nothing here names a machine: the
``machine`` label is one the user chooses.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foundation import _ops
from foundation.errors import AmbiguousRunIdError, RunNotFoundError
from foundation.runtime import Workspace
from mason.report import summarize
from mason.session import transcript_for, transcript_groups
from slab._version import __version__

#: Where a project keeps its scored campaigns, relative to the project root.
RECORDS_FILE = Path("benchmarks") / "results.jsonl"

#: Lifecycle states a cited run must have reached for its number to count.
PASSING_STATES = frozenset({"verified", "promoted", "archived"})

#: Engine classes the tolerance bands are keyed by.
DFT = "dft"
MLIP = "mlip"

_MARKER = "<!-- benchmark:{name}:{edge} -->"


class BenchmarkError(Exception):
    """A benchmark verb could not do what was asked; the message says why."""


@dataclass(frozen=True)
class Question:
    """One fixed campaign: the science question and how its answer is judged.

    ``results`` maps each result name to its unit; ``reference`` and
    ``tolerance`` are keyed by engine class (``dft`` or ``mlip``), then by
    result name. A ``band`` question passes when every value lies within
    ``tolerance`` of ``reference``; a ``threshold`` question passes when
    every value is at most ``tolerance`` (no reference).
    """

    key: str
    number: int
    question: str
    results: dict[str, str]
    reference: dict[str, dict[str, float]]
    tolerance: dict[str, dict[str, float]]
    kind: str = "band"
    experiment: str = ""
    skills: tuple[str, ...] = ()
    notes: str = ""

    @property
    def instruction(self) -> str:
        """The exact text given to the agent: the question plus the reporting clause.

        Examples:
            >>> QUESTIONS[0].instruction.endswith("and the run ids that produced it.")
            True
        """
        named = ", ".join(f"`{name}` in {unit}" for name, unit in self.results.items())
        plural = "them" if len(self.results) > 1 else "it"
        return (
            f"{self.question} Finish with results key {named}, "
            f"and the run ids that produced {plural}."
        )


_A0 = 3.632
_SURFACE = 1.33
_VACANCY = 1.07
_MELT = 1357.77

QUESTIONS: tuple[Question, ...] = (
    Question(
        key="a0",
        number=1,
        question="Determine the equilibrium lattice constant of fcc Cu from an equation of state.",
        results={"a0": "Å"},
        reference={DFT: {"a0": _A0}, MLIP: {"a0": _A0}},
        tolerance={DFT: {"a0": 0.03}, MLIP: {"a0": 0.05}},
        experiment="3.615 Å",
        skills=("equation-of-state",),
    ),
    Question(
        key="surface",
        number=2,
        question="Compute the Cu(111) surface energy.",
        results={"gamma_111": "J/m^2"},
        reference={DFT: {"gamma_111": _SURFACE}, MLIP: {"gamma_111": _SURFACE}},
        tolerance={DFT: {"gamma_111": 0.2}, MLIP: {"gamma_111": 0.4}},
        experiment="1.83 J/m^2 (polycrystalline average)",
        skills=("surface-energy",),
    ),
    Question(
        key="vacancy",
        number=3,
        question="Compute the monovacancy formation energy in fcc Cu.",
        results={"e_vac": "eV"},
        reference={DFT: {"e_vac": _VACANCY}, MLIP: {"e_vac": _VACANCY}},
        tolerance={DFT: {"e_vac": 0.15}, MLIP: {"e_vac": 0.3}},
        experiment="1.29 ± 0.02 eV",
        skills=("atomsk-defects", "convergence-study"),
    ),
    Question(
        key="melting",
        number=4,
        question=(
            "Estimate the melting point of Cu with the two-phase method under the served MLIP."
        ),
        results={"t_melt": "K"},
        reference={DFT: {"t_melt": _MELT}, MLIP: {"t_melt": _MELT}},
        tolerance={DFT: {"t_melt": 100.0}, MLIP: {"t_melt": 100.0}},
        experiment="1357.77 K",
        skills=("two-phase-melting", "melt-quench"),
        notes="The reference is experiment for every engine; EMT and classical potentials miss it.",
    ),
    Question(
        key="finetune",
        number=5,
        question=(
            "Fine-tune a GRACE potential on DFT labels for strained fcc Cu and validate it "
            "against held-out DFT single points."
        ),
        results={"energy_rmse": "meV/atom", "force_rmse": "eV/Å"},
        reference={},
        tolerance={
            DFT: {"energy_rmse": 5.0, "force_rmse": 0.1},
            MLIP: {"energy_rmse": 5.0, "force_rmse": 0.1},
        },
        kind="threshold",
        experiment="",
        skills=("mlip-training",),
        notes="The held-out set is generated in the campaign, so the reference is internal.",
    ),
)


def find_question(selector: str) -> Question:
    """A question by number or key.

    Examples:
        >>> find_question("1").key
        'a0'
        >>> find_question("vacancy").number
        3
    """
    for question in QUESTIONS:
        if selector == question.key or selector == str(question.number):
            return question
    known = ", ".join(f"{q.number}/{q.key}" for q in QUESTIONS)
    raise BenchmarkError(f"no benchmark question {selector!r}; known: {known}")


def question_for(transcript: Path) -> Question | None:
    """The question a transcript's opening user message asked, or None."""
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message":
            continue
        message = event.get("message") or {}
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        for question in QUESTIONS:
            if text == question.instruction:
                return question
        return None
    return None


# -- scoring ------------------------------------------------------------------


def _cited_runs(ws: Workspace, run_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Every cited run's details; an unknown or ambiguous id raises BenchmarkError."""
    details = []
    for run_id in run_ids:
        try:
            details.append(_ops.run_details(ws, run_id))
        except RunNotFoundError as e:
            raise BenchmarkError(f"cited run {run_id!r} does not exist: {e}") from e
        except AmbiguousRunIdError as e:
            raise BenchmarkError(f"cited run id {run_id!r} is ambiguous: {e}") from e
    return details


def _engines(details: Iterable[dict[str, Any]]) -> tuple[list[str], str]:
    """The engines the cited runs' tasks used, and the class the band is keyed by.

    Read from the task recipes, never from the report's prose. ``dft`` when
    any task ran the built-in ``qe`` or a registry alias built on SLAB's qe
    factory; ``mlip`` for everything else (served checkpoints, classical
    potentials, EMT).
    """
    engines: set[str] = set()
    dft = False
    for run in details:
        for task in run.get("tasks") or []:
            recipe = task.get("recipe") or {}
            engine = (recipe.get("params") or {}).get("engine")
            if engine is None:
                continue
            engines.add(str(engine))
            if str(engine).strip().lower() == "qe" or "qe_calculator" in json.dumps(recipe):
                dft = True
    return sorted(engines), (DFT if dft else MLIP)


def _judge(question: Question, engine_class: str, results: dict[str, Any]) -> str | None:
    """None when every result value satisfies the question; else the reason."""
    tolerance = question.tolerance[engine_class]
    reference = question.reference.get(engine_class, {})
    for name, unit in question.results.items():
        entry = results.get(name)
        value = entry.get("value") if isinstance(entry, dict) else None
        if not isinstance(value, int | float) or isinstance(value, bool):
            return f"no numeric result for {name!r} ({unit})"
        if question.kind == "threshold":
            if value > tolerance[name]:
                return f"{name} = {value} {unit} exceeds {tolerance[name]} {unit}"
        else:
            expected = reference[name]
            if abs(value - expected) > tolerance[name]:
                return (
                    f"{name} = {value} {unit} is outside {expected} ± {tolerance[name]} "
                    f"{unit} ({engine_class} band)"
                )
    return None


def score_session(
    root: Path,
    session: str,
    *,
    question: Question | None = None,
    machine: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Score one campaign and return its record. Never raises for a failed
    campaign — the record carries ``passed`` and ``reason``.

    Raises :class:`BenchmarkError` only when there is nothing to score: no
    such session, or a session that is not a benchmark campaign.
    """
    transcript = transcript_for(root, session)
    siblings = next(
        (group for conversation, group in transcript_groups(root) if conversation == transcript),
        [],
    )
    summary = summarize(transcript, siblings)
    asked = question or question_for(transcript)
    if asked is None:
        raise BenchmarkError(
            f"session {transcript.stem} did not open with a benchmark instruction; "
            "pass --question to score it as one anyway"
        )
    finish = summary["finish"]
    record: dict[str, Any] = {
        "question": asked.number,
        "key": asked.key,
        "session": transcript.stem,
        "model": model or summary.get("model") or "unknown",
        "provider": summary.get("provider"),
        "endpoint_origin": summary.get("endpoint_origin"),
        "machine": machine or summary.get("compute_profile") or "unknown",
        "engine_class": None,
        "engines": [],
        "run_ids": list(finish.get("run_ids") or []),
        "results": dict(finish.get("results") or {}),
        "reference": {},
        "tolerance": {},
        "passed": False,
        "reason": None,
        "steps": summary.get("total_steps"),
        "prompt_tokens": summary.get("total_prompt_tokens"),
        "completion_tokens": summary.get("total_completion_tokens"),
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slab_version": __version__,
    }

    def fail(reason: str) -> dict[str, Any]:
        record["reason"] = reason
        return record

    if not finish.get("reported"):
        return fail("no finish report")
    if not record["results"]:
        return fail("the finish carried no structured results")
    if not record["run_ids"]:
        return fail("the finish cited no run ids")
    with Workspace(root) as ws:
        try:
            details = _cited_runs(ws, record["run_ids"])
        except BenchmarkError as e:
            return fail(str(e))
        unverified = [
            f"{run['run']['id'][:10]} ({run['run']['state']})"
            for run in details
            if run["run"]["state"] not in PASSING_STATES
        ]
        if unverified:
            return fail("cited runs never verified: " + ", ".join(unverified))
        engines, engine_class = _engines(details)
    record["engines"] = engines
    record["engine_class"] = engine_class
    record["reference"] = dict(asked.reference.get(engine_class, {}))
    record["tolerance"] = dict(asked.tolerance[engine_class])
    reason = _judge(asked, engine_class, record["results"])
    if reason is not None:
        return fail(reason)
    record["passed"] = True
    return record


# -- records ------------------------------------------------------------------


def records_path(project: Path | None = None) -> Path:
    return (project or Path.cwd()) / RECORDS_FILE


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    """Every record in the file, oldest first; malformed lines are skipped."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def recorded_sessions(records: Iterable[dict[str, Any]]) -> set[str]:
    return {str(r.get("session")) for r in records}


def latest_by_cell(records: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """The newest record per (model, machine, question key); a re-score wins."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:  # file order is chronological
        cells[(str(record.get("model")), str(record.get("machine")), str(record.get("key")))] = (
            record
        )
    return cells


# -- running ------------------------------------------------------------------


def run_campaign(
    question: Question,
    *,
    workspace: Path | None,
    model: str | None = None,
    provider: str | None = None,
    endpoint: str | None = None,
    max_turns: int | None = None,
    agent: str | None = None,
) -> tuple[str, Any]:
    """Run one campaign in this process and return ``(session_id, TurnResult)``.

    The same composition ``slab mason run --auto`` uses, so the campaign
    is exactly what a person would have started by hand.
    """
    from mason import Mason
    from mason.cli import open_session, resolve_spec

    session = open_session(
        workspace,
        auto=True,
        model=model,
        endpoint=endpoint,
        provider=provider,
        max_turns=max_turns,
        interactive=False,
    )
    spec, roster = resolve_spec(agent)
    mason = Mason(session, spec=spec, roster=roster)
    try:
        result = mason.run_turn(question.instruction)
    finally:
        session.release_session_lock()
    return session.session_id, result


# -- rendering ----------------------------------------------------------------


def _fmt(value: float) -> str:
    return f"{value:g}"


def questions_table() -> str:
    lines = [
        "| # | Instruction to the agent | Reference (DFT-PBE) | Experiment "
        "| Passes when | Skills |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for q in QUESTIONS:
        if q.kind == "threshold":
            reference = "internal to the campaign"
            passes = "; ".join(
                f"{name} ≤ {_fmt(q.tolerance[DFT][name])} {unit}"
                for name, unit in q.results.items()
            )
        else:
            reference = ", ".join(
                f"{_fmt(q.reference[DFT][name])} {unit}" for name, unit in q.results.items()
            )
            passes = "; ".join(
                f"{name} within ±{_fmt(q.tolerance[DFT][name])} (DFT) or "
                f"±{_fmt(q.tolerance[MLIP][name])} (MLIP, classical, EMT) {unit}"
                for name, unit in q.results.items()
            )
        lines.append(
            f"| {q.number} | {q.instruction} | {reference} | {q.experiment or '—'} | "
            f"{passes} | {', '.join(q.skills)} |"
        )
    return "\n".join(lines)


def _cell(record: dict[str, Any] | None, question: Question) -> str:
    if record is None:
        return "—"
    values = ", ".join(
        f"{_fmt(entry['value'])} {unit}"
        for name, unit in question.results.items()
        if isinstance((entry := (record.get("results") or {}).get(name)), dict)
        and isinstance(entry.get("value"), int | float)
    )
    if record.get("passed"):
        return f"pass ({values})" if values else "pass"
    reason = str(record.get("reason") or "failed")
    return f"fail ({values}; {reason})" if values else f"fail ({reason})"


def results_table(records: Iterable[dict[str, Any]]) -> str:
    cells = latest_by_cell(records)
    rows = sorted({(model, machine) for model, machine, _ in cells})
    if not rows:
        return "No campaign has been scored yet."
    header = "| Model | Machine | " + " | ".join(f"Q{q.number} {q.key}" for q in QUESTIONS)
    rule = "| --- | --- | " + " | ".join("---" for _ in QUESTIONS) + " | --- |"
    lines = [header + " | Passed |", rule]
    for model, machine in rows:
        passed = 0
        cols = []
        for q in QUESTIONS:
            record = cells.get((model, machine, q.key))
            passed += 1 if record is not None and record.get("passed") else 0
            cols.append(_cell(record, q))
        lines.append(
            f"| {model} | {machine} | " + " | ".join(cols) + f" | {passed}/{len(QUESTIONS)} |"
        )
    return "\n".join(lines)


def readme_summary(records: Iterable[dict[str, Any]], link: str) -> str:
    cells = latest_by_cell(records)
    rows = sorted({(model, machine) for model, machine, _ in cells})
    if not rows:
        return f"No campaign has been scored yet. See [the benchmark]({link})."
    lines = ["| Model | Machine | Passed |", "| --- | --- | --- |"]
    for model, machine in rows:
        passed = sum(
            1
            for q in QUESTIONS
            if (r := cells.get((model, machine, q.key))) is not None and r.get("passed")
        )
        lines.append(f"| {model} | {machine} | {passed}/{len(QUESTIONS)} |")
    lines.append("")
    lines.append(f"Five copper questions with known answers; [the benchmark]({link}) has the rule.")
    return "\n".join(lines)


def rewrite_region(path: Path, name: str, body: str) -> bool:
    """Replace the text between the ``benchmark:<name>`` markers in *path*.

    Returns True when the file changed. A file without the marker pair is
    refused: the region is the contract that keeps hand-written prose safe.
    """
    start = _MARKER.format(name=name, edge="start")
    end = _MARKER.format(name=name, edge="end")
    text = path.read_text(encoding="utf-8")
    head, sep, rest = text.partition(start)
    if not sep:
        raise BenchmarkError(f"{path} has no '{start}' marker")
    _middle, sep, tail = rest.partition(end)
    if not sep:
        raise BenchmarkError(f"{path} has no '{end}' marker")
    replaced = f"{head}{start}\n{body}\n{end}{tail}"
    if replaced == text:
        return False
    path.write_text(replaced, encoding="utf-8")
    return True


def render(
    records: Iterable[dict[str, Any]],
    *,
    docs: Path | None,
    readme: Path | None,
    link: str = "https://tarbaugh.github.io/SLAB/benchmark/",
) -> list[Path]:
    """Rewrite the marker regions; return the files that changed."""
    records = list(records)
    changed: list[Path] = []
    if docs is not None:
        touched = rewrite_region(docs, "questions", questions_table())
        touched = rewrite_region(docs, "results", results_table(records)) or touched
        if touched:
            changed.append(docs)
    if readme is not None and rewrite_region(readme, "summary", readme_summary(records, link)):
        changed.append(readme)
    return changed


__all__ = [
    "DFT",
    "MLIP",
    "PASSING_STATES",
    "QUESTIONS",
    "RECORDS_FILE",
    "BenchmarkError",
    "Question",
    "append_record",
    "find_question",
    "latest_by_cell",
    "load_records",
    "question_for",
    "questions_table",
    "readme_summary",
    "recorded_sessions",
    "records_path",
    "render",
    "results_table",
    "rewrite_region",
    "run_campaign",
    "score_session",
]
