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

After the verdict, the review (:mod:`slab_stack.review`) reads the same
evidence and raises flags: attributable defects, each naming the skill,
card, or tool a revision would edit. The flags travel in the record
beside ``passed`` and ``reason``.

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
from typing import TYPE_CHECKING, Any

from foundation import _ops
from foundation.errors import (
    AmbiguousRunIdError,
    ArtifactNotFoundError,
    RunNotFoundError,
    SerializationError,
)
from foundation.runtime import Workspace
from foundation.serialize import loads
from mason.report import summarize
from mason.session import transcript_for, transcript_groups
from slab._version import __version__
from slab_stack import review

if TYPE_CHECKING:
    from mason.loop import ChatBackend
    from mason.skills import Skill

#: Where a project keeps its scored campaigns, relative to the project root.
RECORDS_FILE = Path("benchmarks") / "results.jsonl"

#: Lifecycle states a cited run must have reached for its number to count.
PASSING_STATES = frozenset({"verified", "promoted", "archived"})

#: Engine classes the reference values and tolerance bands are keyed by. A
#: DFT run is judged against the functional it actually used, because PBE
#: and PBEsol give different numbers for the same crystal; everything that
#: is not DFT (served checkpoints, classical potentials, EMT) shares the
#: wider ``mlip`` band around the PBE value.
PBE = "pbe"
PBESOL = "pbesol"
MLIP = "mlip"
DFT_CLASSES = (PBE, PBESOL)
CLASSES = (PBE, PBESOL, MLIP)

_MARKER = "<!-- benchmark:{name}:{edge} -->"


class BenchmarkError(Exception):
    """A benchmark verb could not do what was asked; the message says why."""


@dataclass(frozen=True)
class Question:
    """One fixed campaign: the science question and how its answer is judged.

    ``results`` maps each result name to its unit; ``reference`` and
    ``tolerance`` are keyed by engine class (``pbe``, ``pbesol``, or
    ``mlip``), then by result name. A ``band`` question passes when every
    value lies within ``tolerance`` of ``reference``; a ``threshold``
    question passes when every value is at most ``tolerance`` (no
    reference). A class missing from ``reference`` has no checked value
    yet, and the scorer refuses to judge a campaign in that class rather
    than guess.
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


def _every_class(**values: float) -> dict[str, dict[str, float]]:
    """The same per-result values for every engine class."""
    return {cls: dict(values) for cls in CLASSES}


# The reference values and their sources are listed in docs/benchmark.md.
# PBE and PBEsol are kept apart because the SSSP families SLAB installs
# are PBEsol by default, and PBEsol binds copper about 2% tighter than PBE.
_A0 = {PBE: 3.632, PBESOL: 3.562}
_SURFACE = {PBE: 1.33, PBESOL: 1.59}
_VACANCY = {PBE: 1.07}  # no checked PBEsol value yet; the scorer says so
_MELT = 1357.77

QUESTIONS: tuple[Question, ...] = (
    Question(
        key="a0",
        number=1,
        question="Determine the equilibrium lattice constant of fcc Cu from an equation of state.",
        results={"a0": "Å"},
        reference={PBE: {"a0": _A0[PBE]}, PBESOL: {"a0": _A0[PBESOL]}, MLIP: {"a0": _A0[PBE]}},
        tolerance={PBE: {"a0": 0.03}, PBESOL: {"a0": 0.03}, MLIP: {"a0": 0.05}},
        experiment="3.615 Å",
        skills=("equation-of-state",),
    ),
    Question(
        key="surface",
        number=2,
        question="Compute the Cu(111) surface energy.",
        results={"gamma_111": "J/m^2"},
        reference={
            PBE: {"gamma_111": _SURFACE[PBE]},
            PBESOL: {"gamma_111": _SURFACE[PBESOL]},
            MLIP: {"gamma_111": _SURFACE[PBE]},
        },
        tolerance={PBE: {"gamma_111": 0.2}, PBESOL: {"gamma_111": 0.2}, MLIP: {"gamma_111": 0.4}},
        experiment="1.79 ± 0.19 J/m^2 (polycrystalline average)",
        skills=("surface-energy",),
    ),
    Question(
        key="vacancy",
        number=3,
        question="Compute the monovacancy formation energy in fcc Cu.",
        results={"e_vac": "eV"},
        reference={PBE: {"e_vac": _VACANCY[PBE]}, MLIP: {"e_vac": _VACANCY[PBE]}},
        tolerance={PBE: {"e_vac": 0.15}, PBESOL: {"e_vac": 0.15}, MLIP: {"e_vac": 0.3}},
        experiment="1.29 ± 0.02 eV",
        skills=("atomsk-defects", "convergence-study"),
        notes="No checked PBEsol reference yet; a PBEsol campaign is refused, not guessed at.",
    ),
    Question(
        key="melting",
        number=4,
        question=(
            "Estimate the melting point of Cu with the two-phase method under the served MLIP."
        ),
        results={"t_melt": "K"},
        reference=_every_class(t_melt=_MELT),
        tolerance=_every_class(t_melt=100.0),
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
        tolerance=_every_class(energy_rmse=5.0, force_rmse=0.1),
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


def _is_qe(engine: str, recipe: dict[str, Any]) -> bool:
    """The built-in ``qe`` engine, or a registry alias built on SLAB's qe factory."""
    return engine.strip().lower() == "qe" or "qe_calculator" in json.dumps(recipe)


def _calculator_options(ws: Workspace, recipe: dict[str, Any]) -> dict[str, Any] | None:
    """A task's ``calculator_options`` as it was traced, or None when unreadable.

    The recipe keeps literals verbatim and larger inputs by hash; the
    options dict is the latter, so its bytes come from the artifact store.
    """
    raw = (recipe.get("params") or {}).get("calculator_options")
    if isinstance(raw, dict) and "$hash" in raw:
        try:
            loaded = loads(ws.artifacts.get(str(raw["$hash"])).read_bytes())
        except (ArtifactNotFoundError, SerializationError, OSError):
            return None
        return loaded if isinstance(loaded, dict) else None
    return raw if isinstance(raw, dict) else None


def functional_of(options: dict[str, Any] | None) -> str | None:
    """The exchange-correlation functional a qe options dict commits to.

    ``input_dft`` wins when set; otherwise the pseudopotential family name
    and then the pseudopotential file names decide, because SSSP names
    carry the functional they were generated for.

    Examples:
        >>> functional_of({"pseudo_family": "SSSP/1.3.0/PBEsol/efficiency"})
        'pbesol'
        >>> functional_of({"pseudopotentials": {"Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF"}})
        'pbe'
        >>> functional_of({"pseudo_family": "SSSP/1.3.0/PBEsol/efficiency",
        ...                "input_data": {"system": {"input_dft": "pbe"}}})
        'pbe'
        >>> functional_of(None) is None
        True
    """
    if not options:
        return None
    system = (options.get("input_data") or {}).get("system") or {}
    clues = [
        str(system.get("input_dft") or ""),
        str(options.get("pseudo_family") or ""),
        " ".join(str(v) for v in (options.get("pseudopotentials") or {}).values()),
    ]
    for clue in clues:
        lowered = clue.lower()
        if "pbesol" in lowered:
            return PBESOL
        if "pbe" in lowered:
            return PBE
    return None


def _engines(ws: Workspace, details: Iterable[dict[str, Any]]) -> tuple[list[str], str]:
    """The engines the cited runs' tasks used, and the class the band is keyed by.

    Read from the task recipes, never from the report's prose. A DFT task
    (the built-in ``qe`` or a registry alias built on SLAB's qe factory)
    puts the campaign in the class of its functional, ``pbe`` or
    ``pbesol``; everything else (served checkpoints, classical potentials,
    EMT) is ``mlip``.

    Raises:
        BenchmarkError: A DFT task whose functional cannot be read from its
            traced options, or cited runs that mix functionals. Either way
            the band is undefined, and the record says why.
    """
    engines: set[str] = set()
    functionals: dict[str, str] = {}
    for run in details:
        run_id = str(run["run"]["id"])[:10]
        for task in run.get("tasks") or []:
            recipe = task.get("recipe") or {}
            engine = (recipe.get("params") or {}).get("engine")
            if engine is None:
                continue
            engines.add(str(engine))
            if not _is_qe(str(engine), recipe):
                continue
            functional = functional_of(_calculator_options(ws, recipe))
            if functional is None:
                raise BenchmarkError(
                    f"cannot tell the functional of cited run {run_id} from its traced "
                    "calculator options, so its band is undefined"
                )
            functionals[functional] = run_id
    if len(functionals) > 1:
        mixed = ", ".join(f"{f} ({r})" for f, r in sorted(functionals.items()))
        raise BenchmarkError(f"cited runs mix functionals, so the band is undefined: {mixed}")
    engine_class = next(iter(functionals), MLIP)
    return sorted(engines), engine_class


def _judge(question: Question, engine_class: str, results: dict[str, Any]) -> str | None:
    """None when every result value satisfies the question; else the reason.

    Raises:
        BenchmarkError: The question has no checked reference for this
            class, so there is nothing to judge against.
    """
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
            if name not in reference:
                raise BenchmarkError(
                    f"Q{question.number} has no checked {engine_class} reference for {name}; "
                    "add one with its source to QUESTIONS before scoring this campaign"
                )
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
    catalog: dict[str, Skill] | None = None,
    referee: ChatBackend | None = None,
) -> dict[str, Any]:
    """Score one campaign, review it, and return its record. Never raises
    for a failed campaign — the record carries ``passed`` and ``reason``,
    and ``flags`` from the review.

    *catalog* is the skill catalog the flags are judged against (the one
    visible from the working directory when omitted); *referee* is a chat
    client for the referee, or None to run the rules alone.

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
        "agent": summary.get("agent") or "pi",
        "engine_class": None,
        "engines": [],
        "skills": review.loaded_skills(transcript),
        "run_ids": list(finish.get("run_ids") or []),
        "results": dict(finish.get("results") or {}),
        "reference": {},
        "tolerance": {},
        "passed": False,
        "reason": None,
        "flags": [],
        "reviewed_by": [],
        "steps": summary.get("total_steps"),
        "prompt_tokens": summary.get("total_prompt_tokens"),
        "completion_tokens": summary.get("total_completion_tokens"),
        "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "slab_version": __version__,
    }
    with Workspace(root) as ws:
        _judge_campaign(ws, asked, finish, record)  # may refuse: no reference
        review.review(
            root, transcript, asked, record, ws=ws, catalog=catalog, referee_client=referee
        )
    return record


def _judge_campaign(
    ws: Workspace, asked: Question, finish: dict[str, Any], record: dict[str, Any]
) -> None:
    """Settle ``passed`` and ``reason`` in *record* from the finish and the runs."""

    def fail(reason: str) -> None:
        record["reason"] = reason

    if not finish.get("reported"):
        return fail("no finish report")
    if not record["results"]:
        return fail("the finish carried no structured results")
    if not record["run_ids"]:
        return fail("the finish cited no run ids")
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
    try:
        engines, engine_class = _engines(ws, details)
    except BenchmarkError as e:
        return fail(str(e))
    record["engines"] = engines
    record["engine_class"] = engine_class
    record["reference"] = dict(asked.reference.get(engine_class, {}))
    record["tolerance"] = dict(asked.tolerance[engine_class])
    reason = _judge(asked, engine_class, record["results"])  # may refuse: no reference
    if reason is not None:
        return fail(reason)
    record["passed"] = True
    return None


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


def _reference_cell(q: Question, cls: str) -> str:
    if q.kind == "threshold":
        return "internal to the campaign"
    values = q.reference.get(cls, {})
    if not values:
        return "no checked value yet"
    return ", ".join(f"{_fmt(values[name])} {unit}" for name, unit in q.results.items())


def questions_table() -> str:
    lines = [
        "| # | Instruction to the agent | Reference (PBE) | Reference (PBEsol) | Experiment "
        "| Passes when | Skills |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for q in QUESTIONS:
        if q.kind == "threshold":
            passes = "; ".join(
                f"{name} ≤ {_fmt(q.tolerance[PBE][name])} {unit}"
                for name, unit in q.results.items()
            )
        else:
            passes = "; ".join(
                f"{name} within ±{_fmt(q.tolerance[PBE][name])} (DFT, against its "
                f"functional's reference) or ±{_fmt(q.tolerance[MLIP][name])} "
                f"(MLIP, classical, EMT, against PBE) {unit}"
                for name, unit in q.results.items()
            )
        lines.append(
            f"| {q.number} | {q.instruction} | {_reference_cell(q, PBE)} | "
            f"{_reference_cell(q, PBESOL)} | {q.experiment or '—'} | "
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
    raised = len(record.get("flags") or [])
    flagged = f"{raised} flag" + ("s" if raised != 1 else "") if raised else ""
    if record.get("passed"):
        detail = "; ".join(part for part in (values, flagged) if part)
        return f"pass ({detail})" if detail else "pass"
    reason = str(record.get("reason") or "failed")
    detail = "; ".join(part for part in (values, reason, flagged) if part)
    return f"fail ({detail})"


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


def flags_table(records: Iterable[dict[str, Any]], catalog: dict[str, Skill] | None = None) -> str:
    """The defect list: every flag on the latest record per cell, with its status."""
    from mason.skills import discover_skills

    rows = review.ledger(records, catalog if catalog is not None else discover_skills(Path.cwd()))
    if not rows:
        return "No flag has been raised on a recorded campaign."
    lines = [
        "| Target | Rule | Status | Revision | Q | Model | Machine | Raised by | Evidence | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = [
            row["target"], row["rule"], row["status"], row["against"] or "—",
            f"Q{row['question']}", row["model"], row["machine"], row["raised_by"],
            row["evidence"], row["note"],
        ]
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
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
    catalog: dict[str, Skill] | None = None,
) -> list[Path]:
    """Rewrite the marker regions; return the files that changed.

    The docs page carries three regions (``questions``, ``results``,
    ``flags``) and the README one (``summary``). *catalog* decides each
    flag's status; the working directory's catalog when omitted.
    """
    records = list(records)
    changed: list[Path] = []
    if docs is not None:
        touched = rewrite_region(docs, "questions", questions_table())
        touched = rewrite_region(docs, "results", results_table(records)) or touched
        touched = rewrite_region(docs, "flags", flags_table(records, catalog)) or touched
        if touched:
            changed.append(docs)
    if readme is not None and rewrite_region(readme, "summary", readme_summary(records, link)):
        changed.append(readme)
    return changed


__all__ = [
    "CLASSES",
    "DFT_CLASSES",
    "MLIP",
    "PASSING_STATES",
    "PBE",
    "PBESOL",
    "QUESTIONS",
    "RECORDS_FILE",
    "BenchmarkError",
    "Question",
    "append_record",
    "find_question",
    "flags_table",
    "functional_of",
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
