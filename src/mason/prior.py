"""What a plan inherits: the project's earlier findings and the runs it names.

A lead's plan passes three checks that tie it to the record:

* The environment block shows the earlier notebook entries of this project
  that the latest-entries tail leaves out, entry by entry with their dates
  (:func:`prior_findings_block`). One real planner centred a temperature
  ladder far above the value an earlier probe had already found, because
  that entry sat outside the tail.
* The ``plan`` tool refuses a plan whose Goal names a quantity the notebook
  already reports, unless the plan carries a line ``prior result: ...``
  (:func:`prior_result_check`). A match by a single shared word is only a
  warning, because one word is weak evidence of the same quantity.
* The ``plan`` tool checks every artifact reference ``run:<id>/<name>``
  against the run store and rewrites a reference to a cache-hit run into
  one to the run that produced the file (:func:`resolve_artifact_refs`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from foundation.project import NotebookEntry, notebook_entries, notebook_path

# -- the prior findings block ---------------------------------------------------

#: How many earlier entries the block shows, and how much of each.
PRIOR_ENTRIES = 6
_ENTRY_CHARS = 600


def _words(text: str) -> str:
    """Lowercase words joined by single spaces, punctuation and underscores dropped.

    Examples:
        >>> _words("W-MLIP melting — probe [dft-expert]")
        'w mlip melting probe dft expert'
    """
    return " ".join(part for part in re.split(r"[\W_]+", text.lower()) if part)


def names_project(heading: str, project: str) -> bool:
    """Whether a notebook heading names the project, as whole words.

    Case, hyphens, underscores, and spaces do not matter.

    Examples:
        >>> names_project("2026-09-12 10:31 UTC — W-MLIP coexistence probe", "w_mlip")
        True
        >>> names_project("2026-09-12 10:31 UTC — tungsten", "w_mlip")
        False
    """
    key = _words(project)
    return bool(key) and f" {key} " in f" {_words(heading)} "


def prior_findings_block(project: Path, tail_chars: int, count: int = PRIOR_ENTRIES) -> str:
    """The ``# Prior findings`` section: earlier entries the notebook tail leaves out.

    The tail shows the last *tail_chars* characters of the notebook, so an
    entry that starts before them is cut or absent. The block shows the
    last *count* of those entries whole up to a cap each, with their
    dated headings. When some heading in the notebook names the project
    directory, only the entries whose heading names it are shown: the
    others belong to another campaign that shares the directory. Empty
    when the tail already shows the whole notebook.

    Examples:
        >>> import tempfile
        >>> from foundation.project import notebook_append
        >>> project = Path(tempfile.mkdtemp())
        >>> _ = notebook_append(project, "T = 1180 K holds solid", heading="probe")
        >>> _ = notebook_append(project, "x" * 400, heading="later")
        >>> block = prior_findings_block(project, tail_chars=300)
        >>> block.splitlines()[0], "T = 1180 K holds solid" in block, "later" in block
        ('# Prior findings (notebook)', True, True)
        >>> prior_findings_block(project, tail_chars=10_000)
        ''
    """
    path = notebook_path(project)
    if not path.exists():
        return ""
    size = len(path.read_text(encoding="utf-8"))
    if size <= tail_chars:
        return ""
    entries = notebook_entries(project)
    older = [entry for entry in entries if entry.offset < size - tail_chars]
    name = Path(project).resolve().name
    filtered = any(names_project(entry.heading, name) for entry in entries)
    pool = [e for e in older if names_project(e.heading, name)] if filtered else older
    chosen = pool[-count:]
    if not chosen:
        return ""
    which = f"whose heading names {name}, " if filtered else ""
    lines = [
        "# Prior findings (notebook)",
        f"Earlier notebook entries {which}which the latest-entries section leaves "
        f"out, newest last. A quantity reported here is a prior result: a plan "
        f"whose Goal names it carries a line `prior result: ...` and starts from it.",
    ]
    skipped = len(pool) - len(chosen)
    if skipped:
        noun = "entry" if skipped == 1 else "entries"
        lines.append(f"[{skipped} earlier {noun} not shown; read {path.name}]")
    for entry in chosen:
        body = entry.body
        if len(body) > _ENTRY_CHARS:
            body = body[:_ENTRY_CHARS].rstrip() + f" [... entry clipped; read {path.name}]"
        lines += ["", f"## {entry.heading}", body]
    return "\n".join(lines)


# -- the prior-result check -----------------------------------------------------

#: The marker a plan uses to state what the notebook already reports.
#: A ``prior result:`` line, with room for a qualifier before the colon, so
#: ``Prior result (run 01abc): 1180 K`` counts as the marker too.
PRIOR_RESULT_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?prior result(?:\*\*)?[^:\n]{0,60}:", re.I | re.M
)

#: A number with a physical unit: the mark of a line that reports a quantity.
_MEASURED = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*"
    r"(?:K|Kelvin|mK|eV/atom|meV/atom|eV/Å|eV/A|eV|meV|Å\^?3|Å|Angstrom|angstrom|GPa|MPa|kbar"
    r"|bar|Pa|ps|fs|ns|nm|g/cm\^?3|J/m\^?2|kJ/mol|kcal/mol|THz|cm-1|W/mK|%)"
    r"(?![A-Za-z])"
)

#: The second word of a phrase that names a quantity ("melting point", "bulk
#: modulus"). A goal phrase ending in one of these that a measured notebook
#: line repeats is the same quantity; any other shared phrase is only words.
_QUANTITY_HEADS = frozenset(
    {
        "angle", "barrier", "capacity", "charge", "coefficient", "conductivity",
        "constant", "constants", "curve", "density", "diffusivity", "distance",
        "energies", "energy", "enthalpy", "entropy", "expansion", "fraction",
        "frequencies", "frequency", "gap", "heat", "length", "magnetization",
        "modulus", "moduli", "moment", "parameter", "point", "pressure", "radius",
        "rate", "ratio", "spacing", "strain", "stress", "temperature", "tension",
        "transition", "velocity", "viscosity", "volume",
    }
)

_STOPWORDS = frozenset(
    {
        "about", "above", "after", "against", "and", "any", "are", "around", "as",
        "at", "below", "between", "both", "but", "by", "calculate", "compute",
        "determine", "each", "estimate", "evaluate", "find", "for", "from", "goal",
        "has", "have", "how", "into", "its", "measure", "not", "obtain", "of", "on",
        "one", "or", "per", "predict", "report", "result", "results", "run", "runs",
        "same", "should", "than", "that", "the", "their", "then", "this", "through",
        "use", "using", "value", "values", "via", "what", "when", "which", "with",
        "within",
    }
)


def _tokens(text: str) -> list[str]:
    """Content words in order: lowercase, stopwords and one- and two-letter words out.

    A snake_case identifier (a result key) stays one token, whatever its length.

    Examples:
        >>> _tokens("Estimate the melting point of W; report `t_melt` in K")
        ['melting', 'point', 't_melt']
    """
    words = re.findall(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in _STOPWORDS and (len(w) >= 3 or "_" in w)]


def plan_goal(plan: str) -> str:
    """The text of the plan's Goal: the line labelled Goal and the lines under it.

    The label may be a heading (``## Goal``), a bold label, or a plain
    ``Goal:``. The goal runs to the next blank line, heading, or list item
    after its first line. Empty when the plan has no Goal.

    Examples:
        >>> plan_goal("# Plan\\n\\n## Goal\\nMelting point of W.\\n\\n## Steps\\n1. probe")
        'Melting point of W.'
        >>> plan_goal("**Goal:** lattice constant of Cu\\n1. relax")
        'lattice constant of Cu'
        >>> plan_goal("1. relax Cu")
        ''
    """
    lines = plan.splitlines()
    for index, line in enumerate(lines):
        label = re.match(r"^\s*(?:#+\s*)?[-*\s]*goal\b[*\s]*:?[*\s]*(.*)$", line, re.I)
        if label is None:
            continue
        found = [label.group(1).strip()] if label.group(1).strip() else []
        for following in lines[index + 1 :]:
            if following.lstrip().startswith("#"):
                break
            if found and (not following.strip() or re.match(r"\s*(\d+[.)]|[-*])\s", following)):
                break
            if following.strip():
                found.append(following.strip())
        return "\n".join(found)
    return ""


@dataclass(frozen=True)
class PriorMatch:
    """A measured notebook line that names what the Goal names."""

    entry: NotebookEntry
    line: str
    phrase: str  # the quantity phrase or the shared word
    strong: bool  # True: a quantity phrase or a result key; False: one shared word

    def cite(self) -> str:
        line = self.line if len(self.line) <= 160 else self.line[:157] + "..."
        return f"{self.entry.heading}: {line!r}"


def prior_matches(goal: str, entries: list[NotebookEntry]) -> list[PriorMatch]:
    """Every notebook entry with a measured line that names a quantity of *goal*.

    One match per entry, strong first within it: a two-word quantity phrase
    of the goal ("melting point") or a result key (``t_melt``) repeated on
    a line that carries a number with a unit is strong; a shared content
    word of five letters or more is weak.

    Examples:
        >>> from foundation.project import NotebookEntry as E
        >>> probe = E("2026-09-12 10:31 UTC — probe", "Melting point near 1180 K.", 0)
        >>> [(m.phrase, m.strong) for m in prior_matches("melting point of W", [probe])]
        [('melting point', True)]
        >>> ramp = E("2026-09-12 11:00 UTC — ramp", "melting starts at 1300 K", 9)
        >>> [(m.phrase, m.strong) for m in prior_matches("melting point of W", [ramp])]
        [('melting', False)]
        >>> prior_matches("melting point of W", [E("x", "melting point unknown", 0)])
        []
    """
    words = _tokens(goal)
    keys = {w for w in words if "_" in w}
    phrases = {
        (first, second)
        for first, second in pairwise(words)
        if second in _QUANTITY_HEADS
    }
    distinctive = {w for w in words if len(w) >= 5 and "_" not in w}
    found: list[PriorMatch] = []
    for entry in entries:
        best: PriorMatch | None = None
        for line in entry.body.splitlines():
            if not _MEASURED.search(line):
                continue
            tokens = _tokens(line)
            pairs = set(pairwise(tokens))
            strong = sorted(" ".join(pair) for pair in phrases & pairs) + sorted(keys & set(tokens))
            if strong:
                best = PriorMatch(entry, line.strip(), strong[0], True)
                break
            shared = sorted(distinctive & set(tokens))
            if shared and best is None:
                best = PriorMatch(entry, line.strip(), shared[0], False)
        if best is not None:
            found.append(best)
    return found


def prior_result_check(plan: str, entries: list[NotebookEntry]) -> tuple[str | None, str]:
    """The refusal and the warning for a plan that ignores a prior result.

    Returns ``(refusal, warning)``: a refusal when the Goal names a
    quantity a measured notebook line reports by phrase or result key, a
    warning when the only match is a shared word, both empty when the
    plan carries a ``prior result:`` line or nothing matches.

    Examples:
        >>> from foundation.project import NotebookEntry as E
        >>> notes = [E("2026-09-12 10:31 UTC — probe", "melting point near 1180 K", 0)]
        >>> refusal, _ = prior_result_check("Goal: melting point of W", notes)
        >>> refusal.startswith("plan not written: the Goal names 'melting point'")
        True
        >>> prior_result_check("Goal: melting point of W\\nprior result: 1180 K", notes)
        (None, '')
    """
    if PRIOR_RESULT_LINE.search(plan):
        return None, ""
    goal = plan_goal(plan)
    if not goal:
        return None, ""
    matches = prior_matches(goal, entries)
    strong = [m for m in matches if m.strong]
    if strong:
        shown = "; ".join(m.cite() for m in strong[-3:])
        return (
            f"plan not written: the Goal names {strong[-1].phrase!r}, which the notebook "
            f"already reports ({shown}). Add a line 'prior result: <value with unit, run "
            f"id, date>' that states the earlier result and how this plan uses it (start "
            f"from it, check it, or say why it does not apply), then call plan again."
        ), ""
    if matches:
        shown = "; ".join(f"{m.phrase!r} in {m.cite()}" for m in matches[-3:])
        return None, (
            f"\n[note] the notebook reports a quantity that shares a word with the Goal "
            f"({shown}). If it is the same quantity, add a line 'prior result: ...' that "
            f"states it and how the plan uses it."
        )
    return None, ""


# -- artifact references ----------------------------------------------------------

#: ``run:<id>/<name>``: a run id or unique prefix, then the artifact's name.
ARTIFACT_REF = re.compile(r"\brun:([0-9a-zA-Z]{4,26})/([\w.+-]*[\w+-])")


def resolve_artifact_refs(workspace_root: Path, text: str) -> tuple[str, list[str], list[str]]:
    """Check every ``run:<id>/<name>`` in *text* against the run store.

    Returns ``(text, notes, errors)``. A reference whose run keeps the
    artifact stands as written. A reference to a run whose task was a
    cache hit is rewritten to the run that produced the file
    (:func:`foundation._ops.artifact_holder`), and *notes* says so. A
    reference that no run in the store answers lands in *errors*. A store
    that cannot be opened checks nothing, and *notes* says that instead.
    """
    import sqlite3

    from foundation._ops import artifact_holder
    from foundation.errors import AmbiguousRunIdError, FoundationError, RunNotFoundError
    from foundation.runtime import Workspace

    refs = list(dict.fromkeys(ARTIFACT_REF.findall(text)))
    if not refs:
        return text, [], []
    notes: list[str] = []
    errors: list[str] = []
    moved: dict[str, str] = {}
    try:
        with Workspace(workspace_root) as ws:
            for run_id, name in refs:
                old = f"run:{run_id}/{name}"
                try:
                    named = ws.runs.resolve(run_id)
                except (RunNotFoundError, AmbiguousRunIdError) as e:
                    errors.append(f"{old}: {e}")
                    continue
                holder = artifact_holder(ws, named, name)
                if holder is None:
                    kept = ", ".join(ref.name for ref in ws.runs.list_artifacts(named)) or "none"
                    errors.append(
                        f"{old}: run {named[:10]} keeps no artifact named {name!r}, and "
                        f"no run its cache hits came from does; it keeps: {kept}"
                    )
                elif holder != named:
                    moved[old] = f"run:{holder}/{name}"
                    notes.append(
                        f"{old} rewritten to {moved[old]}: run {named[:10]} was a cache "
                        f"hit, and the file is on the run that produced it"
                    )
    except (FoundationError, sqlite3.Error, OSError) as e:
        note = f"the artifact references were not checked: the run store could not be opened ({e})"
        return text, [note], []
    rewritten = ARTIFACT_REF.sub(lambda m: moved.get(m.group(0), m.group(0)), text)
    return rewritten, notes, errors
