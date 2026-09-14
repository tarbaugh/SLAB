"""Machine memory: what one session learns, every later session starts knowing.

A memory is one fact about *this machine*: a package that behaves unlike its
documentation, an engine flag that turns out to matter, a workaround that
took an afternoon to find. An agent that discovers such a fact records it
once; every session afterwards, in any project on the machine, reads it in
the system prompt.

Memory is the fourth knowledge surface, and each has one job:

* the curated software notes (``mason.notes``) — machine-scoped, written by
  humans and shipped with the package
* ``NOTEBOOK.md`` — project-scoped, the scientific record
* runs and artifacts — project-scoped, results with provenance
* memory (this module) — machine-scoped, written by agents, learned quirks

The store lives in Foundation because three consumers need it and they sit
in different layers: the agent (prompt and tools), the ``slab`` CLI
(human management), and, later, the MCP server. Foundation is the one layer
all three may import.

Layout is a directory of markdown files, one per memory, at
``~/.config/slab/memory/`` (``$XDG_CONFIG_HOME`` honored, ``$SLAB_MEMORY_DIR``
overriding both). The format follows the Agent Skills frontmatter shape that
``foundation.skills`` already teaches, minus the parts a single file does not
need::

    ---
    description: One line stating the fact and when it applies.
    created: 2026-08-28
    updated: 2026-08-28
    agent: pi
    model: qwen3-30b
    evidence: run 01k2x7... completed with the flag set
    against:
      gracemaker: 0.6.0
    ---
    The body: the fact itself, in full.

``against`` is the version stamp: the software the memory names, at the
versions present when it was written. A later session compares the stamp
with the machine it runs on and flags the memories whose software changed,
so the agent re-checks those and trusts the rest without probing.

``evidence`` is what confirmed the fact: a run id, a dry run, or a failure
record. A write without it is refused unless the writer says the fact is
unverified, which stamps ``unverified: true``. A memory with no evidence
reads as unverified whatever its frontmatter says, so a file written before
this rule, or by hand, is flagged for review rather than trusted.

Replacing a memory keeps the replaced file under ``.history/<name>/``, so a
contradiction is visible: recall shows the previous body beside the new
one when they differ, and a wrong replacement can be undone.

There is no index file. The catalog is a directory scan, which stays cheap at
the enforced cap and, unlike an index, never becomes a write-contention point
between concurrent jobs.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from foundation.errors import MemoryStoreError
from slab.config import user_config_path

#: The description is the recall trigger every session reads, so it is capped
#: like a skill's. The body is capped an order of magnitude below a skill's
#: 500-line budget: a memory is a fact, not a procedure.
MAX_DESCRIPTION_CHARS = 1_024
MAX_BODY_CHARS = 4_000

#: How many memories one machine may hold. The catalog enters every system
#: prompt, so it is a context budget before it is a storage budget; an agent
#: that hits the cap is told to consolidate.
MAX_MEMORIES = 100

#: What a piece of evidence may hold: a run id with a line of context, or a
#: failure record's first lines. Longer evidence belongs in the run it cites.
MAX_EVIDENCE_CHARS = 500

#: How many replaced versions one memory keeps under ``.history/<name>/``.
#: The oldest goes first; the cap bounds an agent that rewrites in a loop.
MAX_VERSIONS = 10

#: The directory under the memory root that holds replaced versions. It
#: starts with a dot, so the catalog scan never reads it as a memory.
HISTORY_DIR = ".history"

#: What a refused write says: the rule and the two ways past it.
EVIDENCE_REQUIRED = (
    "a memory needs evidence: the run id, the dry run, or the failure record that "
    "confirmed the fact. Pass evidence, or pass unverified=true to record it as an "
    "unverified claim that recall flags and 'slab memory review' lists"
)

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class _PlainDumper(yaml.SafeDumper):
    """A dumper that never writes anchors.

    ``created`` and ``updated`` hold the same date on the day a memory is
    written, and PyYAML would collapse the second into an alias
    (``updated: *id001``). It parses back correctly and reads as line noise
    to the person editing the file, who might then delete the anchor and
    leave a dangling alias behind.
    """

    def ignore_aliases(self, data: Any) -> bool:
        return True


@dataclass(frozen=True)
class Memory:
    """One recorded fact: its trigger line, its home, and who wrote it."""

    name: str
    description: str
    path: Path
    created: str | None = None
    updated: str | None = None
    agent: str | None = None
    model: str | None = None
    #: The version stamp: software the memory names, at the versions present
    #: when it was written. Empty for a memory nobody stamped.
    against: dict[str, str] = field(default_factory=dict)
    #: What confirmed the fact, as the writer gave it. None when nobody said.
    evidence: str | None = None
    #: Whether the fact is an unconfirmed claim: the writer said so, or no
    #: evidence was recorded.
    unverified: bool = True
    #: Set by :func:`write` only: the history file holding the version this
    #: write replaced, or None when the write created the memory.
    replaced: Path | None = None

    def body(self) -> str:
        """The fact itself: everything in the file after the frontmatter."""
        _, text = split_frontmatter(self.path.read_text(encoding="utf-8"))
        return text

    def provenance(self) -> str:
        """One line naming who recorded this memory, when, and on what evidence.

        Examples:
            >>> m = Memory("x", "d", Path("x.md"), created="2026-08-28", agent="pi")
            >>> m.provenance()
            'recorded by pi on 2026-08-28, no evidence recorded'
            >>> Memory("x", "d", Path("x.md"), evidence="run 01abc", unverified=False).provenance()
            'evidence: run 01abc'
        """
        parts = []
        if self.agent and self.created:
            parts.append(f"recorded by {self.agent} on {self.created}")
        elif self.agent:
            parts.append(f"recorded by {self.agent}")
        elif self.created:
            parts.append(f"recorded {self.created}")
        if self.updated and self.updated != self.created:
            parts.append(f"updated {self.updated}")
        if self.model:
            parts.append(f"model {self.model}")
        if self.against:
            stamped = ", ".join(f"{name} {version}" for name, version in self.against.items())
            parts.append(f"against {stamped}")
        parts.append(f"evidence: {self.evidence}" if self.evidence else "no evidence recorded")
        return ", ".join(parts)

    def drift(self, live: Mapping[str, str]) -> list[str]:
        """What changed since the stamp: one phrase per software that differs.

        Compares the stamp with *live*, the versions present now. Software
        the stamp names but *live* lacks reads as "not found now": the
        conservative reading, since a probe that failed and a tool that was
        removed look the same, and either is reason to re-check the fact.
        An unstamped memory never drifts, because it makes no claim.

        Examples:
            >>> m = Memory("x", "d", Path("x.md"), against={"gracemaker": "0.5.2"})
            >>> m.drift({"gracemaker": "0.6.0", "atomsk": "0.13.1"})
            ['gracemaker was 0.5.2, now 0.6.0']
            >>> m.drift({"gracemaker": "0.5.2"})
            []
            >>> m.drift({})
            ['gracemaker was 0.5.2, not found now']
        """
        changed = []
        for name, was in self.against.items():
            now = live.get(name)
            if now is None:
                changed.append(f"{name} was {was}, not found now")
            elif now != was:
                changed.append(f"{name} was {was}, now {now}")
        return changed


def valid_name(name: str) -> bool:
    """Whether a name may address a memory.

    The rule is the Agent Skills naming rule, which memories borrow so that
    one convention covers every named thing an agent writes: lowercase
    alphanumerics and single hyphens, 1 to 64 characters, no hyphen at
    either end.

    Examples:
        >>> valid_name("vllm-mamba-cache")
        True
        >>> valid_name("Vllm_Mamba")
        False
        >>> valid_name("-quirk")
        False
    """
    return len(name) <= 64 and bool(_NAME.match(name))


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a memory file into (frontmatter mapping, body).

    A deliberate copy of :func:`foundation.skills.split_frontmatter`: Foundation
    sits below Mason and may not import it, and the two files answer to
    different rules below the frontmatter anyway.

    Examples:
        >>> meta, body = split_frontmatter("---\\ndescription: d\\n---\\nThe fact.\\n")
        >>> meta["description"], body
        ('d', 'The fact.\\n')
    """
    if not text.startswith("---\n"):
        raise MemoryStoreError("no YAML frontmatter (the file must start with '---')")
    closing = text.find("\n---\n", 3)
    if closing < 0:
        raise MemoryStoreError("frontmatter never closes (no line with '---' after the first)")
    raw = text[4:closing]
    body = text[closing + len("\n---\n") :]
    try:
        meta = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise MemoryStoreError(f"frontmatter is not valid YAML: {e}") from e
    if not isinstance(meta, dict):
        raise MemoryStoreError("frontmatter must be a YAML mapping")
    return meta, body.lstrip("\n")


def memory_dir() -> Path:
    """Where memories live: ``$SLAB_MEMORY_DIR``, else the user config root.

    Never created here. The directory comes into being on the first write,
    so a machine whose agents have learned nothing yet has no stray empty
    directory to explain.
    """
    override = os.environ.get("SLAB_MEMORY_DIR")
    if override:
        return Path(override).expanduser()
    return user_config_path().parent / "memory"


def _provenance(meta: dict[str, Any], key: str) -> str | None:
    """Read one optional provenance field as a string, or refuse loudly."""
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, date | datetime):
        return value.isoformat()[:10] if key in ("created", "updated") else value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise MemoryStoreError(f"frontmatter {key!r} must be a date or a non-empty string")


def _against(meta: dict[str, Any]) -> dict[str, str]:
    """Read the version stamp: a mapping of software name to version string."""
    value = meta.get("against")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MemoryStoreError("frontmatter 'against' must be a mapping of software to version")
    stamp: dict[str, str] = {}
    for name, version in value.items():
        if not isinstance(name, str) or not name.strip():
            raise MemoryStoreError("frontmatter 'against' keys must be software names")
        if version is None or isinstance(version, dict | list):
            raise MemoryStoreError(f"frontmatter 'against' {name!r} must be a version string")
        # A version a person typed unquoted may have loaded as a number
        # (2024, 1.5); keep the text, since only equality matters.
        stamp[name.strip()] = str(version).strip()
    return stamp


def _as_date(value: str) -> date | str:
    """A stored date back as a ``date``, or unchanged when a human wrote prose."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        return value


def parse_memory(path: Path) -> Memory:
    """Read and validate one memory file, or raise :class:`MemoryStoreError`."""
    name = path.stem
    try:
        if not valid_name(name):
            raise MemoryStoreError(
                f"file name {path.name!r} is not a valid memory name (lowercase "
                f"alphanumerics and single hyphens, not at the ends, then '.md')"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise MemoryStoreError(f"cannot read it: {e}") from e
        meta, body = split_frontmatter(text)
        declared = meta.get("name")
        if declared is not None and declared != name:
            raise MemoryStoreError(
                f"frontmatter name {declared!r} disagrees with the file name {name!r}; "
                f"the file name is the memory's name, so drop the key or rename the file"
            )
        description = meta.get("description")
        if description is None:
            raise MemoryStoreError("frontmatter is missing the required 'description' field")
        if not isinstance(description, str) or not description.strip():
            raise MemoryStoreError("frontmatter 'description' must be a non-empty string")
        if len(description) > MAX_DESCRIPTION_CHARS:
            raise MemoryStoreError(
                f"frontmatter 'description' exceeds {MAX_DESCRIPTION_CHARS} characters "
                f"({len(description)})"
            )
        if not body.strip():
            raise MemoryStoreError("the body is empty; a memory must state the fact it holds")
        flagged = meta.get("unverified", False)
        if not isinstance(flagged, bool):
            raise MemoryStoreError("frontmatter 'unverified' must be true or false")
        evidence = _provenance(meta, "evidence")
        return Memory(
            name=name,
            description=" ".join(description.split()),
            path=path.resolve(),
            created=_provenance(meta, "created"),
            updated=_provenance(meta, "updated"),
            agent=_provenance(meta, "agent"),
            model=_provenance(meta, "model"),
            against=_against(meta),
            evidence=evidence,
            unverified=flagged or evidence is None,
        )
    except MemoryStoreError as e:
        raise MemoryStoreError(f"{path}: {e}") from None


def discover(directory: Path | None = None) -> dict[str, Memory]:
    """Every memory on this machine, by name, in name order.

    A malformed file is a loud error, never a silent absence — the same
    doctrine skills follow. A memory that quietly vanished from the catalog
    would be undebuggable, and the agent would go on not knowing the fact it
    holds.
    """
    root = directory if directory is not None else memory_dir()
    if not root.is_dir():
        return {}
    found: dict[str, Memory] = {}
    for path in sorted(root.glob("*.md")):
        if path.name.startswith((".", "_")) or not path.is_file():
            continue
        memory = parse_memory(path)
        found[memory.name] = memory
    return found


def written_since(when: datetime, directory: Path | None = None) -> list[Memory]:
    """The memories whose file changed at or after *when*, newest first.

    The judgement is the file's modification time, which every write and
    every replacement sets, so the answer covers what a process wrote
    whether it created the memory or rewrote one. A naive *when* is read
    as UTC. A malformed file is a loud error, as for :func:`discover`.

    Examples:
        >>> import tempfile
        >>> from datetime import timedelta
        >>> root = Path(tempfile.mkdtemp())
        >>> before = datetime.now(UTC)
        >>> _ = write("qe-on-gpu", "pw.x needs -nk 1 here", "One pool.", unverified=True,
        ...           directory=root)
        >>> [m.name for m in written_since(before, root)]
        ['qe-on-gpu']
        >>> written_since(datetime.now(UTC) + timedelta(seconds=1), root)
        []
    """
    cutoff = when if when.tzinfo is not None else when.replace(tzinfo=UTC)
    found = [
        (datetime.fromtimestamp(memory.path.stat().st_mtime, tz=UTC), memory)
        for memory in discover(directory).values()
    ]
    return [memory for written, memory in sorted(found, reverse=True) if written >= cutoff]


def write(
    name: str,
    description: str,
    body: str,
    *,
    agent: str | None = None,
    model: str | None = None,
    against: Mapping[str, str] | None = None,
    evidence: str | None = None,
    unverified: bool = False,
    directory: Path | None = None,
) -> Memory:
    """Record a fact, creating the memory or replacing it whole.

    *evidence* is what confirmed the fact: a run id, a dry run, a failure
    record. Without it the write is refused unless *unverified* is true,
    which records the fact as a claim that recall flags and review lists.

    Replacing is how an agent consolidates: the ``created`` date survives,
    ``updated`` moves to today, and the writer's attribution is refreshed.
    The replaced file is kept under ``.history/<name>/`` (see
    :func:`versions`), and the returned memory names it in ``replaced``.
    *against* is the version stamp (see :func:`stamp`); it is written as
    given, so a replacement carries the stamp of its own writing and not
    the one it replaced.
    The write is atomic (a temporary file in the same directory, then
    ``os.replace``), so a concurrent reader sees either the old file or the
    new one, never a half-written one. Two writers racing on one name is
    last-writer-wins, and both versions stay in the history.
    """
    root = directory if directory is not None else memory_dir()
    cited = " ".join((evidence or "").split())
    if not cited and not unverified:
        raise MemoryStoreError(EVIDENCE_REQUIRED)
    if len(cited) > MAX_EVIDENCE_CHARS:
        raise MemoryStoreError(
            f"the evidence is {len(cited)} characters, over the {MAX_EVIDENCE_CHARS}-"
            f"character limit; cite the run id and one line, not the output itself"
        )
    if not valid_name(name):
        raise MemoryStoreError(
            f"{name!r} is not a valid memory name: use lowercase alphanumerics and "
            f"single hyphens, 1 to 64 characters, no hyphen at either end "
            f"(for example 'vllm-mamba-cache')"
        )
    collapsed = " ".join(description.split())
    if not collapsed:
        raise MemoryStoreError(
            "a memory needs a description: one line stating the fact and when it "
            "applies, since that line is what a later session reads first"
        )
    if len(collapsed) > MAX_DESCRIPTION_CHARS:
        raise MemoryStoreError(
            f"the description is {len(collapsed)} characters, over the "
            f"{MAX_DESCRIPTION_CHARS}-character limit; state the fact in one line and "
            f"put the detail in the body"
        )
    if not body.strip():
        raise MemoryStoreError("a memory needs a body: the fact itself, in full")
    if len(body) > MAX_BODY_CHARS:
        raise MemoryStoreError(
            f"the body is {len(body)} characters, over the {MAX_BODY_CHARS}-character "
            f"limit; split it into separate memories, or fold it into an existing one"
        )
    path = root / f"{name}.md"
    existing = discover(root)
    if name not in existing and len(existing) >= MAX_MEMORIES:
        raise MemoryStoreError(
            f"this machine already holds {len(existing)} memories, the limit "
            f"({MAX_MEMORIES}); update an existing memory instead, or ask the user to "
            f"prune with 'slab memory forget <name>'"
        )
    today = datetime.now(UTC).date()
    previous = existing.get(name)
    frontmatter: dict[str, Any] = {
        "description": collapsed,
        # Written as YAML dates, not strings, so the file a person opens in
        # an editor reads as 'created: 2026-08-28' rather than quoted text.
        "created": _as_date(previous.created) if previous and previous.created else today,
        "updated": today,
    }
    if agent:
        frontmatter["agent"] = agent
    if model:
        frontmatter["model"] = model
    if cited:
        frontmatter["evidence"] = cited
    if unverified:
        frontmatter["unverified"] = True
    if against:
        # Versions are written as strings whatever they look like, so a
        # stamp of "1.10" survives the round trip as text.
        frontmatter["against"] = {name: str(version) for name, version in sorted(against.items())}
    rendered = yaml.dump(
        frontmatter, Dumper=_PlainDumper, sort_keys=False, allow_unicode=True, width=88
    )
    text = f"---\n{rendered}---\n{body.strip()}\n"
    root.mkdir(parents=True, exist_ok=True)
    replaced = _archive(root, name) if previous is not None else None
    try:
        _put(path, text)
    except MemoryStoreError:
        # The replacement never landed, so the kept copy duplicates the
        # file that is still current.
        if replaced is not None:
            replaced.unlink(missing_ok=True)
        raise
    return replace(parse_memory(path), replaced=replaced)


def _put(path: Path, text: str) -> None:
    """Write *text* to *path* atomically: a staged file, then ``os.replace``."""
    descriptor, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staged, path)
    except OSError as e:
        Path(staged).unlink(missing_ok=True)
        raise MemoryStoreError(f"cannot write {path}: {e}") from e


def _history(root: Path, name: str) -> Path:
    return root / HISTORY_DIR / name


def _archive(root: Path, name: str) -> Path:
    """Copy the current file of *name* into its history, pruning past the cap.

    The copy is made before the replacement lands, so a reader never finds
    the memory missing. The file name is the UTC time to the microsecond,
    which sorts in write order.
    """
    current = root / f"{name}.md"
    history = _history(root, name)
    history.mkdir(parents=True, exist_ok=True)
    kept = history / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.md"
    try:
        _put(kept, current.read_text(encoding="utf-8"))
    except (OSError, MemoryStoreError) as e:
        for empty in (history, history.parent):
            with suppress(OSError):
                empty.rmdir()
        raise MemoryStoreError(f"cannot keep the replaced version of {name!r}: {e}") from e
    for old in sorted(history.glob("*.md"))[:-MAX_VERSIONS]:
        old.unlink(missing_ok=True)
    return kept


@dataclass(frozen=True)
class Version:
    """One replaced version of a memory, as its history file holds it."""

    path: Path
    description: str
    body: str
    updated: str | None
    evidence: str | None
    unverified: bool


def versions(name: str, directory: Path | None = None) -> list[Version]:
    """The replaced versions of *name*, newest first. Empty when never replaced.

    A history file that no longer parses is skipped: the history is a
    record for a reader, and one bad file must not hide the rest.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _ = write("qe-pools", "Use -nk 2.", "Two pools.", unverified=True, directory=root)
        >>> _ = write("qe-pools", "Use -nk 1.", "One pool.", evidence="run 01abc", directory=root)
        >>> [v.body for v in versions("qe-pools", root)]
        ['Two pools.']
    """
    root = directory if directory is not None else memory_dir()
    if not valid_name(name):
        return []
    found = []
    for path in sorted(_history(root, name).glob("*.md"), reverse=True):
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
            evidence = _provenance(meta, "evidence")
            found.append(
                Version(
                    path=path,
                    description=" ".join(str(meta.get("description", "")).split()),
                    body=body.strip(),
                    updated=_provenance(meta, "updated"),
                    evidence=evidence,
                    unverified=meta.get("unverified") is True or evidence is None,
                )
            )
        except (OSError, MemoryStoreError):
            continue
    return found


def restore(name: str, version: Path, directory: Path | None = None) -> Memory:
    """Put an earlier version of *name* back, keeping the one it displaces.

    This undoes a replacement: *version* is a file from :func:`versions`
    (or a write's ``replaced``). The current file goes into the history
    first, so nothing is lost, and the restored text lands atomically.
    """
    root = directory if directory is not None else memory_dir()
    if version.parent.resolve() != _history(root, name).resolve() or not version.is_file():
        raise MemoryStoreError(f"{version} is not a kept version of the memory {name!r}")
    text = version.read_text(encoding="utf-8")
    if (root / f"{name}.md").is_file():
        _archive(root, name)
    _put(root / f"{name}.md", text)
    return parse_memory(root / f"{name}.md")


def delete(name: str, directory: Path | None = None) -> Path:
    """Forget one memory and its kept versions, returning the file removed.

    The deletion surface for a person. An agent may only undo a write its
    own session made (``forget`` in Mason), so nothing an agent does can
    erase a fact a person still wants.
    """
    root = directory if directory is not None else memory_dir()
    path = root / f"{name}.md"
    if not valid_name(name) or not path.is_file():
        known = ", ".join(discover(root)) or "none"
        raise MemoryStoreError(f"no memory named {name!r} (memories here: {known})")
    path.unlink()
    shutil.rmtree(_history(root, name), ignore_errors=True)
    return path


#: A run id, whole or as a prefix of at least eight characters: Crockford
#: base32, lowercase, and starting with the ULID's leading zero.
_RUN_ID = re.compile(r"(?<![A-Za-z0-9_])0[0-9a-hjkmnp-tv-z]{7,25}(?![A-Za-z0-9_])")


def run_ids(evidence: str | None) -> list[str]:
    """The run ids (or id prefixes) an evidence line cites, in order, once each.

    A token counts when it looks like a run id and holds a letter, so a
    number such as a step count is not mistaken for one.

    Examples:
        >>> run_ids("run 01k2x7abcd failed, then 01k2x7abcd again; 20000000 steps")
        ['01k2x7abcd']
        >>> run_ids(None)
        []
    """
    found: list[str] = []
    for token in _RUN_ID.findall(evidence or ""):
        if any(c.isalpha() for c in token) and token not in found:
            found.append(token)
    return found


def needs_review(
    memories: Mapping[str, Memory], live: Mapping[str, str]
) -> list[tuple[Memory, list[str]]]:
    """The memories a person should confirm or forget, each with its reasons.

    A memory needs review when it is unverified or when software it is
    stamped against has changed since it was written. *live* is the
    software present now. Name order.

    Examples:
        >>> checked = Memory("a", "d", Path("a.md"), evidence="run 01abc", unverified=False)
        >>> claim = Memory("b", "d", Path("b.md"))
        >>> old = Memory("c", "d", Path("c.md"), evidence="run 01def", unverified=False,
        ...              against={"lammps": "2Aug2023"})
        >>> [(m.name, why) for m, why in needs_review(
        ...     {"a": checked, "b": claim, "c": old}, {"lammps": "22Jul2025"})]
        [('b', ['unverified']), ('c', ['lammps was 2Aug2023, now 22Jul2025'])]
    """
    listed = []
    for _, memory in sorted(memories.items()):
        reasons = (["unverified"] if memory.unverified else []) + memory.drift(live)
        if reasons:
            listed.append((memory, reasons))
    return listed


#: Other names a memory may use for stamped software. The key is the name
#: :func:`slab._ops.software_versions` reports; the values are matched as
#: whole words, case-insensitively, like the key itself.
SOFTWARE_ALIASES: dict[str, tuple[str, ...]] = {
    "qe": ("pw.x", "quantum espresso", "espresso"),
    "lammps": ("lmp",),
    "gracemaker": ("tensorpotential", "grace"),
    "slab-stack": ("slab", "foundation", "mason"),
}


def _mentions(text: str, name: str) -> bool:
    words = (name, *SOFTWARE_ALIASES.get(name, ()))
    pattern = "|".join(re.escape(word) for word in words)
    return re.search(rf"(?<![A-Za-z0-9_])(?:{pattern})(?![A-Za-z0-9_])", text, re.I) is not None


#: What a remember reply adds when the fact describes SLAB itself.
ABOUT_SLAB_NOTE = (
    "; this describes SLAB itself, not the machine: it is stamped against slab-stack "
    "{version} and will be flagged stale on upgrade; if a skill is wrong, say so in "
    "your finish report so the skill gets fixed"
)

#: Names of SLAB's own surfaces. A memory that names one describes SLAB
#: itself, not the machine, and is worth flagging: it goes stale on an
#: upgrade, and a wrong skill is fixed in the skill, not remembered around.
SLAB_SURFACES: tuple[str, ...] = (
    "slab-stack",
    "slab_stack",
    "run_lammps",
    "launch_workflow",
    "series",
    "show_run",
    "read_artifact",
    "list_runs",
    "wait_for_run",
    "list_engines",
    "free_resources",
    "submit_job",
)


def about_slab(text: str, names: Iterable[str] = ()) -> bool:
    """Whether *text* describes SLAB itself: a result shape, a tool, the package.

    Matches ``result[`` or ``info[`` anywhere, and any of
    :data:`SLAB_SURFACES` or *names* (a toolbox's tool names) as a whole
    word.

    Examples:
        >>> about_slab("run_lammps returns thermo as the last row.")
        True
        >>> about_slab("The keys are result['tables'][0]['loop'].")
        True
        >>> about_slab("gracemaker needs TF_FORCE_GPU_ALLOW_GROWTH set.")
        False
        >>> about_slab("Call recall before shell.", names=("recall", "shell"))
        True
    """
    if "result[" in text or "info[" in text:
        return True
    words = [*SLAB_SURFACES, *names]
    pattern = "|".join(re.escape(word) for word in words)
    return re.search(rf"(?<![A-Za-z0-9_])(?:{pattern})(?![A-Za-z0-9_])", text) is not None


def stamp(text: str, live: Mapping[str, str]) -> dict[str, str]:
    """The version stamp for a memory: the software its text names, from *live*.

    *live* maps software names to the versions present now, as
    :func:`slab._ops.software_versions` reports them. A memory is stamped
    only with what it mentions, so a gracemaker upgrade flags the memories
    about gracemaker and leaves the one about vLLM alone. Names are matched
    as whole words, with the aliases in :data:`SOFTWARE_ALIASES`.

    Examples:
        >>> live = {"gracemaker": "0.6.0", "atomsk": "0.13.1", "slab-stack": "0.1.0"}
        >>> stamp("gracemaker needs TF_FORCE_GPU_ALLOW_GROWTH set.", live)
        {'gracemaker': '0.6.0'}
        >>> stamp("Set it in slab.toml before a GRACE fit.", live)
        {'gracemaker': '0.6.0', 'slab-stack': '0.1.0'}
        >>> stamp("vLLM refuses a big batch.", live)
        {}
    """
    return {name: live[name] for name in sorted(live) if _mentions(text, name)}


def catalog_block(memories: dict[str, Memory], live: Mapping[str, str] | None = None) -> str:
    """The ``# Memory`` section of the system prompt.

    One line per memory when the store is populated; a shorter form when it
    is empty, so a fresh machine still tells the agent this surface exists
    and when to write to it. The body stays on disk until ``recall``, so the
    always-loaded cost is the trigger lines — the same progressive
    disclosure the skill catalog uses.

    *live* is the software present now. A memory whose stamp differs from
    it gets a note on its line naming what changed, so the agent re-checks
    that memory and relies on the others without probing.

    Examples:
        >>> block = catalog_block(
        ...     {"vllm-cache": Memory("vllm-cache", "vLLM refuses a big batch.", Path("x"))}
        ... )
        >>> block.splitlines()[0]
        '# Memory'
        >>> block.splitlines()[-1]
        '- vllm-cache: vLLM refuses a big batch. [unverified]'
        >>> catalog_block({}).splitlines()[0]
        '# Memory'
        >>> stamped = Memory(
        ...     "grace-gpu", "gracemaker needs X.", Path("y"), against={"gracemaker": "0.5.2"},
        ...     evidence="run 01abc", unverified=False,
        ... )
        >>> catalog_block({"grace-gpu": stamped}, live={"gracemaker": "0.6.0"}).splitlines()[-1]
        '- grace-gpu: gracemaker needs X. [changed since: gracemaker was 0.5.2, now 0.6.0]'
    """
    listed = [memory for _, memory in sorted(memories.items())]
    if not listed:
        return "\n".join(
            [
                "# Memory",
                "",
                "No machine facts recorded on this machine yet. When you find a "
                "quirk of this machine or its software worth keeping — a package "
                "flag, a workaround, a path that surprised you — call the "
                "remember tool with a name, a one-line description, the "
                "detail, and the evidence that confirmed it (a run id, a dry "
                "run, a failure record). Machine facts only: results belong in "
                "runs, project decisions in the notebook, and credentials nowhere.",
            ]
        )
    lines = [
        "# Memory",
        "",
        "Facts earlier sessions recorded about this machine and its software. "
        "Call the recall tool with a name before you rely on one: the line "
        "below is a summary, and the memory itself holds the detail. Each "
        "memory is stamped with the versions of the software it names. A "
        "line that reports a change since, a newer version or a tool not "
        "found now, is a memory you must confirm before you build on it. A "
        "line that reports none names software that is unchanged, so rely "
        "on that memory without probing. A line marked [unverified] is a "
        "claim no run confirmed; test it before you rely on it. When you find "
        "a quirk of this machine or its software worth keeping, record it "
        "with remember once a run has confirmed it, and cite that run as the "
        "evidence. Machine facts only: results belong in runs, project "
        "decisions in the notebook, and credentials nowhere.",
        "",
    ]
    for m in listed:
        changed = m.drift(live) if live is not None else []
        note = " [unverified]" if m.unverified else ""
        note += f" [changed since: {'; '.join(changed)}]" if changed else ""
        lines.append(f"- {m.name}: {m.description}{note}")
    return "\n".join(lines)
