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
``mason.skills`` already teaches, minus the parts a single file does not
need::

    ---
    description: One line stating the fact and when it applies.
    created: 2026-08-28
    updated: 2026-08-28
    agent: pi
    model: qwen3-30b
    against:
      gracemaker: 0.6.0
    ---
    The body: the fact itself, in full.

``against`` is the version stamp: the software the memory names, at the
versions present when it was written. A later session compares the stamp
with the machine it runs on and flags the memories whose software changed,
so the agent re-checks those and trusts the rest without probing.

There is no index file. The catalog is a directory scan, which stays cheap at
the enforced cap and, unlike an index, never becomes a write-contention point
between concurrent jobs.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
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

    def body(self) -> str:
        """The fact itself: everything in the file after the frontmatter."""
        _, text = split_frontmatter(self.path.read_text(encoding="utf-8"))
        return text

    def provenance(self) -> str:
        """One line naming who recorded this memory and when.

        Examples:
            >>> m = Memory("x", "d", Path("x.md"), created="2026-08-28", agent="pi")
            >>> m.provenance()
            'recorded by pi on 2026-08-28'
            >>> Memory("x", "d", Path("x.md")).provenance()
            'no provenance recorded'
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
        return ", ".join(parts) if parts else "no provenance recorded"

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

    A deliberate copy of :func:`mason.skills.split_frontmatter`: Foundation
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
        return Memory(
            name=name,
            description=" ".join(description.split()),
            path=path.resolve(),
            created=_provenance(meta, "created"),
            updated=_provenance(meta, "updated"),
            agent=_provenance(meta, "agent"),
            model=_provenance(meta, "model"),
            against=_against(meta),
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


def write(
    name: str,
    description: str,
    body: str,
    *,
    agent: str | None = None,
    model: str | None = None,
    against: Mapping[str, str] | None = None,
    directory: Path | None = None,
) -> Memory:
    """Record a fact, creating the memory or replacing it whole.

    Replacing is how an agent consolidates: the ``created`` date survives,
    ``updated`` moves to today, and the writer's attribution is refreshed.
    *against* is the version stamp (see :func:`stamp`); it is written as
    given, so a replacement carries the stamp of its own writing and not
    the one it replaced.
    The write is atomic (a temporary file in the same directory, then
    ``os.replace``), so a concurrent reader sees either the old file or the
    new one, never a half-written one. Two writers racing on one name is
    last-writer-wins.
    """
    root = directory if directory is not None else memory_dir()
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
    if against:
        # Versions are written as strings whatever they look like, so a
        # stamp of "1.10" survives the round trip as text.
        frontmatter["against"] = {name: str(version) for name, version in sorted(against.items())}
    rendered = yaml.dump(
        frontmatter, Dumper=_PlainDumper, sort_keys=False, allow_unicode=True, width=88
    )
    text = f"---\n{rendered}---\n{body.strip()}\n"
    root.mkdir(parents=True, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(dir=root, prefix=f".{name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(staged, path)
    except OSError as e:
        Path(staged).unlink(missing_ok=True)
        raise MemoryStoreError(f"cannot write {path}: {e}") from e
    return parse_memory(path)


def delete(name: str, directory: Path | None = None) -> Path:
    """Forget one memory, returning the file that was removed.

    The only deletion surface, and it belongs to the human: agents
    consolidate by rewriting, so nothing an agent does can erase a fact a
    person still wants.
    """
    root = directory if directory is not None else memory_dir()
    path = root / f"{name}.md"
    if not valid_name(name) or not path.is_file():
        known = ", ".join(discover(root)) or "none"
        raise MemoryStoreError(f"no memory named {name!r} (memories here: {known})")
    path.unlink()
    return path


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
        '- vllm-cache: vLLM refuses a big batch.'
        >>> catalog_block({}).splitlines()[0]
        '# Memory'
        >>> stamped = Memory(
        ...     "grace-gpu", "gracemaker needs X.", Path("y"), against={"gracemaker": "0.5.2"}
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
                "remember tool with a name, a one-line description, and the "
                "detail. Machine facts only: results belong in runs, project "
                "decisions in the notebook, and credentials nowhere.",
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
        "on that memory without probing. When you find a quirk of this "
        "machine or its software worth keeping, record it with remember once "
        "you have confirmed it. Machine facts only: results belong in runs, "
        "project decisions in the notebook, and credentials nowhere.",
        "",
    ]
    for m in listed:
        changed = m.drift(live) if live is not None else []
        note = f" [changed since: {'; '.join(changed)}]" if changed else ""
        lines.append(f"- {m.name}: {m.description}{note}")
    return "\n".join(lines)
