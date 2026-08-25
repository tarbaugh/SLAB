"""Agent cards: the roster that turns Mason into a research group.

An agent card is one markdown file. The frontmatter declares what the agent
is; the body is the role section of its system prompt. The shared harness
discipline lives in :mod:`mason.prompts` and is identical for every card,
so a card stays short: identity and domain doctrine only. ::

    ---
    name: dft-expert
    description: Plans and runs DFT calculations. Delegate anything that
      needs cutoffs, k-meshes, or SCF diagnosis.
    tools: read_file shell skill launch_workflow notebook finish
    skills: matching
    delegates: false
    ---
    You are the DFT specialist of a SLAB research group...

Fields beyond ``name`` and ``description`` are optional. ``tools`` is a
space-separated allowlist validated against the universal tool vocabulary,
so a typo is refused even when the tool it misspells is absent from the
current session. ``skills`` is ``matching`` (the skills whose
``mason-agents`` include this card, plus the unrestricted ones) or ``all``
(the full catalog; the PI uses this so solo mode loses nothing).
``delegates`` grants the ``delegate`` tool — only ever at delegation depth
zero, so no combination of cards can recurse.

Cards are discovered like skills, and a name in a higher layer shadows the
lower ones whole: project ``<cwd>/agents/*.md``, then user
``~/.config/slab/agents/*.md``, then the built-ins shipped in the package
(``pi``, ``dft-expert``, ``md-expert``, ``analysis-expert``). A project
card named ``pi.md`` therefore replaces the default agent entirely.

Cards are portable content and never name models. Machine facts — which
model, which endpoint, what budgets — live in config as
``[agent.roster.<name>]`` tables (:mod:`mason.config`).
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mason.config import AgentConfig
from mason.errors import MasonError
from mason.skills import Skill, SkillError, split_frontmatter, valid_name, visible_catalog
from mason.tools import TOOL_VOCABULARY
from slab.config import user_config_path

Source = Literal["built-in", "user", "project"]

_KNOWN_KEYS = frozenset({"name", "description", "tools", "skills", "delegates"})


class RosterError(MasonError):
    """An agent card that cannot be used, naming the file and the rule."""


@dataclass(frozen=True)
class AgentSpec:
    """One agent card, parsed: who it is, what it may use, what it sees."""

    name: str
    description: str
    prompt: str  # the card body: the role section of the system prompt
    tools: frozenset[str] | None  # None = every tool the session offers
    skills_scope: Literal["matching", "all"]
    delegates: bool
    source: Source
    path: Path


def _required_string(meta: dict[str, Any], key: str, limit: int) -> str:
    value = meta.get(key)
    if value is None:
        raise RosterError(f"frontmatter is missing the required {key!r} field")
    if not isinstance(value, str) or not value.strip():
        raise RosterError(f"frontmatter {key!r} must be a non-empty string")
    if len(value) > limit:
        raise RosterError(f"frontmatter {key!r} exceeds {limit} characters ({len(value)})")
    return value


def _tools_allowlist(meta: dict[str, Any]) -> frozenset[str] | None:
    raw = meta.get("tools")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.split():
        raise RosterError(
            "frontmatter 'tools' must be a space-separated string of tool names; "
            "remove the key to allow every tool"
        )
    names = frozenset(raw.split())
    unknown = sorted(names - TOOL_VOCABULARY)
    if unknown:
        raise RosterError(
            f"frontmatter 'tools' names {', '.join(repr(n) for n in unknown)}, which "
            f"no session offers; the vocabulary: {', '.join(sorted(TOOL_VOCABULARY))}"
        )
    return names


def parse_agent_card(path: Path, source: Source) -> AgentSpec:
    """Read and validate one agent card, or raise a :class:`RosterError`."""
    try:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            # An unreadable card must surface as an error line, not a
            # traceback: the CLI catches RosterError, never a bare OSError.
            raise RosterError(f"cannot read it: {e}") from e
        try:
            meta, body = split_frontmatter(text)
        except SkillError as e:
            raise RosterError(str(e)) from None
        unknown = sorted(set(meta) - _KNOWN_KEYS)
        if unknown:
            raise RosterError(
                f"unknown frontmatter key(s) {', '.join(repr(k) for k in unknown)}; "
                f"an agent card takes: {', '.join(sorted(_KNOWN_KEYS))}"
            )
        name = _required_string(meta, "name", limit=64)
        if not valid_name(name):
            raise RosterError(
                f"name {name!r} breaks the naming rules (lowercase alphanumerics and "
                f"single hyphens, not at the ends)"
            )
        if name != path.stem:
            raise RosterError(f"name {name!r} must equal the file name {path.stem!r}")
        description = _required_string(meta, "description", limit=1024)
        scope = meta.get("skills", "matching")
        if scope not in ("matching", "all"):
            raise RosterError(f"frontmatter 'skills' must be 'matching' or 'all', not {scope!r}")
        delegates = meta.get("delegates", False)
        if not isinstance(delegates, bool):
            raise RosterError("frontmatter 'delegates' must be true or false")
        if not body.strip():
            raise RosterError("the card has no body; the body is the agent's role prompt")
        return AgentSpec(
            name=name,
            description=" ".join(description.split()),
            prompt=body.strip(),
            tools=_tools_allowlist(meta),
            skills_scope=scope,
            delegates=delegates,
            source=source,
            path=path.resolve(),
        )
    except RosterError as e:
        raise RosterError(f"{path}: {e}") from None


def _builtin_root() -> Path:
    return Path(str(importlib.resources.files("mason"))) / "agents"


def _user_root() -> Path:
    return user_config_path().parent / "agents"


def _layer(root: Path, source: Source) -> dict[str, AgentSpec]:
    if not root.is_dir():
        return {}
    found: dict[str, AgentSpec] = {}
    for path in sorted(root.glob("*.md")):
        # Well-known non-card files may sit beside cards; anything else that
        # fails to parse is a loud error, never a silent skip.
        if path.name.startswith((".", "_")) or path.name in ("AGENTS.md", "README.md"):
            continue
        spec = parse_agent_card(path, source)
        found[spec.name] = spec
    return found


def discover_roster(cwd: Path) -> dict[str, AgentSpec]:
    """Every agent card visible from *cwd*: built-in, then user, then project.

    A later layer's name shadows the earlier ones whole. The built-in layer
    guarantees ``pi`` exists (a project card may replace it, never remove
    it). A malformed card is a loud :class:`RosterError` naming the file.
    """
    roster: dict[str, AgentSpec] = {}
    roster.update(_layer(_builtin_root(), "built-in"))
    roster.update(_layer(_user_root(), "user"))
    roster.update(_layer(Path(cwd) / "agents", "project"))
    return roster


def check_overrides(agent: AgentConfig, roster: dict[str, AgentSpec]) -> None:
    """Refuse ``[agent.roster.<name>]`` tables that name no discovered card.

    Config that silently does nothing is a trap: a mistyped table would sit
    in the file looking effective. The refusal names the roster, the same
    philosophy as the moved-key refusal in the config loader.
    """
    unknown = sorted(set(agent.roster) - set(roster))
    if unknown:
        tables = ", ".join(f"[agent.roster.{name}]" for name in unknown)
        raise RosterError(
            f"{tables} name{'s' if len(unknown) == 1 else ''} no agent card; "
            f"the roster: {', '.join(sorted(roster))}"
        )


def skills_for(spec: AgentSpec, skills: dict[str, Skill]) -> dict[str, Skill]:
    """The slice of the skill catalog this agent sees in its prompt and tool."""
    return visible_catalog(skills, spec.name, spec.skills_scope)
