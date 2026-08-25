"""Skills: reusable procedure packages in the Agent Skills format.

A skill is a directory whose ``SKILL.md`` satisfies the Agent Skills
specification (https://agentskills.io/specification): YAML frontmatter with a
required ``name`` and ``description``, then a markdown body of instructions,
beside optional ``scripts/``, ``references/``, and ``assets/``. Mason adds
nothing to the format. Its one extension rides where the spec sends
extensions, the ``metadata`` map::

    metadata:
      mason-agents: "dft-expert analysis-expert"

names the agent cards that see the skill (space-separated, mirroring the
spec's own ``allowed-tools`` convention). Absent means every agent sees it.
Frontmatter keys this module does not know are ignored, so skills written
for other Agent Skills consumers load unmodified; the experimental
``allowed-tools`` field is accepted, ignored, and reported as ignored,
because the toolbox already gates approval per call.

Skills are discovered in three layers, and a name in a higher layer shadows
the lower ones whole:

1. project — ``<cwd>/skills/*/SKILL.md``
2. user — ``~/.config/slab/skills/*/SKILL.md`` (``$XDG_CONFIG_HOME``
   honored, matching the config loader)
3. built-in — shipped inside the ``mason`` package

Progressive disclosure follows the spec: only ``name: description`` lines
enter the system prompt; the body loads when the model calls the ``skill``
tool; bundled files load only when the model reads or runs them through the
ordinary primitives. A skill script therefore runs under exactly the
approval gate and allowlist that govern every other shell command — there
is no separate execution surface.
"""

from __future__ import annotations

import importlib.resources
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from mason.errors import MasonError
from slab.config import user_config_path

_MAX_LISTED_FILES = 50
_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

Source = Literal["built-in", "user", "project"]


class SkillError(MasonError):
    """A skill that cannot be used, naming the file and the violated rule."""


@dataclass(frozen=True)
class Skill:
    """One discovered skill: its metadata, its home, and where it came from."""

    name: str
    description: str
    root: Path
    source: Source
    agents: frozenset[str] | None  # None = visible to every agent
    compatibility: str | None
    license: str | None
    ignored_allowed_tools: bool

    def visible_to(self, agent_name: str) -> bool:
        """Whether the named agent card sees this skill in its catalog."""
        return self.agents is None or agent_name in self.agents

    def body(self) -> str:
        """The instructions: everything in ``SKILL.md`` after the frontmatter."""
        _, text = split_frontmatter((self.root / "SKILL.md").read_text(encoding="utf-8"))
        return text


def valid_name(name: str) -> bool:
    """Whether a name satisfies the spec's rules for skill and agent names.

    Lowercase alphanumerics and hyphens, 1-64 characters, no hyphen at
    either end, no consecutive hyphens.

    Examples:
        >>> valid_name("equation-of-state")
        True
        >>> valid_name("PDF-Processing")
        False
        >>> valid_name("-eos")
        False
        >>> valid_name("eos--fit")
        False
        >>> valid_name("x" * 65)
        False
    """
    return len(name) <= 64 and bool(_NAME.match(name))


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``SKILL.md`` text into (frontmatter mapping, body).

    Examples:
        >>> meta, body = split_frontmatter("---\\nname: eos\\n---\\nSteps.\\n")
        >>> meta["name"], body
        ('eos', 'Steps.\\n')
    """
    if not text.startswith("---\n"):
        raise SkillError("no YAML frontmatter (the file must start with '---')")
    closing = text.find("\n---\n", 3)
    if closing < 0:
        raise SkillError("frontmatter never closes (no line with '---' after the first)")
    raw = text[4:closing]
    body = text[closing + len("\n---\n") :]
    try:
        meta = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise SkillError(f"frontmatter is not valid YAML: {e}") from e
    if not isinstance(meta, dict):
        raise SkillError("frontmatter must be a YAML mapping")
    return meta, body.lstrip("\n")


def _string_field(meta: dict[str, Any], key: str, *, limit: int, required: bool) -> str | None:
    value = meta.get(key)
    if value is None:
        if required:
            raise SkillError(f"frontmatter is missing the required {key!r} field")
        return None
    if not isinstance(value, str) or not value.strip():
        raise SkillError(f"frontmatter {key!r} must be a non-empty string")
    if len(value) > limit:
        raise SkillError(f"frontmatter {key!r} exceeds {limit} characters ({len(value)})")
    return value


def _agents_filter(meta: dict[str, Any]) -> frozenset[str] | None:
    metadata = meta.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise SkillError("frontmatter 'metadata' must be a mapping of strings to strings")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SkillError(
                f"frontmatter 'metadata' values must be strings (metadata.{key} is not)"
            )
    raw = metadata.get("mason-agents")
    if raw is None:
        return None
    names = raw.split()
    if not names:
        raise SkillError("'mason-agents' names no agents; remove the key to mean every agent")
    for name in names:
        if not valid_name(name):
            raise SkillError(f"'mason-agents' entry {name!r} is not a valid agent name")
    return frozenset(names)


def parse_skill(skill_dir: Path, source: Source) -> Skill:
    """Read and validate one skill directory, or raise a :class:`SkillError`."""
    manifest = skill_dir / "SKILL.md"
    try:
        if not manifest.is_file():
            raise SkillError("the directory has no SKILL.md")
        meta, _ = split_frontmatter(manifest.read_text(encoding="utf-8"))
        name = _string_field(meta, "name", limit=64, required=True)
        assert name is not None  # required=True raised otherwise
        if not valid_name(name):
            raise SkillError(
                f"name {name!r} breaks the naming rules (lowercase alphanumerics and "
                f"single hyphens, not at the ends)"
            )
        if name != skill_dir.name:
            raise SkillError(f"name {name!r} must equal the directory name {skill_dir.name!r}")
        description = _string_field(meta, "description", limit=1024, required=True)
        assert description is not None
        return Skill(
            name=name,
            description=" ".join(description.split()),
            root=skill_dir.resolve(),
            source=source,
            agents=_agents_filter(meta),
            compatibility=_string_field(meta, "compatibility", limit=500, required=False),
            license=_string_field(meta, "license", limit=500, required=False),
            ignored_allowed_tools="allowed-tools" in meta,
        )
    except SkillError as e:
        raise SkillError(f"{manifest}: {e}") from None


def _builtin_root() -> Path:
    return Path(str(importlib.resources.files("mason"))) / "skills"


def _user_root() -> Path:
    return user_config_path().parent / "skills"


def _layer(root: Path, source: Source) -> dict[str, Skill]:
    if not root.is_dir():
        return {}
    found: dict[str, Skill] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        skill = parse_skill(child, source)
        found[skill.name] = skill
    return found


def discover_skills(cwd: Path) -> dict[str, Skill]:
    """Every skill visible from *cwd*: built-in, then user, then project.

    A later layer's name shadows the earlier ones whole. A directory in a
    skills root that fails validation is a loud :class:`SkillError`, never a
    silent absence — a skill that quietly vanishes from the catalog is
    undebuggable.
    """
    skills: dict[str, Skill] = {}
    skills.update(_layer(_builtin_root(), "built-in"))
    skills.update(_layer(_user_root(), "user"))
    skills.update(_layer(Path(cwd) / "skills", "project"))
    return skills


def catalog_block(skills: dict[str, Skill]) -> str:
    """The ``# Skills`` section of the system prompt, or empty when none apply.

    One line per skill — the spec's progressive disclosure keeps the always-
    loaded cost to the name and description. The caller passes the catalog
    already narrowed to the current agent.
    """
    visible = [skill for _, skill in sorted(skills.items())]
    if not visible:
        return ""
    lines = [
        "# Skills",
        "",
        "Procedure packages available in this workspace. When a task matches "
        "one, call the skill tool with its name before working, follow the "
        "instructions it returns, and prefer its bundled scripts over "
        "writing your own.",
        "",
    ]
    lines.extend(f"- {s.name}: {s.description}" for s in visible)
    return "\n".join(lines)


def listing(skill: Skill) -> str:
    """What the ``skill`` tool returns: the body, the root, the bundled files."""
    # Hidden-part filtering looks below the skill root only: the root itself
    # may live under a dotted or underscored parent and still deserves a
    # listing (the same rule the search tool applies).
    files = sorted(
        str(p.relative_to(skill.root))
        for p in skill.root.rglob("*")
        if p.is_file()
        and not any(part.startswith((".", "_")) for part in p.relative_to(skill.root).parts)
    )
    shown = files[:_MAX_LISTED_FILES]
    note = (
        f"\n[... {len(files) - len(shown)} more files not listed]"
        if len(files) > len(shown)
        else ""
    )
    inventory = "\n".join(f"  {name}" for name in shown)
    return (
        f"{skill.body().rstrip()}\n\n"
        f"skill root: {skill.root}\n"
        f"files (paths relative to the root):\n{inventory}{note}"
    )
