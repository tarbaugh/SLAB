"""Mason's view of the skill catalog: what one card sees, and how a prompt lists it.

The format, the three discovery layers, and the built-in skills live in
:mod:`foundation.skills`, where an external harness over MCP loads the
same catalog. Mason adds the per-card categorization (the ``mason-agents``
metadata key names which cards see a skill) and the prompt rendering, and
re-exports the rest so the roster and the tools keep one import path.
"""

from __future__ import annotations

from foundation.skills import (
    Skill,
    SkillError,
    Source,
    bundled_files,
    discover_skills,
    listing,
    parse_skill,
    skill_digest,
    split_frontmatter,
    valid_name,
)

__all__ = [
    "Skill",
    "SkillError",
    "Source",
    "bundled_files",
    "catalog_block",
    "discover_skills",
    "listing",
    "parse_skill",
    "skill_digest",
    "split_frontmatter",
    "valid_name",
    "visible_catalog",
]


def visible_catalog(
    skills: dict[str, Skill], agent_name: str, scope: str = "matching"
) -> dict[str, Skill]:
    """The slice of the catalog one agent sees.

    ``scope="all"`` is the full catalog (the PI's view, so solo mode loses
    nothing to categorization); ``"matching"`` keeps the skills whose
    ``mason-agents`` include *agent_name*, plus the unrestricted ones.
    """
    if scope == "all":
        return dict(skills)
    return {name: skill for name, skill in skills.items() if skill.visible_to(agent_name)}


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
