"""Curated software notes: what the agent should know about each engine.

The transcript of a session that lacks this context is easy to picture: the
model spends its first ten steps grepping the filesystem to rediscover what
``slab`` already knows — which engines exist, what they are for, and which
mistakes each one invites. These notes put that knowledge in the system
prompt up front, so the reasoning starts from the answer instead of the
search.

Selection follows the configuration. A note ships for each built-in engine;
the always-available ones (``emt``, ``lj``, ``mace``) load unconditionally,
and ``qe``, ``lammps``, and ``rootstock`` load when ``slab.toml`` configures
their ``[engines.<name>]`` table — configuring the table is what enables the
software on a machine, so it is also what enables its note. ``[agent]
software_notes = false`` turns the whole block off.

A user with local tweaks to their software can replace any note: a file at
``~/.config/slab/notes/<name>.md`` (``$XDG_CONFIG_HOME`` honored) wins over
the packaged note, whole-file. This is an escape hatch, not a content
system — the packaged notes are the curated surface, and ``list_engines``
stays the live truth about what actually exists.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from slab.config import EnginesConfig, user_config_path

#: Prompt order: the engines the agent reaches for first come first.
_CANONICAL = ("rootstock", "qe", "lammps", "emt", "lj")

#: Notes that load regardless of configuration. ``emt`` and ``lj`` are ASE
#: built-ins that need no table. ``rootstock`` is here because it is the only
#: route to an MLIP the agent has, and the note names the route a machine
#: without ``[engines.rootstock]`` must configure to reach one.
_ALWAYS = frozenset({"emt", "lj", "rootstock"})


def _builtin_root() -> Path:
    return Path(str(importlib.resources.files("mason"))) / "notes"


def _user_root() -> Path:
    return user_config_path().parent / "notes"


def enabled_notes(engines: EnginesConfig) -> tuple[str, ...]:
    """Which note names apply under this ``[engines]`` configuration.

    Examples:
        >>> enabled_notes(EnginesConfig())
        ('rootstock', 'emt', 'lj')
        >>> cfg = EnginesConfig.model_validate({"qe": {"command": "pw.x"}})
        >>> enabled_notes(cfg)
        ('rootstock', 'qe', 'emt', 'lj')
    """
    names = []
    for name in _CANONICAL:
        if name in _ALWAYS or getattr(engines, name) != type(getattr(engines, name))():
            names.append(name)
    return tuple(names)


def note_text(name: str) -> str:
    """One note's text — the user's override when present, else the packaged note."""
    override = _user_root() / f"{name}.md"
    if override.is_file():
        return override.read_text(encoding="utf-8").strip()
    return (_builtin_root() / f"{name}.md").read_text(encoding="utf-8").strip()


def notes_block(engines: EnginesConfig) -> str:
    """The ``# Software notes`` section of the system prompt.

    Stable for a given configuration, so a prefix-caching server reuses it
    across sessions on the same machine.
    """
    lines = [
        "# Software notes",
        "",
        "Curated notes on the computational software configured here. They are "
        "starting context, not a live inventory: call list_engines for what "
        "actually exists right now.",
    ]
    for name in enabled_notes(engines):
        lines.append(f"\n## {name}\n\n{note_text(name)}")
    return "\n".join(lines)
