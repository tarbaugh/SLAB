"""Curated software notes: what the agent should know about each resource.

The transcript of a session that lacks this context is easy to picture: the
model spends its first ten steps grepping the filesystem to rediscover what
``slab`` already knows — which engines exist, what they are for, and which
mistakes each one invites. These notes put that knowledge in the system
prompt up front, so the reasoning starts from the answer instead of the
search.

Selection follows the configuration. A note ships for each built-in engine,
for the ``mp`` data snapshot, and for the ``gracemaker`` trainer; the
always-available ones (``emt``, ``lj``, ``rootstock``) load unconditionally,
and ``qe``, ``lammps``, ``mp``, and ``gracemaker`` load when ``slab.toml``
configures their table (``[engines.<name>]``, or ``[builders.<name>]``) —
configuring the table is what enables the resource on a machine, so it is
also what enables its note. ``[agent] software_notes = false`` turns the
whole block off.

A user with local tweaks to their software can replace any note: a file at
``~/.config/slab/notes/<name>.md`` (``$XDG_CONFIG_HOME`` honored) wins over
the packaged note, whole-file. This is an escape hatch, not a content
system — the packaged notes are the curated surface, and ``list_engines``
stays the live truth about what actually exists.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

from slab.config import SlabConfig, user_config_path

#: Prompt order: the engines the agent reaches for first come first; the
#: mp data snapshot follows the engines because it feeds them structures,
#: and the gracemaker trainer comes last because training is the rarest move.
_CANONICAL = ("rootstock", "qe", "lammps", "emt", "lj", "mp", "gracemaker")

#: Notes that load regardless of configuration. ``emt`` and ``lj`` are ASE
#: built-ins that need no table. ``rootstock`` is here because it is the only
#: route to an MLIP the agent has, and the note names the route a machine
#: without ``[engines.rootstock]`` must configure to reach one.
_ALWAYS = frozenset({"emt", "lj", "rootstock"})


def _builtin_root() -> Path:
    return Path(str(importlib.resources.files("mason"))) / "notes"


def _user_root() -> Path:
    return user_config_path().parent / "notes"


def enabled_notes(config: SlabConfig) -> tuple[str, ...]:
    """Which note names apply under this configuration.

    Examples:
        >>> enabled_notes(SlabConfig())
        ('rootstock', 'emt', 'lj')
        >>> cfg = SlabConfig.model_validate({"engines": {"qe": {"command": "pw.x"}}})
        >>> enabled_notes(cfg)
        ('rootstock', 'qe', 'emt', 'lj')
        >>> cfg = SlabConfig.model_validate({"builders": {"mp": {"root": "/data/mp"}}})
        >>> enabled_notes(cfg)
        ('rootstock', 'emt', 'lj', 'mp')
        >>> cfg = SlabConfig.model_validate(
        ...     {"builders": {"gracemaker": {"command": "gracemaker"}}}
        ... )
        >>> enabled_notes(cfg)
        ('rootstock', 'emt', 'lj', 'gracemaker')
    """
    names = []
    for name in _CANONICAL:
        if name in _ALWAYS:  # emt/lj have no config table to look at
            names.append(name)
            continue
        sub = _sub_model(config, name)
        if sub != type(sub)():
            names.append(name)
    return tuple(names)


def _sub_model(config: SlabConfig, name: str) -> object:
    """The config sub-model whose non-default state enables note *name*."""
    if name == "mp":
        return config.builders.mp
    if name == "gracemaker":
        return config.builders.gracemaker
    return getattr(config.engines, name)


def note_text(name: str) -> str:
    """One note's text — the user's override when present, else the packaged note."""
    override = _user_root() / f"{name}.md"
    if override.is_file():
        return override.read_text(encoding="utf-8").strip()
    return (_builtin_root() / f"{name}.md").read_text(encoding="utf-8").strip()


def notes_block(config: SlabConfig) -> str:
    """The ``# Software notes`` section of the system prompt.

    Stable for a given configuration, so a prefix-caching server reuses it
    across sessions on the same machine.
    """
    lines = [
        "# Software notes",
        "",
        "Curated notes on the computational software and data configured "
        "here. They are starting context, not a live inventory: call "
        "list_engines for what actually exists right now.",
    ]
    for name in enabled_notes(config):
        lines.append(f"\n## {name}\n\n{note_text(name)}")
    return "\n".join(lines)
