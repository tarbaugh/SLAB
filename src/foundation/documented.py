"""Behaviour the bundled skills already document, and the memories it refuses.

A machine memory is a fact about *this* machine. What a LAMMPS command does
everywhere is not such a fact, and a memory that restates it costs every
later session prompt space and, when the restatement is wrong, sends that
session down a wrong path. The skills under :mod:`foundation.skills` already
carry this material, so the memory store refuses a write that restates it and
names the skill section that holds the real answer.

The table below covers the commands the bundled skills document. Each entry
names the command, the skill and section that document it, and the subject
the skills state there. A memory is refused when its description names the
command and its text touches that subject. The test suite reads the table and
checks that every entry names a skill and a section that exist, so a skill
that loses a section cannot leave the table pointing at nothing.

The table is deliberately small. It holds the five subjects an agent has been
seen to re-derive and record, not every documented keyword, because a wide
table would refuse real machine facts about the same commands. A memory that
says the build segfaults in ``cna/atom`` above eight threads is a machine
fact and is recorded; one that says what the ``cna/atom`` codes mean is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Documented:
    """One documented subject: the command, where it is written down, and the words that name it."""

    #: The command or keyword, as a skill writes it. Matched in a memory's
    #: description as a whole word, case-insensitively.
    command: str
    #: The bundled skill that documents it.
    skill: str
    #: The section number in that skill, as its heading numbers it.
    section: str
    #: What the skill states there, for the refusal message.
    subject: str
    #: Patterns that show a text is about that subject. One match is enough.
    #: Searched case-insensitively over the name, description, and body.
    restates: tuple[str, ...]


#: The documented subjects a memory may not restate.
DOCUMENTED: tuple[Documented, ...] = (
    Documented(
        command="cna/atom",
        skill="two-phase-melting",
        section="3",
        subject="what the common-neighbour-analysis codes mean",
        restates=(
            r"\b(?:1|one)\b[^.]{0,40}\bfcc\b",
            r"\bfcc\b[^.]{0,40}\b(?:1|one)\b",
            r"\b(?:2|two)\b[^.]{0,40}\bhcp\b",
            r"\bhcp\b[^.]{0,40}\b(?:2|two)\b",
            r"\b(?:5|five)\b[^.]{0,40}\bunknown\b",
            r"\bunknown\b[^.]{0,40}\b(?:5|five)\b",
        ),
    ),
    Documented(
        command="dilate",
        skill="two-phase-melting",
        section="2",
        subject="which atoms a barostat remaps",
        restates=(
            r"\bdilate\b[^.]{0,60}\b(?:remap\w*|rescal\w*|scal\w*|position\w*|coordinat\w*)",
            r"\b(?:remap\w*|rescal\w*|scal\w*)\b[^.]{0,60}\bdilate\b",
        ),
    ),
    Documented(
        command="fix_modify",
        skill="lammps-scripting",
        section="8",
        subject="that a thermostat refuses fix_modify energy and econserve replaces it",
        restates=(
            r"\benergy\s+yes\b",
            r"\beconserve\b",
            r"does not support fix_modify",
        ),
    ),
    Documented(
        command="velocity",
        skill="lammps-scripting",
        section="3",
        subject="what velocity create and velocity scale set",
        restates=(
            r"\bvelocity\b[^.]{0,60}\b(?:create|scale)\b[^.]{0,60}\btemperature\b",
            r"\bvelocity\s+\w+\s+(?:create|scale)\b[^.]{0,60}\b(?:sets?|rescales?|draws?)\b",
            r"\b(?:mom|rot)\s+yes\b",
        ),
    ),
    Documented(
        command="halt",
        skill="lammps-scripting",
        section="7",
        subject="the syntax of fix halt and what it does at the condition",
        restates=(
            r"\bhalt\b[^.]{0,60}\b(?:error\s+continue|keyword|syntax|argument)",
            r"\bhalt\b[^.]{0,60}\bstops?\b[^.]{0,60}\brun\b",
            r"\bfix\s+\w+\s+\w+\s+halt\s+\d",
        ),
    ),
)


def _names(text: str, command: str) -> bool:
    """Whether *text* names *command* as a whole word, case-insensitively."""
    word = rf"(?<![A-Za-z0-9_]){re.escape(command)}(?![A-Za-z0-9_])"
    return re.search(word, text, re.I) is not None


def documented(description: str, text: str) -> Documented | None:
    """The table entry a memory restates, or None when it states a machine fact.

    *description* is the memory's one-line description, which must name the
    command. *text* is everything the memory holds, which must touch the
    documented subject. Both conditions are needed, so a memory that names a
    command while stating something the skills do not cover is recorded.

    Examples:
        >>> entry = documented(
        ...     "The build shifts the cna/atom codes.",
        ...     "Here 1 is hcp and 2 is fcc, not the documented mapping.",
        ... )
        >>> entry.skill, entry.section
        ('two-phase-melting', '3')
        >>> documented(
        ...     "cna/atom segfaults above eight threads on this build.",
        ...     "Run the compute on one thread until the build is replaced.",
        ... ) is None
        True
    """
    for entry in DOCUMENTED:
        if not _names(description, entry.command):
            continue
        if any(re.search(pattern, text, re.I) for pattern in entry.restates):
            return entry
    return None


def refusal(entry: Documented) -> str:
    """What a write refused for restating documentation says."""
    return (
        f"this is documented behaviour, not a fact about this machine: see "
        f"{entry.skill} section {entry.section} for {entry.subject}. If this machine "
        f"really departs from that, say so in your finish report so the skill gets "
        f"fixed, and record only what the machine does differently"
    )
