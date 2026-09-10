"""Exception types raised by SLAB.

All errors here derive from :class:`SlabError`. Messages are written to be
actionable for the caller — including LLM agents, who read them verbatim — so
they state what was attempted, why it was refused, and what would be allowed
instead.

SLAB is the bottom package, so this vocabulary is small: engines and the
scheduler. Run, artifact, and storage errors belong to Foundation, and the
agent's own errors to Mason. :class:`slab.config.ConfigError` derives from
:class:`SlabError` because the config loader lives here.
"""

from __future__ import annotations


class SlabError(Exception):
    """Base class for all SLAB errors."""


class EngineNotAvailableError(SlabError):
    """A requested calculation engine is unknown, or known but not installed."""


class BuilderNotAvailableError(SlabError):
    """A structure builder's executable cannot be found on this machine."""


class BuilderError(SlabError):
    """A structure builder ran and failed, or was invoked incorrectly.

    Carries the builder's full captured output on ``log`` so a caller can
    keep it as evidence; the message itself holds the extracted error lines.
    """

    def __init__(self, message: str, *, log: str = "") -> None:
        super().__init__(message)
        self.log = log


class LammpsScriptError(SlabError):
    """A LAMMPS input script ran and failed, or was staged incorrectly.

    Carries the log file's text on ``log`` and the captured screen output
    on ``screen`` so a caller can keep both as evidence; the message holds
    the extracted ``ERROR`` lines with one line of context.
    """

    def __init__(self, message: str, *, log: str = "", screen: str = "") -> None:
        super().__init__(message)
        self.log = log
        self.screen = screen
