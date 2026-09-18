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


class JobSizeError(SlabError):
    """A requested job size does not fit the partition's declared node, or the
    partition declares no node to size against."""


class ResourcesError(SlabError):
    """A launch's size does not fit: the slice it asks for, or the build it runs.

    Carries ``free`` (the cpu and gpu ids free on the host when a slice
    was refused; empty when the refusal is about the launch's own shape)
    so a caller can size the next request without asking again.
    Foundation's :class:`foundation.errors.ResourcesError` derives from
    this one, so a caller that catches it sees both.
    """

    def __init__(self, message: str, *, free: dict[str, list[object]] | None = None) -> None:
        super().__init__(message)
        self.free = free if free is not None else {"cpus": [], "gpus": []}


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


class QeToolError(SlabError):
    """A Quantum ESPRESSO post-processing tool ran and failed.

    ``dos.x`` and ``projwfc.x`` tell their failure story on their standard
    output, in the same ``%%%%``-fenced block ``pw.x`` uses. The message
    holds the extracted block or the tail of the output, and ``log``
    carries the whole capture so a caller can keep it as evidence.
    """

    def __init__(self, message: str, *, tool: str = "", log: str = "") -> None:
        super().__init__(message)
        self.tool = tool
        self.log = log


class LammpsScriptError(SlabError):
    """A LAMMPS input script ran and failed, or was staged incorrectly.

    Carries the log file's text on ``log`` and the captured screen output
    on ``screen`` so a caller can keep both as evidence; the message holds
    the extracted ``ERROR`` lines with one line of context. A failure of a
    LAMMPS that started also carries the ``command`` it ran and
    ``elapsed_s``, the seconds from its start to its end, so a device that
    refused within seconds can be told from a fault later in the run.
    """

    def __init__(
        self,
        message: str,
        *,
        log: str = "",
        screen: str = "",
        command: str | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.log = log
        self.screen = screen
        self.command = command
        self.elapsed_s = elapsed_s
