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
