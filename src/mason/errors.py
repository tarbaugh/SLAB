"""Exception types raised by Mason.

All errors here derive from :class:`MasonError`. Messages are written to be
actionable for the caller — including the model itself, which reads several of
them verbatim as tool observations — so they state what was attempted, why it
was refused, and what would be allowed instead.

:class:`MasonError` deliberately does not derive from
:class:`foundation.errors.FoundationError` or :class:`slab.errors.SlabError`.
The three packages are peers with separate vocabularies, so Mason's CLI
catches all three bases and lets each report itself in its own words.
"""

from __future__ import annotations


class MasonError(Exception):
    """Base class for all Mason errors."""
