"""SLAB — Simplest Layer for Atomistic Backends.

Access to computational software, and nothing above it. SLAB maps engine
names to ASE calculators, reads the cluster engine registry, expands named
Quantum ESPRESSO protocols, installs and verifies pseudopotential families,
and renders SLURM batch scripts. It implements no physics, keeps no state,
and never decides what to compute.

Start here:

- :func:`slab.backends.get_calculator` — an ASE calculator by engine name
- :mod:`slab.engines` — the cluster engine registry
- :mod:`slab.protocols` — named QE input protocols
- :mod:`slab.pseudos` — pseudopotential families
- :mod:`slab.hpc` — SLURM batch scripts and job state
- :mod:`slab.config` — the layered ``slab.toml`` loader

Workflows, runs, and state live in :mod:`foundation`; the resident agent in
:mod:`mason`. Neither is imported from here, and SLAB imports neither.

The heavy imports (ASE, numpy, torch) stay behind :mod:`slab.backends`, so
importing this package stays cheap.

Examples:
    >>> from slab.backends import get_calculator
    >>> type(get_calculator("emt")).__name__
    'EMT'
"""

from slab._version import __version__
from slab.config import ConfigError
from slab.errors import EngineNotAvailableError, SlabError

__all__ = [
    "ConfigError",
    "EngineNotAvailableError",
    "SlabError",
    "__version__",
]
