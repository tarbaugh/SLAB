"""Distribution-level housekeeping for the SLAB stack.

The three packages own their layers; this one owns the broom. It sits
above all of them because a cleanup that means anything has to cross
them: the run store and artifact bytes are ``foundation``'s, the session
transcripts are ``mason``'s, and the scheduler that says whether a job
file is still live is ``slab``'s. Two verbs, deliberately blunt:

* ``slab-stack fast-forward`` — every unpromoted run goes to ``expired``,
  now. Promotion is the only shelter; this is the lifecycle's TTL sweep
  with the clock removed.
* ``slab-stack purge`` — everything expired goes away for real: run rows,
  unshared artifact bytes, session transcripts, and finished jobs'
  scripts and SLURM output files.
"""

from slab._version import __version__

__all__ = ["__version__"]
