The cluster's pre-built MLIP serving layer, and the only route SLAB has
to a machine-learned interatomic potential. Any canonical checkpoint id
its install declares works directly as an engine name —
`engine="mace-mp-0-medium"`, `engine="uma-s-1p1"` — and the model runs in
the site's own environment, so nothing is installed into yours. Call
`list_engines` to see the checkpoint ids available here; a curated
registry alias for the same model wins over the bare id when both exist.
If `list_engines` shows no checkpoint ids, this machine has no MLIP —
configure `[engines.rootstock]` first, or ask the site admin to declare
the checkpoint you need.

A served calculator runs in a worker subprocess. The `foundation.tasks`
functions close it automatically; only long-lived hand-rolled scripts
need to care.

A declared checkpoint id is a promise the install may not keep: if a
worker dies while trying to download model weights, the checkpoint is
declared but its weights are not cached in the install. The sandbox has
no network and the install is mounted read-only, so no workaround exists
from here. Record the gap with `remember`, report it as a machine
blocker naming the checkpoint id, and move on or stop. Do not explore
the install's directories, and do not attempt a download — the install
belongs to the site, not to this session.

Trust its forces and geometries across broad chemistry. Do not trust its
absolute energies as DFT-grade, and do not use it for charged defects,
magnetism, or electronic properties. Energy differences between similar
structures are useful for screening; label those numbers MLIP-level, not
DFT.
