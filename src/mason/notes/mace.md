The MACE machine-learned interatomic potential. This is the default cheap
engine for geometry: relax under MACE, then `single_point` the relaxed
structure under `qe` and check the DFT residual force.

Two routes to it. `engine="mace"` runs in-process and needs the `slab[mace]`
extra; options forward to `mace.calculators.mace_mp` (`model="small"` /
`"medium"` / `"large"` or a checkpoint file path, `device="cuda"`). First use
downloads the checkpoint to `~/.cache/mace`, so on a firewalled compute node
prefer the second route: a served rootstock checkpoint id used directly as
the engine name (`engine="mace-mp-0-medium"`). Call `list_engines` to see
which checkpoint ids exist here, and never pip-install an MLIP on a cluster.

Trust its forces and geometries across broad chemistry. Do not trust its
absolute energies as DFT-grade, and do not use it for charged defects,
magnetism, or electronic properties. Energy differences between similar
structures are useful for screening; label them as MLIP-level, not DFT.
