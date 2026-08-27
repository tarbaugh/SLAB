Classical molecular dynamics and statics through the LAMMPS binary. Fast at
real system sizes — thousands of atoms and beyond — and the accuracy is
entirely the interatomic potential's, so name the potential and its source
in every report.

The potential is required and is a science decision: pass `pair_style=` and
`pair_coeff=` (plus `files=` for potential files) in `calculator_options`.
There is no usable default. Entries in `files=` are staged into the
calculation's scratch, and bare basenames in `pair_coeff` resolve to the
staged copies, so an absolute path in `files=` works from any directory.

If a run dies with only `Failed to retrieve any thermo_style-output`, that
is LAMMPS masking its real error. The actual `ERROR: ...` line is in the
LAMMPS log kept in the run's failure evidence — read it with `show_run`.
