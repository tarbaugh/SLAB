---
name: nemd-transport
description: Thermal conductivity and thermal boundary resistance from
  non-equilibrium MD temperature profiles. Use when asked for kappa, a
  Kapitza or interface resistance, a TBR, or to analyze a Muller-Plathe
  or heat-source/sink run.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# NEMD transport

The bundled script turns a steady-state temperature profile plus the
imposed heat flux into a conductivity or an interface resistance; it
launches nothing. Producing the profile is the MD run's job: a
Muller-Plathe (reverse-NEMD) velocity swap, or an explicit heat source
and sink.

## Produce the inputs

- The profile: bin the cell along the transport direction and time-average
  the temperature per bin *after* the profile stops drifting. Save it as
  a JSON list of `{"x": position, "T": kelvin}` rows or as two whitespace
  columns; positions default to Angstrom.
- The flux, in W/m^2, comes from the run itself. For Muller-Plathe it is
  the accumulated exchanged kinetic energy divided by (2 * cross-section
  area * exchange time); the 2 counts the two half-profiles of the
  periodic cell. Record how it was computed in the run.

## Fit

    python <skill root>/scripts/fit_nemd.py kappa profile.json --flux 2.1e9
    python <skill root>/scripts/fit_nemd.py tbr profile.dat --flux 2.1e9 --interface 42.5

- `kappa` fits one linear gradient and prints k = flux/|dT/dx| in
  W/(m K).
- `tbr` fits a line on each side of `--interface`, extrapolates both to
  the interface, and prints R = |dT|/flux in m^2 K/W, plus each side's
  gradient and implied conductivity.
- `--json` gives the numbers machine-readable; `--x-unit` accepts A, nm,
  um, m.

## Verify and report

- Exclude the bins nearest the exchange slabs (and the interface, for
  tbr) before fitting; the script's linearity warnings flag when you
  have not.
- Finite cells understate k: phonons longer than the cell cannot carry
  heat. For a real number, repeat at two or three cell lengths and
  extrapolate 1/k against 1/L; for a screening number, state the length
  next to the value.
- Check the two half-profiles of a Muller-Plathe cell give the same k;
  a mismatch means the profile had not converged.
- Anisotropic crystals have a k per axis; name the transport direction
  in every report, with the potential, the cell, and the run id.

## When not to use this

- Electronic conductors: MD carries only the lattice part of k, and in a
  metal that is the minority. Say so instead of reporting a lattice
  number as the conductivity.
- Ballistic (sub-micron mean-free-path) regimes and strongly
  size-dependent geometries need the length extrapolation, not one box.
