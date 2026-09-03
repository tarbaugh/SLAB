---
name: nemd-transport
description: Thermal conductivity and thermal boundary resistance from
  non-equilibrium MD temperature profiles, with the fit window, the
  half-profile check, errors, and the length extrapolation. Use when
  asked for kappa, a Kapitza or interface resistance or conductance, a
  TBR, or to analyze a Muller-Plathe or heat-source/sink run.
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

## 1. Produce the inputs

- Run the production part under NVE in the gradient region. A global
  thermostat flattens the gradient it is supposed to measure; with a
  source and sink, thermostat only the reservoirs.
- The profile: bin the cell along the transport direction and
  time-average the temperature per bin *after* the profile stops
  drifting. Save it as a JSON list of `{"x": position, "T": kelvin}`
  rows or as two whitespace columns; positions default to Angstrom.
  Save two or three time blocks as separate files for an error bar.
- The flux, in W/m^2, comes from the run itself. For Muller-Plathe in a
  periodic cell it is the accumulated exchanged kinetic energy divided
  by (2 * cross-section area * exchange time); the 2 counts the two
  half-profiles. With fixed walls and a source and sink there is one
  profile and no factor 2. Record how it was computed in the run.
- Stay in linear response. Swapping too often makes the profile curve;
  keep the temperature drop across the fitted window under about 30 K
  and report k at the window's mean temperature.

## 2. Fit

    python <skill root>/scripts/fit_nemd.py kappa profile.json --flux 2.1e9 --fold --drop-ends 3
    python <skill root>/scripts/fit_nemd.py kappa block1.dat block2.dat --flux 2.1e9 --xmin 20 --xmax 80
    python <skill root>/scripts/fit_nemd.py tbr profile.dat --flux 2.1e9 --interface 42.5 --exclude-interface 8

- `kappa` fits one linear gradient and prints k = flux/|dT/dx| in
  W/(m K) with the slope's standard error, the mean temperature of the
  window, and R^2. Choose the window with `--xmin`/`--xmax` or
  `--drop-ends N` (the bins where the exchange slabs sit). A periodic
  Muller-Plathe profile is a sawtooth; `--fold` splits it at the middle,
  fits each half, and prints both k values and their mismatch. Several
  files are time blocks and give a mean with a standard error.
- `tbr` fits a line on each side of `--interface`, dropping the points
  within `--exclude-interface` of it, extrapolates both to the
  interface, and prints R = |dT|/flux in m^2 K/W and the conductance
  G = 1/R in MW/(m^2 K), plus each side's gradient and implied
  conductivity.
- `--json` gives the numbers machine-readable; `--x-unit` accepts A, nm,
  um, m.

## 3. Verify and report

- The two half-profiles of a Muller-Plathe cell must give the same k;
  the script warns above a 10% mismatch, which means the profile had
  not converged or the exchange slabs are inside the window.
- Finite cells understate k: phonons longer than the cell cannot carry
  heat. For a real number, repeat at four or more cell lengths spanning
  a decade and extrapolate 1/k against 1/L, and check that the points
  are linear; the extrapolation underpredicts by up to 2.5 times when
  the smallest cell is far shorter than the dominant mean free path.
  Green-Kubo and homogeneous NEMD need no length series and are the
  cross-check. For a screening number, state the length next to the
  value.
- Anisotropic crystals have a k per axis; name the transport direction
  in every report, with the potential, the cell, the mean temperature
  of the window, and the run id.
- A boundary resistance depends on the cell length and the reservoir
  temperature difference; report both with R and G, and expect NEMD and
  equilibrium estimates of an interface conductance to differ by
  several times.

## When not to use this

- Electronic conductors: MD carries only the lattice part of k, and in a
  metal that is the minority. Say so instead of reporting a lattice
  number as the conductivity.
- Temperatures well below the Debye temperature: classical MD populates
  every mode, and the quantum corrections are unreliable there; flag the
  regime rather than correcting it.
- Ballistic (sub-micron mean-free-path) regimes and strongly
  size-dependent geometries need the length extrapolation, not one box.
