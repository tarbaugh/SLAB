---
name: two-phase-melting
description: Melting temperature and crystal-growth velocities from
  solid-liquid coexistence MD. Use when asked for a melting point from
  simulation, an interface or growth velocity, growth anisotropy, or a
  v(T) table for kinetics.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# Two-phase melting and growth

A coexistence cell puts crystal and melt in direct contact and lets the
interface vote: below T_m the crystal grows, above it the crystal
shrinks, and the velocity crosses zero at T_m. This skill bundles no
script of its own; the fits live in the kinetic-fits skill. The
procedure has the traps.

## 1. Build the coexistence cell

- Make the cell long along the growth direction, with the crystal slab
  spanning the cross-section. Melt one half by holding only those atoms
  hot (or splice an equilibrated melt onto the crystal), then anneal the
  whole cell briefly so the two interfaces are clean.
- The periodic cell holds *two* interfaces; both move. Account for that
  factor in any velocity you extract.
- For an anisotropic crystal, build one cell per growth direction; the
  anisotropy is the ratio of the fitted velocities, direction by
  direction.

## 2. Run the temperature ladder

- Hold each copy of the cell at one temperature under NPT with an
  isotropic-enough barostat, spanning both sides of the expected T_m.
- Track the interface position over time: the fraction of crystalline
  atoms (a bond-order parameter), or the potential energy, which drifts
  linearly while an interface moves at constant velocity. Convert the
  drift to velocity via the cell cross-section, the latent heat per
  atom, and the two interfaces.
- Latent heat matters: a global thermostat must absorb the heat the
  moving interface releases, or the interface warms its own
  neighborhood and the velocity stalls. Large cross-sections and gentle
  thermostatting keep this honest; halve the thermostat coupling and
  confirm v does not move.

## 3. Fit and report

- Collect `{"T": kelvin, "value": velocity}` rows (signed: growth
  positive, melting negative) and use the kinetic-fits skill:
  `--mode crossing` gives T_m from v = 0; `--mode vft` on the growth
  branch magnitudes gives the kinetic prefactor, B, and T0 of a
  Vogel-Fulcher-Tammann growth law.
- T_m from coexistence carries a finite-size shift; repeat at a larger
  cross-section once and state the change next to the number.
- Report T_m with the potential, the cell geometry, the direction, the
  ladder, and the run ids; report v(T) as the table plus the fitted law,
  never the law alone.

## When not to use this

- One-phase heating until the crystal collapses measures superheating,
  not T_m; it overshoots by hundreds of kelvin. Use coexistence.
- Compounds that melt incongruently, or off-stoichiometric cells,
  couple melting to composition; the plain coexistence T_m applies to
  congruent melting at the cell's own stoichiometry, and the report
  must say so.
