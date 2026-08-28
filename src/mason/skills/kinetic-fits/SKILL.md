---
name: kinetic-fits
description: Fit Arrhenius or Vogel-Fulcher-Tammann laws to rate-versus-
  temperature tables, or locate a zero crossing. Use when asked for an
  activation energy, a diffusion or growth-rate law, VFT parameters, an
  attempt frequency, or a melting temperature from velocity data.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Kinetic fits

The bundled script fits temperature laws to recorded rate tables; it
launches nothing. The input is a JSON list of `{"T": kelvin, "value":
rate}` rows in whatever unit the rate carries; the fitted prefactor comes
back in the same unit.

## Run

    python <skill root>/scripts/fit_rates.py rates.json --mode arrhenius
    python <skill root>/scripts/fit_rates.py growth.json --mode vft --json
    python <skill root>/scripts/fit_rates.py velocity.json --mode crossing

- `arrhenius` fits ln(value) against 1/T and prints the activation energy
  in eV and J with the prefactor. Use it for diffusion coefficients,
  escape or desorption rates, and any thermally activated process over a
  window where one barrier dominates.
- `vft` fits value = A exp(-B/(T-T0)) with T0 searched on a grid. Use it
  for supercooled-liquid kinetics: growth velocities and viscous rates
  that slow faster than Arrhenius on cooling.
- `crossing` locates where the value changes sign and prints the
  temperature and local slope. Use it on a two-phase growth-velocity
  table: v = 0 is the melting temperature.

## Read the result

- Trust an Arrhenius fit only over the window it was fitted on; state
  the window with the energy. A curved ln(value) vs 1/T means the
  mechanism changes; fit regimes separately.
- A VFT T0 pinned at its search edge (the script warns) means the data
  does not reach deep enough undercooling to constrain the divergence;
  report B and the prefactor as conditional on that.
- Multiple sign changes in crossing mode mean the velocity data is noisy
  around the transition; run longer at the temperatures nearest the
  crossing before quoting T_m.

## Caveats the numbers carry

- Rates in a table are only as good as the runs behind them; each row
  should trace to a run id, and the fit's report should list them.
- An activation energy from three points has no error bar worth the
  name. Five or more temperatures spanning at least a factor of ~1.5 in
  1/T is a reasonable floor.
- Fitting velocity *magnitudes* from both sides of a crossing as one
  Arrhenius set is wrong; the two branches are different processes.
