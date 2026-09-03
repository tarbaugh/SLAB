---
name: kinetic-fits
description: Fit Arrhenius, Vogel-Fulcher-Tammann, or MYEGA laws to
  rate-versus-temperature tables with standard errors, test for
  curvature, or locate a zero crossing with an error bar. Use when asked
  for an activation energy, a diffusion or growth-rate law, VFT or MYEGA
  parameters, an attempt frequency, or a melting temperature from
  velocity data.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Kinetic fits

The bundled script fits temperature laws to recorded rate tables; it
launches nothing. The input is a JSON list of `{"T": kelvin, "value":
rate}` rows in whatever unit the rate carries, with an optional `"err"`
(one standard deviation of the value). Repeated temperatures are
replicas; keep them, because they are the best source of a per-row
error. The fitted prefactor comes back in the input unit.

## Run

    python <skill root>/scripts/fit_rates.py rates.json --mode arrhenius
    python <skill root>/scripts/fit_rates.py rates.json --mode arrhenius --jump-length 2.5 --value-unit cm2/s
    python <skill root>/scripts/fit_rates.py growth.json --mode vft --json
    python <skill root>/scripts/fit_rates.py growth.json --mode myega
    python <skill root>/scripts/fit_rates.py velocity.json --mode crossing --window 60

- `arrhenius` fits ln(value) against 1/T by weighted least squares
  (weights from `err`) and prints the activation energy in eV and J with
  its standard error, ln A with its error and the Ea–lnA correlation
  (near +1: a higher barrier comes with a higher prefactor, which is why
  a prefactor quoted alone misleads), the rate at T_ref (the harmonic
  mean of the temperatures, where the two parameters decorrelate), and a
  curvature verdict: a significant quadratic term in 1/T (F-test,
  p < 0.05) means the mechanism changes across the window. Use it for
  diffusion coefficients, escape or desorption rates,
  and any thermally activated process over a window where one barrier
  dominates. With `--jump-length a` (Å), `--dim d`, and `--value-unit`
  the diffusion prefactor becomes an effective attempt frequency
  nu = 2 d D0 / a^2; a prefactor is not an attempt frequency by itself.
- `vft` fits value = A exp(-B/(T-T0)) with T0 searched on a grid and
  gives bootstrap errors for B and T0, their correlation, and the
  strength D = B/T0. Use it for supercooled-liquid kinetics that slow
  faster than Arrhenius on cooling.
- `myega` fits value = A exp(-(K/T) exp(C/T)), the Mauro–Yue–Ellison–
  Gupta–Allan form with the same three parameters and no finite
  divergence temperature. Prefer it when the law will be extrapolated
  to lower temperature than the data.
- `crossing` locates where the value changes sign, then fits
  value = k (T_m − T) over the rows within `--window` kelvin of it and
  prints T_m with a bootstrap error and the kinetic coefficient k. Use
  it on a two-phase growth-velocity table: v = 0 is the melting
  temperature.

## Read the result

- Trust an Arrhenius fit only over the window it was fitted on; state
  the window with the energy. A `curved` verdict means the mechanism
  changes; fit regimes separately, or use VFT or MYEGA.
- A VFT T0 pinned at its search edge (the script warns) means the data
  does not reach deep enough undercooling to constrain the divergence;
  report B and the prefactor as conditional on that, or switch to MYEGA.
- Multiple sign changes in crossing mode mean the velocity data is noisy
  around the transition; run longer at the temperatures nearest the
  crossing before quoting T_m. Fewer than three points in the window
  leaves a two-point interpolation with no error bar; the script says so.
- Report every parameter with its error. Publishable melting points
  from coexistence carry about ±5 to 10 K from repeats.

## Caveats the numbers carry

- Rates in a table are only as good as the runs behind them; each row
  should trace to a run id, and the fit's report should list them.
- An activation energy from three points has no error bar (the script
  prints none). Five or more temperatures spanning at least a factor of
  ~1.5 in 1/T is a reasonable floor.
- Fitting velocity *magnitudes* from both sides of a crossing as one
  Arrhenius set is wrong; the two branches are different processes.
- The bootstrap resamples rows, so its error reflects the scatter
  between rows, not a bias shared by all of them.
