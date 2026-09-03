---
name: msd-diffusion
description: Compute the mean-squared displacement from a trajectory and
  estimate the diffusion coefficient by the Einstein relation, with the
  diffusive-regime test, a block standard error, per-axis values, and
  the finite-size correction. Use when asked whether atoms diffuse, for
  a diffusion coefficient, or to tell a solid from a liquid dynamically.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Mean-squared displacement and diffusion

The bundled script computes MSD(t) averaged over atoms and time origins,
fits the Einstein relation MSD = 2 d D t over a window of lags, and
reports D with its standard error. It reads recorded frames; it launches
nothing.

## Run

    python <skill root>/scripts/msd.py md.traj --dt-fs 5.0
    python <skill root>/scripts/msd.py md.traj --dt-fs 5.0 --skip 200 --species Li --json
    python <skill root>/scripts/msd.py md.traj --dt-fs 5.0 --yeh-hummer 1.5e-3 --temperature 1400

- `--dt-fs` is the time between *saved frames*: the MD timestep times the
  save interval. Getting this wrong scales D linearly, so read it from
  the run record, not from memory.
- `--skip N` drops unequilibrated frames before any origin is taken.
- `--fit-from` and `--fit-to` bound the linear fit as fractions of the
  maximum lag (defaults 0.05 and 0.25). Lags near half the run have one
  independent sample each and triple the scatter; the script refuses a
  window past 0.5. The early ballistic and cage regimes stay out below
  0.05.
- `--species X` restricts the average to one species (the mobile one in
  a solid electrolyte, for example). The whole cell's centre-of-mass
  drift is subtracted first (`--keep-com` disables it), so a thermostat
  drift does not enter the mobile species' MSD as a t^2 term.
- `--axes xy` (or `x`, `xz`, ...) sets the dimensionality d for a surface
  or a channel diffuser; the default `xyz` is d = 3. D per axis is
  printed in every case, and an unexpected anisotropy is itself a check.
- `--blocks 5` splits the trajectory into contiguous blocks and reports
  the standard error of D over them.
- `--yeh-hummer ETA --temperature T` adds the Yeh–Hummer estimate
  D_inf = D + xi k_B T/(6 pi eta L) for a liquid in a cubic cell of edge
  L, which is 10 to 15% for small cells.

## Read the result

- The script prints D in A^2/fs and cm^2/s with its block standard
  error, the log-log slope beta of the MSD over the window, D per axis,
  and the MSD table. Trust D only when beta is within 0.1 of 1; the
  script warns outside that, and calls a beta below 0.3 a plateau.
- A solid's MSD saturates at a plateau (vibration amplitude); reporting
  a "diffusion coefficient" from a plateau is wrong. Say the system is
  solid instead.
- Liquids near melting have D around 1e-5 cm^2/s; use that scale to
  sanity-check the exponent, which is where unit errors land.

## Caveats the number carries

- Positions must be unwrapped: frames whose coordinates were wrapped
  back into the cell corrupt MSD at every boundary crossing. ASE's MD
  writes unwrapped positions unless something wrapped them explicitly.
- Use an NVE or NVT production run. An NPT trajectory carries the
  barostat's rescaling in its positions and inflates the MSD at long
  lags; the script warns when the cell changes between frames.
- One trajectory gives one estimate; the origin average reduces noise
  but not bias, and the block error captures noise only. For a
  publishable D, average three or more independent runs and state the
  spread.
- Finite cells understate a liquid's D (hydrodynamic self-interaction);
  report the cell edge next to the number, and the Yeh–Hummer value when
  the viscosity is known. The correction does not apply to a mobile
  species in a rigid host.
