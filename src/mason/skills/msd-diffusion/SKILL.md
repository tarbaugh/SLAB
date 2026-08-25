---
name: msd-diffusion
description: Compute the mean-squared displacement from a trajectory and
  estimate the diffusion coefficient by the Einstein relation. Use when
  asked whether atoms diffuse, for a diffusion coefficient, or to tell a
  solid from a liquid dynamically.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Mean-squared displacement and diffusion

The bundled script computes MSD(t) averaged over atoms and time origins,
fits the Einstein relation MSD = 6 D t over the tail, and reports D. It
reads recorded frames; it launches nothing.

## Run

    python <skill root>/scripts/msd.py md.traj --dt-fs 5.0
    python <skill root>/scripts/msd.py md.traj --dt-fs 5.0 --skip 200 --species Li --json

- `--dt-fs` is the time between *saved frames*: the MD timestep times the
  save interval. Getting this wrong scales D linearly, so read it from
  the run record, not from memory.
- `--skip N` drops unequilibrated frames before any origin is taken.
- `--fit-from` sets where the linear fit starts, as a fraction of the
  maximum lag (default 0.5): the early ballistic and cage regimes must
  stay out of the fit.
- `--species X` restricts the average to one species (the mobile one in
  a solid electrolyte, for example).

## Read the result

- The script prints D in A^2/fs and cm^2/s, and the MSD table so you can
  see the regime. Trust D only if the tail is actually linear; a curved
  tail means the run is too short or the system is not diffusive.
- A solid's MSD saturates at a plateau (vibration amplitude); reporting
  a "diffusion coefficient" from a plateau is wrong - say the system is
  solid instead.
- Liquids near melting have D around 1e-5 cm^2/s; use that scale to
  sanity-check the exponent, which is where unit errors land.

## Caveats the number carries

- Positions must be unwrapped: frames whose coordinates were wrapped
  back into the cell corrupt MSD at every boundary crossing. ASE's MD
  writes unwrapped positions unless something wrapped them explicitly.
- One trajectory gives one estimate; the origin average reduces noise
  but not bias. For a publishable D, average over independent runs and
  state the spread.
- Finite cells understate D somewhat (hydrodynamic self-interaction);
  say the cell size next to the number.
