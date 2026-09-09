---
name: mlip-training
description: Train or fine-tune a GRACE machine-learned interatomic
  potential with gracemaker — assemble a labeled dataset from recorded
  runs, hold out test data by run, author the input.yaml, fit on a GPU
  partition, and judge the metrics against stated thresholds and
  physical checks. Use when asked to train, fit, refit, or fine-tune an
  MLIP.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Training a GRACE potential with gracemaker

Gracemaker is the only MLIP-training route on this machine. It runs
through `foundation.tasks.train_potential`, which needs
`[builders.gracemaker]` configured — check `list_engines` for the
"gracemaker" entry before planning a fit.

## 1. Assemble the dataset

Training data is labeled structures: energies and forces (stress when
available) from an engine you trust for the target chemistry. Decide
first whether you fine-tune a foundation model or train from scratch,
because the two need different datasets (1a and 1b below). The rules
here apply to both.

- From recorded runs: `collect_training_data([run_id, ...])` gathers
  the results of completed `relax`, `relax_cell`, and `single_point`
  tasks into one extended-XYZ file. `frames="all"` also includes every
  labeled frame of the kept relaxation trajectories. Nothing else
  enters: an MD snapshot, a rattled cell, or a dimer becomes a label
  only after a `single_point` on it under the dataset's protocol.
- Split by run, not by frame. Adjacent frames of one relaxation are
  near-duplicates, and a random `test_size` split over them puts copies
  of the training frames in the test set and reports errors that are
  artificially low. Collect the held-out runs into a second file and
  pass it as `data: test_filename:`; never use a random split with
  `frames="all"`. Hold out whole sources, not just whole runs: one
  entire MD trajectory, one defect type, one temperature. A test set
  that looks like the training set measures interpolation only.
- Never mix engines in one dataset. The collector refuses mixed
  sources; pass `engine=` to select one. `allow_mixed=True` exists but
  is almost always wrong. Keep the labels consistent too: one
  functional, one pseudopotential family, one cutoff, one smearing, a
  k-mesh converged for the smallest cell in the set, force-consistent
  energies. Labels from `single_point` carry no stress, so drop the
  `stress` term from the loss unless every label has one.
- Count atoms, not structures. Each atom contributes three force
  labels, so many cells of 30 to 100 atoms teach more per DFT hour
  than a few large ones, and the potential is local (cutoff 6 to 7 Å)
  so the labels transfer to any cell size.
- Decorrelate. Space MD snapshots by at least a few hundred
  femtoseconds; a `rattle` of 0.05 to 0.2 Å on a relaxed cell is a
  cheap independent near-equilibrium sample; the collector's
  `n_duplicates` count tells you when two runs contributed the same
  structure.
- An existing extended-XYZ or `.pkl.gz` dataset works directly as
  `dataset=`.

### 1a. A dataset for training from scratch

A from-scratch potential knows nothing outside its data, so coverage is
the whole job. Sample every region the study will visit, and sample
past it:

| Region | How to generate the structures | Why |
|---|---|---|
| Equilibrium phases | `relax_cell` of each phase, `frames="all"` | anchors the energy ordering |
| Thermal | MD at the target temperature and at 1.5× it, snapshots labeled by `single_point` | the forces the study will actually ask for |
| Volume | the equation-of-state ladder, about ±15 % in volume | bulk modulus and pressure |
| Shear and strain | the elastic-constants strains, ±2 % | elastic response, transferable to defects |
| Short range | compressed cells and dimers at 60 to 80 % of the bond length | so a close approach is never extrapolation |
| Defects and surfaces | the atomsk-defects and atomsk-interfaces skills, relaxed and rattled | vacancies, interstitials, boundaries |
| Liquid or amorphous | melt-quench snapshots, when the study touches them | disordered environments |

- Tens of structures make a smoke-test fit; production potentials need
  hundreds to thousands of well-spread configurations. Keep the regions
  balanced: a set that is 90 % relaxation frames fits the equilibrium
  beautifully and fails at the first hot snapshot.
- Run one round of active learning before you trust the fit: run MD
  with the potential (or with two seeds of it), harvest the frames with
  the largest extrapolation grade or seed disagreement, label them with
  `single_point`, and refit. This closes the gaps a hand-built set
  always leaves.
- `reference_energy: auto` fits one E0 per element by least squares.
  Include labeled isolated atoms only when the study needs cohesive or
  dissociation energies on the DFT scale.
- Keep one test set of held-out sources for the metrics, and one
  "transfer" set of a structure type absent from training (a defect
  type, a higher temperature). The second number is the honest one.

### 1b. A dataset for fine-tuning a foundation model

The foundation model already covers general chemistry. Fine-tuning data
has two jobs: correct the model in the region the study needs, and move
its energy scale onto your protocol. Do not try to re-cover everything.

- Size: tens to a few hundred structures, concentrated on the target
  (the phases, temperatures, defects, and compositions the study will
  ask about). Fine-tuning with labels that are scarce beats a from-
  scratch fit, and it rarely needs more than a few hundred.
- Protocol compatibility: the GRACE OAM models were trained on PBE
  labels with Materials Project settings. Your labels under another
  functional or pseudopotential family carry a different per-element
  offset, which `shift: auto` absorbs, and different relative energies
  and forces, which the fine-tune pulls the model toward only where you
  have data. State the functional of the foundation labels and the
  functional of yours in the report, and treat the fine-tuned model as
  valid for your protocol inside the covered region only. Remember that
  SLAB's QE protocols default to PBEsol.
- Anchor against forgetting. A narrow dataset moves the model in the
  target region and degrades it everywhere else. Add a small anchor
  slice under your protocol: the relaxed bulk phases, a few
  equation-of-state points, and a few rattled cells of the same
  elements. Keep the learning rate small and `maxiter` short (the
  values in section 2), and stop when the test error stops falling.
- Set `eval_init_stats: True` so the log records the foundation model's
  own error on your test set before any update. That baseline is the
  proof that fine-tuning helped, and its absence makes the claim
  unverifiable.
- After the fit, compute one property you did not train on (an
  equation of state, an elastic constant, a phonon) with both the
  foundation model and the fine-tuned one. If the fine-tuned model is
  worse there, the anchor slice is too small or `maxiter` too long.
- Use only elements the foundation model knows, hold out by run
  exactly as above, and keep every label from one engine and one
  protocol. The collector's rules do not relax because the model is
  pre-trained.

Choose fine-tuning when the labels are scarce or the chemistry sits
inside the foundation model's training distribution. Choose from
scratch when your protocol differs from the foundation labels and the
whole model must be consistent with it, when the study needs the speed
of the `FS` preset, or when the chemistry is far from anything the
foundation model saw.

## 2. Author the input.yaml

`train_potential` takes the input.yaml **text**, verbatim — write it
yourself and reference the dataset by bare basename. This shape ran a
real fit (tensorpotential 0.6.0):

    seed: 1
    cutoff: 6.0
    data:
      filename: training.extxyz
      test_filename: test.extxyz     # held out by run
      reference_energy: auto         # per-element E0 by least squares
    potential:
      preset: FS
      kwargs: {n_rad_base: 8, embedding_size: 16}
    fit:
      loss: {energy: {weight: 1.0}, forces: {weight: 5.0}, stress: {weight: 0.1}}
      optimizer: L-BFGS-B
      opt_params: {"maxcor": 50, "maxls": 20, "gtol": 1.e-8, "iprint": -1}
      maxiter: 500
      batch_size: 8

- `fit: loss:` is required — a missing loss block dies with
  `KeyError: 'loss'` before any training. Drop the `stress` term when
  the labels carry no stress. The `Adam` optimizer additionally requires
  `scheduler` and `scheduler_params`; the quasi-Newton optimizers
  (`L-BFGS-B`, `BFGS`) need neither.
- `reference_energy: auto` (or an isolated-atom table `{Cu: -0.12, ...}`)
  matters with raw DFT totals; `0` makes the model absorb hundreds of eV
  per atom of offset.
- Presets: `FS` (fast, CPU-friendly, exports a C++-ready
  `FS_model.yaml`), `GRACE_1LAYER_latest` (GPU, local),
  `GRACE_2LAYER_latest` (GPU, semi-local, most accurate). Use a cutoff
  of 6 to 7 Å for the GRACE presets. Start small; scale the preset only
  when the metrics demand it.
- Fine-tuning a foundation model is its own key, not a preset:

      potential:
        finetune_foundation_model: GRACE-1L-OAM
        shift: auto
      fit:
        loss: {energy: {weight: 1.0}, forces: {weight: 5.0}}
        optimizer: Adam
        opt_params: {learning_rate: 1.e-4}
        scheduler: ReduceLROnPlateau
        scheduler_params: {factor: 0.8, patience: 5, min_lr: 1.e-6}
        eval_init_stats: True
        maxiter: 200

  Section 1b says what the dataset must hold. The foundation weights
  must already sit in the machine's grace cache; a missing model is a
  machine blocker to report, never to download.
- One fit per call: one `seed`, one dataset, one task invocation. Run
  two seeds when you need a variance on the metrics.

## 3. Run the fit

A real fit is a GPU batch job, never login-node work:

1. Write the workflow script: `collect_training_data(...)` for the
   training runs and again for the held-out runs, then
   `train_potential(input_yaml, dataset=...)` inside the same run.
2. Submit `slab run workflow.py` with `submit_job` on the GPU
   partition, then poll `job_status`.
3. `train_potential` keeps the training log, the model architecture,
   the final metrics, and the exported `saved_model` tar as run
   artifacts, and copies the exports into `{label}/` in the project.
   Pass `export_fs=True` for FS-preset fits that LAMMPS should read.

A failed fit keeps its log tail, partial metrics, and checkpoints as
run evidence — read them with `show_run`, change something, and state
what you changed. A fit that finished with bad metrics is not a
failure: judge it.

## 4. Verify and report

- Check the returned metrics with a `@check` in the workflow against
  thresholds you state in advance. Publishable bulk fits reach an energy
  RMSE of 1 to 5 meV/atom and a force RMSE of 50 to 100 meV/Å; a
  fine-tuned foundation model reaches under 2 meV/atom and about
  30 meV/Å.
- Test-set RMSE is not a proxy for extrapolation. Before trusting the
  model, compute what the study needs against DFT: lattice constants
  and an equation of state, elastic constants, a phonon or a rattled
  cell's forces, and an MD run at the target temperature that stays
  stable and keeps its extrapolation grades low.
- Report the run id, the artifact names, the final train and test
  metrics, the split (which runs were held out), and the dataset
  provenance (which runs, which engine, how many structures) together.
  A potential without its training provenance is fiction with good
  statistics.
- To *use* the model, the routes are fixed by where tensorpotential
  lives. It lives in gracemaker's own environment, never in SLAB's, so
  a registry entry pointing at `tensorpotential.calculator.TPCalculator`
  cannot import here; do not spend steps trying. The routes that work:
  LAMMPS `pair_style grace` (or `grace/fs` with the FS export) when the
  site's LAMMPS was built with the ML-GRACE package, or asking the site
  to serve the checkpoint through rootstock. When neither exists on this
  machine, report the trained artifact and its metrics, and name the
  missing route as a machine fact. The trained model is never an
  `engine=` name by itself.

## When not to use this

- No `[builders.gracemaker]` on this machine means no training here —
  say so; do not pip-install tensorpotential or any other training
  stack into SLAB's environment.
- There is no other trainer: MACE, NequIP, and pacemaker training do
  not exist on this machine, and requests for them route here or
  nowhere.
- A served foundation checkpoint (`list_engines`) often makes training
  unnecessary — screening and geometry work rarely justify a bespoke
  potential. Train when the chemistry or property is outside what the
  served models handle, and say why.
- Labels from a mixed bag of engines, protocols, or cutoffs make a
  potential that averages physics; regenerate consistent labels
  instead.
