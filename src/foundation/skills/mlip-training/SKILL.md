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
  with the potential, grade the frames (section 5), harvest those with
  the largest gamma or seed disagreement, label them with
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
   A Kokkos LAMMPS run needs the weights export of section 5.

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
  stable and whose largest gamma (section 5) stays inside the training
  distribution.
- Report the run id, the artifact names, the final train and test
  metrics, the split (which runs were held out), and the dataset
  provenance (which runs, which engine, how many structures) together.
  A potential without its training provenance is fiction with good
  statistics.
- To *use* the model, the routes are fixed by where tensorpotential
  lives. It lives in gracemaker's own environment, never in SLAB's, so
  a registry entry pointing at `tensorpotential.calculator.TPCalculator`
  cannot import here; do not spend steps trying. The routes that work:
  LAMMPS with `pair_style grace` on the saved model (a TensorFlow
  build), the `/kk` styles on the Kokkos weights of section 5 (a KOKKOS
  build, no TensorFlow), or `grace/fs` on the FS export, with the lines
  and the switches in the lammps-potentials skill; or asking the site
  to serve the checkpoint through rootstock. When neither exists on this
  machine, report the trained artifact and its metrics, and name the
  missing route as a machine fact. The trained model is never an
  `engine=` name by itself.

## 5. Uncertainty, acceleration, and active learning

Everything here runs in gracemaker's environment, as a shell step under
the same setup lines as the fit, inside the job that trains or after
it. `train_potential` runs gracemaker only, so these commands ride in
the workflow's job script, and their outputs enter the record as
artifacts of a run or as files the report names by path. Nothing here
is imported into SLAB's environment.

### The extrapolation grade

GRACE's uncertainty signal is gamma, one value per atom: the
Mahalanobis distance of the atom's environment to the nearest cluster
in the model's own latent space, divided by a calibrated per-cluster
threshold. Gamma below about 1 is inside the training distribution,
near 1 is the boundary, and far above 1 is extrapolation, where the
forces are not to be trusted.

- 1L, 2L, and 3L models get gamma from an artifact built once, after
  the fit, from the training data. The command from the gracemaker
  documentation:

      grace_uq build --model-yaml model.yaml \
                     --checkpoint checkpoints/checkpoint.best_test_loss.index \
                     --train-data training_set.pkl.gz \
                     --artifact-path UQ/gmm_artifacts.npz

  `--train-data` takes `.pkl.gz` datasets; convert an extended-XYZ file
  with `extxyz2df` first. `--n-workers 4` and `--gpus 0,1` spread the
  work, `--n-clusters 1 2 4 8 16` lets the elbow method choose the
  cluster count, and `grace_uq info UQ/gmm_artifacts.npz` inspects the
  result. The build also exports a saved model with a `compute_uq`
  signature, so `TPCalculator` returns `gamma` and `atomic_sigma` in its
  results. Foundation models with a tick in the UQ column of
  `grace_models list` ship the artifact with their checkpoint.
- GRACE/FS has no such artifact. Build an active set with D-optimality
  instead (python-ace must be installed in the same environment; when
  it is not, that is a machine fact to report):

      cd seed/1
      pace_activeset -d training_set.pkl.gz FS_model.yaml

  The result, `FS_model.asi`, sits next to the export. LAMMPS reads it
  through `pair_style grace/fs extrapolation`, and `PyGRACEFSCalculator`
  through `set_active_set`; the lammps-potentials skill has the lines.
  This gamma is on the PACE scale: above 5 is outside the training set,
  above 25 far outside.
- An ensemble is the third signal, and the only one that needs no
  artifact: fit the same input.yaml under two or more seeds, then
  evaluate with `TPCalculator(model=[...])` over the saved models. The
  results carry `energy_std`, `forces_std`, and `stress_std` next to
  the ensemble means. Use it when the artifact cannot be built, or to
  confirm what gamma says.

Report the largest gamma over the frames a study used next to every
number that depends on the potential. A trajectory that spent time at
gamma far above 1 is not evidence; it is a list of frames to label.

### Acceleration

- Precision. Foundation models resolve to fp32 by default, about twice
  the speed and half the memory of fp64 at a negligible accuracy cost.
  The `-fp64` suffix on the name selects full precision, and
  `grace_utils cast_model_param` converts a fit. The `-mx` models are
  natively mixed precision, and the 3L models exist only in fp32.
- Kokkos weights for LAMMPS. Export once after the fit:

      grace_utils -p model.yaml -c checkpoints/checkpoint.index export_kokkos -o grace_weights.npz

  Add `--uq-artifacts UQ/gmm_artifacts.npz` to bake the gamma thresholds
  into the weights. A foundation model's weights come from
  `grace_models download NAME --kokkos`. The `/kk` pair styles read the
  file without TensorFlow, and the lammps-potentials skill gives the
  `-k on g 1 -sf kk -pk kokkos newton on neigh half` switches that run
  them on a GPU. This is the route for MD with a 1L, 2L, or 3L model
  whenever the machine declares a gpu build: the script names
  `grace/1l/kk`, `grace/2l/kk`, or `grace/3l/kk` (or a `/mixed` or
  `/fp32` variant) and passes the `.npz` in `files=`, because `-sf kk`
  cannot derive those styles from `pair_style grace`. The export takes
  the standard architectures only and refuses a custom one with a clear
  error; a refused model runs through the TensorFlow styles or as
  GRACE/FS.
- The FS preset. Choose it when the campaign needs millions of atoms,
  CPU-only nodes, or MPI across nodes: `grace/fs` is a C++
  implementation with MPI parallelisation and no TensorFlow, at lower
  accuracy than the GRACE presets. `export_fs=True` on `train_potential`
  writes `FS_model.yaml`.
- Padding. The TensorFlow calculator compiles the model for each new
  input shape. `TPCalculator` pads adaptively by default, and
  `pair_style grace padding 0.05` does the same in LAMMPS; large cells
  use the chunk variants. A log line saying that adaptive padding grew
  its margins is normal, not an error.

### One round of active learning

1. Run MD with the potential at the study's conditions as a recorded
   run, with frames written often enough to catch the excursions (every
   few hundred femtoseconds).
2. Grade the frames. For a 1L, 2L, or 3L model,
   `grace_uq predict --model-yaml model.yaml --checkpoint checkpoints/checkpoint.index --artifact-path UQ/gmm_artifacts.npz --data frames.pkl.gz --output graded.pkl.gz`
   writes energies, forces, and per-atom gamma for every frame. For FS,
   `PyGRACEFSCalculator` with the `.asi` gives the same per frame.
3. Select what to label:
   `grace_uq select --artifact-path UQ/gmm_artifacts.npz --candidate-data frames.pkl.gz --n-select 100 --strategy extrapolation`
   picks frames by extrapolation and diversity. Without an artifact,
   take the frames with the largest seed disagreement.
4. Label the selection with `single_point` under the dataset's
   protocol, append the labels to the training file, and refit with the
   same input.yaml. The graded MD frames are training data now, so keep
   them out of the test set.
5. Stop when a production-length MD keeps its largest gamma inside the
   training distribution and the transfer-set error stops falling.
   Report the number of rounds, the frames labelled per round, and the
   largest gamma before and after.

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
