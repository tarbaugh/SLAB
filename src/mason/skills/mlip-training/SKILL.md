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
available) from an engine you trust for the target chemistry.

- From recorded runs: `collect_training_data([run_id, ...])` gathers
  the results of completed `relax`, `relax_cell`, and `single_point`
  tasks into one extended-XYZ file. `frames="all"` also includes every
  labeled frame of the kept relaxation trajectories.
- Split by run, not by frame. Adjacent frames of one relaxation are
  near-duplicates, and a random `test_size` split over them puts copies
  of the training frames in the test set and reports errors that are
  artificially low. Collect the held-out runs into a second file and
  pass it as `data: test_filename:`; never use a random split with
  `frames="all"`.
- Never mix engines in one dataset. The collector refuses mixed
  sources; pass `engine=` to select one. `allow_mixed=True` exists but
  is almost always wrong. Keep the labels consistent too: one
  functional, one cutoff, one smearing, force-consistent energies.
- Cover what the potential will see. Relaxation paths sit near
  equilibrium; add high-temperature MD snapshots, compressed cells and
  dimers (or a core repulsion) so close approaches are not
  extrapolation, strained cells, surfaces, and defects (the atomsk-*
  skills), all labeled with DFT `single_point` under one protocol. Tens
  of structures make a smoke-test fit; production potentials need
  hundreds to thousands of well-spread configurations, and one round of
  active learning (run MD with the fit, harvest the frames with the
  largest extrapolation grade, relabel, refit) closes the gaps a hand-
  built set leaves.
- An existing extended-XYZ or `.pkl.gz` dataset works directly as
  `dataset=`.

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

  Fine-tuning usually beats from-scratch when labels are scarce, and
  freezing most layers matches a from-scratch fit with a fifth of the
  data. The foundation weights must already sit in the machine's grace
  cache; a missing model is a machine blocker to report, never to
  download.
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
