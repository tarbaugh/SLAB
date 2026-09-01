---
name: mlip-training
description: Train or fine-tune a GRACE machine-learned interatomic
  potential with gracemaker — assemble a labeled dataset from recorded
  runs, author the input.yaml, fit on a GPU partition, and judge the
  metrics. Use when asked to train, fit, refit, or fine-tune an MLIP.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Training a GRACE potential with gracemaker

Gracemaker is the only MLIP-training route on this machine. It runs
through `foundation.tasks.train_potential`, which needs
`[builders.gracemaker]` configured — check `list_engines` for the
"gracemaker" entry before planning a fit.

## Assemble the dataset

Training data is labeled structures: energies and forces (stress when
available) from an engine you trust for the target chemistry.

- From recorded runs: `collect_training_data([run_id, ...])` gathers
  the results of completed `relax`, `relax_cell`, and `single_point`
  tasks into one extended-XYZ file. `frames="all"` also includes every
  labeled frame of the kept relaxation trajectories — cheap extra
  labels along the optimization paths.
- Never mix engines in one dataset. The collector refuses mixed
  sources; pass `engine=` to select one. `allow_mixed=True` exists but
  is almost always wrong.
- Generate labels the canonical way first: build or fetch structures,
  perturb them (rattle, strain, vacancies — the atomsk-* skills), and
  label with DFT `single_point` under one protocol. Tens of structures
  make a smoke-test fit; production potentials need hundreds to
  thousands of well-spread configurations.
- An existing extended-XYZ or `.pkl.gz` dataset works directly as
  `dataset=`.

## Author the input.yaml

`train_potential` takes the input.yaml **text**, verbatim — write it
yourself and reference the dataset by bare basename. This shape ran a
real fit (tensorpotential 0.6.0):

    seed: 1
    cutoff: 5.0
    data:
      filename: training.extxyz
      test_size: 0.1
      reference_energy: 0          # or auto (least-squares per-element E0)
    potential:
      preset: FS
      kwargs: {n_rad_base: 8, embedding_size: 16}
    fit:
      loss: {energy: {weight: 1.0}, forces: {weight: 5.0}}
      optimizer: L-BFGS-B
      opt_params: {"maxcor": 50, "maxls": 20, "gtol": 1.e-8, "iprint": -1}
      maxiter: 500
      batch_size: 8

- `fit: loss:` is required — a missing loss block dies with
  `KeyError: 'loss'` before any training. The `Adam` optimizer
  additionally requires `scheduler` and `scheduler_params`; the
  quasi-Newton optimizers (`L-BFGS-B`, `BFGS`) need neither.
- Presets: `FS` (fast, CPU-friendly, exports a C++-ready
  `FS_model.yaml`), `GRACE_1LAYER_latest` (GPU, local),
  `GRACE_2LAYER_latest` (GPU, semi-local, most accurate). Start small;
  scale the preset only when the metrics demand it.
- Fine-tuning: name a GRACE foundation model under `potential:`
  (for example `GRACE-2L-OMAT-medium-ft-E`) instead of a preset.
  Fine-tuning usually beats from-scratch when labels are scarce. The
  foundation weights must already sit in the machine's grace cache; a
  missing model is a machine blocker to report, never to download.
- One fit per call: one `seed`, one dataset, one task invocation.

## Run the fit

A real fit is a GPU batch job, never login-node work:

1. Write the workflow script: `collect_training_data(...)` then
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

## Verify and report

- Check the returned metrics with a `@check` in the workflow: energy
  RMSE per atom and force RMSE against thresholds you state in
  advance.
- Validate against held-out data before trusting the model: label a
  few structures the fit never saw, compare.
- Report the run id, the artifact names, the final train/test metrics,
  and the dataset provenance (which runs, which engine, how many
  structures) together. A potential without its training provenance is
  fiction with good statistics.
- To *use* the model: LAMMPS `pair_style grace` (or `grace/fs` with
  the FS export), a registry engine entry pointing at
  `tensorpotential.calculator.TPCalculator`, or ask the site to serve
  it through rootstock. The trained model is never an `engine=` name
  by itself.

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
