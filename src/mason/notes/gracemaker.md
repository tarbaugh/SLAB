Gracemaker trains GRACE machine-learned interatomic potentials, and it
is the only MLIP-training route on this machine. There is no MACE,
NequIP, or pacemaker training here, and none can be installed — never
pip-install a training stack into SLAB's environment. Gracemaker lives
in its own python environment, reached through `[builders.gracemaker]`
setup lines; the `train_potential` task is the only way to run it.

Training is a GPU batch job. Write the workflow script
(`collect_training_data` then `train_potential`), and submit
`slab run workflow.py` through `submit_job` on the GPU partition — a
real fit on a login node is never acceptable. The task's `timeout_s`
and the job's time limit are the guards; poll with `job_status`.

Datasets are labeled structures: energies and forces from completed
runs (`collect_training_data`), or an existing extended-XYZ file. Never
mix labels from different engines in one training set — the collector
refuses this unless you explicitly allow it, and you should almost
never allow it. Before training from scratch, check whether fine-tuning
a GRACE foundation model (named under `potential:` in the input.yaml)
fits the data budget better; foundation weights live in the site cache,
and a missing model there is a machine blocker to report, not to
download.

You author gracemaker's input.yaml yourself and pass its text to
`train_potential`. The mlip-training skill carries the schema and the
recipe. Judge the fit by the returned train/test metrics with a
`@check`, and validate the model against held-out DFT before trusting
it. Report the model artifacts, the final metrics, and the run id
together.

A trained model is not an engine here. To *use* it, deploy it: LAMMPS
`pair_style grace` with the exported model, a registry engine entry
(`tensorpotential.calculator.TPCalculator`), or ask the site to serve
it through rootstock. Serving stays rootstock's territory; training
stays gracemaker's.
