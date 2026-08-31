# Engines

SLAB implements no physics. Every engine is reached through exactly one
seam, the ASE `Calculator` contract, whether that engine is the EMT toy, a
`pw.x` binary, LAMMPS on a cluster, or an MLIP served from a rootstock
install. This page follows that seam from laptop built-ins to cluster
registries.

## One seam, three sources

`get_calculator(engine, **options)` maps an engine name to a ready ASE
calculator. Three sources feed the mapping, tried in order:

1. **Built-ins.** `emt` and `lj` (ASE toys), `qe` (Quantum ESPRESSO's
   `pw.x`, no extra needed), `lammps` (the `lmp` binary, likewise no
   extra), and `rootstock` (cluster-served MLIPs,
   `slab-stack[rootstock]`).
2. **The cluster engine registry.** Names that a cluster maintainer declared
   in an `engines.json` that lives with the install, such as `vasp`, curated
   site aliases like `qe-delta`, and MLIP aliases.
3. **Rootstock checkpoint ids.** Any canonical id that a cluster's rootstock
   install declares works directly as an engine name, with no registry
   entry needed.

Registry entries deliberately win over bare checkpoint ids, so a
maintainer's curated alias, with its baked-in options, beats bare
resolution. Nothing in the tracing, lifecycle, or retention layers knows
that engines exist. To add a backend, add a registry entry, and never touch
SLAB.

## Built-ins

`available_engines()` lists what resolves right now, and `get_calculator`
builds it.

```python
from slab.backends import available_engines, get_calculator

print(available_engines())
calc = get_calculator("emt")
print(type(calc).__name__)
```

```text
('emt', 'lammps', 'lj', 'qe', 'rootstock')
EMT
```

Tasks take the same name. `relax` forwards `calculator_options` verbatim to
the engine factory, and stamps the resolved identity into its `info` dict:

```python
from ase.build import bulk
from foundation import Workspace
from foundation.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
atoms.rattle(stdev=0.05, seed=42)

ws = Workspace("workspace")
with ws.start_run(name="cu-relax", intent="engine seam demo") as run:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05,
                          calculator_options={"asap_cutoff": False})
print(info["engine"], info["engine_source"], info["engine_version"])
print(f"E = {info['energy']:.6f} eV   fmax = {info['fmax']:.4f}   steps = {info['steps']}")
```

```text
emt builtin None
E = -0.053417 eV   fmax = 0.0388   steps = 9
```

EMT is deterministic and the rattle seed is fixed, so these numbers reproduce
exactly. EMT and LJ run in milliseconds per step, and they fit only the
elements they parametrize, which makes them ideal for tests and tutorials
but not for science. For real work SLAB has no in-process MLIP path; the
route to a MACE, UMA, or any other foundation model is `rootstock`, covered
below.

## Quantum ESPRESSO

`qe` is a built-in that drives `pw.x` through ASE's file-IO calculator, so
it works wherever the executable and the pseudopotentials exist, whether
that is a laptop build or a cluster module, with no extra to install. Two
options locate the code, `command` and `pseudo_dir`, or you can configure an
`[espresso]` section in ASE's own config file once and pass neither.
Everything else is standard `Espresso` options, forwarded verbatim.
`resolve_pseudopotentials` maps elements to `.upf` files in a directory, and
it refuses ambiguity rather than guessing, because which pseudopotential to
use is a science decision:

<!-- no-verify -->
```python
from slab.backends import resolve_pseudopotentials
from foundation.tasks import relax

pseudos = resolve_pseudopotentials(atoms, "/opt/pseudos/sssp")  # {'Si': 'Si.pz-vbc.UPF'}
relaxed, info = relax(
    atoms,
    engine="qe",
    fmax=0.05,
    label="si",
    calculator_options={
        "command": "pw.x",                    # or "mpirun -np 8 pw.x"
        "pseudo_dir": "/opt/pseudos/sssp",
        "pseudopotentials": pseudos,
        "input_data": {"system": {"ecutwfc": 40.0}},
        "kpts": (2, 2, 2),
    },
)
print(info["engine_version"], info["converged"], info["steps"])
```

<!-- no-verify -->
```text
7.4.1 True 2
```

The details that keep runs honest and directories clean:

- **Scratch, not cwd.** Each calculation runs in a slab-managed temporary
  directory that is removed when the task finishes, or pass `directory=` to
  manage the files yourself. Inside a run, the final SCF's `espresso.pwo` is
  kept as an intermediate artifact (`si.pwo` here) next to the trajectory.
  ASE reruns `pw.x` for every force evaluation and overwrites its output, so
  the kept file is the last step's.
- **The `pw.x` identity is detected and cached against.** SLAB probes the
  binary once per executable and parses the `Program PWSCF v.7.4.1` banner.
  The probe is memoized on the binary's path and mtime, and it runs under a
  timeout. The version, the resolved command, and `pseudo_dir` go into
  provenance (`info["engine_version"]`) and the cache key, so an executable
  upgrade or a pseudopotential-library switch honestly invalidates cached
  results. Pseudo file contents are not hashed, because the directory path
  is the identity.
- **Forces default on.** `pw.x` omits forces unless `tprnfor` is set, and
  SLAB's tasks drive optimizers with forces, so the factory sets it. An
  explicit `tprnfor` in `input_data` still wins.
- **Failure speaks QE.** A crashed `pw.x` is a bare `CalledProcessError` in
  Python, but the evidence lands in the failure record: QE's own
  `Error in routine ...` message, and the kept input, output, and `CRASH`
  files. See
  [Debugging failures](debugging-failures.md#when-the-engine-writes-files).

Two patterns adopted from AiiDA remove the ceremony from the options above.
Installable **pseudopotential families** replace `pseudo_dir` plus
`pseudopotentials` with one name (`slab pseudos install sssp`, then
`pseudo_family="SSSP/1.3/PBEsol/efficiency"`), and named **input
protocols** expand to curated cutoffs, k-mesh, smearing, and thresholds
(`qe_protocol_options(atoms, protocol="balanced")`). Both have their own
page, [Protocols & pseudopotentials](protocols-and-pseudos.md).

A cluster's curated QE setup (fixed module, shared pseudo library, MPI
launcher) still belongs in the registry below, under a distinct alias like
`qe-delta`, because entries may not shadow built-in names.

## LAMMPS

`lammps` is a built-in on the same terms as `qe`. It drives the `lmp` binary
through ASE's `lammpsrun` calculator, so it works wherever the executable
exists, with no extra to install. One option locates the code, `command`,
or you can set it once under `[engines.lammps]` in the slab config or export
`$ASE_LAMMPSRUN_COMMAND`. Bare `lmp` is the default.

The interatomic potential has no default at all. `pair_style` and
`pair_coeff` are required, and SLAB refuses a `lammps` engine without them.
The refusal is deliberate, because ASE's own fallback is a dimensionless
`lj/cut` toy that would "relax" any material and return meaningless numbers,
and which potential describes a system is a science decision, exactly like
which pseudopotential.

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms,
    engine="lammps",
    fmax=0.05,
    label="cu",
    calculator_options={
        "command": "lmp",                     # or "mpirun -np 8 lmp"
        "pair_style": "eam",
        "pair_coeff": ["1 1 Cu_u3.eam"],
        "files": ["/opt/potentials/Cu_u3.eam"],
    },
)
print(info["engine_version"], info["converged"], info["energy"])
```

<!-- no-verify -->
```text
22 Jul 2025 - Update 4 True -28.31961277087714
```

The details that keep runs honest and directories clean:

- **Scratch, not cwd, and staged potentials.** Each calculator runs in a
  slab-managed scratch directory that is removed when the task finishes, or
  pass `tmp_dir=` to manage the files yourself. `files=` entries are staged
  into that scratch, and bare-basename references to them in `pair_coeff`
  are rewritten to the staged copies, so the options above work from any
  cwd. (Stock `lammpsrun` resolves that bare `Cu_u3.eam` against the
  caller's cwd, which a traced task must never depend on.) Inside a run, the
  final force evaluation's log, thermo table included, is kept as an
  intermediate artifact (`cu.log` here) next to the trajectory.
- **The `lmp` identity is detected and cached against.** SLAB probes
  `<command> -h` once per executable and parses the `Large-scale
  Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025` banner. The
  probe is memoized on the binary's path and mtime, and it runs under a
  timeout. The version and the resolved command go into provenance and the
  cache key. Potential file contents are not hashed, because their paths,
  which ride in the traced options, are the identity.
- **Units come back converted.** Whatever `units=` the potential requires
  (`metal`, `real`, ...), ASE converts results to eV and eV/Å, so `relax`'s
  `energy_unit` stays `"eV"`.
- **Failure speaks LAMMPS, because the scratch exists.** A crashed `lmp`
  surfaces in Python as a useless
  `RuntimeError: Failed to retrieve any thermo_style-output`, because the
  real `ERROR: Unrecognized pair style 'eam/aloy' (src/force.cpp:275)` is
  raised inside a `lammpsrun` reader thread, where no caller can catch it.
  The log file is the only place the message survives, and stock `lammpsrun`
  retains files only when given a working directory, which the slab-managed
  scratch provides. The failure record gets the `ERROR` line(s), one line of
  preceding context (the echoed command that died, or the last thermo row
  before a blow-up), and the kept `-failed.{in,log,data}` files.
  See [Debugging failures](debugging-failures.md#when-the-engine-writes-files).

A cluster's curated LAMMPS setup (fixed module, MPI launcher) belongs in the
registry below under a distinct alias like `lammps-delta`, the same as QE.

## Per-engine environments

Each engine runs isolated by construction. QE and LAMMPS are external
binaries in private slab-managed scratch directories, and rootstock and the
Mason model server live behind HTTP in their own environments. What crosses
the seam is the command line, and when one engine needs its own environment
variables, the command line carries those too:

```toml
[engines.qe]
command = "env OMP_NUM_THREADS=4 pw.x"
```

ASE execs engine commands as a plain argv, with no shell, so `/usr/bin/env`
applies the assignments to that engine's subprocess alone. Nothing leaks
into the Python process, the cache, or any other engine.

For a custom Quantum ESPRESSO install, you can name the install instead of
writing the command:

```toml
[engines.qe]
bin = "/shared/sw/qe-7.4/bin"
```

SLAB then constructs the command as `mpirun -np N <bin>/pw.x`. `N` comes
from `$SLURM_NTASKS`, so a batch job uses its whole allocation and a
login-node smoke test stays serial. An `mpirun` bundled in the same bin
directory wins over the one on `PATH`, because a custom install usually
links against its own MPI. `bin` and `command` are exclusive, and the
constructed line enters cache identity exactly as a hand-written command
would. `mason sandbox render` also binds the whole install read-only
automatically, so the bin form needs no `[agent.sandbox]` entry.

Sometimes an engine's install needs more than variables, such as a
`module load`, an `LD_LIBRARY_PATH` change, or anything else that takes a
shell. Declare that as the engine's own `setup`:

```toml
[engines.qe]
command = "pw.x"
setup = ["module purge", "module load qe/7.4", "export OMP_NUM_THREADS=4"]

[engines.lammps]
command = "lmp"
setup = ["module purge", "module load lammps/2025.07"]
```

SLAB materializes each engine's `setup` into a private `#!/bin/bash -l`
wrapper that runs `set -e`, then your lines, then `exec` of the real
command, and the wrapper is scoped to that engine's subprocess alone. Where
`[hpc] setup` lines apply to the whole job, `[engines.X] setup` lines apply
to one engine, so two engines with conflicting module stacks can share one
job without sharing an environment.

The guards follow the semantics. With `setup` lines in play, the process PATH
proves nothing, because the module load may be exactly what provides `pw.x`.
So SLAB checks existence, and probes the version banner, inside the same
login shell the wrapper uses, and a setup that still cannot find the binary
refuses loudly, with the shell's own words. Cache identity stamps the logical
command, the setup lines, and the setup-resolved binary. The wrapper file
itself is an implementation detail that SLAB creates with the calculator
and removes with it. `setup=` also works per call in `calculator_options`,
where it overrides the config's, and inside a registry alias's options.

The isolation is lateral, not a sandbox. Each engine gets its own short-lived
shell that dies with its subprocess, so one engine's `module load` never
reaches another, but every engine starts from the same base. Three layers
arrive in the engine's process, and each can override the one before:

1. the driver's environment, which holds whatever `sbatch` exported, the
   `[hpc] setup` lines, the driver's venv, and any registry `env` values slab
   applied;
2. the login profile that `bash -l` sources;
3. your `setup` lines, last.

The profile can rewrite `PATH` before your lines run, so do not assume that a
`PATH` the driver built arrives intact. This is why the template leads with
`module purge`, the line that decides what the engine inherits. Drop it to
build on the job's modules, or keep it when the engine's stack must not see
them. For a sealed userland, the same command seam takes a container, and
`command = "apptainer exec /sw/qe.sif pw.x"` isolates the whole userland.

The guards look through the `env` wrapper, for plain assignments and for
env's portable flags (`-i`, `-u NAME`) alike:

- The PATH check judges `pw.x`, not `env`. When the wrapper assigns `PATH=`
  itself, the payload is resolved under that PATH, which is the
  module-load-replacement case.
- An `env`-wrapped `srun` still trips the allocation refusal.
- The version-probe memo watches every binary the command names, payload
  included.
- Version probes never run through an MPI launcher, because `srun pw.x`
  would queue or consume a job step and `mpirun -np 64 pw.x` would fan ranks
  out on a login node. So the probe runs the bare payload, which prints the
  same banner.

The bare shell idiom (`command = "OMP_NUM_THREADS=4 pw.x"`) is refused by
name, because only a shell would apply it and there is no shell. The refusal
message spells out the working form.

Three boundaries to know about:

- A batch job is still one environment. `[hpc]` and partition `setup` lines
  apply to the whole job, so anything one engine needs and another must not
  see belongs in that engine's own `[engines.X] setup`, or in the `env`
  command wrapper for plain variables, and never in job-global setup.
- Rootstock-served MLIPs live in the site's own pre-built environment, so
  compute nodes never touch a checkpoint download or a torch install. A
  named checkpoint id used as an engine name is the whole surface.
- Slab-managed scratch directories default to `$TMPDIR`, which clusters often
  point at small node-local tmpfs. Set `[paths] scratch` to a shared scratch
  root, so pw.x's wavefunctions fit and MPI ranks on other nodes can see
  their input.

## Two fidelities, one run

The seam's payoff is that engines compose. `foundation.tasks` ships two traced
tasks that take the same `engine` argument: `relax` runs BFGS on positions,
and `single_point` runs one energy and forces evaluation with no
optimization. The canonical workflow is therefore a chain that relaxes under
a cheap engine and then evaluates the relaxed geometry under the expensive
one, with the DFT residual force as the check that says whether the cheap
geometry held up:

<!-- no-verify -->
```python
from ase.build import bulk
from foundation import check, converged
from slab.protocols import qe_protocol_options
from foundation.tasks import relax, single_point

atoms = bulk("Si", "diamond", a=5.43)
atoms.rattle(stdev=0.05, seed=11)

relaxed, cheap = relax(atoms, engine="mace-mp-0-medium", fmax=0.02, label="si-mace",
                       calculator_options={"cluster": "delta"})

options = qe_protocol_options(relaxed, protocol="fast")
final, dft = single_point(relaxed, engine="qe", label="si-scf",
                          calculator_options=options)

@check
def dft_confirms_geometry():
    return converged(dft["fmax"], below=0.1, label="dft fmax")
```

Executed for real, with a rootstock-served MACE-MP-0 medium checkpoint and
Quantum ESPRESSO 7.5 with the SSSP PBEsol efficiency family through the
`fast` protocol:

<!-- no-verify -->
```text
MLIP relax: converged=True E=-10.7384560555 eV fmax=0.013715 eV/A steps=4
QE single point: E=-308.5312843034 eV fmax=0.021275 eV/A version=7.5 n_atoms=2
run 01m08kqbd1sv4cadjxcre2tcaa  si-two-fidelity  state=verified status=completed checks=3/3 tasks=2
```

`single_point`'s info deliberately has no `converged` key, because nothing
was optimized, and an engine whose own self-consistency fails raises, with
the engine's error report attached as notes, instead of returning. Its
artifacts follow relax's rules. The SCF's output is kept as the intermediate
`si-scf.pwo`. Both tasks cache, so rerunning the script serves both results
without re-invoking the MLIP or `pw.x`. Provenance links the chain by hash
equality, because `single_point`'s `atoms` input hash is `relax`'s first
output hash.

## Builders: atomsk

Some external codes make structures, not energies. SLAB names them
*builders*, and ships one: [atomsk](https://atomsk.univ-lille.fr), which
creates unit cells, supercells, defects, interfaces, and polycrystals, and
converts between the file formats the engines read. A builder is not an
engine. `engine="atomsk"` does not exist, and nothing about it passes
through `get_calculator`.

Point SLAB at the install in `slab.toml`:

<!-- no-verify -->
```toml
[builders.atomsk]
command = "atomsk"                       # or an absolute path
setup = ["module load atomsk/0.13"]      # per-builder scoped shell, like the engines
```

The traced task is `foundation.tasks.build_structure`. It runs one atomsk
invocation in a private scratch directory, reads the produced file back as
ASE `Atoms`, and keeps the file and the atomsk log as artifacts. The atomsk
version and the resolved command enter the cache key, exactly like an
engine's:

<!-- no-verify -->
```python
from foundation.tasks import build_structure, relax

supercell, build = build_structure(
    "--create fcc 4.046 Al -duplicate 4 4 4 al.xsf", label="al-444"
)
relaxed, opt = relax(supercell, engine="emt", fmax=0.05, label="al-444")
```

Executed for real, with atomsk built from source:

<!-- no-verify -->
```text
built: Al256 atomsk=master-2026-07-24 output=al.xsf
relax: converged=True E=-0.5030025151477417 eV n_atoms=256
```

The argument list is atomsk's own. File names must be bare, because the
invocation runs in a fresh scratch directory; structures and parameter
files the invocation reads enter through the task's `inputs=` mapping, and
each is a traced input. On failure the `X!X ERROR` lines from atomsk's log
ride on the exception as notes, and the full log is kept as
`{label}-failed.log` — the same evidence contract as the engine tasks. The
`atomsk-structures`, `atomsk-defects`, and `atomsk-interfaces` skills carry
the recipes and a bundled structure checker.

## The cluster engine registry

For VASP, site-specific MLIP aliases, and curated site setups of the built-in
engines, SLAB generalizes rootstock's management pattern. The client is only
a bootstrap, and a JSON file that lives with the cluster declares how each
canonical name is built here. Workflow code says `engine="vasp"` and runs
unchanged on any cluster whose registry declares `vasp`.

```json
{
  "layout_version": 1,
  "cluster": "delta",
  "engines": {
    "mace-mp": {
      "calculator": "rootstock.RootstockCalculator",
      "options": {"cluster": "delta", "checkpoint": "mace-mp-0-medium", "device": "cuda"},
      "version": "rootstock/mace-mp-0-medium",
      "probe": ["rootstock", "list"]
    },
    "qe-delta": {
      "calculator": "slab.backends.qe_calculator",
      "options": {"command": "srun pw.x", "pseudo_dir": "/sw/pseudos/sssp"},
      "version": "7.3.1",
      "probe": ["pw.x", "-h"]
    }
  }
}
```

Every entry is a dotted path to an ASE calculator class or factory, and even
rootstock enters through the same seam. A curated QE or LAMMPS alias goes
through SLAB's own factories (`slab.backends.qe_calculator` and
`lammps_calculator`), which carry the built-in engines' guards and take
JSON-able options. The fields:

- `options` are defaults that the caller overrides key-by-key.
- `env` declares variables the calculator reads at run time
  (`VASP_PP_PATH`, `ASE_VASP_COMMAND`). SLAB applies them process-wide at
  build time, because those calculators consult the environment when they
  calculate. This is also why `ASE_CONFIG_PATH` is refused: ASE parses its
  config file exactly once, at import, before any registry entry runs, so
  that declaration would silently never apply. Applied env stays in this
  process, and job submission hands `sbatch` the environment as it was
  before any registry entry ran, so the kept batch script means the same
  thing no matter which engines the submitting process built first.
- `probe` is a cheap command that proves the engine actually works here.

The slab-factory aliases are deliberately strict, because their spec, plus
the caller's traced options, is their whole cache identity. So
`qe_calculator` refuses to fall back to `[engines.qe]` or ASE config for
`command` and `pseudo_dir`, and it requires an explicit k-point policy
(`kpts=` or `kspacing=`), since the task-level k-point refusal only
recognizes the literal name `qe`.

Discovery order is an explicit path, else `$SLAB_ENGINES`, else
`~/.config/slab/engines.json`, so a maintainer ships the file at a shared
path and exports `SLAB_ENGINES` from a module file. The full worked example
is
[examples/engines.example.json](https://github.com/tarbaugh/SLAB/blob/main/examples/engines.example.json).

<!-- no-verify -->
```bash
slab engines list      # built-ins + everything this cluster declares
slab engines verify    # run every entry's probe; exit nonzero on failure
```

Two refusals keep resolution honest. An entry that shadows a built-in name
(`qe`, `emt`, ...) is rejected loudly at load, because it would silently
win or lose depending on resolution order, and either way you would get a
different engine than someone intended. Names retired from the built-ins
(`mace`) are legal, and declaring one is how a site keeps `engine="mace"`
working in existing scripts by pointing at a rootstock checkpoint. A
registry that declares a newer `layout_version` than the client
understands refuses rather than misreads.

You can exercise the whole chain locally, because the registry is just a
file, and nothing says its calculator must live on a cluster:

```python
import json, os
from slab.engines import load_registry

registry = {
    "layout_version": 1,
    "cluster": "laptop",
    "engines": {
        "emt-cluster": {
            "calculator": "ase.calculators.emt.EMT",
            "version": "1.0",
            "description": "EMT masquerading as cluster software",
        }
    },
}
with open("engines.json", "w") as f:
    json.dump(registry, f)
os.environ["SLAB_ENGINES"] = os.path.abspath("engines.json")

print(available_engines(load_registry()))
with ws.start_run(name="cu-relax-registry", intent="resolve through the registry") as run:
    relaxed, info = relax(atoms, engine="emt-cluster", fmax=0.05)
print(info["engine"], info["engine_source"], info["engine_version"])
```

```text
('emt', 'lammps', 'lj', 'qe', 'rootstock', 'emt-cluster')
emt-cluster registry:laptop 1.0
```

The declared `version` lands in the task recipe as provenance and in the
cache key. When a maintainer bumps `qe-delta` from 7.3 to 7.4, SLAB honestly
invalidates cached results instead of serving the old engine's numbers, and
not just for versions, because any spec edit (options, env, calculator)
changes the fingerprint. Watch it happen:

```python
with ws.start_run(name="cu-relax-cached", intent="same registry, cache hit") as again:
    relax(atoms, engine="emt-cluster", fmax=0.05)
print("cache hit:", ws.runs.list_tasks(again.id)[0].cache_hit)

registry["engines"]["emt-cluster"]["version"] = "2.0"
with open("engines.json", "w") as f:
    json.dump(registry, f)

with ws.start_run(name="cu-relax-bumped", intent="registry bumped, recompute") as bumped:
    relax(atoms, engine="emt-cluster", fmax=0.05)
print("cache hit:", ws.runs.list_tasks(bumped.id)[0].cache_hit)
ws.close()
```

```text
cache hit: True
cache hit: False
```

For how the cache key is built, see [Caching & resume](caching-and-resume.md).
The point here is that engine identity is part of it.

## Rootstock checkpoint ids, served silently

On a cluster with a [Garden-AI/rootstock](https://github.com/Garden-AI/rootstock)
install, any canonical checkpoint id works directly as the engine name.
Rootstock resolves the hosting environment and serves the model from a worker
subprocess, so your Python environment stays free of torch and model
packages, and `pip install 'slab-stack[rootstock]'` adds only a thin client:

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms,
    engine="mace-mp-0-medium",                        # a checkpoint id IS an engine
    calculator_options={"cluster": "delta", "device": "cuda"},
)
```

SLAB finds the install via `cluster=` or `root=` in `calculator_options`,
else `[engines.rootstock]` in the slab config, else rootstock's own defaults
(`$ROOTSTOCK_ROOT`, `~/.config/rootstock/config.toml`). The config section is
the machine-fact home, exactly like `[engines.qe] command`, so a local
install declares its path once:

```toml
[engines.rootstock]
root = "/scratch/me/rootstock-install"   # a local install: the path form
# cluster = "delta"                      # or a site-maintained install, by
#                                        # rootstock's name for it — unrelated
#                                        # to [hpc] cluster, the SLURM label
```

Checkpoint ids then work as engine names with no per-call options. Because
resolution tries the registry first, a maintainer's alias always beats a
bare id. The explicit `engine="rootstock"` form remains for full control,
for example
`calculator_options={"checkpoint": "uma:custom", "weights": ..., "cluster": "delta"}`
with your own weights, and `relax` closes the worker subprocess when the
task finishes.

Cache identity for silently-served engines is deliberately the checkpoint id
plus the rootstock client version, never the install's path or hosting
environment. Rootstock's contract is that canonical ids are stable
identities, so the same id on another install is the same computation, and
cluster-side internals (env rebuilds, in-place weight edits) are invisible
to any client.

## Trust model

Stated plainly: registry entries execute maintainer-declared code and
environment variables as the calling user, and rootstock installs run
maintainer-built environments the same way. SLAB isolates configuration,
not privilege, so trusting a cluster's `engines.json` is trusting its module
farm. `slab engines verify` tells you the declarations work, not that they
are benign.

When an engine misbehaves mid-run, the evidence survives, as described in
[Debugging failures](debugging-failures.md). For where the seam sits in the
overall design, see [Architecture](../architecture.md).
