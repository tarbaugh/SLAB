# Engines

SLAB implements no physics. Every engine — EMT toy, in-process MACE, a `pw.x`
binary, LAMMPS on a cluster, an MLIP served from a rootstock install — is
reached through exactly one seam: the ASE `Calculator` contract. This page
walks that seam from laptop built-ins to cluster registries.

## One seam, three sources

`get_calculator(engine, **options)` maps an engine *name* to a ready ASE
calculator. Three sources feed the mapping, tried in order:

1. **Built-ins** — `emt`, `lj` (ASE toys), `mace` (in-process, `slab[mace]`),
   `qe` (Quantum ESPRESSO's `pw.x`, no extra needed), `lammps` (the `lmp`
   binary, likewise no extra), `rootstock` (cluster-served MLIPs,
   `slab[rootstock]`).
2. **The cluster engine registry** — names a cluster maintainer declared in an
   `engines.json` that lives with the install (`vasp`, curated site aliases
   like `qe-delta`, MLIP aliases).
3. **Rootstock checkpoint ids, served silently** — any canonical id a cluster's
   rootstock install declares works directly as an engine name, no registry
   entry needed.

Registry entries deliberately win over bare checkpoint ids: a maintainer's
curated alias with baked-in options beats bare resolution. Nothing in the
tracing, lifecycle, or retention layers knows engines exist — adding a backend
means adding a registry entry, never touching SLAB.

## Built-ins

`available_engines()` lists what resolves right now; `get_calculator` builds it.

```python
from slab.backends import available_engines, get_calculator

print(available_engines())
calc = get_calculator("emt")
print(type(calc).__name__)
```

```text
('emt', 'lammps', 'lj', 'mace', 'qe', 'rootstock')
EMT
```

Tasks take the same name. `relax` forwards `calculator_options` verbatim to the
engine factory, and stamps the resolved identity into its `info` dict:

```python
from ase.build import bulk
from slab import Workspace
from slab.tasks import relax

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
exactly. EMT and LJ run in milliseconds per step and are fit only for the
elements they parametrize — ideal for tests and tutorials, not for science. For
real work in-process, the MACE foundation model:

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms,
    engine="mace",
    calculator_options={"model": "medium", "device": "cuda"},
)
```

!!! note
    Options for `mace` are forwarded to `mace.calculators.mace_mp`; defaults are
    `model="small"`, `device="cpu"`, `default_dtype="float64"`. First use
    downloads the checkpoint to `~/.cache/mace` — on a cluster, do that once
    from a node with internet (compute nodes are typically firewalled), or
    skip in-process MACE entirely and use a rootstock-served checkpoint id as
    the engine name. The resolved model and the mace-torch version are both
    part of the engine's cache identity.

## Quantum ESPRESSO

`qe` is a built-in: it drives `pw.x` through ASE's file-IO calculator, so it
works wherever the executable and pseudopotentials exist — a laptop build or a
cluster module, no extra to install. Two options locate the code: `command`
and `pseudo_dir` (or configure an `[espresso]` section in ASE's own config
file once and pass neither). Everything else is standard `Espresso` options,
forwarded verbatim. `resolve_pseudopotentials` maps elements to `.upf` files
in a directory and refuses ambiguity rather than guessing — *which*
pseudopotential to use is a science decision:

<!-- no-verify -->
```python
from slab.backends import resolve_pseudopotentials
from slab.tasks import relax

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
  directory, removed when the task finishes; pass `directory=` to manage the
  files yourself. Inside a run, the final SCF's `espresso.pwo` is kept as an
  intermediate artifact (`si.pwo` here) next to the trajectory — ASE reruns
  `pw.x` for every force evaluation, overwriting its output, so what survives
  is the last step's.
- **The `pw.x` identity is detected and cached against.** SLAB probes the
  binary (once per executable — memoized on its path and mtime, under a
  timeout), parses the `Program PWSCF v.7.4.1` banner, and folds the version
  plus the resolved command and `pseudo_dir` into provenance
  (`info["engine_version"]`) and the cache key: upgrading the executable or
  switching pseudopotential libraries honestly invalidates cached results.
  Pseudo file *contents* are not hashed — the directory path is the identity.
- **Forces default on.** `pw.x` omits forces unless `tprnfor` is set, and
  SLAB's tasks drive optimizers with forces — so the factory sets it (an
  explicit `tprnfor` in `input_data` still wins).
- **Failure speaks QE.** A crashed `pw.x` is a bare `CalledProcessError` in
  Python; the evidence — QE's own `Error in routine ...` message, the kept
  input/output/`CRASH` files — lands in the failure record. See
  [Debugging failures](debugging-failures.md#when-the-engine-writes-files).

Two adopted AiiDA patterns take the ceremony out of the options above:
installable **pseudopotential families** (`slab pseudos install sssp`, then
`pseudo_family="SSSP/1.3/PBEsol/efficiency"` replaces
`pseudo_dir`+`pseudopotentials`) and named **input protocols**
(`qe_protocol_options(atoms, protocol="balanced")` expands to curated
cutoffs, k-mesh, smearing, and thresholds). Both are their own story:
[Protocols & pseudopotentials](protocols-and-pseudos.md).

A cluster's *curated* QE setup (fixed module, shared pseudo library, MPI
launcher) still belongs in the registry below, under a distinct alias like
`qe-delta` — entries may not shadow built-in names.

## LAMMPS

`lammps` is a built-in on the same terms as `qe`: it drives the `lmp` binary
through ASE's `lammpsrun` calculator, so it works wherever the executable
exists — no extra to install. One option locates the code: `command` (or set
it once under `[engines.lammps]` in the slab config, or export
`$ASE_LAMMPSRUN_COMMAND`; bare `lmp` is the default). The *interatomic
potential* has no default at all: `pair_style` and `pair_coeff` are required,
and a `lammps` engine without them is refused. That refusal is deliberate —
ASE's own fallback is a dimensionless `lj/cut` toy that would happily "relax"
any material and return numbers meaning nothing, and *which* potential
describes a system is a science decision, exactly like which pseudopotential.

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

- **Scratch, not cwd — and staged potentials.** Each calculator runs in a
  slab-managed scratch directory, removed when the task finishes (pass
  `tmp_dir=` to manage the files yourself). `files=` entries are staged into
  that scratch and bare-basename references to them in `pair_coeff` are
  rewritten to the staged copies — so the options above work from any cwd.
  (Stock `lammpsrun` resolves that bare `Cu_u3.eam` against the *caller's*
  cwd, which a traced task must never depend on.) Inside a run, the final
  force evaluation's log — thermo table included — is kept as an intermediate
  artifact (`cu.log` here) next to the trajectory.
- **The `lmp` identity is detected and cached against.** SLAB probes
  `<command> -h` (once per executable — memoized on its path and mtime, under
  a timeout), parses the `Large-scale Atomic/Molecular Massively Parallel
  Simulator - 22 Jul 2025` banner, and folds the version plus the resolved
  command into provenance and the cache key. Potential file *contents* are
  not hashed — their paths, riding in the traced options, are the identity.
- **Units come back converted.** Whatever `units=` the potential requires
  (`metal`, `real`, ...), ASE converts results to eV and eV/Å — `relax`'s
  `energy_unit` stays `"eV"`.
- **Failure speaks LAMMPS — because the scratch exists.** A crashed `lmp`
  surfaces in Python as a useless
  `RuntimeError: Failed to retrieve any thermo_style-output`: the real
  `ERROR: Unrecognized pair style 'eam/aloy' (src/force.cpp:275)` is raised
  inside a `lammpsrun` reader *thread*, where no caller can catch it. The log
  file is the only place the story survives, and stock `lammpsrun` retains
  files only when given a working directory — which is exactly what the
  slab-managed scratch provides. The failure record gets the `ERROR` line(s),
  one line of preceding context (the echoed command that died, or the last
  thermo row before a blow-up), and the kept `-failed.{in,log,data}` files.
  See [Debugging failures](debugging-failures.md#when-the-engine-writes-files).

A cluster's curated LAMMPS setup (fixed module, MPI launcher) belongs in the
registry below under a distinct alias like `lammps-delta`, the same as QE.

## Per-engine environments

Each engine runs isolated by construction: QE and LAMMPS are external
binaries in private slab-managed scratch directories, rootstock and the
Mason model server live behind HTTP in their own environments, and only
mace-torch is deliberately in-process (its cluster-grade isolation *is*
rootstock). What crosses the seam is the command line — and when one engine
needs its own environment variables, the command line carries those too:

```toml
[engines.qe]
command = "env OMP_NUM_THREADS=4 pw.x"
```

ASE execs engine commands as a plain argv (no shell), so `/usr/bin/env`
applies the assignments to that engine's subprocess alone — nothing leaks
into the Python process, the cache, or any other engine. The guards look
through the wrapper — plain assignments and env's portable flags (`-i`,
`-u NAME`) alike: the PATH check judges `pw.x`, not `env` (and when the
wrapper assigns `PATH=` itself, the payload is resolved under *that* PATH,
the module-load-replacement case), an `env`-wrapped `srun` still trips the
allocation refusal, and the version-probe memo watches every binary the
command names, payload included. Version probes also never run *through* an
MPI launcher — `srun pw.x` would queue or eat a job step and
`mpirun -np 64 pw.x` would fan ranks out on a login node, so the probe runs
the bare payload, which prints the same banner. The bare shell idiom
(`command = "OMP_NUM_THREADS=4 pw.x"`) is refused by name — only a shell
would apply it, and there is no shell — with the working form spelled out
in the message. The same seam takes a container:
`command = "apptainer exec /sw/qe.sif pw.x"` isolates the whole userland.

Three boundaries to know about. A batch job is still one environment:
`[hpc]` and partition `setup` lines (module loads) apply to the whole job,
so a module whose libraries conflict with the driver's venv belongs on the
engine command via `env` (say, `env LD_LIBRARY_PATH=/opt/qe/lib pw.x`), not
in job-global setup. A *named* MACE checkpoint downloads on first use with
no offline mode — on firewalled compute nodes slab bounds the attempt and
refuses with instructions instead of hanging, but the real fix is the
pre-warm (or rootstock) described under [Built-ins](#built-ins). And
slab-managed scratch directories default to $TMPDIR, which clusters often
point at small node-local tmpfs — set `[paths] scratch` to a shared scratch
root so pw.x's wavefunctions fit and MPI ranks on other nodes can see their
input.

## Two fidelities, one run

The seam's payoff is that engines compose. `slab.tasks` ships two traced
tasks — `relax` (BFGS on positions) and `single_point` (one energy+forces
evaluation, no optimization) — that take the same `engine` argument, so the
canonical workflow is a chain: relax under a cheap engine, then evaluate the
relaxed geometry under the expensive one. The DFT residual force is the
check that says whether the cheap geometry held up:

<!-- no-verify -->
```python
from ase.build import bulk
from slab import check, converged
from slab.protocols import qe_protocol_options
from slab.tasks import relax, single_point

atoms = bulk("Si", "diamond", a=5.43)
atoms.rattle(stdev=0.05, seed=11)

relaxed, cheap = relax(atoms, engine="mace", fmax=0.02, label="si-mace")

options = qe_protocol_options(relaxed, protocol="fast")
final, dft = single_point(relaxed, engine="qe", label="si-scf",
                          calculator_options=options)

@check
def dft_confirms_geometry():
    return converged(dft["fmax"], below=0.1, label="dft fmax")
```

Executed for real (MACE-MP small on CPU; Quantum ESPRESSO 7.5 with the
SSSP PBEsol efficiency family through the `fast` protocol):

<!-- no-verify -->
```text
MACE relax: converged=True E=-10.7384560555 eV fmax=0.013715 eV/A steps=4
QE single point: E=-308.5312843034 eV fmax=0.021275 eV/A version=7.5 n_atoms=2
run 01m08kqbd1sv4cadjxcre2tcaa  si-two-fidelity  state=verified status=completed checks=3/3 tasks=2
```

`single_point`'s info deliberately has no `converged` key — nothing was
optimized, and an engine whose own self-consistency fails raises (with the
engine's own error report attached as notes) instead of returning. Its
artifacts follow relax's rules: the SCF's output is kept as the intermediate
`si-scf.pwo`, both tasks cache (rerunning the script serves both results
without re-invoking MACE or `pw.x`), and provenance links the chain by hash
equality — `single_point`'s `atoms` input hash *is* `relax`'s first output
hash.

## The cluster engine registry

For VASP, site-specific MLIP aliases, and curated site setups of the built-in
engines, SLAB generalizes rootstock's management pattern: the client is only
a bootstrap, and a JSON file that lives with the cluster declares how each
canonical name is built *here*. Workflow code says `engine="vasp"` and runs
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

Every entry is a dotted path to an ASE calculator class or factory — even
rootstock enters through the same seam, and a curated QE or LAMMPS alias
goes through SLAB's own factories (`slab.backends.qe_calculator` /
`lammps_calculator`), which carry the built-in engines' guards and take
JSON-able options. `options` are defaults the caller overrides key-by-key;
`env` declares variables the calculator reads *at run time*
(`VASP_PP_PATH`, `ASE_VASP_COMMAND`), applied process-wide at build time
because those calculators consult the environment when they calculate —
which is also why `ASE_CONFIG_PATH` is refused: ASE parses its config file
exactly once, at import, before any registry entry runs, so that
declaration would silently never apply. Applied env stays in *this*
process: job submission hands `sbatch` the environment as it was before any
registry entry ran, so the kept batch script means the same thing no matter
which engines the submitting process happened to build first. `probe` is a
cheap command proving the engine actually works here. And the slab-factory
aliases are deliberately strict: their spec (plus the caller's traced
options) is their whole cache identity, so `qe_calculator` refuses to fall
back to `[engines.qe]` or ASE config for `command`/`pseudo_dir`, and asks
for an explicit k-point policy (`kpts=`/`kspacing=`) — the task-level
k-point refusal only recognizes the literal name `qe`. Discovery order: an explicit path, else
`$SLAB_ENGINES`, else `~/.config/slab/engines.json` — a maintainer ships the
file at a shared path and exports `SLAB_ENGINES` from a module file. The full
worked example is
[examples/engines.example.json](https://github.com/tarbaugh/SLAB/blob/main/examples/engines.example.json).

<!-- no-verify -->
```bash
slab engines list      # built-ins + everything this cluster declares
slab engines verify    # run every entry's probe; exit nonzero on failure
```

Two refusals keep resolution honest. An entry that shadows a built-in name
(`mace`, `emt`, ...) is rejected loudly at load — it would silently win or lose
depending on resolution order, and either way you get a different engine than
someone intended. And a registry declaring a newer `layout_version` than the
client understands refuses rather than misreads.

You can exercise the whole chain locally — the registry is just a file, and
nothing says its calculator must live on a cluster:

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
('emt', 'lammps', 'lj', 'mace', 'qe', 'rootstock', 'emt-cluster')
emt-cluster registry:laptop 1.0
```

The declared `version` lands in the task recipe as provenance *and* in the
cache key. When a maintainer bumps `qe-delta` from 7.3 to 7.4, cached results
are honestly invalidated instead of the old engine's numbers being served —
and not just versions: any spec edit (options, env, calculator) changes the
fingerprint. Watch it happen:

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

How the cache key is built is [Caching & resume](caching-and-resume.md)'s
story; the point here is that engine identity is part of it.

## Rootstock checkpoint ids, served silently

On a cluster with a [Garden-AI/rootstock](https://github.com/Garden-AI/rootstock)
install, any canonical checkpoint id works *directly as the engine name* —
rootstock resolves the hosting environment and serves the model from a worker
subprocess, so your Python environment stays free of torch and model packages
(`pip install 'slab[rootstock]'` adds only a thin client):

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms,
    engine="mace-mp-0-medium",                        # a checkpoint id IS an engine
    calculator_options={"cluster": "delta", "device": "cuda"},
)
```

The install is found via `cluster=` or `root=` in `calculator_options`, else
rootstock's own defaults (`$ROOTSTOCK_ROOT`, `~/.config/rootstock/config.toml`).
Because resolution tries the registry first, a maintainer's alias always beats
a bare id. The explicit `engine="rootstock"` form remains for full control —
e.g. `calculator_options={"checkpoint": "uma:custom", "weights": ..., "cluster": "delta"}`
with your own weights. `relax` closes the worker subprocess when the task
finishes.

Cache identity for silently-served engines is deliberately the checkpoint id
plus the rootstock *client* version — not the install's path or hosting
environment. Rootstock's contract is that canonical ids are stable identities:
the same id on another install is the same computation, while cluster-side
internals (env rebuilds, in-place weight edits) are invisible to any client.

## Trust model

Stated plainly: registry entries execute maintainer-declared code and
environment variables as the calling user, and rootstock installs run
maintainer-built environments the same way. SLAB isolates *configuration*, not
*privilege* — trusting a cluster's `engines.json` is trusting its module farm.
`slab engines verify` tells you the declarations work; it does not tell you
they are benign.

When an engine misbehaves mid-run, the evidence survives — see
[Debugging failures](debugging-failures.md). For where the seam sits in the
overall design, see [Architecture](../architecture.md).
