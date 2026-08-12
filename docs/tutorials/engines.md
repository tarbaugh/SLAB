# Engines

SLAB implements no physics. Every engine — EMT toy, in-process MACE, a `pw.x`
binary, LAMMPS on a cluster, an MLIP served from a rootstock install — is
reached through exactly one seam: the ASE `Calculator` contract. This page
walks that seam from laptop built-ins to cluster registries.

## One seam, three sources

`get_calculator(engine, **options)` maps an engine *name* to a ready ASE
calculator. Three sources feed the mapping, tried in order:

1. **Built-ins** — `emt`, `lj` (ASE toys), `mace` (in-process, `slab[mace]`),
   `qe` (Quantum ESPRESSO's `pw.x`, no extra needed), `rootstock`
   (cluster-served MLIPs, `slab[rootstock]`).
2. **The cluster engine registry** — names a cluster maintainer declared in an
   `engines.json` that lives with the install (`qe`, `lammps`, `vasp`, curated
   MLIP aliases).
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
('emt', 'lj', 'mace', 'qe', 'rootstock')
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
    downloads the checkpoint to `~/.cache/mace`.

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

## The cluster engine registry

For LAMMPS, VASP, site-specific MLIP aliases, and curated site setups of the
built-in engines, SLAB generalizes rootstock's management pattern: the client
is only a bootstrap, and a JSON file that lives with the cluster declares how
each canonical name is built *here*. Workflow code says `engine="lammps"` and
runs unchanged on any cluster whose registry declares `lammps`.

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
      "calculator": "ase.calculators.espresso.Espresso",
      "env": {"ASE_CONFIG_PATH": "/sw/slab/ase-delta.ini"},
      "version": "7.3.1",
      "probe": ["pw.x", "-h"]
    }
  }
}
```

Every entry is a dotted path to an ASE calculator class or factory — even
rootstock enters through the same seam. `options` are defaults the caller
overrides key-by-key; `env` declares variables the code needs
(`ASE_CONFIG_PATH`, `VASP_PP_PATH`), applied process-wide at build time because
ASE calculators read configuration at run time; `probe` is a cheap command
proving the engine actually works here. Discovery order: an explicit path, else
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
('emt', 'lj', 'mace', 'qe', 'rootstock', 'emt-cluster')
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
