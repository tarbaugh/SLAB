# Protocols & pseudopotential families

To run Quantum ESPRESSO well, you must choose cutoffs, a k-mesh, smearing,
and convergence thresholds. Those are physics decisions, and SLAB refuses to
make them silently. This page covers the two pieces that make those
decisions explicit, curated, and reproducible, both adopted from the AiiDA
ecosystem. Named input **protocols** come from
[aiida-quantumespresso](https://aiida-quantumespresso.readthedocs.io/en/stable/topics/protocol.html),
and installable **pseudopotential families** come from
[aiida-pseudo](https://github.com/aiidateam/aiida-pseudo).

## Pseudopotential families

A family is a curated, versioned set of one pseudopotential per element, with
per-element recommended cutoffs as metadata. Install one from the official
[Materials Cloud SSSP archive](https://www.materialscloud.org/discover/sssp).
Every file is verified against the published MD5 checksums, and one mismatch
aborts the whole install:

<!-- no-verify -->
```bash
slab pseudos install sssp --version 1.3 --functional PBEsol --precision efficiency
```

<!-- no-verify -->
```text
downloading SSSP 1.3 PBEsol efficiency from Materials Cloud ...
installed SSSP/1.3.0/PBEsol/efficiency (103 elements, digest b20acfb68f77) at
/Users/you/.local/share/slab/pseudos/SSSP_1.3.0_PBEsol_efficiency
```

Families land under `$SLAB_PSEUDOS`, which defaults to
`$XDG_DATA_HOME/slab/pseudos`, that is `~/.local/share/slab/pseudos`. They
are addressed by name, git-style on the version, so
`SSSP/1.3/PBEsol/efficiency` finds the installed `1.3.0` as long as exactly
one `1.3.x` exists. Ambiguity is refused, never guessed. `slab pseudos list`
shows the inventory, and `slab pseudos verify <name>` re-hashes every file
against the manifest.

From then on the `qe` engine takes the family instead of a directory:

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms, engine="qe",
    calculator_options={"pseudo_family": "SSSP/1.3/PBEsol/efficiency",
                        "input_data": {"system": {"ecutwfc": 30.0, "ecutrho": 240.0}}},
)
```

The element-to-file mapping comes from the family manifest, and the cache
identity upgrades from a directory path to the family name plus a **digest
of its per-element checksums**. That identity is content-derived, so it is
portable across machines.

Only SSSP installs today. PseudoDojo's archives are served over unverified
HTTP upstream, and SLAB does not download physics inputs over a channel it
cannot authenticate, so point `pseudo_dir=` at your own files instead.

## Protocols: fast, balanced, stringent

A protocol is a named, versioned bundle of input choices. SLAB adopts the
three from aiida-quantumespresso v4.10, and the values live in
[`slab/data/qe_protocols.json`](https://github.com/tarbaugh/SLAB/blob/main/src/slab/data/qe_protocols.json)
as data, not code:

```python
from slab.protocols import protocol_details

for key, value in sorted(protocol_details("balanced").items()):
    print(f"{key}: {value}")
```

```text
conv_thr_per_atom: 2e-10
degauss: 0.02
description: Balanced accuracy and speed for most production runs (AiiDA's default).
electron_maxstep: 80
etot_conv_thr_per_atom: 1e-05
forc_conv_thr: 0.0001
forc_conv_thr_ev_per_ang: 0.0025711033738162943
kpoints_distance: 0.15
mixing_beta: 0.4
name: balanced
pseudo_family: SSSP/1.3/PBEsol/efficiency
smearing: cold
```

Units are QE-native (Ry, Ry/atom, Ry/bohr; `kpoints_distance` in Å⁻¹).
`fast` trades accuracy for speed, with a 0.30 Å⁻¹ mesh and loose
thresholds, while `stringent` tightens everything and switches to the
`precision` family. AiiDA's pre-rename names are refused with a pointer, not
aliased, because the rename came with retuned values:

```python
from slab.protocols import ProtocolError, qe_protocol_options

try:
    protocol_details("moderate")
except ProtocolError as e:
    print(e)
```

```text
protocol 'moderate' was renamed 'balanced' (and retuned) in aiida-quantumespresso v4.10; ask for 'balanced' explicitly
```

## Expansion: from name to numbers

A protocol is structure-dependent, because cutoffs come from the elements
present and the k-mesh from the cell. Expansion is therefore an explicit,
atoms-aware call, and it returns a plain `calculator_options` dict of
**concrete values**:

<!-- no-verify -->
```python
from ase.build import bulk
from slab.protocols import qe_protocol_options
from slab.tasks import relax

atoms = bulk("Si", "diamond", a=5.43)
options = qe_protocol_options(atoms, protocol="balanced")
relaxed, info = relax(atoms, engine="qe", fmax=0.05, label="si",
                      calculator_options=options)
```

Expanded against the real SSSP family above, `options` is as follows, with
the pseudopotentials key elided:

<!-- no-verify -->
```json
{
 "pseudo_family": "SSSP/1.3.0/PBEsol/efficiency",
 "kpts": [14, 14, 14],
 "input_data": {
  "control": {"etot_conv_thr": 2e-05, "forc_conv_thr": 0.0001,
              "tprnfor": true, "tstress": true},
  "system": {"ecutwfc": 30.0, "ecutrho": 240.0,
             "occupations": "smearing", "smearing": "cold", "degauss": 0.02},
  "electrons": {"conv_thr": 4e-10, "electron_maxstep": 80, "mixing_beta": 0.4}
 }
}
```

The arithmetic is exactly AiiDA's:

- `ecutwfc` and `ecutrho` are element-wise maxima of the family's
  recommendations (Si: 30/240 Ry).
- The mesh is `ceil(|b_i| / 0.15)` per reciprocal vector, with 2π included,
  which gives 14³ for this 2-atom primitive cell.
- `etot_conv_thr` and `conv_thr` scale with the atom count (2 × 1e-5,
  2 × 2e-10), while `forc_conv_thr` does not, because it is per force
  component.

Insulators drop the smearing, so
`qe_protocol_options(atoms, protocol="balanced", electronic_type="insulator")`
sets fixed occupations. Overrides merge recursively, right wins, AiiDA-style:

<!-- no-verify -->
```python
options = qe_protocol_options(atoms, protocol="balanced",
                              overrides={"input_data": {"system": {"ecutwfc": 90.0}}})
```

To drive relaxation with ASE optimizers instead of pw.x's internal loop, use
the protocol's force threshold on the `fmax` scale, which `protocol_details`
reports as `forc_conv_thr_ev_per_ang` (balanced ≈ 0.0026 eV/Å).

The same expanded options drive `single_point`, the usual closing step of a
two-fidelity chain that relaxes under an MLIP and then runs one SCF under
`qe` on the relaxed geometry. The executed chain is in
[Engines](engines.md#two-fidelities-one-run).

## Why the cache never sees a protocol's name

The expanded numbers flow into the task as traced inputs, not the word
"balanced". Run the Si relax above and the recorded cache identity is:

<!-- no-verify -->
```json
{
 "engine": "qe", "source": "builtin", "version": "7.4.1",
 "command": ".../pw.x",
 "pseudo_dir": ".../slab/pseudos/SSSP_1.3.0_PBEsol_efficiency",
 "pseudo_family": "SSSP/1.3.0/PBEsol/efficiency",
 "pseudo_family_digest": "b20acfb68f77"
}
```

plus every expanded value in the traced `calculator_options`. If the protocol
data file is ever retuned, previously cached results stay valid for the
numbers they were computed with, and nothing is silently re-served under a
new meaning. Record the protocol name in the run's `intent` for the
narrative ("balanced protocol on Si"), because the recipe holds the physics.

Scope, honestly stated: SLAB adopts the `pw` base protocol. AiiDA's
spin/magnetization handling, 2D `assume_isolated` treatment, and the
PwRelaxWorkChain meta-convergence loop are not adopted (yet), so structures
must be fully periodic, and magnetism is yours to configure via
`input_data`.

Where to go next: for the engine seam these options feed, see
[Engines](engines.md#quantum-espresso), and for what happens when pw.x
rejects your protocol-expanded input, see
[Debugging failures](debugging-failures.md#when-the-engine-writes-files).
