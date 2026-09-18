Quantum ESPRESSO (`pw.x`), plane-wave DFT. This is the expensive, trusted
engine, and it is the verification leg of the two-fidelity flow.

Never invent cutoffs or k-meshes. Expand a named protocol:

    from slab.protocols import qe_protocol_options
    options = qe_protocol_options(atoms, protocol="balanced")

and pass the result as `calculator_options`. The protocols are `fast` (smoke
test), `balanced` (production default), and `stringent` (publishable). Each
protocol also names its pseudopotential family; `list_engines` shows the
families installed on this machine. If the protocol's family is not
installed, the expansion fails loudly — pick an installed family with the
`family=` argument instead of hand-picking files. For non-metals pass
`electronic_type="insulator"` (fixed occupations instead of smearing), and
adjust single parameters with `overrides=`, never by editing the expansion
by hand.

Cost grows steeply with atom count, k-point density, and cutoff. A single
point on tens of atoms takes minutes to hours on a node, so anything past a
few minutes goes through `submit_job`. Each calculation runs in slab-managed
scratch. On failure the input, output, and the parsed `Error in routine`
message are kept as run evidence — read them with `show_run` before retrying.

The build follows the slice, as for LAMMPS. Where the `qe` entry of
`list_engines` shows a `gpu` build, a launch sized with `gpus=` runs it,
and an unsized launch runs the plain build. The gpu build runs one MPI
rank per GPU, so size a GPU launch with `gpus=` alone or with `ntasks`
equal to `gpus`. Never name a build; `engine="qe"` is all you pass.
