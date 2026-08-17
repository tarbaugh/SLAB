# Configuration templates

Fill-in-the-blank starting points for putting SLAB on a new machine. Copy,
fill, keep local: any file named `slab.toml` or `engines.json` inside this
checkout is gitignored (only these templates are exempt), so machine facts —
accounts, partitions, paths, commands — never end up in a commit.

| template | copy to | read by |
|---|---|---|
| `slab.toml` | `./slab.toml` (project layer), `~/.config/slab/config.toml` (user layer), or any path `$SLAB_SITE_CONFIG` points at (site layer) | the layered config; inspect with `slab config show` |
| `engines.json` | `~/.config/slab/engines.json`, or any path `$SLAB_ENGINES` or `[paths] engines` points at | the engine registry; inspect with `slab engines list` |

`slab.toml` here is byte-for-byte the text `slab config init` writes (a test
pins the two together). Every key ships commented out with its meaning
inline — uncomment what your machine needs; whatever stays commented falls
back to built-in defaults. `engines.json` is only needed for codes slab has
no built-in for (VASP) or for curated site aliases; the built-in `qe` and
`lammps` engines are configured in `slab.toml` itself.

The walkthrough for a new cluster is
[Configuring SLAB for your HPC](../docs/tutorials/hpc-config.md); the
registry's rules and semantics are in
[Engines](../docs/tutorials/engines.md).
