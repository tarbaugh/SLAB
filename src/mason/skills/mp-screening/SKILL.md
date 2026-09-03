---
name: mp-screening
description: Screen candidate materials from the offline Materials
  Project snapshot — indexed SQL search, shortlist hygiene (deprecated,
  theoretical, hull bands), fetch structures, compute. Use when asked to
  find, screen, or rank known materials by composition or reported
  properties, or to start a study from a Materials Project entry.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Screening from the Materials Project snapshot

The snapshot is a local, read-only copy of Materials Project metadata
and structures. There is no network and no online fallback: what the
snapshot holds is the entire search space, and a missing entry is a
finding to report. Every step here is cheapest in this order: SQL
first, one metadata record second, structure files last.

## 1. Search, then shortlist

Reduce the candidate set with `search_materials` before touching any
structure. Filters compose: `elements` (all must be present, so Fe and
O also return ternaries; use `nelements` or `chemsys` for an exact
composition space when the snapshot has them), `exclude_elements`, and
column comparisons with `__lte`/`__gte`/`__lt`/`__gt`/`__ne` suffixes:

    search_materials(filters={"elements": ["Fe", "O"], "nelements": 2,
                              "energy_above_hull__lte": 0.025,
                              "energy_above_hull__ne": None},
                     columns=["material_id", "formula_pretty",
                              "energy_above_hull", "band_gap"],
                     order_by="energy_above_hull", limit=25)

For queries the filters cannot express (aggregates, joins), use
`query_materials` with one SELECT and an explicit LIMIT. A wrong column
name is refused with the real column list, so one failed call teaches
the schema. Do not list the `cifs/` tree, and do not pull thousands of
rows into context; shortlists of tens are the working size.

Shortlist hygiene, applied before any structure is fetched:

- Drop retired records. When the schema has a `deprecated` column,
  filter `deprecated = 0`; Materials Project hides those rows by
  default and a snapshot may not.
- Say what "stable" means. `is_stable = 1` (booleans are integers) is
  on the hull. `energy_above_hull` is in eV/atom: observed metastable
  phases have a median near 15 meV/atom and a 90th percentile near
  67 meV/atom, so 25 to 50 meV/atom is "plausibly synthesizable" and
  above 200 meV/atom is not a candidate. The band is chemistry
  dependent; state the one you used.
- "Known materials" means `theoretical = 0` when the column exists
  (an experimental structure matched the entry); without it, say the
  shortlist mixes predicted and observed compounds.
- NULL is "not populated", never a value: NULL band gap is not
  metallic, NULL magnetization is not non-magnetic, NULL
  energy-above-hull is not stable. Add `"column__ne": None` when a
  property must be present, and order by a column only after filtering
  its NULLs, because SQLite sorts NULL first. Consult the `units` table
  (`query_materials`) rather than inferring units from names.

Read the snapshot's numbers for what they are. Band gaps are PBE (or
r2SCAN) values, about 40% below experiment, and a 0.0 gap is weak
evidence of a metal because several known insulators come out metallic;
a filter at 1.0 eV targets roughly 1.7 eV measured. Hull energies are
corrected, mixed-functional quantities (GGA, GGA+U, and r2SCAN under the
MP2020 scheme) and are never comparable to your own engine's total
energies; a hull distance you compute needs the same references and
corrections. Structures are DFT equilibria with lattice constants 1 to
3% above experiment (PBE) or at a different equilibrium (r2SCAN);
record the functional column when the snapshot carries one, and never
report a snapshot lattice constant as experimental.

## 2. Fetch and compute

Fetch shortlisted structures inside the workflow script, traced:

    from foundation.tasks import fetch_structure, relax, single_point

    atoms, meta = fetch_structure("mp-149")
    relaxed, opt = relax(atoms, engine="mace-mp-0-medium")

Then the canonical two-fidelity chain: relax under a cheap engine (a
served MLIP checkpoint id from `list_engines`), `single_point` under
the expensive one. `fetch_structure` keeps the consumed CIF with the
run, and its cache key carries the snapshot release, so a newer
snapshot honestly recomputes.

## 3. Verify and report

- Report every result as the pair (snapshot release, material_id) —
  `meta["release"]` from `fetch_structure`, or the `mp` entry in
  `list_engines`. Never a formula alone: entries share compositions,
  releases revise and merge records, and identifiers change between
  releases.
- Snapshot properties are reference metadata, not your results. When
  you compute a quantity the snapshot also lists, report both values
  and the run id; do not average them or silently prefer one.
- State the hygiene filters and the hull band with the shortlist.
- Cite run ids for computed numbers, as always.

## When not to use this

- No `[builders.mp]` root configured: the tools are absent and the
  snapshot does not exist here. Say so; do not go looking for one.
- Site-resolved magnetic detail, charged defects, or full nested MP
  provenance: the snapshot's parquet is its canonical metadata and this
  install does not read it. Surface the limit instead of working around
  it, and surface (never silently reconcile) a CIF that contradicts the
  metadata.
- Materials not in the snapshot: build them with the atomsk skills
  instead of searching for a lookalike entry.
