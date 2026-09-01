An offline Materials Project snapshot mounted on this machine: real MP
metadata and structures, no network, no API key, no `mp-api`. Reduce in
SQL first: `search_materials` (structured filters) or `query_materials`
(read-only SELECT) against the indexed `materials` table, then fetch the
few structures that survive with `fetch_structure(material_id)` in a
workflow script. Never enumerate the `cifs/` tree or pull broad
unfiltered rows into context; put a LIMIT on every query.

A SQL NULL means the property was not populated for that record. It is
never a measurement: NULL band gap does not mean metallic, NULL
magnetization does not mean non-magnetic, NULL energy-above-hull does not
mean stable. Filter with that in mind, and consult the `units` table
instead of inferring units from field names. Energy-per-atom and total
energies both appear; do not compare one against the other.

Identity is the pair (snapshot release, material_id). Report both with
every result, and never cache a conclusion by formula alone — entries
share compositions, and releases revise records. The release comes from
`get_material` context or the `list_engines` overview's `mp` entry.

Absence is absence. A material id the snapshot does not hold is a
finding to report, never a reason to attempt an online lookup — no such
route exists here. The CIFs are derivative interchange files; the
snapshot's parquet is its canonical metadata, and this install does not
read it, so surface a suspected CIF inconsistency instead of silently
reconciling it. The `source_license` column and the snapshot manifest
govern any redistribution of raw records.
