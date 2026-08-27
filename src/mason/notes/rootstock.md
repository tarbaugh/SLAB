The cluster's pre-built MLIP serving layer. Any canonical checkpoint id its
install declares works directly as an engine name — `engine="mace-mp-0-medium"`,
`engine="uma-s-1p1"` — and the model runs in the site's own environment, so
nothing is installed into yours. Call `list_engines` to see the checkpoint
ids available here; a curated registry alias for the same model wins over
the bare id when both exist.

A served calculator runs in a worker subprocess. The `foundation.tasks`
functions close it automatically; only long-lived hand-rolled scripts need
to care. Prefer a served checkpoint over `engine="mace"` in-process whenever
one is declared: it needs no download and no extra dependencies.
