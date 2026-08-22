Pre-split workspace fixture
===========================

Recorded on 2026-08-22 with the single-package `slab` layout, at commit
39e49dc, immediately before the three-package split (slab / foundation /
mason). Its purpose is to prove that a workspace written by the old layout
still opens, lists, and renders under the new one.

Contents
--------
  runs.db      the SQLite run store (schema unchanged by the split)
  cas/         the content-addressed artifact store
  cu_relax.py  the workflow that produced it, kept for provenance

Runs
----
  01m0m08zrwmykakpqvzvr1fxj7  promoted  cu_relax  cache_hit=False
      intent: "fixture: baseline Cu relax recorded pre-split"
      promoted with reason "fixture: the promoted run a post-split
      workspace must still render"
  01m0m090615m4v9z6bksew7bgj  verified  cu_relax  cache_hit=True
      intent: "fixture: rerun proves the cache hit"

Both runs completed with 1/1 checks passing. The task recipe records
module "slab.tasks" and the recipe key "slab", which is exactly what the
compatibility test asserts: those strings are historical, and the split
must not rewrite them.

Do not regenerate this fixture. Its value is that it predates the split.
