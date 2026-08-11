# SLAB — Simplest Layer for Atomistic Backends

Agent-native workflow orchestration for atomistic materials modeling.

The core idea: **runs are born ephemeral and promoted to permanent — never born
permanent and deleted.** Every run starts in quarantine with a TTL; machine-checkable
verification hooks gate it to `verified`; an explicit, one-command promotion makes it
permanent. Anything never promoted expires silently.

**Status: pre-alpha, under construction.** Current slice: run lifecycle state machine +
SQLite persistence. The full README (install, demo walkthrough, non-goals) lands with
the MVP demo.
