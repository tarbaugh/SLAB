"""Tests for the @task tracer: recording, hashing, caching, DAG derivation."""

import typing
from pathlib import Path

import pytest

from foundation import (
    ExecutionStatus,
    SerializationError,
    Workspace,
    current_run,
    fingerprint,
    task,
)


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


@task
def double(x: float) -> float:
    return 2 * x


@task
def add(a: float, b: float = 10.0) -> float:
    return a + b


@task
def relax(x: float) -> tuple[float, dict[str, float]]:
    return x * 0.5, {"fmax": 0.03}


def test_untraced_outside_run(ws: Workspace) -> None:
    assert double(21) == 42
    assert ws.runs.list_runs() == []  # nothing was recorded anywhere


def test_traced_call_records_task(ws: Workspace) -> None:
    with ws.start_run(name="demo") as run:
        assert double(21) == 42

    (record,) = ws.runs.list_tasks(run.id)
    assert record.name == "double"
    assert record.status is ExecutionStatus.COMPLETED
    assert record.cache_hit is False
    assert record.error is None
    assert set(record.inputs) == {"x"}
    assert record.inputs["x"] == fingerprint(21)
    assert set(record.outputs) == {"return"}
    assert record.outputs["return"] == fingerprint(42)
    assert ws.artifacts.has(record.inputs["x"])  # inputs stored: recompute roots
    assert ws.artifacts.has(record.outputs["return"])


def test_defaults_are_bound_and_hashed(ws: Workspace) -> None:
    with ws.start_run() as run:
        add(1.0)
    (record,) = ws.runs.list_tasks(run.id)
    assert set(record.inputs) == {"a", "b"}
    assert record.recipe["params"] == {"a": 1.0, "b": 10.0}


def test_recipe_contents(ws: Workspace) -> None:
    with ws.start_run() as run:
        double(3)
    (record,) = ws.runs.list_tasks(run.id)
    recipe = record.recipe
    assert recipe["task"] == "double"
    assert recipe["module"] == __name__
    assert recipe["qualname"] == "double"
    assert len(recipe["code_sha256"]) == 64
    assert recipe["slab-stack"] == "0.1.0"
    assert recipe["python"].count(".") == 2


def test_engine_versions_recorded_and_keyed(ws: Workspace) -> None:
    @task(engines=("pytest", "not-a-real-distribution"))
    def probe(x: int) -> int:
        return x

    with ws.start_run() as run:
        probe(1)
    (record,) = ws.runs.list_tasks(run.id)
    assert record.recipe["engines"]["pytest"] == pytest.__version__
    assert record.recipe["engines"]["not-a-real-distribution"] is None


def test_single_string_engines_not_split_into_chars(ws: Workspace) -> None:
    @task(engines="pytest")
    def probe(x: int) -> int:
        return x

    with ws.start_run() as run:
        probe(1)
    (record,) = ws.runs.list_tasks(run.id)
    assert list(record.recipe["engines"]) == ["pytest"]


def test_tuple_return_stored_elementwise(ws: Workspace) -> None:
    with ws.start_run() as run:
        structure, info = relax(1.0)
    assert (structure, info) == (0.5, {"fmax": 0.03})
    (record,) = ws.runs.list_tasks(run.id)
    assert set(record.outputs) == {"return[0]", "return[1]"}
    assert record.outputs["return[0]"] == fingerprint(0.5)


def test_dag_edge_derived_by_hash_equality(ws: Workspace) -> None:
    with ws.start_run() as run:
        y = double(3)
        add(y)
    first, second = ws.runs.list_tasks(run.id)
    assert first.outputs["return"] == second.inputs["a"]  # the edge, no graph API needed


def test_task_failure_recorded_and_raises(ws: Workspace) -> None:
    @task
    def explode(x: int) -> int:
        raise RuntimeError("SCF diverged")

    with ws.start_run() as run:
        with pytest.raises(RuntimeError, match="SCF diverged"):
            explode(1)
        recovered = double(2)  # script caught the error; the run continues
    assert recovered == 4

    failed, ok = ws.runs.list_tasks(run.id)
    assert failed.status is ExecutionStatus.FAILED
    assert failed.error == "RuntimeError: SCF diverged"
    assert failed.failure["type"] == "RuntimeError"
    assert 'raise RuntimeError("SCF diverged")' in failed.failure["traceback"]
    assert failed.outputs == {}
    assert ok.status is ExecutionStatus.COMPLETED
    assert ok.failure is None
    assert ws.runs.get(run.id).status is ExecutionStatus.COMPLETED


def test_task_failure_notes_reach_the_record(ws: Workspace) -> None:
    """A task that annotates its exception (add_note) gets the notes stored —
    the mechanism relax uses to attach last-known-state diagnostics."""

    @task
    def annotated_explosion() -> None:
        error = RuntimeError("forces exploded")
        error.add_note("atom 17: |F| = 812.4 eV/Å")
        raise error

    with ws.start_run() as run, pytest.raises(RuntimeError):
        annotated_explosion()
    (record,) = ws.runs.list_tasks(run.id)
    assert record.failure["notes"] == ["atom 17: |F| = 812.4 eV/Å"]
    assert "atom 17" in record.failure["traceback"]  # notes ride the traceback too


def test_unserializable_output_fails_the_task(ws: Workspace) -> None:
    @task
    def bad_output() -> object:
        return lambda: None

    with pytest.raises(SerializationError), ws.start_run() as run:
        bad_output()
    # the run itself is failed by the propagating exception
    assert ws.runs.get(run.id).status is ExecutionStatus.FAILED
    (record,) = ws.runs.list_tasks(run.id)
    assert record.status is ExecutionStatus.FAILED


def test_source_unavailable_falls_back_to_no_code_hash(ws: Workspace) -> None:
    namespace: dict[str, object] = {}
    exec("def dynamic(x):\n    return x + 5", namespace)
    dynamic = task(namespace["dynamic"])  # type: ignore[arg-type]
    with ws.start_run() as run:
        assert dynamic(1) == 6
    (record,) = ws.runs.list_tasks(run.id)
    assert record.recipe["code_sha256"] is None


def test_name_override(ws: Workspace) -> None:
    @task(name="scf")
    def underlying(x: int) -> int:
        return x

    with ws.start_run() as run:
        underlying(1)
    assert ws.runs.list_tasks(run.id)[0].name == "scf"


def test_params_lite_hashes_bulky_values(ws: Workspace) -> None:
    @task
    def consume(structure: dict, tag: str, n: int) -> int:  # type: ignore[type-arg]
        return n

    big = {"positions": list(range(100))}
    with ws.start_run() as run:
        consume(big, "si", 3)
    params = ws.runs.list_tasks(run.id)[0].recipe["params"]
    assert params["tag"] == "si"
    assert params["n"] == 3
    assert params["structure"] == {"$hash": fingerprint(big)}


# -- caching ---------------------------------------------------------------------------

# Execution counters live at module level ON PURPOSE: closure cells are part of
# the cache key (bound parameters change the computation), so a counter inside
# a closure would invalidate the cache on every call. Globals are documented as
# outside the key.
CALLS: dict[str, int] = {}


def make_counted_task(key: str):
    CALLS[key] = 0

    @task
    def costly(x: float) -> float:
        CALLS[key] += 1
        return x * 3

    return costly


def test_cache_hit_within_and_across_runs(ws: Workspace) -> None:
    costly = make_counted_task("hit")
    with ws.start_run(name="first") as first:
        assert costly(2.0) == 6.0
        assert costly(2.0) == 6.0  # same call again: served from cache
    with ws.start_run(name="second") as second:
        assert costly(2.0) == 6.0  # cache spans runs in the workspace

    assert CALLS["hit"] == 1
    first_records = ws.runs.list_tasks(first.id)
    assert [r.cache_hit for r in first_records] == [False, True]
    assert first_records[0].outputs == first_records[1].outputs
    (second_record,) = ws.runs.list_tasks(second.id)
    assert second_record.cache_hit is True


def test_cache_misses_on_different_input(ws: Workspace) -> None:
    costly = make_counted_task("miss")
    with ws.start_run():
        costly(2.0)
        costly(3.0)
    assert CALLS["miss"] == 2


def test_cache_restores_tuple_returns(ws: Workspace) -> None:
    CALLS["multi"] = 0

    @task
    def multi(x: float) -> tuple[float, dict[str, float]]:
        CALLS["multi"] += 1
        return x, {"fmax": 0.01}

    with ws.start_run():
        first = multi(1.0)
    with ws.start_run():
        again = multi(1.0)
    assert CALLS["multi"] == 1
    assert again == first
    assert isinstance(again, tuple) and isinstance(again[1], dict)


def test_cache_keyed_on_code_not_just_name(ws: Workspace) -> None:
    def fn(x: int) -> int:
        return x + 1

    v1 = task(fn)

    def fn(x: int) -> int:
        return x + 2

    v2 = task(fn)

    with ws.start_run() as run:
        assert v1(10) == 11
        assert v2(10) == 12  # same module+qualname, different source -> cache miss
    records = ws.runs.list_tasks(run.id)
    assert [r.cache_hit for r in records] == [False, False]
    assert records[0].cache_key != records[1].cache_key


def test_failed_tasks_never_populate_the_cache(ws: Workspace) -> None:
    CALLS["flaky"] = 0

    @task
    def flaky(x: int) -> int:
        CALLS["flaky"] += 1
        if CALLS["flaky"] == 1:
            raise RuntimeError("transient")
        return x

    with ws.start_run():
        with pytest.raises(RuntimeError):
            flaky(1)
        assert flaky(1) == 1  # retried for real, not served from a poisoned cache
    assert CALLS["flaky"] == 2


def test_discarded_bytes_mean_cache_miss_and_recompute(ws: Workspace) -> None:
    costly = make_counted_task("discard")
    with ws.start_run() as first:
        costly(2.0)
    (record,) = ws.runs.list_tasks(first.id)
    ws.artifacts.discard(record.outputs["return"])  # retention took the bytes

    with ws.start_run():
        assert costly(2.0) == 6.0  # recomputed on demand
    assert CALLS["discard"] == 2
    assert ws.artifacts.has(record.outputs["return"])  # bytes restored


# -- cache-key soundness (regression tests from adversarial review) --------------------


def _scaler_factory(factor: float):
    @task(name="scale")
    def scale(x: float) -> float:
        return factor * x

    return scale


def test_closure_parameterized_tasks_do_not_collide(ws: Workspace) -> None:
    """Two factory products with different bound parameters are different
    computations: serving one's cached output for the other would be a wrong
    scientific result."""
    with ws.start_run() as first:
        assert _scaler_factory(2.0)(10.0) == 20.0
    with ws.start_run() as second:
        assert _scaler_factory(3.0)(10.0) == 30.0  # not a stale 20.0
    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is False
    assert ws.runs.list_tasks(first.id)[0].cache_key != ws.runs.list_tasks(second.id)[0].cache_key


def test_same_closure_still_hits_cache(ws: Workspace) -> None:
    with ws.start_run():
        assert _scaler_factory(2.0)(10.0) == 20.0
    with ws.start_run() as second:
        assert _scaler_factory(2.0)(10.0) == 20.0
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True


def test_unserializable_closure_makes_call_uncacheable(ws: Workspace) -> None:
    CALLS["uncacheable"] = 0

    def apply_factory(fn):
        @task(name="apply")
        def apply(x: int) -> int:
            CALLS["uncacheable"] += 1
            return fn(x)

        return apply

    bump = apply_factory(lambda v: v + 1)
    with ws.start_run() as first:
        assert bump(1) == 2
    with ws.start_run() as second:
        assert bump(1) == 2  # computed honestly again, never served from cache
    assert CALLS["uncacheable"] == 2
    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is False


def test_exec_defined_tasks_keyed_by_bytecode(ws: Workspace) -> None:
    """Source-less functions (REPL/exec) must not collapse to one cache key."""
    ns1: dict[str, object] = {}
    exec("def dyn(x):\n    return x + 1", ns1)
    ns2: dict[str, object] = {}
    exec("def dyn(x):\n    return x * 100", ns2)
    v1 = task(ns1["dyn"])  # type: ignore[arg-type]
    v2 = task(ns2["dyn"])  # type: ignore[arg-type]
    with ws.start_run():
        assert v1(5) == 6
        assert v2(5) == 500  # not a stale 6 from v1's cache entry


class RelaxResult(typing.NamedTuple):  # module-level: local classes don't pickle
    energy: float
    converged: bool


def test_namedtuple_returns_survive_cache_restore(ws: Workspace) -> None:
    @task
    def fake_relax(x: float) -> RelaxResult:
        return RelaxResult(energy=-10.5 * x, converged=True)

    with ws.start_run():
        first = fake_relax(2.0)
    with ws.start_run() as second:
        again = fake_relax(2.0)
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True
    assert type(again) is RelaxResult
    assert again.energy == first.energy  # attribute access works on the restore


def test_cache_extra_contributes_to_key_and_recipe(ws: Workspace) -> None:
    """cache_extra captures computation identity that lives outside pip (e.g.
    a cluster registry's declared engine version): changing it must miss."""
    CALLS["extra"] = 0
    environment = {"version": "7.3"}

    @task(cache_extra=lambda arguments: dict(environment))
    def compute(x: int) -> int:
        CALLS["extra"] += 1
        return x * 2

    with ws.start_run() as first:
        assert compute(5) == 10
    with ws.start_run():
        assert compute(5) == 10  # same extra: cache hit
    assert CALLS["extra"] == 1
    assert ws.runs.list_tasks(first.id)[0].recipe["extra"] == {"version": "7.3"}

    environment["version"] = "7.4"  # the maintainer bumped the engine
    with ws.start_run() as third:
        assert compute(5) == 10
    assert CALLS["extra"] == 2  # honest recompute, not a stale hit
    assert ws.runs.list_tasks(third.id)[0].cache_hit is False


def test_provisional_row_visible_during_execution(ws: Workspace) -> None:
    """The tracer commits a running row before executing, so a concurrent
    reader (retention, a dashboard) sees the task and its input references."""
    observed: dict[str, object] = {}

    @task
    def probe(x: int) -> int:
        rows = ws.runs.list_tasks(current_run().id)  # type: ignore[union-attr]
        observed["status"] = rows[0].status
        observed["inputs"] = rows[0].inputs
        return x

    with ws.start_run():
        probe(7)
    assert observed["status"] is ExecutionStatus.RUNNING
    assert observed["inputs"] == {"x": fingerprint(7)}


def test_a_refusal_before_the_call_leaves_no_orphan_bytes(ws: Workspace) -> None:
    """cache_extra (an unknown pseudo family, an unreadable dataset) runs
    before any input lands in the store, so a refused call leaves nothing
    that no row will ever name."""

    def refuse(arguments: dict[str, object]) -> dict[str, object]:
        raise ValueError("no such pseudo family")

    @task(cache_extra=refuse)
    def picky(x: float) -> float:
        return x

    with pytest.raises(ValueError, match="no such pseudo family"), ws.start_run(name="refused"):
        picky(2.0)
    assert list(ws.artifacts.hashes()) == []
