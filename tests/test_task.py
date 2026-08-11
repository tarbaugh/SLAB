"""Tests for the @task tracer: recording, hashing, caching, DAG derivation."""

from pathlib import Path

import pytest

from slab import ExecutionStatus, SerializationError, Workspace, fingerprint, task


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
    assert recipe["slab"] == "0.1.0"
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
    assert failed.outputs == {}
    assert ok.status is ExecutionStatus.COMPLETED
    assert ws.runs.get(run.id).status is ExecutionStatus.COMPLETED


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


def make_counted_task():
    calls = {"n": 0}

    @task
    def costly(x: float) -> float:
        calls["n"] += 1
        return x * 3

    return costly, calls


def test_cache_hit_within_and_across_runs(ws: Workspace) -> None:
    costly, calls = make_counted_task()
    with ws.start_run(name="first") as first:
        assert costly(2.0) == 6.0
        assert costly(2.0) == 6.0  # same call again: served from cache
    with ws.start_run(name="second") as second:
        assert costly(2.0) == 6.0  # cache spans runs in the workspace

    assert calls["n"] == 1
    first_records = ws.runs.list_tasks(first.id)
    assert [r.cache_hit for r in first_records] == [False, True]
    assert first_records[0].outputs == first_records[1].outputs
    (second_record,) = ws.runs.list_tasks(second.id)
    assert second_record.cache_hit is True


def test_cache_misses_on_different_input(ws: Workspace) -> None:
    costly, calls = make_counted_task()
    with ws.start_run():
        costly(2.0)
        costly(3.0)
    assert calls["n"] == 2


def test_cache_restores_tuple_returns(ws: Workspace) -> None:
    calls = {"n": 0}

    @task
    def multi(x: float) -> tuple[float, dict[str, float]]:
        calls["n"] += 1
        return x, {"fmax": 0.01}

    with ws.start_run():
        first = multi(1.0)
    with ws.start_run():
        again = multi(1.0)
    assert calls["n"] == 1
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
    attempts = {"n": 0}

    @task
    def flaky(x: int) -> int:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return x

    with ws.start_run():
        with pytest.raises(RuntimeError):
            flaky(1)
        assert flaky(1) == 1  # retried for real, not served from a poisoned cache
    assert attempts["n"] == 2


def test_discarded_bytes_mean_cache_miss_and_recompute(ws: Workspace) -> None:
    costly, calls = make_counted_task()
    with ws.start_run() as first:
        costly(2.0)
    (record,) = ws.runs.list_tasks(first.id)
    ws.artifacts.discard(record.outputs["return"])  # retention took the bytes

    with ws.start_run():
        assert costly(2.0) == 6.0  # recomputed on demand
    assert calls["n"] == 2
    assert ws.artifacts.has(record.outputs["return"])  # bytes restored
