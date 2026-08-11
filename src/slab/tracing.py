"""The ``@task`` tracing decorator: plain functions, recorded runs.

Decorating a function changes nothing about how it is called. Outside a run
context it is exactly the function it was. Inside ``Workspace.start_run`` each
call is traced:

1. Bound arguments (defaults applied) are serialized, content-hashed, and
   stored — inputs are part of the recipe, so recompute-on-demand has roots.
2. A cache key is fingerprinted from the function's identity and environment:
   module + qualname, the source hash, a *bytecode* hash (which covers
   REPL/``exec``-defined functions where source is unavailable), the
   fingerprints of every closure cell value (two functions from the same
   factory with different bound parameters are different computations),
   resolved engine versions, the task's ``cache_extra`` contribution (see
   :func:`task`), and the input hashes. If a completed task with
   the same key exists *and its output bytes are still present*, the stored
   outputs are returned without executing — restart a crashed script and
   finished work is skipped. Discarded bytes mean a cache miss and honest
   recomputation. A closure cell that cannot be serialized makes the call
   *uncacheable* (computed honestly every time) rather than a collision risk.
   Functions whose behavior depends on mutated globals are outside what the
   key can see — treat globals as constants or pass them as arguments.
3. A provisional :class:`~slab.models.TaskRecord` (status ``running``) is
   committed *before* the function executes, so the run's input references
   are visible to concurrent retention sweeps for the whole execution; the
   row is finalized (``completed``/``failed``) when the call returns. A
   failed row records a one-line ``error`` plus a structured ``failure``
   record — trimmed traceback and any diagnostic notes the task attached
   via ``Exception.add_note`` (see :func:`slab.errors.failure_record`). The
   return value is serialized and stored — an exact ``tuple`` return is
   stored element-wise (``return[0]``, ``return[1]``, ...) so the common
   ``atoms, info = relax(...)`` pattern leaves per-value hashes; tuple
   *subclasses* (e.g. NamedTuple) are stored whole so cache restores preserve
   their type.

The DAG is *derived*, not declared: task B consuming task A's output is
visible because B's input hash equals A's output hash. No graph API to learn,
nothing to get wrong.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import platform
import textwrap
import types
from collections.abc import Callable, Sequence
from importlib import metadata
from typing import Any, ParamSpec, TypeVar, overload

from slab._version import __version__
from slab.artifacts import ArtifactStore
from slab.errors import SerializationError, failure_record
from slab.lifecycle import ExecutionStatus
from slab.models import TaskRecord, utcnow
from slab.runtime import ActiveRun, current_run
from slab.serialize import dumps, fingerprint, loads

P = ParamSpec("P")
R = TypeVar("R")

_MAX_LITERAL_PARAM = 500  # chars; longer strings are recorded by hash, not verbatim


@overload
def task(fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def task(
    *,
    name: str | None = None,
    engines: str | Sequence[str] = (),
    cache_extra: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def task(
    fn: Callable[P, R] | None = None,
    *,
    name: str | None = None,
    engines: str | Sequence[str] = (),
    cache_extra: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Make a plain function a traced task, without changing how it is called.

    Args:
        name: Display name for records (defaults to the function's name).
        engines: Distribution names whose installed versions should be pinned
            into the recipe and the cache key (e.g. ``engines=("mace", "ase")``).
            A version bump of a listed engine invalidates the cache — same
            inputs through a different engine version is a different result.
        cache_extra: Callable receiving the bound arguments (as a dict) at
            call time, returning a JSON-able dict folded into both the cache
            key and the recipe. Use it for computation identity that lives
            outside pip — e.g. the cluster engine registry's declared version
            for the engine an argument names: bumping the registry version
            then honestly invalidates the cache.

    Examples:
        >>> @task
        ... def double(x):
        ...     return 2 * x
        >>> double(21)  # outside a run: just the function, untraced
        42
    """

    def decorate(f: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(f)
        task_name = name or f.__name__
        engine_names = (engines,) if isinstance(engines, str) else tuple(engines)
        try:
            source = textwrap.dedent(inspect.getsource(f))
            code_hash: str | None = hashlib.sha256(source.encode()).hexdigest()
        except (OSError, TypeError):  # REPL/exec-defined functions have no source
            code_hash = None
        bytecode_hash = _code_identity(f.__code__).hexdigest()

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            active = current_run()
            if active is None:
                return f(*args, **kwargs)
            return _traced_call(
                active,
                f,
                signature,
                task_name,
                code_hash,
                bytecode_hash,
                engine_names,
                cache_extra,
                args,
                kwargs,
            )

        return wrapper

    return decorate if fn is None else decorate(fn)


def _traced_call(
    active: ActiveRun,
    f: Callable[P, R],
    signature: inspect.Signature,
    task_name: str,
    code_hash: str | None,
    bytecode_hash: str,
    engine_names: tuple[str, ...],
    cache_extra: Callable[[dict[str, Any]], dict[str, Any]] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    input_hashes: dict[str, str] = {}
    for arg_name, value in bound.arguments.items():
        payload = dumps(value)  # SerializationError here names a real problem: fix the input
        input_hashes[arg_name] = active.artifacts.put_bytes(payload)

    engine_versions = {engine: _dist_version(engine) for engine in engine_names}
    extra = dict(cache_extra(dict(bound.arguments))) if cache_extra is not None else {}
    recipe: dict[str, Any] = {
        "task": task_name,
        "module": f.__module__,
        "qualname": f.__qualname__,
        "code_sha256": code_hash,
        "engines": engine_versions,
        "python": platform.python_version(),
        "slab": __version__,
        "params": _params_lite(bound.arguments, input_hashes),
    }
    if extra:
        recipe["extra"] = extra
    closure_fp = _closure_fingerprints(f)
    cacheable = closure_fp is not None
    cache_key = fingerprint(
        {
            "module": f.__module__,
            "qualname": f.__qualname__,
            "code": code_hash,
            "bytecode": bytecode_hash,
            "closure": closure_fp if cacheable else f"uncacheable-{os.urandom(16).hex()}",
            "engines": engine_versions,
            "extra": extra,
            "inputs": input_hashes,
        }
    )

    cached = active.runs.find_cached_task(cache_key) if cacheable else None
    if cached is not None and all(active.artifacts.has(h) for h in cached.outputs.values()):
        now = utcnow()
        active.runs.add_task(
            TaskRecord(
                run_id=active.id,
                name=task_name,
                status=ExecutionStatus.COMPLETED,
                cache_hit=True,
                cache_key=cache_key,
                recipe=recipe,
                inputs=input_hashes,
                outputs=cached.outputs,
                started_at=now,
                finished_at=now,
            )
        )
        return _restore(active.artifacts, cached.outputs)  # type: ignore[return-value]

    # Provisional row BEFORE execution: the run's input references are visible
    # to concurrent gc for the whole (possibly very long) execution.
    provisional = active.runs.add_task(
        TaskRecord(
            run_id=active.id,
            name=task_name,
            status=ExecutionStatus.RUNNING,
            cache_key=cache_key,
            recipe=recipe,
            inputs=input_hashes,
            started_at=utcnow(),
        )
    )
    try:
        result = f(*args, **kwargs)
        outputs = _store_outputs(active.artifacts, result)
    except Exception as e:
        # The failed record carries the evidence an agent needs to correct and
        # retry: a one-line error for listings, and a structured failure record
        # (trimmed traceback + any add_note diagnostics) for show_run.
        record = failure_record(e)
        active.runs.update_task(
            provisional.seq,
            status=ExecutionStatus.FAILED,
            error=f"{record['type']}: {record['message']}",
            failure=record,
            finished_at=utcnow(),
        )
        raise
    active.runs.update_task(
        provisional.seq,
        status=ExecutionStatus.COMPLETED,
        outputs=outputs,
        finished_at=utcnow(),
    )
    return result


def _code_identity(code: types.CodeType) -> Any:
    """Hash of a code object's behavior-bearing parts (stable across processes).

    Covers bytecode, constants (nested code objects recursively), and the
    global names the code references — so exec/REPL-defined functions with no
    retrievable source still get distinct identities when their bodies differ.
    """
    hasher = hashlib.sha256()
    hasher.update(code.co_code)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            hasher.update(_code_identity(const).digest())
        else:
            hasher.update(repr(const).encode())
        hasher.update(b"\x00")
    hasher.update(" ".join(code.co_names).encode())
    return hasher


def _closure_fingerprints(f: Callable[..., Any]) -> list[str] | None:
    """Fingerprint each closure cell value, or None if any is unserializable.

    None means "uncacheable": a factory-made task whose bound state cannot be
    hashed must compute honestly every time — a collision would silently serve
    a wrong result, the one failure direction the cache must never take.
    """
    cells = f.__closure__ or ()
    fingerprints: list[str] = []
    for cell in cells:
        try:
            fingerprints.append(fingerprint(cell.cell_contents))
        except (SerializationError, ValueError):
            return None
    return fingerprints


def _store_outputs(artifacts: ArtifactStore, result: object) -> dict[str, str]:
    """Serialize a return value into the CAS; exact tuples go element-wise.

    Tuple *subclasses* (NamedTuple etc.) are stored whole so a cache restore
    reproduces the original type, not a bare tuple.
    """
    if type(result) is tuple:
        return {
            f"return[{i}]": artifacts.put_bytes(dumps(element)) for i, element in enumerate(result)
        }
    return {"return": artifacts.put_bytes(dumps(result))}


def _restore(artifacts: ArtifactStore, outputs: dict[str, str]) -> object:
    """Rebuild a return value from stored output blobs."""
    if set(outputs) == {"return"}:
        return loads(artifacts.get(outputs["return"]).read_bytes())
    ordered = sorted(outputs.items(), key=lambda item: int(item[0][7:-1]))
    return tuple(loads(artifacts.get(digest).read_bytes()) for _, digest in ordered)


def _params_lite(arguments: dict[str, Any], input_hashes: dict[str, str]) -> dict[str, Any]:
    """Human-readable parameters for the recipe: literals verbatim, data by hash."""
    lite: dict[str, Any] = {}
    for arg_name, value in arguments.items():
        is_literal = value is None or isinstance(value, (bool, int, float))
        if is_literal or (isinstance(value, str) and len(value) <= _MAX_LITERAL_PARAM):
            lite[arg_name] = value
        else:
            lite[arg_name] = {"$hash": input_hashes[arg_name]}
    return lite


def _dist_version(distribution: str) -> str | None:
    """Installed version of a distribution, or None if it is not installed."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
