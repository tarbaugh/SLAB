"""A workspace recorded before the split still opens after it.

The split moved the run machinery from ``slab`` to ``foundation``, which
changes every cache key: a task's module path is part of its identity, and
``slab.tasks.relax`` became ``foundation.tasks.relax``. Recorded rows keep the
old strings, and that is correct — a recipe describes the computation that
actually ran. What must not change is whether the workspace opens, lists,
renders, and accepts new runs.

``tests/data/workspace-pre-split/`` holds the evidence: a real workspace
written by the single-package code at 39e49dc. See its ``README.txt``.
"""

import shutil
from pathlib import Path

from ase.build import bulk
from typer.testing import CliRunner

from foundation import Workspace, check, converged
from foundation._ops import run_details
from foundation.cli import app
from foundation.tasks import relax

runner = CliRunner()

FIXTURE = Path(__file__).parent / "data" / "workspace-pre-split"
PROMOTED = "01m0m08zrwmykakpqvzvr1fxj7"
VERIFIED = "01m0m090615m4v9z6bksew7bgj"


def _copy(tmp_path: Path) -> Path:
    """A writable copy of the fixture, holding only the workspace itself."""
    root = tmp_path / "pre-split"
    root.mkdir()
    shutil.copy(FIXTURE / "runs.db", root / "runs.db")
    shutil.copytree(FIXTURE / "cas", root / "cas")
    return root


def _fixture_atoms():
    """The structure the fixture's workflow relaxed, rebuilt exactly."""
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 1, 1)
    atoms.rattle(stdev=0.04, seed=11)
    return atoms


def test_pre_split_workspace_opens_and_lists(tmp_path: Path) -> None:
    with Workspace(_copy(tmp_path)) as ws:
        runs = {r.id: r for r in ws.runs.list_runs()}
    assert set(runs) == {PROMOTED, VERIFIED}
    assert runs[PROMOTED].state.value == "promoted"
    assert runs[VERIFIED].state.value == "verified"
    assert all(r.status.value == "completed" for r in runs.values())


def test_pre_split_run_details_render_with_their_historical_recipe(tmp_path: Path) -> None:
    with Workspace(_copy(tmp_path)) as ws:
        details = run_details(ws, PROMOTED)

    assert details["run"]["state"] == "promoted"
    assert [c["name"] for c in details["checks"]] == ["forces_converged"]
    assert details["checks"][0]["passed"] is True
    assert [a["name"] for a in details["artifacts"]] == ["cu.traj"]

    (task,) = details["tasks"]
    assert task["name"] == "relax"
    assert task["cache_hit"] is False
    # The recorded provenance is history and stays verbatim. Rewriting these
    # to the new module path would be a lie about what ran.
    assert task["recipe"]["module"] == "slab.tasks"
    assert task["recipe"]["qualname"] == "relax"
    assert task["recipe"]["slab"] == "0.1.0"
    assert "slab-stack" not in task["recipe"]


def test_pre_split_workspace_renders_through_the_foundation_cli(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    listed = runner.invoke(app, ["list", "-w", str(root)])
    assert listed.exit_code == 0
    assert PROMOTED[:10] in listed.output
    assert VERIFIED[:10] in listed.output

    shown = runner.invoke(app, ["show", PROMOTED[:10], "-w", str(root)])
    assert shown.exit_code == 0
    assert "promoted" in shown.output
    assert "forces_converged" in shown.output


def test_the_same_computation_misses_the_pre_split_cache(tmp_path: Path) -> None:
    """The module path moved, so the identity moved. A miss here is correct."""
    root = _copy(tmp_path)
    with Workspace(root) as ws:
        with ws.start_run(name="cu-relax", intent="after the split") as run:
            _, info = relax(_fixture_atoms(), engine="emt", fmax=0.05, label="cu")

            @check
            def forces_converged():
                return converged(info["fmax"], below=0.05)

        details = run_details(ws, run.id)

    (task,) = details["tasks"]
    assert task["cache_hit"] is False
    assert task["recipe"]["module"] == "foundation.tasks"
    assert task["recipe"]["slab-stack"] == "0.1.0"
    assert details["run"]["state"] == "verified"

    # The old rows are untouched by the new one landing beside them.
    with Workspace(root) as ws:
        assert len(ws.runs.list_runs()) == 3
        assert ws.runs.get(PROMOTED).state.value == "promoted"
