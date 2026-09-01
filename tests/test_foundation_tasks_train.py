"""Tests for collect_training_data and train_potential (a fake gracemaker).

The fake is protocol-shaped from gracemaker's documented behavior: a fit
writes a ``seed/<N>/`` tree (N from the input.yaml, never assumed to be 1)
holding the log, the model architecture, and the metrics files; a ``-r -s``
re-invocation exports ``saved_model/``, and ``-sf`` adds ``FS_model.yaml``.
Failures exit nonzero with a python traceback. A gated test runs the real
trainer when ``$SLAB_TEST_GRACEMAKER`` names it.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read as ase_read

from foundation import Workspace
from foundation.errors import FoundationError
from foundation.tasks import collect_training_data, relax, single_point, train_potential
from slab.errors import BuilderError


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _fake_env(tmp_path: Path, name: str, *, version: str = "0.5.1", body: str) -> str:
    """A fake environment: a gracemaker script beside a python echoing a version."""
    bin_dir = tmp_path / name
    bin_dir.mkdir(exist_ok=True)
    _script(bin_dir / "python", f'echo "{version}"\n')
    return _script(bin_dir / "gracemaker", body)


_TRAINER_BODY = """\
seed=$(sed -n 's/^seed:[[:space:]]*//p' input.yaml)
[ -n "$seed" ] || seed=1
d="seed/$seed"
case "$*" in
  *" -r"*|"-r"*)
    mkdir -p "$d/saved_model/variables"
    echo "pb" > "$d/saved_model/saved_model.pb"
    echo "weights" > "$d/saved_model/variables/variables.data-00000-of-00001"
    case "$*" in *-sf*) echo "fs-model" > "$d/FS_model.yaml";; esac
    echo "exported"
    ;;
  *)
    mkdir -p "$d/checkpoints"
    echo "epoch 1 loss 0.5" > "$d/log.txt"
    echo "preset: FS" > "$d/model.yaml"
    {
      printf -- '- {"rmse/depa": 0.0123, "rmse/f_comp": 0.0450, "epoch": 1}\\n'
    } > "$d/train_metrics.yaml"
    {
      printf -- '- {"rmse/depa": 0.0199, "rmse/f_comp": 0.0700, "epoch": 1}\\n'
      printf -- '- {"rmse/depa": 0.0210, "rmse/f_comp": 0.0610, "epoch": 2}\\n'
    } > "$d/test_metrics.yaml"
    echo "ck" > "$d/checkpoints/checkpoint.best_test_loss.index"
    echo "training done"
    ;;
esac
"""


@pytest.fixture()
def fake_gracemaker(tmp_path: Path) -> str:
    return _fake_env(tmp_path, "grace-env", body=_TRAINER_BODY)


INPUT_YAML = """\
seed: 7
cutoff: 5.0
data:
  filename: training.extxyz
potential:
  preset: FS
fit:
  maxiter: 10
"""


def _labeled_runs(ws: Workspace) -> tuple[str, str]:
    """Two completed source runs: an lj relax and an lj single point."""
    atoms = bulk("Cu", cubic=True)
    atoms.rattle(0.05, seed=1)
    with ws.start_run(name="src-relax") as run_a:
        relax(atoms, engine="lj", fmax=0.5)
    with ws.start_run(name="src-sp") as run_b:
        single_point(bulk("Cu"), engine="lj")
    return run_a.id, run_b.id


# -- collect_training_data ----------------------------------------------------


def test_collect_gathers_recorded_labels_into_extxyz(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _labeled_runs(ws)
    with ws.start_run(name="collect") as run:
        path, info = collect_training_data([a, b])
    assert info["n_structures"] == 2
    assert info["engines"] == ["lj"]
    assert info["elements"] == ["Cu"]
    assert info["sources"] == {a: 1, b: 1}
    structures = ase_read(path, index=":")
    assert len(structures) == 2
    for atoms in structures:
        assert np.isfinite(atoms.get_potential_energy())
        assert atoms.get_forces().shape == (len(atoms), 3)
    ref = ws.runs.get_artifact(run.id, "training.extxyz")
    assert info["dataset_hash"] == ref.hash


def test_collect_frames_all_reads_the_trajectories(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _labeled_runs(ws)
    with ws.start_run(name="collect-all"):
        _, info = collect_training_data([a, b], frames="all")
    # The relax run contributes its trajectory frames on top of the two
    # task results (the endpoint frame appears twice, by documented design).
    assert info["n_structures"] > 2


def test_collect_refuses_mixed_engines_and_engine_filters(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    a, _ = _labeled_runs(ws)
    with ws.start_run(name="src-both") as run_c:
        single_point(bulk("Cu"), engine="emt")
        single_point(bulk("Al"), engine="lj")
    with ws.start_run(name="collect-mixed"):
        with pytest.raises(FoundationError, match="emt, lj"):
            collect_training_data([run_c.id])
        _, filtered = collect_training_data(
            [run_c.id], engine="emt", output="emt-only.extxyz"
        )
        assert filtered["engines"] == ["emt"]
        assert filtered["n_structures"] == 1
        _, mixed = collect_training_data(
            [run_c.id], allow_mixed=True, output="mixed.extxyz"
        )
        assert sorted(mixed["engines"]) == ["emt", "lj"]
        # A named engine filter that empties a run is a loud error, never a
        # silently thinned dataset.
        with pytest.raises(FoundationError, match="under engine 'emt'"):
            collect_training_data([a], engine="emt")


def test_collect_refuses_a_run_with_nothing_to_give(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with ws.start_run(name="empty") as empty:
        pass
    with (
        ws.start_run(name="collect-empty"),
        pytest.raises(FoundationError, match="no labeled structures"),
    ):
        collect_training_data([empty.id])


def test_collect_refuses_discarded_bytes_loudly(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _labeled_runs(ws)
    record = ws.runs.list_tasks(b)[0]
    ws.artifacts.discard(record.outputs["return[0]"])
    with (
        ws.start_run(name="collect-gone"),
        pytest.raises(FoundationError, match="discarded by retention"),
    ):
        collect_training_data([a, b])


def test_collect_dedupes_identical_results_across_runs(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with ws.start_run(name="one") as run_a:
        single_point(bulk("Cu"), engine="lj")
    with ws.start_run(name="two") as run_b:
        single_point(bulk("Cu"), engine="lj")  # cache hit: identical output hash
    with ws.start_run(name="collect-dup"):
        _, info = collect_training_data([run_a.id, run_b.id])
    assert info["n_structures"] == 1
    assert info["n_duplicates"] == 1


def test_collect_caches_on_the_resolved_content(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    a, b = _labeled_runs(ws)
    with ws.start_run(name="collect-1"):
        collect_training_data([a, b])
    with ws.start_run(name="collect-2") as second:
        collect_training_data([a, b])
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True


def test_collect_works_as_a_plain_function_outside_a_run(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAB_WORKSPACE", str(ws.root))
    a, b = _labeled_runs(ws)
    path, info = collect_training_data([a, b], output="plain.extxyz")
    assert Path(path).is_file()
    assert info["dataset_hash"] is None  # nothing traced outside a run


# -- train_potential ----------------------------------------------------------


def _dataset(tmp_path: Path, ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAB_WORKSPACE", str(ws.root))
    a, b = _labeled_runs(ws)
    path, _ = collect_training_data([a, b])
    return path


def test_train_refuses_bad_staging_up_front(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    with pytest.raises(BuilderError, match="never mentions the dataset's basename"):
        train_potential(
            "seed: 1\ndata:\n  filename: other.pkl.gz\n",
            dataset=dataset,
            command=fake_gracemaker,
        )
    yaml_file = tmp_path / "input.yaml"
    yaml_file.write_text(INPUT_YAML)
    with pytest.raises(BuilderError, match="looks like a path"):
        train_potential(str(yaml_file), dataset=dataset, command=fake_gracemaker)
    with pytest.raises(BuilderError, match="does not exist"):
        train_potential(INPUT_YAML, dataset="no-such.extxyz", command=fake_gracemaker)


def test_train_refuses_a_dirty_output_dir_before_the_fit(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    dirty = tmp_path / "potential"
    dirty.mkdir()
    (dirty / "old-model.yaml").write_text("stale")
    with pytest.raises(BuilderError, match="exists and is not empty"):
        train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker)


def test_train_success_keeps_evidence_and_exports_the_model(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    with ws.start_run(name="train") as run:
        model, info = train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker)
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert {
        "potential.log",
        "potential-model.yaml",
        "potential-train-metrics.yaml",
        "potential-test-metrics.yaml",
        "potential-saved-model.tar.gz",
    } <= names
    assert model["seed_dir"] == "seed/7"  # from the input.yaml, never assumed 1
    assert Path(model["saved_model"]).is_dir()
    assert (Path(model["saved_model"]) / "saved_model.pb").is_file()
    assert info["version"] == "0.5.1"
    # The final epoch's row wins; the fake writes two test rows on purpose.
    assert info["test_metrics"]["rmse/depa"] == pytest.approx(0.0210)
    assert info["test_metrics"]["epoch"] == 2
    assert info["train_metrics"]["rmse/f_comp"] == pytest.approx(0.0450)
    for kept_name, kept_hash in model["artifacts"].items():
        assert ws.runs.get_artifact(run.id, kept_name).hash == kept_hash


def test_train_export_fs_ships_the_fs_model(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    with ws.start_run(name="train-fs") as run:
        model, _ = train_potential(
            INPUT_YAML, dataset=dataset, command=fake_gracemaker, export_fs=True,
            label="fs-fit",
        )
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert "fs-fit-FS-model.yaml" in names
    assert Path(model["fs_model"]).is_file()


def test_train_tarball_is_deterministic(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    """Identical model bytes tar to identical bytes: the CAS dedupes them."""
    dataset = _dataset(tmp_path, ws, monkeypatch)
    with ws.start_run(name="t1"):
        model1, _ = train_potential(
            INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="fit-a"
        )
    with ws.start_run(name="t2"):
        model2, _ = train_potential(
            INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="fit-b"
        )
    hash1 = model1["artifacts"]["fit-a-saved-model.tar.gz"]
    hash2 = model2["artifacts"]["fit-b-saved-model.tar.gz"]
    assert hash1 == hash2


def test_train_failure_keeps_bounded_evidence(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    dying = _fake_env(
        tmp_path,
        "dying-env",
        body=(
            "mkdir -p seed/7/checkpoints\n"
            'echo "epoch 1" > seed/7/log.txt\n'
            'echo "ck" > seed/7/checkpoints/checkpoint.index\n'
            'echo "Traceback (most recent call last):"\n'
            "echo \"ValueError: loss diverged\"\n"
            "exit 1\n"
        ),
    )
    with (
        ws.start_run(name="train-fail") as run,
        pytest.raises(BuilderError, match="loss diverged") as excinfo,
    ):
        train_potential(INPUT_YAML, dataset=dataset, command=dying)
    notes = "\n".join(excinfo.value.__notes__)
    assert "training evidence kept as artifacts" in notes
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert "potential-failed.log" in names
    assert "potential-failed-checkpoint-checkpoint.index" in names
    assert "potential-failed-console.log" in names


def test_train_export_failure_names_the_completed_fit(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    exporterless = _fake_env(
        tmp_path,
        "exporterless-env",
        body=_TRAINER_BODY.replace(
            'echo "exported"',
            'echo "Traceback (most recent call last):"; echo "OSError: disk"; exit 1',
        ),
    )
    with (
        ws.start_run(name="train-noexp") as run,
        pytest.raises(BuilderError, match="OSError: disk") as excinfo,
    ):
        train_potential(INPUT_YAML, dataset=dataset, command=exporterless)
    assert any("fit itself completed" in n for n in excinfo.value.__notes__)
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert "potential.log" in names  # the fit's evidence survived the export


def test_train_refuses_seed_dir_ambiguity(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, ws, monkeypatch)
    twins = _fake_env(
        tmp_path,
        "twins-env",
        body='mkdir -p seed/1 seed/2\necho "done"\n',
    )
    with pytest.raises(BuilderError, match="more than one seed directory"):
        train_potential(INPUT_YAML, dataset=dataset, command=twins)


def test_train_caches_on_dataset_bytes_and_trainer_version(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gracemaker: str
) -> None:
    """The serializer hashes a dataset path by its string; cache_extra must
    hash the bytes, so changed content at the same path misses."""
    dataset = _dataset(tmp_path, ws, monkeypatch)
    with ws.start_run(name="c1"):
        train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="c")
    shutil.rmtree(tmp_path / "c")  # the cwd copy would refuse a second cold run
    with ws.start_run(name="c2") as hit:
        train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="c")
    assert ws.runs.list_tasks(hit.id)[0].cache_hit is True

    Path(dataset).write_text(Path(dataset).read_text() + "\n")  # same path, new bytes
    with ws.start_run(name="c3") as miss:
        train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="c2")
    assert ws.runs.list_tasks(miss.id)[0].cache_hit is False

    # A trainer upgrade behind the same command re-probes and misses too.
    grace = Path(fake_gracemaker)
    (grace.parent / "python").write_text('#!/bin/sh\necho "0.6.0"\n')
    grace.touch()  # new mtime -> new probe identity
    with ws.start_run(name="c4") as upgraded:
        train_potential(INPUT_YAML, dataset=dataset, command=fake_gracemaker, label="c3")
    assert ws.runs.list_tasks(upgraded.id)[0].cache_hit is False


# -- the real trainer, when present -------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SLAB_TEST_GRACEMAKER"),
    reason="set SLAB_TEST_GRACEMAKER to a real gracemaker executable",
)
def test_real_gracemaker_fits_a_tiny_fs_model(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real (CPU) FS fit on a handful of lj-labeled Cu cells."""
    command = os.environ["SLAB_TEST_GRACEMAKER"]
    monkeypatch.chdir(tmp_path)
    run_ids = []
    for step in range(6):
        atoms = bulk("Cu", cubic=True)
        atoms.rattle(0.02 * (step + 1), seed=step)
        with ws.start_run(name=f"label-{step}") as run:
            single_point(atoms, engine="lj")
        run_ids.append(run.id)
    with ws.start_run(name="real-train") as run:
        dataset, _ = collect_training_data(run_ids)
        input_yaml = (
            "seed: 1\n"
            "cutoff: 4.5\n"
            "data:\n"
            f"  filename: {Path(dataset).name}\n"
            "  test_size: 0.2\n"
            "  reference_energy: 0\n"
            "potential:\n"
            "  preset: FS\n"
            "  kwargs: {n_rad_base: 4, embedding_size: 8}\n"
            "fit:\n"
            "  loss: {energy: {weight: 1.0}, forces: {weight: 5.0}}\n"
            "  optimizer: L-BFGS-B\n"
            '  opt_params: {"maxcor": 50, "maxls": 20, "gtol": 1.e-8, "iprint": -1}\n'
            "  maxiter: 5\n"
            "  batch_size: 2\n"
        )
        model, info = train_potential(
            input_yaml, dataset=dataset, command=command, timeout_s=1800.0
        )
    assert Path(model["saved_model"]).is_dir()
    assert info["version"] is not None
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert "potential-saved-model.tar.gz" in names
