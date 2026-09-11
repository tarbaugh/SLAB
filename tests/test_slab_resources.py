"""slab.resources: the budget, the envelope, the placeholders, and job sizes."""

from __future__ import annotations

import os

import pytest

from slab.config import Partition
from slab.errors import EngineNotAvailableError, JobSizeError
from slab.resources import (
    Budget,
    Envelope,
    JobSize,
    apply,
    budget,
    check_size,
    env_for,
    envelope,
    fill,
    job_size,
    placeholders,
)

SLAB_VARS = ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS")


@pytest.fixture()
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        *SLAB_VARS,
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
    ):
        monkeypatch.delenv(name, raising=False)
    # No nvidia-smi probe reaches a real device from the suite.
    monkeypatch.setattr("slab.resources._probed_gpus", lambda: ())


# -- budget --------------------------------------------------------------------


def test_budget_gpus_precedence(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA_VISIBLE_DEVICES first (empty means none), then SLURM's job or step
    gpus, then the probe, then none. SLURM's variables hold the node's global
    ids, which a cgroup-constrained job sees renumbered from zero, so only
    their count is taken and the ids become 0, 1, ..."""
    monkeypatch.setattr("slab.resources._probed_gpus", lambda: ("0", "1", "2", "3"))
    assert budget().gpus == ("0", "1", "2", "3")
    monkeypatch.setenv("SLURM_STEP_GPUS", "2,3")
    assert budget().gpus == ("0", "1")
    monkeypatch.setenv("SLURM_JOB_GPUS", "4-6")
    assert budget().gpus == ("0", "1", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert budget().gpus == ("1",)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert budget().gpus == ()


def test_budget_cpus_are_the_affinity_mask_or_the_machine(clean_env: None) -> None:
    found = budget().cpus
    assert found and found == tuple(sorted(set(found)))
    getter = getattr(os, "sched_getaffinity", None)
    expected = sorted(getter(0)) if getter is not None else list(range(os.cpu_count() or 1))
    assert list(found) == expected


# -- envelope ------------------------------------------------------------------


def test_envelope_precedence(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """SLAB_* first, then the budget, SLURM_NTASKS, and SLURM_CPUS_PER_TASK, then 1."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    whole = envelope()
    assert whole.cpus == budget().cpus and whole.gpus == ("0", "1")
    assert (whole.ntasks, whole.threads) == (1, 1)
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert (envelope().ntasks, envelope().threads) == (4, 2)
    monkeypatch.setenv("SLAB_CPUS", "2,3")
    monkeypatch.setenv("SLAB_GPUS", "")
    monkeypatch.setenv("SLAB_NTASKS", "2")
    monkeypatch.setenv("SLAB_THREADS", "1")
    assert envelope() == Envelope(cpus=(2, 3), gpus=(), ntasks=2, threads=1)
    monkeypatch.setenv("SLAB_NTASKS", "zero")
    monkeypatch.setenv("SLAB_THREADS", "-1")
    assert (envelope().ntasks, envelope().threads) == (4, 2)  # unusable: the fallback


def test_env_for_round_trips_through_envelope(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    given = Envelope(cpus=(4, 5, 6, 7), gpus=("1",), ntasks=2, threads=2)
    exported = env_for(given)
    assert exported == {
        "SLAB_CPUS": "4,5,6,7",
        "SLAB_GPUS": "1",
        "SLAB_NTASKS": "2",
        "SLAB_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "1",
    }
    for name, value in exported.items():
        monkeypatch.setenv(name, value)
    assert envelope() == given


def test_apply_pins_where_the_platform_allows(clean_env: None) -> None:
    if not hasattr(os, "sched_setaffinity"):
        assert apply(Envelope(cpus=(0,))) is False
        pytest.skip("no sched_setaffinity on this platform")
    before = os.sched_getaffinity(0)
    try:
        assert apply(Envelope(cpus=(min(before),))) is True
        assert os.sched_getaffinity(0) == {min(before)}
        assert apply(Envelope(cpus=(10**6,))) is False  # outside the mask: refused
    finally:
        os.sched_setaffinity(0, before)


# -- fill ----------------------------------------------------------------------


def test_fill_replaces_only_the_three_placeholders() -> None:
    env = Envelope(cpus=(0, 1, 2, 3), gpus=("0", "1"), ntasks=2, threads=2)
    assert fill("mpirun -np {ntasks} lmp -k on g {gpus} t {threads} -sf kk", env) == (
        "mpirun -np 2 lmp -k on g 2 t 2 -sf kk"
    )
    assert fill("srun pw.x", env) == "srun pw.x"
    assert fill("lmp -var n {other}", env) == "lmp -var n {other}"


def test_fill_leaves_shell_variables_alone() -> None:
    """A ${var} is the shell's; only a bare {ntasks} is SLAB's."""
    env = Envelope(cpus=(0, 1), ntasks=2, threads=1)
    line = "env OMP_NUM_THREADS=${threads} mpirun -np ${ntasks} pw.x"
    assert fill(line, env) == line
    assert placeholders(line) == []


def test_fill_refuses_gpus_when_the_launch_holds_none() -> None:
    with pytest.raises(EngineNotAvailableError, match=r"route 'lammps-gpu' asks for \{gpus\}"):
        fill("lmp -k on g {gpus} -sf kk", Envelope(cpus=(0,)), route="lammps-gpu")
    with pytest.raises(EngineNotAvailableError, match=r"asks for \{gpus\} but this launch"):
        fill("lmp -k on g {gpus}", Envelope(cpus=(0,)))
    # ${gpus} is not a request for GPUs.
    assert fill("lmp -k on g ${gpus}", Envelope(cpus=(0,))) == "lmp -k on g ${gpus}"


# -- job sizes -----------------------------------------------------------------

GPU = Partition.model_validate(
    {"gres": "gpu:a100:4", "node": {"cpus": 64, "gpus": 4, "mem": "480G"}, "max_nodes": 2}
)


def test_job_size_needs_the_rank_count() -> None:
    assert job_size() is None
    assert job_size(ntasks_per_node=8) == JobSize(ntasks_per_node=8)
    with pytest.raises(JobSizeError, match="pass ntasks_per_node"):
        job_size(gpus_per_node=2)


def test_check_size_accepts_what_fits() -> None:
    size = JobSize(nodes=2, ntasks_per_node=4, cpus_per_task=16, gpus_per_node=4, mem="480G")
    assert check_size(size, "gpu", GPU) is size


def test_check_size_refuses_too_many_nodes() -> None:
    with pytest.raises(JobSizeError, match=r"nodes=3 exceeds the 2 node\(s\).*max_nodes"):
        check_size(JobSize(nodes=3, ntasks_per_node=1), "gpu", GPU)


def test_check_size_refuses_too_many_cores() -> None:
    with pytest.raises(JobSizeError, match=r"= 80 exceeds the 64 cpus.*\.node\] cpus"):
        check_size(JobSize(ntasks_per_node=8, cpus_per_task=10), "gpu", GPU)


def test_check_size_refuses_too_many_gpus() -> None:
    with pytest.raises(JobSizeError, match=r"gpus_per_node=5 exceeds the 4 gpus.*\.node\] gpus"):
        check_size(JobSize(ntasks_per_node=1, gpus_per_node=5), "gpu", GPU)


def test_check_size_refuses_too_much_memory() -> None:
    with pytest.raises(JobSizeError, match=r"mem=1T exceeds the 480G.*\.node\] mem"):
        check_size(JobSize(ntasks_per_node=1, mem="1T"), "gpu", GPU)
    without_mem = Partition.model_validate({"node": {"cpus": 8}})
    with pytest.raises(JobSizeError, match="declares no node memory"):
        check_size(JobSize(ntasks_per_node=1, mem="1G"), "cpu", without_mem)


def test_check_size_refuses_a_partition_without_a_node_table() -> None:
    with pytest.raises(JobSizeError, match=r"add \[hpc\.partitions\.cpu\.node\] with cpus"):
        check_size(JobSize(ntasks_per_node=1), "cpu", Partition())


def test_job_size_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="not a SLURM memory form"):
        JobSize(ntasks_per_node=1, mem="lots")
    with pytest.raises(ValueError):
        JobSize(ntasks_per_node=0)


def test_budget_counts() -> None:
    assert Budget(cpus=(0, 1), gpus=()).counts == {"cpus": 2, "gpus": 0}
