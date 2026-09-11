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
    gres_gpus,
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
    with pytest.raises(EngineNotAvailableError, match=r"the lammps build 'gpu' asks for \{gpus\}"):
        fill("lmp -k on g {gpus} -sf kk", Envelope(cpus=(0,)), route="gpu")
    with pytest.raises(EngineNotAvailableError, match=r"asks for \{gpus\} but this launch"):
        fill("lmp -k on g {gpus}", Envelope(cpus=(0,)))
    # ${gpus} is not a request for GPUs.
    assert fill("lmp -k on g ${gpus}", Envelope(cpus=(0,))) == "lmp -k on g ${gpus}"


# -- job sizes -----------------------------------------------------------------

# A partition declares its caps through its own fields: nodes, ranks per node,
# cores per rank, the gpu count in gres, and memory per node.
GPU = Partition.model_validate(
    {"nodes": 2, "ntasks_per_node": 4, "cpus_per_task": 16, "gres": "gpu:a100:4", "mem": "480G"}
)


def test_job_size_needs_the_rank_count() -> None:
    assert job_size() is None
    assert job_size(ntasks_per_node=8) == JobSize(ntasks_per_node=8)
    with pytest.raises(JobSizeError, match="pass ntasks_per_node"):
        job_size(gpus_per_node=2)


def test_gres_gpus_counts_the_gpu_entry_alone() -> None:
    assert gres_gpus("gpu:a100:4") == 4
    assert gres_gpus("gpu:2") == 2
    assert gres_gpus("nvme:1,gpu:a100:8") == 8
    assert gres_gpus("gpu") is None
    assert gres_gpus("nvme:1") is None
    assert gres_gpus(None) is None


def test_check_size_accepts_what_fits() -> None:
    size = JobSize(nodes=2, ntasks_per_node=4, cpus_per_task=16, gpus_per_node=4, mem="480G")
    assert check_size(size, "gpu", GPU) is size


def test_check_size_refuses_too_many_nodes() -> None:
    """The cap is the partition's own nodes field, and the refusal names it."""
    with pytest.raises(
        JobSizeError,
        match=r"nodes=3 exceeds the 2 node\(s\) gpu declares \(\[hpc\.partitions\.gpu\] nodes\)",
    ):
        check_size(JobSize(nodes=3, ntasks_per_node=1), "gpu", GPU)


def test_check_size_refuses_too_many_ranks_per_node() -> None:
    with pytest.raises(
        JobSizeError,
        match=r"ntasks_per_node=5 exceeds the 4 ranks per node gpu declares "
        r"\(\[hpc\.partitions\.gpu\] ntasks_per_node\)",
    ):
        check_size(JobSize(ntasks_per_node=5), "gpu", GPU)


def test_check_size_caps_ranks_by_ntasks_when_only_that_is_set() -> None:
    """A partition that declares ntasks but no ntasks_per_node caps ranks with ntasks,
    and the refusal names that field."""
    spec = Partition.model_validate({"ntasks": 32})
    assert check_size(JobSize(ntasks_per_node=32), "cpu", spec).ntasks_per_node == 32
    with pytest.raises(
        JobSizeError,
        match=r"ntasks_per_node=33 exceeds the 32 ranks per node cpu declares "
        r"\(\[hpc\.partitions\.cpu\] ntasks\)",
    ):
        check_size(JobSize(ntasks_per_node=33), "cpu", spec)


def test_check_size_refuses_too_many_cores() -> None:
    """Cores per node are ranks times cpus_per_task; 4 x 20 = 80 exceeds 4 x 16 = 64."""
    with pytest.raises(
        JobSizeError,
        match=r"ntasks_per_node=4 x cpus_per_task=20 = 80 exceeds the 64 cores per node gpu "
        r"declares \(\[hpc\.partitions\.gpu\] ntasks_per_node x cpus_per_task\)",
    ):
        check_size(JobSize(ntasks_per_node=4, cpus_per_task=20), "gpu", GPU)
    # A partition with ranks but no cpus_per_task caps cores at one per rank.
    spec = Partition.model_validate({"ntasks_per_node": 8})
    with pytest.raises(JobSizeError, match=r"= 16 exceeds the 8 cores per node"):
        check_size(JobSize(ntasks_per_node=8, cpus_per_task=2), "cpu", spec)


def test_check_size_refuses_too_many_gpus() -> None:
    with pytest.raises(
        JobSizeError,
        match=r"gpus_per_node=5 exceeds the 4 gpus per node gpu declares "
        r"\(\[hpc\.partitions\.gpu\] gres\)",
    ):
        check_size(JobSize(ntasks_per_node=1, gpus_per_node=5), "gpu", GPU)


def test_check_size_refuses_gpus_on_a_partition_without_gres() -> None:
    """A partition with no gres has no gpus to give, whatever else it leaves unset."""
    with pytest.raises(
        JobSizeError,
        match=r"gpus_per_node=2 asks for gpus on cpu, which declares no gres "
        r"\(\[hpc\.partitions\.cpu\] gres\)",
    ):
        check_size(JobSize(ntasks_per_node=1, gpus_per_node=2), "cpu", Partition())


def test_check_size_refuses_too_much_memory() -> None:
    with pytest.raises(
        JobSizeError,
        match=r"mem=1T exceeds the 480G per node gpu declares \(\[hpc\.partitions\.gpu\] mem\)",
    ):
        check_size(JobSize(ntasks_per_node=1, mem="1T"), "gpu", GPU)


def test_check_size_leaves_unset_fields_uncapped() -> None:
    """A field the partition leaves unset is no cap; SLURM enforces its own limit."""
    size = JobSize(nodes=64, ntasks_per_node=1024, cpus_per_task=8, mem="4T")
    assert check_size(size, "cpu", Partition()) is size
    # A gres without a count allows any gpu count.
    uncounted = Partition.model_validate({"gres": "gpu"})
    eight = JobSize(ntasks_per_node=1, gpus_per_node=8)
    assert check_size(eight, "gpu", uncounted) is eight
    # Only gres and mem declared: ranks, cores, and nodes stay uncapped.
    partial = Partition.model_validate({"gres": "gpu:a100:4", "mem": "480G"})
    size = JobSize(nodes=8, ntasks_per_node=256, cpus_per_task=4, gpus_per_node=4, mem="480G")
    assert check_size(size, "gpu", partial) is size


def test_job_size_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="not a SLURM memory form"):
        JobSize(ntasks_per_node=1, mem="lots")
    with pytest.raises(ValueError):
        JobSize(ntasks_per_node=0)


def test_budget_counts() -> None:
    assert Budget(cpus=(0, 1), gpus=()).counts == {"cpus": 2, "gpus": 0}
