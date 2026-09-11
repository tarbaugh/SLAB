"""SLURM layer tests — fake sbatch/squeue/sacct keep them cluster-free."""

import stat
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slab.cli import app
from slab.config import HpcConfig
from slab.hpc import (
    JobState,
    SchedulerError,
    SchedulerNotAvailableError,
    cancel,
    job_state,
    render_sbatch,
    submit,
)

runner = CliRunner()

HPC = HpcConfig.model_validate(
    {
        "cluster": "testcluster",
        "account": "abc-123",
        "default_partition": "cpu",
        "setup": ["module load quantum-espresso/7.4"],
        "partitions": {
            "cpu": {
                "time_limit": "04:00:00",
                "ntasks": 8,
                "launcher": "srun",
                "setup": ["export OMP_NUM_THREADS=1"],
            },
            "gpu": {
                "account": "gpu-999",
                "gres": "gpu:a100:4",
                "qos": "gpu",
                "sbatch_extra": ["--exclusive"],
            },
        },
    }
)

SIZED = HpcConfig.model_validate(
    {
        "account": "abc-123",
        "default_partition": "cpu",
        "partitions": {
            # A partition's own fields are its caps: nodes, ranks per node
            # (ntasks when only that is set), cores per rank, the gpu count in
            # gres, and mem. An unset field is no cap.
            "cpu": {
                "time_limit": "04:00:00",
                "nodes": 1,
                "ntasks": 64,
                "mem": "240G",
                "launcher": "srun",
            },
            "gpu": {
                "nodes": 2,
                "ntasks_per_node": 4,
                "cpus_per_task": 16,
                "gres": "gpu:a100:4",
                "mem": "480G",
            },
            "untyped": {"gres": "gpu:4"},
            "bare": {"gres": "gpu:4", "mem": "100G"},
        },
    }
)


def _fake(bin_dir: Path, name: str, body: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.fixture()
def scheduler_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "fake-slurm"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    return bin_dir


# -- rendering ---------------------------------------------------------------


def test_render_only_declared_fields_become_directives() -> None:
    script = render_sbatch("pw.x -in si.pwi", job_name="si", config=HPC)
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --ntasks=8" in script
    assert "--nodes" not in script  # unset -> omitted, no silent defaults
    assert "--mem" not in script
    assert script.startswith("#!/bin/bash -l\n")


def test_render_setup_order_and_launcher_prefix() -> None:
    script = render_sbatch("pw.x -in si.pwi", job_name="si", config=HPC)
    lines = script.splitlines()
    module = lines.index("module load quantum-espresso/7.4")
    omp = lines.index("export OMP_NUM_THREADS=1")
    payload = lines.index("srun pw.x -in si.pwi")
    assert module < omp < payload  # hpc-level setup, then partition's, then work


def test_render_partition_account_beats_hpc_account() -> None:
    script = render_sbatch("cmd", job_name="j", partition="gpu", config=HPC)
    assert "#SBATCH --account=gpu-999" in script
    assert "abc-123" not in script
    assert "#SBATCH --gres=gpu:a100:4" in script
    assert "#SBATCH --exclusive" in script
    assert "srun cmd" not in script  # gpu partition declares no launcher
    assert script.rstrip().endswith("cmd")


def test_render_time_and_output_overrides() -> None:
    script = render_sbatch(
        "cmd", job_name="j", config=HPC, time_limit="00:10:00", output="probe.log"
    )
    assert "#SBATCH --time=00:10:00" in script
    assert "04:00:00" not in script
    assert "#SBATCH --output=probe.log" in script


def test_render_unknown_partition_refuses() -> None:
    from slab.config import ConfigError

    with pytest.raises(ConfigError, match="not declared"):
        render_sbatch("cmd", job_name="j", partition="bigmem", config=HPC)


def test_render_reads_ambient_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\ntime_limit = "01:00:00"\n'
    )
    monkeypatch.chdir(tmp_path)
    script = render_sbatch("cmd", job_name="j")
    assert "#SBATCH --time=01:00:00" in script


# -- submission --------------------------------------------------------------


def test_submit_writes_script_and_parses_parsable_id(scheduler_bin: Path, tmp_path: Path) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "4242;testcluster"')
    job = submit("#!/bin/bash\ntrue\n", job_name="si", partition="cpu", directory=tmp_path / "wd")
    assert job.job_id == "4242"
    assert job.partition == "cpu"
    kept = Path(job.script_path)
    assert kept.read_text() == "#!/bin/bash\ntrue\n"
    assert kept.name == "si-4242.sbatch"  # per-job-id, so a second job cannot clobber it


def test_submit_falls_back_to_prose_job_id(scheduler_bin: Path, tmp_path: Path) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "Submitted batch job 777"')
    job = submit("x", job_name="j", partition="cpu", directory=tmp_path)
    assert job.job_id == "777"


def test_submit_without_sbatch_refuses(scheduler_bin: Path, tmp_path: Path) -> None:
    with pytest.raises(SchedulerNotAvailableError, match="sbatch"):
        submit("x", job_name="j", partition="cpu", directory=tmp_path)


def test_submit_failure_carries_stderr(scheduler_bin: Path, tmp_path: Path) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "sbatch: error: invalid partition" >&2\nexit 1')
    with pytest.raises(SchedulerError, match="invalid partition"):
        submit("x", job_name="j", partition="cpu", directory=tmp_path)


def test_submit_zero_exit_without_job_id_is_an_error(
    scheduler_bin: Path, tmp_path: Path
) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "cheerful nonsense"')
    with pytest.raises(SchedulerError, match="no job id"):
        submit("x", job_name="j", partition="cpu", directory=tmp_path)


def test_scheduler_timeout_is_a_loud_error(
    scheduler_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake(scheduler_bin, "sbatch", "true")

    def explode(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="sbatch", timeout=60)

    monkeypatch.setattr("slab.hpc.subprocess.run", explode)
    with pytest.raises(SchedulerError, match="did not answer within"):
        submit("x", job_name="j", partition="cpu", directory=tmp_path)


def test_job_name_is_validated_against_directive_injection(tmp_path: Path) -> None:
    with pytest.raises(SchedulerError, match="job name"):
        render_sbatch("cmd", job_name="evil\n#SBATCH --uid=0", config=HPC)
    with pytest.raises(SchedulerError, match="job name"):
        render_sbatch("cmd", job_name="../escape", config=HPC)
    with pytest.raises(SchedulerError, match="time limit"):
        render_sbatch("cmd", job_name="ok", config=HPC, time_limit="4h; rm -rf /")
    with pytest.raises(SchedulerError, match="whitespace"):
        render_sbatch("cmd", job_name="ok", config=HPC, output="a b")


def test_submit_validates_job_name_before_writing(scheduler_bin: Path, tmp_path: Path) -> None:
    _fake(scheduler_bin, "sbatch", 'echo "1"')
    workdir = tmp_path / "jobs"
    with pytest.raises(SchedulerError, match="job name"):
        submit("x", job_name="../../etc/cron.d/evil", partition="cpu", directory=workdir)
    assert not workdir.exists() or list(workdir.iterdir()) == []  # nothing written


# -- polling -----------------------------------------------------------------


def test_job_state_aggregates_array_rows(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", 'printf "COMPLETED\\nRUNNING\\nPENDING\\n"')
    status = job_state("123")
    assert status.state is JobState.RUNNING  # a partly-running array is running
    assert status.raw is not None and "RUNNING" in status.raw


def test_job_state_from_squeue(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", 'echo "RUNNING"')
    status = job_state("123")
    assert status.state is JobState.RUNNING
    assert status.raw == "RUNNING"
    assert not status.state.is_terminal


def test_job_state_falls_back_to_sacct(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", "exit 1")
    _fake(scheduler_bin, "sacct", 'echo "COMPLETED|0:0"')
    status = job_state("123")
    assert status.state is JobState.COMPLETED
    assert status.detail == "exit code 0:0"
    assert status.state.is_terminal


def test_job_state_cancelled_by_user_collapses(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", "exit 1")
    _fake(scheduler_bin, "sacct", 'echo "CANCELLED by 501|0:0"')
    assert job_state("123").state is JobState.CANCELLED


def test_job_state_unknown_raw_state_is_undetermined(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", 'echo "SPECIAL_SAUCE"')
    status = job_state("123")
    assert status.state is JobState.UNDETERMINED
    assert status.raw == "SPECIAL_SAUCE"


def test_job_state_without_sacct_is_undetermined_with_teaching(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", "exit 1")
    status = job_state("123")
    assert status.state is JobState.UNDETERMINED
    assert status.detail is not None and "sacct is not on PATH" in status.detail


def test_job_state_sacct_empty_record_is_undetermined(scheduler_bin: Path) -> None:
    _fake(scheduler_bin, "squeue", "exit 1")
    _fake(scheduler_bin, "sacct", "true")
    status = job_state("123")
    assert status.state is JobState.UNDETERMINED
    assert status.detail is not None and "no record" in status.detail


def test_job_state_without_squeue_refuses(scheduler_bin: Path) -> None:
    with pytest.raises(SchedulerNotAvailableError, match="squeue"):
        job_state("123")


# -- cancel ------------------------------------------------------------------


def test_cancel_invokes_scancel_and_is_idempotent(scheduler_bin: Path, tmp_path: Path) -> None:
    marker = tmp_path / "scancel-called"
    _fake(scheduler_bin, "scancel", f'echo "$@" > "{marker}"')
    cancel("123")
    assert marker.read_text().split() == ["-Q", "123"]
    _fake(scheduler_bin, "scancel", "exit 1")  # already-finished job: not an error
    cancel("123")


def test_cancel_without_scancel_refuses(scheduler_bin: Path) -> None:
    with pytest.raises(SchedulerNotAvailableError, match="scancel"):
        cancel("123")


# -- CLI ---------------------------------------------------------------------


def _config_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        "[hpc]\n"
        'cluster = "testcluster"\n'
        'default_partition = "cpu"\n'
        "[hpc.partitions.cpu]\n"
        'time_limit = "04:00:00"\n'
        'description = "general nodes"\n'
        "[hpc.partitions.gpu]\n"
        'gres = "gpu:a100:4"\n'
    )


def test_cli_hpc_partitions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hpc", "partitions"])
    assert result.exit_code == 0
    assert "cluster: testcluster" in result.output
    assert "(default)" in result.output
    assert "gpu:a100:4" in result.output


def test_cli_hpc_partitions_none_declared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hpc", "partitions"])
    assert result.exit_code == 0
    assert "no partitions declared" in result.output


def test_cli_hpc_render(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hpc", "render", "slab run relax.py", "--name", "si"])
    assert result.exit_code == 0
    assert "#SBATCH --job-name=si" in result.output
    assert "slab run relax.py" in result.output


def test_cli_hpc_render_without_partitions_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hpc", "render", "cmd"])
    assert result.exit_code == 1
    assert "no default_partition" in result.output


def test_cli_hpc_submit_and_status_and_cancel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduler_bin: Path
) -> None:
    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    _fake(scheduler_bin, "sbatch", 'echo "31337"')
    _fake(scheduler_bin, "squeue", 'echo "PENDING"')
    _fake(scheduler_bin, "scancel", "true")
    result = runner.invoke(app, ["hpc", "submit", "slab run relax.py", "--name", "si"])
    assert result.exit_code == 0
    assert "submitted job 31337 (si) to cpu" in result.output
    assert (tmp_path / "si-31337.sbatch").exists()  # kept under the job id
    result = runner.invoke(app, ["hpc", "status", "31337"])
    assert result.exit_code == 0
    assert "job 31337: pending" in result.output
    result = runner.invoke(app, ["hpc", "cancel", "31337"])
    assert result.exit_code == 0
    assert "cancel requested" in result.output


def test_cli_engines_list_shows_hpc_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "hpc partitions [testcluster]: cpu, gpu (default cpu)" in result.output


def test_engines_overview_hpc_section(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from slab._ops import engines_overview

    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    overview = engines_overview()
    assert overview["hpc"] is not None
    assert overview["hpc"]["cluster"] == "testcluster"
    assert list(overview["hpc"]["partitions"]) == ["cpu", "gpu"]


def test_engines_overview_hpc_error_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With an explicit registry path the config is only read for [hpc] —
    a broken file must be reported there, not hide the engines."""
    import json

    from slab._ops import engines_overview

    (tmp_path / "slab.toml").write_text("[hpc\n")
    registry = tmp_path / "engines.json"
    registry.write_text(json.dumps({"cluster": "t", "engines": {}}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    overview = engines_overview(registry)
    assert overview["hpc"] is None
    assert "not valid TOML" in overview["hpc_error"]


def test_driver_payload_is_never_launched() -> None:
    """srun on the slab driver would replicate the workflow once per task."""
    from slab.config import HpcConfig
    from slab.hpc import render_sbatch

    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"launcher": "srun", "ntasks_per_node": 64}},
        }
    )
    script = render_sbatch("slab run relax.py", job_name="wf", config=hpc)
    assert "\nslab run relax.py" in script
    assert "srun slab run" not in script
    assert "launcher omitted" in script
    engine_script = render_sbatch("pw.x -in si.pwi", job_name="pw", config=hpc)
    assert "srun pw.x -in si.pwi" in engine_script  # engines still get the launcher
    # An env assignment prefixing the payload must not hide the driver.
    prefixed = render_sbatch(
        "OMP_NUM_THREADS=4 slab run relax.py", job_name="env", config=hpc
    )
    assert "\nOMP_NUM_THREADS=4 slab run relax.py" in prefixed
    assert "srun OMP_NUM_THREADS" not in prefixed
    assert "launcher omitted" in prefixed
    # The explicit env wrapper is the same payload in different clothes.
    wrapped = render_sbatch(
        "env OMP_NUM_THREADS=4 slab run relax.py", job_name="w", config=hpc
    )
    assert "\nenv OMP_NUM_THREADS=4 slab run relax.py" in wrapped
    assert "srun env" not in wrapped
    assert "launcher omitted" in wrapped
    # ...but an env-wrapped engine command still gets the launcher.
    wrapped_engine = render_sbatch("env OMP_NUM_THREADS=4 pw.x", job_name="we", config=hpc)
    assert "srun env OMP_NUM_THREADS=4 pw.x" in wrapped_engine


def test_submission_env_restores_pre_registry_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sbatch --export=ALL would carry registry-engine residue into the job's
    fresh process (where an ASE_CONFIG_PATH that no-oped in-process WOULD
    apply); the submission env is the environment as it was before any
    registry entry ran."""
    import os

    import slab.engines as engines
    from slab.hpc import _submission_env

    monkeypatch.setattr(engines, "_APPLIED_ENV", {})
    assert _submission_env() is None  # nothing applied: env passes through as-is

    monkeypatch.setenv("SLAB_TEST_OVERWRITTEN", "registry-value")
    monkeypatch.setenv("SLAB_TEST_CREATED", "registry-value")
    monkeypatch.setenv("SLAB_TEST_USERSET", "user-fresh-export")
    monkeypatch.setattr(
        engines,
        "_APPLIED_ENV",
        {
            "SLAB_TEST_OVERWRITTEN": ("shell-value", "registry-value"),
            "SLAB_TEST_CREATED": (None, "registry-value"),
            "SLAB_TEST_USERSET": ("old-shell-value", "registry-value"),
        },
    )
    env = _submission_env()
    assert env is not None
    assert env["SLAB_TEST_OVERWRITTEN"] == "shell-value"  # unchanged residue: restored
    assert "SLAB_TEST_CREATED" not in env  # originally unset: dropped
    # A value the USER re-set after application is intent, not residue.
    assert env["SLAB_TEST_USERSET"] == "user-fresh-export"
    assert os.environ["SLAB_TEST_CREATED"] == "registry-value"  # process untouched


def test_the_console_script_is_treated_as_a_driver() -> None:
    """The CLI is single-process; srun would replicate the whole workflow."""
    driver = "slab"
    from slab.config import HpcConfig
    from slab.hpc import render_sbatch

    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"launcher": "srun", "ntasks_per_node": 64}},
        }
    )
    payload = f"{driver} run relax.py"
    script = render_sbatch(payload, job_name="wf", config=hpc)
    assert f"\n{payload}" in script
    assert f"srun {driver}" not in script
    # The comment names the driver it found, so the reader knows which one.
    assert f"# partition launcher omitted: '{driver}' is a single-process driver;" in script


def test_a_driver_reached_by_absolute_path_is_still_a_driver() -> None:
    """Path(token).name is what decides, so /opt/venv/bin/slab counts."""
    from slab.config import HpcConfig
    from slab.hpc import render_sbatch

    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"launcher": "srun", "ntasks_per_node": 64}},
        }
    )
    script = render_sbatch("/opt/venv/bin/slab run wf.py", job_name="abs", config=hpc)
    assert "srun /opt/venv/bin/slab" not in script
    assert "'slab' is a single-process driver" in script


def test_a_lookalike_command_is_not_a_driver() -> None:
    """Only the exact console-script names; 'foundationctl' is someone's tool."""
    from slab.config import HpcConfig
    from slab.hpc import render_sbatch

    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"launcher": "srun", "ntasks_per_node": 64}},
        }
    )
    script = render_sbatch("foundationctl sync", job_name="lookalike", config=hpc)
    assert "srun foundationctl sync" in script
    assert "launcher omitted" not in script


# -- sized rendering ---------------------------------------------------------


def test_render_without_a_size_is_unchanged() -> None:
    """The plain form is byte for byte what it was: the size is opt-in."""
    assert render_sbatch("slab run a.py", job_name="a", config=SIZED) == (
        "#!/bin/bash -l\n"
        "#SBATCH --job-name=a\n"
        "#SBATCH --partition=cpu\n"
        "#SBATCH --output=a-%j.out\n"
        "#SBATCH --account=abc-123\n"
        "#SBATCH --time=04:00:00\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks=64\n"
        "#SBATCH --mem=240G\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        "# partition launcher omitted: 'slab' is a single-process driver;\n"
        '# engines bring their own MPI (e.g. [engines.qe] command = "srun pw.x")\n'
        "slab run a.py"
    )
    assert render_sbatch("slab run a.py", job_name="a", config=SIZED, size=None) == (
        render_sbatch("slab run a.py", job_name="a", config=SIZED)
    )


def test_render_with_a_size_byte_for_byte() -> None:
    """The size's directives replace the partition's own, and the partition's
    ntasks is dropped because the size states the whole shape."""
    from slab.resources import JobSize

    size = JobSize(nodes=1, ntasks_per_node=8, cpus_per_task=4, mem="120G")
    assert render_sbatch("slab run a.py", job_name="a", config=SIZED, size=size) == (
        "#!/bin/bash -l\n"
        "#SBATCH --job-name=a\n"
        "#SBATCH --partition=cpu\n"
        "#SBATCH --output=a-%j.out\n"
        "#SBATCH --account=abc-123\n"
        "#SBATCH --time=04:00:00\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --ntasks-per-node=8\n"
        "#SBATCH --cpus-per-task=4\n"
        "#SBATCH --mem=120G\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        "# partition launcher omitted: 'slab' is a single-process driver;\n"
        '# engines bring their own MPI (e.g. [engines.qe] command = "srun pw.x")\n'
        "slab run a.py"
    )


def test_render_with_a_multi_node_size_byte_for_byte() -> None:
    """On a partition that declares nodes, ntasks_per_node, and cpus_per_task,
    a size within them replaces every one of those directives."""
    from slab.resources import JobSize

    size = JobSize(nodes=2, ntasks_per_node=2, cpus_per_task=8, gpus_per_node=2)
    assert render_sbatch("cmd", job_name="j", partition="gpu", config=SIZED, size=size) == (
        "#!/bin/bash -l\n"
        "#SBATCH --job-name=j\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --output=j-%j.out\n"
        "#SBATCH --account=abc-123\n"
        "#SBATCH --nodes=2\n"
        "#SBATCH --ntasks-per-node=2\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --mem=480G\n"
        "#SBATCH --gres=gpu:a100:2\n"
        "\n"
        "set -euo pipefail\n"
        "\n"
        "cmd"
    )


def test_render_sized_on_a_partition_with_only_gres_and_mem() -> None:
    """A partition that declares only gres and mem caps only those; the rest of
    the size renders as asked."""
    from slab.resources import JobSize

    size = JobSize(nodes=8, ntasks_per_node=128, cpus_per_task=2, gpus_per_node=4, mem="100G")
    rendered = render_sbatch("cmd", job_name="j", partition="bare", config=SIZED, size=size)
    assert (
        "#SBATCH --nodes=8\n#SBATCH --ntasks-per-node=128\n#SBATCH --cpus-per-task=2\n"
        "#SBATCH --mem=100G\n#SBATCH --gres=gpu:4\n"
    ) in rendered


def test_render_sized_gres_keeps_the_partition_type() -> None:
    from slab.resources import JobSize

    size = JobSize(ntasks_per_node=2, gpus_per_node=2)
    typed = render_sbatch("cmd", job_name="j", partition="gpu", config=SIZED, size=size)
    assert "#SBATCH --gres=gpu:a100:2\n" in typed
    assert "#SBATCH --mem=480G\n" in typed  # the size said nothing about mem
    untyped = render_sbatch("cmd", job_name="j", partition="untyped", config=SIZED, size=size)
    assert "#SBATCH --gres=gpu:2\n" in untyped
    none = render_sbatch(
        "cmd", job_name="j", partition="gpu", config=SIZED, size=JobSize(ntasks_per_node=2)
    )
    assert "--gres" not in none


@pytest.mark.parametrize(
    ("partition", "size", "message"),
    [
        (
            "cpu",
            {"nodes": 2, "ntasks_per_node": 1},
            r"nodes=2 exceeds the 1 node\(s\) cpu declares \(\[hpc\.partitions\.cpu\] nodes\)",
        ),
        (
            "gpu",
            {"ntasks_per_node": 5},
            r"ntasks_per_node=5 exceeds the 4 ranks per node gpu declares "
            r"\(\[hpc\.partitions\.gpu\] ntasks_per_node\)",
        ),
        (
            "cpu",
            {"ntasks_per_node": 65},
            r"ntasks_per_node=65 exceeds the 64 ranks per node cpu declares "
            r"\(\[hpc\.partitions\.cpu\] ntasks\)",
        ),
        (
            "gpu",
            {"ntasks_per_node": 4, "cpus_per_task": 17},
            r"= 68 exceeds the 64 cores per node gpu declares "
            r"\(\[hpc\.partitions\.gpu\] ntasks_per_node x cpus_per_task\)",
        ),
        (
            "gpu",
            {"ntasks_per_node": 1, "gpus_per_node": 8},
            r"gpus_per_node=8 exceeds the 4 gpus per node gpu declares "
            r"\(\[hpc\.partitions\.gpu\] gres\)",
        ),
        (
            "cpu",
            {"ntasks_per_node": 1, "gpus_per_node": 1},
            r"gpus_per_node=1 asks for gpus on cpu, which declares no gres "
            r"\(\[hpc\.partitions\.cpu\] gres\)",
        ),
        (
            "bare",
            {"ntasks_per_node": 1, "mem": "101G"},
            r"mem=101G exceeds the 100G per node bare declares \(\[hpc\.partitions\.bare\] mem\)",
        ),
    ],
)
def test_render_sized_refuses_before_rendering(partition: str, size: dict, message: str) -> None:
    """Each capped field refuses a size past it, naming the partition field,
    before anything renders."""
    from slab.errors import JobSizeError
    from slab.resources import JobSize

    with pytest.raises(JobSizeError, match=message):
        render_sbatch("cmd", job_name="j", partition=partition, config=SIZED, size=JobSize(**size))


def test_render_sized_leaves_unset_fields_uncapped() -> None:
    """untyped declares a gres count only, so nodes, ranks, cores, and mem are
    whatever the size asks."""
    from slab.resources import JobSize

    size = JobSize(nodes=16, ntasks_per_node=8, cpus_per_task=32, gpus_per_node=4, mem="2T")
    rendered = render_sbatch("cmd", job_name="j", partition="untyped", config=SIZED, size=size)
    assert "#SBATCH --nodes=16\n#SBATCH --ntasks-per-node=8\n#SBATCH --cpus-per-task=32\n" in (
        rendered
    )
    assert "#SBATCH --mem=2T\n#SBATCH --gres=gpu:4\n" in rendered


def _sized_config_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        "[hpc]\n"
        'default_partition = "gpu"\n'
        "[hpc.partitions.gpu]\n"
        "nodes = 2\n"
        "ntasks_per_node = 8\n"
        "cpus_per_task = 8\n"
        'gres = "gpu:a100:4"\n'
        'mem = "480G"\n'
        "[hpc.partitions.cpu]\n"
        'time_limit = "01:00:00"\n'
    )


def test_cli_hpc_render_size_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _sized_config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["hpc", "render", "slab run md.py", "--ntasks-per-node", "8", "--gpus-per-node", "2",
         "--cpus-per-task", "4", "--nodes", "2", "--mem", "200G"],
    )
    assert result.exit_code == 0, result.output
    assert "#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=8\n#SBATCH --cpus-per-task=4\n" in (
        result.output
    )
    assert "#SBATCH --mem=200G\n#SBATCH --gres=gpu:a100:2\n" in result.output
    refused = runner.invoke(app, ["hpc", "render", "cmd", "--gpus-per-node", "8"])
    assert refused.exit_code == 1 and "pass ntasks_per_node" in refused.output
    too_big = runner.invoke(app, ["hpc", "render", "cmd", "-n", "j", "--ntasks-per-node", "9"])
    assert too_big.exit_code == 1
    assert "exceeds the 8 ranks per node gpu declares ([hpc.partitions.gpu] ntasks_per_node)" in (
        too_big.output
    )
    # cpu declares no caps, so a size on it renders as asked.
    uncapped = runner.invoke(
        app, ["hpc", "render", "cmd", "-n", "j", "-p", "cpu", "--ntasks-per-node", "256"]
    )
    assert uncapped.exit_code == 0, uncapped.output
    assert "#SBATCH --ntasks-per-node=256\n" in uncapped.output
    no_gres = runner.invoke(
        app, ["hpc", "render", "cmd", "-n", "j", "-p", "cpu", "--ntasks-per-node", "1",
              "--gpus-per-node", "1"]
    )
    assert no_gres.exit_code == 1
    assert "asks for gpus on cpu, which declares no gres ([hpc.partitions.cpu] gres)" in (
        no_gres.output
    )


def test_cli_hpc_submit_size_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduler_bin: Path
) -> None:
    _sized_config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    _fake(scheduler_bin, "sbatch", 'echo "4242"')
    result = runner.invoke(
        app, ["hpc", "submit", "slab run md.py", "--name", "md", "--ntasks-per-node", "4",
              "--gpus-per-node", "4"]
    )
    assert result.exit_code == 0, result.output
    kept = (tmp_path / "md-4242.sbatch").read_text()
    assert "#SBATCH --ntasks-per-node=4\n" in kept and "#SBATCH --gres=gpu:a100:4\n" in kept
    refused = runner.invoke(
        app, ["hpc", "submit", "cmd", "--ntasks-per-node", "1", "--gpus-per-node", "5"]
    )
    assert refused.exit_code == 1
    assert "exceeds the 4 gpus per node gpu declares ([hpc.partitions.gpu] gres)" in refused.output


def test_cli_hpc_partitions_prints_no_node_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The caps are the partition's own fields, so there is no separate node line."""
    _sized_config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["hpc", "partitions"])
    assert result.exit_code == 0
    assert "node:" not in result.output and "node(s) per job" not in result.output
    assert "gpu:a100:4, mem 480G" in result.output
    assert len(result.output.strip().splitlines()) == 2  # one line per partition


def test_engines_overview_reports_partition_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab._ops import engines_overview

    _sized_config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    partitions = engines_overview()["hpc"]["partitions"]
    gpu = partitions["gpu"]
    assert (gpu["nodes"], gpu["ntasks_per_node"], gpu["cpus_per_task"]) == (2, 8, 8)
    assert (gpu["gres"], gpu["mem"]) == ("gpu:a100:4", "480G")
    cpu = partitions["cpu"]
    assert (cpu["nodes"], cpu["ntasks_per_node"], cpu["cpus_per_task"], cpu["mem"]) == (
        None, None, None, None,
    )
    assert "node" not in gpu and "max_nodes" not in gpu


def test_sized_gres_forms() -> None:
    from slab.hpc import sized_gres

    assert sized_gres("gpu:a100:4", 3) == "gpu:a100:3"
    assert sized_gres("gpu:a100", 1) == "gpu:a100:1"
    assert sized_gres("gpu", 2) == "gpu:2"
    assert sized_gres("gpu:2", 1) == "gpu:1"
    assert sized_gres(None, 2) == "gpu:2"
    assert sized_gres("gpu:a100:4", 0) is None


def test_allocated_tasks_and_cpu_budget_are_thin(monkeypatch: pytest.MonkeyPatch) -> None:
    from slab.hpc import allocated_tasks, cpu_budget
    from slab.resources import budget

    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "12")
    assert allocated_tasks() == 12
    monkeypatch.setenv("SLAB_NTASKS", "3")
    assert allocated_tasks() == 3
    assert cpu_budget() == len(budget().cpus)


def test_sized_gres_sizes_only_the_gpu_entry_of_a_list() -> None:
    """A partition's gres may list more than gpus; a sized job keeps the
    rest as written and sizes the gpu entry alone."""
    from slab.hpc import sized_gres

    assert sized_gres("gpu:a100:4,nvme:1", 2) == "gpu:a100:2,nvme:1"
    assert sized_gres("gpu:4,shard:8", 2) == "gpu:2,shard:8"
    assert sized_gres("nvme:1", 1) == "nvme:1,gpu:1"
    assert sized_gres("gpu:a100:4,nvme:1", 0) == "nvme:1"


def test_cli_hpc_cancel_settles_the_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scheduler_bin: Path
) -> None:
    """The front door's 'slab hpc cancel' takes --workspace: it fails the
    job's running runs, releases their reservations, and lists the memories
    written since the job started, one line each."""
    from foundation.models import Run
    from foundation.runtime import Workspace
    from slab_stack.cli import app as front_door

    _config_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))
    _fake(scheduler_bin, "scancel", "true")
    root = tmp_path / "ws"
    with Workspace(root) as ws:
        run = ws.runs.create(Run(name="si-relax", job_id="31337"))
        ws.runs.set_status(run.id, "running", pid=1, host="node7")
        bystander = ws.runs.create(Run(name="other", job_id="31338"))
        ws.runs.set_status(bystander.id, "running", pid=1, host="node7")
    from foundation import memory as memory_store

    memory_store.write("qe-on-node7", "pw.x wants -nk 1 there", "One pool.")

    result = runner.invoke(front_door, ["hpc", "cancel", "31337", "-w", str(root)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "cancel requested for job 31337"
    assert lines[1] == (
        f"failed  {run.id}  si-relax  job 31337 cancelled by the operator; the process died with it"
    )
    assert lines[2].startswith("memory  qe-on-node7  written 0s ago")
    assert lines[2].endswith("('slab memory show qe-on-node7' to review)")
    with Workspace(root) as ws:
        assert ws.runs.get(run.id).status.value == "failed"
        assert ws.runs.get(bystander.id).status.value == "running"
    assert (tmp_path / "memory" / "qe-on-node7.md").exists()
