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
    prefixed = render_sbatch("OMP_NUM_THREADS=4 slab run relax.py", job_name="env", config=hpc)
    assert "\nOMP_NUM_THREADS=4 slab run relax.py" in prefixed
    assert "srun OMP_NUM_THREADS" not in prefixed
    assert "launcher omitted" in prefixed
    # The explicit env wrapper is the same payload in different clothes.
    wrapped = render_sbatch("env OMP_NUM_THREADS=4 slab run relax.py", job_name="w", config=hpc)
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


@pytest.mark.parametrize("driver", ["slab", "foundation", "mason"])
def test_every_console_script_is_treated_as_a_driver(driver: str) -> None:
    """All three CLIs are single-process; srun would replicate any of them."""
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
    """Path(token).name is what decides, so /opt/venv/bin/foundation counts."""
    from slab.config import HpcConfig
    from slab.hpc import render_sbatch

    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"launcher": "srun", "ntasks_per_node": 64}},
        }
    )
    script = render_sbatch("/opt/venv/bin/foundation run wf.py", job_name="abs", config=hpc)
    assert "srun /opt/venv/bin/foundation" not in script
    assert "'foundation' is a single-process driver" in script


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
