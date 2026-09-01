"""Gracemaker builder tests — a fake gracemaker keeps them TensorFlow-free.

The fakes are protocol-shaped from gracemaker's documented behavior: a
python CLI that exits nonzero with a traceback on failure, has no
``--version`` flag (the version probe asks the owning environment for the
``tensorpotential`` distribution version instead), and trains inside a
``seed/<N>/`` working tree. The real-execution vehicle is the
``$SLAB_TEST_GRACEMAKER``-gated test in ``test_foundation_tasks_train.py``.
"""

import os
import re
import time
from pathlib import Path

import pytest

from slab.errors import BuilderError, BuilderNotAvailableError
from slab.gracemaker import (
    GracemakerOutcome,
    describe_gracemaker,
    error_lines,
    gracemaker_command,
    gracemaker_setup,
    gracemaker_version,
    run_gracemaker,
)

TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/grace/bin/gracemaker", line 8, in <module>\n'
    "    sys.exit(main())\n"
    "KeyError: 'cutoff'\n"
)


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _env_with_python(tmp_path: Path, *, version: str = "0.5.1") -> str:
    """A fake environment bin dir: a gracemaker script with a python sibling."""
    bin_dir = tmp_path / "env-bin"
    bin_dir.mkdir(exist_ok=True)
    _script(bin_dir / "python", f'echo "{version}"\n')
    return _script(bin_dir / "gracemaker", 'echo "training"\n')


# -- command and setup resolution ---------------------------------------------


def test_command_resolution_prefers_call_then_config_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert gracemaker_command() == "gracemaker"
    assert gracemaker_command("/opt/grace/bin/gracemaker") == "/opt/grace/bin/gracemaker"
    (tmp_path / "slab.toml").write_text(
        "[builders.gracemaker]\n"
        'command = "gracemaker-config"\n'
        'setup = ["module load cuda"]\n'
    )
    monkeypatch.chdir(tmp_path)
    assert gracemaker_command() == "gracemaker-config"
    assert gracemaker_command("per-call") == "per-call"
    assert gracemaker_setup() == ("module load cuda",)
    assert gracemaker_setup("override") == ("override",)


# -- version probe ------------------------------------------------------------


def test_version_probe_asks_the_sibling_python(tmp_path: Path) -> None:
    """A console script's environment is its parent directory; the probe must
    use the interpreter beside the script, never a bare ``python`` from
    PATH (which would be SLAB's own environment)."""
    fake = _env_with_python(tmp_path)
    assert gracemaker_version(command=fake) == "0.5.1"


def test_version_probe_follows_the_shebang_without_a_sibling(tmp_path: Path) -> None:
    interpreter_dir = tmp_path / "pybin"
    interpreter_dir.mkdir()
    _script(interpreter_dir / "python", 'echo "0.6.0"\n')
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    script = lonely / "gracemaker"
    script.write_text(f"#!{interpreter_dir / 'python'}\n")
    script.chmod(0o755)
    assert gracemaker_version(command=str(script)) == "0.6.0"


def test_version_probe_degrades_to_none(tmp_path: Path) -> None:
    assert gracemaker_version(command=str(tmp_path / "no-such-gracemaker")) is None
    # A sibling python that fails, or prints something that is not a version.
    bad = tmp_path / "bad-bin"
    bad.mkdir()
    _script(bad / "python", "exit 1\n")
    failing = _script(bad / "gracemaker", "echo hi\n")
    assert gracemaker_version(command=failing) is None
    junk = tmp_path / "junk-bin"
    junk.mkdir()
    _script(junk / "python", 'echo "not a version"\n')
    chatty = _script(junk / "gracemaker", "echo hi\n")
    assert gracemaker_version(command=chatty) is None


def test_version_probe_runs_inside_the_setup_shell(tmp_path: Path) -> None:
    """With setup lines the environment only exists inside the setup shell —
    the probe must run where the fit will."""
    hidden = tmp_path / "hidden-bin"
    hidden.mkdir()
    _script(hidden / "python", 'echo "0.7.2"\n')
    version = gracemaker_version(
        command="modular-gracemaker", setup=(f'PATH="{hidden}:$PATH"',)
    )
    assert version == "0.7.2"


# -- identity -----------------------------------------------------------------


def test_describe_stamps_command_version_and_setup(tmp_path: Path) -> None:
    fake = _env_with_python(tmp_path)
    identity = describe_gracemaker(command=fake)
    assert identity["builder"] == "gracemaker"
    assert identity["command"] == fake
    assert identity["version"] == "0.5.1"
    assert "setup" not in identity

    hidden = tmp_path / "setup-bin"
    hidden.mkdir()
    _script(hidden / "python", 'echo "0.5.1"\n')
    with_setup = describe_gracemaker(
        command="gracemaker", setup=(f'PATH="{hidden}:$PATH"',)
    )
    assert with_setup["setup"] == [f'PATH="{hidden}:$PATH"']
    assert with_setup["version"] == "0.5.1"


def test_versionless_identity_falls_back_to_a_fingerprint(tmp_path: Path) -> None:
    lonely = tmp_path / "lonely-bin"
    lonely.mkdir()
    silent = _script(lonely / "gracemaker", 'echo "training"\n')
    identity = describe_gracemaker(command=silent)
    assert identity["version"] is None
    fingerprint = identity["executable_fingerprint"]
    assert silent in fingerprint  # resolved path + mtime discriminate installs


# -- run guards ---------------------------------------------------------------


def test_run_argument_guards(tmp_path: Path) -> None:
    fake = _env_with_python(tmp_path)
    with pytest.raises(BuilderError, match="no gracemaker arguments"):
        run_gracemaker([], cwd=tmp_path, command=fake)
    with pytest.raises(BuilderError, match="list of tokens"):
        run_gracemaker("input.yaml", cwd=tmp_path, command=fake)  # type: ignore[arg-type]
    # Unlike atomsk, path-shaped tokens are the task's concern, not the seam's.
    outcome = run_gracemaker(["-p", "sub/model.yaml"], cwd=tmp_path, command=fake)
    assert isinstance(outcome, GracemakerOutcome)


def test_missing_binary_is_refused_up_front(tmp_path: Path) -> None:
    with pytest.raises(
        BuilderNotAvailableError, match=re.escape("[builders.gracemaker]")
    ):
        run_gracemaker(["input.yaml"], cwd=tmp_path, command="no-such-gracemaker-anywhere")


# -- run outcomes -------------------------------------------------------------


def test_run_success_returns_the_log(tmp_path: Path) -> None:
    fake = _env_with_python(tmp_path)
    outcome = run_gracemaker(["input.yaml"], cwd=tmp_path, command=fake)
    assert outcome.log.strip() == "training"
    assert outcome.args == ("input.yaml",)


def test_a_traceback_is_failure_even_at_exit_zero(tmp_path: Path) -> None:
    liar = _script(
        tmp_path / "liar",
        f"cat <<'EOF'\nepoch 1\n{TRACEBACK}EOF\nexit 0\n",
    )
    with pytest.raises(BuilderError, match="KeyError: 'cutoff'") as excinfo:
        run_gracemaker(["input.yaml"], cwd=tmp_path, command=liar)
    assert "Traceback" in str(excinfo.value)
    assert "epoch 1" in excinfo.value.log


def test_nonzero_exit_is_failure_even_with_a_clean_log(tmp_path: Path) -> None:
    dying = _script(tmp_path / "dying", "echo fine\nexit 3\n")
    with pytest.raises(BuilderError, match="exit 3"):
        run_gracemaker(["input.yaml"], cwd=tmp_path, command=dying)


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    """TensorFlow training spawns workers; on timeout the child of the child
    must die too, or GPU processes are orphaned on a shared node."""
    spawning = _script(
        tmp_path / "spawning",
        "sleep 30 &\necho $! > child.pid\nwait\n",
    )
    with pytest.raises(BuilderError, match="did not finish within 1s"):
        run_gracemaker(["input.yaml"], cwd=tmp_path, command=spawning, timeout_s=1.0)
    child_pid = int((tmp_path / "child.pid").read_text())
    for _ in range(40):  # the group is SIGKILLed; give init time to reap
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {child_pid} survived the process-group kill")


def test_setup_lines_run_before_the_invocation(tmp_path: Path) -> None:
    fake = _env_with_python(tmp_path)
    work = tmp_path / "work-setup"
    work.mkdir()
    run_gracemaker(
        ["input.yaml"], cwd=work, command=fake, setup=("touch setup-ran",)
    )
    assert (work / "setup-ran").exists()


def test_a_failing_setup_line_kills_the_run(tmp_path: Path) -> None:
    fake = _env_with_python(tmp_path)
    with pytest.raises(BuilderError):
        run_gracemaker(["input.yaml"], cwd=tmp_path, command=fake, setup=("false",))


# -- evidence extraction ------------------------------------------------------


def test_error_lines_keep_the_traceback_tail() -> None:
    log = "epoch 1\nepoch 2\n" + TRACEBACK
    lines = error_lines(log)
    assert lines[0] == "Traceback (most recent call last):"
    assert lines[-1] == "KeyError: 'cutoff'"
    assert "epoch 1" not in lines


def test_error_lines_keep_the_last_traceback_only() -> None:
    log = TRACEBACK + "retrying\n" + TRACEBACK.replace("cutoff", "elements")
    lines = error_lines(log)
    assert lines[-1] == "KeyError: 'elements'"
    assert not any("cutoff" in line for line in lines)


def test_error_lines_fall_back_to_error_mentions_then_the_tail() -> None:
    assert error_lines("fine\nERROR: loss is NaN\nfine\n") == ["ERROR: loss is NaN"]
    assert error_lines("first\n\nsecond\nthird\n") == ["first", "second", "third"]
    assert error_lines("") == ["(gracemaker produced no output)"]
