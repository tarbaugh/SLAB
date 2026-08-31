"""Atomsk builder tests — a fake atomsk keeps them real-atomsk-free.

The fake replays outputs captured from a real atomsk build (master-2026-07-24,
gfortran + Accelerate on macOS): the ``--version`` banner, the log of a real
``--create fcc 4.046 Al -duplicate 2 2 2``, the 32-atom XSF file it wrote,
and the log of a failing ``--create`` with an unknown species — which real
atomsk ends with **exit code 0**. Failure detection must read the log, and
these tests pin that.
"""

import re
from pathlib import Path

import pytest

from slab.atomsk import (
    AtomskOutcome,
    atomsk_command,
    atomsk_setup,
    atomsk_version,
    describe_atomsk,
    error_lines,
    run_atomsk,
)
from slab.errors import BuilderError, BuilderNotAvailableError

DATA = Path(__file__).parent / "data"
XSF = DATA / "atomsk-al-fcc-222.xsf"
CREATE_LOG = DATA / "atomsk-create.log"
BAD_CREATE_LOG = DATA / "atomsk-bad-create.log"

VERSION_BANNER = "(C) P. Hirel 2010 - Version master-2026-07-24 (Beta)"


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def _fake_atomsk(tmp_path: Path) -> str:
    """Protocol-faithful fake: version banner, error-replay on species Xx
    (exit 0, like the real binary), else copy the recorded XSF to the last
    argument and print the recorded create log."""
    return _script(
        tmp_path / "fake-atomsk",
        f"""\
if [ "$1" = "--version" ]; then echo "{VERSION_BANNER}"; exit 0; fi
for a in "$@"; do
  if [ "$a" = "Xx" ]; then cat "{BAD_CREATE_LOG}"; exit 0; fi
done
for a in "$@"; do out="$a"; done
cp "{XSF}" "$out"
cat "{CREATE_LOG}"
""",
    )


# -- command and setup resolution ---------------------------------------------


def test_command_resolution_prefers_call_then_config_then_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert atomsk_command() == "atomsk"
    assert atomsk_command("/opt/atomsk") == "/opt/atomsk"
    (tmp_path / "slab.toml").write_text(
        '[builders.atomsk]\ncommand = "atomsk-config"\nsetup = ["module load atomsk"]\n'
    )
    monkeypatch.chdir(tmp_path)
    assert atomsk_command() == "atomsk-config"
    assert atomsk_command("per-call") == "per-call"
    assert atomsk_setup() == ("module load atomsk",)
    assert atomsk_setup("override") == ("override",)


# -- version probe ------------------------------------------------------------


def test_version_probe_parses_the_recorded_banner(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    assert atomsk_version(command=fake) == "master-2026-07-24"


def test_version_probe_degrades_to_none(tmp_path: Path) -> None:
    assert atomsk_version(command=str(tmp_path / "no-such-atomsk")) is None
    silent = _script(tmp_path / "silent", "exit 0\n")
    assert atomsk_version(command=silent) is None


def test_version_probe_runs_inside_the_setup_shell(tmp_path: Path) -> None:
    """With setup lines the binary may only exist inside the setup shell —
    the probe must run where the build will."""
    hidden = tmp_path / "hidden-bin"
    hidden.mkdir()
    _script(hidden / "modular-atomsk", f'echo "{VERSION_BANNER}"\n')
    version = atomsk_version(
        command="modular-atomsk", setup=(f'PATH="{hidden}:$PATH"',)
    )
    assert version == "master-2026-07-24"


# -- identity -----------------------------------------------------------------


def test_describe_stamps_command_version_and_setup(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    identity = describe_atomsk(command=fake)
    assert identity["builder"] == "atomsk"
    assert identity["command"] == fake
    assert identity["version"] == "master-2026-07-24"
    assert "setup" not in identity

    with_setup = describe_atomsk(command=fake, setup=("true",))
    assert with_setup["setup"] == ["true"]


def test_versionless_identity_falls_back_to_a_fingerprint(tmp_path: Path) -> None:
    silent = _script(tmp_path / "silent", "exit 0\n")
    identity = describe_atomsk(command=silent)
    assert identity["version"] is None
    fingerprint = identity["executable_fingerprint"]
    assert silent in fingerprint  # resolved path + mtime discriminate binaries


# -- run guards ---------------------------------------------------------------


def test_run_refuses_path_arguments(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    for bad in ("/etc/passwd", "sub/dir.xsf", "..", "~home.xsf", "back\\slash"):
        with pytest.raises(BuilderError, match="outside the working directory"):
            run_atomsk(["--create", bad], cwd=tmp_path, command=fake)
    with pytest.raises(BuilderError, match="no atomsk arguments"):
        run_atomsk([], cwd=tmp_path, command=fake)
    with pytest.raises(BuilderError, match="list of tokens"):
        run_atomsk("--create fcc", cwd=tmp_path, command=fake)  # type: ignore[arg-type]


def test_missing_binary_is_refused_up_front(tmp_path: Path) -> None:
    with pytest.raises(BuilderNotAvailableError, match=re.escape("[builders.atomsk]")):
        run_atomsk(["--create", "x.xsf"], cwd=tmp_path, command="no-such-atomsk-anywhere")


# -- run outcomes -------------------------------------------------------------


def test_run_success_returns_the_log_and_writes_the_file(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    outcome = run_atomsk(
        ["--create", "fcc", "4.046", "Al", "al.xsf"], cwd=work, command=fake
    )
    assert isinstance(outcome, AtomskOutcome)
    assert "Successfully wrote XSF file" in outcome.log
    assert (work / "al.xsf").read_bytes() == XSF.read_bytes()


def test_failure_is_detected_from_the_log_despite_exit_zero(tmp_path: Path) -> None:
    """Real atomsk exits 0 after 'X!X ERROR: non-conform statement in mode:
    create' — the recorded log proves it, and detection must not trust the
    exit code."""
    fake = _fake_atomsk(tmp_path)
    with pytest.raises(BuilderError, match="X!X ERROR") as excinfo:
        run_atomsk(["--create", "fcc", "4.046", "Xx", "bad.xsf"], cwd=tmp_path, command=fake)
    assert "non-conform statement" in str(excinfo.value)
    assert excinfo.value.log == BAD_CREATE_LOG.read_text()


def test_nonzero_exit_is_failure_even_with_a_clean_log(tmp_path: Path) -> None:
    dying = _script(tmp_path / "dying", "echo fine\nexit 3\n")
    with pytest.raises(BuilderError, match="exit 3"):
        run_atomsk(["convert.xsf", "out.cfg"], cwd=tmp_path, command=dying)


def test_timeout_kills_and_reports(tmp_path: Path) -> None:
    slow = _script(tmp_path / "slow", "sleep 30\n")
    with pytest.raises(BuilderError, match="did not finish within 1s"):
        run_atomsk(["x.xsf"], cwd=tmp_path, command=slow, timeout_s=1.0)


def test_setup_lines_run_before_the_invocation(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    work = tmp_path / "work-setup"
    work.mkdir()
    run_atomsk(
        ["--create", "fcc", "4.046", "Al", "al.xsf"],
        cwd=work,
        command=fake,
        setup=("touch setup-ran",),
    )
    assert (work / "setup-ran").exists()
    assert (work / "al.xsf").exists()


def test_a_failing_setup_line_kills_the_run(tmp_path: Path) -> None:
    fake = _fake_atomsk(tmp_path)
    with pytest.raises(BuilderError):
        run_atomsk(
            ["--create", "fcc", "4.046", "Al", "al.xsf"],
            cwd=tmp_path,
            command=fake,
            setup=("false",),
        )


# -- evidence extraction ------------------------------------------------------


def test_error_lines_extract_markers_from_the_recorded_log() -> None:
    lines = error_lines(BAD_CREATE_LOG.read_text())
    assert any("non-conform statement" in line for line in lines)
    assert len(lines) <= 10


def test_error_lines_fall_back_to_the_tail() -> None:
    assert error_lines("first\n\nsecond\nthird\n") == ["first", "second", "third"]
    assert error_lines("") == ["(atomsk produced no output)"]
    fortran = "At line 170\nFortran runtime error: End of file\n"
    assert error_lines(fortran) == ["Fortran runtime error: End of file"]
