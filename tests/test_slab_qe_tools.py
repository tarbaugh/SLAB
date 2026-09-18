"""The post-processing tool commands, and the runner that drives them."""

import re
from pathlib import Path

import pytest

from slab.errors import EngineNotAvailableError, QeToolError
from slab.qe_tools import (
    PW_ONLY_FLAGS,
    namelist_text,
    qe_tool_command,
    run_qe_tool,
)


def _install(tmp_path: Path, *names: str) -> Path:
    """A directory of executables with the given names."""
    root = tmp_path / "bin"
    root.mkdir(exist_ok=True)
    for name in names:
        script = root / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
    return root


def test_a_tool_follows_the_pw_x_install_and_launcher(tmp_path: Path) -> None:
    root = _install(tmp_path, "pw.x", "dos.x", "projwfc.x")
    command = qe_tool_command("dos.x", {"command": f"mpirun -np 4 {root}/pw.x -nk 4"})
    assert command == f"mpirun -np 4 {root}/dos.x"
    assert qe_tool_command("projwfc.x", {"command": str(root / "pw.x")}) == str(
        root / "projwfc.x"
    )
    # An env wrapper is a wrapper, not the payload, and it is kept.
    wrapped = qe_tool_command("dos.x", {"command": f"env OMP_NUM_THREADS=1 {root}/pw.x"})
    assert wrapped == f"env OMP_NUM_THREADS=1 {root}/dos.x"


def test_every_pw_only_flag_is_dropped_from_the_tail(tmp_path: Path) -> None:
    root = _install(tmp_path, "pw.x", "dos.x")
    tail = " ".join(f"{flag} 2" for flag in sorted(PW_ONLY_FLAGS))
    assert qe_tool_command("dos.x", {"command": f"{root}/pw.x {tail}"}) == str(root / "dos.x")


def test_a_bare_pw_x_maps_to_a_bare_tool_and_an_override_wins(tmp_path: Path) -> None:
    assert qe_tool_command("dos.x", {"command": "pw.x"}) == "dos.x"
    assert qe_tool_command("dos.x", {"command": "pw.x", "dos_command": "srun dos.x"}) == (
        "srun dos.x"
    )
    assert qe_tool_command("projwfc.x", {"projwfc_command": "projwfc.x -i"}) == "projwfc.x -i"


def test_a_missing_sibling_names_the_file_and_the_key(tmp_path: Path) -> None:
    root = _install(tmp_path, "pw.x")
    with pytest.raises(EngineNotAvailableError) as excinfo:
        qe_tool_command("dos.x", {"command": str(root / "pw.x")})
    assert str(root / "dos.x") in str(excinfo.value)
    assert "'dos_command'" in str(excinfo.value)
    with pytest.raises(EngineNotAvailableError, match=re.escape("names no 'pw.x' token")):
        qe_tool_command("dos.x", {"command": "lmp"})
    with pytest.raises(ValueError, match=re.escape("runs dos.x and projwfc.x, not 'bands.x'")):
        qe_tool_command("bands.x", {"command": "pw.x"})


def test_namelist_text_writes_fortran_spellings() -> None:
    written = namelist_text("PROJWFC", {"prefix": "pwscf", "lsym": False, "n": 3, "gone": None})
    assert written == "&PROJWFC\n  prefix = 'pwscf'\n  lsym = .false.\n  n = 3\n/\n"


def test_run_qe_tool_writes_the_input_and_captures_the_output(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    root.mkdir()
    (root / "pw.x").write_text("#!/bin/sh\nexit 0\n")
    (root / "dos.x").write_text('#!/bin/sh\ncat "$2"\necho "   JOB DONE."\n')
    for name in ("pw.x", "dos.x"):
        (root / name).chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    outcome = run_qe_tool(
        "dos.x",
        {"prefix": "pwscf", "DeltaE": 0.05},
        cwd=work,
        options={"command": str(root / "pw.x")},
    )
    assert outcome.input_path == work / "dos.in"
    assert (work / "dos.in").read_text() == "&DOS\n  prefix = 'pwscf'\n  DeltaE = 0.05\n/\n"
    assert outcome.output_path == work / "dos.out"
    assert "JOB DONE" in outcome.output == (work / "dos.out").read_text()
    assert outcome.elapsed_s >= 0.0


def test_run_qe_tool_raises_on_an_error_block_and_keeps_the_output(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    root.mkdir()
    fence = "%" * 40
    (root / "pw.x").write_text("#!/bin/sh\nexit 0\n")
    (root / "projwfc.x").write_text(
        f'#!/bin/sh\necho " {fence}"\necho "     Error in routine projwave (1):"\n'
        f'echo "     no atomic wavefunctions"\necho " {fence}"\nexit 3\n'
    )
    for name in ("pw.x", "projwfc.x"):
        (root / name).chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(QeToolError) as excinfo:
        run_qe_tool("projwfc.x", {"prefix": "pwscf"}, cwd=work, options={
            "command": str(root / "pw.x")
        })
    assert "Error in routine projwave (1): no atomic wavefunctions" in str(excinfo.value)
    assert excinfo.value.tool == "projwfc.x"
    assert "Error in routine" in excinfo.value.log
    assert (work / "projwfc.out").read_text() == excinfo.value.log


def test_run_qe_tool_reports_a_nonzero_exit_with_the_tail(tmp_path: Path) -> None:
    root = tmp_path / "bin"
    root.mkdir()
    (root / "pw.x").write_text("#!/bin/sh\nexit 0\n")
    (root / "dos.x").write_text('#!/bin/sh\necho "stopping: no charge density"\nexit 1\n')
    for name in ("pw.x", "dos.x"):
        (root / name).chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(QeToolError, match=r"exit 1.*no charge density"):
        run_qe_tool("dos.x", {"prefix": "pwscf"}, cwd=work, options={
            "command": str(root / "pw.x")
        })


def test_run_qe_tool_refuses_a_command_this_machine_does_not_have(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(EngineNotAvailableError, match="not on PATH"):
        run_qe_tool(
            "dos.x",
            {"prefix": "pwscf"},
            cwd=work,
            options={"dos_command": "slab-has-no-such-dos.x"},
        )
