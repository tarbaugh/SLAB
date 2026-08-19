"""LAMMPS engine tests — a fake lmp keeps them real-LAMMPS-free.

The fake success script replays a genuine LAMMPS ``22 Jul 2025 - Update 4``
run captured from a real binary (``tests/data/lammps-cu-relax-final.log`` and
``.dump``: bulk Cu under the classic ``Cu_u3.eam`` potential), speaking
lammpsrun's actual stdin/stdout protocol — commands in on stdin, thermo out
on stdout, the dump written where the input asked, the ASE end mark last —
so ASE's parser and slab's artifact capture are exercised against the real
file formats. The fake failure script reproduces LAMMPS's on-disk failure
surface exactly: the echoed command, the ``ERROR: ... (src/...)`` line, a
nonzero exit — and, crucially, the *useless* Python-side exception (LAMMPS's
real error dies in a lammpsrun reader thread; only the log tells the story).
The optional integration test at the bottom runs against an actual ``lmp``
when ``$SLAB_TEST_LMP`` and ``$SLAB_TEST_LAMMPS_CU_EAM`` point at one.
"""

import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase.build import bulk

# The constant lammpsrun itself writes and scans for — imported, not
# hardcoded, so ASE drift breaks these tests loudly instead of hanging them.
from ase.calculators.lammps import CALCULATION_END_MARK

from slab import EngineNotAvailableError, ExecutionStatus, Workspace
from slab.backends import (
    _lammps_locator,
    _lammps_version,
    close_calculator,
    collect_engine_outputs,
    collect_failure_evidence,
    describe_engine,
    get_calculator,
)
from slab.tasks import relax

FIXTURE_LOG = Path(__file__).parent / "data" / "lammps-cu-relax-final.log"
FIXTURE_DUMP = Path(__file__).parent / "data" / "lammps-cu-relax-final.dump"
FIXTURE_ENERGY_EV = -28.31961276572313  # the fixture log's PotEng, in eV
END_MARK = CALCULATION_END_MARK

POTENTIAL = {"pair_style": "eam", "pair_coeff": ["1 1 Cu_u3.eam"]}


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _cu_atoms():
    atoms = bulk("Cu", "fcc", a=3.615) * (2, 2, 2)
    atoms.rattle(stdev=0.02, seed=7)
    return atoms


def _script(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def _fake_lmp_success(tmp_path: Path, log_source: Path, banner_version: str) -> Path:
    """A lammpsrun-protocol fake: replays a captured log, writes the dump.

    Reads commands line-by-line from stdin like the real binary under
    lammpsrun (which pipes the whole input and waits for the end mark on
    stdout); on each end-mark request it writes the recorded dump to the
    path the input declared and replays the recorded log. Persistent across
    invocations, exactly like a keep-alive lmp. ``-h`` prints a real-shaped
    version banner for the probe.
    """
    # The recorded log ends with the end mark the real run printed; replaying
    # it verbatim would leave a stray mark for the *next* invocation's reader,
    # which would then stop with an empty thermo table. Strip marks; the fake
    # emits exactly one per invocation.
    log_text = "\n".join(
        line for line in log_source.read_text().splitlines() if line.strip() != END_MARK
    )
    return _script(
        tmp_path / "fake-lmp",
        f'''#!{sys.executable}
import sys

if "-h" in sys.argv:
    print("Large-scale Atomic/Molecular Massively Parallel Simulator - {banner_version}")
    sys.exit(0)

LOG = {log_text!r}
DUMP = {FIXTURE_DUMP.read_text()!r}

dump_path = None
for line in sys.stdin:
    stripped = line.strip()
    if stripped.startswith('variable dump_file string "'):
        dump_path = stripped.split('"')[1]
    if stripped.startswith("print") and "{END_MARK}" in stripped:
        if dump_path:
            with open(dump_path, "w") as handle:
                handle.write(DUMP)
        sys.stdout.write(LOG + "\\n")
        sys.stdout.write("{END_MARK}\\n")
        sys.stdout.flush()
''',
    )


def _fake_lmp_failure(tmp_path: Path) -> Path:
    """A lmp that fails the way LAMMPS fails: echoed command, ERROR line,
    exit 1 — after consuming all input, so the write side never races."""
    return _script(
        tmp_path / "failing-lmp",
        f'''#!{sys.executable}
import sys

if "-h" in sys.argv:
    print("Large-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025")
    sys.exit(0)

for line in sys.stdin:
    if "{END_MARK}" in line:
        break
print("pair_style eam/aloy")
print("ERROR: Unrecognized pair style 'eam/aloy' (src/force.cpp:275)")
sys.stdout.flush()
sys.exit(1)
''',
    )


# -- calculator factory ----------------------------------------------------------------


def test_lammps_potential_is_required() -> None:
    """ASE's silent default is a dimensionless lj/cut toy that would 'work'
    for any material; which potential describes a system is a science
    decision, so a lammps engine without one is refused, not defaulted."""
    with pytest.raises(EngineNotAvailableError, match="pair_style"):
        get_calculator("lammps", command="/bin/echo")
    with pytest.raises(EngineNotAvailableError, match="pair_coeff"):
        get_calculator("lammps", command="/bin/echo", pair_style="eam")
    with pytest.raises(EngineNotAvailableError, match="pair_style"):
        get_calculator("lammps", command="/bin/echo", pair_coeff=["1 1 x"])


def test_lammps_missing_binary_is_loud_and_creates_no_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import slab.backends as backends

    calls: list[str] = []
    real_mkdtemp = backends.tempfile.mkdtemp
    monkeypatch.setattr(
        backends.tempfile, "mkdtemp", lambda **kw: calls.append("x") or real_mkdtemp(**kw)
    )
    with pytest.raises(EngineNotAvailableError, match="not on PATH") as excinfo:
        get_calculator("lammps", command="definitely-not-lmp", **POTENTIAL)
    assert "[engines.lammps]" in str(excinfo.value)  # the error teaches the config fix
    assert calls == []  # refused before any scratch existed


def test_lammps_scratch_lifecycle_and_close_hook(tmp_path: Path) -> None:
    calc = get_calculator("lammps", command="/bin/echo", **POTENTIAL)
    scratch = calc._slab_scratch
    assert scratch.is_dir()
    assert scratch.name.startswith("slab-lammps-")
    # tmp_dir is what makes evidence exist at all: lammpsrun only retains
    # input/log/data files when one is supplied.
    assert Path(calc.parameters["tmp_dir"]) == scratch.resolve()
    assert calc.parameters["keep_tmp_files"] is True
    # lammpsrun has no close(); the persistent lmp subprocess is released
    # through the generic hook close_calculator invokes.
    assert calc._slab_close.__func__ is type(calc)._lmp_end
    close_calculator(calc)
    assert not scratch.exists()
    close_calculator(calc)  # idempotent


def test_lammps_explicit_tmp_dir_is_respected_and_kept(tmp_path: Path) -> None:
    mine = tmp_path / "mine"
    calc = get_calculator("lammps", command="/bin/echo", tmp_dir=str(mine), **POTENTIAL)
    assert getattr(calc, "_slab_scratch", None) is None
    assert Path(calc.parameters["tmp_dir"]) == mine.resolve()
    close_calculator(calc)
    assert mine.exists()  # not slab's to delete


def test_lammps_files_are_staged_and_pair_coeff_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASE's documented shape — files=["pot.eam"] plus a bare basename in
    pair_coeff — resolves against the *caller's cwd* in stock lammpsrun,
    because the spawned lmp inherits it. Slab absolutizes the sources and
    points the pair_coeff reference at the staged copy, so the same options
    work from any cwd (a traced task must not depend on where it ran)."""
    (tmp_path / "pot.eam").write_text("fake potential\n")
    monkeypatch.chdir(tmp_path)
    options = {
        "command": "/bin/echo",
        "pair_style": "eam",
        "pair_coeff": ["1 1 pot.eam"],
        "files": ["pot.eam"],
    }
    before = deepcopy(options)
    calc = get_calculator("lammps", **options)
    scratch = Path(calc.parameters["tmp_dir"])
    assert calc.parameters["pair_coeff"] == [f"1 1 {scratch / 'pot.eam'}"]
    assert calc.parameters["files"] == [str(tmp_path / "pot.eam")]
    assert (scratch / "pot.eam").read_text() == "fake potential\n"
    # calculator_options is a traced task input: never mutated.
    assert options == before
    close_calculator(calc)


def test_lammps_staging_leaves_absolute_and_ordinary_tokens_alone(tmp_path: Path) -> None:
    pot = tmp_path / "pot.eam"
    pot.write_text("fake\n")
    other = tmp_path / "other.eam"
    other.write_text("fake\n")
    calc = get_calculator(
        "lammps",
        command="/bin/echo",
        pair_style="eam/alloy",
        # An absolute reference and a non-file coefficient must survive
        # untouched; only the exact basename of a declared file is rewritten.
        pair_coeff=[f"* * {other} Cu", "1 2 4.0 pot.eam"],
        files=[str(pot)],
    )
    scratch = Path(calc.parameters["tmp_dir"])
    assert calc.parameters["pair_coeff"] == [
        f"* * {other} Cu",
        f"1 2 4.0 {scratch / 'pot.eam'}",
    ]
    close_calculator(calc)


def test_lammps_duplicate_file_basenames_are_refused(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "pot.eam").write_text("A\n")
    (tmp_path / "b" / "pot.eam").write_text("B\n")
    with pytest.raises(EngineNotAvailableError, match="share basename"):
        get_calculator(
            "lammps",
            command="/bin/echo",
            files=[str(tmp_path / "a" / "pot.eam"), str(tmp_path / "b" / "pot.eam")],
            **POTENTIAL,
        )


def test_lammps_duplicate_basenames_refused_case_insensitively(tmp_path: Path) -> None:
    """The overwrite this guard prevents happens at filesystem level, and
    macOS's default filesystem collapses case — so the guard must too.
    Refusing a legal pair on Linux beats a silent overwrite on a Mac."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "Pot.eam").write_text("A\n")
    (tmp_path / "b" / "pot.eam").write_text("B\n")
    with pytest.raises(EngineNotAvailableError, match="case-insensitively"):
        get_calculator(
            "lammps",
            command="/bin/echo",
            files=[str(tmp_path / "a" / "Pot.eam"), str(tmp_path / "b" / "pot.eam")],
            **POTENTIAL,
        )


def test_lammps_str_pair_coeff_is_one_line_even_without_files(tmp_path: Path) -> None:
    """ASE's input writer iterates pair_coeff; an unwrapped string would be
    walked character by character into one garbage line each. The wrap must
    not depend on whether files= happens to be present."""
    calc = get_calculator(
        "lammps", command="/bin/echo", pair_style="lj/cut 2.5", pair_coeff="* * 1 1"
    )
    assert calc.parameters["pair_coeff"] == ["* * 1 1"]
    close_calculator(calc)


def test_lammps_files_as_single_string_is_one_file(tmp_path: Path) -> None:
    """A str files= is one file, never a character sequence of 1-char paths."""
    pot = tmp_path / "pot.eam"
    pot.write_text("fake\n")
    calc = get_calculator(
        "lammps",
        command="/bin/echo",
        pair_style="eam",
        pair_coeff=["1 1 pot.eam"],
        files=str(pot),
    )
    scratch = Path(calc.parameters["tmp_dir"])
    assert calc.parameters["files"] == [str(pot)]
    assert calc.parameters["pair_coeff"] == [f"1 1 {scratch / 'pot.eam'}"]
    close_calculator(calc)


def test_lammps_element_symbol_file_reference_is_refused(tmp_path: Path) -> None:
    """'pair_coeff * * alloy.eam Cu' ends in element names; a staged file
    named exactly 'Cu' makes that token undecidable — file reference or
    element? — so it is refused, never guessed."""
    (tmp_path / "CuNi.eam.alloy").write_text("fake\n")
    (tmp_path / "Cu").write_text("per-element parameters\n")
    with pytest.raises(EngineNotAvailableError, match="element symbol"):
        get_calculator(
            "lammps",
            command="/bin/echo",
            pair_style="eam/alloy",
            pair_coeff=["* * CuNi.eam.alloy Cu Ni"],
            files=[str(tmp_path / "CuNi.eam.alloy"), str(tmp_path / "Cu")],
        )
    # The same file is fine when nothing in pair_coeff references it.
    calc = get_calculator(
        "lammps",
        command="/bin/echo",
        pair_style="eam/alloy",
        pair_coeff=["* * CuNi.eam.alloy Cu Ni"],
        files=[str(tmp_path / "CuNi.eam.alloy")],
    )
    close_calculator(calc)


def test_lammps_command_none_means_absent_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """calculator_options={'command': None} — a JSON null, an os.environ.get
    miss — must resolve identically in the factory, the locator, and the
    stamped cache identity. A stamped command of the literal string 'None'
    would cache results under an identity no binary matches, so a later
    config swap to a different LAMMPS build could re-serve stale results."""
    from ase.config import Config

    monkeypatch.setattr("ase.config.cfg", Config())
    monkeypatch.setenv("ASE_LAMMPSRUN_COMMAND", "/bin/echo")
    assert _lammps_locator({"command": None}) == "/bin/echo"
    identity = describe_engine("lammps", {"command": None})
    assert identity["command"] == "/bin/echo"
    calc = get_calculator("lammps", command=None, **POTENTIAL)
    assert calc.parameters["command"] == "/bin/echo"
    close_calculator(calc)


# -- command resolution ----------------------------------------------------------------


def test_lammps_command_resolution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit option > [engines.lammps] in the slab config > ASE's
    $ASE_LAMMPSRUN_COMMAND convention > bare lmp."""
    # A real ASE config file on the machine running the suite must not leak
    # into the chain under test (ASE reads Path.home(), not $XDG_CONFIG_HOME).
    from ase.config import Config

    monkeypatch.setattr("ase.config.cfg", Config())
    assert _lammps_locator({"command": "/opt/lmp"}) == "/opt/lmp"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        'schema_version = 1\n[engines.lammps]\ncommand = "srun configured-lmp"\n'
    )
    monkeypatch.setenv("ASE_LAMMPSRUN_COMMAND", "env-lmp")
    assert _lammps_locator({}) == "srun configured-lmp"

    (tmp_path / "slab.toml").write_text("schema_version = 1\n")
    assert _lammps_locator({}) == "env-lmp"

    monkeypatch.delenv("ASE_LAMMPSRUN_COMMAND")
    assert _lammps_locator({}) == "lmp"


def test_lammps_missing_binary_error_names_the_resolved_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        'schema_version = 1\n[engines.lammps]\ncommand = "no-such-configured-lmp"\n'
    )
    with pytest.raises(EngineNotAvailableError, match="no-such-configured-lmp"):
        get_calculator("lammps", **POTENTIAL)


# -- version detection -----------------------------------------------------------------


def test_lammps_version_parsed_from_banner(tmp_path: Path) -> None:
    script = _script(
        tmp_path / "lmp",
        '#!/bin/sh\necho "Large-scale Atomic/Molecular Massively Parallel'
        ' Simulator - 2 Apr 2038 - Update 9"\n',
    )
    assert _lammps_version({"command": str(script)}) == "2 Apr 2038 - Update 9"
    assert describe_engine("lammps", {"command": str(script)}) == {
        "engine": "lammps",
        "source": "builtin",
        "version": "2 Apr 2038 - Update 9",
        "command": str(script),
    }


def test_lammps_identity_differs_by_command(tmp_path: Path) -> None:
    """Pointing at a different binary must change the cache identity even
    when neither binary answers a version probe."""
    a = describe_engine("lammps", {"command": "/bin/echo"})
    b = describe_engine("lammps", {"command": "/bin/cat"})
    assert a != b
    assert a["command"] == "/bin/echo"


def test_lammps_identity_resolves_relative_potential_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The traced options carry only the literal string 'pot.eam'; from a
    different cwd that names different bytes. The stamped identity must
    resolve it, or two genuinely different computations share a cache key."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    options = {"command": "/bin/echo", "files": ["pot.eam"]}
    monkeypatch.chdir(tmp_path / "a")
    in_a = describe_engine("lammps", options)
    monkeypatch.chdir(tmp_path / "b")
    in_b = describe_engine("lammps", options)
    assert in_a["files"] == [str(tmp_path / "a" / "pot.eam")]
    assert in_b["files"] == [str(tmp_path / "b" / "pot.eam")]
    assert in_a != in_b
    # No files: no key in the identity at all (absence and [] must differ).
    assert "files" not in describe_engine("lammps", {"command": "/bin/echo"})


def test_lammps_version_probe_is_memoized_until_the_binary_changes(tmp_path: Path) -> None:
    counter = tmp_path / "count"
    script = _script(
        tmp_path / "lmp",
        f'#!/bin/sh\necho x >> "{counter}"\n'
        'echo "Large-scale Atomic/Molecular Massively Parallel Simulator - 1 Jan 2031"\n',
    )
    assert _lammps_version({"command": str(script)}) == "1 Jan 2031"
    assert _lammps_version({"command": str(script)}) == "1 Jan 2031"
    assert counter.read_text().count("x") == 1
    os.utime(script, ns=(1, 1))  # "upgrade": same path, new mtime
    assert _lammps_version({"command": str(script)}) == "1 Jan 2031"
    assert counter.read_text().count("x") == 2


def test_lammps_version_degrades_to_none() -> None:
    assert _lammps_version({"command": "definitely-not-lmp"}) is None
    assert describe_engine("lammps", {"command": "definitely-not-lmp"})["version"] is None
    assert _lammps_version({"command": "/bin/echo"}) is None  # no banner to parse


# -- failure-evidence parsing ----------------------------------------------------------


def _fake_lammpsrun_calc(tmp_path: Path) -> SimpleNamespace:
    """The duck-type surface the collectors dispatch on."""
    return SimpleNamespace(name="lammpsrun", parameters={"tmp_dir": str(tmp_path)})


def _write_eval(directory: Path, call: int, log_text: str, when: int) -> None:
    """One force evaluation's files, stamped with an mtime for ordering."""
    for prefix, text in (("in_", "units metal\n"), ("log_", log_text), ("data_", "8 atoms\n")):
        path = directory / f"{prefix}lammps{call:06d}x{call}"
        path.write_text(text)
        os.utime(path, ns=(when, when))


def test_collect_failure_evidence_surfaces_error_line_and_context(tmp_path: Path) -> None:
    """The real failure shape: the Python-side exception says only 'Failed to
    retrieve any thermo_style-output' (the ERROR dies in a reader thread);
    the log carries the actual message, and -echo log put the dying command
    right above it."""
    _write_eval(
        tmp_path,
        1,
        "units metal\npair_style eam/aloy\n"
        "ERROR: Unrecognized pair style 'eam/aloy' (src/force.cpp:275)\n",
        when=1_000,
    )
    notes, files = collect_failure_evidence(_fake_lammpsrun_calc(tmp_path))
    assert notes[0].startswith("engine error (log_lammps000001x1): ERROR: Unrecognized")
    assert notes[1] == "engine log context (log_lammps000001x1): pair_style eam/aloy"
    assert {suffix for suffix, _ in files} == {"in", "log", "data"}


def test_collect_failure_evidence_reads_the_latest_evaluation(tmp_path: Path) -> None:
    """A relax makes many force evaluations in one scratch; the story of the
    failure is in the newest files, not the first ones."""
    _write_eval(tmp_path, 1, "old log, fine\n", when=1_000)
    _write_eval(
        tmp_path, 2, "ERROR: Lost atoms: original 8 current 5 (src/thermo.cpp)\n", when=2_000
    )
    notes, files = collect_failure_evidence(_fake_lammpsrun_calc(tmp_path))
    assert "Lost atoms" in notes[0]
    assert all(path.name.endswith("x2") for _suffix, path in files)


def test_collect_failure_evidence_flags_and_tail_fallbacks(tmp_path: Path) -> None:
    _write_eval(tmp_path, 1, "step one\nMaximum CPU time exceeded, stopping\n", when=1_000)
    notes, _files = collect_failure_evidence(_fake_lammpsrun_calc(tmp_path))
    assert notes == [
        "engine output flagged (log_lammps000001x1): Maximum CPU time exceeded, stopping"
    ]

    plain = tmp_path / "plain"
    plain.mkdir()
    _write_eval(plain, 1, "one\ntwo\nthree\nfour\n", when=1_000)
    notes, _files = collect_failure_evidence(_fake_lammpsrun_calc(plain))
    assert notes == ["engine output tail (log_lammps000001x1): two | three | four"]


def test_lammps_collectors_empty_when_nothing_written(tmp_path: Path) -> None:
    assert collect_failure_evidence(_fake_lammpsrun_calc(tmp_path)) == ([], [])
    assert collect_engine_outputs(_fake_lammpsrun_calc(tmp_path)) == []
    ghost = SimpleNamespace(name="lammpsrun", parameters={"tmp_dir": str(tmp_path / "ghost")})
    assert collect_failure_evidence(ghost) == ([], [])


def test_collect_engine_outputs_returns_latest_log(tmp_path: Path) -> None:
    _write_eval(tmp_path, 1, "old\n", when=1_000)
    _write_eval(tmp_path, 2, "new\n", when=2_000)
    outputs = collect_engine_outputs(_fake_lammpsrun_calc(tmp_path))
    assert [(suffix, path.name) for suffix, path in outputs] == [("log", "log_lammps000002x2")]


def test_latest_file_mtime_beats_name(tmp_path: Path) -> None:
    """Modification time decides; the name is only a tiebreak. The names here
    disagree with the mtimes on purpose — a name-ordered implementation
    would pick the wrong file."""
    from slab.backends import _latest_lammps_file

    newer_by_time = tmp_path / "log_lammps000001zz"
    newer_by_time.write_text("the actual latest\n")
    os.utime(newer_by_time, ns=(2_000, 2_000))
    newer_by_name = tmp_path / "log_lammps000002aa"
    newer_by_name.write_text("older but bigger counter\n")
    os.utime(newer_by_name, ns=(1_000, 1_000))
    assert _latest_lammps_file(tmp_path, "log_") == newer_by_time


def test_empty_newest_log_means_killed_not_previous_evals_story(tmp_path: Path) -> None:
    """OOM and scheduler kills leave the dying evaluation's log empty. An
    older evaluation's healthy log would be *misleading* evidence — the note
    must say the process died writing nothing, and no log is kept."""
    _write_eval(tmp_path, 1, "healthy old log, thermo and all\n", when=1_000)
    _write_eval(tmp_path, 2, "", when=2_000)  # killed before writing
    notes, files = collect_failure_evidence(_fake_lammpsrun_calc(tmp_path))
    assert "died before writing" in notes[0]
    assert "log_lammps000002x2" in notes[0]
    kept = {suffix: path.name for suffix, path in files}
    assert "log" not in kept
    assert kept["in"] == "in_lammps000002x2"  # current eval's input still kept
    assert kept["data"] == "data_lammps000002x2"


def test_unreadable_log_still_keeps_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log that cannot be read costs its notes, never the already-found
    evidence files (the collectors run inside exception handlers)."""
    _write_eval(tmp_path, 1, "ERROR: something real\n", when=1_000)
    real_read_text = Path.read_text

    def broken_log_read(self: Path, *args: object, **kwargs: object) -> str:
        if self.name.startswith("log_"):
            raise PermissionError(f"unreadable: {self}")
        return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", broken_log_read)
    notes, files = collect_failure_evidence(_fake_lammpsrun_calc(tmp_path))
    assert notes == ["engine log (log_lammps000001x1) could not be read"]
    assert {suffix for suffix, _ in files} == {"in", "log", "data"}


def test_close_calculator_invokes_the_slab_close_hook() -> None:
    """The hook is what releases lammps's persistent subprocess; asserting
    its identity alone would let a close_calculator that never calls it
    pass. Called on close, and safe to close twice."""
    calls: list[str] = []
    calc = SimpleNamespace(_slab_close=lambda: calls.append("closed"))
    close_calculator(calc)
    assert calls == ["closed"]
    close_calculator(calc)
    assert calls == ["closed", "closed"]


# -- relax through a fake lmp ----------------------------------------------------------


def _replay_options(script: Path, tmp_path: Path) -> dict:
    # A declared potential file plus a bare-basename pair_coeff reference:
    # the staging contract runs in every full-loop test, so its rewrite is
    # exercised through relax, not just in unit tests.
    potential = tmp_path / "Cu_u3.eam"
    if not potential.exists():
        potential.write_text("fake potential contents\n")
    return {
        "command": str(script),
        "pair_style": "eam",
        "pair_coeff": ["1 1 Cu_u3.eam"],
        "files": [str(potential)],
        # Text dump so the replayed fixture is human-diffable; the real
        # recording was made the same way.
        "binary_dump": False,
    }


# The unhandled-thread-exception warning IS the finding under test: lammpsrun
# raises the real ERROR inside its reader thread, where it can reach no caller.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_relax_lammps_failure_keeps_engine_evidence(ws: Workspace, tmp_path: Path) -> None:
    script = _fake_lmp_failure(tmp_path)
    atoms = _cu_atoms()
    with (
        ws.start_run(name="lammps-fail", intent="fake lmp failure") as run,
        # Which RuntimeError surfaces is a race in lammpsrun itself (did
        # poll() see the exit before the thermo sanity check?); both are
        # equally uninformative — the point of this test is that the notes
        # carry the real story either way.
        pytest.raises(RuntimeError, match=r"exit code|thermo_style") as excinfo,
    ):
        relax(
            atoms,
            engine="lammps",
            label="cu",
            calculator_options=_replay_options(script, tmp_path),
        )
    notes = excinfo.value.__notes__
    assert notes[0] == "relax failed after 0 completed step(s)"
    assert any(
        "ERROR: Unrecognized pair style 'eam/aloy' (src/force.cpp:275)" in note
        for note in notes
    )
    assert any("engine log context" in note and "pair_style eam/aloy" in note for note in notes)
    assert any(note.startswith("engine files kept as artifacts:") for note in notes)

    record = ws.runs.list_tasks(run.id)[0]
    assert record.status is ExecutionStatus.FAILED
    assert record.failure is not None
    assert "Unrecognized pair style" in " ".join(record.failure["notes"])

    artifacts = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(artifacts) == {"cu-failed.in", "cu-failed.log", "cu-failed.data"}
    kept_in = ws.artifacts.get(artifacts["cu-failed.in"].hash).read_text()
    assert "pair_style eam" in kept_in
    # The staging contract, end to end through relax: the input the engine
    # actually received references the staged copy, not the bare basename.
    assert re.search(r"pair_coeff 1 1 \S*slab-lammps-\S*/Cu_u3\.eam", kept_in)
    kept_log = ws.artifacts.get(artifacts["cu-failed.log"].hash).read_text()
    assert "ERROR: Unrecognized pair style" in kept_log


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_relax_lammps_failure_untraced_still_gets_notes(tmp_path: Path) -> None:
    script = _fake_lmp_failure(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        relax(_cu_atoms(), engine="lammps", calculator_options=_replay_options(script, tmp_path))
    notes = " ".join(excinfo.value.__notes__)
    assert "Unrecognized pair style" in notes
    assert "kept as artifacts" not in notes  # no run, no store


def test_relax_lammps_success_replays_real_output(ws: Workspace, tmp_path: Path) -> None:
    replay_log = tmp_path / "replay.log"
    replay_log.write_text(FIXTURE_LOG.read_text())
    script = _fake_lmp_success(tmp_path, replay_log, "22 Jul 2025 - Update 4")
    atoms = _cu_atoms()
    options = _replay_options(script, tmp_path)

    with ws.start_run(name="lammps-ok", intent="fake lmp replaying a real log") as run:
        relaxed, info = relax(
            atoms, engine="lammps", fmax=0.05, label="cu", calculator_options=options
        )

    assert info["engine"] == "lammps"
    assert info["engine_source"] == "builtin"
    assert info["engine_version"] == "22 Jul 2025 - Update 4"
    assert info["converged"] is True
    assert info["energy"] == pytest.approx(FIXTURE_ENERGY_EV)
    assert info["energy_unit"] == "eV"
    assert relaxed.get_potential_energy() == pytest.approx(FIXTURE_ENERGY_EV)

    artifacts = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(artifacts) == {"cu.traj", "cu.log"}
    kept = ws.artifacts.get(artifacts["cu.log"].hash).read_text()
    assert "Step" in kept  # the thermo table travelled into the artifact

    # Same inputs, same lmp -> cache hit; an upgraded binary (new banner at
    # the same command) -> honest miss via the version in the cache key.
    with ws.start_run(name="lammps-again", intent="cache hit") as again:
        relax(atoms, engine="lammps", fmax=0.05, label="cu", calculator_options=options)
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True

    _fake_lmp_success(tmp_path, replay_log, "1 Jan 2031")  # rewrite = fresh mtime
    with ws.start_run(name="lammps-upgraded", intent="version bump invalidates") as bumped:
        _, info_bumped = relax(
            atoms, engine="lammps", fmax=0.05, label="cu", calculator_options=options
        )
    assert ws.runs.list_tasks(bumped.id)[0].cache_hit is False
    assert info_bumped["engine_version"] == "1 Jan 2031"


def test_fake_lmp_persistent_process_serves_repeated_evaluations(tmp_path: Path) -> None:
    """The keep-alive half of the protocol: one lmp process, many inputs on
    one stdin stream. A fake (or a wiring bug) that dies after the first
    end mark would hang or fail the second evaluation."""
    replay_log = tmp_path / "replay.log"
    replay_log.write_text(FIXTURE_LOG.read_text())
    script = _fake_lmp_success(tmp_path, replay_log, "22 Jul 2025 - Update 4")
    calc = get_calculator("lammps", **_replay_options(script, tmp_path))
    scratch = Path(calc.parameters["tmp_dir"])
    try:
        atoms = _cu_atoms()
        atoms.calc = calc
        first = atoms.get_potential_energy()
        atoms.rattle(stdev=0.01, seed=11)  # changed positions force a recalculation
        second = atoms.get_potential_energy()
        assert first == second == pytest.approx(FIXTURE_ENERGY_EV)
        assert len(list(scratch.glob("log_*"))) == 2  # two evaluations, one process
    finally:
        close_calculator(calc)


# -- the real thing, when present ------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("SLAB_TEST_LMP") and os.environ.get("SLAB_TEST_LAMMPS_CU_EAM")),
    reason="set SLAB_TEST_LMP and SLAB_TEST_LAMMPS_CU_EAM to test against a real lmp",
)
def test_relax_lammps_real_integration(ws: Workspace) -> None:
    lmp = os.environ["SLAB_TEST_LMP"]
    potential = os.environ["SLAB_TEST_LAMMPS_CU_EAM"]
    atoms = _cu_atoms()

    with ws.start_run(name="lammps-real", intent="real lmp integration") as run:
        _relaxed, info = relax(
            atoms,
            engine="lammps",
            fmax=0.05,
            label="cu",
            calculator_options={
                "command": lmp,
                "pair_style": "eam",
                "pair_coeff": [f"1 1 {Path(potential).name}"],
                "files": [potential],
            },
        )

    assert info["converged"] is True
    assert info["engine_version"] is not None
    # EAM Cu cohesive energy is ~-3.54 eV/atom; a number far outside that
    # band would mean the potential file was not the one we asked for.
    assert -4.5 < info["energy"] / info["n_atoms"] < -2.5
    artifacts = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert {"cu.traj", "cu.log"} <= set(artifacts)
    # Independent check: the PotEng LAMMPS itself printed into the kept log
    # must match the energy relax reported (same float from two routes —
    # the parsed thermo table vs. the artifact bytes).
    kept_log = ws.artifacts.get(artifacts["cu.log"].hash).read_text()
    lines = kept_log.splitlines()
    header_at = max(i for i, line in enumerate(lines) if line.strip().startswith("Step "))
    row = dict(zip(lines[header_at].split(), lines[header_at + 1].split(), strict=False))
    assert float(row["PotEng"]) == pytest.approx(info["energy"], rel=1e-12)


# -- per-engine environments: the env wrapper -------------------------------------------


def test_lammps_bare_env_assignment_prefix_is_refused_by_name() -> None:
    """Same rule as qe: lammpsrun execs argv without a shell, so a leading
    assignment would be exec'd as the program — teach the env form."""
    with pytest.raises(EngineNotAvailableError, match="env OMP_NUM_THREADS=2 lmp"):
        get_calculator("lammps", command="OMP_NUM_THREADS=2 lmp", **POTENTIAL)


def test_lammps_env_wrapped_command_checks_the_payload() -> None:
    """The PATH refusal must name the missing engine binary, not pass on
    /usr/bin/env's existence."""
    with pytest.raises(EngineNotAvailableError, match="definitely-not-lmp"):
        get_calculator("lammps", command="env OMP_NUM_THREADS=2 definitely-not-lmp", **POTENTIAL)


def test_lammps_env_wrapped_command_builds(tmp_path: Path) -> None:
    calc = get_calculator("lammps", command="env OMP_NUM_THREADS=2 /bin/echo", **POTENTIAL)
    try:
        assert calc.parameters["command"] == "env OMP_NUM_THREADS=2 /bin/echo"
    finally:
        close_calculator(calc)


def test_ambient_potential_resolution_is_stamped_into_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair_coeff potential outside files= is resolved by lmp itself (cwd,
    then $LAMMPS_POTENTIALS) — allowed, but never silently: the resolved
    source lands in cache identity, so repointing the module farm's
    potentials directory honestly invalidates cached results."""
    farm_a = tmp_path / "farm-a"
    farm_b = tmp_path / "farm-b"
    for farm in (farm_a, farm_b):
        farm.mkdir()
        (farm / "Cu_u3.eam").write_text("fake potential\n")
    monkeypatch.chdir(tmp_path)  # nothing resolvable in cwd
    monkeypatch.setenv("LAMMPS_POTENTIALS", str(farm_a))
    options = {"command": "/bin/echo", **POTENTIAL}
    a = describe_engine("lammps", options)
    assert a["pair_coeff_files"] == [str((farm_a / "Cu_u3.eam").resolve())]
    monkeypatch.setenv("LAMMPS_POTENTIALS", str(farm_b))
    b = describe_engine("lammps", options)
    assert a["pair_coeff_files"] != b["pair_coeff_files"]  # swap invalidates
    monkeypatch.delenv("LAMMPS_POTENTIALS")
    unresolved = describe_engine("lammps", options)
    assert "pair_coeff_files" not in unresolved  # nothing ambient to stamp


def test_identity_accepts_singular_files_forms(tmp_path: Path) -> None:
    """files= as one str or one PathLike is one file everywhere — identity
    construction must never iterate a path per character or raise."""
    pot = tmp_path / "Cu_u3.eam"
    pot.write_text("fake\n")
    for form in (str(pot), Path(str(pot))):
        identity = describe_engine(
            "lammps",
            {"command": "/bin/echo", "pair_style": "eam",
             "pair_coeff": ["1 1 Cu_u3.eam"], "files": form},
        )
        assert identity["files"] == [str(pot)]
