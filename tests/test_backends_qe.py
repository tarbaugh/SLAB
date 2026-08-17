"""Quantum ESPRESSO engine tests — a fake pw.x keeps them real-QE-free.

The fake failure script reproduces QE's on-disk failure surface exactly (the
``%%%%``-fenced error block on stdout, the ``CRASH`` file, a nonzero exit);
the fake success script replays a genuine pw.x 7.4.1 output captured from a
real run (``tests/data/qe-si-relax-final.pwo``), so ASE's parser and slab's
artifact capture are exercised against the real file format. The optional
integration test at the bottom runs against an actual ``pw.x`` when
``$SLAB_TEST_PW`` and ``$SLAB_TEST_PSEUDO_DIR`` point at one.
"""

import os
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace

import pytest
from ase.build import bulk

from slab import EngineNotAvailableError, ExecutionStatus, Workspace
from slab.backends import (
    _error_blocks,
    _qe_version,
    close_calculator,
    collect_engine_outputs,
    collect_failure_evidence,
    describe_engine,
    get_calculator,
    resolve_pseudopotentials,
)
from slab.tasks import relax

FIXTURE_PWO = Path(__file__).parent / "data" / "qe-si-relax-final.pwo"
FIXTURE_ENERGY_EV = -212.79352142972365  # "!    total energy" of the fixture, in eV


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def _fake_pw_success(tmp_path: Path, pwo_source: Path) -> Path:
    """A pw.x that replays a captured output file (stdout -> espresso.pwo)."""
    return _script(tmp_path / "fake-pw.x", f'cat "{pwo_source}"\n')


def _fake_pw_failure(tmp_path: Path) -> Path:
    """A pw.x that fails the way QE fails: fenced error block, CRASH, exit 3."""
    fence = " " + "%" * 40
    return _script(
        tmp_path / "failing-pw.x",
        f"""echo "     Program PWSCF v.7.4.1 starts"
echo "{fence}"
echo "     Error in routine cheese (1):"
echo "     the wheel is not round"
echo "{fence}"
printf '%s\\n     Error in routine cheese (1):\\n     the wheel is not round\\n' "{fence}" > CRASH
exit 3
""",
    )


# -- calculator factory ----------------------------------------------------------------


def test_qe_missing_binary_is_loud_and_cleans_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slab.backends as backends

    scratch = tmp_path / "observed-scratch"

    def fake_mkdtemp(prefix: str) -> str:
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(backends.tempfile, "mkdtemp", fake_mkdtemp)
    with pytest.raises(EngineNotAvailableError, match="not on PATH"):
        get_calculator("qe", command="definitely-not-pw-x", pseudo_dir=str(tmp_path))
    assert not scratch.exists()


def test_qe_command_without_pseudo_dir_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EngineNotAvailableError, match="pseudo_dir"):
        get_calculator("qe", command="/bin/echo")


def test_qe_profile_conflicts_with_command(tmp_path: Path) -> None:
    from ase.calculators.espresso import EspressoProfile

    profile = EspressoProfile(command="/bin/echo", pseudo_dir=str(tmp_path))
    with pytest.raises(EngineNotAvailableError, match="not both"):
        get_calculator("qe", profile=profile, command="/bin/echo")


def test_qe_unconfigured_points_at_both_setups(monkeypatch: pytest.MonkeyPatch) -> None:
    """No options and no ASE config file: the error teaches both fixes."""
    from ase.calculators.genericfileio import GenericFileIOCalculator
    from ase.config import Config

    monkeypatch.setattr(GenericFileIOCalculator, "cfg", Config())
    with pytest.raises(EngineNotAvailableError, match=r"pseudo_dir.*ASE config") as excinfo:
        get_calculator("qe")
    assert "command" in str(excinfo.value)


def test_qe_scratch_lifecycle_and_option_forwarding(tmp_path: Path) -> None:
    calc = get_calculator(
        "qe", command="/bin/echo", pseudo_dir="~/nowhere-pseudos", kpts=(2, 2, 2)
    )
    scratch = calc._slab_scratch
    assert scratch.is_dir()
    assert scratch.name.startswith("slab-qe-")
    assert calc.directory == scratch
    assert calc.parameters["kpts"] == (2, 2, 2)
    assert calc.profile.pseudo_dir == str(Path("~/nowhere-pseudos").expanduser())
    close_calculator(calc)
    assert not scratch.exists()
    close_calculator(calc)  # idempotent


def test_qe_forces_printing_defaults_on(tmp_path: Path) -> None:
    """pw.x omits forces unless tprnfor is set; slab tasks need them. The
    default survives flat-form input_data and yields to an explicit value."""
    calc = get_calculator(
        "qe", command="/bin/echo", pseudo_dir=str(tmp_path), input_data={"ecutwfc": 14.0}
    )
    input_data = calc.parameters["input_data"]
    assert input_data["control"]["tprnfor"] is True
    assert input_data["system"]["ecutwfc"] == 14.0  # flat keys sorted into sections
    close_calculator(calc)

    calc = get_calculator(
        "qe",
        command="/bin/echo",
        pseudo_dir=str(tmp_path),
        input_data={"control": {"tprnfor": False}},
    )
    assert calc.parameters["input_data"]["control"]["tprnfor"] is False
    close_calculator(calc)


def test_qe_never_mutates_caller_options(tmp_path: Path) -> None:
    """calculator_options is a traced task input: growing keys behind the
    tracer's back (tprnfor injection into a shared nested dict) would make a
    reused options dict hash differently on its second use — a spurious
    cache miss."""
    from copy import deepcopy

    calc_options = {
        "command": "/bin/echo",
        "pseudo_dir": str(tmp_path),
        "input_data": {"control": {"calculation": "scf"}, "system": {"ecutwfc": 14.0}},
    }
    before = deepcopy(calc_options)
    calc = get_calculator("qe", **calc_options)
    assert calc.parameters["input_data"]["control"]["tprnfor"] is True
    close_calculator(calc)
    assert calc_options == before


def test_qe_explicit_directory_is_respected_and_kept(tmp_path: Path) -> None:
    mine = tmp_path / "mine"
    mine.mkdir()
    calc = get_calculator("qe", command="/bin/echo", pseudo_dir=str(tmp_path), directory=mine)
    assert getattr(calc, "_slab_scratch", None) is None
    assert calc.directory == mine
    close_calculator(calc)
    assert mine.exists()  # not slab's to delete


# -- version detection -----------------------------------------------------------------


def test_qe_version_parsed_from_banner(tmp_path: Path) -> None:
    script = _script(tmp_path / "pw.x", 'echo "     Program PWSCF v.9.9.9 starts"\n')
    assert _qe_version({"command": str(script)}) == "9.9.9"
    assert describe_engine("qe", {"command": str(script), "pseudo_dir": "~/ps"}) == {
        "engine": "qe",
        "source": "builtin",
        "version": "9.9.9",
        "command": str(script),
        "pseudo_dir": str(Path("~/ps").expanduser()),
    }


def test_qe_identity_includes_resolved_pseudo_dir(tmp_path: Path) -> None:
    """Switching pseudopotential libraries must change the cache identity
    even when filenames overlap — the directory path is the identity."""
    a = describe_engine("qe", {"command": "/bin/echo", "pseudo_dir": str(tmp_path / "a")})
    b = describe_engine("qe", {"command": "/bin/echo", "pseudo_dir": str(tmp_path / "b")})
    assert a != b
    assert a["pseudo_dir"] == str(tmp_path / "a")


def test_qe_version_probe_is_memoized_until_the_binary_changes(tmp_path: Path) -> None:
    """One spawn per executable identity — not one per task call — but a
    replaced binary (new mtime) is re-probed, so long-lived processes still
    see upgrades."""
    counter = tmp_path / "count"
    script = _script(
        tmp_path / "pw.x",
        f'echo x >> "{counter}"\necho "     Program PWSCF v.1.1.1 starts"\n',
    )
    assert _qe_version({"command": str(script)}) == "1.1.1"
    assert _qe_version({"command": str(script)}) == "1.1.1"
    assert counter.read_text().count("x") == 1
    os.utime(script, ns=(1, 1))  # "upgrade": same path, new mtime
    assert _qe_version({"command": str(script)}) == "1.1.1"
    assert counter.read_text().count("x") == 2


def test_qe_command_none_means_absent_in_the_locator(tmp_path: Path) -> None:
    """{'command': None} (a JSON null) must resolve like an absent key, not
    stamp the literal string 'None' into the cache identity while the
    factory quietly resolves a real binary from the config."""
    from slab.backends import _qe_locator

    command, _pseudo_dir = _qe_locator({"command": None})
    assert command == "pw.x"
    assert describe_engine("qe", {"command": None})["command"] == "pw.x"


def test_qe_version_prefers_profile_command(tmp_path: Path) -> None:
    script = _script(tmp_path / "pw.x", 'echo "     Program PWSCF v.8.8.8 starts"\n')
    profile = SimpleNamespace(command=str(script))
    assert _qe_version({"profile": profile, "command": "ignored"}) == "8.8.8"


def test_qe_version_degrades_to_none() -> None:
    assert _qe_version({"command": "definitely-not-pw-x"}) is None
    assert describe_engine("qe", {"command": "definitely-not-pw-x"})["version"] is None
    assert _qe_version({"command": "/bin/echo"}) is None  # no banner to parse


# -- pseudopotential resolution --------------------------------------------------------


def test_resolve_pseudopotentials_unambiguous(tmp_path: Path) -> None:
    (tmp_path / "Si.pz-vbc.UPF").write_text("<UPF/>")
    (tmp_path / "o_pbe_v1.uspp.F.UPF").write_text("<UPF/>")
    (tmp_path / "README.txt").write_text("not a pseudo")
    mapping = resolve_pseudopotentials(["Si", "O", "Si"], tmp_path)
    assert mapping == {"Si": "Si.pz-vbc.UPF", "O": "o_pbe_v1.uspp.F.UPF"}


def test_resolve_pseudopotentials_accepts_atoms(tmp_path: Path) -> None:
    (tmp_path / "Si.pz-vbc.UPF").write_text("<UPF/>")
    atoms = bulk("Si", "diamond", a=5.43)
    assert resolve_pseudopotentials(atoms, tmp_path) == {"Si": "Si.pz-vbc.UPF"}


def test_resolve_pseudopotentials_element_prefix_is_exact(tmp_path: Path) -> None:
    """'Si.pz.UPF' must never satisfy sulfur: S is not a prefix-match free-for-all."""
    (tmp_path / "Si.pz.UPF").write_text("<UPF/>")
    with pytest.raises(EngineNotAvailableError, match="no pseudopotential for 'S'"):
        resolve_pseudopotentials(["S"], tmp_path)


def test_resolve_pseudopotentials_ambiguity_is_refused(tmp_path: Path) -> None:
    (tmp_path / "Si.pz-vbc.UPF").write_text("<UPF/>")
    (tmp_path / "Si.pbe-rrkjus.UPF").write_text("<UPF/>")
    with pytest.raises(EngineNotAvailableError, match=r"ambiguous.*Si\.pbe-rrkjus"):
        resolve_pseudopotentials(["Si"], tmp_path)


def test_resolve_pseudopotentials_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(EngineNotAvailableError, match="not a directory"):
        resolve_pseudopotentials(["Si"], tmp_path / "ghost")


# -- failure-evidence parsing ----------------------------------------------------------


def test_error_blocks_parse_dedupe_and_flush() -> None:
    fence = " " + "%" * 40
    text = "\n".join(
        [
            "preamble",
            fence,
            "     Error in routine electrons (1):",
            "     charge is wrong: smearing is needed",
            fence,
            "middle",
            fence,
            "     Error in routine electrons (1):",
            "     charge is wrong: smearing is needed",
            fence,
            fence,
            "     Error in routine davcio (10):",  # unclosed: truncated output
        ]
    )
    assert _error_blocks(text) == [
        "Error in routine electrons (1): charge is wrong: smearing is needed",
        "Error in routine davcio (10):",
    ]
    assert _error_blocks("nothing fenced here\n%%\n") == []


def _fake_fileio_calc(tmp_path: Path) -> SimpleNamespace:
    template = SimpleNamespace(
        inputname="espresso.pwi", outputname="espresso.pwo", errorname="espresso.err"
    )
    return SimpleNamespace(template=template, directory=tmp_path)


def test_collect_failure_evidence_prefers_fenced_block(tmp_path: Path) -> None:
    fence = " " + "%" * 40
    (tmp_path / "espresso.pwi").write_text("&CONTROL\n/\n")
    (tmp_path / "espresso.pwo").write_text(
        f"banner\n{fence}\n  Error in routine cheese (1):\n  the wheel is not round\n{fence}\n"
    )
    (tmp_path / "espresso.err").write_text("STOP 1\n")
    (tmp_path / "CRASH").write_text("crash copy\n")
    notes, files = collect_failure_evidence(_fake_fileio_calc(tmp_path))
    assert notes[0] == (
        "engine error (espresso.pwo): Error in routine cheese (1): the wheel is not round"
    )
    assert notes[1] == "engine stderr tail (espresso.err): STOP 1"
    assert {suffix for suffix, _ in files} == {"pwi", "pwo", "err", "crash"}


def test_collect_failure_evidence_flags_unfenced_stop_lines(tmp_path: Path) -> None:
    (tmp_path / "espresso.pwo").write_text(
        "banner\n     convergence NOT achieved after   1 iterations: stopping\n\n JOB DONE.\n"
    )
    notes, files = collect_failure_evidence(_fake_fileio_calc(tmp_path))
    assert notes == [
        "engine output flagged (espresso.pwo): "
        "convergence NOT achieved after   1 iterations: stopping"
    ]
    assert [suffix for suffix, _ in files] == ["pwo"]


def test_collect_failure_evidence_tail_when_nothing_flagged(tmp_path: Path) -> None:
    (tmp_path / "espresso.pwo").write_text("one\ntwo\nthree\nfour\n")
    notes, _files = collect_failure_evidence(_fake_fileio_calc(tmp_path))
    assert notes == ["engine output tail (espresso.pwo): two | three | four"]


def test_collect_failure_evidence_crash_only_is_attributed_to_crash(tmp_path: Path) -> None:
    """MPI-abort shape: ASE pre-created an empty pwo, the story is in CRASH.
    The note must say CRASH — the empty pwo is not even kept as an artifact,
    so pointing at it would send an agent to a file that does not exist."""
    (tmp_path / "espresso.pwo").write_text("")
    (tmp_path / "CRASH").write_text("  Error in routine setup:\n  dead\n")
    notes, files = collect_failure_evidence(_fake_fileio_calc(tmp_path))
    assert notes == ["engine error (CRASH): Error in routine setup: dead"]
    assert [suffix for suffix, _ in files] == ["crash"]


def test_collectors_empty_for_in_process_and_empty_dirs(tmp_path: Path) -> None:
    emt = get_calculator("emt")
    assert collect_failure_evidence(emt) == ([], [])
    assert collect_engine_outputs(emt) == []
    assert collect_failure_evidence(_fake_fileio_calc(tmp_path)) == ([], [])
    assert collect_engine_outputs(_fake_fileio_calc(tmp_path)) == []


# -- relax through a fake pw.x ---------------------------------------------------------


def test_relax_qe_failure_keeps_engine_evidence(ws: Workspace, tmp_path: Path) -> None:
    script = _fake_pw_failure(tmp_path)
    atoms = bulk("Si", "diamond", a=5.43)
    with (
        ws.start_run(name="qe-fail", intent="fake pw.x failure") as run,
        pytest.raises(CalledProcessError) as excinfo,
    ):
        relax(
            atoms,
            engine="qe",
            label="si",
            calculator_options={
                "command": str(script),
                "pseudo_dir": str(tmp_path),
                "pseudopotentials": {"Si": "fake.upf"},
            },
        )
    notes = excinfo.value.__notes__
    assert notes[0] == "relax failed after 0 completed step(s)"
    assert (
        "engine error (espresso.pwo): Error in routine cheese (1): the wheel is not round"
        in notes
    )
    assert any(note.startswith("engine files kept as artifacts:") for note in notes)

    record = ws.runs.list_tasks(run.id)[0]
    assert record.status is ExecutionStatus.FAILED
    assert record.failure is not None
    assert "Error in routine cheese" in " ".join(record.failure["notes"])

    artifacts = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(artifacts) == {"si-failed.pwi", "si-failed.pwo", "si-failed.crash"}
    pwi = ws.artifacts.get(artifacts["si-failed.pwi"].hash).read_text()
    assert "&CONTROL" in pwi.upper()
    crash = ws.artifacts.get(artifacts["si-failed.crash"].hash).read_text()
    assert "Error in routine cheese" in crash


def test_relax_qe_failure_untraced_still_gets_notes(tmp_path: Path) -> None:
    script = _fake_pw_failure(tmp_path)
    atoms = bulk("Si", "diamond", a=5.43)
    with pytest.raises(CalledProcessError) as excinfo:
        relax(
            atoms,
            engine="qe",
            calculator_options={
                "command": str(script),
                "pseudo_dir": str(tmp_path),
                "pseudopotentials": {"Si": "fake.upf"},
            },
        )
    notes = excinfo.value.__notes__
    assert "Error in routine cheese (1): the wheel is not round" in " ".join(notes)
    assert not any("kept as artifacts" in note for note in notes)  # no run, no store


def test_relax_qe_success_replays_real_output(ws: Workspace, tmp_path: Path) -> None:
    replay = tmp_path / "replay.pwo"
    replay.write_text(FIXTURE_PWO.read_text())
    script = _fake_pw_success(tmp_path, replay)
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.rattle(stdev=0.03, seed=7)
    options = {
        "command": str(script),
        "pseudo_dir": str(tmp_path),
        "pseudopotentials": {"Si": "Si.pz-vbc.UPF"},
    }

    with ws.start_run(name="qe-ok", intent="fake pw.x replaying a real pwo") as run:
        relaxed, info = relax(atoms, engine="qe", fmax=0.05, label="si", calculator_options=options)

    assert info["engine"] == "qe"
    assert info["engine_source"] == "builtin"
    assert info["engine_version"] == "7.4.1"  # parsed from the replayed banner
    assert info["converged"] is True
    assert info["energy"] == pytest.approx(FIXTURE_ENERGY_EV)
    assert info["energy_unit"] == "eV"
    assert relaxed.get_potential_energy() == pytest.approx(FIXTURE_ENERGY_EV)

    artifacts = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(artifacts) == {"si.traj", "si.pwo"}
    kept = ws.artifacts.get(artifacts["si.pwo"].hash).read_text()
    assert "Program PWSCF v.7.4.1" in kept
    assert "JOB DONE" in kept

    # Same inputs, same pw.x -> cache hit; a pw.x upgrade (new banner at the
    # same command) -> honest miss via the version in the cache key.
    with ws.start_run(name="qe-again", intent="cache hit") as again:
        relax(atoms, engine="qe", fmax=0.05, label="si", calculator_options=options)
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True

    # An upgrade replaces the executable: new content at the replayed output
    # AND a fresh mtime on the command itself (the probe memoizes on the
    # executable's path + mtime, so an untouched binary is not re-spawned).
    replay.write_text(FIXTURE_PWO.read_text().replace("v.7.4.1", "v.7.5.0"))
    script.touch()
    with ws.start_run(name="qe-upgraded", intent="version bump invalidates") as bumped:
        _, info_bumped = relax(
            atoms, engine="qe", fmax=0.05, label="si", calculator_options=options
        )
    assert ws.runs.list_tasks(bumped.id)[0].cache_hit is False
    assert info_bumped["engine_version"] == "7.5.0"


# -- the real thing, when present ------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("SLAB_TEST_PW") and os.environ.get("SLAB_TEST_PSEUDO_DIR")),
    reason="set SLAB_TEST_PW and SLAB_TEST_PSEUDO_DIR to test against a real pw.x",
)
def test_relax_qe_real_integration(ws: Workspace) -> None:
    pw = os.environ["SLAB_TEST_PW"]
    pseudo_dir = os.environ["SLAB_TEST_PSEUDO_DIR"]
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.rattle(stdev=0.02, seed=7)
    mapping = resolve_pseudopotentials(atoms, pseudo_dir)

    with ws.start_run(name="qe-real", intent="real pw.x integration") as run:
        relaxed, info = relax(
            atoms,
            engine="qe",
            fmax=0.1,
            label="si",
            calculator_options={
                "command": pw,
                "pseudo_dir": pseudo_dir,
                "pseudopotentials": mapping,
                "input_data": {"system": {"ecutwfc": 14.0}},
            },
        )

    assert info["converged"] is True
    assert info["engine_version"] is not None
    assert info["energy"] < 0
    assert relaxed.get_potential_energy() == pytest.approx(info["energy"])
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert {"si.traj", "si.pwo"} <= names
