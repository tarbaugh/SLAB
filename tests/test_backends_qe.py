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

from slab import EngineNotAvailableError, ExecutionStatus, Workspace, backends
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
        "qe",
        command="/bin/echo",
        pseudo_dir="~/nowhere-pseudos",
        kpts=(2, 2, 2),
        input_data={"system": {"ecutwfc": 30.0}},
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
        input_data={"control": {"tprnfor": False}, "system": {"ecutwfc": 14.0}},
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
    calc = get_calculator(
        "qe",
        command="/bin/echo",
        pseudo_dir=str(tmp_path),
        directory=mine,
        input_data={"system": {"ecutwfc": 30.0}},
    )
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
                "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
                "input_data": {"system": {"ecutwfc": 30.0}},
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
                "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
                "input_data": {"system": {"ecutwfc": 30.0}},
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
        "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
        "input_data": {"system": {"ecutwfc": 30.0}},
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


def test_single_point_qe_replay_keeps_pwo_and_caches(ws: Workspace, tmp_path: Path) -> None:
    """One SCF, no optimizer: the pwo is the only artifact, and identity caches."""
    from slab.tasks import single_point

    script = _fake_pw_success(tmp_path, FIXTURE_PWO)
    atoms = bulk("Si", "diamond", a=5.43)
    options = {
        "command": str(script),
        "pseudo_dir": str(tmp_path),
        "pseudopotentials": {"Si": "Si.pz-vbc.UPF"},
        "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
        "input_data": {"system": {"ecutwfc": 30.0}},
    }

    with ws.start_run(name="qe-scf", intent="fake pw.x single point") as run:
        evaluated, info = single_point(atoms, engine="qe", label="si", calculator_options=options)

    assert info["engine_version"] == "7.4.1"
    assert info["energy"] == pytest.approx(FIXTURE_ENERGY_EV)
    assert "converged" not in info
    assert info["fmax"] >= 0  # forces parsed from the replayed pwo
    assert evaluated.get_potential_energy() == pytest.approx(FIXTURE_ENERGY_EV)
    artifacts = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert artifacts == {"si.pwo"}  # no trajectory: nothing was optimized

    with ws.start_run(name="qe-scf-again", intent="cache hit") as again:
        single_point(atoms, engine="qe", label="si", calculator_options=options)
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True


def test_single_point_qe_failure_untraced_still_gets_notes(tmp_path: Path) -> None:
    from slab.tasks import single_point

    script = _fake_pw_failure(tmp_path)
    with pytest.raises(CalledProcessError) as excinfo:
        single_point(
            bulk("Si", "diamond", a=5.43),
            engine="qe",
            calculator_options={
                "command": str(script),
                "pseudo_dir": str(tmp_path),
                "pseudopotentials": {"Si": "fake.upf"},
                "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
                "input_data": {"system": {"ecutwfc": 30.0}},
            },
        )
    notes = " ".join(excinfo.value.__notes__)
    assert "Error in routine cheese (1): the wheel is not round" in notes
    assert "kept as artifacts" not in notes  # no run, no store


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


# -- first-contact guards (protocol=, ecutwfc, launchers) ------------------------------


def test_qe_refuses_protocol_name_in_options(tmp_path: Path) -> None:
    """A protocol *name* would silently vanish inside ASE's input writer."""
    with pytest.raises(EngineNotAvailableError, match="qe_protocol_options"):
        get_calculator("qe", command="/bin/echo", pseudo_dir=str(tmp_path), protocol="balanced")


def test_qe_refuses_missing_ecutwfc(tmp_path: Path) -> None:
    """pw.x aborts at runtime without ecutwfc; slab refuses at build time."""
    with pytest.raises(EngineNotAvailableError, match="ecutwfc"):
        get_calculator("qe", command="/bin/echo", pseudo_dir=str(tmp_path))


def _fake_launcher_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    bins = tmp_path / "bin"
    bins.mkdir(exist_ok=True)
    for name in names:
        _script(bins / name, "exit 0\n")
    monkeypatch.setenv("PATH", f"{bins}:{os.environ.get('PATH', '')}")


def test_qe_srun_outside_allocation_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """srun outside a job queues for a fresh allocation — a silent hang, since
    ASE runs engine commands with no timeout. Refused loudly instead."""
    _fake_launcher_env(tmp_path, monkeypatch, "srun", "pw.x")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(EngineNotAvailableError, match="not inside a SLURM allocation"):
        get_calculator(
            "qe",
            command="srun pw.x",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_srun_inside_allocation_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_launcher_env(tmp_path, monkeypatch, "srun", "pw.x")
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    calc = get_calculator(
        "qe",
        command="srun pw.x",
        pseudo_dir=str(tmp_path),
        input_data={"system": {"ecutwfc": 30.0}},
    )
    close_calculator(calc)


def test_qe_launcher_with_missing_payload_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """which(argv[0]) alone validates the *launcher* and would miss this."""
    _fake_launcher_env(tmp_path, monkeypatch, "srun")
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    with pytest.raises(EngineNotAvailableError, match="module load"):
        get_calculator(
            "qe",
            command="srun definitely-not-pw",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_srun_probe_short_circuits_outside_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The version probe must not burn its timeout on a doomed srun."""
    from slab.backends import _srun_without_allocation

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert _srun_without_allocation("srun pw.x") is True
    assert _srun_without_allocation("mpirun pw.x") is False
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    assert _srun_without_allocation("srun pw.x") is False


# -- per-engine environments: the env wrapper -------------------------------------------


def test_qe_env_wrapped_command_builds_and_checks_the_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'env VAR=val pw.x' scopes variables to this engine's subprocess alone
    (ASE execs argv, no shell) — and the PATH check must judge pw.x, not
    /usr/bin/env."""
    _fake_launcher_env(tmp_path, monkeypatch, "pw.x")
    calc = get_calculator(
        "qe",
        command="env OMP_NUM_THREADS=1 pw.x",
        pseudo_dir=str(tmp_path),
        input_data={"system": {"ecutwfc": 30.0}},
    )
    close_calculator(calc)
    with pytest.raises(EngineNotAvailableError, match="definitely-not-pw"):
        get_calculator(
            "qe",
            command="env OMP_NUM_THREADS=1 definitely-not-pw",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_env_wrapped_srun_still_hits_the_allocation_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env-wrapped srun is still srun: the wrapper must not become a hole
    in the silent-hang refusal."""
    _fake_launcher_env(tmp_path, monkeypatch, "srun", "pw.x")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(EngineNotAvailableError, match="not inside a SLURM allocation"):
        get_calculator(
            "qe",
            command="env OMP_NUM_THREADS=1 srun pw.x",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_bare_env_assignment_prefix_is_refused_by_name(tmp_path: Path) -> None:
    """A shell idiom in a no-shell seam: the refusal must teach the env form,
    not report a missing binary called 'OMP_NUM_THREADS=4'."""
    with pytest.raises(EngineNotAvailableError, match=r"env OMP_NUM_THREADS=4 pw\.x"):
        get_calculator(
            "qe",
            command="OMP_NUM_THREADS=4 pw.x",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_command_payload_sees_through_the_wrapper() -> None:
    from slab.backends import _command_payload

    assert _command_payload("env OMP_NUM_THREADS=1 srun pw.x") == ["srun", "pw.x"]
    assert _command_payload("srun pw.x") == ["srun", "pw.x"]
    assert _command_payload("env A=1 B=2") == []  # sets environment, names no program
    # env's portable flags are understood — the guards must not disengage
    # for -i or -u, the forms people actually write.
    assert _command_payload("env -i pw.x") == ["pw.x"]
    assert _command_payload("env -u DISPLAY srun pw.x") == ["srun", "pw.x"]
    assert _command_payload("env --unset=DISPLAY pw.x") == ["pw.x"]
    assert _command_payload("env -S pw.x") is None  # -S re-splits: env's business
    assert _command_payload('un"balanced') is None


def test_qe_env_wrapped_version_probe_keys_on_the_payload_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memoized version probe must watch pw.x's mtime, not /usr/bin/env's
    — an env-keyed memo would never notice a module swap."""
    from slab.backends import _executable_identity

    _fake_launcher_env(tmp_path, monkeypatch, "pw.x")
    identity = _executable_identity("env OMP_NUM_THREADS=1 pw.x")
    assert identity is not None
    assert identity[0] == str(tmp_path / "bin" / "pw.x")


def test_env_wrapper_setting_path_resolves_payload_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'env PATH=/opt/qe/bin pw.x' execs the payload via the ASSIGNED PATH —
    the module-load-replacement use case — so the which-check must judge it
    there, not against the process PATH."""
    bins = tmp_path / "qe-bin"
    bins.mkdir()
    _script(bins / "pw.x", "exit 0\n")
    # pw.x is NOT on the process PATH; only the wrapper's assignment finds it.
    calc = get_calculator(
        "qe",
        command=f"env PATH={bins} pw.x",
        pseudo_dir=str(tmp_path),
        input_data={"system": {"ecutwfc": 30.0}},
    )
    close_calculator(calc)
    # And the reverse: 'env -i pw.x' clears PATH, so a bare payload name
    # genuinely cannot exec — the refusal is telling the truth.
    with pytest.raises(EngineNotAvailableError, match="not on PATH"):
        get_calculator(
            "qe",
            command="env -i pw.x",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_unparseable_command_is_refused_at_the_factory(tmp_path: Path) -> None:
    """An unbalanced quote raised loudly before the guards existed; it must
    not have regressed into a deferred calculate-time failure."""
    with pytest.raises(EngineNotAvailableError, match="not parseable"):
        get_calculator(
            "qe",
            command='pw.x "unbalanced',
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_version_probe_never_runs_through_a_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'mpirun -np 64 pw.x' at identity time would fan ranks out on a login
    node; the probe must run the bare payload (two-token form) or nothing."""
    from slab.backends import _banner_version

    bins = tmp_path / "bin"
    bins.mkdir(exist_ok=True)
    marker = tmp_path / "mpirun-ran"
    _script(bins / "mpirun", f"touch {marker}\nexit 0\n")
    _script(bins / "pw.x", 'echo "Program PWSCF v.7.5 starts"\nexit 0\n')
    monkeypatch.setenv("PATH", f"{bins}:{os.environ.get('PATH', '')}")
    assert _banner_version("mpirun pw.x") == "7.5"  # probed via bare pw.x
    assert not marker.exists()  # the launcher itself never executed
    assert _banner_version("mpirun -np 64 pw.x") is None  # flagged form: no probe
    assert not marker.exists()


def test_versionless_identity_carries_a_binary_fingerprint(tmp_path: Path) -> None:
    """When the banner probe yields None, the command STRING alone would let
    two different binaries share a cache key; the resolved path+mtime keeps
    them apart."""
    a = describe_engine("qe", {"command": "/bin/echo", "pseudo_dir": str(tmp_path)})
    assert a["version"] is None
    assert "/bin/echo" in a["executable_fingerprint"]
    b = describe_engine("qe", {"command": "/bin/cat", "pseudo_dir": str(tmp_path)})
    assert a["executable_fingerprint"] != b["executable_fingerprint"]


def test_scratch_root_config_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[paths] scratch redirects slab-managed scratches off node-local
    $TMPDIR — pw.x wavefunctions overflow tmpfs, and MPI ranks on other
    nodes cannot see node-local files."""
    from slab.backends import _scratch_dir

    root = tmp_path / "shared-scratch"
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(f'[paths]\nscratch = "{root}"\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_CONFIG", raising=False)
    monkeypatch.delenv("SLAB_SITE_CONFIG", raising=False)
    scratch = _scratch_dir("slab-qe-")
    try:
        assert scratch.parent == root
        assert scratch.name.startswith("slab-qe-")
    finally:
        scratch.rmdir()


def test_wrapper_path_override_follows_exec_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PATH= wrapper makes exec resolve the payload ONLY under the assigned
    PATH — identity must stat the binary that runs, never a process-PATH
    shadow of the same name; and 'env -i' falls back to the system default
    path, not to nothing."""
    from slab.backends import _executable_identity, _which_payload

    override_bin = tmp_path / "override-bin"
    shadow_bin = tmp_path / "shadow-bin"
    for bins in (override_bin, shadow_bin):
        bins.mkdir()
        _script(bins / "pw.x", "exit 0\n")
    monkeypatch.setenv("PATH", f"{shadow_bin}:/usr/bin:/bin")
    identity = _executable_identity(f"env PATH={override_bin} pw.x")
    assert identity is not None
    assert str(override_bin / "pw.x") in identity  # the binary exec runs
    assert str(shadow_bin / "pw.x") not in identity  # the shadow never runs
    # env -i: exec falls back to the system default path (confstr CS_PATH).
    assert _which_payload("ls", "env -i ls") is not None


def test_env_flags_after_assignments_are_payload(tmp_path: Path) -> None:
    """env stops option parsing at the first assignment, so a flag after one
    is the utility name — the guards must refuse it by its own name instead
    of parsing it as a flag and passing a command that exits 127."""
    with pytest.raises(EngineNotAvailableError, match="'-u'"):
        get_calculator(
            "qe",
            command="env A=1 -u X pw.x",
            pseudo_dir=str(tmp_path),
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_shaped_recognizes_registry_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """The task-level qe guards are engine-semantic: a registry alias built
    on slab.backends.qe_calculator earns them too."""
    import slab.tasks as tasks

    def fake_describe(engine: str, options: object = None) -> dict:
        return {"calculator": "slab.backends.qe_calculator"}

    monkeypatch.setattr(tasks, "describe_engine", fake_describe)
    assert tasks._qe_shaped("qe-delta", None) is True
    monkeypatch.setattr(tasks, "describe_engine", lambda e, o=None: {"calculator": "x.Y"})
    assert tasks._qe_shaped("lammps-delta", None) is False
    assert tasks._qe_shaped("qe", None) is True  # the literal name, no lookup needed


# -- per-engine setup: module loads and exports, scoped by a wrapper --------------------


def _pw_behind_setup(tmp_path: Path) -> tuple[Path, list[str]]:
    """A fake pw.x reachable ONLY through setup lines, never the process PATH."""
    bins = tmp_path / "module-bin"
    bins.mkdir(exist_ok=True)
    _script(bins / "pw.x", 'echo "     Program PWSCF v.7.4.9 starts"\n')
    return bins, [f'export PATH="{bins}:$PATH"']


def test_qe_setup_wraps_the_engine_in_its_own_login_shell(tmp_path: Path) -> None:
    """setup lines (module loads, exports) are THIS engine's dependencies:
    materialized into a private #!/bin/bash -l wrapper that execs the real
    command — never applied job-wide, and removed with the calculator."""
    _bins, setup = _pw_behind_setup(tmp_path)
    calc = get_calculator(
        "qe",
        command="pw.x",
        pseudo_dir=str(tmp_path),
        setup=setup,
        input_data={"system": {"ecutwfc": 30.0}},
    )
    wrapper = Path(str(calc.profile.command))
    text = wrapper.read_text()
    assert text.startswith("#!/bin/bash -l\nset -e\n")
    assert setup[0] in text
    assert 'exec pw.x "$@"' in text
    setup_dir = calc._slab_setup_dir
    assert wrapper.parent == setup_dir
    close_calculator(calc)
    assert not setup_dir.exists()


def test_qe_setup_runs_the_scenario_end_to_end(tmp_path: Path, ws: Workspace) -> None:
    """The wrapper is not decoration: a relax replay runs pw.x found only by
    the setup shell, with artifacts collected as ever."""
    bins = tmp_path / "module-bin"
    bins.mkdir()
    _script(bins / "pw.x", f'cat "{FIXTURE_PWO}"\n')
    setup = [f'export PATH="{bins}:$PATH"']
    atoms = bulk("Si", "diamond", a=5.43)
    with ws.start_run(name="qe-setup", intent="relax through a setup wrapper"):
        _relaxed, info = relax(
            atoms,
            engine="qe",
            fmax=10.0,
            label="si-setup",
            calculator_options={
                "command": "pw.x",
                "pseudo_dir": str(tmp_path),
                "pseudopotentials": {"Si": "Si.pz-vbc.UPF"},
                "setup": setup,
                "kpts": None,  # explicit Γ-only opt-in; the replay ignores inputs
                "input_data": {"system": {"ecutwfc": 30.0}},
            },
        )
    assert info["energy"] == pytest.approx(FIXTURE_ENERGY_EV)
    assert info["engine_version"] == "7.4.1"  # the fixture's banner, probed in-shell


def test_qe_setup_identity_and_version_probe_in_shell(tmp_path: Path) -> None:
    """Identity stamps the LOGICAL command plus the setup lines, and the
    version comes from a probe inside the setup shell — the module may be
    what provides the binary."""
    _bins, setup = _pw_behind_setup(tmp_path)
    options = {"command": "pw.x", "pseudo_dir": str(tmp_path), "setup": setup}
    identity = describe_engine("qe", options)
    assert identity["command"] == "pw.x"  # never the wrapper path
    assert identity["setup"] == setup
    assert identity["version"] == "7.4.9"
    # And a versionless setup engine still gets a resolved fingerprint.
    _script(tmp_path / "module-bin" / "quiet.x", "exit 0\n")
    quiet = describe_engine(
        "qe", {"command": "quiet.x", "pseudo_dir": str(tmp_path), "setup": setup}
    )
    assert quiet["version"] is None
    assert quiet["executable_fingerprint"][0] == str(tmp_path / "module-bin" / "quiet.x")


def test_qe_setup_failure_is_loud_with_the_shells_words(tmp_path: Path) -> None:
    with pytest.raises(EngineNotAvailableError, match="after its setup lines ran"):
        get_calculator(
            "qe",
            command="definitely-not-pw",
            pseudo_dir=str(tmp_path),
            setup=["echo module load qe/9.9 failed >&2"],
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_setup_failure_is_remembered_briefly_then_re_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal costs a login shell — up to the probe timeout of one — and
    the guard runs per calculator, so an agent retrying a typo'd module name
    would pay for one on every task. The refusal is remembered instead, and
    only for a TTL: a failure has none of the staleness signal a success has
    (no resolved path, so no mtime to fold into identity)."""
    monkeypatch.setattr(backends, "_SETUP_WHICH_CACHE", {})
    monkeypatch.setattr(backends, "_SETUP_WHICH_FAIL_CACHE", {})
    probes = tmp_path / "probes"
    setup = [f'echo probe >> "{probes}"']

    def refuse() -> str:
        match = "after its setup lines ran"
        with pytest.raises(EngineNotAvailableError, match=match) as excinfo:
            get_calculator(
                "qe",
                command="definitely-not-pw",
                pseudo_dir=str(tmp_path),
                setup=setup,
                input_data={"system": {"ecutwfc": 30.0}},
            )
        return str(excinfo.value)

    first = refuse()
    assert probes.read_text().count("probe") == 1
    assert "remembered" not in first  # the shell's own words, freshly read

    second = refuse()  # inside the TTL: no second login shell is spawned
    assert probes.read_text().count("probe") == 1
    assert "remembered" in second  # and it says so, rather than posing as fresh

    monkeypatch.setattr(backends, "_SETUP_WHICH_TTL_S", 0.0)
    third = refuse()  # TTL lapsed: the environment is asked again
    assert probes.read_text().count("probe") == 2
    assert "remembered" not in third


def test_qe_setup_recovers_when_the_module_farm_is_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the refusal is remembered and not cached: a binary that appears
    behind unchanged setup lines is found once the TTL lapses, with no
    process restart — and the refusal is forgotten rather than left to
    shadow the success."""
    monkeypatch.setattr(backends, "_SETUP_WHICH_CACHE", {})
    monkeypatch.setattr(backends, "_SETUP_WHICH_FAIL_CACHE", {})
    bins = tmp_path / "module-bin"
    bins.mkdir()
    setup = [f'export PATH="{bins}:$PATH"']
    options = {
        "command": "pw.x",
        "pseudo_dir": str(tmp_path),
        "setup": setup,
        "input_data": {"system": {"ecutwfc": 30.0}},
    }
    with pytest.raises(EngineNotAvailableError, match="after its setup lines ran"):
        get_calculator("qe", **options)

    _script(bins / "pw.x", 'echo "     Program PWSCF v.7.4.9 starts"\n')
    monkeypatch.setattr(backends, "_SETUP_WHICH_TTL_S", 0.0)

    calc = get_calculator("qe", **options)
    close_calculator(calc)
    key = (tuple(setup), "pw.x")
    assert key in backends._SETUP_WHICH_CACHE
    assert key not in backends._SETUP_WHICH_FAIL_CACHE


def test_setup_resolution_restats_but_memoizes_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the login shell is memoized, never the mtime. A binary rebuilt in
    place must reach the next cache identity exactly as it does without setup
    lines — otherwise a rebuilt pw.x would keep serving results keyed to the
    binary it replaced."""
    monkeypatch.setattr(backends, "_SETUP_WHICH_CACHE", {})
    monkeypatch.setattr(backends, "_SETUP_WHICH_FAIL_CACHE", {})
    bins, setup = _pw_behind_setup(tmp_path)
    probes = tmp_path / "probes"
    setup = [f'echo probe >> "{probes}"', *setup]

    first = backends._setup_which(tuple(setup), "pw.x")[0]
    assert first is not None
    _script(bins / "pw.x", 'echo "     Program PWSCF v.8.0 starts"\n')
    os.utime(bins / "pw.x", (0, 0))  # a rebuild, at the same path

    second = backends._setup_which(tuple(setup), "pw.x")[0]
    assert second is not None
    assert second[0] == first[0]  # same path, resolved once
    assert second[1] != first[1]  # fresh mtime, so the identity moves
    assert probes.read_text().count("probe") == 1  # and no second login shell


def test_setup_resolution_re_resolves_when_the_window_lapses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stat cannot see a module farm REPOINT the name at a different path —
    only the shell can, so the memo asks it again once the window lapses."""
    monkeypatch.setattr(backends, "_SETUP_WHICH_CACHE", {})
    monkeypatch.setattr(backends, "_SETUP_WHICH_FAIL_CACHE", {})
    old_bins = tmp_path / "old-bin"
    new_bins = tmp_path / "new-bin"
    for bins in (old_bins, new_bins):
        bins.mkdir()
        _script(bins / "pw.x", 'echo "     Program PWSCF v.7.4.9 starts"\n')
    pointer = tmp_path / "which-bin"
    pointer.write_text(str(old_bins))
    setup = (f'export PATH="$(cat {pointer}):$PATH"',)

    first = backends._setup_which(setup, "pw.x")[0]
    assert first is not None and first[0] == str(old_bins / "pw.x")

    pointer.write_text(str(new_bins))
    assert backends._setup_which(setup, "pw.x")[0] == first  # inside the window

    monkeypatch.setattr(backends, "_SETUP_WHICH_TTL_S", 0.0)
    third = backends._setup_which(setup, "pw.x")[0]
    assert third is not None and third[0] == str(new_bins / "pw.x")


def test_qe_setup_conflicts_with_profile(tmp_path: Path) -> None:
    from ase.calculators.espresso import EspressoProfile

    profile = EspressoProfile(command="/bin/echo", pseudo_dir=str(tmp_path))
    with pytest.raises(EngineNotAvailableError, match="not both"):
        get_calculator("qe", profile=profile, setup=["true"])


def test_qe_setup_srun_guard_still_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bins, setup = _pw_behind_setup(tmp_path)
    _script(bins / "srun", "exit 0\n")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(EngineNotAvailableError, match="not inside a SLURM allocation"):
        get_calculator(
            "qe",
            command="srun pw.x",
            pseudo_dir=str(tmp_path),
            setup=setup,
            input_data={"system": {"ecutwfc": 30.0}},
        )


def test_qe_setup_from_config_and_per_call_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[engines.qe] setup is the machine fact; a per-call setup= overrides it."""
    _bins, setup = _pw_behind_setup(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    setup_toml = ", ".join(f"'{line}'" for line in setup)  # literal strings: lines hold quotes
    (project / "slab.toml").write_text(
        f'[engines.qe]\ncommand = "pw.x"\npseudo_dir = "{tmp_path}"\nsetup = [{setup_toml}]\n'
    )
    monkeypatch.chdir(project)
    calc = get_calculator("qe", input_data={"system": {"ecutwfc": 30.0}})
    try:
        assert setup[0] in Path(str(calc.profile.command)).read_text()
    finally:
        close_calculator(calc)
    identity = describe_engine("qe", {})
    assert identity["setup"] == setup
    with pytest.raises(EngineNotAvailableError, match="after its setup lines ran"):
        get_calculator(
            "qe",
            setup=["export PATH=/nowhere"],  # per-call wins; pw.x vanishes
            input_data={"system": {"ecutwfc": 30.0}},
        )
