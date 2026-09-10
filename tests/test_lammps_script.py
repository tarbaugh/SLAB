"""run_lammps: a LAMMPS input script run whole, traced. A fake lmp keeps it
real-LAMMPS-free.

The fake speaks LAMMPS's command-line protocol (``-in``, ``-log``, ``-h``),
echoes the script into the log the way ``echo log`` does, writes one thermo
table per ``run`` with the header ``thermo_style custom`` asks for, the
``Loop time`` and ``Total wall time`` lines, and the files that ``dump``,
``write_data``, ``write_restart``, and ``fix ave/time`` name. It fails the
way LAMMPS fails: the echoed command, an ``ERROR: ... (src/...)`` line, exit
1. A gated test at the bottom runs the skill's template against a real
binary when ``$SLAB_TEST_LMP`` names one.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from ase.build import bulk

from foundation import ExecutionStatus, Workspace
from foundation.tasks import run_lammps
from slab.errors import EngineNotAvailableError, LammpsScriptError
from slab.lammps import (
    LOG_NAME,
    SCREEN_NAME,
    describe_lammps,
    error_lines,
    kokkos_report,
    kokkos_switches,
    lammps_route,
    lammps_routes,
    run_lammps_script,
)
from slab.outputs import lammps_thermo

FIXTURE_LOG = Path(__file__).parent / "data" / "lammps-cu-relax-final.log"
SKILLS = Path(__file__).parent.parent / "src" / "foundation" / "skills"

_FAKE = f'''#!{sys.executable}
"""A LAMMPS that never integrates anything, but writes what LAMMPS writes."""
import os
import re
import sys
import time

args = sys.argv[1:]
if "-h" in args:
    print("Large-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025 - Update 4")
    sys.exit(0)
script = open(args[args.index("-in") + 1]).read()
log_name = args[args.index("-log") + 1] if "-log" in args else "log.lammps"
NAMES = {{"step": "Step", "temp": "Temp", "pe": "PotEng", "ke": "KinEng",
         "etotal": "TotEng", "press": "Press", "vol": "Volume"}}
lines = ["LAMMPS (22 Jul 2025 - Update 4)"]
suffix_kk = "-sf" in args and args[args.index("-sf") + 1] == "kk"
if "-k" in args and args[args.index("-k") + 1] == "on":
    kk = args[args.index("-k") + 2:args.index("-k") + 6]
    gpus = int(kk[kk.index("g") + 1]) if "g" in kk else 0
    threads = int(kk[kk.index("t") + 1]) if "t" in kk else 1
    lines.append("KOKKOS mode with Kokkos version 4.6.1 is enabled (src/KOKKOS/kokkos.cpp:72)")
    lines.append("  will use up to %d GPU(s) per node" % gpus)
    lines.append("  using %d OpenMP thread(s) per MPI task" % threads)
if os.environ.get("FAKE_MARK"):
    lines.append("FAKE_MARK=" + os.environ["FAKE_MARK"])
natoms, every, temp, step = 0, 1, 300.0, 0
columns = ["Step", "Temp", "E_pair", "E_mol", "TotEng", "Press"]
failed = False
for raw in script.splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    lines.append(raw)
    tok = line.split()
    cmd = tok[0]
    if cmd == "read_data":
        for data_line in open(tok[1]):
            m = re.match(r"\\s*(\\d+)\\s+atoms", data_line)
            if m:
                natoms = int(m.group(1))
        lines.append("  %d atoms" % natoms)
    elif cmd == "create_atoms":
        natoms = 108
        lines.append("Created %d atoms" % natoms)
    elif cmd == "thermo":
        every = int(tok[1])
    elif cmd == "thermo_style" and tok[1] == "custom":
        columns = [NAMES.get(t, t) for t in tok[2:]]
    elif cmd == "velocity" and "create" in tok:
        temp = float(tok[tok.index("create") + 1])
    elif cmd == "pair_style" and tok[1] == "nonsense":
        lines.append("ERROR: Unrecognized pair style 'nonsense' (src/force.cpp:275)")
        failed = True
        break
    elif cmd == "pair_style" and suffix_kk:
        lines.append("Neighbor list info ...")
        lines.append("  (1) pair %s/kk, perpetual" % tok[1])
    elif cmd == "dump":
        with open(tok[5], "w") as handle:
            handle.write("ITEM: TIMESTEP\\n0\\nITEM: NUMBER OF ATOMS\\n%d\\n" % natoms)
    elif cmd == "write_data":
        with open(tok[1], "w") as handle:
            handle.write("LAMMPS data file via write_data\\n\\n%d atoms\\n" % natoms)
    elif cmd == "write_restart":
        with open(tok[1], "wb") as handle:
            handle.write(b"restart")
    elif cmd == "fix" and "ave/time" in tok and "file" in tok:
        with open(tok[tok.index("file") + 1], "w") as handle:
            handle.write("# Time-averaged data\\n")
    elif cmd == "print" and "slow" in line:
        time.sleep(30)
    elif cmd == "run":
        n = int(tok[1])
        lines.append(" ".join(columns))
        for s in range(step, step + n + 1, every):
            wobble = 1.0 + 0.01 * ((s // every) % 3 - 1)
            row = {{"Step": s, "Temp": temp * wobble, "E_pair": -3.5 * natoms,
                   "PotEng": -3.5 * natoms, "KinEng": 0.0388 * natoms * wobble,
                   "E_mol": 0.0, "TotEng": -3.46 * natoms, "Press": 1500.0 * wobble,
                   "Volume": 3929.35}}
            cells = ["%d" % row[c] if c == "Step" else "%.6g" % row.get(c, 0.0) for c in columns]
            lines.append(" ".join(cells))
            if s == step + every and "warn-inside" in script:
                lines.append("WARNING: Inconsistent image flags (src/domain.cpp:1)")
        lines.append("Loop time of 0.0123 on 1 procs for %d steps with %d atoms" % (n, natoms))
        step += n
if not failed:
    lines.append("Total wall time: 0:00:01")
with open(log_name, "w") as handle:
    handle.write("\\n".join(lines) + "\\n")
sys.stdout.write("LAMMPS (22 Jul 2025 - Update 4)\\n")
if failed:
    sys.stdout.write(lines[-1] + "\\n")
    sys.exit(1)
'''

_NO_LOG = f'''#!{sys.executable}
import sys
if "-h" in sys.argv:
    print("Large-scale Atomic/Molecular Massively Parallel Simulator - 22 Jul 2025")
    sys.exit(0)
sys.stderr.write("dyld: Library not loaded: @rpath/libmpi.12.dylib\\n")
sys.exit(127)
'''

SCRIPT = """\
units metal
atom_style atomic
boundary p p p
read_data structure.data
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
velocity all create 300.0 4928459 mom yes rot yes dist gaussian
timestep 0.002
fix nvt all nvt temp 300.0 300.0 0.2
thermo 100
thermo_style custom step temp pe ke etotal press vol
dump traj all custom 500 ar.dump id type x y z
fix avg all ave/time 10 10 100 c_thermo_temp file ar-temp.txt
run 1000
write_data ar-final.data
write_restart ar.restart
"""

EAM_SCRIPT = """\
units metal
atom_style atomic
lattice fcc 3.615
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
pair_style eam
pair_coeff 1 1 Cu_u3.eam
thermo 10
run 100
"""


def _script(path: Path, body: str) -> str:
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


@pytest.fixture()
def fake_lmp(tmp_path: Path) -> str:
    return _script(tmp_path / "fake-lmp", _FAKE)


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _argon():
    return bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)


# -- the thermo parser and the runner -------------------------------------------


def test_lammps_thermo_reads_a_real_log_and_a_synthetic_one() -> None:
    real = lammps_thermo(FIXTURE_LOG.read_text())
    assert real and real[0]["columns"][0] == "Step"
    assert real[0]["loop"]["steps"] == 0  # an ASE-driven run 0
    text = (
        "Step Temp PotEng\n0 300 -3.5\nWARNING: something (src/x.cpp:1)\n100 298.2 -3.49\n"
        "Loop time of 0.5 on 4 procs for 100 steps with 32 atoms\nrun 50\n"
        "Step Temp PotEng\n100 298.2 -3.49\n150 297 -3.48\n"
    )
    tables = lammps_thermo(text)
    assert [len(t["rows"]) for t in tables] == [2, 2]
    assert tables[0]["rows"][1] == [100, 298.2, -3.49]  # the warning did not end the table
    assert tables[0]["loop"] == {"seconds": 0.5, "procs": 4, "steps": 100, "atoms": 32}
    assert tables[1]["loop"] is None  # the log stopped before the second loop line


def test_run_lammps_script_runs_whole_and_keeps_log_and_screen(
    tmp_path: Path, fake_lmp: str
) -> None:
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    script = EAM_SCRIPT.replace("pair_coeff 1 1 Cu_u3.eam", "pair_coeff 1 1 x")
    (cwd / "in.lammps").write_text(script)
    outcome = run_lammps_script(cwd=cwd, command=fake_lmp)
    assert outcome.argv[-4:] == (fake_lmp, "-in", "in.lammps", "-log", "log.lammps")[-4:]
    assert "Created 108 atoms" in outcome.log
    assert "Total wall time" in outcome.log
    assert (cwd / LOG_NAME).is_file() and (cwd / SCREEN_NAME).read_text() == outcome.screen
    assert describe_lammps(fake_lmp)["version"] == "22 Jul 2025 - Update 4"


def test_run_lammps_script_fails_the_way_lammps_fails(tmp_path: Path, fake_lmp: str) -> None:
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    (cwd / "in.lammps").write_text("units metal\npair_style nonsense\nrun 10\n")
    with pytest.raises(LammpsScriptError, match="Unrecognized pair style") as excinfo:
        run_lammps_script(cwd=cwd, command=fake_lmp)
    message = str(excinfo.value)
    assert "exit 1" in message
    assert "context: pair_style nonsense" in message  # the echoed command that died
    assert "ERROR: Unrecognized" in excinfo.value.log


def test_run_lammps_script_says_when_lammps_wrote_nothing(tmp_path: Path) -> None:
    dead = _script(tmp_path / "no-log-lmp", _NO_LOG)
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    (cwd / "in.lammps").write_text("units metal\n")
    with pytest.raises(LammpsScriptError, match="exit 127") as excinfo:
        run_lammps_script(cwd=cwd, command=dead)
    assert "Library not loaded" in excinfo.value.screen
    assert excinfo.value.log == ""
    assert error_lines("") == ["LAMMPS wrote nothing: the process died before the script started"]


def test_run_lammps_script_kills_the_process_group_on_timeout(
    tmp_path: Path, fake_lmp: str
) -> None:
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    (cwd / "in.lammps").write_text('units metal\nprint "slow"\nrun 10\n')
    with pytest.raises(LammpsScriptError, match="did not finish within 1s"):
        run_lammps_script(cwd=cwd, command=fake_lmp, timeout_s=1.0)


def test_run_lammps_script_refuses_a_missing_binary_and_runs_setup_lines(
    tmp_path: Path, fake_lmp: str
) -> None:
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    (cwd / "in.lammps").write_text("units metal\nrun 10\n")
    with pytest.raises(EngineNotAvailableError, match="not on PATH"):
        run_lammps_script(cwd=cwd, command="definitely-not-a-lammps-binary")
    outcome = run_lammps_script(cwd=cwd, command=fake_lmp, setup=["export FAKE_MARK=seen"])
    assert "FAKE_MARK=seen" in outcome.log  # the script ran inside the setup shell


# -- the task ----------------------------------------------------------------------


def test_run_lammps_refuses_bad_staging_up_front(tmp_path: Path, fake_lmp: str) -> None:
    potential = tmp_path / "Cu_u3.eam"
    potential.write_text("comment\n29 63.55 3.615 fcc\n")
    as_file = tmp_path / "in.lammps"
    as_file.write_text(EAM_SCRIPT)
    with pytest.raises(LammpsScriptError, match="looks like a path"):
        run_lammps(str(as_file), command=fake_lmp)
    with pytest.raises(LammpsScriptError, match="is empty"):
        run_lammps("   \n", command=fake_lmp)
    with pytest.raises(LammpsScriptError, match="does not exist"):
        run_lammps(EAM_SCRIPT, files=[str(tmp_path / "missing.eam")], command=fake_lmp)
    with pytest.raises(LammpsScriptError, match="never mentions the staged file"):
        run_lammps("units metal\nrun 0\n", files=[str(potential)], command=fake_lmp)
    twin = tmp_path / "twin" / "Cu_u3.eam"
    twin.parent.mkdir()
    twin.write_text("other bytes\n")
    with pytest.raises(LammpsScriptError, match="share the basename"):
        run_lammps(EAM_SCRIPT, files=[str(potential), str(twin)], command=fake_lmp)
    with pytest.raises(LammpsScriptError, match=r"never reads 'structure\.data'"):
        run_lammps("units metal\nrun 0\n", atoms=_argon(), command=fake_lmp)
    with pytest.raises(LammpsScriptError, match="pass specorder="):
        run_lammps(SCRIPT, atoms=bulk("CuAu", "rocksalt", a=4.0), command=fake_lmp)
    with pytest.raises(LammpsScriptError, match="lacks Au"):
        run_lammps(
            SCRIPT, atoms=bulk("CuAu", "rocksalt", a=4.0), specorder=["Cu"], command=fake_lmp
        )
    with pytest.raises(LammpsScriptError, match="repeats a symbol"):
        run_lammps(
            SCRIPT, atoms=bulk("CuAu", "rocksalt", a=4.0), specorder=["Cu", "Cu"],
            command=fake_lmp,
        )
    with pytest.raises(LammpsScriptError, match="pass atoms too"):
        run_lammps(EAM_SCRIPT, specorder=["Cu"], command=fake_lmp)


def test_run_lammps_keeps_the_log_the_thermo_and_what_the_script_wrote(
    ws: Workspace, fake_lmp: str
) -> None:
    with ws.start_run(name="ar-nvt") as run:
        result, info = run_lammps(SCRIPT, atoms=_argon(), label="ar", command=fake_lmp)
    assert result["thermo"]["Step"] == 1000
    assert result["thermo"]["Temp"] == pytest.approx(300.0)
    assert set(result["thermo"]) == {
        "Step", "Temp", "PotEng", "KinEng", "TotEng", "Press", "Volume",
    }
    table = result["tables"][0]
    assert table["rows"] == 11 and table["loop"]["atoms"] == 32 and table["loop"]["steps"] == 1000
    assert table["tail"]["rows"] == 6
    assert table["tail"]["mean"]["Temp"] == pytest.approx(300.0, abs=2.0)
    assert result["steps"] == 1000 and result["wall_time"] == "0:00:01"
    assert info["types"] == {1: "Ar"} and info["version"] == "22 Jul 2025 - Update 4"
    assert info["command"] == fake_lmp and info["n_warnings"] == 0
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert {
        "ar.in", "ar.log", "ar.screen", "ar-structure.data", "ar-thermo.json",
        "ar.dump", "ar-temp.txt", "ar-final.data", "ar.restart",
    } <= names
    assert sorted(info["files"]) == ["ar-final.data", "ar-temp.txt", "ar.dump", "ar.restart"]
    for kept, digest in info["artifacts"].items():
        assert ws.runs.get_artifact(run.id, kept).hash == digest
    thermo = json.loads(ws.artifacts.get(info["artifacts"]["ar-thermo.json"]).read_text())
    assert thermo[0]["columns"][1] == "Temp" and len(thermo[0]["rows"]) == 11
    data = ws.artifacts.get(info["artifacts"]["ar-structure.data"]).read_text()
    assert "32 atoms" in data and "Masses" in data and "39.9" in data
    kept_script = ws.artifacts.get(info["artifacts"]["ar.in"]).read_text()
    assert kept_script == SCRIPT
    assert ws.runs.get(run.id).status is ExecutionStatus.COMPLETED


def test_kokkos_switches_read_the_command_and_kokkos_report_reads_the_log() -> None:
    """SLAB adds no switch, so the command alone says whether KOKKOS is on."""
    assert kokkos_switches("lmp")["enabled"] is False
    assert kokkos_switches("env OMP_NUM_THREADS=4 mpirun -np 2 lmp -k on t 8 -sf kk") == {
        "enabled": True, "gpus": None, "threads": 8, "suffix": True, "package": None,
    }
    gpu = kokkos_switches(
        "srun lmp -kokkos on g 2 t 1 -suffix kk -package kokkos newton on neigh half -in x"
    )
    assert gpu == {
        "enabled": True, "gpus": 2, "threads": 1, "suffix": True, "package": "newton on neigh half",
    }
    assert kokkos_switches("lmp -k off g 1")["enabled"] is False
    assert kokkos_switches("lmp -k on g notanumber")["gpus"] is None
    assert kokkos_switches("lmp 'unterminated")["enabled"] is False
    log = FIXTURE_LOG.read_text()
    assert kokkos_report(log) == {"enabled": False, "gpus": None, "threads": None, "styles": []}
    kokkos_log = (
        "LAMMPS (22 Jul 2025 - Update 4)\n"
        "KOKKOS mode with Kokkos version 4.6.1 is enabled (src/KOKKOS/kokkos.cpp:72)\n"
        "  will use up to 2 GPU(s) per node\n"
        "  using 1 OpenMP thread(s) per MPI task\n"
        "Neighbor list info ...\n"
        "  (1) pair grace/2l/kk, perpetual\n"
        "  (2) fix nvt/kk, occasional\n"
    )
    assert kokkos_report(kokkos_log) == {
        "enabled": True, "gpus": 2, "threads": 1, "styles": ["grace/2l/kk", "nvt/kk"],
    }


def test_run_lammps_records_what_kokkos_did_and_the_exact_argv(
    ws: Workspace, fake_lmp: str
) -> None:
    """A GPU run has to show it: the switches asked for, and what the log says ran."""
    accelerated = f"{fake_lmp} -k on g 1 -sf kk -pk kokkos newton on neigh half"
    with ws.start_run(name="kk"):
        _, info = run_lammps(SCRIPT, atoms=_argon(), label="kk", command=accelerated)
    assert info["command"] == accelerated
    assert info["argv"][0] == fake_lmp and info["argv"][-4:] == [
        "-in", "in.lammps", "-log", "log.lammps",
    ]
    assert info["kokkos"] == {
        "enabled": True,
        "gpus": 1,
        "threads": 1,
        "styles": ["lj/cut/kk"],
        "switches": {
            "enabled": True, "gpus": 1, "threads": None, "suffix": True,
            "package": "newton on neigh half",
        },
    }
    with ws.start_run(name="plain"):
        _, plain = run_lammps(SCRIPT, atoms=_argon(), label="plain", command=fake_lmp)
    assert plain["kokkos"]["enabled"] is False and plain["kokkos"]["styles"] == []
    assert plain["kokkos"]["switches"]["enabled"] is False


def _registry_with_routes(
    tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A registry with a KOKKOS LAMMPS alias, a plain alias, and a non-LAMMPS alias."""
    registry = tmp_path / "engines.json"
    registry.write_text(json.dumps({
        "cluster": "delta",
        "engines": {
            "lammps-gpu": {
                "calculator": "slab.backends.lammps_calculator",
                "options": {
                    "command": f"{fake_lmp} -k on g 1 -sf kk",
                    "setup": ["export FAKE_MARK=gpu"],
                },
                "version": "22 Jul 2025 - Update 4",
            },
            "lammps-plain": {
                "calculator": "slab.backends.lammps_calculator",
                "options": {"command": fake_lmp},
            },
            "emt-cluster": {"calculator": "ase.calculators.emt.EMT"},
        },
    }))
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    monkeypatch.setenv("ASE_LAMMPSRUN_COMMAND", fake_lmp)
    return registry


def test_lammps_routes_name_every_build_and_refuse_the_rest(
    tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain build and a KOKKOS build coexist as routes; a run picks one by name."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("ASE_LAMMPSRUN_COMMAND", fake_lmp)
    assert list(lammps_routes()) == ["lammps"]
    _registry_with_routes(tmp_path, fake_lmp, monkeypatch)
    routes = lammps_routes()
    assert list(routes) == ["lammps", "lammps-gpu", "lammps-plain"]
    assert routes["lammps"]["source"] == "builtin" and routes["lammps"]["command"] == fake_lmp
    assert routes["lammps"]["kokkos"]["enabled"] is False
    gpu = routes["lammps-gpu"]
    assert gpu["source"] == "registry:delta" and gpu["setup"] == ["export FAKE_MARK=gpu"]
    assert gpu["kokkos"] == {
        "enabled": True, "gpus": 1, "threads": None, "suffix": True, "package": None,
    }
    assert lammps_route("lammps-gpu")["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    assert lammps_route(None) == lammps_route("lammps") == lammps_route(" LAMMPS ")
    with pytest.raises(EngineNotAvailableError, match=r"names no engine here.*lammps-gpu"):
        lammps_route("lammps-tpu")
    with pytest.raises(EngineNotAvailableError, match="is not a LAMMPS route"):
        lammps_route("emt-cluster")


def test_run_lammps_takes_a_route_by_name_and_records_it(
    ws: Workspace, tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    _registry_with_routes(tmp_path, fake_lmp, monkeypatch)
    with ws.start_run(name="gpu") as run:
        _, gpu = run_lammps(SCRIPT, atoms=_argon(), label="gpu", engine="lammps-gpu")
    assert gpu["route"] == "lammps-gpu" and gpu["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    assert gpu["setup"] == ["export FAKE_MARK=gpu"] and gpu["kokkos"]["gpus"] == 1
    assert "FAKE_MARK=gpu" in ws.artifacts.get(gpu["artifacts"]["gpu.log"]).read_text()
    (task,) = ws.runs.list_tasks(run.id)
    assert task.recipe["extra"]["route"] == "lammps-gpu"
    assert task.recipe["extra"]["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    with ws.start_run(name="plain"):
        _, plain = run_lammps(SCRIPT, atoms=_argon(), label="plain")
    assert plain["route"] == "lammps" and plain["kokkos"]["enabled"] is False
    with ws.start_run(name="override"):
        _, over = run_lammps(
            SCRIPT, atoms=_argon(), label="o", engine="lammps-gpu", command=fake_lmp
        )
    assert over["route"] == "lammps-gpu" and over["command"] == fake_lmp
    assert over["setup"] == ["export FAKE_MARK=gpu"]
    with pytest.raises(EngineNotAvailableError, match="is not a LAMMPS route"):
        run_lammps(SCRIPT, atoms=_argon(), engine="emt-cluster")


def test_run_lammps_types_follow_specorder_and_warnings_are_collected(
    ws: Workspace, fake_lmp: str
) -> None:
    alloy = bulk("CuAu", "rocksalt", a=4.0)
    script = SCRIPT.replace(
        "pair_coeff 1 1 0.0104 3.40", "pair_coeff * * 0.0104 3.40\n# warn-inside"
    )
    with ws.start_run(name="alloy") as run:
        result, info = run_lammps(
            script, atoms=alloy, specorder=["Au", "Cu"], label="alloy", command=fake_lmp
        )
    assert info["types"] == {1: "Au", 2: "Cu"}
    data = ws.artifacts.get(info["artifacts"]["alloy-structure.data"]).read_text()
    assert "2 atom types" in data
    assert info["warnings"] == ["WARNING: Inconsistent image flags (src/domain.cpp:1)"]
    assert result["tables"][0]["rows"] == 11  # the warning inside the table did not cut it
    assert ws.runs.get(run.id).status is ExecutionStatus.COMPLETED


def test_run_lammps_failure_keeps_evidence_and_names_the_error(
    ws: Workspace, fake_lmp: str
) -> None:
    dying = SCRIPT.replace("pair_style lj/cut 8.5", "pair_style nonsense")
    # Contexts exit innermost first: the run records the failure, then pytest sees it.
    with (
        pytest.raises(LammpsScriptError, match="Unrecognized pair style 'nonsense'") as excinfo,
        ws.start_run(name="dies") as run,
    ):
        run_lammps(dying, atoms=_argon(), label="ar", command=fake_lmp)
    notes = "\n".join(excinfo.value.__notes__)
    assert "LAMMPS files kept as artifacts" in notes
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert {"ar-failed.in", "ar-failed.log", "ar-failed.screen"} <= names
    failed = ws.runs.get(run.id)
    assert failed.status is ExecutionStatus.FAILED
    assert "Unrecognized pair style" in (failed.error or "")


def test_run_lammps_cache_identity_follows_the_script_the_files_and_the_command(
    ws: Workspace, tmp_path: Path, fake_lmp: str
) -> None:
    potential = tmp_path / "Cu_u3.eam"
    potential.write_text("comment\n29 63.55 3.615 fcc\n")
    with ws.start_run(name="one"):
        run_lammps(EAM_SCRIPT, files=[str(potential)], command=fake_lmp)
    with ws.start_run(name="two") as again:
        run_lammps(EAM_SCRIPT, files=[str(potential)], command=fake_lmp)
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True
    potential.write_text("comment\n29 63.55 3.700 fcc\n")  # same path, other bytes
    with ws.start_run(name="three") as changed:
        run_lammps(EAM_SCRIPT, files=[str(potential)], command=fake_lmp)
    assert ws.runs.list_tasks(changed.id)[0].cache_hit is False
    other = _script(tmp_path / "other-lmp", _FAKE)
    with ws.start_run(name="four") as elsewhere:
        run_lammps(EAM_SCRIPT, files=[str(potential)], command=other)
    assert ws.runs.list_tasks(elsewhere.id)[0].cache_hit is False
    with ws.start_run(name="five") as edited:
        run_lammps(EAM_SCRIPT.replace("run 100", "run 200"), files=[str(potential)], command=other)
    assert ws.runs.list_tasks(edited.id)[0].cache_hit is False


def test_run_lammps_is_in_the_task_catalog() -> None:
    from foundation._ops import describe_task, task_catalog

    names = [entry["name"] for entry in task_catalog()]
    assert "run_lammps" in names
    described = describe_task("run_lammps")
    assert "script" in described["signature"] and "structure.data" in described["doc"]


# -- the real thing ------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("SLAB_TEST_LMP"), reason="set $SLAB_TEST_LMP to a real lmp")
def test_the_md_template_runs_verified_under_a_real_lammps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The skill's own promise, executed: template -> verified run."""
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        f'[engines.lammps]\ncommand = "{os.environ["SLAB_TEST_LMP"]}"\n'
    )
    template = SKILLS / "lammps-scripting" / "assets" / "md_nvt.py"
    result = launch_script(
        tmp_path / ".slab",
        template,
        name="ar-nvt-shakeout",
        intent="skill template shakeout (argon LJ, 2000 steps)",
        capture_output=True,
    )
    assert result["state"] == "verified", result
    assert result["checks_passed"] == result["checks_total"] == 2
    assert "tail of" in result["output"]
