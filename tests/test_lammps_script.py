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
    lammps_build,
    lammps_builds,
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
if os.environ.get("FAKE_CUDA_BUSY") and "-k" in args:
    # A GPU build whose device is held: the banner reaches the log, the
    # Kokkos abort reaches the screen alone, and the exit is an abort.
    with open(log_name, "w") as handle:
        handle.write("\\n".join(lines) + "\\n")
    sys.stdout.write("\\n".join(lines) + "\\n")
    sys.stdout.write("terminate called after throwing an instance of 'std::runtime_error'\\n")
    sys.stdout.write("  what():  cudaSetDevice(cuda_device_id) error( cudaErrorDevicesUnavailable):"
                     " CUDA-capable device(s) is/are busy or unavailable"
                     " /opt/lammps/lib/kokkos/core/src/Cuda/Kokkos_Cuda_Instance.cpp:135\\n")
    sys.exit(134)
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
            handle.write("# Time-averaged data for fix %s\\n" % tok[1])
            handle.write("# TimeStep c_thermo_temp\\n")
            for k in range(1, 11):
                handle.write("%d %.3f\\n" % (100 * k, temp * (1.0 + 0.01 * (k % 3 - 1))))
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
    assert table["n_rows"] == 11 and table["loop"]["atoms"] == 32 and table["loop"]["steps"] == 1000
    assert table["tail"]["n_rows"] == 6
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


def _registry_with_builds(
    tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A registry with a KOKKOS LAMMPS alias, a plain alias, and a non-LAMMPS alias."""
    registry = tmp_path / "engines.json"
    registry.write_text(json.dumps({
        "cluster": "delta",
        "engines": {
            "lammps-kokkos": {
                "calculator": "slab.backends.lammps_calculator",
                "options": {
                    "command": f"{fake_lmp} -k on g 1 -sf kk",
                    "setup": ["export FAKE_MARK=kokkos"],
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


def _gpu_table(tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slab.toml whose plain build is the fake and whose gpu build is the fake
    under KOKKOS, asking for the launch's gpus."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        "[engines.lammps]\n"
        f'command = "{fake_lmp}"\n'
        "[engines.lammps.gpu]\n"
        f'command = "{fake_lmp} -k on g {{gpus}} -sf kk"\n'
        'setup = ["export FAKE_MARK=gpu"]\n'
    )
    monkeypatch.setenv("SLAB_CPUS", "0,1")
    monkeypatch.setenv("SLAB_NTASKS", "1")
    monkeypatch.setenv("SLAB_THREADS", "1")


def test_lammps_builds_name_every_build_and_refuse_the_rest(
    tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cpu build and the registry aliases coexist as builds; a script names
    an alias or lets the slice choose, and nothing else."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("SLAB_GPUS", "")
    monkeypatch.setenv("ASE_LAMMPSRUN_COMMAND", fake_lmp)
    assert list(lammps_builds()) == ["cpu"]
    _registry_with_builds(tmp_path, fake_lmp, monkeypatch)
    builds = lammps_builds()
    assert list(builds) == ["cpu", "lammps-kokkos", "lammps-plain"]
    assert builds["cpu"]["source"] == "builtin" and builds["cpu"]["command"] == fake_lmp
    assert builds["cpu"]["kokkos"]["enabled"] is False
    kokkos = builds["lammps-kokkos"]
    assert kokkos["source"] == "registry:delta" and kokkos["setup"] == ["export FAKE_MARK=kokkos"]
    assert kokkos["kokkos"] == {
        "enabled": True, "gpus": 1, "threads": None, "suffix": True, "package": None,
    }
    assert lammps_build("lammps-kokkos")["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    assert lammps_build("lammps-kokkos")["build"] == "lammps-kokkos"
    assert lammps_build(None) == lammps_build("lammps") == lammps_build(" LAMMPS ")
    assert lammps_build()["build"] == "cpu"
    with pytest.raises(EngineNotAvailableError, match=r"names no engine here.*cpu, lammps-kokkos"):
        lammps_build("lammps-tpu")
    with pytest.raises(EngineNotAvailableError, match="is not a LAMMPS build"):
        lammps_build("emt-cluster")


def test_run_lammps_takes_a_build_by_alias_and_records_it(
    ws: Workspace, tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """info['build'] is 'cpu' by default and the alias name for engine=<alias>;
    the cache identity carries the same key, and command= overrides."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SLAB_GPUS", "")
    _registry_with_builds(tmp_path, fake_lmp, monkeypatch)
    with ws.start_run(name="kokkos") as run:
        _, kokkos = run_lammps(SCRIPT, atoms=_argon(), label="kk", engine="lammps-kokkos")
    assert kokkos["build"] == "lammps-kokkos"
    assert kokkos["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    assert kokkos["setup"] == ["export FAKE_MARK=kokkos"] and kokkos["kokkos"]["gpus"] == 1
    assert "FAKE_MARK=kokkos" in ws.artifacts.get(kokkos["artifacts"]["kk.log"]).read_text()
    (task,) = ws.runs.list_tasks(run.id)
    assert task.recipe["extra"]["build"] == "lammps-kokkos"
    assert "route" not in task.recipe["extra"]
    assert task.recipe["extra"]["command"] == f"{fake_lmp} -k on g 1 -sf kk"
    with ws.start_run(name="plain"):
        _, plain = run_lammps(SCRIPT, atoms=_argon(), label="plain")
    assert plain["build"] == "cpu" and plain["kokkos"]["enabled"] is False
    assert "route" not in plain
    with ws.start_run(name="override"):
        _, over = run_lammps(
            SCRIPT, atoms=_argon(), label="o", engine="lammps-kokkos", command=fake_lmp
        )
    assert over["build"] == "lammps-kokkos" and over["command"] == fake_lmp
    assert over["setup"] == ["export FAKE_MARK=kokkos"]
    with pytest.raises(EngineNotAvailableError, match="is not a LAMMPS build"):
        run_lammps(SCRIPT, atoms=_argon(), engine="emt-cluster")


def test_run_lammps_follows_the_slice_to_the_gpu_build(
    ws: Workspace, tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With engine='lammps' the build follows the slice: under a launch that
    holds gpus and a declared [engines.lammps.gpu], info['build'] is 'gpu', the
    gpu command is filled from the envelope, and the gpu setup ran; without
    gpus the same call runs the plain build as 'cpu'. command= still wins."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    _gpu_table(tmp_path, fake_lmp, monkeypatch)
    monkeypatch.setenv("SLAB_GPUS", "0,1")
    with ws.start_run(name="gpu") as run:
        _, gpu = run_lammps(SCRIPT, atoms=_argon(), label="gpu")
    assert gpu["build"] == "gpu"
    assert gpu["command"] == f"{fake_lmp} -k on g 2 -sf kk"
    assert gpu["setup"] == ["export FAKE_MARK=gpu"] and gpu["kokkos"]["gpus"] == 2
    assert "FAKE_MARK=gpu" in ws.artifacts.get(gpu["artifacts"]["gpu.log"]).read_text()
    (task,) = ws.runs.list_tasks(run.id)
    assert task.recipe["extra"]["build"] == "gpu"
    # The template is the identity; the filled line is provenance.
    assert task.recipe["extra"]["command"] == f"{fake_lmp} -k on g {{gpus}} -sf kk"
    assert task.recipe["extra"]["provenance"]["command"] == f"{fake_lmp} -k on g 2 -sf kk"
    monkeypatch.setenv("SLAB_GPUS", "")
    with ws.start_run(name="cpu") as run:
        _, cpu = run_lammps(SCRIPT, atoms=_argon(), label="cpu")
    assert cpu["build"] == "cpu" and cpu["command"] == fake_lmp
    assert cpu["setup"] == [] and cpu["kokkos"]["enabled"] is False
    (task,) = ws.runs.list_tasks(run.id)
    assert task.recipe["extra"]["build"] == "cpu"
    monkeypatch.setenv("SLAB_GPUS", "0,1")
    with ws.start_run(name="override"):
        _, over = run_lammps(SCRIPT, atoms=_argon(), label="o", command=fake_lmp)
    assert over["build"] == "gpu" and over["command"] == fake_lmp
    assert over["kokkos"]["enabled"] is False


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
    assert result["tables"][0]["n_rows"] == 11  # the warning inside the table did not cut it
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


def test_a_device_error_is_quoted_from_the_screen_and_names_the_slice(
    tmp_path: Path, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign's failure: a GPU build dies in Kokkos::initialize within
    seconds. The log holds the banner alone, so the evidence comes from the
    screen, and the failure record's notes name the gpu ids the launch held,
    the budget they came from, and its source."""
    from foundation._ops import launch_script, run_details

    monkeypatch.setenv("FAKE_CUDA_BUSY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("SLAB_GPU_SOURCE", "slurm_job_gpus")
    monkeypatch.setenv("SLAB_GPUS", "1")
    monkeypatch.setenv("SLAB_CPUS", "0")
    workflow = tmp_path / "gpu.py"
    workflow.write_text(
        "from foundation.tasks import run_lammps\n"
        f"run_lammps({EAM_SCRIPT.replace('pair_style eam', 'pair_style lj/cut 8.5')!r}"
        f".replace('pair_coeff 1 1 Cu_u3.eam', 'pair_coeff 1 1 0.0104 3.40'),"
        f" label='cu', command={fake_lmp + ' -k on g 1 -sf kk'!r})\n"
    )
    result = launch_script(tmp_path / "ws", workflow, name="gpu", capture_output=True)
    assert (result["state"], result["status"]) == ("quarantined", "failed")
    message = result["failure"]["message"]
    assert "cudaErrorDevicesUnavailable" in message, message
    assert "context: terminate called" in message
    notes = result["failure"]["notes"]
    assert (
        "the launch held gpu id(s) 1 from a budget of 1 (source: slurm_job_gpus); a device "
        "that refuses within seconds is held by another process or is outside this job's "
        "allocation; check nvidia-smi inside the job"
    ) in notes
    with Workspace(tmp_path / "ws") as ws:
        details = run_details(ws, result["run_id"])
        names = {a.name for a in ws.runs.list_artifacts(result["run_id"])}
    assert {"cu-failed.log", "cu-failed.screen"} <= names
    (task_entry,) = details["tasks"]
    assert "the launch held gpu id(s) 1" in "\n".join(task_entry["failure"]["notes"])


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


# -- the result needs no memory ------------------------------------------------------


def test_run_lammps_result_carries_rate_seconds_atoms_and_parsed_averages(
    ws: Workspace, fake_lmp: str
) -> None:
    """Every number the campaign computed by hand is on the result, typed:
    seconds and rate are floats, n_rows is a count, wall_time stays text."""
    with ws.start_run(name="ar-nvt") as run:
        result, info = run_lammps(SCRIPT, atoms=_argon(), label="ar", command=fake_lmp)
    assert result["label"] == "ar"
    assert result["steps"] == 1000 and result["atoms"] == 32
    assert result["seconds"] == pytest.approx(0.0123)
    assert result["rate"]["steps_per_s"] == pytest.approx(1000 / 0.0123)
    assert result["rate"]["atom_steps_per_s"] == pytest.approx(32 * 1000 / 0.0123)
    assert isinstance(result["wall_time"], str)
    table = result["tables"][0]
    assert table["n_rows"] == 11 and "rows" not in table
    assert table["tail"]["n_rows"] == 6
    # The fix ave/time file came back parsed, in the shape of a table.
    assert list(result["averages"]) == ["ar-temp.txt"]
    average = result["averages"]["ar-temp.txt"]
    assert average["columns"] == ["TimeStep", "c_thermo_temp"]
    assert average["n_rows"] == 10 and average["loop"] is None
    assert average["first"]["TimeStep"] == 100 and average["last"]["TimeStep"] == 1000
    assert average["tail"]["mean"]["c_thermo_temp"] == pytest.approx(300.0, rel=0.02)
    # The parsed files are artifacts, and the result names their hashes.
    assert set(info["artifacts"]) >= {"ar-thermo.json", "ar-averages.json"}
    assert result["artifacts"]["thermo"] == info["artifacts"]["ar-thermo.json"]
    assert result["artifacts"]["averages"] == info["artifacts"]["ar-averages.json"]
    full = json.loads(ws.artifacts.get(result["artifacts"]["averages"]).read_text())
    assert full["ar-temp.txt"]["fix"] == "avg" and len(full["ar-temp.txt"]["rows"]) == 10
    assert ws.runs.get(run.id).status is ExecutionStatus.COMPLETED


def test_series_is_the_one_way_to_a_time_series(
    ws: Workspace, fake_lmp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """series() reads the full rows back from the parsed artifacts, keyed by
    column, from the result, from the info, or from the run id; after a cache
    hit the hashes still resolve, so a later run reads the earlier series."""
    from foundation.errors import FoundationError
    from foundation.tasks import series

    monkeypatch.setenv("SLAB_WORKSPACE", str(ws.root))
    with ws.start_run(name="first") as first:
        result, info = run_lammps(SCRIPT, atoms=_argon(), label="ar", command=fake_lmp)
        rows = series(result, 0)
        assert len(rows) == 11 and rows[0]["Step"] == 0 and rows[-1]["Step"] == 1000
        assert set(rows[0]) == set(result["tables"][0]["columns"])
        assert series(result, -1) == rows
        averaged = series(result, "ar-temp.txt")
        assert [row["TimeStep"] for row in averaged] == list(range(100, 1100, 100))
        assert series(info, "ar-temp.txt") == averaged
        with pytest.raises(FoundationError, match=r"no fix ave/time file 'nope.dat'.*ar-temp.txt"):
            series(result, "nope.dat")
        with pytest.raises(FoundationError, match="no thermo table 3; the run printed 1"):
            series(result, 3)
    assert series(first.id, 0, label="ar") == rows
    with ws.start_run(name="second") as second:
        again, _ = run_lammps(SCRIPT, atoms=_argon(), label="ar", command=fake_lmp)
        assert series(again, "ar-temp.txt") == averaged
    assert ws.runs.list_tasks(second.id)[0].cache_hit
    assert not ws.runs.list_tasks(first.id)[0].cache_hit


def test_run_lammps_refuses_an_output_path_with_a_directory(ws: Workspace, fake_lmp: str) -> None:
    """A fix ave/time file written into the project directory is not an
    artifact of any run; the script is refused before LAMMPS starts."""
    for line in (
        "fix avg all ave/time 10 10 100 c_thermo_temp file /tmp/proj/avg.dat",
        "dump traj all custom 500 out/ar.dump id type x y z",
        "write_data results/final.data",
        "write_restart ~/ar.restart",
        "restart 1000 chk/ar.restart",
    ):
        script = SCRIPT.replace("run 1000\n", f"{line}\nrun 1000\n")
        with (
            ws.start_run(name="bad"),
            pytest.raises(LammpsScriptError, match="a path with a directory component"),
        ):
            run_lammps(script, atoms=_argon(), label="ar", command=fake_lmp)
    # Refused before LAMMPS started: no run kept failure evidence.
    for run in ws.runs.list_runs():
        assert not [a for a in ws.runs.list_artifacts(run.id) if "failed" in a.name]

# -- dry run ---------------------------------------------------------------------------


STAGED_EAM = EAM_SCRIPT + "unfix nosuch\nminimize 1e-6 1e-8 1000 10000\n"


def test_rewrite_loops_empties_every_run_and_minimize() -> None:
    from slab.lammps import rewrite_loops

    text, lines = rewrite_loops(STAGED_EAM)
    assert "run 0\n" in text and "minimize 1e-6 1e-8 0 0\n" in text
    assert lines == ("run 100", "minimize 1e-6 1e-8 1000 10000")
    assert text.count("\n") == STAGED_EAM.count("\n")


def test_run_lammps_script_dry_run_rewrites_the_loops_in_place(
    tmp_path: Path, fake_lmp: str
) -> None:
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    (cwd / "in.lammps").write_text(EAM_SCRIPT.replace("Cu_u3.eam", "x"))
    outcome = run_lammps_script(cwd=cwd, command=fake_lmp, dry_run=True)
    assert outcome.dry_run is True
    assert outcome.rewritten == ("run 100",)
    assert "run 0\n" in (cwd / "in.lammps").read_text()
    assert outcome.argv[-4:] == ("-in", "in.lammps", "-log", "log.lammps")
    (table,) = lammps_thermo(outcome.log)
    assert len(table["rows"]) == 1 and table["loop"]["steps"] == 0
    plain = run_lammps_script(cwd=cwd, command=fake_lmp)
    assert plain.dry_run is False and plain.rewritten == ()


def test_run_lammps_inside_a_dry_run_empties_the_loops_and_records_them(
    ws: Workspace, tmp_path: Path, fake_lmp: str
) -> None:
    potential = tmp_path / "Cu_u3.eam"
    potential.write_text("comment\n29 63.55 3.615 fcc\n")
    with ws.start_run(name="rehearsal", dry_run=True) as run:
        assert run.dry_run is True
        result, info = run_lammps(
            EAM_SCRIPT, files=[str(potential)], command=fake_lmp, label="cu"
        )
    assert info["dry_run"] is True
    assert info["rewritten_lines"] == ["run 100"]
    assert result["steps"] == 0 and len(result["tables"]) == 1
    assert result["tables"][0]["n_rows"] == 1
    assert "run 0\n" in ws.artifacts.get(info["artifacts"]["cu.in"]).read_text()
    with ws.start_run(name="real") as real:
        assert real.dry_run is False
        result, info = run_lammps(EAM_SCRIPT, files=[str(potential)], command=fake_lmp)
    assert info["dry_run"] is False and info["rewritten_lines"] == []
    assert result["steps"] == 100


def test_dry_run_reports_each_run_lammps_call_and_the_python_after_it(
    tmp_path: Path, fake_lmp: str
) -> None:
    """The report lists every run_lammps call in order with its outcome, the
    files it would keep, and the traceback of the Python that died after it."""
    from foundation._ops import launch_script

    potential = tmp_path / "Cu_u3.eam"
    potential.write_text("comment\n29 63.55 3.615 fcc\n")
    dying = tmp_path / "dying.py"
    dying.write_text(
        "from foundation.tasks import run_lammps\n"
        f"script = {EAM_SCRIPT!r}\n"
        f"result, info = run_lammps(script, files=[{str(potential)!r}], "
        f"command={fake_lmp!r}, label='cu')\n"
        "print(result['tables'][-1]['loop']['steps'])\n"
        "print(result['msd'])\n"
    )
    report = launch_script(tmp_path / "ws", dying, dry_run=True, capture_output=True)
    assert report["reached_end"] is False
    assert report["output"] == "0\n"  # the table indexing before the KeyError ran
    assert "KeyError: 'msd'" in report["traceback"]
    assert report["lammps"] == [{"label": "cu", "outcome": "setup ok"}]
    assert report["outputs"] == [
        "cu.in", "cu.log", "cu-thermo.json", "cu-averages.json", "cu.screen",
    ]
    assert not (tmp_path / "ws").exists()
    broken = tmp_path / "broken.py"
    broken.write_text(
        "from foundation.tasks import run_lammps\n"
        f"run_lammps({EAM_SCRIPT!r}, files=[{str(potential)!r}], command={fake_lmp!r})\n"
        f"run_lammps('units metal\\npair_style nonsense\\nrun 10\\n', command={fake_lmp!r})\n"
    )
    report = launch_script(tmp_path / "ws", broken, dry_run=True)
    assert report["reached_end"] is False
    assert report["lammps"] == [
        {"label": "lammps", "outcome": "setup ok"},
        {
            "label": "lammps",
            "outcome": "ERROR: Unrecognized pair style 'nonsense' (src/force.cpp:275)",
        },
    ]


@pytest.mark.skipif(not os.environ.get("SLAB_TEST_LMP"), reason="set $SLAB_TEST_LMP to a real lmp")
def test_a_real_lammps_dry_run_sets_up_every_loop_and_integrates_nothing(
    tmp_path: Path,
) -> None:
    lmp = os.environ["SLAB_TEST_LMP"]
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    script = SCRIPT + "unfix nvt\nminimize 1e-6 1e-8 1000 10000\n"
    (cwd / "in.lammps").write_text(script)
    from ase.io import write as ase_write

    ase_write(cwd / "structure.data", _argon(), format="lammps-data", masses=True)
    outcome = run_lammps_script(cwd=cwd, command=lmp, dry_run=True)
    assert outcome.rewritten == ("run 1000", "minimize 1e-6 1e-8 1000 10000")
    assert "Total wall time: 0:00:00" in outcome.log
    tables = lammps_thermo(outcome.log)
    assert [len(t["rows"]) for t in tables] == [1, 1]
    assert [t["loop"]["steps"] for t in tables] == [0, 0]
    assert tables[0]["rows"][0][0] == 0  # the step-0 row
    assert (cwd / "ar.dump").read_text().count("ITEM: TIMESTEP") == 1  # the step-0 frame
    assert (cwd / "ar-temp.txt").read_text().count("\n") == 2  # the two header lines
    assert "atoms" in (cwd / "ar-final.data").read_text()
    assert (cwd / "ar.restart").stat().st_size > 0


STAGED_SCRIPT = """\
from ase.build import bulk

from foundation.tasks import run_lammps

atoms = bulk("Ar", "fcc", a=5.26, cubic=True) * (3, 3, 3)
script = '''\\
units metal
atom_style atomic
boundary p p p
read_data structure.data
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
velocity all create 300.0 4928459 mom yes rot yes dist gaussian
timestep 0.002
fix nvt all nvt temp 300.0 300.0 0.2
fix avg all ave/time 10 10 100 c_thermo_temp file ar-temp.txt
thermo 100
thermo_style custom step temp pe ke etotal press vol
run 1000
unfix nvt
fix npt all npt temp 300.0 300.0 0.2 iso 0.0 0.0 2.0
run 1000
unfix nosuch
write_data ar-final.data
'''
result, info = run_lammps(script, atoms=atoms, label="ar")
"""


@pytest.mark.skipif(not os.environ.get("SLAB_TEST_LMP"), reason="set $SLAB_TEST_LMP to a real lmp")
def test_a_real_dry_run_finds_a_stage_three_error_before_any_run_is_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        f'[engines.lammps]\ncommand = "{os.environ["SLAB_TEST_LMP"]}"\n'
    )
    staged = tmp_path / "staged.py"
    staged.write_text(STAGED_SCRIPT)
    report = launch_script(tmp_path / ".slab", staged, dry_run=True)
    assert report["reached_end"] is False
    (entry,) = report["lammps"]
    assert entry["label"] == "ar"
    assert "Could not find fix ID nosuch" in entry["outcome"]
    assert "Could not find fix ID nosuch" in report["traceback"]
    assert not (tmp_path / ".slab").exists()
    clean = tmp_path / "clean.py"
    clean.write_text(
        STAGED_SCRIPT.replace("unfix nosuch\n", "")
        + "table = result['tables'][-1]\n"
        "print('steps', result['steps'], 'tables', len(result['tables']), 'rows', "
        "table['n_rows'], 'loop', table['loop']['steps'], 'rewritten', info['rewritten_lines'])\n"
        "from foundation import check\n"
        "@check\ndef held():\n    assert abs(table['tail']['mean']['Temp'] - 300.0) < 30.0\n"
    )
    report = launch_script(tmp_path / ".slab", clean, dry_run=True, capture_output=True)
    assert report["reached_end"] is True, report
    assert report["lammps"] == [{"label": "ar", "outcome": "setup ok"}]
    assert (
        "steps 0 tables 2 rows 1 loop 0 rewritten ['run 1000', 'run 1000']" in report["output"]
    )
    assert report["checks"] == [
        {"name": "held", "passed": True, "message": "completed without assertion errors"}
    ]
    assert {"ar-temp.txt", "ar-final.data", "ar.in", "ar.log"} <= set(report["outputs"])
