"""Digests of engine output files: what a reader needs, without the noise.

A pw.x output for a 54-atom cell runs to 300 KB, and most of it is band
eigenvalues. A model that reads it in 400-line windows sees the eigenvalue
lists and not the SCF trace; one real session read an ascending list of
band energies in eV as a diverging total energy in Ry, declared the run
broken, retracted, declared it again, and compacted its context six times
in sixteen minutes doing so. The digests here are the summary a colleague
would give first: the system, the convergence trace, the final numbers,
the warnings, and whether the job finished. The raw text stays one
argument away.

Three formats are recognised: pw.x output (any name; the header says
``Program PWSCF``), a LAMMPS log (the first lines say ``LAMMPS (``, or open
with the commands an ASE-driven run echoes), and extended XYZ (a frame
count line, then a comment line with ``Lattice=`` or ``Properties=``).
:func:`digest` returns ``None`` for anything else, and the caller shows
the text as it always did. A source file is never digested: the format is
decided by the name's extension and by the file's own header, not by
keywords that a script mentions in its text. A workflow script that
builds a LAMMPS input was once read as a LAMMPS log.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

__all__ = [
    "THERMO_FORMATS",
    "digest",
    "extxyz_digest",
    "lammps_log_digest",
    "lammps_run_progress",
    "lammps_thermo",
    "lammps_thermo_format",
    "lammps_yaml_thermo",
    "pwscf_digest",
]

#: Ry/bohr to eV/Å, for the force line.
_RY_PER_BOHR_TO_EV_PER_A = 25.71104309541616
#: How many entries of a trace to show at each end.
_TRACE_ENDS = 3
#: How many per-cycle energies of a relaxation to show before eliding.
_CYCLE_ENERGIES_SHOWN = 12
#: How many warning or error lines to carry.
_MAX_NOTES = 6
#: How much of a file to look at when deciding its format.
_SNIFF_CHARS = 4_000
#: Extensions of files that are code or configuration, never engine output.
_SOURCE_EXTENSIONS = frozenset(
    {".py", ".sh", ".bash", ".zsh", ".toml", ".json", ".yaml", ".yml", ".md", ".cfg", ".ini"}
)


def digest(name: str, text: str) -> str | None:
    """The digest for *text*, by format, or ``None`` when no reader applies.

    Examples:
        >>> digest("notes.txt", "just a note") is None
        True
        >>> comment = 'Lattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3 energy=-1.5'
        >>> print(digest("cu.extxyz", f"1\\n{comment}\\nCu 0 0 0\\n"))
        extended XYZ digest: cu.extxyz (1 frame, 3 lines)
        atoms per frame: 1; species: Cu
        energy: present on 1 frame(s), -1.5 to -1.5
        forces: absent; lattice: present
        spacing: n/a (no frame holds two atoms)
        >>> digest("make_input.py", "lines = ['units metal', 'pair_style eam']\\n") is None
        True
    """
    if _is_source_file(name):
        return None
    head = text[:_SNIFF_CHARS]
    if "Program PWSCF" in head:
        return pwscf_digest(name, text)
    if _looks_like_lammps_log(head):
        return lammps_log_digest(name, text)
    if _looks_like_ave_time(head):
        return lammps_ave_time_digest(name, text)
    if _looks_like_extxyz(head, name):
        return extxyz_digest(name, text)
    return None


def _is_source_file(name: str) -> bool:
    """Whether *name* has the extension of code or configuration.

    Examples:
        >>> _is_source_file("build_cell.py"), _is_source_file("run.log")
        (True, False)
    """
    dot = name.rfind(".")
    return dot >= 0 and name[dot:].lower() in _SOURCE_EXTENSIONS


# -- pw.x ----------------------------------------------------------------------

_PW_FACTS = {
    "version": re.compile(r"Program PWSCF v\.(\S+) starts on"),
    "atoms": re.compile(r"number of atoms/cell\s*=\s*(\d+)"),
    "electrons": re.compile(r"number of electrons\s*=\s*([\d.]+)"),
    "ks_states": re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)"),
    "ecutwfc": re.compile(r"kinetic-energy cutoff\s*=\s*([\d.]+)\s*Ry"),
    "ecutrho": re.compile(r"charge density cutoff\s*=\s*([\d.]+)\s*Ry"),
    "mixing_beta": re.compile(r"mixing beta\s*=\s*([\d.]+)"),
    "xc": re.compile(r"Exchange-correlation\s*=\s*(.+?)\s*$", re.MULTILINE),
    "kpoints": re.compile(r"number of k points=\s*(\d+)"),
    "volume": re.compile(r"unit-cell volume\s*=\s*([\d.]+)\s*\(a\.u\.\)\^3"),
}
_PW_ITERATION = re.compile(r"^\s*iteration #\s*(\d+)")
_PW_TRACE_ENERGY = re.compile(r"^\s*total energy\s*=\s*(-?[\d.]+)\s*Ry")
_PW_FINAL_ENERGY = re.compile(r"^!\s*total energy\s*=\s*(-?[\d.]+)\s*Ry")
_PW_ACCURACY = re.compile(r"estimated scf accuracy\s*<\s*([\d.Ee+-]+)\s*Ry")
_PW_CONVERGED = re.compile(r"convergence has been achieved in\s*(\d+)\s*iterations")
_PW_NOT_CONVERGED = re.compile(r"convergence NOT achieved after\s*(\d+)\s*iterations")
_PW_FERMI = re.compile(r"the Fermi energy is\s*(-?[\d.]+)\s*ev")
_PW_HOMO = re.compile(r"highest occupied(?:, lowest unoccupied)? level\(s\)?\s*\(ev\):\s*(.+)")
_PW_FORCE = re.compile(
    r"^\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)"
)
_PW_TOTAL_FORCE = re.compile(r"Total force\s*=\s*([\d.]+)\s*Total SCF correction\s*=\s*([\d.]+)")
_PW_PRESSURE = re.compile(r"total\s+stress.*P=\s*(-?[\d.]+)")
_PW_BFGS = re.compile(r"bfgs converged in\s*(\d+)\s*scf cycles and\s*(\d+)\s*bfgs steps")
_PW_WALL = re.compile(r"^\s*PWSCF\s*:\s*(.+?)\s+CPU\s+(.+?)\s+WALL")
_PW_ERROR_FENCE = re.compile(r"^\s*%{10,}")
_PW_CARD_IGNORED = re.compile(r"Warning: card .* ignored")


@dataclass
class _ScfCycle:
    iterations: int = 0
    trace: list[float] = field(default_factory=list)
    final_energy: float | None = None
    accuracy: float | None = None
    converged: bool | None = None


def pwscf_digest(name: str, text: str) -> str:
    """The digest of one pw.x output: system, SCF trace, final numbers, status."""
    lines = text.splitlines()
    facts = {key: (m.group(1) if (m := rx.search(text)) else None) for key, rx in _PW_FACTS.items()}
    cycles: list[_ScfCycle] = []
    forces_max = 0.0
    forces_seen = False
    total_force: tuple[str, str] | None = None
    pressure: str | None = None
    fermi: str | None = None
    bfgs: tuple[str, str] | None = None
    wall: str | None = None
    warnings: list[str] = []
    errors: list[str] = []
    in_error = False
    for line in lines:
        if _PW_ERROR_FENCE.match(line):
            in_error = not in_error
            continue
        if in_error:
            if line.strip() and len(errors) < _MAX_NOTES:
                errors.append(line.strip())
            continue
        if m := _PW_ITERATION.match(line):
            if not cycles or cycles[-1].converged is not None:
                cycles.append(_ScfCycle())
            cycles[-1].iterations = int(m.group(1))
            continue
        if cycles and cycles[-1].converged is None:
            cycle = cycles[-1]
            if m := _PW_FINAL_ENERGY.match(line):
                cycle.final_energy = float(m.group(1))
                continue
            if m := _PW_TRACE_ENERGY.match(line):
                cycle.trace.append(float(m.group(1)))
                continue
            if m := _PW_ACCURACY.search(line):
                cycle.accuracy = float(m.group(1))
                continue
            if m := _PW_CONVERGED.search(line):
                cycle.converged = True
                cycle.iterations = int(m.group(1))
                continue
            if m := _PW_NOT_CONVERGED.search(line):
                cycle.converged = False
                cycle.iterations = int(m.group(1))
                warnings.append(line.strip())
                continue
        if m := _PW_FORCE.match(line):
            forces_seen = True
            forces_max = max(forces_max, *(abs(float(c)) for c in m.groups()))
            continue
        if m := _PW_TOTAL_FORCE.search(line):
            total_force = (m.group(1), m.group(2))
            continue
        if m := _PW_PRESSURE.search(line):
            pressure = m.group(1)
            continue
        if m := _PW_FERMI.search(line):
            fermi = m.group(1)
            continue
        if m := _PW_BFGS.search(line):
            bfgs = (m.group(1), m.group(2))
            continue
        if m := _PW_WALL.match(line):
            wall = m.group(2)
            continue
        lowered = line.lower()
        if ("warning" in lowered or "error in routine" in lowered) and len(warnings) < _MAX_NOTES:
            if _PW_CARD_IGNORED.search(line):
                continue  # pw.x says this for every namelist ASE writes and it never uses
            warnings.append(line.strip())
    finished = "JOB DONE" in text
    out = [
        f"pw.x output digest: {name} ({len(lines)} lines, PWSCF v.{facts['version'] or '?'}, "
        f"{'finished: JOB DONE' if finished else 'NOT finished: no JOB DONE line'})"
    ]
    system = []
    if facts["atoms"]:
        system.append(f"{facts['atoms']} atoms")
    if facts["electrons"]:
        system.append(f"{facts['electrons']} electrons")
    if facts["ks_states"]:
        system.append(f"{facts['ks_states']} KS states")
    if facts["volume"]:
        system.append(f"volume {facts['volume']} bohr^3")
    settings = []
    if facts["ecutwfc"]:
        settings.append(f"ecutwfc {facts['ecutwfc']} Ry")
    if facts["ecutrho"]:
        settings.append(f"ecutrho {facts['ecutrho']} Ry")
    if facts["kpoints"]:
        settings.append(f"{facts['kpoints']} k-points")
    if facts["mixing_beta"]:
        settings.append(f"mixing beta {facts['mixing_beta']}")
    if facts["xc"]:
        settings.append(f"xc {facts['xc']}")
    if system:
        out.append("system: " + ", ".join(system))
    if settings:
        out.append("settings: " + "; ".join(settings))
    if not cycles:
        out.append("scf: no iteration found (the run may have stopped before the first step)")
    elif len(cycles) == 1:
        out.append("scf: 1 cycle (single point)")
        out.append("  " + _cycle_line(cycles[0]))
    else:
        done = sum(1 for c in cycles if c.converged)
        out.append(f"scf: {len(cycles)} cycles (a relaxation), {done} converged")
        out.append("  first " + _cycle_line(cycles[0]))
        out.append("  last " + _cycle_line(cycles[-1]))
        energies = [c.final_energy for c in cycles if c.final_energy is not None]
        if energies:
            out.append("  cycle energies (Ry): " + _elide([f"{e:.6f}" for e in energies]))
        if bfgs:
            out.append(f"  bfgs converged in {bfgs[0]} scf cycles and {bfgs[1]} bfgs steps")
    if fermi:
        out.append(f"fermi energy: {fermi} eV")
    if forces_seen:
        force = (
            f"forces: max |component| {forces_max:.6f} Ry/bohr "
            f"({forces_max * _RY_PER_BOHR_TO_EV_PER_A:.4f} eV/Å)"
        )
        if total_force:
            force += f"; Total force {total_force[0]}, Total SCF correction {total_force[1]}"
        out.append(force)
    else:
        out.append("forces: none printed")
    if pressure:
        out.append(f"stress: P = {pressure} kbar")
    out.append("warnings: " + ("; ".join(warnings) if warnings else "none"))
    if errors:
        out.append("errors: " + " | ".join(errors))
    if wall:
        out.append(f"wall: {wall} (PWSCF total)")
    return "\n".join(out)


def _cycle_line(cycle: _ScfCycle) -> str:
    if cycle.converged is True:
        status = f"converged in {cycle.iterations} iterations"
    elif cycle.converged is False:
        status = f"NOT converged after {cycle.iterations} iterations"
    else:
        status = f"{cycle.iterations} iterations, no convergence line (cut off?)"
    parts = [status]
    if cycle.trace:
        parts.append("trace (Ry): " + _elide([f"{e:.4f}" for e in cycle.trace]))
    if cycle.final_energy is not None:
        parts.append(f"final ! {cycle.final_energy:.8f} Ry")
    if cycle.accuracy is not None:
        parts.append(f"accuracy < {cycle.accuracy:.2e} Ry")
    return "; ".join(parts)


def _elide(items: list[str], ends: int = _TRACE_ENDS) -> str:
    if len(items) <= 2 * ends + 1:
        return ", ".join(items)
    head, tail = ", ".join(items[:ends]), ", ".join(items[-ends:])
    return f"{head} ... ({len(items) - 2 * ends} more) ... {tail}"


# -- LAMMPS log ----------------------------------------------------------------

_LMP_VERSION = re.compile(r"^LAMMPS \((.+)\)")
_LMP_UNITS = re.compile(r"^units\s+(\S+)")
_LMP_ATOMS = re.compile(r"^\s*(?:Created\s+)?(\d+)\s+atoms\s*$")
_LMP_PAIR = re.compile(r"^pair_style\s+(.+)")
_LMP_THERMO_HEAD = re.compile(r"^\s*Step\s+\S")
_LMP_RUN = re.compile(r"^\s*run\s+(\d+)(\s+upto)?(?:\s|$)")
_LMP_LOOP = re.compile(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) atoms")
_LMP_STOP = re.compile(r"^\s*Stopping criterion\s*=\s*(.+)")
_LMP_WALL = re.compile(r"^Total wall time:\s*(\S+)")
_LMP_INPUT_ECHO = re.compile(r"^(atom_style|thermo_style|pair_style|thermo)\s", re.MULTILINE)
#: ``thermo_modify line yaml`` prints each table as one YAML document between
#: these two lines. Both are exact: the ``----...`` rule of the timing
#: breakdown is longer than three dashes.
_LMP_YAML_OPEN = "---"
_LMP_YAML_CLOSE = "..."
_LMP_YAML_ROW = "  - ["
THERMO_FORMATS = ("yaml", "text", "mixed")
#: The commands an ASE-driven run echoes first, before any banner would come.
_LMP_OPENING = re.compile(
    r"^(log|clear|echo|units|atom_style|dimension|boundary|newton|package|"
    r"processors|variable)\s"
)


def _looks_like_lammps_log(head: str) -> bool:
    """A LAMMPS log opens with its banner, or (under ASE, which logs to
    stdout without one) with an echoed command, then ``units`` plus a
    style line. The header decides: a file whose first line is anything
    else, a script that mentions these commands in its text, is not a log.

    Examples:
        >>> _looks_like_lammps_log("LAMMPS (2 Aug 2023)\\nunits metal\\n")
        True
        >>> _looks_like_lammps_log("log /dev/stdout\\nclear\\nunits metal\\npair_style eam\\n")
        True
        >>> _looks_like_lammps_log("import ase\\n# units metal\\npair_style = 'eam'\\n")
        False
    """
    first = next((line for line in head.splitlines() if line.strip()), "")
    if first.startswith("LAMMPS ("):
        return True
    if not _LMP_OPENING.match(first):
        return False
    has_units = bool(re.search(r"^units\s+\w+", head, re.MULTILINE))
    return has_units and bool(_LMP_INPUT_ECHO.search(head))


def lammps_log_digest(name: str, text: str) -> str:
    """The digest of one LAMMPS log: setup, each thermo table's ends, warnings.

    A table printed as YAML (``thermo_modify line yaml``) digests to the
    same lines as a text table, and a ``thermo:`` line says which form
    the log took.

    Examples:
        >>> log = (
        ...     "LAMMPS (22 Jul 2025)\\nunits metal\\nCreated 4 atoms\\n---\\n"
        ...     "keywords: ['Step', 'Temp', ]\\ndata:\\n  - [0, 300, ]\\n  - [10, 1e+20, ]\\n"
        ...     "...\\nLoop time of 0.5 on 1 procs for 10 steps with 4 atoms\\n"
        ...     "Total wall time: 0:00:01\\n"
        ... )
        >>> print(lammps_log_digest("a.log", log))
        LAMMPS log digest: a.log (11 lines, LAMMPS 22 Jul 2025, finished: Total wall time 0:00:01)
        setup: units metal; 4 atoms
        thermo: yaml
        thermo table 1 (2 rows): Step Temp
          first: 0 300
          last:  10 1e+20
          loop: 10 steps, 4 atoms, 1 procs, 0.5 s
        warnings: none
    """
    lines = text.splitlines()
    version = units = pair = None
    atoms: str | None = None
    loops: list[str] = []
    stops: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    wall = None
    for line in lines:
        if m := _LMP_VERSION.match(line):
            version = m.group(1)
        elif m := _LMP_UNITS.match(line):
            units = m.group(1)
        elif m := _LMP_ATOMS.match(line):
            atoms = m.group(1)
        elif m := _LMP_PAIR.match(line):
            pair = m.group(1).strip()
        elif m := _LMP_LOOP.match(line):
            loops.append(
                f"{m.group(3)} steps, {m.group(4)} atoms, {m.group(2)} procs, {m.group(1)} s"
            )
        elif m := _LMP_STOP.match(line):
            stops.append(m.group(1).strip())
        elif m := _LMP_WALL.match(line):
            wall = m.group(1)
        elif line.startswith("WARNING") and len(warnings) < _MAX_NOTES:
            if line.strip() not in warnings:
                warnings.append(line.strip())
        elif line.startswith("ERROR") and len(errors) < _MAX_NOTES:
            errors.append(line.strip())
    scanned = _scan_thermo(text)
    if wall:
        status = f"finished: Total wall time {wall}"
    elif loops:
        status = (
            f"{len(loops)} loop(s) completed, no Total wall time line "
            f"(an ASE-driven log ends without one)"
        )
    else:
        status = "NOT finished: no loop completed"
    out = [f"LAMMPS log digest: {name} ({len(lines)} lines, LAMMPS {version or '?'}, {status})"]
    setup = []
    if units:
        setup.append(f"units {units}")
    if atoms:
        setup.append(f"{atoms} atoms")
    if pair:
        setup.append(f"pair_style {pair}")
    if setup:
        out.append("setup: " + "; ".join(setup))
    if not scanned:
        out.append("thermo: no table printed")
    else:
        out.append(f"thermo: {_thermo_format(scanned)}")
    for i, (_, table, texts) in enumerate(scanned, start=1):
        out.append(f"thermo table {i} ({len(texts)} rows): {' '.join(table['columns'])}")
        if texts:
            out.append(f"  first: {texts[0]}")
            if len(texts) > 1:
                out.append(f"  last:  {texts[-1]}")
        if i <= len(loops):
            out.append(f"  loop: {loops[i - 1]}")
    for stop in stops:
        out.append(f"minimization stopped: {stop}")
    out.append("warnings: " + ("; ".join(warnings) if warnings else "none"))
    if errors:
        out.append("errors: " + " | ".join(errors))
    return "\n".join(out)


def _row_text(row: list[int | float]) -> str:
    """One YAML thermo row as LAMMPS's default text format prints it (``%.8g``)."""
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8g}" for v in row)


def lammps_thermo(text: str) -> list[dict[str, Any]]:
    """Every thermo table of a LAMMPS log: columns, rows, and its loop line.

    A table is either a YAML document, which ``thermo_modify line yaml``
    prints between a ``---`` line and a ``...`` line, or a text table
    that starts at a ``Step ...`` header and holds every numeric row that
    follows; ``WARNING`` lines inside a table are skipped, any other line
    ends a text table. The two forms come back in order of appearance,
    so a log that switches to YAML between two runs parses fully, and a
    YAML document is never read as a text table. The ``Loop time`` line
    that follows a table is attached to it as ``loop`` (seconds, procs,
    steps, atoms). ``minimize`` is True for a table a ``minimize``
    command printed, which LAMMPS marks with a ``Minimization stats:``
    block after the loop line; its rows and its loop steps count
    minimizer iterations, not time steps. In a YAML table the step is an int and every other
    value a float; in a text table integers stay integers (the step) and
    everything else is a float.

    Examples:
        >>> log = (
        ...     "Step Temp PotEng\\n0 300 -3.5\\n100 298.2 -3.49\\n"
        ...     "Loop time of 0.5 on 1 procs for 100 steps with 32 atoms\\n"
        ... )
        >>> tables = lammps_thermo(log)
        >>> tables[0]["columns"], tables[0]["rows"][-1]
        (['Step', 'Temp', 'PotEng'], [100, 298.2, -3.49])
        >>> tables[0]["loop"], tables[0]["minimize"]
        ({'seconds': 0.5, 'procs': 1, 'steps': 100, 'atoms': 32}, False)
        >>> lammps_thermo("no table here\\n")
        []
        >>> mixed = log + (
        ...     "---\\nkeywords: ['Step', 'Temp', ]\\ndata:\\n  - [100, 300, ]\\n"
        ...     "  - [200, 1e+20, ]\\n...\\n"
        ...     "Loop time of 0.4 on 1 procs for 100 steps with 32 atoms\\n"
        ... )
        >>> [t["rows"] for t in lammps_thermo(mixed)]
        [[[0, 300, -3.5], [100, 298.2, -3.49]], [[100, 300.0], [200, 1e+20]]]
        >>> lammps_thermo(mixed)[1]["loop"]["seconds"]
        0.4
    """
    return [table for _, table, _ in _scan_thermo(text)]


def lammps_thermo_format(text: str) -> str | None:
    """Which form the thermo tables of a log took: ``yaml``, ``text``,
    ``mixed`` (both forms in one log), or None when no table was printed.

    Examples:
        >>> lammps_thermo_format("Step Temp\\n0 300\\n")
        'text'
        >>> lammps_thermo_format("---\\nkeywords: ['Step', ]\\ndata:\\n  - [0, ]\\n...\\n")
        'yaml'
        >>> lammps_thermo_format("Step Temp\\n0 300\\n---\\nkeywords: ['Step', ]\\ndata:\\n...\\n")
        'mixed'
        >>> lammps_thermo_format("Loop time of 0.5 on 1 procs for 0 steps with 1 atoms\\n") is None
        True
    """
    scanned = _scan_thermo(text)
    return _thermo_format(scanned) if scanned else None


def lammps_run_progress(text: str) -> dict[str, int | None] | None:
    """How far the last thermo table of a log has got: its step and the run's end.

    Reads a log that may still be written, so a table without its
    ``Loop time`` line counts. A last line without its newline is a row
    LAMMPS is still writing, and it is left out. ``step`` is the last
    row's step and ``start`` the first row's. ``target`` is the step the
    ``run`` command in force when the table opened will stop at: the
    start plus N for ``run N``, and N for ``run N upto``. LAMMPS echoes a command
    with variables to the log twice, once as written and once
    substituted, so the substituted count is the one read. A table that
    a ``minimize`` opened, or one no ``run`` preceded, has no target.
    None when no table with a ``Step`` column has a row yet.

    Examples:
        >>> log = (
        ...     "run ${n}\\nrun 200\\n---\\nkeywords: ['Step', 'Temp', ]\\ndata:\\n"
        ...     "  - [0, 1.44, ]\\n  - [100, 0.79, ]\\n...\\n"
        ...     "Loop time of 0.01 on 1 procs for 200 steps with 256 atoms\\n"
        ...     "run 300 upto\\nStep Temp\\n200 0.76\\n250 0.74\\n"
        ... )
        >>> lammps_run_progress(log)
        {'step': 250, 'start': 200, 'target': 300}
        >>> lammps_run_progress("minimize 1e-6 1e-8 100 1000\\nStep PotEng\\n0 -3.5\\n5 -3.6\\n")
        {'step': 5, 'start': 0, 'target': None}
        >>> lammps_run_progress(log + "300 0.7")["step"]
        250
        >>> lammps_run_progress("run 1000\\n") is None
        True
    """
    if not text.endswith("\n"):
        text = text[: text.rfind("\n") + 1]
    tables = lammps_thermo(text)
    if not tables or "Step" not in tables[-1]["columns"] or not tables[-1]["rows"]:
        return None
    column = tables[-1]["columns"].index("Step")
    rows = [row for row in tables[-1]["rows"] if len(row) > column]
    if not rows:
        return None
    start, step = int(rows[0][column]), int(rows[-1][column])
    # The run in force at each table's opening; only the last one is read.
    pending: tuple[int, bool] | None = None
    opened: tuple[int, bool] | None = None
    for line in text.splitlines():
        if match := _LMP_RUN.match(line):
            pending = (int(match.group(1)), match.group(2) is not None)
        elif line.startswith("minimize"):
            pending = None
        elif line == _LMP_YAML_OPEN or _LMP_THERMO_HEAD.match(line):
            opened, pending = pending, None
    target: int | None = None
    if opened is not None:
        count, upto = opened
        target = count if upto else start + count
    return {"step": step, "start": start, "target": target}


def _thermo_format(scanned: list[_ThermoScan]) -> str:
    forms = {form for form, _, _ in scanned}
    return "mixed" if len(forms) > 1 else forms.pop()


def lammps_yaml_thermo(text: str) -> list[dict[str, Any]]:
    """The thermo tables a log printed as YAML documents, in order.

    Only the lines that belong to a document are read: the ``keywords:``
    line, the ``data:`` line, and the ``  - [...]`` rows between a
    ``---`` line and a ``...`` line. A ``fix print`` line or a WARNING
    inside a document is skipped. A document that never closes, because
    LAMMPS died inside the run, still yields the rows it printed. Values
    PyYAML leaves as text (``1e+20``, ``inf``, ``-nan``) become floats.

    Examples:
        >>> doc = (
        ...     "---\\nkeywords: ['Step', 'Temp', 'Press', ]\\ndata:\\n"
        ...     "  - [0, 300, 1e+20, ]\\nWARNING: x (src/a.cpp:1)\\n"
        ...     "  - [50, 298.5, -1.5, ]\\n...\\n"
        ...     "Loop time of 0.5 on 1 procs for 50 steps with 4 atoms\\n"
        ... )
        >>> table = lammps_yaml_thermo(doc)[0]
        >>> table["columns"], table["rows"]
        (['Step', 'Temp', 'Press'], [[0, 300.0, 1e+20], [50, 298.5, -1.5]])
        >>> table["loop"]["steps"]
        50
        >>> lammps_yaml_thermo("---\\nno keywords here\\n...\\n")
        []
    """
    return [table for form, table, _ in _scan_thermo(text) if form == "yaml"]


#: One scanned table: its form (``yaml`` or ``text``), the table, and each
#: row as text for the digest (the log's own line for a text table).
_ThermoScan = tuple[str, dict[str, Any], list[str]]


def _scan_thermo(text: str) -> list[_ThermoScan]:
    """Every thermo table with its form (``yaml`` or ``text``), in log order."""
    tables: list[_ThermoScan] = []
    current: dict[str, Any] | None = None  # a text table being read
    texts: list[str] = []
    document: list[str] | None = None  # the lines of a YAML document being read

    def close_document() -> None:
        nonlocal document
        if document is not None:
            table = _yaml_table(document)
            if table is not None:
                tables.append(("yaml", table, [_row_text(row) for row in table["rows"]]))
            document = None

    for line in text.splitlines():
        stripped = line.strip()
        if document is not None:
            if line == _LMP_YAML_CLOSE:
                close_document()
            elif line.startswith(("keywords:", "data:", _LMP_YAML_ROW)):
                document.append(line)
            continue
        if current is not None:
            if stripped and _is_numeric_row(stripped):
                current["rows"].append([_number(token) for token in stripped.split()])
                texts.append(" ".join(stripped.split()))
                continue
            if stripped.startswith("WARNING"):
                continue
            tables.append(("text", current, texts))
            current, texts = None, []
        if line == _LMP_YAML_OPEN:
            document = []
        elif _LMP_THERMO_HEAD.match(line):
            current = {"columns": stripped.split(), "rows": [], "loop": None, "minimize": False}
        elif (m := _LMP_LOOP.match(line)) and tables and tables[-1][1]["loop"] is None:
            tables[-1][1]["loop"] = {
                "seconds": float(m.group(1)),
                "procs": int(m.group(2)),
                "steps": int(m.group(3)),
                "atoms": int(m.group(4)),
            }
        elif stripped == "Minimization stats:" and tables and tables[-1][1]["loop"] is not None:
            tables[-1][1]["minimize"] = True
    close_document()
    if current is not None:
        tables.append(("text", current, texts))
    return tables


def _yaml_table(lines: list[str]) -> dict[str, Any] | None:
    """One thermo table from the lines of a YAML document, or None when the
    lines hold no ``keywords`` list (a ``---`` line that opened no table)."""
    try:
        loaded = yaml.safe_load("\n".join(lines))
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict) or not isinstance(loaded.get("keywords"), list):
        return None
    columns = [str(key) for key in loaded["keywords"]]
    rows: list[list[int | float]] = []
    for row in loaded.get("data") or []:
        if not isinstance(row, list) or len(row) != len(columns):
            continue
        rows.append(
            [int(v) if name == "Step" else float(v) for name, v in zip(columns, row, strict=True)]
        )
    return {"columns": columns, "rows": rows, "loop": None, "minimize": False}


def _number(token: str) -> int | float:
    try:
        return int(token)
    except ValueError:
        return float(token)


def _is_numeric_row(stripped: str) -> bool:
    try:
        for token in stripped.split():
            float(token)
    except ValueError:
        return False
    return True


# -- fix ave/time --------------------------------------------------------------

_AVE_TIME_HEAD = "# Time-averaged data for fix "


def _looks_like_ave_time(head: str) -> bool:
    """A ``fix ave/time`` file opens with its own first line.

    Examples:
        >>> _looks_like_ave_time("# Time-averaged data for fix avg\\n# TimeStep c_t\\n")
        True
        >>> _looks_like_ave_time("# TimeStep c_t\\n100 1.0\\n")
        False
    """
    first = next((line for line in head.splitlines() if line.strip()), "")
    return first.startswith(_AVE_TIME_HEAD)


def lammps_ave_time(text: str) -> dict[str, Any]:
    """Parse one ``fix ave/time`` output file.

    The file opens with ``# Time-averaged data for fix <id>`` and a column
    line ``# TimeStep col1 col2 ...``. In scalar mode every later line is
    one numeric row, returned as ``rows`` with integers kept as integers.
    In vector mode (``mode vector``) a third header line ``# Row col1
    ...`` names the per-row columns, and each block is a ``TimeStep
    Number-of-rows`` line followed by that many rows. The vector layout
    is recognised and reported, with ``columns`` from the ``# Row`` line
    and ``rows`` left empty, because the blocks belong to the artifact,
    not to a summary. A file that is not a ``fix ave/time`` file raises
    ``ValueError``.

    Examples:
        >>> scalar = (
        ...     "# Time-averaged data for fix avg\\n"
        ...     "# TimeStep c_thermo_temp c_thermo_press\\n"
        ...     "100 168.698 4991.83\\n200 178.055 6233.87\\n"
        ... )
        >>> parsed = lammps_ave_time(scalar)
        >>> parsed["fix"], parsed["mode"], parsed["columns"]
        ('avg', 'scalar', ['TimeStep', 'c_thermo_temp', 'c_thermo_press'])
        >>> parsed["rows"]
        [[100, 168.698, 4991.83], [200, 178.055, 6233.87]]
        >>> vector = (
        ...     "# Time-averaged data for fix vec\\n"
        ...     "# TimeStep Number-of-rows\\n"
        ...     "# Row c_rdf[1] c_rdf[2]\\n"
        ...     "100 2\\n1 0.425 0\\n2 1.275 0.5\\n"
        ... )
        >>> lammps_ave_time(vector)
        {'fix': 'vec', 'mode': 'vector', 'columns': ['Row', 'c_rdf[1]', 'c_rdf[2]'], 'rows': []}
        >>> lammps_ave_time("# TimeStep c_t\\n100 1.0\\n")
        Traceback (most recent call last):
        ...
        ValueError: not a fix ave/time file: the first line is '# TimeStep c_t'
    """
    lines = [line for line in text.splitlines() if line.strip()]
    first = lines[0].strip() if lines else ""
    if not first.startswith(_AVE_TIME_HEAD):
        raise ValueError(f"not a fix ave/time file: the first line is {first!r}")
    fix = first[len(_AVE_TIME_HEAD) :].strip()
    headers = [line.strip()[1:].split() for line in lines[1:3] if line.strip().startswith("#")]
    if not headers:
        raise ValueError(f"fix ave/time file for fix {fix!r} has no column line")
    if len(headers) > 1 and headers[1][:1] == ["Row"]:
        return {"fix": fix, "mode": "vector", "columns": headers[1], "rows": []}
    columns = headers[0]
    rows = [
        [_number(token) for token in line.split()]
        for line in lines[2:]
        if not line.strip().startswith("#") and _is_numeric_row(line.strip())
    ]
    return {"fix": fix, "mode": "scalar", "columns": columns, "rows": rows}


def lammps_ave_time_digest(name: str, text: str) -> str:
    """The digest of one ``fix ave/time`` file: the fix, the columns, the ends.

    Examples:
        >>> text = (
        ...     "# Time-averaged data for fix avg\\n# TimeStep c_thermo_temp\\n"
        ...     "100 168.698\\n200 178.055\\n300 225.661\\n"
        ... )
        >>> print(lammps_ave_time_digest("ar-avg.dat", text))
        fix ave/time digest: ar-avg.dat (fix avg, scalar mode, 3 rows)
        columns: TimeStep c_thermo_temp
        first: 100 168.698
        last:  300 225.661
    """
    try:
        parsed = lammps_ave_time(text)
    except ValueError as e:
        return f"fix ave/time digest: {name} (unreadable: {e})"
    out = []
    if parsed["mode"] == "vector":
        blocks = sum(
            1
            for line in text.splitlines()
            if _is_numeric_row(line.strip()) and len(line.split()) == 2
        )
        out.append(
            f"fix ave/time digest: {name} (fix {parsed['fix']}, vector mode, "
            f"{blocks} block(s); the blocks are in the file itself)"
        )
        out.append("columns: " + " ".join(parsed["columns"]))
        return "\n".join(out)
    rows = parsed["rows"]
    out.append(
        f"fix ave/time digest: {name} (fix {parsed['fix']}, scalar mode, {len(rows)} rows)"
    )
    out.append("columns: " + " ".join(parsed["columns"]))
    if rows:
        out.append("first: " + " ".join(str(v) for v in rows[0]))
        if len(rows) > 1:
            out.append("last:  " + " ".join(str(v) for v in rows[-1]))
    return "\n".join(out)


# -- extended XYZ --------------------------------------------------------------

_XYZ_ENERGY = re.compile(r"(?:^|\s)energy=(-?[\d.Ee+-]+)")


def _looks_like_extxyz(head: str, name: str) -> bool:
    first, _, rest = head.partition("\n")
    second = rest.partition("\n")[0]
    if not first.strip().isdigit():
        return False
    if "Lattice=" in second or "Properties=" in second:
        return True
    return name.endswith((".xyz", ".extxyz"))


def extxyz_digest(name: str, text: str) -> str:
    """The digest of one extended-XYZ file: frames, sizes, species, labels."""
    lines = text.splitlines()
    frames = 0
    sizes: list[int] = []
    species: set[str] = set()
    energies: list[float] = []
    forces = lattice = False
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head.isdigit():
            i += 1
            continue
        n = int(head)
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        frames += 1
        sizes.append(n)
        if m := _XYZ_ENERGY.search(comment):
            with contextlib.suppress(ValueError):
                energies.append(float(m.group(1)))
        if "forces:R:3" in comment or "force:R:3" in comment:
            forces = True
        if "Lattice=" in comment:
            lattice = True
        for line in lines[i + 2 : i + 2 + n]:
            token = line.split(maxsplit=1)
            if token:
                species.add(token[0])
        i += 2 + n
    plural = "s" if frames != 1 else ""
    out = [f"extended XYZ digest: {name} ({frames} frame{plural}, {len(lines)} lines)"]
    if sizes:
        size = f"{min(sizes)}" if min(sizes) == max(sizes) else f"{min(sizes)} to {max(sizes)}"
        out.append(f"atoms per frame: {size}; species: {', '.join(sorted(species)) or '?'}")
    if energies:
        out.append(
            f"energy: present on {len(energies)} frame(s), {min(energies):g} to {max(energies):g}"
        )
    else:
        out.append("energy: absent from the comment lines")
    out.append(
        f"forces: {'present' if forces else 'absent'}; "
        f"lattice: {'present' if lattice else 'absent'}"
    )
    if frames:
        out.append(_spacing_line(text))
    return "\n".join(out)


#: Pair distances are O(N^2) per frame; past this many pair evaluations the
#: spacing check samples frames evenly instead of reading every one.
_SPACING_PAIR_BUDGET = 50_000_000
#: A pair closer than this fraction of the sum of covalent radii is flagged.
_SPACING_SUSPECT_FRACTION = 0.6


def _spacing_line(text: str) -> str:
    """The closest pair of atoms in the file, with periodic images, and a flag.

    A dataset with two atoms 0.3 Å apart trains a potential on a
    configuration no engine could have labelled honestly; the number a
    reader wants is the smallest distance anywhere in the file, which
    frame it sits in, and whether it is short against the covalent radii.
    Best-effort: a file ASE cannot parse yields a line saying so.
    """
    import io

    import numpy as np
    from ase.data import atomic_numbers, covalent_radii
    from ase.io import read

    try:
        frames = read(io.StringIO(text), index=":", format="extxyz")
    except Exception as e:
        return f"spacing: not computed (ASE could not parse the file: {e})"
    if not isinstance(frames, list):
        frames = [frames]
    pairs = sum(len(a) * (len(a) - 1) // 2 for a in frames)
    step = max(1, -(-pairs // _SPACING_PAIR_BUDGET))
    sampled = frames[::step]
    best: tuple[float, int, str, str] | None = None
    nearest: list[float] = []
    for index, atoms in enumerate(frames):
        if index % step or len(atoms) < 2:
            continue
        distances = np.asarray(
            atoms.get_all_distances(mic=bool(atoms.pbc.any()))  # type: ignore[no-untyped-call]
        )
        np.fill_diagonal(distances, np.inf)
        nearest.append(float(distances.min(axis=1).mean()))
        i, j = np.unravel_index(int(distances.argmin()), distances.shape)
        d = float(distances[i, j])
        if best is None or d < best[0]:
            best = (d, index, atoms[i].symbol, atoms[j].symbol)
    if best is None:
        return "spacing: n/a (no frame holds two atoms)"
    d, index, a, b = best
    radii = covalent_radii[atomic_numbers[a]] + covalent_radii[atomic_numbers[b]]
    verdict = (
        f"SUSPECT: under {_SPACING_SUSPECT_FRACTION:.0%} of the covalent-radii sum {radii:.2f} Å"
        if d < _SPACING_SUSPECT_FRACTION * radii
        else f"plausible against the covalent-radii sum {radii:.2f} Å"
    )
    if step == 1:
        scope = f"all {len(frames)} frames"
    else:
        scope = f"{len(sampled)} of {len(frames)} frames sampled"
    return (
        f"spacing: closest pair {d:.3f} Å ({a}-{b}, frame {index}), {verdict}; "
        f"mean nearest-neighbour distance {sum(nearest) / len(nearest):.3f} Å over {scope}"
    )
