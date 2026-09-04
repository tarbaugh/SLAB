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
``Program PWSCF``), a LAMMPS log (the first lines say ``LAMMPS (``), and
extended XYZ (a frame count line, then a comment line with ``Lattice=`` or
``Properties=``). :func:`digest` returns ``None`` for anything else, and
the caller shows the text as it always did.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field

__all__ = ["digest", "extxyz_digest", "lammps_log_digest", "pwscf_digest"]

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
    """
    head = text[:_SNIFF_CHARS]
    if "Program PWSCF" in head:
        return pwscf_digest(name, text)
    if _looks_like_lammps_log(head):
        return lammps_log_digest(name, text)
    if _looks_like_extxyz(head, name):
        return extxyz_digest(name, text)
    return None


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
_LMP_LOOP = re.compile(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) atoms")
_LMP_STOP = re.compile(r"^\s*Stopping criterion\s*=\s*(.+)")
_LMP_WALL = re.compile(r"^Total wall time:\s*(\S+)")
_LMP_INPUT_ECHO = re.compile(r"^(atom_style|thermo_style|pair_style|thermo)\s", re.MULTILINE)


def _looks_like_lammps_log(head: str) -> bool:
    """A LAMMPS log opens with its banner, or (under ASE, which logs to
    stdout without one) with the echoed input: ``units`` plus a style line."""
    if "LAMMPS (" in head:
        return True
    has_units = bool(re.search(r"^units\s+\w+", head, re.MULTILINE))
    return has_units and bool(_LMP_INPUT_ECHO.search(head))


def lammps_log_digest(name: str, text: str) -> str:
    """The digest of one LAMMPS log: setup, each thermo table's ends, warnings."""
    lines = text.splitlines()
    version = units = pair = None
    atoms: str | None = None
    tables: list[tuple[str, list[str], int]] = []  # header, [first row, last row], rows
    loops: list[str] = []
    stops: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    wall = None
    header: str | None = None
    rows: list[str] = []
    for line in lines:
        if header is not None:
            stripped = line.strip()
            if stripped and _is_numeric_row(stripped):
                rows.append(" ".join(stripped.split()))
                continue
            tables.append((header, [rows[0], rows[-1]] if rows else [], len(rows)))
            header, rows = None, []
        if m := _LMP_VERSION.match(line):
            version = m.group(1)
        elif m := _LMP_UNITS.match(line):
            units = m.group(1)
        elif m := _LMP_ATOMS.match(line):
            atoms = m.group(1)
        elif m := _LMP_PAIR.match(line):
            pair = m.group(1).strip()
        elif _LMP_THERMO_HEAD.match(line):
            header = " ".join(line.split())
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
    if header is not None:
        tables.append((header, [rows[0], rows[-1]] if rows else [], len(rows)))
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
    if not tables:
        out.append("thermo: no table printed")
    for i, (head, ends, count) in enumerate(tables, start=1):
        out.append(f"thermo table {i} ({count} rows): {head}")
        if ends:
            out.append(f"  first: {ends[0]}")
            if count > 1:
                out.append(f"  last:  {ends[1]}")
        if i <= len(loops):
            out.append(f"  loop: {loops[i - 1]}")
    for stop in stops:
        out.append(f"minimization stopped: {stop}")
    out.append("warnings: " + ("; ".join(warnings) if warnings else "none"))
    if errors:
        out.append("errors: " + " | ".join(errors))
    return "\n".join(out)


def _is_numeric_row(stripped: str) -> bool:
    try:
        for token in stripped.split():
            float(token)
    except ValueError:
        return False
    return True


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
    return "\n".join(out)
