"""Named input protocols for Quantum ESPRESSO: AiiDA's balanced protocol, adopted.

aiida-quantumespresso curates three named parameter sets — ``fast``,
``balanced``, ``stringent`` — that turn "run a DFT calculation" into a
reproducible recipe: plane-wave cutoffs from the pseudopotential family's
per-element recommendations, a k-point mesh from a reciprocal-space spacing,
cold smearing for metals, convergence thresholds scaled per atom. SLAB adopts
the protocols (values from aiida-quantumespresso v4.10, MIT-licensed) with
one philosophical adjustment: **a protocol is only ever applied by name,
explicitly**. Nothing in the engine layer defaults to one; you call
:func:`qe_protocol_options` yourself and pass the result on (the call itself
is the explicit act — its ``protocol`` argument defaults to ``"balanced"``,
AiiDA's default, when you don't specify).

The protocol numbers live in a versioned data file
(``slab/data/qe_protocols.json``), not code — policy-as-data, like retention
and the engine registry. And because the helper returns fully *expanded*
options (concrete cutoffs, a concrete mesh, concrete thresholds) that flow
into the task as traced inputs, the physics values themselves are the cache
identity. A protocol data-file edit can never silently re-serve stale cached
results, because the cache never saw the protocol's name — only its numbers.

Usage::

    from slab.protocols import qe_protocol_options
    from foundation.tasks import relax

    options = qe_protocol_options(atoms, protocol="balanced")
    relaxed, info = relax(atoms, engine="qe", calculator_options=options)

Requires the protocol's pseudopotential family installed locally
(``slab pseudos install sssp ...`` — see :mod:`slab.pseudos`); pass
``family=`` to use a different installed family.

Scope, honestly stated: this covers the ``pw`` base protocol (SCF/relax
inputs). AiiDA's spin/magnetization handling, 2D ``assume_isolated``
treatment, and the PwRelaxWorkChain meta-convergence loop are not adopted
yet. AiiDA's pre-v4.10 names are not aliased: asking for ``moderate`` or
``precise`` is an error that points at ``balanced``/``stringent`` — the
rename came with retuned values, and serving different numbers under an old
name would be a silent guess.
"""

from __future__ import annotations

import json
import math
from functools import cache
from importlib import resources
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from slab.errors import SlabError
from slab.pseudos import PseudoFamily, family_cutoffs, family_digest, family_pseudos, find_family

_RENAMED = {"moderate": "balanced", "precise": "stringent"}
_RY_TO_EV = 13.605693122994
_BOHR_TO_ANGSTROM = 0.529177210903


class ProtocolError(SlabError):
    """A protocol name is unknown, or a structure cannot be expanded under it."""


class QeProtocol(BaseModel):
    """One named protocol's data, as the data file declares it.

    Units follow Quantum ESPRESSO's input conventions verbatim (Ry, Ry/atom,
    Ry/bohr); ``kpoints_distance`` is a reciprocal-space spacing in 1/Å
    including the 2π factor.
    """

    model_config = ConfigDict(frozen=True)

    description: str
    kpoints_distance: float = Field(gt=0)
    degauss: float = Field(gt=0)
    smearing: str = "cold"
    conv_thr_per_atom: float = Field(gt=0)
    etot_conv_thr_per_atom: float = Field(gt=0)
    forc_conv_thr: float = Field(gt=0)
    mixing_beta: float = Field(gt=0)
    electron_maxstep: int = Field(gt=0)
    pseudo_family: str


@cache
def _load_protocols() -> dict[str, QeProtocol]:
    raw = json.loads(
        (resources.files("slab") / "data" / "qe_protocols.json").read_text(encoding="utf-8")
    )
    return {
        name: QeProtocol.model_validate(data) for name, data in sorted(raw["protocols"].items())
    }


def available_protocols() -> tuple[str, ...]:
    """The protocol names the data file declares.

    Examples:
        >>> available_protocols()
        ('balanced', 'fast', 'stringent')
    """
    return tuple(_load_protocols())


def get_protocol(name: str) -> QeProtocol:
    """The named protocol's data, refusing unknown and renamed names loudly.

    Examples:
        >>> get_protocol("balanced").kpoints_distance
        0.15
    """
    protocols = _load_protocols()
    normalized = name.strip().lower()
    if normalized in protocols:
        return protocols[normalized]
    if normalized in _RENAMED:
        raise ProtocolError(
            f"protocol {name!r} was renamed {_RENAMED[normalized]!r} (and retuned) in "
            f"aiida-quantumespresso v4.10; ask for {_RENAMED[normalized]!r} explicitly"
        )
    raise ProtocolError(f"unknown protocol {name!r}; available: {', '.join(protocols)}")


def protocol_details(name: str) -> dict[str, Any]:
    """A protocol's data plus derived values, for humans and agents.

    Adds ``forc_conv_thr_ev_per_ang`` — the eV/Å equivalent of the Ry/bohr
    force threshold, the natural ``fmax`` scale when driving relaxation with
    ASE optimizers instead of pw.x's internal loop.

    Examples:
        >>> details = protocol_details("balanced")
        >>> round(details["forc_conv_thr_ev_per_ang"], 5)
        0.00257
    """
    protocol = get_protocol(name)
    details = protocol.model_dump(mode="json")
    details["name"] = name.strip().lower()
    details["forc_conv_thr_ev_per_ang"] = (
        protocol.forc_conv_thr * _RY_TO_EV / _BOHR_TO_ANGSTROM
    )
    return details


def qe_protocol_options(
    atoms: Any,
    protocol: str = "balanced",
    *,
    family: str | None = None,
    electronic_type: str = "metal",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expand a named protocol into concrete ``calculator_options`` for ``qe``.

    The expansion is AiiDA's arithmetic exactly:

    * ``ecutwfc``/``ecutrho`` — element-wise maxima of the family's
      per-element recommended cutoffs over the elements present (Ry).
    * ``kpts`` — a Monkhorst-Pack mesh from ``kpoints_distance``:
      ``n_i = max(1, ceil(|b_i| / distance))`` with the 2π-included
      reciprocal vectors.
    * ``etot_conv_thr = natoms * etot_conv_thr_per_atom`` and
      ``conv_thr = natoms * conv_thr_per_atom``; ``forc_conv_thr`` is not
      scaled (it is per force component).
    * metals get cold smearing with the protocol's ``degauss``; insulators
      (``electronic_type="insulator"``) get fixed occupations and no
      smearing.

    The returned dict contains only concrete values — those values become
    traced task inputs, so the cache identity is the physics, not the
    protocol's name. Record the name in the run's ``intent`` for narrative
    provenance.

    Args:
        atoms: The structure (``ase.Atoms``); must be fully periodic with a
            nonzero cell — protocol k-point generation needs one.
        protocol: ``"fast"``, ``"balanced"``, or ``"stringent"``.
        family: Installed pseudopotential family to use instead of the
            protocol's recommendation (see :mod:`slab.pseudos`).
        electronic_type: ``"metal"`` (smearing) or ``"insulator"`` (fixed
            occupations) — AiiDA's ``ElectronicType``, minus the guessing.
        overrides: Recursive right-wins merge on top of the expanded options
            (AiiDA's override semantics): e.g.
            ``{"input_data": {"system": {"ecutwfc": 90.0}}}``.

    Returns:
        A ``calculator_options`` dict: ``pseudo_family`` (resolved name),
        ``pseudopotentials`` (exactly the elements present), ``kpts``, and
        ``input_data`` (control/system/electrons). Every value is
        JSON-faithful (``kpts`` is a list, not a tuple), so the traced cache
        identity is canonical sorted-key JSON — two logically identical
        expansions can never hash differently.

    Raises:
        ProtocolError: Unknown protocol/electronic_type, or a structure the
            protocol cannot expand (no periodic cell).
        PseudoFamilyError: The family is not installed or lacks an element.
    """
    data = get_protocol(protocol)
    resolved_family, _directory = find_family(family or data.pseudo_family)

    symbols = list(atoms.get_chemical_symbols())
    natoms = len(symbols)
    if natoms == 0:
        raise ProtocolError("cannot expand a protocol for an empty structure")
    ecutwfc, ecutrho = family_cutoffs(resolved_family, symbols)

    options: dict[str, Any] = {
        "pseudo_family": resolved_family.name,
        "pseudopotentials": family_pseudos(resolved_family, symbols),
        "kpts": _kpoints_mesh(atoms, data.kpoints_distance),
        "input_data": {
            "control": {
                "etot_conv_thr": natoms * data.etot_conv_thr_per_atom,
                "forc_conv_thr": data.forc_conv_thr,
                "tprnfor": True,
                "tstress": True,
            },
            "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho},
            "electrons": {
                "conv_thr": natoms * data.conv_thr_per_atom,
                "mixing_beta": data.mixing_beta,
                "electron_maxstep": data.electron_maxstep,
            },
        },
    }
    system = options["input_data"]["system"]
    if electronic_type == "metal":
        system["occupations"] = "smearing"
        system["smearing"] = data.smearing
        system["degauss"] = data.degauss
    elif electronic_type == "insulator":
        system["occupations"] = "fixed"
    else:
        raise ProtocolError(
            f"unknown electronic_type {electronic_type!r}; use 'metal' or 'insulator'"
        )
    if overrides:
        options = _recursive_merge(options, overrides)
    return options


def _kpoints_mesh(atoms: Any, distance: float) -> list[int]:
    """Monkhorst-Pack mesh from a maximum reciprocal-space spacing (1/Å).

    ``n_i = max(1, ceil(round(|b_i| / distance, 5)))`` where ``b_i`` are the
    reciprocal cell vectors *including* the 2π factor (ASE's
    ``cell.reciprocal()`` excludes it) — AiiDA's convention, including its
    5-decimal rounding guard so float noise at an exact-integer ratio cannot
    bump the mesh (10.000000000000002 must mean 10, not 11).

    Returned as a list, deliberately: the expanded options must stay
    JSON-faithful so their traced identity is canonical sorted-key JSON,
    independent of dict construction order.
    """
    pbc = getattr(atoms, "pbc", None)
    cell = atoms.cell
    if pbc is None or not all(bool(p) for p in pbc) or float(cell.volume) <= 0.0:
        raise ProtocolError(
            "protocol k-point generation needs a fully periodic structure with a "
            "nonzero cell; got pbc="
            f"{None if pbc is None else tuple(bool(p) for p in pbc)}"
        )
    reciprocal = cell.reciprocal()
    mesh = []
    for row in reciprocal:
        length = 2.0 * math.pi * math.sqrt(sum(float(c) ** 2 for c in row))
        mesh.append(max(1, math.ceil(round(length / distance, 5))))
    return mesh


def _recursive_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Right-wins merge at every nesting level (AiiDA's override semantics)."""
    merged = dict(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _recursive_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


__all__ = [
    "ProtocolError",
    "PseudoFamily",
    "QeProtocol",
    "available_protocols",
    "family_digest",
    "get_protocol",
    "protocol_details",
    "qe_protocol_options",
]
