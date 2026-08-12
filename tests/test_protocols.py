"""Protocol expansion tests — golden numbers pinned to AiiDA's arithmetic."""

import math
from pathlib import Path

import pytest
from ase.build import bulk

from conftest import make_family
from slab.protocols import (
    ProtocolError,
    available_protocols,
    get_protocol,
    protocol_details,
    qe_protocol_options,
)


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    return tmp_path / "pseudos"


@pytest.fixture()
def sssp(root: Path):
    family, _ = make_family(root, symbols=("Si", "O", "Cu"))
    return family


# -- the data file ----------------------------------------------------------------------


def test_protocol_names(sssp) -> None:
    assert available_protocols() == ("balanced", "fast", "stringent")


def test_balanced_matches_aiida_v4_10() -> None:
    balanced = get_protocol("balanced")
    assert balanced.kpoints_distance == 0.15
    assert balanced.degauss == 0.02
    assert balanced.smearing == "cold"
    assert balanced.conv_thr_per_atom == 2e-10
    assert balanced.etot_conv_thr_per_atom == 1e-5
    assert balanced.forc_conv_thr == 1e-4
    assert balanced.mixing_beta == 0.4
    assert balanced.electron_maxstep == 80
    assert balanced.pseudo_family == "SSSP/1.3/PBEsol/efficiency"


def test_fast_and_stringent_deltas() -> None:
    fast, stringent = get_protocol("fast"), get_protocol("stringent")
    assert (fast.kpoints_distance, stringent.kpoints_distance) == (0.3, 0.1)
    assert (fast.degauss, stringent.degauss) == (0.0275, 0.0125)
    assert (fast.forc_conv_thr, stringent.forc_conv_thr) == (1e-3, 5e-5)
    assert stringent.pseudo_family == "SSSP/1.3/PBEsol/precision"


def test_renamed_aiida_protocols_refused_with_teaching() -> None:
    with pytest.raises(ProtocolError, match="renamed 'balanced'"):
        get_protocol("moderate")
    with pytest.raises(ProtocolError, match="renamed 'stringent'"):
        get_protocol("precise")
    with pytest.raises(ProtocolError, match="available: balanced, fast, stringent"):
        get_protocol("extreme")


def test_protocol_details_derives_ase_fmax_scale() -> None:
    details = protocol_details("balanced")
    assert details["name"] == "balanced"
    assert details["forc_conv_thr_ev_per_ang"] == pytest.approx(0.00257, abs=1e-5)


# -- expansion --------------------------------------------------------------------------


def test_balanced_expansion_golden(sssp) -> None:
    atoms = bulk("Si", "diamond", a=5.43)  # 2-atom primitive cell
    options = qe_protocol_options(atoms, protocol="balanced")

    assert options["pseudo_family"] == "SSSP/1.3.0/PBEsol/efficiency"
    assert options["pseudopotentials"] == {"Si": "Si.test.upf"}

    # |b| = 2*pi*sqrt(3)/5.43 = 2.0036 1/A; / 0.15 -> ceil(13.36) = 14
    assert options["kpts"] == [14, 14, 14]

    system = options["input_data"]["system"]
    assert (system["ecutwfc"], system["ecutrho"]) == (30.0, 240.0)
    assert system["occupations"] == "smearing"
    assert system["smearing"] == "cold"
    assert system["degauss"] == 0.02

    control = options["input_data"]["control"]
    assert control["etot_conv_thr"] == pytest.approx(2 * 1e-5)  # natoms * per-atom
    assert control["forc_conv_thr"] == 1e-4  # NOT scaled with natoms
    assert control["tprnfor"] is True
    assert control["tstress"] is True

    electrons = options["input_data"]["electrons"]
    assert electrons["conv_thr"] == pytest.approx(2 * 2e-10)
    assert electrons["mixing_beta"] == 0.4
    assert electrons["electron_maxstep"] == 80


def test_cutoffs_are_elementwise_maxima_over_structure(sssp) -> None:
    from ase import Atoms

    atoms = Atoms("SiO", positions=[(0, 0, 0), (0, 0, 1.6)], cell=[8, 8, 8], pbc=True)
    options = qe_protocol_options(atoms, protocol="balanced")
    system = options["input_data"]["system"]
    assert (system["ecutwfc"], system["ecutrho"]) == (75.0, 600.0)  # O wins both
    assert options["pseudopotentials"] == {"Si": "Si.test.upf", "O": "O.test.upf"}
    # cubic 8 A cell: |b| = 2*pi/8 = 0.7854; / 0.15 -> ceil(5.24) = 6
    assert options["kpts"] == [6, 6, 6]


def test_natoms_scaling(sssp) -> None:
    atoms = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)  # 32 atoms
    options = qe_protocol_options(atoms, protocol="balanced")
    assert options["input_data"]["control"]["etot_conv_thr"] == pytest.approx(32 * 1e-5)
    assert options["input_data"]["electrons"]["conv_thr"] == pytest.approx(32 * 2e-10)
    assert options["input_data"]["control"]["forc_conv_thr"] == 1e-4


def test_kpoints_distance_drives_mesh(sssp) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    fast = qe_protocol_options(atoms, protocol="fast")
    stringent = qe_protocol_options(
        atoms, protocol="stringent", family="SSSP/1.3.0/PBEsol/efficiency"
    )
    assert fast["kpts"] == [7, 7, 7]  # 2.0036 / 0.3 -> ceil(6.68)
    assert stringent["kpts"] == [21, 21, 21]  # 2.0036 / 0.1 -> ceil(20.04)


def test_insulator_gets_fixed_occupations(sssp) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    options = qe_protocol_options(atoms, protocol="balanced", electronic_type="insulator")
    system = options["input_data"]["system"]
    assert system["occupations"] == "fixed"
    assert "smearing" not in system
    assert "degauss" not in system
    with pytest.raises(ProtocolError, match="electronic_type"):
        qe_protocol_options(atoms, protocol="balanced", electronic_type="semimetal")


def test_overrides_merge_recursively_right_wins(sssp) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    options = qe_protocol_options(
        atoms,
        protocol="balanced",
        overrides={"kpts": (4, 4, 4), "input_data": {"system": {"ecutwfc": 90.0}}},
    )
    assert options["kpts"] == (4, 4, 4)
    system = options["input_data"]["system"]
    assert system["ecutwfc"] == 90.0  # overridden
    assert system["ecutrho"] == 240.0  # sibling survives the merge
    assert options["input_data"]["electrons"]["mixing_beta"] == 0.4


def test_explicit_family_overrides_protocol_recommendation(sssp, root: Path) -> None:
    make_family(root, name="SSSP/1.2.1/PBE/precision", symbols=("Si",))
    atoms = bulk("Si", "diamond", a=5.43)
    options = qe_protocol_options(atoms, protocol="balanced", family="SSSP/1.2/PBE/precision")
    assert options["pseudo_family"] == "SSSP/1.2.1/PBE/precision"


def test_missing_family_error_teaches_install(root: Path) -> None:
    from slab.pseudos import PseudoFamilyError

    atoms = bulk("Si", "diamond", a=5.43)
    with pytest.raises(PseudoFamilyError, match="slab pseudos install"):
        qe_protocol_options(atoms, protocol="balanced")


def test_nonperiodic_structure_refused(sssp) -> None:
    from ase import Atoms

    molecule = Atoms("SiO", positions=[(0, 0, 0), (0, 0, 1.6)])  # no cell, no pbc
    with pytest.raises(ProtocolError, match="periodic"):
        qe_protocol_options(molecule, protocol="balanced")
    slab_2d = Atoms(
        "Si", positions=[(0, 0, 0)], cell=[4, 4, 20], pbc=(True, True, False)
    )
    with pytest.raises(ProtocolError, match="pbc"):
        qe_protocol_options(slab_2d, protocol="balanced")


def test_empty_structure_refused(sssp) -> None:
    from ase import Atoms

    with pytest.raises(ProtocolError, match="empty structure"):
        qe_protocol_options(Atoms(cell=[4, 4, 4], pbc=True), protocol="balanced")


def test_mesh_arithmetic_matches_formula(sssp) -> None:
    atoms = bulk("Si", "diamond", a=5.43)
    reciprocal_norm = 2 * math.pi * math.sqrt(3) / 5.43
    for name in available_protocols():
        options = qe_protocol_options(atoms, protocol=name, family="SSSP/1.3.0/PBEsol/efficiency")
        expected = max(1, math.ceil(reciprocal_norm / get_protocol(name).kpoints_distance))
        assert options["kpts"] == [expected, expected, expected]


# -- expansion feeds the engine ---------------------------------------------------------


def test_expanded_options_build_a_qe_calculator(sssp, tmp_path: Path) -> None:
    from slab.backends import close_calculator, describe_engine, get_calculator

    atoms = bulk("Si", "diamond", a=5.43)
    options = qe_protocol_options(atoms, protocol="balanced")
    calc = get_calculator("qe", command="/bin/echo", **options)
    try:
        assert calc.parameters["kpts"] == [14, 14, 14]
        assert calc.parameters["input_data"]["system"]["ecutwfc"] == 30.0
        assert calc.parameters["input_data"]["control"]["tprnfor"] is True
        assert calc.profile.pseudo_dir.endswith("SSSP_1.3.0_PBEsol_efficiency")
    finally:
        close_calculator(calc)

    identity = describe_engine("qe", {"command": "/bin/echo", **options})
    assert identity["pseudo_family"] == "SSSP/1.3.0/PBEsol/efficiency"
    assert len(identity["pseudo_family_digest"]) == 12
    assert identity["pseudo_dir"] is None  # content identity, never a local path


def test_family_digest_changes_engine_identity(root: Path) -> None:
    from slab.backends import describe_engine

    make_family(root, symbols=("Si",))
    a = describe_engine(
        "qe", {"command": "/bin/echo", "pseudo_family": "SSSP/1.3/PBEsol/efficiency"}
    )
    import shutil

    shutil.rmtree(root)
    make_family(root, symbols=("Si", "O"))  # different contents, same name
    b = describe_engine(
        "qe", {"command": "/bin/echo", "pseudo_family": "SSSP/1.3/PBEsol/efficiency"}
    )
    assert a["pseudo_family_digest"] != b["pseudo_family_digest"]


def test_unknown_family_never_produces_a_cache_identity(root: Path) -> None:
    from slab.backends import describe_engine
    from slab.pseudos import PseudoFamilyError

    with pytest.raises(PseudoFamilyError, match="not installed"):
        describe_engine("qe", {"command": "/bin/echo", "pseudo_family": "SSSP/9.9/PBE/ghost"})


def test_pseudo_family_conflicts_with_pseudo_dir(sssp, tmp_path: Path) -> None:
    from slab import EngineNotAvailableError
    from slab.backends import get_calculator

    with pytest.raises(EngineNotAvailableError, match="not both"):
        get_calculator(
            "qe",
            command="/bin/echo",
            pseudo_dir=str(tmp_path),
            pseudo_family="SSSP/1.3/PBEsol/efficiency",
        )


def test_mesh_rounding_guard_matches_aiida(sssp) -> None:
    """AiiDA rounds |b|/distance to 5 decimals before ceil, so float noise at
    an exact-integer ratio must not bump the mesh: a = 2*pi/1.5 under the
    balanced protocol gives a ratio of 10.000000000000002 -> 10, not 11."""
    atoms = bulk("Po", "sc", a=2 * math.pi / 1.5)  # simple cubic
    atoms.symbols = ["Si"]
    options = qe_protocol_options(atoms, protocol="balanced")
    assert options["kpts"] == [10, 10, 10]


def test_expanded_options_are_json_faithful(sssp) -> None:
    """The expanded options must survive a JSON round-trip unchanged, so the
    traced identity is canonical sorted-key JSON — never a pickle whose bytes
    depend on container types and dict insertion order."""
    import json as _json

    atoms = bulk("Si", "diamond", a=5.43)
    options = qe_protocol_options(atoms, protocol="balanced")
    assert _json.loads(_json.dumps(options)) == options


def test_family_identity_is_portable_across_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same family bytes under two different roots -> identical qe identity
    (the local install path must not leak into the cache key)."""
    from slab.backends import describe_engine

    identities = []
    for where in ("root-a", "root-b"):
        monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / where))
        make_family(tmp_path / where)
        identities.append(
            describe_engine(
                "qe", {"command": "/bin/echo", "pseudo_family": "SSSP/1.3/PBEsol/efficiency"}
            )
        )
    assert identities[0] == identities[1]
    assert identities[0]["pseudo_dir"] is None
