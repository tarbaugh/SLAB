"""Tests for the cluster engine registry and its integration with backends."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from slab import EngineNotAvailableError
from slab.backends import available_engines, close_calculator, describe_engine, get_calculator
from slab.engines import (
    EngineRegistry,
    EngineSpec,
    build_engine,
    find_registry_path,
    load_registry,
    verify_engines,
)

EMT_SPEC = {"calculator": "ase.calculators.emt.EMT", "version": "ase-built-in"}


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests blind to any real $SLAB_ENGINES or ~/.config/slab/engines.json."""
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _write_registry(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


# -- schema ----------------------------------------------------------------------------


def test_registry_shape_and_defaults() -> None:
    registry = EngineRegistry.model_validate(
        {"cluster": "delta", "engines": {"qe": {"calculator": "ase.calculators.espresso.Espresso"}}}
    )
    assert registry.layout_version == 1
    assert registry.engines["qe"].kind == "ase"
    assert registry.engines["qe"].options == {}


def test_registry_refuses_newer_layout() -> None:
    with pytest.raises(ValidationError, match="upgrade slab"):
        EngineRegistry.model_validate({"layout_version": 99, "engines": {}})


def test_registry_refuses_builtin_shadowing() -> None:
    with pytest.raises(ValidationError, match="mace-mp"):
        EngineRegistry.model_validate({"engines": {"mace": EMT_SPEC}})


@pytest.mark.parametrize("sneaky", ["MACE", "Emt", "rootstock ", " lj", "EMT"])
def test_registry_refuses_case_and_whitespace_variants(sneaky: str) -> None:
    """A case-variant of a built-in would validate and then silently resolve
    to the built-in (lookups normalize); canonical names close the bypass."""
    with pytest.raises(ValidationError, match="lowercase"):
        EngineRegistry.model_validate({"engines": {sneaky: EMT_SPEC}})


def test_registry_refuses_non_canonical_names_generally() -> None:
    with pytest.raises(ValidationError, match="lowercase"):
        EngineRegistry.model_validate({"engines": {"My-QE": EMT_SPEC}})


def test_registry_refuses_empty_probe() -> None:
    with pytest.raises(ValidationError):
        EngineSpec.model_validate({**EMT_SPEC, "probe": []})


def test_spec_refuses_unknown_kind_and_fields() -> None:
    with pytest.raises(ValidationError):
        EngineSpec.model_validate({"calculator": "x.Y", "kind": "container"})
    with pytest.raises(ValidationError):
        EngineSpec.model_validate({"calculator": "x.Y", "bogus": 1})


def test_example_registry_file_is_valid() -> None:
    payload = json.loads(Path("examples/engines.example.json").read_text())
    registry = EngineRegistry.model_validate(payload)
    assert {"mace-mp", "uma", "lammps", "qe", "vasp"} <= set(registry.engines)


# -- discovery -------------------------------------------------------------------------


def test_no_registry_configured(tmp_path: Path) -> None:
    assert find_registry_path() is None
    assert load_registry() is None


def test_explicit_path_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = _write_registry(tmp_path / "a.json", {"engines": {}})
    other = _write_registry(tmp_path / "b.json", {"engines": {}})
    monkeypatch.setenv("SLAB_ENGINES", str(other))
    assert find_registry_path(explicit) == explicit
    assert find_registry_path() == other


def test_user_config_fallback(tmp_path: Path) -> None:
    config = tmp_path / "home" / ".config" / "slab"
    config.mkdir(parents=True)
    _write_registry(config / "engines.json", {"cluster": "laptop", "engines": {}})
    assert load_registry().cluster == "laptop"


def test_missing_explicit_or_env_path_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        find_registry_path(tmp_path / "ghost.json")
    monkeypatch.setenv("SLAB_ENGINES", str(tmp_path / "ghost.json"))
    with pytest.raises(FileNotFoundError, match="SLAB_ENGINES"):
        find_registry_path()


def test_invalid_registry_is_loud_never_empty(tmp_path: Path) -> None:
    bad = _write_registry(tmp_path / "bad.json", {"engines": {"qe": {}}})  # no calculator
    with pytest.raises(ValidationError):
        load_registry(bad)


# -- building --------------------------------------------------------------------------


def test_build_engine_dotted_path_and_option_merge() -> None:
    spec = EngineSpec.model_validate(
        {"calculator": "ase.calculators.lj.LennardJones", "options": {"sigma": 2.0, "epsilon": 1.0}}
    )
    calc = build_engine("lj-tuned", spec, sigma=3.5)  # caller wins key-by-key
    assert type(calc).__name__ == "LennardJones"
    assert calc.parameters["sigma"] == 3.5
    assert calc.parameters["epsilon"] == 1.0


def test_build_engine_applies_declared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAB_TEST_ENGINE_ENV", raising=False)
    spec = EngineSpec.model_validate(
        {
            "calculator": "ase.calculators.emt.EMT",
            "env": {"SLAB_TEST_ENGINE_ENV": "set-by-registry"},
        }
    )
    build_engine("emt-cluster", spec)
    import os

    assert os.environ["SLAB_TEST_ENGINE_ENV"] == "set-by-registry"


def test_build_engine_warns_on_env_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Declared env is process-wide by design; overwriting an existing value
    must be visible, never silent cross-engine poisoning."""
    monkeypatch.setenv("SLAB_TEST_CONFLICT", "from-first-engine")
    spec = EngineSpec.model_validate(
        {"calculator": "ase.calculators.emt.EMT", "env": {"SLAB_TEST_CONFLICT": "from-second"}}
    )
    with pytest.warns(UserWarning, match="process-wide"):
        build_engine("second", spec)
    import os

    assert os.environ["SLAB_TEST_CONFLICT"] == "from-second"
    # same value again: no warning
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        build_engine("second-again", spec)


def test_build_engine_import_failures_are_actionable() -> None:
    missing = EngineSpec.model_validate({"calculator": "not_a_real_module.Calc"})
    with pytest.raises(EngineNotAvailableError, match="not_a_real_module"):
        build_engine("ghost", missing)
    bad_attr = EngineSpec.model_validate({"calculator": "ase.calculators.emt.NoSuchThing"})
    with pytest.raises(EngineNotAvailableError, match="no attribute"):
        build_engine("ghost", bad_attr)
    not_dotted = EngineSpec.model_validate({"calculator": "EMT"})
    with pytest.raises(EngineNotAvailableError, match="dotted"):
        build_engine("ghost", not_dotted)


# -- backends integration --------------------------------------------------------------


def test_get_calculator_resolves_registry_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(
        tmp_path / "engines.json", {"cluster": "delta", "engines": {"emt-cluster": EMT_SPEC}}
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    assert type(get_calculator("emt-cluster")).__name__ == "EMT"
    assert available_engines(load_registry()) == ("emt", "lj", "mace", "rootstock", "emt-cluster")


def test_unknown_engine_lists_builtin_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _write_registry(tmp_path / "engines.json", {"engines": {"emt-cluster": EMT_SPEC}})
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    with pytest.raises(EngineNotAvailableError, match="emt-cluster"):
        get_calculator("vasp")


def test_unknown_engine_without_registry_hints_at_setup() -> None:
    with pytest.raises(EngineNotAvailableError, match="SLAB_ENGINES"):
        get_calculator("vasp")


def test_describe_engine_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert describe_engine("emt") == {"engine": "emt", "source": "builtin", "version": None}
    registry = _write_registry(
        tmp_path / "engines.json",
        {"cluster": "delta", "engines": {"qe": {**EMT_SPEC, "version": "7.3.1"}}},
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    described = describe_engine("qe")
    assert described["source"] == "registry:delta"
    assert described["version"] == "7.3.1"
    # full spec is part of the identity: options/env edits change the fingerprint
    assert described["spec"]["calculator"] == EMT_SPEC["calculator"]
    assert describe_engine("QE ")["source"] == "registry:delta"  # normalized lookup
    assert describe_engine("nope")["source"] == "unknown"


def test_rootstock_builtin_requires_checkpoint() -> None:
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    with pytest.raises(EngineNotAvailableError, match="checkpoint"):
        get_calculator("rootstock")


def test_rootstock_builtin_forwards_options(monkeypatch: pytest.MonkeyPatch) -> None:
    rootstock = pytest.importorskip("rootstock", reason="rootstock extra not installed")
    captured: dict[str, object] = {}

    class StubCalculator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rootstock, "RootstockCalculator", StubCalculator)
    calc = get_calculator("rootstock", checkpoint="mace-mp-0-medium", cluster="delta", device="cpu")
    assert isinstance(calc, StubCalculator)
    assert captured == {"checkpoint": "mace-mp-0-medium", "cluster": "delta", "device": "cpu"}


def test_close_calculator_calls_close_when_present() -> None:
    class Worker:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    worker = Worker()
    close_calculator(worker)
    close_calculator(worker)  # safe twice
    assert worker.closed == 2
    close_calculator(object())  # no close(): a no-op, not an error


# -- verify ----------------------------------------------------------------------------


def test_verify_probe_pass_fail_and_importcheck() -> None:
    registry = EngineRegistry.model_validate(
        {
            "engines": {
                "good": {**EMT_SPEC, "probe": [sys.executable, "-c", "pass"]},
                "bad": {**EMT_SPEC, "probe": [sys.executable, "-c", "import sys; sys.exit(3)"]},
                "unprobed": EMT_SPEC,
                "broken-import": {"calculator": "not_a_real_module.Calc"},
                "missing-binary": {**EMT_SPEC, "probe": ["definitely-not-a-command-xyz"]},
            }
        }
    )
    by_name = {r.engine: r for r in verify_engines(registry)}
    assert by_name["good"].ok
    assert not by_name["bad"].ok and "exited 3" in by_name["bad"].detail
    assert by_name["unprobed"].ok and "no probe" in by_name["unprobed"].detail
    assert not by_name["broken-import"].ok
    assert not by_name["missing-binary"].ok


def test_verify_probe_receives_declared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAB_PROBE_ENV", raising=False)
    registry = EngineRegistry.model_validate(
        {
            "engines": {
                "envy": {
                    **EMT_SPEC,
                    "env": {"SLAB_PROBE_ENV": "yes"},
                    "probe": [
                        sys.executable,
                        "-c",
                        "import os, sys; "
                        "sys.exit(0 if os.environ.get('SLAB_PROBE_ENV') == 'yes' else 1)",
                    ],
                }
            }
        }
    )
    (result,) = verify_engines(registry)
    assert result.ok
