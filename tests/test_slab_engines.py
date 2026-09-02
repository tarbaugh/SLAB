"""Tests for the cluster engine registry and its integration with backends."""

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from slab import EngineNotAvailableError, engines
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
    """Keep tests blind to any real $SLAB_ENGINES, $ROOTSTOCK_ROOT, or user config,
    and keep what they write to the environment from outliving them."""
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # build_engine writes registry env process-wide, and monkeypatch only
    # restores what it set itself: snapshot and restore the whole map, and
    # the module's own record of what it applied.
    monkeypatch.setattr(engines, "_APPLIED_ENV", {})
    before = dict(os.environ)
    yield
    for key in set(os.environ) - set(before):
        del os.environ[key]
    for key, value in before.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def _write_registry(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


# -- schema ----------------------------------------------------------------------------


def test_registry_shape_and_defaults() -> None:
    registry = EngineRegistry.model_validate(
        {
            "cluster": "delta",
            "engines": {"qe-delta": {"calculator": "ase.calculators.espresso.Espresso"}},
        }
    )
    assert registry.layout_version == 1
    assert registry.engines["qe-delta"].kind == "ase"
    assert registry.engines["qe-delta"].options == {}


def test_registry_refuses_newer_layout() -> None:
    with pytest.raises(ValidationError, match="upgrade slab"):
        EngineRegistry.model_validate({"layout_version": 99, "engines": {}})


@pytest.mark.parametrize("shadowing", ["qe", "lammps"])
def test_registry_refuses_builtin_shadowing(shadowing: str) -> None:
    with pytest.raises(ValidationError, match="qe-delta"):
        EngineRegistry.model_validate({"engines": {shadowing: EMT_SPEC}})


def test_registry_permits_the_retired_mace_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the mace built-in retired, a site may declare ``mace`` as an
    alias for its rootstock checkpoint. Reserving the name would block the
    migration path; check the registry accepts it and it resolves."""
    registry = _write_registry(
        tmp_path / "engines.json", {"cluster": "delta", "engines": {"mace": EMT_SPEC}}
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    assert type(get_calculator("mace")).__name__ == "EMT"


def test_retired_mace_message_names_both_migration_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script still saying engine='mace' must land on a message that points
    at both migration routes: a registry alias, and a rootstock-served
    checkpoint id used directly as the engine name."""
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    with pytest.raises(EngineNotAvailableError) as excinfo:
        get_calculator("mace")
    text = str(excinfo.value)
    assert "'mace'" in text
    assert "emt, lammps, lj, qe, rootstock" in text
    # Points at the two migration paths: a registry, or a served checkpoint.
    assert "engine registry" in text or "SLAB_ENGINES" in text
    assert "rootstock" in text or "checkpoint" in text


@pytest.mark.parametrize("sneaky", ["QE", "Emt", "rootstock ", " lj", "EMT"])
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
    repo_root = Path(__file__).resolve().parent.parent  # cwd-independent
    payload = json.loads((repo_root / "examples" / "engines.example.json").read_text())
    registry = EngineRegistry.model_validate(payload)
    assert {"mace-mp", "uma", "lammps-delta", "qe-delta", "vasp"} <= set(registry.engines)


def test_template_registry_file_is_valid() -> None:
    """templates/engines.json must satisfy the schema its copies will be held to."""
    repo_root = Path(__file__).resolve().parent.parent  # cwd-independent
    payload = json.loads((repo_root / "templates" / "engines.json").read_text())
    registry = EngineRegistry.model_validate(payload)
    assert {"vasp", "qe-mycluster"} <= set(registry.engines)


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


def test_user_config_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user registry lives under $XDG_CONFIG_HOME when set (as the user
    config does), else ~/.config; the suite isolates through the former."""
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    (xdg / "slab").mkdir(parents=True)
    _write_registry(xdg / "slab" / "engines.json", {"cluster": "laptop", "engines": {}})
    assert load_registry().cluster == "laptop"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    config = tmp_path / "home" / ".config" / "slab"
    config.mkdir(parents=True)
    _write_registry(config / "engines.json", {"cluster": "home", "engines": {}})
    assert load_registry().cluster == "home"


def test_missing_explicit_or_env_path_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        find_registry_path(tmp_path / "ghost.json")
    monkeypatch.setenv("SLAB_ENGINES", str(tmp_path / "ghost.json"))
    with pytest.raises(FileNotFoundError, match="SLAB_ENGINES"):
        find_registry_path()


def test_invalid_registry_is_loud_never_empty(tmp_path: Path) -> None:
    bad = _write_registry(tmp_path / "bad.json", {"engines": {"vasp": {}}})  # no calculator
    with pytest.raises(EngineNotAvailableError, match=str(bad)):
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
    assert available_engines(load_registry()) == (
        "emt",
        "lammps",
        "lj",
        "qe",
        "rootstock",
        "emt-cluster",
    )


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
        {"cluster": "delta", "engines": {"qe-delta": {**EMT_SPEC, "version": "7.3.1"}}},
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    described = describe_engine("qe-delta")
    assert described["source"] == "registry:delta"
    assert described["version"] == "7.3.1"
    # full spec is part of the identity: options/env edits change the fingerprint
    assert described["spec"]["calculator"] == EMT_SPEC["calculator"]
    assert describe_engine("QE-Delta ")["source"] == "registry:delta"  # normalized lookup
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


# -- silent rootstock checkpoint serving -----------------------------------------------


@pytest.fixture()
def rootstock_root(tmp_path: Path) -> Path:
    """A minimal fake rootstock install: envs/<name>/env_source.py with CHECKPOINTS."""
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    root = tmp_path / "rootstock-install"
    env_dir = root / "envs" / "fake-mace"
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text(
        'CHECKPOINTS = {"fake-mace-checkpoint": "small", "fake-mace-large": "large"}\n'
    )
    return root


def test_checkpoint_id_is_an_engine_name(
    rootstock_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The requested UX: engine='<checkpoint-id>' and rootstock serves it silently."""
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
    calc = get_calculator("fake-mace-checkpoint")
    try:
        assert type(calc).__name__ == "RootstockCalculator"
        assert calc.checkpoint == "fake-mace-checkpoint"
    finally:
        close_calculator(calc)


def test_checkpoint_via_root_option(rootstock_root: Path) -> None:
    calc = get_calculator("fake-mace-large", root=str(rootstock_root), device="cpu")
    try:
        assert calc.checkpoint == "fake-mace-large"
    finally:
        close_calculator(calc)


def test_registry_alias_beats_checkpoint_id(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A maintainer's curated registry alias wins over bare checkpoint resolution."""
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
    registry = _write_registry(
        tmp_path / "engines.json", {"engines": {"fake-mace-checkpoint": EMT_SPEC}}
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    assert type(get_calculator("fake-mace-checkpoint")).__name__ == "EMT"


def test_checkpoint_engine_rejects_redundant_checkpoint_option(rootstock_root: Path) -> None:
    with pytest.raises(EngineNotAvailableError, match="itself a rootstock checkpoint"):
        get_calculator("fake-mace-checkpoint", root=str(rootstock_root), checkpoint="other")


def test_unknown_name_mentions_rootstock_install(
    rootstock_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
    with pytest.raises(EngineNotAvailableError, match="not declared as a checkpoint"):
        get_calculator("nope-checkpoint")


def test_unknown_name_hints_when_root_unconfigured() -> None:
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    with pytest.raises(EngineNotAvailableError, match="ROOTSTOCK_ROOT"):
        get_calculator("nope-checkpoint")


def test_suite_is_blind_to_an_ambient_rootstock_user_config(rootstock_root: Path) -> None:
    """A real ``~/.config/rootstock/config.toml`` must not reach the tests.

    Rootstock resolves that path at import time from the literal home, so
    neither ``$XDG_CONFIG_HOME`` nor a late ``$HOME`` redirect blinds it —
    only the root conftest's patch of ``DEFAULT_CONFIG_FILE`` does. This
    proves the patch points at a throwaway location AND that the location is
    the one discovery actually reads: a config written there is honored.
    """
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    from rootstock.config import DEFAULT_CONFIG_FILE

    assert Path.home() not in Path(DEFAULT_CONFIG_FILE).parents
    DEFAULT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_FILE.write_text(f'root = "{rootstock_root}"\n')
    with pytest.raises(EngineNotAvailableError, match="not declared as a checkpoint"):
        get_calculator("nope-checkpoint")


def test_unknown_cluster_surfaces_rootstock_error(rootstock_root: Path) -> None:
    with pytest.raises(EngineNotAvailableError, match=r"[Uu]nknown cluster"):
        get_calculator("fake-mace-checkpoint", cluster="not-a-real-cluster")


def test_describe_checkpoint_engine(rootstock_root: Path) -> None:
    described = describe_engine("fake-mace-checkpoint", {"root": str(rootstock_root)})
    assert described["source"] == "rootstock"
    assert described["checkpoint"] == "fake-mace-checkpoint"
    assert described["version"]  # the rootstock client version


def test_engines_overview_lists_checkpoints(
    rootstock_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slab._ops import engines_overview

    monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
    overview = engines_overview()
    assert overview["rootstock"]["root"] == str(rootstock_root)
    assert overview["rootstock"]["checkpoints"] == {
        "fake-mace": ["fake-mace-checkpoint", "fake-mace-large"]
    }


def test_checkpoint_resolution_silent_without_rootstock_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra, unknown names never mention rootstock — silent serving
    is opt-in, and laptop users get the plain engine error."""
    import builtins

    real_import = builtins.__import__

    def hide_rootstock(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("rootstock"):
            raise ImportError("No module named 'rootstock'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", hide_rootstock)
    with pytest.raises(EngineNotAvailableError) as excinfo:
        get_calculator("some-checkpoint-id")
    assert "rootstock is installed" not in str(excinfo.value)

    from slab._ops import engines_overview

    assert engines_overview()["rootstock"] is None


def test_unreadable_rootstock_install_reported_not_crashed(
    rootstock_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os as _os

    envs = rootstock_root / "envs"
    _os.chmod(envs, 0o000)
    try:
        with pytest.raises(EngineNotAvailableError, match="could not read"):
            get_calculator("fake-mace-checkpoint", root=str(rootstock_root))
        monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
        from slab._ops import engines_overview

        overview = engines_overview()
        assert overview["rootstock"]["error"]
        assert overview["rootstock"]["checkpoints"] == {}
    finally:
        _os.chmod(envs, 0o755)


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


# -- env isolation: import-time refusal, builtin steering, the qe factory ---------------


def test_registry_refuses_import_time_env() -> None:
    """ASE parses its config file once at import — before any registry entry
    runs — so an entry promising ASE_CONFIG_PATH would silently never apply."""
    with pytest.raises(ValidationError, match="ASE_CONFIG_PATH"):
        EngineSpec.model_validate(
            {
                "calculator": "ase.calculators.espresso.Espresso",
                "env": {"ASE_CONFIG_PATH": "/sw/slab/ase.ini"},
            }
        )


def test_registry_refusal_names_the_entry_at_load(tmp_path: Path) -> None:
    payload = {
        "engines": {
            "qe-broken": {
                "calculator": "ase.calculators.espresso.Espresso",
                "env": {"ASE_CONFIG_PATH": "/sw/ase.ini"},
            }
        }
    }
    path = tmp_path / "engines.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(EngineNotAvailableError, match="qe-broken") as excinfo:
        load_registry(path)
    # The message names the file (several registry locations may exist) and
    # the offending variable, so the maintainer knows what to fix where.
    assert "ASE_CONFIG_PATH" in str(excinfo.value)
    assert str(path) in str(excinfo.value)


def test_registry_warns_when_env_steers_builtin_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting ASE_LAMMPSRUN_COMMAND redirects the BUILT-IN lammps engine's
    command resolution for the rest of the process — visible, never silent,
    even when the variable was previously unset."""
    import os

    monkeypatch.delenv("ASE_LAMMPSRUN_COMMAND", raising=False)
    spec = EngineSpec.model_validate(
        {"calculator": "ase.calculators.emt.EMT", "env": {"ASE_LAMMPSRUN_COMMAND": "lmp_site"}}
    )
    with pytest.warns(UserWarning, match="built-in"):
        build_engine("emt-steering", spec)
    assert os.environ["ASE_LAMMPSRUN_COMMAND"] == "lmp_site"


def test_registry_qe_factory_builds_the_builtin_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSON-able route to a curated QE alias: slab's own factory, with
    the built-in engine's guards, instead of an env-mediated ASE config."""
    import os

    bins = tmp_path / "bin"
    bins.mkdir()
    pw = bins / "pw.x"
    pw.write_text("#!/bin/sh\nexit 0\n")
    pw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bins}:{os.environ.get('PATH', '')}")
    spec = EngineSpec.model_validate(
        {
            "calculator": "slab.backends.qe_calculator",
            "options": {"command": "pw.x", "pseudo_dir": str(tmp_path)},
            "version": "7.4.1",
        }
    )
    calculator = build_engine(
        "qe-site", spec, kpts=None, input_data={"system": {"ecutwfc": 30.0}}
    )
    try:
        assert str(calculator.profile.command) == "pw.x"
    finally:
        close_calculator(calculator)
    # And the guards came along: a protocol name is refused here too.
    with pytest.raises(EngineNotAvailableError, match="qe_protocol_options"):
        build_engine("qe-site", spec, kpts=None, protocol="balanced")


def test_registry_qe_factory_is_self_contained(tmp_path: Path) -> None:
    """An alias's spec (plus traced caller options) is its ENTIRE cache
    identity — so nothing may come from slab.toml or ASE's config, and the
    k-point policy the task-level guard would enforce for engine="qe" must
    be stated explicitly here (the guard never sees alias names)."""
    from slab.backends import lammps_calculator, qe_calculator

    with pytest.raises(EngineNotAvailableError, match="command="):
        qe_calculator(pseudo_dir=str(tmp_path), kpts=None)
    with pytest.raises(EngineNotAvailableError, match="pseudo_dir="):
        qe_calculator(command="pw.x", kpts=None)
    with pytest.raises(EngineNotAvailableError, match="k-point"):
        qe_calculator(command="pw.x", pseudo_dir=str(tmp_path))
    with pytest.raises(EngineNotAvailableError, match="command="):
        lammps_calculator(pair_style="eam", pair_coeff=["* * x"])
    with pytest.raises(EngineNotAvailableError, match="absolute"):
        lammps_calculator(
            command="/bin/echo",
            pair_style="eam/alloy",
            pair_coeff=["* * Cu.eam.alloy Cu"],
            files=["Cu.eam.alloy"],
        )


def test_template_qe_alias_uses_the_factory_not_import_time_env() -> None:
    """templates/engines.json must model the working mechanism."""
    repo_root = Path(__file__).resolve().parent.parent
    payload = json.loads((repo_root / "templates" / "engines.json").read_text())
    spec = payload["engines"]["qe-mycluster"]
    assert spec["calculator"] == "slab.backends.qe_calculator"
    assert "env" not in spec
    # No entry may carry the import-time var (the notes may explain the rule).
    assert "ASE_CONFIG_PATH" not in json.dumps(payload["engines"])


def test_applied_env_restores_originals_and_skips_no_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The submission boundary needs the environment as it was BEFORE any
    registry entry ran: originals for overwritten variables, absence for
    created ones, and no entry at all for values that already matched."""
    import os

    import slab.engines as engines

    monkeypatch.setattr(engines, "_APPLIED_ENV", {})
    monkeypatch.setenv("SLAB_TEST_KEPT", "shell-value")
    monkeypatch.setenv("SLAB_TEST_SAME", "same")
    monkeypatch.delenv("SLAB_TEST_NEW", raising=False)
    spec = EngineSpec.model_validate(
        {
            "calculator": "ase.calculators.emt.EMT",
            "env": {
                "SLAB_TEST_KEPT": "registry-value",
                "SLAB_TEST_SAME": "same",
                "SLAB_TEST_NEW": "created",
            },
        }
    )
    with pytest.warns(UserWarning):
        build_engine("emt-env", spec)
    applied = engines.applied_env()
    assert applied["SLAB_TEST_KEPT"] == ("shell-value", "registry-value")
    assert applied["SLAB_TEST_NEW"] == (None, "created")  # originally unset
    assert "SLAB_TEST_SAME" not in applied  # no mutation happened
    assert os.environ["SLAB_TEST_KEPT"] == "registry-value"


def test_poisoning_warning_is_not_deduplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived process alternating between entries must warn EVERY time,
    not once per process — Python's default warning registry would swallow
    the repeats."""
    import warnings as warnings_module

    monkeypatch.setenv("SLAB_TEST_FLIP", "a")
    spec_b = EngineSpec.model_validate(
        {"calculator": "ase.calculators.emt.EMT", "env": {"SLAB_TEST_FLIP": "b"}}
    )
    spec_a = EngineSpec.model_validate(
        {"calculator": "ase.calculators.emt.EMT", "env": {"SLAB_TEST_FLIP": "a"}}
    )
    with warnings_module.catch_warnings(record=True) as seen:
        warnings_module.simplefilter("default")
        build_engine("flip-b", spec_b)
        build_engine("flip-a", spec_a)
        build_engine("flip-b", spec_b)
        build_engine("flip-a", spec_a)
    messages = [str(w.message) for w in seen]
    assert len(messages) == 4, messages


def test_verify_probe_refuses_srun_outside_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    registry = EngineRegistry.model_validate(
        {
            "engines": {
                "qe-site": {
                    "calculator": "ase.calculators.emt.EMT",
                    "probe": ["srun", "pw.x", "-h"],
                }
            }
        }
    )
    (result,) = verify_engines(registry)
    assert not result.ok
    assert "srun" in result.detail and "allocation" in result.detail


def test_verify_probe_runs_in_a_private_cwd_with_closed_stdin(tmp_path: Path) -> None:
    """pw.x-style probes write debris into cwd and lmp-style probes block on
    stdin; the registry probe path must be as careful as the version probes."""
    import os

    registry = EngineRegistry.model_validate(
        {
            "engines": {
                "debris": {
                    "calculator": "ase.calculators.emt.EMT",
                    "probe": ["sh", "-c", "touch debris.txt && head -c1 >/dev/null; exit 0"],
                }
            }
        }
    )
    before = set(os.listdir(tmp_path))
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        (result,) = verify_engines(registry)
    finally:
        os.chdir(cwd)
    assert result.ok, result.detail  # closed stdin: head reads EOF, no hang
    assert set(os.listdir(tmp_path)) == before  # debris landed in the private cwd


# -- [engines.rootstock]: the local-install machine fact --------------------------------


def test_checkpoint_id_resolves_via_config_root(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LOCAL rootstock install is a machine fact like [engines.qe] command:
    declared once in slab.toml, checkpoint ids then work as engine names with
    no per-call options and no $ROOTSTOCK_ROOT export."""
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(
        f'[engines.rootstock]\nroot = "{rootstock_root}"\n'
    )
    monkeypatch.chdir(project)
    calc = get_calculator("fake-mace-checkpoint")
    try:
        assert type(calc).__name__ == "RootstockCalculator"
        assert calc.checkpoint == "fake-mace-checkpoint"
    finally:
        close_calculator(calc)


def test_explicit_options_beat_config_rootstock_defaults(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rootstock

    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text('[engines.rootstock]\ncluster = "config-cluster"\n')
    monkeypatch.chdir(project)
    captured: dict[str, object] = {}

    class StubCalculator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rootstock, "RootstockCalculator", StubCalculator)
    get_calculator("rootstock", checkpoint="x", root=str(rootstock_root))
    assert captured.get("root") == str(rootstock_root)
    assert "cluster" not in captured  # caller named a location: config stays out


def test_config_root_feeds_the_builtin_rootstock_engine(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rootstock

    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(f'[engines.rootstock]\nroot = "{rootstock_root}"\n')
    monkeypatch.chdir(project)
    captured: dict[str, object] = {}

    class StubCalculator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(rootstock, "RootstockCalculator", StubCalculator)
    get_calculator("rootstock", checkpoint="fake-mace-checkpoint")
    assert captured.get("root") == str(rootstock_root)


def test_unconfigured_hint_names_the_config_section() -> None:
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    with pytest.raises(EngineNotAvailableError, match=r"\[engines.rootstock\]"):
        get_calculator("nope-checkpoint")


def test_engines_overview_lists_checkpoints_from_config_root(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """list_engines used to return `rootstock: null` on machines that
    declared the install in slab.toml (not $ROOTSTOCK_ROOT), while a
    served engine=<checkpoint-id> call succeeded — a real trap that made
    an agent believe the machine had no MLIP. Fixed: the overview
    consults [engines.rootstock] first, then rootstock's own defaults."""
    from slab._ops import engines_overview

    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(f'[engines.rootstock]\nroot = "{rootstock_root}"\n')
    monkeypatch.chdir(project)
    section = engines_overview()["rootstock"]
    assert section is not None
    assert section["root"] == str(rootstock_root)
    assert section["root_source"] == "engines.rootstock.root"
    assert section["checkpoints"] == {
        "fake-mace": ["fake-mace-checkpoint", "fake-mace-large"]
    }


def test_engines_overview_labels_the_root_source(
    rootstock_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overview names WHERE the root came from, so a viewer sees why
    the id list is what it is — rootstock defaults, engine-config root,
    or engine-config cluster (with a hint when the cluster is unknown)."""
    from slab._ops import engines_overview

    # rootstock defaults path: no slab config, $ROOTSTOCK_ROOT points at the install.
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(rootstock_root))
    section = engines_overview()["rootstock"]
    assert section is not None
    assert section["root_source"] == "rootstock defaults"

    # engine-config cluster path: an unknown cluster is a live config problem,
    # surfaced as an empty checkpoints dict with an actionable source label.
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text('[engines.rootstock]\ncluster = "no-such-cluster"\n')
    monkeypatch.chdir(project)
    section = engines_overview()["rootstock"]
    assert section is not None
    assert section["checkpoints"] == {}
    assert section["root_source"] == "engines.rootstock.cluster (unknown)"
    assert "no-such-cluster" in section["error"]


def test_installed_rootstock_with_nothing_configured_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one null that reached a real cluster session: the package is
    installed but no install root is declared anywhere (no [engines.rootstock]
    in reach of the working directory, no $ROOTSTOCK_ROOT). A bare null hid
    which of the three preconditions was missing; the overview must say.
    Fake modules make this path testable on machines without the extra."""
    import sys
    import types

    fake = types.ModuleType("rootstock")
    fake_config = types.ModuleType("rootstock.config")
    fake_config.resolve_default_root = lambda: None  # type: ignore[attr-defined]
    fake_env = types.ModuleType("rootstock.environment")
    fake_env.list_declared_checkpoints = lambda root: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rootstock", fake)
    monkeypatch.setitem(sys.modules, "rootstock.config", fake_config)
    monkeypatch.setitem(sys.modules, "rootstock.environment", fake_env)
    monkeypatch.delenv("ROOTSTOCK_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)  # no slab.toml here

    from slab._ops import engines_overview

    section = engines_overview()["rootstock"]
    assert section is not None
    assert section["root"] is None
    assert section["root_source"] == "rootstock defaults"
    assert "[engines.rootstock]" in section["error"]
    assert section["checkpoints"] == {}
