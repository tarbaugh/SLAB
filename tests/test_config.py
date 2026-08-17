"""Layered configuration tests: discovery, merging, refusals, integration."""

import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slab.cli import app
from slab.config import (
    CONFIG_TEMPLATE,
    AgentConfig,
    ConfigError,
    HpcConfig,
    LammpsEngineConfig,
    Partition,
    PathsConfig,
    QeEngineConfig,
    ServeConfig,
    SlabConfig,
    config_value,
    find_config_files,
    load_config,
    load_config_with_origins,
    user_config_path,
    write_template,
)

runner = CliRunner()


def _user_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> Path:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    path = xdg / "slab" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


# -- discovery and layering --------------------------------------------------


def test_no_files_yields_pure_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config == SlabConfig()
    assert config.agent.resolved_endpoint == "http://localhost:11434/v1"


def test_layers_merge_project_over_user_over_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site = tmp_path / "site.toml"
    site.write_text(
        '[paths]\nworkspace = "/site/ws"\npseudos = "/site/pseudos"\n'
        '[hpc]\ncluster = "delta"\n'
    )
    monkeypatch.setenv("SLAB_SITE_CONFIG", str(site))
    _user_file(monkeypatch, tmp_path, '[paths]\nworkspace = "/user/ws"\n')
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "slab.toml").write_text('[engines.qe]\ncommand = "srun pw.x"\n')

    config, origins = load_config_with_origins(project_dir)
    assert config.paths.workspace == "/user/ws"  # user beats site
    assert config.paths.pseudos == "/site/pseudos"  # untouched site value survives
    assert config.hpc.cluster == "delta"
    assert config.engines.qe.command == "srun pw.x"
    assert origins["paths.workspace"] == f"{_user_path(tmp_path)} (user)"
    assert origins["paths.pseudos"] == f"{site} (site)"
    assert origins["engines.qe.command"].endswith("(project)")


def _user_path(tmp_path: Path) -> Path:
    return tmp_path / "xdg" / "slab" / "config.toml"


def test_slab_config_env_var_overrides_project_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "elsewhere.toml"
    explicit.write_text('[paths]\nworkspace = "/explicit/ws"\n')
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "slab.toml").write_text('[paths]\nworkspace = "/discovered/ws"\n')
    monkeypatch.setenv("SLAB_CONFIG", str(explicit))
    assert load_config(project_dir).paths.workspace == "/explicit/ws"


def test_dangling_explicit_files_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLAB_CONFIG", str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError, match=r"\$SLAB_CONFIG points to"):
        find_config_files(tmp_path)
    monkeypatch.delenv("SLAB_CONFIG")
    monkeypatch.setenv("SLAB_SITE_CONFIG", str(tmp_path / "missing-site.toml"))
    with pytest.raises(ConfigError, match=r"\$SLAB_SITE_CONFIG points to"):
        find_config_files(tmp_path)


def test_user_layer_honors_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = _user_file(monkeypatch, tmp_path, '[hpc]\ncluster = "laptop"\n')
    assert user_config_path() == path
    assert load_config(tmp_path).hpc.cluster == "laptop"


# -- refusals ----------------------------------------------------------------


def test_malformed_toml_is_a_loud_config_error(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text("[paths\nworkspace=")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(tmp_path)


def test_unknown_key_is_refused_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[agent]\nmodle = "typo"\n')
    with pytest.raises(ConfigError, match=r"agent\.modle") as excinfo:
        load_config(tmp_path)
    assert "slab.toml" in str(excinfo.value)
    assert "unknown key" in str(excinfo.value)


def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text("schema_version = 99\n")
    with pytest.raises(ConfigError, match="newer than this slab understands"):
        load_config(tmp_path)


def test_each_layer_declares_its_own_schema_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A project file must not mask a site file written for a newer slab."""
    site = tmp_path / "site.toml"
    site.write_text('schema_version = 99\n[paths]\nworkspace = "/site/ws"\n')
    monkeypatch.setenv("SLAB_SITE_CONFIG", str(site))
    (tmp_path / "slab.toml").write_text("schema_version = 1\n")
    with pytest.raises(ConfigError, match=r"site\.toml declares schema_version 99"):
        load_config(tmp_path)


def test_wrong_type_is_refused_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[agent]\nmax_turns = "many"\n')
    with pytest.raises(ConfigError, match=r"agent\.max_turns"):
        load_config(tmp_path)


def test_errors_inside_lists_still_name_the_file(tmp_path: Path) -> None:
    """Pydantic reports list positions; the origin walk must still find the file."""
    (tmp_path / "slab.toml").write_text("[hpc]\nsetup = [42]\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(tmp_path)
    assert "slab.toml" in str(excinfo.value)
    assert "configuration" not in str(excinfo.value).split("\n")[-1]


# -- path expansion ----------------------------------------------------------


def test_path_values_expand_home_and_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLAB_TEST_SCRATCH", "/scratch/tom")
    (tmp_path / "slab.toml").write_text(
        '[paths]\nworkspace = "${SLAB_TEST_SCRATCH}/ws"\npseudos = "~/pseudos"\n'
    )
    config = load_config(tmp_path)
    assert config.paths.workspace == "/scratch/tom/ws"
    assert config.paths.pseudos == str(Path("~/pseudos").expanduser())


def test_unset_variable_in_path_is_refused(tmp_path: Path) -> None:
    os.environ.pop("SLAB_NO_SUCH_VAR", None)
    (tmp_path / "slab.toml").write_text('[paths]\nworkspace = "/scratch/$SLAB_NO_SUCH_VAR/ws"\n')
    with pytest.raises(ConfigError, match=r"references \$SLAB_NO_SUCH_VAR"):
        load_config(tmp_path)


def test_setup_lines_are_never_expanded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """[hpc] setup lines are shell for the compute node, not for load time."""
    (tmp_path / "slab.toml").write_text(
        '[hpc]\nsetup = ["export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK"]\n'
    )
    config = load_config(tmp_path)
    assert config.hpc.setup == ("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",)


# -- hpc schema --------------------------------------------------------------


def test_resolve_partition_default_named_and_refusals() -> None:
    hpc = HpcConfig.model_validate(
        {
            "default_partition": "cpu",
            "partitions": {"cpu": {"time_limit": "24:00:00"}, "gpu": {"gres": "gpu:a100:4"}},
        }
    )
    assert hpc.resolve_partition()[0] == "cpu"
    assert hpc.resolve_partition("gpu")[1].gres == "gpu:a100:4"
    with pytest.raises(ConfigError, match="not declared"):
        hpc.resolve_partition("bigmem")
    bare = HpcConfig()
    with pytest.raises(ConfigError, match="no default_partition"):
        bare.resolve_partition()


def test_partition_names_with_whitespace_are_refused(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[hpc.partitions."two words"]\nqos = "x"\n')
    with pytest.raises(ConfigError, match="whitespace"):
        load_config(tmp_path)


def test_default_partition_pointing_nowhere_is_allowed_until_resolved() -> None:
    # Declaring a default before its partition table lands in a higher layer
    # is legal; resolve_partition is where a dangling name refuses.
    hpc = HpcConfig.model_validate({"default_partition": "cpu"})
    with pytest.raises(ConfigError, match="not declared"):
        hpc.resolve_partition()


# -- template ----------------------------------------------------------------


def test_template_round_trips_through_the_loader(tmp_path: Path) -> None:
    target = tmp_path / "slab.toml"
    write_template(target)
    config = load_config(tmp_path)  # commented template = defaults only
    assert config.schema_version == 1
    assert config == SlabConfig()


def test_template_declares_no_table_twice_even_in_comments(tmp_path: Path) -> None:
    """Uncommenting a template line must never yield a duplicate-table error.

    A commented alternative like ``# [agent]`` reads as an invitation, and
    accepting it produced 'Cannot declare (agent,) twice' — so the template
    puts alternatives inside the one table they belong to.
    """
    target = tmp_path / "slab.toml"
    write_template(target)
    headers = re.findall(
        r"^\s*#?\s*(\[[a-z0-9_.\"]+\])\s*$", target.read_text(), flags=re.MULTILINE
    )
    duplicated = {name for name in headers if headers.count(name) > 1}
    assert not duplicated, f"template declares {duplicated} more than once"


def test_every_key_the_template_shows_is_a_key_the_schema_accepts() -> None:
    """The template is documentation; a stale key in it teaches a load error."""
    models = {
        "": SlabConfig,
        "[paths]": PathsConfig,
        "[engines.qe]": QeEngineConfig,
        "[engines.lammps]": LammpsEngineConfig,
        "[hpc]": HpcConfig,
        "[hpc.partitions.cpu]": Partition,
        "[hpc.partitions.gpu]": Partition,
        "[agent]": AgentConfig,
        "[agent.serve]": ServeConfig,
    }
    table = ""
    unknown: list[str] = []
    for line in CONFIG_TEMPLATE.splitlines():
        bare = line.lstrip("# ").rstrip()
        if re.fullmatch(r"\[[a-z0-9_.]+\]", bare):
            assert bare in models, f"template shows an untested table {bare}"
            table = bare
            continue
        key = re.match(r"^([a-z_]+) = ", bare)
        # The Anthropic alternative lives inside [agent] by design.
        if key and key.group(1) not in models[table].model_fields:
            unknown.append(f"{table} {key.group(1)}")
    assert not unknown, f"template names keys the schema does not have: {unknown}"


def test_committed_template_file_is_the_config_template() -> None:
    """templates/slab.toml is byte-for-byte what 'slab config init' writes.

    The committed fill-in-the-blank file and the CLI's template must never
    drift: both claim to show every key the schema accepts.
    """
    repo_root = Path(__file__).resolve().parent.parent  # cwd-independent
    committed = (repo_root / "templates" / "slab.toml").read_text(encoding="utf-8")
    assert committed == CONFIG_TEMPLATE


def test_template_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "slab.toml"
    write_template(target)
    with pytest.raises(ConfigError, match="already exists"):
        write_template(target)
    write_template(target, force=True)  # explicit force replaces


# -- config_value ------------------------------------------------------------


def test_config_value_walks_dotted_paths(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[engines.qe]\ncommand = "pw.x -nk 4"\n')
    assert config_value("engines.qe.command", tmp_path) == "pw.x -nk 4"
    assert config_value("engines.qe.pseudo_dir", tmp_path) is None
    assert config_value("hpc.no.such.path", tmp_path) is None


# -- integration: resolution chains ------------------------------------------


def test_resolve_root_uses_config_below_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab._ops import resolve_root

    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text('[paths]\nworkspace = "/config/ws"\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    assert resolve_root(None) == Path("/config/ws")
    monkeypatch.setenv("SLAB_WORKSPACE", "/env/ws")
    assert resolve_root(None) == Path("/env/ws")  # env beats config
    assert resolve_root("/explicit/ws") == Path("/explicit/ws")


def test_pseudos_root_uses_config_below_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab.pseudos import pseudos_root

    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text('[paths]\npseudos = "/config/pseudos"\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_PSEUDOS", raising=False)
    assert pseudos_root() == Path("/config/pseudos")
    monkeypatch.setenv("SLAB_PSEUDOS", "/env/pseudos")
    assert pseudos_root() == Path("/env/pseudos")


def test_registry_path_from_config_must_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab.engines import find_registry_path

    project = tmp_path / "project"
    project.mkdir()
    registry = tmp_path / "engines.json"
    registry.write_text(json.dumps({"cluster": "t", "engines": {}}))
    (project / "slab.toml").write_text(f'[paths]\nengines = "{registry}"\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    assert find_registry_path() == registry
    (project / "slab.toml").write_text('[paths]\nengines = "/nowhere/engines.json"\n')
    with pytest.raises(FileNotFoundError, match=r"\[paths\] engines"):
        find_registry_path()


def test_qe_resolution_uses_config_between_options_and_ase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab.backends import _qe_locator

    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(
        f'[engines.qe]\ncommand = "config-pw.x"\npseudo_dir = "{tmp_path}/config-pseudos"\n'
    )
    monkeypatch.chdir(project)
    monkeypatch.setattr("slab.backends._qe_configured", lambda key: f"ase-{key}")
    command, pseudo_dir = _qe_locator({})
    assert command == "config-pw.x"  # slab config beats ASE config
    assert pseudo_dir == f"{tmp_path}/config-pseudos"
    command, pseudo_dir = _qe_locator({"command": "explicit-pw.x"})
    assert command == "explicit-pw.x"  # explicit option beats slab config


def test_qe_ase_config_still_reached_below_slab_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab.backends import _qe_locator

    monkeypatch.chdir(tmp_path)  # no slab.toml anywhere
    monkeypatch.setattr("slab.backends._qe_configured", lambda key: f"ase-{key}")
    command, pseudo_dir = _qe_locator({})
    assert command == "ase-command"
    assert pseudo_dir == "ase-pseudo_dir"


def test_qe_factory_builds_profile_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab.backends import close_calculator, get_calculator

    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text(
        f'[engines.qe]\ncommand = "/bin/echo"\npseudo_dir = "{pseudo_dir}"\n'
    )
    monkeypatch.chdir(project)
    calc = get_calculator("qe", input_data={"system": {"ecutwfc": 30.0}})
    try:
        assert str(calc.profile.command) == "/bin/echo"
        assert calc.profile.pseudo_dir == str(pseudo_dir)
    finally:
        close_calculator(calc)


# -- CLI ---------------------------------------------------------------------


def test_cli_config_show_reports_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "no config files found" in result.output


def test_cli_config_show_renders_values_with_origins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text('[hpc]\ncluster = "delta"\n')
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "project:" in result.output
    assert "hpc.cluster = 'delta'" in result.output
    assert "(project)" in result.output


def test_cli_config_show_json_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text('[hpc]\ncluster = "delta"\n')
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["config"]["hpc"]["cluster"] == "delta"
    assert payload["origins"]["hpc.cluster"].endswith("(project)")


def test_cli_config_show_fails_loud_on_broken_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text('[agent]\nmodle = "typo"\n')
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 1
    assert "agent.modle" in result.output


def test_cli_config_init_writes_and_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert (tmp_path / "slab.toml").exists()
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 1
    assert "already exists" in result.output
    result = runner.invoke(app, ["config", "init", "--force"])
    assert result.exit_code == 0


def test_cli_config_init_user_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    result = runner.invoke(app, ["config", "init", "--user"])
    assert result.exit_code == 0
    assert (tmp_path / "xdg" / "slab" / "config.toml").exists()


def test_cli_broken_config_fails_workspace_commands_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text("[paths\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "not valid TOML" in result.output
