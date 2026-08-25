"""Configuration as Mason sees it: the ``[agent]`` tables it owns."""

from pathlib import Path

import pytest

from mason.config import AgentConfig, MasonConfig, ServeConfig, load_config
from slab.config import ConfigError
from slab.config import load_config as load_slab_config

# -- the tables Mason owns ----------------------------------------------------


def test_no_files_yields_pure_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config == MasonConfig()
    assert config.agent.resolved_endpoint == "http://localhost:11434/v1"


def test_agent_table_is_read_from_the_shared_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "llama3.1:8b"\nmax_turns = 40\n\n'
        '[agent.serve]\npartition = "gpu"\nport = 8123\n'
    )
    agent = load_config(tmp_path).agent
    assert agent.model == "llama3.1:8b"
    assert agent.max_turns == 40
    assert agent.serve.partition == "gpu"
    assert agent.serve.port == 8123


def test_a_typo_inside_agent_is_refused_when_mason_loads(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[agent]\nmodle = "typo"\n')
    with pytest.raises(ConfigError, match=r"agent\.modle") as excinfo:
        load_config(tmp_path)
    assert "slab.toml" in str(excinfo.value)
    assert "unknown key" in str(excinfo.value)


def test_the_same_typo_is_invisible_to_slab(tmp_path: Path) -> None:
    """Each package validates its own tables, so the refusal has one owner."""
    (tmp_path / "slab.toml").write_text('[agent]\nmodle = "typo"\n')
    config = load_slab_config(tmp_path)  # no raise
    assert not hasattr(config, "agent")


def test_a_wrong_type_inside_agent_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[agent]\nmax_turns = "many"\n')
    with pytest.raises(ConfigError, match=r"agent\.max_turns"):
        load_config(tmp_path)


def test_tables_mason_does_not_own_are_ignored_here(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n\n[hpc]\ncluster = "delta"\n\n[workspace]\nroot = "/ws"\n'
    )
    config = load_config(tmp_path)
    assert config.agent.model == "m"
    assert not hasattr(config, "hpc")


# -- serve shell is never expanded at load ------------------------------------


def test_serve_command_and_setup_keep_their_variables_literal(tmp_path: Path) -> None:
    """That shell runs on the GPU node; expanding it here would resolve it
    against the login node's environment instead."""
    (tmp_path / "slab.toml").write_text(
        "[agent.serve]\n"
        'setup = ["module load cuda/$CUDA_VERSION"]\n'
        'command = "vllm serve $MODEL --port $port"\n'
    )
    serve = load_config(tmp_path).agent.serve
    assert serve.setup == ("module load cuda/$CUDA_VERSION",)
    assert serve.command == "vllm serve $MODEL --port $port"


# -- provider defaults --------------------------------------------------------


def test_anthropic_provider_defaults_its_endpoint_and_key_variable() -> None:
    agent = AgentConfig.model_validate({"provider": "anthropic", "model": "claude-opus-5"})
    assert agent.resolved_endpoint == "https://api.anthropic.com/v1"
    assert agent.resolved_api_key_env == "ANTHROPIC_API_KEY"


def test_openai_provider_defaults_to_ollama_and_needs_no_key() -> None:
    agent = AgentConfig.model_validate({"provider": "openai"})
    assert agent.resolved_endpoint == "http://localhost:11434/v1"
    assert agent.resolved_api_key_env is None


def test_agent_models_forbid_unknown_keys() -> None:
    with pytest.raises(Exception, match="modle"):
        AgentConfig.model_validate({"modle": "typo"})
    with pytest.raises(Exception, match="prot"):
        ServeConfig.model_validate({"prot": 8000})
