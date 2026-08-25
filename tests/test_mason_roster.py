"""The roster: agent cards, per-agent config overrides, and the pi default."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mason.cli import app
from mason.client import ChatReply
from mason.config import AgentConfig, MasonConfig, roster_agent_config
from mason.loop import Mason
from mason.roster import (
    RosterError,
    check_overrides,
    discover_roster,
    parse_agent_card,
    skills_for,
)
from mason.session import MasonSession
from mason.skills import discover_skills
from mason.tools import build_toolbox
from slab.config import ConfigError, HpcConfig

runner = CliRunner()


def _write_card(root: Path, name: str, *, frontmatter: str = "", body: str = "A role.\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: Does {name} things.\n{frontmatter}---\n{body}",
        encoding="utf-8",
    )
    return path


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), auto_approve=True,
    )


class _Idle:
    """A backend that must never be called; construction-time tests only."""

    def chat(self, messages: object, tools: object = None) -> ChatReply:
        raise AssertionError("the model was called during construction")


# -- the built-in roster ------------------------------------------------------


def test_builtin_roster_holds_the_four_cards(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    assert {"pi", "dft-expert", "md-expert", "analysis-expert"} <= set(roster)
    assert roster["pi"].delegates
    assert roster["pi"].skills_scope == "all"
    assert not roster["dft-expert"].delegates
    assert all(spec.source == "built-in" for spec in roster.values())


def test_analysis_expert_cannot_launch_new_physics(tmp_path: Path) -> None:
    """The card's allowlist is doctrine in force, not prose."""
    spec = discover_roster(tmp_path)["analysis-expert"]
    assert spec.tools is not None
    assert "launch_workflow" not in spec.tools
    box = build_toolbox(_session(tmp_path), spec)
    assert "launch_workflow" not in box.tools
    assert "show_run" in box.tools
    assert "finish" in box.tools


def test_eos_skill_maps_to_its_specialists(tmp_path: Path) -> None:
    roster = discover_roster(tmp_path)
    skills = discover_skills(tmp_path)
    assert "equation-of-state" in skills_for(roster["dft-expert"], skills)
    assert "equation-of-state" in skills_for(roster["analysis-expert"], skills)
    assert "equation-of-state" not in skills_for(roster["md-expert"], skills)
    # pi's scope is 'all': solo mode loses nothing to categorization.
    assert "equation-of-state" in skills_for(roster["pi"], skills)


# -- card validation ----------------------------------------------------------


def test_a_card_with_unknown_keys_is_refused(tmp_path: Path) -> None:
    path = _write_card(tmp_path / "agents", "typo-card", frontmatter="tool: shell\n")
    with pytest.raises(RosterError, match="unknown frontmatter key") as excinfo:
        parse_agent_card(path, "project")
    assert str(path) in str(excinfo.value)


def test_a_card_tools_typo_is_refused_naming_the_vocabulary(tmp_path: Path) -> None:
    path = _write_card(tmp_path / "agents", "shel-card", frontmatter="tools: shel finish\n")
    with pytest.raises(RosterError, match="'shel'") as excinfo:
        parse_agent_card(path, "project")
    assert "launch_workflow" in str(excinfo.value)  # the vocabulary is named


def test_a_card_naming_a_conditional_tool_is_valid(tmp_path: Path) -> None:
    """submit_job is vocabulary even on a laptop; the session just lacks it."""
    path = _write_card(tmp_path / "agents", "hpc-card", frontmatter="tools: submit_job finish\n")
    spec = parse_agent_card(path, "project")
    box = build_toolbox(_session(tmp_path), spec)  # no partitions here
    assert "submit_job" not in box.tools
    assert set(box.tools) == {"finish"}


def test_card_name_must_match_file_and_naming_rules(tmp_path: Path) -> None:
    path = (tmp_path / "agents")
    path.mkdir()
    (path / "other.md").write_text("---\nname: not-other\ndescription: d\n---\nrole\n")
    with pytest.raises(RosterError, match="must equal the file name"):
        parse_agent_card(path / "other.md", "project")


def test_card_without_a_body_is_refused(tmp_path: Path) -> None:
    path = _write_card(tmp_path / "agents", "empty-role", body="\n")
    with pytest.raises(RosterError, match="no body"):
        parse_agent_card(path, "project")


def test_bad_skills_scope_and_bad_delegates_are_refused(tmp_path: Path) -> None:
    path = _write_card(tmp_path / "agents", "bad-scope", frontmatter="skills: some\n")
    with pytest.raises(RosterError, match="'matching' or 'all'"):
        parse_agent_card(path, "project")
    path = _write_card(tmp_path / "agents", "bad-flag", frontmatter="delegates: maybe\n")
    with pytest.raises(RosterError, match="true or false"):
        parse_agent_card(path, "project")


def test_a_project_card_shadows_the_builtin_pi(tmp_path: Path) -> None:
    _write_card(tmp_path / "agents", "pi", body="You are a replacement PI.\n")
    roster = discover_roster(tmp_path)
    assert roster["pi"].source == "project"
    assert roster["pi"].prompt == "You are a replacement PI."
    assert not roster["pi"].delegates  # the replacement's own frontmatter rules


def test_agents_md_next_to_cards_is_not_a_card(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "AGENTS.md").write_text("conventions, not a card\n")
    assert "agents" not in discover_roster(tmp_path)


# -- config overrides ---------------------------------------------------------


def test_roster_override_typo_is_refused_naming_the_key(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "m"\n\n[agent.roster.dft-expert]\nmodle = "typo"\n'
    )
    from mason.config import load_config

    with pytest.raises(ConfigError, match=r"agent\.roster\.dft-expert\.modle"):
        load_config(tmp_path)


def test_roster_override_merges_under_the_base(tmp_path: Path) -> None:
    agent = AgentConfig.model_validate(
        {
            "model": "qwen3:30b",
            "temperature": 0.4,
            "roster": {"dft-expert": {"temperature": 0.0, "max_turns": 20}},
        }
    )
    effective = roster_agent_config(agent, "dft-expert")
    assert effective.temperature == 0.0
    assert effective.max_turns == 20
    assert effective.model == "qwen3:30b"  # unset fields inherit
    assert roster_agent_config(agent, "md-expert") is agent


def test_provider_override_clears_a_stale_endpoint() -> None:
    """A vLLM URL must not survive a per-agent switch to the Anthropic API."""
    agent = AgentConfig.model_validate(
        {
            "model": "local",
            "endpoint": "http://gpu-node:8000/v1",
            "roster": {"pi": {"provider": "anthropic", "model": "claude-opus-5"}},
        }
    )
    effective = roster_agent_config(agent, "pi")
    assert effective.resolved_endpoint == "https://api.anthropic.com/v1"


def test_overrides_for_unknown_agents_are_refused(tmp_path: Path) -> None:
    agent = AgentConfig.model_validate({"roster": {"dft-exprt": {"model": "m"}}})
    with pytest.raises(RosterError, match=r"\[agent\.roster\.dft-exprt\]") as excinfo:
        check_overrides(agent, discover_roster(tmp_path))
    assert "dft-expert" in str(excinfo.value)  # the real roster is named


# -- the loop resolves the card -----------------------------------------------


def test_mason_defaults_to_the_pi_card(tmp_path: Path) -> None:
    mason = Mason(_session(tmp_path), client=_Idle())
    assert mason.spec.name == "pi"
    assert mason.session.agent_name == "pi"
    (system,) = mason.messages
    content = system["content"]
    assert content.index("You are Mason") < content.index("# How you work")
    # The delegate tool exists for the pi, so the team block is promised.
    assert "delegate" in mason.toolbox.tools
    assert "# Your team" in content
    assert "- dft-expert:" in content


def test_specialist_prompt_carries_its_role_and_filtered_skills(tmp_path: Path) -> None:
    session = _session(tmp_path)
    roster = discover_roster(tmp_path)
    mason = Mason(session, client=_Idle(), spec=roster["md-expert"], roster=roster)
    (system,) = mason.messages
    assert "molecular-dynamics specialist" in system["content"]
    assert "- equation-of-state:" not in system["content"]  # not md's skill
    assert session.agent_name == "md-expert"


def test_entry_agent_honors_its_roster_table(tmp_path: Path) -> None:
    session = _session(tmp_path, roster={"pi": {"temperature": 0.0, "max_turns": 7}})
    mason = Mason(session, client=_Idle())
    assert mason.session.agent.temperature == 0.0
    assert mason.session.agent.max_turns == 7


def test_cli_flags_stay_on_top_of_the_roster_table(tmp_path: Path) -> None:
    session = _session(tmp_path, roster={"pi": {"model": "from-config"}})
    session.flag_updates = {"model": "from-flag"}
    mason = Mason(session, client=_Idle())
    assert mason.session.agent.model == "from-flag"


def test_unknown_roster_table_fails_at_the_loop(tmp_path: Path) -> None:
    session = _session(tmp_path, roster={"nobody": {"model": "m"}})
    with pytest.raises(RosterError, match=r"\[agent\.roster\.nobody\]"):
        Mason(session, client=_Idle())


# -- CLI ----------------------------------------------------------------------


def test_cli_roster_lists_the_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        '[agent]\nmodel = "qwen3:30b"\n\n[agent.roster.pi]\nmodel = "claude-opus-5"\n'
    )
    result = runner.invoke(app, ["roster"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines[0].startswith("pi ")
    assert "claude-opus-5" in lines[0]
    assert "[delegates]" in lines[0]
    assert any(line.startswith("dft-expert") and "qwen3:30b" in line for line in lines)


def test_cli_roster_refuses_an_unknown_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text('[agent.roster.nobody]\nmodel = "m"\n')
    result = runner.invoke(app, ["roster"])
    assert result.exit_code == 1
    assert "[agent.roster.nobody]" in result.output


def test_cli_skills_lists_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["skills"])
    assert result.exit_code == 0
    assert "equation-of-state" in result.output
    assert "built-in" in result.output
    filtered = runner.invoke(app, ["skills", "--agent", "md-expert"])
    assert filtered.exit_code == 0
    assert "equation-of-state" not in filtered.output


def test_cli_unknown_agent_lists_the_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "hi", "--agent", "dft-exprt"])
    assert result.exit_code == 1
    assert "no agent named 'dft-exprt'" in result.output
    assert "dft-expert" in result.output


def test_a_readme_beside_cards_is_not_a_card(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "README.md").write_text("about these cards\n")
    roster = discover_roster(tmp_path)
    assert "readme" not in roster and "README" not in roster
