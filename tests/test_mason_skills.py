"""Skills: the Agent Skills format, three discovery layers, one loading tool."""

import json
from pathlib import Path

import pytest

from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.session import MasonSession
from mason.skills import (
    SkillError,
    catalog_block,
    discover_skills,
    listing,
    parse_skill,
)
from mason.tools import build_toolbox
from slab.config import HpcConfig


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "Does a thing. Use when the thing is asked for.",
    extra_frontmatter: str = "",
    body: str = "# Steps\n\nDo the thing.\n",
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra_frontmatter}---\n{body}",
        encoding="utf-8",
    )
    return directory


def _session(tmp_path: Path) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake"}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), auto_approve=True,
    )


# -- the format ---------------------------------------------------------------


def test_a_minimal_skill_parses(tmp_path: Path) -> None:
    directory = _write_skill(tmp_path, "surface-energy")
    skill = parse_skill(directory, "project")
    assert skill.name == "surface-energy"
    assert skill.agents is None  # no metadata: visible to every agent
    assert skill.source == "project"
    assert not skill.ignored_allowed_tools


@pytest.mark.parametrize(
    "bad", ["PDF-Processing", "-eos", "eos-", "eos--fit", "eos_fit", "x" * 65]
)
def test_names_breaking_the_spec_rules_are_refused(tmp_path: Path, bad: str) -> None:
    directory = tmp_path / bad
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        f"---\nname: {bad}\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillError, match=r"SKILL\.md"):
        parse_skill(directory, "project")


def test_the_name_must_equal_the_directory_name(tmp_path: Path) -> None:
    directory = tmp_path / "one-name"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(SkillError, match="must equal the directory name"):
        parse_skill(directory, "project")


def test_missing_description_and_oversized_fields_are_refused(tmp_path: Path) -> None:
    directory = tmp_path / "no-description"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: no-description\n---\nbody\n")
    with pytest.raises(SkillError, match="required 'description'"):
        parse_skill(directory, "project")

    with pytest.raises(SkillError, match="'description' exceeds 1024"):
        parse_skill(
            _write_skill(tmp_path, "long-description", description="x" * 1025), "project"
        )
    with pytest.raises(SkillError, match="'compatibility' exceeds 500"):
        parse_skill(
            _write_skill(
                tmp_path, "long-compat", extra_frontmatter=f"compatibility: {'y' * 501}\n"
            ),
            "project",
        )


def test_metadata_must_map_strings_to_strings(tmp_path: Path) -> None:
    directory = _write_skill(
        tmp_path, "bad-metadata", extra_frontmatter="metadata:\n  version: 3\n"
    )
    with pytest.raises(SkillError, match=r"metadata\.version"):
        parse_skill(directory, "project")


def test_mason_agents_categorizes_and_bad_entries_are_refused(tmp_path: Path) -> None:
    directory = _write_skill(
        tmp_path,
        "dft-only",
        extra_frontmatter='metadata:\n  mason-agents: "dft-expert analysis-expert"\n',
    )
    skill = parse_skill(directory, "project")
    assert skill.agents == frozenset({"dft-expert", "analysis-expert"})
    assert skill.visible_to("dft-expert")
    assert not skill.visible_to("md-expert")

    broken = _write_skill(
        tmp_path, "bad-agents", extra_frontmatter='metadata:\n  mason-agents: "DFT Expert"\n'
    )
    with pytest.raises(SkillError, match="not a valid agent name"):
        parse_skill(broken, "project")


def test_allowed_tools_is_accepted_ignored_and_flagged(tmp_path: Path) -> None:
    """The experimental spec field must not break loading; the gate already
    governs approval, so the field is recorded as ignored instead."""
    directory = _write_skill(
        tmp_path, "with-allowed", extra_frontmatter='allowed-tools: "Bash(git:*) Read"\n'
    )
    skill = parse_skill(directory, "project")
    assert skill.ignored_allowed_tools


def test_unknown_frontmatter_keys_are_tolerated(tmp_path: Path) -> None:
    """Skills written for other Agent Skills consumers load unmodified."""
    directory = _write_skill(tmp_path, "foreign", extra_frontmatter="version: '2'\n")
    assert parse_skill(directory, "project").name == "foreign"


def test_frontmatter_that_is_not_yaml_or_missing_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "no-frontmatter"
    directory.mkdir()
    (directory / "SKILL.md").write_text("# Just markdown\n")
    with pytest.raises(SkillError, match="must start with '---'"):
        parse_skill(directory, "project")


# -- discovery ----------------------------------------------------------------


def test_builtin_skills_are_discovered(tmp_path: Path) -> None:
    skills = discover_skills(tmp_path)
    assert "equation-of-state" in skills
    eos = skills["equation-of-state"]
    assert eos.source == "built-in"
    assert (eos.root / "scripts" / "fit_eos.py").is_file()


def test_project_shadows_user_shadows_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    project = tmp_path / "project"
    project.mkdir()

    _write_skill(xdg / "slab" / "skills", "local-lore", description="User-level lore.")
    _write_skill(
        xdg / "slab" / "skills", "equation-of-state", description="User replacement."
    )
    skills = discover_skills(project)
    assert skills["local-lore"].source == "user"
    assert skills["equation-of-state"].source == "user"
    assert skills["equation-of-state"].description == "User replacement."

    _write_skill(project / "skills", "local-lore", description="Project lore.")
    skills = discover_skills(project)
    assert skills["local-lore"].source == "project"
    assert skills["local-lore"].description == "Project lore."


def test_a_malformed_project_skill_fails_loudly_naming_the_file(tmp_path: Path) -> None:
    directory = tmp_path / "skills" / "broken"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\ndescription: no name\n---\nbody\n")
    with pytest.raises(SkillError) as excinfo:
        discover_skills(tmp_path)
    assert str(directory / "SKILL.md") in str(excinfo.value)


def test_a_skill_directory_without_a_manifest_is_refused(tmp_path: Path) -> None:
    (tmp_path / "skills" / "empty-dir").mkdir(parents=True)
    with pytest.raises(SkillError, match=r"no SKILL\.md"):
        discover_skills(tmp_path)


def test_hidden_and_underscore_directories_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "skills" / ".git").mkdir(parents=True)
    (tmp_path / "skills" / "__pycache__").mkdir()
    (tmp_path / "skills" / "notes.md").write_text("not a skill\n")
    assert "notes" not in discover_skills(tmp_path)


# -- the catalog and the tool -------------------------------------------------


def test_catalog_block_lists_names_and_descriptions(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "aaa-skill", description="First thing.")
    skills = discover_skills(tmp_path)
    block = catalog_block(skills)
    assert block.startswith("# Skills")
    assert "- aaa-skill: First thing." in block
    assert "- equation-of-state:" in block
    assert catalog_block({}) == ""


def test_the_system_prompt_carries_the_skill_catalog(tmp_path: Path) -> None:
    from mason.prompts import system_messages

    session = _session(tmp_path)
    (content,) = [m["content"] for m in system_messages(session, skills=discover_skills(tmp_path))]
    assert "# Skills" in content
    assert "- equation-of-state:" in content
    (plain,) = [m["content"] for m in system_messages(session)]
    assert "# Skills" not in plain


def box_skill_digest(session: MasonSession) -> str:
    return discover_skills(session.cwd)["equation-of-state"].digest


def test_skill_tool_returns_body_root_and_files(tmp_path: Path) -> None:
    session = _session(tmp_path)
    box = build_toolbox(session)
    call = ToolCall(
        id="c1", name="skill", arguments={"name": "equation-of-state"},
        arguments_raw='{"name": "equation-of-state"}',
    )
    answer = box.dispatch(call)
    assert "# Equation of state" in answer
    assert "skill root: " in answer
    assert "scripts/fit_eos.py" in answer
    assert "assets/eos_scan.py" in answer
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    assert {"type": "skill", "name": "equation-of-state", "source": "built-in"}.items() <= (
        events[-1].items()
    )
    # The revision that loaded, so a benchmark flag can be pinned to it.
    assert events[-1]["digest"] == box_skill_digest(session)


def test_unknown_skill_answers_with_the_available_names(tmp_path: Path) -> None:
    box = build_toolbox(_session(tmp_path))
    call = ToolCall(id="c1", name="skill", arguments={"name": "nope"}, arguments_raw="{}")
    answer = box.dispatch(call)
    assert "no skill named 'nope'" in answer
    assert "equation-of-state" in answer


def test_no_skills_means_no_skill_tool(tmp_path: Path) -> None:
    box = build_toolbox(_session(tmp_path), skills={})
    assert "skill" not in box.tools


def test_long_skill_bodies_pass_through_output_truncation(tmp_path: Path) -> None:
    _write_skill(tmp_path / "skills", "huge-skill", body="# Huge\n" + "line\n" * 20_000)
    config = MasonConfig.model_validate(
        {"agent": {"model": "fake", "max_tool_output_chars": 2_000}}
    )
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent,
        hpc=HpcConfig(), auto_approve=True,
    )
    box = build_toolbox(session)
    call = ToolCall(id="c1", name="skill", arguments={"name": "huge-skill"}, arguments_raw="{}")
    answer = box.dispatch(call)
    assert "characters truncated" in answer
    assert len(answer) < 3_000


def test_listing_caps_the_file_inventory(tmp_path: Path) -> None:
    directory = _write_skill(tmp_path / "skills", "many-files")
    for i in range(60):
        (directory / "references" / f"ref-{i:02d}.md").parent.mkdir(exist_ok=True)
        (directory / "references" / f"ref-{i:02d}.md").write_text("x\n")
    text = listing(parse_skill(directory, "project"))
    assert "more files not listed" in text


# -- end to end through the loop ----------------------------------------------


class _FakeClient:
    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, object]]] = []

    def chat(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]] | None = None
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)


def test_a_scripted_turn_loads_a_skill_and_reads_its_script(tmp_path: Path) -> None:
    """The two-step the prompt teaches: load the skill, then use its files."""
    session = _session(tmp_path)
    skills = discover_skills(tmp_path)
    script_path = skills["equation-of-state"].root / "scripts" / "fit_eos.py"
    client = _FakeClient(
        [
            ChatReply(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="c1", name="skill",
                        arguments={"name": "equation-of-state"}, arguments_raw="{}",
                    ),
                ),
                prompt_tokens=10, completion_tokens=1,
            ),
            ChatReply(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="c2", name="read_file",
                        arguments={"path": str(script_path)}, arguments_raw="{}",
                    ),
                ),
                prompt_tokens=10, completion_tokens=1,
            ),
            ChatReply(content="the fit script exists", prompt_tokens=10, completion_tokens=1),
        ]
    )
    mason = Mason(session, client=client)
    result = mason.run_turn("prepare an EOS fit")
    assert result.stop_reason == "answer"
    skill_result = next(
        m for m in client.requests[1] if m.get("role") == "tool" and m.get("tool_call_id") == "c1"
    )
    assert "Birch-Murnaghan" in str(skill_result["content"])
    read_result = next(
        m for m in client.requests[2] if m.get("role") == "tool" and m.get("tool_call_id") == "c2"
    )
    assert "EquationOfState" in str(read_result["content"])


def test_an_unreadable_manifest_is_an_error_line_not_a_traceback(tmp_path: Path) -> None:
    """The CLI catches SkillError; a permissions problem must arrive as one."""
    directory = _write_skill(tmp_path / "skills", "locked-away")
    (directory / "SKILL.md").chmod(0o000)
    try:
        with pytest.raises(SkillError, match="cannot read it"):
            discover_skills(tmp_path)
    finally:
        (directory / "SKILL.md").chmod(0o644)
