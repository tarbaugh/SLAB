"""Tool contract tests: file primitives, gating, truncation, SLAB integration."""

import contextlib
import json
import os
from pathlib import Path

import pytest

from mason.client import ToolCall
from mason.config import MasonConfig
from mason.session import MasonSession
from mason.tools import Toolbox, _truncate_middle, build_toolbox
from slab.config import HpcConfig


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": agent} if agent else {})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, auto_approve=True
    )


def _call(tool: str, /, **arguments: object) -> ToolCall:
    import json

    return ToolCall(
        id="t1", name=tool, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )


@pytest.fixture()
def box(tmp_path: Path) -> Toolbox:
    return build_toolbox(_session(tmp_path))


def _command_events(session: MasonSession) -> list[dict[str, object]]:
    """The command events a session's transcript holds, in order."""
    if not session.transcript_path.exists():
        return []
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    return [event for event in events if event.get("type") == "command"]


COMMAND_WORKFLOW = """\
from foundation import task

IDENTITY = {"engine": "lammps", "command": "mpirun -np 1 lmp -k on g 1 -sf kk",
            "setup": ["module load lammps"], "version": "22 Jul 2025"}

@task(cache_extra=lambda arguments: IDENTITY)
def probe(x):
    return x

probe(1)
probe(2)
print("probed")
"""


# -- dispatch plumbing -------------------------------------------------------


def test_unknown_tool_lists_the_known(box: Toolbox) -> None:
    answer = box.dispatch(_call("teleport"))
    assert "unknown tool 'teleport'" in answer
    assert "read_file" in answer


def test_malformed_arguments_are_reported_not_run(box: Toolbox) -> None:
    call = ToolCall(
        id="t1", name="shell", arguments={}, arguments_raw="{", arguments_error="bad JSON"
    )
    assert box.dispatch(call) == "tool shell not run: bad JSON"


def test_missing_required_arguments_teach_the_schema(box: Toolbox) -> None:
    answer = box.dispatch(_call("read_file"))  # missing required 'path'
    assert "missing required argument(s) path" in answer
    assert "required: path" in answer and "optional: offset, limit" in answer
    answer = box.dispatch(_call("launch_workflow", intent="x"))
    assert "missing required argument(s) script" in answer


def test_handler_exception_becomes_evidence(box: Toolbox) -> None:
    answer = box.dispatch(_call("shell", command="true", timeout_s="soonish"))
    # A crashing handler is evidence, not a dead loop:
    assert answer.startswith("tool shell failed: ValueError")


def test_python_writes_get_an_immediate_syntax_check(box: Toolbox, tmp_path: Path) -> None:
    answer = box.dispatch(
        _call("write_file", path="broken.py", content="from ase import\\ndef f(:")
    )
    assert "WARNING: the file does not parse as Python" in answer
    answer = box.dispatch(_call("write_file", path="fine.py", content="x = 1\n"))
    assert "WARNING" not in answer
    box.dispatch(_call("read_file", path="fine.py"))
    answer = box.dispatch(_call("edit_file", path="fine.py", old_string="x = 1", new_string="x ="))
    assert "WARNING: the file does not parse as Python" in answer


def test_hpc_tools_only_exist_with_partitions(tmp_path: Path) -> None:
    plain = build_toolbox(_session(tmp_path))
    assert "submit_job" not in plain.tools
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    session = MasonSession(tmp_path, workspace_root=tmp_path / ".slab", hpc=hpc)
    clustered = build_toolbox(session)
    assert {"submit_job", "job_status", "cancel_job"} <= set(clustered.tools)


def test_specs_and_catalog_render_every_tool(box: Toolbox) -> None:
    specs = box.specs()
    assert all(spec["type"] == "function" for spec in specs)
    names = {spec["function"]["name"] for spec in specs}
    assert {"read_file", "edit_file", "shell", "launch_workflow", "finish"} <= names
    catalog = box.catalog_text()
    assert "- read_file(path: string, offset?: integer, limit?: integer, raw?: boolean)" in catalog


# -- file primitives ---------------------------------------------------------


def test_read_file_numbers_lines_and_windows(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("alpha\nbeta\ngamma\n")
    answer = box.dispatch(_call("read_file", path="data.txt", offset=2, limit=1))
    assert "     2\tbeta" in answer
    assert "alpha" not in answer
    assert "[file has 3 lines; showing 2-2]" in answer


def test_read_file_refuses_binary(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    assert "looks binary" in box.dispatch(_call("read_file", path="blob.bin"))


def test_edit_requires_read_first_then_unique_match(box: Toolbox, tmp_path: Path) -> None:
    target = tmp_path / "code.py"
    target.write_text("x = 1\ny = 1\n")
    answer = box.dispatch(_call("edit_file", path="code.py", old_string="1", new_string="2"))
    assert "staleness guard" in answer
    box.dispatch(_call("read_file", path="code.py"))
    answer = box.dispatch(_call("edit_file", path="code.py", old_string="1", new_string="2"))
    assert "matches 2 places" in answer
    answer = box.dispatch(
        _call("edit_file", path="code.py", old_string="x = 1", new_string="x = 2")
    )
    assert "replaced 1 occurrence" in answer
    assert target.read_text() == "x = 2\ny = 1\n"
    target.write_text("a = 1\nb = 1\n")
    box.dispatch(_call("read_file", path="code.py"))
    answer = box.dispatch(
        _call("edit_file", path="code.py", old_string="1", new_string="3", replace_all=True)
    )
    assert "replaced 2 occurrence" in answer
    assert target.read_text() == "a = 3\nb = 3\n"


def test_edit_no_match_teaches_about_line_numbers(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("content\n")
    box.dispatch(_call("read_file", path="a.txt"))
    answer = box.dispatch(
        _call("edit_file", path="a.txt", old_string="  1\tcontent", new_string="x")
    )
    assert "line numbers" in answer


def test_write_file_creates_parents(box: Toolbox, tmp_path: Path) -> None:
    answer = box.dispatch(_call("write_file", path="deep/dir/new.txt", content="hello"))
    assert "wrote 5 characters" in answer
    assert (tmp_path / "deep" / "dir" / "new.txt").read_text() == "hello"


def test_list_dir_marks_directories(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")
    answer = box.dispatch(_call("list_dir"))
    assert "sub/" in answer
    assert "file.txt  (1 B)" in answer


def test_search_works_when_the_project_lives_under_a_dotted_parent(tmp_path: Path) -> None:
    """Hidden-dir filtering must apply below the search root only."""
    project = tmp_path / ".research" / "proj"
    project.mkdir(parents=True)
    (project / "a.py").write_text("def relax():\n    pass\n")
    box = build_toolbox(_session_at(project))
    answer = box.dispatch(_call("search", pattern=r"def relax"))
    assert "a.py:1: def relax():" in answer


def _session_at(path: Path) -> MasonSession:
    return MasonSession(
        path, workspace_root=path / ".slab", agent=MasonConfig().agent, auto_approve=True
    )


def test_list_dir_survives_a_dangling_symlink(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("x")
    (tmp_path / "gone").symlink_to(tmp_path / "no-such-target")
    answer = box.dispatch(_call("list_dir"))
    assert "good.txt" in answer
    assert "gone" in answer and "unreadable" in answer


def test_approval_preview_names_the_load_bearing_keys(tmp_path: Path) -> None:
    previews: list[str] = []

    def approver(tool: str, preview: str) -> bool:
        previews.append(preview)
        return False

    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=MasonConfig().agent, approver=approver
    )
    box = build_toolbox(session)
    box.dispatch(_call("write_file", content="x" * 5_000, path="important.py"))
    assert "path='important.py'" in previews[-1]  # the giant content cannot hide the path


def test_search_finds_and_skips_hidden(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def relax():\n    pass\n")
    hidden = tmp_path / ".secret"
    hidden.mkdir()
    (hidden / "b.py").write_text("def relax():\n    pass\n")
    answer = box.dispatch(_call("search", pattern=r"def relax"))
    assert "a.py:1: def relax():" in answer
    assert ".secret" not in answer
    assert "bad regex" in box.dispatch(_call("search", pattern="("))


# -- shell -------------------------------------------------------------------


def test_shell_reports_exit_code_and_stderr(box: Toolbox) -> None:
    answer = box.dispatch(_call("shell", command="echo out; echo err >&2; exit 3"))
    assert answer.startswith("exit 3\n")
    assert "out" in answer and "[stderr]" in answer and "err" in answer


def test_shell_keeps_binary_output_as_evidence(box: Toolbox) -> None:
    """A stray byte on stdout once failed the whole call with a
    UnicodeDecodeError; the exit code and the readable part are the evidence."""
    answer = box.dispatch(_call("shell", command="printf 'head \\xd8\\xb4 tail'"))
    assert answer.startswith("exit 0\n")
    assert "head" in answer and "tail" in answer
    assert "UnicodeDecodeError" not in answer


def test_shell_timeout_returns_partial_evidence(box: Toolbox) -> None:
    answer = box.dispatch(_call("shell", command="echo started; sleep 5", timeout_s=0.2))
    assert "timed out after 0s" in answer or "timed out" in answer


def test_shell_approval_gate_and_allowlist(tmp_path: Path) -> None:
    asked: list[str] = []

    def approver(tool: str, preview: str) -> bool:
        asked.append(preview)
        return False

    config = MasonConfig.model_validate({"agent": {"shell_allowlist": ["echo"]}})
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, approver=approver
    )
    box = build_toolbox(session)
    assert box.dispatch(_call("shell", command="echo hi")).startswith("exit 0")  # allowlisted
    answer = box.dispatch(_call("shell", command="rm -rf /tmp/x"))
    assert "not approved" in answer
    assert asked == ["rm -rf /tmp/x"]
    # An allowlisted prefix must not smuggle chained or redirected commands:
    assert "not approved" in box.dispatch(_call("shell", command="echo hi; rm -rf /tmp/x"))
    assert "not approved" in box.dispatch(_call("shell", command="echo hi > /tmp/x"))
    assert "not approved" in box.dispatch(_call("shell", command="echoxx hi"))


def test_write_gated_when_not_auto(tmp_path: Path) -> None:
    session = MasonSession(tmp_path, workspace_root=tmp_path / ".slab", agent=MasonConfig().agent)
    box = build_toolbox(session)  # default approver refuses
    answer = box.dispatch(_call("write_file", path="x.txt", content="c"))
    assert "not approved" in answer
    assert not (tmp_path / "x.txt").exists()
    assert "alpha" not in box.dispatch(_call("read_file", path="x.txt"))  # read tools still work


# -- truncation --------------------------------------------------------------


def test_truncate_middle_keeps_head_and_tail(tmp_path: Path) -> None:
    lines = "\n".join(f"line-{i:04d} {'x' * 20}" for i in range(400))
    session = _session(tmp_path, max_tool_output_chars=1_000)
    box = build_toolbox(session)
    (tmp_path / "big.txt").write_text(lines)
    answer = box.dispatch(_call("read_file", path="big.txt", limit=400))
    assert answer.startswith("     1\tline-0000")  # head survives
    assert "characters truncated" in answer  # the drop is announced
    assert "line-0399" in answer  # tail survives
    assert len(answer) < 1_200


def test_truncate_middle_reports_exact_drop() -> None:
    out = _truncate_middle("a" * 300, 250)
    assert "characters truncated" in out
    assert out.startswith("a" * 10) and out.endswith("a" * 10)


# -- slab tools --------------------------------------------------------------


def test_list_runs_empty_then_launch_then_show(box: Toolbox, tmp_path: Path) -> None:
    assert box.dispatch(_call("list_runs")) == "no runs in this workspace yet"
    script = tmp_path / "wf.py"
    script.write_text(
        "from foundation import check, converged\n"
        "from foundation.tasks import relax\n"
        "from ase.build import bulk\n"
        "atoms = bulk('Cu', 'fcc', a=3.6)\n"
        "relaxed, info = relax(atoms, engine='emt', fmax=0.05, label='cu')\n"
        "print('energy (eV):', info['energy'])\n"
        "@check\n"
        "def forces_converged():\n"
        "    return converged(info['fmax'], below=0.05)\n"
    )
    answer = box.dispatch(_call("launch_workflow", script="wf.py", intent="mason test"))
    assert "state=verified" in answer
    assert "checks=1/1" in answer
    assert "energy (eV):" in answer  # script output captured
    run_line = box.dispatch(_call("list_runs"))
    assert "verified" in run_line and "wf" in run_line
    run_id = run_line.split()[0]
    details = box.dispatch(_call("show_run", run_id=run_id))
    assert '"intent": "mason test"' in details
    assert '"passed": true' in details


def test_list_runs_filters_by_session_id(tmp_path: Path) -> None:
    """The session filter mirrors the CLI and the MCP server: a full id or
    a unique prefix returns only that session's runs, no session shows all."""
    import os

    from foundation.runtime import Workspace

    session = _session(tmp_path)
    box = build_toolbox(session)
    with Workspace(session.workspace_root) as ws:
        with ws.start_run(name="other", intent="other chat", session="other-abc") as _:
            pass
        os.environ["SLAB_SESSION"] = session.session_id
        try:
            with ws.start_run(name="mine", intent="this chat") as _:
                pass
        finally:
            os.environ.pop("SLAB_SESSION", None)
    all_runs = box.dispatch(_call("list_runs"))
    assert "other" in all_runs and "mine" in all_runs
    filtered = box.dispatch(_call("list_runs", session=session.session_id))
    assert "mine" in filtered and "other" not in filtered
    # An unknown id raises loudly through the tool — the store's
    # SessionNotFoundError surfaces with the recovery hint intact.
    missing = box.dispatch(_call("list_runs", session="unknown-session-id"))
    assert "unknown-session-id" in missing and "slab sessions" in missing


def test_launch_workflow_failure_carries_the_record(box: Toolbox, tmp_path: Path) -> None:
    script = tmp_path / "boom.py"
    script.write_text("raise ValueError('SCF exploded')\n")
    answer = box.dispatch(_call("launch_workflow", script="boom.py"))
    assert "status=failed" in answer
    assert "failure record:" in answer
    assert "SCF exploded" in answer


def test_list_engines_reports_capabilities(box: Toolbox) -> None:
    answer = box.dispatch(_call("list_engines"))
    assert '"builtin"' in answer and '"qe"' in answer


def test_list_and_describe_task_expose_the_vocabulary(box: Toolbox) -> None:
    """The agent can ask what foundation.tasks offers without shelling into
    the source (which is what the 82-of-120 M0 transcript spent 38 steps on)."""
    listing = box.dispatch(_call("list_tasks"))
    lines = listing.splitlines()
    assert any(line.startswith("relax(atoms, ") for line in lines)
    assert any(line.startswith("single_point(atoms, ") for line in lines)
    import json as _json

    describe_relax = ToolCall(
        id="t", name="describe_task",
        arguments={"name": "relax"}, arguments_raw=_json.dumps({"name": "relax"}),
    )
    detail = box.dispatch(describe_relax)
    assert detail.startswith("relax(atoms, ")
    # The engine kwarg is required post-Part-A: no default in the signature.
    assert "engine, " in detail.splitlines()[0]
    assert "Positions only" in detail
    describe_missing = ToolCall(
        id="t", name="describe_task",
        arguments={"name": "not-a-task"}, arguments_raw=_json.dumps({"name": "not-a-task"}),
    )
    bad = box.dispatch(describe_missing)
    assert "no task 'not-a-task'" in bad and "relax" in bad


def test_read_file_can_reach_installed_slab_sources(box: Toolbox) -> None:
    """The file fence lets read_file inspect the four slab-stack packages
    so an agent can answer 'does foundation.tasks define relax_cell?' as
    one Read tool call, not a source-code shell expedition."""
    for module in (
        "foundation/tasks.py",
        "slab/backends.py",
        "mason/loop.py",
        "slab_stack/__init__.py",
    ):
        import importlib.resources

        pkg, _, tail = module.partition("/")
        path = Path(str(importlib.resources.files(pkg))) / tail
        answer = box.dispatch(_call("read_file", path=str(path)))
        assert "refused:" not in answer, f"{module}: {answer[:200]}"
    # Writes into installed sources stay refused (read scope only).
    from foundation import tasks

    tasks_path = Path(tasks.__file__)
    refused = box.dispatch(_call("write_file", path=str(tasks_path), content="broken\n"))
    assert refused.startswith("refused:")


# -- memory tools ------------------------------------------------------------


def test_notebook_appends_and_plan_recites(box: Toolbox, tmp_path: Path) -> None:
    answer = box.dispatch(_call("notebook", entry="a0 = 3.615 A (run abc123)", heading="Cu bulk"))
    assert "recorded in NOTEBOOK.md" in answer
    notebook = (tmp_path / "NOTEBOOK.md").read_text()
    assert notebook.startswith("# Lab notebook")
    assert "Cu bulk" in notebook and "run abc123" in notebook
    answer = box.dispatch(_call("plan", content="1. [done] relax Cu\n2. [next] EOS"))
    assert "PLAN.md updated:" in answer
    assert "[next] EOS" in answer  # recitation: the plan comes back into context
    assert (tmp_path / "PLAN.md").read_text() == "1. [done] relax Cu\n2. [next] EOS\n"


def test_finish_echoes_the_report(box: Toolbox) -> None:
    assert box.dispatch(_call("finish", report="done: a0=3.615 A (run abc)")).startswith("done:")


def test_crashing_approver_is_a_refusal_not_a_crash(tmp_path: Path) -> None:
    """dispatch never raises — an approver dying on closed stdin included."""

    def broken(tool: str, preview: str) -> bool:
        raise RuntimeError("stdin exploded")

    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=MasonConfig().agent, approver=broken
    )
    box = build_toolbox(session)
    answer = box.dispatch(_call("write_file", path="x.txt", content="c"))
    assert "not approved" in answer
    assert not (tmp_path / "x.txt").exists()


def test_preview_shows_enough_content_to_review(tmp_path: Path) -> None:
    """Approving a workflow script means reading it: head AND tail survive."""
    from mason.tools import _preview

    body = "\n".join(f"line {i}" for i in range(400))
    preview = _preview(_call("write_file", path="wf.py", content=body))
    assert "wf.py" in preview
    assert "line 0" in preview
    assert "line 399" in preview


def test_shell_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    """Killing only /bin/sh would leave backgrounded children running while
    the model reads 'timed out' as the command being gone."""
    import time

    box = build_toolbox(_session(tmp_path, shell_timeout_s=60.0))
    marker = tmp_path / "orphan-survived"
    command = f"(sleep 2 && touch {marker}) & sleep 30"
    started = time.monotonic()
    result = box.dispatch(_call("shell", command=command, timeout_s=1.0))
    assert "timed out" in result
    assert "process group were killed" in result
    time.sleep(2.5)  # give the would-be orphan time to prove itself
    assert not marker.exists()
    assert time.monotonic() - started < 20


def test_the_vocabulary_matches_an_all_features_toolbox(tmp_path: Path) -> None:
    """TOOL_VOCABULARY is what cards validate against, so it must equal what a
    session with everything enabled actually builds — no phantom names, no
    unlisted tools."""
    from conftest import build_mp_snapshot
    from mason.roster import discover_roster
    from mason.tools import TOOL_VOCABULARY

    snapshot = build_mp_snapshot(tmp_path / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    session = MasonSession(tmp_path, workspace_root=tmp_path / ".slab", hpc=hpc)
    roster = discover_roster(tmp_path)
    box = build_toolbox(session, roster["pi"], roster=roster)
    assert set(box.tools) == TOOL_VOCABULARY


def test_the_looking_tools_are_real_tools_that_never_act() -> None:
    from mason.tools import LOOKING_TOOLS, READ_ONLY_TOOLS, TOOL_VOCABULARY

    assert LOOKING_TOOLS <= TOOL_VOCABULARY
    assert READ_ONLY_TOOLS - {"finish"} <= LOOKING_TOOLS
    assert not LOOKING_TOOLS & {"launch_workflow", "submit_job", "plan", "notebook", "write_file"}


# -- the file fence ----------------------------------------------------------


def test_file_tools_refuse_paths_outside_the_fence(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The sandbox principle at the tool layer: work happens in the project."""
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    secret = elsewhere / "secret.txt"
    secret.write_text("credentials\n")
    box = build_toolbox(_session(tmp_path))

    for call in (
        _call("read_file", path=str(secret)),
        _call("write_file", path=str(elsewhere / "new.txt"), content="x"),
        _call("edit_file", path=str(secret), old_string="a", new_string="b"),
        _call("list_dir", path=str(elsewhere)),
        _call("search", pattern="credentials", path=str(elsewhere)),
        _call("launch_workflow", script=str(elsewhere / "wf.py"), intent="x"),
    ):
        answer = box.dispatch(call)
        assert "outside this session's file scope" in answer, call.name
        assert "file_scope" in answer, call.name
        # The refusal names the concrete roots, so the retry can land
        # in-fence instead of falling back to shell introspection.
        assert str(tmp_path) in answer, call.name
    assert not (elsewhere / "new.txt").exists()
    assert secret.read_text() == "credentials\n"


def test_relative_escapes_and_symlinks_stay_inside(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    import os

    elsewhere = tmp_path_factory.mktemp("elsewhere-links")
    (elsewhere / "secret.txt").write_text("hidden\n")
    box = build_toolbox(_session(tmp_path))

    dotted = os.path.relpath(elsewhere / "secret.txt", tmp_path)
    assert dotted.startswith("..")
    assert "outside this session's file scope" in box.dispatch(_call("read_file", path=dotted))

    link = tmp_path / "innocent.txt"
    link.symlink_to(elsewhere / "secret.txt")
    assert "outside this session's file scope" in box.dispatch(
        _call("read_file", path="innocent.txt")
    )


def test_the_fence_admits_project_workspace_and_skill_roots(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reads reach the project, the workspace, and discovered skill roots;
    writes reach only the first two."""
    xdg = tmp_path_factory.mktemp("xdg-fence")
    skill_dir = xdg / "slab" / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: a fence test skill\n---\n\nBody.\n"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    session = _session(tmp_path)
    box = build_toolbox(session)
    (tmp_path / "notes.txt").write_text("in project\n")
    workspace_file = session.workspace_root / "mason" / "roster.json"
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    workspace_file.write_text("{}\n")

    assert "in project" in box.dispatch(_call("read_file", path="notes.txt"))
    assert "{}" in box.dispatch(_call("read_file", path=str(workspace_file)))
    assert "fence test skill" in box.dispatch(_call("read_file", path=str(skill_dir / "SKILL.md")))
    answer = box.dispatch(
        _call("write_file", path=str(skill_dir / "SKILL.md"), content="clobbered")
    )
    assert "outside this session's file scope" in answer
    assert "fence test skill" in (skill_dir / "SKILL.md").read_text()


def test_past_session_transcripts_are_refused_with_the_doctrine(tmp_path: Path) -> None:
    """The sessions directory sits inside the workspace, but past sessions
    are not context: a real run burned six steps excavating an old
    campaign's compaction file and inherited its stale decisions. The
    refusal points at the sanctioned channels instead."""
    session = _session(tmp_path)
    box = build_toolbox(session)
    old = session.sessions_dir / "20260829-134153-23.compactions.md"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("# Context compactions\nstale decisions\n")

    for call in (
        _call("read_file", path=str(old)),
        _call("list_dir", path=str(session.sessions_dir)),
        _call("search", pattern="stale", path=str(session.sessions_dir)),
    ):
        answer = box.dispatch(call)
        assert "past sessions are not context" in answer, call.name
        assert "recall" in answer and "remember" in answer, call.name
    assert "stale decisions" not in box.dispatch(_call("read_file", path=str(old)))


def test_file_scope_anywhere_lifts_the_fence(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere-open")
    (elsewhere / "data.txt").write_text("visible\n")
    box = build_toolbox(_session(tmp_path, file_scope="anywhere"))
    assert "visible" in box.dispatch(_call("read_file", path=str(elsewhere / "data.txt")))


def test_submit_job_files_land_in_the_workspace_jobs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job scripts and SLURM output stay out of the project directory (so
    'slab-stack purge' can sweep them), while a prologue cd keeps the
    payload running in the project."""
    import slab.hpc as hpc_module
    from slab.hpc import SubmittedJob

    captured: dict[str, object] = {}

    def fake_submit(script: str, *, job_name: str, partition: str, directory=None):
        captured["script"] = script
        captured["directory"] = Path(directory)
        return SubmittedJob(
            job_id="42", job_name=job_name, partition=partition, script_path="x"
        )

    monkeypatch.setattr(hpc_module, "submit", fake_submit)
    hpc = HpcConfig.model_validate({"default_partition": "cpu", "partitions": {"cpu": {}}})
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", hpc=hpc, auto_approve=True
    )
    box = build_toolbox(session)
    import json

    call = ToolCall(
        id="t1",
        name="submit_job",
        arguments={"command": "foundation run wf.py", "name": "cu"},
        arguments_raw=json.dumps({"command": "foundation run wf.py", "name": "cu"}),
    )
    answer = box.dispatch(call)
    assert "submitted job 42" in answer
    assert captured["directory"] == session.workspace_root / "jobs"
    recorded = _command_events(session)
    assert len(recorded) == 1 and recorded[0]["kind"] == "job"
    assert recorded[0]["command"] == "foundation run wf.py" and recorded[0]["job_id"] == "42"
    assert recorded[0]["partition"] == "cpu" and recorded[0]["by"] == "pi"
    assert f"cd {tmp_path}" in str(captured["script"])
    # the batch job's runs join this chat, so they promote with it
    assert f"export SLAB_SESSION={session.session_id}" in str(captured["script"])


def test_oversubscribed_launches_are_refused_where_they_run_here(
    box: Toolbox, tmp_path: Path
) -> None:
    """The shell and launch_workflow execute in this session's allocation, so
    a hand-written mpirun asking for more ranks than the CPU budget is
    refused as a tool result the model can read and adapt to."""
    result = box.dispatch(_call("shell", command="mpirun -np 99999 hostname"))
    assert "refused" in result and "99999 MPI rank(s)" in result
    assert "cpu(s) are usable" in result

    script = tmp_path / "over.py"
    script.write_text('import subprocess\nsubprocess.run("srun --ntasks=99999 pw.x")\n')
    result = box.dispatch(_call("launch_workflow", script=str(script)))
    assert "refused" in result and "99999" in result

    # Within budget passes through to real execution.
    result = box.dispatch(_call("shell", command="echo mpirun -np 1 ok"))
    assert "exit 0" in result


def test_the_environment_states_the_budget_and_what_is_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_gpus: None
) -> None:
    from foundation import Workspace
    from mason.prompts import environment_block

    monkeypatch.setenv("SLURM_NTASKS", "16")
    session = _session(tmp_path)
    block = environment_block(session)
    assert "cpus:" in block and "gpus: 0" in block
    assert "16 rank(s)" in block
    assert "ntasks, threads, and gpus" in block
    assert "refused" in block  # the promise the tools actually keep
    cpus = int(block.split("cpus: ")[1].split()[0])
    assert f"free right now: {cpus} cpu(s), 0 gpu(s)" in block
    with Workspace(session.workspace_root) as ws:
        ws.reserve(ntasks=1, holder_pid=os.getpid())
    assert f"free right now: {cpus - 1} cpu(s), 0 gpu(s)" in environment_block(session)


# -- sized launches -----------------------------------------------------------


@pytest.fixture()
def no_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS", "SLURM_NTASKS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


SIZED_WORKFLOW = """\
import os
from foundation import check
print("env", os.environ["SLAB_NTASKS"], os.environ["OMP_NUM_THREADS"],
      repr(os.environ["CUDA_VISIBLE_DEVICES"]))
@check
def ok():
    return True
"""


def _budget_cpus() -> int:
    from slab.resources import budget

    return len(budget().cpus)


def test_a_sized_launch_that_does_not_fit_is_refused_with_the_free_amounts(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    (tmp_path / "wf.py").write_text(SIZED_WORKFLOW)
    answer = box.dispatch(_call("launch_workflow", script="wf.py", gpus=1))
    assert answer.startswith("refused:") and "1 gpu(s) asked, but only 0 of 0" in answer
    too_many = _budget_cpus() + 1
    answer = box.dispatch(_call("launch_workflow", script="wf.py", ntasks=too_many))
    assert answer.startswith("refused:") and f"{too_many} cpus asked" in answer
    assert "are free on" in answer
    with Workspace(box.session.workspace_root) as ws:
        assert ws.runs.list_reservations() == []  # nothing leaked
    assert _command_events(box.session) == []  # a refusal ran nothing


def test_a_foreground_sized_launch_runs_as_a_child_and_the_run_carries_resources(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    (tmp_path / "wf.py").write_text(SIZED_WORKFLOW)
    answer = box.dispatch(
        _call("launch_workflow", script="wf.py", intent="sized", ntasks=1, threads=1)
    )
    assert "state=verified" in answer and "checks=1/1" in answer
    assert "resources held: 1 cpu(s)" in answer and "1 rank(s) x 1 thread(s)" in answer
    assert "env 1 1 ''" in answer  # the child ran inside the envelope
    run_id = answer.split()[1].rstrip(":")
    (launch,) = [e for e in _command_events(box.session) if e["kind"] == "launch"]
    assert launch["sized"] is True and launch["background"] is False
    assert "--reservation " in launch["command"]
    assert launch["resources"]["ntasks"] == 1 and len(launch["resources"]["cpus"]) == 1
    with Workspace(box.session.workspace_root) as ws:
        run = ws.runs.get(run_id)
        assert run.pid != os.getpid()  # a child, never this process
        assert run.resources["reservation"] == launch["reservation"]
        assert run.resources["cpus"] == launch["resources"]["cpus"]
        assert ws.runs.list_reservations() == []  # released when the run ended


def test_two_background_launches_get_disjoint_slices(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    if _budget_cpus() < 2:
        pytest.skip("two one-cpu slices need two cpus")
    (tmp_path / "slow.py").write_text("import time\ntime.sleep(1.5)\n")
    first = box.dispatch(
        _call("launch_workflow", script="slow.py", name="one", background=True, ntasks=1)
    )
    second = box.dispatch(
        _call("launch_workflow", script="slow.py", name="two", background=True, ntasks=1)
    )
    assert "launched in the background" in first and "launched in the background" in second
    one, two = [e for e in _command_events(box.session) if e["kind"] == "launch"]
    assert one["resources"]["cpus"] != two["resources"]["cpus"]
    assert not set(one["resources"]["cpus"]) & set(two["resources"]["cpus"])
    box.dispatch(_call("wait_for_run", timeout_s=60))
    with Workspace(box.session.workspace_root) as ws:
        held = {run.name: run.resources["cpus"] for run in ws.runs.list_runs()}
        assert held["one"] == one["resources"]["cpus"]
        assert held["two"] == two["resources"]["cpus"]
        assert ws.runs.list_reservations() == []


def test_an_unsized_foreground_launch_reserves_the_whole_free_budget_in_process(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    (tmp_path / "wf.py").write_text("print('ok')\n")
    answer = box.dispatch(_call("launch_workflow", script="wf.py"))
    assert f"resources held: {_budget_cpus()} cpu(s)" in answer
    run_id = answer.split()[1].rstrip(":")
    with Workspace(box.session.workspace_root) as ws:
        run = ws.runs.get(run_id)
        assert run.pid == os.getpid()  # in-process, as before
        assert len(run.resources["cpus"]) == _budget_cpus()
        assert ws.runs.list_reservations() == []


def test_a_hand_written_mpirun_is_judged_against_the_launches_own_slice(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    script = tmp_path / "wide.py"
    script.write_text('import subprocess\nsubprocess.run("mpirun -np 2 hostname")\n')
    answer = box.dispatch(_call("launch_workflow", script="wide.py", ntasks=1))
    assert answer.startswith("refused") and "2 MPI rank(s)" in answer
    assert "1 cpu(s) are in this launch's slice" in answer
    with Workspace(box.session.workspace_root) as ws:
        assert ws.runs.list_reservations() == []  # the refused launch gave it back


def test_the_shell_refuses_the_run_driver(box: Toolbox) -> None:
    for command in (
        "slab run wf.py",
        "foundation run wf.py -w .slab",
        "python -m foundation.cli run wf.py",
        "python -m slab_stack.cli run wf.py",
    ):
        answer = box.dispatch(_call("shell", command=command))
        assert answer.startswith("refused") and "launch_workflow" in answer
    assert _command_events(box.session) == []
    assert box.dispatch(_call("shell", command="echo slab list")).startswith("exit 0")


def _clustered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Toolbox, dict[str, object]]:
    import slab.hpc as hpc_module
    from slab.hpc import SubmittedJob

    captured: dict[str, object] = {}

    def fake_submit(script: str, *, job_name: str, partition: str, directory=None):
        captured["script"] = script
        return SubmittedJob(job_id="7", job_name=job_name, partition=partition, script_path="x")

    monkeypatch.setattr(hpc_module, "submit", fake_submit)
    hpc = HpcConfig.model_validate(
        {
            "default_partition": "gpu",
            "partitions": {
                "gpu": {"gres": "gpu:a100:4", "node": {"cpus": 64, "gpus": 4, "mem": "480G"}},
                "cpu": {},
            },
        }
    )
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", hpc=hpc, auto_approve=True
    )
    return build_toolbox(session), captured


def test_submit_job_takes_a_size_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box, captured = _clustered(tmp_path, monkeypatch)
    answer = box.dispatch(
        _call(
            "submit_job", command="slab run md.py", name="md", ntasks_per_node=4, gpus_per_node=2
        )
    )
    assert "submitted job 7" in answer
    assert "sized 1 node(s) x 4 rank(s) x 1 cpu(s), 2 gpu(s) per node" in answer
    script = str(captured["script"])
    assert "#SBATCH --ntasks-per-node=4\n" in script and "#SBATCH --gres=gpu:a100:2\n" in script
    (event,) = _command_events(box.session)
    assert event["kind"] == "job" and event["size"]["gpus_per_node"] == 2
    assert event["size"]["ntasks_per_node"] == 4 and event["size"]["nodes"] == 1


def test_submit_job_refuses_a_size_the_node_cannot_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    box, captured = _clustered(tmp_path, monkeypatch)
    answer = box.dispatch(
        _call("submit_job", command="c", name="n", ntasks_per_node=1, gpus_per_node=5)
    )
    assert answer.startswith("refused:") and "gpus_per_node=5 exceeds the 4 gpus" in answer
    assert "[hpc.partitions.gpu.node] gpus" in answer
    answer = box.dispatch(_call("submit_job", command="c", name="n", gpus_per_node=1))
    assert answer.startswith("refused:") and "pass ntasks_per_node" in answer
    answer = box.dispatch(
        _call("submit_job", command="c", name="n", partition="cpu", ntasks_per_node=1)
    )
    assert answer.startswith("refused:") and "declares no node" in answer
    assert "[hpc.partitions.cpu.node]" in answer
    assert "script" not in captured and _command_events(box.session) == []


# -- session stamps ----------------------------------------------------------


def test_launch_workflow_stamps_the_chat_session(tmp_path: Path) -> None:
    from foundation import Workspace

    session = _session(tmp_path)
    (tmp_path / "wf.py").write_text("x = 1\n")
    build_toolbox(session).dispatch(_call("launch_workflow", script="wf.py", intent="stamp me"))
    with Workspace(session.workspace_root) as ws:
        (run,) = ws.runs.list_runs()
        assert run.session == session.session_id
        assert run.session == session.transcript_path.stem


def test_a_delegated_child_launches_into_the_parents_session(tmp_path: Path) -> None:
    """One chat, one id: a specialist's runs promote with the PI's."""
    from foundation import Workspace
    from mason.config import MasonConfig

    parent = _session(tmp_path)
    child = parent.spawn("md-expert", MasonConfig.model_validate({}).agent)
    grandchild = child.spawn("analysis-expert", MasonConfig.model_validate({}).agent)
    assert child.session_id == parent.session_id
    assert grandchild.session_id == parent.session_id
    assert child.transcript_path != parent.transcript_path

    (tmp_path / "wf.py").write_text("x = 1\n")
    build_toolbox(child).dispatch(_call("launch_workflow", script="wf.py"))
    with Workspace(tmp_path / ".slab") as ws:
        assert ws.runs.list_runs()[0].session == parent.session_id


# -- background runs and waiting ---------------------------------------------


def test_launch_workflow_passes_script_args(box: Toolbox, tmp_path: Path) -> None:
    """A parametrized workflow gets its argv — a real campaign duplicated a
    whole script because the tool could not pass one argument."""
    (tmp_path / "argv_wf.py").write_text(
        "import sys\nprint('got:', sys.argv[1])\n"
    )
    answer = box.dispatch(
        _call("launch_workflow", script="argv_wf.py", args=["quench-10Kps.traj"])
    )
    assert "status=completed" in answer
    assert "got: quench-10Kps.traj" in answer


def test_list_runs_session_this_means_the_current_session(
    box: Toolbox, tmp_path: Path
) -> None:
    (tmp_path / "wf.py").write_text("print('ok')\n")
    box.dispatch(_call("launch_workflow", script="wf.py"))
    mine = box.dispatch(_call("list_runs", session="this"))
    assert "wf" in mine
    missing = box.dispatch(_call("list_runs", session="some-other-session"))
    assert "failed" in missing or "no run" in missing


def test_background_launch_detaches_and_wait_for_run_collects(
    box: Toolbox, tmp_path: Path
) -> None:
    (tmp_path / "slow_wf.py").write_text(
        "import time\ntime.sleep(1.0)\nprint('background done')\n"
    )
    arguments = {
        "script": "slow_wf.py",
        "name": "bg-test",
        "intent": "background launch test",
        "background": True,
    }
    answer = box.dispatch(
        ToolCall(id="t1", name="launch_workflow", arguments=arguments, arguments_raw="{}")
    )
    assert "launched in the background: pid" in answer
    assert "wait_for_run" in answer
    launch = _command_events(box.session)[-1]
    assert launch["kind"] == "launch" and launch["background"] is True
    # A background launch is a reserved child: the record carries the
    # driver with the reservation it hands over, and the slice it holds.
    assert launch["command"].startswith("slab run ") and "--name bg-test" in launch["command"]
    assert "--reservation " in launch["command"] and launch["sized"] is False
    assert launch["resources"]["cpus"] and launch["reservation"]
    waited = box.dispatch(_call("wait_for_run", timeout_s=60))
    assert "bg-test" in waited
    assert "running" not in waited.split("bg-test")[1].splitlines()[0]
    log = (tmp_path / "slow_wf.launch.log").read_text()
    assert "background done" in log


def test_wait_for_run_reports_a_finished_run_by_id(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "wf.py").write_text("print('ok')\n")
    launched = box.dispatch(_call("launch_workflow", script="wf.py"))
    run_id = launched.split()[1].rstrip(":")
    waited = box.dispatch(_call("wait_for_run", run_id=run_id, timeout_s=5))
    assert f"run {run_id}" in waited
    assert "status=completed" in waited


def test_wait_for_run_with_nothing_launched_says_so(box: Toolbox) -> None:
    answer = box.dispatch(_call("wait_for_run", timeout_s=0.2))
    assert "no runs yet" in answer


def test_wait_for_run_timeout_reports_still_running(
    box: Toolbox, tmp_path: Path
) -> None:
    (tmp_path / "very_slow.py").write_text(
        "import time\ntime.sleep(20)\nprint('done')\n"
    )
    arguments = {"script": "very_slow.py", "name": "slowpoke", "background": True}
    answer = box.dispatch(
        ToolCall(id="t1", name="launch_workflow", arguments=arguments, arguments_raw="{}")
    )
    assert "launched in the background" in answer
    import time as _time

    for _ in range(40):  # let the subprocess register its run
        listed = box.dispatch(_call("list_runs", session="this"))
        if "slowpoke" in listed:
            break
        _time.sleep(0.5)
    try:
        waited = box.dispatch(_call("wait_for_run", timeout_s=1))
        assert "still running after 1s" in waited
        assert "slowpoke" in waited
        assert "call wait_for_run again" in waited
        assert "running; tasks:" in waited  # the tally says whether it is moving
    finally:
        # The detached run must not outlive the test on a shared machine.
        import os as _os
        import re as _re
        import signal as _signal

        pid = int(_re.search(r"pid (\d+)", answer).group(1))
        with contextlib.suppress(ProcessLookupError, PermissionError):
            _os.killpg(pid, _signal.SIGKILL)


# -- the mp snapshot tools ---------------------------------------------------


def _snapshot_session(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> MasonSession:
    """A session whose config names a snapshot OUTSIDE the project — the
    deployment shape, and the case the fence carve-out exists for."""
    from conftest import build_mp_snapshot

    snapshot = build_mp_snapshot(tmp_path_factory.mktemp("data") / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    return _session(tmp_path)


def test_mp_tools_exist_only_when_a_snapshot_is_configured(
    box: Toolbox, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    for name in ("search_materials", "get_material", "query_materials"):
        assert name not in box.tools
    configured = build_toolbox(_snapshot_session(tmp_path, tmp_path_factory))
    for name in ("search_materials", "get_material", "query_materials"):
        assert name in configured.tools


def test_mp_search_and_lookup_answer_from_the_snapshot(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    import json

    box = build_toolbox(_snapshot_session(tmp_path, tmp_path_factory))
    rows = json.loads(
        box.dispatch(
            _call(
                "search_materials",
                filters={"elements": ["Fe"], "energy_above_hull__lte": 0.05},
                columns=["material_id", "formula_pretty"],
            )
        )
    )
    assert rows == [{"material_id": "mp-13", "formula_pretty": "Fe"}]
    record = json.loads(box.dispatch(_call("get_material", material_id="mp-13")))
    assert record["elements"] == ["Fe"]
    # The archived CIF is inside the file fence: readable, not writable.
    cif = record["cif_file"]
    assert "Fe" in box.dispatch(_call("read_file", path=cif))
    answer = box.dispatch(
        _call("edit_file", path=cif, old_string="Fe", new_string="Xx")
    )
    assert "outside this session's file scope" in answer
    result = json.loads(
        box.dispatch(
            _call(
                "query_materials",
                sql="SELECT count(*) AS n FROM materials",
            )
        )
    )
    assert result["rows"] == [{"n": 4}]


def test_mp_tool_errors_are_observations(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    box = build_toolbox(_snapshot_session(tmp_path, tmp_path_factory))
    answer = box.dispatch(_call("search_materials", filters={"bandgap__lte": 1}))
    assert answer.startswith("tool search_materials failed:")
    assert "band_gap" in answer  # the refusal teaches the real schema
    answer = box.dispatch(_call("get_material", material_id="mp-404"))
    assert "no online fallback" in answer
    answer = box.dispatch(_call("query_materials", sql="DROP TABLE materials"))
    assert "only read-only queries" in answer


# -- the fences hold under a relative or symlinked workspace -----------------


def test_the_sessions_fence_holds_for_a_relative_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default workspace is the relative '.slab'; resolving the request
    path but not the sessions directory left the fence permanently open."""
    monkeypatch.chdir(tmp_path)
    config = MasonConfig.model_validate({})
    session = MasonSession(
        Path("."), workspace_root=Path(".slab"), agent=config.agent, auto_approve=True
    )
    sessions = Path(".slab") / "mason" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "old.jsonl").write_text('{"role": "user", "content": "OLD TRANSCRIPT"}\n')
    (tmp_path / "link.txt").symlink_to(sessions / "old.jsonl")
    box = build_toolbox(session)
    assert "refused" in box.dispatch(_call("read_file", path=".slab/mason/sessions/old.jsonl"))
    assert "refused" in box.dispatch(_call("list_dir", path=".slab/mason/sessions"))
    found = box.dispatch(_call("search", pattern="OLD TRANSCRIPT", path="."))
    assert found.startswith("no matches")  # neither directly nor through the symlink


def test_shell_and_launches_never_see_the_model_key(
    box: Toolbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`env` is a common diagnostic move; the key must not be in the answer."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-FAKE-anthropic")
    monkeypatch.setenv("SLAB_TEST_MODEL_KEY", "sk-FAKE-named")
    session = _session(tmp_path, api_key_env="SLAB_TEST_MODEL_KEY")
    named = build_toolbox(session)
    answer = named.dispatch(_call("shell", command="env"))
    assert "sk-FAKE-anthropic" not in answer and "sk-FAKE-named" not in answer
    assert "PATH=" in answer  # the rest of the environment still arrives


# -- the run tools after two campaigns' transcripts --------------------------


_RELAX_SCRIPT = (
    "from foundation import check, converged\n"
    "from foundation.tasks import relax\n"
    "from ase.build import bulk\n"
    "atoms = bulk('Cu', 'fcc', a=3.6)\n"
    "relaxed, info = relax(atoms, engine='emt', fmax=0.05, label='cu')\n"
    "@check\n"
    "def forces_converged():\n"
    "    return converged(info['fmax'], below=0.05)\n"
)


def _launch_named(box: Toolbox, script: str, name: str) -> str:
    arguments = {"script": script, "name": name}
    call = ToolCall(id="t1", name="launch_workflow", arguments=arguments, arguments_raw="{}")
    return box.dispatch(call)


def test_show_run_folds_finished_tasks_unless_asked_for_everything(
    box: Toolbox, tmp_path: Path
) -> None:
    """The full record of an 88-task run was 190,000 characters, and a real
    session polled it six times; finished tasks now fold to one line, and
    the recipes come back on request."""
    (tmp_path / "wf.py").write_text(_RELAX_SCRIPT)
    launched = box.dispatch(_call("launch_workflow", script="wf.py", intent="fold test"))
    run_id = launched.split()[1].rstrip(":")
    compact = box.dispatch(_call("show_run", run_id=run_id))
    assert '"tasks_summary": "1 completed"' in compact
    assert '"label": "cu"' in compact
    assert '"recipe"' not in compact
    assert "full=true" in compact
    full = box.dispatch(_call("show_run", run_id=run_id, full=True))
    assert '"recipe"' in full and '"tasks_summary"' not in full


def test_show_run_returns_one_task_in_full_on_request(box: Toolbox, tmp_path: Path) -> None:
    """A real session read the full record, saw it cut at the cap, and went
    digging in the artifact store by hand for one task's output."""
    (tmp_path / "wf.py").write_text(_RELAX_SCRIPT)
    launched = box.dispatch(_call("launch_workflow", script="wf.py", intent="one task"))
    run_id = launched.split()[1].rstrip(":")
    by_label = json.loads(box.dispatch(_call("show_run", run_id=run_id, task="cu")))
    assert by_label["task"]["name"] == "relax" and '"recipe"' not in json.dumps(by_label["run"])
    assert "recipe" in by_label["task"] and "outputs" in by_label["task"]
    assert "tasks_summary" not in by_label and by_label["checks"]
    by_seq = json.loads(box.dispatch(_call("show_run", run_id=run_id, task="1")))
    assert by_seq["task"] == by_label["task"]
    missing = json.loads(box.dispatch(_call("show_run", run_id=run_id, task="nope")))
    assert missing["error"] == "no task 'nope'; the tasks: 1 relax (cu)"


def test_read_artifact_reads_a_runs_file_by_name_windowed(box: Toolbox, tmp_path: Path) -> None:
    """The store is content-addressed; a real session guessed its layout by
    hand for six minutes to read one .pwo. The tool reads it by name."""
    (tmp_path / "wf.py").write_text(_RELAX_SCRIPT)
    launched = box.dispatch(_call("launch_workflow", script="wf.py", intent="artifact"))
    run_id = launched.split()[1].rstrip(":")
    record = json.loads(box.dispatch(_call("show_run", run_id=run_id)))
    assert record["artifacts"], "the relax run keeps at least one artifact"
    name = record["artifacts"][0]["name"]

    def read(**arguments: object) -> str:  # _call's own 'name' parameter is the tool's
        call = ToolCall(id="ra", name="read_artifact", arguments=arguments, arguments_raw="{}")
        return box.dispatch(call)

    shown = read(run_id=run_id, name=name, limit=3)
    assert shown.startswith(f"{name} ({record['artifacts'][0]['size_bytes']} bytes, sha256 ")
    assert "looks binary" in shown or "\n     1\t" in shown
    by_hash = read(run_id=run_id, name=record["artifacts"][0]["hash"][:8], limit=3)
    assert by_hash.startswith(f"{name} (")
    missing = read(run_id=run_id, name="nope.pwo")
    assert missing.startswith("no artifact named 'nope.pwo'") and name in missing


def test_run_tools_resolve_a_run_by_the_name_the_model_remembers(
    box: Toolbox, tmp_path: Path
) -> None:
    """A real transcript passed the script's name to wait_for_run twice and
    was told no run matched; the name resolves now, with a note."""
    (tmp_path / "wf.py").write_text(_RELAX_SCRIPT)
    _launch_named(box, "wf.py", "cu-relax")
    shown = box.dispatch(_call("show_run", run_id="cu-relax"))
    assert shown.startswith("(resolved 'cu-relax' by name to run ")
    assert '"name": "cu-relax"' in shown
    waited = box.dispatch(_call("wait_for_run", run_id="cu-relax", timeout_s=5))
    assert "status=completed" in waited
    assert "tasks: 1 completed; checks: 1/1 passed" in waited
    missing = box.dispatch(_call("show_run", run_id="no-such-run"))
    assert "failed: RunNotFoundError" in missing


def test_list_runs_takes_a_status_and_forgives_the_swap(box: Toolbox, tmp_path: Path) -> None:
    """``state="running"`` raised a ValueError at a real session; the word
    names a status, and it is read as one."""
    (tmp_path / "wf.py").write_text("print('ok')\n")
    _launch_named(box, "wf.py", "quick")
    assert "quick" in box.dispatch(_call("list_runs", status="completed"))
    assert "no runs" in box.dispatch(_call("list_runs", state="running"))
    assert "no runs" in box.dispatch(_call("list_runs", status="failed"))


def test_shell_refuses_to_rewrite_the_run_store_by_hand(box: Toolbox, tmp_path: Path) -> None:
    """A real session deleted a live database's write-ahead log from the
    shell. The store's files are SQLite's; the session reports, it does
    not repair."""
    workspace = tmp_path / ".slab"
    refused = box.dispatch(_call("shell", command=f"rm -f {workspace}/runs.db-wal"))
    assert refused.startswith("refused: this command would delete or move files of the run store")
    refused = box.dispatch(_call("shell", command=f"cd {workspace} && rm -f runs.db-shm"))
    assert "refused" in refused
    allowed = box.dispatch(_call("shell", command="rm -f nothing-here.txt"))
    assert allowed.startswith("exit 0")
    copied = box.dispatch(_call("shell", command=f"ls {workspace}/runs.db >/dev/null; echo ok"))
    assert "ok" in copied


def test_a_locked_run_store_is_reported_with_its_recovery(
    box: Toolbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bare "database is locked" sent a real session into an hour of
    forensics; the fault now arrives with what to do instead."""
    from foundation.errors import StorageError

    def refuse(self: object, root: object = None) -> None:
        raise StorageError(f"cannot open workspace at {root}: database is locked")

    monkeypatch.setattr("foundation.runtime.Workspace.__init__", refuse)
    answer = box.dispatch(_call("list_runs"))
    assert answer.startswith("tool list_runs failed: RunStoreUnavailable: the run store at ")
    assert "database is locked" in answer
    assert "Wait about a minute and retry this call once" in answer
    assert "Do not inspect, modify, or delete files under" in answer
    assert "RunStoreUnavailable" in box.dispatch(_call("wait_for_run", timeout_s=1))


def test_background_launches_write_their_log_line_by_line(
    box: Toolbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A block-buffered log stayed empty for two hours of labeling while a
    real session polled it; the launch asks python for unbuffered output."""
    import types

    seen: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> object:
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setattr("mason.tools.subprocess.Popen", fake_popen)
    (tmp_path / "wf.py").write_text("print('ok')\n")
    answer = box.dispatch(
        ToolCall(
            id="t1",
            name="launch_workflow",
            arguments={"script": "wf.py", "background": True},
            arguments_raw="{}",
        )
    )
    assert "pid 4242" in answer
    env = seen["env"]
    assert isinstance(env, dict) and env["PYTHONUNBUFFERED"] == "1"


# -- digests of engine outputs -------------------------------------------------

_PWO = Path(__file__).parent / "data" / "qe-si-relax-final.pwo"


def test_read_file_digests_an_engine_output_unless_asked_for_the_text(
    box: Toolbox, tmp_path: Path
) -> None:
    """A real session read a 305 KB pw.x output in 400-line windows and took
    a band-eigenvalue block for a diverging SCF energy. The digest comes
    first; the text is one argument away."""
    target = tmp_path / "si.pwo"
    target.write_text(_PWO.read_text())
    shown = box.dispatch(_call("read_file", path="si.pwo"))
    assert shown.startswith(
        "pw.x output digest: si.pwo (248 lines, PWSCF v.7.4.1, finished: JOB DONE)"
    )
    assert "final ! -15.64003672 Ry" in shown
    assert shown.endswith("[digest of 248 lines; pass raw=true, or offset/limit, for the text]")
    assert "\n     1\t" not in shown
    raw = box.dispatch(_call("read_file", path="si.pwo", raw=True))
    assert "\n     2\t     Program PWSCF v.7.4.1 starts on" in raw and "digest" not in raw
    windowed = box.dispatch(_call("read_file", path="si.pwo", offset=183, limit=1))
    assert windowed.startswith("   183\t!    total energy")
    plain = box.dispatch(_call("read_file", path="wf.py")) if (tmp_path / "wf.py").exists() else ""
    assert "digest" not in plain


def test_read_artifact_digests_a_kept_engine_output(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "si.pwo").write_text(_PWO.read_text())
    (tmp_path / "keep.py").write_text(
        "from pathlib import Path\nfrom foundation import current_run\n"
        f"current_run().keep('si.pwo', Path({str(tmp_path / 'si.pwo')!r}))\n"
    )
    launched = box.dispatch(_call("launch_workflow", script="keep.py", intent="digest"))
    run_id = launched.split()[1].rstrip(":")

    def read(**arguments: object) -> str:
        call = ToolCall(id="ra", name="read_artifact", arguments=arguments, arguments_raw="{}")
        return box.dispatch(call)

    shown = read(run_id=run_id, name="si.pwo")
    assert shown.startswith("si.pwo (")
    assert "\npw.x output digest: si.pwo (248 lines" in shown
    assert "converged in 4 iterations" in shown
    raw = read(run_id=run_id, name="si.pwo", raw=True)
    assert "\n     2\t     Program PWSCF" in raw
    assert "[artifact has 248 lines; showing 1-248]" not in raw  # the whole file fit the window


def test_the_workflow_script_is_kept_as_the_runs_input_artifact(
    box: Toolbox, tmp_path: Path
) -> None:
    """One real lead searched the project, the workspace, and scratch for the
    script behind a run; the run record now holds it by name."""
    (tmp_path / "wf.py").write_text(_RELAX_SCRIPT)
    launched = box.dispatch(_call("launch_workflow", script="wf.py", intent="script kept"))
    run_id = launched.split()[1].rstrip(":")
    record = json.loads(box.dispatch(_call("show_run", run_id=run_id)))
    scripts = [a for a in record["artifacts"] if a["name"] == "wf.py"]
    assert scripts and scripts[0]["role"] == "input"
    arguments = {"run_id": run_id, "name": "wf.py"}
    call = ToolCall(id="ra", name="read_artifact", arguments=arguments, arguments_raw="{}")
    assert "relax(atoms, engine='emt'" in box.dispatch(call)


# -- run liveness through the tools ---------------------------------------------


def _dead_run(root: Path, name: str) -> str:
    """A run at status running whose recorded process has already exited."""
    import subprocess
    import sys

    from foundation.models import Run
    from foundation.runtime import Workspace, this_host

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    with Workspace(root) as ws:
        run = ws.runs.create(Run(name=name))
        ws.runs.set_status(run.id, "running", pid=child.pid, host=this_host())
    return run.id


def test_wait_for_run_and_list_runs_report_a_dead_process(box: Toolbox, tmp_path: Path) -> None:
    """A hard-killed run is not waited on and is not listed as running: the
    tool that meets it first marks it failed and says so in its answer."""
    dead = _dead_run(tmp_path / ".slab", "killed")
    waited = box.dispatch(_call("wait_for_run", run_id=dead, timeout_s=30))
    assert waited.startswith("this run's process is gone: run " + dead)
    assert "marked failed by wait_for_run" in waited
    assert "read it with show_run" in waited

    second = _dead_run(tmp_path / ".slab", "killed-too")
    listed = box.dispatch(_call("list_runs"))
    assert listed.startswith(f"(marked failed by list_runs: {second[:10]};")
    lines = listed.splitlines()
    assert all("running" not in line for line in lines[1:])
    assert sum("failed" in line for line in lines[1:]) == 2


def test_list_runs_and_show_run_word_the_initial_state_of_a_running_run(
    box: Toolbox, tmp_path: Path
) -> None:
    import os

    from foundation.models import Run
    from foundation.runtime import Workspace, this_host

    with Workspace(tmp_path / ".slab") as ws:
        run = ws.runs.create(Run(name="live"))
        ws.runs.set_status(run.id, "running", pid=os.getpid(), host=this_host())
    listed = box.dispatch(_call("list_runs"))
    assert "quarantined (initial state)" in listed
    assert f"[process {os.getpid()} on {this_host()} is alive]" in listed
    details = json.loads(box.dispatch(_call("show_run", run_id=run.id)))
    assert details["run"]["state"] == "quarantined (initial state)"
    assert details["run"]["liveness"] == f"process {os.getpid()} on {this_host()} is alive"
    assert (details["run"]["pid"], details["run"]["host"]) == (os.getpid(), this_host())
    with Workspace(tmp_path / ".slab") as ws:
        ws.runs.set_status(run.id, "completed")
    finished = box.dispatch(_call("list_runs"))
    assert "quarantined (initial state)" not in finished and "quarantined" in finished
    details = json.loads(box.dispatch(_call("show_run", run_id=run.id)))
    assert details["run"]["state"] == "quarantined" and "liveness" not in details["run"]


def test_the_transcript_records_every_command_that_ran(tmp_path: Path) -> None:
    """shell, a launch, and the engine commands the run resolved, each with who and when."""
    session = _session(tmp_path)
    box = build_toolbox(session)
    assert box.dispatch(_call("shell", command="echo hi")).startswith("exit 0")
    (tmp_path / "wf.py").write_text(COMMAND_WORKFLOW)
    arguments = {"script": "wf.py", "name": "cmd-test", "intent": "record commands"}
    launched = box.dispatch(
        ToolCall(id="t2", name="launch_workflow", arguments=arguments, arguments_raw="{}")
    )
    run_id = launched.split()[1].rstrip(":")
    shell, launch, engine = _command_events(session)
    assert shell["kind"] == "shell" and shell["tool"] == "shell"
    assert shell["command"] == "echo hi" and shell["cwd"] == str(tmp_path)
    assert shell["by"] == "pi" and shell["at"] < launch["at"] <= engine["at"]
    assert launch["kind"] == "launch" and launch["background"] is False
    assert launch["command"].startswith(f"slab run {tmp_path / 'wf.py'} --name cmd-test")
    assert launch["script"] == str(tmp_path / "wf.py")
    assert engine["kind"] == "engine" and engine["tool"] == "launch_workflow"
    assert engine["run_id"] == run_id and engine["task"] == "probe" and engine["tasks"] == 2
    assert engine["command"] == "mpirun -np 1 lmp -k on g 1 -sf kk"
    assert engine["setup"] == ["module load lammps"] and engine["version"] == "22 Jul 2025"
    assert engine["kokkos"]["enabled"] is True and engine["kokkos"]["gpus"] == 1
    # a wait on the same run records its commands no second time
    box.dispatch(_call("wait_for_run", run_id=run_id))
    assert len(_command_events(session)) == 3


def test_a_delegated_child_records_commands_under_its_own_card(tmp_path: Path) -> None:
    parent = _session(tmp_path)
    child = parent.spawn("md-expert", parent.agent)
    box = build_toolbox(child)
    box.dispatch(_call("shell", command="true"))
    (event,) = _command_events(child)
    assert event["by"] == "md-expert" and event["command"] == "true"
    assert _command_events(parent) == []


# -- review fixes: sizes, comments, the store ---------------------------------


def test_size_arguments_that_are_not_positive_integers_are_refused(
    box: Toolbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A word, a zero, or a negative size is a refusal the model reads, never
    an exception and never a launch of one rank."""
    from foundation import Workspace

    (tmp_path / "wf.py").write_text("print('never')\n")
    for arguments, text in (
        ({"ntasks": "two"}, "ntasks must be a positive integer, not 'two'"),
        ({"ntasks": 0}, "ntasks must be a positive integer, not 0"),
        ({"ntasks": 1, "threads": -2}, "threads must be a positive integer, not -2"),
        ({"gpus": -1}, "gpus must be zero or a positive integer, not -1"),
    ):
        answer = box.dispatch(_call("launch_workflow", script="wf.py", **arguments))
        assert answer == f"refused: {text}"
    with Workspace(box.session.workspace_root) as ws:
        assert ws.runs.list_reservations() == [] and ws.runs.list_runs() == []
    assert _command_events(box.session) == []
    clustered, captured = _clustered(tmp_path, monkeypatch)
    answer = clustered.dispatch(_call("submit_job", command="c", name="n", ntasks_per_node=0))
    assert answer == "refused: ntasks_per_node must be a positive integer, not 0"
    answer = clustered.dispatch(_call("submit_job", command="c", name="n", ntasks_per_node="two"))
    assert answer == "refused: ntasks_per_node must be a positive integer, not 'two'"
    assert "script" not in captured


def test_a_comment_naming_an_mpirun_is_not_an_mpirun(box: Toolbox, tmp_path: Path) -> None:
    (tmp_path / "noted.py").write_text("# note: mpirun -np 64 was too wide\nprint('ran')\n")
    answer = box.dispatch(_call("launch_workflow", script="noted.py", ntasks=1))
    assert answer.startswith("run ") and "script output:\nran" in answer
    answer = box.dispatch(_call("shell", command="echo ok  # mpirun -np 99999"))
    assert answer.startswith("exit 0")
    # A real mpirun after a comment line is still judged.
    (tmp_path / "wide.py").write_text("# mpirun -np 1\nimport os\nos.system('mpirun -np 64 x')\n")
    answer = box.dispatch(_call("launch_workflow", script="wide.py", ntasks=1))
    assert answer.startswith("refused") and "64 MPI rank(s)" in answer


def test_a_missing_script_gives_its_reservation_back(box: Toolbox) -> None:
    from foundation import Workspace

    answer = box.dispatch(_call("launch_workflow", script="nope.py", ntasks=1))
    assert answer.startswith("could not start the run:") and "no such workflow script" in answer
    with Workspace(box.session.workspace_root) as ws:
        assert ws.runs.list_reservations() == []


def test_the_unsized_refusal_names_the_free_gpus_too(
    box: Toolbox, tmp_path: Path, no_gpus: None
) -> None:
    from foundation import Workspace

    (tmp_path / "wf.py").write_text("print('never')\n")
    with Workspace(box.session.workspace_root) as ws:
        ws.reserve(holder_pid=os.getpid())  # the whole free budget
    answer = box.dispatch(_call("launch_workflow", script="wf.py"))
    assert answer.startswith("refused: no cpu is free on") and "free gpus: 0 of 0" in answer


def test_list_engines_still_answers_when_the_store_cannot_be_opened(tmp_path: Path) -> None:
    """A workspace root that cannot be created is a fault of the workspace,
    not of the engines: the tool lists them, with free unknown and a note."""
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    sealed.chmod(0o500)
    try:
        session = MasonSession(
            tmp_path, workspace_root=sealed / ".slab", agent=MasonConfig().agent, auto_approve=True
        )
        answer = json.loads(build_toolbox(session).dispatch(_call("list_engines")))
    finally:
        sealed.chmod(0o700)
    assert "builtin" in answer and answer["budget"]["cpus"] >= 1
    assert answer["free"] is None
    assert answer["resources_note"].startswith("run store unavailable: the run store at")


# -- edit events and the cut-child hand-back ---------------------------------


def _edit_events(session: MasonSession) -> list[dict[str, object]]:
    if not session.transcript_path.exists():
        return []
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    return [event for event in events if event.get("type") == "edit"]


def test_write_and_edit_record_the_file_under_the_card(box: Toolbox, tmp_path: Path) -> None:
    box.dispatch(_call("write_file", path="notes.py", content="x = 1\n"))
    box.dispatch(_call("read_file", path="notes.py"))
    box.dispatch(_call("edit_file", path="notes.py", old_string="x = 1", new_string="x = 2"))
    box.dispatch(_call("edit_file", path="missing.py", old_string="a", new_string="b"))
    events = _edit_events(box.session)
    assert [e["tool"] for e in events] == ["write_file", "edit_file"]  # the miss recorded nothing
    assert all(e["path"] == str(tmp_path / "notes.py") and e["by"] == "pi" for e in events)


def test_partial_outcome_names_the_files_and_runs_a_cut_child_left(tmp_path: Path) -> None:
    """A specialist cut mid-script hands its parent what it did, so the
    re-brief says 'continue from' instead of starting over."""
    from mason.tools import partial_outcome

    parent = _session(tmp_path)
    child = parent.spawn("md-expert", MasonConfig.model_validate({}).agent)
    assert "wrote no file and launched no run" in partial_outcome(child)
    box = build_toolbox(child)
    box.dispatch(_call("write_file", path="md.py", content="print('a')\n"))
    box.dispatch(_call("read_file", path="md.py"))
    box.dispatch(_call("edit_file", path="md.py", old_string="'a'", new_string="'b'"))
    (tmp_path / "wf.py").write_text("x = 1\n")
    answer = box.dispatch(_call("launch_workflow", script="wf.py"))
    run_id = answer.split("run ")[1].split(":")[0]
    outcome = partial_outcome(child)
    assert outcome.startswith("[partial outcome: the specialist's turn ended cut")
    assert outcome.count(str(tmp_path / "md.py")) == 1  # two edits, one file
    assert f"runs it launched: {run_id}" in outcome
    assert "continue from these" in outcome
