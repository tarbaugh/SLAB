"""Tool contract tests: file primitives, gating, truncation, SLAB integration."""

import contextlib
import json
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


def _call(name: str, **arguments: object) -> ToolCall:
    import json

    return ToolCall(
        id="t1", name=name, arguments=dict(arguments), arguments_raw=json.dumps(arguments)
    )


@pytest.fixture()
def box(tmp_path: Path) -> Toolbox:
    return build_toolbox(_session(tmp_path))


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
    assert "- read_file(path: string, offset?: integer, limit?: integer)" in catalog


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


def test_the_environment_states_the_cpu_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mason.prompts import environment_block

    monkeypatch.setenv("SLURM_NTASKS", "16")
    block = environment_block(_session(tmp_path))
    assert "cpus:" in block
    assert "16 rank(s)" in block
    assert "refused" in block  # the promise the tools actually keep


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
