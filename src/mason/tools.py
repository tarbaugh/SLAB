"""Mason's tools: few, orthogonal, with crisp machine-checkable failure modes.

The tool set follows the SWE-agent lesson that agent performance is driven
by interface design, not tool count: six file/shell primitives whose error
messages teach recovery, plus the tools that make Mason a *research* agent —
SLAB runs (list/show/launch), the SLURM plumbing when this machine has
partitions configured, and the memory instruments: the notebook and plan
for this project, recall and remember for what this machine has taught
earlier sessions.

Contracts the primitives enforce in code, not prompt text:

* ``edit_file`` is exact-string replacement — the old text must match the
  file exactly once (or ``replace_all``), and the file must have been read
  this session first (the staleness guard).
* ``read_file`` numbers lines (numbers ground later edits) and refuses
  binary content.
* every tool failure is returned as the tool *result*, never raised — the
  loop continues and the model sees the evidence.
* mutating tools pass through the session's approval gate; read-only tools
  never ask.
* large outputs are middle-truncated with an explicit marker (the head
  usually holds the command echo, the tail the verdict).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundation.runtime import Workspace
    from mason.roster import AgentSpec

from foundation import memory as memory_store
from foundation.errors import MemoryStoreError
from mason.client import ToolCall
from mason.session import MasonSession
from mason.skills import Skill, discover_skills, listing, visible_catalog

#: Every tool name any session can build. Agent cards validate their
#: ``tools:`` allowlists against this set, so a typo is refused even when the
#: tool it misspells is absent from the current session (no partitions, no
#: skills). Keep it in step with the builders below; a test enforces that.
TOOL_VOCABULARY = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "search",
        "shell",
        "list_runs",
        "show_run",
        "launch_workflow",
        "wait_for_run",
        "list_engines",
        "list_tasks",
        "describe_task",
        "search_materials",
        "get_material",
        "query_materials",
        "submit_job",
        "job_status",
        "cancel_job",
        "notebook",
        "plan",
        "skill",
        "recall",
        "remember",
        "delegate",
        "finish",
    }
)

_MAX_READ_LINES = 400
_MAX_LINE_CHARS = 500
_MAX_DIR_ENTRIES = 200
_MAX_SEARCH_MATCHES = 100
_MAX_SHELL_TIMEOUT_S = 600.0
# wait_for_run: one blocking call replaces a chain of sleep-and-poll shell
# commands. The cap bounds a single tool call, not the wait — the tool says
# to call again, and each re-issue costs one step instead of six.
_MAX_WAIT_TIMEOUT_S = 1800.0
_WAIT_POLL_S = 5.0
_WAIT_GRACE_S = 10.0

Handler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    """One tool: its schema for the model, its handler, its permission class."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    requires_approval: bool = False
    # A dynamic gate (e.g. shell allowlists) overrides requires_approval per call.
    gate: Callable[[dict[str, Any]], bool] | None = None

    def needs_approval(self, arguments: dict[str, Any]) -> bool:
        if self.gate is not None:
            return self.gate(arguments)
        return self.requires_approval


@dataclass
class Toolbox:
    """The session's tools, in a stable order (stable prompts cache well)."""

    session: MasonSession
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-style tool declarations for the request body."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def catalog_text(self) -> str:
        """A plain-text catalog for the fenced protocol (no tools= support)."""
        lines = []
        for tool in self.tools.values():
            properties = tool.parameters.get("properties", {})
            required = set(tool.parameters.get("required", ()))
            arguments = ", ".join(
                f"{name}{'' if name in required else '?'}: {spec.get('type', 'any')}"
                for name, spec in properties.items()
            )
            lines.append(f"- {tool.name}({arguments}): {tool.description}")
        return "\n".join(lines)

    def dispatch(self, call: ToolCall) -> str:
        """Execute one tool call; the answer is always a string, never a raise."""
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(self.tools)
            return f"unknown tool {call.name!r}; available tools: {known}"
        if call.arguments_error is not None:
            return f"tool {call.name} not run: {call.arguments_error}"
        missing = [
            key
            for key in tool.parameters.get("required", ())
            if key not in call.arguments
        ]
        if missing:
            properties = tool.parameters.get("properties", {})
            required = list(tool.parameters.get("required", ()))
            optional = [key for key in properties if key not in required]
            return (
                f"tool {call.name} not run: missing required argument(s) "
                f"{', '.join(missing)} (required: {', '.join(required)}"
                + (f"; optional: {', '.join(optional)}" if optional else "")
                + ")"
            )
        preview = self.session.attribution() + _preview(call)
        if tool.needs_approval(call.arguments):
            try:
                approved = self.session.allows(call.name, preview, requires_approval=True)
            except Exception:
                # dispatch never raises: a crashing approver (closed stdin,
                # broken terminal) is a refusal, not a dead process.
                approved = False
            if not approved:
                return (
                    f"tool {call.name} was not approved by the user; explain what you "
                    f"wanted it for, or work another way"
                )
        try:
            result = tool.handler(call.arguments)
        except Exception as e:  # evidence for the model, never a dead loop
            result = f"tool {call.name} failed: {type(e).__name__}: {e}"
        return _truncate_middle(result, self.session.agent.max_tool_output_chars)


def _preview(call: ToolCall) -> str:
    """What the human sees before approving — the load-bearing keys first.

    A raw JSON dump truncates arbitrarily: a giant ``content`` can push the
    ``path`` it writes to right out of the preview. Name the keys that
    decide whether to approve.
    """
    arguments = call.arguments
    if "command" in arguments:
        return str(arguments["command"])
    parts = []
    for key in (
        "path",
        "script",
        "name",
        "partition",
        "old_string",
        "new_string",
        "description",
        "content",
        "body",
    ):
        if key in arguments:
            value = str(arguments[key])
            # content and body are the thing being approved — a human shown
            # 120 chars of a workflow script, or of a fact about to be
            # written into every future session's prompt, is approving
            # blind. Head and tail, with the elision announced, is enough to
            # actually read it.
            limit = 1200 if key in ("content", "body") else 120
            shown = value if len(value) <= limit else _truncate_middle(value, limit)
            parts.append(f"{key}={shown!r}")
    return " ".join(parts)[:1600] if parts else json.dumps(arguments)[:200]


def _truncate_middle(text: str, limit: int) -> str:
    """Keep head and tail; announce exactly what was dropped.

    Examples:
        >>> _truncate_middle("x" * 30, 21)
        'xxxxxxxxxxxxxx\\n[... 15 characters truncated ...]\\nx'
        >>> _truncate_middle("short", 21)
        'short'
    """
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters truncated ...]\n"
    head = (limit * 2) // 3
    tail = limit - head - len(marker) + 2  # the {} placeholder frees ~2 chars
    tail = max(tail, 40) if limit > 200 else max(tail, 1)
    dropped = len(text) - head - tail
    return text[:head] + marker.format(dropped) + text[-tail:]


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def build_toolbox(
    session: MasonSession,
    spec: AgentSpec | None = None,
    *,
    depth: int = 0,
    skills: dict[str, Skill] | None = None,
    roster: dict[str, AgentSpec] | None = None,
    parent_client: Any | None = None,
) -> Toolbox:
    """Every tool this session gets; SLURM tools only where partitions exist.

    *spec* is the agent card in force: its ``tools`` allowlist narrows the
    box (``finish`` always stays), and ``None`` means no narrowing. *depth*
    is the delegation depth: a delegated agent (depth > 0) loses ``plan``,
    because ``PLAN.md`` belongs to the turn owner, and can never delegate
    onward. *skills* is the full catalog: the ``skill`` tool sees the
    card's slice of it, while ``delegate`` hands the whole catalog down so
    each child re-narrows by its own card; ``None`` discovers the catalog
    from the session's project directory. *roster* and *parent_client*
    feed the ``delegate`` tool, which exists only when the card delegates,
    the depth is zero, ``[agent] delegation`` is on, and the roster holds
    someone to delegate to.
    """
    if skills is None:
        skills = discover_skills(session.cwd)
    visible = (
        visible_catalog(skills, spec.name, spec.skills_scope) if spec is not None else skills
    )
    box = Toolbox(session)
    # The file fence (_out_of_scope): writes stay in the project and the
    # workspace; reads and launches also reach the skill directories the
    # harness itself advertises.
    write_roots = (
        _scope_root(session, session.cwd),
        _scope_root(session, session.workspace_root),
    )
    snapshot_root = _mp_snapshot_root(session)
    read_roots = (
        write_roots
        + tuple(_scope_root(session, s.root) for s in skills.values())
        + _installed_package_roots(session)
        + ((snapshot_root,) if snapshot_root is not None else ())
    )
    _add_file_tools(box, session, read_roots, write_roots)
    _add_shell_tool(box, session)
    _add_workflow_tools(box, session, read_roots)
    _add_engine_tools(box, session)
    if snapshot_root is not None:
        _add_mp_tools(box, snapshot_root)
    if session.hpc.partitions:
        _add_hpc_tools(box, session)
    _add_memory_tools(box, session)
    if session.agent.memory:
        _add_machine_memory_tools(box, session)
    if visible:
        _add_skill_tool(box, session, visible)
    if spec is not None and spec.delegates and depth == 0 and session.agent.delegation:
        from mason.roster import hands

        if roster is not None and hands(spec, roster):
            _add_delegate_tool(box, session, spec, roster, skills, parent_client)
    if spec is not None and spec.tools is not None:
        for name in [n for n in box.tools if n not in spec.tools and n != "finish"]:
            del box.tools[name]
    if depth > 0:
        box.tools.pop("plan", None)
    box.add(
        Tool(
            name="finish",
            description=(
                "End the current task with a final report. Cite run ids for every "
                "number; list what was verified and what remains open. When the "
                "task names a result key, also pass the quantity in `results` "
                "under that name with its unit, and list the run ids that "
                "produced it in `run_ids` — that is how a campaign is scored. "
                "Before calling it: a machine fact this session learned the hard "
                "way (a workaround, a missing utility, a device limit) belongs in "
                "`remember` first, or the next session pays for it again. Call "
                "finish alone, as the only tool call of its message, after the "
                "evidence it cites has been read."
            ),
            parameters=_schema(
                {
                    "report": {"type": "string"},
                    "results": {
                        "type": "object",
                        "description": "result name -> {value, unit}",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                            "required": ["value", "unit"],
                        },
                    },
                    "run_ids": {"type": "array", "items": {"type": "string"}},
                },
                ["report"],
            ),
            handler=lambda arguments: str(arguments.get("report", "")),
        )
    )
    return box


# -- file primitives ---------------------------------------------------------


def _resolve(session: MasonSession, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else session.cwd / candidate


def _scope_root(session: MasonSession, root: Path) -> Path:
    """A scope root in comparable form: absolute (against the session's cwd,
    not the process's) and symlink-resolved."""
    absolute = root if root.is_absolute() else session.cwd / root
    return absolute.resolve()


def _installed_package_roots(session: MasonSession) -> tuple[Path, ...]:
    """The roots of the four packages in the slab-stack distribution.

    Adds read-only reach into the harness's own source so ``read_file`` can
    answer questions the vocabulary alone cannot — 'does foundation.tasks
    have a cell relaxation task?' is one Read tool call, not thirty shell
    calls poking at sed and grep. In an editable install the roots are the
    ``src/<pkg>/`` directories; in a site-packages install they point there.
    Write scope is unchanged: the fence still refuses edits into the source.
    """
    import importlib.resources

    roots: list[Path] = []
    for name in ("slab", "foundation", "mason", "slab_stack"):
        try:
            root = Path(str(importlib.resources.files(name)))
        except (ModuleNotFoundError, TypeError):  # pragma: no cover - always installed
            continue
        if root.is_dir():
            roots.append(_scope_root(session, root))
    return tuple(roots)


def _mp_snapshot_root(session: MasonSession) -> Path | None:
    """The configured Materials Project snapshot root, or None.

    A configured snapshot gains read-only reach (the agent opens archived
    CIFs with ``read_file`` and hands their paths to scripts) and switches
    on the search tools. A broken slab.toml answers None here — the config
    error surfaces where config is read for real, not from fence assembly.
    """
    from slab.config import config_value
    from slab.errors import SlabError

    try:
        root = config_value("builders.mp.root", session.cwd)
    except SlabError:
        return None
    if root is None:
        return None
    return _scope_root(session, Path(str(root)))


#: Variables that hold model credentials whatever the config names: a
#: subprocess the model drives must never see them, or `env` in a shell
#: call puts the key into the tool result, the context, and the transcript.
_CREDENTIAL_VARS = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PORTKEY_API_KEY"})


def _subprocess_env(session: MasonSession) -> dict[str, str]:
    """The process environment minus the model's credentials.

    Examples:
        >>> import os, types
        >>> os.environ["SLAB_DOCTEST_SECRET"] = "sk-x"
        >>> fake = types.SimpleNamespace(agent=types.SimpleNamespace(
        ...     resolved_api_key_env="SLAB_DOCTEST_SECRET"))
        >>> "SLAB_DOCTEST_SECRET" in _subprocess_env(fake)
        False
        >>> del os.environ["SLAB_DOCTEST_SECRET"]
    """
    hidden = set(_CREDENTIAL_VARS)
    named = getattr(session.agent, "resolved_api_key_env", None)
    if named:
        hidden.add(str(named))
    return {key: value for key, value in os.environ.items() if key not in hidden}


class RunStoreUnavailable(Exception):
    """The workspace's run store could not be opened, and what to do about it.

    Raised in place of the bare ``StorageError`` or ``sqlite3`` error so the
    text the model reads carries the recovery. A database another process
    holds, or a filesystem refusing locks, is a fault of the workspace and
    not of the work: a real session read a bare "database is locked", spent
    an hour on byte-level forensics, and deleted the store's write-ahead
    log by hand.
    """

    def __init__(self, root: Path, cause: BaseException) -> None:
        super().__init__(
            f"the run store at {root} could not be opened ({cause}). This is a fault "
            f"of the workspace, not of your work: another process may hold the "
            f"database, or the filesystem may be refusing locks. Wait about a "
            f"minute and retry this call once. If it fails again, record the fault "
            f"in the notebook and finish with a report naming it. Do not inspect, "
            f"modify, or delete files under {root}."
        )


def _open_workspace(session: MasonSession) -> Workspace:
    """Open the session's workspace, or raise :class:`RunStoreUnavailable`."""
    from foundation.errors import FoundationError
    from foundation.runtime import Workspace

    try:
        return Workspace(session.workspace_root)
    except (FoundationError, sqlite3.Error) as e:
        raise RunStoreUnavailable(Path(session.workspace_root), e) from e


_STORE_MUTATION = re.compile(r"(?:^|[\s;|&(])(?:rm|rmdir|mv|truncate|shred|unlink|dd)\s")


def _store_mutation(command: str, workspace_root: Path) -> str | None:
    """The refusal when a shell command would rewrite the run store by hand.

    The store's database and its sidecar files (``runs.db`` and its
    ``-wal``, ``-shm``, and ``-journal`` companions) belong to SQLite: a
    session that cannot open them reports the fault, it does not repair
    them. The guard is a name match, not a parser: a deleting verb in a
    command that names the database or the workspace root. A copy made to
    inspect elsewhere passes.

    Examples:
        >>> root = Path("/ws")
        >>> _store_mutation("rm -f runs.db-wal runs.db-shm", root) is None
        False
        >>> _store_mutation("cd /ws && rm -rf cas/ab", root) is None
        False
        >>> _store_mutation("cp /ws/runs.db /tmp/copy.db", root) is None
        True
        >>> _store_mutation("rm -f build/*.o", root) is None
        True
    """
    if not _STORE_MUTATION.search(command):
        return None
    root = str(workspace_root)
    if "runs.db" not in command and root not in command:
        return None
    return (
        f"refused: this command would delete or move files of the run store under "
        f"{root}. The workspace is SLAB's record of every run, and its database is "
        f"never repaired by hand. If the store cannot be opened, wait about a "
        f"minute, retry the tool once, then report the fault with finish."
    )


def _tally_line(statuses: list[str], hits: int) -> str:
    """``3 completed, 1 running (2 cache hits)`` from a run's task statuses."""
    tally = Counter(statuses)
    order = ("completed", "running", "failed")
    parts = [f"{tally[status]} {status}" for status in order if tally.get(status)]
    parts += [f"{n} {status}" for status, n in sorted(tally.items()) if status not in order]
    line = ", ".join(parts) or "no tasks"
    if hits:
        line += f" ({hits} cache hit{'s' if hits != 1 else ''})"
    return line


def _progress(ws: Workspace, run_id: str) -> str:
    """One line of what a run has done so far: its task tally and its checks."""
    tasks = ws.runs.list_tasks(run_id)
    line = "tasks: " + _tally_line(
        [task.status.value for task in tasks], sum(1 for task in tasks if task.cache_hit)
    )
    checks = ws.runs.list_check_results(run_id)
    if checks:
        line += f"; checks: {sum(1 for c in checks if c.passed)}/{len(checks)} passed"
    return line


def _compact_details(details: dict[str, Any]) -> dict[str, Any]:
    """The run record with its finished tasks folded to one line each.

    A full record carries every task's recipe, inputs, and outputs. A
    labeling run of 88 tasks made that 190,000 characters, and a real
    session polled it six times, spending its context on setup lines.
    Checks, the run's failure record, and every field of a failed task
    stay verbatim, because they are what a correction is computed from.

    Examples:
        >>> details = {"run": {"id": "r"}, "checks": [], "tasks": [
        ...     {"seq": 1, "name": "single_point", "status": "completed",
        ...      "cache_hit": True, "duration_s": 0.0, "error": None, "failure": None,
        ...      "recipe": {"params": {"label": "rattle_T300_0"},
        ...                 "extra": {"setup": ["..."] * 32}},
        ...      "inputs": {"atoms": "sha256:..."}, "outputs": {}},
        ...     {"seq": 2, "name": "single_point", "status": "failed",
        ...      "cache_hit": False, "duration_s": 1.5, "error": "boom",
        ...      "failure": {"message": "boom"}, "recipe": {}, "inputs": {},
        ...      "outputs": {}}], "artifacts": [], "history": []}
        >>> compact = _compact_details(details)
        >>> compact["tasks_summary"]
        '1 completed, 1 failed (1 cache hit)'
        >>> sorted(compact["tasks"][0])
        ['cache_hit', 'duration_s', 'label', 'name', 'seq', 'status']
        >>> compact["tasks"][1]["failure"]
        {'message': 'boom'}
        >>> list(compact)
        ['run', 'checks', 'tasks_summary', 'tasks', 'artifacts', 'history', 'note']
    """
    tasks = list(details.get("tasks") or [])
    folded: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("status") == "failed" or task.get("error") or task.get("failure"):
            folded.append(task)
            continue
        line = {key: task.get(key) for key in ("seq", "name", "status", "cache_hit", "duration_s")}
        recipe = task.get("recipe")
        params = recipe.get("params") if isinstance(recipe, dict) else None
        label = params.get("label") if isinstance(params, dict) else None
        if label:
            line["label"] = label
        folded.append(line)
    summary = _tally_line(
        [str(task.get("status")) for task in tasks],
        sum(1 for task in tasks if task.get("cache_hit")),
    )
    compact: dict[str, Any] = {}
    for key, value in details.items():
        if key == "tasks":
            compact["tasks_summary"] = summary
            compact["tasks"] = folded
        else:
            compact[key] = value
    compact["note"] = (
        "finished tasks are folded to one line each; call show_run with full=true "
        "for their recipes, inputs, and outputs"
    )
    return compact


def _out_of_scope(session: MasonSession, path: Path, roots: tuple[Path, ...]) -> str | None:
    """The refusal observation when *path* leaves the file fence, else None.

    The fence is the sandbox principle at the tool layer: the agent reads and
    runs within its project, its workspace, and the skills it was shown, and
    writes only within the project and workspace. One carve-out inside the
    workspace: the sessions directory (transcripts and compaction files) is
    refused, because past sessions are not context. Comparison happens on
    fully resolved paths, so a symlink pointing out of the fence counts as
    outside it. This is a workflow control, not a security boundary — the
    shell tool remains the honest escape, behind its own gate.
    """
    if session.agent.file_scope == "anywhere":
        return None
    resolved = path.expanduser().resolve()
    # The sessions directory sits inside the workspace but is not context.
    # A past transcript is a losing substitute for durable state: it is
    # huge, it records what *seemed* true mid-investigation, and it may
    # describe a different campaign entirely. Everything a past session
    # kept on purpose arrives through the project files and the memories.
    # Resolved against resolved: a relative workspace root (the default
    # .slab) or a symlinked one would otherwise never match, and the fence
    # would silently open.
    if resolved.is_relative_to(session.sessions_dir.expanduser().resolve()):
        return (
            f"refused: {path} is a session transcript, and past sessions are "
            f"not context. What earlier sessions kept for you arrives three "
            f"ways: the goal text, the project files (BRIEF/PLAN/notebook), "
            f"and machine memories — call `recall`. A fact worth carrying "
            f"between sessions belongs in `remember`, not in a transcript."
        )
    for root in roots:
        if resolved.is_relative_to(root):
            return None
    listed = "\n".join(f"  {root}" for root in roots)
    return (
        f"refused: {path} is outside this session's file scope. This "
        f"operation works within these roots — retry with a path under one "
        f"of them:\n{listed}\n"
        f"Otherwise use the shell tool (approval-gated), or set "
        f"[agent] file_scope = \"anywhere\"."
    )


def _python_syntax_note(path: Path, content: str) -> str:
    """Post-write verification for Python files: a syntax check, immediately.

    Open models sometimes write ``\\n`` as literal text or truncate a file;
    running the broken script later wastes a whole round trip. ``compile``
    parses without executing.
    """
    if path.suffix != ".py":
        return ""
    try:
        compile(content, str(path), "exec")
    except SyntaxError as e:
        return (
            f"\nWARNING: the file does not parse as Python (line {e.lineno}: {e.msg}); "
            f"read it back and fix it before running"
        )
    return ""


def _add_file_tools(
    box: Toolbox,
    session: MasonSession,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...],
) -> None:
    def read_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, read_roots):
            return denied
        offset = int(arguments.get("offset", 1))
        limit = int(arguments.get("limit", _MAX_READ_LINES))
        if not path.is_file():
            return f"no such file: {path}"
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            return f"{path} looks binary ({len(raw)} bytes); read_file only reads text"
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        window = lines[max(offset - 1, 0) : max(offset - 1, 0) + limit]
        numbered = []
        for i, line in enumerate(window, start=max(offset, 1)):
            if len(line) > _MAX_LINE_CHARS:
                line = line[:_MAX_LINE_CHARS] + " [line truncated]"
            numbered.append(f"{i:6d}\t{line}")
        session.read_files.add(path)
        shown = "\n".join(numbered) if numbered else "(no lines in this window)"
        note = "" if len(lines) <= len(window) + offset - 1 else (
            f"\n[file has {len(lines)} lines; showing {max(offset,1)}"
            f"-{max(offset,1) + len(window) - 1}]"
        )
        return shown + note

    box.add(
        Tool(
            name="read_file",
            description=(
                "Read a text file with line numbers. Use offset/limit to window "
                "large files. You must read a file before you may edit it."
            ),
            parameters=_schema(
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based first line"},
                    "limit": {"type": "integer"},
                },
                ["path"],
            ),
            handler=read_file,
        )
    )

    def write_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, write_roots):
            return denied
        content = str(arguments["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        session.read_files.add(path)
        return f"wrote {len(content)} characters to {path}" + _python_syntax_note(path, content)

    box.add(
        Tool(
            name="write_file",
            description="Create or overwrite one text file with the given content.",
            parameters=_schema(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            handler=write_file,
            requires_approval=True,
        )
    )

    def edit_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, write_roots):
            return denied
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))
        if not path.is_file():
            return f"no such file: {path}"
        if path not in session.read_files:
            return f"read {path} with read_file before editing it (staleness guard)"
        if old == new:
            return "old_string and new_string are identical; nothing to do"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return (
                "no exact match for old_string — re-read the file; whitespace and "
                "line numbers from read_file output are not part of the text"
            )
        if count > 1 and not replace_all:
            return (
                f"old_string matches {count} places; extend it until it is unique, "
                f"or set replace_all: true"
            )
        updated = text.replace(old, new)
        path.write_text(updated, encoding="utf-8")
        return f"replaced {count if replace_all else 1} occurrence(s) in {path}" + (
            _python_syntax_note(path, updated)
        )

    box.add(
        Tool(
            name="edit_file",
            description=(
                "Exact-string replacement in a text file: old_string must match the "
                "current content exactly once (or set replace_all)."
            ),
            parameters=_schema(
                {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                ["path", "old_string", "new_string"],
            ),
            handler=edit_file,
            requires_approval=True,
        )
    )

    def list_dir(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments.get("path", ".")))
        if denied := _out_of_scope(session, path, read_roots):
            return denied
        if not path.is_dir():
            return f"no such directory: {path}"
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = []
        for entry in entries[:_MAX_DIR_ENTRIES]:
            try:
                if entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name}  ({entry.stat().st_size} B)")
            except OSError:  # dangling symlink: report it, don't fail the listing
                lines.append(f"{entry.name}  (unreadable: broken link?)")
        if len(entries) > _MAX_DIR_ENTRIES:
            lines.append(f"[... {len(entries) - _MAX_DIR_ENTRIES} more entries]")
        return "\n".join(lines) or "(empty directory)"

    box.add(
        Tool(
            name="list_dir",
            description="List one directory (directories end with '/').",
            parameters=_schema({"path": {"type": "string"}}, []),
            handler=list_dir,
        )
    )

    def search(arguments: dict[str, Any]) -> str:
        pattern = str(arguments["pattern"])
        root = _resolve(session, str(arguments.get("path", ".")))
        if denied := _out_of_scope(session, root, read_roots):
            return denied
        glob = str(arguments.get("glob", "*"))
        try:
            expression = re.compile(pattern)
        except re.error as e:
            return f"bad regex {pattern!r}: {e}"
        if not root.is_dir():
            return f"no such directory: {root}"
        matches: list[str] = []
        for candidate in sorted(root.rglob(glob)):
            if not candidate.is_file():
                continue
            # Hidden-dir filtering must look below *root* only: the project may
            # itself live under a dotted parent (~/.research/proj) and still
            # deserves search results.
            relative_parts = candidate.relative_to(root).parts
            if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
                continue
            # A symlink inside the project can point anywhere; the fence
            # that guards read_file guards what search prints too.
            if _out_of_scope(session, candidate, read_roots):
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    shown = line.strip()[:200]
                    matches.append(f"{candidate.relative_to(root)}:{number}: {shown}")
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        matches.append("[... more matches exist; narrow the pattern]")
                        return "\n".join(matches)
        return "\n".join(matches) or f"no matches for {pattern!r}"

    box.add(
        Tool(
            name="search",
            description="Regex search across text files under a directory (recursive).",
            parameters=_schema(
                {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "filename filter, e.g. *.py"},
                },
                ["pattern"],
            ),
            handler=search,
        )
    )


# A parallel launch spelled out in a command or script: mpirun/mpiexec/srun
# with an explicit rank count. The configured engines size their own
# launches; this catches the hand-written ones.
_RANK_FLAG = re.compile(r"\b(?:mpirun|mpiexec|srun)\b[^\n;|&]*?(?:-np|--ntasks|-n)[=\s]+(\d+)")


def _rank_overcommit(text: str) -> str | None:
    """A refusal when *text* asks for more MPI ranks than this session has.

    Guards the two surfaces that execute HERE (the shell and
    launch_workflow); submit_job is deliberately exempt, because its
    payload runs in its own allocation with its own budget. The refusal is
    a tool result the model reads and adapts to, never an exception.
    """
    from slab.hpc import cpu_budget

    requested = max(
        (int(m.group(1)) for m in _RANK_FLAG.finditer(text)), default=0
    )
    budget = cpu_budget()
    if requested > budget:
        return (
            f"refused: this launches {requested} MPI rank(s) but only {budget} "
            f"cpu(s) are usable in this session. Size the launch within that "
            f"budget — and prefer the configured engine (engine='qe' with "
            f"calculator_options) over a hand-written mpirun: it already "
            f"launches at the right width."
        )
    return None


# -- shell -------------------------------------------------------------------


def _add_shell_tool(box: Toolbox, session: MasonSession) -> None:
    def shell(arguments: dict[str, Any]) -> str:
        command = str(arguments["command"])
        if refused := _rank_overcommit(command):
            return refused
        if refused := _store_mutation(command, Path(session.workspace_root)):
            return refused
        timeout = min(
            float(arguments.get("timeout_s", session.agent.shell_timeout_s)),
            _MAX_SHELL_TIMEOUT_S,
        )
        # start_new_session puts the whole pipeline in its own process
        # group: on timeout, killing only the immediate /bin/sh would leave
        # pipeline stages, backgrounded children, or mpirun ranks running
        # detached while the model reads "timed out" as the command being
        # gone — so the timeout kills the group.
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=session.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=_subprocess_env(session),
            )
        except OSError as e:
            return f"could not start the command: {e}"
        import contextlib as _contextlib
        import os as _os
        import signal as _signal

        def _kill_group() -> None:
            with _contextlib.suppress(ProcessLookupError, PermissionError):  # raced exit
                _os.killpg(process.pid, _signal.SIGKILL)

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except KeyboardInterrupt:
            # Ctrl-C in the REPL reaches only this process now (the child
            # runs in its own session), so the interrupt must kill the
            # command tree itself or it keeps running detached.
            _kill_group()
            raise
        except subprocess.TimeoutExpired as e:
            _kill_group()
            # Reap the shell itself — wait() cannot block on pipes — but do
            # NOT communicate(): a child that escaped the group (setsid)
            # still holds the pipe ends, and reading to EOF would hang the
            # agent turn on a daemon that never exits. Abandon the pipes.
            with _contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5.0)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with _contextlib.suppress(OSError):
                        stream.close()
            raw_partial = e.stdout
            if isinstance(raw_partial, bytes):
                partial = raw_partial.decode(errors="replace")
            else:
                partial = raw_partial or ""
            return (
                f"command timed out after {timeout:.0f}s; the command and its "
                f"process group were killed; partial output:\n{partial}"
            )
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        output = completed.stdout + (
            f"\n[stderr]\n{completed.stderr}" if completed.stderr.strip() else ""
        )
        return f"exit {completed.returncode}\n{output.rstrip()}"

    box.add(
        Tool(
            name="shell",
            description=(
                "Run one shell command in the project directory (stdout+stderr, "
                "with the exit code). A timeout kills the command's whole process "
                "group — nohup and '&' do not survive it. Not for long "
                "calculations — use launch_workflow (background=true for long "
                "ones) or submit_job for those."
            ),
            parameters=_schema(
                {"command": {"type": "string"}, "timeout_s": {"type": "number"}},
                ["command"],
            ),
            handler=shell,
            requires_approval=True,
            gate=lambda arguments: not session.shell_allowlisted(
                str(arguments.get("command", ""))
            ),
        )
    )


# -- workflows (foundation) ---------------------------------------------------


def _add_workflow_tools(
    box: Toolbox, session: MasonSession, read_roots: tuple[Path, ...]
) -> None:
    def _run_line(run: Any) -> str:
        return (
            f"{run.id[:10]}  {run.state.value:<11} {run.status.value:<10} "
            f"{run.name[:24]:<24} {run.intent or ''}"
        )

    def _session_filter(raw: object) -> str | None:
        # 'this' names the current session, so the agent never has to know
        # its own id — a real transcript showed the guess being made.
        value = str(raw) if raw else None
        return session.session_id if value == "this" else value

    def list_runs(arguments: dict[str, Any]) -> str:
        state = arguments.get("state")
        status = arguments.get("status")
        if status is None and state in ("running", "completed", "failed"):
            # The two words are easy to swap; take the meaning, not the key.
            state, status = None, state
        limit = int(arguments.get("limit", 10))
        session_filter = _session_filter(arguments.get("session"))
        with _open_workspace(session) as ws:
            runs = ws.runs.list_runs(
                state=state, status=status, session=session_filter, limit=limit
            )
            if not runs:
                where = f" for session {session_filter!r}" if session_filter else ""
                return f"no runs in this workspace yet{where}"
            lines = [_run_line(run) for run in runs]
        return "\n".join(lines)

    box.add(
        Tool(
            name="list_runs",
            description=(
                "List SLAB runs in this workspace, newest first. Pass "
                "session='this' to see only runs this session created "
                "(a full id or unique prefix also works — the same filter "
                "'slab list --session' takes)."
            ),
            parameters=_schema(
                {
                    "state": {
                        "type": "string",
                        "description": "quarantined | verified | promoted | expired",
                    },
                    "status": {
                        "type": "string",
                        "description": "running | completed | failed",
                    },
                    "session": {
                        "type": "string",
                        "description": (
                            "'this' for the current session; or a session id "
                            "(unique prefix ok); omit for all runs"
                        ),
                    },
                    "limit": {"type": "integer"},
                },
                [],
            ),
            handler=list_runs,
        )
    )

    def _resolve_run(ws: Workspace, value: str) -> tuple[str, str]:
        """A run id or unique prefix, else this session's newest run of that name.

        The name is what the model remembers (a real transcript passed the
        script's name twice and was told no run matched), so it resolves
        too, with a note saying which run was taken.
        """
        from foundation.errors import RunNotFoundError, SessionNotFoundError

        try:
            return ws.runs.resolve(value), ""
        except RunNotFoundError as not_found:
            try:
                runs = ws.runs.list_runs(session=session.session_id, limit=200)
            except SessionNotFoundError:
                runs = []
            named = [run for run in runs if run.name == value]
            if not named:
                raise not_found from None
            run = named[0]  # newest first
            return run.id, (
                f"(resolved {value!r} by name to run {run.id[:10]}, this session's "
                f"newest run of that name)\n"
            )

    def show_run(arguments: dict[str, Any]) -> str:
        from foundation._ops import run_details

        with _open_workspace(session) as ws:
            run_id, note = _resolve_run(ws, str(arguments["run_id"]))
            details = run_details(ws, run_id)
        if not arguments.get("full"):
            details = _compact_details(details)
        return note + json.dumps(details, indent=1, ensure_ascii=False)

    box.add(
        Tool(
            name="show_run",
            description=(
                "One run's record: its fields, checks with observed/expected values, "
                "the task tally with finished tasks folded to one line each, failed "
                "tasks and failure records in full, artifacts, and history. Pass "
                "full=true for every task's recipe, inputs, and outputs. run_id "
                "takes an id, a unique prefix, or the name of a run this session "
                "created. Read this before correcting a failed run."
            ),
            parameters=_schema(
                {
                    "run_id": {"type": "string"},
                    "full": {
                        "type": "boolean",
                        "description": "include every task's recipe, inputs, and outputs",
                    },
                },
                ["run_id"],
            ),
            handler=show_run,
        )
    )

    def _launch_background(
        script: Path, name: str | None, intent: str | None, args: list[str]
    ) -> str:
        """Detach the run as its own process so no shell timeout can kill it."""
        import sys

        log_path = Path(script).with_suffix(".launch.log")
        command = [sys.executable, "-m", "foundation.cli", "run", str(script), *args]
        if name:
            command += ["--name", name]
        if intent:
            command += ["--intent", intent]
        command += ["--session", session.session_id, "-w", str(session.workspace_root)]
        try:
            with open(log_path, "ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=session.cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    # Block-buffered stdout reaches the log only at exit; a
                    # real session polled an empty log eight times while
                    # the run was labeling structures. Line by line instead.
                    env={**_subprocess_env(session), "PYTHONUNBUFFERED": "1"},
                )
        except OSError as e:
            return f"could not start the background run: {e}"
        return (
            f"launched in the background: pid {process.pid}, output -> {log_path}\n"
            f"the run appears in list_runs (session='this') once it starts; "
            f"block on it with wait_for_run instead of polling in shell."
        )

    def launch_workflow(arguments: dict[str, Any]) -> str:
        from foundation._ops import launch_script

        script = _resolve(session, str(arguments["script"]))
        if denied := _out_of_scope(session, script, read_roots):
            return denied
        try:
            script_text = Path(script).read_text(encoding="utf-8", errors="replace")
        except OSError:
            script_text = ""  # launch_script reports the unreadable file itself
        if refused := _rank_overcommit(script_text):
            return refused
        args = [str(a) for a in arguments.get("args") or []]
        if arguments.get("background"):
            return _launch_background(
                Path(script), arguments.get("name"), arguments.get("intent"), args
            )
        result = launch_script(
            session.workspace_root,
            script,
            name=arguments.get("name"),
            intent=arguments.get("intent"),
            session=session.session_id,
            argv=tuple(args),
            capture_output=True,
        )
        lines = [
            f"run {result['run_id']}: state={result['state']} status={result['status']} "
            f"checks={result['checks_passed']}/{result['checks_total']} "
            f"tasks={result['tasks_recorded']}"
        ]
        if result.get("failure"):
            lines.append("failure record:")
            lines.append(json.dumps(result["failure"], indent=1, ensure_ascii=False))
        elif result.get("traceback"):
            lines.append(str(result["traceback"]))
        output = str(result.get("output") or "").rstrip()
        if output:
            lines.append(f"script output:\n{output}")
        return "\n".join(lines)

    box.add(
        Tool(
            name="launch_workflow",
            description=(
                "Execute a SLAB workflow script (plain Python with @task calls and "
                "@check verification) as a traced run. This is how calculations "
                "run: results get provenance, caching, and verification gates. "
                "For work longer than a few minutes, pass background=true: the "
                "run detaches from this process (no tool timeout can kill it) "
                "and wait_for_run blocks until it finishes."
            ),
            parameters=_schema(
                {
                    "script": {"type": "string", "description": "path to the workflow script"},
                    "name": {"type": "string"},
                    "intent": {"type": "string", "description": "why this run exists"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arguments passed to the script (sys.argv[1:])",
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "detach and return immediately; follow with wait_for_run"
                        ),
                    },
                },
                ["script"],
            ),
            handler=launch_workflow,
            requires_approval=True,
        )
    )

    def wait_for_run(arguments: dict[str, Any]) -> str:
        import time

        from foundation.errors import SessionNotFoundError

        wanted = arguments.get("run_id")
        timeout = min(float(arguments.get("timeout_s", 900.0)), _MAX_WAIT_TIMEOUT_S)
        deadline = time.monotonic() + timeout
        # A background launch takes a moment to register its run, so an
        # empty workspace gets a grace window before "nothing is running"
        # counts as an answer.
        grace_until = time.monotonic() + min(_WAIT_GRACE_S, timeout)
        run_id: str | None = None
        note = ""
        while True:
            with _open_workspace(session) as ws:
                if wanted:
                    if run_id is None:
                        run_id, note = _resolve_run(ws, str(wanted))
                    run = ws.runs.get(run_id)
                    if run.status.value != "running":
                        return (
                            f"{note}run {run.id}: state={run.state.value} "
                            f"status={run.status.value}; {_progress(ws, run.id)}; "
                            f"read it with show_run"
                        )
                    running = [run]
                else:
                    try:
                        runs = ws.runs.list_runs(session=session.session_id, limit=50)
                    except SessionNotFoundError:
                        runs = []
                    running = [r for r in runs if r.status.value == "running"]
                    if not running and time.monotonic() >= grace_until:
                        if not runs:
                            return (
                                "this session has no runs yet; launch one first "
                                "(a fresh background launch needs a few seconds "
                                "to register)"
                            )
                        finished = "\n".join(_run_line(r) for r in runs[:10])
                        return f"no run of this session is running; the record:\n{finished}"
                if time.monotonic() >= deadline:
                    # The tally answers "is it moving?" without a show_run
                    # of the whole record: a real session queried the
                    # database by hand for exactly this count.
                    lines = "\n".join(
                        f"{r.id[:10]}  {r.name}  running; {_progress(ws, r.id)}"
                        for r in running
                    ) or "(none registered yet)"
                    return (
                        f"{note}still running after {timeout:.0f}s:\n{lines}\n"
                        f"call wait_for_run again to keep waiting"
                    )
            time.sleep(min(_WAIT_POLL_S, max(0.05, deadline - time.monotonic())))

    box.add(
        Tool(
            name="wait_for_run",
            description=(
                "Block until a run finishes (or the timeout passes), then report "
                "its state and task tally. run_id takes an id, a unique prefix, or "
                "the name of a run this session created; without it, waits for "
                "every running run this session created — the partner of "
                "launch_workflow background=true. One call replaces a chain of "
                "sleep-and-poll shell commands, and the timeout answer says how "
                "far each run has got."
            ),
            parameters=_schema(
                {
                    "run_id": {
                        "type": "string",
                        "description": "run id or unique prefix; omit for this session's runs",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "seconds to wait before reporting back (default 900)",
                    },
                },
                [],
            ),
            handler=wait_for_run,
        )
    )


# -- engines (slab) -----------------------------------------------------------


def _add_engine_tools(box: Toolbox, session: MasonSession) -> None:
    def list_engines(arguments: dict[str, Any]) -> str:
        from slab._ops import engines_overview

        return json.dumps(engines_overview(), indent=1, ensure_ascii=False)

    box.add(
        Tool(
            name="list_engines",
            description=(
                "What can be computed here: engines, QE protocols, pseudopotential "
                "families, HPC partitions. Call this BEFORE choosing an engine — "
                "there is no in-process MLIP fallback, so the available checkpoint "
                "ids are the entire runnable-MLIP surface on this machine "
                "(training a new one is the train_potential task, not an engine)."
            ),
            parameters=_schema({}, []),
            handler=list_engines,
        )
    )

    def list_tasks(arguments: dict[str, Any]) -> str:
        entries = [f"{name}({sig}) — {summary}" for name, sig, summary in _catalog_tasks()]
        return "\n".join(entries)

    box.add(
        Tool(
            name="list_tasks",
            description=(
                "The traced tasks foundation.tasks exposes to workflow scripts. "
                "One line per task: name, signature, one-sentence summary. Call "
                "describe_task for the full docstring."
            ),
            parameters=_schema({}, []),
            handler=list_tasks,
        )
    )

    def describe_task(arguments: dict[str, Any]) -> str:
        name = str(arguments.get("name", "")).strip()
        if not name:
            raise ValueError("describe_task requires 'name' — call list_tasks to see them")
        catalog = {entry[0]: entry for entry in _catalog_tasks()}
        if name not in catalog:
            raise ValueError(f"no task {name!r}; known: {', '.join(sorted(catalog))}")
        from foundation import tasks as _tasks

        function = getattr(_tasks, name)
        _, signature, _ = catalog[name]
        doc = (function.__doc__ or "").strip() or "(no docstring)"
        return f"{name}({signature})\n\n{doc}"

    box.add(
        Tool(
            name="describe_task",
            description=(
                "Full signature and docstring of one foundation.tasks task, so an "
                "agent can consult the harness's own vocabulary instead of reading "
                "its source through shell."
            ),
            parameters=_schema(
                {"name": {"type": "string", "description": "e.g. 'relax'"}},
                ["name"],
            ),
            handler=describe_task,
        )
    )


def _add_mp_tools(box: Toolbox, snapshot_root: Path) -> None:
    """The offline Materials Project snapshot ([builders.mp] is configured).

    Search and lookup only: the traced route from a material id to a
    structure is ``foundation.tasks.fetch_structure`` inside a workflow
    script. All three tools are read-only by construction, and they bind
    the root resolved from the session's own project config — the process
    cwd may be elsewhere.
    """

    def search_materials(arguments: dict[str, Any]) -> str:
        from slab.mp import search_materials as mp_search

        rows = mp_search(
            arguments.get("filters") or {},
            columns=arguments.get("columns"),
            limit=int(arguments.get("limit", 20)),
            order_by=arguments.get("order_by"),
            root=snapshot_root,
        )
        return json.dumps(rows, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="search_materials",
            description=(
                "Search the offline Materials Project snapshot's materials table "
                "(local, read-only; there is no online fallback). filters maps "
                "keys to values: 'elements' (all must be present) and "
                "'exclude_elements' take element lists; other keys are columns, "
                "bare for equality (null matches SQL NULL) or suffixed "
                "__lte/__gte/__lt/__gt/__ne — e.g. {\"elements\": [\"Fe\"], "
                "\"energy_above_hull__lte\": 0.025}. A wrong column name is "
                "refused with the real column list. Search first and fetch one "
                "structure second (fetch_structure in a workflow); never "
                "enumerate the cifs/ tree. NULL means not populated, never "
                "zero. Report results as (snapshot release, material_id)."
            ),
            parameters=_schema(
                {
                    "filters": {"type": "object"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "description": "1-500, default 20"},
                    "order_by": {
                        "type": "string",
                        "description": "column name; leading - for descending",
                    },
                },
                [],
            ),
            handler=search_materials,
        )
    )

    def get_material(arguments: dict[str, Any]) -> str:
        from slab.mp import get_material as mp_get

        record = mp_get(str(arguments["material_id"]), root=snapshot_root)
        return json.dumps(record, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="get_material",
            description=(
                "One material's full metadata record from the snapshot: the "
                "materials row, its elements, and cif_file — the absolute path "
                "of its archived CIF, readable here or via "
                "fetch_structure(material_id) in a workflow script. Absence is "
                "absence: an id the snapshot lacks is an error, not a reason "
                "to look elsewhere."
            ),
            parameters=_schema(
                {"material_id": {"type": "string", "description": "e.g. 'mp-149'"}},
                ["material_id"],
            ),
            handler=get_material,
        )
    )

    def query_materials(arguments: dict[str, Any]) -> str:
        from slab.mp import query_materials as mp_query

        result = mp_query(
            str(arguments["sql"]),
            limit=int(arguments.get("limit", 200)),
            root=snapshot_root,
        )
        return json.dumps(result, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="query_materials",
            description=(
                "One read-only SELECT (or WITH) over the snapshot's "
                "metadata.sqlite, for queries the search_materials filters "
                "cannot express. Tables: materials (keyed by material_id), "
                "material_elements(material_id, element), dataset_info, units "
                "(consult it instead of guessing units). Rows are capped and "
                "the result says when it truncated — put LIMIT in the query."
            ),
            parameters=_schema(
                {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "description": "row cap, default 200"},
                },
                ["sql"],
            ),
            handler=query_materials,
        )
    )


def _catalog_tasks() -> list[tuple[str, str, str]]:
    """(name, signature, first-line summary) for every public foundation task.

    Public = not underscore-prefixed and callable with `@task` applied, i.e.
    every symbol foundation.tasks exports that a workflow script may name.
    """
    import inspect

    from foundation import tasks as _tasks

    entries: list[tuple[str, str, str]] = []
    for name in sorted(vars(_tasks)):
        if name.startswith("_"):
            continue
        obj = getattr(_tasks, name)
        if not callable(obj) or not getattr(obj, "__doc__", None):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        # A traced task decorated with @task wraps the underlying function;
        # signature() correctly returns the wrapped signature. Skip helpers
        # by requiring the function be defined in the tasks module itself.
        if getattr(obj, "__module__", "") != _tasks.__name__:
            continue
        summary = (obj.__doc__ or "").strip().splitlines()[0]
        entries.append((name, _short_signature(sig), summary))
    return entries


def _short_signature(sig: Any) -> str:
    """Signature rendering the agent can read: drop annotations, keep names."""
    parts: list[str] = []
    for param in sig.parameters.values():
        kind = param.kind
        if kind is param.VAR_POSITIONAL:
            parts.append(f"*{param.name}")
            continue
        if kind is param.VAR_KEYWORD:
            parts.append(f"**{param.name}")
            continue
        if kind is param.KEYWORD_ONLY and "*" not in parts:
            parts.append("*")
        default = "" if param.default is param.empty else f"={param.default!r}"
        parts.append(f"{param.name}{default}")
    return ", ".join(parts)


# -- hpc ---------------------------------------------------------------------


def _add_hpc_tools(box: Toolbox, session: MasonSession) -> None:
    def submit_job(arguments: dict[str, Any]) -> str:
        from slab.hpc import render_sbatch, submit

        name = str(arguments["name"])
        command = str(arguments["command"])
        partition, _spec = session.hpc.resolve_partition(arguments.get("partition"))
        # Scripts and SLURM .out files live under the workspace so
        # 'slab purge' can sweep them; the prologue cd keeps the
        # payload running in the project directory as before (the .out is
        # opened before the cd, so it stays in jobs/).
        script = render_sbatch(
            command,
            job_name=name,
            partition=partition,
            config=session.hpc,
            time_limit=arguments.get("time_limit"),
            # The exported session id stamps every run the job launches, so a
            # batch result joins the chat that asked for it. The export is
            # explicit rather than inherited: a cluster may submit with
            # --export=NONE.
            prologue=(
                f"export SLAB_SESSION={shlex.quote(session.session_id)}",
                f"cd {shlex.quote(str(session.cwd))}",
            ),
        )
        jobs_dir = session.workspace_root / "jobs"
        job = submit(script, job_name=name, partition=partition, directory=jobs_dir)
        return (
            f"submitted job {job.job_id} ({job.job_name}) to partition {job.partition}; "
            f"script kept at {job.script_path}; poll with job_status"
        )

    box.add(
        Tool(
            name="submit_job",
            description=(
                "Submit a command as a SLURM batch job (typically 'slab run "
                "workflow.py' so the result is still a traced, verified run)."
            ),
            parameters=_schema(
                {
                    "command": {"type": "string"},
                    "name": {"type": "string"},
                    "partition": {"type": "string"},
                    "time_limit": {"type": "string", "description": "HH:MM:SS"},
                },
                ["command", "name"],
            ),
            handler=submit_job,
            requires_approval=True,
        )
    )

    def job_status(arguments: dict[str, Any]) -> str:
        from slab.hpc import job_state

        status = job_state(str(arguments["job_id"]))
        pieces = [f"job {status.job_id}: {status.state.value}"]
        if status.raw and status.raw != status.state.value.upper():
            pieces.append(f"({status.raw})")
        if status.detail:
            pieces.append(status.detail)
        return " ".join(pieces)

    box.add(
        Tool(
            name="job_status",
            description="State of one SLURM job (pending/running/completed/failed/...).",
            parameters=_schema({"job_id": {"type": "string"}}, ["job_id"]),
            handler=job_status,
        )
    )

    def cancel_job(arguments: dict[str, Any]) -> str:
        from slab.hpc import cancel

        cancel(str(arguments["job_id"]))
        return f"cancel requested for job {arguments['job_id']}"

    box.add(
        Tool(
            name="cancel_job",
            description="Cancel a SLURM job (a no-op if it already finished).",
            parameters=_schema({"job_id": {"type": "string"}}, ["job_id"]),
            handler=cancel_job,
            requires_approval=True,
        )
    )


# -- delegation ---------------------------------------------------------------


def _add_delegate_tool(
    box: Toolbox,
    session: MasonSession,
    spec: AgentSpec,
    roster: dict[str, AgentSpec],
    skills: dict[str, Skill],
    parent_client: Any | None,
) -> None:
    def delegate(arguments: dict[str, Any]) -> str:
        # Local imports: tools must not import the loop at module scope
        # (the loop imports tools), and delegation is the one place a tool
        # spins a loop of its own.
        from mason.config import override_agent, roster_agent_config
        from mason.loop import Mason, client_from_config, connection_profile
        from mason.roster import hands

        name = str(arguments["agent"])
        team = hands(spec, roster)
        others = ", ".join(team)
        if name == spec.name:
            return f"you cannot delegate to yourself; your team: {others}"
        target = team.get(name)
        if target is None:
            if name in roster:
                return f"{name} leads a group of its own and takes no briefs; your team: {others}"
            return f"no agent named {name!r}; your team: {others}"
        task = str(arguments["task"])
        context = arguments.get("context")
        # The child derives from the *base* config so the entry agent's own
        # [agent.roster] table never leaks into a specialist; CLI flags are
        # re-asserted on top because a flag outranks config for everyone.
        effective = roster_agent_config(session.base_agent, name)
        if session.flag_updates:
            effective = override_agent(effective, dict(session.flag_updates))
        child_session = session.spawn(name, effective)
        reuse = parent_client is not None and connection_profile(
            child_session.agent
        ) == connection_profile(session.agent)
        client = (
            parent_client
            if reuse
            else client_from_config(child_session.agent, child_session.api_keys)
        )
        child = Mason(
            child_session, client=client, skills=skills, spec=target, roster=roster, depth=1
        )
        brief = task if not context else f"{task}\n\nContext from {spec.name}:\n{context}"
        result = child.run_turn(brief)
        session.record(
            {
                "type": "delegate",
                "agent": name,
                "task": task,
                "transcript": child_session.transcript_path.name,
                "stop": result.stop_reason,
                "steps": result.steps,
            }
        )
        footer = (
            f"[{name}: {result.stop_reason} after {result.steps} step(s); "
            f"tokens {child_session.prompt_tokens}+{child_session.completion_tokens}; "
            f"transcript {child_session.transcript_path.name}]"
        )
        return f"{result.text}\n\n{footer}"

    box.add(
        Tool(
            name="delegate",
            description=(
                "Hand one scoped task to a specialist from your team. The "
                "specialist runs its own tool loop against the shared workspace "
                "and notebook, and you receive its final report. Brief it with "
                "the goal, the constraints (engine, protocol, budget), and what "
                "to return; its report ends with a bracketed harness line "
                "stating how it stopped."
            ),
            parameters=_schema(
                {
                    "agent": {"type": "string", "description": "a name from Your team"},
                    "task": {
                        "type": "string",
                        "description": "the scoped goal, self-contained and checkable",
                    },
                    "context": {
                        "type": "string",
                        "description": "optional background the task needs",
                    },
                },
                ["agent", "task"],
            ),
            handler=delegate,
        )
    )


# -- skills ------------------------------------------------------------------


def _add_skill_tool(box: Toolbox, session: MasonSession, skills: dict[str, Skill]) -> None:
    def skill_tool(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        found = skills.get(name)
        if found is None:
            known = ", ".join(sorted(skills))
            return f"no skill named {name!r}; available skills: {known}"
        # The digest names the revision that loaded, so a benchmark flag
        # raised on this campaign is attributable to one revision of the skill.
        session.record(
            {"type": "skill", "name": name, "source": found.source, "digest": found.digest}
        )
        return listing(found)

    box.add(
        Tool(
            name="skill",
            description=(
                "Load a skill by name: returns its full instructions, its root "
                "path, and its bundled files (scripts, references, assets). Call "
                "this before doing a task a listed skill covers, and prefer its "
                "bundled scripts over writing your own."
            ),
            parameters=_schema({"name": {"type": "string"}}, ["name"]),
            handler=skill_tool,
        )
    )


# -- project memory: the notebook and the plan --------------------------------


def _add_memory_tools(box: Toolbox, session: MasonSession) -> None:
    def notebook(arguments: dict[str, Any]) -> str:
        session.notebook_append(
            str(arguments["entry"]),
            heading=arguments.get("heading"),
        )
        return f"recorded in {session.notebook_path.name}"

    box.add(
        Tool(
            name="notebook",
            description=(
                "Append an entry to the lab notebook (NOTEBOOK.md): decisions, "
                "results with run ids, failures and their diagnosis. The notebook "
                "outlives the context window — write it as if for a colleague."
            ),
            parameters=_schema(
                {"entry": {"type": "string"}, "heading": {"type": "string"}},
                ["entry"],
            ),
            handler=notebook,
        )
    )

    def plan(arguments: dict[str, Any]) -> str:
        content = str(arguments["content"]).rstrip() + "\n"
        session.plan_path.write_text(content, encoding="utf-8")
        return f"PLAN.md updated:\n{content}"

    box.add(
        Tool(
            name="plan",
            description=(
                "Rewrite the living plan (PLAN.md): goal, steps with status, open "
                "questions. Keep it current — it is re-read at session start and "
                "after compaction."
            ),
            parameters=_schema({"content": {"type": "string"}}, ["content"]),
            handler=plan,
        )
    )


# -- machine memory: what this machine taught an earlier session --------------


def _add_machine_memory_tools(box: Toolbox, session: MasonSession) -> None:
    """``recall`` and ``remember``: the store in :mod:`foundation.memory`.

    The notebook holds the project's record; this holds the machine's. Both
    are memory, and the split is the scope: a fact about how software behaves
    here outlives the project that discovered it, so it must not be buried in
    one project's notebook.
    """

    def recall(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        memories = memory_store.discover()
        found = memories.get(name)
        if found is None:
            known = ", ".join(sorted(memories)) or "none recorded yet"
            return f"no memory named {name!r}; memories on this machine: {known}"
        session.record({"type": "recall", "name": name})
        answer = f"{found.body().rstrip()}\n\n[{found.provenance()}]"
        changed = found.drift(session.software_versions()) if found.against else []
        if changed:
            answer += (
                f"\n[changed since: {'; '.join(changed)}. Confirm the fact before "
                f"you build on it; remember it again once you have.]"
            )
        return answer

    box.add(
        Tool(
            name="recall",
            description=(
                "Read one memory in full by name. The catalog under '# Memory' "
                "lists what this machine knows; each line is a summary, and this "
                "returns the fact itself with who recorded it and when."
            ),
            parameters=_schema({"name": {"type": "string"}}, ["name"]),
            handler=recall,
        )
    )

    def remember(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        description = str(arguments["description"])
        body = str(arguments["body"])
        try:
            written = memory_store.write(
                name,
                description,
                body,
                agent=session.agent_name,
                model=session.agent.model,
                # Stamped with the software the text names, at today's
                # versions, so a later session can tell whether it still holds.
                against=memory_store.stamp(f"{description}\n{body}", session.software_versions()),
            )
        except MemoryStoreError as e:
            # A refusal is an observation the model can act on, not a crash:
            # the message says which rule stopped the write.
            return f"not recorded: {e}"
        session.record({"type": "remember", "name": name, "path": str(written.path)})
        answer = (
            f"recorded as memory {written.name!r} in {written.path}; "
            f"every later session on this machine reads it"
        )
        if written.against:
            stamped = ", ".join(f"{n} {v}" for n, v in written.against.items())
            answer += f" (stamped against {stamped})"
        return answer

    box.add(
        Tool(
            name="remember",
            description=(
                "Record one confirmed fact about this machine or its software so "
                "later sessions start knowing it: a package that behaves unlike "
                "its documentation, a flag that matters, a workaround. Write the "
                "description as the line a future session reads when deciding "
                "whether the fact applies. Re-using a name replaces that memory, "
                "which is how you consolidate. Not for results (they belong to "
                "runs), project decisions (the notebook), or credentials (nowhere)."
            ),
            parameters=_schema(
                {
                    "name": {
                        "type": "string",
                        "description": "lowercase-with-hyphens, e.g. 'vllm-mamba-cache'",
                    },
                    "description": {
                        "type": "string",
                        "description": "one line: the fact and when it applies",
                    },
                    "body": {"type": "string", "description": "the fact in full"},
                },
                ["name", "description", "body"],
            ),
            handler=remember,
            requires_approval=True,
        )
    )
