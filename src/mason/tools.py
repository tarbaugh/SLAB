"""Mason's tools: few, orthogonal, with crisp machine-checkable failure modes.

The tool set follows the SWE-agent lesson that agent performance is driven
by interface design, not tool count: six file/shell primitives whose error
messages teach recovery, plus the tools that make Mason a *research* agent —
SLAB runs (list/show/launch), the SLURM plumbing when this machine has
partitions configured, and the two memory instruments (notebook, plan).

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
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mason.roster import AgentSpec

from mason.client import ToolCall
from mason.session import MasonSession
from mason.skills import Skill, discover_skills, listing

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
        "list_engines",
        "submit_job",
        "job_status",
        "cancel_job",
        "notebook",
        "plan",
        "skill",
        "finish",
    }
)

_MAX_READ_LINES = 400
_MAX_LINE_CHARS = 500
_MAX_DIR_ENTRIES = 200
_MAX_SEARCH_MATCHES = 100
_MAX_SHELL_TIMEOUT_S = 600.0

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
        preview = _preview(call)
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
    for key in ("path", "script", "name", "partition", "old_string", "new_string", "content"):
        if key in arguments:
            value = str(arguments[key])
            # content is the thing being approved — a human shown 120 chars
            # of a workflow script is approving blind. Head and tail, with
            # the elision announced, is enough to actually read it.
            limit = 1200 if key == "content" else 120
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
) -> Toolbox:
    """Every tool this session gets; SLURM tools only where partitions exist.

    *spec* is the agent card in force: its ``tools`` allowlist narrows the
    box (``finish`` always stays), and ``None`` means no narrowing. *depth*
    is the delegation depth: a delegated agent (depth > 0) loses ``plan``,
    because ``PLAN.md`` belongs to the turn owner. *skills* is the skill
    catalog for this agent (the ``skill`` tool appears only when it is
    non-empty); ``None`` discovers the full catalog from the session's
    project directory.
    """
    if skills is None:
        skills = discover_skills(session.cwd)
    box = Toolbox(session)
    _add_file_tools(box, session)
    _add_shell_tool(box, session)
    _add_workflow_tools(box, session)
    _add_engine_tools(box, session)
    if session.hpc.partitions:
        _add_hpc_tools(box, session)
    _add_memory_tools(box, session)
    if skills:
        _add_skill_tool(box, session, skills)
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
                "number; list what was verified and what remains open."
            ),
            parameters=_schema({"report": {"type": "string"}}, ["report"]),
            handler=lambda arguments: str(arguments.get("report", "")),
        )
    )
    return box


# -- file primitives ---------------------------------------------------------


def _resolve(session: MasonSession, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else session.cwd / candidate


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


def _add_file_tools(box: Toolbox, session: MasonSession) -> None:
    def read_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
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


# -- shell -------------------------------------------------------------------


def _add_shell_tool(box: Toolbox, session: MasonSession) -> None:
    def shell(arguments: dict[str, Any]) -> str:
        command = str(arguments["command"])
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
                "with the exit code). Not for long calculations — use launch_workflow "
                "or submit_job for those."
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


def _add_workflow_tools(box: Toolbox, session: MasonSession) -> None:
    def list_runs(arguments: dict[str, Any]) -> str:
        from foundation.runtime import Workspace

        state = arguments.get("state")
        limit = int(arguments.get("limit", 10))
        with Workspace(session.workspace_root) as ws:
            runs = ws.runs.list_runs(state=state, limit=limit)
            if not runs:
                return "no runs in this workspace yet"
            lines = [
                f"{run.id[:10]}  {run.state.value:<11} {run.status.value:<10} "
                f"{run.name[:24]:<24} {run.intent or ''}"
                for run in runs
            ]
        return "\n".join(lines)

    box.add(
        Tool(
            name="list_runs",
            description="List SLAB runs in this workspace, newest first.",
            parameters=_schema(
                {
                    "state": {
                        "type": "string",
                        "description": "quarantined | verified | promoted | expired",
                    },
                    "limit": {"type": "integer"},
                },
                [],
            ),
            handler=list_runs,
        )
    )

    def show_run(arguments: dict[str, Any]) -> str:
        from foundation._ops import run_details
        from foundation.runtime import Workspace

        with Workspace(session.workspace_root) as ws:
            details = run_details(ws, str(arguments["run_id"]))
        return json.dumps(details, indent=1, ensure_ascii=False)

    box.add(
        Tool(
            name="show_run",
            description=(
                "Everything about one run: checks with observed/expected values, "
                "tasks, artifacts, failure records, history. Read this before "
                "correcting a failed run."
            ),
            parameters=_schema({"run_id": {"type": "string"}}, ["run_id"]),
            handler=show_run,
        )
    )

    def launch_workflow(arguments: dict[str, Any]) -> str:
        from foundation._ops import launch_script

        script = _resolve(session, str(arguments["script"]))
        result = launch_script(
            session.workspace_root,
            script,
            name=arguments.get("name"),
            intent=arguments.get("intent"),
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
                "run: results get provenance, caching, and verification gates."
            ),
            parameters=_schema(
                {
                    "script": {"type": "string", "description": "path to the workflow script"},
                    "name": {"type": "string"},
                    "intent": {"type": "string", "description": "why this run exists"},
                },
                ["script"],
            ),
            handler=launch_workflow,
            requires_approval=True,
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
                "families, HPC partitions."
            ),
            parameters=_schema({}, []),
            handler=list_engines,
        )
    )


# -- hpc ---------------------------------------------------------------------


def _add_hpc_tools(box: Toolbox, session: MasonSession) -> None:
    def submit_job(arguments: dict[str, Any]) -> str:
        from slab.hpc import render_sbatch, submit

        name = str(arguments["name"])
        command = str(arguments["command"])
        partition, _spec = session.hpc.resolve_partition(arguments.get("partition"))
        script = render_sbatch(
            command,
            job_name=name,
            partition=partition,
            config=session.hpc,
            time_limit=arguments.get("time_limit"),
        )
        job = submit(script, job_name=name, partition=partition, directory=session.cwd)
        return (
            f"submitted job {job.job_id} ({job.job_name}) to partition {job.partition}; "
            f"script kept at {job.script_path}; poll with job_status"
        )

    box.add(
        Tool(
            name="submit_job",
            description=(
                "Submit a command as a SLURM batch job (typically 'foundation run "
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


# -- skills ------------------------------------------------------------------


def _add_skill_tool(box: Toolbox, session: MasonSession, skills: dict[str, Skill]) -> None:
    def skill_tool(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        found = skills.get(name)
        if found is None:
            known = ", ".join(sorted(skills))
            return f"no skill named {name!r}; available skills: {known}"
        session.record({"type": "skill", "name": name, "source": found.source})
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


# -- memory ------------------------------------------------------------------


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
