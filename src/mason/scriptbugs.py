"""Telling a bug in the agent's script from a failure of the science.

A campaign writes Python. A workflow script, an analysis script, a skill
script: the agent wrote them, and they carry the ordinary faults of code
written quickly. A ``NameError`` in the post-processing of an MD run is
not a physics question, and the specialist that owns the physics is the
wrong reader for it. This module says which failures are which, so the
harness can hand the code to the coding helper and leave the science
where it belongs (:mod:`mason.tools`, the ``script-bug-handoff``
mechanism).

The rule, stated once:

* The result must carry a Python traceback. A real run carries one on its
  ``failure`` record, and a run whose failure escaped the run context (and
  a dry run that stopped before its end) carries the raw ``traceback``.
* A real run must have ended at status ``failed``, and a dry run must not
  have reached its end.
* The exception type must not be one of :data:`NOT_A_SCRIPT_BUG`: an
  engine that ran and failed, a pseudopotential family that is missing, a
  slice that was refused, a timeout, an interrupt, a ``sys.exit``.
* The failure record must not carry the note a task attaches when it kept
  an engine's files (:data:`ENGINE_FILES_NOTE`). An engine that ran and
  wrote output is an engine question, whatever exception type carried it.

Everything else with a traceback is a script bug, the exceptions SLAB's
own tasks raise for wrong arguments included. A ``ValueError`` from a
guard in :mod:`foundation.tasks`, and a ``TypeError`` for a keyword the
task does not take, say that the script called the API wrongly, which is
what the coding helper fixes.

Three failures carry no traceback at all and are therefore never a script
bug: a ``@check`` that returned False (the run completes and records the
check), a run a cancel or a reap marked failed (an error line, no
traceback), and a job that ended under its own scheduler.

The shell tool gets a narrower rule, because its output is arbitrary
text. A shell result counts only when the command names a ``.py`` file,
the exit status is not zero, and the output holds a traceback block whose
exception passes the same type test.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The exception types a script bug is not. An engine that ran and failed
#: is a science or an engine question and stays with the specialist; the
#: rest are the machine, the clock, or the person, and no edit to the
#: script fixes them. Matched on the class name alone, which is what a
#: failure record stores.
NOT_A_SCRIPT_BUG: frozenset[str] = frozenset(
    {
        # An engine, a builder, or a scheduler ran and refused.
        "LammpsScriptError",
        "BuilderError",
        "BuilderNotAvailableError",
        "EngineNotAvailableError",
        "PseudoFamilyError",
        "ProtocolError",
        "SchedulerError",
        "SchedulerNotAvailableError",
        "JobSizeError",
        "QeToolError",
        # The store failed, which no edit to the script repairs.
        "StorageError",
        "SchemaVersionError",
        # ASE's own names for a calculator that ran and failed.
        "CalculationFailed",
        "ReadError",
        "BadConfiguration",
        # The slice, the clock, the person, and a deliberate exit.
        "ResourcesError",
        "TimeoutError",
        "KeyboardInterrupt",
        "SystemExit",
        "CancelledError",
        "ScriptExitError",
    }
)

#: What opens a Python traceback block.
TRACEBACK_HEAD = "Traceback (most recent call last):"

_FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)', re.MULTILINE)
_EXCEPTION = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)\s*:\s?(?P<message>.*)$")
#: A ``.py`` argument in a shell command line.
_PY_ARGUMENT = re.compile(r"(?<![\w.])((?:[\w./~+-]*/)?[\w.+-]+\.py)(?![\w.])")


@dataclass(frozen=True)
class ScriptBug:
    """One Python failure of code the agent wrote.

    ``script`` is the path the tool call named, so two failures of one
    script are recognised as the same script whatever the run is called.
    ``line`` is the line in that file the traceback blames, when the
    traceback holds a frame in it.
    """

    exception: str
    message: str
    script: str
    line: int | None
    traceback: str

    @property
    def where(self) -> str:
        """The script and the line, in the words a note uses.

        Examples:
            >>> ScriptBug("NameError", "x", "/p/md.py", 12, "").where
            'md.py line 12'
            >>> ScriptBug("NameError", "x", "/p/md.py", None, "").where
            'md.py'
        """
        name = Path(self.script).name
        return f"{name} line {self.line}" if self.line is not None else name

    @property
    def headline(self) -> str:
        """The exception and its message, in one line.

        Examples:
            >>> ScriptBug("NameError", "name 'n' is not defined", "md.py", 3, "").headline
            "NameError: name 'n' is not defined"
            >>> ScriptBug("SyntaxError", "", "md.py", 3, "").headline
            'SyntaxError'
        """
        return f"{self.exception}: {self.message}" if self.message else self.exception


def last_line(text: str) -> str:
    """The last non-blank line of *text*, stripped.

    Examples:
        >>> last_line("first\\nlast\\n\\n")
        'last'
        >>> last_line("   ")
        ''
    """
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def exception_of(trace: str) -> tuple[str, str]:
    """The exception type and message a traceback ends with; the bare name.

    A dotted name keeps its last component, which is what a failure
    record's ``type`` holds, so both paths compare the same way.

    Examples:
        >>> exception_of("Traceback...\\nNameError: name 'n' is not defined")
        ('NameError', "name 'n' is not defined")
        >>> exception_of("slab.errors.LammpsScriptError: LAMMPS failed (exit 1)")
        ('LammpsScriptError', 'LAMMPS failed (exit 1)')
        >>> exception_of("KeyboardInterrupt")
        ('KeyboardInterrupt', '')
        >>> exception_of("no exception here")
        ('', '')
    """
    line = last_line(trace)
    match = _EXCEPTION.match(line)
    if match is None:
        bare = line.strip()
        return (bare.rpartition(".")[2], "") if bare.isidentifier() else ("", "")
    return match["type"].rpartition(".")[2], match["message"].strip()


def line_in(trace: str, script: str | Path) -> int | None:
    """The last line of *script* a traceback blames, or None.

    Frames are matched on the file name, so a run that executed the
    script from another directory still points at the agent's file.

    Examples:
        >>> trace = ('  File "/w/run.py", line 4, in <module>\\n'
        ...          '  File "/pkg/tasks.py", line 9, in relax\\n')
        >>> line_in(trace, "/project/run.py"), line_in(trace, "other.py")
        (4, None)
    """
    wanted = Path(str(script)).name
    found = [
        int(match["line"])
        for match in _FRAME.finditer(trace)
        if Path(match["path"]).name == wanted
    ]
    return found[-1] if found else None


def tail(trace: str, frames: int = 2) -> str:
    """The last *frames* frames of a traceback and the exception after them.

    The whole traceback goes to the helper; a note the acting agent reads
    needs the failing end of it only.

    Examples:
        >>> trace = ('Traceback (most recent call last):\\n'
        ...          '  File "/w/run.py", line 4, in <module>\\n'
        ...          '    relax(atoms)\\n'
        ...          'NameError: name \\'atoms\\' is not defined')
        >>> print(tail(trace, frames=1))
          File "/w/run.py", line 4, in <module>
            relax(atoms)
        NameError: name 'atoms' is not defined
    """
    lines = trace.rstrip().splitlines()
    starts = [index for index, line in enumerate(lines) if _FRAME.match(line)]
    if not starts:
        return trace.strip()
    return "\n".join(lines[starts[-min(frames, len(starts))] :])


def script_bug(result: Mapping[str, Any], script: str | Path) -> ScriptBug | None:
    """The script bug *result* reports, or None.

    *result* is what a launch returns: the mapping
    :func:`foundation._ops.launch_script` gives for a real run, the same
    mapping read back off a finished background run, or the dry-run
    report. *script* is the path the tool call named.

    Examples:
        >>> failed = {
        ...     "status": "failed",
        ...     "failure": {"type": "NameError", "message": "name 'n' is not defined",
        ...                 "traceback": '  File "/w/md.py", line 3, in <module>\\n'
        ...                              "NameError: name 'n' is not defined"},
        ... }
        >>> bug = script_bug(failed, "/w/md.py")
        >>> bug.exception, bug.line
        ('NameError', 3)
        >>> engine = {"status": "failed", "failure": {"type": "LammpsScriptError",
        ...           "message": "ERROR: Unrecognized pair style", "traceback": "x"}}
        >>> script_bug(engine, "/w/md.py") is None
        True
        >>> script_bug({"status": "completed", "checks_passed": 0}, "/w/md.py") is None
        True
        >>> rehearsed = {"dry_run": True, "reached_end": False,
        ...              "traceback": "KeyError: 'thermo'"}
        >>> script_bug(rehearsed, "/w/md.py").exception
        'KeyError'
        >>> script_bug({"dry_run": True, "reached_end": True}, "/w/md.py") is None
        True
    """
    if "status" in result:
        if str(result.get("status")) != "failed":
            return None
    elif result.get("dry_run") and result.get("reached_end"):
        return None
    failure = result.get("failure")
    if isinstance(failure, Mapping):
        trace = str(failure.get("traceback") or "")
        kind = str(failure.get("type") or "").rpartition(".")[2]
        message = last_line(str(failure.get("message") or ""))
    else:
        trace = str(result.get("traceback") or "")
        kind, message = exception_of(trace)
    if not trace or not kind or kind in NOT_A_SCRIPT_BUG:
        return None
    if isinstance(failure, Mapping) and _engine_ran(failure):
        return None
    return ScriptBug(kind, message, str(script), line_in(trace, script), trace.strip())


#: The note a task attaches when it kept the files of an engine that ran
#: and failed (``foundation.tasks._attach_engine_evidence``). An engine
#: that wrote output is an engine or science question, whatever exception
#: type carried it.
ENGINE_FILES_NOTE = "engine files kept as artifacts:"


def _engine_ran(failure: Mapping[str, Any]) -> bool:
    """Whether the failure record says an engine ran and left its files.

    Examples:
        >>> _engine_ran({"notes": ["engine files kept as artifacts: 'si-failed.pwo'"]})
        True
        >>> _engine_ran({"notes": ["relax failed after 3 steps"]})
        False
    """
    notes = failure.get("notes") or []
    return any(str(note).startswith(ENGINE_FILES_NOTE) for note in notes)


def python_script_in(command: str) -> str | None:
    """The ``.py`` file a shell command names, or None; the last one wins.

    Examples:
        >>> python_script_in("python analysis.py --temp 300")
        'analysis.py'
        >>> python_script_in("cd out && python3 ../bin/fit.py")
        '../bin/fit.py'
        >>> python_script_in("ls -l") is None
        True
    """
    found = _PY_ARGUMENT.findall(command)
    return found[-1] if found else None


def shell_script_bug(result: str, command: str) -> ScriptBug | None:
    """The script bug a shell result reports, or None.

    The rule is deliberately narrow. The command must name a ``.py``
    file, the result must carry a non-zero exit status, and the output
    must hold a traceback whose exception passes the same type test a
    launch's does.

    Examples:
        >>> out = ("exit 1\\n"
        ...        "Traceback (most recent call last):\\n"
        ...        '  File "analysis.py", line 7, in <module>\\n'
        ...        "KeyError: 'temp'")
        >>> bug = shell_script_bug(out, "python analysis.py")
        >>> bug.script, bug.line, bug.exception
        ('analysis.py', 7, 'KeyError')
        >>> shell_script_bug(out.replace("exit 1", "exit 0"), "python analysis.py") is None
        True
        >>> shell_script_bug("exit 1\\nls: no such file", "python analysis.py") is None
        True
        >>> shell_script_bug(out, "grep -r Traceback .") is None
        True
    """
    head, _, body = result.partition("\n")
    if not head.startswith("exit "):
        return None
    try:
        status = int(head[len("exit ") :].strip())
    except ValueError:
        return None
    if status == 0 or TRACEBACK_HEAD not in body:
        return None
    named = python_script_in(command)
    if named is None:
        return None
    trace = body[body.rindex(TRACEBACK_HEAD) :]
    kind, message = exception_of(trace)
    if not kind or kind in NOT_A_SCRIPT_BUG:
        return None
    # The traceback's own frame in that file is the better path, because
    # the command may name it relative to another directory.
    blamed = [
        match["path"]
        for match in _FRAME.finditer(trace)
        if Path(match["path"]).name == Path(named).name
    ]
    script = blamed[-1] if blamed else named
    return ScriptBug(kind, message, script, line_in(trace, script), trace.strip())


__all__ = [
    "NOT_A_SCRIPT_BUG",
    "TRACEBACK_HEAD",
    "ScriptBug",
    "exception_of",
    "last_line",
    "line_in",
    "python_script_in",
    "script_bug",
    "shell_script_bug",
    "tail",
]
