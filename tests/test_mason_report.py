"""``slab mason report``: the session digest is honest arithmetic.

The fixtures write transcripts in the exact vocabulary
:meth:`mason.session.MasonSession.record` uses, so a schema drift breaks
these tests before it breaks the report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from foundation.runtime import Workspace
from mason.cli import app
from mason.report import summarize

runner = CliRunner()


def _write(path: Path, events: list[dict[str, Any] | str]) -> Path:
    lines = [
        event if isinstance(event, str) else json.dumps(event) for event in events
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _usage(at: str, prompt: int = 100, completion: int = 10) -> dict[str, Any]:
    return {"at": at, "type": "usage", "prompt_tokens": prompt, "completion_tokens": completion}


def _assistant(at: str, *tool_names: str) -> dict[str, Any]:
    return {
        "at": at,
        "type": "message",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": "{}"}}
                for name in tool_names
            ],
        },
    }


def _tool_result(at: str, content: str) -> dict[str, Any]:
    return {"at": at, "type": "message", "message": {"role": "tool", "content": content}}


def _campaign(path: Path) -> Path:
    """Three model calls: probe (refused, errored), launch, then finish."""
    return _write(
        path,
        [
            {"at": "2026-08-31T10:00:00+00:00", "type": "message",
             "message": {"role": "user", "content": "the goal"}},
            _usage("2026-08-31T10:00:05+00:00"),
            _assistant("2026-08-31T10:00:06+00:00", "read_file", "list_runs"),
            _tool_result("2026-08-31T10:00:07+00:00", "refused: outside the fence"),
            _tool_result(
                "2026-08-31T10:00:08+00:00",
                "tool list_runs failed: SessionNotFoundError: no run carries it",
            ),
            {"at": "2026-08-31T10:00:09+00:00", "type": "skill",
             "name": "elastic-constants", "source": "builtin"},
            _usage("2026-08-31T10:00:20+00:00"),
            _assistant("2026-08-31T10:00:21+00:00", "recall", "launch_workflow"),
            _tool_result("2026-08-31T10:00:22+00:00", "no memories match"),
            _tool_result("2026-08-31T10:00:23+00:00", "run 01m0000000 launched"),
            {"at": "2026-08-31T10:05:00+00:00", "type": "compaction", "summary": "so far"},
            "{not json",
            _usage("2026-08-31T10:10:00+00:00"),
            {"at": "2026-08-31T10:10:01+00:00", "type": "finish",
             "report": "a0 = 3.30 A for bcc Nb, MLIP-level"},
        ],
    )


def test_the_digest_counts_every_dimension(tmp_path: Path) -> None:
    transcript = _campaign(tmp_path / "20260831-100000-11.jsonl")
    summary = summarize(transcript)
    assert summary["session"] == "20260831-100000-11"
    assert summary["steps"] == 3
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 30
    assert summary["span_s"] == 601.0
    assert summary["tools"] == {
        "read_file": 1, "list_runs": 1, "recall": 1, "launch_workflow": 1
    }
    assert summary["refusals"] == 1
    assert summary["refused_tools"] == {"read_file": 1}
    assert summary["errored_calls"] == 1
    assert summary["errored_tools"] == {"list_runs": 1}
    assert summary["recall"] == 1
    assert summary["remember"] == 0
    assert summary["skills"] == ["elastic-constants"]
    assert summary["compactions"] == 1
    assert summary["malformed_lines"] == 1
    # The launch call was made by the second model call.
    assert summary["first_launch_step"] == 2
    assert summary["finish"] == {
        "reported": True,
        "head": "a0 = 3.30 A for bcc Nb, MLIP-level",
        "report": "a0 = 3.30 A for bcc Nb, MLIP-level",
        "results": {},
        "run_ids": [],
    }


def test_delegations_roll_into_the_totals(tmp_path: Path) -> None:
    transcript = _campaign(tmp_path / "20260831-100000-11.jsonl")
    sibling = _write(
        tmp_path / "20260831-100000-11-dft-expert-1.jsonl",
        [_usage("2026-08-31T10:02:00+00:00", prompt=50, completion=5)],
    )
    summary = summarize(transcript, [sibling])
    assert summary["delegations"] == [
        {
            "agent": "dft-expert",
            "transcript": str(sibling),
            "steps": 1,
            "prompt_tokens": 50,
            "completion_tokens": 5,
        }
    ]
    assert summary["total_steps"] == 4
    assert summary["total_prompt_tokens"] == 350
    assert summary["total_completion_tokens"] == 35


def test_an_empty_transcript_is_a_zero_report(tmp_path: Path) -> None:
    transcript = _write(tmp_path / "20260831-100000-11.jsonl", [])
    summary = summarize(transcript)
    assert summary["steps"] == 0
    assert summary["span_s"] is None
    assert summary["first_launch_step"] is None
    assert summary["finish"] == {
        "reported": False,
        "head": None,
        "report": None,
        "results": {},
        "run_ids": [],
    }
    assert summary["model"] is None  # no header: an older transcript


def test_the_header_and_the_structured_finish_are_surfaced(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "20260831-100000-11.jsonl",
        [
            {
                "at": "2026-08-31T10:00:00+00:00",
                "type": "session",
                "agent": "pi",
                "model": "muse-glimmer-30b",
                "provider": "openai",
                "endpoint": "http://node:8000/v1",
                "endpoint_origin": "job 42 on node",
                "compute_profile": "cluster",
                "max_turns": 60,
            },
            _usage("2026-08-31T10:00:01+00:00"),
            {
                "at": "2026-08-31T10:00:02+00:00",
                "type": "finish",
                "report": "a0 = 3.63 A " + "x" * 300,
                "results": {"a0": {"value": 3.63, "unit": "A"}},
                "run_ids": ["01m1"],
            },
        ],
    )
    summary = summarize(transcript)
    assert summary["model"] == "muse-glimmer-30b"
    assert summary["provider"] == "openai"
    assert summary["endpoint_origin"] == "job 42 on node"
    assert summary["compute_profile"] == "cluster"
    assert summary["agent"] == "pi"
    assert len(summary["finish"]["head"]) == 200
    assert summary["finish"]["report"].startswith("a0 = 3.63 A")
    assert len(summary["finish"]["report"]) > 200
    assert summary["finish"]["results"] == {"a0": {"value": 3.63, "unit": "A"}}
    assert summary["finish"]["run_ids"] == ["01m1"]


def test_cli_finds_a_session_by_id_or_unique_prefix(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    sessions = root / "mason" / "sessions"
    _write(sessions / "20260830-090000-7.jsonl", [_usage("2026-08-30T09:00:00+00:00")])
    _campaign(sessions / "20260831-100000-11.jsonl")
    Workspace(root).close()
    by_id = runner.invoke(app, ["report", "-w", str(root), "--session", "20260830-090000-7"])
    assert by_id.exit_code == 0, by_id.output
    assert "session 20260830-090000-7 — 1 step(s)" in by_id.output
    by_prefix = runner.invoke(app, ["report", "-w", str(root), "--session", "20260831"])
    assert by_prefix.exit_code == 0, by_prefix.output
    assert "20260831-100000-11" in by_prefix.output
    ambiguous = runner.invoke(app, ["report", "-w", str(root), "--session", "2026"])
    assert ambiguous.exit_code == 1
    assert "ambiguous" in ambiguous.output
    missing = runner.invoke(app, ["report", "-w", str(root), "--session", "1999"])
    assert missing.exit_code == 1
    assert "no session transcript matches" in missing.output


def test_cli_reports_the_newest_conversation_and_its_runs(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    sessions = root / "mason" / "sessions"
    _write(sessions / "20260830-090000-7.jsonl", [_usage("2026-08-30T09:00:00+00:00")])
    _campaign(sessions / "20260831-100000-11.jsonl")
    _write(
        sessions / "20260831-100000-11-md-expert-1.jsonl",
        [_usage("2026-08-31T10:03:00+00:00")],
    )
    with Workspace(root) as ws, ws.start_run(
        name="nb-a0", session="20260831-100000-11"
    ) as run:
        run.keep("answer", 3.30)
    result = runner.invoke(app, ["report", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert "session 20260831-100000-11 — 4 step(s)" in result.output
    assert "delegation md-expert: 1 step(s)" in result.output
    assert "nb-a0" in result.output
    assert "refusals: 1 (read_file x1)" in result.output
    assert "errored calls: 1 (list_runs x1)" in result.output
    assert "first launch at step 2" in result.output
    assert "finish reported: a0 = 3.30 A" in result.output


def test_cli_json_is_machine_readable(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    _campaign(root / "mason" / "sessions" / "20260831-100000-11.jsonl")
    result = runner.invoke(app, ["report", "-w", str(root), "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["session"] == "20260831-100000-11"
    assert summary["runs"] == []
    assert summary["tools"]["launch_workflow"] == 1


def test_cli_without_transcripts_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "no session transcripts" in result.output


def test_cli_refuses_a_missing_explicit_transcript(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(tmp_path / "gone.jsonl"), "-w", str(tmp_path / "ws")]
    )
    assert result.exit_code == 1
    assert "no transcript at" in result.output


def test_a_session_that_launched_nothing_reports_none(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    _write(
        root / "mason" / "sessions" / "20260831-100000-11.jsonl",
        [_usage("2026-08-31T10:00:00+00:00")],
    )
    Workspace(root).close()
    result = runner.invoke(app, ["report", "-w", str(root)])
    assert result.exit_code == 0, result.output
    assert "runs this session created: none" in result.output
    assert "no finish report" in result.output
