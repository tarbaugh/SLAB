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
            {"at": "2026-08-31T10:00:22+00:00", "type": "command", "kind": "launch",
             "by": "pi", "tool": "launch_workflow", "command": "slab run wf.py"},
            {"at": "2026-08-31T10:00:23+00:00", "type": "command", "kind": "engine",
             "by": "pi", "tool": "launch_workflow", "run_id": "01m0000000",
             "task": "relax", "tasks": 1, "engine": "qe", "command": "srun pw.x"},
            _tool_result("2026-08-31T10:00:23+00:00", "run 01m0000000 launched"),
            {"at": "2026-08-31T10:05:00+00:00", "type": "compaction", "summary": "so far"},
            "{not json",
            _usage("2026-08-31T10:10:00+00:00"),
            {"at": "2026-08-31T10:10:01+00:00", "type": "finish",
             "report": "a0 = 3.30 A for bcc Nb, MLIP-level"},
            {"at": "2026-08-31T10:10:01+00:00", "type": "retire", "mode": "expire",
             "runs_promoted": 1, "runs_expired": 1, "runs_total": 2,
             "bytes_promoted": 10, "bytes_expired": 5, "bytes_total": 15},
        ],
    )


def test_the_digest_counts_every_dimension(tmp_path: Path) -> None:
    transcript = _campaign(tmp_path / "20260831-100000-11.jsonl")
    summary = summarize(transcript)
    assert summary["session"] == "20260831-100000-11"
    assert summary["retire"]["runs_promoted"] == 1 and "type" not in summary["retire"]
    assert summary["commands"] == {"launch": 1, "engine": 1}
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
            "handle": "dft-expert-1",
            "transcript": str(sibling),
            "turns": 1,
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
    assert "delegation md-expert-1: 1 turn(s), 1 step(s)" in result.output
    assert "nb-a0" in result.output
    assert "refusals: 1 (read_file x1)" in result.output
    assert "errored calls: 1 (list_runs x1)" in result.output
    assert "first launch at step 2" in result.output
    assert "finish reported: a0 = 3.30 A" in result.output
    assert (
        "commands recorded: 2 (launch 1, engine 1); 'slab mason read --full' shows each"
    ) in result.output


def _header(at: str, budget: dict[str, int] | None) -> dict[str, Any]:
    event: dict[str, Any] = {"at": at, "type": "session", "agent": "pi", "model": "m"}
    if budget is not None:
        event["budget"] = budget
    return event


def _row(cpus: list[int], gpus: list[str], seconds: float | None) -> dict[str, Any]:
    return {"id": "x", "name": "r", "state": "verified", "status": "completed",
            "resources": {"cpus": cpus, "gpus": gpus, "ntasks": 1, "threads": 1},
            "started": None, "ended": None, "seconds": seconds}


def test_two_sized_runs_sum_their_held_hours_against_the_budget(tmp_path: Path) -> None:
    """Two runs, 2 cpus + 1 gpu for 30 min and 1 cpu for 1 h, in a 2 h session
    on a 4-cpu, 2-gpu budget: 2 cpu-h of 8 and 0.5 gpu-h of 4."""
    transcript = _write(
        tmp_path / "s.jsonl",
        [
            _header("2026-09-10T10:00:00+00:00", {"cpus": 4, "gpus": 2}),
            _usage("2026-09-10T12:00:00+00:00"),
        ],
    )
    rows = [_row([0, 1], ["0"], 1800.0), _row([2], [], 3600.0)]
    summary = summarize(transcript, runs=rows)
    assert summary["budget"] == {"cpus": 4, "gpus": 2}
    assert (summary["cpu_hours_held"], summary["gpu_hours_held"]) == (2.0, 0.5)
    assert summary["wall_hours"] == 2.0
    assert summary["utilisation"] == {"cpu": 0.25, "gpu": 0.125}
    assert (summary["runs_sized"], summary["runs_unsized"], summary["runs_open"]) == (2, 0, 0)
    # Without the rows the fields are absent, not zero: nothing was looked up.
    assert summarize(transcript)["utilisation"] is None
    assert summarize(transcript)["cpu_hours_held"] is None


def test_a_run_without_a_slice_contributes_nothing_and_is_counted(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "s.jsonl",
        [
            _header("2026-09-10T10:00:00+00:00", {"cpus": 4, "gpus": 0}),
            _usage("2026-09-10T11:00:00+00:00"),
        ],
    )
    unsized = {**_row([0], [], 3600.0), "resources": None}
    still_running = _row([0, 1, 2, 3], [], None)
    summary = summarize(transcript, runs=[_row([0], [], 3600.0), unsized, still_running])
    assert summary["cpu_hours_held"] == 1.0
    assert (summary["runs_sized"], summary["runs_unsized"], summary["runs_open"]) == (1, 1, 1)
    # A budget with no gpus gives no gpu share; the cpu share stands.
    assert summary["utilisation"] == {"cpu": 0.25, "gpu": None}


def test_no_header_budget_means_hours_held_but_no_percentage(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "s.jsonl",
        [_header("2026-09-10T10:00:00+00:00", None), _usage("2026-09-10T11:00:00+00:00")],
    )
    summary = summarize(transcript, runs=[_row([0, 1], [], 1800.0)])
    assert summary["budget"] is None
    assert summary["cpu_hours_held"] == 1.0
    assert summary["utilisation"] is None


def test_cli_reads_the_slice_and_the_span_from_the_run_record(tmp_path: Path) -> None:
    from slab.resources import Budget

    root = tmp_path / "ws"
    session = "20260910-100000-5"
    _write(
        root / "mason" / "sessions" / f"{session}.jsonl",
        [
            _header("2026-09-10T10:00:00+00:00", {"cpus": 4, "gpus": 1}),
            _usage("2026-09-10T10:00:30+00:00"),
        ],
    )
    budget = Budget(cpus=(0, 1, 2, 3), gpus=("0",))
    with Workspace(root) as ws:
        held = ws.reserve(ntasks=2, threads=1, gpus=1, budget=budget)
        with ws.start_run(name="sized", session=session, reservation=held) as run:
            run.keep("answer", 1)
        with ws.start_run(name="plain", session=session) as run:
            run.keep("answer", 2)
    result = runner.invoke(app, ["report", "-w", str(root), "--json"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    by_name = {row["name"]: row for row in summary["runs"]}
    sized, plain = by_name["sized"], by_name["plain"]
    assert sized["resources"]["cpus"] == [0, 1] and sized["resources"]["gpus"] == ["0"]
    assert sized["started"] and sized["ended"] and sized["seconds"] >= 0
    assert plain["resources"] is None and plain["seconds"] >= 0
    assert summary["cpu_hours_held"] == 2 * sized["seconds"] / 3600
    assert summary["gpu_hours_held"] == sized["seconds"] / 3600
    assert (summary["runs_sized"], summary["runs_unsized"]) == (1, 1)
    assert summary["utilisation"]["cpu"] == 2 * sized["seconds"] / (4 * 30)
    text = runner.invoke(app, ["report", "-w", str(root)])
    assert text.exit_code == 0, text.output
    assert "cpu-h and" in text.output and "over 30s on 4 cpus and 1 gpus (" in text.output
    assert text.output.rstrip().count("1 run(s) without a slice") == 1


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


def test_the_cached_share_the_peak_and_the_clearings_are_counted(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "20260901-120000-1.jsonl",
        [
            {"at": "2026-09-01T12:00:00+00:00", "type": "usage", "prompt_tokens": 9_000,
             "completion_tokens": 100, "cached_prompt_tokens": 8_000},
            {"at": "2026-09-01T12:00:10+00:00", "type": "usage", "prompt_tokens": 31_000,
             "completion_tokens": 200, "cached_prompt_tokens": 27_000},
            {"at": "2026-09-01T12:00:20+00:00", "type": "clearing", "cleared": 4,
             "chars": 14_000},
            {"at": "2026-09-01T12:00:30+00:00", "type": "usage", "prompt_tokens": 20_000,
             "completion_tokens": 50},
        ],
    )
    summary = summarize(transcript)
    assert summary["prompt_tokens"] == 60_000
    assert summary["cached_prompt_tokens"] == 35_000
    assert summary["total_cached_prompt_tokens"] == 35_000
    assert summary["peak_prompt_tokens"] == 31_000
    assert summary["clearings"] == 1
    shown = runner.invoke(app, ["report", str(transcript)])
    assert shown.exit_code == 0, shown.output
    assert "tokens 60000+350 (35000 cached)" in shown.output
    assert "context: peak prompt 31000 tokens, 1 clearing(s), 0 compaction(s)" in shown.output


def test_a_json_line_that_is_not_an_object_is_counted_as_malformed(tmp_path: Path) -> None:
    transcript = _write(
        tmp_path / "20260901-130000-1.jsonl",
        [_usage("2026-09-01T13:00:00+00:00"), "[]", "42"],
    )
    summary = summarize(transcript)
    assert summary["steps"] == 1
    assert summary["malformed_lines"] == 2


def test_a_continued_specialist_is_one_row_with_its_turn_count(tmp_path: Path) -> None:
    """A lead that continued one specialist twice reads one row, not three,
    and the briefs it sized too tight are counted apart from the card's cap."""
    transcript = _write(
        tmp_path / "20260901-090000-11.jsonl",
        [
            {"at": "2026-09-01T09:00:00+00:00", "type": "delegate", "agent": "dft-expert",
             "stop": "finish", "steps": 9, "turn": 1},
            {"at": "2026-09-01T09:10:00+00:00", "type": "delegate", "agent": "dft-expert",
             "stop": "max_turns", "steps": 6, "turn": 2, "continues": "dft-expert-1",
             "steps_budget": 6},
            {"at": "2026-09-01T09:20:00+00:00", "type": "delegate", "agent": "dft-expert",
             "stop": "finish", "steps": 6, "turn": 3, "continues": "dft-expert-1"},
        ],
    )
    sibling = _write(
        tmp_path / "20260901-090000-11-dft-expert-1.jsonl",
        [
            {"at": "2026-09-01T09:00:00+00:00", "type": "turn", "n": 1},
            _usage("2026-09-01T09:01:00+00:00"),
            {"at": "2026-09-01T09:10:00+00:00", "type": "turn", "n": 2},
            _usage("2026-09-01T09:11:00+00:00"),
            {"at": "2026-09-01T09:20:00+00:00", "type": "turn", "n": 3},
            _usage("2026-09-01T09:21:00+00:00"),
        ],
    )
    summary = summarize(transcript, [sibling])
    (row,) = summary["delegations"]
    assert row["handle"] == "dft-expert-1"
    assert (row["turns"], row["steps"]) == (3, 3)
    assert summary["briefs"] == 3
    assert summary["brief_budget_stops"] == 1


def _cut(at: str, tokens: int, after_tool: str | None) -> dict[str, Any]:
    return {
        "at": at,
        "type": "cut",
        "case": 1,
        "continued": False,
        "tokens": tokens,
        "after_tool": after_tool,
    }


def test_the_report_sums_the_tokens_the_ceiling_cost(tmp_path: Path) -> None:
    """Benchmark-4 session 1 lost 544,000 completion tokens to seventeen
    cuts and the report said nothing about them. It sums them now, and
    names the tool most of them followed."""
    sessions = tmp_path / "ws" / "mason" / "sessions"
    conversation = _write(
        sessions / "20260917-100000-1.jsonl",
        [
            _usage("2026-09-17T10:00:00+00:00"),
            _assistant("2026-09-17T10:00:01+00:00", "read_artifact"),
            _tool_result("2026-09-17T10:00:02+00:00", "1 2\n3 4\n"),
            _cut("2026-09-17T10:00:03+00:00", 32_000, "read_artifact"),
            _cut("2026-09-17T10:00:04+00:00", 16_000, "read_artifact"),
        ],
    )
    sibling = _write(
        sessions / "20260917-100000-1-md-expert-1.jsonl",
        [_usage("2026-09-17T10:01:00+00:00"), _cut("2026-09-17T10:01:01+00:00", 8_000, "shell")],
    )
    summary = summarize(conversation, siblings=[sibling])
    assert summary["cuts"] == 2 and summary["cut_tokens"] == 48_000
    assert summary["cut_after_tools"] == {"read_artifact": 2}
    # The specialist's cut is the session's cost too.
    assert summary["total_cuts"] == 3 and summary["total_cut_tokens"] == 56_000
    assert next(iter(summary["total_cut_after_tools"])) == "read_artifact"
    result = runner.invoke(app, ["report", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 0, result.output
    assert (
        "replies cut at the ceiling: 3, 56000 completion tokens lost; "
        "most after read_artifact" in result.output
    )


def test_a_continued_cut_is_counted_but_lost_nothing(tmp_path: Path) -> None:
    """A cut the loop continued kept its text in the history, so its
    tokens were not lost; the report counts the cut and not the tokens.
    A single cut names its tool with "after", and a tie names none."""
    sessions = tmp_path / "ws" / "mason" / "sessions"
    continued = {**_cut("2026-09-17T12:00:02+00:00", 9_000, "shell"), "case": 2, "continued": True}
    conversation = _write(
        sessions / "20260917-120000-3.jsonl",
        [_usage("2026-09-17T12:00:00+00:00"), continued],
    )
    summary = summarize(conversation)
    assert summary["total_cuts"] == 1 and summary["total_cut_tokens"] == 0
    result = runner.invoke(app, ["report", "-w", str(tmp_path / "ws")])
    assert "replies cut at the ceiling: 1, 0 completion tokens lost; after shell" in result.output
    _write(
        sessions / "20260917-120000-3-md-expert-1.jsonl",
        [
            _usage("2026-09-17T12:01:00+00:00"),
            _cut("2026-09-17T12:01:01+00:00", 4_000, "read_file"),
        ],
    )
    result = runner.invoke(app, ["report", "-w", str(tmp_path / "ws")])
    assert "replies cut at the ceiling: 2, 4000 completion tokens lost\n" in result.output


def test_a_session_without_cuts_says_nothing_about_them(tmp_path: Path) -> None:
    conversation = _campaign(tmp_path / "ws" / "mason" / "sessions" / "20260917-110000-2.jsonl")
    assert summarize(conversation)["total_cuts"] == 0
    result = runner.invoke(app, ["report", "-w", str(tmp_path / "ws")])
    assert "replies cut at the ceiling" not in result.output
