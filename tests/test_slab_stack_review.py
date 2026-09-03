"""The science review: flags are attributable, revisions are gated.

The fixtures write transcripts in the loop's vocabulary against a real
workspace, the way the benchmark tests do, so the rules read the evidence
a campaign leaves. The referee is a scripted client: what matters is the
contract (the evidence pack it receives, the flags it may raise), not a
model's opinion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk
from typer.testing import CliRunner

from foundation.runtime import Workspace
from foundation.tasks import single_point
from mason.client import ChatReply
from mason.skills import discover_skills, parse_skill
from slab_stack import benchmark, review
from slab_stack.cli import app

runner = CliRunner()

Q1 = benchmark.find_question("a0")
EOS = "equation-of-state"
LONG_AGO = "2000-01-01T00:00:00+00:00"
FAR_AHEAD = "2999-01-01T00:00:00+00:00"


def _header(agent: str = "pi") -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:00+00:00",
        "type": "session",
        "agent": agent,
        "model": "fake-30b",
        "provider": "openai",
        "endpoint": "http://localhost:11434/v1",
        "endpoint_origin": "[agent] endpoint",
        "compute_profile": "laptop",
        "max_turns": 40,
    }


def _user(text: str) -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:01+00:00",
        "type": "message",
        "message": {"role": "user", "content": text},
    }


def _skill(name: str = EOS, digest: str = "abc123def456", at: str = LONG_AGO) -> dict[str, Any]:
    return {"at": at, "type": "skill", "name": name, "source": "built-in", "digest": digest}


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:02+00:00",
        "type": "message",
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            ],
        },
    }


def _result(content: str) -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:03+00:00",
        "type": "message",
        "message": {"role": "tool", "content": content},
    }


def _finish(
    results: dict[str, Any] | None, run_ids: list[str] | None, report: str = "done"
) -> dict[str, Any]:
    event: dict[str, Any] = {"at": "2026-09-01T10:05:00+00:00", "type": "finish", "report": report}
    if results is not None:
        event["results"] = results
    if run_ids is not None:
        event["run_ids"] = run_ids
    return event


def _transcript(root: Path, stem: str, events: list[dict[str, Any]]) -> Path:
    path = root / "mason" / "sessions" / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def _run(root: Path, session: str, *, check: bool | None = True, name: str = "eos") -> str:
    """A real run: verified (check True), failing its check (False), or unchecked (None)."""
    with Workspace(root) as ws, ws.start_run(name=name, session=session) as run:
        single_point(bulk("Cu"), engine="emt")
        if check is not None:

            @run.check
            def gate() -> bool:
                return bool(check)

    return run.id


def _flags(record: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(f["rule"], f["target"]): f for f in record["flags"]}


class ScriptedReferee:
    """Answers one reply; records the messages it was sent."""

    model = "referee-70b"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[dict[str, Any]] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatReply:
        self.messages = messages
        return ChatReply(content=self.reply, prompt_tokens=10, completion_tokens=5)


# -- the rules ----------------------------------------------------------------


def test_a_campaign_that_never_loaded_the_skill_is_flagged_for_it(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-1"
    _transcript(root, session, [_header(), _user(Q1.instruction), _finish(None, None)])
    Workspace(root).close()
    record = benchmark.score_session(root, session)
    assert record["passed"] is False
    assert record["reason"] == "the finish carried no structured results"
    assert record["reviewed_by"] == ["rules"] and record["skills"] == {}
    assert record["agent"] == "pi" and record["referee_model"] is None
    flags = _flags(record)
    assert set(flags) == {("skill-not-loaded", f"skill:{EOS}"), ("finish-incomplete", "card:pi")}
    assert "description did not trigger" in flags[("skill-not-loaded", f"skill:{EOS}")]["note"]
    assert "no structured results" in flags[("finish-incomplete", "card:pi")]["note"]
    assert all(f["raised_by"] == "rules" for f in record["flags"])


def test_a_loaded_skill_whose_script_never_ran_is_flagged(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-2"
    run_id = _run(root, session)
    _transcript(
        root,
        session,
        [
            _header(),
            _user(Q1.instruction),
            _skill(),
            _call("shell", command="python my_own_fit.py eos.json"),
            _result("exit 0\nV0 = 11.8"),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["passed"] is True
    assert record["skills"] == {EOS: "abc123def456"}
    flags = _flags(record)
    assert set(flags) == {("script-not-used", f"skill:{EOS}")}
    flag = flags[("script-not-used", f"skill:{EOS}")]
    assert flag["evidence"] == "loaded at step 0; 1 shell call(s) after it"
    assert "fit_eos.py" in flag["note"]


def test_a_bundled_script_that_failed_is_flagged_with_its_first_line(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-3"
    run_id = _run(root, session)
    _transcript(
        root,
        session,
        [
            _header(),
            _user(Q1.instruction),
            _call("list_runs"),
            _result("no runs"),
            _skill(),
            _call("shell", command="python /skills/equation-of-state/scripts/fit_eos.py eos.json"),
            _result("exit 2\nnot enough points: 3 (need 5)"),
            _call("shell", command="python /skills/equation-of-state/scripts/fit_eos.py eos.json"),
            _result("exit 0\nV0 = 11.8"),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id]),
        ],
    )
    record = benchmark.score_session(root, session)
    flags = _flags(record)
    assert set(flags) == {("script-failed", f"skill:{EOS}")}
    flag = flags[("script-failed", f"skill:{EOS}")]
    assert flag["evidence"].startswith("step 2: python /skills/equation-of-state/scripts/fit_eos")
    assert flag["note"] == "exit 2"  # the first line of the result


def test_an_unverified_run_is_blamed_on_the_skill_in_force(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-4"
    cited = _run(root, session)
    unchecked = _run(root, session, check=None, name="draft")
    failing = _run(root, session, check=False, name="scan")
    _transcript(
        root,
        session,
        [
            _header(),
            _user(Q1.instruction),
            _skill(at=LONG_AGO),
            _call("shell", command="python x/scripts/fit_eos.py eos.json"),
            _result("exit 0\nfine"),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [cited]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["passed"] is True
    by_evidence = {f["evidence"].split()[1]: f for f in record["flags"]}
    assert set(by_evidence) == {unchecked[:10], failing[:10]}
    assert by_evidence[unchecked[:10]]["target"] == f"skill:{EOS}"
    assert "no checks were registered" in by_evidence[unchecked[:10]]["note"]
    assert by_evidence[failing[:10]]["rule"] == "run-not-verified"
    assert "checks failed: gate" in by_evidence[failing[:10]]["note"]

    # With no skill loaded before the runs started, the card carries the flag.
    session = "20260901-100000-5"
    cited = _run(root, session)
    _run(root, session, check=None, name="draft")
    _transcript(
        root,
        session,
        [
            _header(agent="dft-expert"),
            _user(Q1.instruction),
            _skill(at=FAR_AHEAD),
            _call("shell", command="python x/scripts/fit_eos.py eos.json"),
            _result("exit 0\nfine"),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [cited]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert [f["target"] for f in record["flags"]] == ["card:dft-expert"]


@pytest.mark.parametrize(("unit", "flagged"), [("nm", True), ("A", False), ("angstrom", False)])
def test_a_unit_that_is_not_the_questions_is_flagged_on_the_finish_tool(
    tmp_path: Path, unit: str, flagged: bool
) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-6"
    run_id = _run(root, session)
    _transcript(
        root,
        session,
        [
            _header(),
            _user(Q1.instruction),
            _skill(),
            _call("shell", command="python x/scripts/fit_eos.py eos.json"),
            _result("exit 0\nfine"),
            _finish({"a0": {"value": 3.60, "unit": unit}}, [run_id]),
        ],
    )
    record = benchmark.score_session(root, session)
    rules = {f["rule"] for f in record["flags"]}
    assert ("unit-mismatch" in rules) is flagged
    if flagged:
        assert _flags(record)[("unit-mismatch", "tool:finish")]["note"].endswith("asks for 'Å'.")


# -- the referee --------------------------------------------------------------


def _clean_campaign(root: Path, session: str) -> str:
    run_id = _run(root, session)
    _transcript(
        root,
        session,
        [
            _header(),
            _user(Q1.instruction),
            _skill(),
            _call("launch_workflow", path="eos_scan.py", intent="scan"),
            _result(f"run {run_id} launched"),
            _call("shell", command="python x/scripts/fit_eos.py eos.json --json"),
            _result('exit 0\n{"v0_A3": 47.2}'),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id], "a0 from a 7-point scan"),
        ],
    )
    return run_id


def test_the_referee_reads_the_evidence_pack_and_its_flags_join_the_rules(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-7"
    run_id = _clean_campaign(root, session)
    judge = ScriptedReferee(
        "Here is my review.\n```json\n"
        + json.dumps(
            {
                "flags": [
                    {
                        "rule": "Scan Not Bracketing",
                        "target": f"skill:{EOS}",
                        "evidence": "step 1",
                        "note": "The scan range was never widened after the fit.",
                    },
                    {
                        "rule": "kpoints-not-converged",
                        "target": "engine:emt",
                        "evidence": f"run {run_id[:10]}",
                        "note": "No k-point ladder was run.",
                    },
                ]
            }
        )
        + "\n```"
    )
    record = benchmark.score_session(root, session, referee=judge)
    assert record["passed"] is True
    assert record["reviewed_by"] == ["rules", "referee"]
    assert record["referee_model"] == "referee-70b" and record["referee_error"] is None
    flags = _flags(record)
    assert set(flags) == {
        ("scan-not-bracketing", f"skill:{EOS}"),
        ("kpoints-not-converged", "card:pi"),  # an unknown target lands on the card
    }
    assert all(f["raised_by"] == "referee" for f in record["flags"])
    assert flags[("kpoints-not-converged", "card:pi")]["note"].startswith(
        "(the referee named target 'engine:emt')"
    )

    system, user = judge.messages
    assert system["role"] == "system" and "adversarial" in system["content"]
    pack = user["content"]
    assert Q1.instruction in pack and "passed" in pack
    assert f"- skill:{EOS}" in pack and "- card:pi" in pack and "- tool:finish" in pack
    assert "# Equation of state" in pack  # the skill body the procedure is judged against
    assert "step 2: shell" in pack and "fit_eos.py" in pack
    assert f"run {run_id[:10]} eos: state verified" in pack and "check gate: passed" in pack
    assert "a0 from a 7-point scan" in pack


def test_a_referee_that_cannot_be_read_leaves_the_rules_in_place(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-8"
    _transcript(root, session, [_header(), _user(Q1.instruction), _finish(None, None)])
    Workspace(root).close()
    record = benchmark.score_session(root, session, referee=ScriptedReferee("no json here"))
    assert record["reviewed_by"] == ["rules"]
    assert "without a JSON object" in record["referee_error"]
    assert {f["rule"] for f in record["flags"]} == {"skill-not-loaded", "finish-incomplete"}


def test_parse_referee_refuses_what_it_cannot_read() -> None:
    with pytest.raises(review.ReviewError, match="no 'flags' list"):
        review.parse_referee('{"verdict": "fine"}', ["card:pi"], "card:pi")
    with pytest.raises(review.ReviewError, match="does not parse"):
        review.parse_referee("{not json}", ["card:pi"], "card:pi")
    assert review.parse_referee('{"flags": []}', ["card:pi"], "card:pi") == []


# -- skill revisions ----------------------------------------------------------


def test_a_skill_digest_names_its_revision(tmp_path: Path) -> None:
    directory = tmp_path / "xrd-pattern"
    (directory / "scripts").mkdir(parents=True)
    (directory / "SKILL.md").write_text("---\nname: xrd-pattern\ndescription: XRD.\n---\nSteps.\n")
    (directory / "scripts" / "sim.py").write_text("print(1)\n")
    first = parse_skill(directory, "project").digest
    assert len(first) == 12
    assert parse_skill(directory, "project").digest == first  # stable
    (directory / "scripts" / "sim.py").write_text("print(2)\n")
    assert parse_skill(directory, "project").digest != first  # a script edit is a revision
    (directory / ".cache").write_text("x")
    assert parse_skill(directory, "project").digest != first  # hidden files do not count
    built_in = discover_skills(tmp_path)[EOS]
    assert built_in.digest == parse_skill(built_in.root, "built-in").digest


# -- the ledger and the gate --------------------------------------------------


def _record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question": 1, "key": "a0", "session": "s", "model": "m-30b", "machine": "laptop",
        "skills": {}, "passed": False, "reason": "x", "flags": [],
    }
    return {**base, **overrides}


def _flag(rule: str, target: str, raised_by: str = "rules") -> dict[str, str]:
    return {"rule": rule, "target": target, "evidence": "e", "note": "n", "raised_by": raised_by}


def test_the_ledger_gives_every_flag_a_status(tmp_path: Path) -> None:
    catalog = discover_skills(tmp_path)
    current = catalog[EOS].digest
    records = [
        _record(session="s1", skills={EOS: "old0old0old0"},
                flags=[_flag("script-not-used", f"skill:{EOS}"),
                       _flag("finish-incomplete", "card:pi")]),
        _record(session="s2", model="tiny-8b", skills={EOS: current},
                flags=[_flag("scan-not-bracketing", f"skill:{EOS}", "referee")]),
        _record(session="s3", model="other", flags=[_flag("r", "skill:not-a-skill")]),
    ]
    rows = review.ledger(records, catalog)
    status = {(r["rule"], r["model"]): r["status"] for r in rows}
    assert status == {
        ("finish-incomplete", "m-30b"): "open",
        ("script-not-used", "m-30b"): "pending",
        ("scan-not-bracketing", "tiny-8b"): "open",
        ("r", "other"): "unknown",
    }
    assert rows[0]["target"] == "card:pi"  # sorted by target
    pending = next(r for r in rows if r["status"] == "pending")
    assert pending["against"] == "old0old0old0" and pending["question"] == 1


def test_the_gate_validates_a_revision_only_against_a_campaign_under_it(tmp_path: Path) -> None:
    catalog = discover_skills(tmp_path)
    current = catalog[EOS].digest
    old = _record(session="s1", skills={EOS: "old0old0old0"}, passed=False,
                  flags=[_flag("script-not-used", f"skill:{EOS}")])

    # No campaign under the current revision: not validated, and the report says why.
    report = review.gate(EOS, [old], catalog)
    assert report.digest == current and report.validated is False
    assert [c.verdict for c in report.cells] == ["not validated"]
    assert f"loaded revision {current}" in report.cells[0].detail

    # A campaign under it that passes and no longer raises the flag: validated, flag closed.
    new = _record(session="s2", skills={EOS: current}, passed=True, reason=None)
    report = review.gate(EOS, [old, new], catalog)
    assert report.validated is True
    assert report.cells[0].verdict == "validated"
    assert report.cells[0].detail == "passes, no flag against the skill; closed script-not-used"

    # The flag recurs under the new revision: still flagged.
    recurring = dict(new, session="s3", flags=[_flag("script-not-used", f"skill:{EOS}")])
    report = review.gate(EOS, [old, new, recurring], catalog)
    assert report.validated is False and report.cells[0].verdict == "still flagged"
    assert "raises script-not-used" in report.cells[0].detail

    # Passed before, fails now: regressed, whatever the flags say.
    was_fine = dict(old, passed=True, reason=None, flags=[])
    broke = dict(new, passed=False, reason="a0 = 3.9 Å is outside 3.632 ± 0.05 Å (mlip band)")
    report = review.gate(EOS, [was_fine, broke], catalog)
    assert report.cells[0].verdict == "regressed"
    assert "now fails: a0 = 3.9" in report.cells[0].detail

    # A cell for another question counts when that campaign loaded the skill.
    elsewhere = _record(session="s4", key="vacancy", question=3, skills={EOS: current},
                        passed=False, reason="no finish report")
    report = review.gate(EOS, [old, new, elsewhere], catalog)
    assert [(c.key, c.verdict) for c in report.cells] == [
        ("a0", "validated"), ("vacancy", "validated")
    ]
    assert report.cells[1].detail.startswith("fails (no finish report), no flag")

    # Nothing exercised the skill: not validated. An unknown skill is refused.
    assert review.gate("nemd-transport", [old], catalog).validated is False
    with pytest.raises(review.ReviewError, match="no skill named"):
        review.gate("nope", [old], catalog)


# -- the CLI ------------------------------------------------------------------


def test_cli_flags_and_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    catalog = discover_skills(tmp_path)
    current = catalog[EOS].digest
    path = tmp_path / "benchmarks" / "results.jsonl"
    benchmark.append_record(
        path,
        _record(session="s1", skills={EOS: "old0old0old0"},
                flags=[_flag("script-not-used", f"skill:{EOS}"),
                       _flag("finish-incomplete", "card:pi")]),
    )

    listed = runner.invoke(app, ["benchmark", "flags"])
    assert listed.exit_code == 0, listed.output
    assert listed.output.startswith("card:pi\n  open     finish-incomplete")
    assert (
        f"skill:{EOS}\n  pending  script-not-used        Q1 m-30b/laptop against old0old0old0"
        in listed.output
    )
    only = runner.invoke(app, ["benchmark", "flags", "--status", "open", "--json"])
    assert [row["rule"] for row in json.loads(only.output)] == ["finish-incomplete"]
    none = runner.invoke(app, ["benchmark", "flags", "--target", "skill:surface-energy"])
    assert none.output == "no flags\n"

    gate = runner.invoke(app, ["benchmark", "gate", EOS])
    assert gate.exit_code == 1, gate.output
    assert f"{EOS} revision {current}" in gate.output
    assert "not validated: no scored campaign loaded revision" in gate.output
    assert gate.output.endswith("not validated\n")

    benchmark.append_record(path, _record(session="s2", skills={EOS: current}, passed=True))
    gate = runner.invoke(app, ["benchmark", "gate", EOS, "--json"])
    assert gate.exit_code == 0, gate.output
    report = json.loads(gate.output)
    assert report["validated"] is True and report["cells"][0]["verdict"] == "validated"

    unknown = runner.invoke(app, ["benchmark", "gate", "nope"])
    assert unknown.exit_code == 1 and "no skill named" in unknown.output


def test_cli_score_prints_the_flags_under_each_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".slab"
    session = "20260901-100000-9"
    _transcript(root, session, [_header(), _user(Q1.instruction), _finish(None, None)])
    Workspace(root).close()
    scored = runner.invoke(app, ["benchmark", "score"])
    assert scored.exit_code == 0, scored.output
    assert "fail: the finish carried no structured results  [2 flags]" in scored.output
    assert "  skill-not-loaded       rules    no skill event in the transcript:" in scored.output
    records = benchmark.load_records(tmp_path / "benchmarks" / "results.jsonl")
    assert records[0]["reviewed_by"] == ["rules"] and len(records[0]["flags"]) == 2
