"""``slab benchmark``: the campaigns are scored from the record, not the prose.

The fixtures write transcripts in the vocabulary the loop records (a
session header, the opening user message, a finish event with structured
results) against a real workspace whose runs went through the lifecycle
for real, so the scorer is exercised on the evidence a campaign leaves.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from ase.build import bulk
from typer.testing import CliRunner

from foundation.models import TaskRecord, utcnow
from foundation.runtime import Workspace
from foundation.serialize import dumps
from foundation.tasks import single_point
from mason.client import ChatReply, ToolCall
from mason.mechanisms import ALL_MECHANISMS, CONDITIONS
from slab_stack import benchmark
from slab_stack.cli import app

runner = CliRunner()

Q1 = benchmark.find_question("a0")


def _events_header(model: str = "fake-30b", profile: str = "laptop") -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:00+00:00",
        "type": "session",
        "agent": "pi",
        "model": model,
        "provider": "openai",
        "endpoint": "http://localhost:11434/v1",
        "endpoint_origin": "[agent] endpoint",
        "compute_profile": profile,
        "max_turns": 40,
    }


def _user(text: str) -> dict[str, Any]:
    return {
        "at": "2026-09-01T10:00:01+00:00",
        "type": "message",
        "message": {"role": "user", "content": text},
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


def _verified_run(root: Path, session: str, *, engine: str = "emt") -> str:
    """A real run that computed something and earned verified from a check."""
    with Workspace(root) as ws, ws.start_run(name="eos", session=session) as run:
        single_point(bulk("Cu"), engine=engine)

        @run.check
        def fine() -> bool:
            return True

    return run.id


def _unverified_run(root: Path, session: str) -> str:
    with Workspace(root) as ws, ws.start_run(name="draft", session=session) as run:
        single_point(bulk("Cu"), engine="emt")  # no checks: stays quarantined
    return run.id


PBESOL_FAMILY = "SSSP/1.3.0/PBEsol/efficiency"
PBE_FAMILY = "SSSP/1.3.0/PBE/efficiency"


def _dft_run(root: Path, session: str, *, family: str | None = PBESOL_FAMILY) -> str:
    """A verified run whose task recipe names the qe engine (no pw.x needed).

    The calculator options are traced the way the loop traces them: by
    hash, with the bytes in the artifact store. ``family=None`` leaves the
    options out, which is a recipe the scorer cannot read a functional from.
    """
    with Workspace(root) as ws:
        with ws.start_run(name="scf", session=session) as run:

            @run.check
            def fine() -> bool:
                return True

        params: dict[str, Any] = {"engine": "qe"}
        if family is not None:
            options = {"pseudo_family": family, "kpts": [8, 8, 8]}
            params["calculator_options"] = {"$hash": ws.artifacts.put_bytes(dumps(options))}
        ws.runs.add_task(
            TaskRecord(
                run_id=run.id,
                name="single_point",
                status="completed",
                cache_key="ab" * 32,
                recipe={"params": params},
                inputs={},
                outputs={},
                started_at=utcnow(),
            )
        )
    return run.id


# -- questions ----------------------------------------------------------------


def test_the_questions_are_fixed_and_addressable() -> None:
    assert [q.number for q in benchmark.QUESTIONS] == [1, 2, 3, 4, 5]
    assert benchmark.find_question("3") is benchmark.find_question("vacancy")
    assert "Finish with results key `a0` in Å" in Q1.instruction
    with pytest.raises(benchmark.BenchmarkError, match="known: 1/a0"):
        benchmark.find_question("zz")
    table = benchmark.questions_table()
    assert table.count("\n") == len(benchmark.QUESTIONS) + 1
    assert "| 3.632 Å | 3.562 Å |" in table and "energy_rmse ≤ 5 meV/atom" in table
    assert "| 1.07 eV | no checked value yet |" in table
    # Every class has a tolerance for every result; references may lag (see the docs).
    for q in benchmark.QUESTIONS:
        assert set(q.tolerance) == set(benchmark.CLASSES)
        assert all(set(t) == set(q.results) for t in q.tolerance.values())


# -- scoring ------------------------------------------------------------------


def test_a_verified_in_band_answer_passes(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-1"
    run_id = _verified_run(root, session)
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(Q1.instruction),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id[:10]]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["passed"] is True and record["reason"] is None
    assert record["question"] == 1 and record["key"] == "a0"
    assert record["model"] == "fake-30b" and record["machine"] == "laptop"
    assert record["engine_class"] == "mlip" and record["engines"] == ["emt"]
    assert record["tolerance"] == {"a0": 0.05} and record["reference"] == {"a0": 3.632}
    assert record["run_ids"] == [run_id[:10]]  # a prefix resolves, git-style


@pytest.mark.parametrize(
    ("finish", "reason"),
    [
        (None, "no finish report"),
        (_finish(None, ["x"]), "no structured results"),
        (_finish({"a0": {"value": 3.6, "unit": "Å"}}, None), "cited no run ids"),
        (_finish({"a0": {"value": 3.6, "unit": "Å"}}, ["zzzz"]), "does not exist"),
        (_finish({"a0": {"value": "3.6", "unit": "Å"}}, ["RUN"]), "no numeric result"),
        (_finish({"a0": {"value": 3.9, "unit": "Å"}}, ["RUN"]), "outside 3.632 ± 0.05"),
    ],
)
def test_each_failure_names_its_reason(
    tmp_path: Path, finish: dict[str, Any] | None, reason: str
) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-2"
    run_id = _verified_run(root, session)
    events = [_events_header(), _user(Q1.instruction)]
    if finish is not None:
        if finish.get("run_ids") == ["RUN"]:
            finish = dict(finish, run_ids=[run_id])
        events.append(finish)
    _transcript(root, session, events)
    record = benchmark.score_session(root, session)
    assert record["passed"] is False
    assert reason in record["reason"]


def test_an_unverified_cited_run_fails_even_with_the_right_number(tmp_path: Path) -> None:
    """The verification condition is what makes 'correct' mean computed."""
    root = tmp_path / "ws"
    session = "20260901-100000-3"
    draft = _unverified_run(root, session)
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(Q1.instruction),
            _finish({"a0": {"value": 3.632, "unit": "Å"}}, [draft]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["passed"] is False
    assert "never verified" in record["reason"] and "quarantined" in record["reason"]


def _score_dft(tmp_path: Path, session: str, value: float, **runs: Any) -> dict[str, Any]:
    """Score a Q1 campaign citing dft runs made with the given families."""
    root = tmp_path / "ws"
    run_ids = [_dft_run(root, session, family=family) for family in runs.values()]
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(Q1.instruction),
            _finish({"a0": {"value": value, "unit": "Å"}}, run_ids),
        ],
    )
    return benchmark.score_session(root, session)


def test_a_dft_run_is_judged_against_its_own_functional(tmp_path: Path) -> None:
    """PBEsol binds Cu ~2% tighter than PBE: the correct PBEsol answer must pass."""
    record = _score_dft(tmp_path, "20260901-100000-4", 3.5645, a=PBESOL_FAMILY)
    assert record["engine_class"] == "pbesol" and record["engines"] == ["qe"]
    assert record["reference"] == {"a0": 3.562} and record["tolerance"] == {"a0": 0.03}
    assert record["passed"] is True, record["reason"]

    # The PBE number is wrong for a PBEsol run, and the reason names the band.
    record = _score_dft(tmp_path, "20260901-100000-5", 3.632, a=PBESOL_FAMILY)
    assert record["passed"] is False and "(pbesol band)" in record["reason"]

    # A PBE run is judged against PBE: 3.67 is inside the mlip band, not the pbe one.
    record = _score_dft(tmp_path, "20260901-100000-6", 3.67, a=PBE_FAMILY)
    assert record["engine_class"] == "pbe" and record["reference"] == {"a0": 3.632}
    assert record["passed"] is False and "(pbe band)" in record["reason"]


def test_an_unreadable_or_mixed_functional_fails_with_that_reason(tmp_path: Path) -> None:
    record = _score_dft(tmp_path, "20260901-100000-7", 3.6, a=None)
    assert record["passed"] is False
    assert "cannot tell the functional of cited run" in record["reason"]
    assert record["engine_class"] is None  # the band was never defined

    record = _score_dft(tmp_path, "20260901-100000-8", 3.6, a=PBE_FAMILY, b=PBESOL_FAMILY)
    assert record["passed"] is False
    assert "mix functionals" in record["reason"]
    assert "pbe (" in record["reason"] and "pbesol (" in record["reason"]


def test_a_class_without_a_checked_reference_is_refused_not_guessed(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-9"
    q3 = benchmark.find_question("vacancy")
    run_id = _dft_run(root, session, family=PBESOL_FAMILY)
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(q3.instruction),
            _finish({"e_vac": {"value": 1.2, "unit": "eV"}}, [run_id]),
        ],
    )
    with pytest.raises(benchmark.BenchmarkError, match="no checked pbesol reference for e_vac"):
        benchmark.score_session(root, session)


def test_functional_of_reads_the_traced_options() -> None:
    assert benchmark.functional_of({"pseudo_family": PBE_FAMILY}) == "pbe"
    assert benchmark.functional_of({"pseudo_family": PBESOL_FAMILY}) == "pbesol"
    by_file = {"pseudopotentials": {"Cu": "cu_pbesol_v1.2.uspp.F.UPF"}}
    assert benchmark.functional_of(by_file) == "pbesol"
    assert benchmark.functional_of({"input_data": {"system": {"input_dft": "PBEsol"}}}) == "pbesol"
    assert benchmark.functional_of({"kpts": [4, 4, 4]}) is None
    assert benchmark.functional_of({}) is None


def test_thresholds_judge_the_finetune_question(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    q5 = benchmark.find_question("finetune")
    session = "20260901-100000-5"
    run_id = _verified_run(root, session)
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(q5.instruction),
            _finish(
                {
                    "energy_rmse": {"value": 2.1, "unit": "meV/atom"},
                    "force_rmse": {"value": 0.14, "unit": "eV/Å"},
                },
                [run_id],
            ),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["question"] == 5 and record["passed"] is False
    assert "force_rmse = 0.14 eV/Å exceeds 0.1" in record["reason"]


def test_a_session_that_is_not_a_campaign_is_refused_unless_named(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-6"
    _transcript(root, session, [_events_header(), _user("relax Cu"), _finish(None, None)])
    Workspace(root).close()
    with pytest.raises(benchmark.BenchmarkError, match="did not open with a benchmark"):
        benchmark.score_session(root, session)
    record = benchmark.score_session(root, session, question=Q1, machine="bench-box", model="m")
    assert record["reason"] == "the finish carried no structured results"
    assert record["machine"] == "bench-box" and record["model"] == "m"


# -- records and rendering ----------------------------------------------------


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "question": 1, "key": "a0", "session": "s1", "model": "m-30b", "machine": "laptop",
        "results": {"a0": {"value": 3.6, "unit": "Å"}}, "passed": True, "reason": None,
    }
    return {**base, **overrides}


def test_records_append_and_the_latest_wins_per_cell(tmp_path: Path) -> None:
    path = tmp_path / "benchmarks" / "results.jsonl"
    benchmark.append_record(path, _record())
    benchmark.append_record(path, _record(session="s2", passed=False, reason="no finish report"))
    path.write_text(path.read_text() + "not json\n")
    records = benchmark.load_records(path)
    assert len(records) == 2
    assert benchmark.recorded_sessions(records) == {"s1", "s2"}
    latest = benchmark.latest_by_cell(records)[("m-30b", "laptop", "slab", "a0")]
    assert latest["session"] == "s2"
    # An ablated arm is its own cell, never folded into the plain one.
    ablated = _record(session="s3", condition="slab", ablated=["budget-hint"])
    cells = benchmark.latest_by_cell([*records, ablated])
    assert cells[("m-30b", "laptop", "slab -budget-hint", "a0")]["session"] == "s3"
    assert cells[("m-30b", "laptop", "slab", "a0")]["session"] == "s2"


def test_render_rewrites_only_the_marker_regions(tmp_path: Path) -> None:
    docs = tmp_path / "benchmark.md"
    docs.write_text(
        "# Page\n\nprose above\n\n<!-- benchmark:questions:start -->\nold\n"
        "<!-- benchmark:questions:end -->\n\nmore prose\n\n"
        "<!-- benchmark:conditions:start -->\n<!-- benchmark:conditions:end -->\n\n"
        "<!-- benchmark:mechanisms:start -->\n<!-- benchmark:mechanisms:end -->\n\n"
        "<!-- benchmark:results:start -->\n<!-- benchmark:results:end -->\n\n"
        "<!-- benchmark:flags:start -->\n<!-- benchmark:flags:end -->\n\ntail\n"
    )
    readme = tmp_path / "README.md"
    readme.write_text("# R\n<!-- benchmark:summary:start -->\nx\n<!-- benchmark:summary:end -->\n")
    records = [
        _record(),
        _record(session="s3", key="vacancy", question=3, passed=False, reason="no finish report",
                results={}),
        _record(session="s4", model="tiny-8b", passed=False, reason="the finish cited no run ids",
                flags=[{"rule": "finish-incomplete", "target": "card:pi", "evidence": "finish",
                        "note": "the finish cited no run ids", "raised_by": "rules"}]),
    ]
    changed = benchmark.render(records, docs=docs, readme=readme)
    assert changed == [docs, readme]
    text = docs.read_text()
    assert "prose above" in text and "more prose" in text and text.endswith("tail\n")
    assert "-->\nold\n" not in text
    assert "| 1 | Determine the equilibrium lattice constant" in text
    assert "| m-30b | laptop | slab | pass (3.6 Å)" in text
    assert "fail (no finish report)" in text
    assert "fail (3.6 Å; the finish cited no run ids; 1 flag)" in text
    assert (
        "| card:pi | finish-incomplete | open | — | Q1 | tiny-8b | laptop | slab | rules |" in text
    )
    assert "| 1/5 |" in text and "| 0/5 |" in text
    assert "| `protocol` | `protocol` |" in text and "| `check-gating` |" in text
    summary = readme.read_text()
    assert "| m-30b | laptop | slab | 1/5 |" in summary
    assert "| tiny-8b | laptop | slab | 0/5 |" in summary
    assert "the benchmark](https://tarbaugh.github.io/SLAB/benchmark/)" in summary
    # Idempotent: a second render changes nothing.
    assert benchmark.render(records, docs=docs, readme=readme) == []
    # A file without the markers is refused, never rewritten.
    bare = tmp_path / "bare.md"
    bare.write_text("no markers\n")
    with pytest.raises(benchmark.BenchmarkError, match="has no"):
        benchmark.rewrite_region(bare, "results", "x")
    assert bare.read_text() == "no markers\n"


def test_the_retention_table_reads_each_records_numbers() -> None:
    table = benchmark.retention_table(
        [
            _record(),
            _record(session="s2", retention={"error": "locked"}),
            _record(
                session="s3",
                retention={"mode": "purge", "runs_total": 4, "runs_promoted": 1,
                           "runs_expired": 3, "bytes_total": 0, "bytes_promoted": 0,
                           "bytes_expired": 0},
            ),
        ]
    )
    lines = table.splitlines()
    assert lines[2].endswith("| not recorded | not recorded | not recorded |")
    assert "retire failed: locked" in lines[3]
    assert lines[4].endswith("| 1/4 (25 %) | 0/0 | purge |")
    assert benchmark.retention_table([]) == "No campaign has been scored yet."


def test_render_with_no_records_says_so(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("<!-- benchmark:summary:start -->\n<!-- benchmark:summary:end -->\n")
    benchmark.render([], docs=None, readme=readme)
    assert "No campaign has been scored yet" in readme.read_text()


# -- the CLI ------------------------------------------------------------------


def test_cli_list_score_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".slab"
    session = "20260901-100000-7"
    run_id = _verified_run(root, session)
    _transcript(
        root,
        session,
        [_events_header("big-70b", "cluster"), _user(Q1.instruction),
         _finish({"a0": {"value": 3.61, "unit": "Å"}}, [run_id]),
         {"at": "2026-09-01T10:05:01+00:00", "type": "retire", "mode": "expire",
          "runs_total": 2, "runs_promoted": 1, "runs_expired": 1,
          "bytes_total": 40, "bytes_promoted": 10, "bytes_expired": 30}],
    )
    _transcript(root, "20260901-100000-8", [_events_header(), _user("unrelated chat")])

    listed = runner.invoke(app, ["benchmark", "list"])
    assert listed.exit_code == 0, listed.output
    assert "1. [a0]" in listed.output and "mlip: a0 = 3.632 ± 0.05" in listed.output

    scored = runner.invoke(app, ["benchmark", "score", "--machine", "hpc-a"])
    assert scored.exit_code == 0, scored.output
    assert (
        "Q1 a0        big-70b                  hpc-a        slab      pass  [1 flag]"
        in scored.output
    )
    assert "  script-not-used        rules    loaded at step" not in scored.output
    assert "  skill-not-loaded       rules    no skill event" in scored.output
    records = benchmark.load_records(tmp_path / "benchmarks" / "results.jsonl")
    assert len(records) == 1 and records[0]["machine"] == "hpc-a"

    again = runner.invoke(app, ["benchmark", "score"])
    assert again.exit_code == 0, again.output
    assert "nothing new to score (1 already recorded)" in again.output

    as_json = runner.invoke(app, ["benchmark", "score", "--rescore", "--json"])
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output)[0]["passed"] is True

    docs = tmp_path / "docs" / "benchmark.md"
    docs.parent.mkdir()
    docs.write_text(
        "<!-- benchmark:questions:start -->\n<!-- benchmark:questions:end -->\n"
        "<!-- benchmark:conditions:start -->\n<!-- benchmark:conditions:end -->\n"
        "<!-- benchmark:mechanisms:start -->\n<!-- benchmark:mechanisms:end -->\n"
        "<!-- benchmark:results:start -->\n<!-- benchmark:results:end -->\n"
        "<!-- benchmark:flags:start -->\n<!-- benchmark:flags:end -->\n"
    )
    (tmp_path / "README.md").write_text(
        "<!-- benchmark:summary:start -->\n<!-- benchmark:summary:end -->\n"
    )
    rendered = runner.invoke(app, ["benchmark", "tables"])
    assert rendered.exit_code == 0, rendered.output
    assert "rewrote docs/benchmark.md" in rendered.output and "rewrote README.md" in rendered.output
    assert "| big-70b | hpc-a | slab | 1/5 |" in (tmp_path / "README.md").read_text()

    retention = runner.invoke(app, ["benchmark", "tables", "--retention"])
    assert retention.exit_code == 0, retention.output
    assert (
        "| 20260901-100000-7 | big-70b | hpc-a | slab | Q1 | 1/2 (50 %) | 10/40 (25 %) | expire |"
        in retention.output
    )
    assert records[0]["retention"]["runs_promoted"] == 1

    unknown = runner.invoke(app, ["benchmark", "score", "--session", "1999"])
    assert unknown.exit_code == 1 and "no session transcript matches" in unknown.output


def test_cli_render_writes_the_job_files_without_submitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render is for tweaks the config cannot express: the files land, nothing runs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(
        "[agent]\nmodel = \"m\"\nendpoint = \"http://gateway.example/v1\"\n"
        "api_key_env = \"GATEWAY_API_KEY\"\n"
        "[agent.sandbox]\nimage = \"/shared/sw/slab-sandbox.sif\"\n"
        "[hpc]\ndefault_partition = \"cpu\"\n[hpc.partitions.cpu]\ntime_limit = \"04:00:00\"\n"
    )
    monkeypatch.setenv("GATEWAY_API_KEY", "not-a-real-key")
    result = runner.invoke(app, ["benchmark", "render", "vacancy", "--partition", "cpu"])
    assert result.exit_code == 0, result.output
    script = tmp_path / "sandbox" / "mason-sandbox.sbatch"
    assert script.is_file()
    assert benchmark.find_question("vacancy").instruction.split(".")[0] in script.read_text()
    assert f"then: sbatch {script}" in result.output
    assert not (tmp_path / "benchmarks").exists()  # nothing scored, nothing recorded


# -- a harness over MCP -------------------------------------------------------


def test_a_harness_session_over_mcp_is_scored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Question 1 driven through the MCP tools by a scripted client, the way
    an external harness would drive it, and scored from the record it leaves."""
    pytest.importorskip("mcp", reason="mcp extra not installed")
    import asyncio

    from foundation.mcp_server import build_server

    monkeypatch.chdir(tmp_path)
    root = tmp_path / ".slab"
    server = build_server(root, project=tmp_path, session="mcp-q1")

    def call(tool: str, args: dict[str, Any] | None = None) -> Any:
        result = asyncio.run(server.call_tool(tool, args or {}))
        structured = getattr(result, "structured_content", None)
        if isinstance(result, tuple):
            structured = result[1]
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured

    # The client looks before it computes, the way the instruction expects.
    assert "emt" in json.dumps(call("list_engines"))
    assert "equation-of-state" in {s["name"] for s in call("list_skills")}
    call("skill", {"name": "equation-of-state"})
    (tmp_path / "a0.py").write_text(
        "from ase.build import bulk\n"
        "from foundation import check, converged\n"
        "from foundation.tasks import relax_cell\n"
        "atoms = bulk('Cu', 'fcc', a=3.6)\n"
        "relaxed, info = relax_cell(atoms, engine='emt', fmax=0.01)\n"
        "a0 = (4 * relaxed.get_volume()) ** (1 / 3)\n"
        "print(f'a0 = {a0:.6f} Å')\n"
        "@check\n"
        "def forces_converged():\n"
        "    return converged(info['fmax'], below=0.01)\n"
    )
    launched = call(
        "launch_workflow", {"script_path": "a0.py", "intent": "Q1: a0 of fcc Cu under emt"}
    )
    assert launched["state"] == "verified", launched
    a0 = float(launched["output"].split("a0 = ")[1].split()[0])
    call("notebook", {"entry": f"a0 = {a0} Å (run {launched['run_id'][:10]})", "heading": "Q1"})
    reported = call(
        "report_results",
        {"results": {"a0": {"value": a0, "unit": "Å"}}, "run_ids": [launched["run_id"]]},
    )
    assert reported["session"] == "mcp-q1"

    with pytest.raises(benchmark.BenchmarkError, match="pass --question"):
        benchmark.score_session(root, "mcp-q1")
    record = benchmark.score_session(root, "mcp-q1", question=Q1)
    assert record["passed"] is True and record["reason"] is None, record
    assert record["session"] == "mcp-q1" and record["agent"] == "mcp"
    assert record["engine_class"] == "mlip" and record["engines"] == ["emt"]
    assert record["run_ids"] == [launched["run_id"]]
    from foundation.skills import discover_skills

    eos = discover_skills(tmp_path)["equation-of-state"]
    assert record["skills"] == {"equation-of-state": eos.digest}
    assert record["reviewed_by"] == [] and record["flags"] == []
    assert record["model"] == "unknown" and record["steps"] is None

    scored = runner.invoke(
        app, ["benchmark", "score", "--session", "mcp-q1", "--question", "1", "--model", "claude"]
    )
    assert scored.exit_code == 0, scored.output
    assert "pass" in scored.output
    (row,) = benchmark.load_records(tmp_path / "benchmarks" / "results.jsonl")
    assert row["session"] == "mcp-q1" and row["model"] == "claude"
# -- the conditions -----------------------------------------------------------

SLAB_TOML = (
    '[agent]\nmodel = "m"\nendpoint = "http://gateway.example/v1"\n'
    'api_key_env = "GATEWAY_API_KEY"\n'
    '[agent.sandbox]\nimage = "/shared/sw/slab-sandbox.sif"\n'
    '[hpc]\ndefault_partition = "cpu"\n[hpc.partitions.cpu]\ntime_limit = "04:00:00"\n'
)

Reply = ChatReply | Callable[[list[dict[str, Any]]], ChatReply]


class _Scripted:
    """A chat backend that answers from a script; an entry may read the messages."""

    def __init__(self, replies: list[Reply]) -> None:
        self.replies = list(replies)

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **_: Any
    ) -> ChatReply:
        answer = self.replies.pop(0)
        return answer(messages) if callable(answer) else answer


def _tool(name: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"call_{name}_{len(json.dumps(arguments))}",
                name=name,
                arguments=dict(arguments),
                arguments_raw=json.dumps(arguments),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _last_tool_result(messages: list[dict[str, Any]]) -> str:
    return str(next(m for m in reversed(messages) if m.get("role") == "tool")["content"])


def _campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str, replies: list[Reply]
) -> dict[str, Any]:
    """Run Q1 under *condition* with a scripted backend, then score it."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    client = _Scripted(replies)
    monkeypatch.setattr("mason.loop.client_from_config", lambda agent, keys=None: client)
    root = tmp_path / "ws"
    session_id, result = benchmark.run_campaign(
        Q1, workspace=root, model="fake", condition=condition, max_turns=12
    )
    assert result.stop_reason == "finish", result.text
    return benchmark.score_session(root, session_id, machine="laptop")


def test_run_campaign_hands_the_questions_result_keys_to_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expected keys travel explicitly from the question, never parsed
    back out of the goal text: a finish under another name is refused,
    and a finish under the asked name is honored."""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    q4 = benchmark.find_question("4")
    refusals: list[str] = []

    def second(messages: list[dict[str, Any]]) -> ChatReply:
        refusals.append(_last_tool_result(messages))
        return _tool(
            "finish", report="melts at 1350 K", results={"t_melt": {"value": 1350, "unit": "K"}}
        )

    client = _Scripted(
        [
            _tool(
                "finish", report="melts at 1350 K", results={"t_m": {"value": 1350, "unit": "K"}}
            ),
            second,
        ]
    )
    monkeypatch.setattr("mason.loop.client_from_config", lambda agent, keys=None: client)
    seen: dict[str, object] = {}
    from mason import loop as mason_loop

    original = mason_loop.Mason.__init__

    def spy(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen["expected_results"] = kwargs.get("expected_results")
        original(self, *args, **kwargs)

    monkeypatch.setattr(mason_loop.Mason, "__init__", spy)
    _session_id, result = benchmark.run_campaign(q4, workspace=tmp_path / "ws", model="fake")
    assert seen["expected_results"] == {"t_melt": "K"}
    assert result.stop_reason == "finish" and result.steps == 2
    assert set(result.results) == {"t_melt"}
    assert refusals == [
        "finish not honored: results name t_m; the goal asks for t_melt in K; "
        "call finish again with results keyed exactly so"
    ]


EOS_WORKFLOW = """\
from ase.build import bulk
from foundation import check
from foundation.tasks import single_point

atoms, info = single_point(bulk("Cu"), engine="emt")
print("a0 (Å): 3.61")


@check
def computed():
    return "energy" in info
"""

EOS_PLAIN = 'print("a0 (Å): 3.61")\n'


def test_the_slab_condition_passes_with_a_verified_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def finish_with_the_run(messages: list[dict[str, Any]]) -> ChatReply:
        launched = _last_tool_result(messages)
        assert launched.startswith("run ") and "state=verified" in launched, launched
        run_id = launched.split()[1].rstrip(":")
        return _tool(
            "finish",
            report=f"a0 = 3.61 Å (run {run_id})",
            results={"a0": {"value": 3.61, "unit": "Å"}},
            run_ids=[run_id],
        )

    record = _campaign(
        tmp_path,
        monkeypatch,
        "slab",
        [
            _tool("write_file", path="eos.py", content=EOS_WORKFLOW),
            _tool("launch_workflow", script="eos.py", intent="the benchmark's eos"),
            finish_with_the_run,
        ],
    )
    assert record["passed"] is True, record["reason"]
    assert record["condition"] == "slab" and record["ablated"] == []
    assert set(record["mechanisms"]) == ALL_MECHANISMS
    assert record["agent"] == "pi" and record["engines"] == ["emt"]
    # The finish promoted the run it cited; the record carries the numbers.
    retention = record["retention"]
    assert (retention["runs_total"], retention["runs_promoted"], retention["runs_expired"]) == (
        1, 1, 0
    )
    assert retention["mode"] == "expire" and retention["bytes_promoted"] > 0


def test_the_protocol_condition_logs_its_provenance_and_still_fails_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same correct number, written down in the log, has no verified run
    behind it. The scorer's rule is the same for every arm, so it fails."""
    python = sys.executable
    record = _campaign(
        tmp_path,
        monkeypatch,
        "protocol",
        [
            _tool("write_file", path="eos.py", content=EOS_PLAIN),
            _tool("shell", command=f"{python} eos.py"),
            _tool(
                "shell",
                command="printf '# Provenance\\n\\n- eos.py: python eos.py; a0 (Å): 3.61; "
                "checked: printed\\n' >> PROVENANCE.md",
            ),
            _tool(
                "finish",
                report="a0 = 3.61 Å (PROVENANCE.md, entry 1)",
                results={"a0": {"value": 3.61, "unit": "Å"}},
            ),
        ],
    )
    assert record["passed"] is False
    assert record["reason"] == "the finish cited no run ids"
    assert record["condition"] == "protocol" and record["agent"] == "protocol"
    assert set(record["mechanisms"]) == CONDITIONS["protocol"].mechanisms
    assert "a0 (Å): 3.61" in (tmp_path / "project" / "PROVENANCE.md").read_text()
    assert record["results"] == {"a0": {"value": 3.61, "unit": "Å"}}


def test_the_bare_condition_runs_a_shell_and_fails_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _campaign(
        tmp_path,
        monkeypatch,
        "bare",
        [
            _tool("shell", command=f"{sys.executable} -c 'print(3.61)'"),
            _tool(
                "finish",
                report="a0 = 3.61 Å",
                results={"a0": {"value": 3.61, "unit": "Å"}},
            ),
        ],
    )
    assert record["passed"] is False
    assert record["reason"] == "the finish cited no run ids"
    assert record["condition"] == "bare" and record["agent"] == "bare"
    assert record["mechanisms"] == []
    # No run tools, no runs: the finish retired nothing, and the record says so.
    assert record["retention"]["runs_total"] == 0
    assert record["retention"]["runs_promoted"] == 0


def test_a_protocol_campaign_passes_only_through_a_verified_run(tmp_path: Path) -> None:
    """The asymmetry cuts one way: an arm without run tools can still pass
    when the agent found the traced path itself and cited the run."""
    root = tmp_path / "ws"
    session = "20260901-100000-9"
    run_id = _verified_run(root, session)
    header = {
        **_events_header(),
        "agent": "protocol",
        "condition": "protocol",
        "mechanisms": sorted(CONDITIONS["protocol"].mechanisms),
        "ablated": [],
    }
    _transcript(
        root,
        session,
        [header, _user(Q1.instruction), _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id])],
    )
    record = benchmark.score_session(root, session)
    assert record["passed"] is True and record["condition"] == "protocol"
    assert benchmark.cell_of(record) == ("fake-30b", "laptop", "protocol", "a0")


def test_a_transcript_from_before_the_switches_scores_as_slab(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    session = "20260901-100000-10"
    run_id = _verified_run(root, session)
    _transcript(
        root,
        session,
        [
            _events_header(),
            _user(Q1.instruction),
            _finish({"a0": {"value": 3.60, "unit": "Å"}}, [run_id]),
        ],
    )
    record = benchmark.score_session(root, session)
    assert record["condition"] == "slab" and set(record["mechanisms"]) == ALL_MECHANISMS


def test_the_cli_carries_the_condition_into_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(SLAB_TOML)
    monkeypatch.setenv("GATEWAY_API_KEY", "not-a-real-key")
    result = runner.invoke(
        app, ["benchmark", "render", "1", "--partition", "cpu", "--condition", "protocol"]
    )
    assert result.exit_code == 0, result.output
    script = (tmp_path / "sandbox" / "mason-sandbox.sbatch").read_text()
    # The unit is non-ASCII, so shlex quotes it, and the inner line is
    # quoted once more inside the apptainer command; assert on the pieces.
    assert "mason run --auto --condition protocol --expect " in script and "a0:Å" in script
    record = json.loads((tmp_path / "sandbox" / "render.json").read_text())
    assert record["condition"] == "protocol" and record["without"] == []
    assert record["expected_results"] == {"a0": "Å"}
    refused = runner.invoke(app, ["benchmark", "render", "1", "--condition", "aicc"])
    assert refused.exit_code == 1 and "no condition named 'aicc'" in refused.output


def test_matrix_cells_and_the_matrix_verb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cells = benchmark.matrix_cells(["slab", "bare"], [Q1], ["budget-hint"])
    assert [c.subdir for c in cells] == [
        "slab/q1-a0", "bare/q1-a0", "slab-without-budget-hint/q1-a0"
    ]
    with pytest.raises(benchmark.BenchmarkError, match="no mechanism named 'hint'"):
        benchmark.matrix_cells(["slab"], [Q1], ["hint"])
    with pytest.raises(benchmark.BenchmarkError, match="no condition named 'aicc'"):
        benchmark.matrix_cells(["aicc"], [Q1])

    monkeypatch.chdir(tmp_path)
    (tmp_path / "slab.toml").write_text(SLAB_TOML)
    monkeypatch.setenv("GATEWAY_API_KEY", "not-a-real-key")
    result = runner.invoke(
        app,
        ["benchmark", "matrix", "--partition", "cpu", "--condition", "slab", "--condition",
         "bare", "--question", "1", "--question", "vacancy", "--without", "budget-hint"],
    )
    assert result.exit_code == 0, result.output
    grid = tmp_path / "sandbox" / "matrix"
    manifest = json.loads((grid / "matrix.json").read_text())
    assert [m["harness"] for m in manifest] == ["slab"] * 2 + ["bare"] * 2 + [
        "slab -budget-hint"
    ] * 2
    assert [m["question"] for m in manifest] == [1, 3, 1, 3, 1, 3]
    for cell in manifest:
        assert Path(cell["script"]).is_file()
    bare = (grid / "bare" / "q3-vacancy" / "mason-sandbox.sbatch").read_text()
    assert "--condition bare" in bare and "--without" not in bare
    assert "monovacancy" in bare
    ablated = (grid / "slab-without-budget-hint" / "q1-a0" / "mason-sandbox.sbatch").read_text()
    assert "--condition slab --without budget-hint" in ablated
    assert "6 job(s) rendered" in result.output and "nothing submitted" in result.output
    assert not (tmp_path / "benchmarks").exists()
    # The default grid is every condition by every question.
    assert len(benchmark.matrix_cells(list(benchmark.CONDITIONS), list(benchmark.QUESTIONS))) == 15
