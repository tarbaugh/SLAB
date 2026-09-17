"""A finish is refused once without verified evidence, and the evidence is reachable.

Two faults from one campaign. The lead finished with four cited runs, none of
them verified, and learned it only from the scorer hours later. The number it
reported came from log files it could not read, because ``read_artifact``
only ever opened registered artifacts, so it briefed a worker to cat them
through the shell.
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from foundation import Workspace
from mason.client import ChatReply, ToolCall
from mason.config import MasonConfig
from mason.loop import Mason
from mason.session import MasonSession
from mason.tools import Toolbox, build_toolbox

LAMMPS_LOG = """\
LAMMPS (2 Aug 2023)
units metal
atom_style atomic
lattice bcc 3.30
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
pair_style eam/alloy
mass 1 92.906
velocity all create 300.0 4928459
fix 1 all nvt temp 300.0 300.0 0.1
thermo 100
Per MPI rank memory allocation (min/avg/max) = 3.1 | 3.1 | 3.1 Mbytes
   Step          Temp          E_pair         E_mol          TotEng         Press
         0   300           -1739.2         0             -1729.3         1421.7
       100   298.41        -1738.9         0             -1729.1         1338.2
       200   301.02        -1738.7         0             -1728.8         1290.5
"""


def _session(tmp_path: Path, **agent: object) -> MasonSession:
    config = MasonConfig.model_validate({"agent": {"model": "fake", **agent}})
    return MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", agent=config.agent, auto_approve=True
    )


class FakeClient:
    """Answers from a script; records every request it saw."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **_: Any
    ) -> ChatReply:
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)


def _tool_reply(name: str, **arguments: object) -> ChatReply:
    return ChatReply(
        content=None,
        tool_calls=(
            ToolCall(
                id=f"call_{name}",
                name=name,
                arguments=dict(arguments),
                arguments_raw=json.dumps(arguments),
            ),
        ),
        prompt_tokens=100,
        completion_tokens=10,
    )


def _events(session: MasonSession) -> list[dict[str, Any]]:
    return [json.loads(line) for line in session.transcript_path.read_text().splitlines()]


def _tool_results(client: FakeClient) -> list[str]:
    """Every tool-role message the client was handed, in order."""
    return [
        str(message["content"])
        for messages in client.requests
        for message in messages
        if message.get("role") == "tool"
    ]


def _stamped_run(session: MasonSession, name: str, *, verified: bool = True) -> str:
    """A run this session made, verified or left quarantined by a failed check."""
    with Workspace(session.workspace_root) as ws, ws.start_run(
        name=name, session=session.session_id
    ) as run:
        run.check(lambda: verified, name="gate")
    return run.id


# -- part A: the finish gate ---------------------------------------------------


def test_a_finish_citing_only_an_unverified_run_is_refused_once(tmp_path: Path) -> None:
    """The refusal names every cited run and how it stands, and the identical
    finish after it stands, recorded as unverified."""
    session = _session(tmp_path)
    quarantined = _stamped_run(session, "eos", verified=False)
    finishing = _tool_reply(
        "finish",
        report="a0 = 3.30 A",
        results={"a0": {"value": 3.30, "unit": "A"}},
        run_ids=[quarantined, "01nosuchrun"],
    )
    client = FakeClient([finishing, finishing])
    result = Mason(session, client=client).run_turn("measure a0")

    refusal = _tool_results(client)[0]
    assert refusal.startswith("finish refused: none of the cited runs is verified (")
    assert f"{quarantined[:10]} quarantined 0/1 checks" in refusal
    assert "01nosuchru no such run" in refusal
    assert "A campaign is scored on verified runs" in refusal
    assert "finish again with the same report" in refusal

    assert result.finished and result.results == {"a0": {"value": 3.30, "unit": "A"}}
    finish = next(e for e in _events(session) if e["type"] == "finish")
    assert finish["unverified"] is True


def test_one_verified_run_among_the_cited_passes_first_time(tmp_path: Path) -> None:
    session = _session(tmp_path)
    good = _stamped_run(session, "eos")
    shaky = _stamped_run(session, "smoke", verified=False)
    client = FakeClient([_tool_reply("finish", report="a0 = 3.30 A", run_ids=[shaky, good])])
    result = Mason(session, client=client).run_turn("measure a0")
    assert result.finished and client.replies == []
    finish = next(e for e in _events(session) if e["type"] == "finish")
    assert "unverified" not in finish


def test_the_gate_leaves_a_delegated_childs_finish_alone(tmp_path: Path) -> None:
    """A specialist's finish hands work back to its lead; the campaign's
    evidence is the lead's to settle."""
    session = _session(tmp_path)
    quarantined = _stamped_run(session, "eos", verified=False)
    client = FakeClient([_tool_reply("finish", report="relaxed", run_ids=[quarantined])])
    result = Mason(session, client=client, depth=1).run_turn("relax it")
    assert result.finished and client.replies == []


def test_the_gate_is_off_without_check_gating(tmp_path: Path) -> None:
    session = _session(tmp_path, mechanisms=["delegation"])
    quarantined = _stamped_run(session, "eos", verified=False)
    client = FakeClient([_tool_reply("finish", report="a0 = 3.30 A", run_ids=[quarantined])])
    assert Mason(session, client=client).run_turn("measure a0").finished
    assert client.replies == []


def test_the_scorer_reports_the_unverified_flag(tmp_path: Path) -> None:
    """``mason report`` carries the flag out of the transcript, and the
    record the scorer writes keeps it beside the failure line."""
    from mason.report import summarize

    transcript = tmp_path / "campaign.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "finish",
                "report": "a0 = 3.30 A",
                "results": {"a0": {"value": 3.30, "unit": "A"}},
                "run_ids": ["01abc"],
                "unverified": True,
            }
        )
        + "\n"
    )
    summary = summarize(transcript)
    assert summary["finish"]["unverified"] is True


# -- part B: evidence of a running or dead run ---------------------------------


def _scratch_of(root: Path, run_id: str, name: str = "slab-lammps-script-t1") -> Path:
    """A scratch directory the run owns, the way an engine leaves one behind."""
    from slab.scratch import Owner, this_host

    made = root / name
    made.mkdir(parents=True)
    owner = Owner(
        pid=os.getpid(),
        host=this_host(),
        created_at="2026-09-17T00:00:00+00:00",
        prefix="slab-lammps-script-",
        run_id=run_id,
    )
    (made / ".slab-owner").write_text(owner.model_dump_json() + "\n", encoding="utf-8")
    return made


@pytest.fixture()
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Toolbox, str, Path]:
    """A run still going, its scratch directory holding a LAMMPS log."""
    from foundation import _ops
    from foundation.lifecycle import ExecutionStatus
    from foundation.models import Run

    root = tmp_path / "scratch"
    monkeypatch.setattr(_ops, "scratch_root", lambda: root)
    session = _session(tmp_path)
    with Workspace(session.workspace_root) as ws:
        run = ws.runs.create(Run(name="md", session=session.session_id))
        ws.runs.set_status(run.id, ExecutionStatus.RUNNING)
    scratch = _scratch_of(root, run.id)
    (scratch / "log.lammps").write_text(LAMMPS_LOG)
    return build_toolbox(session), run.id, scratch


def _read(box: Toolbox, **arguments: object) -> str:
    call = ToolCall(id="ra", name="read_artifact", arguments=arguments, arguments_raw="{}")
    return box.dispatch(call)


def test_read_artifact_reads_a_running_runs_live_log(
    live: tuple[Toolbox, str, Path],
) -> None:
    """No artifact is registered while the run is going, and the log is the
    only evidence it has. It reads through the LAMMPS digest, and raw."""
    box, run_id, _ = live
    digested = _read(box, run_id=run_id, name="log.lammps")
    head = digested.splitlines()[0]
    assert head == (
        f"log.lammps (live file of run {run_id}, {len(LAMMPS_LOG)} bytes so far, not an artifact)"
    )
    assert "LAMMPS log digest" in digested
    raw = _read(box, run_id=run_id, name="log.lammps", raw=True)
    assert raw.splitlines()[0] == head
    assert "     1\tLAMMPS (2 Aug 2023)" in raw
    missing = _read(box, run_id=run_id, name="dump.final")
    assert missing.startswith("no artifact named 'dump.final'")


def test_show_run_lists_the_live_files_of_a_running_run(
    live: tuple[Toolbox, str, Path],
) -> None:
    box, run_id, scratch = live
    (scratch / "tables").mkdir()
    (scratch / "tables" / "temp.dat").write_text("# Time Temp\n0 300\n")
    record = json.loads(box.dispatch(ToolCall(
        id="sr", name="show_run", arguments={"run_id": run_id}, arguments_raw="{}"
    )))
    assert record["live_files"] == [
        {"name": "log.lammps", "bytes": len(LAMMPS_LOG)},
        {"name": "tables/temp.dat", "bytes": 18},
    ]


def test_a_registered_artifact_wins_over_a_live_file_of_the_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foundation import _ops

    root = tmp_path / "scratch"
    monkeypatch.setattr(_ops, "scratch_root", lambda: root)
    session = _session(tmp_path)
    kept = tmp_path / "log.lammps"
    kept.write_text("the registered bytes\n")
    with Workspace(session.workspace_root) as ws, ws.start_run(
        name="md", session=session.session_id
    ) as run:
        run.keep("log.lammps", kept)
    _scratch_of(root, run.id)
    (root / "slab-lammps-script-t1" / "log.lammps").write_text("the live bytes\n")
    shown = _read(build_toolbox(session), run_id=run.id, name="log.lammps", raw=True)
    assert "the registered bytes" in shown
    assert "live file of run" not in shown


def test_a_dead_runs_scratch_file_is_readable_until_the_sweep_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that died before it registered anything leaves its files where
    they lie. That is the only evidence of what went wrong."""
    import shutil

    from foundation import _ops
    from foundation.lifecycle import ExecutionStatus
    from foundation.models import Run

    root = tmp_path / "scratch"
    monkeypatch.setattr(_ops, "scratch_root", lambda: root)
    session = _session(tmp_path)
    with Workspace(session.workspace_root) as ws:
        run = ws.runs.create(Run(name="md", session=session.session_id))
        ws.runs.set_status(run.id, ExecutionStatus.RUNNING)
        ws.runs.set_status(run.id, ExecutionStatus.FAILED, error="killed")
    scratch = _scratch_of(root, run.id)
    (scratch / "md.screen").write_text("ERROR on proc 0: Out of memory\n")
    box = build_toolbox(session)
    shown = _read(box, run_id=run.id, name="md.screen")
    assert "Out of memory" in shown
    assert f"live file of run {run.id}" in shown
    # A failed run is not running, so show_run lists no live files for it.
    record = json.loads(box.dispatch(ToolCall(
        id="sr", name="show_run", arguments={"run_id": run.id}, arguments_raw="{}"
    )))
    assert "live_files" not in record
    shutil.rmtree(scratch)
    assert _read(box, run_id=run.id, name="md.screen").startswith("no artifact named")


# -- part C: dry-run records ---------------------------------------------------


@pytest.fixture()
def dry(tmp_path: Path) -> tuple[Toolbox, str]:
    """A dry run whose LAMMPS step failed, so its files were kept as a record."""
    from test_lammps_script import _FAKE, _script

    fake = _script(tmp_path / "fake-lmp", _FAKE)
    session = _session(tmp_path)
    box = build_toolbox(session)
    (tmp_path / "md.py").write_text(
        "from foundation.tasks import run_lammps\n"
        f"run_lammps('units metal\\npair_style nonsense\\nrun 10\\n', label='md', "
        f"command={fake!r})\n"
    )
    answer = box.dispatch(ToolCall(
        id="lw",
        name="launch_workflow",
        arguments={"script": "md.py", "dry_run": True},
        arguments_raw="{}",
    ))
    (line,) = [ln for ln in answer.splitlines() if ln.startswith("dry-run record ")]
    return box, line.split()[2]


def test_a_dry_id_reads_through_hash_as_well_as_run_id(dry: tuple[Toolbox, str]) -> None:
    """The footer names a dry id, and a lead that read it as a hash got an
    error. A dry id is an id wherever it is offered."""
    box, record_id = dry
    shown = _read(box, hash=record_id, name="md-failed.log", raw=True)
    assert shown.startswith("md-failed.log (")
    assert f"dry-run record {record_id})" in shown


def test_a_dry_id_without_a_name_lists_the_files_it_holds(dry: tuple[Toolbox, str]) -> None:
    box, record_id = dry
    listed = _read(box, run_id=record_id)
    assert listed.startswith(f"dry-run record {record_id} holds 3 file(s)")
    assert "  md-failed.log" in listed
    assert f"read_artifact run_id={record_id} name=<file>" in listed


def test_show_run_on_a_dry_id_returns_the_record(dry: tuple[Toolbox, str]) -> None:
    box, record_id = dry
    shown = box.dispatch(ToolCall(
        id="sr", name="show_run", arguments={"run_id": record_id}, arguments_raw="{}"
    ))
    assert shown.startswith(f"dry-run record {record_id};")
    record = json.loads(shown.split("\n", 1)[1])
    assert record["id"] == record_id
    assert record["files"] == ["md-failed.in", "md-failed.log", "md-failed.screen"]
    assert record["script"].endswith(".in") or record["script"]
    assert Path(record["path"]).is_dir()
    gone = box.dispatch(ToolCall(
        id="sr",
        name="show_run",
        arguments={"run_id": "dry-20260101-000000-0000"},
        arguments_raw="{}",
    ))
    assert gone.startswith("no dry-run record dry-20260101-000000-0000")
