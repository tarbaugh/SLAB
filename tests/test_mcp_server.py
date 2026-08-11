"""MCP server tests: tool registration and behavior through the real tool layer."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="mcp extra not installed")

from slab import Workspace
from slab.mcp_server import build_server

EXPECTED_TOOLS = {"list_runs", "show_run", "promote_run", "expire_runs", "gc", "launch_workflow"}


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def _call(server: Any, tool: str, args: dict[str, Any] | None = None) -> Any:
    """Invoke a tool through the real MCP tool layer; normalize across mcp versions."""
    result = asyncio.run(server.call_tool(tool, args or {}))
    if isinstance(result, tuple):  # some mcp versions: (content_blocks, structured)
        blocks, structured = result
    else:  # mcp 2.x: CallToolResult
        blocks = getattr(result, "content", [])
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        # the SDK wraps non-object results as {"result": ...}
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [b.text for b in blocks if getattr(b, "text", None)]
    return json.loads(texts[0]) if texts else None


def _seed(root: Path, *, verified: bool = True) -> str:
    with Workspace(root) as ws:
        with ws.start_run(name="seeded", intent="mcp test") as run:
            # content differs per outcome so gc tests don't hit shared-hash dedup
            run.keep("out", {"e": -2.0, "verified": verified})
            run.check(lambda: verified, name="gate")
        return run.id


def test_expected_tools_registered(root: Path) -> None:
    server = build_server(root)
    tools = asyncio.run(server.list_tools())
    by_name = {t.name: t for t in tools}
    assert set(by_name) == EXPECTED_TOOLS
    for tool in by_name.values():
        assert tool.description  # agents read these; empty would be a regression


def test_list_and_show(root: Path) -> None:
    run_id = _seed(root)
    server = build_server(root)

    runs = _call(server, "list_runs", {"state": "verified"})
    assert [r["id"] for r in runs] == [run_id]

    details = _call(server, "show_run", {"run_id": run_id[:8]})
    assert details["run"]["id"] == run_id
    assert details["checks"][0]["passed"] is True
    assert details["artifacts"][0]["bytes_available"] is True


def test_promote_expire_gc_flow(root: Path) -> None:
    keeper = _seed(root, verified=True)
    goner = _seed(root, verified=False)
    server = build_server(root)

    promoted = _call(server, "promote_run", {"run_id": keeper, "reason": "agent liked it"})
    assert promoted["state"] == "promoted"

    untouched = _call(server, "expire_runs", {})  # policy default: nothing is overdue
    assert untouched["count"] == 0

    expired = _call(server, "expire_runs", {"older_than": "0d"})
    assert expired["count"] == 1
    assert expired["expired"][0]["id"] == goner

    report = _call(server, "gc", {})
    assert len(report["dropped"]) == 1
    assert report["freed_bytes"] > 0

    with Workspace(root) as ws:
        assert ws.runs.get(keeper).state.value == "promoted"
        assert not ws.artifacts.has(ws.runs.get_artifact(goner, "out").hash)


def test_launch_workflow(root: Path, tmp_path: Path) -> None:
    script = tmp_path / "wf.py"
    script.write_text(
        "from slab import check, task\n"
        "@task\ndef triple(x):\n    return 3 * x\n"
        "print('result:', triple(5))\n"
        "@check\ndef ok():\n    return True\n"
    )
    server = build_server(root)
    result = _call(
        server,
        "launch_workflow",
        {"script_path": str(script), "intent": "launched by agent"},
    )
    assert result["state"] == "verified"
    assert "result: 15" in result["output"]
    with Workspace(root) as ws:
        assert ws.runs.get(result["run_id"]).intent == "launched by agent"


def test_tool_errors_surface_helpfully(root: Path) -> None:
    _seed(root)
    server = build_server(root)
    with pytest.raises(Exception, match="no run matches"):
        _call(server, "show_run", {"run_id": "zzzz"})
