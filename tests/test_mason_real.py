"""Gated integration test against a real open-model endpoint.

Runs only when ``$SLAB_TEST_LLM`` points at an OpenAI-compatible server and
``$SLAB_TEST_LLM_MODEL`` names a served model, e.g.::

    SLAB_TEST_LLM=http://localhost:11434/v1 SLAB_TEST_LLM_MODEL=llama3.1:8b \\
        pytest tests/test_mason_real.py

Small open models are nondeterministic; the assertions check that the loop
closed and tools actually ran, not any particular wording.
"""

import json
import os
from pathlib import Path

import pytest

from slab.config import SlabConfig
from slab.mason.loop import Mason
from slab.mason.session import MasonSession

ENDPOINT = os.environ.get("SLAB_TEST_LLM")
MODEL = os.environ.get("SLAB_TEST_LLM_MODEL")

pytestmark = pytest.mark.skipif(
    not (ENDPOINT and MODEL),
    reason="set SLAB_TEST_LLM and SLAB_TEST_LLM_MODEL to run against a real endpoint",
)


def test_real_model_reads_a_file_and_answers(tmp_path: Path) -> None:
    # Temperature 0 plus explicit sequencing ("may not finish until read_file
    # has returned") is what makes 8B-class models take the tool path reliably;
    # softer phrasings make them guess. One retry absorbs residual sampling noise.
    (tmp_path / "data.txt").write_text("the secret word is perovskite\n")
    config = SlabConfig.model_validate(
        {"agent": {"endpoint": ENDPOINT, "model": MODEL, "max_turns": 6, "temperature": 0.0}}
    )
    goal = (
        "Use the read_file tool on data.txt. You may not call finish until "
        "read_file has returned. Then call finish reporting the secret word verbatim."
    )
    last = ""
    for _attempt in range(2):
        session = MasonSession(
            tmp_path, workspace_root=tmp_path / ".slab", config=config, auto_approve=True
        )
        result = Mason(session).run_turn(goal)
        assert result.stop_reason in ("answer", "finish")
        last = result.text
        if "perovskite" in last.lower():
            break
    # The word can only come from actually reading the file:
    assert "perovskite" in last.lower()
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    assert any(event["type"] == "usage" for event in events)


ANTHROPIC_MODEL = os.environ.get("SLAB_TEST_ANTHROPIC_MODEL", "claude-opus-5")


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set ANTHROPIC_API_KEY to run against the live Anthropic API",
)
def test_real_anthropic_provider_reads_a_file_and_answers(tmp_path: Path) -> None:
    """Same probe, the Claude provider — exercises the real Messages wire shape.

    Run it with::

        ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_mason_real.py

    Override the model with ``$SLAB_TEST_ANTHROPIC_MODEL``.
    """
    (tmp_path / "data.txt").write_text("the secret word is perovskite\n")
    config = SlabConfig.model_validate(
        {
            "agent": {
                "provider": "anthropic",
                "model": ANTHROPIC_MODEL,
                "max_turns": 6,
                "effort": "low",
            }
        }
    )
    session = MasonSession(
        tmp_path, workspace_root=tmp_path / ".slab", config=config, auto_approve=True
    )
    result = Mason(session).run_turn(
        "Use the read_file tool on data.txt, then call finish reporting the secret word."
    )
    assert result.stop_reason in ("answer", "finish")
    assert "perovskite" in result.text.lower()
    events = [json.loads(line) for line in session.transcript_path.read_text().splitlines()]
    usage = [event for event in events if event["type"] == "usage"]
    assert usage and any(event["prompt_tokens"] for event in usage)
