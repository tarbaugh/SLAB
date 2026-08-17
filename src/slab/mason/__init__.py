"""Mason — the resident research agent: a coding-agent harness for open models.

The mason works the slab. This subpackage is a Claude-Code-class agent loop
built directly on SLAB's own operations and tuned for long-running atomistic
research projects, designed to run against *open-weight* models served on
your own hardware (vLLM on a compute node, Ollama on a laptop) through the
OpenAI-compatible chat-completions API — stdlib HTTP only, no SDK.

The design distills the 2025-2026 agent-harness literature; each mechanism
and its provenance is documented in ``docs/tutorials/mason.md``:

* a single ReAct-style tool loop (multi-agent orchestration deliberately
  omitted — it pays off for breadth-first search, not interdependent
  research work),
* context as a budgeted resource: token-accounted turns, compaction well
  below the window, notes that outlive the context,
* file-first memory: a lab notebook (``NOTEBOOK.md``) and a living plan
  (``PLAN.md``) in the project directory, under version control — not in
  an opaque store,
* verification-gated physics: calculations run as SLAB workflow scripts
  with ``@check`` assertions, so every number Mason reports traces to a
  run id and its checks,
* failure evidence kept in context, with a diagnose-before-retry
  discipline enforced by the harness, not the prompt alone.
"""

from slab.mason.client import ChatClient, ChatReply, ContextOverflowError, LlmError, ToolCall
from slab.mason.loop import Mason, TurnResult
from slab.mason.session import MasonSession
from slab.mason.tools import Tool, Toolbox, build_toolbox

__all__ = [
    "ChatClient",
    "ChatReply",
    "ContextOverflowError",
    "LlmError",
    "Mason",
    "MasonSession",
    "Tool",
    "ToolCall",
    "Toolbox",
    "TurnResult",
    "build_toolbox",
]
