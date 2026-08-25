"""Mason — the resident research agent: a coding-agent harness for open models.

The mason works the slab. This package is a Claude-Code-class agent loop
built directly on SLAB's own operations and tuned for long-running atomistic
research projects, designed to run against *open-weight* models served on
your own hardware (vLLM on a compute node, Ollama on a laptop) through the
OpenAI-compatible chat-completions API — stdlib HTTP only, no SDK.

The design distills the 2025-2026 agent-harness literature; each mechanism
and its provenance is documented in ``docs/tutorials/mason.md``:

* one ReAct-style tool loop as the unit of work, composed one level deep
  by the roster: agent cards (a PI and specialists) whose ``delegate``
  tool runs a specialist's own loop and returns its report — sequential,
  depth-limited in code, never a swarm,
* skills in the open Agent Skills format, categorized per specialist,
  carrying tested analysis scripts an agent loads instead of re-deriving,
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

from mason.anthropic import AnthropicClient, ModelRefusalError
from mason.client import ChatClient, ChatReply, ContextOverflowError, LlmError, ToolCall
from mason.errors import MasonError
from mason.loop import Mason, TurnResult
from mason.roster import AgentSpec, RosterError, discover_roster
from mason.session import MasonSession
from mason.skills import Skill, SkillError, discover_skills
from mason.tools import Tool, Toolbox, build_toolbox
from slab._version import __version__

__all__ = [
    "AgentSpec",
    "AnthropicClient",
    "ChatClient",
    "ChatReply",
    "ContextOverflowError",
    "LlmError",
    "Mason",
    "MasonError",
    "MasonSession",
    "ModelRefusalError",
    "RosterError",
    "Skill",
    "SkillError",
    "Tool",
    "ToolCall",
    "Toolbox",
    "TurnResult",
    "__version__",
    "build_toolbox",
    "discover_roster",
    "discover_skills",
]
