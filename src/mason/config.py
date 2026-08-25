"""Mason's slice of the shared ``slab.toml``: the ``[agent]`` tables.

The file, the loader, the layering, and the origin tracking all live in
:mod:`slab.config`. What lives here are the two models Mason owns and the
view that validates them, so a typo inside ``[agent]`` is refused when Mason
loads and is invisible to a ``slab engines list`` that never reads the table.

``[agent]`` describes the model and the harness around it; ``[agent.serve]``
describes how to start that model as a batch job. Neither is expanded at load
beyond the fields typed as paths: a serve ``command`` and its ``setup`` lines
are shell for the GPU node, so their variables must reach it literally.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from slab.config import load_merged, validate

_OLLAMA = "http://localhost:11434/v1"


class ServeConfig(BaseModel):
    """How to stand the agent's own model server up on this cluster (``[agent.serve]``).

    Mason talks to an OpenAI-compatible endpoint. On a laptop that endpoint is
    Ollama at a fixed localhost URL; on a cluster it is a server *you* start,
    on a GPU node the scheduler picks — so the URL cannot be written in a
    config file. This section declares the launch (which partition, which
    port, which flags); the node and the URL are discovered at run time from
    the record the job writes (:mod:`mason.serve`).

    ``command`` is the escape hatch for a server this schema does not model;
    it may reference ``$port``, which the rendered script binds. Everything
    here is shell for the *compute node*, so no variable is expanded at load.

    Examples:
        >>> ServeConfig(partition="gpu", tool_call_parser="llama4_pythonic").port
        8000
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: str | None = None
    time_limit: str | None = None
    port: int = Field(default=8000, ge=1024, le=65535)
    job_name: str = "mason-serve"
    tool_call_parser: str | None = None
    args: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    command: str | None = None
    ready_timeout_s: float = Field(default=1800.0, gt=0)
    # The serve job brings a self-contained environment (its own venv via
    # setup), and [hpc]-level setup exists to load ENGINE software — module
    # stacks whose libraries fight the server's. So global setup is excluded
    # by default; the partition's own setup (GPU drivers) still applies.
    include_hpc_setup: bool = False


class AgentConfig(BaseModel):
    """The resident research agent's model connection and budgets (``[agent]``).

    Two providers, one harness. ``provider = "openai"`` (the default) talks to
    any OpenAI-compatible ``/v1`` server — vLLM on a compute node, Ollama on a
    laptop — and keeps working where compute nodes have no internet, which is
    the normal HPC case. ``provider = "anthropic"`` talks to the Claude
    Messages API, which needs reachable internet and *billed API access* (a
    Claude subscription is a separate product and does not grant it).

    ``endpoint`` may be left unset on a cluster: :func:`mason.serve.
    discover_endpoint` reads the URL from the running server job's record.

    ``api_key_env`` names an environment variable holding the key; the key
    itself never belongs in a config file (config files get committed and
    shipped). ``compute_profile`` shapes what the agent *chooses* to run, not
    what SLAB permits — see :func:`mason.prompts.compute_profile_block`.

    Examples:
        >>> AgentConfig().resolved_endpoint
        'http://localhost:11434/v1'
        >>> AgentConfig(provider="anthropic").resolved_endpoint
        'https://api.anthropic.com/v1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai", "anthropic"] = "openai"
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    compute_profile: Literal["laptop", "workstation", "cluster"] | None = None
    context_window: int = Field(default=65_536, ge=4_096)
    compact_at: float = Field(default=0.7, gt=0.0, le=1.0)
    max_turns: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    request_timeout_s: float = Field(default=600.0, gt=0)
    max_reply_tokens: int | None = Field(default=None, ge=256)
    max_tool_output_chars: int = Field(default=24_000, ge=1_000)
    shell_timeout_s: float = Field(default=120.0, gt=0)
    tool_protocol: Literal["native", "fenced"] = "native"
    approval: Literal["ask", "auto"] = "ask"
    shell_allowlist: tuple[str, ...] = ()
    serve: ServeConfig = ServeConfig()

    @property
    def resolved_endpoint(self) -> str:
        """The endpoint to call: explicit, else the provider's own default."""
        if self.endpoint:
            return self.endpoint
        return "https://api.anthropic.com/v1" if self.provider == "anthropic" else _OLLAMA

    @property
    def resolved_api_key_env(self) -> str | None:
        """The env var holding the key — Anthropic's is required and defaulted."""
        if self.api_key_env:
            return self.api_key_env
        return "ANTHROPIC_API_KEY" if self.provider == "anthropic" else None


class MasonConfig(BaseModel):
    """The ``[agent]`` view of the merged configuration.

    ``extra="ignore"`` at this level: the same file carries ``[paths]``,
    ``[engines]``, ``[hpc]``, and ``[workspace]``, which other packages
    validate. Inside ``[agent]`` the models still forbid unknown keys, so a
    misspelled agent setting is refused naming the file.

    Examples:
        >>> MasonConfig().agent.resolved_endpoint
        'http://localhost:11434/v1'
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    agent: AgentConfig = AgentConfig()


def load_config(cwd: str | os.PathLike[str] | None = None) -> MasonConfig:
    """Load every layer and validate the tables Mason owns."""
    return validate(load_merged(cwd), MasonConfig)


