"""The no-network sandbox for autonomous runs: preflight, bridge, render.

``--auto`` removes the approval gate, so the boundary for an unattended run
must come from the operating system, not from the harness. The shape this
module renders: one batch job that runs ``mason run --auto`` inside an
Apptainer container with an empty network namespace (``--net --network
none``), no home directory, a clean environment, and file access limited to
explicit bind mounts. The model stays reachable through exactly one path — a
unix socket, bridged on the host side to the recorded serve endpoint by a
``socat`` with a fixed destination — so the agent's shell can reach the
model and nothing else.

Everything machine-specific derives from the loaded configuration: the
workspace from ``[workspace]``, the pseudopotential and scratch roots from
``[paths]``, the rootstock install from ``[engines.rootstock]``, the Python
environment from the running interpreter, and the endpoint from the serve
record at job runtime. The rendered artifacts therefore live on the machine
they describe, and nothing site-specific belongs in a repository — the
template here stays generic.

Engine ``setup`` lines get the same treatment. ``module load`` works on the
host and means nothing inside the container, so the render runs each
engine's setup once, on the host, and snapshots what it did: the resolved
binary, the environment delta, and the binary's library closure. The
snapshot becomes bind mounts and explicit ``export`` lines in the rendered
``slab.toml`` — the real config keeps its modules as the single source of
truth, and the frozen copy is reviewable output like everything else. A
module upgrade means re-rendering.

The rendered job fails closed: before starting the agent, it proves inside
the container that the internet is unreachable and that the bridged
endpoint answers. Either proof failing aborts the job. Like the serve
script, the result is pure text — render it, read it, then submit it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mason.config import AgentConfig
from mason.errors import MasonError
from mason.serve import read_record, record_path
from slab.config import HpcConfig, SlabConfig

#: Where the bridge surfaces inside the container. The port exists only in
#: the sandbox's private namespace, so it can never collide with the host.
BRIDGE_PORT = 8000
_SOCKET_IN_CONTAINER = "/run/llm.sock"

#: Reads the serve record and prints ``host:port`` — run on the host at job
#: start, so the bridge follows the server wherever the scheduler put it.
_UPSTREAM_SNIPPET = (
    "import json, sys; from urllib.parse import urlsplit; "
    'print(urlsplit(json.load(open(sys.argv[1]))["endpoint"]).netloc)'
)


class SandboxError(MasonError):
    """The sandbox cannot be rendered, verified, or bridged."""


# -- the bridge (runs inside the container) -----------------------------------


def forward(socket_path: str, port: int = BRIDGE_PORT) -> None:
    """Serve ``127.0.0.1:port`` by relaying every connection to a unix socket.

    This is the container half of the bridge. The namespace has no network,
    so the only route out is the bound socket file, whose other end is a
    host-side ``socat`` pointed at one fixed destination. Runs until killed.
    """

    def pump(source: socket.socket, sink: socket.socket) -> None:
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for side in (source, sink):
                with contextlib.suppress(OSError):
                    side.shutdown(socket.SHUT_RDWR)

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen()
    while True:
        client, _addr = server.accept()
        upstream = socket.socket(socket.AF_UNIX)
        try:
            upstream.connect(socket_path)
        except OSError:
            client.close()
            continue
        threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
        threading.Thread(target=pump, args=(upstream, client), daemon=True).start()


def verify(
    port: int = BRIDGE_PORT,
    *,
    probe_url: str = "http://example.com",
    ready_timeout_s: float = 30.0,
) -> list[str]:
    """Prove the sandbox is dark and the bridge answers, or refuse to start.

    Two checks, both mandatory. *probe_url* must be unreachable — any
    response at all means the namespace has a route out, and an autonomous
    run must not start. The bridged endpoint must list its models within
    *ready_timeout_s* (retried, because the forwarder starts moments before
    this check). Returns the served model names.
    """
    try:
        with urllib.request.urlopen(probe_url, timeout=3.0):
            pass
        raise SandboxError(
            f"the sandbox can reach {probe_url} — the network namespace is not "
            f"isolated. Refusing to start an autonomous run. Check that the "
            f"container was launched with --net --network none."
        )
    except SandboxError:
        raise
    except (urllib.error.URLError, OSError, TimeoutError):
        pass  # dark, as required

    deadline = time.monotonic() + ready_timeout_s
    last = "no attempt made"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5.0
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return [str(m["id"]) for m in payload.get("data", [])]
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            last = str(e)
            time.sleep(1.0)
    raise SandboxError(
        f"the bridged endpoint at 127.0.0.1:{port} did not answer within "
        f"{ready_timeout_s:.0f}s (last error: {last}). Is the serve job "
        f"running, and did the host-side socat start?"
    )


# -- preflight (runs on the login node) ---------------------------------------


def preflight(agent: AgentConfig, workspace_root: str | os.PathLike[str]) -> list[tuple[str, str]]:
    """What the sandbox needs from this machine, as ``(mark, message)`` rows.

    ``+`` is satisfied, ``-`` is a hard requirement that failed, ``?`` could
    not be tested here. The caller decides how loudly to fail.
    """
    rows: list[tuple[str, str]] = []

    def tool(name: str, why: str) -> None:
        if shutil.which(name):
            rows.append(("+", f"{name} found ({why})"))
        else:
            rows.append(("-", f"{name} not found on PATH ({why})"))

    tool("apptainer", "the container runtime")
    tool("socat", "the host side of the endpoint bridge")

    if shutil.which("unshare"):
        probe = subprocess.run(
            ["unshare", "--user", "--net", "true"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            rows.append(("+", "unprivileged user+network namespaces work"))
        else:
            rows.append(
                (
                    "-",
                    "unprivileged network namespaces are disabled here "
                    "(unshare --user --net failed); --network none needs them — "
                    "ask your administrators",
                )
            )
    else:
        rows.append(("?", "cannot test namespaces here (no unshare); try on a compute node"))

    image = agent.sandbox.image
    if not image:
        rows.append(("-", "[agent.sandbox] image is not set; render needs a container image"))
    elif Path(image).expanduser().is_file():
        rows.append(("+", f"container image exists: {image}"))
    else:
        rows.append(("-", f"[agent.sandbox] image does not exist yet: {image}"))

    record = read_record(workspace_root)
    if record is None:
        rows.append(
            (
                "?",
                f"no serve record at {record_path(workspace_root)}; the rendered "
                f"job reads the endpoint from it at start, so the server must be "
                f"up by then ('mason serve start')",
            )
        )
    else:
        rows.append(("+", f"serve record: {record.model} at {record.endpoint}"))
    return rows


# -- the setup snapshot (runs on the host, at render time) --------------------

#: Environment keys a login shell churns on its own; never part of what a
#: setup line meant to communicate. BASH_FUNC_* entries (exported shell
#: functions, the module command itself among them) are dropped too — they
#: cannot ride an export line, and the container has no module system for
#: them to drive anyway.
_ENV_NOISE = frozenset(
    {"PWD", "OLDPWD", "SHLVL", "_", "PS1", "PS2", "PROMPT_COMMAND", "LS_COLORS"}
)

#: Library directories the container's base image provides itself; binding
#: the host's copy over them would shadow the image's own loader setup.
_BASE_IMAGE_LIBS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64")

#: What binary to resolve per engine when the command does not name one.
_DEFAULT_PAYLOAD = {"qe": "pw.x", "lammps": "lmp"}

_LAUNCHERS = frozenset({"env", "srun", "mpirun", "mpiexec", "nice", "time"})


class SetupSnapshot:
    """What one engine's ``setup`` lines did when run on the host.

    ``payload`` is the resolved binary, ``env`` the plain environment
    variables the setup introduced or changed, ``path_prepends`` the
    components it added to colon-separated list variables (``PATH``,
    ``LD_LIBRARY_PATH``, ...) — only the additions, so the container's own
    base value survives underneath. ``lib_dirs`` are the binary's
    shared-library directories outside the container base image. A snapshot
    with ``error`` set means the setup failed or the binary never appeared —
    the render then keeps the hand-configuration warning instead of
    pretending.
    """

    def __init__(
        self,
        engine: str,
        payload: str,
        env: dict[str, str],
        lib_dirs: tuple[str, ...],
        error: str | None = None,
        path_prepends: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.engine = engine
        self.payload = payload
        self.env = env
        self.lib_dirs = lib_dirs
        self.error = error
        self.path_prepends = path_prepends or {}

    def setup_lines(self) -> list[str]:
        """The frozen replacement for the module loads: explicit exports."""
        lines = [
            f"export {key}={shlex.quote(value)}" for key, value in sorted(self.env.items())
        ]
        for key, added in sorted(self.path_prepends.items()):
            joined = ":".join(added)
            for hostile in ("\\", '"', "$", "`"):
                joined = joined.replace(hostile, "\\" + hostile)
            lines.append(f'export {key}="{joined}${{{key}:+:${key}}}"')
        return lines

    def export_count(self) -> int:
        return len(self.env) + len(self.path_prepends)


def payload_name(engine: str, command: str | None) -> str:
    """The binary a setup must make resolvable, read off the command line.

    Skips launchers, options, assignments, and bare numbers, so
    ``mpirun -np 4 pw.x`` and ``env OMP_NUM_THREADS=4 pw.x`` both name
    ``pw.x``. With no command configured, the engine's canonical binary.

    Examples:
        >>> payload_name("qe", "mpirun -np 4 pw.x")
        'pw.x'
        >>> payload_name("lammps", None)
        'lmp'
    """
    for token in shlex.split(command) if command else ():
        if token in _LAUNCHERS or token.startswith("-") or "=" in token or token.isdigit():
            continue
        return token
    return _DEFAULT_PAYLOAD[engine]


def _env_entries(raw: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    for chunk in raw.split(b"\0"):
        if not chunk:
            continue
        key, _sep, value = chunk.decode("utf-8", errors="replace").partition("=")
        if key in _ENV_NOISE or key.startswith("BASH_FUNC_"):
            continue
        entries[key] = value
    return entries


def _lib_dirs(ldd_output: str) -> tuple[str, ...]:
    dirs: set[str] = set()
    for line in ldd_output.splitlines():
        _name, arrow, rest = line.partition("=>")
        target = rest.split()[0] if arrow and rest.split() else ""
        if not target.startswith("/"):
            continue
        parent = str(Path(target).parent)
        if any(
            parent == base or parent.startswith(base + "/") for base in _BASE_IMAGE_LIBS
        ):
            continue
        dirs.add(parent)
    return tuple(sorted(dirs))


def snapshot_setup(
    engine: str, setup: tuple[str, ...], payload: str, *, timeout_s: float = 120.0
) -> SetupSnapshot:
    """Run one engine's setup on the host and record what it did.

    A login shell dumps its environment, runs the setup under ``set -e``,
    resolves *payload*, dumps the environment again, and asks ``ldd`` for
    the binary's closure (best-effort: a missing ldd or a static binary
    yields no library dirs, and the payload's own prefix still gets bound).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        files = {name: Path(scratch) / name for name in ("before", "after", "which", "ldd")}
        script = "\n".join(
            [
                f"env -0 > {shlex.quote(str(files['before']))}",
                "set -e",
                *setup,
                f"command -v {shlex.quote(payload)} > {shlex.quote(str(files['which']))}",
                f"env -0 > {shlex.quote(str(files['after']))}",
                f'ldd "$(command -v {shlex.quote(payload)})" '
                f"> {shlex.quote(str(files['ldd']))} 2>/dev/null || true",
            ]
        )
        try:
            result = subprocess.run(
                ["bash", "-lc", script], capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return SetupSnapshot(
                engine, "", {}, (), error=f"setup did not finish within {timeout_s:.0f}s"
            )
        resolved = files["which"].read_text().strip() if files["which"].exists() else ""
        if result.returncode != 0 or not resolved:
            detail = result.stderr.strip().splitlines()
            return SetupSnapshot(
                engine,
                "",
                {},
                (),
                error=(detail[-1] if detail else f"{payload!r} not found after setup"),
            )
        before = _env_entries(files["before"].read_bytes())
        after = _env_entries(files["after"].read_bytes())
        plain: dict[str, str] = {}
        prepends: dict[str, tuple[str, ...]] = {}
        for key, value in after.items():
            if before.get(key) == value:
                continue
            if key.endswith("PATH"):
                # A list variable: keep only what the setup added, so the
                # container's own base value survives underneath instead of
                # being shadowed by the host's entire list.
                have = set(filter(None, before.get(key, "").split(":")))
                added = tuple(c for c in value.split(":") if c and c not in have)
                if added:
                    prepends[key] = added
            else:
                plain[key] = value
        libs = _lib_dirs(files["ldd"].read_text()) if files["ldd"].exists() else ()
        return SetupSnapshot(engine, resolved, plain, libs, path_prepends=prepends)


def snapshot_engines(slab_cfg: SlabConfig) -> dict[str, SetupSnapshot]:
    """A snapshot per engine whose config declares setup lines."""
    snapshots: dict[str, SetupSnapshot] = {}
    for engine in ("qe", "lammps"):
        table = getattr(slab_cfg.engines, engine)
        if table.setup:
            snapshots[engine] = snapshot_setup(
                engine, table.setup, payload_name(engine, table.command)
            )
    return snapshots


def _snapshot_binds(snapshot: SetupSnapshot) -> list[str]:
    """Read-only binds for a snapshotted engine: its install, its libraries."""
    if snapshot.error:
        return []
    bin_dir = Path(snapshot.payload).parent
    prefix = bin_dir.parent if bin_dir.name == "bin" else bin_dir
    return [f"{d}:{d}:ro" for d in dict.fromkeys([str(prefix), *snapshot.lib_dirs])]


def _collapse_binds(binds: list[str]) -> list[str]:
    """Drop exact duplicates and read-only binds nested inside another bind."""

    def src(spec: str) -> str:
        return spec.split(":", 1)[0]

    kept: list[str] = []
    for spec in binds:
        redundant = False
        for other in binds:
            if other == spec:
                continue
            inside = src(spec) == src(other) or src(spec).startswith(src(other) + "/")
            wider = spec.endswith(":ro") or other.endswith(":rw")
            if inside and wider and (other in kept or binds.index(other) < binds.index(spec)):
                redundant = True
                break
        if not redundant and spec not in kept:
            kept.append(spec)
    return kept


# -- rendering ----------------------------------------------------------------


def _python() -> str:
    return sys.executable


def _mason_bin() -> Path:
    """The mason console script next to the running interpreter."""
    candidate = Path(sys.executable).parent / "mason"
    if not candidate.is_file():
        raise SandboxError(
            f"no mason console script next to {sys.executable}; render from the "
            f"environment SLAB is installed in, since the job binds and reuses it"
        )
    return candidate


def default_binds(
    project: Path,
    workspace_root: Path,
    slab_cfg: SlabConfig,
    snapshots: dict[str, SetupSnapshot] | None = None,
) -> tuple[list[str], list[str]]:
    """The bind mounts the configuration implies, plus warnings for the gaps.

    Read-write: the project, the workspace, and the scratch root. Read-only:
    the pseudopotential roots, the rootstock install, the Python environment
    (with the repository checkout when the install is editable), and — for
    each engine in *snapshots* — the install and library directories its
    setup resolved to on the host. Everything else does not exist inside
    the container.
    """
    warnings: list[str] = []
    binds = [f"{project}:{project}:rw", f"{workspace_root}:{workspace_root}:rw"]
    if slab_cfg.paths.scratch:
        binds.append(f"{slab_cfg.paths.scratch}:{slab_cfg.paths.scratch}:rw")
    if slab_cfg.paths.pseudos:
        binds.append(f"{slab_cfg.paths.pseudos}:{slab_cfg.paths.pseudos}:ro")
    if slab_cfg.engines.qe.pseudo_dir:
        binds.append(f"{slab_cfg.engines.qe.pseudo_dir}:{slab_cfg.engines.qe.pseudo_dir}:ro")
    if not slab_cfg.paths.pseudos and not slab_cfg.engines.qe.pseudo_dir:
        warnings.append(
            "neither [paths] pseudos nor [engines.qe] pseudo_dir is set: "
            "no pseudopotentials will be visible"
        )
    if slab_cfg.paths.engines:
        binds.append(f"{slab_cfg.paths.engines}:{slab_cfg.paths.engines}:ro")
    for snapshot in (snapshots or {}).values():
        binds.extend(_snapshot_binds(snapshot))
    if slab_cfg.engines.qe.bin:
        # The whole install, not just bin/: pw.x usually links ../lib.
        prefix = Path(slab_cfg.engines.qe.bin).parent
        binds.append(f"{prefix}:{prefix}:ro")
    rootstock = slab_cfg.engines.rootstock
    if rootstock.root:
        binds.append(f"{rootstock.root}:{rootstock.root}:ro")
    elif rootstock.cluster:
        warnings.append(
            "[engines.rootstock] uses the cluster form, so its install path is "
            "not in this config: add the site's rootstock root to "
            "[agent.sandbox] binds or served checkpoints will not resolve"
        )

    prefix = Path(sys.prefix).resolve()
    binds.append(f"{prefix}:{prefix}:ro")
    base = Path(sys.base_prefix).resolve()
    if base != prefix:
        binds.append(f"{base}:{base}:ro")
    package_root = Path(__file__).resolve().parent
    if not package_root.is_relative_to(prefix):
        # An editable install: the package lives in a checkout (src/mason),
        # so the checkout itself must exist inside the container.
        repo = package_root.parent.parent
        binds.append(f"{repo}:{repo}:ro")
    return binds, warnings


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _emit_table(name: str, mapping: dict[str, Any], out: list[str]) -> None:
    scalars = {k: v for k, v in mapping.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in mapping.items() if isinstance(v, dict)}
    if scalars or not tables:
        out.append(f"[{name}]")
        for key, value in scalars.items():
            out.append(f"{key} = {_toml_value(value)}")
        out.append("")
    for key, sub in tables.items():
        if sub:
            _emit_table(f"{name}.{key}", sub, out)


#: [agent] keys that must not follow the config into the sandbox: the serve
#: and roster machinery, the sandbox's own table, connection details the
#: render replaces (--endpoint points at the bridge), and the approval mode
#: (--auto on the command line is the explicit, visible choice).
_AGENT_KEYS_DROPPED = frozenset(
    {"serve", "roster", "sandbox", "endpoint", "api_key_env", "approval"}
)


def sandbox_toml(
    slab_cfg: SlabConfig,
    agent: AgentConfig,
    workspace_root: Path,
    snapshots: dict[str, SetupSnapshot] | None = None,
) -> tuple[str, list[str]]:
    """The configuration the sandboxed session loads, plus render warnings.

    Deliberately absent: ``[hpc]`` — with no partitions, the scheduler tools
    do not exist in the tool vocabulary, which is what an empty network
    namespace requires (the SLURM controller is unreachable). Calculations
    run in-process inside the job's own allocation.

    An engine present in *snapshots* has its ``setup`` lines replaced by the
    snapshot's explicit exports — the frozen equivalent of what the module
    loads did on the host, which the container cannot run itself.
    """
    warnings: list[str] = []
    engines = slab_cfg.engines.model_dump(exclude_defaults=True)
    qe_command = str(engines.get("qe", {}).get("command", ""))
    if "srun" in qe_command.split():
        warnings.append(
            "[engines.qe] command uses srun, which cannot work inside the "
            "sandbox (no route to the controller): edit the rendered slab.toml "
            "to an mpirun-style command sized to the job's allocation"
        )
    for name, table in engines.items():
        if not table.get("setup"):
            continue
        snapshot = (snapshots or {}).get(name)
        if snapshot is not None and snapshot.error is None:
            table["setup"] = snapshot.setup_lines()
            warnings.append(
                f"[engines.{name}] setup snapshotted from the host: "
                f"{snapshot.payload}, {snapshot.export_count()} export(s), "
                f"{len(_snapshot_binds(snapshot))} bind(s)"
            )
            continue
        because = f" (snapshot failed: {snapshot.error})" if snapshot is not None else ""
        warnings.append(
            f"[engines.{name}] has setup lines the render could not snapshot"
            f"{because}: module loads resolve against the host, not the "
            f"container — bind the software read-only via [agent.sandbox] "
            f"binds and set PATH/LD_LIBRARY_PATH in setup instead of "
            f"'module load'"
        )

    agent_table = {
        k: v
        for k, v in agent.model_dump(exclude_defaults=True).items()
        if k not in _AGENT_KEYS_DROPPED
    }
    out = [
        "# Rendered by 'mason sandbox render'. The scheduler table is deliberately",
        "# absent: the sandbox has no route to the controller, so the scheduler",
        "# tools must not exist. Review before submitting.",
        "",
    ]
    _emit_table("workspace", {"root": str(workspace_root)}, out)
    paths = slab_cfg.paths.model_dump(exclude_defaults=True)
    if paths:
        _emit_table("paths", paths, out)
    if engines:
        _emit_table("engines", engines, out)
    _emit_table("agent", agent_table, out)
    return "\n".join(out).rstrip() + "\n", warnings


def render_sandbox_script(
    agent: AgentConfig,
    hpc: HpcConfig,
    slab_cfg: SlabConfig,
    workspace_root: Path,
    project: Path,
    goal: str,
    *,
    toml_path: Path,
    partition: str | None = None,
    time_limit: str | None = None,
    snapshots: dict[str, SetupSnapshot] | None = None,
) -> tuple[str, list[str]]:
    """The batch script for one autonomous, network-dark session.

    Host side: resolve the serve record's endpoint, bridge it onto a unix
    socket with a fixed-destination ``socat``. Container side: forward the
    socket to loopback, prove darkness and reachability with
    ``mason sandbox verify`` (either failing aborts the job), then run the
    goal with ``mason run --auto``.
    """
    from slab.hpc import render_sbatch

    if not agent.sandbox.image:
        raise SandboxError(
            "[agent.sandbox] image is not set: name the Apptainer image the "
            "sandbox job should run in (build one with e.g. "
            "'apptainer build slab-sandbox.sif docker://rockylinux:9')"
        )
    binds, warnings = default_binds(project, workspace_root, slab_cfg, snapshots)
    binds.extend(agent.sandbox.binds)
    binds = _collapse_binds(binds)
    mason = _mason_bin()

    record = record_path(workspace_root).resolve()
    prologue = [
        'command -v socat >/dev/null || { echo "socat not found on the host" >&2; exit 1; }',
        f"RECORD={shlex.quote(str(record))}",
        '[ -f "$RECORD" ] || { echo "no serve record at $RECORD;'
        " start the model server first ('mason serve start')\" >&2; exit 1; }",
        f'UPSTREAM=$({shlex.quote(_python())} -c {shlex.quote(_UPSTREAM_SNIPPET)} "$RECORD")',
        'BRIDGE="$(mktemp -d)/llm.sock"',
        'socat "UNIX-LISTEN:$BRIDGE,fork" "TCP:$UPSTREAM" &',
        "BRIDGE_PID=$!",
        "trap 'kill \"$BRIDGE_PID\" 2>/dev/null || true' EXIT",
        'for _ in $(seq 50); do [ -S "$BRIDGE" ] && break; sleep 0.1; done',
    ]

    inner = "\n".join(
        [
            f"{mason} sandbox forward {_SOCKET_IN_CONTAINER} --port {BRIDGE_PORT} &",
            "FORWARD_PID=$!",
            "trap 'kill \"$FORWARD_PID\" 2>/dev/null || true' EXIT",
            f"{mason} sandbox verify --port {BRIDGE_PORT}",
            f"cd {shlex.quote(str(project))}",
            f"{mason} run --auto --endpoint http://127.0.0.1:{BRIDGE_PORT}/v1 "
            f"{shlex.quote(goal)}",
        ]
    )
    command = " \\\n  ".join(
        [
            "apptainer exec",
            "--containall --no-home --cleanenv --net --network none",
            *(f"--bind {shlex.quote(spec)}" for spec in binds),
            f'--bind "$BRIDGE":{_SOCKET_IN_CONTAINER}',
            f"--env SLAB_WORKSPACE={shlex.quote(str(workspace_root))}",
            f"--env SLAB_CONFIG={shlex.quote(str(toml_path))}",
            # --cleanenv would strip it, and the bin-form qe command sizes
            # its mpirun from it — 1 if the scheduler did not set it.
            '--env SLURM_NTASKS="${SLURM_NTASKS:-1}"',
            shlex.quote(str(Path(agent.sandbox.image).expanduser())),
            f"bash -c {shlex.quote(inner)}",
        ]
    )
    script = render_sbatch(
        command,
        job_name="mason-sandbox",
        partition=partition,
        config=hpc,
        time_limit=time_limit,
        prologue=prologue,
        use_launcher=False,
        # Host-level module loads serve the engines OUTSIDE a container;
        # inside one they resolve against the wrong filesystem anyway.
        include_global_setup=False,
    )
    return script, warnings
