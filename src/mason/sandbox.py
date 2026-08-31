"""The no-network sandbox for autonomous runs: preflight, bridge, render.

``--auto`` removes the approval gate, so the boundary for an unattended run
must come from the operating system, not from the harness. The shape this
module renders: one batch job that runs ``mason run --auto`` inside an
Apptainer container with an empty network namespace (``--net --network
none``), no home directory, a clean environment, and file access limited to
explicit bind mounts. The model stays reachable through exactly one path — a
unix socket, bridged on the host side to the recorded serve endpoint by
:func:`bridge` with a fixed destination — so the agent's shell can reach
the model and nothing else. Both halves of the bridge are stdlib Python
run by the ``mason`` script itself, so the host needs no relay tool.

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
import functools
import http.client
import http.server
import json
import os
import re
import shlex
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from foundation.memory import memory_dir
from mason.config import AgentConfig
from mason.errors import MasonError
from mason.serve import read_record, record_path
from slab.config import HpcConfig, SlabConfig

#: Where the bridge surfaces inside the container. The port exists only in
#: the sandbox's private namespace, so it can never collide with the host.
BRIDGE_PORT = 8000
_SOCKET_IN_CONTAINER = "/run/llm.sock"
_CONNECT_TIMEOUT_S = 30.0

#: Reads the serve record and prints ``host:port`` — run on the host at job
#: start, so the bridge follows the server wherever the scheduler put it.
_UPSTREAM_SNIPPET = (
    "import json, sys; from urllib.parse import urlsplit; "
    'print(urlsplit(json.load(open(sys.argv[1]))["endpoint"]).netloc)'
)


class SandboxError(MasonError):
    """The sandbox cannot be rendered, verified, or bridged."""


# -- the bridge (runs inside the container) -----------------------------------


def _pump(source: socket.socket, sink: socket.socket) -> None:
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


def _relay(server: socket.socket, connect: Callable[[], socket.socket]) -> None:
    """Accept forever; pair every client with a fresh upstream connection.

    Concurrency is capped at :data:`_MAX_INFLIGHT`. The client is an
    untrusted sandbox, and each accepted connection opens a fresh upstream
    connection and two pump threads; without a bound a flood of connections
    would grow the host's thread and socket count without end. Past the cap
    a new client is closed at once, and a slot frees when both of a pair's
    pumps finish.
    """
    server.listen()
    slots = threading.BoundedSemaphore(_MAX_INFLIGHT)

    def serve(client: socket.socket, upstream: socket.socket) -> None:
        try:
            a = threading.Thread(target=_pump, args=(client, upstream), daemon=True)
            b = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
            a.start()
            b.start()
            a.join()
            b.join()
        finally:
            slots.release()

    while True:
        client, _addr = server.accept()
        if not slots.acquire(blocking=False):
            client.close()
            continue
        try:
            upstream = connect()
        except OSError:
            client.close()
            slots.release()
            continue
        threading.Thread(target=serve, args=(client, upstream), daemon=True).start()


def forward(socket_path: str, port: int = BRIDGE_PORT) -> None:
    """Serve ``127.0.0.1:port`` by relaying every connection to a unix socket.

    This is the container half of the bridge. The namespace has no network,
    so the only route out is the bound socket file, whose other end is the
    host-side :func:`bridge` pointed at one fixed destination. Runs until
    killed.
    """

    def connect() -> socket.socket:
        upstream = socket.socket(socket.AF_UNIX)
        upstream.connect(socket_path)
        return upstream

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    _relay(server, connect)


def bridge(
    socket_path: str,
    upstream: str,
    *,
    key_env: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Serve a unix socket, relaying every connection to one fixed upstream.

    The host half of the bridge, run outside the container by the rendered
    job. *upstream* is fixed at start — the agent inside can neither see nor
    change the destination. Pure stdlib on purpose: the host already has
    this environment, so the sandbox needs no relay tool of its own. Runs
    until killed.

    Two forms, and the argument's shape picks between them:

    - ``host:port`` — a byte relay, for a model this cluster serves.
    - a URL (``https://gateway.example/v1``) — an HTTP forwarder
      (:func:`forward_http`), which speaks TLS and authenticates, for a
      model behind a gateway.

    The shape decides rather than a flag, so every job script rendered
    before the forwarder existed keeps working unchanged. Those files live
    on disk and get resubmitted for months.

    Raises:
        SandboxError: *upstream* is neither form, or a key or injected
            headers are named for a byte relay, which cannot send them.
    """
    if "://" in upstream:
        forward_http(socket_path, upstream, key_env=key_env, headers=headers)
        return
    if key_env is not None:
        raise SandboxError(
            f"a key can only be sent to a URL upstream; {upstream!r} is a byte "
            f"relay to a host and port, which forwards what the client sends"
        )
    if headers:
        raise SandboxError(
            f"headers can only be injected toward a URL upstream; {upstream!r} "
            f"is a byte relay to a host and port, which forwards what the "
            f"client sends"
        )
    host, _sep, port_text = upstream.rpartition(":")
    if not host or not port_text.isdigit():
        raise SandboxError(f"bridge upstream must be host:port, got {upstream!r}")

    def connect() -> socket.socket:
        connection = socket.create_connection(
            (host, int(port_text)), timeout=_CONNECT_TIMEOUT_S
        )
        # The timeout above bounds the CONNECT only. Left in place it would
        # also bound every read — and an inference server legitimately says
        # nothing for minutes while it generates, so a timed-out idle read
        # here tears the relay down mid-request and the client sees the
        # server "close without response" while the server is fine. Blocking
        # mode from here on; the client's own request timeout is the bound.
        connection.settimeout(None)
        return connection

    with contextlib.suppress(OSError):
        os.unlink(socket_path)
    server = socket.socket(socket.AF_UNIX)
    server.bind(socket_path)
    _relay(server, connect)


#: Headers that describe one hop and never survive being forwarded
#: (RFC 9110 7.6.1). ``host`` and ``content-length`` are reframed per hop
#: too, and ``authorization`` is replaced rather than relayed.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_DROPPED_RESPONSE_HEADERS = _HOP_BY_HOP | {"content-length"}

#: The ONLY request headers that cross to the gateway. Everything the
#: sandbox sends is dropped unless it is on this list: ``Authorization`` is
#: replaced with the host's key, ``Host`` and ``Content-Length`` are
#: reframed per hop, and every other client header is discarded. A deny
#: list would be wrong here. The upstream is a gateway, and a gateway
#: routes by request header: forwarding an attacker-chosen ``x-portkey-*``,
#: ``x-*``, or ``forwarded`` header would let a sandbox that is supposed to
#: reach one fixed model repoint the gateway at a provider or host of its
#: choosing, on the host's credential. Only the two headers a plain
#: OpenAI-compatible request needs are allowed through; if a deployment's
#: gateway needs a fixed routing header, the bridge must inject it host-side
#: with a fixed value, never trust the sandbox to send it.
_FORWARDED_REQUEST_HEADERS = frozenset({"content-type", "accept"})

#: A request body larger than this is refused rather than read. The sandbox
#: is untrusted; without a cap a single request can name an unbounded
#: ``Content-Length`` and pin a host thread reading it. OpenAI-shaped
#: requests are comfortably under this.
_MAX_REQUEST_BODY = 64 * 1024 * 1024

#: A per-operation timeout on the connection the sandbox speaks to. It
#: bounds reading the request only — the long, legitimate wait is on the
#: upstream socket while the model generates, and that connection is left
#: untimed (see :func:`bridge`). A sandbox that opens a connection and
#: dribbles a request would otherwise hold a host thread forever.
_REQUEST_READ_TIMEOUT_S = 60.0

#: The most connections the host-side bridge serves at once. Past it, new
#: connections are refused instead of spawning an unbounded number of host
#: threads. Generous: a real session's concurrency is far below it.
_MAX_INFLIGHT = 64

#: The version prefix the container always speaks, because the render
#: writes its endpoint. Stripped here and replaced by the upstream's own
#: base path, or a gateway based at ``https://gw/v1`` would be asked for
#: ``/v1/v1/chat/completions``.
_CONTAINER_PREFIX = "/v1"


def upstream_path(base_path: str, request_path: str) -> str:
    """Map a request path from the container onto the upstream's base path.

    Examples:
        >>> upstream_path("/v1", "/v1/chat/completions")
        '/v1/chat/completions'
        >>> upstream_path("/llm/openai/v1", "/v1/models")
        '/llm/openai/v1/models'
        >>> upstream_path("/", "/v1/models")
        '/models'
    """
    suffix = (
        request_path[len(_CONTAINER_PREFIX) :]
        if request_path.startswith(_CONTAINER_PREFIX)
        else request_path
    )
    return (base_path.rstrip("/") + suffix) or "/"


def _safe_upstream_path(base_path: str, request_path: str) -> str | None:
    """Map the container's request target onto the gateway, or None if unsafe.

    The sandbox controls the request line, so the target is validated before
    it is used:

    - Origin-form only. It must be an absolute path (start with ``/``).
      Absolute-form (``http://elsewhere/x``) and authority-form name a
      destination, and a header- or URL-routed gateway would honour it,
      which is exactly the pin this bridge is supposed to hold.
    - No control characters, so nothing can be smuggled into the request
      line sent upstream.
    - No ``..`` segment, so the sandbox cannot climb out of the gateway's
      own base path to another route on the same host.

    A rejected target returns None; the caller answers 400 rather than
    forwarding it.
    """
    if not request_path.startswith("/"):
        return None
    if any(ord(ch) < 0x20 or ch == "\x7f" for ch in request_path):
        return None
    path_only = request_path.split("?", 1)[0]
    if ".." in path_only.split("/"):
        return None
    return upstream_path(base_path, request_path)


#: Header names the bridge itself owns. An injected header must not collide
#: with them: ``Authorization`` comes from ``--key-env``, ``Host`` and
#: ``Content-Length`` are reframed per hop, and the hop-by-hop set never
#: crosses a hop at all.
_RESERVED_INJECTED = _HOP_BY_HOP | {"authorization", "host", "content-length"}

_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


def parse_bridge_headers(specs: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Parse operator-chosen ``Name: value`` specs into injectable headers.

    These are the fixed routing headers a deployment's gateway requires
    (``x-portkey-provider: openai``, an ``x-portkey-config`` id, ...). They
    are chosen host-side by the operator, which is exactly why they are safe
    to send where sandbox-chosen headers are not; the forwarder injects them
    into every request, overriding a sandbox-sent header of the same name.

    Raises:
        SandboxError: a spec has no ``:``, an invalid or reserved name, a
            control character in the value, or repeats a name.

    Examples:
        >>> parse_bridge_headers(["x-portkey-provider: openai"])
        {'x-portkey-provider': 'openai'}
    """
    parsed: dict[str, str] = {}
    for spec in specs:
        name, separator, value = spec.partition(":")
        name, value = name.strip(), value.strip()
        if not separator or not name or not value:
            raise SandboxError(
                f"header {spec!r} must look like 'Name: value' (both sides non-empty)"
            )
        if not _HEADER_NAME.match(name):
            raise SandboxError(f"header name {name!r} is not a valid HTTP header name")
        if name.lower() in _RESERVED_INJECTED:
            raise SandboxError(
                f"header {name!r} cannot be injected: the bridge sets it itself "
                f"(Authorization comes from --key-env; Host and Content-Length "
                f"are reframed per hop; hop-by-hop headers never cross)"
            )
        if any(ord(ch) < 0x20 or ch == "\x7f" for ch in value):
            raise SandboxError(f"header {name!r} has a control character in its value")
        if name.lower() in {existing.lower() for existing in parsed}:
            raise SandboxError(f"header {name!r} is given twice")
        parsed[name] = value
    return parsed


@functools.cache
def _bridge_user_agent() -> str:
    """What the bridge calls itself upstream.

    The allow list drops the client's ``User-Agent``, and some gateway
    front ends refuse a request that has none at all — so the bridge sends
    a fixed, host-chosen one.
    """
    try:
        from importlib.metadata import version

        return f"slab-mason-bridge/{version('slab-stack')}"
    except Exception:
        return "slab-mason-bridge"


def _unreachable_hint(error: OSError) -> str:
    """One actionable clause for the 502 body, keyed on how the trip failed.

    The exception text says what happened; this says what to check. The
    reader is the agent (or the operator reading its transcript), on a
    cluster, mid-job — name the likely cause and the concrete probe.
    """
    if isinstance(error, ssl.SSLCertVerificationError):
        return (
            " (the gateway's TLS certificate did not verify on this host; if the "
            "gateway is signed by a site CA, point $SSL_CERT_FILE at the site's "
            "CA bundle before starting the bridge)"
        )
    if isinstance(error, http.client.RemoteDisconnected):
        return (
            " (the gateway, or a front end before it, closed the connection "
            "without answering; front ends often drop requests missing a header "
            "they require — start the bridge with --header to inject the fixed "
            "header your gateway documents)"
        )
    if isinstance(error, socket.gaierror):
        return " (the gateway's hostname did not resolve on this host)"
    if isinstance(error, ConnectionRefusedError):
        return " (the host resolved and answered, but refused the port)"
    if isinstance(error, TimeoutError):
        return (
            f" (no connection within {_CONNECT_TIMEOUT_S:.0f}s; the bridge needs "
            f"direct egress from this host and does not use $https_proxy — probe "
            f"with: curl --noproxy '*' -sS -m 10 <gateway URL>/models)"
        )
    return ""


def _resolve_key(key_env: str | None) -> str | None:
    """The key the forwarder will send, refusing a named-but-unset variable.

    Starting without the key the job was told to use would make every
    request fail as "the gateway rejected us", when the truth is that we
    sent nothing.
    """
    if key_env is None:
        return None
    key = os.environ.get(key_env)
    if not key:
        raise SandboxError(
            f"${key_env} is not set in the job's environment, so the bridge has "
            f"no key for the gateway (sbatch exports the submitting environment "
            f"by default; a cluster set to --export=NONE needs it exported in "
            f"the job itself)"
        )
    return key


def forward_http(
    socket_path: str,
    upstream: str,
    *,
    key_env: str | None = None,
    headers: dict[str, str] | None = None,
) -> None:
    """Serve a unix socket as an HTTP forwarder to one fixed gateway URL.

    The authenticating half of the bridge. It terminates the plain HTTP the
    container speaks, re-issues each request to *upstream* over TLS when the
    URL says so, and sends back the answer. The key named by *key_env* is
    read from this process's environment, which lives on the host: the
    container is launched ``--cleanenv``, and the rendered ``slab.toml``
    carries no ``api_key_env``, so the agent can neither read the key nor
    change where its requests go.

    *headers* are fixed, operator-chosen headers injected into every
    upstream request (the routing header a gateway deployment requires);
    they override a sandbox-sent header of the same name and never come
    from the sandbox. See :func:`parse_bridge_headers`.

    Nothing is logged per request. A SLURM ``.out`` file is readable by
    anyone who can read the job's directory, and headers do not belong in
    one.

    Raises:
        SandboxError: The URL is unusable, or the named key variable is unset.
    """
    parts = urllib.parse.urlsplit(upstream)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SandboxError(
            f"bridge upstream must be host:port or an http(s) URL, got {upstream!r}"
        )
    key = _resolve_key(key_env)
    handler = _forwarder(parts, key, headers or {})
    with contextlib.suppress(OSError):
        os.unlink(socket_path)
    server = _UnixHTTPServer(socket_path, handler)
    server.serve_forever()


class _UnixHTTPServer(socketserver.ThreadingUnixStreamServer):
    """An HTTP server on a unix socket: one thread per connection, capped.

    The default threading server spawns a thread per connection without
    limit. The peer is an untrusted sandbox, so the count is bounded: past
    :data:`_MAX_INFLIGHT` concurrent connections, new ones are closed at
    once rather than growing the host's thread count without end.
    """

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(_MAX_INFLIGHT)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        threading.Thread(
            target=self._serve, args=(request, client_address), daemon=True
        ).start()

    def _serve(self, request: Any, client_address: Any) -> None:
        try:
            self.finish_request(request, client_address)
        except Exception:  # mirror ThreadingMixIn's own broad guard
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            self._slots.release()


def _forwarder(
    parts: urllib.parse.SplitResult, key: str | None, injected: dict[str, str]
) -> type[http.server.BaseHTTPRequestHandler]:
    """Build the request handler bound to one upstream, one key, fixed headers."""

    class Handler(http.server.BaseHTTPRequestHandler):
        # The container's client opens a connection per request through the
        # relay, so HTTP/1.0's close-when-done framing is the whole protocol
        # needed here, and it keeps every response's length explicit.
        protocol_version = "HTTP/1.0"
        # Bounds reading a request from the untrusted sandbox; the upstream
        # connection is timed separately (untimed) so a slow generation is
        # never cut short.
        timeout = _REQUEST_READ_TIMEOUT_S

        def do_GET(self) -> None:  # http.server's naming contract
            self._forward("GET")

        def do_POST(self) -> None:  # http.server's naming contract
            self._forward("POST")

        def _forward(self, method: str) -> None:
            # The sandbox chooses the request target. Pin it to an
            # origin-form path under the gateway's base before anything
            # else: absolute-form ('http://elsewhere/') or a '..' climb
            # would otherwise become a request the gateway routes away from
            # the one model this bridge exists to reach.
            path = _safe_upstream_path(parts.path, self.path)
            if path is None:
                self._answer_error(
                    "the bridge refused a request target that was not a plain "
                    "path under the gateway",
                    status=400,
                    reason="Bad Request",
                )
                return
            length = self._body_length()
            if length is None:
                self._answer_error(
                    "the bridge refused a request with a bad or oversized "
                    "Content-Length",
                    status=413,
                    reason="Payload Too Large",
                )
                return
            body = self.rfile.read(length) if length else None
            # An allow list, not a deny list: only these client headers cross
            # to the gateway. See _FORWARDED_REQUEST_HEADERS.
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() in _FORWARDED_REQUEST_HEADERS
            }
            # Operator-injected headers are fixed at bridge start and win
            # over anything the sandbox sent under the same name.
            for name, value in injected.items():
                for existing in [h for h in headers if h.lower() == name.lower()]:
                    del headers[existing]
                headers[name] = value
            # The allow list dropped the client's User-Agent; a request with
            # none at all trips some gateway front ends, so the bridge sends
            # its own fixed one (an injected header may replace it).
            if not any(name.lower() == "user-agent" for name in headers):
                headers["User-Agent"] = _bridge_user_agent()
            # Replaced, never appended: the container sends its client's
            # placeholder, and that must not reach the gateway.
            if key is not None:
                headers["Authorization"] = f"Bearer {key}"
            if body is not None:
                headers["Content-Length"] = str(len(body))
            try:
                status, reason, answer, payload = self._issue(method, path, headers, body)
            except OSError as e:
                # A transport failure has no status of its own. Say so as one
                # the agent's client can read, rather than dropping the
                # connection and leaving it to guess — and say what to check,
                # keyed on how the trip failed.
                self._answer_error(
                    f"the bridge could not reach the gateway: "
                    f"{e}{_unreachable_hint(e)}"
                )
                return
            self.send_response(status, reason)
            for name, value in answer:
                if name.lower() not in _DROPPED_RESPONSE_HEADERS:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _body_length(self) -> int | None:
            """The request body length to read, or None if it is unusable.

            A missing length is no body. A non-numeric, negative, or
            oversized length is refused: ``read(-1)`` would drain the socket
            to EOF and pin a thread, and an unbounded length invites the same
            by another name.
            """
            raw = self.headers.get("Content-Length")
            if raw is None:
                return 0
            try:
                length = int(raw)
            except ValueError:
                return None
            if length < 0 or length > _MAX_REQUEST_BODY:
                return None
            return length

        def _issue(
            self, method: str, path: str, headers: dict[str, str], body: bytes | None
        ) -> tuple[int, str, list[tuple[str, str]], bytes]:
            """One upstream round trip, returning its status, headers, body.

            http.client decodes a chunked response transparently, so the
            body here is the real one and the caller reframes its length.
            Relaying the upstream's own framing headers would describe a
            body that no longer exists.
            """
            connection: http.client.HTTPConnection
            if parts.scheme == "https":
                connection = http.client.HTTPSConnection(
                    parts.hostname or "",
                    parts.port,
                    timeout=_CONNECT_TIMEOUT_S,
                    context=ssl.create_default_context(),
                )
            else:
                connection = http.client.HTTPConnection(
                    parts.hostname or "", parts.port, timeout=_CONNECT_TIMEOUT_S
                )
            try:
                connection.connect()
                # The timeout bounds the connect only, the same lesson the
                # byte relay records: a model legitimately says nothing for
                # minutes while it generates, and a timed-out idle read
                # would tear down a request the gateway is still serving.
                if connection.sock is not None:
                    connection.sock.settimeout(None)
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                return (
                    response.status,
                    response.reason or "",
                    response.getheaders(),
                    response.read(),
                )
            finally:
                connection.close()

        def _answer_error(
            self, message: str, *, status: int = 502, reason: str = "Bad Gateway"
        ) -> None:
            payload = json.dumps({"error": {"message": message, "type": "bridge_error"}})
            raw = payload.encode("utf-8")
            self.send_response(status, reason)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def address_string(self) -> str:
            # A unix socket peer has no address, and the stdlib's default
            # would index an empty tuple.
            return "-"

        def log_message(self, *args: Any) -> None:
            """Silent by design; see :func:`forward_http`."""

    return Handler


#: The darkness probes run when the caller names none. Two on purpose, and
#: one is a raw IP: a namespace with a route but no resolver would let a
#: DNS-name probe "fail" and read as dark when it is not, so a probe that
#: needs no DNS closes that gap. Neither proves isolation on its own — only
#: ``--net --network none`` does that — so this stays a fail-closed sanity
#: check, not the boundary.
_DARKNESS_PROBES = ("http://example.com/", "http://1.1.1.1/")


def verify(
    port: int = BRIDGE_PORT,
    *,
    probe_url: str | None = None,
    ready_timeout_s: float = 30.0,
) -> list[str]:
    """Prove the sandbox is dark and the bridge answers, or refuse to start.

    Two checks, both mandatory. Every darkness probe must be unreachable —
    any response at all means the namespace has a route out, and an
    autonomous run must not start. *probe_url* names one probe; left unset,
    the defaults in :data:`_DARKNESS_PROBES` run, which include a raw-IP
    probe so a broken resolver cannot masquerade as darkness. The bridged
    endpoint must list its models within *ready_timeout_s* (retried, because
    the forwarder starts moments before this check). Returns the served
    model names.
    """
    probes = (probe_url,) if probe_url is not None else _DARKNESS_PROBES
    for probe in probes:
        try:
            with urllib.request.urlopen(probe, timeout=3.0):
                pass
            raise SandboxError(
                f"the sandbox can reach {probe} — the network namespace is not "
                f"isolated. Refusing to start an autonomous run. Check that the "
                f"container was launched with --net --network none."
            )
        except SandboxError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError):
            continue  # dark, as required

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
        f"{ready_timeout_s:.0f}s (last error: {last}). Did the host-side "
        f"'mason sandbox bridge' start, and is its upstream answering?"
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

    if agent.endpoint:
        rows.append(("+", f"upstream: {agent.endpoint} [[agent] endpoint]"))
        key_env = agent.resolved_api_key_env
        if key_env is None:
            rows.append(("?", "no [agent] api_key_env: the bridge will send no key"))
        elif os.environ.get(key_env):
            rows.append(("+", f"${key_env} is set; the bridge reads it on the host"))
        else:
            rows.append(
                (
                    "-",
                    f"${key_env} is not set here; sbatch exports the submitting "
                    f"environment, so export it before you submit",
                )
            )
        rows.append(
            (
                "?",
                "only a compute node can prove a compute node reaches that host "
                "(srun curl); this node's own reachability proves nothing",
            )
        )
        return rows

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
        extra_binaries: tuple[str, ...] = (),
        system_libs: tuple[str, ...] = (),
    ) -> None:
        self.engine = engine
        self.payload = payload
        self.env = env
        self.lib_dirs = lib_dirs
        self.error = error
        self.path_prepends = path_prepends or {}
        # Host-installed system libraries (ordinary RPMs in /usr/lib64 the
        # base image does not ship); bound file-by-file, never by directory.
        self.system_libs = system_libs
        # Launchers and helpers the engine command needs besides the payload
        # (mpirun, notably). Each contributes its install prefix to the
        # binds: a launcher reachable on PATH but with an unbound bin/ is
        # exactly the 'mpirun: not found' the first real run produced.
        self.extra_binaries = extra_binaries

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


#: Sonames the container image MUST provide itself: binding the host's
#: copies would splice one machine's core runtime into another's loader.
_CORE_SONAMES = (
    "libc.so",
    "libm.so",
    "libpthread.so",
    "libdl.so",
    "librt.so",
    "libutil.so",
    "libresolv.so",
    "ld-linux",
    "libgcc_s.so",
    "libstdc++.so",
)


def _lib_closure(ldd_output: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(site_dirs, system_files)`` from an ldd listing.

    A library under a site prefix means its whole directory gets bound. A
    library under a *system* directory is a different case: the base image
    has its own /usr/lib64, but only with the distro's default packages —
    an ordinary RPM like libpciaccess (the first real run's actual missing
    piece) is host-installed and absent from a minimal image. Those are
    bound as individual files, so they fill the gap without shadowing the
    image's own core runtime, which stays on the deny list above.
    """
    dirs: set[str] = set()
    files: set[str] = set()
    for line in ldd_output.splitlines():
        _name, arrow, rest = line.partition("=>")
        target = rest.split()[0] if arrow and rest.split() else ""
        if not target.startswith("/"):
            continue
        parent = str(Path(target).parent)
        system = any(
            parent == base or parent.startswith(base + "/") for base in _BASE_IMAGE_LIBS
        )
        if not system:
            dirs.add(parent)
        elif not any(core in Path(target).name for core in _CORE_SONAMES):
            files.add(target)
    return tuple(sorted(dirs)), tuple(sorted(files))


def snapshot_setup(
    engine: str,
    setup: tuple[str, ...],
    payload: str,
    *,
    extras: tuple[str, ...] = (),
    timeout_s: float = 120.0,
) -> SetupSnapshot:
    """Run one engine's setup on the host and record what it did.

    A login shell dumps its environment, runs the setup under ``set -e``,
    resolves *payload* and every *extras* executable (the launcher the
    engine command needs, typically ``mpirun``), dumps the environment
    again, and asks ``ldd`` for each binary's closure (best-effort: a
    missing ldd or a static binary yields no library dirs, and each
    binary's own prefix still gets bound). Any target failing to resolve
    fails the snapshot — a command that cannot resolve on the host cannot
    work inside the container either.
    """
    import tempfile

    targets = (payload, *extras)
    with tempfile.TemporaryDirectory() as scratch:
        files = {name: Path(scratch) / name for name in ("before", "after", "ldd")}
        probes = [Path(scratch) / f"which{i}" for i in range(len(targets))]
        probe_lines = []
        for target, probe in zip(targets, probes, strict=True):
            # Probes must not trip set -e: a target that fails to resolve is
            # its own diagnosis, distinct from a failing setup.
            probe_lines.append(
                f"command -v {shlex.quote(target)} > {shlex.quote(str(probe))} || true"
            )
            probe_lines.append(
                f'ldd "$(command -v {shlex.quote(target)})" '
                f">> {shlex.quote(str(files['ldd']))} 2>/dev/null || true"
            )
        script = "\n".join(
            [
                f"env -0 > {shlex.quote(str(files['before']))}",
                "set -e",
                *setup,
                *probe_lines,
                f"env -0 > {shlex.quote(str(files['after']))}",
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
        resolved = [p.read_text().strip() if p.exists() else "" for p in probes]
        missing = [t for t, r in zip(targets, resolved, strict=True) if not r]
        if result.returncode != 0 or missing:
            # Module systems chatter on stderr while succeeding, so name the
            # actual failure first and quote stderr only as supporting detail.
            tail = " | ".join(result.stderr.strip().splitlines()[-3:])
            detail = f" (stderr: {tail})" if tail else ""
            if result.returncode != 0:
                cause = f"setup exited {result.returncode}{detail}"
            else:
                cause = f"{missing[0]!r} did not resolve after setup{detail}"
            return SetupSnapshot(engine, "", {}, (), error=cause)
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
        ldd_text = files["ldd"].read_text() if files["ldd"].exists() else ""
        lib_dirs, system_libs = _lib_closure(ldd_text)
        return SetupSnapshot(
            engine,
            resolved[0],
            plain,
            lib_dirs,
            path_prepends=prepends,
            extra_binaries=tuple(resolved[1:]),
            system_libs=system_libs,
        )


def snapshot_engines(slab_cfg: SlabConfig) -> dict[str, SetupSnapshot]:
    """A snapshot per engine whose config declares setup lines.

    Each snapshot probes the payload binary and every launcher the engine
    command references, because a launcher resolves through the setup's
    PATH exactly like the payload does — and its install must be bound or
    the command dies inside the container at its first token.
    """
    snapshots: dict[str, SetupSnapshot] = {}
    for engine in ("qe", "lammps"):
        table = getattr(slab_cfg.engines, engine)
        if not table.setup:
            continue
        tokens = shlex.split(table.command) if table.command else []
        extras = tuple(t for t in tokens if t in ("mpirun", "mpiexec"))
        if engine == "qe" and getattr(table, "bin", None):
            # The bin form keeps pw.x off PATH on purpose; the setup lines
            # exist for its runtime libraries, so resolve the binary by its
            # absolute path rather than expecting the setup to export it.
            # The constructed command launches through mpirun, which must
            # resolve too unless the install bundles its own.
            payload = str(Path(table.bin) / "pw.x")
            if not (Path(table.bin) / "mpirun").is_file():
                extras = ("mpirun",)
        else:
            payload = payload_name(engine, table.command)
        snapshots[engine] = snapshot_setup(engine, table.setup, payload, extras=extras)
    return snapshots


def _snapshot_binds(snapshot: SetupSnapshot) -> list[str]:
    """Read-only binds for a snapshotted engine: installs and libraries.

    Every resolved binary contributes its install prefix — the launcher's
    included, because ``ldd`` of the payload names the launcher's ``lib``
    but never its ``bin``, and a bound ``lib`` next to an absent ``bin``
    is precisely how 'mpirun: not found' happens inside the container.
    """
    if snapshot.error:
        return []
    prefixes = []
    for binary in (snapshot.payload, *snapshot.extra_binaries):
        bin_dir = Path(binary).parent
        prefixes.append(str(bin_dir.parent if bin_dir.name == "bin" else bin_dir))
    return [
        f"{d}:{d}:ro"
        for d in dict.fromkeys([*prefixes, *snapshot.lib_dirs, *snapshot.system_libs])
    ]


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
    # With no [hpc] table, an unset profile would derive to "laptop" — and
    # the prompt would tell a whole compute allocation to think small (the
    # first real run's agent spent turns reasoning from "this is a laptop").
    # The sandbox owns one real node: workstation is the honest size.
    agent_table.setdefault("compute_profile", "workstation")
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


def _upstream_lines(agent: AgentConfig, workspace_root: Path) -> list[str]:
    """The prologue lines that settle ``$UPSTREAM``, by where the model is.

    The precedence is Mason's everywhere else
    (:func:`mason.serve.discover_endpoint`): a configured endpoint wins, and
    only without one does the job read a serve record. A machine pointed at
    a gateway must not render a job that demands a server nobody started.
    """
    if agent.endpoint:
        lines = [f"UPSTREAM={shlex.quote(agent.endpoint)}"]
        key_env = agent.resolved_api_key_env
        if key_env:
            # By name, and never echoed. sbatch exports the submitting
            # environment by default, but a cluster set to --export=NONE does
            # not, and the job should die here saying which variable is
            # missing rather than at the first request hours later.
            lines.append(
                # The name is escaped in the message: an unescaped $NAME would
                # expand, printing an empty word when unset and the key itself
                # if it ever were set.
                f'[ -n "${key_env}" ] || {{ echo "\\${key_env} is not set in this '
                f'job\'s environment, and the gateway at $UPSTREAM needs it" >&2; '
                f"exit 1; }}"
            )
        return lines
    record = record_path(workspace_root).resolve()
    return [
        f"RECORD={shlex.quote(str(record))}",
        '[ -f "$RECORD" ] || { echo "no serve record at $RECORD;'
        " start the model server first ('mason serve start')\" >&2; exit 1; }",
        f'UPSTREAM=$({shlex.quote(_python())} -c {shlex.quote(_UPSTREAM_SNIPPET)} "$RECORD")',
    ]


def _key_flag(agent: AgentConfig) -> str:
    """``--key-env NAME`` when a configured gateway needs a key, else nothing."""
    key_env = agent.resolved_api_key_env if agent.endpoint else None
    return f" --key-env {shlex.quote(key_env)}" if key_env else ""


def _header_flags(agent: AgentConfig) -> str:
    """``--header 'Name: value'`` per configured gateway header, else nothing.

    Follows :func:`_key_flag`'s rule: only a URL upstream (a configured
    endpoint) takes them — a byte relay to a served model forwards what the
    client sends and the bridge would refuse the flag.
    """
    if not agent.endpoint:
        return ""
    return "".join(
        f" --header {shlex.quote(spec)}" for spec in agent.sandbox.gateway_headers
    )


def _sandbox_context(
    binds: list[str],
    *,
    gpu: bool,
    memories: Path | None,
    engine_tasks: int | None,
) -> str:
    """The sandbox facts as a prompt block, written at render time.

    Every autonomous run used to spend its opening steps reading the
    submission script to learn its own cage — the binds, the darkness, the
    missing scheduler. The render already knows all of it, so it says so
    once, and the session prompt carries it from step one.
    """
    rows = []
    for spec in binds:
        parts = spec.split(":")
        destination = parts[1] if len(parts) > 1 else parts[0]
        mode = parts[2] if len(parts) > 2 else "rw"
        rows.append(f"    {mode:<2}  {destination}")
    lines = [
        "# Sandbox",
        "",
        "This session runs inside an Apptainer container on a compute node.",
        "These facts come from the job that launched you; do not spend steps",
        "inspecting the submission script to rediscover them.",
        "",
        "- The network is dark: no internet, no DNS. Downloads fail by",
        "  design, and HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are set so hub",
        "  clients refuse up front instead of hanging.",
        "- There is no scheduler here: no sbatch, srun, or squeue. Engine",
        "  legs run in this container, on this job's allocation.",
        "- The model endpoint is a local bridge to the outside; do not",
        "  reconfigure it.",
        "- $HOME is not mounted."
        + (f" Machine memories are at {memories} (rw)." if memories else ""),
        "- GPU: the host driver stack is mounted (--nv); the job holds a GPU."
        if gpu
        else "- GPU: none on this partition.",
        "- Mounted paths and their modes (nothing else exists in here):",
        *rows,
    ]
    if engine_tasks is not None:
        lines.append(f"- MPI engines launch with {engine_tasks} rank(s).")
    return "\n".join(lines)


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
    engine_tasks: int | None = None,
) -> tuple[str, list[str], str]:
    """The batch script for one autonomous, network-dark session.

    Returns ``(script, warnings, context)`` — *context* is the ``# Sandbox``
    prompt block describing the rendered cage (binds, darkness, GPU, no
    scheduler). The CLI writes it next to the script as ``context.md``, the
    job exports ``SLAB_SANDBOX_CONTEXT`` naming it, and the session prompt
    then carries the facts from step one.

    Host side: settle the upstream (a configured ``[agent] endpoint``, else
    the serve record's), and bridge it onto a unix socket with one fixed
    destination. Container side: forward the socket to loopback, prove
    darkness and reachability with ``mason sandbox verify`` (either failing
    aborts the job), then run the goal with ``mason run --auto``.

    A gateway upstream authenticates with the key named by ``[agent]
    api_key_env``, read on the host at job start. The container is launched
    ``--cleanenv`` and its rendered config drops ``api_key_env``, so the key
    never crosses the boundary.
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
    # The machine's memory, read-write. --no-home hides ~/.config, so without
    # this the job neither reads what earlier sessions learned nor keeps what
    # it learns itself, which is exactly the case memory exists for: the
    # overnight job that trips on a quirk at 03:00. Apptainer refuses a bind
    # whose source is missing, so an untouched machine gets the directory now.
    memories = memory_dir()
    if agent.memory:
        memories.mkdir(parents=True, exist_ok=True)
        binds.append(f"{memories}:{memories}:rw")
    binds = _collapse_binds(binds)
    mason = _mason_bin()

    # GPU passthrough derives from the target partition: a gres that names
    # gpus means the job will hold one, and --nv mounts the host's driver
    # stack so a torch-backed served engine (a rootstock MLIP worker) can
    # see it. A CPU partition renders without it, and the container stays
    # minimal.
    _, partition_spec = hpc.resolve_partition(partition)
    isolation_flags = "--containall --no-home --cleanenv --net --network none"
    if partition_spec.gres and "gpu" in partition_spec.gres.lower():
        isolation_flags += " --nv"

    image = str(Path(agent.sandbox.image).expanduser())
    prologue = [
        f"IMAGE={shlex.quote(image)}",
        '[ -f "$IMAGE" ] || { echo "no container image at $IMAGE; build it '
        "(e.g. 'apptainer build $IMAGE docker://rockylinux:9') on a filesystem "
        'the compute nodes mount" >&2; exit 1; }',
        *_upstream_lines(agent, workspace_root),
        'BRIDGE="$(mktemp -d)/llm.sock"',
        f'{mason} sandbox bridge "$BRIDGE" "$UPSTREAM"'
        f"{_key_flag(agent)}{_header_flags(agent)} &",
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
            isolation_flags,
            *(f"--bind {shlex.quote(spec)}" for spec in binds),
            f'--bind "$BRIDGE":{_SOCKET_IN_CONTAINER}',
            f"--env SLAB_WORKSPACE={shlex.quote(str(workspace_root))}",
            f"--env SLAB_CONFIG={shlex.quote(str(toml_path))}",
            # Name the memory directory rather than letting it derive from a
            # $HOME the container does not have.
            *(
                [f"--env SLAB_MEMORY_DIR={shlex.quote(str(memories))}"]
                if agent.memory
                else []
            ),
            # --cleanenv would strip it, and the bin-form qe command sizes
            # its mpirun from it — 1 if the scheduler did not set it. An
            # explicit --engine-tasks pins it instead: a whole node's rank
            # count overwhelms a small cell's plane-wave distribution.
            (
                f"--env SLURM_NTASKS={engine_tasks}"
                if engine_tasks is not None
                else '--env SLURM_NTASKS="${SLURM_NTASKS:-1}"'
            ),
            # OpenMPI inside the namespace: there is no ssh and no
            # scheduler, so component selection must not go looking for
            # either — 'isolated' launches local ranks with no agent, and
            # CMA single-copy is unavailable in user namespaces. Other MPI
            # implementations ignore both variables.
            "--env OMPI_MCA_plm=isolated",
            "--env OMPI_MCA_btl_vader_single_copy_mechanism=none",
            # The network namespace is dark, but ML stacks that do not know
            # that will still *attempt* a model download — burning a worker
            # on a doomed fetch against a read-only bind. These flags make
            # hub clients refuse up front with a message that says offline,
            # which is the truth, instead of a half-written cache.
            "--env HF_HUB_OFFLINE=1",
            "--env TRANSFORMERS_OFFLINE=1",
            "--env HF_DATASETS_OFFLINE=1",
            # The prompt-block description of this cage, written by the
            # render next to this script; the session reads it at startup.
            f"--env SLAB_SANDBOX_CONTEXT="
            f"{shlex.quote(str(toml_path.with_name('context.md')))}",
            # The venv on PATH, so the agent's shell probes find python and
            # the console scripts without knowing the install layout.
            f"--env PATH={shlex.quote(str(Path(_python()).parent))}"
            ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            '"$IMAGE"',
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
    context = _sandbox_context(
        binds,
        gpu="--nv" in isolation_flags,
        memories=memories if agent.memory else None,
        engine_tasks=engine_tasks,
    )
    context_path = toml_path.with_name("context.md")
    if not context_path.is_relative_to(project):
        warnings.append(
            f"{context_path} is outside the project bind, so the in-job "
            f"prompt cannot carry the sandbox context; render into the "
            f"project (the default) to keep it"
        )
    return script, warnings, context
