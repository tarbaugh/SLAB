"""What this process may use, what one launch asks for, and what a job may take.

Three sizes live here, and nothing else in SLAB guesses at any of them.

* The :class:`Budget` is what the current process may use: the cpu ids of
  its affinity mask and the gpu ids it can see. :func:`budget` discovers it
  from the platform and the scheduler's variables.
* The :class:`Envelope` is what one launch runs under: a slice of cpu ids
  and gpu ids, a rank count, and a thread count. :func:`envelope` reads the
  per-launch ``SLAB_*`` variables a parent exported, and falls back to the
  budget. :func:`env_for` is the parent's half and :func:`apply` the
  child's: export the envelope, then take the affinity mask.
* The :class:`JobSize` is what a batch job asks of the scheduler, and
  :func:`check_size` refuses one that does not fit the partition's declared
  node.

:func:`fill` connects the engines to the envelope. An engine command holds
the placeholders ``{ntasks}``, ``{threads}``, and ``{gpus}`` where it wants
the launch's numbers, and SLAB fills only the placeholders the command asks
for. A command without one runs as written, so a route that hardcodes its
rank count keeps doing what it says.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slab.config import Partition, memory_mb
from slab.errors import EngineNotAvailableError, JobSizeError

PLACEHOLDERS = ("ntasks", "threads", "gpus")
_PLACEHOLDER = re.compile(r"(?<!\$)\{(ntasks|threads|gpus)\}")
_NVIDIA_SMI_TIMEOUT_S = 10


@dataclass(frozen=True)
class Budget:
    """The cpu ids and gpu ids this process may use.

    Examples:
        >>> Budget(cpus=(0, 1, 2, 3), gpus=("0",)).counts
        {'cpus': 4, 'gpus': 1}
    """

    cpus: tuple[int, ...]
    gpus: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """The two counts, for a listing."""
        return {"cpus": len(self.cpus), "gpus": len(self.gpus)}


@dataclass(frozen=True)
class Envelope:
    """What one launch runs under: its slice, its rank count, its thread count.

    Examples:
        >>> Envelope(cpus=(0, 1, 2, 3), gpus=("0",), ntasks=2, threads=2).as_dict()
        {'cpus': [0, 1, 2, 3], 'gpus': ['0'], 'ntasks': 2, 'threads': 2}
    """

    cpus: tuple[int, ...]
    gpus: tuple[str, ...] = ()
    ntasks: int = 1
    threads: int = 1

    def as_dict(self) -> dict[str, object]:
        """The JSON form a run record keeps."""
        return {
            "cpus": list(self.cpus),
            "gpus": list(self.gpus),
            "ntasks": self.ntasks,
            "threads": self.threads,
        }


def budget() -> Budget:
    """Discover what this process may use.

    Cpus come from the affinity mask where the platform reports one (a
    SLURM cgroup shrinks it to the allocation) and from the machine's count
    elsewhere. Gpus come from ``CUDA_VISIBLE_DEVICES`` when it is set (an
    empty value means none), else ``SLURM_JOB_GPUS`` or ``SLURM_STEP_GPUS``,
    else one cached ``nvidia-smi -L`` probe, else none.

    Examples:
        >>> import os
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        >>> budget().gpus
        ('0', '1')
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
        >>> budget().gpus
        ()
        >>> del os.environ["CUDA_VISIBLE_DEVICES"]
        >>> len(budget().cpus) >= 1
        True
    """
    return Budget(cpus=_affinity_cpus(), gpus=_visible_gpus())


def envelope() -> Envelope:
    """What this launch runs under.

    The per-launch variables ``SLAB_CPUS`` (comma list), ``SLAB_GPUS``,
    ``SLAB_NTASKS``, and ``SLAB_THREADS`` win when a parent exported them.
    Otherwise the slice is the whole :func:`budget`, the rank count is
    ``SLURM_NTASKS``, and the thread count is ``SLURM_CPUS_PER_TASK``; each
    of the two counts is 1 when unset or not a number, so an interactive
    smoke test stays serial.

    Examples:
        >>> import os
        >>> for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS"):
        ...     _ = os.environ.pop(name, None)
        >>> os.environ["SLURM_NTASKS"] = "16"
        >>> envelope().ntasks
        16
        >>> os.environ["SLURM_NTASKS"] = "not-a-number"
        >>> envelope().ntasks
        1
        >>> del os.environ["SLURM_NTASKS"]
        >>> os.environ["SLAB_CPUS"] = "4,5"
        >>> os.environ["SLAB_GPUS"] = "1"
        >>> os.environ["SLAB_NTASKS"] = "2"
        >>> os.environ["SLAB_THREADS"] = "1"
        >>> envelope()
        Envelope(cpus=(4, 5), gpus=('1',), ntasks=2, threads=1)
        >>> for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS"):
        ...     del os.environ[name]
    """
    cpus_text = os.environ.get("SLAB_CPUS")
    gpus_text = os.environ.get("SLAB_GPUS")
    if cpus_text is not None or gpus_text is not None:
        cpus = _int_list(cpus_text) if cpus_text is not None else _affinity_cpus()
        gpus = _id_list(gpus_text) if gpus_text is not None else _visible_gpus()
    else:
        found = budget()
        cpus, gpus = found.cpus, found.gpus
    ntasks = _positive_int(os.environ.get("SLAB_NTASKS")) or _positive_int(
        os.environ.get("SLURM_NTASKS")
    )
    threads = _positive_int(os.environ.get("SLAB_THREADS")) or _positive_int(
        os.environ.get("SLURM_CPUS_PER_TASK")
    )
    return Envelope(cpus=cpus, gpus=gpus, ntasks=ntasks or 1, threads=threads or 1)


def env_for(env: Envelope) -> dict[str, str]:
    """The variables a child process needs to run inside *env*.

    The four ``SLAB_*`` variables :func:`envelope` reads back, plus
    ``OMP_NUM_THREADS`` for the threaded engines and ``CUDA_VISIBLE_DEVICES``
    for the GPU ones. An envelope without gpus exports an empty
    ``CUDA_VISIBLE_DEVICES``, which hides every device.

    Examples:
        >>> env_for(Envelope(cpus=(0, 1), gpus=(), ntasks=2, threads=1))["CUDA_VISIBLE_DEVICES"]
        ''
        >>> env_for(Envelope(cpus=(2, 3), gpus=("1",), ntasks=1, threads=2))["OMP_NUM_THREADS"]
        '2'
    """
    return {
        "SLAB_CPUS": ",".join(str(cpu) for cpu in env.cpus),
        "SLAB_GPUS": ",".join(env.gpus),
        "SLAB_NTASKS": str(env.ntasks),
        "SLAB_THREADS": str(env.threads),
        "OMP_NUM_THREADS": str(env.threads),
        "CUDA_VISIBLE_DEVICES": ",".join(env.gpus),
    }


def apply(env: Envelope) -> bool:
    """Pin the current process to the envelope's cpus; True when a mask was set.

    Where the platform has ``sched_setaffinity`` the mask bounds every
    rank and thread this process starts. Elsewhere this is a no-op that
    returns False. Cpus outside the current mask are dropped from the
    request, because the kernel would refuse the whole call.
    """
    setter = getattr(os, "sched_setaffinity", None)
    getter = getattr(os, "sched_getaffinity", None)
    if setter is None or getter is None or not env.cpus:
        return False
    try:
        allowed = set(getter(0))
        wanted = set(env.cpus) & allowed
        if not wanted:
            return False
        setter(0, wanted)
    except OSError:
        return False
    return True


def placeholders(command: str) -> list[str]:
    """The placeholders a command asks for, in order of first appearance.

    Examples:
        >>> placeholders("mpirun -np {ntasks} lmp -k on g {gpus} t {threads} -sf kk")
        ['ntasks', 'gpus', 'threads']
        >>> placeholders("srun pw.x")
        []
    """
    return list(dict.fromkeys(_PLACEHOLDER.findall(command)))


def fill(command: str, env: Envelope, *, route: str | None = None) -> str:
    """Replace ``{ntasks}``, ``{threads}``, and ``{gpus}`` in an engine command.

    ``{gpus}`` becomes the number of gpus in the envelope. A command that
    asks for it under an envelope with none is refused, naming the route,
    because a GPU build launched with ``g 0`` would fail later and less
    clearly. A command without a placeholder comes back unchanged, and
    shell forms like ``${OMP_NUM_THREADS}`` are left alone.

    Examples:
        >>> env = Envelope(cpus=(0, 1, 2, 3), gpus=("0", "1"), ntasks=2, threads=2)
        >>> fill("mpirun -np {ntasks} lmp -k on g {gpus} t {threads} -sf kk", env)
        'mpirun -np 2 lmp -k on g 2 t 2 -sf kk'
        >>> fill("env OMP_NUM_THREADS=${threads} pw.x", env)
        'env OMP_NUM_THREADS=${threads} pw.x'
        >>> fill("lmp -k on g {gpus}", Envelope(cpus=(0,)), route="lammps-gpu")
        Traceback (most recent call last):
        ...
        slab.errors.EngineNotAvailableError: route 'lammps-gpu' asks for {gpus} but ...
    """
    if "gpus" in placeholders(command) and not env.gpus:
        who = f"route {route!r}" if route else f"the command {command!r}"
        raise EngineNotAvailableError(
            f"{who} asks for {{gpus}} but this launch holds no GPU (the budget "
            f"lists none, or the launch was not sized with gpus=); size the launch "
            f"with gpus= or pick a route without a GPU switch"
        )
    values = {"ntasks": str(env.ntasks), "threads": str(env.threads), "gpus": str(len(env.gpus))}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], command)


class JobSize(BaseModel):
    """What one batch job asks of the scheduler, per node.

    ``ntasks_per_node`` has no default: a sized job says its rank count.

    Examples:
        >>> JobSize(ntasks_per_node=8, gpus_per_node=2).cpus_per_task
        1
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: int = Field(default=1, ge=1)
    ntasks_per_node: int = Field(ge=1)
    cpus_per_task: int = Field(default=1, ge=1)
    gpus_per_node: int = Field(default=0, ge=0)
    mem: str | None = None

    @field_validator("mem")
    @classmethod
    def _mem_parses(cls, value: str | None) -> str | None:
        if value is not None:
            memory_mb(value)
        return value


def job_size(
    *,
    nodes: int | None = None,
    ntasks_per_node: int | None = None,
    cpus_per_task: int | None = None,
    gpus_per_node: int | None = None,
    mem: str | None = None,
) -> JobSize | None:
    """A :class:`JobSize` from optional fields, or None when none is given.

    The CLI flags and the tool arguments are all optional, so an unsized
    request stays unsized. A partial size must still name
    ``ntasks_per_node``.

    Examples:
        >>> job_size() is None
        True
        >>> job_size(ntasks_per_node=4, gpus_per_node=1).nodes
        1
        >>> job_size(gpus_per_node=1)
        Traceback (most recent call last):
        ...
        slab.errors.JobSizeError: a sized job names its rank count: pass ntasks_per_node ...
    """
    given = {
        "nodes": nodes,
        "ntasks_per_node": ntasks_per_node,
        "cpus_per_task": cpus_per_task,
        "gpus_per_node": gpus_per_node,
        "mem": mem,
    }
    fields = {key: value for key, value in given.items() if value is not None}
    if not fields:
        return None
    if "ntasks_per_node" not in fields:
        raise JobSizeError(
            "a sized job names its rank count: pass ntasks_per_node (with cpus_per_task, "
            "gpus_per_node, nodes, and mem as needed), or pass no size at all to take "
            "the partition's own directives"
        )
    return JobSize.model_validate(fields)


def check_size(size: JobSize, partition_name: str, spec: Partition) -> JobSize:
    """Refuse a size the partition's declared node cannot hold; return it otherwise.

    Each refusal names the cap and the config field that declares it. A
    partition without a ``node`` table cannot be sized at all.

    Examples:
        >>> spec = Partition.model_validate(
        ...     {"node": {"cpus": 64, "gpus": 4, "mem": "480G"}, "max_nodes": 2})
        >>> check_size(JobSize(ntasks_per_node=4, gpus_per_node=4), "gpu", spec).gpus_per_node
        4
        >>> check_size(JobSize(ntasks_per_node=4, gpus_per_node=5), "gpu", spec)
        Traceback (most recent call last):
        ...
        slab.errors.JobSizeError: gpus_per_node=5 exceeds the 4 gpus of one gpu node ...
        >>> check_size(JobSize(ntasks_per_node=1), "cpu", Partition())
        Traceback (most recent call last):
        ...
        slab.errors.JobSizeError: partition 'cpu' declares no node, so ...
    """
    node = spec.node
    if node is None:
        raise JobSizeError(
            f"partition {partition_name!r} declares no node, so a job on it cannot be "
            f"sized; add [hpc.partitions.{partition_name}.node] with cpus (and gpus, "
            f"mem) to the slab config, or submit without a size"
        )
    where = f"[hpc.partitions.{partition_name}"
    if size.nodes > spec.max_nodes:
        raise JobSizeError(
            f"nodes={size.nodes} exceeds the {spec.max_nodes} node(s) one job may take on "
            f"{partition_name} ({where}] max_nodes)"
        )
    cores = size.ntasks_per_node * size.cpus_per_task
    if cores > node.cpus:
        raise JobSizeError(
            f"ntasks_per_node={size.ntasks_per_node} x cpus_per_task={size.cpus_per_task} "
            f"= {cores} exceeds the {node.cpus} cpus of one {partition_name} node "
            f"({where}.node] cpus)"
        )
    if size.gpus_per_node > node.gpus:
        raise JobSizeError(
            f"gpus_per_node={size.gpus_per_node} exceeds the {node.gpus} gpus of one "
            f"{partition_name} node ({where}.node] gpus)"
        )
    if size.mem is not None:
        if node.mem is None:
            raise JobSizeError(
                f"mem={size.mem} cannot be checked: {partition_name} declares no node "
                f"memory ({where}.node] mem)"
            )
        if memory_mb(size.mem) > memory_mb(node.mem):
            raise JobSizeError(
                f"mem={size.mem} exceeds the {node.mem} of one {partition_name} node "
                f"({where}.node] mem)"
            )
    return size


# -- discovery ---------------------------------------------------------------


def _affinity_cpus() -> tuple[int, ...]:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            found = tuple(sorted(getter(0)))
            if found:
                return found
        except OSError:  # pragma: no cover - platform quirk
            pass
    return tuple(range(os.cpu_count() or 1))


def _visible_gpus() -> tuple[str, ...]:
    """The gpu ids this process may name in ``CUDA_VISIBLE_DEVICES``.

    ``CUDA_VISIBLE_DEVICES`` is taken as written. ``SLURM_JOB_GPUS`` and
    ``SLURM_STEP_GPUS`` hold the node's global ids (``2,3`` on a node with
    four), but a job under cgroup device constraints sees its devices
    renumbered from zero, so exporting those ids to a child would name
    devices it cannot open. Only their count is used, and the ids become
    ``0, 1, ...``.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return _id_list(visible)
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        value = os.environ.get(name)
        if value:
            return tuple(str(index) for index in range(len(_id_list(value))))
    return _probed_gpus()


@functools.lru_cache(maxsize=1)
def _probed_gpus() -> tuple[str, ...]:
    """The gpu ids ``nvidia-smi -L`` lists, probed once per process."""
    if shutil.which("nvidia-smi") is None:
        return ()
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()
    found = re.findall(r"^GPU (\d+):", result.stdout, flags=re.MULTILINE)
    return tuple(found)


def _id_list(text: str) -> tuple[str, ...]:
    """A comma list of ids, ranges expanded (``0-3`` is ``0,1,2,3``).

    Examples:
        >>> _id_list("0,2-3, 5")
        ('0', '2', '3', '5')
        >>> _id_list("")
        ()
    """
    ids: list[str] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        low, dash, high = piece.partition("-")
        if dash and low.isdigit() and high.isdigit():
            ids.extend(str(n) for n in range(int(low), int(high) + 1))
        else:
            ids.append(piece)
    return tuple(dict.fromkeys(ids))


def _int_list(text: str) -> tuple[int, ...]:
    """Cpu ids from a comma list; a piece that is not a number is dropped.

    Examples:
        >>> _int_list("0-1,4,x")
        (0, 1, 4)
    """
    return tuple(sorted({int(piece) for piece in _id_list(text) if piece.isdigit()}))


def _positive_int(text: str | None) -> int | None:
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


__all__ = [
    "PLACEHOLDERS",
    "Budget",
    "Envelope",
    "JobSize",
    "apply",
    "budget",
    "check_size",
    "env_for",
    "envelope",
    "fill",
    "job_size",
    "placeholders",
]
