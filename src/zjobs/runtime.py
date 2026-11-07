from __future__ import annotations

import inspect
import multiprocessing as mp
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import cloudpickle
import zmq

from ._protocol import configure, stop_processes
from ._remote import RemoteCancelled, RemoteTransport
from ._worker import worker_main
from .graph import Node, ReplicaPlacement

_SHIPPING_LOCK = threading.Lock()


def _ship(task: Any) -> bytes:
    # Importable user modules otherwise pickle by reference and would have to
    # exist on every agent. Dependencies outside this module remain installed.
    module = inspect.getmodule(task) or inspect.getmodule(type(task))
    with _SHIPPING_LOCK:
        registered = cloudpickle.list_registry_pickle_by_value()
        temporary = (
            module is not None and module.__name__ not in ("__main__", "builtins")
            and module.__name__ not in registered
        )
        if temporary:
            cloudpickle.register_pickle_by_value(module)
        try:
            return cloudpickle.dumps(task, protocol=5)
        finally:
            if temporary:
                cloudpickle.unregister_pickle_by_value(module)


class RunError(RuntimeError):
    """A contextual execution, transport, deadline, or serialization failure."""


@dataclass
class _Stage:
    node: Node
    downstream: list[int] = field(default_factory=list)
    pending: deque[bytes] = field(default_factory=deque)
    complete: bool = False


@dataclass
class _Replica:
    stage: int
    rank: int
    process: mp.Process | None
    machine: str = "local"
    state: str = "starting"
    since: float = field(default_factory=time.monotonic)
    payload: bytes | None = None
    work: int | None = None
    exited_since: float | None = None


class Run(Iterator[Any]):
    """Bounded output iterator; close explicitly or use ``with graph.run()``."""

    def __init__(
        self, nodes: tuple[Node, ...], capacity: int, output_capacity: int,
        worker_timeout: float, consumer_timeout: float, max_payload_bytes: int,
        agents: Mapping[str, str] | None = None, agent_timeout: float = 3.0,
    ) -> None:
        self._nodes = nodes
        self._capacity = capacity
        self._worker_timeout = worker_timeout
        self._consumer_timeout = consumer_timeout
        self._payload_limit = max_payload_bytes
        self._agent_timeout = agent_timeout
        self._run_id = uuid.uuid4().hex
        self._agents = dict(agents) if agents is not None else None
        if self._agents is not None:
            if not self._agents:
                raise ValueError("agents must contain at least one machine endpoint")
            if any(
                not isinstance(machine, str) or not machine
                or not isinstance(endpoint, str) or not endpoint.startswith("tcp://")
                for machine, endpoint in self._agents.items()
            ):
                raise ValueError("agents require nonempty machine identities and TCP endpoints")
            if len(set(self._agents.values())) != len(self._agents):
                raise ValueError("duplicate agent endpoints are not allowed")
        placements = []
        for node in nodes:
            if self._agents is None:
                if node.placement:
                    raise ValueError("stage placement requires agents at run startup")
                hosts = (("local", node.replicas),)
            else:
                if not node.placement:
                    raise ValueError(f"stage {node.name!r} requires explicit placement when using agents")
                hosts = node.placement
                for machine, _ in hosts:
                    if machine not in self._agents:
                        raise ValueError(f"stage {node.name!r}: missing agent {machine!r}")
            rank = 0
            for machine, count in hosts:
                for local_rank in range(count):
                    placements.append(ReplicaPlacement(
                        node.name, machine, rank, node.replicas, local_rank, count,
                    ))
                    rank += 1
        self._placement = tuple(placements)
        self._outputs: queue.Queue[bytes] = queue.Queue(output_capacity)
        self._cancel = threading.Event()
        self._finished = threading.Event()
        self._error: RunError | None = None
        self._lock = threading.Lock()
        self._last_consumed = time.monotonic()
        self._stats: dict[str, Any] = {
            "state": "starting", "queued": {}, "peak_queued": {},
            "output_buffered": 0, "peak_output_buffered": 0,
            "produced": {}, "completed_work": {}, "worker_pids": [],
            "workers": [], "run_id": self._run_id,
        }
        # Fail code shipping before creating any owned processes.
        self._definitions: list[bytes] = []
        for node in nodes:
            try:
                definition = _ship(node.task)
            except Exception as exc:
                raise RunError(f"stage {node.name!r}: serialize callable: {exc}") from exc
            if len(definition) > max_payload_bytes:
                raise RunError(f"stage {node.name!r}: serialized callable exceeds payload limit")
            self._definitions.append(definition)
        self._thread = threading.Thread(target=self._drive, name="zjobs-controller", daemon=True)
        self._thread.start()

    def __iter__(self) -> Run:
        return self

    def __next__(self) -> Any:
        while True:
            if self._error is not None:
                self.close()
                raise self._error
            try:
                payload = self._outputs.get(timeout=0.05)
            except queue.Empty:
                if self._finished.is_set():
                    if self._error is not None:
                        raise self._error
                    raise StopIteration
                continue
            with self._lock:
                self._last_consumed = time.monotonic()
            try:
                return cloudpickle.loads(payload)
            except Exception as exc:
                self.close()
                error = RunError(f"final output: deserialize payload: {exc}")
                self._record_error(error)
                raise error from exc

    def batches(self, size: int) -> Iterator[list[Any]]:
        """Group final outputs, retaining a final partial batch."""
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("batch size must be a positive integer")
        batch: list[Any] = []
        for item in self:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __enter__(self) -> Run:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Cancel remaining work and join/terminate only this run's processes."""
        self._cancel.set()
        self._thread.join()
        while True:
            try:
                self._outputs.get_nowait()
            except queue.Empty:
                break

    @property
    def placement(self) -> tuple[ReplicaPlacement, ...]:
        """Immutable per-replica placement resolved before processes launch."""
        return self._placement

    @property
    def status(self) -> dict[str, Any]:
        """Snapshot of controller counts; queues are bounded by item count."""
        with self._lock:
            result = {
                key: value.copy() if isinstance(value, (dict, list)) else value
                for key, value in self._stats.items()
            }
            result["workers"] = [worker.copy() for worker in self._stats["workers"]]
        result["output_buffered"] = self._outputs.qsize()
        result["error"] = str(self._error) if self._error else None
        result["placement"] = [
            {
                "stage": p.stage, "machine": p.machine, "rank": p.rank,
                "replicas": p.replicas, "local_rank": p.local_rank,
                "local_replicas": p.local_replicas,
            } for p in self._placement
        ]
        return result

    def _drive(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self._record_error(exc)
        finally:
            self._finished.set()

    def _record_error(self, exc: Exception) -> None:
        if isinstance(exc, RemoteCancelled) and self._cancel.is_set():
            with self._lock:
                self._stats["state"] = "cancelled"
            return
        if self._error is None:
            self._error = (
                exc if isinstance(exc, RunError)
                else RunError(f"controller: {type(exc).__name__}: {exc}")
            )
        with self._lock:
            self._stats["state"] = "failed"

    def _execute(self) -> None:
        with zmq.Context() as context:
            if self._agents is not None:
                used = {p.machine for p in self._placement}
                remote = RemoteTransport(
                    context, {key: value for key, value in self._agents.items() if key in used},
                    self._run_id, self._agent_timeout,
                    self._cancel,
                )
                try:
                    self._coordinate(remote)
                finally:
                    remote.stop()
                return
            with context.socket(zmq.ROUTER) as socket:
                configure(socket)
                socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                self._coordinate(socket)

    def _coordinate(self, socket: zmq.Socket | RemoteTransport) -> None:
        stages = [_Stage(node) for node in self._nodes]
        indices = {node: index for index, node in enumerate(self._nodes)}
        for index, stage in enumerate(stages):
            for parent in stage.node.inputs:
                stages[indices[parent]].downstream.append(index)
        replicas: dict[bytes, _Replica] = {}
        produced = {node.name: 0 for node in self._nodes}
        completed_work = {node.name: 0 for node in self._nodes}
        peak_queued = produced.copy()
        blocked_since: float | None = None
        next_work = 0
        try:
            endpoint = ""
            if isinstance(socket, zmq.Socket):
                port = socket.bind_to_random_port("tcp://127.0.0.1")
                endpoint = f"tcp://127.0.0.1:{port}"
            spawn = mp.get_context("spawn")
            stage_indices = {node.name: index for index, node in enumerate(self._nodes)}
            specs: dict[str, list[dict]] = {}
            definitions = {}
            for placement in self._placement:
                if self._cancel.is_set():
                    return
                index = stage_indices[placement.stage]
                stage = stages[index]
                identity = f"{self._run_id}:{index}:{placement.rank}".encode("ascii")
                process = None
                if isinstance(socket, zmq.Socket):
                    process = spawn.Process(
                        target=worker_main,
                        args=(
                            endpoint, identity, self._definitions[index],
                            not stage.node.inputs, placement.rank, stage.node.replicas,
                            self._payload_limit,
                            placement.machine, placement.local_rank, placement.local_replicas,
                        ),
                        name=f"zjobs-{stage.node.name}-{placement.rank}",
                    )
                    process.start()
                    with self._lock:
                        self._stats["worker_pids"].append(process.pid)
                else:
                    specs.setdefault(placement.machine, []).append({
                        "identity": identity.decode(), "stage": stage.node.name,
                        "source": not stage.node.inputs, "rank": placement.rank,
                        "replicas": placement.replicas, "local_rank": placement.local_rank,
                        "local_replicas": placement.local_replicas,
                    })
                    definitions[identity] = self._definitions[index]
                replicas[identity] = _Replica(index, placement.rank, process, placement.machine)
            if isinstance(socket, RemoteTransport):
                pids = socket.start(specs, definitions, self._payload_limit, self._worker_timeout)
                for replica in replicas.values():
                    replica.since = time.monotonic()
                with self._lock:
                    self._stats["worker_pids"] = list(pids.values())
                    self._stats["workers"] = [
                        {"machine": replicas[key].machine, "pid": pid,
                         "stage": self._nodes[replicas[key].stage].name, "rank": replicas[key].rank}
                        for key, pid in pids.items()
                    ]
            with self._lock:
                self._stats["state"] = "running"
            order = deque(replicas)
            while not self._cancel.is_set():
                now = time.monotonic()
                # One result credit per replica. Fan-out is atomic, so a slow
                # branch backpressures its producer instead of dropping copies.
                order.rotate(-1)
                for identity in order:
                    replica = replicas[identity]
                    stage = stages[replica.stage]
                    if replica.state != "output":
                        continue
                    if stage.downstream:
                        if any(len(stages[i].pending) >= self._capacity for i in stage.downstream):
                            continue
                        assert replica.payload is not None
                        for index in stage.downstream:
                            stages[index].pending.append(replica.payload)
                    else:
                        assert replica.payload is not None
                        try:
                            self._outputs.put_nowait(replica.payload)
                        except queue.Full:
                            continue
                    produced[stage.node.name] += 1
                    replica.payload = None
                    replica.state = "ready"
                for index, stage in enumerate(stages):
                    group = [r for r in replicas.values() if r.stage == index]
                    if not stage.node.inputs:
                        stage.complete = all(r.state == "done" for r in group)
                    elif (
                        all(stages[indices[parent]].complete for parent in stage.node.inputs)
                        and not stage.pending
                        and all(r.state == "ready" for r in group)
                    ):
                        stage.complete = True
                    peak_queued[stage.node.name] = max(peak_queued[stage.node.name], len(stage.pending))
                with self._lock:
                    self._stats.update(
                        queued={s.node.name: len(s.pending) for s in stages},
                        peak_queued=peak_queued.copy(), produced=produced.copy(),
                        completed_work=completed_work.copy(),
                        peak_output_buffered=max(
                            self._stats["peak_output_buffered"], self._outputs.qsize()
                        ),
                    )
                if all(stage.complete for stage in stages):
                    with self._lock:
                        self._stats["state"] = "completed"
                    return
                if self._outputs.full():
                    if blocked_since is None:
                        blocked_since = now
                    with self._lock:
                        last_consumed = self._last_consumed
                    if now - max(blocked_since, last_consumed) > self._consumer_timeout:
                        raise RunError(
                            f"final consumer stalled for {self._consumer_timeout}s with full "
                            f"output buffer ({self._outputs.maxsize} items); consume or close the run"
                        )
                else:
                    blocked_since = None
                for identity in order:
                    replica = replicas[identity]
                    stage = stages[replica.stage]
                    label = f"stage {stage.node.name!r} rank {replica.rank} work {replica.work}"
                    label += f" machine {replica.machine!r}"
                    if (
                        replica.state in ("starting", "busy")
                        and now - replica.since > self._worker_timeout
                    ):
                        raise RunError(f"{label}: {replica.state} exceeded {self._worker_timeout}s")
                    if replica.process is not None and replica.process.exitcode is not None:
                        # Briefly drain a pending traceback, but never let traffic
                        # from healthy ranks indefinitely hide a dead worker.
                        if replica.exited_since is None:
                            replica.exited_since = now
                        if not socket.poll(0) or now - replica.exited_since >= 0.1:
                            raise RunError(f"{label}: worker exited with code {replica.process.exitcode}")
                        continue
                    if replica.state != "ready" or stage.complete:
                        continue
                    if not stage.node.inputs:
                        command, payload = b"next", b""
                    elif stage.pending:
                        command, payload = b"task", stage.pending.popleft()
                    else:
                        continue
                    replica.work = next_work
                    next_work += 1
                    socket.send_multipart([identity, command, payload])
                    replica.state = "busy"
                    replica.since = time.monotonic()
                if socket.poll(10):
                    identity, kind, payload = socket.recv_multipart()
                    replica = replicas[identity]
                    stage = stages[replica.stage]
                    label = f"stage {stage.node.name!r} rank {replica.rank} work {replica.work}"
                    label += f" machine {replica.machine!r}"
                    if kind == b"error":
                        raise RunError(f"{label}: {payload.decode('utf-8', errors='replace')}")
                    if kind == b"ready" and replica.state == "starting":
                        replica.state = "ready"
                    elif kind == b"done" and replica.state == "busy" and not stage.node.inputs:
                        replica.state = "done"
                    elif kind == b"item" and replica.state == "busy":
                        if len(payload) > self._payload_limit:
                            raise RunError(f"{label}: received payload exceeds limit")
                        replica.payload = payload
                        replica.state = "output"
                        completed_work[stage.node.name] += 1
                    else:
                        raise RunError(f"{label}: unexpected message {kind!r} in {replica.state}")
        except Exception as exc:
            self._record_error(exc)
        finally:
            if not isinstance(socket, RemoteTransport):
                for identity in replicas:
                    try:
                        socket.send_multipart([identity, b"stop", b""], flags=zmq.DONTWAIT)
                    except zmq.ZMQError:
                        pass
                stop_processes([r.process for r in replicas.values() if r.process is not None])
            with self._lock:
                if self._stats["state"] not in ("completed", "failed"):
                    self._stats["state"] = "cancelled"
