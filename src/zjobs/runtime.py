from __future__ import annotations

import copy
import inspect
import queue
import threading
import time
import uuid
from typing import Any, Iterator, Mapping

import cloudpickle
import zmq

from ._protocol import configure
from ._remote import RemoteCancelled, RemoteTransport
from .graph import Node, ReplicaPlacement

_SHIPPING_LOCK = threading.Lock()


def _ship(task: Any) -> bytes:
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
    """Contextual execution, routing, transport or capacity failure."""


class Run(Iterator[Any]):
    def __init__(
        self, nodes: tuple[Node, ...], capacity: int, output_capacity: int,
        worker_timeout: float, consumer_timeout: float, max_payload_bytes: int,
        agents: Mapping[str, str] | None = None, agent_timeout: float = 3.0,
        policy: str = "local_first", retained_items: int = 128,
        retained_bytes: int = 1024 * 1024 * 1024, queue_bytes: int = 256 * 1024 * 1024,
        output_bytes: int = 256 * 1024 * 1024, remote_after: int = 8,
    ) -> None:
        self._nodes = nodes
        self._capacity = capacity
        self._worker_timeout = worker_timeout
        self._consumer_timeout = consumer_timeout
        self._payload_limit = max_payload_bytes
        self._agent_timeout = agent_timeout
        self._retained_items = retained_items
        self._retained_bytes = retained_bytes
        self._queue_bytes = queue_bytes
        self._output_bytes = output_bytes
        self._remote_after = remote_after
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
                        dict(node.rank_policies).get(rank, node.policy or policy),
                    ))
                    rank += 1
        self._placement = tuple(placements)
        for node in nodes:
            for parent in node.inputs:
                for producer in (p for p in placements if p.stage == parent.name):
                    if not any(
                        p.stage == node.name and (p.policy != "local_only" or p.machine == producer.machine)
                        for p in placements
                    ):
                        raise RunError(
                            f"local_only impossible routing: stage {parent.name!r} machine "
                            f"{producer.machine!r} -> stage {node.name!r} has no eligible consumer"
                        )
        self._outputs: queue.Queue[bytes] = queue.Queue(output_capacity)
        self._buffered_bytes = 0
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
            "obligations_created": 0, "obligations_settled": 0, "obligations_pending": 0,
            "local_items": 0, "local_bytes": 0, "remote_items": 0, "remote_bytes": 0,
            "final_bytes": 0, "peak_output_bytes": 0,
        }
        self._definitions = []
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
                self._buffered_bytes -= len(payload)
                self._last_consumed = time.monotonic()
            try:
                return cloudpickle.loads(payload)
            except Exception as exc:
                self.close()
                error = RunError(f"final output: deserialize payload: {exc}")
                self._record_error(error)
                raise error from exc

    def batches(self, size: int) -> Iterator[list[Any]]:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("batch size must be a positive integer")
        batch = []
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
        self._cancel.set()
        self._thread.join()
        with self._lock:
            while True:
                try:
                    self._outputs.get_nowait()
                except queue.Empty:
                    break
            self._buffered_bytes = 0

    @property
    def placement(self) -> tuple[ReplicaPlacement, ...]:
        return self._placement

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._stats)
            result["output_buffered_bytes"] = self._buffered_bytes
        result["output_buffered"] = self._outputs.qsize()
        result["error"] = str(self._error) if self._error else None
        result["placement"] = [
            {
                "stage": p.stage, "machine": p.machine, "rank": p.rank,
                "replicas": p.replicas, "local_rank": p.local_rank,
                "local_replicas": p.local_replicas, "policy": p.policy,
            } for p in self._placement
        ]
        return result

    def _record_error(self, exc: Exception) -> None:
        if isinstance(exc, RemoteCancelled) and self._cancel.is_set():
            with self._lock:
                self._stats["state"] = "cancelled"
            return
        if self._error is None:
            self._error = exc if isinstance(exc, RunError) else RunError(
                f"controller: {type(exc).__name__}: {exc}"
            )
        with self._lock:
            self._stats["state"] = "failed"

    def _drive(self) -> None:
        from ._scheduler import coordinate

        try:
            with zmq.Context() as context:
                if self._agents is not None:
                    used = {p.machine for p in self._placement}
                    remote = RemoteTransport(
                        context, {key: value for key, value in self._agents.items() if key in used},
                        self._run_id, self._agent_timeout, self._cancel,
                    )
                    try:
                        coordinate(self, remote)
                    except Exception as exc:
                        self._record_error(exc)
                    finally:
                        try:
                            remote.stop()
                        finally:
                            with self._lock:
                                self._stats["agents"] = copy.deepcopy(remote.agent_stats)
                else:
                    with context.socket(zmq.ROUTER) as socket:
                        configure(socket)
                        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                        coordinate(self, socket)
        except Exception as exc:
            self._record_error(exc)
        finally:
            with self._lock:
                if self._stats["state"] not in ("completed", "failed"):
                    self._stats["state"] = "cancelled"
            self._finished.set()
