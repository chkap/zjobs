from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

import cloudpickle
import zmq

from ._worker import worker_main
from .graph import Node


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
    process: mp.Process
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
    ) -> None:
        self._nodes = nodes
        self._capacity = capacity
        self._worker_timeout = worker_timeout
        self._consumer_timeout = consumer_timeout
        self._payload_limit = max_payload_bytes
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
        }
        # Fail code shipping before creating any owned processes.
        self._definitions: list[bytes] = []
        for node in nodes:
            try:
                definition = cloudpickle.dumps(node.task, protocol=5)
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
    def status(self) -> dict[str, Any]:
        """Snapshot of controller counts; queues are bounded by item count."""
        with self._lock:
            result = {
                key: value.copy() if isinstance(value, (dict, list)) else value
                for key, value in self._stats.items()
            }
        result["output_buffered"] = self._outputs.qsize()
        result["error"] = str(self._error) if self._error else None
        return result

    def _drive(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self._record_error(exc)
        finally:
            self._finished.set()

    def _record_error(self, exc: Exception) -> None:
        if self._error is None:
            self._error = (
                exc if isinstance(exc, RunError)
                else RunError(f"controller: {type(exc).__name__}: {exc}")
            )
        with self._lock:
            self._stats["state"] = "failed"

    def _execute(self) -> None:
        with zmq.Context() as context:
            with context.socket(zmq.ROUTER) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDHWM, 2)
                socket.setsockopt(zmq.RCVHWM, 2)
                socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
                socket.setsockopt(zmq.SNDTIMEO, 1000)
                self._coordinate(socket)

    def _coordinate(self, socket: zmq.Socket) -> None:
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
            port = socket.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"
            spawn = mp.get_context("spawn")
            for index, stage in enumerate(stages):
                for rank in range(stage.node.replicas):
                    if self._cancel.is_set():
                        return
                    identity = f"{index}:{rank}".encode("ascii")
                    process = spawn.Process(
                        target=worker_main,
                        args=(
                            endpoint, identity, self._definitions[index],
                            not stage.node.inputs, rank, stage.node.replicas,
                            self._payload_limit,
                        ),
                        name=f"zjobs-{stage.node.name}-{rank}",
                    )
                    process.start()
                    replicas[identity] = _Replica(index, rank, process)
                    with self._lock:
                        self._stats["worker_pids"].append(process.pid)
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
                    if (
                        replica.state in ("starting", "busy")
                        and now - replica.since > self._worker_timeout
                    ):
                        raise RunError(f"{label}: {replica.state} exceeded {self._worker_timeout}s")
                    if replica.process.exitcode is not None:
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
            # Global grace periods avoid multiplying shutdown time by rank count.
            for identity in replicas:
                try:
                    socket.send_multipart([identity, b"stop", b""], flags=zmq.DONTWAIT)
                except zmq.ZMQError:
                    pass
            deadline = time.monotonic() + 0.5
            for replica in replicas.values():
                replica.process.join(max(0, deadline - time.monotonic()))
            for replica in replicas.values():
                if replica.process.is_alive():
                    replica.process.terminate()
            deadline = time.monotonic() + 0.5
            for replica in replicas.values():
                replica.process.join(max(0, deadline - time.monotonic()))
            for replica in replicas.values():
                if replica.process.is_alive():
                    replica.process.kill()
                replica.process.join()
                replica.process.close()
            with self._lock:
                if self._stats["state"] not in ("completed", "failed"):
                    self._stats["state"] = "cancelled"
