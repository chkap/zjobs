from __future__ import annotations

import inspect
import os
import threading
import time
import traceback
from typing import Any

import cloudpickle
import zmq

from .graph import SourceContext


def _encode(value: Any, limit: int) -> bytes:
    payload = cloudpickle.dumps(value, protocol=5)
    if len(payload) > limit:
        raise ValueError(f"serialized payload is {len(payload)} bytes; limit is {limit}")
    return payload


def worker_main(
    endpoint: str, identity: bytes, definition: bytes,
    source: bool, rank: int, replicas: int, payload_limit: int,
    machine: str = "local", local_rank: int | None = None,
    local_replicas: int | None = None,
    parent_pid: int | None = None,
) -> None:
    if parent_pid is not None:
        def watch_parent() -> None:
            while True:
                if os.getppid() != parent_pid:
                    os._exit(143)
                time.sleep(0.1)

        threading.Thread(target=watch_parent, name="zjobs-agent-watch", daemon=True).start()
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, identity)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.connect(endpoint)
    operation = "initialize callable"
    try:
        task = cloudpickle.loads(definition)
        if inspect.isclass(task):
            task = task()
        if not callable(task):
            raise TypeError("constructed worker instance is not callable")
        iterator = None
        socket.send_multipart([b"ready", b""])
        while True:
            command, payload = socket.recv_multipart()
            if command == b"stop":
                return
            if command == b"next" and source:
                operation = "source iteration"
                if iterator is None:
                    iterator = iter(task(SourceContext(
                        rank, replicas, machine,
                        rank if local_rank is None else local_rank,
                        replicas if local_replicas is None else local_replicas,
                    )))
                try:
                    value = next(iterator)
                except StopIteration:
                    socket.send_multipart([b"done", b""])
                    continue
            elif command == b"task" and not source:
                operation = "deserialize input"
                value = cloudpickle.loads(payload)
                operation = "execute task"
                value = task(value)
            else:
                raise RuntimeError(f"unexpected worker command {command!r}")
            operation = "serialize output"
            socket.send_multipart([b"item", _encode(value, payload_limit)])
    except BaseException:
        error = f"{operation}:\n{traceback.format_exc()[-32700:]}".encode("utf-8", errors="replace")
        try:
            socket.send_multipart([b"error", error])
            # Especially during initialization, closing a DEALER immediately
            # can discard an error before its asynchronous connection completes.
            if socket.poll(1000):
                socket.recv_multipart()
        except zmq.ZMQError:
            # A disconnected controller detects process exit or its work deadline.
            pass
    finally:
        socket.close()
        context.term()
