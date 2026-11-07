from __future__ import annotations

import threading
import time
from collections import deque

import zmq

from ._protocol import capabilities, configure, decode, encode


class RemoteError(RuntimeError):
    pass


class RemoteCancelled(RemoteError):
    pass


class RemoteTransport:
    """Controller-owned sockets; agent heartbeats never carry payload data."""

    def __init__(
        self, context: zmq.Context, endpoints: dict[str, str], run: str, timeout: float,
        cancel: threading.Event,
    ) -> None:
        self.run = run
        self.timeout = timeout
        self.cancel = cancel
        self.sockets: dict[str, zmq.Socket] = {}
        self.poller = zmq.Poller()
        self.last_seen: dict[str, float] = {}
        self.active: set[str] = set()
        self.routes: dict[bytes, str] = {}
        self.messages: deque[list[bytes]] = deque()
        self.responses: dict[tuple[str, str], dict] = {}
        self.hello: dict[str, dict] = {}
        self.last_sent = 0.0
        self.agent_stats: dict[str, dict] = {}
        self.finals: dict[str, bytes] = {}
        self.endpoints = endpoints.copy()
        try:
            for machine, endpoint in endpoints.items():
                socket = context.socket(zmq.DEALER)
                configure(socket)
                socket.setsockopt(zmq.IDENTITY, run.encode("ascii"))
                self.sockets[machine] = socket
                self.poller.register(socket, zmq.POLLIN)
                socket.connect(endpoint)
                self.last_seen[machine] = time.monotonic()
                self._send(machine, {"type": "hello"})
            self._wait(set(endpoints), "hello", timeout)
            for machine in endpoints:
                response = self.responses.pop((machine, "hello"))
                if response.get("machine") != machine:
                    raise RemoteError(
                        f"agent {machine!r}: duplicate/mismatched machine identity "
                        f"{response.get('machine')!r}"
                    )
                actual = {key: response.get(key) for key in capabilities()}
                if actual != capabilities():
                    raise RemoteError(f"agent {machine!r}: incompatible capabilities: {actual}")
                lease = response.get("lease_timeout")
                if not isinstance(lease, (int, float)) or lease < 0.5:
                    raise RemoteError(f"agent {machine!r}: invalid lease timeout")
                self.hello[machine] = response
        except Exception:
            self.close_sockets()
            raise

    def _send(self, machine: str, message: dict, *frames: bytes) -> None:
        try:
            self.sockets[machine].send_multipart([encode(message), *frames])
        except zmq.ZMQError as exc:
            raise RemoteError(f"agent {machine!r}: connection/send failure: {exc}") from exc

    def _pump(self, milliseconds: int, *, deadlines: bool = True) -> None:
        now = time.monotonic()
        if now - self.last_sent >= 0.1:
            for machine in self.active:
                self._send(machine, {"type": "heartbeat", "run": self.run})
            self.last_sent = now
        events = dict(self.poller.poll(milliseconds))
        for machine, socket in self.sockets.items():
            if socket not in events:
                continue
            frames = socket.recv_multipart()
            message = decode(frames[0])
            kind = message.get("type")
            if kind != "hello" and message.get("run") not in (None, self.run):
                raise RemoteError(f"agent {machine!r}: response from unexpected run")
            self.last_seen[machine] = time.monotonic()
            if kind == "error":
                raise RemoteError(f"agent {machine!r}: {message.get('error')}")
            if kind == "worker":
                identity = message["worker"].encode("ascii")
                if self.routes.get(identity) != machine or len(frames) != 3:
                    raise RemoteError(f"agent {machine!r}: invalid worker response")
                self.messages.append([identity, frames[1], frames[2]])
            elif kind in ("hello", "started", "stopped"):
                self.responses[(machine, kind)] = message
                if kind == "stopped":
                    self.agent_stats[machine] = message
            elif kind == "final":
                if len(frames) != 2:
                    raise RemoteError(f"agent {machine!r}: invalid final payload")
                self.finals[message["output"]] = frames[1]
            elif kind == "heartbeat":
                if "transfers" in message:
                    self.agent_stats[machine] = message
            elif kind != "heartbeat":
                raise RemoteError(f"agent {machine!r}: unexpected response {kind!r}")
        if deadlines:
            now = time.monotonic()
            for machine, seen in self.last_seen.items():
                if now - seen > self.timeout:
                    raise RemoteError(
                        f"agent {machine!r}: unreachable or heartbeat lost for {self.timeout}s"
                    )

    def _wait(self, machines: set[str], kind: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not all((machine, kind) in self.responses for machine in machines):
            if self.cancel.is_set():
                raise RemoteCancelled("agent startup cancelled")
            if time.monotonic() >= deadline:
                raise RemoteError(f"agents {sorted(machines)}: {kind} deadline exceeded ({timeout}s)")
            self._pump(20)

    def start(
        self, specs: dict[str, list[dict]], definitions: dict[bytes, bytes],
        limit: int, timeout: float,
        retained_items: int, retained_bytes: int,
    ) -> dict[bytes, int]:
        for machine, workers in specs.items():
            if len(workers) > self.hello[machine]["max_workers"]:
                raise RemoteError(f"agent {machine!r}: placement exceeds worker capacity")
        for machine, workers in specs.items():
            for worker in workers:
                self.routes[worker["identity"].encode("ascii")] = machine
            # Include agents with a sent start in cleanup even if its ack is lost.
            self.active.add(machine)
            self._send(machine, {
                "type": "start", "run": self.run, "workers": workers,
                "payload_limit": limit, "capabilities": capabilities(),
                "retained_items": retained_items, "retained_bytes": retained_bytes,
                "peers": self.endpoints,
            }, *(definitions[worker["identity"].encode("ascii")] for worker in workers))
        self._wait(set(specs), "started", timeout)
        pids = {}
        for machine in specs:
            response = self.responses.pop((machine, "started"))
            for identity, pid in response["pids"].items():
                key = identity.encode("ascii")
                if self.routes.get(key) != machine:
                    raise RemoteError(f"agent {machine!r}: invalid started worker")
                pids[key] = pid
        if set(pids) != set(self.routes):
            raise RemoteError("agents did not start all required replicas")
        return pids

    def release(self, owner: str, output: str) -> None:
        self._send(owner, {"type": "release", "run": self.run, "output": output})

    def final(self, owner: str, output: str) -> None:
        self._send(owner, {"type": "final", "run": self.run, "output": output})

    def send_multipart(self, frames: list[bytes]) -> None:
        identity, command, payload = frames
        machine = self.routes[identity]
        self._send(machine, {
            "type": "command", "run": self.run, "worker": identity.decode("ascii"),
        }, command, payload)

    def poll(self, milliseconds: int) -> int:
        self._pump(milliseconds)
        return int(bool(self.messages))

    def recv_multipart(self) -> list[bytes]:
        return self.messages.popleft()

    def stop(self) -> None:
        waiting = self.active.copy()
        send_error: RemoteError | None = None
        try:
            for machine in tuple(waiting):
                try:
                    self._send(machine, {"type": "stop", "run": self.run})
                except RemoteError as exc:
                    send_error = exc
                    waiting.remove(machine)
            deadline = time.monotonic() + self.timeout + 1.5
            while waiting and time.monotonic() < deadline:
                # Stop sending leases while awaiting cleanup; disconnected agents
                # still release their children when the last lease expires.
                self.active.clear()
                self._pump(20, deadlines=False)
                waiting = {machine for machine in waiting if (machine, "stopped") not in self.responses}
            if waiting:
                raise RemoteError(f"agents {sorted(waiting)} did not acknowledge owned-worker cleanup")
            if send_error is not None:
                raise send_error
        finally:
            self.close_sockets()

    def close_sockets(self) -> None:
        for socket in self.sockets.values():
            socket.close()
