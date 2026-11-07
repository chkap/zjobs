from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import copy
import json
import multiprocessing as mp
import threading
import time
import uuid
from typing import Any

import cloudpickle
import zmq

from . import _payload
from ._selection import Selection
from ._worker import worker
from .graph import Graph, Limits, Locality, PeerConfig, SourceContext, Stage


class RunError(RuntimeError):
    pass


@dataclass
class Output:
    stage: str
    frames: _payload.Payload
    obligations: dict[str, tuple[str, str, float] | None]
    queued: bool = False


@dataclass
class Input:
    owner: str
    output: str
    parent: str
    child: str
    token: str
    size: int
    frames: _payload.Payload = ()


@dataclass
class Slot:
    stage: Stage
    rank: int
    socket: Any
    process: Any
    started: float
    ready: bool = False
    eof: bool = False
    reserved: bool = False
    task: str = ""
    task_started: float = 0
    retries: int = 0
    pending: Input | None = None
    inputs: list[Input] = field(default_factory=list)
    selection: Selection = field(default_factory=Selection)


class Run:
    """Own one graph's workers, output store, peer mesh, and local iterator."""

    def __init__(self, graph: Graph, peer: PeerConfig, limits: Limits):
        self.graph = cloudpickle.loads(cloudpickle.dumps(graph))
        self.peer, self.limits = peer, limits
        for s in self.graph.stages:
            if s.kind != "source" and (
                limits.claim_items < s.replicas or limits.claim_bytes < s.replicas
            ):
                raise ValueError(f"{s.name}: claim limits must admit each local replica")
        self._condition = threading.Condition()
        self._outputs: deque[tuple[str, _payload.Payload]] = deque()
        self._consumed: deque[str] = deque()
        self._output_bytes = 0
        self._last_consume = time.monotonic()
        self._error: RunError | None = None
        self._iteration_done = False
        self._closing = False
        self._cancel = False
        self._status: dict[str, Any] = {}
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._main, name=f"zjobs-{peer.identity}")
        self._thread.start()

    @property
    def status(self) -> dict[str, Any]:
        with self._condition:
            return copy.deepcopy(self._status)

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        with self._condition:
            while not self._outputs and not self._error and not self._iteration_done:
                self._condition.wait(.1)
            if self._error:
                raise self._error
            if not self._outputs:
                raise StopIteration
            identity, frames = self._outputs.popleft()
            self._output_bytes -= _payload.size(frames)
            self._last_consume = time.monotonic()
        try:
            value = _payload.loads(frames)
        except Exception as exc:
            self._cancel = True
            raise RunError(f"{self.peer.identity}: local consumer decode failed: {exc}") from exc
        with self._condition:
            self._consumed.append(identity)
            self._condition.notify_all()
        return value

    def close(self):
        if not self._iteration_done and not self._finished.is_set():
            self._cancel = True
        self._closing = True
        self._thread.join(self.limits.shutdown_timeout + 2)
        if self._thread.is_alive():
            raise RunError(f"{self.peer.identity}: runtime shutdown exceeded deadline")
        if self._error:
            raise self._error

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except RunError:
            if exc_type is None:
                raise
        return False

    def _fail(self, message):
        raise RunError(f"{self.peer.identity}: {message}")

    def _setup(self):
        self._context = zmq.Context()
        self._incarnation = uuid.uuid4().hex
        self._members = sorted(self.peer.peers) or [self.peer.identity]
        self._others = [p for p in self._members if p != self.peer.identity]
        self._key = [1, self.peer.run_id, self.graph.fingerprint(),
                     [list(v) for v in sorted(self.peer.peers.items())]]
        self._router = None
        self._sockets = {}
        self._incarnations: dict[str, str] = {}
        self._seen: dict[str, float] = {}
        self._states: dict[str, dict] = {}
        self._sequence = 0
        self._ready = False
        self._bye = False
        self._seen_ready: set[str] = set()
        self._seen_bye: set[str] = set()
        self._store: dict[str, Output] = {}
        self._slots: list[Slot] = []
        self._sealed: set[str] = set()
        self._counter = 0
        self._produced = {s.name: 0 for s in self.graph.stages}
        self._rank_cursor = {s.name: 0 for s in self.graph.stages}
        self._usage = {s.name: [0, 0] for s in self.graph.stages}
        self._peaks = {s.name: [0, 0] for s in self.graph.stages}
        self._claim_peaks = {s.name: [0, 0] for s in self.graph.stages}
        self._children = {s.name: [c.name for c in self.graph.stages if s in c.inputs]
                          for s in self.graph.stages}
        self._claim_queue: list[tuple[str, dict]] = []
        self._grant_selection = {s.name: Selection() for s in self.graph.stages}
        self._denied: set[tuple[str, str, str]] = set()
        self._metrics = {
            "obligations_created": 0, "obligations_acked": 0,
            "local_items": {}, "peer_items": {}, "local_bytes": {}, "peer_bytes": {},
            "retries": 0, "stale_messages": 0, "renewals": 0,
            "peak_output_items": 0, "peak_output_bytes": 0,
        }
        self._all_pids: list[int] = []
        if self._others:
            self._router = self._context.socket(zmq.ROUTER)
            self._configure(self._router)
            self._router.bind(self.peer.listen)
            for p in self._others:
                sock = self._context.socket(zmq.DEALER)
                self._configure(sock)
                sock.setsockopt(zmq.IDENTITY, self.peer.identity.encode())
                sock.connect(self.peer.peers[p])
                self._sockets[p] = sock
        for stage in self.graph.stages:
            for rank in range(stage.replicas):
                self._slots.append(self._spawn(stage, rank))

    def _configure(self, sock):
        sock.setsockopt(zmq.LINGER, 0)
        # Queued control and payload messages are finite, independent per peer.
        hwm = max(16, self.limits.claim_items + self.limits.retained_items *
                  max(1, len(self.graph.stages)) + len(self.graph.stages) * 4)
        sock.setsockopt(zmq.SNDHWM, hwm)
        sock.setsockopt(zmq.RCVHWM, hwm)
        sock.setsockopt(zmq.MAXMSGSIZE, max(1024 * 1024, self.limits.max_payload_bytes))

    def _spawn(self, stage, rank):
        sock = self._context.socket(zmq.PAIR)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 1)
        sock.setsockopt(zmq.RCVHWM, 1)
        sock.setsockopt(zmq.MAXMSGSIZE, max(65536, self.limits.max_payload_bytes))
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        source_context = SourceContext(self.peer.identity, rank, stage.replicas,
                                       tuple(self._members))
        definition = cloudpickle.dumps(stage.task) if stage.task is not None else None
        process = mp.get_context("spawn").Process(
            target=worker, args=(f"tcp://127.0.0.1:{port}", definition, source_context,
                                 self.limits.max_payload_bytes, stage.kind),
            name=f"zjobs-{self.peer.identity}-{stage.name}-{rank}",
        )
        try:
            process.start()
        except BaseException:
            sock.close()
            raise
        self._all_pids.append(process.pid)
        return Slot(stage, rank, sock, process, time.monotonic())

    def _send(self, recipient, kind, body, frames=(), *, optional=False):
        if recipient == self.peer.identity:
            self._dispatch(recipient, kind, body, frames)
            return
        header = {
            "key": self._key, "sender": self.peer.identity,
            "incarnation": self._incarnation, "kind": kind, "body": body,
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        if len(encoded) > 1024 * 1024:
            self._fail("control snapshot exceeds 1 MiB; reduce graph/retention size")
        try:
            self._sockets[recipient].send_multipart([encoded, *frames], zmq.DONTWAIT)
        except zmq.Again:
            if not optional:
                self._fail(f"peer {recipient}: bounded transport queue congested during {kind}")

    def _poll_peers(self):
        if self._router is None:
            return
        # Bound processing per turn so active worker/claim maintenance cannot starve.
        for _ in range(256):
            try:
                message = self._router.recv_multipart(zmq.DONTWAIT)
            except zmq.Again:
                break
            header = json.loads(message[1])
            sender = header["sender"]
            if sender not in self._sockets or message[0] != sender.encode():
                self._fail("message from unconfigured peer")
            if header["key"] != self._key:
                self._fail(f"peer {sender}: incompatible run, graph, or membership")
            incarnation = header["incarnation"]
            previous = self._incarnations.setdefault(sender, incarnation)
            if previous != incarnation:
                self._fail(f"peer {sender}: active-run incarnation replacement is unsupported")
            self._seen[sender] = time.monotonic()
            self._dispatch(sender, header["kind"], header["body"], tuple(message[2:]))

    def _dispatch(self, sender, kind, body, frames):
        if kind == "state":
            previous = self._states.get(sender)
            if previous and body["sequence"] <= previous["sequence"]:
                self._metrics["stale_messages"] += 1
                return
            if previous:
                if not set(previous["sealed"]).issubset(body["sealed"]):
                    self._fail(f"peer {sender}: regressed production seal")
                if previous["ready"] and not body["ready"]:
                    self._fail(f"peer {sender}: regressed completion")
                if any(previous["watermarks"][name] != body["watermarks"][name]
                       for name in previous["sealed"]):
                    self._fail(f"peer {sender}: changed sealed production watermark")
            self._states[sender] = body
            self._denied = {v for v in self._denied if v[0] != sender}
            if body["ready"]:
                self._seen_ready.add(sender)
            if body["bye"]:
                self._seen_bye.add(sender)
        elif kind == "claim":
            if len(self._claim_queue) >= self._request_bound():
                self._fail("bounded owner claim arbitration queue exceeded")
            self._claim_queue.append((sender, body))
        elif kind in ("grant", "reject"):
            self._receive_claim(sender, kind, body, frames)
        elif kind in ("ack", "renew"):
            output = self._store.get(body["output"])
            claim = output.obligations.get(body["child"]) if output else None
            if not claim or claim[:2] != (sender, body["token"]):
                self._metrics["stale_messages"] += 1
                return
            if time.monotonic() > claim[2]:
                self._fail(f"{output.stage}->{body['child']}: expired claim cannot be renewed/ACKed")
            if kind == "renew":
                output.obligations[body["child"]] = (
                    sender, body["token"], time.monotonic() + self.peer.lease_timeout)
            else:
                del output.obligations[body["child"]]
                self._metrics["obligations_acked"] += 1
                if not output.obligations:
                    self._retire(body["output"])
        elif kind == "fail":
            self._fail(f"peer {sender} failed: {body['error']}")
        else:
            self._fail(f"peer {sender}: unknown protocol message {kind}")

    def _request_bound(self):
        counts = sum(s.replicas for s in self.graph.stages)
        counts += sum(sum(s["replicas"].values()) for s in self._states.values())
        return max(1, counts * 2)

    def _snapshot(self):
        offers = {s.name: [] for s in self.graph.stages}
        outstanding = {s.name: {c: 0 for c in self._children[s.name]}
                       for s in self.graph.stages}
        for identity, output in self._store.items():
            edges = [c for c, claim in output.obligations.items()
                     if claim is None and c != "@consumer"]
            if edges:
                offers[output.stage].append([identity, _payload.size(output.frames), edges])
            for c in output.obligations:
                if c != "@consumer":
                    outstanding[output.stage][c] += 1
        return {
            "sequence": self._sequence, "sealed": sorted(self._sealed),
            "watermarks": {name: self._produced[name] for name in self._sealed},
            "offers": offers, "outstanding": outstanding,
            "replicas": {s.name: s.replicas for s in self.graph.stages},
            "pids": {s.name: [slot.process.pid for slot in self._slots if slot.stage is s]
                     for s in self.graph.stages},
            "receivers": {s.name: [
                {"mode": self.graph.policy(s, rank).mode,
                 "bytes": min(self.limits.max_payload_bytes,
                              self.limits.claim_bytes // s.replicas,
                              s.max_bytes if s.kind == "batch" else self.limits.claim_bytes)}
                for rank in range(s.replicas)] for s in self.graph.stages},
            "blocked": {s.name: (
                self._usage[s.name][0] >= self.limits.retained_items or
                self._usage[s.name][1] + self.limits.max_payload_bytes >
                self.limits.retained_bytes) for s in self.graph.stages},
            "ready": self._ready, "seen_ready": sorted(self._seen_ready),
            "bye": self._bye, "seen_bye": sorted(self._seen_bye),
        }

    def _states_all(self):
        return {self.peer.identity: self._snapshot(), **self._states}

    def _parents_sealed(self, stage):
        return all(p.name in state["sealed"] for state in self._states_all().values()
                   for p in stage.inputs) and len(self._states) == len(self._others)

    def _parents_drained(self, stage):
        return self._parents_sealed(stage) and all(
            state["outstanding"][p.name][stage.name] == 0
            for state in self._states_all().values() for p in stage.inputs)

    def _reserve(self, slot):
        if slot.reserved:
            return True
        items, size = self._usage[slot.stage.name]
        if (items >= self.limits.retained_items or
                size + self.limits.max_payload_bytes > self.limits.retained_bytes):
            return False
        self._usage[slot.stage.name] = [items + 1, size + self.limits.max_payload_bytes]
        slot.reserved = True
        self._rank_cursor[slot.stage.name] = (slot.rank + 1) % slot.stage.replicas
        self._peak(slot.stage.name)
        return True

    def _peak(self, name):
        self._peaks[name] = [max(a, b) for a, b in zip(self._peaks[name], self._usage[name])]

    def _release_reservation(self, slot):
        if slot.reserved:
            self._usage[slot.stage.name][0] -= 1
            self._usage[slot.stage.name][1] -= self.limits.max_payload_bytes
            slot.reserved = False

    def _publish(self, slot, frames):
        if not slot.reserved or slot.stage.name in self._sealed:
            self._fail(f"{slot.stage.name}/{slot.rank}: result without live production reservation")
        size = _payload.size(frames)
        if len(frames) > _payload.MAX_FRAMES or size > self.limits.max_payload_bytes:
            self._fail(f"{slot.stage.name}/{slot.rank}: oversized worker result")
        self._counter += 1
        self._produced[slot.stage.name] += 1
        identity = f"{self._incarnation}:{slot.stage.name}:{slot.rank}:{self._counter}"
        children = self._children[slot.stage.name] or ["@consumer"]
        for child in children:
            if child == "@consumer":
                continue
            eligible = [
                receiver
                for owner, state in self._states_all().items()
                for receiver in state["receivers"][child]
                if owner == self.peer.identity or receiver["mode"] != "local_only"
            ]
            if not any(receiver["bytes"] >= size for receiver in eligible):
                self._fail(f"{slot.stage.name}->{child}: required {size}-byte output "
                           "has no eligible receiver with sufficient byte capacity")
        self._store[identity] = Output(slot.stage.name, frames, dict.fromkeys(children))
        self._usage[slot.stage.name][1] -= self.limits.max_payload_bytes - size
        slot.reserved = False
        self._metrics["obligations_created"] += len(children)
        # This is the commit boundary: only retained output permits input ACK.
        for inp in slot.inputs:
            self._send(inp.owner, "ack", {
                "output": inp.output, "child": inp.child, "token": inp.token})
        slot.inputs.clear()
        slot.retries = 0

    def _retire(self, identity):
        output = self._store.pop(identity)
        self._usage[output.stage][0] -= 1
        self._usage[output.stage][1] -= _payload.size(output.frames)

    def _receive_claim(self, sender, kind, body, frames):
        slot = next((s for s in self._slots
                     if s.pending and s.pending.token == body["token"]), None)
        if slot is None:
            self._metrics["stale_messages"] += 1
            return
        inp = slot.pending
        if sender != inp.owner or body["output"] != inp.output or body["child"] != inp.child:
            self._fail("claim grant does not match reserved candidate")
        slot.pending = None
        if kind == "reject":
            self._denied.add((sender, inp.output, inp.child))
            slot.selection.rejected()
            return
        if _payload.size(frames) != inp.size or not 2 <= len(frames) <= _payload.MAX_FRAMES:
            self._fail("grant payload disagrees with concrete byte reservation")
        inp.frames = tuple(frames)
        slot.inputs.append(inp)
        slot.selection.admitted(sender, self.peer.identity)
        edge = f"{inp.parent}->{inp.child}"
        prefix = "local" if sender == self.peer.identity else "peer"
        for suffix, amount in (("items", 1), ("bytes", inp.size)):
            key = f"{prefix}_{suffix}"
            self._metrics[key][edge] = self._metrics[key].get(edge, 0) + amount

    def _arbitrate(self):
        while self._claim_queue:
            child = self._claim_queue[0][1]["child"]
            choices = list({p for p, b in self._claim_queue if b["child"] == child})
            selector = self._grant_selection[child]
            policy = self.graph.locality
            if policy.mode == "local_only":
                policy = Locality(max_local_streak=policy.max_local_streak)
            owner = selector.choose(choices, self.peer.identity, policy)
            index = next(i for i, (p, b) in enumerate(self._claim_queue)
                         if p == owner and b["child"] == child)
            sender, body = self._claim_queue.pop(index)
            output = self._store.get(body["output"])
            if (output is None or child not in output.obligations or
                    output.obligations[child] is not None):
                self._send(sender, "reject", body)
                selector.rejected()
                continue
            if _payload.size(output.frames) != body["size"]:
                self._fail(f"peer {sender}: incorrect claim size")
            receiver = self._states_all()[sender]["receivers"][child][body["rank"]]
            if (body["size"] > receiver["bytes"] or
                    (sender != self.peer.identity and receiver["mode"] == "local_only")):
                self._fail(f"peer {sender}: claim violates receiver capacity/locality")
            output.obligations[child] = (
                sender, body["token"], time.monotonic() + self.peer.lease_timeout)
            self._send(sender, "grant", body, output.frames)
            selector.admitted(sender, self.peer.identity)

    def _candidate(self, slot):
        stage = slot.stage
        selected = {
            (i.owner, i.output, i.child)
            for s in self._slots for i in [*s.inputs, *([s.pending] if s.pending else [])]
        }
        candidates: dict[str, list] = {}
        _, byte_cap = self._input_capacity(slot)
        byte_cap = min(byte_cap, self.limits.max_payload_bytes)
        if stage.kind == "batch":
            byte_cap = min(byte_cap, stage.max_bytes)
        for owner, state in self._states_all().items():
            for parent in stage.inputs:
                for identity, size, edges in state["offers"][parent.name]:
                    key = (owner, identity, stage.name)
                    if (stage.name in edges and key not in selected and
                            key not in self._denied and size <= byte_cap):
                        candidates.setdefault(owner, []).append((parent.name, identity, size))
        owner = slot.selection.choose(list(candidates), self.peer.identity,
                                       self.graph.policy(stage, slot.rank))
        if owner is None:
            return None
        parent, identity, size = candidates[owner][0]
        return Input(owner, identity, parent, stage.name, uuid.uuid4().hex, size)

    def _input_capacity(self, slot):
        return (self.limits.claim_items // slot.stage.replicas,
                self.limits.claim_bytes // slot.stage.replicas)

    def _command(self, slot):
        slot.task = uuid.uuid4().hex
        slot.task_started = time.monotonic()
        if slot.stage.kind == "batch":
            frames = _payload.group_frames([i.frames for i in slot.inputs])
        else:
            frames = slot.inputs[0].frames if slot.inputs else ()
        slot.socket.send_multipart([
            json.dumps({"kind": "run", "token": slot.task}).encode(), *frames], zmq.DONTWAIT)

    def _schedule(self):
        ordered = [
            slot for stage in self.graph.stages
            for slot in sorted(
                (s for s in self._slots if s.stage is stage),
                key=lambda s: (s.rank - self._rank_cursor[stage.name]) % stage.replicas)
        ]
        for slot in ordered:
            if not slot.ready or slot.eof or slot.task or slot.pending:
                continue
            if slot.stage.name in self._sealed:
                continue
            if slot.stage.kind == "source":
                if self._reserve(slot):
                    self._command(slot)
                continue
            if not self._reserve(slot):
                continue
            item_cap, byte_cap = self._input_capacity(slot)
            group_size = sum(i.size for i in slot.inputs)
            inp = self._candidate(slot)
            if slot.stage.kind == "batch":
                byte_cap = min(byte_cap, slot.stage.max_bytes)
                full = len(slot.inputs) >= min(slot.stage.size, item_cap)
                sealed_partial = self._parents_sealed(slot.stage) and inp is None
                pressure_partial = (inp is None and any(
                    state["blocked"][p.name] for state in self._states_all().values()
                    for p in slot.stage.inputs))
                if slot.inputs and (full or sealed_partial or pressure_partial):
                    self._command(slot)
                    continue
            elif slot.inputs:
                self._command(slot)
                continue
            if inp is None:
                if not slot.inputs:
                    self._release_reservation(slot)
                continue
            if inp.size > byte_cap:
                self._fail(f"{slot.stage.name}/{slot.rank}: input {inp.output} ({inp.size} bytes) "
                           f"exceeds per-replica claim/batch capacity {byte_cap}")
            if group_size + inp.size > byte_cap:
                self._command(slot)
                continue
            slot.pending = inp
            # Peaks include pending (pre-transfer) reservations.
            stage_usage = [
                sum(len(s.inputs) + bool(s.pending) for s in self._slots if s.stage is slot.stage),
                sum(sum(i.size for i in s.inputs) + (s.pending.size if s.pending else 0)
                    for s in self._slots if s.stage is slot.stage),
            ]
            self._claim_peaks[slot.stage.name] = [
                max(a, b) for a, b in zip(self._claim_peaks[slot.stage.name], stage_usage)]
            self._send(inp.owner, "claim", {
                "output": inp.output, "child": inp.child, "token": inp.token,
                "size": inp.size,
                "rank": slot.rank,
            })

    def _worker_messages(self):
        for slot in self._slots:
            try:
                frames = slot.socket.recv_multipart(zmq.DONTWAIT)
            except zmq.Again:
                continue
            header = json.loads(frames[0])
            if header["kind"] == "ready":
                slot.ready = True
                continue
            if not slot.task or header.get("token") != slot.task:
                if header["kind"] == "error" and not slot.ready:
                    self._fail(f"{slot.stage.name}/{slot.rank}: initialization: {header['error']}")
                self._metrics["stale_messages"] += 1
                continue
            slot.task = ""
            if header["kind"] == "error":
                self._fail(f"{slot.stage.name}/{slot.rank}: task error: {header['error']}")
            elif header["kind"] == "eof":
                if slot.stage.kind != "source":
                    self._fail("non-source worker emitted EOF")
                slot.eof = True
                self._release_reservation(slot)
            elif header["kind"] == "result":
                self._publish(slot, tuple(frames[1:]))
            else:
                self._fail(f"unknown worker response {header['kind']}")

    def _deadlines(self, now):
        if self._cancel:
            self._fail("local consumer cancelled/abandoned its stream")
        for peer in self._others:
            if peer in self._seen and now - self._seen[peer] > self.peer.heartbeat_timeout:
                self._fail(f"peer {peer}: heartbeat deadline exceeded")
        for slot in list(self._slots):
            if slot.eof:
                continue
            expired = ((not slot.ready and now - slot.started > self.limits.task_timeout) or
                       (slot.task and now - slot.task_started > self.limits.task_timeout))
            if not slot.process.is_alive() or expired:
                reason = "task/initialization stalled" if expired else "worker crashed"
                if (slot.stage.kind != "source" and slot.task and
                        slot.retries < self.limits.max_retries):
                    self._stop_worker(slot)
                    replacement = self._spawn(slot.stage, slot.rank)
                    replacement.reserved = slot.reserved
                    replacement.inputs = slot.inputs
                    replacement.retries = slot.retries + 1
                    replacement.selection = slot.selection
                    self._slots[self._slots.index(slot)] = replacement
                    self._metrics["retries"] += 1
                else:
                    detail = ("source replay is unsupported" if slot.stage.kind == "source"
                              else "retries exhausted")
                    self._fail(f"{slot.stage.name}/{slot.rank}: {reason}; {detail}")
        for output in self._store.values():
            for child, claim in output.obligations.items():
                if claim and now > claim[2]:
                    self._fail(f"{output.stage}->{child}: active claim lease expired; "
                               "run failed rather than overlapping committed attempts")
        with self._condition:
            if self._outputs and now - self._last_consume > self.limits.consumer_timeout:
                self._fail("local consumer stalled beyond consumer_timeout")

    def _renew(self):
        for slot in self._slots:
            for inp in [*slot.inputs, *([slot.pending] if slot.pending else [])]:
                self._send(inp.owner, "renew", {
                    "output": inp.output, "child": inp.child, "token": inp.token})
                self._metrics["renewals"] += 1

    def _consume(self):
        with self._condition:
            while self._consumed:
                identity = self._consumed.popleft()
                output = self._store.get(identity)
                if output is None:
                    self._fail("local consumer acknowledged unknown output")
                self._metrics["obligations_acked"] += 1
                self._retire(identity)
            for identity, output in self._store.items():
                if "@consumer" not in output.obligations or output.queued:
                    continue
                size = _payload.size(output.frames)
                if (len(self._outputs) >= self.limits.output_items or
                        self._output_bytes + size > self.limits.output_bytes):
                    break
                if not self._outputs:
                    self._last_consume = time.monotonic()
                output.queued = True
                self._outputs.append((identity, output.frames))
                self._output_bytes += size
                self._metrics["peak_output_items"] = max(
                    self._metrics["peak_output_items"], len(self._outputs))
                self._metrics["peak_output_bytes"] = max(
                    self._metrics["peak_output_bytes"], self._output_bytes)
            self._condition.notify_all()

    def _completion(self):
        for stage in self.graph.stages:
            slots = [s for s in self._slots if s.stage is stage]
            if stage.kind == "source":
                done = all(s.eof for s in slots)
            else:
                done = self._parents_drained(stage) and all(
                    not s.inputs and not s.pending and not s.task for s in slots)
            if done:
                for slot in slots:
                    self._release_reservation(slot)
                self._sealed.add(stage.name)
        if len(self._sealed) == len(self.graph.stages) and not self._store:
            self._ready = True
            self._seen_ready.add(self.peer.identity)
        states = self._states_all()
        if (self._ready and len(states) == len(self._members) and
                all(s["ready"] and self.peer.identity in s["seen_ready"] for s in states.values())):
            with self._condition:
                self._iteration_done = True
                self._condition.notify_all()
        if self._iteration_done and self._closing:
            self._bye = True
            self._seen_bye.add(self.peer.identity)
        return (self._bye and all(s["bye"] and self.peer.identity in s["seen_bye"]
                                 for s in self._states_all().values()))

    def _report(self):
        stages = {}
        for stage in self.graph.stages:
            slots = [s for s in self._slots if s.stage is stage]
            stages[stage.name] = {
                "pids": [s.process.pid for s in slots], "replicas": stage.replicas,
                "policies": [
                    {"mode": self.graph.policy(stage, s.rank).mode,
                     "max_local_streak": self.graph.policy(stage, s.rank).max_local_streak}
                    for s in slots],
                "ready": sum(s.ready for s in slots),
                "in_flight": sum(bool(s.task) for s in slots),
                "pending": sum(bool(s.pending) for s in slots),
                "claimed": sum(len(s.inputs) for s in slots),
                "retained_items": self._usage[stage.name][0],
                "retained_bytes": self._usage[stage.name][1],
                "peak_retained_items": self._peaks[stage.name][0],
                "peak_retained_bytes": self._peaks[stage.name][1],
                "peak_claim_items": self._claim_peaks[stage.name][0],
                "peak_claim_bytes": self._claim_peaks[stage.name][1],
                "production_sealed": stage.name in self._sealed,
                "produced": self._produced[stage.name],
                "production_watermark": (self._produced[stage.name]
                                         if stage.name in self._sealed else None),
            }
        with self._condition:
            self._status = {
                "peer": self.peer.identity, "incarnation": self._incarnation,
                "members": self._members, "stages": stages,
                "peers": {p: {"incarnation": self._incarnations.get(p),
                              "ready": s["ready"], "production_sealed": s["sealed"],
                              "production_watermarks": s["watermarks"],
                              "replicas": s["replicas"], "pids": s["pids"],
                              "available": {name: len(offers) for name, offers in s["offers"].items()},
                              "outstanding": s["outstanding"],
                              "heartbeat_age": time.monotonic() - self._seen[p]}
                          for p, s in self._states.items()},
                **copy.deepcopy(self._metrics),
                "output_items": len(self._outputs), "output_bytes": self._output_bytes,
                "complete_ready": self._ready, "iteration_done": self._iteration_done,
                "error": str(self._error) if self._error else None,
                "worker_pids": list(self._all_pids),
            }

    def _stop_worker(self, slot):
        if slot.process.is_alive():
            try:
                slot.socket.send_multipart([
                    json.dumps({"kind": "stop", "token": ""}).encode()], zmq.DONTWAIT)
            except zmq.Again:
                pass
            slot.process.join(.05)
            if slot.process.is_alive():
                slot.process.terminate()
        slot.process.join(.5)
        if slot.process.is_alive():
            slot.process.kill()
            slot.process.join(.5)
        slot.socket.close()

    def _cleanup_workers(self):
        slots = getattr(self, "_slots", [])
        for slot in slots:
            if slot.process.is_alive():
                try:
                    slot.socket.send_multipart([
                        json.dumps({"kind": "stop", "token": ""}).encode()], zmq.DONTWAIT)
                except zmq.Again:
                    pass
        for action, duration in (("stop", .2), ("terminate", .5), ("kill", .5)):
            if action != "stop":
                for slot in slots:
                    if slot.process.is_alive():
                        getattr(slot.process, action)()
            deadline = time.monotonic() + duration
            while any(slot.process.is_alive() for slot in slots) and time.monotonic() < deadline:
                time.sleep(.01)
        for slot in slots:
            slot.process.join(0)
            slot.socket.close()
            slot.inputs.clear()
            slot.pending = None
            slot.task = ""
            slot.reserved = False
        if any(slot.process.is_alive() for slot in slots):
            with self._condition:
                self._error = RunError(f"{self.peer.identity}: worker cleanup deadline exceeded")
                self._condition.notify_all()

    def _main(self):
        started = time.monotonic()
        closing_started = None
        linger_started = None
        last_state = last_renew = 0
        try:
            self._setup()
            while True:
                now = time.monotonic()
                self._poll_peers()
                self._worker_messages()
                joined = len(self._states) == len(self._others)
                if not joined and now - started > self.peer.join_timeout:
                    self._fail(f"join deadline exceeded; missing {set(self._others) - self._states.keys()}")
                self._deadlines(now)
                self._consume()
                if joined:
                    self._arbitrate()
                    self._schedule()
                    shutdown_agreed = self._completion()
                else:
                    shutdown_agreed = False
                if now - last_renew >= min(.5, self.peer.lease_timeout / 3):
                    self._renew()
                    last_renew = now
                if now - last_state >= .05:
                    self._sequence += 1
                    state = self._snapshot()
                    for p in self._others:
                        self._send(p, "state", state, optional=True)
                    last_state = now
                if self._closing:
                    closing_started = closing_started or now
                    if now - closing_started > self.limits.shutdown_timeout:
                        self._fail("symmetric shutdown acknowledgement deadline exceeded")
                if shutdown_agreed:
                    linger_started = linger_started or now
                    # Keep responding and retransmitting after the last ACK;
                    # no first observer is permitted to immediately close.
                    if now - linger_started >= .25:
                        break
                self._report()
                time.sleep(.002)
        except BaseException as exc:
            error = exc if isinstance(exc, RunError) else RunError(
                f"{self.peer.identity}: runtime failure: {type(exc).__name__}: {exc}")
            with self._condition:
                self._error = error
                self._condition.notify_all()
            if hasattr(self, "_sockets"):
                for p in self._others:
                    try:
                        self._send(p, "fail", {"error": str(error)}, optional=True)
                    except Exception:
                        # The original contextual failure remains visible; peers
                        # independently fail on their heartbeat deadline.
                        pass
        finally:
            self._cleanup_workers()
            if hasattr(self, "_store"):
                self._store.clear()
                for name in self._usage:
                    self._usage[name] = [0, 0]
                with self._condition:
                    self._outputs.clear()
                    self._output_bytes = 0
                self._report()
            for sock in getattr(self, "_sockets", {}).values():
                sock.close()
            if getattr(self, "_router", None) is not None:
                self._router.close()
            if hasattr(self, "_context"):
                self._context.term()
            self._finished.set()
