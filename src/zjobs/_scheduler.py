from __future__ import annotations

import copy
import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import zmq

from ._protocol import decode, encode, stop_processes
from ._remote import RemoteTransport
from ._worker import worker_main
from .graph import ReplicaPlacement
from .runtime import RunError

if TYPE_CHECKING:
    from .runtime import Run


@dataclass
class Output:
    key: str
    owner: str
    size: int
    producer: bytes
    edges: set[int]
    published: bool = False
    final_requested: bool = False


@dataclass
class Replica:
    placement: ReplicaPlacement
    stage: int
    process: mp.Process | None = None
    state: str = "starting"
    since: float = field(default_factory=time.monotonic)
    input: Output | None = None
    output: Output | None = None
    work: int | None = None
    local_streak: int = 0
    exited_since: float | None = None


def choose(
    pending: deque[Output], machine: str, policy: str, streak: int, remote_after: int,
) -> Output | None:
    eligible = [item for item in pending if policy != "local_only" or item.owner == machine]
    if not eligible:
        return None
    if policy == "balanced":
        return eligible[0]
    local = [item for item in eligible if item.owner == machine]
    remote = [item for item in eligible if item.owner != machine]
    if remote and (not local or streak >= remote_after):
        return remote[0]
    return local[0] if local else eligible[0]


def coordinate(run: Run, transport: zmq.Socket | RemoteTransport) -> None:
    nodes = run._nodes
    indices = {node: i for i, node in enumerate(nodes)}
    by_name = {node.name: i for i, node in enumerate(nodes)}
    downstream = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        for parent in node.inputs:
            downstream[indices[parent]].append(index)
    pending: list[deque[Output]] = [deque() for _ in nodes]
    complete = [False for _ in nodes]
    replicas: dict[bytes, Replica] = {}
    outputs: dict[str, Output] = {}
    local_payloads: dict[str, bytes] = {}
    used_items = {p.machine: 0 for p in run.placement}
    used_bytes = used_items.copy()
    peak_items = used_items.copy()
    peak_bytes = used_items.copy()
    produced = {node.name: 0 for node in nodes}
    completed_work = produced.copy()
    peak_queued = produced.copy()
    peak_queue_bytes = produced.copy()
    created = settled = sequence = work = 0
    last_progress = time.monotonic()
    blocked_since: float | None = None
    final_reserved = 0
    final_reserved_bytes = 0
    recent_assignments: deque[dict] = deque(maxlen=64)

    def release(output: Output) -> None:
        used_items[output.owner] -= 1
        used_bytes[output.owner] -= output.size
        outputs.pop(output.key)
        if isinstance(transport, RemoteTransport):
            transport.release(output.owner, output.key)
        else:
            local_payloads.pop(output.key)

    def settle(output: Output, edge: int) -> None:
        nonlocal settled
        if edge not in output.edges:
            raise RunError(f"output {output.key}: duplicate/missing edge settlement {edge}")
        output.edges.remove(edge)
        settled += 1
        if not output.edges:
            release(output)

    def label(replica: Replica) -> str:
        p = replica.placement
        return f"stage {p.stage!r} rank {p.rank} work {replica.work} machine {p.machine!r}"

    try:
        endpoint = ""
        if isinstance(transport, zmq.Socket):
            port = transport.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"
        specs: dict[str, list[dict]] = {}
        definitions = {}
        for p in run.placement:
            if run._cancel.is_set():
                return
            index = by_name[p.stage]
            identity = f"{run._run_id}:{index}:{p.rank}".encode()
            replica = Replica(p, index)
            replicas[identity] = replica
            if isinstance(transport, RemoteTransport):
                specs.setdefault(p.machine, []).append({
                    "identity": identity.decode(), "stage": p.stage,
                    "source": not nodes[index].inputs, "rank": p.rank, "replicas": p.replicas,
                    "local_rank": p.local_rank, "local_replicas": p.local_replicas,
                })
                definitions[identity] = run._definitions[index]
            else:
                process = mp.get_context("spawn").Process(
                    target=worker_main,
                    args=(endpoint, identity, run._definitions[index], not nodes[index].inputs,
                          p.rank, p.replicas, run._payload_limit, p.machine, p.local_rank, p.local_replicas),
                    name=f"zjobs-{p.stage}-{p.rank}",
                )
                process.start()
                replica.process = process
                with run._lock:
                    run._stats["worker_pids"].append(process.pid)
        if isinstance(transport, RemoteTransport):
            pids = transport.start(specs, definitions, run._payload_limit, run._worker_timeout,
                                   run._retained_items, run._retained_bytes)
        else:
            pids = {key: r.process.pid for key, r in replicas.items() if r.process is not None}
        for replica in replicas.values():
            replica.since = time.monotonic()
        with run._lock:
            run._stats["state"] = "running"
            run._stats["worker_pids"] = list(pids.values())
            run._stats["workers"] = [
                {"machine": replicas[key].placement.machine, "pid": pid,
                 "stage": replicas[key].placement.stage, "rank": replicas[key].placement.rank}
                for key, pid in pids.items()
            ]
        order = deque(replicas)
        while not run._cancel.is_set():
            now = time.monotonic()
            progress = False
            order.rotate(-1)
            # Publish every required branch atomically. Retention continues after
            # admission and dispatch until successful consumer execution settles it.
            for identity in order:
                replica = replicas[identity]
                output = replica.output
                if output is None or output.published:
                    continue
                if downstream[replica.stage]:
                    if any(
                        len(pending[i]) >= run._capacity
                        or sum(item.size for item in pending[i]) + output.size > run._queue_bytes
                        for i in downstream[replica.stage]
                    ):
                        continue
                    for index in downstream[replica.stage]:
                        pending[index].append(output)
                        name = nodes[index].name
                        peak_queued[name] = max(peak_queued[name], len(pending[index]))
                        peak_queue_bytes[name] = max(
                            peak_queue_bytes[name], sum(item.size for item in pending[index]),
                        )
                    output.published = True
                    replica.output = None
                    replica.state = "ready"
                    produced[replica.placement.stage] += 1
                    progress = True
                else:
                    with run._lock:
                        fits = (
                            run._outputs.qsize() + final_reserved < run._outputs.maxsize
                            and run._buffered_bytes + final_reserved_bytes + output.size <= run._output_bytes
                        )
                    if not fits:
                        continue
                    if isinstance(transport, RemoteTransport):
                        transport.final(output.owner, output.key)
                        output.final_requested = True
                        final_reserved += 1
                        final_reserved_bytes += output.size
                        output.published = True
                    else:
                        payload = local_payloads[output.key]
                        with run._lock:
                            run._buffered_bytes += len(payload)
                            run._stats["peak_output_buffered"] = max(
                                run._stats["peak_output_buffered"], run._outputs.qsize() + 1,
                            )
                            run._stats["peak_output_bytes"] = max(
                                run._stats["peak_output_bytes"], run._buffered_bytes,
                            )
                            run._outputs.put_nowait(payload)
                        settle(output, -1)
                        replica.output = None
                        replica.state = "ready"
                        produced[replica.placement.stage] += 1
                    progress = True
            if isinstance(transport, RemoteTransport):
                for key, payload in list(transport.finals.items()):
                    output = outputs[key]
                    if len(payload) != output.size or not output.final_requested:
                        raise RunError(f"output {key}: invalid final payload length or state")
                    with run._lock:
                        run._buffered_bytes += len(payload)
                        run._stats["peak_output_buffered"] = max(
                            run._stats["peak_output_buffered"], run._outputs.qsize() + final_reserved,
                        )
                        run._stats["peak_output_bytes"] = max(
                            run._stats["peak_output_bytes"],
                            run._buffered_bytes + final_reserved_bytes - output.size,
                        )
                        run._outputs.put_nowait(payload)
                        run._stats["final_bytes"] += len(payload)
                    final_reserved -= 1
                    final_reserved_bytes -= output.size
                    replica = replicas[output.producer]
                    settle(output, -1)
                    replica.output = None
                    replica.state = "ready"
                    produced[replica.placement.stage] += 1
                    transport.finals.pop(key)
                    progress = True
            for index, node in enumerate(nodes):
                group = [r for r in replicas.values() if r.stage == index]
                if not node.inputs:
                    complete[index] = all(r.state == "done" for r in group)
                elif (
                    all(complete[indices[parent]] for parent in node.inputs)
                    and not pending[index] and all(r.state == "ready" for r in group)
                ):
                    complete[index] = True
            def dispatch_priority(identity: bytes) -> int:
                replica = replicas[identity]
                p = replica.placement
                if replica.state != "ready" or p.policy == "balanced":
                    return 1
                item = choose(
                    pending[replica.stage], p.machine, p.policy,
                    replica.local_streak, run._remote_after,
                )
                return 0 if item is not None and item.owner == p.machine else 1

            # Match ready local consumers before another ready rank can steal
            # that payload remotely. Balanced replicas retain rotating order.
            dispatch_order = sorted(order, key=dispatch_priority)
            for identity in dispatch_order:
                replica = replicas[identity]
                p = replica.placement
                if replica.state in ("starting", "busy") and now - replica.since > run._worker_timeout:
                    raise RunError(f"{label(replica)}: {replica.state} exceeded {run._worker_timeout}s")
                if replica.process is not None and replica.process.exitcode is not None:
                    if replica.exited_since is None:
                        replica.exited_since = now
                    if not transport.poll(0) or now - replica.exited_since >= .1:
                        raise RunError(f"{label(replica)}: worker exited with code {replica.process.exitcode}")
                    continue
                if replica.state != "ready" or complete[replica.stage]:
                    continue
                # A producer has one retained-output credit until all of its
                # branches settle. Fast sources cannot exhaust a host with an
                # unbounded sequence while downstream ranks are initializing.
                if any(output.producer == identity for output in outputs.values()):
                    continue
                if (
                    used_items[p.machine] >= run._retained_items
                    or used_bytes[p.machine] + run._payload_limit > run._retained_bytes
                ):
                    continue
                if nodes[replica.stage].inputs:
                    item = choose(
                        pending[replica.stage], p.machine, p.policy,
                        replica.local_streak, run._remote_after,
                    )
                    if item is None:
                        continue
                    pending[replica.stage].remove(item)
                    replica.input = item
                    if isinstance(transport, RemoteTransport):
                        payload = encode({"owner": item.owner, "output": item.key})
                    else:
                        payload = local_payloads[item.key]
                    is_local = item.owner == p.machine
                    replica.local_streak = replica.local_streak + 1 if is_local else 0
                    recent_assignments.append({
                        "output": item.key, "stage": p.stage, "rank": p.rank,
                        "owner": item.owner, "consumer": p.machine, "policy": p.policy,
                        "local": is_local, "local_streak": replica.local_streak,
                    })
                    command = b"task"
                else:
                    command, payload = b"next", b""
                used_items[p.machine] += 1
                used_bytes[p.machine] += run._payload_limit
                peak_items[p.machine] = max(peak_items[p.machine], used_items[p.machine])
                peak_bytes[p.machine] = max(peak_bytes[p.machine], used_bytes[p.machine])
                replica.work = work
                work += 1
                transport.send_multipart([identity, command, payload])
                replica.state = "busy"
                replica.since = time.monotonic()
                progress = True
            if transport.poll(10):
                identity, kind, payload = transport.recv_multipart()
                replica = replicas[identity]
                if kind == b"error":
                    raise RunError(f"{label(replica)}: {payload.decode(errors='replace')}")
                if kind == b"ready" and replica.state == "starting":
                    replica.state = "ready"
                elif kind == b"done" and replica.state == "busy" and not nodes[replica.stage].inputs:
                    replica.state = "done"
                    used_items[replica.placement.machine] -= 1
                    used_bytes[replica.placement.machine] -= run._payload_limit
                elif kind == b"item" and replica.state == "busy":
                    sequence += 1
                    if isinstance(transport, RemoteTransport):
                        reference = decode(payload)
                        key, owner, size = reference["output"], reference["owner"], reference["size"]
                        if owner != replica.placement.machine or key in outputs:
                            raise RunError(f"{label(replica)}: invalid output owner/identity")
                    else:
                        key, owner, size = f"{run._run_id}:local:{sequence}", "local", len(payload)
                        local_payloads[key] = payload
                    if not isinstance(size, int) or not 0 <= size <= run._payload_limit:
                        raise RunError(f"{label(replica)}: received payload exceeds limit")
                    used_bytes[owner] -= run._payload_limit - size
                    if replica.input is not None:
                        with run._lock:
                            kind = "local" if replica.input.owner == replica.placement.machine else "remote"
                            run._stats[f"{kind}_items"] += 1
                            run._stats[f"{kind}_bytes"] += replica.input.size
                        settle(replica.input, replica.stage)
                        replica.input = None
                    edges = set(downstream[replica.stage]) or {-1}
                    output = Output(key, owner, size, identity, edges)
                    outputs[key] = output
                    created += len(edges)
                    replica.output = output
                    replica.state = "output"
                    completed_work[replica.placement.stage] += 1
                else:
                    raise RunError(f"{label(replica)}: unexpected message {kind!r} in {replica.state}")
                progress = True
            if progress:
                last_progress = now
            with run._lock:
                buffered = run._buffered_bytes + final_reserved_bytes
                for index, node in enumerate(nodes):
                    peak_queued[node.name] = max(peak_queued[node.name], len(pending[index]))
                    peak_queue_bytes[node.name] = max(peak_queue_bytes[node.name], sum(item.size for item in pending[index]))
                run._stats.update(
                    produced=produced.copy(), completed_work=completed_work.copy(),
                    queued={node.name: len(pending[i]) for i, node in enumerate(nodes)},
                    peak_queued=peak_queued.copy(), peak_queue_bytes=peak_queue_bytes.copy(),
                    retained_items=used_items.copy(), retained_bytes=used_bytes.copy(),
                    peak_retained_items=peak_items.copy(), peak_retained_bytes=peak_bytes.copy(),
                    obligations_created=created, obligations_settled=settled,
                    obligations_pending=created - settled,
                    recent_assignments=list(recent_assignments),
                    pending_outputs=[
                        {"output": item.key, "owner": item.owner, "bytes": item.size,
                         "edges": [nodes[edge].name if edge >= 0 else "final" for edge in sorted(item.edges)]}
                        for item in list(outputs.values())[:32]
                    ],
                    peak_output_buffered=max(run._stats["peak_output_buffered"], run._outputs.qsize() + final_reserved),
                    peak_output_bytes=max(run._stats["peak_output_bytes"], buffered),
                )
                if isinstance(transport, RemoteTransport):
                    run._stats["agents"] = copy.deepcopy(transport.agent_stats)
                last_consumed = run._last_consumed
            if all(complete) and not outputs and created == settled:
                with run._lock:
                    run._stats["state"] = "completed"
                return
            blocked = any(
                r.output is not None and not downstream[r.stage] and not r.output.published
                for r in replicas.values()
            )
            if blocked:
                blocked_since = now if blocked_since is None else blocked_since
                if now - max(blocked_since, last_consumed) > run._consumer_timeout:
                    raise RunError("final consumer stalled with full item/byte buffer; consume or close the run")
            else:
                blocked_since = None
            if now - last_progress > run._worker_timeout:
                raise RunError(
                    f"routing/capacity stalled: {created - settled} required obligations; "
                    f"retained items={used_items}, bytes={used_bytes}; increase capacity or fix placement"
                )
    finally:
        if isinstance(transport, zmq.Socket):
            for identity in replicas:
                try:
                    transport.send_multipart([identity, b"stop", b""], zmq.DONTWAIT)
                except zmq.ZMQError:
                    pass
            stop_processes([r.process for r in replicas.values() if r.process is not None])
