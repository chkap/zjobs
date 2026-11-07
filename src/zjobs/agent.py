"""Manually launched trusted-user host agent: ``python -m zjobs.agent``."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass

import zmq

from ._protocol import capabilities, configure, decode, encode, stop_processes
from ._worker import worker_main
from ._payload import Payload, size, group_frames, MAX_FRAMES


@dataclass
class _Worker:
    process: mp.Process
    reported_exit: bool = False
    exited_since: float | None = None


def serve(
    machine: str, bind: str, advertise: str, *,
    lease_timeout: float = 3.0, max_workers: int = 256,
) -> None:
    """Serve one active run at a time, leaving the agent alive between runs."""
    stopping = False

    def interrupt(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with zmq.Context() as context:
        with context.socket(zmq.ROUTER) as control, context.socket(zmq.ROUTER) as workers:
            configure(control)
            configure(workers)
            control.setsockopt(zmq.ROUTER_MANDATORY, 1)
            workers.setsockopt(zmq.ROUTER_MANDATORY, 1)
            control.bind(bind)
            port = workers.bind_to_random_port("tcp://127.0.0.1")
            worker_endpoint = f"tcp://127.0.0.1:{port}"
            poller = zmq.Poller()
            poller.register(control, zmq.POLLIN)
            poller.register(workers, zmq.POLLIN)
            owner: bytes | None = None
            run: str | None = None
            last_seen = 0.0
            last_heartbeat = 0.0
            group: dict[bytes, _Worker] = {}
            payloads: dict[str, Payload] = {}
            payload_workers: dict[str, bytes] = {}
            accepted: set[str] = set()
            sequence = 0
            peers: dict[str, zmq.Socket] = {}
            fetching: dict[bytes, str] = {}
            groups: dict[bytes, list[dict]] = {}
            group_payloads: dict[bytes, list[Payload]] = {}
            byte_limit = 0
            item_limit = 0
            payload_limit = 0
            retired: set[bytes] = set()
            retained_size = 0
            peak_retained_size = 0
            peak_retained_items = 0
            transfers = {"local_items": 0, "local_bytes": 0, "remote_items": 0, "remote_bytes": 0,
                         "served_remote_bytes": 0, "final_bytes": 0}

            def send(recipient: bytes, message: dict, *frames: bytes) -> None:
                control.send_multipart([recipient, encode(message), *frames])

            def cleanup() -> None:
                nonlocal owner, run, group, retained_size, transfers, peak_retained_size, peak_retained_items
                for identity in group:
                    try:
                        workers.send_multipart([identity, b"stop", b""], zmq.DONTWAIT)
                    except zmq.ZMQError:
                        pass
                stop_processes([worker.process for worker in group.values()])
                group = {}
                payloads.clear()
                payload_workers.clear()
                accepted.clear()
                fetching.clear()
                groups.clear()
                group_payloads.clear()
                retired.clear()
                retained_size = 0
                peak_retained_size = peak_retained_items = 0
                for peer in peers.values():
                    poller.unregister(peer)
                    peer.close()
                peers.clear()
                transfers = {key: 0 for key in transfers}
                owner = None
                run = None
                # Old run identities include a UUID and are never reused.

            print(encode({
                "event": "ready", "machine": machine, "endpoint": advertise,
                **capabilities(), "lease_timeout": lease_timeout,
            }).decode(), flush=True)

            def deliver_group(identity: bytes) -> None:
                while groups[identity]:
                    reference = groups[identity][0]
                    if reference["owner"] != machine:
                        fetching[identity] = reference["output"]
                        peers[reference["owner"]].send_multipart([encode({
                            "type": "fetch", "run": run, "output": reference["output"],
                            "worker": identity.decode(),
                        })])
                        return
                    payload = payloads[reference["output"]]
                    transfers["local_items"] += 1
                    transfers["local_bytes"] += size(payload)
                    group_payloads[identity].append(payload)
                    groups[identity].pop(0)
                workers.send_multipart([identity, b"batch", *group_frames(group_payloads.pop(identity))])
                groups.pop(identity)

            try:
                while not stopping:
                    now = time.monotonic()
                    if owner is not None and now - last_seen > lease_timeout:
                        print(f"agent {machine}: run {run} controller lease expired; cleaning workers",
                              file=sys.stderr, flush=True)
                        cleanup()
                    events = dict(poller.poll(20))
                    if control in events:
                        frames = control.recv_multipart()
                        recipient = frames[0]
                        message = {}
                        try:
                            message = decode(frames[1])
                            kind = message["type"]
                            if kind == "hello":
                                send(recipient, {
                                    "type": "hello", "machine": machine,
                                    "endpoint": advertise, **capabilities(),
                                    "lease_timeout": lease_timeout,
                                    "max_workers": max_workers,
                                })
                            elif kind == "start":
                                if owner is not None:
                                    raise ValueError(f"agent {machine} is busy with run {run}")
                                if message["capabilities"] != capabilities():
                                    raise ValueError("incompatible protocol/Python/cloudpickle/transport")
                                specs = message["workers"]
                                if not specs or len(specs) > max_workers:
                                    raise ValueError(f"worker count exceeds agent capacity {max_workers}")
                                if len(frames) != 2 + len(specs):
                                    raise ValueError("worker definition count does not match placement")
                                owner, run = recipient, message["run"]
                                byte_limit = message["retained_bytes"]
                                item_limit = message["retained_items"]
                                payload_limit = message["payload_limit"]
                                for peer_machine, endpoint in message["peers"].items():
                                    if peer_machine == machine:
                                        continue
                                    peer = context.socket(zmq.DEALER)
                                    configure(peer)
                                    peer.connect(endpoint)
                                    peers[peer_machine] = peer
                                    poller.register(peer, zmq.POLLIN)
                                last_seen = time.monotonic()
                                spawn = mp.get_context("spawn")
                                for spec, definition in zip(specs, frames[2:]):
                                    if stopping:
                                        raise RuntimeError("agent is stopping")
                                    identity = spec["identity"].encode("ascii")
                                    if identity in group:
                                        raise ValueError("duplicate worker identity")
                                    process = spawn.Process(
                                        target=worker_main,
                                        args=(
                                            worker_endpoint, identity, definition,
                                            spec["source"], spec["rank"], spec["replicas"],
                                            message["payload_limit"], machine,
                                            spec["local_rank"], spec["local_replicas"],
                                            os.getpid(),
                                        ),
                                        name=f"zjobs-{spec['stage']}-{spec['rank']}",
                                    )
                                    process.start()
                                    group[identity] = _Worker(process)
                                    send(recipient, {"type": "heartbeat", "run": run})
                                last_seen = time.monotonic()
                                send(recipient, {
                                    "type": "started", "run": run,
                                    "pids": {key.decode(): value.process.pid for key, value in group.items()},
                                })
                            elif kind == "fetch":
                                if message["run"] != run or message["output"] not in payloads:
                                    raise ValueError("requested payload is not available in this run")
                                payload = payloads[message["output"]]
                                transfers["served_remote_bytes"] += size(payload)
                                send(recipient, {
                                    "type": "fetched", "run": run, "worker": message["worker"],
                                    "output": message["output"],
                                }, *payload)
                            elif recipient != owner or message.get("run") != run:
                                raise ValueError("command does not own this agent run")
                            elif kind == "heartbeat":
                                last_seen = time.monotonic()
                            elif kind == "command":
                                last_seen = time.monotonic()
                                identity = message["worker"].encode("ascii")
                                if identity not in group or len(frames) != 4:
                                    raise ValueError("unknown worker or invalid command frames")
                                if frames[2] == b"batch":
                                    groups[identity] = decode(frames[3])["members"]
                                    group_payloads[identity] = []
                                    deliver_group(identity)
                                elif frames[2] == b"task":
                                    reference = decode(frames[3])
                                    if reference["owner"] == machine:
                                        payload = payloads[reference["output"]]
                                        transfers["local_items"] += 1
                                        transfers["local_bytes"] += size(payload)
                                        workers.send_multipart([identity, b"task", *payload])
                                    else:
                                        fetching[identity] = reference["output"]
                                        peers[reference["owner"]].send_multipart([encode({
                                            "type": "fetch", "run": run,
                                            "output": reference["output"], "worker": identity.decode(),
                                        })])
                                else:
                                    workers.send_multipart([identity, frames[2], frames[3]])
                            elif kind == "release":
                                payload = payloads.pop(message["output"])
                                payload_workers.pop(message["output"])
                                accepted.discard(message["output"])
                                retained_size -= size(payload)
                            elif kind == "accept":
                                if message["output"] not in payloads:
                                    raise ValueError("cannot acknowledge unavailable output")
                                accepted.add(message["output"])
                            elif kind == "replace":
                                old = message["old"].encode("ascii")
                                spec = message["worker"]
                                identity = spec["identity"].encode("ascii")
                                if old not in group or identity in group or len(frames) != 3:
                                    raise ValueError("invalid worker replacement")
                                retired.add(old)
                                for output in list(payloads):
                                    if payload_workers[output] == old and output not in accepted:
                                        retained_size -= size(payloads.pop(output))
                                        payload_workers.pop(output)
                                fetching.pop(old, None)
                                groups.pop(old, None)
                                group_payloads.pop(old, None)
                                try:
                                    workers.send_multipart([old, b"stop", b""], zmq.DONTWAIT)
                                except zmq.ZMQError:
                                    pass
                                previous = group.pop(old).process
                                if previous.is_alive():
                                    previous.terminate()
                                stop_processes([previous])
                                process = mp.get_context("spawn").Process(
                                    target=worker_main,
                                    args=(worker_endpoint, identity, frames[2], spec["source"],
                                          spec["rank"], spec["replicas"], payload_limit, machine,
                                          spec["local_rank"], spec["local_replicas"], os.getpid()),
                                    name=f"zjobs-{spec['stage']}-{spec['rank']}",
                                )
                                process.start()
                                group[identity] = _Worker(process)
                                last_seen = time.monotonic()
                                send(recipient, {"type": "replaced", "run": run,
                                                 "worker": identity.decode(), "pid": process.pid})
                            elif kind == "final":
                                payload = payloads[message["output"]]
                                transfers["final_bytes"] += size(payload)
                                send(recipient, {
                                    "type": "final", "run": run, "output": message["output"],
                                }, *payload)
                            elif kind == "stop":
                                stopped_run = run
                                final_transfers = transfers.copy()
                                peak_bytes, peak_items = peak_retained_size, peak_retained_items
                                cleanup()
                                send(recipient, {
                                    "type": "stopped", "run": stopped_run,
                                    "transfers": final_transfers, "retained_items": 0, "retained_bytes": 0,
                                    "peak_retained_bytes": peak_bytes, "peak_retained_items": peak_items,
                                })
                            else:
                                raise ValueError(f"unknown agent command {kind!r}")
                        except (ValueError, KeyError, TypeError, RuntimeError, OSError, zmq.ZMQError) as exc:
                            print(f"agent {machine}: {exc}", file=sys.stderr, flush=True)
                            failed_run = message.get("run")
                            if recipient == owner:
                                cleanup()
                            try:
                                error = {"type": "error", "run": failed_run, "error": str(exc)}
                                if message.get("type") == "fetch":
                                    error.update(worker=message.get("worker"), output=message.get("output"))
                                send(recipient, error)
                            except zmq.ZMQError as send_error:
                                print(f"agent {machine}: cannot report error: {send_error}",
                                      file=sys.stderr, flush=True)
                    if workers in events:
                        worker_frames = workers.recv_multipart()
                        identity, kind = worker_frames[:2]
                        payload = tuple(worker_frames[2:])
                        if identity in group and owner is not None:
                            try:
                                if kind == b"item":
                                    total = size(payload)
                                    if len(payload) > MAX_FRAMES or total > payload_limit:
                                        raise RuntimeError("agent received oversized logical payload")
                                    if len(payloads) >= item_limit or retained_size + total > byte_limit:
                                        raise RuntimeError("agent retained payload capacity exceeded")
                                    sequence += 1
                                    output = f"{run}:{machine}:{sequence}"
                                    payloads[output] = payload
                                    payload_workers[output] = identity
                                    retained_size += total
                                    peak_retained_size = max(peak_retained_size, retained_size)
                                    peak_retained_items = max(peak_retained_items, len(payloads))
                                    payload = (encode({
                                        "output": output, "owner": machine, "size": total,
                                    }),)
                                send(owner, {
                                    "type": "worker", "run": run, "worker": identity.decode(),
                                }, kind, *payload)
                            except RuntimeError as exc:
                                print(f"agent {machine}: payload accounting failure: {exc}",
                                      file=sys.stderr, flush=True)
                                send(owner, {"type": "error", "run": run, "error": str(exc)})
                                cleanup()
                            except zmq.ZMQError as exc:
                                print(f"agent {machine}: controller disconnected: {exc}",
                                      file=sys.stderr, flush=True)
                                cleanup()
                    for peer_machine, peer in peers.copy().items():
                        if peer not in events:
                            continue
                        frames = peer.recv_multipart()
                        try:
                            message = decode(frames[0])
                            identity = message.get("worker", "").encode("ascii")
                            if identity in retired:
                                continue
                            if (
                                message.get("type") != "fetched" or message.get("run") != run
                                or fetching.get(identity) != message.get("output") or len(frames) < 3
                            ):
                                raise ValueError(f"invalid/missing payload from agent {peer_machine}")
                            fetching.pop(identity)
                            payload = tuple(frames[1:])
                            if len(payload) > MAX_FRAMES or size(payload) > payload_limit:
                                raise ValueError("peer returned oversized logical payload")
                            transfers["remote_items"] += 1
                            transfers["remote_bytes"] += size(payload)
                            if identity in groups:
                                group_payloads[identity].append(payload)
                                groups[identity].pop(0)
                                deliver_group(identity)
                            else:
                                workers.send_multipart([identity, b"task", *payload])
                        except (ValueError, zmq.ZMQError) as exc:
                            if owner is not None:
                                send(owner, {"type": "error", "run": run, "error": str(exc)})
                            cleanup()
                            break
                    now = time.monotonic()
                    if owner is not None:
                        try:
                            if now - last_heartbeat >= 0.1:
                                send(owner, {
                                    "type": "heartbeat", "run": run, "transfers": transfers,
                                    "retained_items": len(payloads), "retained_bytes": retained_size,
                                    "peak_retained_bytes": peak_retained_size, "peak_retained_items": peak_retained_items,
                                })
                                last_heartbeat = now
                            for identity, worker in group.items():
                                if worker.process.exitcode is None or worker.reported_exit:
                                    continue
                                if worker.exited_since is None:
                                    worker.exited_since = now
                                if now - worker.exited_since < 0.1:
                                    continue
                                send(owner, {
                                    "type": "worker", "run": run, "worker": identity.decode(),
                                }, b"error", f"worker exited with code {worker.process.exitcode}".encode())
                                worker.reported_exit = True
                        except zmq.ZMQError as exc:
                            print(f"agent {machine}: connection failure: {exc}",
                                  file=sys.stderr, flush=True)
                            cleanup()
            finally:
                cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted zjobs static host agent (no authentication)")
    parser.add_argument("--machine", required=True, help="Unique configured machine identity")
    parser.add_argument("--bind", required=True, help="TCP listen endpoint, e.g. tcp://0.0.0.0:5555")
    parser.add_argument("--advertise", required=True, help="Reachable TCP endpoint, e.g. tcp://host:5555")
    parser.add_argument("--lease-timeout", type=float, default=3.0)
    parser.add_argument("--max-workers", type=int, default=256)
    args = parser.parse_args()
    if (
        not args.machine or not args.bind.startswith("tcp://")
        or not args.advertise.startswith("tcp://")
        or not math.isfinite(args.lease_timeout) or args.lease_timeout < 0.5
        or args.max_workers < 1
    ):
        parser.error("require a machine identity, TCP endpoints, lease-timeout >= 0.5 and positive max-workers")
    serve(args.machine, args.bind, args.advertise,
          lease_timeout=args.lease_timeout, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
