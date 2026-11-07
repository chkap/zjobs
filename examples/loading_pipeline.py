"""File IO -> CPU preprocessing -> independently scaled collection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

from zjobs import Graph

SALT = b"zjobs-loading-v1"


def generate(directory: Path, count: int, payload_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        seed = hashlib.sha256(str(index).encode()).digest()
        data = (seed * ((payload_bytes + len(seed) - 1) // len(seed)))[:payload_bytes]
        (directory / f"{index:06d}.bin").write_bytes(data)


def load_files(directory: str, count: int, context):
    # Shard cheap IDs BEFORE reading files: each rank loads only its own work.
    for index in context.shard(range(count)):
        started, cpu = time.perf_counter(), time.process_time()
        data = (Path(directory) / f"{index:06d}.bin").read_bytes()
        yield {
            "id": index, "data": data,
            "read_seconds": time.perf_counter() - started,
            "read_cpu_seconds": time.process_time() - cpu,
            "source_machine": context.machine, "source_rank": context.global_rank,
        }


class FileSource:
    def __init__(self, directory: str, count: int):
        self.directory, self.count = directory, count

    def __call__(self, context):
        yield from load_files(self.directory, self.count, context)


class Preprocess:
    def __init__(self, rounds: int):
        self.rounds = rounds

    def __call__(self, record):
        started, cpu = time.perf_counter(), time.process_time()
        digest = hashlib.pbkdf2_hmac("sha256", record["data"], SALT, self.rounds).hex()
        return {
            **record, "digest": digest,
            "preprocess_seconds": time.perf_counter() - started,
            "preprocess_cpu_seconds": time.process_time() - cpu,
        }


def build_graph(
    directory: Path, count: int, rounds: int, batch_size: int,
    io_ranks: int = 2, preprocess_ranks: int = 2, collect_ranks: int = 1,
    policy: str = "local_first", agents: dict[str, str] | None = None,
):
    graph = Graph(policy=policy)

    def scaling(ranks):
        # Counts are per machine when agents are present, total in local mode.
        return {"placement": {machine: ranks for machine in agents}} if agents else {"replicas": ranks}

    io = graph.source("io", FileSource(str(directory), count),
                      **scaling(io_ranks))
    processed = graph.map("preprocess", Preprocess(rounds), inputs=[io],
                          **scaling(preprocess_ranks))
    graph.batch("collect", inputs=[processed], size=batch_size,
                **scaling(collect_ranks))
    return graph


def expected_outputs(directory: Path, count: int, rounds: int):
    expected = {}
    for index in range(count):
        data = (directory / f"{index:06d}.bin").read_bytes()
        expected[index] = (
            len(data), hashlib.sha256(data).hexdigest(),
            hashlib.pbkdf2_hmac("sha256", data, SALT, rounds).hex(),
        )
    return expected


def checksum(digests: dict[int, str]) -> str:
    return hashlib.sha256("".join(
        f"{index}:{digests[index]}\n" for index in sorted(digests)
    ).encode()).hexdigest()


def run_options(expected, batch_size):
    largest = max((value[0] for value in expected.values()), default=1)
    limit = (largest + 2048) * batch_size + 4096
    return {
        "max_payload_bytes": limit, "retained_items": max(64, batch_size * 8),
        "retained_bytes": max(64 * 1024 * 1024, limit * 32),
        "queue_bytes": limit * 8, "output_bytes": limit * 4,
        "capacity": 16, "output_capacity": 4,
    }


def trial(graph, agents, expected, batch_size):
    options = run_options(expected, batch_size)
    digests = {}
    service = {name: 0.0 for name in (
        "read_seconds", "read_cpu_seconds", "preprocess_seconds", "preprocess_cpu_seconds",
    )}
    started = time.perf_counter()
    first = None
    batch_lengths = {}
    sources = {}
    with graph.run(agents=agents, **options) as run:
        for batch in run:
            if first is None:
                first = time.perf_counter()
            batch_lengths[len(batch)] = batch_lengths.get(len(batch), 0) + 1
            for record in batch:
                index = record["id"]
                if index not in expected or index in digests:
                    raise AssertionError(f"unexpected/duplicate item {index}")
                length, data_hash, digest = expected[index]
                if (
                    len(record["data"]) != length
                    or hashlib.sha256(record["data"]).hexdigest() != data_hash
                    or record["digest"] != digest
                ):
                    raise AssertionError(f"incorrect payload/preprocessing for item {index}")
                digests[index] = record["digest"]
                for name in service:
                    service[name] += record[name]
                source = f"{record['source_machine']}:{record['source_rank']}"
                sources[source] = sources.get(source, 0) + 1
    elapsed = time.perf_counter() - started
    status = run.status
    if set(digests) != set(expected):
        raise AssertionError("missing required input IDs")
    if status["obligations_created"] != status["obligations_settled"]:
        raise AssertionError("unsettled required delivery")
    if any(agent["retained_bytes"] for agent in status.get("agents", {}).values()):
        raise AssertionError("agent store not released")
    return {
        "elapsed_seconds": elapsed, "items_per_second": len(digests) / elapsed,
        "time_to_first_batch_seconds": None if first is None else first - started,
        "items": len(digests), "checksum": checksum(digests), "source_items": sources,
        "batch_length_counts": batch_lengths, "service": service,
        "placement": status["placement"], "run_options": options,
        "state": status["state"], "obligations_created": status["obligations_created"],
        "obligations_settled": status["obligations_settled"],
        "local_items": status["local_items"], "remote_items": status["remote_items"],
        "local_bytes": status["local_bytes"], "remote_bytes": status["remote_bytes"],
        "final_bytes": status["final_bytes"], "agents": status.get("agents", {}),
        "peak_queue_items": status["peak_queued"], "peak_queue_bytes": status["peak_queue_bytes"],
        "peak_retained_items": status["peak_retained_items"],
        "peak_retained_bytes": status["peak_retained_bytes"],
        "peak_output_items": status["peak_output_buffered"],
        "peak_output_bytes": status["peak_output_bytes"],
        "worker_pids": status["worker_pids"],
    }


def environment():
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "versions": {name: importlib.metadata.version(name) for name in ("zjobs", "cloudpickle", "pyzmq")},
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path)
    result.add_argument("--generate", action="store_true", help="Write deterministic fixture files before running")
    result.add_argument("--generate-only", action="store_true", help="Write fixtures and exit; useful on each agent host")
    result.add_argument("--items", type=int, default=32)
    result.add_argument("--kib", type=int, default=256)
    result.add_argument("--rounds", type=int, default=100000, help="PBKDF2 CPU preprocessing iterations")
    result.add_argument("--io-ranks", type=int, default=2)
    result.add_argument("--preprocess-ranks", type=int, default=2)
    result.add_argument("--collect-ranks", type=int, default=1)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--policy", choices=("local_first", "balanced", "local_only"), default="local_first")
    result.add_argument("--agent", action="append", default=[], metavar="NAME=ENDPOINT")
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--trials", type=int, default=3)
    return result


def agent_endpoints(values):
    endpoints = {}
    for value in values:
        name, separator, endpoint = value.partition("=")
        if not separator or not name or not endpoint.startswith("tcp://") or name in endpoints:
            raise ValueError(f"invalid/duplicate agent mapping {value!r}; use NAME=tcp://HOST:PORT")
        endpoints[name] = endpoint
    return endpoints or None


def main():
    cli = parser()
    args = cli.parse_args()
    if min(args.items, args.kib, args.rounds, args.io_ranks, args.preprocess_ranks,
           args.collect_ranks, args.batch_size, args.trials) < 1 or args.warmup < 0:
        cli.error("counts/sizes/rounds/ranks/trials must be positive; warmup must be nonnegative")
    try:
        agents = agent_endpoints(args.agent)
    except ValueError as exc:
        cli.error(str(exc))
    if agents and args.data_dir is None:
        cli.error("--data-dir must exist at the same path on controller and every agent host")
    if args.generate_only and args.data_dir is None:
        cli.error("--generate-only requires --data-dir")
    with tempfile.TemporaryDirectory(prefix="zjobs-loading-") as temporary:
        directory = (args.data_dir or Path(temporary)).resolve()
        if args.generate or args.generate_only or args.data_dir is None:
            generate(directory, args.items, args.kib * 1024)
        if args.generate_only:
            print(json.dumps({"directory": str(directory), "items": args.items, "payload_bytes": args.kib * 1024}))
            return
        expected = expected_outputs(directory, args.items, args.rounds)
        graph = build_graph(directory, args.items, args.rounds, args.batch_size,
                            args.io_ranks, args.preprocess_ranks, args.collect_ranks, args.policy, agents)
        warmups = [trial(graph, agents, expected, args.batch_size) for _ in range(args.warmup)]
        trials = [trial(graph, agents, expected, args.batch_size) for _ in range(args.trials)]
        print(json.dumps({
            "environment": environment(), "config": {
                **vars(args), "data_dir": str(directory),
                "agent": agents, "rank_counts": "per machine" if agents else "local totals",
            },
            "expected_checksum": checksum({i: value[2] for i, value in expected.items()}),
            "warmups": warmups, "trials": trials,
            "median_elapsed_seconds": statistics.median(t["elapsed_seconds"] for t in trials),
            "physical_host_evidence": "not established by agent identities",
        }))


if __name__ == "__main__":
    main()
