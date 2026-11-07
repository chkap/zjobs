"""Reproducible file/CPU application rank and locality comparison; JSON output."""

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from loading_pipeline import (
    agent_endpoints, build_graph, checksum, environment, expected_outputs, generate, trial,
)
from measure_payload import start_agents, stop_agents


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, help="Existing identical corpus path on all hosts")
    cli.add_argument("--items", type=int, default=48)
    cli.add_argument("--kib", type=int, default=256)
    cli.add_argument("--rounds", type=int, default=200000)
    cli.add_argument("--batch-size", type=int, default=4)
    cli.add_argument("--preprocess-ranks", type=int, nargs="+", default=[1, 2, 4])
    cli.add_argument("--warmup", type=int, default=1)
    cli.add_argument("--trials", type=int, default=3)
    cli.add_argument("--network-host", help="Start two one-physical-host agents on this bindable IP")
    cli.add_argument("--agent", action="append", default=[], metavar="NAME=ENDPOINT")
    cli.add_argument("--output", type=Path, help="Write generated machine-readable evidence")
    args = cli.parse_args()
    if min(args.items, args.kib, args.rounds, args.batch_size, args.trials,
           *args.preprocess_ranks) < 1 or args.warmup < 0:
        cli.error("counts/bytes/rounds/ranks/trials must be positive; warmup nonnegative")
    if args.agent and args.network_host:
        cli.error("--agent and --network-host are alternatives")
    if args.agent and args.data_dir is None:
        cli.error("external agents require identical --data-dir on controller/hosts")
    try:
        endpoints = agent_endpoints(args.agent)
    except ValueError as exc:
        cli.error(str(exc))
    processes = []
    with tempfile.TemporaryDirectory(prefix="zjobs-application-") as temporary:
        directory = (args.data_dir or Path(temporary) / "data").resolve()
        if args.data_dir is None:
            generate(directory, args.items, args.kib * 1024)
        expected = expected_outputs(directory, args.items, args.rounds)
        results = []
        try:
            configs = [
                ("local", ranks, "local_first") for ranks in args.preprocess_ranks
            ]
            if args.network_host:
                processes, endpoints = start_agents(args.network_host, temporary)
            if endpoints:
                configs += [("agents", 1, policy) for policy in ("local_first", "balanced")]
            for mode, ranks, policy in configs:
                agents = endpoints if mode == "agents" else None
                io_ranks = 1 if agents else 2
                graph = build_graph(directory, args.items, args.rounds, args.batch_size,
                                    io_ranks, ranks, 1, policy, agents)
                warmups = [trial(graph, agents, expected, args.batch_size) for _ in range(args.warmup)]
                trials = [trial(graph, agents, expected, args.batch_size) for _ in range(args.trials)]
                results.append({
                    "mode": mode, "policy": policy, "io_ranks": io_ranks,
                    "preprocess_ranks": ranks, "collect_ranks": 1,
                    "rank_counts": "per machine" if agents else "local totals",
                    "warmups": warmups, "trials": trials,
                    "median_elapsed_seconds": statistics.median(t["elapsed_seconds"] for t in trials),
                    "median_items_per_second": statistics.median(t["items_per_second"] for t in trials),
                })
        finally:
            stop_agents(processes)
        output = {
            "environment": environment(),
            "config": {**vars(args), "data_dir": str(directory), "output": str(args.output) if args.output else None},
            "expected_checksum": checksum({i: value[2] for i, value in expected.items()}),
            "agents": endpoints,
            "physical_hosts": 1 if args.network_host or not endpoints else None,
            "physical_host_evidence": "external placement not independently attested" if args.agent else "one physical host only",
            "results": results,
            "timing_scope": "worker initialization, file IO, CPU preprocessing, serialization, collection, verification and cleanup; excludes fixture generation, expected digest computation and agent startup",
        }
        text = json.dumps(output, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(json.dumps({
                "output": str(args.output), "expected_checksum": output["expected_checksum"],
                "medians": [{k: result[k] for k in ("mode", "policy", "preprocess_ranks",
                                                  "median_elapsed_seconds", "median_items_per_second")}
                            for result in results],
            }))
        else:
            print(text, end="")


if __name__ == "__main__":
    main()
