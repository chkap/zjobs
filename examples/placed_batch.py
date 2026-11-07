"""Narrow in-graph batching demo; omit endpoint arguments for local launch."""

import argparse

from zjobs import Graph


def source(context):
    yield from context.shard(range(23))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-a")
    parser.add_argument("--host-b")
    args = parser.parse_args()
    if bool(args.host_a) != bool(args.host_b):
        parser.error("supply both agent endpoints or neither")
    agents = {"host-a": args.host_a, "host-b": args.host_b} if args.host_a else None
    graph = Graph()
    root = graph.source("source", source, **(
        {"placement": {"host-a": 2}} if agents else {"replicas": 2}
    ))
    collected = graph.batch("collect", inputs=[root], size=5, max_bytes=4096, **(
        {"placement": {"host-b": 2}} if agents else {"replicas": 2}
    ))
    graph.map("consume", lambda batch: batch, inputs=[collected], **(
        {"placement": {"host-b": 1}} if agents else {}
    ))
    with graph.run(agents=agents, max_payload_bytes=8192, retained_bytes=262144) as run:
        batches = list(run)
        assert sorted(member for batch in batches for member in batch) == list(range(23))
        print(batches)
        print({
            "state": run.status["state"], "items": sum(map(len, batches)),
            "batch_lengths": list(map(len, batches)),
            "obligations_created": run.status["obligations_created"],
            "obligations_settled": run.status["obligations_settled"],
            "retries": run.status["retries"],
        })


if __name__ == "__main__":
    main()
