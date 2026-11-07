"""Start two agents as documented, then pass their reachable endpoints."""

import argparse

from zjobs import Graph, SourceContext


def load(context: SourceContext):
    for index in context.shard(range(25)):
        yield {
            "index": index, "bytes": bytes([index]),
            "source_machine": context.machine, "source_rank": context.rank,
        }


class Preprocess:
    def __init__(self):
        self.processed = 0

    def __call__(self, item):
        self.processed += 1
        return {
            "index": item["index"], "value": item["bytes"][0] ** 2,
            "source_machine": item["source_machine"], "source_rank": item["source_rank"],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-a", required=True, help="Reachable agent TCP endpoint")
    parser.add_argument("--host-b", required=True, help="Reachable agent TCP endpoint")
    args = parser.parse_args()
    graph = Graph()
    source = graph.source("load", load, placement={"host-a": 2, "host-b": 2})
    graph.map("preprocess", Preprocess, inputs=[source], placement={"host-b": 2})
    with graph.run(agents={"host-a": args.host_a, "host-b": args.host_b},
                   capacity=8, output_capacity=4) as run:
        seen = []
        for batch in run.batches(6):
            seen.extend(item["index"] for item in batch)
            print(batch)
        assert sorted(seen) == list(range(25)), "incomplete or duplicate sharding"
        print(run.status)


if __name__ == "__main__":
    main()
