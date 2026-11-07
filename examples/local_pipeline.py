"""Run with: python examples/local_pipeline.py"""

from zjobs import Graph, SourceContext


def load(context: SourceContext):
    for item in context.shard(range(25)):
        yield {"index": item, "data": bytes([item])}


class Preprocess:
    def __init__(self):
        self.processed = 0

    def __call__(self, item):
        self.processed += 1
        return {"index": item["index"], "value": item["data"][0] ** 2}


def main():
    graph = Graph()
    source = graph.source("load", load, replicas=4)
    graph.map("preprocess", Preprocess, inputs=[source], replicas=2)
    with graph.run(capacity=8, output_capacity=4) as run:
        for batch in run.batches(6):
            print(batch)
        print(run.status)


if __name__ == "__main__":
    main()
