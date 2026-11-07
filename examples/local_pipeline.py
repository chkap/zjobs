"""Run: python examples/local_pipeline.py"""
from zjobs import Graph


def main():
    graph = Graph()
    raw = graph.source("numbers", lambda ctx: range(7))
    squared = graph.map("square", lambda x: x * x, inputs=[raw], replicas=2)
    graph.batch("collect", inputs=[squared], size=3)
    with graph.run() as stream:
        for batch in stream:
            print(batch)
    print(stream.status)


if __name__ == "__main__":
    main()
