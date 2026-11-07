"""Optional CPU IterableDataset; one locally owned graph and consumer per machine."""
import argparse
import json

from zjobs import Graph, Limits, PeerConfig

try:
    import torch
except ImportError as exc:
    raise ImportError("This optional example requires torch; see docs/examples.md") from exc


class LocalGraphDataset(torch.utils.data.IterableDataset):
    def __init__(self, graph, *, peer=None, limits=None):
        super().__init__()
        self.graph, self.peer, self.limits = graph, peer, limits
        self.stream = None
        self.last_status = None

    def __enter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("LocalGraphDataset requires DataLoader num_workers=0")
        if self.stream is not None:
            raise RuntimeError("dataset context is already active")
        self.stream = self.graph.run(peer=self.peer, limits=self.limits)
        self._iterated = False
        return self

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("LocalGraphDataset requires DataLoader num_workers=0")
        if self.stream is None:
            raise RuntimeError("enter the LocalGraphDataset context before iteration")
        if self._iterated:
            raise RuntimeError("one iterator per dataset context; create a new run for repetition")
        self._iterated = True
        return iter(self.stream)

    def __exit__(self, exc_type, exc, traceback):
        stream, self.stream = self.stream, None
        try:
            return stream.__exit__(exc_type, exc, traceback)
        finally:
            self.last_status = stream.status


def tensor_graph(count):
    graph = Graph()
    def source(ctx):
        for i in range(count):
            # Explicit disjoint fixture coordinates; not a distributed sampler.
            yield (ctx.peer, i)
    raw = graph.source("coordinates", source)
    class Tensorize:
        def __init__(self):
            import torch
            torch.set_num_threads(1)
        def __call__(self, value):
            import torch
            return {"source_peer": value[0], "index": value[1],
                    "tensor": torch.tensor([value[1], value[1] + 1], dtype=torch.float32)}
    graph.map("cpu_tensor", Tensorize, inputs=[raw], replicas=2)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--identity", default="local")
    parser.add_argument("--peers", type=json.loads, default={})
    parser.add_argument("--run-id", default="tensor-example")
    args = parser.parse_args()
    torch.set_num_threads(1)
    peer = PeerConfig(args.identity, args.run_id, args.peers.get(args.identity), args.peers)
    dataset = LocalGraphDataset(tensor_graph(args.count), peer=peer,
                                limits=Limits(task_timeout=60))
    values = []
    with dataset:
        loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None)
        for value in loader:
            if value["tensor"].device.type != "cpu":
                raise RuntimeError("example supports CPU tensors only")
            values.append({**value, "tensor": value["tensor"].tolist()})
    print(json.dumps({"identity": args.identity, "values": values,
                      "status": dataset.last_status}, sort_keys=True))


if __name__ == "__main__":
    main()
