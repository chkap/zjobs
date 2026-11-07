"""Same entrypoint on each independent peer; see README for separate commands."""
import argparse
import json
import time

from zjobs import Graph, Limits, Locality, PeerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("identity", choices=["A", "B", "C"])
    parser.add_argument("--base-port", type=int, default=5570)
    parser.add_argument("--run-id", default="peer-demo")
    parser.add_argument("--locality", choices=["local_first", "balanced", "local_only"],
                        default="local_first")
    parser.add_argument("--count", type=int, default=18)
    args = parser.parse_args()
    peers = {p: f"tcp://127.0.0.1:{args.base_port + i}" for i, p in enumerate("ABC")}
    identity = args.identity
    graph = Graph(locality=Locality(args.locality, max_local_streak=2))
    def read(ctx):
        time.sleep({"A": .01, "B": .4, "C": .8}[ctx.peer])
        for i in range(args.count):
            yield (ctx.peer, ctx.rank, i)
    raw = graph.source("io", read, replicas=2 if identity == "A" else 1)
    def decode(x):
        time.sleep({"A": .08, "B": .025, "C": .005}[identity])
        return x
    cpu = graph.map("cpu", decode, inputs=[raw], replicas=1 if identity == "A" else 2)
    def transform(x):
        time.sleep({"A": .002, "B": .06, "C": .015}[identity])
        return x
    cooked = graph.map("cooked", transform, inputs=[cpu])
    graph.batch("collect", inputs=[cooked], size=5)
    peer = PeerConfig(identity, args.run_id, peers[identity], peers)
    started = time.monotonic()
    with graph.run(peer=peer, limits=Limits(task_timeout=20)) as stream:
        values = [value for batch in stream for value in batch]
    print(json.dumps({
        "identity": identity, "values": values,
        "elapsed": time.monotonic() - started, "status": stream.status,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
