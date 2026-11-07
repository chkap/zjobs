"""Practical independently launched file -> CPU -> branch/union -> batch graph."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from zjobs import Graph, Limits, Locality, PeerConfig


def make_fixture(directory: Path, count: int):
    if count < 0:
        raise ValueError("fixture count must be nonnegative")
    directory.mkdir(parents=True, exist_ok=False)
    for index in range(count):
        (directory / f"{index:04d}.txt").write_text(
            f"record {index} has meaningful cpu text\n" * 16, encoding="utf-8")


def build_graph(directory, *, identity="local", locality="local_first",
                io_workers=2, cpu_workers=2, collect_workers=1,
                batch_size=5, rank_zero_local_only=False, delays=False):
    graph = Graph(locality=Locality(locality, max_local_streak=2))
    paths = sorted(Path(directory).glob("*.txt"))
    def read(ctx):
        if delays:
            time.sleep({"A": .01, "B": .2, "C": .4}.get(ctx.peer, 0))
        for index, path in enumerate(paths):
            # Application-defined two-level partition works with unequal local
            # IO rank counts, without global scheduling or runtime deduplication.
            if index % len(ctx.members) != ctx.peer_index:
                continue
            if (index // len(ctx.members)) % ctx.replicas != ctx.rank:
                continue
            yield (path.name, path.read_bytes())
    raw = graph.source("read", read, replicas=io_workers)
    class Normalize:
        def __init__(self):
            self.processed = 0
        def __call__(self, value):
            self.processed += 1
            if delays:
                time.sleep({"A": .08, "B": .025, "C": .005}.get(identity, 0))
            name, data = value
            words = data.decode("utf-8").lower().split()
            return (name, " ".join(words), len(words))
    overrides = {0: Locality("local_only")} if rank_zero_local_only else None
    normalized = graph.map("normalize", Normalize, inputs=[raw], replicas=cpu_workers,
                           rank_locality=overrides)
    def summarize(value):
        if delays:
            time.sleep({"A": .002, "B": .06, "C": .015}.get(identity, 0))
        return {"branch": "words", "file": value[0], "value": value[2]}
    def digest(value):
        if delays:
            time.sleep({"A": .002, "B": .06, "C": .015}.get(identity, 0))
        return {"branch": "sha256", "file": value[0],
                "value": hashlib.sha256(value[1].encode()).hexdigest()}
    words = graph.map("words", summarize, inputs=[normalized])
    hashes = graph.map("sha256", digest, inputs=[normalized])
    graph.batch("collect", inputs=[words, hashes], size=batch_size,
                max_bytes=65536, replicas=collect_workers)
    return graph


def limits():
    return Limits(max_payload_bytes=131072, retained_items=16, retained_bytes=2097152,
                  claim_bytes=1048576, output_items=4, output_bytes=524288)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--make-fixture", type=int, metavar="COUNT")
    parser.add_argument("--identity", default="local")
    parser.add_argument("--peers", type=json.loads, default={})
    parser.add_argument("--run-id", default="files")
    parser.add_argument("--locality", choices=["local_first", "balanced", "local_only"],
                        default="local_first")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--rank-zero-local-only", action="store_true")
    parser.add_argument("--delays", action="store_true")
    args = parser.parse_args()
    if args.make_fixture is not None:
        make_fixture(args.directory, args.make_fixture)
        print(json.dumps({"fixture": str(args.directory), "files": args.make_fixture}))
        return
    if not args.directory.is_dir():
        parser.error(f"input directory does not exist: {args.directory}")
    graph = build_graph(
        args.directory, identity=args.identity, locality=args.locality,
        io_workers=args.io_workers, cpu_workers=args.cpu_workers,
        collect_workers=args.collect_workers, batch_size=args.batch_size,
        rank_zero_local_only=args.rank_zero_local_only, delays=args.delays)
    peer = PeerConfig(args.identity, args.run_id, args.peers.get(args.identity), args.peers)
    batches = []
    with graph.run(peer=peer, limits=limits()) as stream:
        for batch in stream:
            batches.append(batch)
    print(json.dumps({
        "identity": args.identity, "batches": batches,
        "batch_sizes": [len(batch) for batch in batches],
        "status": stream.status,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
