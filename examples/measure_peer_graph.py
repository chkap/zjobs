"""Application-level overlap, byte and peer traffic evidence, not a socket benchmark."""
import argparse
import json
import resource
import time

from zjobs import Graph, Limits, Locality, PeerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", default="local")
    parser.add_argument("--peers", type=json.loads, default={})
    parser.add_argument("--run-id", default="measurement")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--collect-workers", type=int, default=1)
    parser.add_argument("--count", type=int, default=16, help="items per local IO rank")
    parser.add_argument("--payload-bytes", type=int, default=65536)
    parser.add_argument("--source-delay", type=float, default=.01)
    parser.add_argument("--cpu-seconds", type=float, default=.01)
    parser.add_argument("--transform-delay", type=float, default=.01)
    parser.add_argument("--locality", choices=["local_first", "balanced", "local_only"],
                        default="local_first")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    graph = Graph(locality=Locality(args.locality, max_local_streak=2))
    def read(ctx):
        for i in range(args.count):
            time.sleep(args.source_delay)
            yield (ctx.peer, ctx.rank, i, b"x" * args.payload_bytes)
    root = graph.source("io", read, replicas=args.io_workers)
    def preprocess(value):
        started = time.process_time()
        total = 0
        while time.process_time() - started < args.cpu_seconds:
            total += sum(value[3][:1024])
        return value[:3] + (len(value[3]),)
    mapped = graph.map("cpu", preprocess, inputs=[root], replicas=args.cpu_workers)
    def transform(value):
        time.sleep(args.transform_delay)
        return value
    transformed = graph.map("transform", transform, inputs=[mapped])
    graph.batch("collect", inputs=[transformed], size=8, replicas=args.collect_workers)
    peer = PeerConfig(args.identity, args.run_id, args.peers.get(args.identity), args.peers)
    plan = graph.describe(peer=peer)
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True))
        return
    limits = Limits(max_payload_bytes=1048576, retained_items=16, retained_bytes=16777216,
                    claim_bytes=8388608, output_bytes=8388608)
    started = time.monotonic()
    with graph.run(peer=peer, limits=limits) as stream:
        values = [value for batch in stream for value in batch]
    elapsed = time.monotonic() - started
    status = stream.status
    execution = [s["execution"] for s in status["stages"].values() if s["execution"]["attempts_completed"]]
    window = (max(s["last_finished"] for s in execution) -
              min(s["first_started"] for s in execution)) if execution else 0.
    busy = sum(s["worker_wall_seconds"] for s in execution)
    print(json.dumps({
        "plan": plan, "elapsed_seconds": elapsed, "values": values, "status": status,
        "execution_window_seconds": window, "summed_worker_wall_seconds": busy,
        "average_executing_workers": busy / window if window else 0.,
        "runtime_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "largest_child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "physical_host_count_demonstrated": 1,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
