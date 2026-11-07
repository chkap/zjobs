"""Runnable bounded backpressure, early-exit and task-failure behavior."""
import argparse
import json
import time

from zjobs import Graph, Limits, RunError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["slow", "early", "error", "idle"])
    args = parser.parse_args()
    graph = Graph()
    root = graph.source("source", lambda ctx: range(12))
    def task(value):
        if args.mode == "error" and value == 3:
            raise ValueError("intentional task failure for fixture 3")
        return value * value
    graph.map("square", task, inputs=[root], replicas=2)
    budget = Limits(max_payload_bytes=65536, retained_items=2, retained_bytes=131072,
                    output_items=1, output_bytes=65536, consumer_timeout=.5)
    values = []
    error = None
    try:
        with graph.run(limits=budget) as stream:
            if args.mode == "idle":
                time.sleep(2)
            for value in stream:
                values.append(value)
                if args.mode == "early":
                    break
                if args.mode == "slow":
                    time.sleep(.05)
    except RunError as exc:
        error = str(exc)
    if args.mode == "slow":
        if error or sorted(values) != [i * i for i in range(12)]:
            raise RuntimeError(f"slow-consumer fixture failed: {error}, {values}")
    elif error is None:
        raise RuntimeError(f"{args.mode} must fail/cancel explicitly, not report success")
    status = stream.status
    if any(s["retained_bytes"] or s["retained_items"] for s in status["stages"].values()):
        raise RuntimeError("owned retention was not released")
    print(json.dumps({"mode": args.mode, "values": values, "expected_error": error,
                      "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
