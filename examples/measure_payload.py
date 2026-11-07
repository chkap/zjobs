"""Compare installed releases using the same payload workload; prints JSON."""

import argparse
import importlib.metadata
import importlib.util
import json
import resource
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import cloudpickle

from zjobs import Graph


def make_value(kind, elements, index):
    if kind == "bytes":
        return bytes([index % 256]) * (elements * 4)
    if kind == "numpy":
        import numpy as np
        return np.full((elements // 1024, 1024), index, dtype=np.float32)
    import torch
    torch.set_num_threads(1)
    return torch.full((elements // 1024, 1024), float(index))


def preprocess(value):
    if isinstance(value, bytes):
        return value
    if type(value).__module__.startswith("torch"):
        import torch
        torch.set_num_threads(1)
    return value + 1


def verify(value, kind, elements, index):
    if kind == "bytes":
        assert value == bytes([index % 256]) * (elements * 4)
    elif kind == "numpy":
        import numpy as np
        assert value.shape == (elements // 1024, 1024)
        assert value.dtype == np.float32 and value.flags.c_contiguous
        assert np.all(value == index)
    else:
        import torch
        assert value.shape == (elements // 1024, 1024)
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        assert value.is_contiguous() and torch.all(value == index).item()


def serialize_trial(args, elements):
    try:
        from zjobs._payload import dumps, loads, size
        framed = True
    except ModuleNotFoundError as exc:
        if exc.name != "zjobs._payload":
            raise
        framed = False
    value = make_value(args.kind, elements, 3)
    encode_times, decode_times, allocations = [], [], []
    for _ in range(args.items):
        tracemalloc.start()
        start = time.perf_counter()
        payload = dumps(value, elements * 8 + 65536) if framed else cloudpickle.dumps(value, protocol=5)
        encode_times.append(time.perf_counter() - start)
        allocations.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        byte_count, frame_count = (size(payload), len(payload)) if framed else (len(payload), 1)
        start = time.perf_counter()
        result = loads(payload) if framed else cloudpickle.loads(payload)
        decode_times.append(time.perf_counter() - start)
        verify(result, args.kind, elements, 3)
        del payload, result
    return {
        "encode_seconds": encode_times, "decode_seconds": decode_times,
        "encode_peak_traced_bytes": allocations,
        "logical_payload_bytes": byte_count, "logical_payload_frames": frame_count,
    }


def start_agents(host, directory):
    processes, endpoints = [], {}
    try:
        for machine in ("a", "b"):
            with socket.socket() as sock:
                sock.bind((host, 0))
                endpoint = f"tcp://{host}:{sock.getsockname()[1]}"
            log = Path(directory) / f"{machine}.log"
            with log.open("w") as output:
                process = subprocess.Popen([
                    sys.executable, "-m", "zjobs.agent", "--machine", machine,
                    "--bind", endpoint, "--advertise", endpoint,
                ], stdout=output, stderr=subprocess.STDOUT)
            processes.append(process)
            deadline = time.monotonic() + 10
            while not log.read_text() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            lines = log.read_text().splitlines()
            if process.poll() is not None or not lines or json.loads(lines[0]).get("event") != "ready":
                raise RuntimeError(f"agent startup failed: {log.read_text()}")
            endpoints[machine] = endpoint
        return processes, endpoints
    except BaseException:
        stop_agents(processes)
        raise


def stop_agents(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def pipeline_trial(args, elements):
    # Imports are performed by each real worker, not replaced by socket-only
    # traffic. Include loading, transformation, collection and consumption.
    processes = []
    with tempfile.TemporaryDirectory(prefix="zjobs-payload-") as directory:
        try:
            endpoints = None
            if args.network_host:
                processes, endpoints = start_agents(args.network_host, directory)
            def placement(machine):
                return {"placement": {machine: 1}} if endpoints else {}
            graph = Graph()
            root = graph.source("load", lambda ctx: (
                make_value(args.kind, elements, i) for i in range(args.items)
            ), **placement("a"))
            mapped = graph.map("preprocess", preprocess, inputs=[root], **placement("b"))
            graph.batch("collect", inputs=[mapped], size=2,
                        max_bytes=elements * 8 + 65536, **placement("a"))
            limit = elements * 8 + 65536
            start = time.perf_counter()
            with graph.run(agents=endpoints, max_payload_bytes=limit,
                           retained_bytes=limit * 12, retained_items=12,
                           queue_bytes=limit * 4, output_bytes=limit * 2) as run:
                count = 0
                for batch in run:
                    for value in batch:
                        verify(value, args.kind, elements, count + (args.kind != "bytes"))
                        count += 1
                assert count == args.items
            elapsed = time.perf_counter() - start
            status = run.status
            assert status["obligations_created"] == status["obligations_settled"]
            assert all(value["retained_bytes"] == 0 for value in status["agents"].values())
            return {
                "elapsed_seconds": elapsed, "verified_items": count,
                "obligations": status["obligations_settled"],
                "peak_reserved_retained_bytes": status["peak_retained_bytes"],
                "peak_queue_bytes": status["peak_queue_bytes"],
                "peak_final_bytes": status["peak_output_bytes"],
                "agents": status["agents"], "remote_success_bytes": status["remote_bytes"],
                "final_transfer_bytes": status["final_bytes"],
            }
        finally:
            stop_agents(processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("bytes", "numpy", "torch"), default="numpy")
    parser.add_argument("--mode", choices=("serialize", "pipeline"), default="pipeline")
    parser.add_argument("--mib", type=int, default=8)
    parser.add_argument("--items", type=int, default=12)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--network-host", help="Bind two simulated host identities on this machine")
    parser.add_argument("--one-trial", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.mib, args.items, args.trials) < 1:
        parser.error("mib/items/trials must be positive")
    if args.one_trial:
        elements = args.mib * 1024 * 1024 // 4
        result = serialize_trial(args, elements) if args.mode == "serialize" else pipeline_trial(args, elements)
        result["controller_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports the maximum child peak, NOT summed concurrent RSS.
        result["max_child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        print(json.dumps(result))
        return
    results = []
    for _ in range(args.trials):
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--one-trial"]
        results.append(json.loads(subprocess.check_output(command, text=True)))
    output = {
        "version": importlib.metadata.version("zjobs"), "python": sys.version.split()[0],
        "kind": args.kind, "mode": args.mode, "mib": args.mib, "items": args.items,
        "network_host": args.network_host, "physical_hosts": 1,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("cloudpickle", "pyzmq", "numpy") if importlib.util.find_spec(name.replace("pyzmq", "zmq"))},
        "trials": results,
    }
    if args.kind == "torch":
        output["versions"]["torch"] = importlib.metadata.version("torch")
    if args.mode == "pipeline":
        output["median_elapsed_seconds"] = statistics.median(r["elapsed_seconds"] for r in results)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
