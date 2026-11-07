# zjobs

A small Python/ZMQ streaming DAG runtime for trusted Python code. **Version
0.1 is a local-process foundation**, not yet a multi-machine framework.
Independent stage scaling, sharded sources, bounded streaming, fan-out/fan-in,
and cancellation work today. Static host agents, machine-based locality
(`local_first`, `balanced`, `local_only` and their precedence), replay/retries,
optimized large-payload transport, and distributed performance measurements
remain follow-on work. No multi-machine execution evidence is claimed.

## Install and run

Use Python 3.10+ in a virtual environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python examples/local_pipeline.py
```

The core depends only on cloudpickle and pyzmq, not NumPy, torch, or GPUs.
For development: `python -m pip install -e '.[test]'` then `python -m pytest`.

```python
from zjobs import Graph


def load(context):
    # Shard cheap paths/IDs before doing IO or preprocessing.
    for index in context.shard(range(100)):
        yield {"index": index, "value": index * 2}


class Normalize:
    def __init__(self):
        self.count = 0  # private to this worker process

    def __call__(self, item):
        self.count += 1
        return item["value"] / 100


def main():
    graph = Graph()
    io = graph.source("io", load, replicas=4)
    graph.map("normalize", Normalize, inputs=[io], replicas=2)
    with graph.run(capacity=8, output_capacity=4) as run:
        for batch in run.batches(12):
            print(batch)  # final partial batch is retained


if __name__ == "__main__":
    main()
```

Run from a Python script with the `if __name__ == "__main__":` guard:
workers use multiprocessing **spawn**, not fork. Interactive notebook/REPL
launch is not supported in this slice. Functions, closures, callable instances,
and zero-argument callable classes are cloudpickle-shipped to workers. Classes
are constructed once per replica in that process; instances are copied, not
shared. Install compatible Python and all third-party dependencies yourself.
Only trusted code/payloads are supported: deserialization executes Python code.

## Execution semantics

Build nodes in topological order with unique names. Each source is called once
per replica with `SourceContext(rank, replicas)` and returns an iterable.
`context.shard(items)` selects indices modulo replica count; each replica still
enumerates the iterable, so **do not shard already-loaded expensive data**.
Use rank/replicas directly for partitioned external queries where appropriate.
Without explicit source sharding, replicas repeat whatever their source returns.

`graph.map(name, task, inputs=[...], replicas=N)` shares incoming work across N
processes. A task returns **one Python object** per input, including `None`,
bytes, containers, or a list. It does not implicitly flatten lists or iterate
generator results. Each logical output is broadcast once to *every downstream
stage*, not every replica. Fan-in interleaves parents without zip/join or order
guarantees. A slow branch backpressures the common producer. All terminal
stages feed one merged final iterator; use tagged values if provenance matters.
Sources and map stages execute concurrently, so outputs arrive before sources
are exhausted. Empty sources finish normally. Completion uses separate control
frames, never a special Python payload.

`run.batches(size)` batches at the consumer, yields a final partial batch,
and yields nothing for empty input. This is output collection, not a stateful
in-graph batching stage. Memory for a batch is additional to runtime buffers.

## Bounds, lifecycle, and errors

`graph.run` accepts:

| Option | Default | Meaning |
| --- | --- | --- |
| `capacity` | 16 | Maximum queued input items per stage, shared across its ranks |
| `output_capacity` | 16 | Maximum final items buffered for the consumer |
| `worker_timeout` | 30 seconds | Deadline for each worker initialization or source/map call |
| `consumer_timeout` | 30 seconds | Deadline without consumption while final buffer stays full |
| `max_payload_bytes` | 64 MiB | Maximum serialized callable or individual output size |

A background controller owns its ZMQ ROUTER socket; each local worker has a
DEALER connected via loopback TCP. The controller maintains bounded stage
queues, one outstanding work/result credit per replica, and a bounded final
queue. Serialized fan-out data is reused in controller memory; transport uses
ordinary cloudpickle protocol-5 bytes (not shared memory or zero-copy array
frames). The controller rotates scheduling order; no stage input ordering is
promised. This intentionally simple central broker favors correctness and
backpressure over peak throughput.

Bounds are **item counts plus accepted serialized-payload limits**, not a
hard process RSS ceiling. Python user code, serialization temporaries, decoded
objects, callable state, ZMQ framing, and consumer-retained outputs add memory.
Do not return unbounded objects or retain unbounded state. Arrays/CPU tensors
can be ordinary pickleable payloads when dependencies are installed, but this
slice makes no large-payload efficiency or performance claims.

Always use `with graph.run() as run:` or call `run.close()` in a `finally`
block. Early exit cancels the graph and joins/terminates its owned workers;
close is idempotent. An abandoned consumer with a full buffer fails after
`consumer_timeout` and cleans up **without another call to next**. A completed
run also cleans up even if its already-buffered results have not been consumed.
Errors invalidate remaining buffered outputs; results already consumed cannot
be rolled back. Iteration is single-consumer; `status` and `close()` may be
called from another thread.

User-code errors, missing dependencies, code/payload serialization errors,
worker crashes, and deadlines raise `RunError`. Worker errors identify the
stage, rank, work ID and operation and retain a traceback. `run.status` exposes
state, error, worker PIDs, produced/completed-work counts, current/peak stage
queues, and current/peak final buffering. No durable recovery is provided.
There are **no retries in this slice**: a lost worker fails the whole run,
even if replay might be possible. Future retries will use at-least-once
execution, not exactly-once external side effects. Resubmitting a run can repeat
source work and side effects. Per-worker state and source iteration are lost
on failure; there is no checkpoint or restoration promise. Tune finite
deadlines for your workload; a long valid call beyond its deadline fails.

Workers are trusted CPU Python processes. Cleanup owns only the controller,
its sockets/context, and its directly launched worker processes; arbitrary
resources or subprocesses created by user code remain the user's responsibility.
