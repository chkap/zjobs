# zjobs

A small Python/ZMQ streaming DAG runtime for trusted Python code. **Version
0.2 supports local processes and manually started static host agents**.
Independent stage scaling, sharded sources, bounded streaming, fan-out/fan-in,
and cancellation work today. Machine-based locality
(`local_first`, `balanced`, `local_only` and their precedence), replay/retries,
optimized large-payload transport, and distributed performance measurements
remain follow-on work. Network execution has been tested with separate agent
processes on one host, including a non-loopback endpoint. **Actual multi-machine
execution evidence is still outstanding**; host aliases are not physical hosts.

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
shared. The callable's defining module is shipped by value (also for imported
user functions/classes); other imported modules and third-party dependencies
must be installed on each host. Dynamic imports of uninstalled user modules
still fail explicitly. Code shipping uses cloudpickle's module-by-value
registration under a serialization lock and restores prior registration.
Install compatible Python and all third-party dependencies yourself.
Only trusted code/payloads are supported: deserialization executes Python code.

## Static hosts

Install the **same zjobs version, Python major/minor and cloudpickle version**
on every host (pin your deployment dependencies). Each agent serves one active
run at a time and stays alive for subsequent runs. Start agents manually in
separate terminals on their respective hosts:

```sh
# On host A; replace host-a.example with its reachable address:
zjobs-agent --machine host-a --bind tcp://0.0.0.0:5555 \
  --advertise tcp://host-a.example:5555
# On host B:
zjobs-agent --machine host-b --bind tcp://0.0.0.0:5555 \
  --advertise tcp://host-b.example:5555
# On the controller machine:
python examples/static_pipeline.py \
  --host-a tcp://host-a.example:5555 --host-b tcp://host-b.example:5555
```

`python -m zjobs.agent` is equivalent to `zjobs-agent`. The controller connects
outbound to agent TCP listeners; allow those ports through your trusted network
firewall. No controller listening port or SSH automation is needed. Listen
only on trusted interfaces: **there is no authentication, encryption or
untrusted-code isolation**. The agent prints a JSON readiness line with its
identity, advertised endpoint, capabilities and lease. `--advertise` is the
published reachable endpoint, not a discovery service; pass that endpoint to
`graph.run(agents=...)`. Control and payload frames currently share this single
public endpoint; workers use a separate agent-owned internal loopback endpoint.

For a two-agent smoke test on one machine, bind/advertise host-a to
`tcp://127.0.0.1:5555` and host-b to `tcp://127.0.0.1:5556`, and pass those
endpoints to the example. This exercises real processes/TCP but is not evidence
of execution on distinct machines.

Use stage `placement` instead of `replicas` when agents are configured:

```python
io = graph.source("io", load, placement={"host-a": 8, "host-b": 8})
graph.map("normalize", Normalize, inputs=[io], placement={"host-b": 2})
with graph.run(agents={"host-a": "tcp://host-a.example:5555",
                       "host-b": "tcp://host-b.example:5555"}) as run:
    print(run.placement)  # tuple of frozen ReplicaPlacement records
    for batch in run.batches(12):
        print(batch)
```

Every stage in an agent run requires explicit placement; missing agents,
duplicate endpoint strings, identity mismatches, incompatible
protocol/Python/cloudpickle/transport capabilities, busy agents and capacity
violations fail explicitly. Machine identity is explicitly configured,
not inferred from numeric ranks or hostnames. Operators must give each
machine a unique identity; we cannot prove physical distinctness from names.
Placement dictionaries and endpoint mappings are copied at startup. Hosts
are sorted by identity, then local rank, to assign deterministic contiguous
global ranks separately per stage. `rank` / `replicas` (also exposed as
`global_rank` / `global_replicas`) span the entire stage placement;
`machine`, `local_rank`, and `local_replicas` describe the local partition.
Eight hosts each placed with count 8 therefore define 64 global source ranks;
`context.shard` partitions once across all 64, not once per host.

An agent defaults to `--max-workers 256` (sum across all stages on that host)
and `--lease-timeout 3` seconds (minimum 0.5). Controller heartbeats keep the
lease alive even when user code blocks. If the controller disappears, the
agent releases workers after lease expiry plus bounded shutdown grace.
SIGINT/SIGTERM cleans agent-owned workers and stops the agent. Workers also
watch the agent's parent PID and exit when the agent dies abruptly, including
during user-code execution. This process-parent check is tested on Linux;
OS reaping of already-exited processes after SIGKILL belongs to the host init.
No agent process is owned or stopped by `run.close()`; it releases only that
run's workers. Concurrent runs on the same agent are rejected, not queued.

**Temporary data path:** agents launch/monitor/forward their local workers, but
all output bytes still travel through the controller before scheduling another
stage. This is deliberately the static-deployment milestone, not a locality
implementation. Even same-agent producers/consumers currently round-trip through
the controller. There is no locality-policy API or transfer-reduction claim.
The immutable machine placement and UUID-scoped run/worker identities support
the next host-owned payload/edge-obligation milestone without changing graph
definitions; the internal protocol will evolve with that implementation.

## Execution semantics

Build nodes in topological order with unique names. Each source is called once
per replica with a `SourceContext` and returns an iterable.
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
| `agents` | None | Machine identity -> reachable TCP endpoint; None launches locally |
| `agent_timeout` | 3 seconds | Registration deadline and maximum agent heartbeat silence |

A background controller owns its ZMQ sockets; each local worker has a
DEALER connected via loopback TCP. The controller maintains bounded stage
queues, one outstanding work/result credit per replica, and a bounded final
queue. Serialized fan-out data is reused in controller memory; transport uses
ordinary cloudpickle protocol-5 bytes (not shared memory or zero-copy array
frames). The controller rotates scheduling order; no stage input ordering is
promised. Agent relays retain only bounded in-flight protocol frames; work
credits, not heartbeats, permit payload production. Agent launch as a whole
must finish within `worker_timeout`, followed by per-worker initialization/call
deadlines. Cleanup requests stop on every contacted agent and waits a finite
`agent_timeout + 1.5` seconds for acknowledgments (each transport send also
has a finite one-second deadline); missing acknowledgments surface as errors,
with disconnected agents independently cleaning after their last lease expires.
This intentionally simple central broker favors correctness and backpressure
over peak throughput.

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
stage, rank, machine, work ID and operation and retain a traceback. `run.status` exposes
state, error, worker PIDs, produced/completed-work counts, current/peak stage
queues, current/peak final buffering, immutable placement coordinates and
machine-qualified worker PID records (PIDs alone are not globally unique).
No durable recovery is provided.
There are **no retries in this slice**: a lost worker fails the whole run,
even if replay might be possible. Future retries will use at-least-once
execution, not exactly-once external side effects. Resubmitting a run can repeat
source work and side effects. Per-worker state and source iteration are lost
on failure; there is no checkpoint or restoration promise. Tune finite
deadlines for your workload; a long valid call beyond its deadline fails.

Workers are trusted CPU Python processes. Cleanup owns only the controller,
its sockets/context, and its local or agent-launched worker processes; arbitrary
resources or subprocesses created by user code remain the user's responsibility.
