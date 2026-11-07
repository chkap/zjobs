# zjobs

A small Python/ZMQ streaming DAG runtime for trusted Python code. **Version
0.6 supports local processes, static host agents, host-owned locality,
bounded replay, independently placed batching and framed large payloads**.
Independent stage scaling, sharded sources, bounded streaming, fan-out/fan-in,
and cancellation work today. Machine-based locality
(`local_first`, `balanced`, `local_only` and their precedence) uses actual
agent-retained payload availability. Application scaling/examples remain
follow-on work. Network execution has been tested with separate agent
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
`graph.run(agents=...)`. Control and direct peer fetch requests share this
single public endpoint;
workers use a separate agent-owned internal loopback endpoint. Agents must
also reach each other's advertised endpoints; the controller does not forward
intermediate payloads.

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

**Host-owned data path:** agents retain successful serialized outputs. The
controller receives only output ID/owner/size metadata and selects a consumer.
For a same-machine assignment, the agent delivers its retained bytes directly
to a local worker. A remote consumer agent fetches bytes directly from the
producer agent; intermediate bytes never traverse the controller. Terminal
outputs alone are fetched by the controller into its bounded final queue.
Fan-out reuses the producer's serialized buffer, not one stored copy per edge.
Protocol version 4 / `host-owned-v4` is incompatible with the earlier agents;
upgrade all participants together.

## Locality and required delivery

`Graph(policy="local_first")` is the default. Each source/map stage may set
`policy=...` and `rank_policies={global_stage_rank: policy}`. Precedence is
**rank > stage > graph**. Resolved policies appear in `run.placement` and status;
they are fixed for the run. Source policies are recorded but sources have no
incoming work to schedule. Local means the successful payload is retained by
the same configured machine agent, not an equal numeric rank.

| Policy | Selection among ready, eligible work |
| --- | --- |
| `local_first` | Prefer same-machine ready work; use remote immediately if local unavailable |
| `balanced` | Oldest admitted eligible item, rotating ready replica order, no machine preference |
| `local_only` | Only same-machine retained inputs; no remote fallback |

For `local_first`, `remote_after` defaults to 8: after at most that many
consecutive local selections by an eligible replica, its next selection takes
the oldest ready remote item if present. The counter is per replica and resets
on remote selection. Thus an oldest eligible remote obligation cannot be
indefinitely bypassed by a replenished local queue: among R eligible ranks,
at most R * (`remote_after` + 1) such selections exhaust their local budgets.
This is a selection bound, not a wall-time promise; task deadlines bound
individual calls. Balanced scheduling admits work only to ready replicas with
output capacity; faster ranks therefore perform more work. No policy bypasses
retention, stage-queue or final-output capacity.
Ready local matches are scheduled before remote-fallback ranks so a rotating
rank order does not steal a payload away from an available same-machine
consumer. A rank whose remote budget is exhausted still requests oldest remote
work on its next eligible selection.

```python
graph = Graph(policy="balanced")
io = graph.source("io", load, placement={"host-a": 2, "host-b": 2})
graph.map("normalize", Normalize, inputs=[io],
          placement={"host-a": 2, "host-b": 2}, policy="local_first",
          rank_policies={0: "local_only"})
```

Before launching workers, each producer-machine/downstream-stage edge must
have at least one eligible rank. An impossible `local_only` route raises
`RunError` rather than spilling or dropping data, even for a source that might
ultimately be empty. An agent disconnect fails the active run; it never causes
an ineligible local-only transfer. Replacement stays on the same machine and
rank with the same effective policy; placement does not change during recovery.

Every successful output has a run-scoped logical ID and one required
obligation for each downstream **stage**, or one final-queue obligation for
a terminal stage. Stage replicas compete for each obligation. Dispatch retains
the input: only a successful consumer result settles that edge. The producer
retains bytes until every branch settles, including branches already dispatched.
An error in one branch fails the run even if another branch finished.
Terminal settlement means acceptance into the bounded final queue, not an
external durable consumption acknowledgment. Completion requires all stages
finished, no active output/edge obligations and final transfers settled.

## Replay and worker replacement

Map calls whose worker **crashes or exceeds its finite call deadline** are
replayed from the retained input, on a replacement process at the same
stage/global rank/machine. `max_retries=2` permits at most three executions of
one logical map input on that rank. `max_retries=0` disables call replay.
`max_worker_restarts=16` also caps total replacements per rank per run,
including failures while idle; zero disables all replacement. Exhausting
either budget fails explicitly with stage, rank, machine, logical input ID,
work ID, attempt index and worker generation.

The input obligation and reserved output capacity remain active across retries,
not re-enqueued/double-reserved. A replacement initializes a fresh callable
class/instance from the original shipped definition. **State from the failed
worker is lost**, including user resources and any counters; already completed
work's arbitrary effects cannot be undone. These are at-least-once attempts,
not exactly-once external effects. User tasks must tolerate repeated execution
or provide their own idempotency.

Every replacement gets a fresh generation-scoped ZMQ identity. Results/errors
from retired identities cannot settle current obligations. Agents retain a
result before notifying the controller; controller acknowledgment marks an
accepted output. Replacement removes the failed worker's *unacknowledged*
results, while accepted outputs remain retained through all downstream
obligations even if their producing worker dies. Losing a publication before
acknowledgment can therefore replay the map call; a late publication from
the old generation is discarded. Partial fan-out retry re-executes only the
failed consuming edge, not already settled siblings. Final queue acceptance
is still the terminal accounting boundary, not a durable delivery guarantee.

Sources are **not replayable** in this version. A crash/stall of an active
source iterator fails explicitly rather than starting an arbitrary generator
again and silently skipping/repeating source partitions. Source exceptions,
dependency/import/serialization failures, callable initialization errors, and
ordinary user exceptions are contextual failures, **not retried**. A
completed source's accepted outputs remain available after its worker exits.
Agent/machine loss fails the whole run: the producer payload store may be gone,
so replacement on another host cannot guarantee required inputs. Controller
or agent restart requires resubmission; there is no durable active-run recovery.

Replacement first terminates/joins the failed owned process, then starts its
successor. Cleanup still respects item/byte credits, consumer abandonment,
cancellation and controller leases. A blocked or crashed replacement must
finish initialization within `worker_timeout`; initialization failure is final
instead of an unbounded restart loop. Healthy agent connectivity and compatible
environments remain prerequisites for replay. Retry warnings retain at most 32
contextual failures in `status["retry_errors"]`; counters distinguish attempts,
retries, replacements, stale messages and successfully settled logical edges.

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

`run.batches(size)` still groups final outputs at the consumer, retains a final
partial group, and yields nothing for empty input. It is separate from
`Graph.batch` below: wrapping an in-graph batch iterator in `run.batches` creates
lists of batches, not an implicit flatten. Consumer-side grouping memory is
additional to runtime buffers.

## Independently placed in-graph batches

```python
graph = Graph()
io = graph.source("io", load, placement={"host-a": 2})
groups = graph.batch("collect", inputs=[io], size=8, max_bytes=8 * 1024 * 1024,
                     placement={"host-b": 2}, policy="local_first")
graph.map("consume", lambda batch: [item["value"] for item in batch],
          inputs=[groups], placement={"host-b": 1})
with graph.run(agents={"host-a": "tcp://host-a.example:5555",
                       "host-b": "tcp://host-b.example:5555"}) as run:
    for batch in run:
        print(batch)
```

For a runnable small demo, `python examples/placed_batch.py` launches locally.
With manually started agents named host-a/host-b, add `--host-a <endpoint>
--host-b <endpoint>`. This demo validates 23 members across two source ranks,
two independently placed collection ranks and one downstream rank; it is not
the later full data-loading/application measurement example.

`graph.batch(name, inputs=[...], size=N, max_bytes=B, replicas=...,
placement=..., policy=..., rank_policies=...)` emits ordinary **list payloads**
that downstream maps can consume. Use either local replicas or placement as
for maps. It is a narrow built-in list primitive, not arbitrary stateful
accumulation or checkpointing. Members from multiple parents interleave
without ordering, keyed joins or zip/barrier semantics. Every incoming edge
member belongs to exactly one group for this batch stage. `None` is a valid
member, not a completion marker.

Each replica has at most one scheduler-owned partial/active group, at most
`size` members and `max_bytes` aggregate **serialized member bytes**. A full
group dispatches immediately. A non-full group also dispatches when the next
eligible member would exceed its byte bound; an individual member exceeding
the bound fails explicitly. After all parents complete, every remaining
nonempty group flushes. Empty input produces no list. With parallel replicas,
each may emit its own final partial list; **globally maximal batches and a
single final partial are not promised**. A serialized list can be larger than
the sum of member frames due to list/type metadata; it must also fit the run's
`max_payload_bytes` or fails explicitly. Python decoded object size is not
the serialized byte bound.

Policies select eligible members *before* joining them to the replica's group.
`local_only` groups contain only payloads available on the chosen machine;
`local_first` uses ready local members preferentially but may combine owners
when locality/fair remote selection requests it; `balanced` interleaves admitted
members without machine preference. Rank > stage > graph precedence and the
finite remote budget remain fixed at startup. Members are fetched directly
to the selected collector agent, never through the controller.

The controller retains every member reference and its upstream edge obligation
while gathering, dispatching and retrying. Only an accepted batch output
settles **all** member obligations together. Unpublished group crash/stall
replays the same members on a fresh collector at the same machine/rank; stale
generations cannot settle them. Accepted batch outputs survive producing-worker
death without regrouping or re-settling members. A dropped branch, exhausted
retry, unavailable agent or non-replayable source fails the run, not successful
completion with missing members. Completion includes no unflushed group,
active group attempt or required downstream output.

Upstream producers feeding a batch stage may retain multiple outputs so
`size` can exceed upstream replica count. Each is still covered by machine
item/byte credits; ordinary map-only paths retain their single-output producer
credit. **Budget enough retention for gathered members plus reserved output
overlap**: a group of 8 on one host needs at least 8 retained member slots plus
an output slot, and enough bytes for members plus `max_payload_bytes`.
Parallel groups and branches add overlap. Too-small capacity fails finitely
with stalled required-obligation diagnostics rather than releasing members to
make room or hanging indefinitely.

`grouping_items` / `grouping_bytes` and their `peak_grouping_*` counters expose
aggregate member references per stage, bounded by replicas * size/max_bytes.
These include active/retrying groups; their payload bytes remain in the
existing retained-store credits. Cancellation/abandonment releases workers,
partial groups and retained payload stores. Last-active controller snapshots
may still show pending references after failure for diagnosis.

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
| `retained_items` | 128 | Per-machine retained-output/reserved-result item credits |
| `retained_bytes` | 1 GiB | Per-machine retained-output/reserved-result byte credits |
| `queue_bytes` | 256 MiB | Maximum logical input bytes queued per stage |
| `output_bytes` | 256 MiB | Maximum buffered/reserved final payload bytes |
| `remote_after` | 8 | Consecutive local selections before forced eligible remote selection |
| `max_retries` | 2 | Additional attempts for one retained map input after worker crash/stall |
| `max_worker_restarts` | 16 | Total replacements per stage rank per active run |

A background controller owns its ZMQ sockets; each local worker has a
DEALER connected via loopback TCP. The controller maintains bounded stage
queues, one outstanding work/result credit per replica, and a bounded final
queue. Each producer also has only one retained output credit until all
branches settle, preventing a fast source from filling a host while downstream
workers initialize. Serialized fan-out data is reused in producer-agent memory;
the controller queues references only. Transport uses a small protocol-5 pickle
envelope plus out-of-band buffers as one logical multipart payload; exact
built-in `bytes` bypass pickle wrapping. The controller rotates scheduling order; no stage input ordering is
promised. Agent payload stores and transfers are bounded; work
credits, not heartbeats, permit payload production. Agent launch as a whole
must finish within `worker_timeout`, followed by per-worker initialization/call
deadlines. Cleanup requests stop on every contacted agent and waits a finite
`agent_timeout + 1.5` seconds for acknowledgments (each transport send also
has a finite one-second deadline); missing acknowledgments surface as errors,
with disconnected agents independently cleaning after their last lease expires.
The centralized metadata scheduler favors correctness and backpressure over
peak scheduling throughput.

Before dispatch, the scheduler reserves `max_payload_bytes` plus one item
on the output machine; a successful result converts that reservation to its
actual serialized size. Retained inputs are *not* released at dispatch, so
input/output overlap is included. Choose `retained_items` and `retained_bytes`
to accommodate concurrently active ranks plus their retained inputs. Conservative
reservations can leave ranks idle; too-small budgets can prevent progress
(e.g. one retained item for a same-host source -> map pipeline cannot retain
an input and reserve its output simultaneously). Such a run fails finitely
with routing/capacity and pending-obligation diagnostics after `worker_timeout`,
not silent dropping or an indefinite hang. Queue and final byte limits must
each fit `max_payload_bytes`, as must machine retention. One final transfer
is reserved before its fetch, so a slow consumer cannot trigger unbounded
fetch buffering. Final consumer deadlines apply when either bytes or items
block admission.

Bounds are **item counts plus accepted serialized-payload limits**, not a
hard process RSS ceiling. Agent stores enforce configured bytes/items, while
per-rank input/result transport frames can coexist with retained buffers.
Python user code, serialization temporaries, decoded
objects, callable state, ZMQ framing, and consumer-retained outputs add memory.
Do not return unbounded objects or retain unbounded state.

### Large payloads and measured tradeoffs

Every logical payload includes its format marker, pickle envelope (when used),
and all buffer frames. Their **sum** counts against `max_payload_bytes`,
retained/queue/final byte budgets and aggregate batch member limits. A payload
has at most 1,024 frames (1,022 out-of-band buffers); excess frames fail explicitly.
Batch command frame directories and agent control envelopes are transport
metadata, not serialized member bytes. Transfer counters sum actual transmitted
logical payload frame bytes, excluding ZMQ wire headers and control traffic.
Fan-out and retries reuse the same immutable retained frame tuple until all
required edges settle. Byte limits remain logical credits, not total RSS limits.

Protocol-5-compatible contiguous NumPy arrays use buffers; other arrays/objects
use their normal pickle reducers. Array shape/dtype/values and C/Fortran layout
are preserved where the reducer supports them; noncontiguous NumPy slices
normally reconstruct contiguous, as with ordinary pickle. Writable buffers are
snapshotted at serialization and reconstructed into isolated writable storage
per consumer, so later user mutation cannot corrupt another branch or replay.
Read-only NumPy arrays remain read-only. This deliberately retains copies:
ZMQ uses its normal copying sends, not `copy=False`; **this is not zero-copy or
shared memory**.

Torch's default reducer in the measured release stays in-band. A narrow
fallback therefore frames plain contiguous CPU `torch.Tensor` values when
NumPy is installed, including scalar/empty, bfloat16 and complex tensors.
It preserves dtype, shape, values and `requires_grad`, not autograd history.
Tensor subclasses, custom attributes/hooks, noncontiguous/quantized/conjugate
layouts and tensors without NumPy use torch's existing reducer. They receive
no promised optimization. Core import/execution requires neither NumPy nor torch.

`examples/measure_payload.py` prints JSON with raw trial durations, correctness,
serialization allocations, controller/max-child peak RSS, retained/queue/final
peaks and agent-observed transfer bytes. Run the **same script** from separate
noneditable baseline/candidate installations:

```sh
# Install numpy for array probes; install CPU torch separately for tensor probes.
python examples/measure_payload.py --mode serialize --kind numpy --mib 16 --items 8 --trials 3
python examples/measure_payload.py --kind numpy --mib 8 --items 12 --trials 3 --network-host 10.0.0.4
# Repeat with --kind bytes and --kind torch. Choose a bindable local IP.
```

On one Linux physical host with Python 3.12, cloudpickle 3.1.2, pyzmq 27.3.0,
NumPy 2.5.4 and CPU torch 2.15.0, comparing accepted 0.5.0 (`6504bb9`) with
0.6.0: serialization-only median traced encoding peaks for 16 MiB values fell
from 18.00 MiB to 276 bytes for exact bytes, 18.00 to 16.01 MiB for NumPy,
and 34.01 to 16.01 MiB for contiguous tensors. These exclude already-created
user values and native allocations; they are not whole-process RSS.

The application probe actually loads twelve 8 MiB values, preprocesses them
(array/tensor addition), collects groups of two and verifies every final value.
One source on agent `a`, one preprocessing rank on `b`, one collector on `a`
forces two remote payload transfers, plus final delivery to the controller.
Agent startup precedes timing; worker startup/imports and consumption are timed.
Three fresh-process trials (seconds):

| Payload | 0.5.0 trials | 0.6.0 trials | Median max-child RSS, before / after |
| --- | --- | --- | --- |
| bytes | 0.728, 0.708, 0.708 | 0.607, 0.618, 0.633 | 131.8 / 108.0 MiB |
| NumPy | 0.961, 0.907, 0.879 | 0.931, 0.937, 1.046 | 143.0 / 127.0 MiB |
| CPU tensor | 4.662, 3.946, 3.923 | 3.907, 3.852, 3.853 | 354.2 / 322.8 MiB |

Linux child RSS is the maximum individual child peak, **not summed concurrent
RSS**. Controller RSS was essentially unchanged for bytes/arrays and lower for
tensors. NumPy was slightly slower despite lower allocation/child RSS: isolated
mutable reconstruction still copies buffers, and scheduling/transport dominate
this small pipeline. Torch imports dominate cold-run latency. There is no
universal speedup claim. Remote payload bytes remain approximately 192 MiB and
final bytes 96 MiB (metadata changes only); framing reduces allocation, not the
data required by placement. Store peaks vary with scheduling and remain under
credits; they do not measure serialization temporaries.
These non-loopback TCP trials use **two host identities on one physical
machine**, not physical multi-host or scaling evidence. Practical IO/scaling
examples and actual distinct-host evidence remain separate work.

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
stage, rank, machine, work ID and operation and retain a traceback.
`run.status` exposes
state, error, worker PIDs, produced/completed-work counts, current/peak stage
queues, current/peak final buffering, immutable placement coordinates and
machine-qualified worker PID records (PIDs alone are not globally unique).
`local_items`/`remote_items` and bytes count successful map executions;
`agents` reports actual local cache deliveries, direct remote fetch/served
bytes, terminal bytes and retained-store peaks. `recent_assignments` is a bounded
64-decision trace with output owner, consumer, policy and selection streak;
`pending_outputs` lists at most 32 unsettled logical outputs/edges. Controller
credit peaks include reservations as well as stored bytes. Agent-observed
transfer counters, not assignment labels alone, establish the data path.
After failure/cancellation, controller credit/edge counters retain the last
active snapshot for diagnosis; agent stop acknowledgments report released
payload stores separately.
`active_attempts` lists at most 32 active initialization/call states, with
logical input/work IDs, attempt and generation. `attempts` counts all dispatched
calls (including source next/completion requests); `completed_work` counts
successful logical outputs, not retries. No durable recovery is provided.
Resubmitting a run can repeat source work and side effects. Per-worker state
and source iteration have no checkpoint/restoration promise. Tune finite
deadlines for your workload; a long valid map call beyond its deadline is
terminated and may be repeated within its retry budget.

Workers are trusted CPU Python processes. Cleanup owns only the controller,
its sockets/context, and its local or agent-launched worker processes; arbitrary
resources or subprocesses created by user code remain the user's responsibility.
