# zjobs: independent local-first peer graphs

Python 3.10+ and ZeroMQ. Each application independently owns its **complete**
streaming DAG, local worker processes, retained outputs and local iterator.
Peers pull eligible upstream values directly from the producing owner. There
is no master, worker agent, metadata coordinator, stage broker or global graph.
Static logical identities describe locality, not proof of separate hardware.

## Breaking replacement

Version 1 replaces the previous central runtime. `zjobs-agent`, placement maps,
agent connections and the old runtime/protocol are removed without compatibility
facades. Old centralized tests/examples/measurements are not evidence for this
runtime. The immutable multipart payload codec is reused as a generic utility.

## Install and standalone quickstart

```sh
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python examples/local_pipeline.py
.venv/bin/python -m pytest -q
```

Core dependencies are cloudpickle and pyzmq, not numpy or torch. The test extra
installs numpy. Guard application launches with `if __name__ == "__main__"`:
workers use multiprocessing **spawn**, not inherited sockets or forked state.

```python
from zjobs import Graph, Limits, Locality

def main():
    g = Graph(locality=Locality())  # local_first, max_local_streak=8
    raw = g.source("read", lambda ctx: range(7), replicas=2)
    decoded = g.map("decode", lambda x: x * 2, inputs=[raw], replicas=3)
    g.batch("collect", inputs=[decoded], size=4, max_bytes=8 * 1024 * 1024)
    with g.run(limits=Limits()) as local_stream:
        for batch in local_stream:
            print(batch)
```

Each source rank invokes `source(SourceContext)` independently. This example
intentionally repeats `range(7)` twice: the runtime does not infer dataset
identity, partition inputs, deduplicate, synchronize epochs or broadcast final
outputs. SourceContext exposes `peer`, `rank`, `replicas`, sorted `members`, and
`peer_index`. Applications decide partitioning/repetition/filtering.

## Three independently launched peers

Run these in **three separate terminals**, in any order (within the join timeout):

```sh
.venv/bin/python examples/peer_pipeline.py A --run-id demo-1
.venv/bin/python examples/peer_pipeline.py B --run-id demo-1
.venv/bin/python examples/peer_pipeline.py C --run-id demo-1
```

Every command runs IO, CPU, transform, collection and its own consumer. There
is no service launch. A uses two IO ranks; B/C use one. Across the three JSON
results expect exactly 72 distinct `(source peer, source rank, sequence)` values,
not 72 per consumer. Delays encourage peer fallback on both IO->CPU and
CPU->transform. Observe `status.peer_items`/`peer_bytes`, local counterparts,
per-stage process IDs, effective policies, pending/in-flight claims and peaks.
User-output counts per peer are intentionally not fixed. Ordering is unspecified.

Use the same `--base-port` on all three commands to change the three endpoints.
Repeat with `--locality local_only` for no peer payload traffic, `balanced` for
no owner preference, or `--count 0` for symmetric empty completion. Choose a new
run ID for a new run; do not overlap runs on the same listening endpoints.
For real hosts replace the endpoint map in the example with reachable IPs,
bind each corresponding listen address, install the same code/dependencies and
launch the same entrypoint separately. These examples demonstrate logical peers
on one physical host, not multi-host throughput.

## Graph and task API

`Graph.source(name, task, replicas=1)`, `map(name, task, inputs=[...], replicas=1)`
and `batch(name, inputs=[...], size=N, max_bytes=8 MiB, replicas=1)` return stages.
Every local stage has positive replicas; there are no remotely placed stages.
Each parent value has one required obligation per child stage, with replicas
across eligible peers competing for that obligation. Branches get independent
deliveries, not competing branches. Multiple parents form an unordered union,
not a keyed join, zip or barrier. Every returned map value, including `None`,
is data. Empty sources produce no data. Terminal outputs stay on their producing
peer and are consumed only by that peer's local iterator.

Batch replicas form independent groups; they do not share a global batch.
Groups flush on item count, serialized member byte cap, final production seal
with no remaining available input, or upstream retention pressure. Pressure can
therefore create early partial groups when buffers cannot hold the requested
batch; empty input emits no empty batch. Batch frame-directory/serialized result
overhead still must fit `max_payload_bytes`, independently of member byte limits.

Functions, closures, callable instances and no-argument callable classes are
automatically cloudpickled **only into local children**. Classes are constructed
in each worker, suitable for opening files/loading CPU resources there. Instances
are cloned per worker; state is private and survives successive calls, but is
lost on replacement. Do not capture open sockets, locks, GPU state or
unserializable resources. Deploy matching imports/dependencies and semantically
equivalent callables on every peer; handshake checks graph shape, batching and
exact membership/run identity, not arbitrary callable semantic equality.
Code/pickle inputs are trusted; this is not an authenticated public service.

## Locality and fairness

Supply `locality=Locality("local_first" | "balanced" | "local_only",
max_local_streak=8)` on the graph or stage, and
`rank_locality={0: Locality(...)}` on a stage. Precedence is rank > stage > graph.
Local means payloads in this application's same-machine store, not matching rank.
`local_first` admits ready local work, immediately falls back when absent, and
gives the remote class an opportunity after at most L local admissions. With P
continuously eligible remote owners, rotating choices give an owner an
opportunity within P*(L+1) admissions at that receiver rank, **assuming unchanged
eligibility, sufficient byte/worker capacity and no raced/rejected claims**.
Rejections rotate owners and reschedule; they are not successful admissions.
Owner-side arbitration also rotates pending claimants with a finite local
streak. Local requests enter each arbitration round before queued peer requests
are granted; no distinguished remote queue runs ahead of local scheduling.
Peer/rank candidate cursors spread requests across ready items to avoid every
receiver racing the oldest snapshot candidate. This is not static partitioning:
all eligible replicas still compete, and rejection rotates/reschedules.
There is no unconditional wall-time or per-owner successful-grant bound
under competing requests, slow workers, changing offers or backpressure.
`balanced` uses round-robin ready owners that fit available capacity; no numeric
rank/machine preference. `local_only` never requests peer payloads. Requiring all
stages locally with positive ranks prevents impossible missing-local-stage
routing. Receivers skip candidates larger than their individual byte capacity;
owners fail contextually if no eligible local/peer receiver can deliver a
required output.

## Limits and memory

`Limits` defaults:

| Setting | Default | Scope |
| --- | --- | --- |
| max_payload_bytes | 8 MiB | serialized result, including multipart envelope |
| retained_items / retained_bytes | 32 / 64 MiB | **per local stage**, including result reservations |
| claim_items / claim_bytes | 64 / 64 MiB | per stage, divided between its local replicas |
| output_items / output_bytes | 8 / 64 MiB | combined local iterator buffer |
| task_timeout / consumer_timeout | 30s / 30s | task/startup deadline and pending-output consumer idle |
| shutdown_timeout | 10s | symmetric context shutdown |
| max_retries | 0 | local map/batch crash/stall replacements |

Unknown-size source/map results reserve one item and the full maximum payload
**before** execution. Admission releases the unused bytes. Each worker has at
most one result/command in its bounded multipart channel. Each remote claim
advertises an individual serialized size and reserves concrete item/byte credit
before transfer. Receivers advertise individual capacity/policy at handshake;
owners validate that each required output fits an eligible destination. An
oversized member cannot be silently dropped or hang waiting for impossible
routing. Result reservations are stage-local
to avoid source credits exhausting every downstream result slot. Sources pull
one `next` per granted reservation; downstream execution overlaps live sources.
Increase stage retention/claim budgets or reduce replicas/payloads if necessary.

NumPy and supported contiguous CPU tensors use pickle-5 out-of-band frames;
ordinary bytes have their own frame. Writable buffers are snapshotted, and each
decode gets independent mutable storage for fan-out/retry isolation. CPU tensor
support is optional and does not import torch into core startup.

These are serialized accounting bounds, **not a hard RSS ceiling**. Decoded
objects, callable state, Python serialization allocations, frame directories,
metadata, bounded ZMQ transport queues/copies, allocator overhead, and values
retained by user code are additional. Multipart payloads are not concatenated
for peer or local-worker transport. Control snapshots are bounded by retained
items and a 1 MiB header cap; transport HWM is finite per connection.

## Ownership, failure and completion

Each peer's store is authoritative only for outputs it produced. Each outgoing
edge transitions available -> token-fenced claimed -> acknowledged. The owner
retains payloads until **all** child obligations settle. Input ACK occurs only
after the downstream worker result is admitted to its own bounded retained
store, including all members of a successful batch. Map crash/stall retries
retain and renew the original input claim, kill/join the old process, and fence
the replacement attempt. Repeated Python effects are possible. There is no
exactly-once external side-effect guarantee, durable replay or active-run restart.

`PeerConfig(identity="local", run_id="standalone", listen=None, peers={},
join_timeout=15, heartbeat_timeout=5, lease_timeout=5, claim_timeout=5)` configures the run.
For peers, the map includes **all** members, including self with its listen
endpoint. Different local replica counts are valid. Incarnation replacement,
incompatible membership/graph, missing peers, peer loss, stalled/crashed
non-replayable sources, task exceptions, exhausted retries, oversized results,
expired leases and congested required-message queues explicitly fail the run.
Lease expiry fails instead of silently reissuing already committed work.
Renewals continue for active input/batch claims even under consumer backpressure.
Pending claims have an independent `claim_timeout` for receiving grant/reject;
heartbeats and safe lease renewals cannot extend a missing response forever.
Duplicate live claim tokens are ignored without retransferring the payload.
Peer failure gossip is best effort; each survivor independently detects missing
heartbeats. A partition does not produce success-shaped completion.

Production EOF/seals and final per-owner/stage output-count watermarks are
**separate from delivery drain**. Monotonic numbered full snapshots carry
available candidates and outstanding edge obligations together with those seals.
Partial batches can
flush once all parent owners sealed production and no eligible available input
remains; other replicas' assigned claims cannot join that group. Non-source
production seals propagate only after parent edge obligations drained and local
attempts settled. Success requires all local stores/consumer obligations drained,
all peers' retained COMPLETE_READY records, and reciprocal observation ACKs.
`StopIteration` signifies that delivery/completion barrier. The runtime remains
alive until context exit, when peers symmetrically announce/acknowledge shutdown
and linger to retransmit final observations. Missing shutdown members fail by
deadline; do not leave a successful iterator's context open indefinitely.

Use the context manager. Breaking out early cancels and raises contextual
`RunError`; peers fail too. A consumer with pending buffered output that makes
no `next` progress for `consumer_timeout` explicitly fails. Slow consumption
within that deadline backpressures the graph. Worker task deadlines also apply
to user initialization and source `next`. Owned workers are stopped/joined,
sockets closed and retention released on normal exit or failure. Catch
`RunError` outside the context for intentional cancellation/failure handling.
Local children monitor parent loss; Linux additionally sets a parent-death
signal so abrupt peer/application death does not leave its workers running.

## Validation and next delivery gates

### Diagnostics and application overlap

`Graph.describe(peer=...)` gives the local launch plan without launching or
contacting peers: stage counts, effective rank policies, inputs, membership and
fingerprint. It is not a global scheduler. `Run.status.stages[name].buffers`
separates actual retained payloads, maximum-size result reservations, received
claims, pending transfer reservations and grouped batch bytes/items.
`buffer_peaks` retains their individual peaks, while `peak_retained_bytes`
includes simultaneous actual + result reservations and `peak_claim_bytes`
includes received + pending claims. Individual peaks need not occur together;
do not add them and report the sum as an observed memory peak. Iterator frames
reference retained snapshots, so those two accounting scopes are not additive
unique allocations. Transport multipart/envelope/decoded memory caveats above
still apply.

Per-stage `execution` reports started/completed attempts, peak commanded
in-flight workers, accumulated **worker-measured** task wall/CPU time and first/
last monotonic timestamps. Wall time includes decode, user work and result
serialization, not worker initialization or socket send wait. CPU is the worker's
process CPU clock. Crashed attempts cannot report their lost execution time;
started counts include retries, completed counts do not invent those samples.
`peak_in_flight` is command capacity, not a claim of continuous CPU utilization.
Owner `claims_granted`/`claims_rejected` counts are by child and requester, while
local/peer traffic measures successful receiver admissions.

```sh
.venv/bin/python examples/measure_peer_graph.py
```

This runs a real timed source -> CPU preprocessing -> transform -> collection -> local
consumer workload with 64 KiB byte payloads. JSON includes actual distinct PIDs,
accounted buffer peaks, traffic, worker CPU/wall time, execution window and
summed worker wall/window ratio. A ratio above one proves overlapping reported
worker intervals, not a universal speedup. Linux maxrss is in KiB; parent peak
and **largest child** peak are separately reported, not summed as simultaneous
total process RSS.

Launch the same measurement entrypoint independently on A/B/C:

```sh
PEERS='{"A":"tcp://127.0.0.1:5580","B":"tcp://127.0.0.1:5581","C":"tcp://127.0.0.1:5582"}'
.venv/bin/python examples/measure_peer_graph.py --identity A --peers "$PEERS" --run-id measure-1 --io-workers 2 --source-delay .005 --cpu-workers 1 --cpu-seconds .03 --transform-delay .002
.venv/bin/python examples/measure_peer_graph.py --identity B --peers "$PEERS" --run-id measure-1 --io-workers 1 --source-delay .03 --cpu-workers 2 --cpu-seconds .01 --transform-delay .04
.venv/bin/python examples/measure_peer_graph.py --identity C --peers "$PEERS" --run-id measure-1 --io-workers 1 --source-delay .05 --cpu-workers 2 --cpu-seconds .005 --transform-delay .01
```

Run the three commands in separate terminals. Expect 64 unique coordinates in
the union of local consumers, every payload length 65536; worker distribution
is local and timing/consumer ownership are nondeterministic. Repeat with
`--locality local_only` and a new run ID/ports for application-level comparison.
Synthetic delays model asymmetric readiness, not real disk throughput; practical
file IO and optional torch examples remain separate delivery work.

For an **8 machines x 8 IO ranks** configuration demonstration without pretending
to launch 64 concurrent workers or requiring eight physical hosts:

```sh
PEERS='{"A":"tcp://127.0.0.1:5590","B":"tcp://127.0.0.1:5591","C":"tcp://127.0.0.1:5592","D":"tcp://127.0.0.1:5593","E":"tcp://127.0.0.1:5594","F":"tcp://127.0.0.1:5595","G":"tcp://127.0.0.1:5596","H":"tcp://127.0.0.1:5597"}'
for PEER in A B C D E F G H; do
    .venv/bin/python examples/measure_peer_graph.py --identity "$PEER" --peers "$PEERS" --io-workers 8 --plan-only
done
```

Each independent plan reports eight local source replicas (64 across eight
plans), one transform, plus separately configurable CPU/collection counts. `--plan-only` starts
no workers or network service. The actual A/B/C resource-bounded launch above
checks the same configuration/API path without claiming physical eight-host
execution.

`tests/test_peer_graph.py` covers standalone streaming, independent peer launches,
accounting, policies, branches/union, batches, multipart isolation, retries,
deadlines and cleanup. Task #27 is the replacement foundation, not full goal
acceptance. #28 extends adversarial delivery/failure/fairness tests; #29 provides
practical file IO, optional IterableDataset, richer diagnostics/performance
measurements and additional exercised examples; #30 independently verifies the
complete goal. Old benchmark results are deliberately removed.
