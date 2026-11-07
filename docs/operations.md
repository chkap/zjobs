# Operations, performance and troubleshooting

[Examples](examples.md) | [Architecture](architecture.md) | [API](api.md)

## Configure and launch

Install compatible Python/dependencies/application code on each machine. Choose
one fixed run ID and exact identity->TCP map, reachable listening addresses,
identical graph shape/edge batching and input corpus policy; local counts/
policies may differ. Launch the same entrypoint manually on every member within
join_timeout. No mandatory master/seed service exists. An owner is required only
as an equal fixed run member retaining its own output obligations.

Use a new run ID for resubmission and unused ports for concurrent runs. A new
process incarnation cannot take over active obligations. After failure wait for
owned cleanup before reusing endpoints; no active-run durability is provided.
For real hosts do not advertise `127.0.0.1` to other machines. Listening self
address must exist locally; use DNS/IP reachability and trusted firewall rules.
The practical example binds its advertised self endpoint directly; it does not
separately accept a wildcard bind address.

## Deadlines and failures

| Deadline | Default | What it bounds |
| --- | --- | --- |
| join_timeout | 15s | Missing fixed member/initial snapshots |
| heartbeat_timeout | 5s | Lost or stalled peer after first contact |
| claim_timeout | 5s | Pending grant/reject response only, not granted execution |
| lease_timeout | 5s | Fenced active owner claim; safely renewed under backpressure |
| task_timeout | 30s | Worker initialization and each source-next/map/batch attempt |
| consumer_timeout | 30s | No `next` progress while local output is buffered |
| shutdown_timeout | 10s | Reciprocal BYE observation at successful context shutdown |

Timeouts are configured detection budgets plus local polling/transport/cleanup
cost, not hard real-time guarantees under OS suspension or exhausted resources.
Long task execution requires task_timeout tuning; safely renewed active inputs
do not expire merely because claim_timeout is short. Normal backpressure stops
new pulls/claims but preserves active renewal. Abandoned consumer deadlines bound
slow/idle cases; terminal delivery is not redistributed to another machine.

RunError reports contextual peer, stage/rank or edge/token failure. Missing/
incompatible member, connection failure, payload too large, task/source error,
exhausted crash/stall retries, expired/lost ACK lease, congested required control
queue and cancellation are explicit failures. Failure gossip is best effort;
each survivor also has independent missing-peer detection. Partition cannot
silently ACK lost required outputs or produce global success.

Use `with graph.run(...) as stream:` and catch RunError outside. Early loop
break means unfinished work cancellation and **raises** on exit. StopIteration
means delivery/completion-ready; successful context exit still requires all peers
to close their contexts. Do not hold a successful peer context open indefinitely.
Processes owned by this run are cooperatively stopped then specifically
terminated/killed/joined if necessary; sockets and retention are released.
Linux parent-death signal plus parent watcher protects local children after
abrupt application death. The library never kills unrelated system processes.

## Diagnostics and tuning

`status.stages` reports actual worker PIDs/counts, effective rank policies, ready/
pending/in-flight/claimed work, production watermarks and buffer scopes. `buffers`
versus `buffer_peaks` distinguishes actual retained/received/grouped payloads from
result/transfer reservations. `peak_retained_bytes` is simultaneous retained +
result reservation accounting; `peak_claim_bytes` includes pending transfer +
received claims. Individual scope peaks may occur at different times, so adding
them is not an observed simultaneous maximum.

Local/peer items+bytes count receiver admissions by edge; `claims_granted` and
`claims_rejected` show producing-owner arbitration by child/requester. Stale/
duplicate controls are counted; they do not count as completed attempts.
Worker `execution` measures returned successful task/EOF wall/CPU intervals,
attempt start/completion counts and peak commanded attempts. A process doing
user IO may have low CPU utilization; result serialization counts toward task
wall/CPU but socket send wait/worker initialization do not. Crashed attempts lose
timing reports rather than inventing work performed.

Tune using actual application bottlenecks: increase local IO counts for slow IO,
CPU counts for parallel CPU, collection counts only when independent partial
groups are acceptable. Claim limits divide by ranks, so too many replicas can
make a large input impossible even when total bytes look generous. Increase
retained budgets or reduce maximum payload/reservations if stage execution is
blocked with available upstream work. Downstream user throughput sets sustainable
rate; additional buffers do not fix an abandoned consumer.

Observe source/CPU/transform overlap and compare local_only vs sharing with the
same explicit corpus accounting. Synthetic delays/byte fixtures are useful
reproducibility tools, not a promised real-workload speedup. Costs include Python
serialization/decode, immutable snapshots, process startup and peer transfers.
NumPy/compatible contiguous CPU tensors use multipart out-of-band frames rather
than concatenating huge messages. Noncontiguous/unusual tensor payloads can use
serialization fallback and need measurement. GPU transport is unsupported.

Parent maxrss and largest-child maxrss in measurement JSON are separately scoped
Linux peaks; their sum is not simultaneous process-tree memory. Decoded Python
objects, allocator overhead, user-retained output, task resources, metadata and
bounded ZMQ queues/copies remain outside serialized payload budgets.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| join deadline / missing peer | Launch every configured member; check identical run/map/shape, reachable self bind and ports |
| incompatible run / graph / membership | Match names/edges/batch semantics and run/map; callable meaning still application responsibility |
| multiprocessing re-import / recursive launch | Run a guarded Python script; do not start graph at import time |
| unpicklable callable/resource | Ship definitions/plain config; construct resources in each worker class |
| input has no eligible byte capacity | Increase per-stage claim bytes or reduce replicas/member size; check local_only route |
| group result exceeds maximum | Account output envelope/list overhead; reduce batch count/member cap or raise result maximum |
| low utilization with pending source results | Check reserved retention limits, slow downstream/consumer and local worker counts |
| frequent raced claims | Inspect owner rejection counts, snapshot eligibility and capacity; fairness is qualified scheduling opportunity |
| expected None missing | Do not filter None in your consumer; None is payload, not EOF |
| context raises after breaking loop | Expected explicit cancellation; catch RunError outside or consume/drain fully |
| shutdown acknowledgement deadline | Every successful peer must exit its context; READY iteration barrier is not BYE shutdown |
| stalled/crashed source | No transparent source replay; fix application and explicitly resubmit a new run |
| nested DataLoader worker | Adapter supports num_workers=0 only; graph already owns local worker replicas |

## Breaking migration boundary

Version 1 removes `zjobs-agent`, central coordinator/global shared Graph,
placement/remote agents and the old scheduler/protocol. There is no compatibility
wrapper. Replace deployment with a full local application on each member using
PeerConfig; final consumption is local and source/sample policy is explicit.
Old centralized test/measurement acceptance does not validate this architecture.
The independently reviewed new foundation/hardening/tasks are not integrated
goal acceptance: final #30 remains the separate acceptance gate.
