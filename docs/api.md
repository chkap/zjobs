# API, task shipping and local resources

[Examples](examples.md) | [Architecture](architecture.md) | [Operations](operations.md)

## Definition and ownership

`from zjobs import Graph, Stage, Locality, SourceContext, PeerConfig, Limits,
Run, RunError`

| API | Defaults and behavior |
| --- | --- |
| `Graph(locality=Locality())` | One locally defined streaming DAG; insertion inputs must already belong to graph |
| `source(name, task, replicas=1, locality=None, rank_locality=None)` | `task(SourceContext)` returns iterable; each local rank invokes independently |
| `map(name, task, inputs=[...], replicas=1, locality=None, rank_locality=None)` | One result per selected input, including None; unordered union for parents |
| `batch(name, inputs=[...], size=N, max_bytes=8 MiB, replicas=1, locality=None, rank_locality=None)` | Bounded independent per-rank groups; list output; pressure/byte/final partial; no empty group |
| `graph.policy(stage, rank)` | Rank override > stage > graph |
| `graph.describe(peer=PeerConfig())` | No-launch local counts/policies/input/fingerprint plan |
| `graph.run(peer=PeerConfig(), limits=Limits())` | Starts runtime/children; use returned Run as context and one local iterator |
| `run.status` | Thread-safe copied diagnostic snapshot |
| `run.close()` | Cancel unfinished run or participate in symmetric successful shutdown; can raise RunError |

Stages require positive local replica counts, unique nonempty names (excluding
reserved `@consumer`), distinct earlier same-graph parents and callable map/source
tasks. Cycles, ordered output, keyed joins/zip and stages placed only remotely
are unsupported. Do not mutate the graph after launch; run snapshots its local
definition. Every application launches all stages; different local counts do
not change the shared graph shape.

## Policies

`Locality(mode="local_first", max_local_streak=8)` accepts `local_first`,
`balanced`, `local_only`. Overrides use local integer ranks within that stage's
replica range: `rank_locality={0: Locality("local_only")}`. Policies are about
same-machine payload availability, never numeric source/consumer rank equality.

Local-first uses ready local work unless the remote-class finite streak
opportunity is due; absent local candidates, ready peers are tried immediately
without waiting for future local production. Balanced rotates ready owners that
fit capacity without machine preference; it is not load prediction/work-cost
scheduling. Local-only makes no remote requests. Required outputs with no
eligible capacity fail at owner publication instead of hanging/dropping.

With L local streak and P continuously eligible remote owners, rotating receiver
opportunities are bounded by P*(L+1) admissions **under stable eligibility, worker/
byte capacity and no rejected/raced claims**. Owner grant arbitration rotates
pending requesters too; dynamic pending sets and rejection races do not imply
a universal per-owner success/wall-time bound. Observe selections versus
admissions/grants/rejections separately, not just totals.

## Source and sample policy

SourceContext fields: `peer`, local `rank`, local `replicas`, sorted `members`;
`peer_index` is its index in that membership. Sources are called independently
per local rank. Corpus partitioning, overlapping source values, repetition,
filtering and training synchronization are **application choices**. The practical
file source demonstrates unequal-rank-safe two-level partitioning; a source
returning `range(7)` on two ranks instead deliberately produces 14 outputs.
There is no automatic deduplication/distributed sampler/epoch/equal-step protocol.
Filtering belongs in source/application logic; returning None from a map does
not filter an item.

## Callable shipping and resource setup

Functions/closures, callable instances and no-argument callable classes are
automatically cloudpickled into local spawn workers. A class is constructed
inside **each** worker, so initialize expensive models/resources there:

```python
class Decode:
    def __init__(self):
        self.processed = 0
        # Open local resource/cache here, not in an unpicklable captured parent.

    def __call__(self, item):
        self.processed += 1
        return item.decode("utf-8")
```

Instances are cloned per worker with their captured Python state; they are not
shared mutable objects or constructors rerun implicitly. Build complex resources
in the class constructor rather than capturing file handles, thread locks,
sockets, active generators, GPU objects or huge data. Captured state/cloudpickle
serialization memory is outside payload buffer budgets. Importable application
modules/dependencies must exist on each machine and spawned local worker.
Top-level import side effects run again under spawn; guard process launches.
Arbitrary callable semantic equivalence is not checked over peer handshake.

Worker private state lives across successful calls, disappears on replacement
and may observe repeated current input on retries. Process teardown releases OS
resources; there is no public destructor/close hook guaranteed for user callable
resources after forced termination. Use idempotent external effects or application
transaction keys if necessary; library ACKs do not make external writes exactly-
once. Exceptions report stage/rank plus traceback and fail the run.

## Peer and limits defaults

`PeerConfig(identity="local", run_id="standalone", listen=None, peers={},
join_timeout=15, heartbeat_timeout=5, lease_timeout=5, claim_timeout=5)`.
For cooperation peers includes self at exactly listen and every other member
with a distinct reachable TCP endpoint. Membership/run/graph must match; local
counts/policies can vary. No automatic discovery, auth, code distribution or
member replacement. Standalone empty peers uses the same ownership lifecycle.

`Limits` defaults and scopes are listed in [README](../README.md#limits-and-memory).
Retained items/bytes are per-stage and include maximum-result pre-reservations;
claim item/byte capacity is per-stage divided among local ranks; output budget
is combined local iterator capacity. `max_payload_bytes` includes pickle envelope
and out-of-band buffers; `batch.max_bytes` counts serialized member inputs, not
guaranteed output-envelope overhead. Oversized single result/group fails.

Sources pull only with a result reservation; maps/batches reserve result capacity
before input claims. Streaming overlaps stages with independent local workers.
Output objects, callable state, decoded values, buffers retained by user code,
serialization transient memory and ZMQ transport overhead are additional:
serialized budgets do **not** impose a hard RSS ceiling.
