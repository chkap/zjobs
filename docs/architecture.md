# Architecture, delivery and completion

[Examples](examples.md) | [API](api.md) | [Operations](operations.md)

```text
Application A                          Application B (same complete graph)
source workers -> A retained store     source workers -> B retained store
        |           ^                         |           ^
local map workers   | direct peer claims      local map workers
        |           +------ grants/ACKs -------+
local branches / collection            local branches / collection
        |                                      |
local user iterator                    local user iterator
```

Every application owns one local runtime thread, local child stage processes,
local retained payload/obligation authority and its own final iterator. A local
source/map/collection stage does not depend on a distinguished machine launching
it. Membership is static, symmetric and manually configured; peers can start in
any order within join deadlines. There is **no** global graph, placement map,
central metadata service, master-to-agent code dispatch or stage broker.
An output's producing application is its only owner; this owner is not empowered
to schedule anyone else's graph.

## Paths and authority

Local worker channels use bounded ZMQ PAIR sockets. Peers use a bound ROUTER and
direct DEALER connections to each configured other member. Worker definitions
are cloudpickled from the local graph into **only local** children. Peer messages
contain membership/run/graph key and incarnation, availability snapshots,
concrete-size claims, immutable multipart grants, token-fenced ACK/renewals and
completion/failure records, never remote deployment instructions.

Peer snapshots describe only that owner's outputs/stage seals/capacity. They
are hints, not a global routing authority. Local downstream ranks independently
select eligible local/peer candidates with sufficient input capacity and reserve
items/serialized bytes before requests. The owner's serialized arbitration
grants a live token only for an available obligation. Duplicate/raced requests
cannot settle another attempt; failed/raced claims do not count as admissions.

Output identities contain owner incarnation, stage/rank and sequence. One
required obligation exists **per child stage**, not per replica or peer.
Competing replicas across eligible machines settle it once. Fan-out retains the
snapshot until every child edge ACKs; each branch independently decodes mutable
storage. Multiple parent stages form an unordered union, not a keyed join, zip,
barrier or duplicate-input inference. User source repetition yields distinct
runtime outputs, even when Python values are equal.

An input is acknowledged only after its local downstream result has been
admitted into its producing owner's retained store. Successful batches commit
all member ACKs together after output admission. This is the retry/commit
boundary. The local terminal iterator is a local-only required consumer
obligation, not a peer sharing edge. Iteration acknowledges a successfully
decoded returned object; application retention/processing after `next` belongs
to the user, not a durable library transaction.

## Production versus delivery drain

Source production seals after each rank reaches real StopIteration and all
results are retained. Seals include immutable final per-owner/stage output-count
watermarks in monotonic snapshots. **Production sealed does not mean ACKed.**
Ready/claimed values can remain retained after EOF.

Each batch rank owns its own input group. It flushes on item/serialized member
cap, pressure, or sealed parent production when no remaining eligible candidate
can join that rank's group. Other ranks' active claims are assigned elsewhere;
they do not join this group. Waiting until that group's own member ACKs before
flushing would be a circular dependency and is deliberately avoided.

A non-source stage seals after all parent owners' production seals and required
edge obligations drain and its local pending inputs/attempts/results settle.
Induction over the DAG, not a leader, derives graph completion. Each peer becomes
COMPLETE_READY only when every stage sealed and its owned stores/terminal user
obligations drained. Records remain/retransmit until reciprocal observation.
`StopIteration` is that all-member delivery-ready barrier. Context close then
symmetrically announces/observes BYE and lingers to retransmit before sockets
close. A missing READY/BYE member fails explicitly by configured deadlines.

## Failure and guarantees

Trusted Python attempts/side effects are at-least-once when local map/batch
crash/stall retry is enabled. The prior process is fenced and stopped before
replacement; claims remain retained/renewed while executing or backpressured.
Stateful replacement loses private state. Source replay is unsupported and
crashed/stalled source generators fail explicitly, not restart at an inferred
index. Deterministic application coordinates help accounting, not durable replay.

Pending claim responses have a separate finite claim_timeout. Granted-active
claim renewals continue regardless of ordinary backpressure; task/consumer/
heartbeat/lease/shutdown deadlines still bound failure. Lease expiry, unsupported
restart/incarnation change and required transport congestion fail, rather than
silently reissuing committed values. No durable active-run recovery or exactly-
once arbitrary external effect guarantee exists. On a partition each reachable
peer fails by direct heartbeat/failure detection; unreachable members do not get
to report success.

Semantic graph/code/environment equivalence and compatible corpus snapshots are
deployment responsibilities. Handshake checks exact run/membership and graph
shape/batching, not arbitrary callable bytecode meaning. Logical identity denotes
same application/machine availability, not physical-host proof. This is trusted
pickle/ZMQ transport, not an authenticated/encrypted hostile network service.
