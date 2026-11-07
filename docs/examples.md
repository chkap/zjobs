# Examples and exercised command matrix

[README](../README.md) | [Architecture](architecture.md) | [API](api.md) |
[Operations](operations.md)

Run from the repository root with Python 3.10+. Core has **no torch or numpy
requirement**. Use one fresh environment for core commands:

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -c "import importlib.util; assert importlib.util.find_spec('torch') is None; import zjobs"
```

For the repository tests install the test extra (pytest/numpy, not torch):
`.venv/bin/pip install -e '.[test]'`, then `.venv/bin/python -m pytest -q`.
All applications must guard launches with `if __name__ == "__main__"` because
local tasks use spawn. The scripts do so; run them as scripts, not snippets
from a notebook/interactive interpreter.

| Purpose | Entry point | Expected invariant |
| --- | --- | --- |
| Minimal standalone | `examples/local_pipeline.py` | Seven squares, full/partial groups, unspecified order |
| General A/B/C same graph | `examples/peer_pipeline.py A/B/C` | 72 unique coordinates across local streams, multi-edge fallback |
| Real text file IO/CPU/branch/union | `examples/file_pipeline.py` | Two exact branch outputs per input file across local streams |
| Check file accounting | `examples/check_files.py` | Exact union, successful drain and zero retention |
| Application overlap/bytes/RSS | `examples/measure_peer_graph.py` | Exact coordinates, measured local/peer paths; no universal speedup |
| Backpressure/cancel/failure | `examples/lifecycle.py` | Slow succeeds; early/error/idle fail explicitly and clean up |
| Optional CPU consumer | `examples/torch_dataset.py` | One graph/consumer per machine, num_workers=0 only |

## Standalone file processing, partial and empty

Choose a new fixture directory; the fixture creator refuses to overwrite an
existing directory. It writes deterministic UTF-8 files (no download). On real
data use your own directory of `.txt` files, subject to the configured maximum
serialized input size. This example is whole-file IO, not unbounded file
streaming. Oversized files fail contextually; chunk large files in application
source code instead.

```sh
.venv/bin/python examples/local_pipeline.py
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-fixture-29 --make-fixture 17
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-fixture-29 > /tmp/zjobs-files-local.json
.venv/bin/python examples/check_files.py /tmp/zjobs-fixture-29 /tmp/zjobs-files-local.json
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-empty-29 --make-fixture 0
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-empty-29 > /tmp/zjobs-empty.json
.venv/bin/python examples/check_files.py /tmp/zjobs-empty-29 /tmp/zjobs-empty.json
```

Expected 17 files -> **34 branch values** (word count and SHA256 of normalized
text), and empty -> **zero batches/values** with successful completion, not an
empty data batch. Default batch size 5, one local collector: full groups plus
final partial, with possible earlier pressure-partials if buffering changes.
`None` is valid map output in the general API, not a filter/EOS marker.

The source explicitly partitions file indices first by sorted configured peer
index, then by **that peer's** local IO rank. Unequal IO counts are supported
without overlapping partitions. This is application fixture policy, **not**
runtime dataset deduplication or a mandatory sampler. Each no-argument Normalize
class is constructed in its local worker; its counter is private per worker.
Two downstream branches each get a required obligation, and batch inputs are
an unordered union, not a zip/join. Read/hash/word tasks run on local CPU workers;
remote upstream values can be claimed at every eligible boundary.

## Independently launch A/B/C practical graphs

Use the SAME `file_pipeline.py`, corpus and run/membership map on all machines.
Start each command in a separate terminal within the join deadline. No service,
seed, master or collector is started first. Each command owns **all five**
stages and its final local consumer.

```sh
PEERS='{"A":"tcp://127.0.0.1:5600","B":"tcp://127.0.0.1:5601","C":"tcp://127.0.0.1:5602"}'
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-fixture-29 --identity A --peers "$PEERS" --run-id files-29 --io-workers 2 --cpu-workers 1 --collect-workers 1 --delays > /tmp/zjobs-files-A.json
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-fixture-29 --identity B --peers "$PEERS" --run-id files-29 --io-workers 1 --cpu-workers 2 --collect-workers 2 --delays > /tmp/zjobs-files-B.json
.venv/bin/python examples/file_pipeline.py /tmp/zjobs-fixture-29 --identity C --peers "$PEERS" --run-id files-29 --io-workers 3 --cpu-workers 2 --collect-workers 1 --delays > /tmp/zjobs-files-C.json
```

After all three exit:

```sh
.venv/bin/python examples/check_files.py /tmp/zjobs-fixture-29 /tmp/zjobs-files-A.json /tmp/zjobs-files-B.json /tmp/zjobs-files-C.json
```

Expected **34 exact values across the three consumers**, not 34 on each machine.
Final collection is local, not globally broadcast; shares at earlier boundaries
change which peer produces terminal batches. Inspect `status.peer_items` for
`read->normalize` and `normalize->words`/`normalize->sha256`: synthetic asymmetric
delays make eligible upstream fallback observable on multiple boundaries. Delays
are not a promise of disk speed or fixed per-peer output counts. Exact branch
fixture equality and sum of created=ACKed obligations are invariant; batching,
distribution, delays/timing and transfer counts are nondeterministic.

Repeat the same three commands with new run ID `files-balanced` and
`--locality balanced`, or `files-local-only` and `--locality local_only`.
All peers must still launch. Local-only has **zero remote payload**; fixture
partition gives A/B/C 6/6/5 files -> 12/12/10 branch values. `--rank-zero-local-only`
overrides only normalization rank 0; other normalization ranks inherit stage/
graph policy, and other stages keep graph policy. Use `--batch-size 100` to
observe pressure/final partial groups or `--collect-workers 2` for independent
parallel group formation. There is no global collection-size/order barrier.

For real hosts use the same map with reachable addresses, e.g.
`{"A":"tcp://10.0.0.11:5600","B":"tcp://10.0.0.12:5600","C":"tcp://10.0.0.13:5600"}`.
Each self endpoint must bind locally; open those TCP ports between trusted peers.
Deploy the same graph code/dependencies and corpus snapshot/path layout on each
machine. The example does not synchronize files or authenticate/encrypt peer
code/data. Logical peers on one host prove execution paths, not physical scale.

## Lifecycle examples and cleanup

```sh
.venv/bin/python examples/lifecycle.py slow
.venv/bin/python examples/lifecycle.py early
.venv/bin/python examples/lifecycle.py error
.venv/bin/python examples/lifecycle.py idle
```

`slow` consumes all 12 squares with output-items=1 and 50ms consumer pauses under
the .5s idle deadline. `early` exits after one value and reports expected explicit
cancellation, not success. `error` reports intentional contextual task failure.
`idle` stops consuming for 2s and reports `.5s consumer_timeout`. These examples
catch `RunError` **outside** the context, print `expected_error` and final status,
and assert retention released. They intentionally exit zero after confirming
the expected failure; inspect the error field, not only shell exit status.
In a peer run cancellation/consumer loss fails the fixed-membership run; peers
do not relocate final outputs to a different user. [Operations](operations.md)
explains deadlines and owned process cleanup.

Fixture/output files belong to the user and are not automatically deleted.
Remove only the explicit example files/directories you created after inspecting
them. The library owns its worker processes/sockets and closes them on context
exit; it does not own unrelated processes or input files.

## Optional CPU PyTorch integration on each machine

This is an **example adapter**, not torch/training logic in core or a supported
GPU transport. Install a separate optional environment:

```sh
python3 -m venv .venv-torch
.venv-torch/bin/pip install -e '.[test]'
.venv-torch/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-torch/bin/python examples/torch_dataset.py
.venv-torch/bin/python -m pytest -q tests/test_examples.py -k torch
```

Use the wheel index appropriate to your Python/platform if that CPU wheel is
unavailable. Core commands do not require this environment. The optional script
fails with installation guidance if torch is missing; tests report skipped
optional coverage separately rather than pretending it passed.

`LocalGraphDataset` is a context-owned IterableDataset. **Only DataLoader
`num_workers=0, batch_size=None`** is supported here: the graph already owns its
worker processes and delivers terminal values. A second iterator in one context,
iteration outside the context, or a nested DataLoader worker fails explicitly.
No hidden DataLoader child may launch another complete graph or user consumer.
Early DataLoader loop exit is still graph cancellation when dataset context
exits. New context/new run is explicit repetition; user decides source policy.

Each optional application launches its own coordinate source and Tensorize CPU
workers, then its own local DataLoader consumer. To demonstrate three peers:

```sh
PEERS='{"A":"tcp://127.0.0.1:5610","B":"tcp://127.0.0.1:5611","C":"tcp://127.0.0.1:5612"}'
.venv-torch/bin/python examples/torch_dataset.py --identity A --peers "$PEERS" --run-id torch-29
.venv-torch/bin/python examples/torch_dataset.py --identity B --peers "$PEERS" --run-id torch-29
.venv-torch/bin/python examples/torch_dataset.py --identity C --peers "$PEERS" --run-id torch-29
```

Run independently in separate terminals. Across streams expect 15 unique
`(source_peer,index)` coordinates, each CPU float32 tensor `[index,index+1]`,
not equal per-peer step counts. If torch import/startup is slow, configure peer
join/heartbeat deadlines for your environment; this example uses task_timeout=60.
DDP equal steps, epoch shuffling, pad/drop-last, samplers, gradient synchronization
and GPU placement belong to the application and are not provided.

## Reproducible measurements and scale plans

Use the [README measurement commands](../README.md#diagnostics-and-application-overlap)
for exact source-byte/CPU/transform work, independent local stage counts,
local-first versus local-only comparison, task wall/CPU, overlap, traffic and
actual/reserved item/byte peaks. A JSON value list is fixture accounting, **not**
a recommended production consumer retention pattern: retaining every result in
this example adds user-owned memory beyond library budgets.

The same section includes eight independent `--plan-only` logical memberships
with eight IO ranks each = **64 configured**, no service or concurrent launch.
For a bounded actual local worker-count probe:

```sh
.venv/bin/python examples/measure_peer_graph.py --io-workers 8 --cpu-workers 1 --count 1 --cpu-seconds .002
```

Expect eight source coordinates and **11 actual processes** (8 IO, CPU, transform,
collect), not a claim of eight physical hosts or 64 concurrent IO. Compare
resource evidence by scope: parent maxrss and largest child maxrss are separately
reported Linux peaks, not total simultaneous process-tree RSS.
