import multiprocessing as mp
import os
import socket
import threading
import time

import numpy as np
import pytest

from zjobs import Graph, Limits, Locality, PeerConfig, Run, RunError
from zjobs._payload import dumps, loads, size
from zjobs._selection import Selection


def wait_until(test, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if test():
            return
        time.sleep(.01)
    raise AssertionError("condition deadline exceeded")


def clean(run):
    assert run._finished.is_set()
    assert not run._thread.is_alive()
    assert all(not slot.process.is_alive() for slot in run._slots)
    assert all(s["retained_items"] == s["retained_bytes"] == 0
               for s in run.status["stages"].values())


@pytest.mark.parametrize("n", [0, 1, 7])
def test_standalone_none_full_partial_and_empty(n):
    graph = Graph()
    raw = graph.source("raw", lambda ctx: range(n))
    mapped = graph.map("mapped", lambda x: None if x == 0 else x, inputs=[raw])
    graph.batch("batch", inputs=[mapped], size=3)
    with graph.run() as run:
        values = list(run)
    assert [len(v) for v in values] == ([3] * (n // 3) + ([n % 3] if n % 3 else []))
    assert [x for batch in values for x in batch] == [None if x == 0 else x for x in range(n)]
    assert run.status["obligations_created"] == run.status["obligations_acked"]
    clean(run)


def test_fanout_union_replicas_stateful_classes_and_streaming():
    class Increment:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            return (x, self.calls, os.getpid())

    graph = Graph()
    def source(ctx):
        for i in range(5):
            if i == 4:
                time.sleep(.2)
            yield (ctx.rank, i)
    root = graph.source("raw", source, replicas=2)
    a = graph.map("a", Increment, inputs=[root], replicas=2)
    b = graph.map("b", lambda x: (x, 0, os.getpid()), inputs=[root])
    graph.map("union", lambda x: x, inputs=[a, b], replicas=2)
    with graph.run() as run:
        first = next(run)
        assert not run.status["stages"]["raw"]["production_sealed"]
        result = [first, *run]
    assert len(result) == 20
    assert sorted(x[0] for x in result) == sorted([(r, i) for r in range(2) for i in range(5)] * 2)
    assert max(x[1] for x in result) > 1
    assert len(run.status["stages"]["a"]["pids"]) == 2
    assert run.status["obligations_created"] == run.status["obligations_acked"] == 60
    clean(run)


def test_locality_preference_fallback_class_and_individual_owner_progress():
    choice = Selection()
    policy = Locality(max_local_streak=8)
    owners = ["A", "B", "C", "D"]
    result = []
    for _ in range(54):
        owner = choice.choose(owners, "A", policy)
        result.append(owner)
        choice.admitted(owner, "A")
    assert result[:9] == ["A"] * 8 + ["B"]
    assert result[17] == "C" and result[26] == "D"
    assert result[35] == "B" and result[44] == "C" and result[53] == "D"
    assert choice.choose(["C"], "A", policy) == "C"
    assert choice.choose(["C"], "A", Locality("local_only")) is None
    balanced = Selection()
    sequence = []
    for _ in range(8):
        owner = balanced.choose(owners, "A", Locality("balanced"))
        sequence.append(owner)
        balanced.admitted(owner, "A")
    assert sequence == owners * 2


def test_rank_stage_graph_policy_precedence():
    graph = Graph(locality=Locality("balanced"))
    source = graph.source("s", lambda ctx: [])
    stage = graph.map("m", lambda x: x, inputs=[source], replicas=3,
                      locality=Locality("local_only"), rank_locality={2: Locality()})
    assert [graph.policy(stage, i).mode for i in range(3)] == [
        "local_only", "local_only", "local_first"]
    assert graph.policy(source, 0).mode == "balanced"


def endpoints(n=3):
    sockets = [socket.socket() for _ in range(n)]
    for sock in sockets:
        sock.bind(("127.0.0.1", 0))
    result = {chr(65 + i): f"tcp://127.0.0.1:{sock.getsockname()[1]}"
              for i, sock in enumerate(sockets)}
    for sock in sockets:
        sock.close()
    return result


def peer_entry(identity, peers, queue, mode, empty=False):
    try:
        graph = Graph(locality=Locality(mode, max_local_streak=2))
        def source(ctx):
            time.sleep({"A": .01, "B": .4, "C": .8}[ctx.peer])
            for i in range(0 if empty else 18):
                yield (ctx.peer, ctx.rank, i)
        raw = graph.source("io", source, replicas=2 if identity == "A" else 1)
        def first(x):
            time.sleep({"A": .08, "B": .025, "C": .005}[identity])
            return x
        cpu = graph.map("cpu", first, inputs=[raw], replicas=1 if identity == "A" else 2)
        def second(x):
            time.sleep({"A": .002, "B": .06, "C": .015}[identity])
            return x
        cooked = graph.map("cooked", second, inputs=[cpu])
        graph.batch("batch", inputs=[cooked], size=5)
        peer = PeerConfig(identity, "test", peers[identity], peers,
                          heartbeat_timeout=3, lease_timeout=1)
        with graph.run(peer=peer, limits=Limits(task_timeout=10)) as run:
            values = [v for batch in run for v in batch]
        clean(run)
        queue.put(("ok", identity, values, run.status))
    except BaseException as exc:
        queue.put(("error", identity, str(exc)))


@pytest.mark.parametrize("mode,empty", [("local_first", False), ("local_only", False),
                                       ("balanced", False), ("local_first", True)])
def test_three_independent_complete_graphs(mode, empty):
    peers = endpoints()
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=peer_entry, args=(p, peers, queue, mode, empty))
                 for p in ["C", "A", "B"]]
    try:
        for process in processes:
            process.start()
            time.sleep(.08)
        rows = [queue.get(timeout=25) for _ in processes]
        assert all(r[0] == "ok" for r in rows), rows
        values = [x for row in rows for x in row[2]]
        expected = [(p, r, i) for p in peers for r in range(2 if p == "A" else 1)
                    for i in range(0 if empty else 18)]
        assert sorted(values) == sorted(expected)
        statuses = [row[3] for row in rows]
        assert sum(s["obligations_created"] for s in statuses) == sum(
            s["obligations_acked"] for s in statuses)
        for status in statuses:
            assert set(status["stages"]) == {"io", "cpu", "cooked", "batch"}
            assert all(s["ready"] == s["replicas"] for s in status["stages"].values())
            assert status["iteration_done"] and status["error"] is None
        for edge in ("io->cpu", "cpu->cooked"):
            count = sum(s["peer_items"].get(edge, 0) for s in statuses)
            assert (count == 0) if mode == "local_only" or empty else (count > 0)
        if mode == "local_only":
            assert all(all(x[0] == row[1] for x in row[2]) for row in rows)
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        queue.close()
        queue.join_thread()


def test_bounded_large_array_fanout_snapshot_and_consumer_pressure():
    graph = Graph()
    raw = graph.source("raw", lambda ctx: (np.full(10000, i) for i in range(8)))
    def mutate(x):
        x[0] = -1
        return x[1]
    graph.map("mutate", mutate, inputs=[raw])
    graph.map("unchanged", lambda x: x[0], inputs=[raw])
    limits = Limits(max_payload_bytes=90000, retained_items=2, retained_bytes=180000,
                    claim_bytes=90000, output_items=1, output_bytes=90000)
    with graph.run(limits=limits) as run:
        time.sleep(.5)
        result = list(run)
    assert sorted(result) == sorted(list(range(8)) * 2)
    for stage in run.status["stages"].values():
        assert stage["peak_retained_items"] <= 2
        assert stage["peak_retained_bytes"] <= 180000
        assert stage["peak_claim_bytes"] <= 90000
    assert run.status["peak_output_items"] <= 1
    clean(run)


def test_batch_larger_than_retention_flushes_under_pressure():
    graph = Graph()
    raw = graph.source("raw", lambda ctx: range(5))
    graph.batch("batch", inputs=[raw], size=10, replicas=2)
    limits = Limits(max_payload_bytes=10000, retained_items=1, retained_bytes=10000,
                    claim_bytes=20000, claim_items=20)
    with graph.run(limits=limits) as run:
        values = list(run)
    assert sorted(x for v in values for x in v) == list(range(5))
    clean(run)


@pytest.mark.parametrize("kind", ["error", "oversize", "stall", "source_crash"])
def test_contextual_source_failures_clean_owned_resources(kind):
    graph = Graph()
    def source(ctx):
        if kind == "error":
            raise ValueError("source exploded")
        if kind == "stall":
            time.sleep(5)
        if kind == "source_crash":
            os._exit(7)
        return [b"x" * 100000] if kind == "oversize" else []
    graph.source("io", source)
    limits = Limits(max_payload_bytes=10000, task_timeout=.7)
    with pytest.raises(RunError, match={
        "error": "source exploded", "oversize": "serialized payload",
        "stall": "stalled", "source_crash": "crashed",
    }[kind]):
        with graph.run(limits=limits) as run:
            list(run)
    clean(run)


def test_map_crash_retry_and_fenced_attempt(tmp_path):
    marker = tmp_path / "once"
    def task(x):
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return x * 2
        os.close(fd)
        os._exit(9)
    graph = Graph()
    root = graph.source("s", lambda ctx: [3])
    graph.map("m", task, inputs=[root])
    with graph.run(limits=Limits(max_retries=1)) as run:
        assert list(run) == [6]
    assert run.status["retries"] == 1
    assert run.status["obligations_created"] == run.status["obligations_acked"] == 2
    clean(run)


def test_slow_consumer_renews_active_claims_and_abandonment_fails():
    graph = Graph()
    raw = graph.source("s", lambda ctx: range(8))
    def task(x):
        time.sleep(.3)
        return x
    graph.map("m", task, inputs=[raw])
    limits = Limits(output_items=1, retained_items=1, consumer_timeout=3)
    peer = PeerConfig(lease_timeout=.1)
    with graph.run(peer=peer, limits=limits) as run:
        time.sleep(.8)
        assert sorted(run) == list(range(8))
    assert run.status["renewals"] > 10
    clean(run)
    with pytest.raises(RunError, match="cancelled/abandoned"):
        with graph.run() as abandoned:
            next(abandoned)
    clean(abandoned)


def test_consumer_idle_and_missing_peer_fail_explicitly():
    graph = Graph()
    graph.source("s", lambda ctx: [None])
    with pytest.raises(RunError, match="consumer stalled"):
        with graph.run(limits=Limits(consumer_timeout=.2)) as run:
            wait_until(lambda: run._finished.is_set())
            next(run)
    clean(run)
    peers = endpoints(2)
    with pytest.raises(RunError, match="join deadline"):
        with graph.run(peer=PeerConfig("A", "absent", peers["A"], peers, join_timeout=.3)) as run:
            list(run)
    clean(run)


@pytest.mark.parametrize("value", [None, b"", b"x" * 100000, {"a": [1, None]}])
def test_codec_exact_limit(value):
    frames = dumps(value, 200000)
    assert loads(frames) == value
    assert dumps(value, size(frames)) == frames
    with pytest.raises(ValueError, match="serialized payload"):
        dumps(value, size(frames) - 1)


def test_array_codec_preserves_isolated_out_of_band_snapshot():
    original = np.arange(262144, dtype=np.float64)
    frames = dumps(original, 3000000)
    assert len(frames) == 3 and len(frames[1]) < 1000
    original[:] = -1
    restored = loads(frames)
    np.testing.assert_array_equal(restored, np.arange(262144))
    restored[:] = 0
    np.testing.assert_array_equal(loads(frames), np.arange(262144))


def test_duplicate_grants_and_stale_acks_do_not_settle_current_work(monkeypatch):
    receive = Run._receive_claim
    publish = Run._publish
    def duplicate(self, sender, kind, body, frames):
        receive(self, sender, kind, body, frames)
        receive(self, sender, kind, body, frames)
    def stale(self, slot, frames):
        old = list(slot.inputs)
        for inp in old:
            self._dispatch(inp.owner, "ack", {
                "output": inp.output, "child": inp.child, "token": "retired-attempt",
            }, ())
        publish(self, slot, frames)
        for inp in old:
            self._dispatch(inp.owner, "ack", {
                "output": inp.output, "child": inp.child, "token": inp.token,
            }, ())
    monkeypatch.setattr(Run, "_receive_claim", duplicate)
    monkeypatch.setattr(Run, "_publish", stale)
    graph = Graph()
    raw = graph.source("raw", lambda ctx: range(6))
    graph.map("map", lambda x: x * 2, inputs=[raw], replicas=2)
    with graph.run() as run:
        assert sorted(run) == [0, 2, 4, 6, 8, 10]
    assert run.status["obligations_created"] == run.status["obligations_acked"] == 12
    assert run.status["stale_messages"] == 18
    clean(run)


def failure_peer(identity, peers, queue, action):
    try:
        graph = Graph()
        def source(ctx):
            for i in range(100):
                time.sleep(.03)
                yield (ctx.peer, i)
        raw = graph.source("raw", source)
        graph.map("map", lambda x: x, inputs=[raw])
        with graph.run(
            peer=PeerConfig(identity, "failure", peers[identity], peers,
                            heartbeat_timeout=.5, join_timeout=3, lease_timeout=2),
            limits=Limits(consumer_timeout=.4, task_timeout=5),
        ) as run:
            if identity == "B":
                wait_until(lambda: run.status.get("local_items", {}).get("raw->map", 0) +
                           run.status.get("peer_items", {}).get("raw->map", 0) > 0)
                if action == "crash":
                    time.sleep(.15)
                    queue.put(("crashing", identity, run.status["worker_pids"]))
                    queue.close()
                    queue.join_thread()
                    os._exit(17)
                time.sleep(2)
            list(run)
        queue.put(("unexpected_success", identity))
    except RunError as exc:
        queue.put(("failed", identity, str(exc), time.monotonic()))


@pytest.mark.parametrize("action", ["crash", "abandon"])
def test_peer_loss_or_abandoned_remote_consumer_fails_survivor(action):
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=failure_peer, args=(p, peers, queue, action))
                 for p in peers]
    started = time.monotonic()
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=8) for _ in range(2)]
        failed = [r for r in rows if r[0] == "failed"]
        assert len(failed) == (1 if action == "crash" else 2), rows
        survivor = next(r for r in failed if r[1] == "A")
        assert "heartbeat" in survivor[2] or "peer B failed" in survivor[2]
        assert survivor[3] - started < 6
        for process in processes:
            process.join(5)
            assert not process.is_alive()
        assert processes[1].exitcode == (17 if action == "crash" else 0)
        if action == "crash":
            pids = next(r[2] for r in rows if r[0] == "crashing")
            def stopped():
                for pid in pids:
                    try:
                        with open(f"/proc/{pid}/stat") as stat:
                            if stat.read().split()[2] != "Z":
                                return False
                    except FileNotFoundError:
                        continue
                return True
            wait_until(stopped, timeout=2)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        queue.close()
        queue.join_thread()


def array_peer(identity, peers, queue):
    try:
        graph = Graph(locality=Locality("balanced"))
        def source(ctx):
            if ctx.peer == "B":
                return []
            return [np.full(10000, i, dtype=np.int64) for i in range(7)]
        raw = graph.source("raw", source)
        mapped = graph.map("map", lambda x: int(x[0]), inputs=[raw])
        graph.batch("batch", inputs=[mapped], size=3, replicas=2)
        with graph.run(
            peer=PeerConfig(identity, "arrays", peers[identity], peers, lease_timeout=.2),
            limits=Limits(max_payload_bytes=90000, retained_items=4, retained_bytes=360000,
                          claim_bytes=180000, consumer_timeout=5),
        ) as run:
            time.sleep(.8)
            values = [x for batch in run for x in batch]
        clean(run)
        queue.put(("ok", values, run.status))
    except BaseException as exc:
        queue.put(("error", str(exc)))


def test_remote_large_payload_empty_owner_delayed_consumer_and_partial_batch():
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=array_peer, args=(p, peers, queue)) for p in peers]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=15) for _ in processes]
        assert all(r[0] == "ok" for r in rows), rows
        assert sorted(x for row in rows for x in row[1]) == list(range(7))
        assert sum(r[2]["peer_bytes"].get("raw->map", 0) for r in rows) > 80000
        for row in rows:
            assert all(s["peak_retained_bytes"] <= 360000 for s in row[2]["stages"].values())
            assert all(s["peak_claim_bytes"] <= 180000 for s in row[2]["stages"].values())
        for process in processes:
            process.join(3)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        queue.close()
        queue.join_thread()


def capacity_peer(identity, peers, queue):
    try:
        graph = Graph(locality=Locality("balanced"))
        raw = graph.source("raw", lambda ctx: [b"x" * 50000] if ctx.peer == "A" else [])
        graph.map("map", lambda x: len(x), inputs=[raw], replicas=1 if identity == "A" else 2)
        with graph.run(
            peer=PeerConfig(identity, "capacity", peers[identity], peers),
            limits=Limits(max_payload_bytes=60000, claim_bytes=60000),
        ) as run:
            values = list(run)
        queue.put(("ok", identity, values, run.status))
    except BaseException as exc:
        queue.put(("error", identity, str(exc)))


def test_receiver_capacity_skips_unfit_offer_without_failing_other_routing():
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=capacity_peer, args=(p, peers, queue)) for p in peers]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=10) for _ in processes]
        assert all(r[0] == "ok" for r in rows), rows
        assert next(r[2] for r in rows if r[1] == "A") == [50000]
        assert next(r[2] for r in rows if r[1] == "B") == []
        assert all(r[3]["peer_items"].get("raw->map", 0) == 0 for r in rows)
        for process in processes:
            process.join(3)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        queue.close()
        queue.join_thread()


def test_impossible_required_byte_routing_fails_at_owner():
    graph = Graph()
    raw = graph.source("raw", lambda ctx: [b"x" * 50000])
    graph.map("map", len, inputs=[raw], replicas=2)
    with pytest.raises(RunError, match="no eligible receiver"):
        with graph.run(limits=Limits(max_payload_bytes=60000, claim_bytes=60000)) as run:
            list(run)
    clean(run)


def test_partial_batch_flush_does_not_wait_for_delayed_ack(monkeypatch):
    original = Run._send
    consume = Run._consume
    delayed = []
    release = threading.Event()
    def send(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "ack" and body["child"] == "batch":
            delayed.append((self, recipient, kind, body, frames))
            return
        return original(self, recipient, kind, body, frames, optional=optional)
    def consume_with_deferred_ack(self):
        if release.is_set():
            while delayed:
                runtime, recipient, kind, body, frames = delayed.pop()
                original(runtime, recipient, kind, body, frames)
        consume(self)
    monkeypatch.setattr(Run, "_send", send)
    monkeypatch.setattr(Run, "_consume", consume_with_deferred_ack)
    graph = Graph()
    raw = graph.source("raw", lambda ctx: [1, 2])
    graph.batch("batch", inputs=[raw], size=3, replicas=2)
    with graph.run() as run:
        wait_until(lambda: len(delayed) == 2)
        values = next(run)
        assert run.status["stages"]["raw"]["production_sealed"]
        assert not run.status["iteration_done"]
        release.set()
        assert sorted([*values, *(x for batch in run for x in batch)]) == [1, 2]
    clean(run)


def test_single_result_credit_rotates_source_ranks(monkeypatch):
    schedule = Run._schedule
    def after_ready(self):
        if all(slot.ready for slot in self._slots):
            schedule(self)
    monkeypatch.setattr(Run, "_schedule", after_ready)
    graph = Graph()
    def source(ctx):
        for i in range(4):
            yield (ctx.rank, i)
    root = graph.source("raw", source, replicas=3)
    graph.map("map", lambda x: x, inputs=[root], replicas=2)
    with graph.run(limits=Limits(retained_items=1)) as run:
        values = list(run)
    assert sorted(values) == [(r, i) for r in range(3) for i in range(4)]
    assert [v[0] for v in values[:3]] == [0, 1, 2]
    clean(run)


def test_final_byte_capped_batches_respect_member_and_claim_caps():
    graph = Graph()
    root = graph.source("raw", lambda ctx: [b"x" * 100 for _ in range(5)])
    graph.batch("batch", inputs=[root], size=10, max_bytes=203)
    with graph.run(limits=Limits(max_payload_bytes=1000, claim_bytes=1000)) as run:
        groups = list(run)
    assert [len(group) for group in groups] == [2, 2, 1]
    assert run.status["stages"]["batch"]["peak_claim_bytes"] <= 203
    clean(run)


def retransmit_peer(identity, peers, queue):
    original = Run._send
    skipped = {}
    def send(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "state" and (body["ready"] or body["bye"]):
            key = (recipient, body["bye"])
            count = skipped.get(key, 0)
            skipped[key] = count + 1
            if count < 2:
                return
        return original(self, recipient, kind, body, frames, optional=optional)
    Run._send = send
    try:
        graph = Graph()
        graph.source("raw", lambda ctx: [])
        with graph.run(peer=PeerConfig(identity, "retransmit", peers[identity], peers)) as run:
            assert list(run) == []
        clean(run)
        queue.put(("ok", identity, skipped))
    except BaseException as exc:
        queue.put(("error", identity, str(exc)))


def test_ready_and_symmetric_shutdown_retransmit_dropped_snapshots():
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=retransmit_peer, args=(p, peers, queue)) for p in peers]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=10) for _ in processes]
        assert all(r[0] == "ok" for r in rows), rows
        assert all(all(count >= 3 for count in r[2].values()) for r in rows)
        for process in processes:
            process.join(3)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        queue.close()
        queue.join_thread()
