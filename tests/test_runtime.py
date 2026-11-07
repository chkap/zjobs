import multiprocessing as mp
import os
import threading
import time

import pytest

from zjobs import Graph, RunError


def numbers(context):
    yield from context.shard(range(24))


def empty(context):
    return ()


def identity(value):
    return value


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition did not become true before deadline")


def assert_clean(run):
    assert run._finished.is_set()
    pids = run.status["worker_pids"]
    assert not any(child.pid in pids for child in mp.active_children())
    assert not run._thread.is_alive()


def test_shared_work_source_sharding_and_worker_instances():
    class Stateful:
        def __init__(self):
            self.count = 0

        def __call__(self, value):
            time.sleep(0.02)
            self.count += 1
            return value, os.getpid(), self.count

    graph = Graph()
    source = graph.source("source", numbers, replicas=3)
    graph.map("map", Stateful, inputs=[source], replicas=2)
    with graph.run(capacity=2, output_capacity=2, worker_timeout=8) as run:
        results = list(run)
    assert sorted(value for value, _, _ in results) == list(range(24))
    per_pid = {}
    for _, pid, count in results:
        per_pid.setdefault(pid, []).append(count)
    assert len(per_pid) == 2
    assert all(sorted(counts) == list(range(1, len(counts) + 1)) for counts in per_pid.values())
    assert len(run.status["worker_pids"]) == 5
    assert run.status["state"] == "completed"
    assert_clean(run)


def test_streaming_before_source_exhaustion(tmp_path):
    exhausted = tmp_path / "exhausted"

    def source(context):
        yield 0
        time.sleep(0.5)
        exhausted.write_text("yes")
        yield 1

    graph = Graph()
    root = graph.source("source", source)
    graph.map("map", lambda value: value + 10, inputs=[root], replicas=2)
    with graph.run(capacity=1, output_capacity=1, worker_timeout=8) as run:
        assert next(run) == 10
        assert not exhausted.exists()
        assert list(run) == [11]
    assert_clean(run)


def test_broadcast_and_interleaved_fanin_with_item_accounting():
    graph = Graph()
    root = graph.source("source", numbers, replicas=2)
    left = graph.map("left", lambda value: ("left", value), inputs=[root], replicas=2)
    right = graph.map("right", lambda value: ("right", value), inputs=[root], replicas=3)
    graph.map("merge", identity, inputs=[left, right], replicas=2)
    with graph.run(capacity=1, output_capacity=2, worker_timeout=8) as run:
        result = list(run)
    assert sorted(result) == [(side, value) for side in ("left", "right") for value in range(24)]
    assert run.status["produced"] == {"source": 24, "left": 24, "right": 24, "merge": 48}
    assert all(peak <= 1 for peak in run.status["peak_queued"].values())
    assert_clean(run)


@pytest.mark.parametrize("replicas", [1, 3])
def test_empty_source_completes_branches_and_batches(replicas):
    graph = Graph()
    root = graph.source("empty", empty, replicas=replicas)
    left = graph.map("left", identity, inputs=[root], replicas=2)
    graph.map("right", identity, inputs=[root, left])
    with graph.run(worker_timeout=8) as run:
        assert list(run.batches(3)) == []
    assert_clean(run)


def test_payloads_none_and_partial_batches():
    payloads = [None, b"\x00\xff", {"a": [1, 2]}, [3], "end"]
    graph = Graph()
    root = graph.source("source", lambda context: iter(payloads))
    graph.map("map", identity, inputs=[root])
    with graph.run(worker_timeout=8) as run:
        assert list(run.batches(2)) == [payloads[:2], payloads[2:4], payloads[4:]]
    assert_clean(run)


def test_multiple_terminal_stages_are_broadcast_outputs():
    graph = Graph()
    root = graph.source("source", lambda context: [None])
    graph.map("left", lambda item: ("left", item), inputs=[root])
    graph.map("right", lambda item: ("right", item), inputs=[root])
    with graph.run(worker_timeout=8) as run:
        assert sorted(run) == [("left", None), ("right", None)]


@pytest.mark.parametrize("where", ["source", "map", "class"])
def test_user_errors_are_contextual_and_cleanup(where):
    def fail(value):
        raise ValueError("task exploded")

    class Broken:
        def __init__(self):
            raise ValueError("task exploded")

        def __call__(self, value):
            return value

    graph = Graph()
    root = graph.source("source", fail if where == "source" else numbers)
    if where != "source":
        graph.map("broken", Broken if where == "class" else fail, inputs=[root])
    with graph.run(worker_timeout=8) as run:
        with pytest.raises(RunError, match=r"(?s)stage .* rank 0 work .*task exploded"):
            list(run)
    assert run.status["state"] == "failed"
    assert_clean(run)


def test_callable_shipping_failure_does_not_start_workers():
    class Unpicklable:
        def __call__(self, context):
            return ()

        def __reduce__(self):
            raise ValueError("cannot ship")

    before = {child.pid for child in mp.active_children()}
    graph = Graph()
    graph.source("bad", Unpicklable())
    with pytest.raises(RunError, match="stage 'bad': serialize callable: cannot ship"):
        graph.run()
    assert {child.pid for child in mp.active_children()} == before


def test_payload_serialization_failure():
    graph = Graph()
    root = graph.source("source", lambda context: [1])
    graph.map("generator", lambda value: (item for item in [value]), inputs=[root])
    with graph.run(worker_timeout=8) as run:
        with pytest.raises(RunError, match="stage 'generator'.*serialize output"):
            list(run)
    assert_clean(run)


@pytest.mark.parametrize("downstream", [False, True])
def test_payload_deserialization_failure(downstream):
    def restore():
        raise ValueError("cannot restore payload")

    class BadPayload:
        def __reduce__(self):
            return restore, ()

    graph = Graph()
    root = graph.source("source", lambda context: [BadPayload()])
    if downstream:
        graph.map("map", identity, inputs=[root])
    with graph.run(worker_timeout=8) as run:
        with pytest.raises(RunError, match=r"(?s)deserialize (input|payload).*cannot restore payload"):
            list(run)
        with pytest.raises(RunError, match="cannot restore payload"):
            next(run)
    assert run.status["state"] == "failed"
    assert "cannot restore payload" in run.status["error"]
    assert_clean(run)


def test_missing_dependency_error_is_explicit():
    def missing(value):
        import zjobs_dependency_deliberately_absent
        return zjobs_dependency_deliberately_absent.process(value)

    graph = Graph()
    root = graph.source("source", numbers)
    graph.map("missing", missing, inputs=[root])
    with graph.run(worker_timeout=8) as run:
        with pytest.raises(RunError, match="ModuleNotFoundError.*zjobs_dependency_deliberately_absent"):
            list(run)
    assert_clean(run)


def test_payload_size_limit():
    graph = Graph()
    graph.source("large", lambda context: [b"x" * 10000])
    with graph.run(max_payload_bytes=4096, worker_timeout=8) as run:
        with pytest.raises(RunError, match="serialized payload is .*limit is 4096"):
            list(run)
    assert_clean(run)


def test_worker_crash_is_finite_and_contextual():
    graph = Graph()
    root = graph.source("source", numbers)
    graph.map("crash", lambda value: os._exit(7), inputs=[root])
    with graph.run(worker_timeout=8) as run:
        with pytest.raises(RunError, match="stage 'crash'.*worker exited with code 7"):
            list(run)
    assert_clean(run)


def test_idle_worker_crash_with_an_independent_active_source():
    class IdleCrash:
        def __init__(self):
            def crash():
                time.sleep(0.2)
                os._exit(9)

            threading.Thread(target=crash, daemon=True).start()

        def __call__(self, item):
            return item

    def active(context):
        while True:
            yield 1

    graph = Graph()
    slow = graph.source("slow", lambda context: (time.sleep(5),))
    graph.map("idle", IdleCrash, inputs=[slow])
    graph.source("active", active)
    with graph.run(worker_timeout=8, consumer_timeout=8) as run:
        wait_until(lambda: run._finished.is_set(), timeout=3)
        with pytest.raises(RunError, match="stage 'idle'.*worker exited with code 9"):
            next(run)
    assert_clean(run)


def test_worker_stall_is_finite_and_cleanup():
    graph = Graph()
    graph.source("stall", lambda context: (time.sleep(10),))
    with graph.run(worker_timeout=1.5) as run:
        with pytest.raises(RunError, match="stage 'stall'.*busy exceeded 1.5s"):
            list(run)
    assert_clean(run)


def test_slow_consumer_backpressure_and_early_cancellation(tmp_path):
    log = tmp_path / "source-log"

    def endless(context):
        index = 0
        while True:
            with log.open("a") as file:
                file.write(f"{index}\n")
            yield index
            index += 1

    graph = Graph()
    root = graph.source("source", endless)
    graph.map("map", identity, inputs=[root], replicas=2)
    with graph.run(capacity=2, output_capacity=2, worker_timeout=8) as run:
        assert isinstance(next(run), int)
        wait_until(lambda: run.status["output_buffered"] == 2)
        time.sleep(0.3)
        first = len(log.read_text().splitlines())
        time.sleep(0.3)
        assert len(log.read_text().splitlines()) == first
        # 1 consumed + 2 final + 2 map result credits + 2 queued + 1 source credit.
        assert first <= 8
        assert run.status["peak_output_buffered"] <= 2
        assert run.status["peak_queued"]["map"] <= 2
    assert run.status["state"] == "cancelled"
    assert_clean(run)
    run.close()
    assert list(run) == []


def test_abandoned_consumer_timeout_cleans_without_next():
    def endless(context):
        while True:
            yield None

    graph = Graph()
    graph.source("source", endless)
    run = graph.run(output_capacity=1, consumer_timeout=0.3, worker_timeout=8)
    wait_until(lambda: run._finished.is_set())
    assert run.status["state"] == "failed"
    assert_clean(run)
    with pytest.raises(RunError, match="final consumer stalled"):
        next(run)


def test_immediate_close_cleans_during_startup():
    before = {child.pid for child in mp.active_children()}
    graph = Graph()
    graph.source("source", numbers, replicas=4)
    run = graph.run(worker_timeout=8)
    run.close()
    assert {child.pid for child in mp.active_children()} == before
    assert_clean(run)


def test_consumption_resets_consumer_deadline():
    def endless(context):
        while True:
            yield None

    graph = Graph()
    graph.source("source", endless)
    with graph.run(output_capacity=1, consumer_timeout=0.5, worker_timeout=8) as run:
        for _ in range(8):
            assert next(run) is None
            time.sleep(0.15)
        assert run.status["state"] == "running"
    assert_clean(run)


def test_graph_validation_and_snapshot():
    graph = Graph()
    with pytest.raises(ValueError, match="at least one source"):
        graph.run()
    root = graph.source("source", empty)
    with pytest.raises(ValueError, match="unique"):
        graph.source("source", empty)
    with pytest.raises(ValueError, match="positive integer"):
        graph.source("invalid", empty, replicas=True)
    with pytest.raises(ValueError, match="at least one input"):
        graph.map("invalid", identity, inputs=[])
    with pytest.raises(ValueError, match="duplicate"):
        graph.map("invalid", identity, inputs=[root, root])
    other = Graph().source("other", empty)
    with pytest.raises(ValueError, match="existing nodes"):
        graph.map("invalid", identity, inputs=[other])
    for settings in ({"capacity": 0}, {"worker_timeout": float("inf")}, {"consumer_timeout": 0}):
        with pytest.raises(ValueError):
            graph.run(**settings)
    with graph.run(worker_timeout=8) as run:
        graph.source("later", lambda context: ["not in snapshot"])
        with pytest.raises(ValueError, match="batch size"):
            list(run.batches(0))
        assert list(run) == []
    assert len(run.status["worker_pids"]) == 1
