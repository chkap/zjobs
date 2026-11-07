import os
import threading
import time

import pytest

from zjobs import Graph, RunError
from zjobs._protocol import decode
from zjobs._remote import RemoteTransport
from test_agents import agents, check_clean, wait_until


@pytest.mark.parametrize("mode", ["crash", "stall"])
@pytest.mark.parametrize("remote", [False, True])
def test_retained_input_recovers_on_same_rank_and_machine(agents, tmp_path, mode, remote):
    _, endpoints = agents
    marker = tmp_path / "attempts"

    class Once:
        def __init__(self):
            self.count = 0

        def __call__(self, value):
            self.count += 1
            with marker.open("a") as file:
                file.write(f"{os.getpid()} {self.count}\n")
            if len(marker.read_text().splitlines()) == 1:
                if mode == "crash":
                    os._exit(7)
                time.sleep(10)
            return value, self.count, os.getpid()

    graph = Graph(policy="local_only" if not remote else "balanced")
    root = graph.source("source", lambda ctx: [None, b"bytes", 3],
                        **({"placement": {"a": 1}} if remote else {}))
    graph.map("map", Once, inputs=[root], **({"placement": {"b": 1}} if remote else {}))
    with graph.run(agents=endpoints if remote else None, worker_timeout=1,
                   max_retries=1, max_payload_bytes=4096, retained_items=4,
                   retained_bytes=16384, queue_bytes=4096, output_bytes=4096) as run:
        results = list(run)
    assert [item[0] for item in results] == [None, b"bytes", 3]
    assert results[0][1] == 1  # replacement constructs a fresh instance
    attempts = marker.read_text().splitlines()
    assert len(attempts) == 4
    assert attempts[0].split()[0] != attempts[1].split()[0]
    assert run.status["retries"] == run.status["worker_restarts"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 6
    assert run.status["obligations_pending"] == 0
    assert len(run.status["retry_errors"]) == 1
    assert "logical" in run.status["retry_errors"][0]
    assert run.status["local_items"] + run.status["remote_items"] == 3
    assert all(value <= 4 for value in run.status["peak_retained_items"].values())
    assert all(value <= 16384 for value in run.status["peak_retained_bytes"].values())
    check_clean(run)


def test_partial_broadcast_branch_retry_does_not_repeat_settled_branch(agents, tmp_path):
    _, endpoints = agents
    fast_log, slow_log = tmp_path / "fast", tmp_path / "slow"

    def fast(value):
        with fast_log.open("a") as file:
            file.write("executed\n")
        return "fast", value

    def slow(value):
        while not fast_log.exists():
            time.sleep(.01)
        with slow_log.open("a") as file:
            file.write("executed\n")
        if len(slow_log.read_text().splitlines()) == 1:
            os._exit(4)
        return "slow", value

    graph = Graph()
    root = graph.source("source", lambda ctx: [42], placement={"a": 1})
    graph.map("fast", fast, inputs=[root], placement={"a": 1})
    graph.map("slow", slow, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=3, max_retries=1) as run:
        assert sorted(run) == [("fast", 42), ("slow", 42)]
    assert len(fast_log.read_text().splitlines()) == 1
    assert len(slow_log.read_text().splitlines()) == 2
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    assert run.status["retries"] == 1
    check_clean(run)


def test_lost_dispatch_before_execution_replays_retained_input(agents, tmp_path, monkeypatch):
    _, endpoints = agents
    calls = tmp_path / "calls"
    original = RemoteTransport.send_multipart
    dropped = []

    def send(self, frames):
        if frames[1] == b"task" and not dropped:
            dropped.append(frames)
            return
        original(self, frames)

    monkeypatch.setattr(RemoteTransport, "send_multipart", send)

    def task(value):
        with calls.open("a") as file:
            file.write("executed\n")
        return value

    graph = Graph(policy="local_only")
    root = graph.source("source", lambda ctx: [17], placement={"a": 1})
    graph.map("map", task, inputs=[root], placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=1, max_retries=1,
                   retained_items=2, max_payload_bytes=4096, retained_bytes=8192,
                   queue_bytes=4096, output_bytes=4096) as run:
        assert list(run) == [17]
    assert calls.read_text().splitlines() == ["executed"]
    assert run.status["retries"] == 1 and run.status["remote_bytes"] == 0
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 2
    check_clean(run)


def test_lost_publication_replayed_and_late_attempt_result_rejected(agents, tmp_path, monkeypatch):
    _, endpoints = agents
    marker = tmp_path / "calls"

    def task(value):
        with marker.open("a") as file:
            file.write(f"{os.getpid()}\n")
        return value + 1

    original_poll = RemoteTransport.poll
    held = []
    replaced = []
    original_replace = RemoteTransport.replace

    def replace(self, old, spec, definition, timeout):
        result = original_replace(self, old, spec, definition, timeout)
        replaced.append(old)
        return result

    def poll(self, milliseconds):
        original_poll(self, milliseconds)
        # Hide the first map result AFTER the agent stores it but BEFORE
        # controller acknowledgment. Reinject it after replacement is complete.
        if self.messages and self.messages[0][1] == b"item":
            reference = decode(self.messages[0][2])
            if reference["owner"] == "b" and not held:
                held.append(self.messages.popleft())
        if replaced and held and not getattr(self, "_injected_old", False):
            self.messages.appendleft(held[0])
            self._injected_old = True
        return int(bool(self.messages))

    monkeypatch.setattr(RemoteTransport, "poll", poll)
    monkeypatch.setattr(RemoteTransport, "replace", replace)
    graph = Graph()
    root = graph.source("source", lambda ctx: [4], placement={"a": 1})
    graph.map("map", task, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=1, max_retries=1, retained_items=2,
                   max_payload_bytes=4096, retained_bytes=8192,
                   queue_bytes=4096, output_bytes=4096) as run:
        assert list(run) == [5]
    assert len(marker.read_text().splitlines()) == 2
    assert run.status["retries"] == run.status["stale_results"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 2
    assert all(record["retained_items"] == 0 for record in run.status["agents"].values())
    check_clean(run)


def test_accepted_output_survives_producer_worker_loss_without_reexecution(agents, tmp_path):
    _, endpoints = agents
    marker = tmp_path / "side_effect"
    gate = tmp_path / "consumer_gate"
    killed = False

    def task(value):
        with marker.open("a") as file:
            file.write("executed\n")
        return value + 1

    def downstream(value):
        while not gate.exists():
            time.sleep(.02)
        return value * 2

    graph = Graph()
    root = graph.source("source", lambda ctx: [2], placement={"a": 1})
    middle = graph.map("middle", task, inputs=[root], placement={"b": 1})
    graph.map("downstream", downstream, inputs=[middle], placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=4) as run:
        wait_until(lambda: run.status["completed_work"].get("middle") == 1)
        record = next(record for record in run.status["workers"] if record["stage"] == "middle")
        os.kill(record["pid"], 9)
        killed = True
        wait_until(lambda: run.status["worker_restarts"] == 1)
        gate.write_text("release")
        assert list(run) == [6]
    assert killed and len(marker.read_text().splitlines()) == 1
    assert run.status["retries"] == 0 and run.status["worker_restarts"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 3
    check_clean(run)


def test_buffered_accepted_output_waits_for_replacement_initialization(agents, tmp_path):
    _, endpoints = agents
    log = tmp_path / "calls"

    class Task:
        def __init__(self):
            if log.exists():
                time.sleep(.3)

        def __call__(self, value):
            with log.open("a") as file:
                file.write("called\n")
            return value

    graph = Graph()
    root = graph.source("source", lambda ctx: [1, 2], placement={"a": 1})
    graph.map("map", Task, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, output_capacity=1, worker_timeout=3) as run:
        wait_until(lambda: run.status["completed_work"].get("map") == 2)
        record = next(record for record in run.status["workers"] if record["stage"] == "map")
        os.kill(record["pid"], 9)
        wait_until(lambda: run.status["worker_restarts"] == 1)
        assert list(run) == [1, 2]
    assert log.read_text().splitlines() == ["called", "called"]
    assert run.status["retries"] == 0
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    check_clean(run)


@pytest.mark.parametrize("remote", [False, True])
def test_retry_exhaustion_has_logical_attempt_context(agents, tmp_path, remote):
    _, endpoints = agents
    log = tmp_path / "attempts"

    def task(value):
        with log.open("a") as file:
            file.write("executed\n")
        os._exit(6)

    graph = Graph()
    root = graph.source("source", lambda ctx: [1], **({"placement": {"a": 1}} if remote else {}))
    graph.map("failed", task, inputs=[root], **({"placement": {"b": 1}} if remote else {}))
    with graph.run(agents=endpoints if remote else None, max_retries=2, worker_timeout=2) as run:
        with pytest.raises(RunError, match="stage 'failed'.*logical .*attempt 2.*retries exhausted"):
            list(run)
    assert len(log.read_text().splitlines()) == 3
    assert run.status["retries"] == 2
    assert run.status["obligations_pending"] == 1
    check_clean(run)


def test_source_failure_is_nonreplayable_even_after_emitting(agents, tmp_path):
    _, endpoints = agents
    log = tmp_path / "source"

    def source(ctx):
        with log.open("a") as file:
            file.write("started\n")
        yield 1
        os._exit(9)

    graph = Graph()
    graph.source("source", source, placement={"a": 1})
    with graph.run(agents=endpoints, max_retries=5) as run:
        assert next(run) == 1
        with pytest.raises(RunError, match="non-replayable source failure.*exited with code 9"):
            next(run)
    assert len(log.read_text().splitlines()) == 1
    assert run.status["retries"] == 0
    check_clean(run)


def test_restart_budget_bounds_idle_crash_loop(agents):
    _, endpoints = agents
    class CrashIdle:
        def __init__(self):
            def crash():
                time.sleep(.2)
                os._exit(5)
            threading.Thread(target=crash, daemon=True).start()

        def __call__(self, item):
            return item

    graph = Graph()
    root = graph.source("source", lambda ctx: (time.sleep(10),), placement={"a": 1})
    graph.map("idle", CrashIdle, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, max_worker_restarts=2, worker_timeout=15) as run:
        with pytest.raises(RunError, match="worker restart budget exhausted"):
            list(run)
    assert run.status["worker_restarts"] == 2
    assert run.status["retries"] == 0
    check_clean(run)


def test_deterministic_user_exception_does_not_retry(agents, tmp_path):
    _, endpoints = agents
    log = tmp_path / "calls"

    def task(value):
        with log.open("a") as file:
            file.write("called\n")
        raise ValueError("invalid user task")

    graph = Graph()
    root = graph.source("source", lambda ctx: [1], placement={"a": 1})
    graph.map("map", task, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, max_retries=5) as run:
        with pytest.raises(RunError, match="invalid user task"):
            list(run)
    assert len(log.read_text().splitlines()) == 1
    assert run.status["retries"] == 0
    check_clean(run)


def test_cancel_retried_blocked_call_releases_workers_and_inputs(agents, tmp_path):
    _, endpoints = agents
    log = tmp_path / "calls"

    def task(value):
        with log.open("a") as file:
            file.write("called\n")
        if len(log.read_text().splitlines()) == 1:
            os._exit(7)
        time.sleep(30)

    graph = Graph(policy="local_only")
    root = graph.source("source", lambda ctx: [b"x" * 1024], placement={"a": 1})
    graph.map("map", task, inputs=[root], placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=5) as run:
        wait_until(lambda: log.exists() and len(log.read_text().splitlines()) == 2)
        assert run.status["retries"] == 1
    assert run.status["state"] == "cancelled"
    assert run.status["remote_bytes"] == 0
    assert all(record["retained_bytes"] == 0 for record in run.status["agents"].values())
    check_clean(run)


def test_abandoned_consumer_after_retry_cleans_without_next(agents, tmp_path):
    _, endpoints = agents
    marker = tmp_path / "calls"

    def task(item):
        with marker.open("a") as file:
            file.write("called\n")
        if len(marker.read_text().splitlines()) == 1:
            os._exit(8)
        return bytes([120]) * 3000

    graph = Graph()
    root = graph.source("source", lambda ctx: range(20), placement={"a": 1})
    graph.map("map", task, inputs=[root], placement={"b": 1})
    run = graph.run(agents=endpoints, max_payload_bytes=4096, retained_items=3,
                    retained_bytes=12288, queue_bytes=4096, output_bytes=4096,
                    output_capacity=8, consumer_timeout=.3, worker_timeout=2)
    wait_until(lambda: run._finished.is_set())
    with pytest.raises(RunError, match="final consumer stalled"):
        next(run)
    assert run.status["retries"] == 1
    assert run.status["peak_output_buffered"] == 1
    assert run.status["peak_output_bytes"] <= 4096
    assert all(value <= 12288 for value in run.status["peak_retained_bytes"].values())
    assert all(record["retained_bytes"] == 0 for record in run.status["agents"].values())
    check_clean(run)


def test_retry_limits_validation():
    graph = Graph()
    graph.source("source", lambda ctx: ())
    for args in ({"max_retries": -1}, {"max_retries": True}, {"max_worker_restarts": -1}):
        with pytest.raises(ValueError, match="nonnegative integer"):
            graph.run(**args)
