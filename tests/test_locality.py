import os
import time
from collections import deque

import pytest

from zjobs import Graph, RunError
from zjobs._scheduler import Output, choose
from zjobs._remote import RemoteTransport
from zjobs._protocol import decode

from test_agents import agents, check_clean, wait_until


def test_ready_selection_and_finite_remote_budget():
    remote = Output("remote", "b", 1, b"producer", {0})
    local = Output("local", "a", 1, b"producer", {0})
    pending = deque([remote, local])
    assert choose(pending, "a", "local_first", 0, 3) is local
    assert choose(pending, "a", "local_first", 3, 3) is remote
    assert choose(pending, "a", "balanced", 0, 3) is remote
    assert choose(pending, "a", "local_only", 99, 3) is local
    assert choose(deque([remote]), "a", "local_first", 0, 3) is remote
    assert choose(deque([remote]), "a", "local_only", 0, 3) is None
    # An oldest remote item competes with an unending replenished local queue.
    selected = []
    streak = 0
    for index in range(4):
        pending.append(Output(f"local-{index}", "a", 1, b"producer", {0}))
        item = choose(pending, "a", "local_first", streak, 3)
        pending.remove(item)
        selected.append(item.key)
        streak = streak + 1 if item.owner == "a" else 0
    assert selected[-1] == "remote"


def test_policy_precedence_resolves_once(agents):
    _, endpoints = agents
    overrides = {0: "local_first", 1: "local_only"}
    graph = Graph(policy="balanced")
    root = graph.source("source", lambda ctx: (), placement={"a": 1, "b": 1})
    stage = graph.map("map", lambda value: value, inputs=[root], placement={"a": 2, "b": 1},
                      policy="local_only", rank_policies=overrides)
    overrides[0] = "balanced"
    assert stage.rank_policies == ((0, "local_first"), (1, "local_only"))
    with graph.run(agents=endpoints) as run:
        assert [p.policy for p in run.placement] == [
            "balanced", "balanced", "local_first", "local_only", "local_only",
        ]
        graph._policy = "local_only"
        assert list(run) == []
    check_clean(run)


def test_local_only_has_real_same_agent_payload_path(agents):
    _, endpoints = agents
    graph = Graph(policy="local_only")

    def source(context):
        for value in context.shard(range(24)):
            yield context.machine, value, b"x" * 2048

    root = graph.source("source", source, placement={"a": 1, "b": 1})
    graph.map("map", lambda item: (item[0], item[1], os.getpid(), len(item[2])),
              inputs=[root], placement={"a": 2, "b": 2})
    with graph.run(agents=endpoints, capacity=2, max_payload_bytes=4096,
                   retained_items=6, retained_bytes=24576, queue_bytes=8192, output_bytes=8192) as run:
        result = list(run)
    pid_hosts = {record["pid"]: record["machine"] for record in run.status["workers"]}
    assert sorted(value for _, value, _, _ in result) == list(range(24))
    assert all(owner == pid_hosts[pid] and size == 2048 for owner, _, pid, size in result)
    assert run.status["remote_items"] == 0
    assert run.status["local_items"] == 24
    actual = run.status["agents"]
    assert sum(record["transfers"]["remote_bytes"] for record in actual.values()) == 0
    assert sum(record["transfers"]["served_remote_bytes"] for record in actual.values()) == 0
    assert sum(record["transfers"]["local_items"] for record in actual.values()) == 24
    assert sum(record["transfers"]["local_bytes"] for record in actual.values()) > 24 * 2048
    # Only the small mapped final results reach the controller, not source data.
    assert run.status["final_bytes"] < 24 * 2048
    assert run.status["peak_output_buffered"] >= 1
    assert run.status["peak_output_bytes"] > 0
    assert run.status["peak_queued"]["map"] >= 1
    assert sum(record["transfers"]["final_bytes"] for record in actual.values()) == run.status["final_bytes"]
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 48
    assert run.status["obligations_pending"] == 0
    check_clean(run)


def test_impossible_local_only_fails_before_agents_start(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda context: [1], placement={"a": 1})
    graph.map("unreachable", lambda value: value, inputs=[root],
              placement={"b": 1}, policy="local_only")
    with pytest.raises(RunError, match="local_only impossible routing.*machine 'a'.*unreachable"):
        graph.run(agents=endpoints)
    # A rank override making one consumer remotely eligible fixes the topology.
    allowed = Graph(policy="local_only")
    root = allowed.source("source", lambda context: [1], placement={"a": 1})
    allowed.map("allowed", lambda value: value, inputs=[root], placement={"b": 2},
                rank_policies={1: "balanced"})
    with allowed.run(agents=endpoints) as run:
        assert list(run) == [1]
    assert run.status["remote_items"] == 1
    check_clean(run)


def test_local_only_agent_loss_fails_without_remote_spill(agents):
    group, endpoints = agents

    def source(ctx):
        while True:
            yield ctx.machine, b"x" * 512

    graph = Graph(policy="local_only")
    root = graph.source("source", source, placement={"a": 1, "b": 1})
    graph.map("map", lambda item: item, inputs=[root], placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, agent_timeout=.5, worker_timeout=8) as run:
        next(run)
        group[1].process.kill()
        group[1].process.wait(timeout=3)
        with pytest.raises(RunError, match="agent 'b'.*(heartbeat lost|connection/send failure)"):
            list(run)
    assert run.status["remote_items"] == 0
    assert all(assignment["local"] for assignment in run.status["recent_assignments"])
    assert run.status["state"] == "failed"
    check_clean(run)


@pytest.mark.parametrize("policy", ["local_first", "balanced"])
def test_direct_remote_fetch_avoids_controller_input_bytes(agents, policy, monkeypatch):
    _, endpoints = agents
    graph = Graph(policy=policy)
    received_sizes = []
    original_receive = RemoteTransport.recv_multipart

    def receive(self):
        frames = original_receive(self)
        if frames[1] == b"item":
            reference = decode(frames[2])
            assert set(reference) == {"output", "owner", "size"}
            received_sizes.append((len(frames[2]), reference["size"]))
        return frames

    monkeypatch.setattr(RemoteTransport, "recv_multipart", receive)
    root = graph.source("source", lambda context: [b"large" * 600], placement={"a": 1})
    graph.map("map", lambda data: (len(data), data[:5]), inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=4096,
                   retained_bytes=16384, queue_bytes=4096, output_bytes=4096) as run:
        assert list(run) == [(3000, b"large")]
    stats = run.status
    assert stats["remote_items"] == 1 and stats["local_items"] == 0
    assert stats["remote_bytes"] > 3000
    assert stats["agents"]["a"]["transfers"]["served_remote_bytes"] == stats["remote_bytes"]
    assert stats["agents"]["b"]["transfers"]["remote_bytes"] == stats["remote_bytes"]
    assert stats["final_bytes"] < 100
    assert any(size > 3000 and frame_size < 256 for frame_size, size in received_sizes)
    assert stats["obligations_created"] == stats["obligations_settled"] == 2
    check_clean(run)


def test_producer_retains_payload_until_all_dispatched_edges_settle(agents, tmp_path):
    _, endpoints = agents
    gate = tmp_path / "release-slow"

    def slow(item):
        while not gate.exists():
            time.sleep(.02)
        return "slow"

    graph = Graph()
    root = graph.source("source", lambda ctx: [b"x" * 3000], placement={"a": 1})
    graph.map("fast", lambda item: "fast", inputs=[root], placement={"a": 1})
    graph.map("slow", slow, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=4096, worker_timeout=8) as run:
        assert next(run) == "fast"
        wait_until(lambda: any(
            output["bytes"] > 3000 and output["edges"] == ["slow"]
            for output in run.status["pending_outputs"]
        ))
        wait_until(lambda: run.status.get("agents", {}).get("a", {}).get("retained_bytes", 0) >= 3000)
        assert run.status["obligations_pending"] >= 1
        gate.write_text("release")
        assert list(run) == ["slow"]
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    assert all(record["retained_items"] == 0 for record in run.status["agents"].values())
    check_clean(run)


def test_local_first_real_ready_work_preference_and_remote_progress(agents, tmp_path):
    _, endpoints = agents
    ready = tmp_path / "ready"
    local_ready = tmp_path / "local"

    def source(context):
        if context.machine == "a":
            local_ready.write_text("yes")
            for value in range(20):
                yield "local", value
        else:
            while not local_ready.exists():
                time.sleep(.01)
            ready.write_text("yes")
            yield "remote", 0

    class Slow:
        def __call__(self, item):
            time.sleep(.04)
            return item

    graph = Graph()
    root = graph.source("source", source, placement={"a": 1, "b": 1})
    graph.map("map", Slow, inputs=[root], placement={"a": 1})
    with graph.run(agents=endpoints, capacity=24, remote_after=3, worker_timeout=8) as run:
        result = list(run)
    assert len(result) == 21
    assert result.index(("remote", 0)) <= 6
    assert run.status["remote_items"] == 1 and run.status["local_items"] == 20
    check_clean(run)


def test_local_first_matches_ready_local_consumer_before_remote_steal(agents):
    _, endpoints = agents

    def source(ctx):
        # Both consumers initialize before the first payload, then stay ready
        # between items. Scheduling order must not cause needless remote steals.
        time.sleep(.2)
        for index in range(12):
            time.sleep(.04)
            yield index

    graph = Graph()
    root = graph.source("source", source, placement={"a": 1})
    graph.map("map", lambda item: (item, os.getpid()), inputs=[root], placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, worker_timeout=8) as run:
        result = list(run)
    hosts = {worker["pid"]: worker["machine"] for worker in run.status["workers"]}
    assert sorted(item for item, _ in result) == list(range(12))
    assert all(hosts[pid] == "a" for _, pid in result)
    assert run.status["local_items"] == 12
    assert run.status["remote_bytes"] == 0
    assert run.status["agents"]["a"]["transfers"]["served_remote_bytes"] == 0
    check_clean(run)


def test_broadcast_obligations_and_partial_failure_never_report_success(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda context: [None, b"bytes", {"key": 1}], placement={"a": 1})
    left = graph.map("left", lambda item: ("left", item), inputs=[root], placement={"a": 2})
    right = graph.map("right", lambda item: ("right", item), inputs=[root], placement={"b": 2})
    graph.map("merge", lambda item: item, inputs=[left, right], placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, capacity=1) as run:
        result = list(run)
    assert len(result) == 6
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 18
    assert run.status["obligations_pending"] == 0
    assert run.status["state"] == "completed"
    check_clean(run)

    def failed(item):
        time.sleep(.2)
        raise ValueError("required branch failed")

    failure = Graph()
    root = failure.source("source", lambda context: [1], placement={"a": 1})
    failure.map("successful", lambda value: value, inputs=[root], placement={"a": 1})
    failure.map("failed", failed, inputs=[root], placement={"b": 1})
    with failure.run(agents=endpoints) as run:
        with pytest.raises(RunError, match="required branch failed"):
            list(run)
    assert run.status["state"] == "failed"
    assert run.status["obligations_pending"] >= 1
    assert run.status["obligations_created"] > run.status["obligations_settled"]
    check_clean(run)


def test_retained_byte_item_and_final_byte_bounds_for_slow_consumers(agents):
    _, endpoints = agents

    def source(context):
        while True:
            yield b"x" * 3000

    graph = Graph()
    root = graph.source("source", source, placement={"a": 1})
    graph.map("map", lambda item: item, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=4096, retained_items=3,
                   retained_bytes=12288, queue_bytes=4096, output_capacity=8,
                   output_bytes=4096, consumer_timeout=.5) as run:
        wait_until(lambda: run._finished.is_set())
        with pytest.raises(RunError, match="final consumer stalled"):
            next(run)
    stats = run.status
    assert stats["peak_output_buffered"] == 1  # bytes, not the eight-item limit
    assert 3000 <= stats["peak_output_bytes"] <= 4096
    assert all(value <= 3 for value in stats["peak_retained_items"].values())
    assert all(value <= 12288 for value in stats["peak_retained_bytes"].values())
    assert all(value <= 4096 for value in stats["peak_queue_bytes"].values())
    assert all(record["retained_bytes"] == 0 for record in stats["agents"].values())
    assert all(record["peak_retained_bytes"] <= 12288 for record in stats["agents"].values())
    assert all(record["peak_retained_items"] <= 3 for record in stats["agents"].values())
    check_clean(run)


def test_inadequate_retention_fails_with_stalled_obligations(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: [b"x"], placement={"a": 1})
    graph.map("map", lambda item: item, inputs=[root], placement={"a": 1})
    with graph.run(agents=endpoints, retained_items=1, max_payload_bytes=4096,
                   retained_bytes=4096, queue_bytes=4096, output_bytes=4096, worker_timeout=1.5) as run:
        with pytest.raises(RunError, match="routing/capacity stalled.*required obligations"):
            list(run)
    assert run.status["state"] == "failed"
    check_clean(run)


def test_invalid_policy_rank_and_byte_limits():
    with pytest.raises(ValueError, match="unknown locality"):
        Graph(policy="pretend")
    graph = Graph()
    with pytest.raises(ValueError, match="unknown locality"):
        graph.source("invalid", lambda ctx: (), policy="pretend")
    with pytest.raises(ValueError, match="global stage ranks"):
        graph.source("invalid", lambda ctx: (), rank_policies={1: "balanced"})
    graph.source("source", lambda ctx: ())
    with pytest.raises(ValueError, match="fit max_payload_bytes"):
        graph.run(max_payload_bytes=4096, retained_bytes=1024)
