import os
import time

import pytest

from zjobs import Graph, RunError
from zjobs._protocol import decode
from zjobs._remote import RemoteTransport
from test_agents import agents, check_clean, wait_until


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("count", [0, 1, 17])
def test_batch_full_partial_empty_and_none(agents, remote, count):
    _, endpoints = agents
    values = [None if i == 0 else i for i in range(count)]
    graph = Graph()
    root = graph.source("source", lambda ctx: values, **({"placement": {"a": 1}} if remote else {}))
    graph.batch("collect", inputs=[root], size=4, **({"placement": {"b": 1}} if remote else {}))
    with graph.run(agents=endpoints if remote else None) as run:
        batches = list(run)
    assert [item for batch in batches for item in batch] == values
    assert [len(batch) for batch in batches] == [4] * (count // 4) + ([count % 4] if count % 4 else [])
    assert run.status["obligations_created"] == run.status["obligations_settled"] == count + len(batches)
    assert run.status["grouping_items"]["collect"] == 0
    check_clean(run)


def test_independent_placed_parallel_batch_fanin_fanout(agents):
    _, endpoints = agents
    graph = Graph(policy="local_only")
    root = graph.source("source", lambda ctx: ctx.shard(range(21)), placement={"a": 2, "b": 2})
    left = graph.map("left", lambda value: ("left", value), inputs=[root], placement={"a": 1, "b": 1})
    right = graph.map("right", lambda value: ("right", value), inputs=[root], placement={"a": 1, "b": 1})
    batches = graph.batch("collect", inputs=[left, right], size=5, placement={"a": 2, "b": 1})
    graph.map("out1", lambda batch: ("one", batch), inputs=[batches], placement={"a": 1, "b": 1})
    graph.map("out2", lambda batch: ("two", batch), inputs=[batches], placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=4096, retained_items=64,
                   retained_bytes=262144, queue_bytes=32768, output_bytes=32768) as run:
        result = list(run)
    for branch in ("one", "two"):
        members = [value for tag, batch in result if tag == branch for value in batch]
        assert sorted(members) == [(side, i) for side in ("left", "right") for i in range(21)]
        assert all(1 <= len(batch) <= 5 for tag, batch in result if tag == branch)
    assert run.status["obligations_created"] == run.status["obligations_settled"]
    assert run.status["remote_bytes"] == 0
    check_clean(run)


def test_aggregate_byte_bound_flushes_small_groups(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: [bytes([i]) * 900 for i in range(7)], placement={"a": 1})
    graph.batch("collect", inputs=[root], size=10, max_bytes=2000, placement={"b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=4096, retained_bytes=65536,
                   queue_bytes=16384, output_bytes=8192) as run:
        batches = list(run)
    assert [len(batch) for batch in batches] == [2, 2, 2, 1]
    assert [item[0] for batch in batches for item in batch] == list(range(7))
    assert run.status["peak_grouping_items"]["collect"] == 2
    assert run.status["peak_grouping_bytes"]["collect"] <= 2000
    check_clean(run)


@pytest.mark.parametrize("mode", ["drop", "stale", "exhaust"])
def test_group_replay_and_member_settlement(agents, monkeypatch, mode):
    _, endpoints = agents
    original_poll = RemoteTransport.poll
    original_send = RemoteTransport.send_multipart
    original_replace = RemoteTransport.replace
    held, replacements, dispatches = [], [], []

    def send(self, frames):
        if frames[1] == b"batch":
            dispatches.append(decode(frames[2])["members"])
            if mode in ("drop", "exhaust") and (len(dispatches) == 1 or mode == "exhaust"):
                return
        original_send(self, frames)

    def replace(self, old, spec, definition, timeout):
        pid = original_replace(self, old, spec, definition, timeout)
        replacements.append(old)
        return pid

    def poll(self, milliseconds):
        original_poll(self, milliseconds)
        if mode == "stale" and self.messages and self.messages[0][1] == b"item":
            if decode(self.messages[0][2])["owner"] == "b" and not held:
                held.append(self.messages.popleft())
        if held and replacements and not getattr(self, "_batch_stale_injected", False):
            self.messages.appendleft(held[0])
            self._batch_stale_injected = True
        return int(bool(self.messages))

    monkeypatch.setattr(RemoteTransport, "send_multipart", send)
    monkeypatch.setattr(RemoteTransport, "replace", replace)
    monkeypatch.setattr(RemoteTransport, "poll", poll)
    graph = Graph()
    root = graph.source("source", lambda ctx: [None, 2, 3], placement={"a": 1})
    graph.batch("collect", inputs=[root], size=3, placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=1, max_retries=1,
                   max_payload_bytes=4096, retained_items=8, retained_bytes=32768,
                   queue_bytes=4096, output_bytes=4096) as run:
        if mode == "exhaust":
            with pytest.raises(RunError, match="retries exhausted"):
                list(run)
        else:
            assert list(run) == [[None, 2, 3]]
    assert len(dispatches) == 2
    assert dispatches[0] == dispatches[1]
    assert run.status["retries"] == 1
    if mode == "exhaust":
        assert run.status["obligations_pending"] == 3
    else:
        assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    if mode == "stale":
        assert run.status["stale_results"] == 1
    check_clean(run)


def test_batch_group_retained_on_slow_branch_and_cancel(agents, tmp_path):
    _, endpoints = agents
    gate = tmp_path / "gate"

    def slow(batch):
        while not gate.exists():
            time.sleep(.02)
        return batch

    graph = Graph()
    root = graph.source("source", lambda ctx: range(6), placement={"a": 1})
    collected = graph.batch("collect", inputs=[root], size=3, placement={"b": 1})
    graph.map("slow", slow, inputs=[collected], placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=5) as run:
        wait_until(lambda: run.status["completed_work"].get("collect", 0) >= 1)
        assert run.status["obligations_pending"] > 0
    assert run.status["state"] == "cancelled"
    assert all(value["retained_bytes"] == 0 for value in run.status["agents"].values())
    check_clean(run)


def test_accepted_group_survives_collection_worker_loss(agents, tmp_path):
    _, endpoints = agents
    gate = tmp_path / "gate"

    def slow(batch):
        while not gate.exists():
            time.sleep(.02)
        return batch

    graph = Graph()
    root = graph.source("source", lambda ctx: [1, 2, 3], placement={"a": 1})
    collected = graph.batch("collect", inputs=[root], size=3, placement={"b": 1})
    graph.map("slow", slow, inputs=[collected], placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=5) as run:
        wait_until(lambda: run.status["completed_work"].get("collect", 0) == 1)
        worker = next(worker for worker in run.status["workers"] if worker["stage"] == "collect")
        os.kill(worker["pid"], 9)
        wait_until(lambda: run.status["worker_restarts"] == 1)
        gate.write_text("release")
        assert list(run) == [[1, 2, 3]]
    assert run.status["retries"] == 0
    assert run.status["completed_work"]["collect"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 5
    check_clean(run)


def test_group_members_survive_actual_worker_crash_before_dispatch(agents, monkeypatch):
    _, endpoints = agents
    original = RemoteTransport.send_multipart
    crashed = []

    def send(self, frames):
        if frames[1] == b"batch" and not crashed:
            os.kill(next(item["pid"] for item in run.status["workers"] if item["stage"] == "collect"), 9)
            crashed.append(True)
            return
        original(self, frames)

    monkeypatch.setattr(RemoteTransport, "send_multipart", send)
    graph = Graph()
    root = graph.source("source", lambda ctx: [1, 2, 3], placement={"a": 1})
    graph.batch("collect", inputs=[root], size=3, placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=3, max_retries=1) as run:
        assert list(run) == [[1, 2, 3]]
    assert crashed and run.status["retries"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    check_clean(run)


def test_abandoned_batch_consumer_byte_bound_and_cleanup(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: (bytes([1]) * 1000 for _ in range(30)),
                        placement={"a": 1})
    graph.batch("collect", inputs=[root], size=3, max_bytes=3500, placement={"b": 1})
    run = graph.run(agents=endpoints, max_payload_bytes=4096, retained_items=12,
                    retained_bytes=49152, queue_bytes=8192, output_bytes=4096,
                    output_capacity=8, consumer_timeout=.3)
    wait_until(lambda: run._finished.is_set())
    with pytest.raises(RunError, match="final consumer stalled"):
        next(run)
    assert run.status["peak_output_buffered"] == 1
    assert run.status["peak_output_bytes"] <= 4096
    assert run.status["peak_grouping_items"]["collect"] <= 3
    assert run.status["peak_grouping_bytes"]["collect"] <= 3500
    assert all(value <= 49152 for value in run.status["peak_retained_bytes"].values())
    assert all(value["retained_bytes"] == 0 for value in run.status["agents"].values())
    check_clean(run)


def test_batch_rejects_oversize_member_explicitly(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: [bytes([1]) * 1000], placement={"a": 1})
    graph.batch("collect", inputs=[root], size=2, max_bytes=500, placement={"b": 1})
    with graph.run(agents=endpoints) as run:
        with pytest.raises(RunError, match="member exceeds batch max_bytes"):
            list(run)
    check_clean(run)


def test_unserializable_oversize_batch_does_not_settle_members(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: [bytes([i]) * 1000 for i in range(2)],
                        placement={"a": 1})
    graph.batch("collect", inputs=[root], size=2, max_bytes=3000, placement={"b": 1})
    with graph.run(agents=endpoints, max_payload_bytes=1500, retained_bytes=12000,
                   queue_bytes=3000, output_bytes=3000) as run:
        with pytest.raises(RunError, match="serialize output.*|serialized payload"):
            list(run)
    assert run.status["obligations_pending"] == 2
    assert run.status["obligations_settled"] == 0
    check_clean(run)

def test_batch_validation():
    graph = Graph()
    root = graph.source("source", lambda ctx: ())
    for settings in ({"size": 0}, {"size": True}, {"size": 1, "max_bytes": 0}):
        with pytest.raises(ValueError, match="positive integer"):
            graph.batch("bad", inputs=[root], **settings)


def test_full_batch_streams_before_source_exhaustion(agents, tmp_path):
    _, endpoints = agents
    marker = tmp_path / "source_finished"

    def source(ctx):
        yield 1
        yield 2
        time.sleep(.5)
        marker.write_text("yes")
        yield 3

    graph = Graph()
    root = graph.source("source", source, placement={"a": 1})
    graph.batch("collect", inputs=[root], size=2, placement={"b": 1})
    with graph.run(agents=endpoints) as run:
        assert next(run) == [1, 2]
        assert not marker.exists()
        assert list(run) == [[3]]
    check_clean(run)


def test_local_first_batch_mixes_ready_owners_without_losing_members(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: [(ctx.machine, i) for i in ctx.shard(range(12))],
                        placement={"a": 1, "b": 1})
    graph.batch("collect", inputs=[root], size=12, placement={"a": 1}, policy="local_first")
    with graph.run(agents=endpoints, remote_after=2, max_payload_bytes=4096,
                   retained_items=32, retained_bytes=131072, queue_bytes=8192, output_bytes=8192) as run:
        batches = list(run)
    assert len(batches) == 1 and len(batches[0]) == 12
    assert sorted(i for owner, i in batches[0]) == list(range(12))
    assert run.status["local_items"] == run.status["remote_items"] == 6
    assert run.status["agents"]["a"]["transfers"]["remote_items"] == 6
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 13
    check_clean(run)


def test_nested_batches_preserve_list_payloads_and_completion(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: range(11), placement={"a": 1})
    first = graph.batch("first", inputs=[root], size=3, placement={"b": 1})
    graph.batch("second", inputs=[first], size=2, placement={"a": 1})
    with graph.run(agents=endpoints) as run:
        result = list(run)
    assert result == [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10]]]
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 17
    check_clean(run)
