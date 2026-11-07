import multiprocessing as mp
import threading
import time

import numpy as np
import pytest
import zmq

from zjobs import Graph, Limits, Locality, PeerConfig, Run, RunError
from test_peer_graph import clean, endpoints, wait_until


@pytest.mark.parametrize("lost", ["grant", "reject"])
def test_missing_claim_response_has_deadline_even_while_renewing(monkeypatch, lost):
    send = Run._send
    def drop(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "grant":
            if lost == "reject":
                # A raced owner rejects; dropping that rejection must also fail.
                return
            return
        return send(self, recipient, kind, body, frames, optional=optional)
    if lost == "reject":
        arbitrate = Run._arbitrate
        def raced(self):
            for _, body in self._claim_queue:
                output = self._store[body["output"]]
                output.obligations[body["child"]] = ("other", "other", time.monotonic() + 5)
            arbitrate(self)
        monkeypatch.setattr(Run, "_arbitrate", raced)
        def drop_reject(self, recipient, kind, body, frames=(), *, optional=False):
            if kind == "reject":
                return
            return send(self, recipient, kind, body, frames, optional=optional)
        monkeypatch.setattr(Run, "_send", drop_reject)
    else:
        monkeypatch.setattr(Run, "_send", drop)
    graph = Graph()
    root = graph.source("raw", lambda ctx: [1])
    graph.map("map", lambda x: x, inputs=[root])
    started = time.monotonic()
    run = graph.run(peer=PeerConfig(claim_timeout=.4, lease_timeout=.3))
    try:
        wait_until(run._finished.is_set, timeout=3)
        with pytest.raises(RunError, match="claim response deadline"):
            next(run)
        assert time.monotonic() - started < 3
        assert run.status["obligations_created"] > run.status["obligations_acked"]
        assert run.status["renewals"] > 0
    finally:
        with pytest.raises(RunError):
            run.close()
    clean(run)


def test_response_timeout_does_not_apply_to_safely_renewed_active_execution():
    graph = Graph()
    root = graph.source("raw", lambda ctx: range(3))
    def task(value):
        time.sleep(.25)
        return value
    graph.map("map", task, inputs=[root])
    with graph.run(peer=PeerConfig(claim_timeout=.05, lease_timeout=.12),
                   limits=Limits(retained_items=1, output_items=1, consumer_timeout=2)) as run:
        wait_until(lambda: run.status.get("output_items", 0) == 1)
        time.sleep(.25)
        assert list(run) == [0, 1, 2]
    assert run.status["renewals"] > 5
    clean(run)


def test_lost_branch_ack_cannot_complete_and_lease_failure_is_bounded(monkeypatch):
    send = Run._send
    def drop(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "ack" and body["child"] == "right":
            return
        return send(self, recipient, kind, body, frames, optional=optional)
    monkeypatch.setattr(Run, "_send", drop)
    graph = Graph()
    root = graph.source("raw", lambda ctx: [None])
    graph.map("left", lambda x: ("left", x), inputs=[root])
    graph.map("right", lambda x: ("right", x), inputs=[root])
    run = graph.run(peer=PeerConfig(lease_timeout=.3))
    with pytest.raises(RunError, match="active claim lease expired"):
        with run:
            list(run)
    assert not run.status["iteration_done"]
    assert run.status["obligations_created"] == 4
    assert run.status["obligations_acked"] == 3
    clean(run)


def test_duplicate_claim_does_not_retransfer_payload_after_grant(monkeypatch):
    send = Run._send
    arbitrate = Run._arbitrate
    requests, grants = [], []
    def duplicate(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "claim":
            requests.append((recipient, dict(body)))
            send(self, recipient, kind, body, frames, optional=optional)
        if kind == "grant":
            grants.append(body["token"])
        return send(self, recipient, kind, body, frames, optional=optional)
    def replay(self):
        arbitrate(self)
        while requests:
            recipient, body = requests.pop()
            send(self, recipient, "claim", body)
        arbitrate(self)
    monkeypatch.setattr(Run, "_send", duplicate)
    monkeypatch.setattr(Run, "_arbitrate", replay)
    graph = Graph()
    root = graph.source("raw", lambda ctx: range(4))
    graph.map("map", lambda x: x, inputs=[root])
    with graph.run() as run:
        assert sorted(run) == list(range(4))
    assert len(grants) == len(set(grants)) == 4
    assert run.status["stale_messages"] >= 8
    clean(run)


def test_combined_large_payload_reservations_and_actual_peak_accounting():
    graph = Graph()
    root = graph.source("raw", lambda ctx: (np.full(20000, i, dtype=np.int64) for i in range(10)))
    first = graph.map("first", lambda x: x, inputs=[root], replicas=2)
    graph.batch("batch", inputs=[first], size=3, max_bytes=330000, replicas=2)
    limits = Limits(max_payload_bytes=400000, retained_items=4, retained_bytes=1600000,
                    claim_items=8, claim_bytes=660000, output_items=1, output_bytes=400000)
    with graph.run(limits=limits) as run:
        wait_until(lambda: run.status.get("output_items", 0) == 1)
        time.sleep(.15)
        values = [int(x[0]) for batch in run for x in batch]
    assert sorted(values) == list(range(10))
    for stage in run.status["stages"].values():
        peaks = stage["buffer_peaks"]
        assert peaks["actual_retained_bytes"] <= limits.retained_bytes
        assert peaks["result_reserved_bytes"] <= limits.retained_bytes
        assert peaks["actual_claim_bytes"] <= limits.claim_bytes
        assert peaks["transfer_reserved_bytes"] <= limits.claim_bytes
        assert stage["peak_retained_bytes"] <= limits.retained_bytes
        assert stage["peak_claim_bytes"] <= limits.claim_bytes
        assert all(v == 0 for v in stage["buffers"].values())
    batch = run.status["stages"]["batch"]["buffer_peaks"]
    assert batch["batch_bytes"] > 160000 and batch["batch_bytes"] <= 660000
    assert run.status["peak_output_bytes"] <= 400000
    clean(run)


def test_pipeline_reports_real_worker_overlap_cpu_and_parallel_capacity():
    graph = Graph()
    def source(ctx):
        for i in range(10):
            time.sleep(.02)
            yield i
    def compute(x):
        started = time.process_time()
        while time.process_time() - started < .015:
            x += 0
        time.sleep(.02)
        return x
    root = graph.source("io", source, replicas=2)
    first = graph.map("cpu", compute, inputs=[root], replicas=2)
    graph.map("final", compute, inputs=[first], replicas=2)
    with graph.run() as run:
        assert sorted(run) == sorted(list(range(10)) * 2)
    execution = [s["execution"] for s in run.status["stages"].values()]
    wall = max(s["last_finished"] for s in execution) - min(s["first_started"] for s in execution)
    busy = sum(s["worker_wall_seconds"] for s in execution)
    assert busy > wall * 1.2  # Returned worker timestamps prove overlapping execution.
    assert sum(s["worker_cpu_seconds"] for s in execution) > .4
    assert run.status["peak_in_flight"] >= 3
    assert all(s["peak_in_flight"] == 2 for s in execution)
    clean(run)


def mesh_probe(identity, peers, queue, mode):
    send, candidate, receive, schedule = Run._send, Run._candidate, Run._receive_claim, Run._schedule
    selections, admissions, grants = [], [], []
    pending = {}
    def choose(self, slot):
        original = slot.selection.choose
        eligible = []
        def observe(owners, local, policy):
            eligible[:] = sorted(owners)
            return original(owners, local, policy)
        slot.selection.choose = observe
        try:
            inp = candidate(self, slot)
        finally:
            slot.selection.choose = original
        if inp and slot.stage.name == "first":
            policy = self.graph.policy(slot.stage, slot.rank)
            record = {
                "owner": inp.owner, "eligible": list(eligible), "streak": slot.selection.streak,
                "policy": policy.mode, "rank": slot.rank, "token": inp.token,
            }
            selections.append(record)
            pending[inp.token] = record
        return inp
    def accepted(self, sender, kind, body, frames):
        receive(self, sender, kind, body, frames)
        if body["token"] in pending:
            record = pending.pop(body["token"])
            record["outcome"] = kind
            if kind == "grant":
                admissions.append(record)
    def transmit(self, recipient, kind, body, frames=(), *, optional=False):
        if kind == "grant" and body["child"] == "first":
            grants.append({"recipient": recipient, "token": body["token"],
                           "pending_owners": sorted(set(p for p, b in self._claim_queue
                                                       if b["child"] == "first"))})
        if kind == "state":
            # Delayed duplicate older snapshots arrive after the current one.
            send(self, recipient, kind, body, frames, optional=optional)
            return send(self, recipient, kind, body, frames, optional=optional)
        return send(self, recipient, kind, body, frames, optional=optional)
    def synchronized(self):
        if len(self._states) == len(self._others) and all(
            "raw" in s["sealed"] for s in self._states_all().values()):
            return schedule(self)
        saved = [(s, s.ready) for s in self._slots if s.stage.kind != "source"]
        for slot, _ in saved:
            slot.ready = False
        try:
            schedule(self)
        finally:
            for slot, ready in saved:
                slot.ready = ready
    Run._candidate, Run._receive_claim, Run._send, Run._schedule = choose, accepted, transmit, synchronized
    try:
        graph = Graph(locality=Locality("balanced"))
        root = graph.source("raw", lambda ctx: [(ctx.peer, i) for i in range(30)])
        first = graph.map("first", lambda x: (time.sleep(.02), x)[1], inputs=[root],
                          locality=Locality("local_only"), replicas=2,
                          rank_locality={0: Locality(mode, max_local_streak=2)})
        left = graph.map("left", lambda x: ("left", x), inputs=[first])
        right = graph.map("right", lambda x: ("right", x), inputs=[first])
        graph.batch("union", inputs=[left, right], size=7, replicas=2)
        with graph.run(peer=PeerConfig(identity, "mesh-"+mode, peers[identity], peers),
                       limits=Limits(retained_items=64)) as run:
            values = [x for batch in run for x in batch]
        clean(run)
        queue.put({"ok": True, "identity": identity, "values": values, "status": run.status,
                   "selections": selections, "admissions": admissions, "grants": grants})
    except BaseException as exc:
        queue.put({"ok": False, "identity": identity, "error": str(exc)})


@pytest.mark.parametrize("mode", ["local_first", "balanced"])
def test_actual_three_peer_selection_owner_arbitration_and_branch_union(mode):
    peers = endpoints()
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=mesh_probe, args=(p, peers, queue, mode)) for p in "CAB"]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=20) for _ in processes]
        assert all(r["ok"] for r in rows), rows
        expected = sorted((branch, (p, i)) for branch in ["left", "right"]
                          for p in peers for i in range(30))
        actual = sorted((v[0], tuple(v[1])) for row in rows for v in row["values"])
        assert actual == expected
        assert sum(r["status"]["obligations_created"] for r in rows) == sum(
            r["status"]["obligations_acked"] for r in rows)
        stable_windows = 0
        for row in rows:
            identity = row["identity"]
            assert row["status"]["stale_messages"] > 0
            assert row["status"]["stages"]["first"]["policies"] == [
                {"mode": mode, "max_local_streak": 2},
                {"mode": "local_only", "max_local_streak": 8},
            ]
            admissions = row["admissions"]
            assert any(v["owner"] != identity for v in admissions)
            assert all(v["owner"] == identity for v in admissions if v["rank"] == 1)
            if mode == "local_first":
                both = [v for v in row["selections"] if identity in v["eligible"] and
                        len(v["eligible"]) > 1 and v["rank"] == 0]
                assert both
                assert all((v["owner"] == identity) == (v["streak"] < 2) for v in both)
            for owner in peers:
                assert any(v["owner"] == owner for v in admissions), (identity, owner, admissions)
            bound = 6 if mode == "local_first" else 3
            sequence = [v for v in row["selections"]
                        if v["rank"] == 0 and "outcome" in v]
            for start in range(len(sequence) - bound + 1):
                window = sequence[start:start + bound]
                if all(v.get("outcome") == "grant" and v["eligible"] == sorted(peers)
                       for v in window):
                    stable_windows += 1
                    required = set(peers) - {identity} if mode == "local_first" else set(peers)
                    assert required.issubset(v["owner"] for v in window), window
        for owner in peers:
            row = next(r for r in rows if r["identity"] == owner)
            assert all(any(g["recipient"] == recipient for g in row["grants"])
                       for recipient in peers)
        assert stable_windows > 0
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


def test_owner_arbitration_local_request_participates_before_remote_grants(monkeypatch):
    schedule, arbitrate = Run._schedule, Run._arbitrate
    rounds = []
    def scheduled(self):
        schedule(self)
        if any(s.pending for s in self._slots):
            rounds.append("local_request")
    def owner(self):
        if self._claim_queue:
            rounds.append("arbitration")
        arbitrate(self)
    monkeypatch.setattr(Run, "_schedule", scheduled)
    monkeypatch.setattr(Run, "_arbitrate", owner)
    graph = Graph()
    root = graph.source("raw", lambda ctx: [1])
    graph.map("map", lambda x: x, inputs=[root])
    with graph.run() as run:
        assert list(run) == [1]
    assert rounds[:2] == ["local_request", "arbitration"]
    clean(run)


def fault_mesh(identity, peers, queue, fault):
    poll = Run._poll_peers
    send = Run._send
    def triggered(self):
        return (self.peer.identity == "B" and
                sum(self._metrics["local_items"].values()) +
                sum(self._metrics["peer_items"].values()) >= 2)
    def delayed(self):
        if fault == "stall" and triggered(self) and not getattr(self, "_injected", False):
            self._injected = True
            time.sleep(.9)
        poll(self)
    class Congested:
        def __init__(self, socket):
            self.socket = socket
        def send_multipart(self, frames, flags):
            raise zmq.Again()
        def close(self):
            self.socket.close()
    def transmit(self, recipient, kind, body, frames=(), *, optional=False):
        if triggered(self):
            if fault == "partition":
                return
            if (fault == "congestion" and recipient != self.peer.identity and
                    not getattr(self, "_injected", False)):
                self._injected = True
                self._sockets[recipient] = Congested(self._sockets[recipient])
        return send(self, recipient, kind, body, frames, optional=optional)
    Run._poll_peers, Run._send = delayed, transmit
    started = time.monotonic()
    try:
        graph = Graph(locality=Locality("balanced"))
        def source(ctx):
            for i in range(100):
                time.sleep(.01)
                yield (ctx.peer, i)
        raw = graph.source("raw", source)
        graph.map("map", lambda x: (time.sleep(.01), x)[1], inputs=[raw])
        with graph.run(peer=PeerConfig(identity, "fault-"+fault, peers[identity], peers,
                                       heartbeat_timeout=.35, lease_timeout=2),
                       limits=Limits(task_timeout=5, shutdown_timeout=2)) as run:
            list(run)
        queue.put({"identity": identity, "unexpected_success": True})
    except RunError as exc:
        clean(run)
        queue.put({"identity": identity, "error": str(exc),
                   "elapsed": time.monotonic() - started, "status": run.status})


@pytest.mark.parametrize("fault", ["partition", "stall", "congestion"])
def test_live_peer_faults_fail_explicitly_and_release_owned_resources(fault):
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=fault_mesh, args=(p, peers, queue, fault)) for p in peers]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=8) for _ in processes]
        assert all("error" in r for r in rows), rows
        assert all(r["elapsed"] < 4 for r in rows), rows
        assert all(not r["status"]["iteration_done"] for r in rows)
        assert any("heartbeat" in r["error"] or "congested" in r["error"] for r in rows), rows
        if fault == "congestion":
            assert "bounded transport queue congested" in next(
                r["error"] for r in rows if r["identity"] == "B")
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


def test_expired_ack_does_not_resurrect_claim(monkeypatch):
    publish = Run._publish
    def expired(self, slot, frames):
        if slot.inputs:
            inp = slot.inputs[0]
            output = self._store[inp.output]
            output.obligations[inp.child] = (inp.owner, inp.token, time.monotonic() - .01)
        publish(self, slot, frames)
    monkeypatch.setattr(Run, "_publish", expired)
    graph = Graph()
    root = graph.source("raw", lambda ctx: [1])
    graph.map("map", lambda x: x, inputs=[root])
    with pytest.raises(RunError, match="expired claim cannot be renewed/ACKed"):
        with graph.run() as run:
            list(run)
    assert not run.status["iteration_done"]
    clean(run)


def test_duplicate_worker_result_cannot_complete_a_new_attempt(monkeypatch):
    spawn = Run._spawn
    class Duplicating:
        def __init__(self, socket):
            self.socket, self.replay = socket, None
        def recv_multipart(self, flags):
            if self.replay is not None:
                replay, self.replay = self.replay, None
                return replay
            frames = self.socket.recv_multipart(flags)
            if b'"result"' in frames[0]:
                self.replay = frames
            return frames
        def send_multipart(self, frames, flags):
            return self.socket.send_multipart(frames, flags)
        def close(self):
            self.socket.close()
    def duplicate(self, stage, rank):
        slot = spawn(self, stage, rank)
        slot.socket = Duplicating(slot.socket)
        return slot
    monkeypatch.setattr(Run, "_spawn", duplicate)
    graph = Graph()
    root = graph.source("raw", lambda ctx: range(8))
    graph.map("map", lambda x: x, inputs=[root], replicas=2)
    with graph.run() as run:
        assert sorted(run) == list(range(8))
    assert run.status["obligations_created"] == run.status["obligations_acked"] == 16
    assert run.status["stale_messages"] >= 8
    assert run.status["stages"]["map"]["execution"]["attempts_completed"] == 8
    clean(run)


def test_abandoned_shutdown_context_has_its_own_deadline():
    # One successful local iterator staying open cannot make the other peer's
    # close report success without its BYE. Use a separate full peer application.
    peers = endpoints(2)
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=unclosed_peer, args=(p, peers, queue)) for p in peers]
    try:
        for process in processes:
            process.start()
        rows = [queue.get(timeout=8) for _ in processes]
        assert all("error" in row for row in rows), rows
        assert "shutdown acknowledgement deadline" in next(
            row["error"] for row in rows if row["identity"] == "A")
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


def unclosed_peer(identity, peers, queue):
    graph = Graph()
    graph.source("raw", lambda ctx: [])
    run = graph.run(peer=PeerConfig(identity, "unclosed", peers[identity], peers),
                    limits=Limits(shutdown_timeout=.3))
    try:
        assert list(run) == []
        if identity == "B":
            time.sleep(1)
        run.close()
        queue.put({"identity": identity, "unexpected_success": True})
    except RunError as exc:
        with pytest.raises(RunError):
            run.close()
        clean(run)
        queue.put({"identity": identity, "error": str(exc)})
