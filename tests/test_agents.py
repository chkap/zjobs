import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError

import pytest

from zjobs import Graph, RunError


def wait_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail("condition did not become true before deadline")


def free_endpoint(host="127.0.0.1"):
    with socket.socket() as sock:
        sock.bind((host, 0))
        return f"tcp://{host}:{sock.getsockname()[1]}"


def pid_running(pid):
    # All integration agents in this suite are on this one Linux host.
    try:
        with open(f"/proc/{pid}/stat") as file:
            return file.read().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


class Agent:
    def __init__(
        self, tmp_path, machine, host="127.0.0.1", max_workers=256,
        capability_override=None,
    ):
        self.endpoint = free_endpoint(host)
        self.log = tmp_path / f"{machine}-{time.monotonic_ns()}.log"
        self.file = self.log.open("w")
        command = [
            sys.executable, "-m", "zjobs.agent", "--machine", machine,
            "--bind", self.endpoint, "--advertise", self.endpoint,
            "--max-workers", str(max_workers),
        ]
        if capability_override:
            script = tmp_path / f"{machine}-incompatible.py"
            script.write_text(f"""
import zjobs.agent as agent
if __name__ == "__main__":
    original = agent.capabilities
    def altered():
        return {{**original(), **{capability_override!r}}}
    agent.capabilities = altered
    agent.serve({machine!r}, {self.endpoint!r}, {self.endpoint!r})
""")
            command = [sys.executable, str(script)]
        self.process = subprocess.Popen(command, stdout=self.file, stderr=subprocess.STDOUT)
        wait_until(lambda: self.log.stat().st_size > 0 or self.process.poll() is not None)
        assert self.process.poll() is None, self.log.read_text()
        assert json.loads(self.log.read_text().splitlines()[0])["event"] == "ready"

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.file.close()


@pytest.fixture
def agents(tmp_path):
    group = [Agent(tmp_path, name) for name in ("a", "b")]
    try:
        yield group, {name: agent.endpoint for name, agent in zip(("a", "b"), group)}
    finally:
        for agent in group:
            agent.close()


def check_clean(run):
    assert run._finished.is_set()
    assert not run._thread.is_alive()
    wait_until(lambda: not any(pid_running(pid) for pid in run.status["worker_pids"]))


def coordinates(context):
    for value in context.shard(range(31)):
        yield (
            value, context.rank, context.replicas, context.machine,
            context.local_rank, context.local_replicas,
            context.global_rank, context.global_replicas,
        )


def test_distributed_placement_sharding_class_instances_and_snapshot(agents):
    _, endpoints = agents
    placement = {"b": 2, "a": 1}

    class Annotate:
        def __init__(self):
            self.count = 0

        def __call__(self, item):
            time.sleep(0.01)
            self.count += 1
            return item, os.getpid(), self.count

    graph = Graph()
    root = graph.source("source", coordinates, placement=placement)
    graph.map("map", Annotate, inputs=[root], placement={"a": 2, "b": 1})
    placement["a"] = 99
    with graph.run(agents=endpoints, capacity=2, output_capacity=2, worker_timeout=8) as run:
        endpoints["a"] = "tcp://127.0.0.1:1"
        assert [(p.machine, p.rank, p.local_rank, p.local_replicas)
                for p in run.placement if p.stage == "source"] == [
                    ("a", 0, 0, 1), ("b", 1, 0, 2), ("b", 2, 1, 2),
                ]
        with pytest.raises(FrozenInstanceError):
            run.placement[0].machine = "changed"
        result = list(run)
    assert sorted(item[0] for item, _, _ in result) == list(range(31))
    for item, _, _ in result:
        value, rank, replicas, machine, local_rank, local_replicas, global_rank, global_replicas = item
        assert value % 3 == rank == global_rank
        assert replicas == global_replicas == 3
        assert (machine, local_rank, local_replicas) == (
            ("a", 0, 1) if rank == 0 else ("b", rank - 1, 2)
        )
    per_pid = {}
    for _, pid, count in result:
        per_pid.setdefault(pid, []).append(count)
    assert len(per_pid) == 3
    assert all(sorted(counts) == list(range(1, len(counts) + 1)) for counts in per_pid.values())
    assert len(run.status["workers"]) == 6
    assert run.status["state"] == "completed"
    check_clean(run)


def test_streaming_fanout_fanin_and_partial_batches(agents, tmp_path):
    _, endpoints = agents
    marker = tmp_path / "exhausted"

    def source(context):
        yield None
        time.sleep(0.5)
        marker.write_text("yes")
        yield b"end"

    graph = Graph()
    root = graph.source("source", source, placement={"a": 1})
    left = graph.map("left", lambda value: ("left", value), inputs=[root], placement={"b": 2})
    right = graph.map("right", lambda value: ("right", value), inputs=[root], placement={"a": 1})
    graph.map("merge", lambda value: value, inputs=[left, right], placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, capacity=1, output_capacity=1, worker_timeout=8) as run:
        first = next(run)
        assert first[1] is None
        assert not marker.exists()
        batches = list(run.batches(2))
        assert [len(batch) for batch in batches] == [2, 1]
        result = [first] + [item for batch in batches for item in batch]
    assert {side for side, _ in result} == {"left", "right"}
    assert sum(value is None for _, value in result) == 2
    assert sum(value == b"end" for _, value in result) == 2
    assert run.status["produced"] == {"source": 2, "left": 2, "right": 2, "merge": 4}
    assert all(peak <= 1 for peak in run.status["peak_queued"].values())
    check_clean(run)


def test_empty_completion_and_agent_reuse(agents):
    group, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda context: (), placement={"a": 1, "b": 2})
    graph.map("map", lambda item: item, inputs=[root], placement={"b": 2})
    for _ in range(2):
        with graph.run(agents=endpoints, worker_timeout=8) as run:
            assert list(run.batches(3)) == []
        assert run.status["state"] == "completed"
        check_clean(run)
    assert all(agent.process.poll() is None for agent in group)


@pytest.mark.parametrize("where", ["class", "dependency", "call"])
def test_remote_initialization_and_task_errors(agents, where):
    _, endpoints = agents

    class Broken:
        def __init__(self):
            raise ValueError("cannot initialize")

        def __call__(self, item):
            return item

    def missing(item):
        import zjobs_dependency_deliberately_absent
        return zjobs_dependency_deliberately_absent.process(item)

    def failed(item):
        raise ValueError("remote task failure")

    graph = Graph()
    root = graph.source("source", lambda context: [1], placement={"a": 1})
    graph.map("bad", {"class": Broken, "dependency": missing, "call": failed}[where],
              inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=8) as run:
        with pytest.raises(RunError, match=r"(?s)stage 'bad'.*machine 'b'.*(cannot initialize|ModuleNotFoundError|remote task failure)"):
            list(run)
    check_clean(run)


@pytest.mark.parametrize("mode", ["crash", "stall"])
def test_remote_worker_crash_and_stall_cleanup(agents, mode):
    _, endpoints = agents

    def fail(item):
        if mode == "crash":
            os._exit(7)
        time.sleep(10)

    graph = Graph()
    root = graph.source("source", lambda context: [1], placement={"a": 1})
    graph.map("failed", fail, inputs=[root], placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=1.5) as run:
        with pytest.raises(RunError, match=r"stage 'failed'.*machine 'b'.*(exited with code 7|busy exceeded)"):
            list(run)
    check_clean(run)


@pytest.mark.parametrize("override", [
    {"protocol": 999}, {"python": [0, 0]}, {"cloudpickle": "incompatible"},
    {"transport": "unsupported"},
])
def test_incompatible_agent_capabilities_fail_before_workers(tmp_path, override):
    agent = Agent(tmp_path, "wrong-version", capability_override=override)
    try:
        graph = Graph()
        graph.source("source", lambda context: (), placement={"wrong-version": 1})
        with graph.run(agents={"wrong-version": agent.endpoint}) as run:
            with pytest.raises(RunError, match="agent 'wrong-version'.*incompatible capabilities"):
                list(run)
        assert run.status["worker_pids"] == []
        check_clean(run)
    finally:
        agent.close()


def test_unreachable_agent_fails_before_launch():
    graph = Graph()
    graph.source("source", lambda context: (), placement={"missing": 1})
    with graph.run(agents={"missing": free_endpoint()}, agent_timeout=0.3) as run:
        with pytest.raises(RunError, match="(agent 'missing'.*unreachable|hello deadline exceeded)"):
            list(run)
    assert run.status["worker_pids"] == []
    check_clean(run)


def test_missing_duplicate_and_explicit_placement_validation(agents):
    _, endpoints = agents
    graph = Graph()
    graph.source("source", lambda context: (), placement={"a": 1})
    with pytest.raises(ValueError, match="missing agent"):
        graph.run(agents={"b": endpoints["b"]})
    with pytest.raises(ValueError, match="duplicate agent endpoints"):
        graph.run(agents={"a": endpoints["a"], "b": endpoints["a"]})
    with pytest.raises(ValueError, match="placement requires agents"):
        graph.run()
    local = Graph()
    local.source("source", lambda context: ())
    with pytest.raises(ValueError, match="explicit placement"):
        local.run(agents=endpoints)
    for placement in ({}, {"a": 0}, {"a": True}, {"": 1}):
        with pytest.raises(ValueError):
            Graph().source("bad", lambda context: (), placement=placement)
    with pytest.raises(ValueError, match="instead of replicas"):
        Graph().source("bad", lambda context: (), replicas=2, placement={"a": 1})
    with graph.run(agents={"a": endpoints["b"]}, agent_timeout=0.5) as run:
        with pytest.raises(RunError, match="duplicate/mismatched machine identity"):
            list(run)
    check_clean(run)


def test_busy_agent_and_capacity_errors(agents, tmp_path):
    _, endpoints = agents
    graph = Graph()
    graph.source("source", lambda context: (time.sleep(10),), placement={"a": 1})
    with graph.run(agents=endpoints, worker_timeout=15) as first:
        wait_until(lambda: first.status["state"] == "running")
        with graph.run(agents=endpoints, worker_timeout=15) as second:
            with pytest.raises(RunError, match="agent 'a'.*busy"):
                list(second)
        assert first.status["state"] == "running"
    check_clean(first)
    constrained = Agent(tmp_path, "small", max_workers=1)
    try:
        graph = Graph()
        graph.source("source", lambda context: (), placement={"small": 2})
        with graph.run(agents={"small": constrained.endpoint}) as run:
            with pytest.raises(RunError, match="exceeds worker capacity"):
                list(run)
        check_clean(run)
    finally:
        constrained.close()


def test_agent_loss_is_finite_and_other_agents_cleanup(agents):
    group, endpoints = agents

    def forever(context):
        if context.machine == "a":
            time.sleep(100)
        while True:
            yield context.machine

    graph = Graph()
    graph.source("source", forever, placement={"a": 1, "b": 1})
    with graph.run(agents=endpoints, output_capacity=2, agent_timeout=0.5, worker_timeout=8) as run:
        next(run)
        group[0].process.kill()
        group[0].process.wait(timeout=3)
        with pytest.raises(RunError, match="agent 'a'.*(heartbeat lost|connection/send failure)"):
            list(run)
    check_clean(run)


def test_cancel_during_agent_registration_is_prompt():
    graph = Graph()
    graph.source("source", lambda context: (), placement={"missing": 1})
    run = graph.run(agents={"missing": free_endpoint()}, agent_timeout=10)
    started = time.monotonic()
    run.close()
    assert time.monotonic() - started < 1
    assert run.status["state"] == "cancelled"
    check_clean(run)


def test_cancellation_and_slow_abandoned_consumer(agents, tmp_path):
    _, endpoints = agents
    marker = tmp_path / "produced"

    def forever(context):
        index = 0
        while True:
            with marker.open("a") as file:
                file.write(f"{index}\n")
            yield index
            index += 1

    graph = Graph()
    root = graph.source("source", forever, placement={"a": 1})
    graph.map("map", lambda item: item, inputs=[root], placement={"b": 2})
    with graph.run(agents=endpoints, capacity=2, output_capacity=2, worker_timeout=8) as run:
        next(run)
        wait_until(lambda: run.status["output_buffered"] == 2)
        time.sleep(0.4)
        first = len(marker.read_text().splitlines())
        time.sleep(0.3)
        assert len(marker.read_text().splitlines()) == first <= 8
        assert run.status["peak_queued"]["map"] <= 2
        assert run.status["peak_output_buffered"] <= 2
    assert run.status["state"] == "cancelled"
    check_clean(run)
    run = graph.run(agents=endpoints, capacity=2, output_capacity=1,
                    consumer_timeout=0.3, worker_timeout=8)
    wait_until(lambda: run._finished.is_set())
    with pytest.raises(RunError, match="final consumer stalled"):
        next(run)
    check_clean(run)


def test_controller_process_loss_expires_agent_lease(agents, tmp_path):
    _, endpoints = agents
    script = tmp_path / "controller.py"
    pidfile = tmp_path / "pids.json"
    script.write_text(f"""
import json
import time
from zjobs import Graph
if __name__ == "__main__":
    g = Graph()
    g.source("source", lambda ctx: (time.sleep(100),), placement={{"a": 1, "b": 1}})
    run = g.run(agents={endpoints!r}, worker_timeout=120)
    while run.status["state"] != "running":
        time.sleep(.02)
    with open({str(pidfile)!r}, "w") as f:
        json.dump(run.status["worker_pids"], f)
    time.sleep(100)
""")
    process = subprocess.Popen([sys.executable, str(script)])
    try:
        wait_until(pidfile.exists)
        pids = json.loads(pidfile.read_text())
        assert len(pids) == 2
        process.kill()
        process.wait(timeout=3)
        wait_until(lambda: not any(pid_running(pid) for pid in pids), timeout=6)
        graph = Graph()
        graph.source("after", lambda context: [context.machine], placement={"a": 1, "b": 1})
        with graph.run(agents=endpoints, worker_timeout=8) as run:
            assert sorted(run) == ["a", "b"]
        check_clean(run)
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def test_non_loopback_tcp_execution(tmp_path):
    host = socket.gethostbyname(socket.gethostname())
    if host.startswith("127."):
        pytest.skip("no non-loopback host address available")
    agent = Agent(tmp_path, "network", host=host)
    try:
        graph = Graph()
        graph.source("source", lambda context: [context.machine, b"\x00\xff"],
                     placement={"network": 2})
        with graph.run(agents={"network": agent.endpoint}, worker_timeout=8) as run:
            result = list(run)
        assert result.count("network") == 2
        assert result.count(b"\x00\xff") == 2
        assert not agent.endpoint.startswith("tcp://127.")
        print(f"non-loopback endpoint: {agent.endpoint} (one physical host)")
        check_clean(run)
    finally:
        agent.close()


def test_eight_by_eight_placement_is_64_global_ranks():
    # Snapshot-only probe: unreachable hosts prevent launch. This is not an
    # eight-host execution claim.
    graph = Graph()
    root = graph.source("io", coordinates, placement={f"host-{i}": 8 for i in range(8)})
    assert root.replicas == 64
    endpoints = {}
    while len(endpoints) < 8:
        endpoint = free_endpoint()
        if endpoint not in endpoints.values():
            endpoints[f"host-{len(endpoints)}"] = endpoint
    run = graph.run(agents=endpoints, agent_timeout=0.1)
    assert len(run.placement) == 64
    assert [p.rank for p in run.placement] == list(range(64))
    assert all(p.replicas == 64 and p.local_replicas == 8 for p in run.placement)
    run.close()
    check_clean(run)
