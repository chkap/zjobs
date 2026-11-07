import numpy as np
import pytest

from zjobs import Graph, RunError
from zjobs._payload import dumps, loads, size, group_frames, load_group
from zjobs._remote import RemoteTransport
from zjobs._protocol import decode
from test_agents import agents, check_clean, wait_until


@pytest.mark.parametrize("value", [None, b"", b"large" * 10000, [1, None, {"x": (2, 3)}]])
def test_ordinary_payloads_and_exact_limit(value):
    frames = dumps(value, 1000000)
    assert loads(frames) == value
    assert dumps(value, size(frames)) == frames
    with pytest.raises(ValueError, match="serialized payload"):
        dumps(value, size(frames) - 1)


def test_bytes_subclass_keeps_type_and_attributes():
    class Tagged(bytes):
        pass
    value = Tagged(b"value")
    value.tag = 42
    result = loads(dumps(value, 10000))
    assert type(result).__name__ == "Tagged"
    assert result.tag == 42 and result == value


@pytest.mark.parametrize("layout", ["c", "f", "slice", "readonly", "empty", "object"])
def test_array_shapes_dtypes_layout_and_isolated_mutability(layout):
    value = np.arange(262144, dtype=np.float64).reshape(512, 512)
    if layout == "f":
        value = np.asfortranarray(value)
    elif layout == "slice":
        value = value[::2, ::3]
    elif layout == "readonly":
        value.flags.writeable = False
    elif layout == "empty":
        value = np.empty((0, 3), dtype=np.int16)
    elif layout == "object":
        value = np.array([None, {"a": 1}], dtype=object)
    frames = dumps(value, 4000000)
    result = loads(frames)
    np.testing.assert_array_equal(result, value)
    assert result.dtype == value.dtype and result.shape == value.shape
    assert result.flags.writeable == value.flags.writeable
    if layout in ("c", "f", "readonly"):
        assert len(frames) == 3
        assert len(frames[1]) < 1000 and len(frames[2]) == value.nbytes
        assert result.flags.c_contiguous == value.flags.c_contiguous
        assert result.flags.f_contiguous == value.flags.f_contiguous
    if result.size and result.flags.writeable and layout != "object":
        original = value.copy()
        result.flat[0] = -1
        np.testing.assert_array_equal(value, original)
        np.testing.assert_array_equal(loads(frames), original)


def test_buffer_snapshot_survives_user_mutation_and_batch_directory():
    value = np.arange(1000)
    frames = dumps(value, 10000)
    value[:] = -1
    members = [frames, dumps(None, 1000), dumps(b"data", 1000)]
    restored = load_group(group_frames(members))
    np.testing.assert_array_equal(restored[0], np.arange(1000))
    assert restored[1:] == [None, b"data"]
    with pytest.raises(ValueError, match="batch"):
        load_group(group_frames(members)[:-1])
    with pytest.raises(ValueError, match="batch"):
        load_group(group_frames(members) + [b"extra"])


def test_array_limit_includes_envelope_as_well_as_buffers():
    value = np.zeros(10000, dtype=np.uint8)
    frames = dumps(value, 20000)
    assert size(frames) > value.nbytes
    with pytest.raises(ValueError, match="serialized payload"):
        dumps(value, value.nbytes)
    assert size(dumps(value, size(frames))) == size(frames)


@pytest.mark.parametrize("kind", ["float32", "bfloat16", "complex64", "int64", "bool", "scalar", "empty", "transpose", "grad", "attributes"])
def test_optional_cpu_tensor_roundtrip(kind):
    torch = pytest.importorskip("torch")
    value = torch.arange(24).reshape(4, 6)
    if kind in ("float32", "bfloat16", "complex64", "int64", "bool"):
        value = value.to(getattr(torch, kind))
    elif kind == "scalar":
        value = torch.tensor(3.5)
    elif kind == "empty":
        value = torch.empty((0, 3))
    elif kind == "transpose":
        value = value.T
    elif kind == "grad":
        value = value.float().requires_grad_()
    elif kind == "attributes":
        value.note = "preserve"
    frames = dumps(value, 1000000)
    result = loads(frames)
    assert torch.equal(result, value)
    assert result.dtype == value.dtype and result.shape == value.shape
    assert result.device.type == "cpu" and result.requires_grad == value.requires_grad
    assert result.is_contiguous() == value.is_contiguous()
    if kind not in ("transpose", "attributes"):
        assert len(frames) == 3
    if kind == "attributes":
        assert result.note == "preserve"
    if result.numel() and not result.requires_grad:
        result.reshape(-1)[0] = 0
        assert torch.equal(loads(frames), value)


def test_payload_frame_count_is_finitely_bounded():
    values = [np.arange(1) for _ in range(1024)]
    with pytest.raises(ValueError, match="frames"):
        dumps(values, 1000000)


def test_optional_tensor_falls_back_without_numpy(monkeypatch):
    torch = pytest.importorskip("torch")
    import zjobs._payload as codec
    original = codec.importlib.util.find_spec
    monkeypatch.setattr(codec.importlib.util, "find_spec",
                        lambda name: None if name == "numpy" else original(name))
    value = torch.arange(10000)
    frames = dumps(value, 1000000)
    assert len(frames) == 2
    assert torch.equal(loads(frames), value)


@pytest.mark.parametrize("remote", [False, True])
def test_framed_array_fanout_batch_and_final_limits(agents, remote):
    _, endpoints = agents
    graph = Graph()
    def placement(machine):
        return {"placement": {machine: 1}} if remote else {}
    def source(ctx):
        for i in range(5):
            yield np.full((256, 512), i, dtype=np.float32)
    root = graph.source("source", source, **placement("a"))
    group = graph.batch("collect", inputs=[root], size=2, max_bytes=1100000, **placement("b"))
    graph.map("arrays", lambda batch: batch, inputs=[group], **placement("a"))
    def mutate(batch):
        batch[0][0, 0] = -123
        return len(batch)
    graph.map("mutate", mutate, inputs=[group], **placement("b"))
    with graph.run(agents=endpoints if remote else None, max_payload_bytes=1200000,
                   retained_bytes=12000000, queue_bytes=2400000, output_bytes=2400000,
                   retained_items=16) as run:
        result = list(run)
    arrays = [batch for batch in result if isinstance(batch, list)]
    assert [len(batch) for batch in arrays] == [2, 2, 1]
    for i, value in enumerate(value for batch in arrays for value in batch):
        np.testing.assert_array_equal(value, np.full((256, 512), i, dtype=np.float32))
    assert run.status["obligations_created"] == run.status["obligations_settled"]
    assert run.status["peak_grouping_bytes"]["collect"] > 2 * 524288
    assert run.status["peak_grouping_bytes"]["collect"] <= 1100000
    assert run.status["peak_output_bytes"] <= 2400000
    if remote:
        assert run.status["agents"]["b"]["transfers"]["remote_bytes"] > 5 * 524288
        assert all(s["retained_bytes"] == 0 for s in run.status["agents"].values())
    check_clean(run)


@pytest.mark.parametrize("remote", [False, True])
def test_array_limit_counts_buffers_not_just_pickle_envelope(agents, remote):
    _, endpoints = agents
    graph = Graph()
    graph.source("source", lambda ctx: [np.arange(65536, dtype=np.int32)],
                 **({"placement": {"a": 1}} if remote else {}))
    with graph.run(agents=endpoints if remote else None, max_payload_bytes=10000) as run:
        with pytest.raises(RunError, match="serialized payload"):
            list(run)
    check_clean(run)


def test_batch_array_limit_includes_member_buffer_bytes(agents):
    _, endpoints = agents
    graph = Graph()
    source = graph.source("source", lambda ctx: [np.zeros(10000)], placement={"a": 1})
    graph.batch("collect", inputs=[source], size=2, max_bytes=1000, placement={"b": 1})
    with graph.run(agents=endpoints) as run:
        with pytest.raises(RunError, match="member exceeds batch max_bytes"):
            list(run)
    check_clean(run)


def test_framed_group_replay_rejects_retired_publication(agents, monkeypatch):
    _, endpoints = agents
    original_poll = RemoteTransport.poll
    original_replace = RemoteTransport.replace
    held, replacements = [], []
    def poll(self, milliseconds):
        original_poll(self, milliseconds)
        if self.messages and self.messages[0][1] == b"item" and not held:
            if decode(self.messages[0][2])["owner"] == "b":
                held.append(self.messages.popleft())
        if held and replacements and not getattr(self, "_payload_injected", False):
            self.messages.appendleft(held[0])
            self._payload_injected = True
        return int(bool(self.messages))
    def replace(self, old, spec, definition, timeout):
        pid = original_replace(self, old, spec, definition, timeout)
        replacements.append(old)
        return pid
    monkeypatch.setattr(RemoteTransport, "poll", poll)
    monkeypatch.setattr(RemoteTransport, "replace", replace)
    graph = Graph()
    root = graph.source("source", lambda ctx: [np.full(10000, i) for i in range(3)],
                        placement={"a": 1})
    graph.batch("collect", inputs=[root], size=3, placement={"b": 1})
    with graph.run(agents=endpoints, worker_timeout=2, max_retries=1) as run:
        result = list(run)
    assert len(result) == 1 and len(result[0]) == 3
    for i, value in enumerate(result[0]):
        np.testing.assert_array_equal(value, np.full(10000, i))
    assert run.status["retries"] == run.status["stale_results"] == 1
    assert run.status["obligations_created"] == run.status["obligations_settled"] == 4
    assert all(s["retained_bytes"] == 0 for s in run.status["agents"].values())
    check_clean(run)


def test_framed_slow_consumer_cleanup_and_aggregate_final_bounds(agents):
    _, endpoints = agents
    graph = Graph()
    root = graph.source("source", lambda ctx: (np.zeros(65536) for _ in range(20)),
                        placement={"a": 1})
    graph.batch("collect", inputs=[root], size=2, placement={"b": 1})
    run = graph.run(agents=endpoints, max_payload_bytes=1100000, retained_items=10,
                    retained_bytes=11000000, queue_bytes=2200000, output_bytes=1100000,
                    output_capacity=8, consumer_timeout=.3)
    wait_until(lambda: run._finished.is_set())
    with pytest.raises(RunError, match="final consumer stalled"):
        next(run)
    assert run.status["peak_output_buffered"] == 1
    assert 1048576 < run.status["peak_output_bytes"] <= 1100000
    assert all(s["retained_bytes"] == 0 for s in run.status["agents"].values())
    check_clean(run)


def test_optional_tensor_real_remote_workers_and_batch(agents):
    torch = pytest.importorskip("torch")
    _, endpoints = agents
    graph = Graph()
    def source(ctx):
        import torch
        torch.set_num_threads(1)
        for i in range(3):
            yield torch.full((256, 256), float(i))
    root = graph.source("source", source, placement={"a": 1})
    graph.batch("collect", inputs=[root], size=2, placement={"b": 1})
    with graph.run(agents=endpoints) as run:
        batches = list(run)
    assert [len(batch) for batch in batches] == [2, 1]
    for i, value in enumerate(value for batch in batches for value in batch):
        assert torch.equal(value, torch.full((256, 256), float(i)))
        assert value.device.type == "cpu"
    assert run.status["remote_bytes"] > 3 * 256 * 256 * 4
    check_clean(run)
