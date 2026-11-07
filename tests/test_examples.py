import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples import loading_pipeline as loading
from test_agents import agents, check_clean, pid_running, wait_until

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("remote", [False, True])
def test_file_pipeline_shards_reads_and_validates_full_partial_batches(tmp_path, agents, remote):
    _, endpoints = agents
    loading.generate(tmp_path, 11, 32768)
    expected = loading.expected_outputs(tmp_path, 11, 1000)
    graph = loading.build_graph(tmp_path, 11, 1000, 4,
                                io_ranks=2, preprocess_ranks=1, collect_ranks=1,
                                agents=endpoints if remote else None)
    result = loading.trial(graph, endpoints if remote else None, expected, 4)
    assert result["items"] == 11
    assert result["checksum"] == loading.checksum({i: value[2] for i, value in expected.items()})
    assert result["obligations_created"] == result["obligations_settled"]
    assert sum(result["source_items"].values()) == 11
    assert all(value > 0 for value in result["source_items"].values())
    assert sum(int(length) * count for length, count in result["batch_length_counts"].items()) == 11
    assert result["service"]["preprocess_cpu_seconds"] > 0
    assert result["time_to_first_batch_seconds"] < result["elapsed_seconds"]
    assert result["peak_output_bytes"] <= result["run_options"]["output_bytes"]
    assert all(value <= result["run_options"]["retained_bytes"] for value in result["peak_retained_bytes"].values())
    if remote:
        assert all(value["retained_bytes"] == 0 for value in result["agents"].values())
        assert sum(value["transfers"]["remote_bytes"] for value in result["agents"].values()) == result["remote_bytes"]
    wait_until(lambda: not any(pid_running(pid) for pid in result["worker_pids"]))


def test_source_shards_ids_before_opening_unowned_files(tmp_path):
    from zjobs import SourceContext
    loading.generate(tmp_path, 6, 1024)
    for index in (1, 3, 5):
        (tmp_path / f"{index:06d}.bin").unlink()
    records = list(loading.load_files(str(tmp_path), 6, SourceContext(0, 2)))
    assert [record["id"] for record in records] == [0, 2, 4]
    assert all(len(record["data"]) == 1024 for record in records)


def test_missing_files_fail_with_stage_and_operation(tmp_path):
    from zjobs import RunError
    graph = loading.build_graph(tmp_path, 1, 1000, 4)
    with graph.run() as run:
        with pytest.raises(RunError, match="stage 'io'.*|source iteration"):
            list(run)
    assert "FileNotFoundError" in str(run.status["error"])
    check_clean(run)


def test_measurement_runner_emits_warmups_trials_and_exact_correctness(tmp_path):
    output = tmp_path / "results.json"
    subprocess.run([
        sys.executable, str(EXAMPLES / "measure_application.py"),
        "--items", "5", "--kib", "8", "--rounds", "1000",
        "--preprocess-ranks", "1", "2", "--warmup", "1", "--trials", "2",
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["physical_hosts"] == 1
    assert len(result["results"]) == 2
    for config in result["results"]:
        assert len(config["warmups"]) == 1 and len(config["trials"]) == 2
        for trial in config["warmups"] + config["trials"]:
            assert trial["checksum"] == result["expected_checksum"]
            assert trial["items"] == 5 and trial["state"] == "completed"
            assert trial["obligations_created"] == trial["obligations_settled"]
            assert trial["elapsed_seconds"] > 0
            assert trial["service"]["preprocess_cpu_seconds"] > 0


@pytest.fixture
def tensor_example(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module("torch_dataset")


@pytest.mark.parametrize("remote", [False, True])
def test_optional_dataset_cpu_values_ids_and_partial_graph_batches(tmp_path, agents, tensor_example, remote):
    import torch
    torch.set_num_threads(1)
    _, endpoints = agents
    loading.generate(tmp_path, 11, 1024)
    with tensor_example.LoadingDataset(tmp_path, 11, 1024, batch_size=4,
                                      agents=endpoints if remote else None) as dataset:
        loader = tensor_example.DataLoader(dataset, batch_size=None, num_workers=0)
        ids, lengths = set(), []
        for batch in loader:
            assert batch["features"].device.type == "cpu"
            assert batch["features"].dtype == torch.float32
            lengths.append(len(batch["ids"]))
            for offset, item in enumerate(batch["ids"].tolist()):
                assert item not in ids
                ids.add(item)
                data = (tmp_path / f"{item:06d}.bin").read_bytes()
                expected = torch.tensor(list(data), dtype=torch.float32).div_(255)
                assert torch.equal(batch["features"][offset], expected)
                assert batch["digests"][offset] == hashlib.pbkdf2_hmac("sha256", data, loading.SALT, 1000).hex()
    assert ids == set(range(11)) and sum(lengths) == 11
    assert any(length < 4 for length in lengths)
    assert dataset.last_status["state"] == "completed"
    wait_until(lambda: not any(pid_running(pid) for pid in dataset.last_status["worker_pids"]))


@pytest.mark.parametrize("remote", [False, True])
def test_dataset_context_closes_abandoned_loader_without_gc(tmp_path, agents, tensor_example, remote):
    _, endpoints = agents
    loading.generate(tmp_path, 64, 4096)
    with tensor_example.LoadingDataset(tmp_path, 64, 4096, rounds=10000,
                                      agents=endpoints if remote else None) as dataset:
        loader = tensor_example.DataLoader(dataset, batch_size=None, num_workers=0)
        iterator = iter(loader)
        assert len(next(iterator)["ids"]) == 4
        active = dataset.run
        assert active is not None
    # Retaining loader+iterator must not keep the owned workers alive.
    assert dataset.last_status["state"] == "cancelled"
    if remote:
        assert all(agent["retained_bytes"] == 0 for agent in dataset.last_status["agents"].values())
    check_clean(active)
    dataset.close()


def test_dataset_rejects_nested_workers_and_concurrent_runs(tmp_path, tensor_example, monkeypatch):
    loading.generate(tmp_path, 8, 1024)
    dataset = tensor_example.LoadingDataset(tmp_path, 8, 1024)
    monkeypatch.setattr(tensor_example, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(dataset))
    assert dataset.run is None
    monkeypatch.setattr(tensor_example, "get_worker_info", lambda: None)
    with dataset:
        iterator = iter(dataset)
        next(iterator)
        with pytest.raises(RuntimeError, match="one active"):
            next(iter(dataset))
        active = dataset.run
    check_clean(active)
    with pytest.raises(RuntimeError, match="closed"):
        next(iter(dataset))


def test_dataset_cleanup_when_source_fails(tmp_path, tensor_example):
    from zjobs import RunError
    with tensor_example.LoadingDataset(tmp_path, 1, 1024) as dataset:
        with pytest.raises(RunError, match="source iteration"):
            next(iter(dataset))
    assert dataset.last_status["state"] == "failed"
    wait_until(lambda: not any(pid_running(pid) for pid in dataset.last_status["worker_pids"]))


def test_agent_options_reject_duplicates_and_implicit_remote_temp_paths():
    with pytest.raises(ValueError, match="duplicate"):
        loading.agent_endpoints(["a=tcp://localhost:1", "a=tcp://localhost:2"])
    completed = subprocess.run([
        sys.executable, str(EXAMPLES / "loading_pipeline.py"),
        "--agent", "a=tcp://localhost:1",
    ], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "--data-dir" in completed.stderr
