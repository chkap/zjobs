import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def example(name):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [0, 1, 7])
def test_file_graph_branch_accounting_full_partial_empty_and_check(tmp_path, count):
    files = example("file_pipeline")
    checker = example("check_files")
    directory = tmp_path / "fixture"
    files.make_fixture(directory, count)
    graph = files.build_graph(directory, io_workers=2, cpu_workers=2, batch_size=3)
    with graph.run(limits=files.limits()) as stream:
        batches = list(stream)
    assert sum(map(len, batches)) == count * 2
    if count == 7:
        assert [len(batch) for batch in batches] == [3, 3, 3, 3, 2]
    log = tmp_path / "log.json"
    log.write_text(json.dumps({"identity": "local", "batches": batches,
                              "batch_sizes": list(map(len, batches)), "status": stream.status}))
    assert checker.check(directory, [log])["branch_values"] == count * 2
    if batches:
        batches[0][0]["value"] = "wrong"
        log.write_text(json.dumps({"identity": "local", "batches": batches,
                                  "batch_sizes": list(map(len, batches)), "status": stream.status}))
        with pytest.raises(AssertionError, match="incorrect"):
            checker.check(directory, [log])


@pytest.mark.parametrize("mode", ["slow", "early", "idle", "error"])
def test_lifecycle_commands_expose_expected_failure_and_cleanup(mode):
    result = subprocess.run([sys.executable, str(EXAMPLES / "lifecycle.py"), mode],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert bool(value["expected_error"]) == (mode != "slow")
    assert value["status"]["iteration_done"] == (mode == "slow")
    assert all(s["retained_items"] == s["retained_bytes"] == 0
               for s in value["status"]["stages"].values())


def test_torch_dataset_context_owns_single_iterator_and_early_cleanup(monkeypatch):
    torch = pytest.importorskip("torch")
    adapter = example("torch_dataset")
    dataset = adapter.LocalGraphDataset(adapter.tensor_graph(3))
    with pytest.raises(RuntimeError, match="context before iteration"):
        iter(dataset)
    with dataset:
        loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None)
        values = list(loader)
        assert sorted(v["index"] for v in values) == [0, 1, 2]
        assert all(v["tensor"].device.type == "cpu" for v in values)
        with pytest.raises(RuntimeError, match="one iterator"):
            iter(dataset)
    assert dataset.last_status["iteration_done"]
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="num_workers=0"):
        with dataset:
            pass
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    from zjobs import RunError
    with pytest.raises(RunError, match="cancelled/abandoned"):
        with adapter.LocalGraphDataset(adapter.tensor_graph(8)) as early:
            next(iter(early))
    assert all(s["retained_bytes"] == 0 for s in early.last_status["stages"].values())


def test_torch_nested_dataloader_worker_rejects_before_graph_launch():
    torch = pytest.importorskip("torch")
    adapter = example("torch_dataset")
    dataset = adapter.LocalGraphDataset(adapter.tensor_graph(3))
    loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=None)
    iterator = iter(loader)
    try:
        with pytest.raises(RuntimeError, match="num_workers=0"):
            next(iterator)
        assert dataset.stream is None
    finally:
        iterator._shutdown_workers()
    assert all(not worker.is_alive() for worker in iterator._workers)
