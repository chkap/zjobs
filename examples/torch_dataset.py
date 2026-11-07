"""Optional CPU IterableDataset; zjobs supplies workers, not DataLoader."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit("This optional example requires CPU torch; install it before running torch_dataset.py.") from exc

from loading_pipeline import agent_endpoints, build_graph, generate, run_options


class LoadingDataset(IterableDataset):
    """Yields graph batches as CPU tensors; one active iterator per dataset."""

    def __init__(
        self, directory: Path, count: int, payload_bytes: int, rounds: int = 1000,
        batch_size: int = 4, agents: dict[str, str] | None = None,
    ):
        super().__init__()
        self.graph = build_graph(directory, count, rounds, batch_size, agents=agents)
        self.agents = agents
        self.options = run_options({0: (payload_bytes, "", "")}, batch_size)
        self.run = None
        self.last_status = None
        self._iterating = False
        self._closed = False

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Use DataLoader(num_workers=0): zjobs owns stage workers; nested DataLoader workers are unsupported.")
        if self._closed:
            raise RuntimeError("dataset is closed")
        if self._iterating:
            raise RuntimeError("only one active dataset iterator/run is supported")
        self._iterating = True
        try:
            with self.graph.run(agents=self.agents, **self.options) as run:
                self.run = run
                for batch in run:
                    if self._closed:
                        return
                    yield {
                        "ids": torch.tensor([record["id"] for record in batch]),
                        "features": torch.stack([
                            torch.frombuffer(bytearray(record["data"]), dtype=torch.uint8)
                            for record in batch
                        ]).float().div_(255),
                        "digests": tuple(record["digest"] for record in batch),
                    }
        finally:
            if self.run is not None:
                self.last_status = self.run.status
                self.run = None
            self._iterating = False

    def close(self):
        self._closed = True
        if self.run is not None:
            self.run.close()
            self.last_status = self.run.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--items", type=int, default=11)
    cli.add_argument("--kib", type=int, default=64)
    cli.add_argument("--data-dir", type=Path, help="Existing corpus path on controller and all agent hosts")
    cli.add_argument("--agent", action="append", default=[], metavar="NAME=ENDPOINT")
    cli.add_argument("--stop-after", type=int, help="Exit after this many graph batches, exercising cancellation")
    args = cli.parse_args()
    if min(args.items, args.kib) < 1 or (args.stop_after is not None and args.stop_after < 1):
        cli.error("items/kib/stop-after must be positive")
    try:
        agents = agent_endpoints(args.agent)
    except ValueError as exc:
        cli.error(str(exc))
    if agents and args.data_dir is None:
        cli.error("external agents require identical --data-dir on all hosts")
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="zjobs-torch-") as temporary:
        directory = (args.data_dir or Path(temporary)).resolve()
        if args.data_dir is None:
            generate(directory, args.items, args.kib * 1024)
        with LoadingDataset(directory, args.items, args.kib * 1024, agents=agents) as dataset:
            # batch_size=None preserves full/partial in-graph batches as-is.
            loader = DataLoader(dataset, batch_size=None, num_workers=0)
            seen = set()
            for index, batch in enumerate(loader):
                print({"ids": batch["ids"].tolist(), "shape": list(batch["features"].shape),
                       "device": str(batch["features"].device)})
                for index_in_batch, item in enumerate(batch["ids"].tolist()):
                    if item in seen:
                        raise AssertionError(f"duplicate item {item}")
                    expected = torch.tensor(list((directory / f"{item:06d}.bin").read_bytes()),
                                            dtype=torch.float32).div_(255)
                    if not torch.equal(batch["features"][index_in_batch], expected):
                        raise AssertionError(f"incorrect tensor for item {item}")
                    seen.add(item)
                if args.stop_after is not None and index + 1 >= args.stop_after:
                    break
        if args.stop_after is None and seen != set(range(args.items)):
            raise AssertionError("missing required IDs")
        print({"items": len(seen), "state": dataset.last_status["state"]})


if __name__ == "__main__":
    main()
