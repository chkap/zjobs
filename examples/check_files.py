"""Check the invariant union of independently consumed file-pipeline JSON logs."""
import argparse
import hashlib
import json
from pathlib import Path


def check(directory, logs):
    expected = []
    for path in sorted(Path(directory).glob("*.txt")):
        words = path.read_text(encoding="utf-8").lower().split()
        expected.extend([
            {"branch": "words", "file": path.name, "value": len(words)},
            {"branch": "sha256", "file": path.name,
             "value": hashlib.sha256(" ".join(words).encode()).hexdigest()},
        ])
    rows = [json.loads(Path(log).read_text()) for log in logs]
    actual = [value for row in rows for batch in row["batches"] for value in batch]
    key = lambda value: (value["file"], value["branch"])
    if sorted(actual, key=key) != sorted(expected, key=key):
        raise AssertionError("branch fixture union has missing, repeated or incorrect results")
    created = sum(row["status"]["obligations_created"] for row in rows)
    acked = sum(row["status"]["obligations_acked"] for row in rows)
    if created != acked:
        raise AssertionError(f"required obligations not drained: {created} created, {acked} ACKed")
    for row in rows:
        status = row["status"]
        if status["error"] or not status["iteration_done"]:
            raise AssertionError(f"{row['identity']}: not successful: {status['error']}")
        if any(s["retained_items"] or s["retained_bytes"] for s in status["stages"].values()):
            raise AssertionError(f"{row['identity']}: retained work after shutdown")
    return {"files": len(expected) // 2, "branch_values": len(actual),
            "created_acked": created,
            "local_consumed": {r["identity"]: sum(r["batch_sizes"]) for r in rows},
            "peer_items": {r["identity"]: r["status"]["peer_items"] for r in rows}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.directory, args.logs), sort_keys=True))


if __name__ == "__main__":
    main()
