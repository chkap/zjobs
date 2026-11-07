"""Static-agent control envelope; only trusted peers may connect."""

import json
import multiprocessing as mp
import sys
import time

import cloudpickle
import zmq

VERSION = 3


def capabilities() -> dict:
    return {
        "protocol": VERSION,
        "python": list(sys.version_info[:2]),
        "cloudpickle": cloudpickle.__version__,
        "transport": "host-owned-v3",
    }


def encode(message: dict) -> bytes:
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def decode(payload: bytes) -> dict:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("control envelope must be an object")
    return value


def configure(socket: zmq.Socket) -> None:
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.setsockopt(zmq.SNDTIMEO, 1000)


def stop_processes(processes: list[mp.Process]) -> None:
    """Global grace periods independent of the number of ranks."""
    deadline = time.monotonic() + 0.5
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 0.5
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join()
        process.close()
