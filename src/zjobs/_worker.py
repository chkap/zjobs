from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback

import cloudpickle
import zmq

from ._payload import dumps, load_group, loads


def worker(endpoint, definition, source_context, limit, kind):
    parent = mp.parent_process()
    if sys.platform == "linux":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "setting worker parent-death signal failed")
    if parent is not None and not parent.is_alive():
        return
    stopped = threading.Event()
    def watch_parent():
        while not stopped.wait(.1):
            if parent is not None and not parent.is_alive():
                os._exit(70)
    guardian = threading.Thread(target=watch_parent, daemon=True)
    guardian.start()
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.MAXMSGSIZE, max(65536, limit))
    socket.connect(endpoint)
    token = ""
    try:
        task = cloudpickle.loads(definition) if definition is not None else None
        if isinstance(task, type):
            task = task()
        iterator = None
        socket.send_multipart([b'{"kind":"ready"}'])
        while True:
            message = socket.recv_multipart()
            header = json.loads(message[0])
            token = header["token"]
            if header["kind"] == "stop":
                break
            started, cpu_started = time.monotonic(), time.process_time()
            if kind == "source":
                if iterator is None:
                    iterator = iter(task(source_context))
                try:
                    value = next(iterator)
                except StopIteration:
                    socket.send_multipart([json.dumps({
                        "kind": "eof", "token": token, "started": started,
                        "finished": time.monotonic(), "cpu_seconds": time.process_time() - cpu_started,
                    }).encode()])
                    continue
            elif kind == "batch":
                value = load_group(message[1:])
            else:
                value = task(loads(message[1:]))
            frames = dumps(value, limit)
            socket.send_multipart(
                [json.dumps({
                    "kind": "result", "token": token, "started": started,
                    "finished": time.monotonic(), "cpu_seconds": time.process_time() - cpu_started,
                }).encode(), *frames])
    except BaseException:
        socket.send_multipart([json.dumps({
            "kind": "error", "token": token, "error": traceback.format_exc()[-8000:],
        }).encode()])
        # Keep the queued error alive until the local owner observes it and
        # shuts us down; LINGER=0 must not discard an initialization failure.
        while parent is None or parent.is_alive():
            if socket.poll(100):
                message = socket.recv_multipart()
                if json.loads(message[0])["kind"] == "stop":
                    break
    finally:
        stopped.set()
        socket.close()
        context.term()
