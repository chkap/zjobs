"""Immutable logical payload snapshots; all frames count against byte credits."""

from __future__ import annotations

import importlib.util
import io
import pickle
import struct
import sys
from typing import Any, Sequence

import cloudpickle

Payload = tuple[bytes, ...]
MAX_FRAMES = 1024


def _restore_tensor(buffer, shape, dtype_name, requires_grad):
    import torch

    dtype = getattr(torch, dtype_name)
    if not buffer:
        value = torch.empty(shape, dtype=dtype)
    else:
        value = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
    return value.requires_grad_(requires_grad)


class _Pickler(cloudpickle.CloudPickler):
    def reducer_override(self, value):
        torch = sys.modules.get("torch")
        if (
            torch is not None and type(value) is getattr(torch, "Tensor", None)
            and value.device.type == "cpu" and value.layout == torch.strided
            and value.is_contiguous() and not value.is_quantized
            and not value.is_conj() and not value.is_neg()
            and not value.__dict__ and not value._backward_hooks
            and importlib.util.find_spec("numpy") is not None
        ):
            raw = value.detach().reshape(-1).view(torch.uint8).numpy()
            return (_restore_tensor, (
                pickle.PickleBuffer(raw), tuple(value.shape),
                str(value.dtype).removeprefix("torch."), value.requires_grad,
            ))
        return super().reducer_override(value)


def size(frames: Sequence[bytes]) -> int:
    return sum(map(len, frames))


def dumps(value: Any, limit: int) -> Payload:
    if type(value) is bytes:
        frames = (b"B", value)
    else:
        buffers = []
        buffered_bytes = 0

        def snapshot(buffer) -> None:
            nonlocal buffered_bytes
            if len(buffers) >= MAX_FRAMES - 2:
                raise ValueError(f"payload exceeds {MAX_FRAMES} frames")
            raw = buffer.raw()
            buffered_bytes += raw.nbytes
            if buffered_bytes > limit:
                raise ValueError(f"serialized payload exceeds {limit} bytes in buffers")
            # Snapshot writable user buffers; retaining a view would let later
            # user mutation change fan-out or replay inputs after publication.
            buffers.append(bytes(raw))

        stream = io.BytesIO()
        _Pickler(stream, protocol=5, buffer_callback=snapshot).dump(value)
        envelope = stream.getvalue()
        frames = (b"P", envelope, *buffers)
    total = size(frames)
    if total > limit:
        raise ValueError(f"serialized payload is {total} bytes; limit is {limit}")
    return frames


def loads(frames: Sequence[bytes]) -> Any:
    if len(frames) < 2 or len(frames) > MAX_FRAMES:
        raise ValueError("invalid logical payload frame count")
    if frames[0] == b"B" and len(frames) == 2:
        return frames[1]
    if frames[0] != b"P":
        raise ValueError("invalid logical payload format")
    # Writable pickle buffers get isolated mutable storage. Pickle's
    # READONLY_BUFFER opcode preserves original read-only array semantics.
    return cloudpickle.loads(frames[1], buffers=[bytearray(buffer) for buffer in frames[2:]])


def group_frames(members: Sequence[Payload]) -> list[bytes]:
    header = struct.pack(f"!{len(members)}I", *(len(member) for member in members))
    return [header, *(frame for member in members for frame in member)]


def load_group(frames: Sequence[bytes]) -> list[Any]:
    if not frames or len(frames[0]) % 4:
        raise ValueError("invalid batch frame directory")
    counts = struct.unpack(f"!{len(frames[0]) // 4}I", frames[0])
    values = []
    offset = 1
    for count in counts:
        if count < 2 or count > MAX_FRAMES or offset + count > len(frames):
            raise ValueError("invalid batch member frame count")
        values.append(loads(frames[offset:offset + count]))
        offset += count
    if offset != len(frames):
        raise ValueError("unexpected trailing batch frames")
    return values
