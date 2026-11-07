from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Literal, Mapping, Sequence


@dataclass(frozen=True)
class Locality:
    mode: Literal["local_first", "balanced", "local_only"] = "local_first"
    max_local_streak: int = 8

    def __post_init__(self):
        if self.mode not in ("local_first", "balanced", "local_only"):
            raise ValueError(f"unknown locality: {self.mode}")
        if type(self.max_local_streak) is not int or self.max_local_streak < 1:
            raise ValueError("max_local_streak must be a positive integer")


@dataclass(frozen=True)
class Limits:
    max_payload_bytes: int = 8 * 1024 * 1024
    retained_items: int = 32
    retained_bytes: int = 64 * 1024 * 1024
    claim_items: int = 64
    claim_bytes: int = 64 * 1024 * 1024
    output_items: int = 8
    output_bytes: int = 64 * 1024 * 1024
    task_timeout: float = 30
    consumer_timeout: float = 30
    shutdown_timeout: float = 10
    max_retries: int = 0

    def __post_init__(self):
        for name in (
            "max_payload_bytes", "retained_items", "retained_bytes",
            "claim_items", "claim_bytes", "output_items", "output_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("task_timeout", "consumer_timeout", "shutdown_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if self.retained_bytes < self.max_payload_bytes:
            raise ValueError("retained_bytes must reserve at least max_payload_bytes")
        if self.output_bytes < self.max_payload_bytes:
            raise ValueError("output_bytes must admit max_payload_bytes")


@dataclass(frozen=True)
class PeerConfig:
    identity: str = "local"
    run_id: str = "standalone"
    listen: str | None = None
    peers: Mapping[str, str] = field(default_factory=dict)
    join_timeout: float = 15
    heartbeat_timeout: float = 5
    lease_timeout: float = 5

    def __post_init__(self):
        if not self.identity or not self.run_id:
            raise ValueError("identity and run_id must be nonempty")
        members = dict(self.peers)
        if members and (not self.listen or members.get(self.identity) != self.listen):
            raise ValueError("peers must include identity with its listen endpoint")
        if len(set(members.values())) != len(members):
            raise ValueError("peer endpoints must be distinct")
        if any(not k or not v.startswith("tcp://") for k, v in members.items()):
            raise ValueError("peers require nonempty identities and tcp:// endpoints")
        for name in ("join_timeout", "heartbeat_timeout", "lease_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "peers", members)


@dataclass(frozen=True)
class SourceContext:
    peer: str
    rank: int
    replicas: int
    members: tuple[str, ...]

    @property
    def peer_index(self) -> int:
        return self.members.index(self.peer)


@dataclass(eq=False)
class Stage:
    name: str
    kind: str
    task: Callable[..., Any] | type | None
    inputs: tuple[Stage, ...]
    replicas: int
    locality: Locality | None
    rank_locality: dict[int, Locality]
    size: int = 1
    max_bytes: int = 8 * 1024 * 1024


class Graph:
    def __init__(self, *, locality: Locality = Locality()):
        self.locality = locality
        self.stages: list[Stage] = []

    def _add(self, name, kind, task, inputs, replicas, locality, rank_locality,
             size=1, max_bytes=8 * 1024 * 1024):
        if (not isinstance(name, str) or not name or name == "@consumer" or
                any(s.name == name for s in self.stages)):
            raise ValueError("stage names must be unique nonempty strings, excluding @consumer")
        if type(replicas) is not int or replicas < 1:
            raise ValueError("each local stage must have positive replicas")
        parents = tuple(inputs)
        if len(set(parents)) != len(parents) or any(p not in self.stages for p in parents):
            raise ValueError("inputs must be distinct stages from this graph")
        if kind != "source" and not parents:
            raise ValueError("downstream stages require inputs")
        overrides = dict(rank_locality or {})
        if any(type(r) is not int or not 0 <= r < replicas for r in overrides):
            raise ValueError("rank locality override outside local replica range")
        if any(not isinstance(p, Locality) for p in [self.locality, locality or self.locality,
                                                     *overrides.values()]):
            raise TypeError("locality policies must be Locality instances")
        if kind != "batch" and not callable(task):
            raise TypeError("task must be callable")
        if type(size) is not int or size < 1 or type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("batch size and max_bytes must be positive integers")
        stage = Stage(name, kind, task, parents, replicas, locality, overrides, size, max_bytes)
        self.stages.append(stage)
        return stage

    def source(self, name, task, *, replicas=1, locality=None, rank_locality=None) -> Stage:
        return self._add(name, "source", task, (), replicas, locality, rank_locality)

    def map(self, name, task, *, inputs: Sequence[Stage], replicas=1,
            locality=None, rank_locality=None) -> Stage:
        return self._add(name, "map", task, inputs, replicas, locality, rank_locality)

    def batch(self, name, *, inputs: Sequence[Stage], size: int,
              max_bytes=8 * 1024 * 1024, replicas=1, locality=None,
              rank_locality=None) -> Stage:
        return self._add(name, "batch", None, inputs, replicas, locality, rank_locality,
                         size, max_bytes)

    def policy(self, stage: Stage, rank: int) -> Locality:
        return stage.rank_locality.get(rank, stage.locality or self.locality)

    def fingerprint(self) -> str:
        # Callable equivalence and local replica counts are deployment concerns,
        # not bytecode equality: cloudpickle captures machine-specific state.
        shape = [(s.name, s.kind, [p.name for p in s.inputs], s.size, s.max_bytes)
                 for s in self.stages]
        return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()

    def run(self, *, peer: PeerConfig | None = None, limits: Limits | None = None):
        from .runtime import Run
        if not self.stages:
            raise ValueError("cannot run an empty graph")
        return Run(self, peer or PeerConfig(), limits or Limits())
