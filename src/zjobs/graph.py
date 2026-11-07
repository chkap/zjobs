from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping

if TYPE_CHECKING:
    from .runtime import Run


@dataclass(frozen=True)
class SourceContext:
    """Coordinates for explicitly partitioning a source across its replicas."""

    rank: int
    replicas: int
    machine: str = "local"
    local_rank: int = 0
    local_replicas: int = 1

    @property
    def global_rank(self) -> int:
        return self.rank

    @property
    def global_replicas(self) -> int:
        return self.replicas

    def shard(self, items: Iterable[Any]) -> Iterator[Any]:
        """Select this rank's items; shard identifiers before expensive loading."""
        return (item for index, item in enumerate(items) if index % self.replicas == self.rank)


@dataclass(frozen=True, eq=False)
class Node:
    name: str
    task: Callable[..., Any]
    replicas: int
    inputs: tuple[Node, ...]
    placement: tuple[tuple[str, int], ...] = ()
    policy: str | None = None
    rank_policies: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class ReplicaPlacement:
    stage: str
    machine: str
    rank: int
    replicas: int
    local_rank: int
    local_replicas: int
    policy: str = "local_first"


POLICIES = frozenset(("local_first", "balanced", "local_only"))


def validate_policy(policy: str) -> None:
    if not isinstance(policy, str) or policy not in POLICIES:
        raise ValueError(f"unknown locality policy {policy!r}; choose local_first, balanced or local_only")


class Graph:
    """Build a DAG in topological order; map stages return one output per input."""

    def __init__(self, *, policy: str = "local_first") -> None:
        validate_policy(policy)
        self._policy = policy
        self._nodes: list[Node] = []

    def source(
        self, name: str, task: Callable[..., Any], *, replicas: int = 1,
        placement: Mapping[str, int] | None = None,
        policy: str | None = None, rank_policies: Mapping[int, str] | None = None,
    ) -> Node:
        """Add a callable(context) -> iterable source, invoked once per replica."""
        return self._add(name, task, replicas, (), placement, policy, rank_policies)

    def map(
        self,
        name: str,
        task: Callable[..., Any],
        *,
        inputs: Iterable[Node],
        replicas: int = 1,
        placement: Mapping[str, int] | None = None,
        policy: str | None = None, rank_policies: Mapping[int, str] | None = None,
    ) -> Node:
        """Add callable(item) -> item; multiple inputs interleave without ordering."""
        parents = tuple(inputs)
        if not parents:
            raise ValueError("map stages require at least one input")
        if len(set(parents)) != len(parents):
            raise ValueError("duplicate input edges are not allowed")
        if any(parent not in self._nodes for parent in parents):
            raise ValueError("inputs must be existing nodes of this graph")
        return self._add(name, task, replicas, parents, placement, policy, rank_policies)

    def _add(
        self, name: str, task: Callable[..., Any], replicas: int, inputs: tuple[Node, ...],
        placement: Mapping[str, int] | None,
        policy: str | None, rank_policies: Mapping[int, str] | None,
    ) -> Node:
        if (
            not isinstance(name, str)
            or not name
            or any(node.name == name for node in self._nodes)
        ):
            raise ValueError("stage names must be nonempty, unique strings")
        if not callable(task):
            raise TypeError(f"stage {name!r} task must be callable")
        if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
            raise ValueError("replicas must be a positive integer")
        hosts: tuple[tuple[str, int], ...] = ()
        if placement is not None:
            if replicas != 1:
                raise ValueError("use placement counts instead of replicas")
            if not placement:
                raise ValueError("placement must contain at least one machine")
            for machine, count in placement.items():
                if not isinstance(machine, str) or not machine:
                    raise ValueError("machine identities must be nonempty strings")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("placement counts must be positive integers")
            hosts = tuple(sorted(placement.items()))
            replicas = sum(count for _, count in hosts)
        if policy is not None:
            validate_policy(policy)
        overrides = dict(rank_policies or {})
        for rank, value in overrides.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < replicas:
                raise ValueError("rank policy indices must be global stage ranks")
            validate_policy(value)
        node = Node(name, task, replicas, inputs, hosts, policy, tuple(sorted(overrides.items())))
        self._nodes.append(node)
        return node

    def run(
        self,
        *,
        capacity: int = 16,
        output_capacity: int = 16,
        worker_timeout: float = 30.0,
        consumer_timeout: float = 30.0,
        max_payload_bytes: int = 64 * 1024 * 1024,
        agents: Mapping[str, str] | None = None,
        agent_timeout: float = 3.0,
        retained_items: int = 128,
        retained_bytes: int = 1024 * 1024 * 1024,
        queue_bytes: int = 256 * 1024 * 1024,
        output_bytes: int = 256 * 1024 * 1024,
        remote_after: int = 8,
    ) -> Run:
        """Launch a snapshot; use the returned iterator as a context manager."""
        from .runtime import Run

        if not self._nodes:
            raise ValueError("a graph must have at least one source")
        for name, value in (
            ("capacity", capacity),
            ("output_capacity", output_capacity),
            ("max_payload_bytes", max_payload_bytes),
            ("retained_items", retained_items), ("retained_bytes", retained_bytes),
            ("queue_bytes", queue_bytes), ("output_bytes", output_bytes),
            ("remote_after", remote_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(retained_bytes, queue_bytes, output_bytes) < max_payload_bytes:
            raise ValueError("byte capacities must each fit max_payload_bytes")
        for name, value in (
            ("worker_timeout", worker_timeout), ("consumer_timeout", consumer_timeout),
            ("agent_timeout", agent_timeout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        return Run(
            tuple(self._nodes), capacity, output_capacity, worker_timeout,
            consumer_timeout, max_payload_bytes, agents, agent_timeout,
            self._policy, retained_items, retained_bytes, queue_bytes, output_bytes, remote_after,
        )
