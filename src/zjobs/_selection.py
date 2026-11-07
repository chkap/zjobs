from __future__ import annotations

from .graph import Locality


class Selection:
    """One receiver-rank's local streak and rotating owner opportunities."""

    def __init__(self):
        self.streak = 0
        self.cursor = 0
        self.remote_cursor = 0

    def choose(self, owners: list[str], local: str, policy: Locality) -> str | None:
        if policy.mode == "local_only":
            return local if local in owners else None
        remote = sorted(o for o in owners if o != local)
        if policy.mode == "local_first":
            if local in owners and (not remote or self.streak < policy.max_local_streak):
                return local
            choices = remote
        else:
            choices = sorted(owners)
        if not choices:
            return None
        cursor = self.remote_cursor if policy.mode == "local_first" else self.cursor
        return choices[cursor % len(choices)]

    def admitted(self, owner: str, local: str):
        if owner == local:
            self.streak += 1
        else:
            self.streak = 0
            self.remote_cursor += 1
        self.cursor += 1

    def rejected(self):
        self.cursor += 1
        self.remote_cursor += 1
