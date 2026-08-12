"""The external store: an exact-match dictionary from a normalised key.

Deliberately the dumbest possible retrieval mechanism. Fuzzy matching, embedding
similarity and learned retrievers all introduce a second thing that can fail, and
the question under test is what happens when fact *values* leave the weights --
not how well a retriever works. An exact dictionary also makes the store's own
FLOP cost genuinely negligible, which the cost accounting relies on.
"""

from __future__ import annotations

import json
from pathlib import Path


def normalize(key: str) -> str:
    """Lowercase and collapse whitespace. Applied on both write and read."""
    return " ".join(key.lower().split())


class Organizer:
    def __init__(self) -> None:
        self._table: dict[str, str] = {}
        self.n_hits = 0
        self.n_misses = 0

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: str) -> bool:
        return normalize(key) in self._table

    def add(self, subject: str, relation: str, value: str) -> None:
        self._table[normalize(f"{subject}, {relation}")] = value

    def lookup(self, key: str) -> str | None:
        value = self._table.get(normalize(key))
        if value is None:
            self.n_misses += 1
        else:
            self.n_hits += 1
        return value

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for key, value in sorted(self._table.items()):
                fh.write(json.dumps({"key": key, "value": value}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Organizer":
        org = cls()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                org._table[normalize(row["key"])] = row["value"]
        return org

    def keys(self) -> list[str]:
        return sorted(self._table)
