"""Acceptance check kept outside the model-writable repository copy."""

from __future__ import annotations

import os
from pathlib import Path
import sys


repo_path = Path(os.environ["HARNESS_REPO"]).resolve()
sys.path.insert(0, str(repo_path))

import model  # noqa: E402
import repository  # noqa: E402
import services  # noqa: E402


class FakeRepository(repository.AbstractRepository):
    def __init__(self):
        self._batches = set()

    def add(self, batch):
        self._batches.add(batch)

    def get(self, reference):
        return next(batch for batch in self._batches if batch.reference == reference)

    def list(self):
        return list(self._batches)


class FakeSession:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


repo = FakeRepository()
session = FakeSession()
services.add_batch("batch-1", "BLUE-PLINTH", 100, None, repo, session)
line = model.OrderLine("order-1", "BLUE-PLINTH", 10)
assert services.allocate(line, repo, session) == "batch-1"
assert repo.get("batch-1").available_quantity == 90
services.deallocate(line, repo, session)
assert repo.get("batch-1").available_quantity == 100
assert session.commits == 3
print("acceptance: service-layer add, allocate, and deallocate passed")

