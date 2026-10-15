"""Run the repository's public unittest suite from outside the target project."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


repo = Path(os.environ["HARNESS_REPO"]).resolve()
service = repo / "target-service"
tests = service / "tests"

sys.path.insert(0, str(service))
suite = unittest.defaultTestLoader.discover(str(tests), pattern="test*.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
