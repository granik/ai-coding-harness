"""Small external static check for obvious generated-code defects."""

from __future__ import annotations

import ast
import os
from pathlib import Path


repo = Path(os.environ["HARNESS_REPO"]).resolve()
problems: list[str] = []


def inspect_body(body: list[ast.stmt], path: Path) -> None:
    terminated = False
    for statement in body:
        if terminated:
            problems.append(
                f"{path.name}:{statement.lineno}: unreachable statement after terminator"
            )
            break
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(statement, field, None)
            if isinstance(nested, list):
                inspect_body(nested, path)
        handlers = getattr(statement, "handlers", None)
        if isinstance(handlers, list):
            for handler in handlers:
                inspect_body(handler.body, path)
        terminated = isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue))


for path in sorted(repo.glob("*.py")):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        problems.append(f"{path.name}: cannot parse: {error}")
        continue
    inspect_body(tree.body, path)

if problems:
    raise SystemExit("\n".join(problems))
print("quality: Python files parse and contain no obvious unreachable statements")

