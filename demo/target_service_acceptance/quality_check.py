"""Small dependency-free syntax and regression-test quality check."""

from __future__ import annotations

import ast
import builtins
import os
from pathlib import Path
import symtable


repo = Path(os.environ["HARNESS_REPO"]).resolve()
pricing = repo / "target-service" / "orders" / "pricing.py"
student_test = repo / "target-service" / "tests" / "test_student.py"

implicit_globals = {
    "__annotations__",
    "__builtins__",
    "__cached__",
    "__doc__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}


def undefined_globals(source: str, filename: str) -> list[str]:
    table = symtable.symtable(source, filename, "exec")
    module_bindings = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    available = module_bindings | set(dir(builtins)) | implicit_globals
    unresolved = set()

    def inspect(scope: symtable.SymbolTable) -> None:
        for symbol in scope.get_symbols():
            if not symbol.is_referenced():
                continue
            name = symbol.get_name()
            if scope.get_type() == "module":
                if name not in available:
                    unresolved.add(name)
            elif symbol.is_global() and name not in available:
                unresolved.add(name)
        for child in scope.get_children():
            inspect(child)

    inspect(table)
    return sorted(unresolved)


trees = {}
for path in (pricing, student_test):
    if not path.is_file():
        raise AssertionError(f"required file is missing: {path.relative_to(repo)}")
    source = path.read_text(encoding="utf-8")
    trees[path] = ast.parse(source, filename=str(path))
    unresolved = undefined_globals(source, str(path))
    if unresolved:
        if path == student_test and "lines" in unresolved:
            raise AssertionError(
                "tests/test_student.py must define its own small 'lines' helper; "
                "import only quote from orders.pricing"
            )
        raise AssertionError(
            f"{path.relative_to(repo)} uses undefined global name(s): "
            + ", ".join(unresolved)
        )

pricing_tree = trees[pricing]
pricing_tests = [
    node
    for node in ast.walk(pricing_tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name.startswith("test")
]
if pricing_tests:
    raise AssertionError("orders/pricing.py must not contain regression tests")
if any(isinstance(node, ast.Assert) for node in ast.walk(pricing_tree)):
    raise AssertionError("orders/pricing.py must not contain test assertions")
if any(
    (isinstance(node, ast.Import) and any(alias.name == "unittest" for alias in node.names))
    or (isinstance(node, ast.ImportFrom) and node.module == "unittest")
    for node in ast.walk(pricing_tree)
):
    raise AssertionError("orders/pricing.py must not import unittest")
if any(
    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "validate_order"
    for node in pricing_tree.body
):
    raise AssertionError(
        "orders/pricing.py must use the existing orders.validation.validate_order"
    )
if not any(
    isinstance(node, ast.ImportFrom)
    and node.module == "validation"
    and node.level == 1
    and any(alias.name == "validate_order" for alias in node.names)
    for node in pricing_tree.body
):
    raise AssertionError(
        "orders/pricing.py must preserve 'from .validation import validate_order'"
    )

tree = trees[student_test]
tests = [
    node
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
]
if len(tests) != 1:
    names = ", ".join(node.name for node in tests) or "(none)"
    raise AssertionError(
        "tests/test_student.py must contain exactly one regression test; "
        f"found {len(tests)}: {names}. Keep one useful test and remove all others."
    )

test = tests[0]
if not any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "quote"
    for node in ast.walk(test)
):
    raise AssertionError("the regression test must call quote")

defined_functions = {
    node.name
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
if "lines" not in defined_functions and any(
    isinstance(node, ast.Name)
    and isinstance(node.ctx, ast.Load)
    and node.id == "lines"
    for node in ast.walk(tree)
):
    raise AssertionError("tests/test_student.py uses undefined helper 'lines'")

print("target-service quality checks passed")
