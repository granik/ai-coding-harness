"""Data-driven tool registry and permission-enforcing dispatcher."""

from __future__ import annotations

import ast
import builtins
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import symtable
from typing import Callable

from .config import CheckSpec, LimitsConfig
from .executor import CheckRunner
from .workspace import DisposableWorkspace, WorkspaceError


@dataclass(frozen=True)
class ToolSpec:
    arguments: tuple[str, ...]
    description: str
    mutating: bool = False


TOOL_SPECS = {
    "list_files": ToolSpec((), "List files in the disposable repository."),
    "read_file": ToolSpec(("path",), "Read one UTF-8 repository file."),
    "search_files": ToolSpec(
        ("query",), "Search text files for a literal string."
    ),
    "write_file": ToolSpec(
        ("path", "content"), "Create one new UTF-8 file.", mutating=True
    ),
    "replace_file": ToolSpec(
        ("path", "content"),
        "Replace the complete content of one existing UTF-8 file.",
        mutating=True,
    ),
    "edit_file": ToolSpec(
        ("path", "old", "new"),
        "Replace one exact unique non-empty string in an existing file.",
        mutating=True,
    ),
    "run_check": ToolSpec(
        ("name",), "Run one configured check that is available to the model."
    ),
}


_IMPLICIT_MODULE_GLOBALS = {
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


def _is_test_path(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    return (
        "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _undefined_python_globals(content: str, filename: str) -> set[str]:
    """Find unresolved global names while allowing pre-existing dynamic globals."""
    table = symtable.symtable(content, filename, "exec")
    module_bindings = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    available = module_bindings | set(dir(builtins)) | _IMPLICIT_MODULE_GLOBALS
    unresolved: set[str] = set()

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
    return unresolved


def _imported_bindings(tree: ast.Module) -> set[str]:
    bindings: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            bindings.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            bindings.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name != "*"
            )
    return bindings


def _defined_module_bindings(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _contains_embedded_main_test(tree: ast.Module) -> bool:
    """Recognize a script-style assertion block added to a non-test module."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        comparison = node.test
        if not (
            isinstance(comparison, ast.Compare)
            and isinstance(comparison.left, ast.Name)
            and comparison.left.id == "__name__"
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq)
            and len(comparison.comparators) == 1
            and isinstance(comparison.comparators[0], ast.Constant)
            and comparison.comparators[0].value == "__main__"
        ):
            continue
        if any(isinstance(candidate, ast.Assert) for candidate in ast.walk(node)):
            return True
    return False


def _contains_test_code(tree: ast.Module) -> bool:
    if _contains_embedded_main_test(tree):
        return True
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test")
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name == "unittest" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "unittest":
            return True
    return False


def _test_function_names(tree: ast.Module) -> list[str]:
    """Return test functions and methods defined by one Python file."""
    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )


@dataclass
class DispatchStats:
    actions: int = 0
    denied: int = 0
    errors: int = 0
    retries: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Dispatcher:
    """Validate, authorize, and execute one model-proposed action."""

    def __init__(
        self,
        *,
        workspace: DisposableWorkspace,
        runner: CheckRunner,
        checks: tuple[CheckSpec, ...],
        limits: LimitsConfig,
    ) -> None:
        self.workspace = workspace
        self.runner = runner
        self.checks = {check.name: check for check in checks}
        self.limits = limits
        self.stats = DispatchStats()
        self._last_failed_signature: str | None = None
        self._consecutive_failures = 0
        self._read_paths: set[str] = set()
        self._search_queries: set[str] = set()
        self._file_evidence: dict[str, dict] = {}
        self._frozen_paths: set[str] = set()
        self._inspection_actions = 0
        self._handlers: dict[str, Callable[..., dict]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_files": self._search_files,
            "write_file": self._write_file,
            "replace_file": self._replace_file,
            "edit_file": self._edit_file,
            "run_check": self._run_check,
        }

    def _tool_variants(
        self,
        *,
        only_tools: set[str] | None = None,
        unread_only: bool = False,
    ) -> list[dict]:
        variants = []
        for name, spec in TOOL_SPECS.items():
            if only_tools is not None and name not in only_tools:
                continue
            properties = {
                argument: {"type": "string"} for argument in spec.arguments
            }
            if name == "read_file":
                existing_files = self.workspace.list_files()
                if unread_only:
                    existing_files = [
                        path for path in existing_files if path not in self._read_paths
                    ]
                if not existing_files:
                    continue
                properties["path"] = {
                    "type": "string",
                    "enum": existing_files,
                    "description": "An existing repository file from this exact list.",
                }
            if name in {"edit_file", "replace_file"}:
                writable_files = []
                for path in self.workspace.list_files():
                    target = self.workspace.resolve(path, must_exist=True)
                    if (
                        self.workspace.is_writable(target)
                        and path not in self._frozen_paths
                    ):
                        writable_files.append(path)
                if not writable_files:
                    continue
                properties["path"] = {
                    "type": "string",
                    "enum": writable_files,
                    "description": "An existing writable file from this exact list.",
                }
            if name == "write_file":
                exact_missing_paths = []
                has_pattern = False
                for pattern in self.workspace.writable:
                    if pattern in {"*", "**"} or any(
                        marker in pattern for marker in ("*", "?", "[")
                    ):
                        has_pattern = True
                        continue
                    try:
                        target = self.workspace.resolve(pattern)
                    except WorkspaceError:
                        continue
                    if not target.exists():
                        exact_missing_paths.append(pattern)
                if exact_missing_paths and not has_pattern:
                    properties["path"] = {
                        "type": "string",
                        "enum": sorted(exact_missing_paths),
                        "description": "A missing writable file from this exact list.",
                    }
                elif not exact_missing_paths and not has_pattern:
                    continue
            if name == "edit_file":
                properties["old"] = {
                    "type": "string",
                    "minLength": 1,
                    "description": "Non-empty exact text occurring once in the file.",
                }
            if name == "run_check":
                allowed = sorted(
                    check.name for check in self.checks.values() if check.model_allowed
                )
                if not allowed:
                    continue
                properties["name"] = {"type": "string", "enum": allowed}
            variants.append(
                {
                    "type": "object",
                    "description": spec.description,
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": [name],
                            "description": spec.description,
                        },
                        "args": {
                            "type": "object",
                            "properties": properties,
                            "required": list(spec.arguments),
                            "additionalProperties": False,
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                }
            )
        return variants

    def schema(
        self,
        *,
        only_tools: set[str] | None = None,
        unread_only: bool = False,
    ) -> dict:
        variants = self._tool_variants(
            only_tools=only_tools, unread_only=unread_only
        )
        variants.append(
            {
                "type": "object",
                "properties": {"final": {"type": "string"}},
                "required": ["final"],
                "additionalProperties": False,
            }
        )
        return {"oneOf": variants}

    def selection_schema(
        self,
        *,
        only_tools: set[str] | None = None,
        unread_only: bool = False,
        detailed_plan: bool = False,
        plan_cases: tuple[str, ...] = (),
        allow_final: bool = True,
    ) -> dict:
        """Small first-stage grammar that selects a tool or finishes."""
        variants = self._tool_variants(
            only_tools=only_tools, unread_only=unread_only
        )
        choices = []
        for variant in variants:
            name = variant["properties"]["tool"]["enum"][0]
            description = TOOL_SPECS[name].description
            path_schema = (
                variant.get("properties", {})
                .get("args", {})
                .get("properties", {})
                .get("path")
            )
            if isinstance(path_schema, dict) and path_schema.get("enum"):
                description += (
                    " Available paths for this tool: "
                    + json.dumps(path_schema["enum"], ensure_ascii=False)
                    + "."
                )
            properties = {
                "tool": {
                    "type": "string",
                    "enum": [name],
                    "description": description,
                }
            }
            if detailed_plan and plan_cases:
                case_properties = {
                    f"case_{index}": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 600,
                        "description": (
                            "State the required behavior and concrete repair for this "
                            f"reported failure: {case}"
                        ),
                    }
                    for index, case in enumerate(plan_cases, start=1)
                }
                properties.update(
                    {
                        "failure_plan": {
                            "type": "object",
                            "properties": case_properties,
                            "required": list(case_properties),
                            "additionalProperties": False,
                        },
                        "implementation_plan": {
                            "type": "string",
                            "minLength": 100,
                            "maxLength": 3_000,
                            "description": (
                                "Combine every case above into one coherent algorithm. "
                                "Specify conditions, operation order, and exact data "
                                "handling needed by the implementation."
                            ),
                        },
                    }
                )
                required = ["tool", "failure_plan", "implementation_plan"]
            else:
                properties["intent"] = {
                    "type": "string",
                    "minLength": 80 if detailed_plan else 1,
                    "maxLength": 2_000 if detailed_plan else 400,
                    "description": (
                        (
                            "A concrete repair plan derived from every retained "
                            "requirement and failure. Identify the required conditions "
                            "and operation order, and describe one coherent "
                            "implementation. Do not give a generic instruction or "
                            "repeat current code."
                        )
                        if detailed_plan
                        else (
                            "A concise description of the concrete next change or "
                            "verification this tool will perform."
                        )
                    ),
                }
                required = ["tool", "intent"]
            choices.append(
                {
                    "type": "object",
                    "description": description,
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                }
            )
        if allow_final:
            choices.append(
                {
                    "type": "object",
                    "properties": {"final": {"type": "string"}},
                    "required": ["final"],
                    "additionalProperties": False,
                }
            )
        return {"oneOf": choices}

    def tool_schema(
        self,
        name: str,
        *,
        unread_only: bool = False,
        required_path: str | None = None,
        content_max_length: int | None = None,
        content_description: str | None = None,
    ) -> dict:
        """Strict second-stage grammar for one previously selected tool."""
        variants = self._tool_variants(
            only_tools={name}, unread_only=unread_only
        )
        if not variants:
            raise ValueError(f"Selected tool is no longer available: {name}")
        variant = variants[0]
        if required_path is not None:
            path_schema = variant["properties"]["args"]["properties"].get("path")
            available = path_schema.get("enum", []) if isinstance(path_schema, dict) else []
            if required_path not in available:
                raise ValueError(
                    f"Selected tool {name!r} cannot operate on {required_path!r}"
                )
            path_schema["enum"] = [required_path]
        content_schema = variant["properties"]["args"]["properties"].get("content")
        if isinstance(content_schema, dict):
            if content_description is not None:
                content_schema["description"] = content_description
            if content_max_length is not None:
                content_schema["maxLength"] = content_max_length
                content_schema.setdefault(
                    "description",
                    (
                        "Complete focused file content. Keep it within "
                        f"{content_max_length} characters."
                    ),
                )
        return variant

    def align_tool_to_explicit_path(
        self,
        selected_tool: str,
        intent: str,
        *,
        only_tools: set[str] | None = None,
        unread_only: bool = False,
    ) -> tuple[str, str | None]:
        """Align a selection when its intent names one unambiguous tool path."""
        tools_by_path: dict[str, set[str]] = {}
        paths_by_tool: dict[str, set[str]] = {}
        for variant in self._tool_variants(
            only_tools=only_tools, unread_only=unread_only
        ):
            name = variant["properties"]["tool"]["enum"][0]
            path_schema = variant["properties"]["args"]["properties"].get("path")
            if not isinstance(path_schema, dict):
                continue
            for path in path_schema.get("enum", []):
                if not isinstance(path, str):
                    continue
                paths_by_tool.setdefault(name, set()).add(path)
                parts = path.split("/")
                candidates = [path]
                candidates.extend(
                    "/".join(parts[index:]) for index in range(1, len(parts))
                )
                if any(candidate in intent for candidate in candidates):
                    tools_by_path.setdefault(path, set()).add(name)
        if len(tools_by_path) == 1:
            path = next(iter(tools_by_path))
            tools = tools_by_path[path]
            if selected_tool in tools:
                return selected_tool, path
            if len(tools) == 1:
                aligned = next(iter(tools))
                return aligned, path
        selected_paths = paths_by_tool.get(selected_tool, set())
        if len(selected_paths) == 1:
            # A single path is a stronger boundary than a weak model's prose
            # intent. Bind it explicitly so argument generation cannot place
            # content meant for an unavailable path in the remaining file.
            return selected_tool, next(iter(selected_paths))
        return selected_tool, None

    def frozen_path_in_text(self, text: str) -> str | None:
        """Return the one frozen path explicitly named in model text, if any."""
        matches = []
        for path in self._frozen_paths:
            parts = path.split("/")
            candidates = [path]
            candidates.extend(
                "/".join(parts[index:]) for index in range(1, len(parts))
            )
            if any(candidate in text for candidate in candidates):
                matches.append(path)
        return matches[0] if len(matches) == 1 else None

    def _recovery_cross_file_copy(
        self, path: str, content: str
    ) -> tuple[str, int] | None:
        """Detect a large copied block from another retained file during recovery."""
        minimum = 400
        for other_path, item in self._file_evidence.items():
            if other_path == path or item.get("truncated"):
                continue
            other_content = item.get("content")
            if not isinstance(other_content, str) or len(other_content) < minimum:
                continue
            match = SequenceMatcher(
                None, content, other_content, autojunk=False
            ).find_longest_match()
            if match.size >= minimum and match.size * 2 >= len(other_content):
                return other_path, match.size
        return None

    @staticmethod
    def replacement_schema(
        path: str,
        *,
        content_max_length: int | None = None,
        content_description: str | None = None,
    ) -> dict:
        """Strict recovery grammar for one known existing file."""
        content_schema = {"type": "string"}
        if content_description is not None:
            content_schema["description"] = content_description
        if content_max_length is not None:
            content_schema["maxLength"] = content_max_length
            content_schema.setdefault(
                "description",
                (
                    "Complete focused file content. Keep it within "
                    f"{content_max_length} characters."
                ),
            )
        return {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": ["replace_file"],
                    "description": TOOL_SPECS["replace_file"].description,
                },
                "args": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "enum": [path]},
                        "content": content_schema,
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        }

    def tool_help(self) -> dict[str, dict]:
        return {
            name: {
                "args": list(spec.arguments),
                "description": spec.description,
            }
            for name, spec in TOOL_SPECS.items()
            if name != "run_check"
            or any(check.model_allowed for check in self.checks.values())
        }

    def read_only_progress(self) -> dict[str, object]:
        return {
            "files_read": sorted(self._read_paths),
            "searches_completed": sorted(self._search_queries),
            "inspection_actions": self._inspection_actions,
            "frozen_paths": self.frozen_paths(),
        }

    def freeze_path(self, path: str) -> str:
        """Remove an existing path from mutation schemas for this run."""
        target = self.workspace.resolve(path, must_exist=True)
        relative = self.workspace.relative(target)
        self._frozen_paths.add(relative)
        return relative

    def unfreeze_path(self, path: str) -> str:
        """Restore a previously frozen path to mutation schemas."""
        target = self.workspace.resolve(path, must_exist=True)
        relative = self.workspace.relative(target)
        self._frozen_paths.discard(relative)
        return relative

    def frozen_paths(self) -> list[str]:
        return sorted(self._frozen_paths)

    def missing_writable_paths(self) -> list[str]:
        """Return exact configured writable paths that do not exist yet."""
        missing = []
        for pattern in self.workspace.writable:
            if pattern in {"*", "**"} or any(
                marker in pattern for marker in ("*", "?", "[")
            ):
                continue
            try:
                target = self.workspace.resolve(pattern, must_exist=False)
            except WorkspaceError:
                continue
            if not target.exists():
                missing.append(pattern)
        return sorted(missing)

    @property
    def inspection_actions(self) -> int:
        return self._inspection_actions

    def file_evidence(self) -> dict[str, dict]:
        remaining = self.limits.max_tool_output_bytes
        evidence: dict[str, dict] = {}
        for path, item in self._file_evidence.items():
            if remaining <= 0:
                break
            content = item["content"]
            encoded = content.encode("utf-8")
            truncated = item["truncated"] or len(encoded) > remaining
            if len(encoded) > remaining:
                marker = "\n[persistent evidence shortened]\n"
                marker_bytes = marker.encode("utf-8")
                content = (
                    encoded[: max(0, remaining - len(marker_bytes))].decode(
                        "utf-8", errors="ignore"
                    )
                    + marker
                )
            evidence[path] = {"content": content, "truncated": truncated}
            remaining -= len(content.encode("utf-8"))
        return evidence

    def _remember_file(self, path: str, content: str, truncated: bool) -> None:
        self._read_paths.add(path)
        self._file_evidence[path] = {
            "content": content,
            "truncated": truncated,
        }

    def _result(self, status: str, output, *, fatal: bool = False) -> dict:
        if status == "denied":
            self.stats.denied += 1
        elif status == "error":
            self.stats.errors += 1
        return {"status": status, "output": output, "fatal": fatal}

    def dispatch(
        self,
        action,
        *,
        allowed_tools: set[str] | None = None,
        required_path: str | None = None,
        required_test_count: int | None = None,
    ) -> dict:
        self.stats.actions += 1
        if not isinstance(action, dict) or set(action) != {"tool", "args"}:
            return self._result(
                "error", "Action must contain exactly 'tool' and 'args'."
            )
        tool = action.get("tool")
        args = action.get("args")
        if not isinstance(tool, str) or tool not in TOOL_SPECS:
            return self._result("error", f"Unknown tool: {tool!r}")
        expected = TOOL_SPECS[tool].arguments
        if not isinstance(args, dict) or set(args) != set(expected):
            return self._result(
                "error",
                f"Invalid arguments for {tool}; expected {list(expected)}.",
            )
        if any(not isinstance(value, str) for value in args.values()):
            return self._result("error", "Every tool argument must be a string.")

        signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
        if (
            signature == self._last_failed_signature
            and self._consecutive_failures > 0
        ):
            self.stats.retries += 1
            if self._consecutive_failures > self.limits.max_retries:
                return self._result(
                    "denied",
                    "Repeated failed action exceeded the configured retry limit.",
                    fatal=True,
                )
        else:
            self._last_failed_signature = None
            self._consecutive_failures = 0

        def finish(result: dict) -> dict:
            if result.get("status") == "ok":
                self._last_failed_signature = None
                self._consecutive_failures = 0
            elif signature == self._last_failed_signature:
                self._consecutive_failures += 1
            else:
                self._last_failed_signature = signature
                self._consecutive_failures = 1
            return result

        if allowed_tools is not None and tool not in allowed_tools:
            return finish(
                self._result(
                    "denied",
                    "This phase permits only: "
                    + ", ".join(sorted(allowed_tools))
                    + ".",
                )
            )
        if required_path is not None and args.get("path") != required_path:
            return finish(
                self._result(
                    "denied",
                    f"This recovery phase permits only path {required_path!r}.",
                )
            )
        if tool in {"write_file", "replace_file", "edit_file"}:
            path = args.get("path")
            if isinstance(path, str):
                try:
                    try:
                        target = self.workspace.resolve(path, must_exist=True)
                    except WorkspaceError:
                        target = self.workspace.resolve(path, must_exist=False)
                    relative = self.workspace.relative(target)
                except WorkspaceError:
                    relative = path
                if relative in self._frozen_paths:
                    return finish(
                        self._result(
                            "denied",
                            (
                                f"Path {relative!r} is frozen after passing its "
                                "quality check. Repair implementation code for "
                                "assertion failures; this path is unlocked only "
                                "when a later check reports a test-code error."
                            ),
                        )
                    )
        if (
            required_test_count is not None
            and tool in {"write_file", "replace_file", "edit_file"}
            and isinstance(args.get("path"), str)
        ):
            try:
                target = self.workspace.resolve(
                    args["path"], must_exist=tool != "write_file"
                )
                if _is_test_path(target) and target.suffix == ".py":
                    current = (
                        target.read_text(encoding="utf-8")
                        if target.exists()
                        else ""
                    )
                    candidate = None
                    if tool in {"write_file", "replace_file"}:
                        candidate = args["content"]
                    elif current.count(args["old"]) == 1:
                        candidate = current.replace(args["old"], args["new"], 1)
                    if candidate is not None:
                        tree = ast.parse(candidate, filename=str(target))
                        problems = []
                        names = _test_function_names(tree)
                        if len(names) != required_test_count:
                            problems.append(
                                f"expected exactly {required_test_count} test function "
                                f"or method; found {len(names)}"
                                + (f": {', '.join(names)}" if names else "")
                            )
                        if tool == "write_file":
                            unresolved = sorted(
                                _undefined_python_globals(candidate, str(target))
                            )
                            if unresolved:
                                problems.append(
                                    "unresolved global name(s): "
                                    + ", ".join(repr(name) for name in unresolved)
                                )
                        if problems:
                            output = {
                                "message": (
                                    "Focused test file rejected: "
                                    + "; ".join(problems)
                                    + ". Define every helper used by the one requested "
                                    "test and include no additional tests."
                                ),
                                "path": self.workspace.relative(target),
                            }
                            if target.exists():
                                visible, truncated = self._bounded_text(current)
                                output.update(
                                    {
                                        "current_content": visible,
                                        "truncated": truncated,
                                    }
                                )
                            return finish(self._result("error", output))
            except SyntaxError:
                # The ordinary Python mutation validation returns the precise syntax
                # error and, for existing files, the normal recovery context.
                pass
            except WorkspaceError as error:
                return finish(self._result("denied", str(error)))
        if required_path is not None and tool == "replace_file":
            copied = self._recovery_cross_file_copy(
                required_path, args["content"]
            )
            if copied is not None:
                other_path, copied_characters = copied
                target = self.workspace.resolve(required_path, must_exist=True)
                current, truncated = self._bounded_text(
                    target.read_text(encoding="utf-8")
                )
                return finish(
                    self._result(
                        "error",
                        {
                            "message": (
                                "Recovery replacement appears to copy "
                                f"{copied_characters} characters from another file "
                                f"({other_path}); keep each file's content separate."
                            ),
                            "path": required_path,
                            "current_content": current,
                            "truncated": truncated,
                            "instruction": (
                                "Replace only this file with code appropriate to its "
                                "path and the failed edit. Do not include content for "
                                "another file."
                            ),
                        },
                    )
                )
        try:
            result = self._handlers[tool](**args)
            return finish(result)
        except WorkspaceError as error:
            return finish(self._result("denied", str(error)))
        except (OSError, UnicodeError, ValueError) as error:
            return finish(self._result("error", str(error)))

    def _bounded_text(self, text: str) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        limit = self.limits.max_tool_output_bytes
        if len(encoded) <= limit:
            return text, False
        marker = "\n[tool output shortened]\n"
        head = encoded[: max(0, limit - len(marker.encode()))].decode(
            "utf-8", errors="ignore"
        )
        return head + marker, True

    def _list_files(self) -> dict:
        files = self.workspace.list_files()
        output, truncated = self._bounded_text(json.dumps(files, ensure_ascii=False))
        self._inspection_actions += 1
        return self._result(
            "ok", {"files": json.loads(output) if not truncated else output, "truncated": truncated}
        )

    def _read_file(self, path: str) -> dict:
        target = self.workspace.resolve(path, must_exist=True)
        raw = target.read_bytes()
        if len(raw) > self.limits.max_file_bytes:
            return self._result("error", "File exceeds the configured read limit.")
        text = raw.decode("utf-8")
        text, truncated = self._bounded_text(text)
        relative = self.workspace.relative(target)
        self._remember_file(relative, text, truncated)
        self._inspection_actions += 1
        return self._result(
            "ok", {"path": relative, "content": text, "truncated": truncated}
        )

    def _search_files(self, query: str) -> dict:
        if not query:
            return self._result("error", "Search query must not be empty.")
        matches = []
        for relative in self.workspace.list_files():
            target = self.workspace.resolve(relative, must_exist=True)
            try:
                if target.stat().st_size > self.limits.max_file_bytes:
                    continue
                with target.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if query in line:
                            matches.append(
                                {
                                    "path": relative,
                                    "line": line_number,
                                    "text": line.rstrip("\r\n")[:300],
                                }
                            )
                            if len(matches) >= 100:
                                break
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= 100:
                break
        output, truncated = self._bounded_text(json.dumps(matches, ensure_ascii=False))
        self._search_queries.add(query)
        self._inspection_actions += 1
        return self._result(
            "ok",
            {
                "matches": matches if not truncated else output,
                "truncated": truncated,
            },
        )

    @staticmethod
    def _validate_python(path: Path, content: str) -> None:
        if path.suffix == ".py":
            try:
                ast.parse(content, filename=str(path))
            except SyntaxError as error:
                raise ValueError(
                    f"Python syntax error on line {error.lineno}: {error.msg}"
                ) from error

    @staticmethod
    def _validate_python_change(path: Path, current: str, updated: str) -> None:
        """Reject structurally broken Python mutations before they reach checks."""
        Dispatcher._validate_python(path, updated)
        if path.suffix != ".py":
            return

        current_tree = ast.parse(current, filename=str(path))
        updated_tree = ast.parse(updated, filename=str(path))
        introduced_undefined = sorted(
            _undefined_python_globals(updated, str(path))
            - _undefined_python_globals(current, str(path))
        )
        if introduced_undefined:
            names = ", ".join(repr(name) for name in introduced_undefined)
            raise ValueError(
                "Python change introduces unresolved global name(s): "
                f"{names}. Restore the required import or definition."
            )

        if not _is_test_path(path):
            removed_imports = _imported_bindings(current_tree) - _imported_bindings(
                updated_tree
            )
            shadowed_imports = sorted(
                removed_imports & _defined_module_bindings(updated_tree)
            )
            if shadowed_imports:
                names = ", ".join(repr(name) for name in shadowed_imports)
                raise ValueError(
                    "Python change replaces existing imported binding(s) with local "
                    f"definitions: {names}. Preserve the existing dependency instead "
                    "of copying or reimplementing it in this file."
                )

        if (
            not _is_test_path(path)
            and not _contains_test_code(current_tree)
            and _contains_test_code(updated_tree)
        ):
            raise ValueError(
                "Python change adds test code to a non-test module. "
                "Keep regression tests in an authorized test file and repair only "
                "implementation code here."
            )

    def _write_file(self, path: str, content: str) -> dict:
        target = self.workspace.resolve(path, must_exist=False)
        self.workspace.require_writable(target)
        if len(content.encode("utf-8")) > self.limits.max_file_bytes:
            return self._result("error", "New file exceeds the configured size limit.")
        self._validate_python_change(target, "", content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        current, truncated = self._bounded_text(content)
        relative = self.workspace.relative(target)
        self._remember_file(relative, current, truncated)
        return self._result(
            "ok",
            {
                "path": relative,
                "created": True,
                "current_content": current,
                "truncated": truncated,
            },
        )

    def _replace_file(self, path: str, content: str) -> dict:
        target = self.workspace.resolve(path, must_exist=True)
        self.workspace.require_writable(target)
        if len(content.encode("utf-8")) > self.limits.max_file_bytes:
            return self._result(
                "error", "Replacement file exceeds the configured size limit."
            )
        current_text = target.read_text(encoding="utf-8")
        if content == current_text:
            visible_current, truncated = self._bounded_text(current_text)
            return self._result(
                "error",
                {
                    "message": "Replacement content is identical to the current file.",
                    "path": self.workspace.relative(target),
                    "current_content": visible_current,
                    "truncated": truncated,
                    "instruction": (
                        "Make a real task-relevant change or choose the next required "
                        "file or check."
                    ),
                },
            )
        try:
            self._validate_python_change(target, current_text, content)
        except ValueError as error:
            return self._edit_recovery_error(target, current_text, str(error))
        temporary = target.with_name(target.name + ".harness-tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
        current, truncated = self._bounded_text(content)
        relative = self.workspace.relative(target)
        self._remember_file(relative, current, truncated)
        return self._result(
            "ok",
            {
                "path": relative,
                "replaced": True,
                "current_content": current,
                "truncated": truncated,
            },
        )

    def _edit_recovery_error(
        self, target: Path, current: str, message: str
    ) -> dict:
        visible_current, truncated = self._bounded_text(current)
        return self._result(
            "error",
            {
                "message": message,
                "path": self.workspace.relative(target),
                "current_content": visible_current,
                "truncated": truncated,
                "instruction": (
                    "Build the next edit from current_content. If the exact edit "
                    "contract is fragile or invalid, use replace_file with the "
                    "corrected full content."
                ),
            },
        )

    def _edit_file(self, path: str, old: str, new: str) -> dict:
        target = self.workspace.resolve(path, must_exist=True)
        self.workspace.require_writable(target)
        current = target.read_text(encoding="utf-8")
        if not old:
            return self._edit_recovery_error(
                target, current, "old must be non-empty exact current text."
            )
        if old == new:
            return self._edit_recovery_error(
                target, current, "old and new are identical."
            )
        count = current.count(old)
        if count != 1:
            return self._edit_recovery_error(
                target,
                current,
                f"old occurs {count} times; expected exactly once.",
            )
        updated = current.replace(old, new, 1)
        if len(updated.encode("utf-8")) > self.limits.max_file_bytes:
            return self._edit_recovery_error(
                target, current, "Edited file exceeds the size limit."
            )
        try:
            self._validate_python_change(target, current, updated)
        except ValueError as error:
            return self._edit_recovery_error(target, current, str(error))
        temporary = target.with_name(target.name + ".harness-tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(target)
        visible_updated, truncated = self._bounded_text(updated)
        relative = self.workspace.relative(target)
        self._remember_file(relative, visible_updated, truncated)
        return self._result(
            "ok",
            {
                "path": relative,
                "edited": True,
                "current_content": visible_updated,
                "truncated": truncated,
            },
        )

    def _run_check(self, name: str) -> dict:
        check = self.checks.get(name)
        if check is None:
            return self._result("error", f"Unknown configured check: {name!r}")
        if not check.model_allowed:
            return self._result(
                "denied", f"Check {name!r} is reserved for final verification."
            )
        command_result = self.runner.run(check)
        status = "ok" if command_result.status == "passed" else "error"
        return self._result(status, command_result.to_dict())

