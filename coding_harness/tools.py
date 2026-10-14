"""Data-driven tool registry and permission-enforcing dispatcher."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
        self._seen: dict[str, int] = {}
        self._handlers: dict[str, Callable[..., dict]] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "search_files": self._search_files,
            "write_file": self._write_file,
            "replace_file": self._replace_file,
            "edit_file": self._edit_file,
            "run_check": self._run_check,
        }

    def schema(self, *, only_tools: set[str] | None = None) -> dict:
        variants = []
        for name, spec in TOOL_SPECS.items():
            if only_tools is not None and name not in only_tools:
                continue
            properties = {
                argument: {"type": "string"} for argument in spec.arguments
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
        variants.append(
            {
                "type": "object",
                "properties": {"final": {"type": "string"}},
                "required": ["final"],
                "additionalProperties": False,
            }
        )
        return {"oneOf": variants}

    @staticmethod
    def replacement_schema(path: str) -> dict:
        """Simple recovery grammar for models that do not enforce oneOf."""
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
                        "content": {"type": "string"},
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
        count = self._seen.get(signature, 0) + 1
        self._seen[signature] = count
        if count > 1:
            self.stats.retries += 1
        if count > self.limits.max_retries + 1:
            return self._result(
                "denied",
                "Repeated action exceeded the configured retry limit.",
                fatal=True,
            )
        if allowed_tools is not None and tool not in allowed_tools:
            return self._result(
                "denied",
                "This recovery phase permits only: "
                + ", ".join(sorted(allowed_tools))
                + ".",
            )
        if required_path is not None and args.get("path") != required_path:
            return self._result(
                "denied",
                f"This recovery phase permits only path {required_path!r}.",
            )
        try:
            result = self._handlers[tool](**args)
            if result.get("status") == "ok" and TOOL_SPECS[tool].mutating:
                # A successful mutation changes the state against which reads,
                # edits, and checks operate, so identical later actions are no
                # longer stalled retries.
                self._seen.clear()
            return result
        except WorkspaceError as error:
            return self._result("denied", str(error))
        except (OSError, UnicodeError, ValueError) as error:
            return self._result("error", str(error))

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
        return self._result(
            "ok", {"path": self.workspace.relative(target), "content": text, "truncated": truncated}
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

    def _write_file(self, path: str, content: str) -> dict:
        target = self.workspace.resolve(path, must_exist=False)
        self.workspace.require_writable(target)
        if len(content.encode("utf-8")) > self.limits.max_file_bytes:
            return self._result("error", "New file exceeds the configured size limit.")
        self._validate_python(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        current, truncated = self._bounded_text(content)
        return self._result(
            "ok",
            {
                "path": self.workspace.relative(target),
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
        self._validate_python(target, content)
        temporary = target.with_name(target.name + ".harness-tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
        current, truncated = self._bounded_text(content)
        return self._result(
            "ok",
            {
                "path": self.workspace.relative(target),
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
            self._validate_python(target, updated)
        except ValueError as error:
            return self._edit_recovery_error(target, current, str(error))
        temporary = target.with_name(target.name + ".harness-tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.replace(target)
        visible_updated, truncated = self._bounded_text(updated)
        return self._result(
            "ok",
            {
                "path": self.workspace.relative(target),
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

