"""Trusted TOML configuration for a harness run."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import tomllib


class ConfigError(ValueError):
    """Raised when trusted harness configuration is invalid."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    source: Path
    runs_dir: Path
    acceptance_dir: Path | None
    start_commit: str
    task: str
    writable: tuple[str, ...]
    mode: str = "edit"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "qwen2.5-coder:7b"
    endpoint: str = "http://127.0.0.1:11434/api/chat"
    timeout_seconds: int = 300
    context_tokens: int = 16_384
    predict_tokens: int = 2_048


@dataclass(frozen=True)
class LimitsConfig:
    max_actions: int = 15
    max_inspection_actions: int = 6
    max_retries: int = 2
    max_file_bytes: int = 100_000
    max_tool_output_bytes: int = 20_000
    max_command_output_bytes: int = 50_000
    max_repo_bytes: int = 50_000_000
    max_files: int = 5_000


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "docker"
    allow_local_process: bool = False
    docker_image: str = "python:3.11-slim"
    memory_mb: int = 512
    cpu_seconds: int = 30
    pids: int = 64


@dataclass(frozen=True)
class CheckSpec:
    name: str
    command: tuple[str, ...]
    cwd: str
    kind: str
    model_allowed: bool
    timeout_seconds: int


@dataclass(frozen=True)
class HarnessConfig:
    path: Path
    project: ProjectConfig
    model: ModelConfig
    limits: LimitsConfig
    execution: ExecutionConfig
    checks: tuple[CheckSpec, ...]

    def with_overrides(
        self,
        *,
        task: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        mode: str | None = None,
    ) -> "HarnessConfig":
        project_cfg = self.project
        if task:
            project_cfg = replace(project_cfg, task=task)
        if mode:
            project_cfg = replace(project_cfg, mode=mode)
        model_cfg = replace(self.model, name=model) if model else self.model
        execution_cfg = (
            replace(self.execution, backend=backend) if backend else self.execution
        )
        return replace(
            self,
            project=project_cfg,
            model=model_cfg,
            execution=execution_cfg,
        )


def _table(data: dict, name: str) -> dict:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a TOML table")
    return value


def _positive(value, label: str, *, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{label} must be at most {maximum}")
    return value


def _string(value, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{label} must be a non-empty string")
    return value


def _project_mode(value) -> str:
    mode = _string(value, "project.mode")
    if mode not in {"edit", "read-only"}:
        raise ConfigError("project.mode must be 'edit' or 'read-only'")
    return mode


def _resolve(base: Path, value: str, label: str) -> Path:
    raw = Path(_string(value, label)).expanduser()
    return (base / raw).resolve() if not raw.is_absolute() else raw.resolve()


def load_config(path: str | Path) -> HarnessConfig:
    """Load and validate a trusted harness configuration file."""
    config_path = Path(path).expanduser().resolve()
    try:
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Cannot load {config_path}: {error}") from error
    base = config_path.parent

    project_data = _table(data, "project")
    source = _resolve(base, project_data.get("source", ""), "project.source")
    if not source.is_dir():
        raise ConfigError(f"project.source is not a directory: {source}")
    runs_dir = _resolve(
        base, project_data.get("runs_dir", ".harness/runs"), "project.runs_dir"
    )
    if runs_dir.is_relative_to(source):
        raise ConfigError("project.runs_dir must be outside the source repository")
    acceptance_value = project_data.get("acceptance_dir")
    acceptance_dir = (
        _resolve(base, acceptance_value, "project.acceptance_dir")
        if acceptance_value
        else None
    )
    if acceptance_dir is not None and not acceptance_dir.is_dir():
        raise ConfigError(f"project.acceptance_dir is not a directory: {acceptance_dir}")
    if acceptance_dir is not None and acceptance_dir.is_relative_to(source):
        raise ConfigError("acceptance_dir must be outside the model-writable source")
    writable_value = project_data.get("writable", ["**"])
    if not isinstance(writable_value, list) or not all(
        isinstance(item, str) and item for item in writable_value
    ):
        raise ConfigError("project.writable must be a list of path patterns")
    project = ProjectConfig(
        name=_string(project_data.get("name", source.name), "project.name"),
        source=source,
        runs_dir=runs_dir,
        acceptance_dir=acceptance_dir,
        start_commit=_string(
            project_data.get("start_commit", ""),
            "project.start_commit",
        ),
        task=_string(project_data.get("task", ""), "project.task"),
        writable=tuple(writable_value),
        mode=_project_mode(project_data.get("mode", "edit")),
    )

    model_data = _table(data, "model")
    model_cfg = ModelConfig(
        name=_string(model_data.get("name", ModelConfig.name), "model.name"),
        endpoint=_string(
            model_data.get("endpoint", ModelConfig.endpoint), "model.endpoint"
        ),
        timeout_seconds=_positive(
            model_data.get("timeout_seconds", ModelConfig.timeout_seconds),
            "model.timeout_seconds",
            maximum=900,
        ),
        context_tokens=_positive(
            model_data.get("context_tokens", ModelConfig.context_tokens),
            "model.context_tokens",
        ),
        predict_tokens=_positive(
            model_data.get("predict_tokens", ModelConfig.predict_tokens),
            "model.predict_tokens",
        ),
    )

    limits_data = _table(data, "limits")
    defaults = LimitsConfig()
    limits = LimitsConfig(
        max_actions=_positive(
            limits_data.get("max_actions", defaults.max_actions),
            "limits.max_actions",
            maximum=50,
        ),
        max_inspection_actions=_positive(
            limits_data.get(
                "max_inspection_actions", defaults.max_inspection_actions
            ),
            "limits.max_inspection_actions",
            maximum=20,
        ),
        max_retries=_positive(
            limits_data.get("max_retries", defaults.max_retries),
            "limits.max_retries",
            maximum=10,
        ),
        max_file_bytes=_positive(
            limits_data.get("max_file_bytes", defaults.max_file_bytes),
            "limits.max_file_bytes",
        ),
        max_tool_output_bytes=_positive(
            limits_data.get(
                "max_tool_output_bytes", defaults.max_tool_output_bytes
            ),
            "limits.max_tool_output_bytes",
        ),
        max_command_output_bytes=_positive(
            limits_data.get(
                "max_command_output_bytes", defaults.max_command_output_bytes
            ),
            "limits.max_command_output_bytes",
        ),
        max_repo_bytes=_positive(
            limits_data.get("max_repo_bytes", defaults.max_repo_bytes),
            "limits.max_repo_bytes",
        ),
        max_files=_positive(
            limits_data.get("max_files", defaults.max_files), "limits.max_files"
        ),
    )

    execution_data = _table(data, "execution")
    execution_defaults = ExecutionConfig()
    backend = execution_data.get("backend", execution_defaults.backend)
    if backend not in {"docker", "process"}:
        raise ConfigError("execution.backend must be 'docker' or 'process'")
    allow_local = execution_data.get(
        "allow_local_process", execution_defaults.allow_local_process
    )
    if not isinstance(allow_local, bool):
        raise ConfigError("execution.allow_local_process must be true or false")
    if backend == "process" and not allow_local:
        raise ConfigError(
            "process backend requires execution.allow_local_process=true"
        )
    execution = ExecutionConfig(
        backend=backend,
        allow_local_process=allow_local,
        docker_image=_string(
            execution_data.get("docker_image", execution_defaults.docker_image),
            "execution.docker_image",
        ),
        memory_mb=_positive(
            execution_data.get("memory_mb", execution_defaults.memory_mb),
            "execution.memory_mb",
        ),
        cpu_seconds=_positive(
            execution_data.get("cpu_seconds", execution_defaults.cpu_seconds),
            "execution.cpu_seconds",
        ),
        pids=_positive(
            execution_data.get("pids", execution_defaults.pids),
            "execution.pids",
        ),
    )

    raw_checks = data.get("checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ConfigError("at least one [[checks]] table is required")
    checks: list[CheckSpec] = []
    names: set[str] = set()
    for index, item in enumerate(raw_checks):
        if not isinstance(item, dict):
            raise ConfigError(f"checks[{index}] must be a table")
        name = _string(item.get("name", ""), f"checks[{index}].name")
        if name in names:
            raise ConfigError(f"duplicate check name: {name}")
        names.add(name)
        command = item.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ConfigError(f"checks[{index}].command must be a string list")
        cwd = item.get("cwd", "repo")
        if cwd not in {"repo", "acceptance"}:
            raise ConfigError(f"checks[{index}].cwd must be repo or acceptance")
        if cwd == "acceptance" and acceptance_dir is None:
            raise ConfigError(f"check {name} needs project.acceptance_dir")
        kind = item.get("kind", "regression")
        if kind not in {"regression", "acceptance", "lint", "other"}:
            raise ConfigError(f"invalid check kind for {name}: {kind}")
        model_allowed = item.get("model_allowed", False)
        if not isinstance(model_allowed, bool):
            raise ConfigError(f"checks[{index}].model_allowed must be boolean")
        checks.append(
            CheckSpec(
                name=name,
                command=tuple(command),
                cwd=cwd,
                kind=kind,
                model_allowed=model_allowed,
                timeout_seconds=_positive(
                    item.get("timeout_seconds", execution.cpu_seconds),
                    f"checks[{index}].timeout_seconds",
                    maximum=600,
                ),
            )
        )
    if not any(check.kind == "acceptance" for check in checks):
        raise ConfigError("at least one acceptance check is required")

    return HarnessConfig(
        path=config_path,
        project=project,
        model=model_cfg,
        limits=limits,
        execution=execution,
        checks=tuple(checks),
    )
