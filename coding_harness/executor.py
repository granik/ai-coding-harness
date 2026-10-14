"""Bounded execution backends for configured checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import time
from typing import Mapping

from .config import CheckSpec, ExecutionConfig, LimitsConfig


@dataclass(frozen=True)
class CommandResult:
    status: str
    command: list[str]
    cwd: str
    exit_code: int | None
    output: str
    timed_out: bool
    truncated: bool
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _limit_child(memory_mb: int, cpu_seconds: int, pids: int):
    """Return a POSIX pre-exec hook with conservative resource limits."""

    # RLIMIT_NPROC is intentionally not used here. On macOS it counts every
    # process owned by the desktop user, so a small per-command value can reject
    # the first legitimate child. Docker applies an isolated --pids-limit.
    del pids

    def apply() -> None:
        memory = memory_mb * 1024 * 1024
        limits = [
            (resource.RLIMIT_CORE, (0, 0)),
            (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
            (resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024)),
        ]
        if hasattr(resource, "RLIMIT_AS"):
            limits.append((resource.RLIMIT_AS, (memory, memory)))
        for key, value in limits:
            try:
                resource.setrlimit(key, value)
            except (OSError, ValueError):
                pass

    return apply


def _bounded_process(
    command: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    output_limit: int,
    memory_mb: int,
    cpu_seconds: int,
    pids: int,
) -> CommandResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(env),
            start_new_session=True,
            preexec_fn=_limit_child(memory_mb, cpu_seconds, pids),
        )
    except (OSError, ValueError) as error:
        return CommandResult(
            status="unavailable",
            command=command,
            cwd=str(cwd),
            exit_code=None,
            output=str(error),
            timed_out=False,
            truncated=False,
            duration_seconds=round(time.monotonic() - started, 3),
        )

    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    captured = bytearray()
    total = 0
    timed_out = False
    deadline = started + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                break
            for key, _ in selector.select(timeout=min(0.2, remaining)):
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                total += len(chunk)
                remaining_capacity = output_limit - len(captured)
                if remaining_capacity > 0:
                    captured.extend(chunk[:remaining_capacity])
        if process.poll() is None:
            process.wait(timeout=1)
    finally:
        selector.close()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        process.stdout.close()

    truncated = total > len(captured)
    output = captured.decode("utf-8", errors="replace")
    if truncated:
        output += f"\n[output shortened: captured {len(captured)} of {total} bytes]\n"
    if timed_out:
        output += f"\n[command stopped after {timeout_seconds} seconds]\n"
    exit_code = process.returncode
    status = "timeout" if timed_out else ("passed" if exit_code == 0 else "failed")
    return CommandResult(
        status=status,
        command=command,
        cwd=str(cwd),
        exit_code=exit_code,
        output=output,
        timed_out=timed_out,
        truncated=truncated,
        duration_seconds=round(time.monotonic() - started, 3),
    )


class CheckRunner:
    """Execute only trusted, configured checks."""

    def __init__(
        self,
        *,
        execution: ExecutionConfig,
        limits: LimitsConfig,
        repo: Path,
        acceptance_dir: Path | None,
        control_dir: Path,
    ) -> None:
        self.execution = execution
        self.limits = limits
        self.repo = repo
        self.acceptance_dir = acceptance_dir
        self.control_dir = control_dir

    def run(self, check: CheckSpec) -> CommandResult:
        if self.execution.backend == "docker":
            return self._run_docker(check)
        return self._run_process(check)

    def _base_env(self) -> dict[str, str]:
        home = self.control_dir / "home"
        tmp = self.control_dir / "tmp"
        home.mkdir(exist_ok=True)
        tmp.mkdir(exist_ok=True)
        # Keep the virtual-environment path itself. Resolving its python symlink
        # would silently move configured checks back to the system interpreter.
        executable_dir = str(Path(sys.executable).parent)
        inherited_path = os.environ.get("PATH", "/usr/bin:/bin")
        return {
            "PATH": executable_dir + os.pathsep + inherited_path,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "HARNESS_REPO": str(self.repo),
        }

    def _run_process(self, check: CheckSpec) -> CommandResult:
        if not self.execution.allow_local_process:
            return CommandResult(
                status="unavailable",
                command=list(check.command),
                cwd=check.cwd,
                exit_code=None,
                output=(
                    "Local process execution is disabled. Use the docker backend or "
                    "set allow_local_process=true after reviewing the target."
                ),
                timed_out=False,
                truncated=False,
                duration_seconds=0.0,
            )
        cwd = self.repo if check.cwd == "repo" else self.acceptance_dir
        if cwd is None:
            return CommandResult(
                status="unavailable",
                command=list(check.command),
                cwd=check.cwd,
                exit_code=None,
                output="Acceptance directory is not configured.",
                timed_out=False,
                truncated=False,
                duration_seconds=0.0,
            )
        return _bounded_process(
            list(check.command),
            cwd=cwd,
            env=self._base_env(),
            timeout_seconds=check.timeout_seconds,
            output_limit=self.limits.max_command_output_bytes,
            memory_mb=self.execution.memory_mb,
            cpu_seconds=min(self.execution.cpu_seconds, check.timeout_seconds),
            pids=self.execution.pids,
        )

    def _run_docker(self, check: CheckSpec) -> CommandResult:
        docker = shutil.which("docker")
        if docker is None:
            return CommandResult(
                status="unavailable",
                command=list(check.command),
                cwd=check.cwd,
                exit_code=None,
                output="Docker is not installed or not on PATH.",
                timed_out=False,
                truncated=False,
                duration_seconds=0.0,
            )
        command = [
            docker,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
            f"--memory={self.execution.memory_mb}m",
            f"--cpus=1",
            f"--pids-limit={self.execution.pids}",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HARNESS_REPO=/workspace",
            "-v",
            f"{self.repo}:/workspace:rw",
        ]
        if self.acceptance_dir is not None:
            command.extend(["-v", f"{self.acceptance_dir}:/acceptance:ro"])
        command.extend(
            [
                "-w",
                "/workspace" if check.cwd == "repo" else "/acceptance",
                self.execution.docker_image,
                *check.command,
            ]
        )
        docker_env = self._base_env()
        # Docker CLI may need the user's Desktop context. These values configure
        # the trusted host client and are not passed into the target container.
        if os.environ.get("HOME"):
            docker_env["HOME"] = os.environ["HOME"]
        if os.environ.get("DOCKER_HOST"):
            docker_env["DOCKER_HOST"] = os.environ["DOCKER_HOST"]
        return _bounded_process(
            command,
            cwd=self.control_dir,
            env=docker_env,
            timeout_seconds=check.timeout_seconds,
            output_limit=self.limits.max_command_output_bytes,
            memory_mb=max(self.execution.memory_mb + 128, 256),
            cpu_seconds=min(self.execution.cpu_seconds, check.timeout_seconds),
            pids=self.execution.pids + 16,
        )

