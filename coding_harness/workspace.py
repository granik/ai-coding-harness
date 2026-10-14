"""Disposable repository copy, path boundary, snapshots, and diffs."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid


DEFAULT_EXCLUDES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".harness",
    ".env",
}


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SnapshotEntry:
    digest: str
    content: bytes


class DisposableWorkspace:
    """A copied repository with a strict relative-path boundary."""

    def __init__(
        self,
        *,
        run_root: Path,
        repo: Path,
        writable: tuple[str, ...],
        max_file_bytes: int,
        max_repo_bytes: int,
        max_files: int,
        baseline: dict[str, SnapshotEntry],
    ) -> None:
        self.run_root = run_root
        self.repo = repo
        self.writable = writable
        self.max_file_bytes = max_file_bytes
        self.max_repo_bytes = max_repo_bytes
        self.max_files = max_files
        self._baseline = baseline

    @classmethod
    def create(
        cls,
        source: Path,
        runs_dir: Path,
        *,
        writable: tuple[str, ...],
        max_file_bytes: int,
        max_repo_bytes: int,
        max_files: int,
    ) -> "DisposableWorkspace":
        source = source.resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = runs_dir.resolve()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_root = runs_dir / f"{stamp}-{uuid.uuid4().hex[:8]}"
        repo = run_root / "repo"
        control = run_root / "control"
        repo.mkdir(parents=True)
        control.mkdir()
        cls._copy_repository(
            source,
            repo,
            max_repo_bytes=max_repo_bytes,
            max_files=max_files,
        )
        run_root = run_root.resolve()
        repo = repo.resolve()
        baseline = cls._snapshot(repo, max_files=max_files)
        return cls(
            run_root=run_root,
            repo=repo,
            writable=writable,
            max_file_bytes=max_file_bytes,
            max_repo_bytes=max_repo_bytes,
            max_files=max_files,
            baseline=baseline,
        )

    @staticmethod
    def _excluded(name: str) -> bool:
        return (
            name in DEFAULT_EXCLUDES
            or name.startswith(".env.")
            or name in {"credentials.json", "service-account.json"}
            or name.endswith((".pyc", ".pyo", ".pem", ".key", ".p12"))
        )

    @staticmethod
    def verify_start_commit(source: Path, expected: str) -> None:
        """Fail closed when a Git source does not match its recorded baseline."""
        git_directory = source / ".git"
        if not git_directory.exists():
            return
        try:
            completed = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise WorkspaceError(f"Cannot verify recorded start commit: {error}") from error
        if completed.returncode != 0:
            raise WorkspaceError(
                "Cannot read source Git commit: " + completed.stderr.strip()
            )
        actual = completed.stdout.strip()
        if actual != expected:
            raise WorkspaceError(
                f"Source commit mismatch: expected {expected}, found {actual}"
            )
        try:
            status = subprocess.run(
                ["git", "-C", str(source), "status", "--porcelain"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise WorkspaceError(f"Cannot verify source Git status: {error}") from error
        if status.returncode != 0:
            raise WorkspaceError(
                "Cannot verify source Git status: " + status.stderr.strip()
            )
        if status.stdout.strip():
            raise WorkspaceError(
                "Source repository has uncommitted or untracked changes; restore the "
                "recorded baseline before creating a disposable copy."
            )

    @classmethod
    def _copy_repository(
        cls,
        source: Path,
        destination: Path,
        *,
        max_repo_bytes: int,
        max_files: int,
    ) -> None:
        total_bytes = 0
        total_files = 0
        for directory, dirnames, filenames in os.walk(source, followlinks=False):
            current = Path(directory)
            relative_dir = current.relative_to(source)
            kept_dirs = []
            for name in sorted(dirnames):
                candidate = current / name
                if cls._excluded(name) or candidate.is_symlink():
                    continue
                kept_dirs.append(name)
                (destination / relative_dir / name).mkdir(parents=True, exist_ok=True)
            dirnames[:] = kept_dirs
            for name in sorted(filenames):
                candidate = current / name
                if cls._excluded(name) or candidate.is_symlink():
                    continue
                if not candidate.is_file():
                    continue
                size = candidate.stat().st_size
                total_files += 1
                total_bytes += size
                if total_files > max_files:
                    raise WorkspaceError(
                        f"Repository has more than {max_files} copied files"
                    )
                if total_bytes > max_repo_bytes:
                    raise WorkspaceError(
                        f"Repository exceeds the {max_repo_bytes}-byte copy limit"
                    )
                target = destination / relative_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target)

    @staticmethod
    def _snapshot(repo: Path, *, max_files: int) -> dict[str, SnapshotEntry]:
        result: dict[str, SnapshotEntry] = {}
        for index, path in enumerate(
            sorted(item for item in repo.rglob("*") if item.is_file()), start=1
        ):
            if index > max_files:
                raise WorkspaceError(f"Workspace has more than {max_files} files")
            if path.is_symlink():
                continue
            content = path.read_bytes()
            relative = path.relative_to(repo).as_posix()
            result[relative] = SnapshotEntry(
                digest=hashlib.sha256(content).hexdigest(), content=content
            )
        return result

    def resolve(self, value: str, *, must_exist: bool | None = None) -> Path:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise WorkspaceError("Path must be a non-empty string")
        candidate = Path(value)
        if candidate.is_absolute() or any(part in {"..", "."} for part in candidate.parts):
            raise WorkspaceError("Path must stay inside the repository")
        if any(part.startswith(".") for part in candidate.parts):
            raise WorkspaceError("Hidden paths are not available to tools")
        target = (self.repo / candidate).resolve(strict=False)
        try:
            target.relative_to(self.repo)
        except ValueError as error:
            raise WorkspaceError("Path escapes the repository") from error
        cursor = self.repo
        for part in candidate.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise WorkspaceError("Symlink paths are not available to tools")
        if must_exist is True and not target.is_file():
            raise WorkspaceError("File does not exist")
        if must_exist is False and target.exists():
            raise WorkspaceError("Path already exists")
        return target

    def relative(self, path: Path) -> str:
        return path.relative_to(self.repo).as_posix()

    def is_writable(self, path: Path) -> bool:
        relative = self.relative(path)
        return any(
            pattern in {"*", "**"}
            or Path("/" + relative).match("/" + pattern)
            for pattern in self.writable
        )

    def require_writable(self, path: Path) -> None:
        if not self.is_writable(path):
            raise WorkspaceError(
                f"Path is outside configured writable patterns: {self.relative(path)}"
            )

    def list_files(self) -> list[str]:
        files = []
        for path in sorted(self.repo.rglob("*")):
            relative_path = path.relative_to(self.repo)
            if any(part.startswith(".") for part in relative_path.parts):
                continue
            if path.is_file() and not path.is_symlink():
                files.append(relative_path.as_posix())
                if len(files) >= self.max_files:
                    break
        return files

    def current_snapshot(self) -> dict[str, SnapshotEntry]:
        return self._snapshot(self.repo, max_files=self.max_files)

    def changed_files(self) -> list[str]:
        current = self.current_snapshot()
        names = set(self._baseline) | set(current)
        return sorted(
            name
            for name in names
            if name not in self._baseline
            or name not in current
            or self._baseline[name].digest != current[name].digest
        )

    @staticmethod
    def _text(content: bytes) -> list[str] | None:
        if b"\x00" in content:
            return None
        try:
            return content.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            return None

    def diff(self, max_bytes: int = 100_000) -> tuple[str, bool]:
        current = self.current_snapshot()
        sections: list[str] = []
        for name in self.changed_files():
            before = self._baseline.get(name)
            after = current.get(name)
            before_bytes = before.content if before else b""
            after_bytes = after.content if after else b""
            before_text = self._text(before_bytes)
            after_text = self._text(after_bytes)
            if before_text is None or after_text is None:
                sections.append(
                    f"Binary file changed: {name} "
                    f"({len(before_bytes)} -> {len(after_bytes)} bytes)\n"
                )
                continue
            diff_lines = difflib.unified_diff(
                before_text,
                after_text,
                fromfile=f"a/{name}" if before else "/dev/null",
                tofile=f"b/{name}" if after else "/dev/null",
            )
            for line in diff_lines:
                if line.endswith(("\n", "\r")):
                    sections.append(line)
                    continue
                sections.append(line + "\n")
                if line.startswith(("+", "-", " ")):
                    sections.append("\\ No newline at end of file\n")
        rendered = "".join(sections)
        encoded = rendered.encode("utf-8")
        if len(encoded) <= max_bytes:
            return rendered, False
        suffix = "\n[diff shortened by harness output limit]\n"
        shortened = encoded[: max(0, max_bytes - len(suffix.encode()))].decode(
            "utf-8", errors="ignore"
        )
        return shortened + suffix, True

