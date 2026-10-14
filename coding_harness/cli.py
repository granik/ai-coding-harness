"""Command-line interface for the coding harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import ConfigError, load_config
from .controller import CodingHarness
from .executor import CheckRunner
from .model import ModelError, OllamaModel, ScriptedModel
from .tools import Dispatcher
from .workspace import DisposableWorkspace, WorkspaceError


class ConsoleReporter:
    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path

    def __call__(self, event: dict) -> None:
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        kind = event.get("event")
        if kind == "model_start":
            print(f"Turn {event['turn']} start: model")
        elif kind == "tool_start":
            location = f" [{event['location']}]" if event.get("location") else ""
            print(f"Turn {event['turn']}: {event['tool']}{location}")
        elif kind == "tool_end":
            print(f"  -> {event['result'].get('status', 'result')}")
        elif kind == "model_end" and event.get("status") not in {"final"}:
            print(f"Turn {event['turn']} model -> {event.get('status')}")
        elif kind == "check_start":
            print(f"Check {event['name']} ({event['kind']}) start")
        elif kind == "check_end":
            result = event["result"]
            print(
                f"Check {event['name']} -> {result['status']} "
                f"(exit={result['exit_code']})"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 local coding harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a coding task in a disposable copy")
    run.add_argument("--config", required=True, help="trusted TOML configuration")
    run.add_argument("--task", help="override the configured task")
    run.add_argument("--model", help="override the Ollama model name")
    run.add_argument("--backend", choices=("docker", "process"))
    run.add_argument(
        "--mode",
        choices=("edit", "read-only"),
        help="override the configured tool mode for this run",
    )
    run.add_argument(
        "--scripted",
        help="JSONL model replies for a repeatable demo; does not call Ollama",
    )

    check = subparsers.add_parser("check-config", help="validate configuration")
    check.add_argument("--config", required=True)
    return parser


def _print_report(report: dict, report_path: Path) -> None:
    print("\nChanged files:")
    if report["changed_files"]:
        for path in report["changed_files"]:
            print(f"  {path}")
    else:
        print("  (none)")
    print("\nDiff:")
    print(report["diff"] or "(no diff)")
    print("Checks:")
    for result in report["checks"]:
        print(
            f"  {result['name']}: {result['status']} "
            f"(exit={result['exit_code']})"
        )
        if result["output"]:
            print(result["output"].rstrip())
    print(f"\nVerification passed: {report['verification_passed']}")
    print(f"Termination: {report['termination']}")
    print(f"Report: {report_path}")
    print(f"Repository copy: {report['repository_copy']}")


def run_command(args: argparse.Namespace) -> int:
    config = load_config(args.config).with_overrides(
        task=args.task, model=args.model, backend=args.backend, mode=args.mode
    )
    DisposableWorkspace.verify_start_commit(
        config.project.source, config.project.start_commit
    )
    workspace = DisposableWorkspace.create(
        config.project.source,
        config.project.runs_dir,
        writable=config.project.writable,
        max_file_bytes=config.limits.max_file_bytes,
        max_repo_bytes=config.limits.max_repo_bytes,
        max_files=config.limits.max_files,
    )
    reporter = ConsoleReporter(workspace.run_root / "trace.jsonl")
    runner = CheckRunner(
        execution=config.execution,
        limits=config.limits,
        repo=workspace.repo,
        acceptance_dir=config.project.acceptance_dir,
        control_dir=workspace.run_root / "control",
    )
    dispatcher = Dispatcher(
        workspace=workspace,
        runner=runner,
        checks=config.checks,
        limits=config.limits,
    )
    model = (
        ScriptedModel.from_jsonl(args.scripted)
        if args.scripted
        else OllamaModel(config.model)
    )
    reporter(
        {
            "event": "run_start",
            "project": config.project.name,
            "model": "scripted" if args.scripted else config.model.name,
            "backend": config.execution.backend,
            "mode": config.project.mode,
            "task": config.project.task,
            "repository_copy": str(workspace.repo),
        }
    )
    report = CodingHarness(
        config=config,
        workspace=workspace,
        model=model,
        dispatcher=dispatcher,
        runner=runner,
        emit=reporter,
    ).run()
    report_path = workspace.run_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _print_report(report, report_path)
    return 0 if report["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check-config":
            config = load_config(args.config)
            print(
                f"Configuration is valid: {config.project.name}; "
                f"{len(config.checks)} checks"
            )
            raise SystemExit(0)
        raise SystemExit(run_command(args))
    except (ConfigError, WorkspaceError, ModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()

