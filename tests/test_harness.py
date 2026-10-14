from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from coding_harness.config import (
    CheckSpec,
    ExecutionConfig,
    HarnessConfig,
    LimitsConfig,
    ModelConfig,
    ProjectConfig,
)
from coding_harness.controller import CodingHarness
from coding_harness.executor import CheckRunner, CommandResult
from coding_harness.model import ScriptedModel
from coding_harness.tools import Dispatcher
from coding_harness.workspace import DisposableWorkspace


class HarnessTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.acceptance = self.base / "acceptance"
        self.runs = self.base / "runs"
        self.source.mkdir()
        self.acceptance.mkdir()
        self.limits = LimitsConfig(
            max_actions=8,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        self.execution = ExecutionConfig(
            backend="process",
            allow_local_process=True,
            memory_mb=1024,
            cpu_seconds=10,
            pids=128,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_parts(
        self,
        *,
        checks: tuple[CheckSpec, ...] = (),
        writable=("**",),
        limits=None,
        task="repair the bug",
    ):
        selected_limits = limits or self.limits
        workspace = DisposableWorkspace.create(
            self.source,
            self.runs,
            writable=tuple(writable),
            max_file_bytes=selected_limits.max_file_bytes,
            max_repo_bytes=selected_limits.max_repo_bytes,
            max_files=selected_limits.max_files,
        )
        runner = CheckRunner(
            execution=self.execution,
            limits=selected_limits,
            repo=workspace.repo,
            acceptance_dir=self.acceptance,
            control_dir=workspace.run_root / "control",
        )
        dispatcher = Dispatcher(
            workspace=workspace,
            runner=runner,
            checks=checks,
            limits=selected_limits,
        )
        config = HarnessConfig(
            path=self.base / "test.toml",
            project=ProjectConfig(
                name="fixture",
                source=self.source,
                runs_dir=self.runs,
                acceptance_dir=self.acceptance,
                start_commit="fixture-commit",
                task=task,
                writable=tuple(writable),
            ),
            model=ModelConfig(),
            limits=selected_limits,
            execution=self.execution,
            checks=checks,
        )
        return config, workspace, runner, dispatcher

    def test_file_tools_and_path_boundary(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.source / "locked.py").write_text("LOCKED = True\n", encoding="utf-8")
        (self.source / ".notes").write_text("hidden\n", encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py",)
        )

        listed = dispatcher.dispatch({"tool": "list_files", "args": {}})
        self.assertEqual(listed["status"], "ok")
        self.assertNotIn(".notes", listed["output"]["files"])
        read = dispatcher.dispatch(
            {"tool": "read_file", "args": {"path": "app.py"}}
        )
        self.assertEqual(read["output"]["content"], "VALUE = 1\n")
        search = dispatcher.dispatch(
            {"tool": "search_files", "args": {"query": "VALUE"}}
        )
        self.assertEqual(search["output"]["matches"][0]["path"], "app.py")
        edited = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {"path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"},
            }
        )
        self.assertEqual(edited["status"], "ok")
        self.assertEqual((workspace.repo / "app.py").read_text(), "VALUE = 2\n")

        protected = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "locked.py",
                    "old": "LOCKED = True",
                    "new": "LOCKED = False",
                },
            }
        )
        escaped = dispatcher.dispatch(
            {
                "tool": "write_file",
                "args": {"path": "../escape.txt", "content": "blocked"},
            }
        )
        self.assertEqual(protected["status"], "denied")
        self.assertEqual(escaped["status"], "denied")
        self.assertFalse((workspace.run_root / "escape.txt").exists())

    def test_unknown_tool_and_invalid_arguments_are_errors(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        _config, _workspace, _runner, dispatcher = self.make_parts()
        unknown = dispatcher.dispatch({"tool": "missing", "args": {}})
        invalid = dispatcher.dispatch(
            {"tool": "read_file", "args": {"wrong": "app.py"}}
        )
        self.assertEqual(unknown["status"], "error")
        self.assertIn("Unknown tool", unknown["output"])
        self.assertEqual(invalid["status"], "error")
        self.assertIn("Invalid arguments", invalid["output"])

    def test_python_syntax_error_does_not_change_file(self):
        path = self.source / "app.py"
        path.write_text("def value():\n    return 1\n", encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts()
        result = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "app.py",
                    "old": "    return 1",
                    "new": "    return (",
                },
            }
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(
            (workspace.repo / "app.py").read_text(), "def value():\n    return 1\n"
        )

    def test_diff_marks_a_missing_final_newline(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py",)
        )
        result = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "app.py", "content": "VALUE = 2"},
            }
        )
        self.assertEqual(result["status"], "ok")
        diff, truncated = workspace.diff()
        self.assertFalse(truncated)
        self.assertIn(
            "+VALUE = 2\n\\ No newline at end of file\n",
            diff,
        )

    def test_edit_mismatch_returns_current_file_for_recovery(self):
        (self.source / "app.py").write_text(
            "def value():\n    return 1\n", encoding="utf-8"
        )
        _config, _workspace, _runner, dispatcher = self.make_parts()
        result = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "app.py",
                    "old": "return 99",
                    "new": "return 2",
                },
            }
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["output"]["path"], "app.py")
        self.assertEqual(
            result["output"]["current_content"],
            "def value():\n    return 1\n",
        )
        self.assertIn("replace_file", result["output"]["instruction"])
        empty_old = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {"path": "app.py", "old": "", "new": "return 2"},
            }
        )
        self.assertEqual(empty_old["status"], "error")
        self.assertEqual(
            empty_old["output"]["current_content"],
            "def value():\n    return 1\n",
        )

    def test_replace_file_is_scoped_atomic_and_syntax_checked(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.source / "locked.py").write_text("LOCKED = True\n", encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py",)
        )
        invalid = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "app.py", "content": "def broken(:\n"},
            }
        )
        self.assertEqual(invalid["status"], "error")
        self.assertEqual((workspace.repo / "app.py").read_text(), "VALUE = 1\n")
        protected = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "locked.py", "content": "LOCKED = False\n"},
            }
        )
        self.assertEqual(protected["status"], "denied")
        replaced = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "app.py", "content": "VALUE = 2\n"},
            }
        )
        self.assertEqual(replaced["status"], "ok")
        self.assertEqual(replaced["output"]["current_content"], "VALUE = 2\n")
        self.assertEqual((workspace.repo / "app.py").read_text(), "VALUE = 2\n")

    def test_failed_command_and_output_remain_visible(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        checks = (
            CheckSpec(
                name="acceptance",
                command=(
                    sys.executable,
                    "-c",
                    "import sys; print('EXPECTED FAILURE'); sys.exit(7)",
                ),
                cwd="repo",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(checks=checks)
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=ScriptedModel(['{"final":"all tests passed"}']),
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertFalse(report["verification_passed"])
        self.assertEqual(report["checks"][0]["exit_code"], 7)
        self.assertIn("EXPECTED FAILURE", report["checks"][0]["output"])
        self.assertEqual(report["status"], "incomplete")

    def test_unavailable_check_remains_visible(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        checks = (
            CheckSpec(
                name="acceptance",
                command=("/definitely/missing/harness-command",),
                cwd="repo",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(checks=checks)
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=ScriptedModel(['{"final":"done"}']),
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["checks"][0]["status"], "unavailable")
        self.assertFalse(report["verification_passed"])

    def test_invalid_json_returns_observation_and_recovers(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        checks = (
            CheckSpec(
                name="acceptance",
                command=(sys.executable, "-c", "print('ok')"),
                cwd="repo",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(checks=checks)
        model = ScriptedModel(["not-json", '{"final":"done"}'])
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["counts"]["invalid_model_replies"], 1)
        self.assertIn("Invalid model JSON", model.calls[1][-1]["content"])
        self.assertEqual(report["status"], "complete")

    def test_failed_final_verification_returns_to_model_for_repair(self):
        (self.source / "logic.py").write_text(
            "def answer():\n    return 41\n", encoding="utf-8"
        )
        (self.acceptance / "check.py").write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.environ['HARNESS_REPO'])\n"
            "from logic import answer\n"
            "assert answer() == 42\n",
            encoding="utf-8",
        )
        checks = (
            CheckSpec(
                name="acceptance",
                command=(sys.executable, "check.py"),
                cwd="acceptance",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=checks, writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                '{"final":"done"}',
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "return 41",
                            "new": "return 42",
                        },
                    }
                ),
                '{"final":"fixed"}',
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["verification_attempts"]), 2)
        self.assertFalse(report["verification_attempts"][0]["passed"])
        self.assertTrue(report["verification_attempts"][1]["passed"])
        self.assertIn("verification_failed", model.calls[1][-1]["content"])

    def test_failed_verification_includes_current_changed_file(self):
        (self.source / "logic.py").write_text(
            "def answer():\n    return 40\n", encoding="utf-8"
        )
        (self.acceptance / "check.py").write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.environ['HARNESS_REPO'])\n"
            "from logic import answer\n"
            "assert answer() == 42\n",
            encoding="utf-8",
        )
        checks = (
            CheckSpec(
                name="acceptance",
                command=(sys.executable, "check.py"),
                cwd="acceptance",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=checks, writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "return 40",
                            "new": "return 41",
                        },
                    }
                ),
                '{"final":"done"}',
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "return 41",
                            "new": "return 42",
                        },
                    }
                ),
                '{"final":"fixed"}',
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        feedback = json.loads(model.calls[2][-1]["content"])
        self.assertEqual(
            feedback["current_changed_files"]["logic.py"]["content"],
            "def answer():\n    return 41\n",
        )
        self.assertEqual(report["status"], "complete")

    def test_edit_mismatch_limits_next_schema_to_full_replacement(self):
        (self.source / "logic.py").write_text(
            "def answer():\n    return 41\n", encoding="utf-8"
        )
        check = CheckSpec(
            name="acceptance",
            command=(
                sys.executable,
                "-c",
                "from logic import answer; assert answer() == 42",
            ),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "return 99",
                            "new": "return 42",
                        },
                    }
                ),
                json.dumps(
                    {"tool": "read_file", "args": {"path": "logic.py"}}
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "logic.py",
                            "content": "def answer():\n    return 42\n",
                        },
                    }
                ),
                '{"final":"fixed"}',
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        second_schema = model.schemas[1]
        self.assertEqual(
            second_schema["properties"]["tool"]["enum"],
            ["replace_file"],
        )
        self.assertEqual(
            second_schema["properties"]["args"]["properties"]["path"]["enum"],
            ["logic.py"],
        )
        self.assertEqual(report["counts"]["denied"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            (workspace.repo / "logic.py").read_text(),
            "def answer():\n    return 42\n",
        )

    def test_all_visible_checks_trigger_independent_completion(self):
        (self.source / "logic.py").write_text(
            "def answer():\n    return 42\n", encoding="utf-8"
        )
        visible = CheckSpec(
            name="regression",
            command=(sys.executable, "-c", "from logic import answer; assert answer() == 42"),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        hidden = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "from logic import answer; assert answer() == 42"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(visible, hidden), writable=("logic.py",)
        )
        model = ScriptedModel(
            [json.dumps({"tool": "run_check", "args": {"name": "regression"}})]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["termination"], "checks_passed")
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["verification_passed"])
        self.assertEqual(report["counts"]["model_turns"], 1)

    def test_action_limit_stops_repeating_model(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=3,
            max_retries=10,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        checks = (
            CheckSpec(
                name="acceptance",
                command=(sys.executable, "-c", "print('ok')"),
                cwd="repo",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=checks, limits=limits
        )
        repeated = json.dumps({"tool": "list_files", "args": {}})
        model = ScriptedModel([repeated, repeated, repeated, repeated])
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["termination"], "action_limit")
        self.assertEqual(report["counts"]["model_turns"], 3)
        self.assertEqual(report["counts"]["actions"], 3)

    def test_retry_limit_stops_same_action(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=6,
            max_retries=1,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        checks = (
            CheckSpec(
                name="acceptance",
                command=(sys.executable, "-c", "print('ok')"),
                cwd="repo",
                kind="acceptance",
                model_allowed=False,
                timeout_seconds=5,
            ),
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=checks, limits=limits
        )
        repeated = json.dumps({"tool": "list_files", "args": {}})
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=ScriptedModel([repeated, repeated, repeated]),
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["termination"], "retry_limit")
        self.assertEqual(report["counts"]["retries"], 2)

    def test_successful_edit_resets_retry_history(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=8,
            max_retries=1,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        check = CheckSpec(
            name="regression",
            command=(sys.executable, "-c", "print('ok')"),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        _config, _workspace, _runner, dispatcher = self.make_parts(
            checks=(check,), limits=limits
        )
        action = {"tool": "run_check", "args": {"name": "regression"}}
        self.assertEqual(dispatcher.dispatch(action)["status"], "ok")
        self.assertEqual(dispatcher.dispatch(action)["status"], "ok")
        edit = {
            "tool": "edit_file",
            "args": {"path": "app.py", "old": "x = 1", "new": "x = 2"},
        }
        self.assertEqual(dispatcher.dispatch(edit)["status"], "ok")
        self.assertEqual(dispatcher.dispatch(action)["status"], "ok")

    def test_large_command_output_is_bounded_and_marked(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=3,
            max_retries=1,
            max_file_bytes=20_000,
            max_tool_output_bytes=2_000,
            max_command_output_bytes=1_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "print('x' * 10000)"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        _config, _workspace, runner, _dispatcher = self.make_parts(
            checks=(check,), limits=limits
        )
        result = runner.run(check)
        self.assertTrue(result.truncated)
        self.assertIn("output shortened", result.output)
        self.assertLess(len(result.output), 1_200)

    def test_timeout_kills_child_process_before_late_write(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        marker = self.base / "late-child-write.txt"
        child_code = (
            "import time; from pathlib import Path; "
            f"time.sleep(2); Path({str(marker)!r}).write_text('unsafe')"
        )
        parent_code = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            "time.sleep(10)"
        )
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", parent_code),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=1,
        )
        _config, _workspace, runner, _dispatcher = self.make_parts(checks=(check,))
        result = runner.run(check)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.status, "timeout")
        time.sleep(2.2)
        self.assertFalse(marker.exists())

    def test_docker_backend_builds_contained_command(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        check = CheckSpec(
            name="acceptance",
            command=("python3", "check.py"),
            cwd="acceptance",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        workspace = DisposableWorkspace.create(
            self.source,
            self.runs,
            writable=("**",),
            max_file_bytes=self.limits.max_file_bytes,
            max_repo_bytes=self.limits.max_repo_bytes,
            max_files=self.limits.max_files,
        )
        runner = CheckRunner(
            execution=ExecutionConfig(
                backend="docker",
                docker_image="test-image:1",
                memory_mb=256,
                cpu_seconds=5,
                pids=32,
            ),
            limits=self.limits,
            repo=workspace.repo,
            acceptance_dir=self.acceptance,
            control_dir=workspace.run_root / "control",
        )
        fake = CommandResult(
            status="passed",
            command=[],
            cwd="",
            exit_code=0,
            output="ok",
            timed_out=False,
            truncated=False,
            duration_seconds=0.1,
        )
        with patch("coding_harness.executor.shutil.which", return_value="/usr/bin/docker"), patch(
            "coding_harness.executor._bounded_process", return_value=fake
        ) as execute:
            result = runner.run(check)
        self.assertEqual(result.status, "passed")
        docker_command = execute.call_args.args[0]
        self.assertIn("--network=none", docker_command)
        self.assertIn("--read-only", docker_command)
        self.assertIn("--cap-drop=ALL", docker_command)
        self.assertIn("--security-opt=no-new-privileges", docker_command)
        self.assertIn("--pids-limit=32", docker_command)
        self.assertIn(f"{self.acceptance}:/acceptance:ro", docker_command)
        self.assertIn(f"{workspace.repo}:/workspace:rw", docker_command)

    def test_full_bug_fix_keeps_external_acceptance_unchanged(self):
        (self.source / "logic.py").write_text(
            "def price(quantity):\n"
            "    if quantity == 2:\n"
            "        return 3\n"
            "    return quantity * 2\n",
            encoding="utf-8",
        )
        acceptance_path = self.acceptance / "check.py"
        acceptance_path.write_text(
            "import os, sys\n"
            "sys.path.insert(0, os.environ['HARNESS_REPO'])\n"
            "from logic import price\n"
            "assert price(2) == 4\n"
            "print('acceptance passed')\n",
            encoding="utf-8",
        )
        protected_hash = hashlib.sha256(acceptance_path.read_bytes()).hexdigest()
        regression = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                "from logic import price; assert price(3) == 6; print('regression passed')",
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        acceptance = CheckSpec(
            name="acceptance",
            command=(sys.executable, "check.py"),
            cwd="acceptance",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        checks = (regression, acceptance)
        config, workspace, runner, dispatcher = self.make_parts(
            checks=checks, writable=("logic.py",)
        )
        starting = runner.run(acceptance)
        self.assertEqual(starting.status, "failed")

        replies = [
            json.dumps({"tool": "read_file", "args": {"path": "logic.py"}}),
            json.dumps(
                {
                    "tool": "edit_file",
                    "args": {
                        "path": "logic.py",
                        "old": (
                            "def price(quantity):\n"
                            "    if quantity == 2:\n"
                            "        return 3\n"
                            "    return quantity * 2\n"
                        ),
                        "new": "def price(quantity):\n    return quantity * 2\n",
                    },
                }
            ),
            json.dumps({"tool": "run_check", "args": {"name": "regression"}}),
            json.dumps({"final": "Fixed the special-case pricing error."}),
        ]
        model = ScriptedModel(replies)
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["verification_passed"])
        self.assertEqual(report["changed_files"], ["logic.py"])
        self.assertIn("-    if quantity == 2:", report["diff"])
        self.assertEqual(
            hashlib.sha256(acceptance_path.read_bytes()).hexdigest(), protected_hash
        )
        self.assertIn("untrusted tool result", model.calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()

