from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
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
from coding_harness.controller import CodingHarness, _regression_test_scaffold
from coding_harness.executor import CheckRunner, CommandResult
from coding_harness.model import ScriptedModel, _generation_temperature
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
        mode="edit",
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
                mode=mode,
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
        shadow = self.source / "nested" / "app.py"
        shadow.parent.mkdir()
        shadow.write_text("VALUE = 99\n", encoding="utf-8")
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
        suffix_shadow = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "nested/app.py",
                    "old": "VALUE = 99",
                    "new": "VALUE = 2",
                },
            }
        )
        self.assertEqual(protected["status"], "denied")
        self.assertEqual(escaped["status"], "denied")
        self.assertEqual(suffix_shadow["status"], "denied")
        self.assertFalse((workspace.run_root / "escape.txt").exists())
        self.assertEqual(
            (workspace.repo / "nested/app.py").read_text(encoding="utf-8"),
            "VALUE = 99\n",
        )

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

    def test_model_adds_variation_only_after_a_rejected_action(self):
        self.assertEqual(
            _generation_temperature([{"role": "user", "content": "continue"}]),
            0,
        )
        self.assertEqual(
            _generation_temperature(
                [
                    {
                        "role": "user",
                        "content": '{"status": "error", "output": "bad edit"}',
                    }
                ]
            ),
            0.1,
        )

    def test_regression_scaffold_keeps_target_api_and_local_helpers(self):
        acceptance = (
            "import unittest\n"
            "from decimal import Decimal as D\n"
            "from orders.pricing import quote\n"
            "from orders.service import place_order\n\n"
            "def lines(price):\n"
            "    return [{'unit_price': D(price), 'quantity': 1}]\n\n"
            "class Acceptance(unittest.TestCase):\n"
            "    def test_existing(self):\n"
            "        self.assertTrue(quote(lines('1')))\n"
        )
        scaffold = _regression_test_scaffold(
            {
                "service/tests/test_acceptance.py": {
                    "content": acceptance,
                    "truncated": False,
                }
            },
            [
                "service/orders/pricing.py",
                "service/orders/service.py",
                "service/tests/test_acceptance.py",
            ],
            (
                "service/orders/pricing.py",
                "service/tests/test_student.py",
            ),
        )

        self.assertIn("from orders.pricing import quote", scaffold)
        self.assertIn("def lines(price):", scaffold)
        self.assertIn("def test_regression", scaffold)
        self.assertNotIn("orders.service", scaffold)

    def test_regression_scaffold_uses_function_for_pytest_style_tests(self):
        scaffold = _regression_test_scaffold(
            {
                "tests/test_pricing.py": {
                    "content": "from orders.pricing import quote\n\ndef helper():\n    return []\n",
                    "truncated": False,
                }
            },
            ["orders/pricing.py", "tests/test_pricing.py"],
            ("orders/pricing.py", "tests/test_student.py"),
        )

        self.assertIn("from orders.pricing import quote", scaffold)
        self.assertIn("def helper():", scaffold)
        self.assertIn("def test_regression():", scaffold)
        self.assertNotIn("unittest.TestCase", scaffold)

    def test_read_file_schema_lists_only_existing_files(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        (self.source / "notes.txt").write_text("notes\n", encoding="utf-8")
        _config, _workspace, _runner, dispatcher = self.make_parts()
        schema = dispatcher.schema(only_tools={"read_file"})
        read_variant = next(
            variant
            for variant in schema["oneOf"]
            if variant.get("properties", {}).get("tool", {}).get("enum")
            == ["read_file"]
        )
        paths = read_variant["properties"]["args"]["properties"]["path"]["enum"]
        self.assertEqual(paths, ["app.py", "notes.txt"])
        self.assertNotIn("missing.py", paths)

        read = {"tool": "read_file", "args": {"path": "app.py"}}
        for _ in range(4):
            self.assertEqual(dispatcher.dispatch(read)["status"], "ok")
        self.assertEqual(dispatcher.stats.retries, 0)

        unread_schema = dispatcher.schema(
            only_tools={"read_file"}, unread_only=True
        )
        unread_variant = next(
            variant
            for variant in unread_schema["oneOf"]
            if variant.get("properties", {}).get("tool", {}).get("enum")
            == ["read_file"]
        )
        unread_paths = unread_variant["properties"]["args"]["properties"]["path"]["enum"]
        self.assertEqual(unread_paths, ["notes.txt"])

    def test_mutation_schema_separates_existing_and_missing_paths(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        _config, _workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py", "new.py")
        )
        variants = {}
        for variant in dispatcher.schema()["oneOf"]:
            tool_values = variant.get("properties", {}).get("tool", {}).get("enum")
            if tool_values:
                variants[tool_values[0]] = variant
        edit_args = variants["edit_file"]["properties"]["args"]["properties"]
        replace_args = variants["replace_file"]["properties"]["args"]["properties"]
        write_args = variants["write_file"]["properties"]["args"]["properties"]
        self.assertEqual(edit_args["path"]["enum"], ["app.py"])
        self.assertEqual(edit_args["old"]["minLength"], 1)
        self.assertEqual(replace_args["path"]["enum"], ["app.py"])
        self.assertEqual(write_args["path"]["enum"], ["new.py"])

        wrong_write = dispatcher.dispatch(
            {"tool": "write_file", "args": {"path": "app.py", "content": "x = 2\n"}}
        )
        wrong_edit = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {"path": "new.py", "old": "x", "new": "y"},
            }
        )
        self.assertEqual(wrong_write["status"], "denied")
        self.assertEqual(wrong_edit["status"], "denied")

    def test_new_python_and_focused_test_are_validated_before_write(self):
        (self.source / "tests").mkdir()
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("new.py", "tests/test_student.py")
        )

        unresolved_module = dispatcher.dispatch(
            {
                "tool": "write_file",
                "args": {
                    "path": "new.py",
                    "content": "def value():\n    return missing_name\n",
                },
            }
        )
        self.assertEqual(unresolved_module["status"], "error")
        self.assertIn("unresolved global", unresolved_module["output"])
        self.assertFalse((workspace.repo / "new.py").exists())

        unfocused = dispatcher.dispatch(
            {
                "tool": "write_file",
                "args": {
                    "path": "tests/test_student.py",
                    "content": (
                        "def test_one():\n    assert lines('1')\n\n"
                        "def test_two():\n    assert lines('2')\n"
                    ),
                },
            },
            required_test_count=1,
        )
        self.assertEqual(unfocused["status"], "error")
        self.assertIn("found 2", unfocused["output"]["message"])
        self.assertIn("'lines'", unfocused["output"]["message"])
        self.assertFalse((workspace.repo / "tests/test_student.py").exists())

        focused = dispatcher.dispatch(
            {
                "tool": "write_file",
                "args": {
                    "path": "tests/test_student.py",
                    "content": (
                        "def lines(value):\n    return [value]\n\n"
                        "def test_one():\n    assert lines('1') == ['1']\n"
                    ),
                },
            },
            required_test_count=1,
        )
        self.assertEqual(focused["status"], "ok")

    def test_frozen_path_is_removed_from_mutation_schemas_and_denied(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        tests = self.source / "tests"
        tests.mkdir()
        student = tests / "test_student.py"
        student.write_text("def test_value():\n    assert True\n", encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py", "tests/test_student.py")
        )

        self.assertEqual(
            dispatcher.freeze_path("tests/test_student.py"),
            "tests/test_student.py",
        )
        variants = {}
        for variant in dispatcher.schema()["oneOf"]:
            tool_values = variant.get("properties", {}).get("tool", {}).get("enum")
            if tool_values:
                variants[tool_values[0]] = variant
        for tool in ("edit_file", "replace_file"):
            paths = variants[tool]["properties"]["args"]["properties"]["path"][
                "enum"
            ]
            self.assertEqual(paths, ["app.py"])
        self.assertEqual(
            dispatcher.frozen_path_in_text(
                "Change tests/test_student.py instead of implementation"
            ),
            "tests/test_student.py",
        )
        self.assertEqual(
            dispatcher.align_tool_to_explicit_path(
                "edit_file",
                "Change tests/test_student.py instead of implementation",
                only_tools={"edit_file", "replace_file"},
            ),
            ("edit_file", "app.py"),
        )

        denied = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "tests/test_student.py",
                    "old": "assert True",
                    "new": "assert False",
                },
            }
        )
        self.assertEqual(denied["status"], "denied")
        self.assertIn("frozen", denied["output"])
        self.assertIn("assert True", student.read_text(encoding="utf-8"))

        dispatcher.unfreeze_path("tests/test_student.py")
        edited = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "tests/test_student.py",
                    "old": "assert True",
                    "new": "assert 1 == 1",
                },
            }
        )
        self.assertEqual(edited["status"], "ok")
        self.assertIn("assert 1 == 1", (workspace.repo / "tests/test_student.py").read_text())

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
        unchanged = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "app.py", "content": "VALUE = 1\n"},
            }
        )
        self.assertEqual(unchanged["status"], "error")
        self.assertIn("identical", unchanged["output"]["message"])
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

    def test_python_mutations_reject_broken_dependencies_and_embedded_tests(self):
        original = (
            "from helpers import validate\n\n"
            "def answer(value):\n"
            "    validate(value)\n"
            "    return value\n"
        )
        (self.source / "app.py").write_text(original, encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py",)
        )

        undefined = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {
                    "path": "app.py",
                    "content": (
                        "def answer(value):\n"
                        "    validate(value)\n"
                        "    return value\n"
                    ),
                },
            }
        )
        self.assertEqual(undefined["status"], "error")
        self.assertIn("unresolved global", undefined["output"]["message"])
        self.assertEqual(undefined["output"]["current_content"], original)

        shadowed = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {
                    "path": "app.py",
                    "content": (
                        "def validate(value):\n"
                        "    return None\n\n"
                        "def answer(value):\n"
                        "    validate(value)\n"
                        "    return value\n"
                    ),
                },
            }
        )
        self.assertEqual(shadowed["status"], "error")
        self.assertIn("imported binding", shadowed["output"]["message"])

        embedded_test = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "app.py",
                    "old": "    return value\n",
                    "new": (
                        "    return value\n\n"
                        "if __name__ == '__main__':\n"
                        "    assert answer(1) == 1\n"
                    ),
                },
            }
        )
        self.assertEqual(embedded_test["status"], "error")
        self.assertIn("non-test module", embedded_test["output"]["message"])

        unittest_in_source = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {
                    "path": "app.py",
                    "content": (
                        original
                        + "\nimport unittest\n\n"
                        + "class TestAnswer(unittest.TestCase):\n"
                        + "    def test_answer(self):\n"
                        + "        self.assertEqual(answer(1), 1)\n"
                    ),
                },
            }
        )
        self.assertEqual(unittest_in_source["status"], "error")
        self.assertIn("non-test module", unittest_in_source["output"]["message"])
        self.assertEqual((workspace.repo / "app.py").read_text(), original)

    def test_test_file_can_replace_bad_helper_import_with_local_helper(self):
        tests = self.source / "tests"
        tests.mkdir()
        original = (
            "from orders.pricing import quote, lines\n\n"
            "def test_quote():\n"
            "    assert quote(lines('10'))\n"
        )
        (tests / "test_student.py").write_text(original, encoding="utf-8")
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("tests/test_student.py",)
        )

        result = dispatcher.dispatch(
            {
                "tool": "edit_file",
                "args": {
                    "path": "tests/test_student.py",
                    "old": "from orders.pricing import quote, lines\n",
                    "new": (
                        "from orders.pricing import quote\n\n"
                        "def lines(price):\n"
                        "    return [{'unit_price': price, 'quantity': 1}]\n"
                    ),
                },
            }
        )

        self.assertEqual(result["status"], "ok")
        updated = (workspace.repo / "tests/test_student.py").read_text()
        self.assertIn("def lines(price):", updated)
        self.assertNotIn("import quote, lines", updated)

    def test_recovery_rejects_large_copy_from_another_retained_file(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        copied = "TEST_DATA = " + repr("x" * 600) + "\n"
        (self.source / "tests").mkdir()
        (self.source / "tests/test_student.py").write_text(
            copied, encoding="utf-8"
        )
        _config, workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py", "tests/test_student.py")
        )
        dispatcher.dispatch(
            {"tool": "read_file", "args": {"path": "tests/test_student.py"}}
        )

        contaminated = dispatcher.dispatch(
            {
                "tool": "replace_file",
                "args": {"path": "app.py", "content": "VALUE = 2\n" + copied},
            },
            allowed_tools={"replace_file"},
            required_path="app.py",
        )

        self.assertEqual(contaminated["status"], "error")
        self.assertIn("copy", contaminated["output"]["message"])
        self.assertIn("tests/test_student.py", contaminated["output"]["message"])
        self.assertEqual((workspace.repo / "app.py").read_text(), "VALUE = 1\n")

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

    def test_failed_check_feedback_is_compact_but_keeps_head_and_tail(self):
        raw_output = (
            "FAILURE SUMMARY\n"
            + ("traceback line\n" * 500)
            + "FINAL ASSERTION\n"
        )
        raw = {
            "status": "error",
            "output": {
                "status": "failed",
                "command": ["docker", "run", "image", "python3", "check.py"],
                "cwd": "/temporary/control",
                "exit_code": 1,
                "output": raw_output,
                "timed_out": False,
            },
            "fatal": False,
        }

        compact = CodingHarness._history_result("run_check", raw)

        feedback = compact["output"]
        self.assertEqual(feedback["command"], ["python3", "check.py"])
        self.assertNotIn("cwd", feedback)
        self.assertIn("FAILURE SUMMARY", feedback["output"])
        self.assertIn("FINAL ASSERTION", feedback["output"])
        self.assertIn("shortened for model", feedback["output"])
        self.assertEqual(raw["output"]["output"], raw_output)

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
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()
        self.assertEqual(report["counts"]["invalid_model_replies"], 1)
        self.assertIn("Invalid model JSON", model.calls[1][-1]["content"])
        invalid = next(
            event for event in events if event.get("status") == "invalid_json"
        )
        self.assertEqual(invalid["response_characters"], len("not-json"))
        self.assertEqual(invalid["response_ending"], "not-json")
        self.assertEqual(report["status"], "complete")

    def test_read_only_mode_hides_and_denies_mutating_tools_and_checks(self):
        (self.source / "app.py").write_text("x = 1\n", encoding="utf-8")
        marker = self.base / "check-ran.txt"
        check = CheckSpec(
            name="quality",
            command=(
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("app.py",), mode="read-only"
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {"path": "new.py", "content": "unsafe = True\n"},
                    }
                ),
                json.dumps({"tool": "run_check", "args": {"name": "quality"}}),
                json.dumps({"final": "The file currently defines x as 1."}),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        tool_names = {
            variant["properties"]["tool"]["enum"][0]
            for variant in model.schemas[0]["oneOf"]
            if "tool" in variant.get("properties", {})
        }
        self.assertEqual(tool_names, {"list_files", "read_file", "search_files"})
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["mode"], "read-only")
        self.assertEqual(report["counts"]["denied"], 2)
        self.assertEqual(report["changed_files"], [])
        self.assertEqual(report["checks"], [])
        self.assertFalse((workspace.repo / "new.py").exists())
        self.assertFalse(marker.exists())
        self.assertIn("enforced read-only", model.calls[0][0]["content"])

    def test_edit_mode_closes_inspection_tools_at_configured_limit(self):
        for name in ("a.py", "b.py", "c.py"):
            (self.source / name).write_text(f"NAME = {name!r}\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=5,
            max_inspection_actions=2,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "print('ok')"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("a.py", "new.py"), limits=limits
        )
        model = ScriptedModel(
            [
                json.dumps({"tool": "read_file", "args": {"path": "a.py"}}),
                json.dumps({"tool": "read_file", "args": {"path": "b.py"}}),
                json.dumps({"tool": "read_file", "args": {"path": "c.py"}}),
                json.dumps({"final": "Inspection complete."}),
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()

        third_turn_tools = {
            variant["properties"]["tool"]["enum"][0]
            for variant in model.schemas[2]["oneOf"]
            if "tool" in variant.get("properties", {})
        }
        first_turn_tools = {
            variant["properties"]["tool"]["enum"][0]
            for variant in model.schemas[0]["oneOf"]
            if "tool" in variant.get("properties", {})
        }
        self.assertNotIn("write_file", first_turn_tools)
        self.assertIn("write_file", third_turn_tools)
        self.assertNotIn("read_file", third_turn_tools)
        self.assertNotIn("search_files", third_turn_tools)
        self.assertEqual(report["counts"]["denied"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertNotIn("c.py", dispatcher.read_only_progress()["files_read"])
        self.assertTrue(
            any(
                "inspection phase is complete" in message["content"].lower()
                for message in model.calls[2]
            )
        )

    def test_persistent_file_evidence_is_not_duplicated_in_history(self):
        unique = "UNIQUE_RETAINED_FILE_BODY = 734951\n"
        (self.source / "app.py").write_text(unique, encoding="utf-8")
        config, workspace, runner, dispatcher = self.make_parts(mode="read-only")
        model = ScriptedModel(
            [
                json.dumps({"tool": "read_file", "args": {"path": "app.py"}}),
                json.dumps({"final": "The file defines one retained value."}),
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()

        second_request = "\n".join(
            message["content"] for message in model.calls[1]
        )
        self.assertEqual(second_request.count(unique.strip()), 1)
        self.assertIn(
            "content_retained_in_current_file_evidence", second_request
        )
        self.assertEqual(report["status"], "complete")

    def test_successful_edit_arguments_remain_exact_in_history_and_trace(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "print('ok')"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("app.py",)
        )
        edit = {
            "tool": "edit_file",
            "args": {"path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"},
        }
        model = ScriptedModel(
            [json.dumps(edit), json.dumps({"final": "Updated the value."})]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        self.assertIn('"old": "VALUE = 1"', model.calls[1][2]["content"])
        tool_start = next(event for event in events if event["event"] == "tool_start")
        self.assertEqual(tool_start["args"], edit["args"])
        self.assertEqual(report["status"], "complete")

    def test_edit_phase_selects_tool_then_requires_its_exact_arguments(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=5,
            max_inspection_actions=1,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "print('ok')"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("app.py",), limits=limits
        )
        model = ScriptedModel(
            [
                json.dumps({"tool": "read_file", "args": {"path": "app.py"}}),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "intent": "Change the configured value from one to two.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "app.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
                json.dumps({"final": "Updated the value."}),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        selector_tools = {
            variant["properties"]["tool"]["enum"][0]
            for variant in model.schemas[1]["oneOf"]
            if "tool" in variant.get("properties", {})
        }
        self.assertIn("edit_file", selector_tools)
        edit_choice = next(
            variant
            for variant in model.schemas[1]["oneOf"]
            if variant.get("properties", {}).get("tool", {}).get("enum")
            == ["edit_file"]
        )
        self.assertIn("app.py", edit_choice["description"])
        strict_args = model.schemas[2]["properties"]["args"]
        self.assertEqual(strict_args["required"], ["path", "old", "new"])
        self.assertTrue(
            any(event["event"] == "model_arguments_start" for event in events)
        )
        edit_request = "\n".join(message["content"] for message in model.calls[1])
        self.assertNotIn('"tool": "read_file"', edit_request)
        self.assertEqual(edit_request.count("VALUE = 1"), 1)
        self.assertEqual(report["counts"]["model_turns"], 3)
        self.assertEqual(report["status"], "complete")

    def test_selection_aligns_tool_to_one_explicit_available_path(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        limits = LimitsConfig(
            max_actions=5,
            max_inspection_actions=1,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        check = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "print('ok')"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("app.py", "project/tests/new.py"), limits=limits
        )
        model = ScriptedModel(
            [
                json.dumps({"tool": "read_file", "args": {"path": "app.py"}}),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "intent": "Create the requested regression in tests/new.py.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {
                            "path": "project/tests/new.py",
                            "content": "CHECK = True\n",
                        },
                    }
                ),
                json.dumps({"final": "Created the requested file."}),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        self.assertEqual(
            model.schemas[2]["properties"]["tool"]["enum"], ["write_file"]
        )
        aligned = next(
            event
            for event in events
            if event["event"] == "model_selection_aligned"
        )
        self.assertEqual(aligned["requested_tool"], "edit_file")
        self.assertEqual(aligned["selected_tool"], "write_file")
        self.assertEqual(aligned["path"], "project/tests/new.py")
        self.assertEqual(
            (workspace.repo / "project/tests/new.py").read_text(), "CHECK = True\n"
        )
        self.assertEqual(report["status"], "complete")

    def test_selection_targets_one_explicit_existing_path(self):
        (self.source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.source / "tests").mkdir()
        (self.source / "tests/test_student.py").write_text(
            "CHECK = False\n", encoding="utf-8"
        )
        _config, _workspace, _runner, dispatcher = self.make_parts(
            writable=("app.py", "tests/test_student.py")
        )

        tool, path = dispatcher.align_tool_to_explicit_path(
            "edit_file",
            "Fix the regression in tests/test_student.py.",
            only_tools={"edit_file", "replace_file"},
        )
        schema = dispatcher.tool_schema(tool, required_path=path)

        self.assertEqual(tool, "edit_file")
        self.assertEqual(path, "tests/test_student.py")
        self.assertEqual(
            schema["properties"]["args"]["properties"]["path"]["enum"],
            ["tests/test_student.py"],
        )

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
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
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
        feedback = next(
            json.loads(message["content"])
            for message in model.calls[2]
            if '"status": "verification_failed"' in message["content"]
        )
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
                        "intent": (
                            "Replace logic.py with the complete corrected function, "
                            "preserving its public API and changing the returned value "
                            "so the retained acceptance requirement is satisfied."
                        ),
                    }
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
        self.assertEqual(len(second_schema["oneOf"]), 1)
        self.assertEqual(
            second_schema["oneOf"][0]["properties"]["tool"]["enum"],
            ["replace_file"],
        )
        self.assertGreaterEqual(
            second_schema["oneOf"][0]["properties"]["intent"]["minLength"],
            80,
        )
        argument_schema = model.schemas[3]
        self.assertEqual(
            argument_schema["properties"]["args"]["properties"]["path"]["enum"],
            ["logic.py"],
        )
        recovery_request = "\n".join(
            message["content"] for message in model.calls[1]
        )
        self.assertIn('"failed_edit"', recovery_request)
        self.assertIn('"old": "return 99"', recovery_request)
        self.assertIn('"new": "return 42"', recovery_request)
        self.assertIn(
            "keep content for other paths out", recovery_request.lower()
        )
        self.assertEqual(report["counts"]["denied"], 1)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            (workspace.repo / "logic.py").read_text(),
            "def answer():\n    return 42\n",
        )

    def test_replace_validation_error_enters_bounded_recovery(self):
        original = (
            "from helpers import validate\n\n"
            "def answer(value):\n"
            "    validate(value)\n"
            "    return value\n"
        )
        (self.source / "logic.py").write_text(original, encoding="utf-8")
        (self.source / "helpers.py").write_text(
            "def validate(value):\n    return None\n", encoding="utf-8"
        )
        acceptance = CheckSpec(
            name="acceptance",
            command=(
                sys.executable,
                "-c",
                "from logic import answer; assert answer(1) == 2",
            ),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(acceptance,), writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "logic.py",
                            "content": (
                                "def answer(value):\n"
                                "    validate(value)\n"
                                "    return value + 1\n"
                            ),
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "intent": (
                            "Restore the imported validation dependency and return a "
                            "complete corrected logic.py while preserving its public "
                            "answer API and all unrelated module structure."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "logic.py",
                            "content": original.replace(
                                "    return value\n", "    return value + 1\n"
                            ),
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

        recovery_schema = model.schemas[1]
        self.assertEqual(len(recovery_schema["oneOf"]), 1)
        self.assertEqual(
            recovery_schema["oneOf"][0]["properties"]["tool"]["enum"],
            ["replace_file"],
        )
        recovery_request = "\n".join(
            message["content"] for message in model.calls[1]
        )
        self.assertIn("validation_error", recovery_request)
        self.assertIn("unresolved global", recovery_request)
        self.assertEqual(report["status"], "complete")
        self.assertIn("return value + 1", (workspace.repo / "logic.py").read_text())

    def test_failed_recovery_does_not_replay_generated_file_content(self):
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
        rejected_marker = "REJECTED_RECOVERY_" + ("x" * 1_000)
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
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "logic.py",
                            "content": f"def broken(:\n# {rejected_marker}\n",
                        },
                    }
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

        retry_request = "\n".join(
            message["content"] for message in model.calls[2]
        )
        self.assertNotIn(rejected_marker, retry_request)
        self.assertIn('"failed_edit"', retry_request)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            (workspace.repo / "logic.py").read_text(),
            "def answer():\n    return 42\n",
        )

    def test_failed_check_evidence_survives_edit_recovery(self):
        (self.source / "logic.py").write_text(
            "def answer():\n    return 41\n", encoding="utf-8"
        )
        check = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                "from logic import answer; assert answer() == 42, 'NEEDS_42'",
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(check,), writable=("logic.py",)
        )
        rejected_marker = "SHOULD_NOT_BE_REPLAYED_" + ("x" * 500)
        model = ScriptedModel(
            [
                json.dumps({"tool": "run_check", "args": {"name": "regression"}}),
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
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "logic.py",
                            "content": f"def broken(:\n# {rejected_marker}\n",
                        },
                    }
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
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        recovery_retry = "\n".join(
            message["content"] for message in model.calls[3]
        )
        self.assertIn("unresolved_check_failure", recovery_retry)
        self.assertIn("NEEDS_42", recovery_retry)
        self.assertNotIn(rejected_marker, recovery_retry)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["termination"], "checks_passed")
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(report["counts"]["actions"], 5)
        self.assertTrue(
            any(
                event.get("event") == "tool_start"
                and event.get("tool") == "run_check"
                and event.get("automatic") is True
                for event in events
            )
        )

    def test_successful_mutation_automatically_runs_lint_check(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        lint = CheckSpec(
            name="quality",
            command=(
                sys.executable,
                "-c",
                "import logic; assert logic.VALUE == 3, 'VALUE_MUST_BE_3'",
            ),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(lint,), writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 2",
                            "new": "VALUE = 3",
                        },
                    }
                ),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        second_request = "\n".join(
            message["content"] for message in model.calls[1]
        )
        self.assertIn("unresolved_check_failure", second_request)
        self.assertIn("VALUE_MUST_BE_3", second_request)
        self.assertEqual(report["termination"], "checks_passed")
        self.assertEqual(report["status"], "complete")
        self.assertEqual((workspace.repo / "logic.py").read_text(), "VALUE = 3\n")
        lint_starts = [
            event
            for event in events
            if event.get("event") == "tool_start"
            and event.get("tool") == "run_check"
            and event.get("automatic") is True
        ]
        self.assertEqual(len(lint_starts), 2)
        self.assertEqual(
            lint_starts[0]["automatic_reason"], "post_mutation_lint"
        )
        self.assertEqual(
            lint_starts[1]["automatic_reason"], "refresh_unresolved_check"
        )

    def test_successful_mutation_cascades_from_lint_to_regression(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        quality = CheckSpec(
            name="quality",
            command=(sys.executable, "-c", "import logic"),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        regression = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                "import logic; assert logic.VALUE == 2",
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(regression, quality), writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                )
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        automatic_starts = [
            event
            for event in events
            if event.get("event") == "tool_start"
            and event.get("tool") == "run_check"
            and event.get("automatic") is True
        ]
        self.assertEqual(
            [(event["location"], event["automatic_reason"]) for event in automatic_starts],
            [
                ("quality", "post_mutation_lint"),
                ("regression", "post_lint_required_check"),
            ],
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["termination"], "checks_passed")

    def test_multiple_check_failures_align_partial_edit_to_full_replacement(self):
        (self.source / "logic.py").write_text("VALUE = 0\n", encoding="utf-8")
        regression = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                (
                    "import unittest, logic\n"
                    "class ValueTests(unittest.TestCase):\n"
                    "    def test_one(self):\n"
                    "        self.assertEqual(logic.VALUE, 2)\n"
                    "    def test_two(self):\n"
                    "        self.assertGreater(logic.VALUE, 1)\n"
                    "unittest.main()\n"
                ),
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=4,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(regression,),
            writable=("logic.py",),
            limits=limits,
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 0",
                            "new": "VALUE = 1",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "failure_plan": {
                            "case_1": "Set VALUE to two for test_one.",
                            "case_2": "Set VALUE above one for test_two.",
                        },
                        "implementation_plan": (
                            "Replace logic.py coherently so VALUE is two and both "
                            "retained regression failures are repaired together."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {"path": "logic.py", "content": "VALUE = 2\n"},
                    }
                ),
            ]
        )
        events = []

        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        argument_schema = model.schemas[2]
        self.assertEqual(
            argument_schema["properties"]["tool"]["enum"], ["replace_file"]
        )
        content_description = argument_schema["properties"]["args"]["properties"][
            "content"
        ]["description"]
        self.assertIn("test_one", content_description)
        self.assertIn("do not repeat current content", content_description)
        self.assertTrue(
            any(
                event.get("event") == "model_selection_aligned"
                and event.get("requested_tool") == "edit_file"
                and event.get("selected_tool") == "replace_file"
                for event in events
            )
        )
        self.assertEqual((workspace.repo / "logic.py").read_text(), "VALUE = 2\n")
        self.assertEqual(report["status"], "complete")

    def test_same_failing_cases_force_full_file_no_progress_recovery(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        regression = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                (
                    "import unittest, logic\n"
                    "class ValueTest(unittest.TestCase):\n"
                    "    def test_value(self):\n"
                    "        self.assertEqual(logic.VALUE, 3)\n"
                    "unittest.main()\n"
                ),
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(regression,), writable=("logic.py",)
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 2",
                            "new": "VALUE = 4",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "failure_plan": {
                            "case_1": (
                                "Set VALUE to the expected value from the retained "
                                "test_value assertion instead of another guess."
                            )
                        },
                        "implementation_plan": (
                            "Replace the complete module with the smallest coherent "
                            "implementation that assigns VALUE to the expected value "
                            "three, preserves the public name, and removes both failed "
                            "intermediate guesses from the current file."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {"path": "logic.py", "content": "VALUE = 3\n"},
                    }
                ),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        recovery_schema = model.schemas[2]
        self.assertEqual(len(recovery_schema["oneOf"]), 1)
        self.assertEqual(
            recovery_schema["oneOf"][0]["properties"]["tool"]["enum"],
            ["replace_file"],
        )
        self.assertIn(
            "case_1",
            recovery_schema["oneOf"][0]["properties"]["failure_plan"][
                "required"
            ],
        )
        replacement_schema = model.schemas[3]
        self.assertEqual(
            replacement_schema["properties"]["args"]["properties"]["path"]["enum"],
            ["logic.py"],
        )
        self.assertTrue(
            any(event.get("event") == "no_progress_recovery" for event in events)
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual((workspace.repo / "logic.py").read_text(), "VALUE = 3\n")

    def test_quality_approved_changed_test_is_frozen_before_source_repair(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        tests = self.source / "tests"
        tests.mkdir()
        (tests / "test_student.py").write_text(
            "def test_value():\n    assert True\n", encoding="utf-8"
        )
        quality = CheckSpec(
            name="quality",
            command=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('tests/test_student.py'); "
                    "compile(p.read_text(), str(p), 'exec')"
                ),
            ),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        acceptance = CheckSpec(
            name="acceptance",
            command=(
                sys.executable,
                "-c",
                "import logic; assert logic.VALUE == 2",
            ),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(quality, acceptance),
            writable=("logic.py", "tests/test_student.py"),
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "old": "assert True",
                            "new": "assert 1 == 1",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        self.assertEqual(report["status"], "complete")
        self.assertEqual(dispatcher.frozen_paths(), ["tests/test_student.py"])
        second_schema = model.schemas[1]
        mutation_paths = []
        for variant in second_schema["oneOf"]:
            tool = variant.get("properties", {}).get("tool", {}).get("enum", [None])[0]
            if tool not in {"edit_file", "replace_file"}:
                continue
            mutation_paths.extend(
                variant["properties"]["args"]["properties"]["path"]["enum"]
            )
        self.assertNotIn("tests/test_student.py", mutation_paths)
        self.assertIn("logic.py", mutation_paths)
        self.assertTrue(
            any(event.get("event") == "path_frozen" for event in events)
        )

    def test_repair_plan_prioritizes_non_generated_failures(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        tests = self.source / "tests"
        tests.mkdir()
        (tests / "test_student.py").write_text(
            "def test_generated():\n    assert False\n", encoding="utf-8"
        )
        output = (
            "FAIL: test_protected (test_acceptance.Protected)\n"
            "FAIL: test_generated (test_student.Generated)\n"
            "FAILED (failures=2)\n"
        )
        regression = CheckSpec(
            name="regression",
            command=(
                sys.executable,
                "-c",
                f"import sys; print({output!r}); sys.exit(1)",
            ),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=2,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(regression,),
            writable=("logic.py", "tests/test_student.py"),
            limits=limits,
        )
        dispatcher.freeze_path("tests/test_student.py")
        model = ScriptedModel(
            [
                json.dumps({"tool": "run_check", "args": {"name": "regression"}}),
                '{"final":"stop"}',
            ]
        )
        CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()

        edit_choice = next(
            choice
            for choice in model.schemas[1]["oneOf"]
            if choice.get("properties", {}).get("tool", {}).get("enum")
            == ["edit_file"]
        )
        case_properties = edit_choice["properties"]["failure_plan"][
            "properties"
        ]
        self.assertEqual(list(case_properties), ["case_1"])
        self.assertIn("test_protected", case_properties["case_1"]["description"])
        self.assertNotIn("test_generated", case_properties["case_1"]["description"])
        self.assertEqual(dispatcher.frozen_paths(), ["tests/test_student.py"])

    def test_generated_test_unfreezes_when_it_is_only_remaining_failure(self):
        tests = self.source / "tests"
        tests.mkdir()
        (tests / "test_student.py").write_text(
            "def test_generated():\n    assert False\n", encoding="utf-8"
        )
        regression = CheckSpec(
            name="regression",
            command=(sys.executable, "-c", "pass"),
            cwd="repo",
            kind="regression",
            model_allowed=True,
            timeout_seconds=5,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(regression,), writable=("tests/test_student.py",)
        )
        dispatcher.freeze_path("tests/test_student.py")
        events = []
        harness = CodingHarness(
            config=config,
            workspace=workspace,
            model=ScriptedModel([]),
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        )
        harness._update_frozen_test_paths(
            "regression",
            {
                "status": "error",
                "output": {
                    "output": (
                        "FAIL: test_generated (test_student.Generated)\n"
                        "FAILED (failures=1)\n"
                    )
                },
            },
            turn=3,
        )

        self.assertEqual(dispatcher.frozen_paths(), [])
        self.assertTrue(
            any(
                event.get("event") == "path_unfrozen"
                and event.get("reason")
                == "only remaining failures are in a generated test"
                for event in events
            )
        )

    def test_frozen_test_intent_is_redirected_to_only_implementation_path(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        tests = self.source / "tests"
        tests.mkdir()
        (tests / "test_student.py").write_text(
            "def test_value():\n    assert True\n", encoding="utf-8"
        )
        quality = CheckSpec(
            name="quality",
            command=(sys.executable, "-c", "pass"),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=4,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(quality,),
            writable=("logic.py", "tests/test_student.py"),
            limits=limits,
        )
        dispatcher.freeze_path("tests/test_student.py")
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "intent": "Fix assertion failures and add new regression tests in logic.py",
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        self.assertEqual(report["status"], "complete")
        argument_prompt = "\n".join(
            message["content"] for message in model.calls[1]
        )
        self.assertIn("Repair the remaining task-relevant implementation", argument_prompt)
        self.assertIn("logic.py", argument_prompt)
        self.assertTrue(
            any(
                event.get("event") == "model_selection_redirected"
                and event.get("path") == "logic.py"
                for event in events
            )
        )

    def test_one_focused_test_task_caps_new_test_file_schema(self):
        (self.source / "tests").mkdir()
        acceptance = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "pass"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=3,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(acceptance,),
            writable=("tests/test_student.py",),
            limits=limits,
            task="Add one useful regression test in tests/test_student.py.",
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "write_file",
                        "intent": "Create one focused regression test in tests/test_student.py.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "content": "def test_regression():\n    assert True\n",
                        },
                    }
                ),
                '{"final":"done"}',
            ]
        )
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()

        content_schema = model.schemas[1]["properties"]["args"]["properties"][
            "content"
        ]
        self.assertEqual(content_schema["maxLength"], 4_000)
        self.assertIn("exactly one", content_schema["description"])
        self.assertIn("def test_regression", content_schema["description"])
        self.assertEqual(report["status"], "complete")

    def test_rejected_new_file_content_is_not_replayed_on_retry(self):
        (self.source / "tests").mkdir()
        acceptance = CheckSpec(
            name="acceptance",
            command=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "p=Path('tests/test_student.py'); "
                    "compile(p.read_text(), str(p), 'exec')"
                ),
            ),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=4,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(acceptance,),
            writable=("tests/test_student.py",),
            limits=limits,
            task="Add one useful regression test in tests/test_student.py.",
        )
        malformed = "def test_regression(:\n    assert True\n"
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "write_file",
                        "intent": "Create the requested focused test file.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "content": malformed,
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "intent": "Retry with one complete syntax-valid test.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "content": "def test_regression():\n    assert True\n",
                        },
                    }
                ),
                '{"final":"done"}',
            ]
        )

        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
        ).run()

        retry_context = "\n".join(
            message["content"] for message in model.calls[2]
        )
        self.assertNotIn(malformed, retry_context)
        self.assertIn("Python syntax error", retry_context)
        self.assertEqual(report["status"], "complete")

    def test_combined_repair_and_test_exposes_write_after_source_edit(self):
        (self.source / "logic.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.source / "tests").mkdir()
        quality = CheckSpec(
            name="quality",
            command=(
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "assert Path('tests/test_student.py').is_file()"
                ),
            ),
            cwd="repo",
            kind="lint",
            model_allowed=True,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=6,
            max_inspection_actions=1,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(quality,),
            writable=("logic.py", "tests/test_student.py"),
            limits=limits,
            task="Repair logic.py and add one focused regression test.",
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {"tool": "read_file", "args": {"path": "logic.py"}}
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "intent": "Repair the existing implementation in logic.py first.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "edit_file",
                        "args": {
                            "path": "logic.py",
                            "old": "VALUE = 1",
                            "new": "VALUE = 2",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "intent": "Add one focused regression test in tests/test_student.py.",
                    }
                ),
                json.dumps(
                    {
                        "tool": "write_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "content": "def test_value():\n    assert True\n",
                        },
                    }
                ),
                '{"final":"done"}',
            ]
        )

        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        inspection_tools = {
            choice.get("properties", {}).get("tool", {}).get("enum", [None])[0]
            for choice in model.schemas[0]["oneOf"]
            if "tool" in choice.get("properties", {})
        }
        before_source_tools = {
            choice.get("properties", {}).get("tool", {}).get("enum", [None])[0]
            for choice in model.schemas[1]["oneOf"]
            if "tool" in choice.get("properties", {})
        }
        after_source_tools = {
            choice.get("properties", {}).get("tool", {}).get("enum", [None])[0]
            for choice in model.schemas[3]["oneOf"]
            if "tool" in choice.get("properties", {})
        }
        self.assertTrue(
            inspection_tools.issubset(
                {"list_files", "read_file", "search_files"}
            )
        )
        self.assertNotIn("write_file", before_source_tools)
        self.assertIn("edit_file", before_source_tools)
        self.assertIn("write_file", after_source_tools)
        automatic_checks = [
            event
            for event in events
            if event.get("event") == "tool_start"
            and event.get("tool") == "run_check"
            and event.get("automatic") is True
        ]
        self.assertEqual(len(automatic_checks), 1)
        self.assertEqual(automatic_checks[0]["turn"], 3)
        self.assertEqual(report["status"], "complete")

    def test_bulk_edit_selection_is_aligned_to_bounded_replacement(self):
        tests = self.source / "tests"
        tests.mkdir()
        (tests / "test_student.py").write_text(
            "def test_one():\n    assert True\n\ndef test_two():\n    assert True\n",
            encoding="utf-8",
        )
        acceptance = CheckSpec(
            name="acceptance",
            command=(sys.executable, "-c", "pass"),
            cwd="repo",
            kind="acceptance",
            model_allowed=False,
            timeout_seconds=5,
        )
        limits = LimitsConfig(
            max_actions=3,
            max_inspection_actions=0,
            max_retries=2,
            max_file_bytes=20_000,
            max_tool_output_bytes=4_000,
            max_command_output_bytes=4_000,
            max_repo_bytes=1_000_000,
            max_files=200,
        )
        config, workspace, runner, dispatcher = self.make_parts(
            checks=(acceptance,),
            writable=("tests/test_student.py",),
            limits=limits,
            task="Keep one useful regression test in tests/test_student.py.",
        )
        model = ScriptedModel(
            [
                json.dumps(
                    {
                        "tool": "edit_file",
                        "intent": (
                            "Remove all regression tests except one in "
                            "tests/test_student.py."
                        ),
                    }
                ),
                json.dumps(
                    {
                        "tool": "replace_file",
                        "args": {
                            "path": "tests/test_student.py",
                            "content": "def test_one():\n    assert True\n",
                        },
                    }
                ),
                '{"final":"done"}',
            ]
        )
        events = []
        report = CodingHarness(
            config=config,
            workspace=workspace,
            model=model,
            dispatcher=dispatcher,
            runner=runner,
            emit=events.append,
        ).run()

        argument_schema = model.schemas[1]
        self.assertEqual(
            argument_schema["properties"]["tool"]["enum"], ["replace_file"]
        )
        self.assertEqual(
            argument_schema["properties"]["args"]["properties"]["content"][
                "maxLength"
            ],
            4_000,
        )
        self.assertTrue(
            any(
                event.get("event") == "model_selection_aligned"
                and event.get("requested_tool") == "edit_file"
                and event.get("selected_tool") == "replace_file"
                for event in events
            )
        )
        self.assertEqual(report["status"], "complete")

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
        repeated = json.dumps(
            {"tool": "read_file", "args": {"path": "missing.py"}}
        )
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
        self.assertTrue(
            any(
                "untrusted tool result" in message["content"]
                for message in model.calls[1]
            )
        )
        self.assertIn("current_file_evidence", model.calls[1][-1]["content"])

    def test_target_service_quality_rejects_cross_file_and_unfocused_tests(self):
        repo = self.base / "quality-repo"
        orders = repo / "target-service" / "orders"
        tests = repo / "target-service" / "tests"
        orders.mkdir(parents=True)
        tests.mkdir(parents=True)
        quality = (
            Path(__file__).resolve().parents[1]
            / "demo"
            / "target_service_acceptance"
            / "quality_check.py"
        )

        pricing = orders / "pricing.py"
        student = tests / "test_student.py"
        pricing.write_text(
            "from .validation import validate_order\n\n"
            "def quote(lines, discount=0):\n"
            "    validate_order(lines, discount)\n"
            "    return {'total': 1}\n",
            encoding="utf-8",
        )
        student.write_text(
            "from orders.pricing import quote\n"
            "def test_one():\n    quote([])\n"
            "def test_two():\n    quote([])\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"HARNESS_REPO": str(repo)}):
            with self.assertRaisesRegex(AssertionError, "exactly one"):
                runpy.run_path(str(quality))

        pricing.write_text(
            "def quote(lines, discount=0):\n    return {'total': 1}\n"
            "def test_misplaced():\n    return quote([])\n",
            encoding="utf-8",
        )
        student.write_text(
            "from orders.pricing import quote\n"
            "def test_one():\n    quote([])\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"HARNESS_REPO": str(repo)}):
            with self.assertRaisesRegex(AssertionError, "must not contain"):
                runpy.run_path(str(quality))

        pricing.write_text(
            "from decimal import Decimal\n\n"
            "def validate_order(lines, discount):\n    return None\n\n"
            "def quote(lines, discount=Decimal('0')):\n"
            "    validate_order(lines, discount)\n"
            "    return {'total': Decimal('1')}\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"HARNESS_REPO": str(repo)}):
            with self.assertRaisesRegex(AssertionError, "existing orders.validation"):
                runpy.run_path(str(quality))

        pricing.write_text(
            "from decimal import Decimal\n"
            "from .validation import validate_order\n\n"
            "def quote(lines, discount=Decimal('0')):\n"
            "    validate_order(lines, discount)\n"
            "    return {'total': Decimal('1')}\n\n"
            "if __name__ == '__main__':\n"
            "    assert quote(lines('1'))\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"HARNESS_REPO": str(repo)}):
            with self.assertRaisesRegex(AssertionError, "undefined global.*lines"):
                runpy.run_path(str(quality))


if __name__ == "__main__":
    unittest.main()

