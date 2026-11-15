from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from dllm.agent import Agent
from dllm.config import Limits
from dllm.llm import Completion, LLMError
from dllm.prompts import SYSTEM_PROMPT
from dllm.tools import ToolContext, scrubbed_env

from helpers import FakeClient, tool_completion


def finish(summary="done", **kw):
    return tool_completion(("finish", {"summary": summary}), **kw)


def write(path, content="x"):
    return tool_completion(("write", {"path": path, "content": content}))


def text_only(text="thinking"):
    return Completion(
        content=text, tool_calls=[], input_tokens=10, output_tokens=5, finish_reason="stop",
        message={"role": "assistant", "content": text},
    )


class AgentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.workdir = self.dir / "work"
        self.workdir.mkdir()
        self.cancel = threading.Event()
        self.ctx = ToolContext(workdir=self.workdir, cancel_event=self.cancel)

    def agent(self, script, limits=None, **kw):
        self.client = FakeClient(script)
        return Agent(self.client, self.ctx, limits or Limits(), **kw)

    def test_finish_without_check(self):
        res = self.agent([finish("did it")]).run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(res.summary, "did it")
        self.assertEqual(res.steps, 1)

    def test_tools_then_finish(self):
        agent = self.agent([write("a.txt", "hi"), finish()])
        res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual((self.workdir / "a.txt").read_text(), "hi")
        self.assertEqual(self.client.calls[1][-1]["role"], "tool")

    def test_tool_message_per_call(self):
        c = tool_completion(
            ("write", {"path": "a.txt", "content": "1"}), ("write", {"path": "b.txt", "content": "2"})
        )
        agent = self.agent([c, finish()])
        agent.run("task")
        tool_msgs = [m for m in self.client.calls[1] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], [t.id for t in c.tool_calls])

    def test_tool_error_reported_to_model(self):
        agent = self.agent([tool_completion(("read", {"path": "missing.txt"})), finish()])
        res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertTrue(self.client.calls[1][-1]["content"].startswith("Error:"))

    def test_unexpected_exception_becomes_tool_message(self):
        agent = self.agent([tool_completion(("read", {"path": "a"})), finish()])
        with mock.patch("dllm.agent.run_tool", side_effect=ValueError("embedded null byte")):
            res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(self.client.calls[1][-1]["content"], "Error: ValueError: embedded null byte")

    def test_tool_result_capped(self):
        agent = self.agent([tool_completion(("read", {"path": "a"})), finish()])
        with mock.patch("dllm.agent.run_tool", return_value="y" * 100_000):
            agent.run("task")
        content = self.client.calls[1][-1]["content"]
        self.assertLess(len(content), 31_000)
        self.assertIn("characters cut", content)

    def test_no_tool_calls_is_format_error(self):
        res = self.agent([text_only(), finish()]).run("task")
        self.assertEqual(res.status, "done")
        last = self.client.calls[1][-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("call a tool", last["content"])
        self.assertIn("finish", last["content"])

    def test_repeated_format_errors(self):
        res = self.agent([text_only(), text_only(), text_only()]).run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("format", res.error)

    def test_format_error_counter_resets(self):
        script = [text_only(), text_only(), write("a.txt"), text_only(), text_only(), finish()]
        res = self.agent(script).run("task")
        self.assertEqual(res.status, "done")

    def test_malformed_arguments(self):
        agent = self.agent([tool_completion(("write", "{not json")), finish()])
        res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertTrue(self.client.calls[1][-1]["content"].startswith("Error: invalid JSON arguments"))

    def test_malformed_arguments_repeated_ends_run(self):
        bad = lambda: tool_completion(("write", "{not json"))
        res = self.agent([bad(), bad(), bad()]).run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("format", res.error)

    def test_unknown_tool_name(self):
        agent = self.agent([tool_completion(("delete_everything", {})), finish()])
        agent.run("task")
        self.assertIn("unknown tool", self.client.calls[1][-1]["content"])

    def test_finish_without_summary_is_tool_error(self):
        agent = self.agent([tool_completion(("finish", {})), finish("ok")])
        res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(res.summary, "ok")
        self.assertTrue(self.client.calls[1][-1]["content"].startswith("Error:"))

    def test_step_limit(self):
        script = [write("a.txt"), write("b.txt"), write("c.txt")]
        res = self.agent(script, Limits(max_steps=2)).run("task")
        self.assertEqual(res.status, "limit_reached")
        self.assertEqual(res.limit, "max_steps")
        self.assertEqual(res.steps, 2)

    def test_time_limit(self):
        now = [0.0]

        def jump(messages):
            now[0] = 1000.0
            return write("a.txt")

        res = self.agent([jump, finish()], Limits(max_seconds=300), clock=lambda: now[0]).run("task")
        self.assertEqual(res.status, "limit_reached")
        self.assertEqual(res.limit, "max_seconds")
        self.assertEqual(self.client.deadlines[0], 300.0)

    def test_token_limit(self):
        script = [tool_completion(("write", {"path": "a", "content": "x"}), tokens=(1_500_000, 600_000))]
        res = self.agent(script).run("task")
        self.assertEqual(res.status, "limit_reached")
        self.assertEqual(res.limit, "max_tokens_total")
        self.assertEqual(res.input_tokens, 1_500_000)
        self.assertEqual(res.output_tokens, 600_000)

    def test_llm_error_ends_run(self):
        def boom(messages):
            raise LLMError("bad tools", 400)

        res = self.agent([boom]).run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("bad tools", res.error)

    def test_unexpected_client_exception_ends_run(self):
        path = self.dir / "traj.json"

        def boom(messages):
            raise AttributeError("'str' object has no attribute 'get'")

        res = self.agent([boom], trajectory_path=path).run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("AttributeError", res.error)
        self.assertEqual(json.loads(path.read_text())["status"], "error")

    def test_on_step_exception_ends_run(self):
        def bad(n, label):
            raise RuntimeError("callback broke")

        res = self.agent([write("a.txt"), finish()], on_step=bad).run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("RuntimeError: callback broke", res.error)

    def test_trajectory_write_failure_does_not_break_run(self):
        blocker = self.dir / "file"
        blocker.write_text("x")
        res = self.agent([finish()], trajectory_path=blocker / "traj.json").run("task")
        self.assertEqual(res.status, "done")

    def test_cancel(self):
        def cancel_after(messages):
            self.cancel.set()
            return write("a.txt")

        res = self.agent([cancel_after, finish()]).run("task")
        self.assertEqual(res.status, "cancelled")
        self.assertEqual(len(self.client.calls), 1)
        self.assertFalse((self.workdir / "a.txt").exists())

    def test_cancel_before_second_step(self):
        agent = self.agent([write("a.txt"), finish()])
        agent.on_step = lambda n, label: self.cancel.set()
        res = agent.run("task")
        self.assertEqual(res.status, "cancelled")
        self.assertEqual(len(self.client.calls), 1)

    def test_check_pass(self):
        res = self.agent([write("a.txt"), finish()], check_command="test -f a.txt").run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(res.check.exit_code, 0)
        self.assertEqual(res.check.attempts, 1)

    def test_check_fail_then_fix(self):
        script = [finish(), write("a.txt"), finish()]
        res = self.agent(script, check_command="test -f a.txt").run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(res.check.attempts, 2)
        self.assertIn("check failed", self.client.calls[1][-1]["content"])

    def test_check_output_tail_in_message(self):
        script = [finish(), finish(), finish()]
        self.agent(script, check_command="echo boom-marker; exit 3").run("task")
        self.assertIn("boom-marker", self.client.calls[1][-1]["content"])

    def test_check_fail_three_times(self):
        res = self.agent([finish(), finish(), finish()], check_command="exit 1").run("task")
        self.assertEqual(res.status, "check_failed")
        self.assertEqual(res.check.attempts, 3)
        self.assertEqual(res.check.exit_code, 1)

    def test_check_runs_with_scrubbed_env(self):
        base = dict(os.environ, FOO_API_KEY="secret")
        self.ctx.env = scrubbed_env(["FOO_API_KEY"], base)
        script = [finish(), finish(), finish()]
        self.agent(script, check_command="echo [$FOO_API_KEY]; exit 1").run("task")
        msg = self.client.calls[1][-1]["content"]
        self.assertIn("[]", msg)
        self.assertNotIn("secret", msg)

    def test_check_spawn_error_ends_run(self):
        with mock.patch("dllm.agent.run_command", side_effect=OSError("no shell")):
            res = self.agent([finish()], check_command="true").run("task")
        self.assertEqual(res.status, "error")
        self.assertIn("no shell", res.error)

    def test_calls_after_finish_not_run(self):
        c = tool_completion(("finish", {"summary": "s"}), ("write", {"path": "b.txt", "content": "x"}))
        agent = self.agent([c])
        res = agent.run("task")
        self.assertEqual(res.status, "done")
        self.assertFalse((self.workdir / "b.txt").exists())
        tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertIn("not run", tool_msgs[1]["content"])

    def test_trajectory_written_each_step(self):
        path = self.dir / "out" / "traj.json"
        seen = []

        def peek(messages):
            seen.append(path.exists())
            return finish()

        res = self.agent([write("a.txt"), peek], trajectory_path=path).run("task")
        self.assertEqual(seen, [True])
        data = json.loads(path.read_text())
        self.assertEqual(data["status"], "done")
        self.assertIn("messages", data)
        self.assertEqual(set(data["stats"]), {"steps", "seconds", "input_tokens", "output_tokens"})
        self.assertEqual(data["stats"]["steps"], res.steps)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_on_step_progress(self):
        seen = []
        self.agent([write("a.txt"), finish()], on_step=lambda n, label: seen.append((n, label))).run("task")
        self.assertEqual(seen[0], (1, "write: a.txt"))
        self.assertEqual(seen[1][0], 2)

    def test_first_messages(self):
        self.agent([finish()]).run("do the thing")
        first = self.client.calls[0]
        self.assertEqual(first[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(first[1], {"role": "user", "content": "do the thing"})
        self.assertEqual(len(first), 2)


SECRET = "sk-test-secret-123456"
# Prints SECRET without the command text containing it, so assistant messages stay clean.
PRINT_SECRET = "printf 'sk-test-%s\\n' 'secret-123456'"


class RedactTest(unittest.TestCase):
    setUp = AgentTest.setUp
    agent = AgentTest.agent

    def test_redacts_tool_output_summary_and_trajectory(self):
        path = self.dir / "traj.json"
        script = [
            tool_completion(("bash", {"command": PRINT_SECRET})),
            tool_completion(("read", {"path": "creds.txt"})),
            finish("found " + SECRET),
        ]
        (self.workdir / "creds.txt").write_text("KEY=%s\n" % SECRET)
        res = self.agent(script, trajectory_path=path, redact=[SECRET]).run("task")
        self.assertEqual(res.status, "done")
        bash_out = self.client.calls[1][-1]["content"]
        read_out = self.client.calls[2][-1]["content"]
        self.assertIn("***", bash_out)
        self.assertIn("KEY=***", read_out)
        self.assertEqual(res.summary, "found ***")
        traj = path.read_text()
        self.assertIn("***", traj)
        for text in (bash_out, read_out, traj, repr(res)):
            self.assertNotIn(SECRET, text)

    def test_redacts_check_output(self):
        path = self.dir / "traj.json"
        res = self.agent([finish()], trajectory_path=path, redact=[SECRET], max_check_failures=1,
                         check_command=PRINT_SECRET + "; exit 1").run("task")
        self.assertEqual(res.status, "check_failed")
        self.assertIn("***", res.check.output_tail)
        self.assertNotIn(SECRET, res.check.output_tail)
        traj = path.read_text()
        self.assertIn("***", traj)  # the failed-check tool message
        self.assertNotIn(SECRET, traj)

    def test_redacts_passing_check_output(self):
        res = self.agent([finish()], redact=[SECRET], check_command=PRINT_SECRET).run("task")
        self.assertEqual(res.status, "done")
        self.assertEqual(res.check.output_tail.strip(), "***")

    def test_redacts_task_message(self):
        self.agent([finish()], redact=[SECRET]).run("file says " + SECRET)
        self.assertEqual(self.client.calls[0][1]["content"], "file says ***")

    def test_redacts_progress_label_and_stored_assistant_message(self):
        seen = []
        script = [tool_completion(("bash", {"command": "echo " + SECRET})), finish()]
        self.agent(script, redact=[SECRET], on_step=lambda n, label: seen.append(label)).run("task")
        self.assertEqual(seen[0], "bash: echo ***")
        stored = self.client.calls[1][2]["tool_calls"][0]["function"]["arguments"]
        self.assertNotIn(SECRET, stored)
        self.assertIn("exit code: 0\n***", self.client.calls[1][-1]["content"])  # ran the raw command

    def test_short_redact_values_ignored(self):
        script = [tool_completion(("bash", {"command": "echo short-ok"})), finish()]
        self.agent(script, redact=["short", "", "ok"]).run("task")
        self.assertIn("short-ok", self.client.calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
