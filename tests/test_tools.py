from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from dllm import tools
from dllm.shell import CommandResult
from dllm.tools import ToolContext, ToolError, resolve_in_workdir, run_tool


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.workdir = self.dir / "work"
        self.workdir.mkdir()
        self.ctx = ToolContext(workdir=self.workdir, cancel_event=threading.Event())

    def test_read_numbered_lines(self):
        (self.workdir / "f.txt").write_text("a\nb\n")
        self.assertEqual(run_tool("read", {"path": "f.txt"}, self.ctx), "     1\ta\n     2\tb")

    def test_read_offset_limit_reports_range(self):
        (self.workdir / "f.txt").write_text("".join("l%d\n" % i for i in range(1, 11)))
        out = run_tool("read", {"path": "f.txt", "offset": 3, "limit": 2}, self.ctx)
        self.assertIn("     3\tl3\n     4\tl4", out)
        self.assertNotIn("l5", out)
        self.assertIn("[lines 3-4 of 10]", out)

    def test_read_long_line_cut(self):
        (self.workdir / "f.txt").write_text("x" * 5000 + "\n")
        out = run_tool("read", {"path": "f.txt"}, self.ctx)
        self.assertLess(len(out), 2100)

    def test_read_binary_file(self):
        (self.workdir / "b.bin").write_bytes(b"ab\x00cd" * 10)
        with self.assertRaises(ToolError) as cm:
            run_tool("read", {"path": "b.bin"}, self.ctx)
        self.assertIn("binary", str(cm.exception))
        self.assertIn("50", str(cm.exception))

    def test_read_missing_file(self):
        with self.assertRaises(ToolError) as cm:
            run_tool("read", {"path": "nope.txt"}, self.ctx)
        self.assertIn("not found", str(cm.exception))

    def test_write_creates_parents(self):
        out = run_tool("write", {"path": "a/b/c.txt", "content": "x"}, self.ctx)
        self.assertEqual((self.workdir / "a" / "b" / "c.txt").read_text(), "x")
        self.assertIn("a/b/c.txt", out)

    def test_edit_applies(self):
        (self.workdir / "f.txt").write_text("hello world")
        out = run_tool(
            "edit",
            {"path": "f.txt", "edits": [{"old_text": "world", "new_text": "there"}]},
            self.ctx,
        )
        self.assertIn("1 replacement", out)
        self.assertEqual((self.workdir / "f.txt").read_text(), "hello there")

    def test_edit_non_utf8_refused(self):
        p = self.workdir / "l.txt"
        p.write_bytes(b"caf\xe9")
        with self.assertRaises(ToolError) as cm:
            run_tool("edit", {"path": "l.txt", "edits": [{"old_text": "caf", "new_text": "x"}]}, self.ctx)
        self.assertIn("UTF-8", str(cm.exception))
        self.assertEqual(p.read_bytes(), b"caf\xe9")

    def test_edit_error_passthrough(self):
        (self.workdir / "f.txt").write_text("abc")
        with self.assertRaises(ToolError) as cm:
            run_tool("edit", {"path": "f.txt", "edits": [{"old_text": "zzz", "new_text": "y"}]}, self.ctx)
        self.assertIn("old_text not found", str(cm.exception))
        self.assertEqual((self.workdir / "f.txt").read_text(), "abc")

    def test_path_escape_rejected(self):
        for p in ("../x", "/etc/hosts"):
            with self.assertRaises(ToolError) as cm:
                run_tool("read", {"path": p}, self.ctx)
            self.assertIn("outside the working directory", str(cm.exception))

    def test_symlink_escape_rejected(self):
        outside = self.dir / "outside.txt"
        outside.write_text("orig")
        os.symlink(str(outside), str(self.workdir / "link"))
        with self.assertRaises(ToolError):
            run_tool("write", {"path": "link", "content": "pwned"}, self.ctx)
        self.assertEqual(outside.read_text(), "orig")

    def test_symlinked_workdir_ok(self):
        link = self.dir / "wlink"
        os.symlink(str(self.workdir), str(link))
        (self.workdir / "f.txt").write_text("a")
        self.assertEqual(resolve_in_workdir(link, "f.txt"), (self.workdir / "f.txt").resolve())
        self.assertEqual(resolve_in_workdir(link, "new.txt"), self.workdir.resolve() / "new.txt")

    def test_bash_runs_in_workdir(self):
        out = run_tool("bash", {"command": "pwd"}, self.ctx)
        self.assertIn(os.path.realpath(str(self.workdir)), out)
        self.assertTrue(out.startswith("exit code: 0"))

    def test_bash_timeout_capped(self):
        res = CommandResult(output="", returncode=0, timed_out=False, cancelled=False)
        with mock.patch.object(tools, "run_command", return_value=res) as m:
            run_tool("bash", {"command": "true", "timeout": 9999}, self.ctx)
        self.assertEqual(m.call_args[0][2], 300)

    def test_bash_timed_out_marker(self):
        res = CommandResult(output="partial", returncode=-9, timed_out=True, cancelled=False)
        with mock.patch.object(tools, "run_command", return_value=res):
            out = run_tool("bash", {"command": "x", "timeout": 5}, self.ctx)
        self.assertIn("[timed out after 5s]", out)

    def test_bash_bad_cwd_is_tool_error(self):
        ctx = ToolContext(workdir=self.dir / "gone", cancel_event=threading.Event())
        with self.assertRaises(ToolError):
            run_tool("bash", {"command": "true"}, ctx)

    def test_bash_guard_blocks(self):
        ctx = ToolContext(
            workdir=self.workdir, cancel_event=threading.Event(), guard=lambda c, w: "no pushing"
        )
        with mock.patch.object(tools, "run_command") as m:
            with self.assertRaises(ToolError) as cm:
                run_tool("bash", {"command": "git push"}, ctx)
        self.assertIn("blocked: no pushing", str(cm.exception))
        m.assert_not_called()

    def test_unknown_tool(self):
        with self.assertRaises(ToolError) as cm:
            run_tool("nope", {}, self.ctx)
        self.assertIn("unknown tool", str(cm.exception))

    def test_bad_args(self):
        with self.assertRaises(ToolError) as cm:
            run_tool("read", {}, self.ctx)
        self.assertIn("path", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
