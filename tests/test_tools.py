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

from helpers import call_near_fifo


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

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs mkfifo")
    def test_fifo_refused_without_blocking(self):
        fifo = self.workdir / "pipe"
        os.mkfifo(str(fifo))
        calls = (("read", {"path": "pipe"}),
                 ("edit", {"path": "pipe", "edits": [{"old_text": "a", "new_text": "b"}]}))
        for name, args in calls:
            with self.subTest(tool=name):
                finished, _, error = call_near_fifo(fifo, lambda n=name, a=args: run_tool(n, a, self.ctx))
                self.assertTrue(finished, "%s blocked on a FIFO" % name)
                self.assertIsInstance(error, ToolError)
                self.assertIn("pipe is not a regular file", str(error))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs mkfifo")
    def test_write_refuses_fifo_without_blocking(self):
        fifo = self.workdir / "pipe"
        os.mkfifo(str(fifo))
        finished, _, error = call_near_fifo(
            fifo, lambda: run_tool("write", {"path": "pipe", "content": "x"}, self.ctx))
        self.assertTrue(finished, "write blocked on a FIFO")
        self.assertIsInstance(error, ToolError)
        self.assertIn("pipe is not a regular file", str(error))

    def test_size_limit_message_follows_the_constant(self):
        with open(str(self.workdir / "big.log"), "wb") as f:
            f.truncate(1024 * 1024 + 1)
        with mock.patch.object(tools, "MAX_FILE_BYTES", 1024 * 1024):
            with self.assertRaises(ToolError) as cm:
                run_tool("read", {"path": "big.log"}, self.ctx)
        self.assertIn("over the 1 MB limit", str(cm.exception))

    def test_oversize_file_refused(self):
        with open(str(self.workdir / "big.log"), "wb") as f:
            f.truncate(tools.MAX_FILE_BYTES + 1)  # sparse: no real disk use
        calls = (("read", {"path": "big.log"}),
                 ("edit", {"path": "big.log", "edits": [{"old_text": "a", "new_text": "b"}]}))
        for name, args in calls:
            with self.subTest(tool=name):
                with self.assertRaises(ToolError) as cm:
                    run_tool(name, args, self.ctx)
                self.assertIn("big.log is %d bytes; over the 10 MB limit, use bash with head/sed"
                              % (tools.MAX_FILE_BYTES + 1), str(cm.exception))

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

    def test_write_and_edit_refuse_git_dir(self):
        git = self.workdir / ".git"
        (git / "hooks").mkdir(parents=True)
        (git / "config").write_text("[core]\n")
        os.symlink(".git/config", str(self.workdir / "cfg-link"))
        for path in (".git/hooks/pre-commit", ".git/config", "sub/.git/x", ".GIT/hooks/x", "cfg-link"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError) as cm:
                    run_tool("write", {"path": path, "content": "x"}, self.ctx)
                self.assertIn("inside a .git directory", str(cm.exception))
        with self.assertRaises(ToolError) as cm:
            run_tool("edit", {"path": ".git/config", "edits": [{"old_text": "core", "new_text": "x"}]},
                     self.ctx)
        self.assertIn("inside a .git directory", str(cm.exception))
        self.assertEqual((git / "config").read_text(), "[core]\n")
        self.assertEqual(sorted(os.listdir(str(git / "hooks"))), [])
        self.assertFalse((self.workdir / "sub").exists())
        # read still works there, and names that only contain ".git" are fine.
        self.assertIn("[core]", run_tool("read", {"path": ".git/config"}, self.ctx))
        run_tool("write", {"path": ".gitignore", "content": "x"}, self.ctx)
        run_tool("write", {"path": "a.git/b.git", "content": "x"}, self.ctx)

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

    def test_redact_replaces_longest_value_first(self):
        text = "a sk-test-secret-123456 b sk-test-secret c"
        self.assertEqual(tools.redact(text, ["sk-test-secret", "sk-test-secret-123456"]), "a *** b *** c")

    def test_scrub_cuts_clears_pieces_beside_markers(self):
        key = "sk-test-secret-123456"
        text = ("head " + key[:10] + "\n... [99 characters cut] ...\n" + key[12:] + " mid " + key[:9]
                + "... [line cut]\nnot cut " + key[:10])
        out = tools.scrub_cuts(text, [key])
        self.assertEqual(out, "head ***\n... [99 characters cut] ...\n*** mid ***... [line cut]\nnot cut "
                         + key[:10])
        # Pieces shorter than 8 characters, and text without markers, are left alone.
        short = "x " + key[:7] + "\n... [5 characters cut] ...\n" + key[-7:]
        self.assertEqual(tools.scrub_cuts(short, [key]), short)
        self.assertEqual(tools.scrub_cuts("plain " + key[:12], [key]), "plain " + key[:12])

    def test_unknown_tool(self):
        with self.assertRaises(ToolError) as cm:
            run_tool("nope", {}, self.ctx)
        self.assertIn("unknown tool", str(cm.exception))

    def test_bad_args(self):
        with self.assertRaises(ToolError) as cm:
            run_tool("read", {}, self.ctx)
        self.assertIn("path", str(cm.exception))


class RunEnvTest(unittest.TestCase):
    BASE = {
        "PATH": "/bin", "HOME": "/h", "USER": "u", "SHELL": "/bin/sh", "LANG": "C", "LC_ALL": "C",
        "TERM": "xterm", "TMPDIR": "/t", "PWD": "/w", "EDITOR": "vi",
        "AWS_REGION": "r", "AWS_DEFAULT_REGION": "r", "AWS_PROFILE": "p",
        "AWS_SECRET_ACCESS_KEY": "s", "AWS_ACCESS_KEY_ID": "i", "GITHUB_TOKEN": "g", "GH_TOKEN": "g",
        "MY_API_KEY": "k", "my_apikey": "k", "DB_PASSWORD": "p", "DB_PASS": "p", "STRIPE_SECRET": "s",
        "X_SECRET_Y": "s", "SECRET_KEY_BASE": "s", "NPM_TOKEN": "n", "ANTHROPIC_BASE_URL": "a",
        "OPENAI_ORG": "o", "CDPATH": "/", "ENV": "/e", "BASH_ENV": "/b", "PROVIDER_KEY": "k",
    }
    KEPT = ("PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TERM", "TMPDIR", "PWD", "EDITOR",
            "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE")

    def test_drops_secret_looking_variables(self):
        env = tools.run_env(["PROVIDER_KEY"], base=self.BASE)
        self.assertEqual(sorted(env), sorted(self.KEPT))

    def test_drops_keys_credentials_dsns_and_urls_with_passwords(self):
        dropped = {
            "AZURE_OPENAI_KEY": "k", "DD_APP_KEY": "k", "GOOGLE_CREDENTIALS": "c", "SENTRY_DSN": "d",
            "MYSQL_PWD": "p", "SSH_AUTH_SOCK": "/s", "DATABASE_URL": "postgres://u:p@h/db",
            "GPG_PASSPHRASE": "p", "AZURE_DEVOPS_EXT_PAT": "p", "NPM_CONFIG__AUTH": "a",
            "npm_config__auth": "a", "MINIO_ACCESS_KEY": "k", "JWT_PRIVATE_KEY": "k",
            "REDIS": "redis://:hunter2@cache:6379",
        }
        kept = {"CACHE_URL": "postgres://h/db", "PWD": "/w", "OLDPWD": "/o", "PATH": "/bin",
                "CLASSPATH": "/c", "GOPATH": "/g", "PROXY": "http://user@proxy:8080/", "SITE": "https://h:8443/x"}
        env = tools.run_env([], base=dict(dropped, **kept))
        self.assertEqual(env, kept)
        self.assertEqual(tools.run_env([], base={"DATABASE_URL": "postgres://h/db"}), {"DATABASE_URL": "postgres://h/db"})

    def test_pass_env_keeps_a_url_with_a_password(self):
        env = tools.run_env([], pass_env=["DATABASE_URL"], base={"DATABASE_URL": "postgres://u:p@h/db"})
        self.assertEqual(env, {"DATABASE_URL": "postgres://u:p@h/db"})
        # An always-passed name wins too.
        self.assertEqual(tools.run_env([], base={"PATH": "x://u:p@h/"}), {"PATH": "x://u:p@h/"})

    def test_pass_env_keeps_named_variables_but_never_provider_keys(self):
        env = tools.run_env(["PROVIDER_KEY"], pass_env=["NPM_TOKEN", "PROVIDER_KEY"], base=self.BASE)
        self.assertEqual(env["NPM_TOKEN"], "n")
        self.assertNotIn("PROVIDER_KEY", env)
        self.assertNotIn("GITHUB_TOKEN", env)


if __name__ == "__main__":
    unittest.main()
