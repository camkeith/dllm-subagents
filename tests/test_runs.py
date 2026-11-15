from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dllm import runs, shell, snapshot
from dllm.config import Limits, Profile, Settings
from dllm.llm import LLMError
from dllm.runs import RunError, RunManager, RunRecord, render_view, runs_dir

from helpers import FakeClient, isolated_git_env, make_repo, tool_completion

KEY_ENV = "DLLM_TEST_FAKE_KEY"
OTHER_ENV = "DLLM_TEST_OTHER_KEY"
SECRET = "sk-test-secret-123abc"
OTHER_SECRET = "sk-other-secret-456def"


def settings():
    profiles = {
        "fake": Profile(name="fake", base_url="http://127.0.0.1:9/v1", api_key_env=KEY_ENV, model="fake-model"),
        "other": Profile(name="other", base_url="http://127.0.0.1:9/v1", api_key_env=OTHER_ENV, model="other-model"),
    }
    return Settings(default_provider="fake", profiles=profiles, limits=Limits(), wait_seconds=1.0)


def finish(summary="did it"):
    return tool_completion(("finish", {"summary": summary}))


def write(path, content="hello\n"):
    return tool_completion(("write", {"path": path, "content": content}))


def bash(command):
    return tool_completion(("bash", {"command": command}))


def signal(event, completion):
    def item(messages):
        event.set()
        return completion
    return item


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, check=True).stdout.decode()


class RunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runs_path = Path(self.tmp) / "runs"
        env = isolated_git_env(self.tmp)
        env.update({
            "DLLM_SUBAGENTS_RUNS_DIR": str(self.runs_path),
            "DLLM_SUBAGENTS_CONFIG_DIR": str(Path(self.tmp) / "config"),
            KEY_ENV: SECRET,
            OTHER_ENV: OTHER_SECRET,
        })
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("CLAUDE_PROJECT_DIR", None)  # restored by patch.dict on stop
        self.addCleanup(snapshot.set_secret_env_names, ())  # RunManager sets them module-wide
        self.repo = make_repo(self.tmp)
        self.release = threading.Event()
        self.clients = []
        self.factory_calls = []

    # -- helpers ---------------------------------------------------------------------

    def manager(self, *scripts):
        """A RunManager whose n-th start gets FakeClient(scripts[n])."""
        queue = list(scripts)

        def factory(profile, key, model):
            self.factory_calls.append((profile.name, key, model))
            client = FakeClient(queue.pop(0))
            self.clients.append(client)
            return client

        m = RunManager(settings(), client_factory=factory)
        self.addCleanup(m.shutdown)
        self.addCleanup(self.release.set)  # runs before shutdown (cleanups are LIFO)
        return m

    def blocked(self, entered=None):
        def item(messages):
            if entered is not None:
                entered.set()
            self.release.wait(30)
            return finish("released")
        return item

    def record(self, run_id):
        return RunRecord.load(self.runs_path / (run_id + ".json"))

    def wait_for_progress(self, m, run_id, text, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            view = m.wait(run_id, 0.05)
            if view["status"] != "running" or text in view.get("progress", ""):
                return view
        self.fail("progress never showed %r" % text)

    def fresh_repo(self):
        return make_repo(Path(self.tmp) / "fresh", commit=False)

    # -- start / wait ----------------------------------------------------------------

    def test_start_wait_done(self):
        repo = self.fresh_repo()
        m = self.manager([write("a.txt"), finish("added a.txt")])
        rid = m.start("make a.txt", workdir=str(repo))
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertEqual(view["run_id"], rid)
        self.assertEqual(view["summary"], "added a.txt")
        files = {f["path"]: f for f in view["changed_files"]}
        self.assertEqual(files["a.txt"]["change"], "added")
        self.assertIn("+hello", view["diff"])
        self.assertFalse(view["diff_truncated"])
        self.assertTrue(Path(view["log_path"]).exists())
        self.assertEqual(view["stats"]["provider"], "fake")
        self.assertEqual(view["stats"]["model"], "fake-model")
        self.assertEqual(view["stats"]["steps"], 2)
        rec = self.record(rid)
        self.assertEqual(rec.status, "done")
        self.assertIsNotNone(rec.after_tree)
        self.assertIsNotNone(rec.ended_at)
        self.assertEqual(self.factory_calls, [("fake", SECRET, None)])

    def test_running_view_then_done(self):
        m = self.manager([self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 0.2)
        self.assertEqual(view["status"], "running")
        self.assertEqual(view["run_id"], rid)
        self.assertTrue(view["progress"])
        self.release.set()
        view = m.wait(rid, 10)
        self.assertEqual(view["status"], "done")
        self.assertEqual(view["summary"], "released")

    def test_running_progress_shows_step_and_action(self):
        entered = threading.Event()
        m = self.manager([write("x.txt"), self.blocked(entered)])
        rid = m.start("t", workdir=str(self.repo))
        self.assertTrue(entered.wait(10))
        progress = m.wait(rid, 0)["progress"]
        self.assertIn("step 1", progress)
        self.assertIn("write: x.txt", progress)
        self.assertIn("elapsed", progress)

    def test_render_running(self):
        m = self.manager([self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        text = render_view(m.wait(rid, 0.1))
        self.assertIn("status: running", text)
        self.assertIn(rid, text)
        self.assertIn("fast_agent_wait", text)
        self.assertNotIn("\u2014", text)

    def test_render_done(self):
        repo = self.fresh_repo()
        m = self.manager([write("a.txt"), finish("added a.txt")])
        rid = m.start("t", workdir=str(repo))
        text = render_view(m.wait(rid, 20))
        self.assertIn("status: done", text)
        self.assertIn(rid, text)
        self.assertIn("summary: added a.txt", text)
        self.assertIn("changed files (1):", text)
        self.assertIn("A a.txt (+1 -0)", text)
        self.assertIn("+hello", text)
        self.assertIn("fast_agent_undo", text)
        self.assertIn(".trajectory.json", text)
        self.assertNotIn("\u2014", text)

    def test_render_truncated_diff_note(self):
        m = self.manager([write("big.txt", "x\n" * 15000), finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertTrue(view["diff_truncated"])
        rec = self.record(rid)
        self.assertEqual(view["before_tree"], rec.before_tree)
        self.assertEqual(view["after_tree"], rec.after_tree)
        self.assertEqual(len(rec.after_tree), 40)
        text = render_view(view)
        self.assertIn("git diff %s %s" % (rec.before_tree, rec.after_tree), text)
        # The suggested command really shows the run's change.
        self.assertIn("big.txt", git(self.repo, "diff", "--stat", rec.before_tree, rec.after_tree))

    def test_render_no_changes_note(self):
        m = self.manager([finish("nothing to do")])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual(view["changed_files"], [])
        text = render_view(view)
        self.assertIn(".gitignore", text)
        self.assertIn("not tracked", text)
        self.assertNotIn("fast_agent_undo", text)

    def test_render_check_stats_binary(self):
        view = {
            "status": "check_failed", "run_id": "20261115-120000-abcdef", "summary": "tried",
            "limit": "max_steps", "error": "something broke",
            "check": {"command": "pytest -q", "exit_code": 1, "output_tail": "1 failed", "attempts": 3},
            "stats": {"steps": 7, "seconds": 12.34, "input_tokens": 1500, "output_tokens": 80,
                      "provider": "fake", "model": "fake-model"},
            "changed_files": [
                {"path": "img.png", "change": "added", "added": -1, "removed": -1},
                {"path": "a.txt", "change": "modified", "added": 2, "removed": 1},
                {"path": "b.txt", "change": "deleted", "added": 0, "removed": 1},
            ],
            "diff": "diff --git a/a.txt b/a.txt\n", "diff_truncated": False,
            "log_path": "/x/run.trajectory.json", "before_tree": "a" * 40, "after_tree": "b" * 40,
        }
        text = render_view(view)
        for part in ("status: check_failed", "limit: max_steps", "error: something broke",
                     "`pytest -q`", "exit code 1", "1 failed", "7 steps", "1500 in", "80 out",
                     "fake/fake-model", "changed files (3):", "A img.png (binary)",
                     "M a.txt (+2 -1)", "D b.txt (+0 -1)", "diff --git", "/x/run.trajectory.json",
                     "fast_agent_undo"):
            self.assertIn(part, text)
        self.assertNotIn("git diff " + "a" * 40, text)
        self.assertNotIn("\u2014", text)

    def test_render_caps_many_files(self):
        # A fresh repo with no .gitignore where the run creates thousands of dependency files.
        repo = self.fresh_repo()
        make = ("python3 -c \"import os; os.makedirs('node_modules/pkg'); "
                "[open('node_modules/pkg/f%d.js' % i, 'w').write('x\\n') for i in range(2999)]\"")
        m = self.manager([bash(make), write("src/app.js", "app\n"), finish("set up the app")])
        rid = m.start("t", workdir=str(repo))
        view = m.wait(rid, 60)
        self.assertEqual(view["status"], "done")
        self.assertEqual(len(view["changed_files"]), 3000)
        text = render_view(view)
        self.assertLess(len(text), 25_000)
        self.assertIn("changed files (3000):", text)
        self.assertIn("A src/app.js (+1 -0)", text)  # the real edit stays visible
        self.assertIn("+ 2900 more under node_modules/ (2900 added)", text)
        rec = self.record(rid)
        hint = "see all: git diff --stat %s %s" % (rec.before_tree, rec.after_tree)
        self.assertIn(hint, text)
        self.assertIn("3000 files changed", git(repo, "diff", "--stat", rec.before_tree, rec.after_tree))
        listed = [l for l in text.splitlines() if l.startswith("A ")]
        self.assertEqual(len(listed), 100)
        self.assertNotIn("—", text)

    def test_render_groups_rest_by_top_level_dir(self):
        changes = [{"path": "big/f%d" % i, "change": "added", "added": 1, "removed": 0} for i in range(150)]
        changes += [{"path": "gen/g%d" % i, "change": "modified" if i % 2 else "deleted",
                     "added": 1, "removed": 1} for i in range(30)]
        changes += [{"path": "top%d.txt" % i, "change": "added", "added": 1, "removed": 0} for i in range(3)]
        view = {
            "status": "done", "run_id": "20261115-120000-abcdef", "summary": "ok", "limit": None,
            "error": None, "check": None, "stats": {}, "changed_files": changes, "diff": "",
            "diff_truncated": False, "log_path": "", "before_tree": "a" * 40, "after_tree": "b" * 40,
        }
        text = render_view(view)
        # The smallest groups are listed in full; the rest is grouped.
        for i in range(3):
            self.assertIn("A top%d.txt" % i, text)
        self.assertIn("D gen/g0 (+1 -1)", text)
        self.assertIn("+ 83 more under big/ (83 added)", text)
        self.assertNotIn("more under gen/", text)
        self.assertIn("see all: git diff --stat %s %s" % ("a" * 40, "b" * 40), text)
        many = [{"path": "d%d/f" % i, "change": "added", "added": 1, "removed": 0} for i in range(400)]
        text = render_view(dict(view, changed_files=many))
        self.assertLess(len(text), 6_000)
        self.assertIn("+ 280 more under 280 other top-level paths", text)

    def test_render_caps_summary(self):
        view = {
            "status": "done", "run_id": "20261115-120000-abcdef", "summary": "s" * 50_000,
            "limit": None, "error": None, "check": None, "stats": {}, "changed_files": [],
            "diff": "", "diff_truncated": False, "log_path": "", "before_tree": "a" * 40,
            "after_tree": "a" * 40,
        }
        text = render_view(view)
        self.assertLess(len(text), 4_600)
        self.assertIn("summary: " + "s" * 2_000 + "\n... [46000 characters cut] ...\n", text)

    def test_workdir_fallback(self):
        os.environ["CLAUDE_PROJECT_DIR"] = str(self.repo)
        m = self.manager([write("from_env.txt"), finish()])
        rid = m.start("t")
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        self.assertTrue((self.repo / "from_env.txt").exists())
        self.assertEqual(self.record(rid).repo, os.path.realpath(str(self.repo)))

    def test_workdir_falls_back_to_server_cwd(self):
        m = self.manager([write("from_cwd.txt"), finish()])
        with mock.patch("os.getcwd", return_value=str(self.repo)):
            rid = m.start("t")
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        self.assertTrue((self.repo / "from_cwd.txt").exists())

    def test_workdir_never_defaults_to_the_plugin_dir(self):
        # Codex starts the server with cwd = the plugin's own directory (codex.mcp.json "cwd": ".").
        self.assertTrue((runs.PLUGIN_ROOT / ".mcp.json").exists())
        plugin = self.repo / "plugin"
        (plugin / "server").mkdir(parents=True)
        m = self.manager([finish()])
        for cwd in (plugin, plugin / "server"):
            with self.subTest(cwd=cwd):
                with mock.patch.object(runs, "PLUGIN_ROOT", plugin), \
                        mock.patch("os.getcwd", return_value=str(cwd)):
                    with self.assertRaises(RunError) as cm:
                        m.start("t")
                self.assertIn("pass workdir: the absolute path of the git repo to work in",
                              str(cm.exception))
                self.assert_no_run_started()

    # -- locking ---------------------------------------------------------------------

    def test_one_run_per_repo(self):
        m = self.manager([self.blocked()], [finish()])
        first = m.start("t", workdir=str(self.repo))
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(self.repo))
        self.assertIn(first, str(cm.exception))

    def test_lock_is_per_repo_root(self):
        (self.repo / "sub").mkdir()
        m = self.manager([self.blocked()], [finish()])
        first = m.start("t", workdir=str(self.repo))
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(self.repo / "sub"))
        self.assertIn(first, str(cm.exception))

    def test_lock_across_managers(self):
        m1 = self.manager([self.blocked()])
        m2 = self.manager([finish()])
        m1.start("t", workdir=str(self.repo))
        with self.assertRaises(RunError) as cm:
            m2.start("t", workdir=str(self.repo))
        self.assertIn("another session", str(cm.exception))

    def test_lock_released_after_run(self):
        m = self.manager([finish()], [finish("second")])
        first = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(first, 20)["status"], "done")
        second = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(second, 20)["summary"], "second")
        # The cross-process lock was released too.
        m2 = self.manager([finish("third")])
        third = m2.start("t", workdir=str(self.repo))
        self.assertEqual(m2.wait(third, 20)["status"], "done")

    def test_runs_in_different_repos_run_together(self):
        other = make_repo(Path(self.tmp) / "second")
        m = self.manager([self.blocked()], [finish("other repo")])
        m.start("t", workdir=str(self.repo))
        rid = m.start("t", workdir=str(other))
        self.assertEqual(m.wait(rid, 20)["summary"], "other repo")

    # -- validation ------------------------------------------------------------------

    def assert_no_run_started(self):
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith("dllm-run-")])
        self.assertFalse(self.runs_path.exists() and list(self.runs_path.iterdir()))

    def test_not_git_repo(self):
        plain = Path(self.tmp) / "plain"
        plain.mkdir()
        m = self.manager([finish()])
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(plain))
        self.assertIn("git init", str(cm.exception))
        self.assert_no_run_started()

    def test_relative_or_missing_workdir(self):
        m = self.manager([finish()])
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir="relative/path")
        self.assertIn("absolute", str(cm.exception))
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(Path(self.tmp) / "missing"))
        self.assertIn("does not exist", str(cm.exception))
        self.assert_no_run_started()

    def test_unknown_provider(self):
        m = self.manager([finish()])
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(self.repo), provider="nope")
        msg = str(cm.exception)
        self.assertIn("nope", msg)
        self.assertIn("fake", msg)
        self.assertIn("other", msg)
        self.assert_no_run_started()

    def test_missing_key_fails_fast(self):
        del os.environ[KEY_ENV]
        m = self.manager([finish()])
        with self.assertRaises(RunError) as cm:
            m.start("t", workdir=str(self.repo))
        self.assertIn(KEY_ENV, str(cm.exception))
        self.assertNotIn(OTHER_SECRET, str(cm.exception))
        self.assert_no_run_started()
        self.assertEqual(self.factory_calls, [])
        # No lock was left behind.
        os.environ[KEY_ENV] = SECRET
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")

    def test_limits_clamped(self):
        seen = {}
        real = runs.Agent

        def spy(client, ctx, limits, **kw):
            seen["limits"] = limits
            return real(client, ctx, limits, **kw)

        m = self.manager([finish()])
        with mock.patch.object(runs, "Agent", spy):
            rid = m.start("t", workdir=str(self.repo), max_steps=999, max_seconds=99999)
            m.wait(rid, 20)
        self.assertEqual(seen["limits"].max_steps, 150)
        self.assertEqual(seen["limits"].max_seconds, 1800)

    def test_files_included_in_first_message(self):
        (self.repo / "a.py").write_text("def special_marker():\n    return 42\n")
        m = self.manager([finish()])
        rid = m.start("t", workdir=str(self.repo), files=["a.py"], check_command="true")
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        first = self.clients[0].calls[0][1]["content"]
        self.assertIn("def special_marker():", first)
        self.assertIn("a.py", first)
        self.assertIn("Check command: true", first)

    def test_provider_and_model_passed_to_factory(self):
        m = self.manager([finish()])
        rid = m.start("t", workdir=str(self.repo), provider="other", model="custom-model")
        view = m.wait(rid, 20)
        self.assertEqual(self.factory_calls, [("other", OTHER_SECRET, "custom-model")])
        self.assertEqual(view["stats"]["provider"], "other")
        self.assertEqual(view["stats"]["model"], "custom-model")

    def test_default_client_factory(self):
        made = []

        def fake_chat_client(profile, key, model=None):
            made.append((profile.name, key, model))
            return FakeClient([finish()])

        m = RunManager(settings())
        self.addCleanup(m.shutdown)
        with mock.patch.object(runs, "ChatClient", fake_chat_client):
            rid = m.start("t", workdir=str(self.repo), model="m2")
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        self.assertEqual(made, [("fake", SECRET, "m2")])

    # -- cancel / interrupt / shutdown ----------------------------------------------

    def test_cancel_stops_and_keeps_changes(self):
        m = self.manager([write("kept.txt", "partial\n"), bash("sleep 30")])
        rid = m.start("t", workdir=str(self.repo))
        self.wait_for_progress(m, rid, "sleep 30")
        view = m.cancel(rid)  # returns a running view if the run did not stop within 5 s
        self.assertEqual(view["status"], "cancelled")
        self.assertTrue((self.repo / "kept.txt").exists())
        self.assertIn("kept.txt", [f["path"] for f in view["changed_files"]])
        self.assertIn("+partial", view["diff"])
        self.assertEqual(self.record(rid).status, "cancelled")

    def test_cancel_kills_all_only_when_no_other_run(self):
        other = make_repo(Path(self.tmp) / "second")
        m = self.manager([self.blocked()], [bash("sleep 30")], [bash("sleep 30")])
        blocked = m.start("t", workdir=str(self.repo))
        b = m.start("t", workdir=str(other))
        self.wait_for_progress(m, b, "sleep 30")
        with mock.patch.object(runs.shell, "kill_all") as kill_all:
            view = m.cancel(b)
        kill_all.assert_not_called()
        self.assertEqual(view["status"], "cancelled")  # the cancel event alone stopped it
        self.release.set()
        self.assertEqual(m.wait(blocked, 20)["status"], "done")
        c = m.start("t", workdir=str(other))
        self.wait_for_progress(m, c, "sleep 30")
        with mock.patch.object(runs.shell, "kill_all", wraps=shell.kill_all) as kill_all:
            view = m.cancel(c)
        kill_all.assert_called_once_with()
        self.assertEqual(view["status"], "cancelled")

    def test_cancel_finished_run_returns_view(self):
        m = self.manager([finish()])
        rid = m.start("t", workdir=str(self.repo))
        m.wait(rid, 20)
        self.assertEqual(m.cancel(rid)["status"], "done")

    def test_interrupt_returns_running(self):
        m = self.manager([self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        interrupt = threading.Event()
        interrupt.set()
        t0 = time.monotonic()
        view = m.wait(rid, 30, interrupt)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(view["status"], "running")

    def test_interrupt_set_while_waiting(self):
        m = self.manager([self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        interrupt = threading.Event()
        threading.Timer(0.2, interrupt.set).start()
        t0 = time.monotonic()
        view = m.wait(rid, 30, interrupt)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(view["status"], "running")

    def test_shutdown_cancels_active(self):
        m = self.manager([write("kept.txt", "partial\n"), bash("sleep 30")])
        rid = m.start("t", workdir=str(self.repo))
        self.wait_for_progress(m, rid, "sleep 30")
        t0 = time.monotonic()
        m.shutdown()
        self.assertLess(time.monotonic() - t0, 10)
        rec = self.record(rid)
        self.assertEqual(rec.status, "cancelled")
        self.assertEqual(len(rec.after_tree or ""), 40)
        self.assertIsNotNone(rec.ended_at)
        with self.assertRaises(RunError):
            m.start("t", workdir=str(self.repo))

    def test_shutdown_marks_stuck_run_terminal(self):
        # The model call never returns within shutdown's grace period.
        m = self.manager([write("kept.txt", "partial\n"), self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        self.wait_for_progress(m, rid, "write: kept.txt")
        with mock.patch.object(runs, "SHUTDOWN_GRACE_SECONDS", 0.3):
            m.shutdown()
        rec = self.record(rid)
        self.assertEqual(rec.status, "cancelled")
        self.assertEqual(len(rec.after_tree or ""), 40)
        self.assertIn("kept.txt", git(self.repo, "diff", "--stat", rec.before_tree, rec.after_tree))

    # -- errors ----------------------------------------------------------------------

    def test_error_view(self):
        def boom(messages):
            raise LLMError("provider exploded", 400)

        m = self.manager([boom], [finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "error")
        self.assertIn("provider exploded", view["error"])
        self.assertIn("error: provider exploded", render_view(view))
        self.assertEqual(self.record(rid).status, "error")
        rid2 = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid2, 20)["status"], "done")

    def test_after_snapshot_failure(self):
        real = snapshot.take_snapshot
        calls = []

        def flaky(repo, exclude=(), keep_from=None):
            calls.append(exclude)
            if len(calls) > 1:
                raise snapshot.SnapshotError("disk on fire")
            return real(repo, exclude, keep_from)

        m = self.manager([write("x.txt"), finish()], [finish()])
        with mock.patch.object(snapshot, "take_snapshot", flaky):
            rid = m.start("t", workdir=str(self.repo))
            view = m.wait(rid, 20)
        self.assertEqual(view["status"], "error")
        self.assertIn("disk on fire", view["error"])
        self.assertIsNone(view["after_tree"])
        rec = self.record(rid)
        self.assertEqual(rec.status, "error")
        self.assertIsNone(rec.after_tree)
        self.assertIn("disk on fire", render_view(view))
        with self.assertRaises(RunError):
            m.undo(rid)
        rid2 = m.start("t", workdir=str(self.repo))  # lock released
        self.assertEqual(m.wait(rid2, 20)["status"], "done")

    def test_diff_failure_reports_unknown_changes(self):
        def broken(*args, **kw):
            raise snapshot.SnapshotError("diff exploded")

        m = self.manager([write("x.txt"), finish()])
        with mock.patch.object(snapshot, "diff_trees", broken):
            rid = m.start("t", workdir=str(self.repo))
            view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertIn("diff exploded", view["error"])
        self.assertIsNone(view["changed_files"])
        text = render_view(view)
        self.assertIn("changed files: unknown", text)
        self.assertNotIn("not tracked", text)

    def test_undo_of_run_active_in_another_session(self):
        m1 = self.manager([self.blocked()])
        rid = m1.start("t", workdir=str(self.repo))
        m2 = self.manager()
        with self.assertRaises(RunError) as cm:
            m2.undo(rid)
        self.assertIn("another session", str(cm.exception))
        with self.assertRaises(RunError) as cm:
            m2.wait(rid, 0)
        self.assertIn("fast_agent_undo", str(cm.exception))

    def test_unknown_run_id(self):
        m = self.manager()
        for bad in ("nope", "../../etc/passwd", "20261115-120000-abcdef"):
            with self.assertRaises(RunError):
                m.wait(bad, 0)
            with self.assertRaises(RunError):
                m.cancel(bad)
            with self.assertRaises(RunError):
                m.undo(bad)

    # -- undo ------------------------------------------------------------------------

    def test_undo_reverts(self):
        m = self.manager([
            write("new.txt"),
            tool_completion(("edit", {"path": "a.txt", "edits": [{"old_text": "two", "new_text": "TWO"}]})),
            bash("rm b.txt"),
            finish(),
        ])
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        self.assertTrue((self.repo / "new.txt").exists())
        msg = m.undo(rid)
        self.assertIn(rid, msg)
        self.assertIn("back to how they were before it started (line endings follow your git settings).", msg)
        self.assertFalse((self.repo / "new.txt").exists())
        self.assertEqual((self.repo / "a.txt").read_text(), "one\ntwo\nthree\n")
        self.assertEqual((self.repo / "b.txt").read_text(), "keep\n")
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        self.assertTrue(self.record(rid).undone)
        with self.assertRaises(RunError) as cm:
            m.undo(rid)
        self.assertIn("already undone", str(cm.exception))

    def test_undo_refused_while_active(self):
        m = self.manager([self.blocked()])
        rid = m.start("t", workdir=str(self.repo))
        with self.assertRaises(RunError) as cm:
            m.undo(rid)
        self.assertIn("cancel", str(cm.exception))

    def test_undo_refused_while_other_run_active_in_repo(self):
        m = self.manager([write("new.txt"), finish()], [self.blocked()])
        first = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(first, 20)["status"], "done")
        second = m.start("t", workdir=str(self.repo))
        with self.assertRaises(RunError) as cm:
            m.undo(first)
        self.assertIn(second, str(cm.exception))
        self.assertTrue((self.repo / "new.txt").exists())

    def test_undo_conflict_passes_message(self):
        m = self.manager([write("new.txt"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        m.wait(rid, 20)
        (self.repo / "new.txt").write_text("user edit\n")
        with self.assertRaises(RunError) as cm:
            m.undo(rid)
        self.assertIn("new.txt", str(cm.exception))
        self.assertEqual((self.repo / "new.txt").read_text(), "user edit\n")
        self.assertFalse(self.record(rid).undone)

    def test_undo_after_restart(self):
        m = self.manager([write("new.txt"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        m.shutdown()
        m2 = RunManager(settings(), client_factory=lambda p, k, mo: FakeClient([]))
        self.addCleanup(m2.shutdown)
        m2.undo(rid)
        self.assertFalse((self.repo / "new.txt").exists())
        self.assertTrue(self.record(rid).undone)
        with self.assertRaises(RunError):
            m2.wait(rid, 0)  # not a live run in this session

    def test_ignored_files_excluded_and_protected(self):
        (self.repo / ".gitignore").write_text("secret.env\n")
        git(self.repo, "add", ".gitignore")
        git(self.repo, "commit", "-q", "-m", "ignore")
        (self.repo / "secret.env").write_text("TOKEN=abc\n")
        # The run un-ignores secret.env and edits it; neither may leak into diff or undo.
        m = self.manager([
            write(".gitignore", ""),
            write("secret.env", "TOKEN=changed\n"),
            write("new.txt"),
            finish(),
        ])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        paths = sorted(f["path"] for f in view["changed_files"])
        self.assertEqual(paths, [".gitignore", "new.txt"])
        self.assertNotIn("TOKEN", view["diff"])
        self.assertEqual(self.record(rid).ignored, ["secret.env"])
        m.shutdown()

        protects = []
        real_undo = snapshot.undo

        def spy(repo, before, after, protect=()):
            protects.append(list(protect))
            return real_undo(repo, before, after, protect=protect)

        m2 = RunManager(settings(), client_factory=lambda p, k, mo: FakeClient([]))
        self.addCleanup(m2.shutdown)
        with mock.patch.object(snapshot, "undo", spy):
            m2.undo(rid)
        self.assertEqual(protects, [["secret.env"]])
        self.assertEqual((self.repo / ".gitignore").read_text(), "secret.env\n")
        self.assertFalse((self.repo / "new.txt").exists())
        self.assertEqual((self.repo / "secret.env").read_text(), "TOKEN=changed\n")

    def test_run_that_ignores_untracked_files_keeps_them(self):
        (self.repo / "dep").mkdir()
        (self.repo / "dep" / "m.js").write_text("module\n")
        m = self.manager([
            write(".gitignore", "dep/\n"),
            tool_completion(("edit", {"path": "a.txt", "edits": [{"old_text": "two", "new_text": "TWO"}]})),
            finish(),
        ])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual(sorted(f["path"] for f in view["changed_files"]), [".gitignore", "a.txt"])
        m.undo(rid)
        self.assertEqual((self.repo / "a.txt").read_text(), "one\ntwo\nthree\n")
        self.assertEqual((self.repo / "dep" / "m.js").read_text(), "module\n")
        self.assertFalse((self.repo / ".gitignore").exists())

    def test_nested_repo_without_commit_is_noted(self):
        vendor = self.repo / "vendor" / "lib"
        vendor.mkdir(parents=True)
        git(vendor, "init", "-q")
        (vendor / "x.js").write_text("x\n")
        m = self.manager([write("vendor/lib/x.js", "changed\n"), write("app.txt"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertEqual([f["path"] for f in view["changed_files"]], ["app.txt"])
        self.assertEqual(view["nested"], ["vendor/lib/"])
        self.assertIn("not tracked: nested git repository vendor/lib/ (changes inside are not in this "
                      "diff, and undo does not revert them)", render_view(view))
        self.assertEqual(self.record(rid).nested, ["vendor/lib/"])

    def test_nested_repo_whose_git_dir_the_run_deletes_is_kept(self):
        vendor = self.repo / "vendor"
        vendor.mkdir()
        git(vendor, "init", "-q")
        (vendor / "lib.py").write_text("LIB = 1\n")
        m = self.manager([bash("rm -rf vendor/.git"), write("app.txt"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertEqual([f["path"] for f in view["changed_files"]], ["app.txt"])
        m.undo(rid)
        self.assertEqual((vendor / "lib.py").read_text(), "LIB = 1\n")
        self.assertFalse((self.repo / "app.txt").exists())

    def test_directory_that_becomes_a_nested_repo_is_not_deleted(self):
        (self.repo / "proj").mkdir()
        (self.repo / "proj" / "a.txt").write_text("a\n")

        def init_proj(messages):
            git(self.repo / "proj", "init", "-q")  # the guard blocks git init, so the test does it
            return write("other.txt", "after\n")

        m = self.manager([init_proj, finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual([f["path"] for f in view["changed_files"]], ["other.txt"])
        self.assertEqual(view["nested"], ["proj/"])
        m.undo(rid)
        self.assertFalse((self.repo / "other.txt").exists())
        self.assertEqual((self.repo / "proj" / "a.txt").read_text(), "a\n")

    def test_unreadable_keys_file_does_not_block_a_run(self):
        # Only the other provider's key would come from keys.env, which is not valid UTF-8.
        del os.environ[OTHER_ENV]
        cfg = Path(os.environ["DLLM_SUBAGENTS_CONFIG_DIR"])
        cfg.mkdir()
        (cfg / "keys.env").write_bytes(OTHER_ENV.encode() + b"=caf\xe9\n")
        m = self.manager([finish()])
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")

    # -- keys ------------------------------------------------------------------------

    def test_subagent_env_drops_secret_looking_variables(self):
        os.environ["GITHUB_TOKEN"] = "ghp-test-token-value-1"
        os.environ["DLLM_TEST_DB_PASSWORD"] = "hunter2-db-password"
        m = self.manager([bash('echo "gh=[$GITHUB_TOKEN] db=[$DLLM_TEST_DB_PASSWORD] path=[${PATH:+set}]"'),
                          finish()])
        rid = m.start("t", workdir=str(self.repo), check_command='test -z "$GITHUB_TOKEN"')
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")  # the check saw no token either
        self.assertIn("gh=[] db=[] path=[set]", self.clients[0].calls[1][-1]["content"])

    def test_pass_env_keeps_a_named_variable(self):
        os.environ["NPM_TOKEN"] = "npm-test-token-value"
        s = settings()
        s.pass_env = ["NPM_TOKEN"]
        made = []
        m = RunManager(s, client_factory=lambda p, k, mo: made.append(FakeClient(
            [bash('echo "npm=[$NPM_TOKEN]"'), finish()])) or made[-1])
        self.addCleanup(m.shutdown)
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        self.assertIn("npm=[npm-test-token-value]", made[0].calls[1][-1]["content"])

    def test_changed_file_named_after_key_is_redacted(self):
        m = self.manager([write("notes-%s.txt" % SECRET), finish()])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        self.assertEqual([f["path"] for f in view["changed_files"]], ["notes-***.txt"])
        for text in (json.dumps(view), render_view(view), (self.runs_path / (rid + ".json")).read_text()):
            self.assertNotIn(SECRET, text)

    def test_manager_keeps_key_variables_from_git(self):
        self.manager()
        self.assertEqual(snapshot._secret_env, frozenset({KEY_ENV, OTHER_ENV}))
        self.assertNotIn(KEY_ENV, snapshot._command(["status"], None)[1])

    def test_subagent_env_has_no_keys(self):
        os.environ["DLLM_TEST_VISIBLE"] = "visible-value"
        m = self.manager([bash('echo "k1=[$%s] k2=[$%s] v=[$DLLM_TEST_VISIBLE]"' % (KEY_ENV, OTHER_ENV)),
                          finish()])
        rid = m.start("t", workdir=str(self.repo), check_command='test -z "$%s"' % KEY_ENV)
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")  # the check saw no key
        out = self.clients[0].calls[1][-1]["content"]
        self.assertIn("k1=[] k2=[] v=[visible-value]", out)

    def test_records_never_contain_key(self):
        m = self.manager([bash("env"), write("x.txt"), finish("ok")])
        rid = m.start("t", workdir=str(self.repo))
        view = m.wait(rid, 20)
        texts = [
            (self.runs_path / (rid + ".json")).read_text(),
            Path(view["log_path"]).read_text(),
            json.dumps(view),
            render_view(view),
        ]
        for text in texts:
            self.assertNotIn(SECRET, text)
            self.assertNotIn(OTHER_SECRET, text)

    def test_keys_in_repo_files_are_redacted(self):
        # The other provider's key is only in keys.env, so it must come from resolve_api_key.
        del os.environ[OTHER_ENV]
        cfg = Path(os.environ["DLLM_SUBAGENTS_CONFIG_DIR"])
        cfg.mkdir()
        (cfg / "keys.env").write_text("%s=%s\n" % (OTHER_ENV, OTHER_SECRET))
        (self.repo / "creds.txt").write_text("A=%s\nB=%s\n" % (SECRET, OTHER_SECRET))
        git(self.repo, "add", "creds.txt")
        git(self.repo, "commit", "-q", "-m", "creds")
        m = self.manager([bash("cat creds.txt"), bash("rm creds.txt"), finish("removed creds")])
        rid = m.start("t", workdir=str(self.repo), check_command="git show HEAD:creds.txt")
        view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertIn("A=***", self.clients[0].calls[1][-1]["content"])  # the model saw *** only
        self.assertEqual([(f["path"], f["change"]) for f in view["changed_files"]], [("creds.txt", "deleted")])
        self.assertIn("-A=***", view["diff"])
        self.assertIn("A=***", view["check"]["output_tail"])
        texts = [
            (self.runs_path / (rid + ".json")).read_text(),
            Path(view["log_path"]).read_text(),
            json.dumps(view),
            render_view(view),
        ]
        for text in texts:
            self.assertNotIn(SECRET, text)
            self.assertNotIn(OTHER_SECRET, text)

    def test_bash_cannot_read_key_file(self):
        m = self.manager([bash("cat $DLLM_SUBAGENTS_CONFIG_DIR/keys.env"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        out = self.clients[0].calls[1][-1]["content"]
        self.assertIn("reading the plugin's key file is not allowed", out)

    # -- finish hardening ------------------------------------------------------------

    def test_finish_failure_gives_error_view(self):
        m = self.manager([finish()], [finish()])
        with mock.patch.object(RunManager, "_final_view", side_effect=RuntimeError("view boom")):
            rid = m.start("t", workdir=str(self.repo))
            view = m.wait(rid, 20)
        self.assertEqual(view["status"], "error")
        self.assertIn("view boom", view["error"])
        self.assertIn("status: error", render_view(view))
        self.assertEqual(m.wait(rid, 0)["status"], "error")  # stays finished, never "running"
        rid2 = m.start("t", workdir=str(self.repo))  # lock released
        self.assertEqual(m.wait(rid2, 20)["status"], "done")

    def test_done_without_view_reports_error(self):
        m = self.manager([finish()])
        rid = m.start("t", workdir=str(self.repo))
        m.wait(rid, 20)
        m._runs[rid].view = None  # simulate a worker that died before building its view
        view = m.wait(rid, 0)
        self.assertEqual(view["status"], "error")
        self.assertIn("status: error", render_view(view))

    def test_record_save_failure_is_reported(self):
        real_save = RunRecord.save
        calls = []

        def save(rec, path):
            calls.append(rec.status)
            if rec.status != "running":
                raise TypeError("not serializable")
            return real_save(rec, path)

        m = self.manager([finish()])
        with mock.patch.object(RunRecord, "save", save):
            rid = m.start("t", workdir=str(self.repo))
            view = m.wait(rid, 20)
        self.assertEqual(view["status"], "done")
        self.assertIn("could not save the run record", view["error"])

    def test_snapshot_failure_after_lock_releases_it(self):
        def boom(*args, **kw):
            raise snapshot.SnapshotError("snap boom")

        m = self.manager([finish()], [finish()], [finish()])  # each start builds a client
        for name in ("take_snapshot", "ignored_paths"):
            with self.subTest(name=name):
                with mock.patch.object(snapshot, name, boom):
                    with self.assertRaises(RunError) as cm:
                        m.start("t", workdir=str(self.repo))
                self.assertIn("snap boom", str(cm.exception))
                self.assert_no_run_started()
        rid = m.start("t", workdir=str(self.repo))  # neither lock was left behind
        self.assertEqual(m.wait(rid, 20)["status"], "done")

    def test_undo_rechecks_record_under_lock(self):
        m = self.manager([write("new.txt"), finish()])
        rid = m.start("t", workdir=str(self.repo))
        self.assertEqual(m.wait(rid, 20)["status"], "done")
        m2 = RunManager(settings(), client_factory=lambda p, k, mo: FakeClient([]))
        self.addCleanup(m2.shutdown)
        m2.undo(rid)  # another session undoes it; m still holds the old in-memory record
        with self.assertRaises(RunError) as cm:
            m.undo(rid)
        self.assertIn("already undone", str(cm.exception))

    def test_render_passing_check_shows_last_lines(self):
        view = {
            "status": "done", "run_id": "20261115-120000-abcdef", "summary": "ok", "limit": None,
            "error": None,
            "check": {"command": "pytest -q", "exit_code": 0, "attempts": 1,
                      "output_tail": "".join("line%d\n" % i for i in range(1, 9))},
            "stats": {}, "changed_files": [], "diff": "", "diff_truncated": False,
            "log_path": "", "before_tree": "a" * 40, "after_tree": "a" * 40,
        }
        text = render_view(view)
        self.assertIn("check output (last 5 lines):", text)
        for i in range(4, 9):
            self.assertIn("line%d" % i, text)
        self.assertNotIn("line3", text)

    # -- records and runs dir --------------------------------------------------------

    def test_record_roundtrip(self):
        rec = RunRecord(
            run_id="20261115-120000-abcdef", workdir="/w", repo="/r", before_tree="a" * 40,
            after_tree=None, status="running", started_at=1.5, ended_at=None, provider="fake",
            model="m", ignored=["node_modules/"],
        )
        path = Path(self.tmp) / "rec.json"
        rec.save(path)
        self.assertEqual(RunRecord.load(path), rec)
        self.assertFalse(rec.undone)

    def test_runs_dir_order(self):
        base = Path(self.tmp)
        env = {"DLLM_SUBAGENTS_RUNS_DIR": str(base / "explicit"),
               "CLAUDE_PLUGIN_DATA": str(base / "claude"),
               "PLUGIN_DATA": str(base / "codex"),
               "HOME": str(base / "home")}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(runs_dir(), base / "explicit")
            del os.environ["DLLM_SUBAGENTS_RUNS_DIR"]
            self.assertEqual(runs_dir(), base / "claude" / "runs")
            del os.environ["CLAUDE_PLUGIN_DATA"]
            self.assertEqual(runs_dir(), base / "codex" / "runs")
            del os.environ["PLUGIN_DATA"]
            home_runs = base / "home" / ".cache" / "dllm-subagents" / "runs"
            self.assertEqual(runs_dir(), home_runs)
        for path in (base / "explicit", base / "claude" / "runs", home_runs):
            self.assertTrue(path.is_dir())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)


if __name__ == "__main__":
    unittest.main()
