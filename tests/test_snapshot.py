from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from dllm.snapshot import (
    FileChange,
    SnapshotError,
    diff_trees,
    git_path,
    ignored_paths,
    repo_root,
    snapshot_tree,
    undo,
)
from helpers import isolated_git_env, make_repo


def run_git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, check=True).stdout


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(os.environ, isolated_git_env(self.tmp))
        patcher.start()
        self.addCleanup(patcher.stop)

    def changes(self, repo, before, after):
        out, _, _ = diff_trees(repo, before, after)
        return {c.path: c for c in out}

    def test_not_a_repo(self):
        d = Path(self.tmp) / "plain"
        d.mkdir()
        with self.assertRaises(SnapshotError) as cm:
            repo_root(d)
        self.assertIn("git init", str(cm.exception))

    def test_snapshot_does_not_touch_index_or_stash(self):
        repo = make_repo(self.tmp)
        (repo / "new.txt").write_text("x\n")
        (repo / "a.txt").write_text("changed\n")
        index = git_path(repo, "index")
        status = run_git(repo, "status", "--porcelain")
        stash = run_git(repo, "stash", "list")
        idx_bytes = index.read_bytes()
        head = run_git(repo, "rev-parse", "HEAD")
        refs = run_git(repo, "for-each-ref")
        sha = snapshot_tree(repo)
        self.assertEqual(len(sha), 40)
        self.assertEqual(run_git(repo, "status", "--porcelain"), status)
        self.assertEqual(run_git(repo, "stash", "list"), stash)
        self.assertEqual(index.read_bytes(), idx_bytes)
        self.assertEqual(run_git(repo, "rev-parse", "HEAD"), head)
        self.assertEqual(run_git(repo, "for-each-ref"), refs)

    def test_diff_add_modify_delete_untracked(self):
        repo = make_repo(self.tmp)
        (repo / "old_untracked.txt").write_text("u\n")
        before = snapshot_tree(repo)
        (repo / "a.txt").write_text("one\nTWO\nthree\nfour\n")
        (repo / "b.txt").unlink()
        (repo / "old_untracked.txt").unlink()
        (repo / "sub").mkdir()
        (repo / "sub" / "new file.txt").write_text("1\n2\n")
        after = snapshot_tree(repo)
        ch = self.changes(repo, before, after)
        self.assertEqual(
            set(ch), {"a.txt", "b.txt", "old_untracked.txt", "sub/new file.txt"}
        )
        self.assertEqual(ch["a.txt"], FileChange("a.txt", "modified", 2, 1))
        self.assertEqual(ch["b.txt"], FileChange("b.txt", "deleted", 0, 1))
        self.assertEqual(ch["old_untracked.txt"], FileChange("old_untracked.txt", "deleted", 0, 1))
        self.assertEqual(ch["sub/new file.txt"], FileChange("sub/new file.txt", "added", 2, 0))

    def test_binary_counts(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "bin.dat").write_bytes(b"\x00\x01\x02\xff")
        after = snapshot_tree(repo)
        ch = self.changes(repo, before, after)
        self.assertEqual(ch["bin.dat"], FileChange("bin.dat", "added", -1, -1))

    def test_ignored_files_excluded(self):
        repo = make_repo(self.tmp)
        (repo / ".gitignore").write_text("node_modules/\n")
        run_git(repo, "add", ".gitignore")
        before = snapshot_tree(repo)
        (repo / "node_modules").mkdir()
        (repo / "node_modules" / "x").write_text("junk\n")
        after = snapshot_tree(repo)
        self.assertEqual(before, after)
        self.assertEqual(diff_trees(repo, before, after)[0], [])

    def test_diff_truncated(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "big.txt").write_text("line of text\n" * 4000)
        after = snapshot_tree(repo)
        changes, text, truncated = diff_trees(repo, before, after)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text), 20_200)
        self.assertEqual(changes[0].added, 4000)
        _, small, trunc2 = diff_trees(repo, before, after, max_chars=10_000_000)
        self.assertFalse(trunc2)
        self.assertIn("+line of text", small)

    def test_diff_read_is_bounded(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "huge.txt").write_text("a fairly long line of generated text\n" * 200_000)  # ~7 MB patch
        after = snapshot_tree(repo)
        procs = []
        real_popen = subprocess.Popen

        def spy(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            procs.append((args[0], proc))
            return proc

        with mock.patch.object(subprocess, "Popen", spy):
            changes, text, truncated = diff_trees(repo, before, after)
        self.assertTrue(truncated)
        self.assertLessEqual(len(text), 20_000 + 30)
        self.assertTrue(text.endswith("[diff truncated]\n"))
        self.assertEqual(changes[0].added, 200_000)  # counts stay complete
        patch = [p for cmd, p in procs if "-p" in cmd]
        self.assertEqual(len(patch), 1)
        # git was stopped once the cap was reached, instead of writing the whole patch.
        self.assertLess(patch[0].returncode, 0)

    def test_diff_redacts_before_cutting(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        key = "sk-diff-secret-0123456789"
        (repo / "k.txt").write_text("x" * 500 + key + "\n" + "tail line\n" * 50)
        after = snapshot_tree(repo)
        _, full, _ = diff_trees(repo, before, after, max_chars=100_000)
        at = full.index(key)
        pieces = {key[i:i + 8] for i in range(len(key) - 7)}
        for cut in (at + 5, at + 12, at + len(key) - 1, at + len(key) + 30):
            with self.subTest(cut=cut):
                _, text, truncated = diff_trees(repo, before, after, max_chars=cut, redact=[key])
                self.assertTrue(truncated)
                self.assertTrue(text.endswith("[diff truncated]\n"))
                for piece in pieces:
                    self.assertNotIn(piece, text)
        _, text, truncated = diff_trees(repo, before, after, max_chars=100_000, redact=[key])
        self.assertFalse(truncated)
        self.assertIn("+" + "x" * 500 + "***\n", text)

    def test_diff_read_reports_git_failure(self):
        repo = make_repo(self.tmp)
        with self.assertRaises(SnapshotError) as cm:
            diff_trees(repo, snapshot_tree(repo), "1" * 40)
        self.assertIn("git diff-tree failed", str(cm.exception))

    def test_undo_restores_exactly(self):
        repo = make_repo(self.tmp)
        (repo / "u.txt").write_text("u\n")
        before = snapshot_tree(repo)
        (repo / "a.txt").write_text("zzz\n")
        (repo / "b.txt").unlink()
        (repo / "u.txt").unlink()
        (repo / "new.txt").write_text("n\n")
        (repo / "bin.dat").write_bytes(b"\x00\xff\x00")
        after = snapshot_tree(repo)
        undo(repo, before, after)
        self.assertEqual(snapshot_tree(repo), before)
        self.assertFalse((repo / "new.txt").exists())
        self.assertTrue((repo / "b.txt").exists())
        self.assertEqual(run_git(repo, "diff", "--cached", "--name-only"), b"")

    def test_undo_conflict_refused(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "a.txt").write_text("one\nTWO\nthree\n")
        after = snapshot_tree(repo)
        (repo / "a.txt").write_text("totally different\n")
        with self.assertRaises(SnapshotError) as cm:
            undo(repo, before, after)
        self.assertIn("a.txt", str(cm.exception))
        self.assertEqual((repo / "a.txt").read_text(), "totally different\n")

    def test_undo_after_gc_pruned_snapshots(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "a.txt").write_text("changed\n")
        after = snapshot_tree(repo)
        loose = git_path(repo, "objects") / after[:2] / after[2:]
        self.assertTrue(loose.exists())
        loose.unlink()  # what git gc does to an unreferenced snapshot tree
        for b, a in ((before, after), (before, "1" * 40), ("1" * 40, before)):
            with self.assertRaises(SnapshotError) as cm:
                undo(repo, b, a)
            self.assertEqual(str(cm.exception),
                             "the run's snapshots were pruned by git gc; undo is no longer possible")
        self.assertEqual((repo / "a.txt").read_text(), "changed\n")

    def test_undo_noop_when_same(self):
        repo = make_repo(self.tmp)
        t = snapshot_tree(repo)
        undo(repo, t, t)

    def test_empty_repo_no_commits(self):
        repo = make_repo(self.tmp, commit=False)
        before = snapshot_tree(repo)
        (repo / "f.txt").write_text("hi\n")
        after = snapshot_tree(repo)
        self.assertEqual(self.changes(repo, before, after)["f.txt"].change, "added")
        undo(repo, before, after)
        self.assertFalse((repo / "f.txt").exists())
        self.assertEqual(snapshot_tree(repo), before)

    def test_worktree_and_subdir(self):
        repo = make_repo(self.tmp)
        wt = Path(self.tmp) / "wt"
        run_git(repo, "worktree", "add", "-q", "-b", "other", str(wt))
        sub = wt / "pkg" / "inner"
        sub.mkdir(parents=True)
        main_index = git_path(repo, "index").read_bytes()
        self.assertEqual(repo_root(sub).resolve(), wt.resolve())
        before = snapshot_tree(wt)
        (sub / "x.txt").write_text("x\n")
        after = snapshot_tree(wt)
        self.assertEqual(self.changes(wt, before, after)["pkg/inner/x.txt"].change, "added")
        undo(wt, before, after)
        self.assertFalse((sub / "x.txt").exists())
        self.assertEqual(git_path(repo, "index").read_bytes(), main_index)

    def test_embedded_repo_is_gitlink(self):
        repo = make_repo(self.tmp)
        inner = make_repo(os.path.join(self.tmp, "x"))
        shutil.move(str(inner), str(repo / "vendored"))
        sha = snapshot_tree(repo)
        out = run_git(repo, "ls-tree", sha, "vendored").decode()
        self.assertTrue(out.startswith("160000 commit"), out)

    def test_temp_index_removed(self):
        repo = make_repo(self.tmp)
        tdir = Path(self.tmp) / "tmpdir"
        tdir.mkdir()
        with mock.patch.dict(os.environ, {"TMPDIR": str(tdir)}):
            tempfile.tempdir = None
            try:
                snapshot_tree(repo)
            finally:
                tempfile.tempdir = None
        self.assertEqual(list(tdir.iterdir()), [])

    def _hostile_config(self):
        script = Path(self.tmp) / "conv.sh"
        script.write_text("#!/bin/sh\necho CONVERTED\n")
        script.chmod(0o755)
        Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text(
            "[diff]\n\tnoprefix = true\n\texternal = /usr/bin/true\n"
            "[apply]\n\twhitespace = fix\n[color]\n\tui = always\n"
            '[diff "conv"]\n\ttextconv = %s\n' % script
        )

    def test_hostile_global_config(self):
        repo = make_repo(self.tmp)
        (repo / "src").mkdir()
        (repo / "src" / "util.py").write_text("x = 1  \n")
        (repo / "data.bin").write_bytes(b"\x00\x01bin\n")
        (repo / ".gitattributes").write_text("*.bin diff=conv\n")
        self._hostile_config()
        before = snapshot_tree(repo)
        (repo / "src" / "util.py").write_text("x = 2  \n")
        (repo / "data.bin").write_bytes(b"\x00\x02bin\n")
        after = snapshot_tree(repo)
        _, text, _ = diff_trees(repo, before, after)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("CONVERTED", text)
        self.assertIn("a/src/util.py", text)
        undo(repo, before, after)
        self.assertEqual(snapshot_tree(repo), before)
        self.assertEqual((repo / "src" / "util.py").read_text(), "x = 1  \n")
        self.assertEqual((repo / "data.bin").read_bytes(), b"\x00\x01bin\n")

    def test_gitignore_drift(self):
        repo = make_repo(self.tmp)
        (repo / ".gitignore").write_text(".env\nnode_modules/\n")
        run_git(repo, "add", ".gitignore")
        run_git(repo, "commit", "-q", "-m", "ign")
        (repo / ".env").write_text("SECRET_KEY=hunter2\n")
        (repo / "node_modules").mkdir()
        for i in range(300):
            (repo / "node_modules" / ("f%d.js" % i)).write_text("x\n")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        self.assertIn(".env", ign)
        self.assertIn("node_modules/", ign)
        (repo / ".gitignore").write_text("")
        (repo / "new.txt").write_text("n\n")
        after = snapshot_tree(repo, exclude=ign)
        changes, text, _ = diff_trees(repo, before, after)
        self.assertEqual({c.path for c in changes}, {".gitignore", "new.txt"})
        self.assertNotIn("hunter2", text)
        undo(repo, before, after, protect=ign)
        self.assertEqual((repo / ".env").read_text(), "SECRET_KEY=hunter2\n")
        self.assertTrue((repo / "node_modules" / "f299.js").exists())
        self.assertFalse((repo / "new.txt").exists())

    def test_exclude_special_names(self):
        repo = make_repo(self.tmp)
        (repo / ".gitignore").write_text("*.log\n")
        weird = ["we ird!#[x].log", " lead.log"]
        for w in weird:
            (repo / w).write_text("l\n")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        (repo / ".gitignore").write_text("")
        after = snapshot_tree(repo, exclude=ign)
        self.assertEqual({c.path for c in diff_trees(repo, before, after)[0]}, {".gitignore"})

    def _drift(self, files, gitignore="", global_ignore=None):
        repo = make_repo(self.tmp)
        if global_ignore:
            Path(os.environ["GIT_CONFIG_GLOBAL"]).write_text(
                "[core]\n\texcludesFile = %s\n" % global_ignore
            )
        for rel, data in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(data)
        return repo

    def test_ignored_dir_not_ignored_itself(self):
        gi = Path(self.tmp) / "global_ignore"
        gi.write_text(".DS_Store\n")
        repo = self._drift({"tools/.DS_Store": "x\n"}, global_ignore=gi)
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        self.assertEqual(ign, ["tools/.DS_Store"])
        (repo / "tools" / "cli.py").write_text("print(1)\n")
        after = snapshot_tree(repo, exclude=ign)
        self.assertEqual([c.path for c in diff_trees(repo, before, after)[0]], ["tools/cli.py"])
        undo(repo, before, after, protect=ign)
        self.assertFalse((repo / "tools" / "cli.py").exists())
        self.assertTrue((repo / "tools" / ".DS_Store").exists())

    def test_nested_ignored_file_only(self):
        repo = self._drift({".gitignore": "*.key\n", "src/conf/prod.key": "k\n"})
        run_git(repo, "add", ".gitignore")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        self.assertEqual(ign, ["src/conf/prod.key"])
        (repo / "src" / "conf" / "app.cfg").write_text("c\n")
        after = snapshot_tree(repo, exclude=ign)
        self.assertEqual([c.path for c in diff_trees(repo, before, after)[0]], ["src/conf/app.cfg"])

    def test_negation_written_by_run(self):
        repo = self._drift({".gitignore": ".env\n", ".env": "SECRET_KEY=hunter2\n"})
        run_git(repo, "add", ".gitignore")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        (repo / ".gitignore").write_text("!.env\n")
        after = snapshot_tree(repo, exclude=ign)
        changes, text, _ = diff_trees(repo, before, after)
        self.assertEqual([c.path for c in changes], [".gitignore"])
        self.assertNotIn("hunter2", text)
        undo(repo, before, after, protect=ign)
        self.assertEqual((repo / ".env").read_text(), "SECRET_KEY=hunter2\n")

    def test_backstop_survives_file_changing_after_add(self):
        import dllm.snapshot as snap

        repo = self._drift({".gitignore": ".env\n", ".env": "SECRET_KEY=hunter2\n"})
        run_git(repo, "add", ".gitignore")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        (repo / ".gitignore").write_text("!.env\n")
        real = snap._git

        def racing(r, args, *rest, **kw):
            out = real(r, args, *rest, **kw)
            if "add" in args:  # the run rewrites the file between git add and the backstop
                (repo / ".env").write_text("SECRET_KEY=rotated\n")
            return out

        with mock.patch.object(snap, "_git", racing):
            after = snapshot_tree(repo, exclude=ign)
        changes, text, _ = diff_trees(repo, before, after)
        self.assertEqual([c.path for c in changes], [".gitignore"])
        self.assertNotIn("SECRET_KEY", text)

    def test_trailing_space_filename(self):
        repo = self._drift({".gitignore": "*.tmp*\n", "a.tmp  ": "t\n"})
        run_git(repo, "add", ".gitignore")
        before = snapshot_tree(repo)
        ign = ignored_paths(repo)
        self.assertEqual(ign, ["a.tmp  "])
        (repo / ".gitignore").write_text("")
        after = snapshot_tree(repo, exclude=ign)
        self.assertEqual([c.path for c in diff_trees(repo, before, after)[0]], [".gitignore"])

    def test_protect_refuses(self):
        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "gen").mkdir()
        (repo / "gen" / "o.txt").write_text("o\n")
        (repo / "gen.txt").write_text("g\n")
        after = snapshot_tree(repo)
        with self.assertRaises(SnapshotError) as cm:
            undo(repo, before, after, protect=["gen.txt"])
        self.assertIn("gen.txt", str(cm.exception))
        with self.assertRaises(SnapshotError):
            undo(repo, before, after, protect=["gen/"])
        self.assertTrue((repo / "gen.txt").exists())
        self.assertTrue((repo / "gen" / "o.txt").exists())

    def test_racy_index_not_lost(self):
        repo = make_repo(self.tmp)
        (repo / "r.txt").write_text("aaaa\n")
        run_git(repo, "add", "r.txt")
        (repo / "r.txt").write_text("bbbb\n")
        time.sleep(1.1)  # make the real index older than "now" so a fresh-mtime copy would hide the edit
        before = snapshot_tree(repo)
        (repo / "r.txt").write_text("cccc\n")
        after = snapshot_tree(repo)
        undo(repo, before, after)
        self.assertEqual((repo / "r.txt").read_text(), "bbbb\n")

    def test_subdir_covers_whole_repo(self):
        repo = make_repo(self.tmp)
        (repo / "pkg").mkdir()
        (repo / "pkg" / "m.py").write_text("m\n")
        sub = repo / "pkg"
        before = snapshot_tree(sub)
        self.assertEqual(before, snapshot_tree(repo))
        (repo / "pkg" / "m.py").write_text("m2\n")
        (repo / "top.txt").write_text("t\n")
        after = snapshot_tree(sub)
        self.assertEqual({c.path for c in diff_trees(sub, before, after)[0]}, {"pkg/m.py", "top.txt"})
        undo(sub, before, after)
        self.assertFalse((repo / "top.txt").exists())
        self.assertEqual((repo / "pkg" / "m.py").read_text(), "m\n")

    def test_empty_patch_with_different_trees_raises(self):
        import dllm.snapshot as snap

        repo = make_repo(self.tmp)
        before = snapshot_tree(repo)
        (repo / "z.txt").write_text("z\n")
        after = snapshot_tree(repo)
        real = snap._git

        def fake(r, args, env=None, input=None):
            if args and args[0] == "diff-tree" and "-p" in args:
                return b""
            return real(r, args, env=env, input=input)

        with mock.patch.object(snap, "_git", fake):
            with self.assertRaises(SnapshotError):
                undo(repo, before, after)
        self.assertTrue((repo / "z.txt").exists())

    def test_inherited_git_dir_ignored(self):
        repo = make_repo(self.tmp)
        with mock.patch.dict(os.environ, {"GIT_DIR": os.path.join(self.tmp, "nope"), "GIT_INDEX_FILE": os.path.join(self.tmp, "nope.idx")}):
            self.assertEqual(len(snapshot_tree(repo)), 40)


if __name__ == "__main__":
    unittest.main()
