from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dllm.snapshot import (
    FileChange,
    SnapshotError,
    diff_trees,
    git_path,
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


if __name__ == "__main__":
    unittest.main()
