"""Git tree snapshots, diffs and undo that never touch the real index, HEAD, refs or stash."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_TIMEOUT = 60
_TRUNC_NOTE = "\n... [diff truncated]\n"


class SnapshotError(Exception):
    pass


@dataclass
class FileChange:
    path: str
    change: str  # "added" | "modified" | "deleted"
    added: int
    removed: int


def _git(
    repo: Path,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    input: Optional[bytes] = None,
) -> bytes:
    cmd = ["git", "-c", "core.quotepath=false", "-c", "core.hooksPath=/dev/null"] + list(args)
    full_env = dict(os.environ)
    full_env["GIT_OPTIONAL_LOCKS"] = "0"
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), env=full_env, input=input, capture_output=True, timeout=_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise SnapshotError("git %s timed out after %d s" % (args[0], _TIMEOUT))
    except OSError as exc:
        raise SnapshotError("could not run git: %s" % exc)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise SnapshotError("git %s failed: %s" % (args[0], stderr))
    return proc.stdout


def repo_root(workdir: Path) -> Path:
    try:
        out = _git(Path(workdir), ["rev-parse", "--show-toplevel"])
    except SnapshotError as exc:
        raise SnapshotError(
            "%s is not inside a git repository; run `git init` there first (%s)" % (workdir, exc)
        )
    return Path(out.decode("utf-8", "replace").strip())


def git_path(repo: Path, name: str) -> Path:
    out = _git(repo, ["rev-parse", "--git-path", name]).decode("utf-8", "replace").strip()
    p = Path(out)
    return p if p.is_absolute() else Path(repo) / p


def snapshot_tree(repo: Path) -> str:
    """Tree SHA of the whole working tree (tracked plus untracked, non-ignored)."""
    tmpdir = tempfile.mkdtemp(prefix="dllm-index-")
    try:
        index = os.path.join(tmpdir, "index")
        real = git_path(repo, "index")
        if real.exists():
            shutil.copyfile(str(real), index)
        env = {"GIT_INDEX_FILE": index}
        _git(repo, ["add", "-A"], env=env)
        return _git(repo, ["write-tree"], env=env).decode().strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def diff_trees(
    repo: Path, before: str, after: str, max_chars: int = 20_000
) -> Tuple[List[FileChange], str, bool]:
    status_out = _git(repo, ["diff", "--no-renames", "--name-status", "-z", before, after])
    parts = status_out.decode("utf-8", "replace").split("\0")
    kinds: Dict[str, str] = {}
    order: List[str] = []
    for i in range(0, len(parts) - 1, 2):
        code, path = parts[i], parts[i + 1]
        kinds[path] = "added" if code.startswith("A") else "deleted" if code.startswith("D") else "modified"
        order.append(path)

    counts: Dict[str, Tuple[int, int]] = {}
    num_out = _git(repo, ["diff", "--no-renames", "--numstat", "-z", before, after])
    for rec in num_out.decode("utf-8", "replace").split("\0"):
        if not rec:
            continue
        a, r, path = rec.split("\t", 2)
        counts[path] = (-1, -1) if a == "-" else (int(a), int(r))

    changes = [FileChange(p, kinds[p], *counts.get(p, (0, 0))) for p in order]

    text = _git(repo, ["diff", "--no-renames", before, after]).decode("utf-8", "replace")
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + _TRUNC_NOTE
    return changes, text, truncated


_CONFLICT_PATTERNS = [
    re.compile(r"^error: patch failed: (.+?):\d+$"),
    re.compile(r"^error: (.+?): (?:patch does not apply|No such file or directory|already exists in working directory|does not match index)$"),
]


def undo(repo: Path, before: str, after: str) -> None:
    if before == after:
        return
    patch = _git(repo, ["diff", "--binary", "--no-renames", after, before])
    if not patch:
        return
    try:
        _git(repo, ["apply", "--check"], input=patch)
    except SnapshotError as exc:
        paths: List[str] = []
        for line in str(exc).splitlines():
            for pat in _CONFLICT_PATTERNS:
                m = pat.match(line.strip())
                if m and m.group(1) not in paths:
                    paths.append(m.group(1))
        detail = "conflicting files: %s" % ", ".join(paths) if paths else str(exc)
        raise SnapshotError("cannot undo, files changed since the run (%s)" % detail)
    _git(repo, ["apply"], input=patch)
