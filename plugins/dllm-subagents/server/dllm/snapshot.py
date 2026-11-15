"""Git tree snapshots, diffs and undo that never touch the real index, HEAD, refs or stash."""
from __future__ import annotations

import codecs
import os
import re
import select
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .tools import redact as _redact
from .tools import scrub_edge

_TIMEOUT = 60
_TRUNC_NOTE = "\n... [diff truncated]\n"
_SCRUB_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
)


class SnapshotError(Exception):
    pass


@dataclass
class FileChange:
    path: str
    change: str  # "added" | "modified" | "deleted"
    added: int
    removed: int


def _command(args: List[str], env: Optional[Dict[str, str]]) -> Tuple[List[str], Dict[str, str]]:
    cmd = ["git", "-c", "core.quotepath=false", "-c", "core.hooksPath=/dev/null"] + list(args)
    full_env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SCRUB_ENV and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    full_env["GIT_OPTIONAL_LOCKS"] = "0"
    if env:
        full_env.update(env)
    return cmd, full_env


def _git(
    repo: Path,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    input: Optional[bytes] = None,
    ok_codes: Sequence[int] = (0,),
) -> bytes:
    cmd, full_env = _command(args, env)
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), env=full_env, input=input, capture_output=True, timeout=_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise SnapshotError("git %s timed out after %d s" % (args[0], _TIMEOUT))
    except OSError as exc:
        raise SnapshotError("could not run git: %s" % exc)
    if proc.returncode not in ok_codes:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise SnapshotError("git %s failed: %s" % (args[0], stderr))
    return proc.stdout


def _git_head(repo: Path, args: List[str], max_chars: int) -> Tuple[str, bool]:
    """The first max_chars characters of git's output, and whether it had more.

    Reads incrementally and stops git once past the limit, so a huge patch is never read whole.
    """
    cmd, full_env = _command(args, None)
    with tempfile.TemporaryFile() as err:  # a file, so a chatty stderr cannot block git
        try:
            proc = subprocess.Popen(cmd, cwd=str(repo), env=full_env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=err)
        except OSError as exc:
            raise SnapshotError("could not run git: %s" % exc)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        parts: List[str] = []
        size, more = 0, False
        deadline = time.monotonic() + _TIMEOUT
        timed_out = SnapshotError("git %s timed out after %d s" % (args[0], _TIMEOUT))
        try:
            fd = proc.stdout.fileno()
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([fd], [], [], left)[0]:
                    raise timed_out
                chunk = os.read(fd, 65536)
                parts.append(decoder.decode(chunk, final=not chunk))
                size += len(parts[-1])
                if not chunk:
                    break
                if size > max_chars:
                    more = True
                    break
            if not more:
                try:
                    proc.wait(max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    raise timed_out
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
        if not more and proc.returncode != 0:
            err.seek(0)
            stderr = err.read().decode("utf-8", "replace").strip()
            raise SnapshotError("git %s failed: %s" % (args[0], stderr))
    return "".join(parts)[:max_chars], more


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


def _user_global_excludes(repo: Path) -> bytes:
    try:
        out = _git(repo, ["config", "--path", "core.excludesFile"]).decode("utf-8", "replace").strip()
    except SnapshotError:
        out = ""
    candidates = []
    if out:
        candidates.append(os.path.join(str(repo), out))
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
        candidates.append(os.path.join(base, "git", "ignore"))
    for c in candidates:
        try:
            with open(c, "rb") as fh:
                data = fh.read()
            return data if data.endswith(b"\n") or not data else data + b"\n"
        except OSError:
            pass
    return b""


def _exclude_pattern(path: str) -> str:
    is_dir = path.endswith("/")
    body = path.rstrip("/") if is_dir else path
    esc = []
    for ch in body:
        if ch in "\\*?[!#":
            esc.append("\\" + ch)
        elif ch == "\n":
            esc.append("?")
        else:
            esc.append(ch)
    text = "".join(esc)
    if text.startswith(" "):
        text = "\\" + text
    stripped = text.rstrip(" ")
    text = stripped + "\\ " * (len(text) - len(stripped))
    return "/" + text + ("/" if is_dir else "")


def ignored_paths(repo: Path) -> List[str]:
    """Untracked paths currently ignored (directories end with '/'), without descending into them."""
    repo = repo_root(repo)
    out = _git(repo, ["ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory"])
    listed = [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]
    if not listed:
        return []
    # ls-files also lists untracked directories whose current contents are all ignored;
    # keep only entries that are themselves ignored (exit 1 means none are).
    checked = _git(
        repo,
        ["check-ignore", "-z", "--stdin"],
        input="\0".join(listed).encode("utf-8", "surrogateescape") + b"\0",
        ok_codes=(0, 1),
    )
    ignored = set(p for p in checked.decode("utf-8", "surrogateescape").split("\0") if p)
    return [p for p in listed if p in ignored]


def snapshot_tree(repo: Path, exclude: Sequence[str] = ()) -> str:
    """Tree SHA of the whole working tree (tracked plus untracked, non-ignored).

    `exclude` lists extra paths (from ignored_paths at before-time) to leave out, so that
    ignore-rule edits made during a run cannot pull previously ignored files into the tree.
    """
    repo = repo_root(repo)
    tmpdir = tempfile.mkdtemp(prefix="dllm-index-")
    try:
        index = os.path.join(tmpdir, "index")
        real = git_path(repo, "index")
        if real.exists():
            shutil.copy2(str(real), index)  # keep mtime so racy-clean entries are re-hashed
        env = {"GIT_INDEX_FILE": index}
        add = ["-c", "core.splitIndex=false"]
        if exclude:
            ex = os.path.join(tmpdir, "exclude")
            with open(ex, "wb") as fh:
                fh.write(_user_global_excludes(repo))
                for p in exclude:
                    fh.write((_exclude_pattern(p) + "\n").encode("utf-8", "surrogateescape"))
            add += ["-c", "core.excludesFile=" + ex]
        _git(repo, add + ["add", "-A"], env=env)
        if exclude:
            # Backstop: drop before-ignored paths even if the run un-ignored them (e.g. a
            # `!path` rule in .gitignore), so they never reach the tree.
            lst = os.path.join(tmpdir, "ignored.lst")
            with open(lst, "wb") as fh:
                for p in exclude:
                    fh.write(p.encode("utf-8", "surrogateescape") + b"\0")
            _git(
                repo,
                ["rm", "--cached", "-r", "-q", "--ignore-unmatch",
                 "--pathspec-from-file=" + lst, "--pathspec-file-nul"],
                env=dict(env, GIT_LITERAL_PATHSPECS="1"),
            )
        return _git(repo, ["-c", "core.splitIndex=false", "write-tree"], env=env).decode().strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_TREE_DIFF = ["diff-tree", "-r", "--no-renames"]


def diff_trees(
    repo: Path, before: str, after: str, max_chars: int = 20_000, redact: Sequence[str] = ()
) -> Tuple[List[FileChange], str, bool]:
    """File changes, the patch (secret values in `redact` replaced, cut to max_chars), and
    whether it was cut."""
    repo = repo_root(repo)
    status_out = _git(repo, _TREE_DIFF + ["--name-status", "-z", before, after])
    parts = status_out.decode("utf-8", "replace").split("\0")
    kinds: Dict[str, str] = {}
    order: List[str] = []
    for i in range(0, len(parts) - 1, 2):
        code, path = parts[i], parts[i + 1]
        kinds[path] = "added" if code.startswith("A") else "deleted" if code.startswith("D") else "modified"
        order.append(path)

    counts: Dict[str, Tuple[int, int]] = {}
    num_out = _git(repo, _TREE_DIFF + ["--numstat", "-z", before, after])
    for rec in num_out.decode("utf-8", "replace").split("\0"):
        if not rec:
            continue
        a, r, path = rec.split("\t", 2)
        counts[path] = (-1, -1) if a == "-" else (int(a), int(r))

    changes = [FileChange(p, kinds[p], *counts.get(p, (0, 0))) for p in order]

    # Read a secret's length past the cap, so one that crosses the cap is whole when redacted.
    margin = max((len(v) for v in redact if v), default=0)
    text, more = _git_head(
        repo,
        _TREE_DIFF + ["-p", "--no-color", "--no-ext-diff", "--no-textconv", before, after],
        max_chars + margin,
    )
    text = _redact(text, redact)
    truncated = more or len(text) > max_chars
    if truncated:
        text = scrub_edge(text[:max_chars], redact, at_end=True) + _TRUNC_NOTE
    return changes, text, truncated


_CONFLICT_PATTERNS = [
    re.compile(r"^error: patch failed: (.+?):\d+$"),
    re.compile(r"^error: (.+?): (?:patch does not apply|No such file or directory|already exists in working directory|does not match index)$"),
]
_APPLY = ["apply", "-p1", "--whitespace=nowarn", "--no-ignore-whitespace"]


def _protected(path: str, protect: Sequence[str]) -> bool:
    for p in protect:
        if path == p or path == p.rstrip("/") or (p.endswith("/") and path.startswith(p)):
            return True
    return False


def undo(repo: Path, before: str, after: str, protect: Sequence[str] = ()) -> None:
    if before == after:
        return
    repo = repo_root(repo)
    if protect:
        names = _git(repo, _TREE_DIFF + ["--name-only", "-z", after, before])
        hit = [n for n in names.decode("utf-8", "replace").split("\0") if n and _protected(n, protect)]
        if hit:
            raise SnapshotError(
                "refusing to undo: would modify protected (ignored) paths: %s" % ", ".join(hit)
            )
    patch = _git(
        repo,
        _TREE_DIFF + ["-p", "--binary", "--no-color", "--no-ext-diff", "--no-textconv", after, before],
    )
    if not patch:
        raise SnapshotError("cannot undo: trees differ but git produced an empty reverse patch")
    try:
        _git(repo, _APPLY + ["--check"], input=patch)
    except SnapshotError as exc:
        paths: List[str] = []
        for line in str(exc).splitlines():
            for pat in _CONFLICT_PATTERNS:
                m = pat.match(line.strip())
                if m and m.group(1) not in paths:
                    paths.append(m.group(1))
        detail = "conflicting files: %s" % ", ".join(paths) if paths else str(exc)
        raise SnapshotError("cannot undo, files changed since the run (%s)" % detail)
    _git(repo, _APPLY, input=patch)
