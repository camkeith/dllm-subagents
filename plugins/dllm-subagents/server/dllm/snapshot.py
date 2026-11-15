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
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

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
_secret_env: FrozenSet[str] = frozenset()  # provider key variables, never passed to git


def set_secret_env_names(names: Iterable[str]) -> None:
    """Name the environment variables (the providers' API key variables) to keep from git.

    Process-wide: it replaces the list for every later git command in this process.
    RunManager sets it from its settings when it is created.
    """
    global _secret_env
    _secret_env = frozenset(names)


class SnapshotError(Exception):
    pass


@dataclass
class FileChange:
    path: str
    change: str  # "added" | "modified" | "deleted"
    added: int
    removed: int


def _command(args: List[str], env: Optional[Dict[str, str]]) -> Tuple[List[str], Dict[str, str]]:
    cmd = ["git", "-c", "core.quotepath=false", "-c", "core.hooksPath=/dev/null",
           "-c", "core.fsmonitor=false"] + list(args)
    full_env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SCRUB_ENV and k not in _secret_env
        and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }
    full_env["GIT_OPTIONAL_LOCKS"] = "0"
    if env:
        full_env.update(env)
    return cmd, full_env


def _subcommand(args: Sequence[str]) -> str:
    """The git subcommand in args, for messages: the first word that is not an option or a -c value."""
    after_c = False
    for a in args:
        if after_c:
            after_c = False
        elif a == "-c":
            after_c = True
        elif not a.startswith("-"):
            return a
    return args[0] if args else ""


def _git(
    repo: Path,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    input: Optional[bytes] = None,
    ok_codes: Sequence[int] = (0,),
) -> bytes:
    cmd, full_env = _command(args, env)
    stdin = {"input": input} if input is not None else {"stdin": subprocess.DEVNULL}
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo), env=full_env, capture_output=True, timeout=_TIMEOUT, **stdin
        )
    except subprocess.TimeoutExpired:
        raise SnapshotError("git %s timed out after %d s" % (_subcommand(args), _TIMEOUT))
    except OSError as exc:
        raise SnapshotError("could not run git: %s" % exc)
    if proc.returncode not in ok_codes:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise SnapshotError("git %s failed: %s" % (_subcommand(args), stderr))
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
        timed_out = SnapshotError("git %s timed out after %d s" % (_subcommand(args), _TIMEOUT))
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
            raise SnapshotError("git %s failed: %s" % (_subcommand(args), stderr))
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


def _clear_assume_unchanged(repo: Path, env: Dict[str, str]) -> None:
    """Clear assume-unchanged bits in the temp index, so add -A sees edits to those files."""
    out = _git(repo, ["ls-files", "-v", "-z"], env=env)
    marked = [e[2:] for e in out.split(b"\0") if e[:1].islower()]  # lowercase tag: assume-unchanged
    if marked:
        _git(repo, ["-c", "core.splitIndex=false", "update-index", "--no-assume-unchanged", "-z", "--stdin"],
             env=env, input=b"\0".join(marked) + b"\0")


def _nested_repos(repo: Path, env: Dict[str, str], config: Sequence[str] = ()) -> List[str]:
    """Untracked, non-ignored directories holding their own git repository (ending with '/').

    add -A would store them as gitlinks, or fail outright for a repository with no commit.
    `config` is leading git options (`-c name=value`) that change what counts as ignored.
    """
    out = _git(repo, list(config) + ["ls-files", "-z", "--others", "--exclude-standard"], env=env)
    listed = out.decode("utf-8", "surrogateescape").split("\0")
    return [p for p in listed if p.endswith("/") and os.path.lexists(os.path.join(str(repo), p, ".git"))]


def _keep_before_files(repo: Path, env: Dict[str, str], keep_from: str, protect: Sequence[str],
                       tmpdir: str) -> None:
    """Put back files of the before tree that add -A left out although they are still on disk.

    The run ignored them (for example by adding `dep/` to .gitignore). Without this they would
    show as deleted, and undo would fail trying to recreate files that still exist.
    """
    listed = _git(repo, ["ls-tree", "-r", "-z", "--name-only", keep_from])
    present = set(_git(repo, ["ls-files", "-z"], env=env).split(b"\0"))
    root = os.fsencode(str(repo))
    real_root = os.path.realpath(root)
    # _protected's rules as a set of exact paths and a tuple of directory prefixes: O(1) per file.
    exact, prefixes = set(), []
    for p in protect:
        b = p.encode("utf-8", "surrogateescape")
        exact.update((b, b.rstrip(b"/")))
        if b.endswith(b"/"):
            prefixes.append(b)
    dir_prefixes = tuple(prefixes)
    keep = []
    for raw in listed.split(b"\0"):
        if not raw or raw in present:
            continue
        full = os.path.join(root, raw)
        if not os.path.lexists(full) or raw in exact or raw.startswith(dir_prefixes):
            continue  # deleted, or excluded
        parent = os.path.dirname(raw)
        if os.path.isdir(full) and not os.path.islink(full):
            continue  # now a directory: force-adding it would pull in ignored files
        if parent and os.path.realpath(os.path.join(root, parent)) != os.path.join(real_root, parent):
            continue  # a parent is now a symlink; git refuses paths beyond one
        keep.append(raw)
    if keep:
        lst = os.path.join(tmpdir, "keep.lst")
        with open(lst, "wb") as fh:
            fh.write(b"\0".join(keep) + b"\0")
        _git(
            repo,
            ["-c", "core.splitIndex=false", "add", "-f", "--pathspec-from-file=" + lst, "--pathspec-file-nul"],
            env=dict(env, GIT_LITERAL_PATHSPECS="1"),
        )
        # add -f exits 0 but adds nothing for a path inside a nested repository (the run ignored
        # tools/ and ran `git init tools`, say): put those entries back from the before tree.
        # Only those: with core.ignorecase a case-only rename also looks skipped, and putting
        # the old name back next to the new one would make undo delete the file.
        added = set(_git(repo, ["ls-files", "-z"], env=env).split(b"\0"))
        skipped = [raw.decode("utf-8", "surrogateescape") for raw in keep
                   if raw not in added and _under_nested_repo(root, raw)]
        if skipped:
            _restore_nested_entries(repo, env, keep_from, skipped)


def _under_nested_repo(root: bytes, raw: bytes) -> bool:
    """Whether a directory between the repo root and raw holds its own .git."""
    parent = os.path.dirname(raw)
    while parent:
        if os.path.lexists(os.path.join(root, parent, b".git")):
            return True
        parent = os.path.dirname(parent)
    return False


def _restore_nested_entries(repo: Path, env: Dict[str, str], keep_from: str, nested: Sequence[str]) -> None:
    """Put back, unchanged, the before tree's entries under directories that are now nested
    repositories (the run ran `git init` in one, say), or at file paths inside one: they did
    not change, so they must not show as deleted, and undo must not touch them."""
    listed = _git(repo, ["ls-tree", "-r", "-z", keep_from, "--"] + [p.rstrip("/") for p in nested],
                  env=dict(env, GIT_LITERAL_PATHSPECS="1"))
    if listed:  # ls-tree's "<mode> <type> <sha>\t<path>" lines are an --index-info format
        _git(repo, ["-c", "core.splitIndex=false", "update-index", "-z", "--index-info"], env=env, input=listed)


def take_snapshot(
    repo: Path, exclude: Sequence[str] = (), keep_from: Optional[str] = None
) -> Tuple[str, List[str]]:
    """Tree SHA of the whole working tree (tracked plus untracked, non-ignored), and the untracked
    nested git repositories left out of it (directories, ending with '/').

    `exclude` lists extra paths (from ignored_paths at before-time) to leave out, so that
    ignore-rule edits made during a run cannot pull previously ignored files into the tree.
    `keep_from` is the before tree: its files that the run's ignore-rule edits would drop are
    kept, so that ignoring a file is not reported as deleting it.
    """
    repo = repo_root(repo)
    tmpdir = tempfile.mkdtemp(prefix="dllm-index-")
    try:
        index = os.path.join(tmpdir, "index")
        real = git_path(repo, "index")
        if real.exists():
            shutil.copy2(str(real), index)  # keep mtime so racy-clean entries are re-hashed
        env = {"GIT_INDEX_FILE": index}
        _clear_assume_unchanged(repo, env)
        found = _nested_repos(repo, env)
        nested = [p for p in found if not _protected(p, exclude)]
        # A path that is a nested repo now is left out by pathspec only: git add refuses a
        # pathspec, even an exclude one, that names a path its exclude patterns ignore.
        still_nested = {p.rstrip("/") for p in found}
        patterns = [p for p in exclude if p.rstrip("/") not in still_nested]
        by_pathspec = found
        add = ["-c", "core.splitIndex=false"]
        if patterns:
            ex = os.path.join(tmpdir, "exclude")
            with open(ex, "wb") as fh:
                fh.write(_user_global_excludes(repo))
                for p in patterns:
                    fh.write((_exclude_pattern(p) + "\n").encode("utf-8", "surrogateescape"))
            excludes = ["-c", "core.excludesFile=" + ex]
            add += excludes
            # Which nested repos add -A will meet is git's call under the same ignore rules: a
            # `!` rule in .gitignore outranks the excludes file, and a nested repo inside a path
            # the excludes file ignores must not be named at all.
            by_pathspec = _nested_repos(repo, env, excludes) if found else []
        add += ["add", "-A"]
        add_env = env
        if by_pathspec:
            # Exclude pathspecs, unlike exclude patterns, cannot be undone by a `!` rule in .gitignore.
            specs = os.path.join(tmpdir, "add.lst")
            with open(specs, "wb") as fh:
                fh.write(b".\0")
                for p in by_pathspec:
                    fh.write(b":(exclude,literal)" + p.rstrip("/").encode("utf-8", "surrogateescape") + b"\0")
            add += ["--pathspec-from-file=" + specs, "--pathspec-file-nul"]
            add_env = dict(env, GIT_LITERAL_PATHSPECS="0")  # the magic must not be read literally
        _git(repo, add, env=add_env)
        if exclude:
            # Backstop: drop before-ignored paths even if the run un-ignored them (e.g. a
            # `!path` rule in .gitignore), so they never reach the tree.
            lst = os.path.join(tmpdir, "ignored.lst")
            with open(lst, "wb") as fh:
                for p in exclude:
                    fh.write(p.encode("utf-8", "surrogateescape") + b"\0")
            _git(
                repo,
                ["rm", "--cached", "-f", "-r", "-q", "--ignore-unmatch",
                 "--pathspec-from-file=" + lst, "--pathspec-file-nul"],
                env=dict(env, GIT_LITERAL_PATHSPECS="1"),
            )
        if keep_from:
            if found:
                _restore_nested_entries(repo, env, keep_from, found)
            _keep_before_files(repo, env, keep_from, list(exclude) + found, tmpdir)
        tree = _git(repo, ["-c", "core.splitIndex=false", "write-tree"], env=env).decode().strip()
        return tree, nested
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def snapshot_tree(repo: Path, exclude: Sequence[str] = (), keep_from: Optional[str] = None) -> str:
    """The tree SHA of take_snapshot."""
    return take_snapshot(repo, exclude, keep_from)[0]


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


def _trees_exist(repo: Path, shas: Sequence[str]) -> bool:
    out = _git(repo, ["cat-file", "--batch-check"], input="".join(s + "^{tree}\n" for s in shas).encode())
    return not any(line.endswith(" missing") for line in out.decode("utf-8", "replace").splitlines())


def undo(repo: Path, before: str, after: str, protect: Sequence[str] = ()) -> None:
    if before == after:
        return
    repo = repo_root(repo)
    if not _trees_exist(repo, (before, after)):
        raise SnapshotError("the run's snapshots were pruned by git gc; undo is no longer possible")
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
        text = str(exc)
        prefix = "git apply failed: "  # added by _git in front of git's first line
        if text.startswith(prefix):
            text = text[len(prefix):]
        paths: List[str] = []
        for line in text.splitlines():
            for pat in _CONFLICT_PATTERNS:
                m = pat.match(line.strip())
                if m and m.group(1) not in paths:
                    paths.append(m.group(1))
        detail = "conflicting files: %s" % ", ".join(paths) if paths else str(exc)
        raise SnapshotError("cannot undo, files changed since the run (%s)" % detail)
    # git apply can still fail partway after --check passed (it accepts creating a file where a
    # directory now is, then cannot write it), so back up what it touches and roll back on failure.
    names = [n for n in _git(repo, _TREE_DIFF + ["--name-only", "-z", after, before]).split(b"\0") if n]
    backup_dir = tempfile.mkdtemp(prefix="dllm-undo-")
    keep_backup = False
    try:
        saved, created_dirs = _back_up(repo, names, backup_dir)
        try:
            _git(repo, _APPLY, input=patch)
        except SnapshotError as exc:
            try:
                _roll_back(saved, created_dirs)
            except OSError as rollback_exc:
                keep_backup = True
                raise SnapshotError(
                    "undo failed partway (%s) and putting the files back also failed (%s); copies of "
                    "the files as they were before the undo are in %s (its MANIFEST.txt says which "
                    "path each numbered copy is)" % (exc, rollback_exc, backup_dir)
                )
            raise SnapshotError("cannot undo, applying the reverse patch failed (%s); nothing was changed" % exc)
    finally:
        if not keep_backup:
            shutil.rmtree(backup_dir, ignore_errors=True)


def _back_up(repo: Path, names: Sequence[bytes], backup_dir: str) -> Tuple[list, List[bytes]]:
    """Copy each path (file, symlink or directory) to backup_dir, or note that it is missing.

    Returns (saved, created_dirs): saved holds (path, kind, data) with kind "file", "link",
    "dir" or "missing"; created_dirs are the missing parent directories git apply may create.
    Also writes backup_dir/MANIFEST.txt, one `<number>\\t<kind>\\t<path>` line per entry (a
    link's target after one more tab), so the copies can be put back by hand if need be.
    """
    root = os.fsencode(str(repo))
    store = os.fsencode(backup_dir)
    saved, created_dirs, manifest = [], [], []
    for i, raw in enumerate(names):
        full = os.path.join(root, raw)
        copy = os.path.join(store, str(i).encode())
        try:
            if os.path.islink(full):
                saved.append((full, "link", os.readlink(full)))
            elif os.path.isdir(full):
                shutil.copytree(full, copy, symlinks=True)
                saved.append((full, "dir", copy))
            elif os.path.isfile(full):
                shutil.copy2(full, copy)
                saved.append((full, "file", copy))
            elif os.path.lexists(full):
                raise SnapshotError("cannot undo: %s is not a regular file, directory or symlink"
                                    % raw.decode("utf-8", "replace"))
            else:
                saved.append((full, "missing", None))
                parent = os.path.dirname(full)
                while len(parent) > len(root) and not os.path.lexists(parent):
                    created_dirs.append(parent)
                    parent = os.path.dirname(parent)
        except OSError as exc:  # shutil.Error too: a FIFO, socket or unreadable file in a directory
            raise SnapshotError("cannot undo: could not back up %s before undoing (%s); nothing was changed"
                                % (raw.decode("utf-8", "replace"), _os_reason(exc, root)))
        _, kind, data = saved[-1]
        manifest.append(b"%d\t%s\t%s%s\n" % (i, kind.encode(), raw, b"\t" + data if kind == "link" else b""))
    try:
        with open(os.path.join(store, b"MANIFEST.txt"), "wb") as fh:
            fh.write(b"".join(manifest))
    except OSError as exc:
        raise SnapshotError("cannot undo: could not write the backup's MANIFEST.txt (%s); nothing was changed"
                            % _os_reason(exc, root))
    return saved, created_dirs


def _os_reason(exc: OSError, root: bytes) -> str:
    """exc as text, with the repo's byte paths shown relative and decoded."""
    if isinstance(exc, shutil.Error) and exc.args and isinstance(exc.args[0], list):
        reasons = []
        for src, _, why in exc.args[0]:  # copytree's (source, destination, reason) for each failure
            if isinstance(src, bytes):
                why = str(why).replace(repr(src), os.path.relpath(src, root).decode("utf-8", "replace"))
            reasons.append(str(why))
        return "; ".join(reasons)
    return exc.strerror or str(exc)


def _remove(path: bytes) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def _roll_back(saved: list, created_dirs: Sequence[bytes]) -> None:
    """Put every backed-up path back exactly as it was (content, mode, type)."""
    for full, _, _ in saved:
        _remove(full)
    for full, kind, data in sorted(saved, key=lambda s: len(s[0])):  # parents before children
        if kind == "missing":
            continue
        os.makedirs(os.path.dirname(full), exist_ok=True)
        if kind == "dir":
            shutil.copytree(data, full, symlinks=True, dirs_exist_ok=True)
            continue
        _remove(full)  # a parent restored above may have put it back already
        if kind == "link":
            os.symlink(data, full)
        else:
            shutil.copy2(data, full)
    for d in sorted(set(created_dirs), key=len, reverse=True):
        try:
            os.rmdir(d)  # only if git apply created it and it is empty again
        except OSError:
            pass
