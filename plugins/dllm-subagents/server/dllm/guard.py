"""Guardrails for the subagent's bash tool.

This is a blocklist that stops common footguns a small model might hit (a stray
`git push`, `rm -rf ~`, `curl | sh`). It is NOT a security boundary: a determined
caller can get around it (variable indirection, scripts written to disk, and so
on). It is pure (no filesystem writes) and never raises.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import FrozenSet, List, Optional, Tuple

from .config import config_dir

Cwds = FrozenSet[Optional[str]]  # directories a command segment may run in; None means unknown

KEY_FILE_REASON = "reading the plugin's key file is not allowed"
READONLY_GIT_COMMANDS = ("status", "diff", "log", "show", "ls-files", "grep", "blame", "rev-parse")
READONLY_GIT = frozenset(READONLY_GIT_COMMANDS)

_MAX_DEPTH = 6
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_FETCHERS = frozenset({"curl", "wget"})
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Words that may precede the real command. Value: option letters that take an argument.
_KEYWORDS = frozenset({"{", "}", "!", "if", "then", "else", "elif", "do", "while", "until", "time"})
_WRAPPERS = {
    "env": {"-u", "-C", "-S"},
    "command": set(),
    "builtin": set(),
    "nohup": set(),
    "exec": {"-a"},
    "nice": {"-n"},
    "timeout": {"-s", "-k"},
    "xargs": {"-n", "-I", "-L", "-P", "-d", "-E", "-s", "-a", "-l", "-i"},
}
_GIT_ARG_OPTS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env"})
_GIT_LIST = ", ".join(READONLY_GIT_COMMANDS)
_ROOTISH = ("/", "~", "$HOME", "${HOME}")
_PRIV = frozenset({"sudo", "doas"})
_POWER = frozenset({"reboot", "shutdown", "halt", "poweroff"})


def _parse_error(detail: str = "") -> str:
    return "could not parse the command%s; rewrite it more simply" % (" (%s)" % detail if detail else "")


def _git_reason(sub: str) -> str:
    return "git %s is not allowed (only read-only git commands: %s)" % (sub, _GIT_LIST)


_HEREDOC = re.compile(r"<<(-?)[ \t]*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")


def _feeds_shell(prefix: str) -> bool:
    """True if the command being built in prefix (last segment) is a shell reading stdin."""
    chunk = re.split(r"[;&|(\n]", prefix)[-1]
    try:
        words = _skip_wrappers(shlex.split(chunk))
    except ValueError:
        return False
    if not words or os.path.basename(words[0]) not in _SHELLS:
        return False
    return not any(a.startswith("-") and not a.startswith("--") and "c" in a[1:] for a in words[1:])


def _prepare(command: str) -> Tuple[str, List[str]]:
    """Prepare text for shlex.

    Turns unquoted newlines into `;`, strips comments and heredoc bodies, and
    collects command-substitution bodies (and bodies of heredocs that feed a
    shell, which are commands).
    """
    out: List[str] = []
    bodies: List[str] = []
    pending: List[Tuple[str, bool, bool]] = []  # (terminator, dash, feeds_shell)
    n = len(command)
    i = 0
    quote = ""
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = ""
            out.append(c)
        elif c == "\\" and i + 1 < n:
            out.append(c + command[i + 1])
            i += 2
            continue
        elif c == "#" and not quote and (i == 0 or command[i - 1] in " \t\n;|&("):
            while i < n and command[i] != "\n":
                i += 1
            continue
        elif c == "<" and not quote and command.startswith("<<", i) and not command.startswith("<<<", i):
            m = _HEREDOC.match(command, i)
            if m:
                pending.append((m.group(3), m.group(1) == "-", _feeds_shell("".join(out))))
            out.append("<<")
            i += 2
            continue
        elif c == "`":
            j = i + 1
            while j < n and command[j] != "`":
                j += 2 if command[j] == "\\" else 1
            bodies.append(command[i + 1:j])
            out.append(command[i:j + 1])
            i = j + 1
            continue
        elif c == "(" and i > 0 and command[i - 1] in "$<>" and quote != "'":
            depth, j = 1, i + 1
            while j < n and depth:
                if command[j] == "\\":
                    j += 1
                elif command[j] == "(":
                    depth += 1
                elif command[j] == ")":
                    depth -= 1
                j += 1
            bodies.append(command[i + 1:j - 1 if depth == 0 else j])
            out.append(c)
        elif c == "'" and not quote:
            quote = "'"
            out.append(c)
        elif c == '"':
            quote = "" if quote == '"' else ('"' if not quote else quote)
            out.append(c)
        elif c == "\n" and not quote:
            out.append(" ; ")
            for term, dash, shell in pending:
                body: List[str] = []
                i += 1
                while i < n:
                    j = command.find("\n", i)
                    line = command[i:] if j < 0 else command[i:j]
                    if (line.lstrip("\t") if dash else line) == term:
                        i = n if j < 0 else j
                        break
                    body.append(line)
                    i = n if j < 0 else j + 1
                else:
                    i = n
                if shell:
                    bodies.append("\n".join(body))
            pending = []
            if i >= n:
                break
        else:
            out.append(c)
        i += 1
    return "".join(out), bodies


def _is_sep(tok: str) -> bool:
    return bool(tok) and all(ch in ";|&()" for ch in tok)


def _segments(tokens: List[str]) -> List[Tuple[str, List[str]]]:
    """Split into (separator_before, words) pairs."""
    segs: List[Tuple[str, List[str]]] = []
    sep, cur = "", []
    for tok in tokens:
        if _is_sep(tok):
            segs.append((sep, cur))
            sep, cur = tok, []
        else:
            cur.append(tok)
    segs.append((sep, cur))
    return segs


def _skip_wrappers(words: List[str]) -> List[str]:
    """Drop leading NAME=value words, shell keywords and wrapper commands."""
    i, n = 0, len(words)
    while i < n:
        w = words[i]
        base = os.path.basename(w)
        if _ASSIGN.match(w) or w in _KEYWORDS:
            i += 1
        elif base in _WRAPPERS:
            argopts = _WRAPPERS[base]
            i += 1
            if base == "timeout":
                while i < n and words[i].startswith("-"):
                    i += 2 if words[i] in argopts else 1
                i += 1  # the duration
                continue
            while i < n and (words[i].startswith("-") or (base == "env" and _ASSIGN.match(words[i]))):
                i += 2 if words[i] in argopts else 1
        else:
            break
    return words[i:]


def _git_subcommand(args: List[str]) -> Optional[str]:
    i, n = 0, len(args)
    while i < n:
        a = args[i]
        if a in _GIT_ARG_OPTS:
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return a
    return None


def _rm_reason(args: List[str], workdir: str, cwds: Cwds) -> Optional[str]:
    recursive = False
    targets: List[str] = []
    opts_done = False
    for a in args:
        if not opts_done and a == "--":
            opts_done = True
        elif not opts_done and a.startswith("--"):
            recursive = recursive or a == "--recursive"
        elif not opts_done and a.startswith("-") and len(a) > 1:
            recursive = recursive or "r" in a[1:] or "R" in a[1:]
        else:
            targets.append(a)
    if not recursive:
        return None
    root = os.path.realpath(workdir)
    bases = [c for c in cwds if c is not None] or [root]
    for t in targets:
        if t in _ROOTISH or t.startswith("~/") or t.startswith("$HOME/") or t.startswith("${HOME}/") or t == "~":
            return "recursive rm of %s is not allowed (only paths inside the working directory)" % t
        if "$" in t or "`" in t:
            return ("recursive rm of %s is not allowed (a variable or command substitution can expand "
                    "to anything, even to nothing; name the path literally)" % t)
        if None in cwds and not os.path.isabs(t):
            return "recursive rm of %s is not allowed (the directory after cd is unknown; use paths inside the working directory)" % t
        outside = 0  # the target must be inside from every directory the rm may run in
        for base in bases:
            path = os.path.realpath(os.path.join(base, t))
            if path == root:
                return "recursive rm of the working directory itself is not allowed (name paths inside it)"
            if not path.startswith(root.rstrip(os.sep) + os.sep):
                outside += 1
        if outside and outside < len(bases):
            return ("recursive rm of %s is not allowed (an earlier cd may have failed or gone through a "
                    "symlink, and then it is outside the working directory; join them with && so rm runs "
                    "only after the cd worked, as in `cd dir && rm -rf ...`, and do not use `..` after a "
                    "cd through a symlink, or use `cd -P`)" % t)
        if outside:
            return "recursive rm of %s is not allowed (only paths inside the working directory)" % t
    return None


def _words_reason(words: List[str], workdir: str, depth: int, cwds: Cwds) -> Optional[str]:
    words = _skip_wrappers(words)
    if not words:
        return None
    cmd = os.path.basename(words[0])
    args = words[1:]
    if cmd in _PRIV:
        return "%s is not allowed (no privilege escalation)" % cmd
    if cmd in _POWER:
        return "%s is not allowed" % cmd
    if cmd.startswith("mkfs"):
        return "%s is not allowed (no formatting filesystems)" % cmd
    if cmd == "dd" and any(a.startswith("of=/dev/") for a in args):
        return "dd to a device is not allowed"
    if cmd == "git":
        sub = _git_subcommand(args)
        if sub is not None and sub not in READONLY_GIT:
            return _git_reason(sub)
    elif cmd == "rm":
        return _rm_reason(args, workdir, cwds)
    elif cmd in _SHELLS:
        for i, a in enumerate(args):
            if a.startswith("-") and not a.startswith("--") and "c" in a[1:] and i + 1 < len(args):
                return _check(args[i + 1], workdir, depth + 1, cwds)
        if "<(" in args and any(os.path.basename(a) in _FETCHERS for a in args[args.index("<(") + 1:]):
            return "running downloaded scripts is not allowed (download to a file and read it first)"
    elif cmd == "eval":
        return _check(" ".join(args), workdir, depth + 1, cwds)
    elif cmd == "find":
        for i, a in enumerate(args):
            if a in ("-exec", "-execdir", "-ok", "-okdir"):
                inner = []
                for b in args[i + 1:]:
                    if b in (";", "+"):
                        break
                    inner.append(b)
                reason = _words_reason(inner, workdir, depth, cwds)
                if reason:
                    return reason
    return None


def _next_cwd(cwd: Optional[str], name: str, args: List[str]) -> Optional[FrozenSet[str]]:
    """Directories `cd`/`pushd`/`popd args` run in cwd may land in; None when it cannot be known
    statically.

    cd is logical by default: `a/..` drops `a` from the path as typed, even when a is a symlink
    (bash may also fall back to the physical path), so both results are candidates. With -P
    (the last of -L and -P wins) symlinks are resolved first, so only the physical one is.
    """
    if name == "popd":
        return None
    physical = False
    for a in args:
        if a in ("-L", "-P"):
            physical = a == "-P"
    args = [a for a in args if a not in ("-L", "-P", "--")]
    if cwd is None or len(args) != 1:
        return None
    target = args[0]
    if target.startswith(("-", "+", "~")) or "$" in target or "`" in target:
        return None
    joined = os.path.join(cwd, target)
    if physical:
        return frozenset({os.path.realpath(joined)})
    return frozenset({os.path.normpath(joined), os.path.realpath(joined)})


def _op(sep: str) -> str:
    """The operator of a separator token, without its parentheses (`);` is `;`)."""
    return sep.replace("(", "").replace(")", "")


def _check(command: str, workdir: str, depth: int, cwds: Cwds) -> Optional[str]:
    """cwds: every directory the next segment may run in (None: unknown)."""
    if depth > _MAX_DEPTH:
        return _parse_error("nested too deeply")
    text, bodies = _prepare(command)
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        tokens = list(lex)
    except ValueError as exc:
        return _parse_error(str(exc).lower())
    segs = _segments(tokens)
    seen = set(cwds)  # every directory any segment may run in, for substitution bodies
    saved: List[Cwds] = []
    before_cd: Optional[Cwds] = None  # directories from before `cd x &&`, back after its && chain
    prev_fetch = False
    for i, (sep, words) in enumerate(segs):
        if before_cd is not None and (_op(sep) != "&&" or ")" in sep):
            # The && chain after the cd ended: if the cd failed, the rest runs where it was before.
            cwds, before_cd = cwds | before_cd, None
        for ch in sep:  # a cd inside (...) may or may not still apply after it
            if ch == "(":
                saved.append(cwds)
            elif ch == ")" and saved:
                cwds = cwds | saved.pop()
        seen |= cwds
        reason = _words_reason(words, workdir, depth, cwds)
        if reason:
            return reason
        stripped = _skip_wrappers(words)
        name = os.path.basename(stripped[0]) if stripped else ""
        if name in ("cd", "pushd", "popd"):
            moved: Cwds = frozenset()
            for c in cwds:
                landed = _next_cwd(c, name, stripped[1:])
                moved |= landed if landed is not None else frozenset({None})
            after = _op(segs[i + 1][0]) if i + 1 < len(segs) else ""
            negated = "!" in words[:len(words) - len(stripped)]  # `! cd x && y` runs y if the cd failed
            # Only `cd x && next` runs next in x for sure (if the cd ran at all: not after || or |),
            # and only until that && chain ends. After ; || | & or a newline, the cd may have
            # failed, so next may run in either.
            if after == "&&" and _op(sep) not in ("||", "|", "|&") and not negated:
                before_cd = cwds if before_cd is None else before_cd | cwds
                cwds = moved
            else:
                cwds = cwds | moved
            seen |= cwds
        if prev_fetch and sep in ("|", "|&") and name in _SHELLS:
            return "piping a download into a shell is not allowed (download to a file and read it first)"
        prev_fetch = name in _FETCHERS
    for body in bodies:
        reason = _check(body, workdir, depth + 1, frozenset(seen))
        if reason:
            return reason
    return None


def check_command(command: str, workdir: str) -> Optional[str]:
    """Return a reason if the command is blocked, else None. Never raises."""
    try:
        if not isinstance(command, str):
            return _parse_error("not a string")
        if "keys.env" in command or str(config_dir()) in command:
            return KEY_FILE_REASON
        return _check(command, workdir, 0, frozenset({os.path.realpath(workdir)}))
    except Exception as exc:  # the guard must never take the run down
        return _parse_error(type(exc).__name__)
