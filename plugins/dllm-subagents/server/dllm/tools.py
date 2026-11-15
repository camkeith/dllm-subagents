"""The tools the fast subagent can call, and their OpenAI function schemas."""
from __future__ import annotations

import fnmatch
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from .edit import EditError, apply_edits
from .guard import check_command
from .shell import CUT_MARKER_RE, run_command

READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE_CHARS = 2000
LINE_CUT_MARKER = "... [line cut]"
MAX_FILE_BYTES = 10 * 1024 * 1024


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "read",
        "Read a text file with line numbers. Use offset (1-based) and limit for large files.",
        {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        ["path"],
    ),
    _tool(
        "write",
        "Create or overwrite a file with content. Creates parent directories.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _tool(
        "edit",
        "Replace exact text in an existing file. Each old_text must match exactly one place "
        "in the file and edits must not overlap; all edits apply or none do.",
        {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"old_text": {"type": "string"}, "new_text": {"type": "string"}},
                    "required": ["old_text", "new_text"],
                },
            },
        },
        ["path", "edits"],
    ),
    _tool(
        "bash",
        "Run a shell command in the working directory. Timeout is in seconds.",
        {"command": {"type": "string"}, "timeout": {"type": "integer"}},
        ["command"],
    ),
    _tool(
        "finish",
        "End the task with a short summary of what changed and anything left undone.",
        {"summary": {"type": "string"}},
        ["summary"],
    ),
]


class ToolError(Exception):
    """A problem the model can fix by changing its call."""


@dataclass
class ToolContext:
    workdir: Path
    cancel_event: threading.Event
    guard: Optional[Callable[[str, str], Optional[str]]] = check_command
    bash_default_timeout: int = 60
    bash_max_timeout: int = 300
    env: Optional[Dict[str, str]] = None


def scrubbed_env(names: Iterable[str], base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """A copy of base (default os.environ) without the named variables."""
    drop = set(names)
    return {k: v for k, v in (os.environ if base is None else base).items() if k not in drop}


_ALWAYS_PASSED = frozenset({"PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR", "PWD"})  # and LC_*
_AWS_NOT_SECRET = frozenset({"AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"})
_SECRET_NAME_PATTERNS = (
    "*_API_KEY", "*_APIKEY", "*_TOKEN", "*_SECRET", "*_SECRET_*", "*SECRET_KEY*", "*PASSWORD*", "*_PASS",
    "*_KEY", "*CREDENTIAL*", "*_DSN", "*_PWD", "*PASSPHRASE*", "*_PAT", "*_AUTH",
    "AWS_*", "ANTHROPIC_*", "OPENAI_*", "GITHUB_TOKEN", "GH_TOKEN", "SSH_AUTH_SOCK",
)
_SHELL_HOOK_NAMES = frozenset({"CDPATH", "ENV", "BASH_ENV"})  # change where cd goes or what a shell runs
_URL_WITH_PASSWORD = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]*:[^/\s@]+@")  # scheme://user:pw@


def _secret_looking(name: str, value: str) -> bool:
    upper = name.upper()
    if upper in _AWS_NOT_SECRET:
        return False
    return (upper in _SHELL_HOOK_NAMES or any(fnmatch.fnmatchcase(upper, p) for p in _SECRET_NAME_PATTERNS)
            or _URL_WITH_PASSWORD.search(value) is not None)


def run_env(secret_names: Iterable[str], pass_env: Iterable[str] = (),
            base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for the subagent's bash and check commands: base (default os.environ)
    without the provider key variables (always), and without secret-looking variables (tokens,
    keys, passwords, credentials, cloud settings, URLs holding a password, CDPATH/ENV/BASH_ENV)
    unless pass_env names them. PATH, HOME, USER, SHELL, LANG, LC_*, TERM, TMPDIR and PWD are
    always passed."""
    allowed = set(pass_env)
    return {
        k: v for k, v in scrubbed_env(secret_names, base).items()
        if k in _ALWAYS_PASSED or k.startswith("LC_") or k in allowed or not _secret_looking(k, v)
    }


REDACT_MIN_CHARS = 8


def _secrets(values: Iterable[str]) -> List[str]:
    """The values long enough to redact, longest first (so a key containing another goes whole)."""
    return sorted({v for v in values if v and len(v) >= REDACT_MIN_CHARS}, key=len, reverse=True)


def redact(text: str, values: Iterable[str]) -> str:
    """Replace each secret value (8+ chars; shorter ones are ignored) in text with ***."""
    for value in _secrets(values):
        text = text.replace(value, "***")
    return text


def scrub_edge(text: str, values: Iterable[str], at_end: bool) -> str:
    """Replace a piece (8+ chars) of a secret at the end (or start) of text with ***.

    For text that truncation cut next to a secret: redact() cannot match the half that is left.
    """
    best = 0
    for value in _secrets(values):
        for k in range(min(len(text), len(value)), max(best, REDACT_MIN_CHARS - 1), -1):
            if (text[-k:] if at_end else text[:k]) in value:
                best = k
                break
    if not best:
        return text
    return text[:-best] + "***" if at_end else "***" + text[best:]


_CUTS = re.compile("(%s|%s)" % (CUT_MARKER_RE.pattern, re.escape(LINE_CUT_MARKER)))


def scrub_cuts(text: str, values: Iterable[str]) -> str:
    """After redact(): clear secret pieces left right before or after a truncation marker
    (bash output's head/tail cut, read's long-line cut)."""
    values = _secrets(values)
    parts = _CUTS.split(text)  # text, marker, text, marker, ..., text
    if not values or len(parts) == 1:
        return text
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            parts[i] = scrub_edge(parts[i], values, at_end=True)
        if i > 0:
            parts[i] = scrub_edge(parts[i], values, at_end=False)
    return "".join(parts)


def resolve_in_workdir(workdir: Path, path: str, for_write: bool = False) -> Path:
    root = Path(os.path.realpath(str(workdir)))
    target = Path(os.path.realpath(os.path.join(str(root), path)))
    try:
        rel = target.relative_to(root)
    except ValueError:
        raise ToolError("path %r is outside the working directory" % path)
    # Git internals are not part of the run's diff, so the run could not report or undo them.
    if for_write and any(part.lower() == ".git" for part in rel.parts):
        raise ToolError("path %r is inside a .git directory; the file tools do not write there" % path)
    return target


def _arg(args: dict, key: str, typ, required: bool = True):
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError("missing required argument: %s" % key)
        return None
    if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
        raise ToolError("argument %s must be %s" % (key, typ.__name__))
    return value


def file_problem(path: Path) -> Optional[str]:
    """Why the file tools will not open path (a FIFO would block, a huge file would flood
    memory), or None. Raises OSError if path cannot be stat'ed."""
    st = os.stat(str(path))
    if stat.S_ISDIR(st.st_mode):
        return "is a directory"
    if not stat.S_ISREG(st.st_mode):
        return "is not a regular file"
    if st.st_size > MAX_FILE_BYTES:
        return "is %d bytes; over the %d MB limit, use bash with head/sed" % (
            st.st_size, MAX_FILE_BYTES // (1024 * 1024))
    return None


def _check_file(path: Path, display: str) -> None:
    try:
        problem = file_problem(path)
    except FileNotFoundError:
        raise ToolError("file not found: %s" % display)
    except OSError as e:
        raise ToolError("cannot read %s: %s" % (display, e.strerror or e))
    if problem:
        raise ToolError("%s %s" % (display, problem))


def _read_text(path: Path, display: str) -> str:
    _check_file(path, display)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ToolError("file not found: %s" % display)
    except OSError as e:
        raise ToolError("cannot read %s: %s" % (display, e.strerror or e))
    if b"\x00" in data[:8192]:
        raise ToolError("%s is a binary file (%d bytes)" % (display, len(data)))
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("%s is not valid UTF-8, treating as binary (%d bytes)" % (display, len(data)))


def _read(args: dict, ctx: ToolContext) -> str:
    display = _arg(args, "path", str)
    offset = _arg(args, "offset", int, False) or 1
    limit = _arg(args, "limit", int, False) or READ_DEFAULT_LIMIT
    if offset < 1 or limit < 1:
        raise ToolError("offset and limit must be at least 1")
    text = _read_text(resolve_in_workdir(ctx.workdir, display), display)
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    if total == 0:
        return "[empty file]"
    if offset > total:
        raise ToolError("offset %d is past the end of the file (%d lines)" % (offset, total))
    chunk = lines[offset - 1 : offset - 1 + limit]
    out = []
    for i, line in enumerate(chunk, start=offset):
        if len(line) > READ_MAX_LINE_CHARS:
            line = line[:READ_MAX_LINE_CHARS] + LINE_CUT_MARKER
        out.append("%6d\t%s" % (i, line))
    last = offset + len(chunk) - 1
    if offset != 1 or last != total:
        out.append("[lines %d-%d of %d]" % (offset, last, total))
    return "\n".join(out)


def _write(args: dict, ctx: ToolContext) -> str:
    display = _arg(args, "path", str)
    content = _arg(args, "content", str)
    path = resolve_in_workdir(ctx.workdir, display, for_write=True)
    try:
        st = os.stat(str(path))
    except FileNotFoundError:
        st = None
    except OSError as e:
        raise ToolError("cannot write %s: %s" % (display, e.strerror or e))
    if st is not None and stat.S_ISDIR(st.st_mode):
        raise ToolError("%s is a directory" % display)
    if st is not None and not stat.S_ISREG(st.st_mode):  # opening a FIFO would block
        raise ToolError("%s is not a regular file" % display)
    data = content.encode("utf-8")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as e:
        raise ToolError("cannot write %s: %s" % (display, e.strerror or e))
    return "wrote %d bytes to %s" % (len(data), display)


def _edit(args: dict, ctx: ToolContext) -> str:
    display = _arg(args, "path", str)
    edits = _arg(args, "edits", list)
    path = resolve_in_workdir(ctx.workdir, display, for_write=True)
    _check_file(path, display)
    try:
        content = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ToolError("file not found: %s" % display)
    except UnicodeDecodeError:
        raise ToolError("%s is not valid UTF-8; edit refused, file left unchanged" % display)
    except OSError as e:
        raise ToolError("cannot read %s: %s" % (display, e.strerror or e))
    try:
        new = apply_edits(content, edits)
    except EditError as e:
        raise ToolError(str(e))
    try:
        path.write_bytes(new.encode("utf-8"))
    except OSError as e:
        raise ToolError("cannot write %s: %s" % (display, e.strerror or e))
    n = len(edits)
    return "%s: %d replacement%s applied" % (display, n, "" if n == 1 else "s")


def _bash(args: dict, ctx: ToolContext) -> str:
    command = _arg(args, "command", str)
    timeout = _arg(args, "timeout", int, False) or ctx.bash_default_timeout
    timeout = max(1, min(timeout, ctx.bash_max_timeout))
    workdir = str(ctx.workdir)
    if ctx.guard is not None:
        reason = ctx.guard(command, workdir)
        if reason:
            raise ToolError("blocked: %s" % reason)
    try:
        res = run_command(command, workdir, timeout, ctx.cancel_event, env=ctx.env)
    except OSError as e:
        raise ToolError("cannot run command: %s" % (e.strerror or e))
    out = "exit code: %d\n%s" % (res.returncode, res.output)
    if res.timed_out:
        out += "\n[timed out after %ds]" % timeout
    if res.cancelled:
        out += "\n[cancelled]"
    return out


_HANDLERS = {"read": _read, "write": _write, "edit": _edit, "bash": _bash}


def run_tool(name: str, args: dict, ctx: ToolContext) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ToolError("unknown tool: %s" % name)
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    return handler(args, ctx)
