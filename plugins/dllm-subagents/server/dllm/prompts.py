"""Static system prompt and first task message for the subagent."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

SYSTEM_PROMPT = (
    "You are a fast coding subagent working for another agent. Complete the task you are "
    "given, then stop.\n"
    "\n"
    "Rules:\n"
    "- Work only inside the working directory. Never commit, push, or change git state; "
    "read-only git (status, diff, log, show) is fine.\n"
    "- Read a file before you change it. Prefer edit for existing files; use write for new "
    "files or full rewrites.\n"
    "- Keep tool output small: use head, tail, and grep -n instead of dumping whole files.\n"
    "- If a check command is given, or relevant tests exist, run them before finishing and "
    "fix failures.\n"
    "- Do exactly what was asked. No extra features or refactors.\n"
    "- When done, call finish with a short summary of what changed and anything left undone. "
    "Do not reply without calling a tool."
)


def load_files(
    workdir: Path, paths: List[str], max_total_bytes: int = 100_000
) -> Tuple[List[Tuple[str, str]], List[str]]:
    # Imported here to keep prompts importable without pulling tool schemas at module load.
    from .tools import ToolError, file_problem, resolve_in_workdir

    files: List[Tuple[str, str]] = []
    notes: List[str] = []
    total = 0
    for rel in paths:
        try:
            path = resolve_in_workdir(workdir, rel)
            problem = file_problem(path)
            if problem:
                notes.append("%s: skipped, %s" % (rel, problem))
                continue
            data = path.read_bytes()
        except ToolError:
            notes.append("%s: skipped, outside the working directory" % rel)
            continue
        except OSError:
            notes.append("%s: skipped, could not be read" % rel)
            continue
        if b"\x00" in data[:8192]:
            notes.append("%s: skipped, binary file" % rel)
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            notes.append("%s: skipped, not valid UTF-8" % rel)
            continue
        if total + len(data) > max_total_bytes:
            notes.append("%s: skipped, size cap of %d bytes reached" % (rel, max_total_bytes))
            continue
        total += len(data)
        files.append((rel, text))
    return files, notes


def build_task_message(
    task: str,
    workdir: str,
    check_command: Optional[str],
    files: List[Tuple[str, str]],
    notes: List[str],
) -> str:
    parts = ["Task:\n" + task, "Working directory: " + workdir]
    if check_command:
        parts.append("Check command: " + check_command)
    for note in notes:
        parts.append("Note: " + note)
    for rel, text in files:
        parts.append("--- %s ---\n%s" % (rel, text))
    return "\n\n".join(parts)
