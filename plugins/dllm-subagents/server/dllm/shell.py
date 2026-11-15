"""Run shell commands with timeout, cancellation and whole-process-group kill.

Adapted from mini-swe-agent's environments/local.py::_run (MIT License,
Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez).
"""
from __future__ import annotations

import codecs
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

# Seconds to keep collecting output after the shell exits while a background
# child still holds the output pipe, before the leftovers are killed.
_BACKGROUND_GRACE = 1.0
_POLL = 0.05

_live: Dict[int, threading.Event] = {}
_live_lock = threading.Lock()


@dataclass
class CommandResult:
    output: str
    returncode: int
    timed_out: bool
    cancelled: bool


def _marker(cut: int) -> str:
    return "\n... [%d characters cut] ...\n" % cut


CUT_MARKER_RE = re.compile(r"\n\.\.\. \[\d+ characters cut\] \.\.\.\n")  # matches _marker


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + _marker(len(text) - max_chars) + (text[len(text) - tail:] if tail else "")


class _Collector:
    """Reads the pipe on a thread, keeping only a bounded head and tail."""

    def __init__(self, stream, max_chars: int) -> None:
        # The thread owns a private dup of the fd and closes it itself, so the
        # fd number cannot be reused while the thread may still be reading.
        self._fd = os.dup(stream.fileno())
        self._lock = threading.Lock()
        self._head_max = max_chars // 2
        self._tail_max = max_chars - self._head_max
        self._head = ""
        self._tail = ""
        self._cut = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _add(self, s: str) -> None:
        with self._lock:
            self._add_locked(s)

    def _add_locked(self, s: str) -> None:
        room = self._head_max - len(self._head)
        if room > 0:
            self._head += s[:room]
            s = s[room:]
        if not s:
            return
        self._tail += s
        if len(self._tail) > self._tail_max:
            self._cut += len(self._tail) - self._tail_max
            self._tail = self._tail[len(self._tail) - self._tail_max:] if self._tail_max else ""

    def _run(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        fd = self._fd
        try:
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                self._add(decoder.decode(chunk))
            self._add(decoder.decode(b"", final=True))
        except (OSError, ValueError):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def text(self) -> str:
        with self._lock:
            if self._cut:
                return self._head + _marker(self._cut) + self._tail
            return self._head + self._tail


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _exited(proc: "subprocess.Popen[bytes]") -> bool:
    """True once the leader has exited. Where possible, do not reap it yet."""
    if hasattr(os, "waitid") and hasattr(os, "WNOWAIT"):
        try:
            return os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
        except ChildProcessError:
            return True
    return proc.poll() is not None


def run_command(
    command: str,
    cwd: str,
    timeout: float,
    cancel_event: Optional[threading.Event] = None,
    max_output_chars: int = 10_000,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pgid = proc.pid
    killed = threading.Event()
    with _live_lock:
        _live[pgid] = killed
    collector: Optional[_Collector] = None
    timed_out = cancelled = False
    deadline = time.monotonic() + timeout
    try:
        collector = _Collector(proc.stdout, max_output_chars)
        while not _exited(proc):
            if killed.is_set() or (cancel_event is not None and cancel_event.is_set()):
                cancelled = True
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL)
        if not (timed_out or cancelled):
            # Shell exited; let a background child holding the pipe finish
            # briefly, but no longer.
            collector.thread.join(_BACKGROUND_GRACE)
    finally:
        # Kill the group before reaping the leader (see _exited) so the pgid
        # cannot have been recycled.
        _killpg(pgid)
        with _live_lock:
            _live.pop(pgid, None)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        if collector is not None:
            collector.thread.join(2)
        # The reader thread has its own dup of the fd, so this is always safe.
        try:
            proc.stdout.close()
        except OSError:
            pass
    output = collector.text() if collector is not None else ""
    if killed.is_set():
        cancelled = True
    returncode = proc.returncode if proc.returncode is not None else -9
    return CommandResult(output, returncode, timed_out, cancelled)


def kill_all() -> None:
    # Hold the lock while killing: entries are removed before the leader is
    # reaped, so a listed pgid still belongs to an unreaped (or live) group.
    with _live_lock:
        for pgid, killed in _live.items():
            killed.set()
            _killpg(pgid)
