from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(ROOT, "plugins", "dllm-subagents", "server")


class JsonRpcPipe:
    """Client side of a newline-delimited JSON-RPC stream over a pair of file objects."""

    def __init__(self, to_server, from_server):
        self._to_server = to_server
        self._from_server = from_server
        self._messages: "queue.Queue[dict]" = queue.Queue()
        self.parse_errors: list = []
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        for line in self._from_server:
            line = line.strip()
            if line:
                try:
                    self._messages.put(json.loads(line))
                except ValueError:
                    self.parse_errors.append(line)

    def send(self, message: dict) -> None:
        self.send_raw(json.dumps(message))

    def send_raw(self, line: str) -> None:
        self._to_server.write(line + "\n")
        self._to_server.flush()

    def read(self, timeout: float = 5.0) -> Optional[dict]:
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty:
            return None

    def close_input(self) -> None:
        self._to_server.close()

    def close(self) -> None:
        for stream in (self._to_server, self._from_server):
            try:
                stream.close()
            except Exception:
                pass
        self._reader.join(2)


class FakeClient:
    """Scripted stand-in for ChatClient. Items are Completions or callables(messages) -> Completion."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.deadlines = []
        self.cancels = []

    def complete(self, messages, tools, deadline=None, cancel=None):
        self.calls.append([dict(m) for m in messages])
        self.deadlines.append(deadline)
        self.cancels.append(cancel)
        if not self._script:
            raise AssertionError("FakeClient script exhausted (call %d)" % len(self.calls))
        item = self._script.pop(0)
        if callable(item):
            return item(messages)
        return item


_call_counter = itertools.count(1)


def tool_completion(*calls, content=None, tokens=(100, 10)):
    """Build a Completion with tool calls. Each call is (name, args); args may be a dict or a raw string."""
    from dllm.llm import Completion, ToolCall

    tool_calls = []
    wire = []
    for name, args in calls:
        raw = args if isinstance(args, str) else json.dumps(args)
        call_id = "call_%d" % next(_call_counter)
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=raw))
        wire.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}})
    message = {"role": "assistant", "content": content}
    if wire:
        message["tool_calls"] = wire
    return Completion(
        content=content,
        tool_calls=tool_calls,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        finish_reason="tool_calls" if calls else "stop",
        message=message,
    )


def call_near_fifo(fifo, fn, timeout: float = 5.0):
    """Run fn() on a thread and return (finished_in_time, result, exception).

    If fn is still blocked opening fifo after timeout, open the other end of the FIFO so the
    thread can end (a reader sees EOF, a writer finishes into the pipe buffer), so a regression
    fails instead of hanging the suite.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = fn()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    finished = not thread.is_alive()
    if not finished:
        try:  # a blocked reader: open and close the write end, so it reads EOF
            os.close(os.open(str(fifo), os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
        try:  # a blocked writer: hold a read end open while it finishes
            reader = os.open(str(fifo), os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            reader = None
        thread.join(timeout)
        if reader is not None:
            os.close(reader)
    return finished, box.get("result"), box.get("error")


def isolated_git_env(tmp: str) -> Dict[str, str]:
    """Environment overrides that keep git away from the user's global config and hooks."""
    home = os.path.join(tmp, "home")
    os.makedirs(home, exist_ok=True)
    return {
        "HOME": home,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.path.join(home, "gitconfig"),
        "XDG_CONFIG_HOME": os.path.join(home, "xdg"),
    }


def make_repo(tmp, commit: bool = True) -> Path:
    """Create a git repo under tmp with local-only identity config; optionally one initial commit.

    Callers should apply isolated_git_env(tmp) to os.environ so git ignores global config.
    """
    repo = Path(tmp) / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(isolated_git_env(str(tmp)))

    def git(*args):
        subprocess.run(["git", *args], cwd=str(repo), env=env, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("config", "commit.gpgsign", "false")
    if commit:
        (repo / "a.txt").write_text("one\ntwo\nthree\n")
        (repo / "b.txt").write_text("keep\n")
        git("add", "-A")
        git("commit", "-q", "-m", "initial")
    return repo
