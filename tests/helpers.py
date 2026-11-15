from __future__ import annotations

import itertools
import json
import os
import queue
import threading
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(ROOT, "plugins", "dllm-subagents", "server")


class JsonRpcPipe:
    """Client side of a newline-delimited JSON-RPC stream over a pair of file objects."""

    def __init__(self, to_server, from_server):
        self._to_server = to_server
        self._from_server = from_server
        self._messages: "queue.Queue[dict]" = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        for line in self._from_server:
            line = line.strip()
            if line:
                self._messages.put(json.loads(line))

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


class FakeClient:
    """Scripted stand-in for ChatClient. Items are Completions or callables(messages) -> Completion."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self.deadlines = []

    def complete(self, messages, tools, deadline=None):
        self.calls.append([dict(m) for m in messages])
        self.deadlines.append(deadline)
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
