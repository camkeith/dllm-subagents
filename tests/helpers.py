from __future__ import annotations

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
