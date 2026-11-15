from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, TextIO

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
_DATE_VERSION = re.compile(r"^\d{4}-\d{2}-\d{2}$")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


@dataclass
class ToolReply:
    text: str
    is_error: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, threading.Event], ToolReply]


def _log(message: str) -> None:
    print("fast-agent: " + message, file=sys.stderr, flush=True)


class McpServer:
    """Minimal MCP server over newline-delimited JSON-RPC 2.0."""

    def __init__(self, name: str, version: str, tools: List[ToolSpec]):
        self._name = name
        self._version = version
        self._tools = {t.name: t for t in tools}
        self._tool_order = list(tools)
        self._write_lock = threading.Lock()
        self._inflight: Dict[object, threading.Event] = {}
        self._inflight_lock = threading.Lock()
        self._out: TextIO = sys.stdout

    def serve(self, instream: TextIO, outstream: TextIO) -> None:
        self._out = outstream
        for line in instream:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self._send({"jsonrpc": "2.0", "id": None, "error": {"code": PARSE_ERROR, "message": "Parse error"}})
                continue
            if not isinstance(message, dict):
                self._send({"jsonrpc": "2.0", "id": None, "error": {"code": INVALID_REQUEST, "message": "Invalid request"}})
                continue
            try:
                self._dispatch(message)
            except Exception as exc:  # keep the read loop alive
                _log("dispatch failed: %r" % (exc,))
        with self._inflight_lock:
            for event in self._inflight.values():
                event.set()

    def _send(self, message: dict) -> None:
        data = json.dumps(message)
        with self._write_lock:
            try:
                self._out.write(data + "\n")
                self._out.flush()
            except (OSError, ValueError) as exc:
                _log("write failed: %r" % (exc,))

    def _result(self, msg_id, result: dict) -> None:
        self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _error(self, msg_id, code: int, text: str) -> None:
        self._send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": text}})

    def _dispatch(self, message: dict) -> None:
        method = message.get("method")
        params = message.get("params")
        if not isinstance(params, dict):
            params = {}
        if "id" not in message:
            if method == "notifications/cancelled":
                self._cancel(params.get("requestId"))
            return
        msg_id = message["id"]
        if method == "initialize":
            version = params.get("protocolVersion")
            if not (isinstance(version, str) and _DATE_VERSION.match(version)):
                version = DEFAULT_PROTOCOL_VERSION
            self._result(msg_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self._name, "version": self._version},
            })
        elif method == "ping":
            self._result(msg_id, {})
        elif method == "tools/list":
            self._result(msg_id, {"tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in self._tool_order
            ]})
        elif method == "tools/call":
            self._start_call(msg_id, params)
        else:
            self._error(msg_id, METHOD_NOT_FOUND, "Method not found: %s" % (method,))

    def _start_call(self, msg_id, params: dict) -> None:
        spec = self._tools.get(params.get("name"))
        if spec is None:
            self._error(msg_id, INVALID_PARAMS, "Unknown tool: %s" % (params.get("name"),))
            return
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        cancel = threading.Event()
        with self._inflight_lock:
            self._inflight[msg_id] = cancel
        threading.Thread(target=self._run_call, args=(msg_id, spec, arguments, cancel), daemon=True).start()

    def _run_call(self, msg_id, spec: ToolSpec, arguments: dict, cancel: threading.Event) -> None:
        try:
            reply = spec.handler(arguments, cancel)
        except Exception as exc:
            _log("tool %s raised %r" % (spec.name, exc))
            reply = ToolReply("%s failed: %s" % (spec.name, exc), is_error=True)
        with self._inflight_lock:
            self._inflight.pop(msg_id, None)
        if cancel.is_set():
            return
        self._result(msg_id, {"content": [{"type": "text", "text": reply.text}], "isError": bool(reply.is_error)})

    def _cancel(self, request_id) -> None:
        with self._inflight_lock:
            event = self._inflight.get(request_id)
        if event is not None:
            event.set()
