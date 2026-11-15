from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest

from helpers import SERVER_DIR, JsonRpcPipe

from dllm import VERSION
from dllm.mcp import McpServer, ToolReply, ToolSpec


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.slow_saw_cancel = threading.Event()

        def echo(args, cancel):
            return ToolReply(args["text"])

        def boom(args, cancel):
            raise RuntimeError("kaboom")

        def slow(args, cancel):
            if cancel.wait(5):
                self.slow_saw_cancel.set()
            return ToolReply("slow done")

        schema = {"type": "object", "properties": {}}
        tools = [
            ToolSpec("echo", "echo text", {"type": "object", "properties": {"text": {"type": "string"}}}, echo),
            ToolSpec("boom", "always raises", schema, boom),
            ToolSpec("slow", "waits for cancel", schema, slow),
        ]
        c2s_r, c2s_w = os.pipe()
        s2c_r, s2c_w = os.pipe()
        self.server_in = os.fdopen(c2s_r, "r", encoding="utf-8")
        self.server_out = os.fdopen(s2c_w, "w", encoding="utf-8")
        self.pipe = JsonRpcPipe(os.fdopen(c2s_w, "w", encoding="utf-8"), os.fdopen(s2c_r, "r", encoding="utf-8"))
        self.server = McpServer("fast-agent", VERSION, tools)
        self.thread = threading.Thread(target=self.server.serve, args=(self.server_in, self.server_out), daemon=True)
        self.thread.start()
        self._next_id = 1

    def tearDown(self):
        try:
            self.pipe.close_input()
        except Exception:
            pass
        self.thread.join(2)

    def call(self, method, params=None):
        msg_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self.pipe.send(message)
        reply = self.pipe.read(5)
        self.assertIsNotNone(reply, "no reply to " + method)
        self.assertEqual(reply["id"], msg_id)
        return reply

    def test_initialize_echoes_protocol_version(self):
        r = self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", r["result"]["capabilities"])
        self.assertEqual(r["result"]["serverInfo"]["name"], "fast-agent")

    def test_initialize_echoes_other_date_version_and_defaults_otherwise(self):
        r = self.call("initialize", {"protocolVersion": "2024-11-05"})
        self.assertEqual(r["result"]["protocolVersion"], "2024-11-05")
        r = self.call("initialize", {"protocolVersion": "banana"})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")

    def test_tools_list(self):
        r = self.call("tools/list")
        tools = r["result"]["tools"]
        self.assertEqual([t["name"] for t in tools], ["echo", "boom", "slow"])
        for t in tools:
            self.assertIn("inputSchema", t)

    def test_tools_call_text_content(self):
        r = self.call("tools/call", {"name": "echo", "arguments": {"text": "hi"}})
        self.assertEqual(r["result"], {"content": [{"type": "text", "text": "hi"}], "isError": False})

    def test_handler_exception_is_error(self):
        r = self.call("tools/call", {"name": "boom", "arguments": {}})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("kaboom", r["result"]["content"][0]["text"])

    def test_unknown_tool_is_invalid_params(self):
        r = self.call("tools/call", {"name": "nope", "arguments": {}})
        self.assertEqual(r["error"]["code"], -32602)

    def test_unknown_method(self):
        r = self.call("does/not/exist")
        self.assertEqual(r["error"]["code"], -32601)

    def test_parse_error(self):
        self.pipe.send_raw("not json")
        r = self.pipe.read(5)
        self.assertEqual(r["error"]["code"], -32700)
        self.assertEqual(self.call("ping")["result"], {})

    def test_ping(self):
        self.assertEqual(self.call("ping")["result"], {})

    def test_notification_gets_no_response(self):
        self.pipe.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertEqual(self.call("ping")["result"], {})

    def test_cancel_sets_event_and_suppresses_response(self):
        self.pipe.send({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "slow", "arguments": {}}})
        self.pipe.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 7}})
        self.assertTrue(self.slow_saw_cancel.wait(2))
        self.assertIsNone(self.pipe.read(0.5))
        self.assertEqual(self.call("ping")["result"], {})

    def test_eof_returns(self):
        self.pipe.close_input()
        self.thread.join(1)
        self.assertFalse(self.thread.is_alive())


class EntryPointTests(unittest.TestCase):
    def test_entry_point_lists_four_tools(self):
        script = os.path.join(SERVER_DIR, "fast_agent_server.py")
        proc = subprocess.Popen(
            [sys.executable, script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, encoding="utf-8",
        )
        try:
            pipe = JsonRpcPipe(proc.stdin, proc.stdout)
            pipe.send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
            self.assertEqual(pipe.read(10)["id"], 1)
            pipe.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            reply = pipe.read(10)
            names = {t["name"] for t in reply["result"]["tools"]}
            self.assertEqual(names, {"fast_agent", "fast_agent_wait", "fast_agent_cancel", "fast_agent_undo"})
            for t in reply["result"]["tools"]:
                self.assertEqual(t["inputSchema"]["type"], "object")
        finally:
            proc.stdin.close()
            proc.wait(5)
            proc.stdout.close()
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
