from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from helpers import SERVER_DIR, JsonRpcPipe, isolated_git_env, make_repo

SCRIPT = os.path.join(SERVER_DIR, "fast_agent_server.py")
KEY = "sk-e2e-fake-key-123"


def _kill_quietly(pid):
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def tool_message(*calls, content=None):
    wire = [
        {"id": "call_%d" % i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]
    return {"role": "assistant", "content": content, "tool_calls": wire}


class FakeProvider:
    """Local chat-completions server that replays scripted (message, delay_seconds) responses."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                with outer.lock:
                    outer.requests.append({"auth": self.headers.get("Authorization"), "body": body})
                    item = outer.script.pop(0) if outer.script else None
                if item is None:
                    self.send_error(500, "script exhausted")
                    return
                message, delay = item
                if delay:
                    time.sleep(delay)
                data = json.dumps({
                    "choices": [{"message": message, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(2)


class E2EBase(unittest.TestCase):
    wait_seconds = 2

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = make_repo(self.tmp)
        self.config_dir = Path(self.tmp) / "config"
        self.runs_dir = Path(self.tmp) / "runs"
        self.config_dir.mkdir()
        self.procs = []
        self.addCleanup(self._cleanup_procs)
        self._next_id = 1

    def _cleanup_procs(self):
        for proc, pipe in self.procs:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
            pipe.close()
            proc.stdout.close()
            proc.stderr.close()

    def provider(self, script):
        fake = FakeProvider(script)
        self.addCleanup(fake.close)
        return fake

    def write_config(self, base_url, config_text=None):
        if config_text is None:
            config_text = json.dumps({
                "default_provider": "fake",
                "wait_seconds": self.wait_seconds,
                "providers": {"fake": {"base_url": base_url, "api_key_env": "FAKE_E2E_KEY", "model": "fake-model"}},
            })
        (self.config_dir / "config.json").write_text(config_text)
        (self.config_dir / "keys.env").write_text("FAKE_E2E_KEY=%s\n" % KEY)

    def start_server(self):
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TMPDIR": self.tmp,
            "LANG": "en_US.UTF-8",
            "DLLM_SUBAGENTS_CONFIG_DIR": str(self.config_dir),
            "DLLM_SUBAGENTS_RUNS_DIR": str(self.runs_dir),
        }
        env.update(isolated_git_env(self.tmp))
        proc = subprocess.Popen(
            [sys.executable, SCRIPT], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, encoding="utf-8",
        )
        pipe = JsonRpcPipe(proc.stdin, proc.stdout)
        self.procs.append((proc, pipe))
        self.proc, self.pipe = proc, pipe
        self.rpc("initialize", {"protocolVersion": "2025-06-18"})
        return proc

    def send_call(self, name, arguments):
        msg_id = self._next_id
        self._next_id += 1
        self.pipe.send({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})
        return msg_id

    def rpc(self, method, params=None, timeout=15):
        msg_id = self._next_id
        self._next_id += 1
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        self.pipe.send(message)
        return self.read_reply(msg_id, timeout)

    def read_reply(self, msg_id, timeout=15):
        reply = self.pipe.read(timeout)
        self.assertIsNotNone(reply, "no reply for id %s" % msg_id)
        self.assertEqual(reply["id"], msg_id)
        return reply

    def call(self, name, arguments, timeout=15):
        msg_id = self.send_call(name, arguments)
        reply = self.read_reply(msg_id, timeout)
        result = reply["result"]
        return result["content"][0]["text"], result["isError"]

    def run_id_from(self, text):
        for line in text.splitlines():
            if line.startswith("run_id: "):
                return line.split(": ", 1)[1].strip()
        self.fail("no run_id in: " + text)

    def run_id_on_disk(self, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = sorted(self.runs_dir.glob("*.json")) if self.runs_dir.exists() else []
            if found:
                return found[0].stem
            time.sleep(0.05)
        self.fail("no run record appeared")


class ServerE2ETests(E2EBase):
    def test_fast_agent_round_trip(self):
        fake = self.provider([
            (tool_message(("write", {"path": "new.txt", "content": "hello\n"})), 0),
            (tool_message(("finish", {"summary": "wrote new.txt"})), 0),
        ])
        self.write_config(fake.base_url)
        self.start_server()
        text, is_error = self.call("fast_agent", {"task": "make new.txt", "workdir": str(self.repo), "provider": "fake"})
        self.assertFalse(is_error, text)
        self.assertIn("status: done", text)
        self.assertIn("wrote new.txt", text)
        self.assertIn("new.txt", text)
        self.assertEqual((self.repo / "new.txt").read_text(), "hello\n")
        self.assertEqual(fake.requests[0]["auth"], "Bearer " + KEY)
        self.assertNotIn(KEY, text)

    def test_wait_path(self):
        delay = 6  # the provider's first reply; wait_seconds is 2
        fake = self.provider([
            (tool_message(("write", {"path": "slow.txt", "content": "x\n"})), delay),
            (tool_message(("finish", {"summary": "slow done"})), 0),
        ])
        self.write_config(fake.base_url)
        self.start_server()
        started = time.monotonic()
        text, is_error = self.call("fast_agent", {"task": "slow job", "workdir": str(self.repo)})
        self.assertLess(time.monotonic() - started, delay - 1)  # returned before the provider did
        self.assertFalse(is_error)
        self.assertIn("status: running", text)
        run_id = self.run_id_from(text)
        for _ in range(10):
            text, is_error = self.call("fast_agent_wait", {"run_id": run_id})
            if "status: running" not in text:
                break
        self.assertFalse(is_error, text)
        self.assertIn("status: done", text)
        self.assertIn("slow done", text)

    def test_host_cancel_stops_run(self):
        fake = self.provider([
            (tool_message(("write", {"path": "late.txt", "content": "x\n"})), 3),
            (tool_message(("finish", {"summary": "never"})), 0),
        ])
        self.write_config(fake.base_url)
        self.start_server()
        msg_id = self.send_call("fast_agent", {"task": "slow job", "workdir": str(self.repo)})
        run_id = self.run_id_on_disk()
        time.sleep(0.3)
        self.pipe.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": msg_id}})
        self.assertIsNone(self.pipe.read(1.0), "a cancelled call must not get a response")
        # Poll for the outcome instead of betting on one 2 s wait: under load the run may still be
        # in its (3 s) model call. A run the cancel did not reach ends "done" and writes late.txt.
        deadline = time.monotonic() + 15
        while True:
            text, is_error = self.call("fast_agent_wait", {"run_id": run_id}, timeout=15)
            self.assertFalse(is_error, text)
            if "status: running" not in text or time.monotonic() > deadline:
                break
        self.assertIn("status: cancelled", text)
        self.assertFalse((self.repo / "late.txt").exists())

    def test_undo_tool(self):
        fake = self.provider([
            (tool_message(("write", {"path": "a.txt", "content": "changed\n"})), 0),
            (tool_message(("finish", {"summary": "edited"})), 0),
        ])
        self.write_config(fake.base_url)
        self.start_server()
        text, _ = self.call("fast_agent", {"task": "edit a", "workdir": str(self.repo)})
        self.assertIn("status: done", text)
        self.assertEqual((self.repo / "a.txt").read_text(), "changed\n")
        undo, is_error = self.call("fast_agent_undo", {"run_id": self.run_id_from(text)})
        self.assertFalse(is_error, undo)
        self.assertIn("Undid run", undo)
        self.assertEqual((self.repo / "a.txt").read_text(), "one\ntwo\nthree\n")

    def test_run_error_is_tool_error(self):
        self.write_config(self.provider([]).base_url)
        self.start_server()
        plain = Path(self.tmp) / "plain"
        plain.mkdir()
        text, is_error = self.call("fast_agent", {"task": "x", "workdir": str(plain)})
        self.assertTrue(is_error)
        self.assertIn("git init", text)

    def test_invalid_argument_types_are_tool_errors(self):
        self.write_config(self.provider([]).base_url)
        self.start_server()
        for args in ({}, {"task": 5}, {"task": "x", "files": "a.txt"}, {"task": "x", "max_steps": "many"}):
            text, is_error = self.call("fast_agent", dict(args, workdir=str(self.repo)))
            self.assertTrue(is_error, args)
        text, is_error = self.call("fast_agent_wait", {"run_id": 7})
        self.assertTrue(is_error)
        # the server is still healthy
        text, is_error = self.call("fast_agent_cancel", {"run_id": "nope"})
        self.assertTrue(is_error)

    def test_config_error_still_serves_and_reports(self):
        self.write_config("http://127.0.0.1:9/v1", config_text="{not json")
        self.start_server()
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual(len(tools), 4)
        text, is_error = self.call("fast_agent", {"task": "x", "workdir": str(self.repo)})
        self.assertTrue(is_error)
        self.assertIn("Invalid JSON", text)
        text, is_error = self.call("fast_agent_undo", {"run_id": "20260101-000000-abcdef"})
        self.assertTrue(is_error)
        self.assertIn("Invalid JSON", text)

    def _assert_signal_stops_run(self, signum):
        pidfile = Path(self.tmp) / "sleep.pid"
        fake = self.provider([
            (tool_message(("bash", {"command": "echo $$ > %s; exec sleep 30" % pidfile})), 0),
        ])
        self.write_config(fake.base_url)
        proc = self.start_server()
        self.send_call("fast_agent", {"task": "sleepy", "workdir": str(self.repo)})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (pidfile.exists() and pidfile.read_text().strip()):
            time.sleep(0.05)
        self.assertTrue(pidfile.exists(), "the bash command never started")
        sleeper = int(pidfile.read_text().strip())
        self.addCleanup(_kill_quietly, sleeper)
        run_id = self.run_id_on_disk()
        os.kill(proc.pid, signum)
        started = time.monotonic()
        proc.wait(5)
        self.assertLess(time.monotonic() - started, 5)
        time.sleep(0.2)
        with self.assertRaises(ProcessLookupError):
            os.kill(sleeper, 0)
        record = json.loads((self.runs_dir / (run_id + ".json")).read_text())
        self.assertEqual(record["status"], "cancelled")

    def test_sigterm_kills_children(self):
        self._assert_signal_stops_run(signal.SIGTERM)

    def test_sighup_kills_children(self):
        self._assert_signal_stops_run(signal.SIGHUP)

    @unittest.skipUnless(hasattr(signal, "SIGQUIT"), "no SIGQUIT on this platform")
    def test_sigquit_kills_children(self):
        self._assert_signal_stops_run(signal.SIGQUIT)

    def test_bad_config_type_still_serves_and_reports(self):
        # QA replay: {"wait_seconds": "abc"} used to crash the server at startup.
        self.write_config("http://127.0.0.1:9/v1", config_text='{"wait_seconds": "abc"}')
        self.start_server()  # initialize is answered
        tools = self.rpc("tools/list")["result"]["tools"]
        self.assertEqual(len(tools), 4)
        text, is_error = self.call("fast_agent", {"task": "x", "workdir": str(self.repo)})
        self.assertTrue(is_error)
        self.assertIn("wait_seconds", text)

    def test_eof_exits_cleanly(self):
        self.write_config(self.provider([]).base_url)
        proc = self.start_server()
        proc.stdin.close()
        self.assertEqual(proc.wait(10), 0)


if __name__ == "__main__":
    unittest.main()
