from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from helpers import ROOT

SMOKE = os.path.join(ROOT, "scripts", "smoke.py")
REAL_KEY_VARS = ("INCEPTION_API_KEY", "CEREBRAS_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")
FAKE_KEY = "sk-fake-secret-123"

CALC = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
TEST_CALC = (
    "import unittest\nfrom calc import add, mul\n\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n\n"
    "    def test_mul(self):\n        self.assertEqual(mul(2, 3), 6)\n"
)


def _call(name, args):
    return {"id": "c_" + name, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


STEPS = [
    _call("write", {"path": "calc.py", "content": CALC}),
    _call("write", {"path": "test_calc.py", "content": TEST_CALC}),
    _call("finish", {"summary": "created calc and tests"}),
]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.server.seen_auth.append(self.headers.get("Authorization"))
        if self.path == "/v1/models":
            self._send({"data": [{"id": "fake-model"}, {"id": "other"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") == "bad-model":
            self.send_error(400)
            return
        done = sum(1 for m in body["messages"] if m.get("role") == "tool")
        call = STEPS[min(done, len(STEPS) - 1)]
        self._send({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.seen_auth = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = os.path.join(self.tmp.name, "cfg")
        os.makedirs(self.cfg)
        with open(os.path.join(self.cfg, "config.json"), "w") as f:
            json.dump({"providers": {"fake": {
                "base_url": "http://127.0.0.1:%d/v1" % self.server.server_address[1],
                "api_key_env": "FAKE_API_KEY", "model": "fake-model"}}}, f)

    def _keys(self, text):
        with open(os.path.join(self.cfg, "keys.env"), "w") as f:
            f.write(text)

    def _run(self, *args):
        env = {k: v for k, v in os.environ.items() if k not in REAL_KEY_VARS + ("FAKE_API_KEY",)}
        env["DLLM_SUBAGENTS_CONFIG_DIR"] = self.cfg
        return subprocess.run([sys.executable, SMOKE] + list(args), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=120)

    def test_default_mode_runs_agent_and_skips_missing_keys(self):
        self._keys("FAKE_API_KEY=%s\n" % FAKE_KEY)
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("done", r.stdout)
        self.assertIn("fake-model", r.stdout)
        self.assertIn("INCEPTION_API_KEY", r.stdout)  # skip note names the env var
        self.assertNotIn(FAKE_KEY, r.stdout)

    def test_keep_prints_paths_and_trajectory(self):
        self._keys("FAKE_API_KEY=%s\n" % FAKE_KEY)
        r = self._run("--provider", "fake", "--keep")
        self.assertEqual(r.returncode, 0, r.stdout)
        paths = [l.split(None, 1)[1].strip() for l in r.stdout.splitlines() if l.startswith("kept ")]
        self.assertTrue(paths, r.stdout)
        workdir = paths[0]
        self.addCleanup(__import__("shutil").rmtree, workdir, True)
        self.assertTrue(os.path.exists(os.path.join(workdir, "calc.py")))
        self.assertIn("trajectory", r.stdout)
        self.assertTrue(any("trajectory" in n for n in os.listdir(workdir)), os.listdir(workdir))

    def test_models_mode(self):
        self._keys("FAKE_API_KEY=%s\n" % FAKE_KEY)
        r = self._run("--provider", "fake", "--models")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("fake-model", r.stdout)
        self.assertIn("present", r.stdout)
        self.assertNotIn(FAKE_KEY, r.stdout)
        self.assertEqual(self.server.seen_auth, ["Bearer " + FAKE_KEY])

    def test_no_keys_exits_1(self):
        self._keys("")
        r = self._run()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("No provider has an API key", r.stdout)

    def test_agent_failure_exits_1(self):
        self._keys("FAKE_API_KEY=%s\n" % FAKE_KEY)
        with open(os.path.join(self.cfg, "config.json")) as f:
            cfg = json.load(f)
        cfg["providers"]["fake"]["model"] = "bad-model"
        with open(os.path.join(self.cfg, "config.json"), "w") as f:
            json.dump(cfg, f)
        r = self._run("--provider", "fake")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("error", r.stdout)


if __name__ == "__main__":
    unittest.main()
