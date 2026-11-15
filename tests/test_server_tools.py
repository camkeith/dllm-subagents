from __future__ import annotations

import threading
import time
import unittest

from dllm.mcp import ToolReply
from dllm.runs import RunError
from dllm.server_tools import build_tool_specs


class FakeManager:
    def __init__(self, start_delay=0.0):
        self.start_delay = start_delay
        self.wait_seconds_seen = None
        self.cancelled = []
        self.start_kwargs = None

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        time.sleep(self.start_delay)
        return "20260101-000000-abcdef"

    def wait(self, run_id, seconds, interrupt=None):
        self.wait_seconds_seen = seconds
        return {"status": "running", "run_id": run_id, "progress": "starting"}

    def cancel(self, run_id):
        self.cancelled.append(run_id)
        return {"status": "cancelled", "run_id": run_id, "summary": "stopped", "stats": {}}

    def undo(self, run_id):
        raise RunError("cannot undo")


def specs(manager, **kwargs):
    return {s.name: s for s in build_tool_specs(manager, **kwargs)}


class ServerToolsTests(unittest.TestCase):
    def test_wait_budget_subtracts_start_time(self):
        manager = FakeManager(start_delay=0.5)
        reply = specs(manager, wait_seconds=2.0)["fast_agent"].handler({"task": "t", "workdir": "/x"}, threading.Event())
        self.assertFalse(reply.is_error)
        self.assertLess(manager.wait_seconds_seen, 1.6)
        self.assertGreater(manager.wait_seconds_seen, 1.0)

    def test_wait_budget_never_negative(self):
        manager = FakeManager(start_delay=0.2)
        specs(manager, wait_seconds=0.1)["fast_agent"].handler({"task": "t"}, threading.Event())
        self.assertEqual(manager.wait_seconds_seen, 0)

    def test_cancel_event_cancels_run(self):
        manager = FakeManager()
        event = threading.Event()
        event.set()
        reply = specs(manager)["fast_agent_wait"].handler({"run_id": "20260101-000000-abcdef"}, event)
        self.assertEqual(manager.cancelled, ["20260101-000000-abcdef"])
        self.assertIn("status: cancelled", reply.text)

    def test_run_error_and_unexpected_exceptions_become_tool_errors(self):
        manager = FakeManager()
        reply = specs(manager)["fast_agent_undo"].handler({"run_id": "r"}, threading.Event())
        self.assertEqual(reply, ToolReply("cannot undo", is_error=True))
        manager.wait = lambda *a, **k: (_ for _ in ()).throw(KeyError("boom"))
        reply = specs(manager)["fast_agent_wait"].handler({"run_id": "r"}, threading.Event())
        self.assertTrue(reply.is_error)
        self.assertTrue(reply.text.startswith("internal error: KeyError"))

    def test_config_error_replies_for_every_tool(self):
        for spec in build_tool_specs(None, config_error="bad config").__iter__():
            reply = spec.handler({"task": "t", "run_id": "r"}, threading.Event())
            self.assertEqual(reply, ToolReply("bad config", is_error=True))


if __name__ == "__main__":
    unittest.main()
