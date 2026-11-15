from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dllm.prompts import SYSTEM_PROMPT, build_task_message, load_files
from dllm.tools import TOOL_SCHEMAS


class PromptsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_prompt_and_tools_are_small(self):
        self.assertLess(len(SYSTEM_PROMPT) + len(json.dumps(TOOL_SCHEMAS)), 4000)

    def test_prompt_is_static(self):
        self.assertNotIn("{", SYSTEM_PROMPT)
        self.assertEqual(
            [t["function"]["name"] for t in TOOL_SCHEMAS],
            ["read", "write", "edit", "bash", "finish"],
        )

    def test_task_message_contents(self):
        msg = build_task_message("do it", "/w", "pytest -q", [("src/a.py", "print(1)\n")], [])
        for s in ("do it", "/w", "Check command: pytest -q", "--- src/a.py ---", "print(1)"):
            self.assertIn(s, msg)

    def test_task_message_without_check(self):
        msg = build_task_message("do it", "/w", None, [], [])
        self.assertNotIn("Check command", msg)

    def test_task_message_notes(self):
        msg = build_task_message("t", "/w", None, [], ["skipped big.txt"])
        self.assertIn("skipped big.txt", msg)

    def test_load_files_cap_and_notes(self):
        (self.dir / "a.txt").write_text("a" * 80_000)
        (self.dir / "b.txt").write_text("b" * 80_000)
        files, notes = load_files(self.dir, ["a.txt", "b.txt"])
        self.assertEqual([f[0] for f in files], ["a.txt"])
        self.assertEqual(len(notes), 1)
        self.assertIn("b.txt", notes[0])
        self.assertIn("size", notes[0])

    def test_load_files_missing_and_outside(self):
        files, notes = load_files(self.dir, ["missing.txt", "../x"])
        self.assertEqual(files, [])
        self.assertEqual(len(notes), 2)

    def test_load_files_binary_noted(self):
        (self.dir / "b.bin").write_bytes(b"\x00\x01")
        files, notes = load_files(self.dir, ["b.bin"])
        self.assertEqual(files, [])
        self.assertIn("b.bin", notes[0])


if __name__ == "__main__":
    unittest.main()
