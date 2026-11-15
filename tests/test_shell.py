from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from dllm.shell import CommandResult, kill_all, run_command, truncate_middle


def pid_gone(pid: int, wait: float = 10.0) -> bool:
    deadline = time.time() + wait
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.05)
    return False


class TruncateTests(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(truncate_middle("abc", 10), "abc")

    def test_keeps_head_and_tail(self):
        out = truncate_middle("a" * 100 + "b" * 100, 20)
        self.assertTrue(out.startswith("a" * 10))
        self.assertTrue(out.endswith("b" * 10))
        self.assertIn("\n... [180 characters cut] ...\n", out)


class RunCommandTests(unittest.TestCase):
    def test_output_and_returncode(self):
        r = run_command("echo hi; exit 3", os.getcwd(), 10)
        self.assertIsInstance(r, CommandResult)
        self.assertEqual((r.output, r.returncode, r.timed_out, r.cancelled), ("hi\n", 3, False, False))

    def test_stderr_merged(self):
        self.assertIn("err", run_command("echo err 1>&2", os.getcwd(), 10).output)

    def test_timeout_kills_group(self):
        start = time.time()
        r = run_command("sleep 30 & echo $!; sleep 30", os.getcwd(), 1)
        self.assertLess(time.time() - start, 3)
        self.assertTrue(r.timed_out)
        self.assertTrue(pid_gone(int(r.output.split()[0])))

    def test_background_process_does_not_block(self):
        start = time.time()
        r = run_command("sleep 999 & echo $!; echo started", os.getcwd(), 10)
        self.assertLess(time.time() - start, 3)
        self.assertFalse(r.timed_out)
        self.assertIn("started", r.output)
        self.assertTrue(pid_gone(int(r.output.split()[0])))

    def test_cancel_event_kills(self):
        ev = threading.Event()
        threading.Timer(0.5, ev.set).start()
        start = time.time()
        r = run_command("sleep 30", os.getcwd(), 20, cancel_event=ev)
        self.assertLess(time.time() - start, 10)
        self.assertTrue(r.cancelled)
        self.assertFalse(r.timed_out)

    def test_large_output_truncated(self):
        r = run_command("python3 -c 'print(\"x\"*50000)'", os.getcwd(), 20)
        self.assertLessEqual(len(r.output), 10_000 + 100)
        self.assertIn("characters cut", r.output)

    def test_no_pipe_deadlock(self):
        r = run_command("python3 -c 'print(\"y\"*200000)'", os.getcwd(), 10, max_output_chars=1_000_000)
        self.assertFalse(r.timed_out)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(r.output), 200001)

    def test_escaped_descendant_does_not_steal_next_output(self):
        code = (
            "import os,time\n"
            "if os.fork()==0:\n"
            "    os.setsid(); print(os.getpid(),flush=True); time.sleep(30)\n"
        )
        start = time.time()
        r = run_command("python3 -c '%s'" % code, os.getcwd(), 10)
        pid = int(r.output.split()[0])

        def cleanup():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass

        self.addCleanup(cleanup)
        self.assertLess(time.time() - start, 6)
        self.assertFalse(r.timed_out)
        for _ in range(3):
            self.assertEqual(run_command("echo hi", os.getcwd(), 10).output, "hi\n")

    def test_kill_all(self):
        out = []
        t = threading.Thread(target=lambda: out.append(run_command("sleep 30", os.getcwd(), 20)))
        t.start()
        time.sleep(0.5)
        start = time.time()
        kill_all()
        t.join(15)
        self.assertFalse(t.is_alive())
        self.assertLess(time.time() - start, 10)
        self.assertTrue(out[0].cancelled)

    def test_quick_command_returns_quickly(self):
        run_command("true", os.getcwd(), 10)  # warm up
        times = []
        for _ in range(10):
            start = time.monotonic()
            run_command("true", os.getcwd(), 10)
            times.append(time.monotonic() - start)
        # The fastest run, not the mean: a few slow runs under load do not fail it, while a
        # fixed 50 ms poll makes every run slower than 30 ms.
        self.assertLess(min(times), 0.030)

    def test_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_command("pwd", d, 10)
            self.assertEqual(os.path.realpath(r.output.strip()), os.path.realpath(d))


if __name__ == "__main__":
    unittest.main()
