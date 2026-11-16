"""Offline tests for the live eval suite (scripts/eval_cases.py and scripts/evals.py). No network."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from helpers import ROOT, isolated_git_env
from test_smoke import FAKE_KEY, _minimal_env, _shipped_key_vars

SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import eval_cases  # noqa: E402

EVALS = os.path.join(SCRIPTS, "evals.py")


def _shell(command, cwd, path_dir):
    """Run command like the runner does, with python3 resolving to this interpreter."""
    env = dict(os.environ, PATH=path_dir + os.pathsep + os.environ.get("PATH", ""),
               PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, shell=True, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=120)


class CaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bin = os.path.join(cls.tmp, "bin")
        os.makedirs(cls.bin)
        os.symlink(sys.executable, os.path.join(cls.bin, "python3"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def repo(self, case, reference=False):
        root = Path(tempfile.mkdtemp(dir=self.tmp))
        eval_cases.write_files(root, eval_cases.starting_files(case))
        if reference:
            eval_cases.apply_reference(case, root)
        return root

    def hidden(self, case, root):
        eval_cases.write_files(root, case.hidden.files)
        return _shell(case.hidden.command, root, self.bin)

    def test_case_table(self):
        self.assertEqual(len(eval_cases.CASES), 8)
        self.assertEqual(sorted(c.category for c in eval_cases.CASES), sorted(eval_cases.CATEGORIES))
        self.assertEqual(len(eval_cases.CASES_BY_NAME), 8)
        for case in eval_cases.CASES:
            with self.subTest(case=case.name):
                files = eval_cases.starting_files(case)
                for rel in case.hint_files:
                    self.assertIn(rel, files)
                for rel in case.hidden.files:
                    self.assertNotIn(rel, files)
                    self.assertNotIn(rel, case.reference)
                self.assertIn("python3", case.check)
                self.assertTrue(set(case.limits) <= {"max_steps", "max_seconds"})

    def test_no_em_dashes(self):
        for name in ("eval_cases.py", "evals.py"):
            with open(os.path.join(SCRIPTS, name), encoding="utf-8") as f:
                self.assertNotIn("\u2014", f.read(), name)
        for case in eval_cases.CASES:
            texts = [case.task] + list(case.files.values()) + list(case.hidden.files.values())
            self.assertFalse(any("\u2014" in t for t in texts), case.name)

    def test_hidden_fails_on_starting_files(self):
        for case in eval_cases.CASES:
            with self.subTest(case=case.name):
                result = self.hidden(case, self.repo(case))
                self.assertNotEqual(result.returncode, 0, result.stdout[-2000:])

    def test_reference_passes_check_and_hidden(self):
        for case in eval_cases.CASES:
            with self.subTest(case=case.name):
                root = self.repo(case, reference=True)
                check = _shell(case.check, root, self.bin)
                self.assertEqual(check.returncode, 0, check.stdout[-2000:])
                result = self.hidden(case, root)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:])

    def test_reference_tests_catch_every_money_mutant(self):
        case = eval_cases.CASES_BY_NAME["write_tests"]
        self.assertEqual(len(eval_cases.MONEY_MUTANTS), 3)
        for name, source in eval_cases.MONEY_MUTANTS.items():
            with self.subTest(mutant=name):
                self.assertNotEqual(source, eval_cases.MONEY)
                root = self.repo(case, reference=True)
                eval_cases.write_files(root, {"money.py": source})
                result = _shell(case.check, root, self.bin)
                self.assertNotEqual(result.returncode, 0, "mutant %s survived" % name)

    def test_hidden_checks_reject_gamed_solutions(self):
        """A reference solution plus one forbidden shortcut must fail the hidden check, for the
        expected reason when one is given."""
        def edit(rel, old, new):
            def apply(root):
                path = Path(root) / rel
                text = path.read_text(encoding="utf-8")
                self.assertIn(old, text)
                path.write_text(text.replace(old, new, 1), encoding="utf-8")
            return apply

        def write(rel, text):
            return lambda root: eval_cases.write_files(root, {rel: text})

        def append(rel, text):
            def apply(root):
                with (Path(root) / rel).open("a", encoding="utf-8") as f:
                    f.write(text)
            return apply

        def delete(rel):
            return lambda root: (Path(root) / rel).unlink()

        weak_spec = ("import unittest\nfrom textutil import slugify\n\n\nclass T(unittest.TestCase):\n"
                     "    def test_plain(self):\n        self.assertEqual(slugify('abc'), 'abc')\n")
        weak_orders = ("import unittest\nfrom handlers import orders\n\n\nclass T(unittest.TestCase):\n"
                       "    def test_exists(self):\n        self.assertTrue(callable(orders.create_order))\n")
        weak_top_n = ("\n\nclass WeakTest(unittest.TestCase):\n    def test_single(self):\n"
                      "        self.assertEqual(report.top_n({'a': 1}), [('a', 1)])\n")
        weak_debug = ("\n\nclass WeakTest(unittest.TestCase):\n    def test_once(self):\n"
                      "        self.assertEqual(RevenueReport(StaticSource(ROWS)).grand_total(DAY), 2300)\n")
        weak_money = ("import unittest\nfrom money import format_money, parse_money\n\n\n"
                      "class T(unittest.TestCase):\n    def test_basic(self):\n"
                      "        self.assertEqual(parse_money('1.50'), 150)\n"
                      "        self.assertEqual(format_money(150), '$1.50')\n")
        no_new = "no new test methods"
        pass_on_original = "pass on the original code"
        tampers = [
            ("fix_failing_test", edit("tests/test_inventory.py", "inv.reserve(\"mug\", 3)\n        with",
                                      "inv.reserve(\"mug\", 2)\n        with"), None),
            ("write_tests", edit("money.py", "Amounts are integers in cents.", "Amounts are in cents."), None),
            ("write_tests", write("test_money.py", weak_money), "survived the tests"),
            ("rename_across_files", edit("crm/billing.py", "def monthly_price(user_id):",
                                         "def monthly_price(user_id):  # was usr_id"), None),
            ("large_file_edit", edit("report.py", "DEFAULT_WIDTH = 72", "DEFAULT_WIDTH = 80"), None),
            ("large_file_edit", write("tests/test_report.py", eval_cases.REPORT_TESTS), no_new),
            ("large_file_edit", write("tests/test_report.py", eval_cases.REPORT_TESTS + weak_top_n),
             pass_on_original),
            ("add_types_docs", edit("utils.py", "return a / b if b else default",
                                    "return a / b if b != 0 else default"), None),
            ("add_types_docs", edit("utils.py", '"""Divide a by b, returning default when b is zero."""',
                                    '"""Divide a by b.\n\n    Returns default when b is zero.\n    """'), None),
            ("add_types_docs", edit("utils.py", "predicate: Optional[Callable[[T], bool]] = None",
                                    "predicate: Callable[[T], bool] | None = None"),
             # 3.10+ imports it, so the AST check must catch it; on 3.9 the import itself fails.
             "breaks Python 3.9" if sys.version_info >= (3, 10) else "unsupported operand"),
            ("debug_symptom", edit("metrics/report.py", "if day not in self._rows_cache:", "if True:"), None),
            ("debug_symptom", write("tests/test_report.py", eval_cases.DS_TESTS), no_new),
            ("debug_symptom", write("tests/test_report.py", eval_cases.DS_TESTS + weak_debug),
             pass_on_original),
            ("follow_pattern", edit("handlers/orders.py", "and not isinstance(value, bool)", ""), None),
            ("follow_pattern", delete("tests/test_orders.py"), no_new),
            ("follow_pattern", write("tests/test_orders.py", weak_orders), pass_on_original),
            ("follow_pattern", append("store.py", "# touched\n"), "store.py"),
            ("follow_pattern", edit("handlers/users.py", '"invalid email"', '"bad email"'), "handlers/users.py"),
            ("spec_module", edit("textutil.py", 'return result or words[0][:max_length]',
                                 'return result or words[0]'), None),
            ("spec_module", write("test_textutil.py", ""), no_new),
            ("spec_module", write("test_textutil.py", weak_spec), pass_on_original),
        ]
        for name, tamper, reason in tampers:
            case = eval_cases.CASES_BY_NAME[name]
            with self.subTest(case=name, reason=reason):
                root = self.repo(case, reference=True)
                tamper(root)
                result = self.hidden(case, root)
                self.assertNotEqual(result.returncode, 0, result.stdout[-2000:])
                if reason:
                    self.assertIn(reason, result.stdout)

    def test_future_import_allows_pep604_unions(self):
        case = eval_cases.CASES_BY_NAME["add_types_docs"]
        root = self.repo(case, reference=True)
        path = root / "utils.py"
        text = path.read_text(encoding="utf-8")
        text = text.replace("predicate: Optional[Callable[[T], bool]] = None",
                            "predicate: Callable[[T], bool] | None = None")
        path.write_text(text.replace("import re\n", "from __future__ import annotations\n\nimport re\n", 1),
                        encoding="utf-8")
        result = self.hidden(case, root)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])


class _FinishHandler(BaseHTTPRequestHandler):
    """A model that calls finish at once, leaking the key in its summary (which must be redacted)."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        call = {"id": "c_finish", "type": "function", "function": {
            "name": "finish", "arguments": json.dumps({"summary": "nothing to do " + FAKE_KEY})}}
        raw = json.dumps({
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [call]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _FinishHandler)
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
        self._keys("FAKE_API_KEY=%s\n" % FAKE_KEY)

    def _keys(self, text):
        with open(os.path.join(self.cfg, "keys.env"), "w") as f:
            f.write(text)

    def _run(self, *args):
        env = _minimal_env()
        for name in _shipped_key_vars():
            self.assertNotIn(name, env)
        env["DLLM_SUBAGENTS_CONFIG_DIR"] = self.cfg
        env["TMPDIR"] = self.tmp.name  # workdirs land here, so a leak would be cleaned up with it
        env.update(isolated_git_env(self.tmp.name))
        return subprocess.run([sys.executable, EVALS] + list(args), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=120)

    def test_finish_at_once_scores_done_and_hidden_fail(self):
        out = os.path.join(self.tmp.name, "rows.json")
        r = self._run("--provider", "fake", "--case", "debug_symptom", "--out", out)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertRegex(r.stdout, r"fake\s+fake-model\s+debug_symptom\s+debug_symptom\s+done\s+1\s")
        self.assertRegex(r.stdout, r"\s0\s+FAIL\s+0\n")  # check passed, hidden failed, nothing changed
        self.assertIn("Per case", r.stdout)
        self.assertRegex(r.stdout, r"fake\s+debug_symptom\s+1\s+1\s+0\s+0\s+0%")
        self.assertIn("Per category", r.stdout)
        self.assertNotIn(FAKE_KEY, r.stdout)
        with open(out) as f:
            text = f.read()
        self.assertNotIn(FAKE_KEY, text)
        rows = json.loads(text)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["status"], row["hidden_passed"], row["passed"]), ("done", False, False))
        self.assertEqual((row["check_exit"], row["changed_files"], row["steps"]), (0, 0, 1))
        self.assertIn("***", row["summary"])
        self.assertIn("FAILED", row["hidden_output"])
        leftovers = [n for n in os.listdir(self.tmp.name) if n.startswith("dllm-eval-")]
        self.assertEqual(leftovers, [])  # workdirs are deleted without --keep

    def test_keep_prints_workdir_and_trajectory(self):
        r = self._run("--provider", "fake", "--case", "spec_module", "--keep")
        self.assertEqual(r.returncode, 0, r.stdout)
        kept = [l.split(": ", 1)[1] for l in r.stdout.splitlines() if l.startswith("kept ")]
        trajs = [l.split(": ", 1)[1] for l in r.stdout.splitlines() if l.startswith("trajectory ")]
        self.assertEqual(len(kept), 1, r.stdout)
        self.assertTrue(os.path.isfile(os.path.join(kept[0], "README.md")))
        self.assertTrue(os.path.isfile(trajs[0]))
        self.assertIn("check_failed", r.stdout)  # finish without test_textutil.py: check fails 3 times

    def test_unknown_case_exits_1(self):
        r = self._run("--case", "nope")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("Unknown case(s): nope", r.stdout)

    def test_no_keys_exits_1_with_smoke_message(self):
        self._keys("")
        r = self._run("--case", "spec_module")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("skip fake: no key (set FAKE_API_KEY)", r.stdout)
        self.assertIn("No provider has an API key", r.stdout)


if __name__ == "__main__":
    unittest.main()
