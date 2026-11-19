#!/usr/bin/env python3
"""Live evals: run realistic delegation cases against each configured provider and score each run
with hidden checks the model never sees.

Usage: python3 scripts/evals.py [--provider NAME ...] [--case NAME ...] [--repeat N] [--keep] [--out FILE]

A run passes only when the agent's status is done AND the hidden check passed. Exits 0 even when
cases fail (it is a measurement tool); exits 1 only for setup errors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import eval_cases
import smoke
from dllm.config import clamp_limits
from dllm.shell import run_command
from dllm.tools import redact, run_env

HIDDEN_TIMEOUT = 300  # write_tests alone runs the model's tests up to 4 times, 60 s each
OUTPUT_TAIL_CHARS = 2000
# Hooks and fsmonitor off: after a run, .git/config is model-controlled (as in dllm.snapshot).
GIT_CONFIG = ["-c", "user.name=dllm-evals", "-c", "user.email=evals@example.invalid",
              "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false"]
HEADER = ("provider", "model", "case", "category", "status", "steps", "seconds", "in_tok", "out_tok",
          "check", "hidden", "changed")


def git(repo, *args, env):
    """Run git in repo with env (the run's scrubbed environment) minus GIT_* overrides."""
    env = {k: v for k, v in env.items() if not k.startswith("GIT_")}
    return subprocess.run(["git"] + GIT_CONFIG + list(args), cwd=str(repo), check=True, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout


def prepare_repo(case, repo, env):
    eval_cases.write_files(repo, eval_cases.starting_files(case))
    git(repo, "init", "-q", env=env)
    git(repo, "add", "-A", env=env)
    git(repo, "commit", "-q", "--no-verify", "-m", "starting files", env=env)


def changed_files(repo, env):
    out = git(repo, "status", "--porcelain", "--untracked-files=all", env=env)
    return len([line for line in out.splitlines() if line.strip()])


def run_hidden(case, repo, env):
    eval_cases.write_files(repo, case.hidden.files)
    res = run_command(case.hidden.command, str(repo), HIDDEN_TIMEOUT, env=env)
    output = res.output + ("\n[hidden check timed out after %ds]" % HIDDEN_TIMEOUT if res.timed_out else "")
    return res.returncode == 0 and not res.timed_out, output


def output_tail(text, known_keys):
    """The last OUTPUT_TAIL_CHARS of text, redacted before the cut so the cut cannot split a key."""
    return redact(text, known_keys)[-OUTPUT_TAIL_CHARS:]


def check_out_path(path):
    """Fail on a --out path that cannot be written, without touching an existing file there."""
    path = Path(path)
    if path.is_dir():
        raise IsADirectoryError(21, "is a directory")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("", encoding="utf-8")
    tmp.unlink()


def write_rows(path, rows, known_keys):
    """Write every row so far as JSON, replacing the file atomically. Values are redacted before
    encoding: JSON escaping could otherwise hide a key containing a quote or backslash."""
    clean = [{k: redact(v, known_keys) if isinstance(v, str) else v for k, v in row.items()} for row in rows]
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def run_case(profile, key, case, settings, secret_names, known_keys, keep, repeat):
    base = Path(tempfile.mkdtemp(prefix="dllm-eval-%s-%s-" % (profile.name, case.name)))
    repo, traj = base / "repo", base / "trajectory.json"  # trajectory outside the repo: not a changed file
    row = OrderedDict([
        ("provider", profile.name), ("model", profile.model), ("case", case.name),
        ("category", case.category), ("repeat", repeat), ("status", "crashed"), ("steps", None),
        ("seconds", None), ("input_tokens", None), ("output_tokens", None), ("check_exit", None),
        ("hidden_passed", None), ("changed_files", None), ("passed", False), ("error", None),
        ("summary", None), ("hidden_output", None),
    ])
    env = run_env(secret_names, settings.pass_env)
    try:
        repo.mkdir()
        prepare_repo(case, repo, env)
        limits = clamp_limits(settings.limits, case.limits.get("max_steps"), case.limits.get("max_seconds"))
        result = smoke.run_agent(profile, key, repo, case.task, case.check, secret_names, limits, traj,
                                 known_keys, settings.pass_env, case.hint_files)
        row.update(status=result.status, steps=result.steps, seconds=round(result.seconds, 1),
                   input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                   check_exit=result.check.exit_code if result.check else None,
                   error=result.error or result.limit, summary=result.summary)
        try:
            row["changed_files"] = changed_files(repo, env)
        except (subprocess.CalledProcessError, OSError):  # e.g. the model broke .git; still score it
            pass
        passed, output = run_hidden(case, repo, env)
        row["hidden_passed"] = passed
        row["hidden_output"] = output_tail(output, known_keys)
        row["passed"] = result.status == "done" and passed
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if not keep:
            shutil.rmtree(str(base), ignore_errors=True)
    return row, repo, traj


def _cell(value):
    return "-" if value is None else str(value)


def table_row(row):
    hidden = {True: "pass", False: "FAIL", None: "-"}[row["hidden_passed"]]
    return (row["provider"], row["model"], row["case"], row["category"], row["status"],
            _cell(row["steps"]), _cell(row["seconds"]), _cell(row["input_tokens"]),
            _cell(row["output_tokens"]), _cell(row["check_exit"]), hidden, _cell(row["changed_files"]))


def print_table(header, rows):
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())


def summary_rows(rows, field):
    """Per (provider, field) totals: runs, done, hidden passes, passes, and pass rate (passes / runs)."""
    groups = OrderedDict()
    for row in rows:
        g = groups.setdefault((row["provider"], row[field]), [0, 0, 0, 0])
        g[0] += 1
        g[1] += row["status"] == "done"
        g[2] += bool(row["hidden_passed"])
        g[3] += row["passed"]
    return [(provider, name, str(runs), str(done), str(hidden), str(passed), "%d%%" % (100 * passed // runs))
            for (provider, name), (runs, done, hidden, passed) in groups.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", action="append", metavar="NAME")
    parser.add_argument("--case", action="append", metavar="NAME",
                        help="one of: %s" % ", ".join(c.name for c in eval_cases.CASES))
    parser.add_argument("--repeat", type=int, default=1, metavar="N")
    parser.add_argument("--keep", action="store_true", help="keep each workdir and trajectory")
    parser.add_argument("--out", metavar="FILE", help="write every row as JSON")
    args = parser.parse_args(argv)

    if args.repeat < 1:
        print("--repeat must be at least 1")
        return 1
    names = args.case or [c.name for c in eval_cases.CASES]
    unknown = [n for n in names if n not in eval_cases.CASES_BY_NAME]
    if unknown:
        print("Unknown case(s): %s (known: %s)" % (", ".join(unknown),
                                                   ", ".join(c.name for c in eval_cases.CASES)))
        return 1
    cases = [eval_cases.CASES_BY_NAME[n] for n in names]
    try:
        settings, keyed, secret_names, known_keys = smoke.load_providers(args.provider)
    except smoke.SetupError as exc:
        print(exc)
        return 1

    if args.out:
        try:
            check_out_path(args.out)  # a bad path fails here, before any paid run
        except OSError as exc:
            print("Cannot write --out file %s: %s" % (args.out, exc.strerror or exc))
            return 1
    rows, kept, out_ok = [], [], True
    for profile, key in keyed:
        for case in cases:
            for repeat in range(1, args.repeat + 1):
                print("running %s %s #%d ..." % (profile.name, case.name, repeat), file=sys.stderr, flush=True)
                row, repo, traj = run_case(profile, key, case, settings, secret_names, known_keys,
                                           args.keep, repeat)
                rows.append(row)
                if args.out:  # after every run, so a crash or Ctrl-C keeps the finished rows
                    try:
                        write_rows(args.out, rows, known_keys)
                        out_ok = True
                    except OSError as exc:  # keep running; the table below still has every row
                        out_ok = False
                        print("could not write %s: %s" % (args.out, exc), file=sys.stderr)
                if args.keep:
                    kept.append(("%s %s #%d" % (profile.name, case.name, repeat), repo, traj))

    def show(text):
        print(redact(text, known_keys))

    print_table(HEADER, [table_row(r) for r in rows])
    print()
    print("Per case (pass = status done and hidden check passed):")
    print_table(("provider", "case", "runs", "done", "hidden", "pass", "rate"), summary_rows(rows, "case"))
    print()
    print("Per category:")
    print_table(("provider", "category", "runs", "done", "hidden", "pass", "rate"),
                summary_rows(rows, "category"))
    problems = [r for r in rows if r["error"] and not r["passed"]]
    if problems:
        print()
    for r in problems:
        show("%s %s #%d %s: %s" % (r["provider"], r["case"], r["repeat"], r["status"], r["error"]))
    for label, repo, traj in kept:
        print("kept %s: %s" % (label, repo))
        print("trajectory %s: %s" % (label, traj))
    if args.out:
        print(("wrote %s" if out_ok else "could not write %s (see the errors above)") % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
