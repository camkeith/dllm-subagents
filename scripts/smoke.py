#!/usr/bin/env python3
"""Live smoke test: run the real agent loop against each configured provider.

Usage: python3 scripts/smoke.py [--provider NAME ...] [--models] [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plugins" / "dllm-subagents" / "server"))

from dllm.agent import Agent  # noqa: E402
from dllm.config import ConfigError, load_settings, resolve_api_key  # noqa: E402
from dllm.llm import ChatClient  # noqa: E402
from dllm.prompts import build_task_message  # noqa: E402
from dllm.tools import ToolContext, scrubbed_env  # noqa: E402

TASK = "Create calc.py with add(a, b) and mul(a, b), and test_calc.py with unittest tests for both."
CHECK = "python3 -m unittest -q test_calc"
MODEL_LIST_LIMIT = 40


def list_models(profile, key):
    request = urllib.request.Request(
        profile.base_url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    items = data.get("data") if isinstance(data, dict) else None
    return sorted(str(m.get("id")) for m in (items or []) if isinstance(m, dict))


def family(model):
    name = model.rsplit("/", 1)[-1].split(":")[0]
    return name.split("-")[0] or name


def show_models(profile, key):
    try:
        ids = list_models(profile, key)
    except (OSError, ValueError) as exc:
        print("%s: could not list models: %s" % (profile.name, type(exc).__name__))
        return False
    shown = ids if len(ids) < MODEL_LIST_LIMIT else [i for i in ids if family(profile.model) in i]
    print("%s: %d models (%s)" % (profile.name, len(ids),
                                  "all" if len(ids) < MODEL_LIST_LIMIT else "matching %r" % family(profile.model)))
    for model_id in shown:
        print("  " + model_id)
    present = profile.model in ids
    print("  configured model %s: %s" % (profile.model, "present" if present else "MISSING"))
    return present


def run_provider(profile, key, secret_names, limits, keep, redact=()):
    workdir = Path(tempfile.mkdtemp(prefix="dllm-smoke-%s-" % profile.name))
    traj = workdir / "trajectory.json"
    try:
        subprocess.run(["git", "init", "-q"], cwd=str(workdir), check=True)
        ctx = ToolContext(workdir, threading.Event(), env=scrubbed_env(secret_names))
        agent = Agent(ChatClient(profile, key), ctx, limits, check_command=CHECK, trajectory_path=traj,
                      redact=redact)
        result = agent.run(build_task_message(TASK, str(workdir), CHECK, [], []))
    finally:
        if not keep:
            shutil.rmtree(str(workdir), ignore_errors=True)
    return result, workdir, traj


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", metavar="NAME")
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print("Config error: %s" % exc)
        return 1
    names = args.provider or list(settings.profiles)
    unknown = [n for n in names if n not in settings.profiles]
    if unknown:
        print("Unknown provider(s): %s (known: %s)" % (", ".join(unknown), ", ".join(settings.profiles)))
        return 1
    secret_names = [p.api_key_env for p in settings.profiles.values()]
    known_keys = []  # every key we can resolve is redacted, like RunManager does
    for other in settings.profiles.values():
        try:
            known_keys.append(resolve_api_key(other))
        except ConfigError:
            pass

    keyed = []
    for name in names:
        profile = settings.profiles[name]
        try:
            keyed.append((profile, resolve_api_key(profile)))
        except ConfigError:
            print("skip %s: no key (set %s)" % (name, profile.api_key_env))
    if not keyed:
        print("No provider has an API key. Put NAME=value lines in %s or export them." %
              (Path(os.environ.get("DLLM_SUBAGENTS_CONFIG_DIR") or "~/.config/dllm-subagents") / "keys.env"))
        return 1

    if args.models:
        for profile, key in keyed:
            show_models(profile, key)
        return 0

    header = ("provider", "model", "status", "steps", "seconds", "in_tok", "out_tok", "check")
    rows, failures, kept = [], [], []
    for profile, key in keyed:
        try:
            result, workdir, traj = run_provider(profile, key, secret_names, settings.limits, args.keep,
                                                 known_keys)
        except Exception as exc:
            rows.append((profile.name, profile.model, "crashed", "-", "-", "-", "-", "-"))
            failures.append((profile.name, "%s: %s" % (type(exc).__name__, exc)))
            continue
        check = str(result.check.exit_code) if result.check else "-"
        rows.append((profile.name, profile.model, result.status, str(result.steps),
                     "%.1f" % result.seconds, str(result.input_tokens), str(result.output_tokens), check))
        if result.status != "done":
            failures.append((profile.name, result.error or result.limit or result.summary))
        if args.keep:
            kept.append((profile.name, workdir, traj))
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    for name, text in failures:
        print("%s failed: %s" % (name, text))
    for name, workdir, traj in kept:
        print("kept %s" % workdir)
        print("trajectory %s: %s" % (name, traj))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
