---
name: delegate-fast
description: Use when a build or edit job is well scoped and would mean writing a lot of code; hand it to the fast subagent with fast_agent instead of typing it yourself.
---

# Delegate to the fast subagent

`fast_agent` runs a fast, cheap coding model in a loop (read, write, edit, bash, finish) inside a git repo. You brief it, review its diff, and stay in charge of design and commits.

<!-- Draft rules: rewrite these in your own words to match how you work. -->
## When to delegate

- Writing a module from a clear spec.
- Adding endpoints or handlers that follow an existing pattern.
- Applying the same change across many files.
- Writing tests for existing code.
- Making a failing test pass.
- Fixing lint or type errors.
- Any job where you would otherwise type a lot of code.

Measured: the repo's eval suite (`scripts/evals.py`) has small jobs for a module from a spec, a handler following a pattern, a rename across files, tests for existing code, a failing test, an edit in a large file, and adding type hints. Mercury passed their hidden checks (2 runs per job, 2026-11-16) when briefed with a check command. Lint and type-checker fixes are not covered yet.

## When not to delegate

- Design decisions or ambiguous requirements.
- Security-sensitive code.
- Subtle debugging.
- Anything you can do in one or two small edits.

## How to brief

Put in `task`: the goal, constraints, existing patterns to follow (name the files), and what done means. The subagent has no memory of this conversation, so say everything it needs.

- Always pass `workdir`: the absolute path of the git repo (or worktree). It must be a git repo.
- Before delegating, make sure dependency and build directories (`node_modules`, `.venv`, `dist`, `build`) are git-ignored, especially in a fresh `git init` repo. Otherwise every file the run installs or builds is snapshotted and listed as a change.
- Pass `files` for paths you already know are relevant; their contents are given to the subagent up front.
- Pass `check_command` whenever one exists (for example `pytest tests/test_api.py`). The harness runs it after the subagent finishes and lets the subagent fix failures a few times.
- Tokens and other secret-looking environment variables are removed from the subagent's shell and check command. If a command needs one (for example `NPM_TOKEN`), the user must list it in `pass_env` in the plugin's `config.json`.
- Optional: `provider` and `model` override the configured defaults; `max_steps` (1 to 150) and `max_seconds` (10 to 1800) override limits.

## Running it

1. Call `fast_agent`. It waits up to about 50 seconds, then returns.
2. While the result says `status: running`, call `fast_agent_wait` with the `run_id` until it finishes.
3. To stop a run, call `fast_agent_cancel` with the `run_id`. It can return `status: running` while a model call is in flight; call `fast_agent_wait` to see the run end.

Setup errors (no API key, a bad or missing `workdir`, not a git repo, another run already active in the repo, an unknown provider) come back as a plain tool error message, not as a `status: error` result. No run was started; fix the cause and call `fast_agent` again.

## Reading the result

`status` is one of:

- `done`: the subagent called finish (and any check passed).
- `check_failed`: it finished but `check_command` still fails; read the check output.
- `limit_reached`: it hit a step, time, or token limit; the work may be partial.
- `cancelled`: the run was stopped.
- `error`: the run failed after it started (see the `error` line, for example the provider returned an error).

The result also has the subagent's `summary`, a `check:` line (command, exit code, attempts), stats, the changed files, and the git `diff` computed by the harness, not by the subagent. Trust the diff over the summary.

Long results are cut. A cut diff ends with a `git diff <before> <after>` command that shows the rest; a long file list ends with a `git diff --stat <before> <after>` command that lists every changed file. Run them in the repo.

## After the run

- Review the diff before moving on. Do not assume it is right because the status is `done`.
- Fix small problems yourself. For larger gaps, send a follow-up `fast_agent` task that says exactly what is wrong.
- If the run went wrong, call `fast_agent_undo` with the `run_id` to revert its changes.
- The subagent can only run read-only git commands (status, diff, log, show, ls-files, grep, blame, rev-parse). Committing is your job.

## While a run is active

- Do not edit files anywhere in that repository, not only in `workdir`: snapshots and undo cover the whole repo, even when `workdir` is a subdirectory. Your edits would show up in the run's diff and `fast_agent_undo` would revert them. Work elsewhere or wait.
- Only one run per repo or worktree at a time. For parallel runs, create separate git worktrees and pass each as its own `workdir`.
