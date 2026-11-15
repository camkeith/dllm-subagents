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

## When not to delegate

- Design decisions or ambiguous requirements.
- Security-sensitive code.
- Subtle debugging.
- Anything you can do in one or two small edits.

## How to brief

Put in `task`: the goal, constraints, existing patterns to follow (name the files), and what done means. The subagent has no memory of this conversation, so say everything it needs.

- Always pass `workdir`: the absolute path of the git repo (or worktree). It must be a git repo.
- Pass `files` for paths you already know are relevant; their contents are given to the subagent up front.
- Pass `check_command` whenever one exists (for example `pytest tests/test_api.py`). The harness runs it after the subagent finishes and lets the subagent fix failures a few times.
- Optional: `provider` and `model` override the configured defaults; `max_steps` (1 to 150) and `max_seconds` (10 to 1800) override limits.

## Running it

1. Call `fast_agent`. It waits up to about 50 seconds, then returns.
2. While the result says `status: running`, call `fast_agent_wait` with the `run_id` until it finishes.
3. To stop a run, call `fast_agent_cancel` with the `run_id`.

## Reading the result

`status` is one of:

- `done`: the subagent called finish (and any check passed).
- `check_failed`: it finished but `check_command` still fails; read the check output.
- `limit_reached`: it hit a step, time, or token limit; the work may be partial.
- `cancelled`: the run was stopped.
- `error`: something failed (see the `error` line, for example a missing API key).

The result also has the subagent's `summary`, a `check:` line (command, exit code, attempts), stats, the changed files, and the full git `diff` computed by the harness, not by the subagent. Trust the diff over the summary.

## After the run

- Review the diff before moving on. Do not assume it is right because the status is `done`.
- Fix small problems yourself. For larger gaps, send a follow-up `fast_agent` task that says exactly what is wrong.
- If the run went wrong, call `fast_agent_undo` with the `run_id` to revert its changes.
- The subagent can only run read-only git commands (status, diff, log, show, ls-files, grep, blame, rev-parse). Committing is your job.

## While a run is active

- Do not edit files in that `workdir`: your edits would show up in the run's diff and `fast_agent_undo` would revert them. Work elsewhere or wait.
- Only one run per repo or worktree at a time. For parallel runs, create separate git worktrees and pass each as its own `workdir`.
