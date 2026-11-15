# dllm-subagents: a fast-model subagent plugin for Claude Code and Codex

Date: 2026-11-14
Status: draft, awaiting approval

## Intent

The user wants to code faster. In Claude Code and Codex, a big slow part of the work is the
frontier model typing out code at tens of tokens per second. This plugin lets the orchestrating
agent (Claude Code or Codex) hand a well-scoped build or edit job to a small agent running on a
very fast model (Inception Mercury, Cerebras, Groq, OpenRouter, or any OpenAI-compatible
provider). That agent does the work in the user's repo with its own tools, then reports back
so the orchestrator can review the result and move on.

The orchestrator keeps the judgment work: understanding the codebase, deciding what to build,
reviewing the result. The fast subagent does the typing: writing modules, applying edits,
running tests and fixing what breaks.

## Goals

- One plugin that installs in both Claude Code and Codex from the GitHub repo
  `camkeith/dllm-subagents`.
- A real agent harness: the subagent loops with read, write, edit, and bash tools until it
  finishes or hits a limit.
- Swappable model: any OpenAI-compatible chat-completions endpoint with tool calling. Ships
  ready-made profiles for Mercury, Cerebras, Groq, and OpenRouter. The user can add more.
- Fast end to end: small prompts, few tool definitions, no heavy runtime start-up.
- Results the orchestrator can trust: a diff computed by the harness from git, not the
  model's claims, and a pass/fail from a check command the harness runs itself.
- A one-call undo of a subagent run.

## Non-goals

- Replacing the orchestrator's model. Claude Code and Codex keep running on their own models.
  (Inception's `mercury-cc` and the community proxies do full replacement; this does not.)
- Proxying the Anthropic Messages API or OpenAI Responses API.
- A TUI, interactive mode, or chat with the subagent.
- An OS-level sandbox (the user chose guardrails, see below).
- Parallel subagent runs in the same working tree (v1 allows one active run per directory).
- Context compaction for very long runs. Runs are meant to be short; limits end them.

## Anti-goals (would make it a failure even if it technically works)

- Slower than the orchestrator just doing the job itself, for the jobs the skill routes to it.
- Silent damage: the subagent changes files and the orchestrator can't see exactly what
  changed or can't undo it.
- Tied to Mercury. The Mercury API shuts down after December 28, 2026
  (https://docs.inceptionlabs.ai/support/wind-down). The plugin must keep working on other
  providers with only a config change.
- A heavy install (a big Node/Python dependency tree, a global CLI, a long first start).

## Constraints and facts the design rests on

All verified during design research on 2026-11-14 unless marked otherwise.

- **Mercury API:** OpenAI-compatible at `https://api.inceptionlabs.ai/v1`, chat completions
  only (no Responses API). `mercury-3`: 1M context, 131,072 max output, tool calling with
  parallel calls supported, `tool_choice` `auto|none|required`, temperature 0.5 to 1.0,
  `reasoning_effort` `instant|low|medium|high`. Docs examples use `max_completion_tokens`.
  Shuts down after 2026-12-28.
- **Cerebras:** OpenAI-compatible at `https://api.cerebras.ai/v1` with tool calling. Free
  trial limits `gpt-oss-120b` to 5 requests/minute and 30K uncached tokens/minute, which is
  unusable for an agent loop; pay-as-you-go is 1K RPM and 500K uncached TPM. Cached tokens
  count separately, so stable message prefixes help. Requests are rate-limited up front using
  `max_completion_tokens` as the output estimate, so set it sensibly.
- **Groq:** OpenAI-compatible at `https://api.groq.com/openai/v1` with tool calling on some
  models. Exact current model IDs not verified.
- **OpenRouter:** OpenAI-compatible at `https://openrouter.ai/api/v1`. The `:nitro` model
  suffix routes to the highest-throughput host. Not verified against this design.
- **Codex MCP:** stdio servers get a `tool_timeout_sec` default of **60 seconds** and a
  `startup_timeout_sec` default of 10 seconds. Codex forwards only the env vars it is told to
  (`env`, `env_vars`), so API keys in the user's shell may not reach the server.
- **Claude Code MCP:** stdio tool calls default to about a 28-hour timeout with a 30-minute
  idle window. Claude Code exports `CLAUDE_PLUGIN_ROOT`, `CLAUDE_PLUGIN_DATA`, and
  `CLAUDE_PROJECT_DIR` to stdio MCP servers and substitutes `${CLAUDE_PLUGIN_ROOT}` in
  `.mcp.json` `command`, `args`, and `env`.
- **Plugin formats:** Claude Code uses `.claude-plugin/plugin.json`, `.mcp.json`, and
  `skills/<name>/SKILL.md` at the plugin root, with a repo-level
  `.claude-plugin/marketplace.json`. Codex docs say Codex "also accepts legacy and
  Claude-compatible manifests", reads a "legacy-compatible marketplace at
  `$REPO_ROOT/.claude-plugin/marketplace.json`" (stated for the ChatGPT desktop app), and sets
  `CLAUDE_PLUGIN_ROOT` for plugin hooks. Whether Codex CLI 0.164 loads a Claude-format
  plugin's `.mcp.json` and substitutes `${CLAUDE_PLUGIN_ROOT}` in it is **not verified**. The
  user has Codex CLI 0.164.0 and Claude Code installed locally for testing.
- **mini-swe-agent** (https://github.com/SWE-agent/mini-swe-agent, MIT): the agent loop is
  `src/minisweagent/agents/default.py` (about 190 lines). It drives the loop with exceptions
  (`FormatError`, `LimitsExceeded`, `TimeExceeded`, `Submitted`) and an `exit` message, and
  enforces `step_limit`, `cost_limit`, `wall_time_limit_seconds`, and
  `max_consecutive_format_errors`. `environments/local.py` runs commands with
  `start_new_session` and kills the whole process group on timeout. It depends on pydantic,
  jinja2, and litellm, and has a single `bash` tool.
- **Pi's edit tool** (https://github.com/earendil-works/pi,
  `packages/coding-agent/src/core/tools/edit.ts`, MIT): `edit(path, edits[])` where each edit is
  `{oldText, newText}`; every `oldText` must match exactly and uniquely in the original file;
  edits are matched against the original, not applied incrementally; edits must not overlap.
  It normalizes CRLF to LF and strips a BOM before matching, and falls back to a fuzzy match
  (trailing whitespace stripped, smart quotes normalized) when the exact match fails.
- **Local machine:** macOS, Python 3.14 (Homebrew) and `uv` on PATH, Node 22. No provider
  keys set in the shell yet (the user has them; see Build order). macOS's system
  `/usr/bin/python3` is 3.9.

## Approaches considered

1. **Vendor mini-swe-agent's core and scale it down (chosen).** Port its small loop, limits,
   and process handling; replace litellm with a direct HTTP client and provider profiles; add
   read/write/edit tools modeled on Pi's edit semantics; wrap it in a small MCP server. About
   1 to 1.5k lines we own. Small prompts and hard limits. We own edit-tool quality.
2. **Install Pi and wrap its CLI** (`pi --mode json`). Best edit tool and least code (about 300
   lines), but no step or cost caps (would have to kill it from outside), a large fast-moving
   dependency, and a global Node install.
3. **Install Qwen Code and wrap its CLI** (`qwen -p --output-format json`). Best built-in
   headless controls, but the heaviest option, with prompts built for big models, so each step
   costs more tokens and time.

Ruled out: OpenCode (~830k lines) and Goose (~358k) can't be trimmed, only installed; Gemini
CLI only speaks Gemini's API; Crush is source-available (FSL), not open source; Aider is not an
autonomous agent loop and was last pushed May 2026.

The user picked approach 1 because it is the "pull in a working harness and scale it down"
route they asked for. If the edit tool proves weak, approach 2 can later sit behind the same
MCP tools without changing the plugin's surface.

## Decisions made with the user

- Full agent harness (not a one-shot writer).
- Model is swappable; support Mercury, Cerebras, Groq, OpenRouter, and any OpenAI-compatible
  provider.
- Build by trimming mini-swe-agent (approach 1).
- Shell safety: **guardrails**, not an OS sandbox and not a shell-less agent. Commands run in
  the repo with a per-command timeout; a short blocklist stops common footguns; a snapshot
  before each run lets the orchestrator undo it. The model still has network access and can
  read outside the repo through bash. The user accepted that.

## Design

### Shape of the plugin

One GitHub repo that is both a Claude Code marketplace and a Codex marketplace, containing one
plugin. The plugin holds:

- an MCP server (stdio) exposing three tools to the orchestrator, and
- one skill that teaches the orchestrator when and how to delegate.

Suggested layout (exact file split is the builder's call):

```text
dllm-subagents/                          repo root, acts as the marketplace
├── .claude-plugin/marketplace.json      Claude Code marketplace (Codex may also read it)
├── .agents/plugins/marketplace.json     Codex marketplace, only if Codex needs its own
├── plugins/dllm-subagents/
│   ├── .claude-plugin/plugin.json
│   ├── .codex-plugin/plugin.json        only if Codex won't load the Claude manifest
│   ├── .mcp.json
│   ├── skills/delegate-fast/SKILL.md
│   └── server/                          the MCP server and harness, Python stdlib only
├── tests/
├── README.md                            install, keys, providers, limits
├── LICENSE                              MIT
└── THIRD_PARTY_NOTICES                  mini-swe-agent and Pi MIT notices for ported code
```

### Runtime: Python standard library only (my call)

The server and harness use only the Python standard library: `urllib.request` for HTTP,
`subprocess` for commands, `json` and `threading`. MCP over stdio is newline-delimited
JSON-RPC 2.0, and only `initialize`, `notifications/initialized`, `ping`, `tools/list`, and
`tools/call` are needed, so a small hand-written loop is enough.

Why: no install step, no virtualenv, nothing to download on first start (Codex's start-up
timeout is 10 seconds), and nothing that breaks when a dependency moves. It means dropping
mini-swe-agent's pydantic, jinja2, and litellm dependencies, which the trim does anyway.

Launch with `python3` from PATH. Code must run on Python 3.9 or newer (macOS system Python is
3.9): use `from __future__ import annotations`, no `match` statements. If the builder finds the
hand-written MCP layer fragile in testing against either host, switching to the official `mcp`
Python SDK via `uv run` is acceptable, as long as first start stays well under 10 seconds.

### Tools exposed to the orchestrator

**`fast_agent`**: start a subagent run and wait for it, up to a cap.

Inputs:

- `task` (string, required): the brief. What to build or change, constraints, and what "done"
  means.
- `workdir` (string, required): absolute path of the repo to work in. The skill tells the
  orchestrator to always pass it. If omitted, fall back to `CLAUDE_PROJECT_DIR`, then the
  server's working directory.
- `files` (list of strings, optional): paths the orchestrator already knows are relevant.
  Their contents go into the first user message so the subagent doesn't spend steps finding
  them. Cap the total (for example 100 KB) and say so in the message when truncated.
- `check_command` (string, optional): a command that proves the work is done, for example
  `npm test -- auth` or `pytest tests/test_api.py`. See "Check loop" below.
- `provider` (string, optional): profile name, for example `mercury`, `cerebras`. Default
  from config.
- `model` (string, optional): override the profile's model.
- `max_steps`, `max_seconds` (optional): override the default limits, within hard caps.

Behavior: validates inputs, takes the pre-run snapshot, starts the run in a background thread,
then waits up to about 50 seconds (configurable, must stay under Codex's 60-second tool
timeout). If the run finishes in time, return the full result. Otherwise return
`status: "running"` with a `run_id` and a short progress line (steps so far, last action).

**`fast_agent_wait`**: `run_id` in; waits up to about 50 seconds more; returns the result or
another `running` status. The orchestrator calls it repeatedly until the run ends.

**`fast_agent_undo`**: `run_id` in; reverts exactly that run's changes (see Snapshots). Refuses,
with a clear message, if the files have changed since the run ended in ways that conflict.
Refuses while the run is still active (cancel it first).

**`fast_agent_cancel`**: `run_id` in; stops the run (my call, added after review). Sets a stop
flag checked before every model call and tool execution, and kills the process group of any
bash command or check that is running. The run ends with `status: cancelled` and keeps its
changes; the result shows the diff so far, and `fast_agent_undo` reverts it if wanted. No
automatic undo, because partial work is sometimes worth keeping.

Host cancellation (my call): if the host cancels a pending `fast_agent` or `fast_agent_wait`
call (the MCP `notifications/cancelled` message, sent when the user presses Esc in Claude Code),
the server cancels the run too, as if `fast_agent_cancel` were called. The user expects Esc to
stop the work, and a run that keeps editing after Esc is the kind of silent change this design
exists to prevent. The 50-second wait cap keeps hosts from cancelling on their own timeouts.

Result of a finished run (returned as MCP text content; the format is the builder's call, but
it must be compact and easy for a model to read):

- `status`: `done`, `check_failed`, `limit_reached` (which limit), `cancelled`, or `error`
  (with message)
- `summary`: the subagent's own summary from its `finish` call
- `changed_files`: path, added/modified/deleted, lines added and removed
- `diff`: unified diff of the run's changes, truncated to a cap (for example 20 KB) with a note
  on how to see the rest (`git diff <before-tree> <after-tree>`)
- `check`: command, exit code, and the tail of its output, when a check was given
- `stats`: steps, seconds, input and output tokens, provider and model
- `run_id` and `log_path` (the saved trajectory)

Run records persist (my call, added after review). When a run starts and ends, write a small
JSON record next to its trajectory log: `run_id`, `workdir`, before-tree and after-tree SHAs,
status, timestamps. The trees are already git objects in the repo, so the record is all undo
needs. `fast_agent_undo` loads the record from disk when the run isn't in memory, so undo works
after a server restart. Unreferenced tree objects survive until `git gc` prunes them (two weeks
by default); if they are gone, undo says so.

Process lifecycle: when the server shuts down (stdin closes, SIGTERM, SIGINT), it cancels active
runs and kills their bash process groups before exiting, so no command outlives it. A run cut
off this way has its record marked `cancelled` with whatever after-tree can be taken; if none
can, undo explains that it can't compute the run's changes.

One active run per `workdir`. A second `fast_agent` call for the same directory while one is
running returns an error saying a run is in progress and naming its `run_id`.

### The subagent's tools

Five tools, short descriptions, to keep every request small:

- `read(path, offset?, limit?)`: returns file contents with line numbers. Default limit around
  2,000 lines; truncate very long lines.
- `write(path, content)`: create or overwrite a file. Creates parent directories.
- `edit(path, edits[])`: Pi's semantics. Each edit is `{old_text, new_text}`; each `old_text`
  must match exactly once in the original file; edits must not overlap; all are matched against
  the original and applied together, or none are. CRLF and BOM handled as Pi does. Fuzzy
  fallback (trailing whitespace, smart quotes) is the builder's call, but if used, it must still
  require a unique match. Errors must say exactly why (not found, found N times, overlap) so the
  model can fix its next call.
- `bash(command, timeout?)`: runs in `workdir` with a fresh shell each call, per-command
  timeout (default 60 seconds, hard cap 300), kills the whole process group on timeout (port
  mini-swe-agent's `_run`). Output is stdout and stderr combined, truncated to keep the head and
  the tail (for example 10K characters total) with a note on how much was cut.
- `finish(summary)`: ends the run. The summary is what the orchestrator sees.

Path rule for `read`, `write`, `edit`: the resolved path (after following symlinks) must be
inside `workdir`. Anything else is an error. Bash is not path-restricted beyond its working
directory; that is the accepted guardrails trade-off.

### Guardrails on bash

These stop common mistakes by a small model. They are not a security boundary, and the README
must say so.

- **Git:** allow read-only subcommands (`status`, `diff`, `log`, `show`, `ls-files`, `grep`,
  `blame`, `rev-parse`). Block every other git subcommand, including `push`, `commit`,
  `reset`, `checkout`, `restore`, `switch`, `clean`, `stash`, `rebase`, `merge`, `branch`,
  `tag`, `worktree`. Committing and branching belong to the orchestrator. (`git stash` matters
  in particular: the stash stack is shared across the user's worktrees and sessions.)
- **Always blocked:** `sudo`, `rm -rf` (or `rm -fr`, `rm -r -f`) aimed at `/`, `~`, `$HOME`, or a
  path outside `workdir`, `mkfs`, `dd of=/dev/...`, `shutdown`, `reboot`, and piping a download
  into a shell (`curl ... | sh`, `wget ... | bash`).
- A blocked command is not run. The tool returns an error naming the rule, so the model can
  choose another way.
- Matching should catch the command anywhere in a compound line (`&&`, `;`, `|`, `$(...)`,
  backticks), not just at the start. Exact matching approach is the builder's call; tests must
  cover the compound cases.

### The agent loop (ported from mini-swe-agent)

Keep mini-swe-agent's structure: `run()` loops `step()` until an `exit` message; `step()` is
`query()` then `execute_actions()`; control flow through exceptions; limits checked before each
model call; consecutive format errors counted and capped; trajectory saved after every step.

Changes from mini-swe-agent:

- Model calls go straight to the provider over HTTP (no litellm). Request body: `model`,
  `messages`, `tools`, `tool_choice: "auto"`, the profile's max-output field and value, and the
  profile's `extra_body` merged in. Non-streaming.
- Five tools instead of bash only; `finish` replaces the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  sentinel.
- A response with no tool calls counts as a format error: the harness replies telling the
  model to call a tool, or `finish` if it is done. (Same as mini-swe-agent.)
- Multiple tool calls in one response run in order, each producing one `tool` message.
- A token budget replaces the dollar `cost_limit`, since providers report tokens but not
  prices. Limits: `max_steps` (model calls), `max_seconds` (wall clock), `max_tokens_total`
  (input plus output across the run). Defaults are the builder's call to tune; starting points:
  40 steps, 300 seconds, 2M tokens. Hard caps on what the orchestrator can request: 150 steps,
  1,800 seconds.
- Retries: on HTTP 429 and 5xx, retry with backoff (honor `Retry-After` when present), up to a
  small number of attempts, and count retry time against `max_seconds`. Other 4xx errors end
  the run with `status: error` and the provider's message.
- Templates are plain Python strings with `str.format` or f-strings; no jinja2.

System prompt: short. Who it is (a fast coding subagent working for another agent), the
working directory, the tools, the rules (stay in the repo, don't commit, prefer `edit` over
rewriting whole files, run the check or tests when they exist, call `finish` with a short
summary of what changed and anything left undone). Target well under 1,000 tokens including
tool definitions. Keep the system prompt and tool definitions byte-identical across runs so
providers with prompt caching (Cerebras, others) get cache hits.

First user message: the task, the working directory, the check command if any, and the
contents of `files` if given.

### Check loop

If `check_command` is set, then when the subagent calls `finish`, the harness runs the check
(same bash runner, timeout up to 300 seconds) before accepting it.

- Exit code 0: the run ends with `status: done`.
- Non-zero: the check's output tail goes back to the subagent as the `finish` tool result
  ("the check failed; fix it and call finish again"), and the loop continues. After 3 failed
  checks, or when a limit is hit, the run ends with `status: check_failed` and the last check
  output.

The check is run by the harness, so a pass is a fact, not the model's claim. Without a check
command, `finish` ends the run with `status: done`.

### Snapshots, diffs, and undo

The harness, not the model, reports what changed.

- `workdir` must be inside a git repository. If not, `fast_agent` returns an error telling the
  orchestrator to run `git init` first. (My call: git gives exact, honest diffs and undo for
  free. Building in an empty directory just needs `git init`.)
- Before the run: write a tree object of the full working tree, including untracked files that
  aren't ignored, without touching the user's index, HEAD, or stash. One way: copy the index to
  a temp file, then `GIT_INDEX_FILE=<temp> git add -A && GIT_INDEX_FILE=<temp> git write-tree`.
- After the run: the same, giving the after-tree.
- `changed_files` and `diff` come from `git diff <before> <after>` (with `--stat` or
  `--numstat` for the counts).
- `fast_agent_undo`: apply the reverse diff (`git diff --binary <after> <before>`) to the
  working tree with `git apply`, after a `git apply --check`. If the check fails because the
  files changed since, refuse and report which files conflict. Undo never touches the index,
  HEAD, or other files.
- Ignored files (for example `node_modules`, build output) are not tracked by the snapshot.
  That is fine; say so in the result notes when a run touched only ignored paths.

### Providers and keys

Profiles live in a JSON file shipped with the plugin (`providers.default.json`) and can be
overridden or extended by a user file at `~/.config/dllm-subagents/config.json`, which also sets
`default_provider` and default limits. A profile has:

- `base_url`
- `api_key_env`: name of the env var holding the key
- `model`
- `max_tokens_field`: `max_completion_tokens` or `max_tokens`, and `max_output_tokens` value
- `extra_body` (optional): provider-specific fields merged into each request, for example
  `{"reasoning_effort": "low"}` for Mercury, or OpenRouter `provider` routing preferences
- `temperature` (optional)

Shipped profiles: `mercury` (`mercury-3`), `cerebras` (`gpt-oss-120b`), `groq`, `openrouter`.
The builder verifies each profile's current model ID and parameters against the provider's live
docs or `/v1/models` before shipping, and records the date checked in the README. Default
provider: `mercury` until it shuts down. The README tells users to switch after 2026-12-28.

Key lookup order: the env var named by `api_key_env`, then a `KEY=value` line in
`~/.config/dllm-subagents/keys.env` (the README tells users to `chmod 600` it). The keys file
exists because Codex doesn't forward the user's whole environment to MCP servers. If no key is
found, `fast_agent` fails fast with a message naming the env var and the keys file. Keys never
appear in results, logs, or trajectories.

### The skill: `delegate-fast`

A `SKILL.md` that loads in both hosts. It tells the orchestrator:

- **When to delegate:** well-scoped build or edit work with a clear finish line. Examples:
  write a module from a clear spec, add endpoints that follow an existing pattern, apply the
  same change across many files, write tests for existing code, make a failing test pass, fix
  lint or type errors. Bias toward jobs where the orchestrator would otherwise type a lot of
  code.
- **When not to:** design decisions, ambiguous requirements, security-sensitive code, subtle
  debugging, or anything the orchestrator can do in one or two small edits.
- **How to brief:** goal, relevant files, constraints and patterns to follow, what done means,
  and a `check_command` whenever one exists. Always pass the absolute `workdir`.
- **After the run:** read the diff, run or trust the harness check, fix small issues itself or
  send a follow-up task, and call `fast_agent_undo` if the run went wrong. Keep calling
  `fast_agent_wait` while the status is `running`.
- **While a run is active:** don't edit files in that `workdir` (the run's diff would include
  them, and undo would revert them). Work elsewhere or wait. To stop a run, call
  `fast_agent_cancel`.
- **Parallel work:** only one run per directory. For parallel runs, use separate git worktrees.

The user may want to write or adjust the delegation rules themselves, since they encode how
they like to work.

### Plugin wiring

- `.mcp.json` starts the server with `python3` and the server script path under
  `${CLAUDE_PLUGIN_ROOT}`.
- Claude Code: the plugin's `.mcp.json` entry sets no `timeout`, since the default is fine. The
  50-second wait cap works the same in both hosts.
- Codex: **verify before building anything else** (see Risks) that Codex CLI 0.164 installs the
  plugin from the repo, loads its skill, starts its MCP server, and resolves the script path. If
  it doesn't accept the Claude-format files, add the Codex-specific files it needs
  (`.codex-plugin/plugin.json`, `.agents/plugins/marketplace.json`, or the portable root
  `plugin.json` plus `mcp.json` with `"type": "stdio"`) alongside, sharing the same `skills/`
  and `server/`.
- The README documents install for both hosts, keys, providers, limits, and the guardrails
  caveat.

### Logs

Save each run's trajectory as JSON (messages, tool calls, results, stats, exit status) under
`$CLAUDE_PLUGIN_DATA` when set, otherwise `~/.cache/dllm-subagents/runs/`. Return the path in the
result. Never write keys.

## Testing

- **Unit tests, no network:** a scripted fake model (returns canned tool calls) drives the real
  loop against throwaway git repos in a temp dir. Cover: edit semantics and error messages
  (not found, not unique, overlap, CRLF, BOM), path escape attempts, guardrail blocks including
  compound commands, bash timeout kills the process group, each limit, format-error cap, check
  loop pass and fail paths, snapshot diff including new and deleted untracked files, undo and
  undo-refusal on conflict, undo from a persisted record after a simulated restart, cancel
  (explicit and via host cancellation) killing a running command, the 50-second wait handing
  back `running` (use a short cap in tests), one-run-per-directory. Python's `unittest`,
  stdlib only.
- **MCP protocol test:** spawn the server as a subprocess and exchange `initialize`,
  `tools/list`, `tools/call` over stdio.
- **Live smoke test (manual, needs keys):** a script that runs one small real task per
  configured provider and prints status, steps, seconds, and tokens. Skips providers without a
  key.
- **Speed check (informal, part of v1 done; my call):** pick three representative jobs in a real
  repo (for example: a new module from a spec, the same change across several files, making a
  failing test pass). Time each one end to end twice: the orchestrator doing it alone, and the
  orchestrator briefing, waiting, and reviewing a delegated run. Record the times in the README.
  If delegation loses on a kind of job, narrow the skill's "when to delegate" rules to exclude
  it. If it loses on all three, stop and revisit the approach (tuning first, then approach 2)
  before polishing. This is a stopwatch check, not an automated test.
- **Host install test (manual):** install from a local path in Claude Code (`claude plugin
  validate`, then install via the marketplace) and in Codex CLI; confirm the skill is listed and
  a `fast_agent` call works in each.

## Build order and keys

The user will have keys for all four providers (Inception, Cerebras pay-as-you-go, Groq,
OpenRouter) from the start. The long-term provider after Mercury's shutdown is undecided.

So the build goes risk-first:

1. Codex plugin-loading check with a throwaway plugin (Risk 1).
2. A thin live spike: the loop, the five subagent tools, and one real task, run against each
   of the four providers. This tests Risk 2 before anything else is built on it. Guardrails,
   snapshots, undo, cancel, and the MCP layer come after.
3. The rest of the design, with unit tests against the scripted fake model.
4. The speed check, run on at least two providers. Its results, plus the per-provider smoke
   test numbers, are how the user will choose the long-term default. Tune limits and the
   prompt so they work on every shipped profile, not just Mercury.

Keys go in env vars or `~/.config/dllm-subagents/keys.env`, never in chat or the repo.

## Risks and unknowns

1. **Codex plugin loading (highest).** Not verified that Codex CLI loads a Claude-format
   plugin's `.mcp.json` or substitutes `${CLAUDE_PLUGIN_ROOT}` in it. Check first with a
   throwaway plugin; adjust the wiring before writing the harness.
2. **Fast models' tool-calling quality.** None of the candidate models was tested with this loop.
   Mercury is roughly Haiku-class by earlier research. The live smoke test should run early,
   once a key is available, to catch format problems (malformed arguments, no tool calls).
3. **Mercury shutdown on 2026-12-28.** Covered by swappable profiles; the default provider has
   to change then.
4. **Rate limits.** Cerebras free trial is too tight for an agent loop. Retries with backoff
   handle bursts; the README should say which tiers work.
5. **Bash guardrails are bypassable.** Accepted by the user; documented in the README.

## Left to the builder

- Module split, internal names, and exact result text format.
- Default limits, output truncation sizes, and the wait cap (under 60 seconds).
- Fuzzy matching in `edit`, and whether to add `ls` or `grep` tools (only if the live smoke test
  shows the model struggling with bash for these).
- The system prompt wording, within the token target.
- How the guardrail matcher parses compound commands, as long as tests cover the cases above.
