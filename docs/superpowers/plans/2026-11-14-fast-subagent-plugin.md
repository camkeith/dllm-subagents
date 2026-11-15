# dllm-subagents Fast Subagent Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Claude Code and Codex plugin whose MCP server runs a small agent on any fast OpenAI-compatible model to build or edit code in the user's repo, then reports a git-computed diff, a harness-run check result, and supports cancel and undo.

**Architecture:** A stdlib-only Python MCP stdio server (`fast_agent_server.py`) exposes `fast_agent`, `fast_agent_wait`, `fast_agent_cancel`, `fast_agent_undo`. Each run executes in a background thread: an agent loop ported from mini-swe-agent calls the provider over HTTP with five tools (read, write, edit, bash, finish), guarded bash, and an optional check loop. Git tree snapshots taken through a temporary index give the before/after diff and undo. A skill teaches the orchestrator when and how to delegate.

**Tech Stack:** Python 3.9+ standard library only (`urllib`, `subprocess`, `threading`, `json`, `fcntl`, `shlex`, `unittest`), git, Claude Code plugin format, Codex plugin format.

**Spec:** `docs/superpowers/specs/2026-11-14-fast-subagent-plugin-design.md` (read it first; this plan argues from it).

## Global Constraints

- Standard library only. No third-party imports anywhere under `plugins/` or `tests/`.
- Must run on Python 3.9: every module starts with `from __future__ import annotations`; no `match`; no `X | Y` outside annotations; no `str.removeprefix`-era assumptions beyond 3.9.
- Server launched as `python3 ${CLAUDE_PLUGIN_ROOT}/server/fast_agent_server.py`.
- Tool waits return within `wait_seconds`, default `50`, always `< 60` (Codex `tool_timeout_sec` default is 60).
- Limit defaults: `max_steps=40`, `max_seconds=300`, `max_tokens_total=2_000_000`. Hard caps on orchestrator overrides: `150` steps, `1800` seconds.
- bash: default timeout `60` s, cap `300` s; output cap `10_000` chars (head and tail kept). Check command timeout `300` s. Max failed checks `3`. Max consecutive format errors `3`.
- Result diff cap `20_000` chars; `files` content cap `100_000` bytes total; `read` default limit `2000` lines, long lines cut at `2000` chars.
- Read-only git allowlist, exactly: `status`, `diff`, `log`, `show`, `ls-files`, `grep`, `blame`, `rev-parse`. Every other git subcommand is blocked.
- Config dir: `$DLLM_SUBAGENTS_CONFIG_DIR` or `~/.config/dllm-subagents` (holds `config.json`, `keys.env`).
- Runs dir: `$DLLM_SUBAGENTS_RUNS_DIR`, else `$CLAUDE_PLUGIN_DATA/runs`, else `$PLUGIN_DATA/runs`, else `~/.cache/dllm-subagents/runs`.
- API keys never appear in results, logs, trajectories, or exceptions.
- One active run per git repository (keyed by repo root), enforced in-process and across processes.
- User-facing text (README, skill, tool descriptions, result text) uses no em dashes.
- Test command (run from repo root): `PYTHONPATH=plugins/dllm-subagents/server python3 -m unittest discover -s tests -v`
- Commits: Conventional Commits, lowercase imperative, under 72 chars, ending with the session's attribution trailers. Commit only with the user's go-ahead for this plan.

## Review Focus

1. **workdir is a git worktree or a subdirectory of the repo** (`.git` is a file; the user works in worktrees daily). Expect snapshots, diffs, undo, and the repo lock to work on the repo root found via `git rev-parse`. Test added to Task 9 and Task 10.
2. **Empty repo with no commits** (`git init` then build from scratch). Expect snapshot, diff, and undo to work with no HEAD. Test added to Task 9.
3. **Provider returns malformed tool calls** (arguments invalid JSON, arguments as an object instead of a string, missing `id`, unknown tool name). Expect a format-error message back to the model, never a crash. Tests added to Task 2 and Task 6.
4. **Subagent starts a background or long-lived process** (`npm run dev &`, `sleep 999 &`). Expect bash to return by its timeout and leave no process behind. Test added to Task 4.
5. **Binary or non-UTF-8 files** read or edited by the subagent. Expect a clear tool error (`read` reports a binary file and its size; `edit` refuses non-UTF-8), not a crash. Tests added to Task 5.

---

## File Structure

```text
.claude-plugin/marketplace.json            Claude Code marketplace (repo root)
.agents/plugins/marketplace.json           Codex marketplace (Task 1 decides if needed)
plugins/dllm-subagents/
  .claude-plugin/plugin.json               Claude manifest
  .codex-plugin/plugin.json                Codex manifest (Task 1 decides if needed)
  .mcp.json                                launches the server
  skills/delegate-fast/SKILL.md            delegation skill (Task 12)
  server/
    fast_agent_server.py                   entry point: settings, RunManager, signals, serve
    providers.default.json                 shipped provider profiles
    dllm/__init__.py                       VERSION = "0.1.0"
    dllm/mcp.py                            JSON-RPC stdio protocol only
    dllm/server_tools.py                   orchestrator tool schemas and handlers
    dllm/config.py                         profiles, limits, settings, key lookup
    dllm/llm.py                            chat-completions client with retry
    dllm/edit.py                           exact-match multi-edit
    dllm/shell.py                          command runner: timeout, cancel, group kill
    dllm/guard.py                          bash guardrails
    dllm/tools.py                          subagent tool schemas and execution
    dllm/prompts.py                        system prompt, task message, file loading
    dllm/agent.py                          agent loop and check loop
    dllm/snapshot.py                       git tree snapshots, diff, undo
    dllm/runs.py                           run registry, locks, persistence, rendering
scripts/smoke.py                           live per-provider smoke test
tests/helpers.py                           temp git repos, FakeClient, scripted completions
tests/test_*.py                            one file per module
README.md, LICENSE (MIT), THIRD_PARTY_NOTICES
```

---

### Task 1: MCP protocol layer, plugin wiring, and host loading check

Highest risk first (spec Risk 1): prove both hosts load the plugin and see its tools before building the harness.

**Files:**
- Create: `plugins/dllm-subagents/server/dllm/__init__.py`, `dllm/mcp.py`, `dllm/server_tools.py`, `server/fast_agent_server.py`
- Create: `.claude-plugin/marketplace.json`, `plugins/dllm-subagents/.claude-plugin/plugin.json`, `plugins/dllm-subagents/.mcp.json`
- Test: `tests/test_mcp.py`, `tests/helpers.py`

**Interfaces:**
- Produces:
  - `dllm.VERSION: str = "0.1.0"`
  - `dllm.mcp.ToolReply(text: str, is_error: bool = False)` (dataclass)
  - `dllm.mcp.ToolSpec(name: str, description: str, input_schema: dict, handler: Callable[[dict, threading.Event], ToolReply])` (dataclass). The event is set when the host cancels that call.
  - `dllm.mcp.McpServer(name: str, version: str, tools: list[ToolSpec])` with `serve(instream: TextIO, outstream: TextIO) -> None`, returning on EOF.
  - `dllm.server_tools.TOOL_SCHEMAS: dict[str, dict]` keyed by the four tool names, and `build_tool_specs(manager, wait_seconds: float = 50.0) -> list[ToolSpec]`. In this task both arguments are unused (the entry point passes `None`) and every handler returns `ToolReply("not implemented yet", is_error=True)`. Task 11 replaces the handlers.

- [ ] **Step 1: Write failing tests in `tests/test_mcp.py`**

Drive `McpServer` in a thread over `os.pipe()` pairs with test-only tools: `echo` (returns its `text` arg), `boom` (raises `RuntimeError("kaboom")`), `slow` (waits on its cancel event up to 5 s, records whether it was set).

```python
def test_initialize_echoes_protocol_version(self):
    r = self.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
    self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
    self.assertIn("tools", r["result"]["capabilities"])
    self.assertEqual(r["result"]["serverInfo"]["name"], "fast-agent")
def test_tools_list(self):            # names == ["echo", "boom", "slow"], each has inputSchema
def test_tools_call_text_content(self):  # result == {"content": [{"type": "text", "text": "hi"}], "isError": False}
def test_handler_exception_is_error(self):  # isError True, "kaboom" in text
def test_unknown_tool_is_invalid_params(self):  # error.code == -32602
def test_unknown_method(self):        # error.code == -32601
def test_parse_error(self):           # line "not json" -> error.code == -32700, server keeps running
def test_ping(self):                  # result == {}
def test_cancel_sets_event_and_suppresses_response(self):
    # send tools/call slow (id 7), then {"method": "notifications/cancelled", "params": {"requestId": 7}}
    # assert slow saw the event within 2 s, no message with id 7 arrives within 0.5 s, ping still answers
def test_eof_returns(self):           # closing the input pipe makes serve() return within 1 s
def test_entry_point_lists_four_tools(self):
    # subprocess: python3 plugins/dllm-subagents/server/fast_agent_server.py; initialize, tools/list
    # names == {"fast_agent", "fast_agent_wait", "fast_agent_cancel", "fast_agent_undo"}
```

`tests/helpers.py` gets `SERVER_DIR` (absolute path of `plugins/dllm-subagents/server`) and a `JsonRpcPipe` helper (send dict, read next message with timeout).

- [ ] **Step 2: Run tests, verify they fail**

Run: the test command. Expected: errors importing `dllm.mcp`.

- [ ] **Step 3: Implement `dllm/mcp.py`**

Newline-delimited JSON-RPC 2.0 on the given streams; one writer lock; each `tools/call` runs its handler in a worker thread so the read loop keeps receiving `notifications/cancelled`. `initialize` echoes the client's `protocolVersion` when it is a `YYYY-MM-DD` string, else `"2025-06-18"`; capabilities `{"tools": {"listChanged": False}}`; serverInfo `{"name": name, "version": version}`. Notifications (no `id`) never get responses. A cancelled call's response is dropped. Logging goes to stderr only.

- [ ] **Step 4: Implement `dllm/server_tools.py` and `fast_agent_server.py`**

Schemas (descriptions short, no em dashes):
- `fast_agent`: `task` (string), `workdir` (string, "Absolute path of the git repo to work in. Always pass it."), `files` (array of string), `check_command` (string), `provider` (string), `model` (string), `max_steps` (integer 1..150), `max_seconds` (integer 10..1800); required `["task"]`.
- `fast_agent_wait`, `fast_agent_cancel`, `fast_agent_undo`: `run_id` (string), required.

Entry point: put its own directory on `sys.path`, build specs, `McpServer("fast-agent", VERSION, specs).serve(sys.stdin, sys.stdout)`.

- [ ] **Step 5: Run tests, verify they pass**

- [ ] **Step 6: Write the Claude wiring**

`.claude-plugin/marketplace.json`: `{"name": "dllm-subagents", "owner": {"name": "camkeith"}, "plugins": [{"name": "dllm-subagents", "source": "./plugins/dllm-subagents", "description": "Hand build and edit jobs to a fast-model subagent"}]}`.
`plugin.json`: `name` `dllm-subagents`, `version` `0.1.0`, `description`, `author` `{"name": "camkeith"}`, `license` `MIT`, `repository` `https://github.com/camkeith/dllm-subagents`.
`.mcp.json`: `{"mcpServers": {"fast-agent": {"command": "python3", "args": ["${CLAUDE_PLUGIN_ROOT}/server/fast_agent_server.py"]}}}`.

Run: `claude plugin validate .` and `claude plugin validate plugins/dllm-subagents`
Expected: both end with `Validation passed`.

- [ ] **Step 7: Claude Code host check**

Run: `claude --plugin-dir plugins/dllm-subagents -p "List the tool names provided by the fast-agent MCP server, one per line, nothing else."`
Expected: the four tool names appear (possibly with an `mcp__...` prefix).

- [ ] **Step 8: Codex host check (ask the user first: this adds a marketplace to `~/.codex/config.toml`)**

Use `codex plugin marketplace add "$PWD"`, then `codex plugin add` (see `codex plugin add --help`) to install `dllm-subagents`, then `codex exec "List the tool names provided by the fast-agent MCP server, one per line."`.
Expected: the four names. If Codex does not load the plugin or the server, add Codex files per the spec's "Plugin wiring" section, in this order, retesting after each: (a) `.agents/plugins/marketplace.json` with `{"name": "dllm-subagents", "interface": {"displayName": "dllm-subagents"}, "plugins": [{"name": "dllm-subagents", "source": {"source": "local", "path": "./plugins/dllm-subagents"}, "policy": {"installation": "AVAILABLE", "authentication": "ON_INSTALL"}, "category": "Coding"}]}`; (b) `plugins/dllm-subagents/.codex-plugin/plugin.json` with the same identity fields plus `"skills": "./skills/"` and `"mcpServers": "./.mcp.json"`; (c) if `${CLAUDE_PLUGIN_ROOT}` is not substituted, the portable root `plugin.json` plus `mcp.json` with `"type": "stdio"`. Record what worked (one short paragraph) for the README in Task 12. Offer to remove the marketplace afterwards (`codex plugin marketplace remove dllm-subagents`).

- [ ] **Step 9: Commit**

```bash
git add .claude-plugin plugins tests   # plus .agents if Step 8 created it
git commit -m "feat: add mcp stdio server skeleton and plugin wiring"
```

---

### Task 2: Config and chat-completions client

**Files:**
- Create: `server/dllm/config.py`, `server/dllm/llm.py`, `server/providers.default.json`
- Test: `tests/test_config.py`, `tests/test_llm.py`

**Interfaces:**
- Produces (`dllm.config`):
  - `class ConfigError(Exception)`
  - `@dataclass Profile(name: str, base_url: str, api_key_env: str, model: str, max_tokens_field: str = "max_completion_tokens", max_output_tokens: int = 8192, extra_body: dict = {}, temperature: float | None = None)` (use `field(default_factory=dict)`)
  - `@dataclass Limits(max_steps: int = 40, max_seconds: int = 300, max_tokens_total: int = 2_000_000)`
  - `@dataclass Settings(default_provider: str, profiles: dict[str, Profile], limits: Limits, wait_seconds: float = 50.0)`
  - `HARD_MAX_STEPS = 150`, `HARD_MAX_SECONDS = 1800`
  - `config_dir() -> Path`
  - `load_settings(defaults_path: Path | None = None, user_path: Path | None = None) -> Settings`. Defaults to `server/providers.default.json` and `config_dir()/config.json` (missing user file is fine). User `providers` entries merge field by field into shipped ones or add new ones; user `default_provider`, `limits`, `wait_seconds` override. `wait_seconds` is clamped to at most 55.
  - `clamp_limits(base: Limits, max_steps: int | None, max_seconds: int | None) -> Limits` (clamp to hard caps, minimum 1 step and 10 seconds)
  - `resolve_api_key(profile: Profile, environ: Mapping[str, str] | None = None, keys_file: Path | None = None) -> str`
- Produces (`dllm.llm`):
  - `class LLMError(Exception)` with attribute `status: int | None`
  - `@dataclass ToolCall(id: str, name: str, arguments: str)`
  - `@dataclass Completion(content: str | None, tool_calls: list[ToolCall], input_tokens: int, output_tokens: int, finish_reason: str | None, message: dict)`. `message` is the sanitized assistant message to append to history: `{"role": "assistant", "content": content, "tool_calls": [{"id", "type": "function", "function": {"name", "arguments"}}]}` (omit `tool_calls` when empty).
  - `Transport = Callable[[str, dict, bytes, float], tuple[int, dict, bytes]]` (url, headers, body, timeout to status, headers, body); `urllib_transport` is the default and returns HTTP error statuses instead of raising.
  - `class ChatClient(profile: Profile, api_key: str, *, model: str | None = None, transport: Transport = urllib_transport, max_retries: int = 4, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic, request_timeout: float = 120.0)` with `complete(messages: list[dict], tools: list[dict], deadline: float | None = None) -> Completion`.

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
def test_shipped_profiles_exist(self):  # {"mercury", "cerebras", "groq", "openrouter"} <= set(s.profiles); s.default_provider == "mercury"
def test_user_config_merges_fields(self):  # user {"providers": {"groq": {"model": "x"}, "local": {...}}} -> groq.model == "x", groq.base_url unchanged, "local" added
def test_limits_and_wait_override(self):  # user limits {"max_steps": 10}; wait_seconds 90 -> clamped to 55
def test_clamp_limits_caps(self):  # clamp_limits(Limits(), 500, 99999) == Limits(150, 1800, 2_000_000); clamp_limits(Limits(), 0, 1) -> (1, 10, ...)
def test_key_from_env_first(self):  # environ {"GROQ_API_KEY": "env"} and keys file GROQ_API_KEY=file -> "env"
def test_key_from_keys_file(self):  # keys.env lines: comment, blank, 'GROQ_API_KEY = "file"' -> "file" (strip spaces and quotes)
def test_missing_key_message(self):  # ConfigError mentions "GROQ_API_KEY" and the keys file path, and no other key values
```

`tests/test_llm.py` (fake transport records requests and returns scripted responses):
```python
def test_request_body_shape(self):
    # profile max_tokens_field "max_completion_tokens", max_output_tokens 8192, extra_body {"reasoning_effort": "low"}, temperature 0.5
    # body == {"model": "m", "messages": [...], "tools": [...], "tool_choice": "auto", "max_completion_tokens": 8192, "reasoning_effort": "low", "temperature": 0.5}
    # url == base_url.rstrip("/") + "/chat/completions"; headers["Authorization"] == "Bearer k"
def test_model_override(self):          # ChatClient(..., model="other") -> body["model"] == "other"
def test_parses_tool_calls_and_usage(self):  # 2 tool calls -> names, ids, raw argument strings; usage prompt_tokens/completion_tokens -> input/output tokens
def test_arguments_object_is_serialized(self):  # provider sends arguments as a dict -> ToolCall.arguments == json.dumps(dict)
def test_missing_tool_call_id_gets_generated(self):  # id missing -> non-empty unique id
def test_sanitized_message_drops_extra_fields(self):  # response message has "reasoning": "..." -> Completion.message has only role, content, tool_calls
def test_retries_429_honoring_retry_after(self):  # 429 with Retry-After "2", then 200 -> sleep called with 2, one Completion
def test_retries_5xx_with_backoff(self):  # 503, 503, 200 -> 3 requests, sleeps increasing
def test_gives_up_after_max_retries(self):  # always 500 -> LLMError(status=500)
def test_4xx_raises_with_provider_message(self):  # 400 {"error": {"message": "bad tools"}} -> LLMError, "bad tools" in str, status 400, no retry
def test_deadline_stops_retries(self):  # clock past deadline before a retry -> LLMError mentioning "time limit"
def test_api_key_not_in_errors(self):   # str(LLMError) and repr never contain the key
```

- [ ] **Step 2: Run, verify failures**

- [ ] **Step 3: Implement `config.py` and `providers.default.json`**

Shipped profiles (model IDs verified live in Task 7):
- `mercury`: `https://api.inceptionlabs.ai/v1`, `INCEPTION_API_KEY`, `mercury-3`, `max_completion_tokens`, `extra_body {"reasoning_effort": "low"}`
- `cerebras`: `https://api.cerebras.ai/v1`, `CEREBRAS_API_KEY`, `gpt-oss-120b`, `max_completion_tokens`, `extra_body {"reasoning_effort": "low"}`
- `groq`: `https://api.groq.com/openai/v1`, `GROQ_API_KEY`, `openai/gpt-oss-120b`, `max_completion_tokens`, `extra_body {"reasoning_effort": "low"}`
- `openrouter`: `https://openrouter.ai/api/v1`, `OPENROUTER_API_KEY`, `openai/gpt-oss-120b:nitro`, `max_tokens`

File also carries `"default_provider": "mercury"`, `"limits"`, `"wait_seconds": 50`.

- [ ] **Step 4: Implement `llm.py`**

Retry on 429 and 5xx and on transport exceptions (`URLError`, timeouts): `Retry-After` seconds when present, else `min(2 ** attempt, 20)`; never sleep past `deadline`. Error messages come from the provider's `error.message` when present, truncated to 500 chars.

- [ ] **Step 5: Run, verify pass. Step 6: Commit** `feat: add provider profiles and chat-completions client`

---

### Task 3: Exact-match edit

**Files:** Create `server/dllm/edit.py`; Test `tests/test_edit.py`

**Interfaces:**
- Produces: `class EditError(Exception)`; `apply_edits(content: str, edits: list[dict]) -> str` where each edit is `{"old_text": str, "new_text": str}`.

- [ ] **Step 1: Write failing tests**

```python
def test_single_replacement(self):      # "a\nb\nc\n", [{"old_text": "b", "new_text": "B"}] -> "a\nB\nc\n"
def test_multiple_disjoint_edits_match_original(self):  # edits applied against the original, not incrementally:
    # "x = 1\ny = 2\n", [{"old_text": "x = 1", "new_text": "y = 2"}, {"old_text": "y = 2", "new_text": "z = 3"}] -> "y = 2\nz = 3\n"
def test_not_found(self):               # EditError, "edit 1" and "not found" in message
def test_not_unique(self):              # "a a", old "a" -> EditError containing "matches 2 times"
def test_overlap(self):                 # old "abc" and "bcd" in "abcde" -> EditError containing "overlap"
def test_empty_old_text(self):          # EditError containing "empty"
def test_all_or_nothing(self):          # second edit fails -> EditError, caller's content unchanged (pure function)
def test_crlf_preserved(self):          # "a\r\nb\r\n", old "a\nb" (LF in old_text) -> "A\r\nB\r\n" for new "A\nB"
def test_bom_preserved(self):           # "﻿hello" , old "hello" -> "﻿bye"
def test_edits_list_required(self):     # [] or missing keys -> EditError
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** Strip a leading BOM; if the text contains `"\r\n"`, normalize CRLF to LF for matching and convert the result back to CRLF; normalize CRLF in `old_text`/`new_text` too; restore the BOM. No fuzzy matching in v1. Messages number edits from 1. **Step 4: Run, verify pass. Step 5: Commit** `feat: add exact-match multi-edit`

---

### Task 4: Command runner

**Files:** Create `server/dllm/shell.py`; Test `tests/test_shell.py`

**Interfaces:**
- Produces:
  - `@dataclass CommandResult(output: str, returncode: int, timed_out: bool, cancelled: bool)`
  - `truncate_middle(text: str, max_chars: int) -> str` (keeps head and tail, inserts `\n... [N characters cut] ...\n`)
  - `run_command(command: str, cwd: str, timeout: float, cancel_event: threading.Event | None = None, max_output_chars: int = 10_000) -> CommandResult`
  - `kill_all() -> None` kills every live process group started by `run_command`.

- [ ] **Step 1: Write failing tests**

```python
def test_output_and_returncode(self):   # "echo hi; exit 3" -> output "hi\n", returncode 3
def test_stderr_merged(self):           # "echo err 1>&2" -> "err" in output
def test_timeout_kills_group(self):     # "sleep 30 & sleep 30" timeout 1 -> timed_out True, returns in < 3 s, `sleep 30` child gone (check its pid via os.kill(pid, 0) raising)
def test_background_process_does_not_block(self):  # Review Focus 4: "sleep 999 & echo started" timeout 2 -> returns by ~2 s, "started" in output, no sleep 999 left
def test_cancel_event_kills(self):      # event set from a timer after 0.5 s on "sleep 30" -> cancelled True, returns in < 2 s
def test_large_output_truncated(self):  # "python3 -c 'print(\"x\"*50000)'" -> len(output) <= 10_000 + 100 and "characters cut" in output
def test_no_pipe_deadlock(self):        # 200 KB of output with timeout 10 -> completes without timing out
def test_kill_all(self):                # start "sleep 30" in a thread, kill_all() -> that run returns promptly
def test_cwd(self):                     # "pwd" in a temp dir -> that dir (realpath)
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement** by porting mini-swe-agent `environments/local.py::_run` (`Popen(shell=True, start_new_session=True, stdout=PIPE, stderr=STDOUT, text=True, errors="replace")`), then looping `communicate(timeout=0.2)` and checking `cancel_event` and the deadline each pass; on either, `os.killpg(pid, SIGKILL)` and collect remaining output. Track live pids in a module-level set under a lock for `kill_all`. Keep the mini-swe-agent copyright line in a header comment. **Step 4: Run, verify pass. Step 5: Commit** `feat: add command runner with timeout and cancel`

---

### Task 5: Subagent tools and prompts

**Files:** Create `server/dllm/tools.py`, `server/dllm/prompts.py`; Test `tests/test_tools.py`, `tests/test_prompts.py`

**Interfaces:**
- Consumes: `apply_edits`, `EditError` (Task 3); `run_command`, `CommandResult` (Task 4).
- Produces (`dllm.tools`):
  - `TOOL_SCHEMAS: list[dict]`: OpenAI function tools in this order: `read(path, offset?, limit?)`, `write(path, content)`, `edit(path, edits: [{old_text, new_text}])`, `bash(command, timeout?)`, `finish(summary)`.
  - `class ToolError(Exception)`
  - `@dataclass ToolContext(workdir: Path, cancel_event: threading.Event, guard: Callable[[str, str], str | None] | None = None, bash_default_timeout: int = 60, bash_max_timeout: int = 300)`
  - `resolve_in_workdir(workdir: Path, path: str) -> Path` (relative to workdir; after `resolve()`, must be inside `workdir.resolve()`)
  - `run_tool(name: str, args: dict, ctx: ToolContext) -> str` for `read`, `write`, `edit`, `bash`; raises `ToolError` for anything the model should fix. `finish` is not handled here.
- Produces (`dllm.prompts`):
  - `SYSTEM_PROMPT: str` (static, no placeholders)
  - `load_files(workdir: Path, paths: list[str], max_total_bytes: int = 100_000) -> tuple[list[tuple[str, str]], list[str]]` (included `(relpath, text)` pairs, notes for skipped or truncated files)
  - `build_task_message(task: str, workdir: str, check_command: str | None, files: list[tuple[str, str]], notes: list[str]) -> str`

- [ ] **Step 1: Write failing tests**

`tests/test_tools.py` (temp dir per test):
```python
def test_read_numbered_lines(self):     # "a\nb\n" -> "     1\ta\n     2\tb"
def test_read_offset_limit_reports_range(self):  # 10 lines, offset 3 limit 2 -> lines 3-4 and "[lines 3-4 of 10]"
def test_read_binary_file(self):        # Review Focus 5: bytes with NUL -> ToolError containing "binary" and the byte size
def test_read_missing_file(self):       # ToolError containing "not found"
def test_write_creates_parents(self):   # write("a/b/c.txt", "x") -> file exists, return mentions "a/b/c.txt"
def test_edit_applies(self):            # uses apply_edits; returns "1 replacement"
def test_edit_non_utf8_refused(self):   # Review Focus 5: latin-1 bytes b"caf\xe9" -> ToolError containing "UTF-8", file unchanged
def test_edit_error_passthrough(self):  # EditError message surfaces in ToolError
def test_path_escape_rejected(self):    # "../x", "/etc/hosts" -> ToolError containing "outside the working directory"
def test_symlink_escape_rejected(self): # link in workdir -> outside file; write via link -> ToolError, outside file unchanged
def test_bash_runs_in_workdir(self):    # "pwd" -> output contains workdir realpath, "exit code: 0"
def test_bash_timeout_capped(self):     # timeout 9999 requested -> run_command called with 300 (patch run_command)
def test_bash_guard_blocks(self):       # ctx.guard returns "no pushing" -> ToolError "blocked: no pushing", command not run
def test_unknown_tool(self):            # ToolError containing "unknown tool"
def test_bad_args(self):                # read without path -> ToolError naming "path"
```
`tests/test_prompts.py`:
```python
def test_prompt_and_tools_are_small(self):  # len(SYSTEM_PROMPT) + len(json.dumps(TOOL_SCHEMAS)) < 4000
def test_prompt_is_static(self):        # "{" not in SYSTEM_PROMPT  (no format placeholders)
def test_task_message_contents(self):   # contains task, workdir, check command line, "--- src/a.py ---" and file text
def test_task_message_without_check(self):  # no "Check command" line
def test_load_files_cap_and_notes(self):  # two 80 KB files -> first included, second noted as skipped for size
def test_load_files_missing_and_outside(self):  # missing and "../x" -> noted, not raised
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** The system prompt covers, in under 4000 characters with the tool schemas: role (a fast coding subagent working for another agent), stay inside the working directory, never commit or change git state, prefer `edit` for changes to existing files, read before editing, run the check command or relevant tests before finishing when they exist, keep tool output small (use `head`, `grep -n`), call `finish` with a short summary of what changed and anything left undone. bash output format: `exit code: N\n<output>` plus `[timed out after Ns]` or `[cancelled]` when set. **Step 4: Run, verify pass. Step 5: Commit** `feat: add subagent tools and prompts`

---

### Task 6: Agent loop and check loop

**Files:** Create `server/dllm/agent.py`, `LICENSE`, `THIRD_PARTY_NOTICES`; Modify `tests/helpers.py`; Test `tests/test_agent.py`

**Interfaces:**
- Consumes: `ChatClient.complete`, `Completion`, `ToolCall`, `LLMError` (Task 2); `Limits` (Task 2); `TOOL_SCHEMAS`, `ToolContext`, `run_tool`, `ToolError` (Task 5); `SYSTEM_PROMPT` (Task 5); `run_command` (Task 4).
- Produces:
  - `@dataclass CheckResult(command: str, exit_code: int, output_tail: str, attempts: int)`
  - `@dataclass AgentResult(status: str, summary: str, limit: str | None, error: str | None, check: CheckResult | None, steps: int, seconds: float, input_tokens: int, output_tokens: int)`. `status` is one of `"done"`, `"check_failed"`, `"limit_reached"`, `"cancelled"`, `"error"`; `limit` is one of `"max_steps"`, `"max_seconds"`, `"max_tokens_total"`.
  - `class Agent(client, ctx: ToolContext, limits: Limits, *, check_command: str | None = None, max_check_failures: int = 3, max_consecutive_format_errors: int = 3, check_timeout: int = 300, on_step: Callable[[int, str], None] | None = None, trajectory_path: Path | None = None, clock: Callable[[], float] = time.monotonic)` with `run(task_message: str) -> AgentResult` and attribute `messages: list[dict]`.
  - `tests.helpers.FakeClient(script: list[Completion | Callable[[list[dict]], Completion]])` with `.calls` (copies of messages per call). A callable item is invoked with the messages, so a test can block on an `Event` or raise. Running past the script raises `AssertionError`.
  - `tests.helpers.tool_completion(*calls: tuple[str, dict], content=None, tokens=(100, 10)) -> Completion`.

- [ ] **Step 1: Write failing tests** (temp git-free dir as workdir; `FakeClient` scripts the model)

```python
def test_finish_without_check(self):    # [finish("did it")] -> status "done", summary "did it", steps 1
def test_tools_then_finish(self):       # write a.txt, then finish -> a.txt exists; second call's messages end with a "tool" message for the write
def test_tool_message_per_call(self):   # one completion with 2 tool calls -> 2 tool messages with matching tool_call_id, in order
def test_tool_error_reported_to_model(self):  # read missing file -> tool message starts "Error:" and run continues
def test_no_tool_calls_is_format_error(self):  # text-only reply then finish -> a user message asking for a tool call was sent; status "done"
def test_repeated_format_errors(self):  # 3 text-only replies -> status "error", error mentions "format"
def test_malformed_arguments(self):     # Review Focus 3: arguments "{not json" -> tool message "Error: invalid JSON arguments", counted as a format error, run continues
def test_unknown_tool_name(self):       # Review Focus 3: name "delete_everything" -> tool message containing "unknown tool"
def test_step_limit(self):              # max_steps 2, model never finishes -> status "limit_reached", limit "max_steps", steps 2
def test_time_limit(self):              # fake clock jumps past max_seconds -> limit "max_seconds"
def test_token_limit(self):             # tokens (1_500_000, 600_000) on first call -> limit "max_tokens_total"
def test_llm_error_ends_run(self):      # client raises LLMError("bad tools", 400) -> status "error", "bad tools" in error
def test_cancel(self):                  # ctx.cancel_event set before the second step -> status "cancelled"
def test_check_pass(self):              # check "test -f a.txt"; write then finish -> status "done", check.exit_code 0, attempts 1
def test_check_fail_then_fix(self):     # finish before writing -> finish tool message contains "check failed" and output tail; then write, finish -> "done", attempts 2
def test_check_fail_three_times(self):  # finish x3 with failing check -> status "check_failed", check.attempts 3
def test_calls_after_finish_not_run(self):  # [finish, write b.txt] in one completion -> b.txt absent
def test_trajectory_written_each_step(self):  # trajectory JSON exists after run, has "messages" and "stats"; no API key string inside
def test_on_step_progress(self):        # on_step called with (1, "write: a.txt") style labels
def test_first_messages(self):          # client.calls[0][0] == {"role": "system", "content": SYSTEM_PROMPT}; [1] is the user task message
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement** by porting `mini-swe-agent/src/minisweagent/agents/default.py`'s structure (run loop until exit, `step = query + execute`, exceptions for control flow, limits checked before each model call, format-error counter reset on a clean step, save after every step), with the changes listed in the spec's "The agent loop" section. Pass `deadline = start + max_seconds` to `client.complete`. Each tool call is checked against `ctx.cancel_event` before it runs. Add `LICENSE` (MIT, holder `camkeith`, 2026) and `THIRD_PARTY_NOTICES` with the MIT notices for mini-swe-agent (Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez) and Pi (copy the holder line from `earendil-works/pi`'s LICENSE). **Step 4: Run, verify pass. Step 5: Commit** `feat: add agent loop with limits and check loop`

---

### Task 7: Live spike against real providers (decision gate)

Spec Risk 2. Runs the real loop on real models before anything else is built on it.

**Files:** Create `scripts/smoke.py`; Create `docs/superpowers/notes/2026-11-14-live-spike.md`; Modify `server/providers.default.json` (verified model IDs and params)

**Interfaces:**
- Consumes: `load_settings`, `resolve_api_key`, `ChatClient`, `Agent`, `ToolContext`, `build_task_message`.
- Produces: `python3 scripts/smoke.py [--provider NAME ...] [--models] [--keep]`.

- [ ] **Step 1: Ask the user to put their keys in `~/.config/dllm-subagents/keys.env` (`chmod 600`)**, one `NAME=value` per line for `INCEPTION_API_KEY`, `CEREBRAS_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`. Never ask for keys in chat.

- [ ] **Step 2: Implement `scripts/smoke.py`**

`--models`: for each provider with a key, `GET {base_url}/models` and print model IDs containing the profile's model family. Default mode: for each provider with a key (skip others with a note), make a temp dir with `git init`, run `Agent` with the task "Create calc.py with add(a, b) and mul(a, b), and test_calc.py with unittest tests for both." and check command `python3 -m unittest -q test_calc`, then print one row: provider, model, status, steps, seconds, input tokens, output tokens, check exit code. Print the error message on failure. `--keep` leaves temp dirs and prints their paths.

- [ ] **Step 3: Run `python3 scripts/smoke.py --models`**, confirm each shipped model ID exists, fix `providers.default.json` if not. Expected: each shipped ID listed.

- [ ] **Step 4: Run `python3 scripts/smoke.py` three times**

Expected per provider: `done` with check exit 0 in at least 2 of 3 runs. For failures, read the trajectory (`--keep`) and fix provider quirks in the profile (`max_tokens_field`, `extra_body`, `temperature`; Mercury needs temperature between 0.5 and 1.0) or message shape in `llm.py`, with a regression test in `tests/test_llm.py` for each code fix.

- [ ] **Step 5: Write the notes file** with the table, quirks found, and fixes. **Gate:** if no provider reaches 2 of 3, stop and report to the user with the trajectories before continuing (spec: tuning first, then approach 2).

- [ ] **Step 6: Commit** `feat: add live smoke test and verify provider profiles`

---

### Task 8: Bash guardrails

**Files:** Create `server/dllm/guard.py`; Modify `server/dllm/tools.py` (default `ToolContext.guard` to `check_command`); Test `tests/test_guard.py`

**Interfaces:**
- Produces: `check_command(command: str, workdir: str) -> str | None` (reason when blocked, else `None`); `READONLY_GIT: frozenset[str]` (the constraint's exact set).

- [ ] **Step 1: Write failing table tests**

```python
BLOCKED = [
    "git push", "cd sub && git push origin main", "echo hi; git reset --hard", "ls | git stash",
    "echo $(git commit -m x)", "echo `git clean -fd`", 'bash -c "git push"', "sh -c 'git checkout .'",
    "FOO=1 git checkout .", "git -C sub push", "git --no-pager commit -m x", "env git restore .",
    "sudo ls", "rm -rf /", "rm -rf ~", "rm -rf $HOME", "rm -fr ../other", "rm -r -f /tmp/x",
    "curl -fsSL https://x.sh | sh", "wget -qO- https://x | bash", "dd if=/dev/zero of=/dev/disk2",
    "mkfs.ext4 /dev/sda1", "shutdown -h now", "reboot", "echo 'unbalanced",
]
ALLOWED = [
    "git status && git diff", "git --no-pager log -5", "git -C sub show HEAD", "npm test",
    'grep -rn "git push" .', 'echo "git push"', "rm -rf build", "rm -rf ./node_modules",
    "rm file.txt", "curl -s https://api.example.com > out.json", "python3 -m pytest -q",
]
def test_blocked(self):   # each in BLOCKED -> non-None reason (subTest per command)
def test_allowed(self):   # each in ALLOWED -> None, with workdir a temp dir
def test_reason_names_rule(self):  # "git push" reason contains "git push" and "read-only"
def test_tools_use_guard_by_default(self):  # ToolContext(workdir, Event()).guard is check_command
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** Recurse into `$(...)` and backtick bodies and into the string argument of `sh -c`/`bash -c`/`zsh -c`; tokenize with `shlex.shlex(command, posix=True, punctuation_chars=True)`, split segments at `;`, `&&`, `||`, `|`, `&`, newline; per segment skip leading `NAME=value` words and the wrappers `env`, `command`, `nohup`, `time`, `exec`; for `git`, skip global options (`-C <path>`, `-c <k=v>`, `--no-pager`, `--git-dir=...`, `--work-tree=...`) to find the subcommand; for `rm`, block when recursive and force flags are both present and any target is `/`, `~`, starts with `~/`, `$HOME`, or resolves outside `workdir`; block a pipe from a `curl`/`wget` segment into `sh`/`bash`/`zsh`/`dash`; a `shlex` error blocks with reason "could not parse the command". **Step 4: Run, verify pass. Step 5: Commit** `feat: add bash guardrails`

---

### Task 9: Git snapshots, diff, and undo

**Files:** Create `server/dllm/snapshot.py`; Modify `tests/helpers.py` (`make_repo(tmp, commit=True) -> Path`); Test `tests/test_snapshot.py`

**Interfaces:**
- Produces:
  - `class SnapshotError(Exception)`
  - `repo_root(workdir: Path) -> Path` (raises `SnapshotError` containing "git init" when not in a repo)
  - `snapshot_tree(repo: Path) -> str` (tree SHA of the full working tree incl. untracked non-ignored files; never modifies the real index, HEAD, refs, or stash)
  - `@dataclass FileChange(path: str, change: str, added: int, removed: int)` with `change` in `"added" | "modified" | "deleted"`; binary files report `-1` counts
  - `diff_trees(repo: Path, before: str, after: str, max_chars: int = 20_000) -> tuple[list[FileChange], str, bool]` (changes, unified diff text, truncated flag)
  - `undo(repo: Path, before: str, after: str) -> None` (raises `SnapshotError` listing conflicting files)
  - `git_path(repo: Path, name: str) -> Path` (`git rev-parse --git-path <name>`, used for the index copy and the repo lock)

- [ ] **Step 1: Write failing tests**

```python
def test_not_a_repo(self):              # SnapshotError, "git init" in message
def test_snapshot_does_not_touch_index_or_stash(self):  # status --porcelain, `git stash list`, index file bytes identical before/after
def test_diff_add_modify_delete_untracked(self):  # modify tracked, delete tracked, add untracked, delete a previously untracked file -> FileChange set matches, counts right
def test_ignored_files_excluded(self):  # .gitignore node_modules/ ; create node_modules/x -> no change reported
def test_diff_truncated(self):          # 50 KB change -> truncated True, len(text) <= 20_000 + 200
def test_undo_restores_exactly(self):   # after undo, snapshot_tree == before tree; new files removed, deleted files back
def test_undo_conflict_refused(self):   # change a file the run changed, then undo -> SnapshotError naming it; file untouched
def test_undo_noop_when_same(self):     # before == after -> returns without error
def test_empty_repo_no_commits(self):   # Review Focus 2: make_repo(commit=False); snapshot, add file, diff, undo all work
def test_worktree_and_subdir(self):     # Review Focus 1: `git worktree add` a second tree; repo_root(sub dir of it) is that worktree; snapshot/diff/undo work; main tree's index unchanged
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** Copy the index at `git_path(repo, "index")` to a temp file when it exists, then run `git add -A` and `git write-tree` with `GIT_INDEX_FILE` set to the temp file; delete the temp file in `finally`. Diff with `git diff --no-renames --numstat` plus `--name-status` plus the full unified diff, all `before after`. Undo pipes `git diff --binary --no-renames <after> <before>` into `git apply --check` then `git apply`. Every git call uses `-c core.quotepath=false` and a 60 s timeout, raising `SnapshotError` on failure with git's stderr. **Step 4: Run, verify pass. Step 5: Commit** `feat: add git tree snapshots, diff, and undo`

---

### Task 10: Run manager

**Files:** Create `server/dllm/runs.py`; Test `tests/test_runs.py`

**Interfaces:**
- Consumes: Settings, `clamp_limits`, `resolve_api_key`, `ChatClient` (Task 2); `Agent`, `AgentResult` (Task 6); `ToolContext` (Task 5); `load_files`, `build_task_message` (Task 5); `repo_root`, `snapshot_tree`, `diff_trees`, `undo`, `git_path` (Task 9); `kill_all` (Task 4).
- Produces:
  - `runs_dir() -> Path`
  - `@dataclass RunRecord(run_id: str, workdir: str, repo: str, before_tree: str, after_tree: str | None, status: str, started_at: float, ended_at: float | None, provider: str, model: str, undone: bool = False)` with `save(path: Path)` and `RunRecord.load(path: Path)`
  - `class RunError(Exception)` (user-facing message)
  - `class RunManager(settings: Settings, *, client_factory: Callable[[Profile, str, str | None], object] | None = None)`:
    - `start(task: str, workdir: str | None = None, files: list[str] | None = None, check_command: str | None = None, provider: str | None = None, model: str | None = None, max_steps: int | None = None, max_seconds: int | None = None) -> str` (run_id; raises `RunError`)
    - `wait(run_id: str, seconds: float, interrupt: threading.Event | None = None) -> dict` (view)
    - `cancel(run_id: str) -> dict` (view after the run stops, waiting up to 5 s)
    - `undo(run_id: str) -> str`
    - `shutdown() -> None`
  - `render_view(view: dict) -> str`
  - View keys: `status`, `run_id`, and when finished `summary`, `limit`, `error`, `check`, `stats`, `changed_files`, `diff`, `diff_truncated`, `log_path`; when running `progress`.

- [ ] **Step 1: Write failing tests** (temp repos via `make_repo`; `client_factory` returns `FakeClient`; `DLLM_SUBAGENTS_RUNS_DIR` set to a temp dir)

```python
def test_start_wait_done(self):         # FakeClient writes a.txt then finishes -> view status "done", changed_files has a.txt "added", diff contains "+", log_path exists
def test_running_view_then_done(self):  # FakeClient blocks on an Event; wait(0.2) -> status "running" with progress; release; wait(5) -> "done"
def test_render_running(self):          # "status: running", run_id, "fast_agent_wait" in text
def test_render_done(self):             # "status: done", "summary:", "A a.txt (+1 -0)", "fast_agent_undo" in text
def test_render_truncated_diff_note(self):  # diff_truncated True -> text contains "git diff <before> <after>" with the real tree SHAs
def test_render_no_changes_note(self):  # done with no changed files -> text notes that files in .gitignore are not tracked
def test_workdir_fallback(self):        # workdir None, CLAUDE_PROJECT_DIR=<repo> in env -> run uses that repo
def test_one_run_per_repo(self):        # second start in same repo while first active -> RunError naming the first run_id
def test_lock_is_per_repo_root(self):   # Review Focus 1: start in repo/sub while a run is active at repo root -> RunError
def test_lock_across_managers(self):    # second RunManager instance, same repo -> RunError mentioning "another session"
def test_lock_released_after_run(self): # after done, a new start succeeds
def test_not_git_repo(self):            # RunError containing "git init"
def test_relative_or_missing_workdir(self):  # RunError
def test_unknown_provider(self):        # RunError listing known providers
def test_missing_key_fails_fast(self):  # RunError naming the env var; no thread started
def test_limits_clamped(self):          # max_steps 999 -> agent built with 150
def test_files_included_in_first_message(self):  # files=["a.py"] -> FakeClient.calls[0][1]["content"] contains a.py text
def test_cancel_stops_and_keeps_changes(self):  # run writes a file then blocks in bash "sleep 30"; cancel -> "cancelled" within 5 s, file still there, diff shown
def test_interrupt_returns_running(self):  # wait with interrupt set -> returns promptly with "running"
def test_undo_reverts(self):            # undo -> files back to before; record.undone True; second undo -> RunError "already undone"
def test_undo_refused_while_active(self):  # RunError mentioning cancel
def test_undo_after_restart(self):      # new RunManager (same runs dir) can undo a finished run from its record
def test_shutdown_cancels_active(self): # shutdown() with a blocked run -> record on disk has status "cancelled" and an after_tree
def test_error_view(self):              # client raises LLMError -> status "error", error text, lock released
def test_records_never_contain_key(self):  # api key string absent from record and trajectory files
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** Run IDs: `time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)`. Lock: in-process dict keyed by repo root, plus `fcntl.flock(LOCK_EX | LOCK_NB)` on `git_path(repo, "dllm-subagents.lock")` held for the run. The worker thread: build `ToolContext`, `Agent`, run, then in `finally` take the after-tree, compute the diff, save the record, release the lock. Records and trajectories go to `runs_dir()/<run_id>.json` and `<run_id>.trajectory.json`. **Step 4: Run, verify pass. Step 5: Commit** `feat: add run manager with locks, persistence, cancel, and undo`

---

### Task 11: Wire real tools into the MCP server

**Files:** Modify `server/dllm/server_tools.py`, `server/fast_agent_server.py`; Test `tests/test_server_e2e.py`

**Interfaces:**
- Consumes: `RunManager`, `render_view`, `RunError` (Task 10); `McpServer`, `ToolSpec`, `ToolReply` (Task 1); `load_settings` (Task 2); `kill_all` (Task 4).
- Produces: `build_tool_specs(manager: RunManager, wait_seconds: float) -> list[ToolSpec]` with real handlers.

- [ ] **Step 1: Write failing end-to-end tests.** Start a local fake chat-completions server (`http.server` in a thread) that replays a scripted list of responses; write a temp `config.json` with a `fake` provider pointing at it and a temp `keys.env`; spawn the real entry point with `DLLM_SUBAGENTS_CONFIG_DIR` and `DLLM_SUBAGENTS_RUNS_DIR` set and `wait_seconds` 2.

```python
def test_fast_agent_round_trip(self):   # tools/call fast_agent (task, workdir=temp repo, provider "fake") -> text has "status: done" and the changed file
def test_wait_path(self):               # fake server delays 4 s -> first reply "status: running"; fast_agent_wait -> "status: done"
def test_host_cancel_stops_run(self):   # notifications/cancelled for a pending fast_agent call -> later fast_agent_wait shows "status: cancelled"
def test_undo_tool(self):               # fast_agent_undo -> file reverted, text confirms
def test_run_error_is_tool_error(self): # workdir not a repo -> isError True, "git init" in text
def test_sigterm_kills_children(self):  # run blocked in bash "sleep 30"; SIGTERM the server -> exits within 5 s, no sleep left
```

- [ ] **Step 2: Run, verify failures. Step 3: Implement.** `fast_agent`: `manager.start(...)`, then `manager.wait(run_id, wait_seconds, interrupt=cancel_event)`; if the event is set, `manager.cancel(run_id)`. `fast_agent_wait` does the same wait and cancel handling. `RunError` becomes `ToolReply(msg, is_error=True)`. Entry point: `load_settings()`, `RunManager`, SIGTERM and SIGINT handlers call `manager.shutdown()` then exit; after `serve()` returns on EOF, call `manager.shutdown()`. **Step 4: Run the whole suite, verify pass. Step 5: Commit** `feat: wire run manager into mcp tools`

---

### Task 12: Skill, README, and end-to-end host test

**Files:** Create `plugins/dllm-subagents/skills/delegate-fast/SKILL.md`, `README.md`

- [ ] **Step 1: Write `SKILL.md`**

Frontmatter: `name: delegate-fast`; `description:` one line saying to use it when a build or edit job is well scoped and would mean writing a lot of code, to hand it to the fast subagent via `fast_agent`. Body sections from the spec's "The skill" section: when to delegate, when not to, how to brief (always absolute `workdir`, a `check_command` when one exists, list known `files`), while a run is active, after the run (review diff, `fast_agent_wait` while running, `fast_agent_undo` if wrong, `fast_agent_cancel` to stop), parallel work. **The "when to delegate" and "when not to" lists are a user contribution point:** leave them as a marked TODO with the spec's examples as comments, and ask the user to write them in their own words before committing.

- [ ] **Step 2: Write `README.md`**: what it does; install in Claude Code (`/plugin marketplace add camkeith/dllm-subagents`, then install `dllm-subagents`); install in Codex (exact commands found in Task 1); keys (env vars or `keys.env` with `chmod 600`, and why Codex needs the file); providers table with the verified models and the date checked; adding a custom provider (example `config.json`); limits and how to change them; guardrails and the warning that they are not a security boundary; Mercury shutdown after 2026-12-28 and how to switch `default_provider`; the spike results summary; troubleshooting (keys not found, Codex 60 s timeout).

- [ ] **Step 3: End-to-end in Claude Code.** In a scratch repo: `claude --plugin-dir <abs path>/plugins/dllm-subagents -p "Use the delegate-fast skill: have the fast subagent add a slugify(text) function to text_utils.py with unittest tests, check command python3 -m unittest -q. Report the status line and changed files." --allowedTools "mcp__plugin_dllm-subagents_fast-agent__fast_agent,mcp__plugin_dllm-subagents_fast-agent__fast_agent_wait"` (adjust tool names to what Task 1 showed). Expected: `status: done` and the two files.

- [ ] **Step 4: End-to-end in Codex** with the install from Task 1: `codex exec` with the same prompt. Expected: `status: done`.

- [ ] **Step 5: Commit** `docs: add delegate-fast skill and readme`

---

### Task 13: Speed check and compatibility pass

**Files:** Modify `README.md`, `docs/superpowers/notes/2026-11-14-live-spike.md`

- [ ] **Step 1: Python 3.9 run.** Run: `PYTHONPATH=plugins/dllm-subagents/server /usr/bin/python3 -m unittest discover -s tests -v`. Expected: all pass on Python 3.9.6.

- [ ] **Step 2: Speed check** per the spec's Testing section: three jobs in a real repo (new module from a spec, the same change across several files, making a failing test pass), each timed end to end with the orchestrator alone and with delegation, on at least two providers. Record a table in the README.

- [ ] **Step 3: Act on results.** If delegation loses on a kind of job, narrow the skill's "when to delegate" list and note it. If it loses on all three, stop and report to the user before any polish.

- [ ] **Step 4: Commit** `docs: record speed check results`
