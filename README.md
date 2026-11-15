# dllm-subagents

A plugin for Claude Code and Codex that lets the main agent hand build and edit jobs to a fast-model coding subagent (Mercury, or gpt-oss-120b on Cerebras, Groq, or OpenRouter). The orchestrator keeps the design work; the fast model types the code.

## What it is and how it works

The plugin has two parts: a stdlib-only Python MCP server that exposes four tools (`fast_agent`, `fast_agent_wait`, `fast_agent_cancel`, `fast_agent_undo`), and a skill, `delegate-fast`, that teaches the host when and how to use them.

Each run is a loop where the subagent can `read`, `write`, `edit`, run `bash`, and `finish`. When it finishes, the harness (not the subagent) computes the git diff of what changed and runs your `check_command` if you gave one, then returns the status, diff, and check result to the orchestrator. Every run takes a snapshot first, so `fast_agent_undo` can revert it.

## Install in Claude Code

```
/plugin marketplace add camkeith/dllm-subagents
/plugin install dllm-subagents@dllm-subagents
```

For local development:

```
claude --plugin-dir <repo>/plugins/dllm-subagents
```

## Install in Codex

Verified locally with Codex CLI 0.164 from a local path:

```
codex plugin marketplace add camkeith/dllm-subagents
codex plugin add dllm-subagents@dllm-subagents
```

Codex copies the plugin into `~/.codex/plugins/cache/`. To pick up updates, run `codex plugin marketplace upgrade` and reinstall.

Codex uses `.codex-plugin/plugin.json` and `codex.mcp.json` instead of `.mcp.json` because it does not expand `${CLAUDE_PLUGIN_ROOT}` in `.mcp.json`; `codex.mcp.json` uses a path relative to the plugin.

## Requirements

- Python 3.9 or newer, available as `python3` on PATH.
- git 2.26 or newer (snapshots use `--pathspec-from-file`).
- macOS or Linux.

## API keys

Provide the key for your provider either as an environment variable or in `~/.config/dllm-subagents/keys.env`, one `NAME=value` per line:

```
INCEPTION_API_KEY=...
CEREBRAS_API_KEY=...
```

```
chmod 600 ~/.config/dllm-subagents/keys.env
```

Codex forwards only the four provider variables listed in `codex.mcp.json` (`INCEPTION_API_KEY`, `CEREBRAS_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`), so the keys file is the reliable option there. Keys are removed from the subagent's shell environment and redacted from tool output, logs, and results.

## Providers

| Name | Base URL | Key env var | Model |
| --- | --- | --- | --- |
| `mercury` (default) | `https://api.inceptionlabs.ai/v1` | `INCEPTION_API_KEY` | `mercury-3` |
| `cerebras` | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | `gpt-oss-120b` |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `openai/gpt-oss-120b:nitro` |

Model IDs as shipped; live verification pending (see Speed and live results).

**Mercury shutdown:** Inception's Mercury API stops after December 28, 2026. Change `default_provider` in `~/.config/dllm-subagents/config.json` before then.

Example `~/.config/dllm-subagents/config.json`:

```json
{
  "default_provider": "cerebras",
  "limits": {"max_steps": 60, "max_seconds": 600},
  "providers": {
    "local": {
      "base_url": "http://localhost:8000/v1",
      "api_key_env": "LOCAL_API_KEY",
      "model": "my-local-model",
      "max_tokens_field": "max_tokens"
    }
  }
}
```

Any OpenAI-compatible chat completions endpoint works (a local vLLM or Ollama, for example). Use it with `provider: "local"` or by setting it as `default_provider`.

Give a `base_url` you add an `https://` address unless it is on your own machine. The key travels in the `Authorization` header, and urllib follows redirects and sends that header again to the new address.

The Cerebras free trial (5 requests per minute on gpt-oss-120b) is too slow for an agent loop; use pay-as-you-go.

## Limits

Defaults: 40 steps, 300 seconds, 2,000,000 tokens per run. Change them under `limits` in `config.json`, or per call with `max_steps` (1 to 150) and `max_seconds` (10 to 1800).

Each `fast_agent` and `fast_agent_wait` call returns within 50 seconds by default, with `status: running` if the run is still going; the orchestrator then calls `fast_agent_wait` again. Set `wait_seconds` in `config.json` to change it, up to 55. The cap exists because Codex's default tool timeout is 60 seconds.

## Safety and guardrails

The subagent's `bash` runs as you, in your repo, outside the permission prompts of Claude Code or Codex. Guardrails block common accidents. A `bash` command is refused when it has:

- a git command that is not read-only (only `status`, `diff`, `log`, `show`, `ls-files`, `grep`, `blame`, and `rev-parse` are allowed, so `git add`, `git commit`, `git stash` and the rest are blocked)
- `sudo` or `doas`
- a recursive `rm` of a path outside the repo or of the repo itself, after a `cd` the guard cannot follow, or on a target that uses a variable or command substitution (such as `rm -rf $DIR/*`)
- `curl` or `wget` piped into a shell, or a shell reading a download (`bash <(curl ...)`)
- `mkfs`, or `dd` writing to a device under `/dev`
- `shutdown`, `reboot`, `poweroff`, or `halt`
- `keys.env`, or the full path of the plugin's config directory, anywhere in it (a `~/...` spelling of that path is not caught)

These are **not a security boundary.** The subagent has network access and can read files outside the repo through bash, and a deliberately encoded read (base64, for example) can get around key redaction. Do not point it at untrusted repos. The file tools (`read`, `write`, `edit`) are confined to the repo, and `write` and `edit` refuse paths inside a `.git` directory.

## Snapshots and undo

- The repo must be a git repo (`git init` if it is not).
- Snapshots never touch your index, HEAD, refs, or stash.
- Files ignored by git before the run are never deleted by undo, and changes the run makes to them are not tracked.
- Make sure dependency and build directories (`node_modules`, `.venv`, `dist`, `build`) are git-ignored, especially in a fresh `git init` repo. A run that creates huge numbers of files git does not ignore makes snapshots slow, and the result then lists only 100 files plus a `git diff --stat` command for the rest.
- One active run per repo or worktree, enforced across sessions.
- Run records and logs live in `$DLLM_SUBAGENTS_RUNS_DIR` when it is set, else in the plugin data directory, or `~/.cache/dllm-subagents/runs` when there is none.

## Speed and live results

Results will be recorded here after the live smoke test (`python3 scripts/smoke.py`) and the speed check.

## Troubleshooting

- **"No API key for provider ..."**: the server looks for the provider's key variable in the environment, then in `~/.config/dllm-subagents/keys.env`. Set it in one of those places.
- **"A fast_agent run from another session is active"**: only one run per repo or worktree is allowed. Wait for it, cancel it, or use a separate git worktree.
- **A run record stuck as `running` after a crash** cannot be undone, because the run never finished and its changes were not recorded as complete.
- **Codex server does not start**: reinstall the plugin (`codex plugin add dllm-subagents@dllm-subagents`) and check `codex mcp list`.

## License

MIT. See `LICENSE`. Ported code from mini-swe-agent and Pi is credited in `THIRD_PARTY_NOTICES`.
