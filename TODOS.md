# TODOS

## P1

- [ ] Live smoke test on real providers, and verify the shipped model IDs: run `python3 scripts/smoke.py --models`, then `python3 scripts/smoke.py` (default mode). Record the date checked in the README.
- [ ] Real end-to-end runs in Claude Code and in Codex (the Codex API key must be renewed first).
- [ ] Speed check: time a delegated job against the orchestrator doing the same job alone, and record the numbers in the README under "Speed and live results".
- [ ] HTTP keep-alive: decide whether to reuse connections based on the speed check numbers.
- [ ] Context-length handling: once the real error shapes are known, map a provider's context-length 400 to `limit_reached` and/or trim old tool outputs.
- [ ] Rewrite the skill's delegation rules (`plugins/dllm-subagents/skills/delegate-fast/SKILL.md`) in the owner's own words.

## P2

- [ ] A case-only rename of an untracked file (`notes.txt` to `Notes.txt`) cannot be undone on a case-insensitive filesystem (the macOS default).
- [ ] Snapshot cost on huge repos: each snapshot now runs extra `git ls-files` walks (nested repositories, assume-unchanged, kept files). Measure it in the speed check.
- [ ] The guard does not follow a `cd` run through `eval` (`eval cd /; rm -rf x`).
- [ ] The 30 ms bound in `tests/test_shell.py` (fastest of 10 quick commands) may fail on very slow hosts.
