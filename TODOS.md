# TODOS

## P1

- [x] Live smoke test and eval suite on Mercury (2026-11-16): `mercury-3` is listed, smoke passes, and `scripts/evals.py` passes 16 of 16 runs (see the README).
- [ ] Eval scoring can be gamed (fix before trusting eval numbers): `run_hidden` passes on exit code alone, so model code that calls `os._exit(0)` on import passes 6 of 8 hidden checks (shown offline); a model-made `eval_hidden_<case>/` package shadows the hidden module; `fix_failing_test` hashes only `tests/test_inventory.py`, not `tests/__init__.py`; and the model's bash can read `scripts/eval_cases.py` and kept runs' hidden files. Fix: run hidden checks through a runner that prints a per-run nonce only when tests ran and passed, refuse leftover `eval_hidden_*` paths, hash all of `tests/`, and point the run's `HOME`/`TMPDIR` at a throwaway directory.
- [ ] Live smoke test and `python3 scripts/evals.py --repeat 2` on Cerebras, Groq and OpenRouter once their keys exist; verify their shipped model IDs with `python3 scripts/smoke.py --models`.
- [ ] Harder eval jobs: every current job passes on Mercury, so the suite cannot yet show where fast models stop being reliable. Add jobs that need more steps or judgment (a multi-module feature, a refactor with a behaviour-preserving check, a non-Python stack).
- [ ] Tighter eval scoring: the `rename_across_files` hidden check does not test that `get_user` still returns a copy, `build_email` for an active user, or the full `profile_card` dict; the model-tests guard (`spec_module`, `follow_pattern`, `large_file_edit`, `debug_symptom`) counts any nonzero exit on the original code as a caught bug, so new tests that merely fail to import on it pass. Hidden checks run with the repo first on `sys.path`, so a model-written `unittest.py` could shadow the stdlib, and `rename_across_files` fails a correct rename when the model leaves a helper script containing the old name. Provider failures (status `error`, such as a rate limit) count as model failures in the pass rate; report them separately.
- [ ] Snapshot git and model-controlled config: `dllm/snapshot.py` runs `git add` on the run's repo with only the provider key variables removed from the environment. A clean filter the subagent writes into `.git/config` plus `.gitattributes` would then run with the server's other secrets (tokens, cloud credentials). Not verified empirically (the probe was blocked); build the snapshot index without filters (`git hash-object --no-filters -w` plus `update-index`), and pass `run_env`'s scrubbed environment to git.
- [ ] Eval harness and symlinks: `eval_cases.write_files` follows a symlink the model planted at a hidden-check path, and the model-tests guard's `copytree` follows symlinks (a link to `/` makes it copy until the hidden-check timeout).
- [ ] Finish guard granularity: it only covers one response (a bare finish in the next response is accepted even if the failed edit was never fixed); a failed edit retried successfully in the same response still refuses that finish; writes through bash or an unknown tool name are not tracked; failure is detected by the `Error:` output prefix rather than an explicit flag.
- [ ] `scripts/evals.py` small fixes: use `tempfile.mkstemp` for the `--out` probe and atomic write (the fixed `<out>.tmp` name clobbers a file of that name and collides across concurrent runs); de-duplicate repeated `--case`/`--provider` names; unit-test `summary_rows` and smoke's unknown-provider error.
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
