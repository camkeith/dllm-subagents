"""Run manager: validates a request, snapshots the repo, runs the agent in a background thread,
keeps one active run per repository (in-process and across processes), persists run records so
undo works after a restart, and renders compact results for the orchestrating model."""
from __future__ import annotations

import collections
import dataclasses
import fcntl
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import shell, snapshot
from .agent import Agent, AgentResult
from .config import ConfigError, Profile, Settings, clamp_limits, resolve_api_key
from .llm import ChatClient
from .prompts import build_task_message, load_files
from .tools import ToolContext, redact, scrubbed_env

CANCEL_WAIT_SECONDS = 5.0
SHUTDOWN_GRACE_SECONDS = 5.0
LOCK_NAME = "dllm-subagents.lock"
PASSED_CHECK_TAIL_LINES = 5
SUMMARY_MAX_CHARS = 4_000
MAX_LISTED_FILES = 100
MAX_GROUP_LINES = 20
_POLL = 0.05
_RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
_UNDO_OWNER = "undo:"


def runs_dir() -> Path:
    env = os.environ
    if env.get("DLLM_SUBAGENTS_RUNS_DIR"):
        path = Path(env["DLLM_SUBAGENTS_RUNS_DIR"])
    elif env.get("CLAUDE_PLUGIN_DATA"):
        path = Path(env["CLAUDE_PLUGIN_DATA"]) / "runs"
    elif env.get("PLUGIN_DATA"):
        path = Path(env["PLUGIN_DATA"]) / "runs"
    else:
        path = Path.home() / ".cache" / "dllm-subagents" / "runs"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


@dataclass
class RunRecord:
    run_id: str
    workdir: str
    repo: str
    before_tree: str
    after_tree: Optional[str]
    status: str
    started_at: float
    ended_at: Optional[float]
    provider: str
    model: str
    undone: bool = False
    ignored: List[str] = field(default_factory=list)  # ignored at before-time; excluded and protected
    error: Optional[str] = None

    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(str(tmp), str(path))

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("run record is not a JSON object")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class RunError(Exception):
    """A problem to show the orchestrator as is."""


@dataclass
class _RepoLock:
    key: str  # resolved repo root
    owner: str  # run_id, or _UNDO_OWNER + run_id
    fd: int


class _Run:
    def __init__(self, record: RunRecord, directory: Path, lock: _RepoLock):
        self.record = record
        self.record_path = directory / (record.run_id + ".json")
        self.log_path = directory / (record.run_id + ".trajectory.json")
        self.lock = lock
        self.cancel_event = threading.Event()
        self.done = threading.Event()  # set after view is built and the lock released
        self.save_lock = threading.Lock()
        self.final_saved = False
        self.started = time.monotonic()
        self.progress = (0, "")
        self.view: Optional[dict] = None

    def on_step(self, step: int, label: str) -> None:
        self.progress = (step, label)


def _undone_on_disk(path: Path) -> bool:
    try:
        return RunRecord.load(path).undone
    except (OSError, ValueError, TypeError):
        return False


def _error_view(run: _Run, error: str) -> dict:
    """Minimal finished view for when the normal one could not be built."""
    return {
        "status": "error", "run_id": run.record.run_id, "summary": "", "limit": None,
        "error": error, "check": None, "stats": {}, "changed_files": None, "diff": "",
        "diff_truncated": False, "log_path": str(run.log_path),
        "before_tree": run.record.before_tree, "after_tree": None,
    }


def _new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def _busy(repo: str, owner: str) -> RunError:
    if owner.startswith(_UNDO_OWNER):
        return RunError("An undo is in progress in %s. Try again in a moment." % repo)
    return RunError(
        "A fast_agent run is already active in %s (run_id %s). Wait for it with fast_agent_wait "
        "or stop it with fast_agent_cancel." % (repo, owner)
    )


def _flock(repo: Path) -> int:
    try:
        path = snapshot.git_path(repo, LOCK_NAME)
    except snapshot.SnapshotError as exc:
        raise RunError(str(exc))
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        raise RunError("cannot open the repository lock %s: %s" % (path, exc.strerror or exc))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RunError(
            "A fast_agent run from another session is active in %s. Wait for it to finish, "
            "or stop it from that session." % repo
        )
    except OSError as exc:
        os.close(fd)
        raise RunError("cannot lock %s: %s" % (path, exc.strerror or exc))
    return fd


class RunManager:
    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Optional[Callable[[Profile, str, Optional[str]], object]] = None,
    ):
        self.settings = settings
        self._client_factory = client_factory or (
            lambda profile, key, model: ChatClient(profile, key, model=model)
        )
        self._lock = threading.Lock()
        self._runs: Dict[str, _Run] = {}
        self._active: Dict[str, str] = {}  # resolved repo root -> owner
        self._closed = False

    # -- public --------------------------------------------------------------------------

    def start(
        self,
        task: str,
        workdir: Optional[str] = None,
        files: Optional[List[str]] = None,
        check_command: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_steps: Optional[int] = None,
        max_seconds: Optional[int] = None,
    ) -> str:
        # Validate everything before taking the lock or a snapshot.
        if self._closed:
            raise RunError("the server is shutting down; not starting a new run")
        if not isinstance(task, str) or not task.strip():
            raise RunError("task must be a non-empty string")
        workdir = workdir or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
        if not os.path.isabs(workdir):
            raise RunError("workdir must be an absolute path, got %r" % workdir)
        if not os.path.isdir(workdir):
            raise RunError("workdir %s does not exist or is not a directory" % workdir)
        try:
            repo = snapshot.repo_root(Path(workdir))
        except snapshot.SnapshotError as exc:
            raise RunError(str(exc))
        name = provider or self.settings.default_provider
        profile = self.settings.profiles.get(name)
        if profile is None:
            raise RunError("unknown provider %r; known providers: %s"
                           % (name, ", ".join(sorted(self.settings.profiles))))
        try:
            key = resolve_api_key(profile)
        except ConfigError as exc:
            raise RunError(str(exc))
        try:
            limits = clamp_limits(self.settings.limits, max_steps, max_seconds)
        except (TypeError, ValueError):
            raise RunError("max_steps and max_seconds must be integers")
        try:
            directory = runs_dir()
        except OSError as exc:
            raise RunError("cannot create the runs directory: %s" % exc)
        loaded, notes = load_files(Path(workdir), list(files or []))
        message = build_task_message(task, workdir, check_command, loaded, notes)
        client = self._client_factory(profile, key, model)
        secret_names = [p.api_key_env for p in self.settings.profiles.values()]
        known_keys = [key]
        for other in self.settings.profiles.values():
            if other is not profile:
                try:
                    known_keys.append(resolve_api_key(other))
                except ConfigError:
                    pass

        run_id = _new_run_id()
        lock = self._acquire(repo, run_id)
        run: Optional[_Run] = None
        try:
            before = snapshot.snapshot_tree(repo)
            ignored = snapshot.ignored_paths(repo)
            record = RunRecord(
                run_id=run_id, workdir=workdir, repo=lock.key, before_tree=before, after_tree=None,
                status="running", started_at=time.time(), ended_at=None, provider=name,
                model=model or profile.model, ignored=ignored,
            )
            run = _Run(record, directory, lock)
            record.save(run.record_path)
            ctx = ToolContext(workdir=Path(workdir), cancel_event=run.cancel_event,
                              env=scrubbed_env(secret_names))
            agent = Agent(client, ctx, limits, check_command=check_command,
                          on_step=run.on_step, trajectory_path=run.log_path, redact=known_keys)
            thread = threading.Thread(target=self._work, args=(run, agent, message),
                                      name="dllm-run-" + run_id, daemon=True)
            with self._lock:
                if self._closed:
                    raise RunError("the server is shutting down; not starting a new run")
                self._runs[run_id] = run
            thread.start()  # last statement: once it runs, the worker owns the lock
        except BaseException as exc:
            with self._lock:
                self._runs.pop(run_id, None)
            self._release(lock)
            if run is not None:
                try:
                    run.record_path.unlink()
                except OSError:
                    pass
            if isinstance(exc, snapshot.SnapshotError):
                raise RunError(str(exc))
            if isinstance(exc, OSError):
                raise RunError("cannot start the run: %s" % exc)
            raise
        return run_id

    def wait(self, run_id: str, seconds: float, interrupt: Optional[threading.Event] = None) -> dict:
        run = self._live_run(run_id)
        deadline = time.monotonic() + max(0.0, float(seconds))
        while not run.done.is_set():
            if interrupt is not None and interrupt.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            run.done.wait(remaining if interrupt is None else min(remaining, _POLL))
        return self._view(run)

    def cancel(self, run_id: str) -> dict:
        run = self._live_run(run_id)
        if not run.done.is_set():
            run.cancel_event.set()
            with self._lock:  # held so no other run can start a command between check and kill
                alone = not any(r is not run and not r.done.is_set() for r in self._runs.values())
                if alone:
                    shell.kill_all()
            run.done.wait(CANCEL_WAIT_SECONDS)
        return self._view(run)

    def undo(self, run_id: str) -> str:
        path = self._record_path(run_id)
        with self._lock:
            run = self._runs.get(run_id)
        if run is not None and not run.done.is_set():
            raise RunError("run %s is still active; cancel it first with fast_agent_cancel" % run_id)
        if run is not None:
            record = run.record
        else:
            try:
                record = RunRecord.load(path)
            except FileNotFoundError:
                raise RunError("unknown run_id %r" % run_id)
            except (OSError, ValueError, TypeError) as exc:
                raise RunError("cannot read the record of run %s: %s" % (run_id, exc))
        if record.undone:
            raise RunError("run %s was already undone" % run_id)
        if record.status == "running":
            raise RunError(
                "run %s has not finished: it is active in another session, or that session ended "
                "without recording the run's result, so its changes are unknown" % run_id
            )
        if record.after_tree is None:
            raise RunError(
                "cannot undo run %s: its changes are unknown because the working tree could not "
                "be snapshotted after it ended%s"
                % (run_id, " (%s)" % record.error if record.error else "")
            )
        lock = self._acquire(Path(record.repo), _UNDO_OWNER + run_id)
        try:
            if _undone_on_disk(path):  # another session undid it since we read the record
                record.undone = True
                raise RunError("run %s was already undone" % run_id)
            try:
                snapshot.undo(Path(record.repo), record.before_tree, record.after_tree,
                              protect=record.ignored)
            except snapshot.SnapshotError as exc:
                raise RunError(str(exc))
            record.undone = True
            try:
                record.save(path)
            except OSError as exc:
                return ("Undid run %s, but could not mark its record as undone (%s); do not undo "
                        "it again." % (run_id, exc))
        finally:
            self._release(lock)
        if record.before_tree == record.after_tree:
            return "Run %s made no changes; nothing to undo." % run_id
        return "Undid run %s: the files it changed are back to how they were before it started." % run_id

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            active = [r for r in self._runs.values() if not r.done.is_set()]
            for run in active:
                run.cancel_event.set()
            shell.kill_all()
        deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        for run in active:
            run.done.wait(max(0.0, deadline - time.monotonic()))
        for run in active:
            if not run.done.is_set():
                self._save_stuck(run)

    # -- locking -------------------------------------------------------------------------

    def _acquire(self, repo: Path, owner: str) -> _RepoLock:
        key = os.path.realpath(str(repo))
        with self._lock:
            holder = self._active.get(key)
            if holder is not None:
                raise _busy(key, holder)
            self._active[key] = owner
        try:
            fd = _flock(Path(key))
        except BaseException:
            with self._lock:
                self._active.pop(key, None)
            raise
        return _RepoLock(key, owner, fd)

    def _release(self, lock: _RepoLock) -> None:
        fd, lock.fd = lock.fd, -1
        if fd >= 0:
            try:
                os.close(fd)  # releases the flock
            except OSError:
                pass
        with self._lock:
            if self._active.get(lock.key) == lock.owner:
                del self._active[lock.key]

    # -- worker --------------------------------------------------------------------------

    def _work(self, run: _Run, agent: Agent, message: str) -> None:
        result: Optional[AgentResult] = None
        crash: Optional[str] = None
        try:
            result = agent.run(message)
        except Exception as exc:  # Agent.run never raises; last line of defence
            crash = "%s: %s" % (type(exc).__name__, exc)
        finally:
            try:
                self._finish(run, result, crash, agent.redact)
            except Exception as exc:  # never leave a finished run without a view
                run.view = _error_view(run, redact("internal error finishing the run: %s: %s"
                                                   % (type(exc).__name__, exc), agent.redact))
            finally:
                self._release(run.lock)
                run.done.set()

    def _finish(self, run: _Run, result: Optional[AgentResult], crash: Optional[str],
                keys: Sequence[str]) -> None:
        rec = run.record
        repo = Path(rec.repo)
        status = result.status if result is not None else "error"
        errors = [e for e in ((result.error if result is not None else crash),) if e]
        after: Optional[str] = None
        try:
            after = snapshot.snapshot_tree(repo, exclude=rec.ignored)
        except Exception as exc:
            status = "error"
            errors.append("could not snapshot the working tree after the run: %s" % exc)
        changes: Optional[List[snapshot.FileChange]] = None  # None: unknown
        diff, truncated = "", False
        if after is not None:
            try:
                # A deleted or edited file may have held a key: redacted before the diff is cut.
                changes, diff, truncated = snapshot.diff_trees(repo, rec.before_tree, after,
                                                               redact=keys)
            except Exception as exc:
                errors.append("could not compute the diff: %s" % exc)
        with run.save_lock:
            rec.after_tree = after
            rec.status = status
            rec.ended_at = time.time()
            rec.error = redact("; ".join(errors), keys) or None
            try:
                rec.save(run.record_path)
                run.final_saved = True
            except Exception as exc:
                errors.append("could not save the run record, so undo will not work after a "
                              "restart: %s" % exc)
        run.view = self._final_view(run, status, result, redact("; ".join(errors), keys) or None,
                                    changes, diff, truncated)

    def _final_view(self, run: _Run, status: str, result: Optional[AgentResult], error: Optional[str],
                    changes: Optional[List[snapshot.FileChange]], diff: str, truncated: bool) -> dict:
        rec = run.record
        return {
            "status": status,
            "run_id": rec.run_id,
            "summary": result.summary if result is not None else "",
            "limit": result.limit if result is not None else None,
            "error": error,
            "check": asdict(result.check) if result is not None and result.check else None,
            "stats": {
                "steps": result.steps if result is not None else 0,
                "seconds": round(time.monotonic() - run.started, 1),
                "input_tokens": result.input_tokens if result is not None else 0,
                "output_tokens": result.output_tokens if result is not None else 0,
                "provider": rec.provider,
                "model": rec.model,
            },
            "changed_files": None if changes is None else [asdict(c) for c in changes],
            "diff": diff,
            "diff_truncated": truncated,
            "log_path": str(run.log_path),
            "before_tree": rec.before_tree,
            "after_tree": rec.after_tree,
        }

    def _save_stuck(self, run: _Run) -> None:
        """Shutdown fallback for a run whose thread did not stop in time: write a terminal record."""
        rec = run.record
        after: Optional[str] = None
        error = "the server shut down before the run stopped"
        try:
            after = snapshot.snapshot_tree(Path(rec.repo), exclude=rec.ignored)
        except Exception as exc:
            error += "; could not snapshot the working tree after the run: %s" % exc
        with run.save_lock:
            if run.final_saved:
                return
            stuck = dataclasses.replace(rec, status="cancelled" if after else "error",
                                        after_tree=after, ended_at=time.time(), error=error)
            try:
                stuck.save(run.record_path)
            except OSError:
                pass

    # -- views ---------------------------------------------------------------------------

    def _record_path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise RunError("unknown run_id %r" % (run_id,))
        try:
            return runs_dir() / (run_id + ".json")
        except OSError as exc:
            raise RunError("cannot open the runs directory: %s" % exc)

    def _live_run(self, run_id: str) -> _Run:
        with self._lock:
            run = self._runs.get(run_id)
        if run is not None:
            return run
        if self._record_path(run_id).exists():
            raise RunError("run %s is not part of this session (it ran in an earlier or another "
                           "session); only fast_agent_undo works for it" % run_id)
        raise RunError("unknown run_id %r" % (run_id,))

    def _view(self, run: _Run) -> dict:
        if run.done.is_set():
            return dict(run.view or _error_view(run, "the run ended without a result"))
        step, label = run.progress
        elapsed = int(time.monotonic() - run.started)
        if step:
            progress = "step %d, %d s elapsed, last action: %s" % (step, elapsed, label)
        else:
            progress = "starting, %d s elapsed" % elapsed
        if run.cancel_event.is_set():
            progress = "cancel requested, stopping; " + progress
        return {"status": "running", "run_id": run.record.run_id, "progress": progress}


_CHANGE_CODES = {"added": "A", "modified": "M", "deleted": "D"}


def _top_level(path: str) -> str:
    return path.split("/", 1)[0] + "/" if "/" in path else ""


def _file_lines(changes: List[dict]) -> List[str]:
    """One line per changed file, up to MAX_LISTED_FILES; the rest grouped by top-level directory."""
    shown, rest = changes, []
    if len(changes) > MAX_LISTED_FILES:
        # List files from the smallest top-level groups first, so a few real edits stay visible
        # next to thousands of generated files; keep them in diff order.
        sizes = collections.Counter(_top_level(c["path"]) for c in changes)
        ranked = sorted(range(len(changes)), key=lambda i: (sizes[_top_level(changes[i]["path"])], i))
        keep = set(ranked[:MAX_LISTED_FILES])
        shown = [c for i, c in enumerate(changes) if i in keep]
        rest = [c for i, c in enumerate(changes) if i not in keep]
    lines = []
    for c in shown:
        counts = "binary" if c["added"] < 0 else "+%d -%d" % (c["added"], c["removed"])
        lines.append("%s %s (%s)" % (_CHANGE_CODES.get(c["change"], "?"), c["path"], counts))
    groups: Dict[str, collections.Counter] = collections.OrderedDict()
    for c in rest:
        groups.setdefault(_top_level(c["path"]), collections.Counter())[c["change"]] += 1
    order = sorted(groups, key=lambda d: -sum(groups[d].values()))
    for d in order[:MAX_GROUP_LINES]:
        kinds = groups[d]
        detail = ", ".join("%d %s" % (kinds[k], k) for k in ("added", "modified", "deleted") if kinds[k])
        lines.append("+ %d more %s (%s)" % (sum(kinds.values()), "under " + d if d else "at the top level",
                                            detail))
    others = order[MAX_GROUP_LINES:]
    if others:
        lines.append("+ %d more under %d other top-level paths"
                     % (sum(sum(groups[d].values()) for d in others), len(others)))
    return lines


def render_view(view: dict) -> str:
    lines = ["status: %s" % view["status"], "run_id: %s" % view["run_id"]]
    if view["status"] == "running":
        lines.append("progress: %s" % view.get("progress", ""))
        lines.append("Still running. Call fast_agent_wait with this run_id to keep waiting, "
                     "or fast_agent_cancel to stop it.")
        return "\n".join(lines)

    summary = view.get("summary") or "(none, the subagent did not call finish)"
    lines.append("summary: %s" % shell.truncate_middle(summary, SUMMARY_MAX_CHARS))
    if view.get("limit"):
        lines.append("limit: %s" % view["limit"])
    if view.get("error"):
        lines.append("error: %s" % view["error"])
    check = view.get("check")
    if check:
        lines.append("check: `%s` exit code %d (attempt %d)"
                     % (check["command"], check["exit_code"], check["attempts"]))
        tail = (check.get("output_tail") or "").rstrip("\n")
        if tail and check["exit_code"] != 0:
            lines.append("check output (tail):")
            lines.append(tail)
        elif tail:
            lines.append("check output (last %d lines):" % PASSED_CHECK_TAIL_LINES)
            lines.extend(tail.splitlines()[-PASSED_CHECK_TAIL_LINES:])
    s = view.get("stats") or {}
    lines.append("stats: %d steps, %.1f s, tokens %d in / %d out, %s/%s" % (
        s.get("steps", 0), s.get("seconds", 0.0), s.get("input_tokens", 0),
        s.get("output_tokens", 0), s.get("provider", "?"), s.get("model", "?")))

    changes = view.get("changed_files")
    if view.get("after_tree") is None or changes is None:
        lines.append("changed files: unknown, the run's changes could not be computed (see error)")
    elif changes:
        lines.append("changed files (%d):" % len(changes))
        lines.extend(_file_lines(changes))
        if len(changes) > MAX_LISTED_FILES:
            lines.append("see all: git diff --stat %s %s" % (view["before_tree"], view["after_tree"]))
        lines.append("diff:")
        lines.append((view.get("diff") or "").rstrip("\n"))
        if view.get("diff_truncated"):
            lines.append("The diff above was cut. See the rest with `git diff %s %s` in the repository."
                         % (view["before_tree"], view["after_tree"]))
    else:
        lines.append("changed files: none. Files ignored by git (for example through .gitignore) "
                     "are not tracked, so changes to them are not shown.")
    if view.get("log_path"):
        lines.append("log: %s" % view["log_path"])
    if changes and view.get("after_tree"):
        lines.append("To revert this run's changes, call fast_agent_undo with this run_id.")
    return "\n".join(lines)
