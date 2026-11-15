"""Agent loop and check loop.

Control flow is ported from mini-swe-agent's DefaultAgent
(https://github.com/SWE-agent/mini-swe-agent, MIT, Copyright (c) 2025 Kilian A. Lieret
and Carlos E. Jimenez): run() loops step() (query + execute) until the run ends, limits are
checked before each model call, consecutive format errors are counted and capped, and the
trajectory is saved after every step.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .config import Limits
from .llm import LLMError
from .prompts import SYSTEM_PROMPT
from .shell import run_command, truncate_middle
from .tools import TOOL_SCHEMAS, ToolContext, ToolError, run_tool
from .tools import redact as _redact

TOOL_RESULT_MAX_CHARS = 30_000
CHECK_TAIL_CHARS = 4_000

FORMAT_ERROR_MESSAGE = (
    "Your last reply did not call a tool. Call exactly one of the available tools. "
    "If you are done, call finish with a short summary of what you changed."
)


@dataclass
class CheckResult:
    command: str
    exit_code: int
    output_tail: str
    attempts: int


@dataclass
class AgentResult:
    status: str
    summary: str
    limit: Optional[str]
    error: Optional[str]
    check: Optional[CheckResult]
    steps: int
    seconds: float
    input_tokens: int
    output_tokens: int


class _Stop(Exception):
    """Control flow: ends the run with the given status."""

    def __init__(self, status: str, *, limit: Optional[str] = None, error: Optional[str] = None):
        super().__init__(status)
        self.status = status
        self.limit = limit
        self.error = error


def _label(name: str, args: dict) -> str:
    hint = ""
    if isinstance(args, dict):
        for key in ("path", "command", "summary"):
            value = args.get(key)
            if isinstance(value, str) and value:
                hint = value.splitlines()[0][:80]
                break
    return "%s: %s" % (name, hint) if hint else name


class Agent:
    def __init__(
        self,
        client,
        ctx: ToolContext,
        limits: Limits,
        *,
        check_command: Optional[str] = None,
        max_check_failures: int = 3,
        max_consecutive_format_errors: int = 3,
        check_timeout: int = 300,
        on_step: Optional[Callable[[int, str], None]] = None,
        trajectory_path: Optional[Path] = None,
        clock: Callable[[], float] = time.monotonic,
        redact: Sequence[str] = (),
    ):
        self.client = client
        self.ctx = ctx
        self.limits = limits
        self.check_command = check_command
        self.max_check_failures = max_check_failures
        self.max_consecutive_format_errors = max_consecutive_format_errors
        self.check_timeout = check_timeout
        self.on_step = on_step
        self.trajectory_path = trajectory_path
        self.clock = clock
        self.redact = tuple(redact)  # secret values scrubbed from everything the model or result sees
        self.messages: List[dict] = []
        self.steps = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.check: Optional[CheckResult] = None
        self.summary = ""
        self._format_errors = 0
        self._start = 0.0
        self._status = "running"

    # -- public --------------------------------------------------------------------------

    def run(self, task_message: str) -> AgentResult:
        self._start = self.clock()
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._scrub(task_message)},
        ]
        stop: Optional[_Stop] = None
        while stop is None:
            try:
                self._step()
            except _Stop as exc:
                stop = exc
            except Exception as exc:  # run() never raises: any unexpected failure ends the run
                stop = _Stop("error", error="%s: %s" % (type(exc).__name__, exc))
            else:
                self._save()
        return self._finish_run(stop)

    # -- loop ----------------------------------------------------------------------------

    def _step(self) -> None:
        completion = self._query()
        self._execute(completion)

    def _query(self):
        self._check_cancel()
        self._check_limits()
        self.steps += 1
        deadline = self._start + self.limits.max_seconds
        try:
            completion = self.client.complete(self.messages, TOOL_SCHEMAS, deadline)
        except LLMError as exc:
            raise _Stop("error", error=str(exc))
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.messages.append(self._scrub_message(completion.message))
        return completion

    def _check_cancel(self) -> None:
        if self.ctx.cancel_event.is_set():
            raise _Stop("cancelled")

    def _check_limits(self) -> None:
        if self.steps >= self.limits.max_steps:
            raise _Stop("limit_reached", limit="max_steps")
        if self.clock() - self._start >= self.limits.max_seconds:
            raise _Stop("limit_reached", limit="max_seconds")
        if self.input_tokens + self.output_tokens >= self.limits.max_tokens_total:
            raise _Stop("limit_reached", limit="max_tokens_total")

    def _scrub(self, text: str) -> str:
        return _redact(text, self.redact)

    def _scrub_message(self, message: dict) -> dict:
        # Stored copy only; tool calls still run with the model's raw arguments.
        return json.loads(self._scrub(json.dumps(message))) if self.redact else message

    def _tool_message(self, call_id: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id,
             "content": truncate_middle(self._scrub(content), TOOL_RESULT_MAX_CHARS)}
        )

    def _progress(self, label: str) -> None:
        if self.on_step is not None:
            self.on_step(self.steps, label)

    def _execute(self, completion) -> None:
        calls = completion.tool_calls
        if not calls:
            self._progress("no tool call")
            self._format_error(FORMAT_ERROR_MESSAGE)
            return
        bad_format = False
        pending_stop: Optional[_Stop] = None
        finished = False
        for i, call in enumerate(calls):
            if finished:
                self._tool_message(call.id, "This call was not run: finish was already called in this response.")
                continue
            if self.ctx.cancel_event.is_set():
                for rest in calls[i:]:
                    self._tool_message(rest.id, "This call was not run: the run was cancelled.")
                raise _Stop("cancelled")
            try:
                args = json.loads(call.arguments) if call.arguments.strip() else {}
                if not isinstance(args, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                bad_format = True
                self._progress("%s: invalid arguments" % call.name)
                self._tool_message(call.id, "Error: invalid JSON arguments: %s" % exc)
                continue
            self._progress(self._scrub(_label(call.name, args)))
            if call.name == "finish":
                finished = True
                pending_stop = self._handle_finish(call.id, args)
            else:
                self._tool_message(call.id, self._run_tool(call.name, args))
        if pending_stop is not None:
            raise pending_stop
        if bad_format:
            self._count_format_error()
        else:
            self._format_errors = 0

    def _run_tool(self, name: str, args: dict) -> str:
        try:
            return run_tool(name, args, self.ctx)
        except ToolError as exc:
            return "Error: %s" % exc
        except Exception as exc:  # unexpected tool failure is reported to the model, not fatal
            return "Error: %s: %s" % (type(exc).__name__, exc)

    def _format_error(self, user_message: str) -> None:
        self.messages.append({"role": "user", "content": user_message})
        self._count_format_error()

    def _count_format_error(self) -> None:
        self._format_errors += 1
        if 0 < self.max_consecutive_format_errors <= self._format_errors:
            raise _Stop("error", error="Stopped after %d consecutive format errors (no valid tool call)"
                        % self._format_errors)

    # -- finish and check loop -----------------------------------------------------------

    def _handle_finish(self, call_id: str, args: dict) -> Optional[_Stop]:
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            self._tool_message(call_id, "Error: finish requires a non-empty 'summary' string. Call finish again with one.")
            return None
        self.summary = self._scrub(summary)
        if not self.check_command:
            self._tool_message(call_id, "Finished.")
            return _Stop("done")
        attempts = (self.check.attempts if self.check else 0) + 1
        try:
            res = run_command(
                self.check_command, str(self.ctx.workdir), self.check_timeout, self.ctx.cancel_event,
                env=self.ctx.env,
            )
        except OSError as exc:
            self._tool_message(call_id, "Error: cannot run check command: %s" % exc)
            return _Stop("error", error="cannot run check command: %s" % (exc.strerror or exc))
        output = self._scrub(res.output)
        if res.timed_out:
            output += "\n[check timed out after %ds]" % self.check_timeout
        self.check = CheckResult(self.check_command, res.returncode, output[-CHECK_TAIL_CHARS:], attempts)
        if res.cancelled:
            self._tool_message(call_id, "The check was cancelled.")
            return _Stop("cancelled")
        if res.returncode == 0 and not res.timed_out:
            self._tool_message(call_id, "Check passed. Finished.")
            return _Stop("done")
        if attempts >= self.max_check_failures:
            self._tool_message(call_id, "The check failed (exit code %d). No attempts left.\n%s"
                               % (res.returncode, self.check.output_tail))
            return _Stop("check_failed")
        self._tool_message(
            call_id,
            "The check failed (exit code %d). Fix it and call finish again.\n%s"
            % (res.returncode, self.check.output_tail),
        )
        return None

    # -- results -------------------------------------------------------------------------

    def _seconds(self) -> float:
        return self.clock() - self._start

    def _finish_run(self, stop: _Stop) -> AgentResult:
        status = stop.status
        if status == "limit_reached" and self.check is not None and self.check.exit_code != 0:
            status = "check_failed"
        self._status = status
        self._save()
        return AgentResult(
            status=status,
            summary=self.summary,
            limit=stop.limit,
            error=stop.error,
            check=self.check,
            steps=self.steps,
            seconds=self._seconds(),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    def _save(self) -> None:
        try:
            self._write_trajectory()
        except Exception:  # a failing trajectory write must not break the run
            pass

    def _write_trajectory(self) -> None:
        if self.trajectory_path is None:
            return
        path = Path(self.trajectory_path)
        data = {
            "messages": self.messages,
            "stats": {
                "steps": self.steps,
                "seconds": self._seconds(),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "status": self._status,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(path))
