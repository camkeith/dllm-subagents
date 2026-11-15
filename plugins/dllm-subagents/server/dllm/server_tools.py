from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

from dllm.config import DEFAULT_WAIT_SECONDS, HARD_MAX_SECONDS, HARD_MAX_STEPS, HARD_MIN_SECONDS, HARD_MIN_STEPS
from dllm.mcp import ToolReply, ToolSpec
from dllm.runs import RunError, render_view

_RUN_ID = {"type": "object", "properties": {"run_id": {"type": "string", "description": "Id returned by fast_agent."}}, "required": ["run_id"]}

TOOL_SCHEMAS: Dict[str, dict] = {
    "fast_agent": {
        "description": "Hand a build or edit job to a fast-model coding subagent working in a git repo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What the subagent should do."},
                "workdir": {"type": "string", "description": "Absolute path of the git repo to work in. Always pass it."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Files to give the subagent as context."},
                "check_command": {"type": "string", "description": "Command that verifies the work."},
                "provider": {"type": "string", "description": "Model provider to use."},
                "model": {"type": "string", "description": "Model name to use."},
                "max_steps": {"type": "integer", "minimum": HARD_MIN_STEPS, "maximum": HARD_MAX_STEPS,
                              "description": "Step limit."},
                "max_seconds": {"type": "integer", "minimum": HARD_MIN_SECONDS, "maximum": HARD_MAX_SECONDS,
                                "description": "Time limit in seconds."},
            },
            "required": ["task"],
        },
    },
    "fast_agent_wait": {"description": "Wait for a fast_agent run to finish or report progress.", "inputSchema": _RUN_ID},
    "fast_agent_cancel": {"description": "Cancel a running fast_agent run.", "inputSchema": _RUN_ID},
    "fast_agent_undo": {"description": "Revert the changes made by a finished fast_agent run.", "inputSchema": _RUN_ID},
}


def _arg(args: dict, key: str, kind, required: bool = False):
    value = args.get(key)
    if value is None:
        if required:
            raise RunError("%s is required" % key)
        return None
    if isinstance(value, bool) or not isinstance(value, kind):
        raise RunError("%s must be of type %s" % (key, kind.__name__))
    return value


def _safe(fn: Callable[[dict, threading.Event], ToolReply]) -> Callable[[dict, threading.Event], ToolReply]:
    def handler(args: dict, cancel: threading.Event) -> ToolReply:
        try:
            return fn(args, cancel)
        except RunError as exc:
            return ToolReply(str(exc), is_error=True)
        except Exception as exc:
            return ToolReply("internal error: %s: %s" % (type(exc).__name__, exc), is_error=True)
    return handler


def _wait_and_render(manager, run_id: str, seconds: float, cancel: threading.Event) -> ToolReply:
    view = manager.wait(run_id, seconds, interrupt=cancel)
    if cancel.is_set():  # the host cancelled the call (Esc): stop the run too
        view = manager.cancel(run_id)
    return ToolReply(render_view(view))


def build_tool_specs(manager, wait_seconds: float = DEFAULT_WAIT_SECONDS,
                     config_error: Optional[str] = None) -> List[ToolSpec]:
    def fast_agent(args: dict, cancel: threading.Event) -> ToolReply:
        began = time.monotonic()
        files = _arg(args, "files", list)
        if files is not None and not all(isinstance(f, str) for f in files):
            raise RunError("files must be a list of strings")
        run_id = manager.start(
            task=_arg(args, "task", str, True),
            workdir=_arg(args, "workdir", str),
            files=files,
            check_command=_arg(args, "check_command", str),
            provider=_arg(args, "provider", str),
            model=_arg(args, "model", str),
            max_steps=_arg(args, "max_steps", int),
            max_seconds=_arg(args, "max_seconds", int),
        )
        remaining = max(0.0, wait_seconds - (time.monotonic() - began))
        return _wait_and_render(manager, run_id, remaining, cancel)

    def fast_agent_wait(args: dict, cancel: threading.Event) -> ToolReply:
        return _wait_and_render(manager, _arg(args, "run_id", str, True), wait_seconds, cancel)

    def fast_agent_cancel(args: dict, cancel: threading.Event) -> ToolReply:
        return ToolReply(render_view(manager.cancel(_arg(args, "run_id", str, True))))

    def fast_agent_undo(args: dict, cancel: threading.Event) -> ToolReply:
        return ToolReply(manager.undo(_arg(args, "run_id", str, True)))

    handlers = {
        "fast_agent": fast_agent,
        "fast_agent_wait": fast_agent_wait,
        "fast_agent_cancel": fast_agent_cancel,
        "fast_agent_undo": fast_agent_undo,
    }
    if config_error is not None:
        def failing(args: dict, cancel: threading.Event) -> ToolReply:
            return ToolReply(config_error, is_error=True)
        handlers = {name: failing for name in handlers}
    return [
        ToolSpec(name, schema["description"], schema["inputSchema"], _safe(handlers[name]))
        for name, schema in TOOL_SCHEMAS.items()
    ]
