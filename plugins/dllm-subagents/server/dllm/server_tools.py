from __future__ import annotations

import threading
from typing import Dict, List

from dllm.mcp import ToolReply, ToolSpec

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
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 150, "description": "Step limit."},
                "max_seconds": {"type": "integer", "minimum": 10, "maximum": 1800, "description": "Time limit in seconds."},
            },
            "required": ["task"],
        },
    },
    "fast_agent_wait": {"description": "Wait for a fast_agent run to finish or report progress.", "inputSchema": _RUN_ID},
    "fast_agent_cancel": {"description": "Cancel a running fast_agent run.", "inputSchema": _RUN_ID},
    "fast_agent_undo": {"description": "Revert the changes made by a finished fast_agent run.", "inputSchema": _RUN_ID},
}


def _not_implemented(args: dict, cancel: threading.Event) -> ToolReply:
    return ToolReply("not implemented yet", is_error=True)


def build_tool_specs(manager, wait_seconds: float = 50.0) -> List[ToolSpec]:
    return [
        ToolSpec(name, schema["description"], schema["inputSchema"], _not_implemented)
        for name, schema in TOOL_SCHEMAS.items()
    ]
