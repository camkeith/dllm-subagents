from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dllm import VERSION  # noqa: E402
from dllm.mcp import McpServer  # noqa: E402
from dllm.server_tools import build_tool_specs  # noqa: E402


def main() -> None:
    McpServer("fast-agent", VERSION, build_tool_specs(None)).serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
