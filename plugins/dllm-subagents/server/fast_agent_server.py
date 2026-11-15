from __future__ import annotations

import os
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dllm import VERSION  # noqa: E402
from dllm.config import ConfigError, load_settings  # noqa: E402
from dllm.mcp import McpServer  # noqa: E402
from dllm.runs import RunManager  # noqa: E402
from dllm.server_tools import build_tool_specs  # noqa: E402


def _configure_stdio() -> None:
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass


def _install_signal_handlers(manager) -> None:
    started = threading.Lock()

    def finish_and_exit() -> None:
        try:
            if manager is not None:
                manager.shutdown()
        finally:
            os._exit(0)

    def handler(signum, frame) -> None:
        # Never shut down on the handler's own stack: the main thread may hold the manager's lock.
        if started.acquire(False):
            threading.Thread(target=finish_and_exit, daemon=True).start()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def main() -> None:
    _configure_stdio()
    manager = None
    config_error = None
    wait_seconds = 50.0
    try:
        settings = load_settings()
        manager = RunManager(settings)
        wait_seconds = settings.wait_seconds
    except ConfigError as exc:
        config_error = str(exc)
        print("fast-agent: config error: " + config_error, file=sys.stderr, flush=True)
    _install_signal_handlers(manager)
    server = McpServer("fast-agent", VERSION, build_tool_specs(manager, wait_seconds, config_error))
    server.serve(sys.stdin, sys.stdout)
    if manager is not None:
        manager.shutdown()


if __name__ == "__main__":
    main()
