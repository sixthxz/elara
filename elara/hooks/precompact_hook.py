"""
PreCompact hook — block Claude Code's built-in compaction when Elara proxy is running.

Claude Code calls this before compacting context. We return {"decision": "block"}
whenever the proxy is live on port 8877, because Elara manages its own compression
and native compaction would destroy the turn history the RRG signals depend on.
"""
import json
import socket
import sys


def proxy_is_running(host: str = "localhost", port: int = 8877) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    try:
        _input = json.loads(sys.stdin.read())
    except Exception:
        _input = {}

    if proxy_is_running():
        sys.stdout.write(json.dumps({"decision": "block"}) + "\n")
    else:
        sys.stdout.write(json.dumps({"decision": "proceed"}) + "\n")


if __name__ == "__main__":
    main()
