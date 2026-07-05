"""
UserPromptSubmit hook — log token estimates and inject Elara compression status.

Receives {"prompt": "...", "session_id": "..."} via stdin.
Writes a JSONL log entry to elara_hook_log.jsonl.
Returns {"additionalContext": "..."} when the proxy has active session data.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_METRICS_PATH = _ROOT / "elara_proxy_metrics.json"
_LOG_PATH = _ROOT / "elara_hook_log.jsonl"


def proxy_is_running(host: str = "localhost", port: int = 8877) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _launch_tray() -> None:
    """Start elara_tray.py in background if proxy is not already running."""
    if proxy_is_running():
        return
    tray_script = _ROOT / "elara_tray.py"
    if not tray_script.exists():
        return
    try:
        kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = True
        subprocess.Popen([sys.executable, str(tray_script)], **kw)
        time.sleep(0.3)  # wait for proxy to bind
    except Exception:
        pass


def _load_compression_summary() -> str:
    """Return a one-line compression status from the last proxy session, or ''."""
    try:
        if not _METRICS_PATH.exists():
            return ""
        payload = json.loads(_METRICS_PATH.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            return ""
        total = len(records)
        compressed = sum(1 for r in records if r.get("compressed"))
        lock_frac = compressed / total
        last = records[-1]
        regime = last.get("regime", "?")
        delta = last.get("delta", 0.0)
        return (
            f"[ELARA] {compressed}/{total} turns compressed "
            f"(lock_frac={lock_frac:.2f})  "
            f"last: regime={regime} δ={delta:+.4f}"
        )
    except Exception:
        return ""


def main() -> None:
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        data = {}

    prompt: str = data.get("prompt", "")
    session_id: str = data.get("session_id", "unknown")

    # Rough token estimate (4 chars ≈ 1 token for English prose)
    token_estimate = max(1, len(prompt) // 4)

    compression_summary = _load_compression_summary()
    proxy_live = proxy_is_running()

    # Launch tray (starts proxy in-process) if proxy is not running
    _launch_tray()

    # Append to log
    try:
        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt_tokens_est": token_estimate,
            "proxy_live": proxy_live,
            "compression_summary": compression_summary,
        }
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    # Inject context only when proxy has data worth showing
    result: dict = {}
    if compression_summary and proxy_live:
        result["additionalContext"] = compression_summary

    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
