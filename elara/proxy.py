"""
elara/proxy.py — Local HTTP proxy for Elara context compression.

Intercepts POST /v1/messages, routes through the Gatekeeper (PROXY-01),
compresses old turns when δ ≤ 0, and forwards to Anthropic.

Usage:
    python elara/proxy.py
    set ANTHROPIC_BASE_URL=http://localhost:8877
    claude

Design notes:
- Single Gatekeeper instance per process (one session at a time).
- tool_use + tool_result pairs are never split (see _find_safe_cut).
- Compression replaces messages[:cut] with a synthetic (user, assistant)
  seed-note pair, then appends messages[cut:] unchanged.
- Streaming (SSE) and non-streaming requests are both supported.
"""

from __future__ import annotations

import json
import logging
import os
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from elara.gatekeeper import Gatekeeper
from elara.metrics import MetricsLogger
from elara.adapters import from_api_stream, PROXY_PARAMS, load_proxy_params

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LISTEN_HOST   = "localhost"
LISTEN_PORT   = 8877
UPSTREAM_BASE = "https://api.anthropic.com"

# Minimum text-turn pairs to keep verbatim (never compressed).
# 2 keeps the last 2 real pairs; the synthetic seed pair provides the 3rd,
# so the gatekeeper always sees W=3 pairs after compression.
MIN_KEEP_PAIRS = 2

METRICS_DB_PATH   = Path(__file__).parent.parent / "elara_proxy_metrics.db"
METRICS_JSON_PATH = Path(__file__).parent.parent / "elara_proxy_metrics.json"
CONFIG_PATH       = Path(__file__).parent.parent / "proxy_active_config.json"
REGISTRY_PATH     = Path(__file__).parent.parent / "calibration_registry.json"
DEFAULT_CONFIG_ID = "PROXY-01"

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _read_active_config_id() -> str:
    """Return the active calibration entry ID from proxy_active_config.json."""
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get(
                "active_id", DEFAULT_CONFIG_ID
            )
    except Exception:
        pass
    return DEFAULT_CONFIG_ID


def _build_gatekeeper(entry_id: str) -> tuple:
    """Load params for entry_id and return (Gatekeeper, entry_id, W, tau_rho)."""
    try:
        params = load_proxy_params(entry_id)
    except Exception as exc:
        logging.warning("[ELARA] config load failed (%s), using fallback: %s", entry_id, exc)
        params = PROXY_PARAMS
        entry_id = DEFAULT_CONFIG_ID
    gk = Gatekeeper(W=params.W, tau_rho=params.tau_lock_rho, tau_lock_dr=params.tau_lock_dr)
    return gk, entry_id, params.W, params.tau_lock_rho


# ---------------------------------------------------------------------------
# Shared session state (single-session proxy)
# ---------------------------------------------------------------------------

_state_lock: threading.Lock = threading.Lock()

# Mutable config state — updated under _state_lock when config file changes
_active_config_id: str
_proxy_W:          int
_proxy_tau_rho:    float
_config_mtime:     float = -1.0

_gatekeeper, _active_config_id, _proxy_W, _proxy_tau_rho = _build_gatekeeper(
    _read_active_config_id()
)
if CONFIG_PATH.exists():
    _config_mtime = CONFIG_PATH.stat().st_mtime

_metrics: MetricsLogger = MetricsLogger(
    db_path=str(METRICS_DB_PATH),
    model="claude-3-5-sonnet-20241022",
    params={"W": _proxy_W, "tau_rho": _proxy_tau_rho, "config": _active_config_id}
)


def _maybe_reload_config() -> None:
    """Rebuild gatekeeper if proxy_active_config.json has been updated.

    Must be called under _state_lock.
    """
    global _gatekeeper, _active_config_id, _proxy_W, _proxy_tau_rho, _config_mtime
    if not CONFIG_PATH.exists():
        return
    mtime = CONFIG_PATH.stat().st_mtime
    if mtime <= _config_mtime:
        return
    new_id = _read_active_config_id()
    if new_id == _active_config_id and mtime == _config_mtime:
        return
    _gatekeeper, _active_config_id, _proxy_W, _proxy_tau_rho = _build_gatekeeper(new_id)
    _config_mtime = mtime
    logging.info(
        "[ELARA] config reloaded → %s  W=%d  tau_lock_dr=%.6f",
        _active_config_id, _proxy_W, _gatekeeper._base_tau_suff,
    )

# ---------------------------------------------------------------------------
# Message utilities
# ---------------------------------------------------------------------------

def _content_text(content) -> str:
    """Return plain-text portion of an Anthropic message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(parts).strip()
    return ""


def _has_tool_results(content) -> bool:
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _has_tool_use(content) -> bool:
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _tool_use_ids(content) -> Set[str]:
    if isinstance(content, list):
        return {
            b["id"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
        }
    return set()


def _tool_result_refs(content) -> Set[str]:
    if isinstance(content, list):
        return {
            b["tool_use_id"]
            for b in content
            if isinstance(b, dict)
            and b.get("type") == "tool_result"
            and "tool_use_id" in b
        }
    return set()

# ---------------------------------------------------------------------------
# Tool-safe compression boundary
# ---------------------------------------------------------------------------

def _find_safe_cut(messages: List[dict], ideal_cut: int) -> int:
    """Return the largest safe cut ≤ ideal_cut where no tool pair is split.

    A cut at index `cut` removes messages[:cut] and keeps messages[cut:].
    The cut is safe when messages[cut] is a user message whose content
    contains no tool_result blocks — i.e., the kept section does not
    reference any tool_use that lives in the removed section.

    Walks left from ideal_cut until it finds such a boundary.
    Returns 0 (no compression) if none found.
    """
    cut = ideal_cut
    while cut > 0:
        msg = messages[cut]
        if msg.get("role") == "user":
            if not _has_tool_results(msg.get("content", [])):
                return cut
        cut -= 1
    return 0


# ---------------------------------------------------------------------------
# Turn extraction for embedding
# ---------------------------------------------------------------------------

def _extract_text_turns(messages: List[dict]) -> List[Tuple[str, str]]:
    """Yield (user_text, asst_text) pairs where both contain plain text.

    Tool-only turns (no visible text on either side) are skipped so they
    don't pollute the cosine-similarity signals.
    """
    turns: List[Tuple[str, str]] = []
    i = 0
    while i < len(messages) - 1:
        a, b = messages[i], messages[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            u = _content_text(a.get("content", ""))
            s = _content_text(b.get("content", ""))
            if u and s:
                turns.append((u, s))
            i += 2
        else:
            i += 1
    return turns


# ---------------------------------------------------------------------------
# Core compression step
# ---------------------------------------------------------------------------

def _compress_messages(
    messages: List[dict],
) -> Tuple[List[dict], bool, dict]:
    """Evaluate gatekeeper; compress messages[] if δ ≤ 0.

    Returns:
        (new_messages, was_compressed, metrics_dict)
    """
    turns = _extract_text_turns(messages)

    # Count message roles for diagnostics
    role_counts = {}
    tool_only_pairs = 0
    for i in range(len(messages) - 1):
        a, b = messages[i], messages[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            u = _content_text(a.get("content", ""))
            s = _content_text(b.get("content", ""))
            if not u or not s:
                tool_only_pairs += 1
        role_counts[a.get("role", "?")] = role_counts.get(a.get("role", "?"), 0) + 1

    logging.info(
        "[ELARA] request: total_msgs=%d  text_turn_pairs=%d  tool_only_skipped=%d  config=%s  W_threshold=%d  meets_W=%s",
        len(messages),
        len(turns),
        tool_only_pairs,
        _active_config_id,
        _proxy_W,
        len(turns) >= _proxy_W,
    )

    if len(turns) < _proxy_W:
        logging.info(
            "[ELARA] early-exit: window_too_small (have %d text pairs, need %d) — gatekeeper NOT called",
            len(turns), _proxy_W,
        )
        return messages, False, {"reason": "window_too_small", "W": _proxy_W}

    try:
        sa, sb = from_api_stream(turns)
    except ImportError:
        logging.warning("[ELARA] sentence-transformers not available — skipping compression")
        return messages, False, {"reason": "missing_sentence_transformers"}
    except Exception as exc:
        logging.warning("[ELARA] from_api_stream error: %s", exc)
        return messages, False, {"reason": f"embedding_error: {exc}"}

    logging.info(
        "[ELARA] calling gatekeeper.evaluate()  sa[:3]=%s  sb[:3]=%s",
        [f"{v:.3f}" for v in sa[:3].tolist()],
        [f"{v:.3f}" for v in sb[:3].tolist()],
    )

    with _state_lock:
        _maybe_reload_config()
        try:
            should_compress, metrics = _gatekeeper.evaluate(
                sa.tolist(), sb.tolist()
            )
            logging.info(
                "[ELARA] gatekeeper result: should_compress=%s  delta=%s  regime=%s",
                should_compress,
                f'{metrics.get("delta", "n/a"):+.4f}' if "delta" in metrics else "n/a",
                metrics.get("regime", "n/a"),
            )
            _metrics.record(metrics, should_compress)
        except Exception as exc:
            logging.warning("[ELARA] gatekeeper error: %s", exc)
            return messages, False, {"reason": f"gatekeeper_error: {exc}"}

    if not should_compress:
        return messages, False, metrics

    # Find safe compression boundary: keep last MIN_KEEP_PAIRS text-turn pairs.
    # MIN_KEEP_PAIRS * 2 because each pair = 2 messages (user + assistant).
    ideal_cut = max(0, len(messages) - MIN_KEEP_PAIRS * 2)
    cut = _find_safe_cut(messages, ideal_cut)

    if cut < 2:
        # Nothing meaningful to remove (boundary too close to start).
        return messages, False, {**metrics, "reason": "no_safe_cut"}

    # Build a brief summary of removed turns for the seed note.
    removed_topics: List[str] = []
    for msg in messages[:cut]:
        if msg.get("role") == "user":
            txt = _content_text(msg.get("content", ""))[:100]
            if txt:
                removed_topics.append(txt)

    topic_str = "; ".join(removed_topics[:4]) or "(no text turns)"
    delta_val = metrics.get("delta", 0.0)
    regime    = metrics.get("regime", "")

    seed_note = (
        f"[ELARA_SEED δ={delta_val:+.4f} regime={regime}] "
        f"Compressed {cut} prior messages. "
        f"Topics: {topic_str}"
    )

    # Synthetic pair replaces the compressed span.
    # Both roles required to maintain the user/assistant alternation.
    synthetic: List[dict] = [
        {"role": "user",      "content": seed_note},
        {"role": "assistant", "content": "[Prior context acknowledged.]"},
    ]

    new_messages = synthetic + messages[cut:]
    return new_messages, True, metrics


# ---------------------------------------------------------------------------
# HTTP proxy handler
# ---------------------------------------------------------------------------

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


class ElaraProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # noqa: D102 — suppress default access log
        pass

    # ------------------------------------------------------------------
    # Request routing
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length)
        logging.info("[ELARA] POST %s  content-length=%d", self.path, length)

        try:
            payload: dict = json.loads(raw_body)
        except json.JSONDecodeError:
            self._error(400, "invalid JSON in request body")
            return

        original_messages = payload.get("messages", [])
        compressed = False
        metrics: dict = {}

        if self.path.startswith("/v1/messages") and original_messages:
            new_messages, compressed, metrics = _compress_messages(original_messages)
            if compressed:
                candidate_body = json.dumps(
                    {**payload, "messages": new_messages}
                ).encode()
                if len(candidate_body) >= len(raw_body):
                    compressed = False
                    logging.info(
                        "[ELARA] compression guard: candidate not smaller (%d >= %d), pass-through",
                        len(candidate_body),
                        len(raw_body),
                    )
                else:
                    saved_bytes = len(raw_body) - len(candidate_body)
                    _metrics.update_last_bytes_saved(saved_bytes)
                    logging.info(
                        "[ELARA] COMPRESS  δ=%+.4f  regime=%s  "
                        "saved=%d bytes (%.1f%%)  turns=%d→%d",
                        metrics.get("delta", 0.0),
                        metrics.get("regime", ""),
                        saved_bytes,
                        100.0 * saved_bytes / len(raw_body),
                        len(original_messages),
                        len(new_messages),
                    )
                    payload = {**payload, "messages": new_messages}
                    raw_body = candidate_body

            if not compressed:
                reason = metrics.get("reason", "")
                logging.info(
                    "[ELARA] pass-thru  δ=%s  %s  turns=%d",
                    f'{metrics.get("delta", 0.0):+.4f}' if "delta" in metrics else "n/a",
                    reason or f'regime={metrics.get("regime", "")}',
                    len(original_messages),
                )

        self._forward(raw_body, payload.get("stream", False))
        _persist_metrics()

    def do_GET(self) -> None:
        self._forward(b"", stream=False)

    # ------------------------------------------------------------------
    # Forwarding
    # ------------------------------------------------------------------

    def _forward(self, body: bytes, stream: bool) -> None:
        """Forward request to Anthropic and pipe response back."""
        url = UPSTREAM_BASE + self.path
        fwd_headers = self._build_forward_headers(len(body))

        try:
            with httpx.Client(timeout=300.0) as client:
                if stream:
                    with client.stream(
                        "POST", url, content=body, headers=fwd_headers
                    ) as resp:
                        self._send_head(resp.status_code, dict(resp.headers), chunked=True)
                        try:
                            for chunk in resp.iter_bytes(chunk_size=4096):
                                if not chunk:
                                    continue
                                self.wfile.write(f"{len(chunk):x}\r\n".encode())
                                self.wfile.write(chunk)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                        except (ConnectionResetError, BrokenPipeError):
                            logging.info("[ELARA] client disconnected mid-stream (normal)")
                            return
                else:
                    method = "POST" if body else "GET"
                    resp = client.request(method, url, content=body, headers=fwd_headers)
                    resp_body = resp.content
                    self._send_head(
                        resp.status_code,
                        dict(resp.headers),
                        chunked=False,
                        content_length=len(resp_body),
                    )
                    self.wfile.write(resp_body)
                    self.wfile.flush()

        except (ConnectionResetError, BrokenPipeError):
            logging.info("[ELARA] client disconnected (normal)")
        except httpx.TimeoutException as exc:
            logging.error("[ELARA] upstream timeout: %s", exc)
            self._error(504, "upstream timeout")
        except Exception as exc:
            logging.error("[ELARA] upstream error: %s", exc)
            self._error(502, f"upstream error: {exc}")

    def _build_forward_headers(self, body_len: int) -> dict:
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in _HOP_BY_HOP
            and k.lower() not in {"host", "content-length"}
        }
        headers["content-length"] = str(body_len)
        # Request uncompressed responses so we don't have to re-encode chunks.
        headers["accept-encoding"] = "identity"
        return headers

    def _send_head(
        self,
        status: int,
        upstream_headers: dict,
        chunked: bool,
        content_length: Optional[int] = None,
    ) -> None:
        self.send_response(status)

        skip = _HOP_BY_HOP | {"content-length", "content-encoding"}
        for k, v in upstream_headers.items():
            if k.lower() not in skip:
                self.send_header(k, v)

        if chunked:
            self.send_header("transfer-encoding", "chunked")
        elif content_length is not None:
            self.send_header("content-length", str(content_length))

        self.end_headers()

    def _error(self, status: int, msg: str) -> None:
        body = json.dumps({"error": {"type": "proxy_error", "message": msg}}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Metrics persistence
# ---------------------------------------------------------------------------

def _persist_metrics() -> None:
    try:
        _metrics.save(str(METRICS_JSON_PATH))
    except Exception as exc:
        logging.warning("[ELARA] metrics save failed: %s", exc)


# ---------------------------------------------------------------------------
# Threaded server
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle each request in a daemon thread."""
    daemon_threads = True


# ---------------------------------------------------------------------------
# Programmatic start/stop (used by ElaraClient)
# ---------------------------------------------------------------------------

def start_server(
    host: str = LISTEN_HOST, port: int = LISTEN_PORT
) -> _ThreadedHTTPServer:
    """Start the proxy in a background daemon thread; return the server.

    Caller is responsible for calling server.shutdown() then server.server_close()
    when done.
    """
    server = _ThreadedHTTPServer((host, port), ElaraProxyHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="elara-proxy")
    t.start()
    return server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    server = _ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), ElaraProxyHandler)
    logging.info("Elara proxy  http://%s:%d  →  %s", LISTEN_HOST, LISTEN_PORT, UPSTREAM_BASE)
    logging.info("  set ANTHROPIC_BASE_URL=http://%s:%d", LISTEN_HOST, LISTEN_PORT)
    logging.info("  database    %s", METRICS_DB_PATH)
    logging.info("  export      %s", METRICS_JSON_PATH)
    logging.info("  config      %s  (W=%d  tau_rho=%.2f  tau_lock_dr=%.6f  min_keep_pairs=%d)",
                 _active_config_id, _proxy_W, _proxy_tau_rho,
                 _gatekeeper._base_tau_suff, MIN_KEEP_PAIRS)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
        logging.info(_metrics.report())
        _persist_metrics()
        _metrics.close()


if __name__ == "__main__":
    main()
