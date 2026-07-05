"""
Elara widget server — port 8878.

Reads elara_proxy_metrics.db directly (no dependency on proxy).
Serves:
  GET /         → widget.html
  GET /metrics  → JSON snapshot (cycles, lock_frac, last record)

Run standalone:
    python widget_server.py

Or import and embed:
    from widget_server import start_widget_server
    server = start_widget_server()   # daemon thread, returns HTTPServer
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_ROOT         = Path(__file__).parent
DB_PATH       = _ROOT / "elara_proxy_metrics.db"
WIDGET_HTML   = _ROOT / "widget.html"
REGISTRY_PATH = _ROOT / "calibration_registry.json"
CONFIG_PATH   = _ROOT / "proxy_active_config.json"

GATE2_THRESHOLD = 0.05   # mirror of gatekeeper constant — no import to avoid heavy deps
WIDGET_HOST     = "localhost"
WIDGET_PORT     = 8878


def _get_configs() -> list:
    """Return all entries from calibration_registry.json (compact form for UI)."""
    if not REGISTRY_PATH.exists():
        return []
    try:
        full = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return [
            {
                "id":                 e["id"],
                "label":              e.get("label", e["id"]),
                "W":                  e.get("W", 3),
                "tau_lock_dr":        e.get("tau_lock_dr"),
                "tau_lock_rho":       e.get("tau_lock_rho"),
                "lock_frac_coherent": e.get("lock_frac_coherent"),
                "calibrated_on":      e.get("calibrated_on"),
            }
            for e in full.get("entries", [])
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


def _get_active_config_id() -> str:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("active_id", "PROXY-01")
        except Exception:
            pass
    return "PROXY-01"


def _set_active_config(entry_id: str) -> bool:
    """Write active_id to proxy_active_config.json. Returns False if ID not in registry."""
    entries = _get_configs()
    if not any(e.get("id") == entry_id for e in entries):
        return False
    CONFIG_PATH.write_text(
        json.dumps({"active_id": entry_id}, indent=2), encoding="utf-8"
    )
    return True


def _get_metrics() -> dict:
    if not DB_PATH.exists():
        return {"cycles": 0, "compressed": 0, "lock_frac": 0.0, "last": None}
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS total, SUM(compressed) AS comp FROM records")
        row  = cur.fetchone()
        total = row["total"] or 0
        comp  = int(row["comp"] or 0)

        cur.execute("SELECT * FROM records ORDER BY id DESC LIMIT 1")
        last_row = cur.fetchone()
        last = None
        if last_row:
            last = dict(last_row)
            series_raw = last.get("d_rho_series", "[]")
            series = json.loads(series_raw) if isinstance(series_raw, str) else series_raw
            last["d_rho_series"] = series
            gate2_max = max(series) if series else 0.0
            last["resonance_lock"] = bool(last.get("compressed", 0)) and gate2_max >= GATE2_THRESHOLD

        conn.close()
        return {
            "cycles":     total,
            "compressed": comp,
            "lock_frac":  comp / total if total > 0 else 0.0,
            "last":       last,
        }
    except Exception as exc:
        return {"cycles": 0, "compressed": 0, "lock_frac": 0.0, "last": None, "error": str(exc)}


class _WidgetHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args): pass  # silence access log

    def do_GET(self) -> None:
        if self.path == "/metrics":
            self._json(_get_metrics())

        elif self.path == "/api/configs":
            self._json({"configs": _get_configs(), "active_id": _get_active_config_id()})

        elif self.path == "/api/config":
            active = _get_active_config_id()
            configs = _get_configs()
            entry = next((c for c in configs if c.get("id") == active), None)
            self._json({"active_id": active, "params": entry})

        elif self.path in ("/", "/widget.html"):
            if WIDGET_HTML.exists():
                body = WIDGET_HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._not_found()
        else:
            self._not_found()

    def do_POST(self) -> None:
        if self.path == "/api/config":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
                entry_id = payload.get("active_id", "")
                if not entry_id:
                    self._json({"ok": False, "error": "missing active_id"}, status=400)
                    return
                if _set_active_config(entry_id):
                    self._json({"ok": True, "active_id": entry_id})
                else:
                    self._json({"ok": False, "error": f"unknown id: {entry_id}"}, status=400)
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, status=400)
        else:
            self._not_found()

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


def start_widget_server(host: str = WIDGET_HOST, port: int = WIDGET_PORT) -> HTTPServer:
    """Start the widget server in a daemon thread; return the HTTPServer instance."""
    server = HTTPServer((host, port), _WidgetHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="elara-widget")
    t.start()
    return server


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.info("Elara widget server  http://%s:%d", WIDGET_HOST, WIDGET_PORT)
    logging.info("  database  %s", DB_PATH)
    logging.info("  widget    %s", WIDGET_HTML)
    server = HTTPServer((WIDGET_HOST, WIDGET_PORT), _WidgetHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
