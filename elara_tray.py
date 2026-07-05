"""
elara_tray.py — System tray entry point for Elara proxy.

Starts the Elara proxy in-process and shows a system tray icon.

Icon colors:
  green  = proxy running
  red    = proxy stopped
  yellow dot overlay = resonance_lock_active

Tooltip: "<config>  cycles=N  lock=X.XXX"

Menu:
  Config  →  <radio items from calibration_registry.json>
  New Session
  Stop Proxy / Start Proxy
  ────
  Quit

Usage:
  python elara_tray.py

The hook (hooks/user_prompt_hook.py) launches this automatically when
the proxy is not running.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

import elara.proxy as _proxy

LISTEN_PORT = _proxy.LISTEN_PORT  # 8877

# ---------------------------------------------------------------------------
# Icon creation
# ---------------------------------------------------------------------------

_ICON_SIZE = 64


def _make_icon(running: bool, resonance_lock: bool = False) -> Image.Image:
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (0, 180, 0, 255) if running else (200, 0, 0, 255)
    m = 4
    draw.ellipse([m, m, _ICON_SIZE - m, _ICON_SIZE - m], fill=color)
    if resonance_lock:
        r = _ICON_SIZE // 8
        x0 = _ICON_SIZE - m - r * 2
        draw.ellipse([x0, m, x0 + r * 2, m + r * 2], fill=(255, 200, 0, 255))
    return img


# ---------------------------------------------------------------------------
# Proxy lifecycle
# ---------------------------------------------------------------------------

_server = None
_server_lock = threading.Lock()
_proxy_running = False
_stop_event = threading.Event()


def _port_in_use() -> bool:
    try:
        with socket.create_connection(("localhost", LISTEN_PORT), timeout=0.3):
            return True
    except OSError:
        return False


def _start_proxy() -> bool:
    global _server, _proxy_running
    with _server_lock:
        if _server is not None:
            return True
        try:
            _server = _proxy.start_server()
            _proxy_running = True
            return True
        except OSError as exc:
            print(f"[ELARA-TRAY] Failed to start proxy on port {LISTEN_PORT}: {exc}", file=sys.stderr)
            _proxy_running = False
            return False


def _stop_proxy() -> None:
    global _server, _proxy_running
    with _server_lock:
        if _server is None:
            return
        _server.shutdown()
        _server.server_close()
        _server = None
    _proxy_running = False


# ---------------------------------------------------------------------------
# Registry / config helpers
# ---------------------------------------------------------------------------

_REGISTRY_PATH = _ROOT / "calibration_registry.json"
_CONFIG_PATH   = _ROOT / "proxy_active_config.json"


def _load_entries() -> list:
    try:
        return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8")).get("entries", [])
    except Exception:
        return []


def _get_active_id() -> str:
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8")).get("active_id", "PROXY-01")
    except Exception:
        return "PROXY-01"


def _set_active_id(entry_id: str) -> None:
    _CONFIG_PATH.write_text(json.dumps({"active_id": entry_id}), encoding="utf-8")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def _make_config_action(eid: str):
    def _action(icon, item):
        _set_active_id(eid)
    return _action


def _build_config_items() -> list:
    items = []
    for e in _load_entries():
        eid = e["id"]
        label = e.get("label", eid)
        if len(label) > 35:
            label = eid
        items.append(
            pystray.MenuItem(
                label,
                _make_config_action(eid),
                checked=lambda item, eid=eid: _get_active_id() == eid,
                radio=True,
            )
        )
    return items


def _new_session() -> None:
    with _proxy._state_lock:
        _proxy._gatekeeper.reset()


def _toggle_proxy(icon, item) -> None:
    if _proxy_running:
        _stop_proxy()
    else:
        _start_proxy()
    icon.icon = _make_icon(_proxy_running, _proxy._gatekeeper._resonance_lock_active)


def _quit(icon, item) -> None:
    _stop_event.set()
    _stop_proxy()
    icon.stop()


def _make_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem("Config", pystray.Menu(*_build_config_items())),
        pystray.MenuItem("New Session", lambda icon, item: _new_session()),
        pystray.MenuItem(
            lambda item: "Stop Proxy" if _proxy_running else "Start Proxy",
            _toggle_proxy,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", _quit),
    )


# ---------------------------------------------------------------------------
# Background update loop
# ---------------------------------------------------------------------------

def _update_loop(icon: pystray.Icon) -> None:
    while not _stop_event.wait(1.0):
        try:
            summary = _proxy._metrics.summary()
            config  = _proxy._active_config_id
            cycles  = summary["cycle_count"]
            lock    = summary["lock_frac"]
            resonance = _proxy._gatekeeper._resonance_lock_active
            icon.title = f"{config}  cycles={cycles}  lock={lock:.3f}"
            icon.icon  = _make_icon(_proxy_running, resonance)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _stop_event
    _stop_event = threading.Event()

    if _port_in_use():
        print(
            f"[ELARA-TRAY] Port {LISTEN_PORT} already in use — "
            "proxy already running. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _start_proxy():
        sys.exit(1)

    icon = pystray.Icon(
        name="elara",
        icon=_make_icon(True, False),
        title=f"{_proxy._active_config_id}  cycles=0  lock=0.000",
        menu=_make_menu(),
    )

    def setup(icon: pystray.Icon) -> None:
        icon.visible = True
        threading.Thread(
            target=_update_loop, args=(icon,), daemon=True, name="elara-tray-updater"
        ).start()

    icon.run(setup=setup)


if __name__ == "__main__":
    main()
