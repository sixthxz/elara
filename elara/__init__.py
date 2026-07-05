"""
elara — RRG context compression for Anthropic API.

Quick start:

    from elara import ElaraClient
    client = ElaraClient()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": "Hello"}],
    )
    client.close()
"""

from __future__ import annotations

import anthropic

from elara.proxy import start_server, LISTEN_HOST, LISTEN_PORT


class ElaraClient:
    """Drop-in replacement for anthropic.Anthropic() with automatic proxy compression.

    Starts the Elara HTTP proxy on localhost:{port} on init, routes all
    messages.create() calls through it, and stops the proxy on close().

    Usage:
        client = ElaraClient()
        # or: with ElaraClient() as client:
        response = client.messages.create(model=..., max_tokens=..., messages=[...])
        client.close()
    """

    def __init__(
        self,
        host: str = LISTEN_HOST,
        port: int = LISTEN_PORT,
        **anthropic_kwargs,
    ) -> None:
        self._host = host
        self._port = port
        self._server = start_server(host, port)
        # Point the Anthropic client at our local proxy.
        self._client = anthropic.Anthropic(
            base_url=f"http://{host}:{port}",
            **anthropic_kwargs,
        )

    @property
    def messages(self):
        return self._client.messages

    @property
    def models(self):
        return self._client.models

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "ElaraClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ElaraClient(proxy=http://{self._host}:{self._port})"
