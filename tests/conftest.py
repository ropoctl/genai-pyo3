"""Shared fixtures: a local fake provider so the suite never touches a network.

The fake speaks enough of the OpenAI chat-completions protocol for
rust-genai's `openai` adapter to drive it, and lets each test dictate the
exact status, headers, and body the provider returns.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

CHAT_COMPLETION_OK: dict[str, Any] = {
    "id": "chatcmpl-fake",
    "object": "chat.completion",
    "model": "fake-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "pong"},
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


@dataclass
class Reply:
    """What the fake provider should return for the next request."""

    status: int = 200
    body: str = json.dumps(CHAT_COMPLETION_OK)
    content_type: str = "application/json"
    headers: dict[str, str] = field(default_factory=dict)


class FakeProvider:
    """A running fake provider whose reply each test can rewrite."""

    def __init__(self) -> None:
        self.reply = Reply()
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        # Trailing slash matters: rust-genai joins the path onto the base
        # URL verbatim, so omitting it silently drops the last segment.
        return f"http://{host}:{port}/v1/"

    def respond(self, **kwargs: Any) -> None:
        self.reply = Reply(**kwargs)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                length = int(self.headers.get("content-length", 0))
                raw = self.rfile.read(length)
                provider.requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": json.loads(raw) if raw else None,
                    }
                )
                reply = provider.reply
                payload = reply.body.encode()
                self.send_response(reply.status)
                self.send_header("content-type", reply.content_type)
                self.send_header("content-length", str(len(payload)))
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        return Handler


@pytest.fixture
def fake_provider():
    provider = FakeProvider()
    try:
        yield provider
    finally:
        provider.shutdown()


@pytest.fixture
def client(fake_provider):
    import genai_pyo3

    return genai_pyo3.Client.with_api_key_and_base_url(
        "openai", "sk-test", fake_provider.base_url
    )


@pytest.fixture
def request_obj():
    import genai_pyo3

    return genai_pyo3.ChatRequest(messages=[genai_pyo3.ChatMessage("user", "ping")])
