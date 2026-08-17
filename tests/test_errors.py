"""Provider failures must arrive as typed exceptions carrying the detail
needed to decide what to do next — not as a formatted string."""

from __future__ import annotations

import json

import genai_pyo3 as g
import pytest


def test_hierarchy_is_rooted_in_runtime_error():
    # Callers written against the old flat mapping caught RuntimeError.
    # Breaking that would fail silently at runtime rather than at import.
    assert issubclass(g.GenaiError, RuntimeError)
    for exc in (
        g.ConfigError,
        g.AuthError,
        g.ResolverError,
        g.ApiError,
        g.TransportError,
        g.StreamError,
        g.ResponseError,
        g.UnsupportedError,
    ):
        assert issubclass(exc, g.GenaiError)
    for exc in (g.BadRequestError, g.RateLimitError, g.ServerError):
        assert issubclass(exc, g.ApiError)


@pytest.mark.parametrize(
    ("status", "expected", "retryable"),
    [
        (400, g.BadRequestError, False),
        (401, g.BadRequestError, False),
        (404, g.BadRequestError, False),
        (408, g.BadRequestError, True),
        (429, g.RateLimitError, True),
        (500, g.ServerError, True),
        (503, g.ServerError, True),
    ],
)
def test_status_maps_to_exception_class(
    client, request_obj, fake_provider, status, expected, retryable
):
    fake_provider.respond(status=status, body=json.dumps({"error": "nope"}))

    with pytest.raises(expected) as excinfo:
        client.chat("fake-model", request_obj)

    err = excinfo.value
    assert err.status == status
    assert err.is_retryable is retryable
    assert "nope" in err.body
    assert err.model_name == "fake-model"
    assert err.adapter_kind == "OpenAI"


def test_rate_limit_exposes_retry_after(client, request_obj, fake_provider):
    fake_provider.respond(
        status=429, body="slow down", headers={"retry-after": "12"}
    )

    with pytest.raises(g.RateLimitError) as excinfo:
        client.chat("fake-model", request_obj)

    assert excinfo.value.retry_after == 12.0


def test_retry_after_absent_is_none(client, request_obj, fake_provider):
    fake_provider.respond(status=429, body="slow down")

    with pytest.raises(g.RateLimitError) as excinfo:
        client.chat("fake-model", request_obj)

    assert excinfo.value.retry_after is None


def test_non_json_body_is_a_response_error(client, request_obj, fake_provider):
    # A proxy or captive portal answering 200 with HTML: the call succeeded
    # at the transport level but the payload is not a chat completion.
    fake_provider.respond(body="<html>gateway</html>", content_type="text/html")

    with pytest.raises(g.ResponseError) as excinfo:
        client.chat("fake-model", request_obj)

    assert "gateway" in excinfo.value.body


def test_transport_failure_is_retryable(request_obj):
    # Port 1 has nothing listening, so the connection is refused outright.
    client = g.Client.with_api_key_and_base_url(
        "openai", "sk-test", "http://127.0.0.1:1/v1/"
    )

    with pytest.raises(g.TransportError) as excinfo:
        client.chat("fake-model", request_obj)

    assert excinfo.value.is_retryable is True
    assert excinfo.value.status is None


def test_missing_credentials_raise_auth_error(request_obj, monkeypatch):
    # Without an explicit key rust-genai falls back to the provider's env
    # var, so clear it — otherwise this passes or fails based on the shell
    # the suite happens to run in.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = g.Client.with_base_url("openai", "http://127.0.0.1:1/v1/")

    with pytest.raises(g.AuthError) as excinfo:
        client.chat("fake-model", request_obj)

    assert excinfo.value.is_retryable is False


def test_async_path_raises_the_same_types(client, request_obj, fake_provider):
    import asyncio

    fake_provider.respond(status=500, body="boom")

    async def run():
        with pytest.raises(g.ServerError) as excinfo:
            await client.achat("fake-model", request_obj)
        return excinfo.value

    err = asyncio.run(run())
    assert err.status == 500
    assert err.is_retryable is True


def test_success_still_returns_a_response(client, request_obj):
    response = client.chat("fake-model", request_obj)
    assert response.text == "pong"
    assert response.usage.total_tokens == 4
