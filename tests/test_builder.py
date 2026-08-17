"""ClientBuilder: one client, several providers, configured up front."""

from __future__ import annotations

import genai_pyo3 as g
import pytest


def test_bare_client_is_unbound():
    client = g.Client()
    assert client.adapter_kind is None
    assert repr(client) == "Client(unbound)"


def test_adapter_kind_binds_the_client():
    client = g.Client.builder().adapter_kind("openai").build()
    assert client.adapter_kind == g.AdapterKind("openai")
    assert repr(client) == 'Client(adapter_kind="openai")'


def test_builder_is_chainable_and_reusable(fake_provider):
    builder = (
        g.Client.builder()
        .adapter_kind("openai")
        .provider(
            "openai",
            endpoint=fake_provider.base_url,
            auth=g.AuthData.key("sk-test"),
        )
    )
    # Building must not consume the builder.
    first, second = builder.build(), builder.build()
    assert first.adapter_kind == second.adapter_kind == g.AdapterKind("openai")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        g.Client.builder().provider("not-a-provider")


def test_provider_calls_merge_rather_than_replace(fake_provider, request_obj):
    # Endpoint and auth arrive in separate calls; both must survive.
    client = (
        g.Client.builder()
        .provider("openai", endpoint=fake_provider.base_url)
        .provider("openai", auth=g.AuthData.key("sk-test"))
        .build()
    )

    assert client.chat(g.ModelIden("openai", "fake-model"), request_obj).text == "pong"
    assert fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-test"


def test_several_providers_on_one_client(fake_provider, request_obj):
    # Two OpenAI-protocol adapters pointed at the same fake server, each
    # with its own credential — proving the per-adapter dispatch really
    # dispatches rather than applying one resolver to everything.
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-openai")
        )
        .provider(
            "open_router",
            endpoint=fake_provider.base_url,
            auth=g.AuthData.key("sk-router"),
        )
        .build()
    )

    client.chat(g.ModelIden("openai", "fake-model"), request_obj)
    client.chat(g.ModelIden("open_router", "fake-model"), request_obj)

    sent = [req["headers"].get("authorization") for req in fake_provider.requests]
    assert sent == ["Bearer sk-openai", "Bearer sk-router"]


def test_unconfigured_adapter_falls_through(request_obj, monkeypatch):
    # The client knows about openai only; a gemini call must not silently
    # borrow openai's credential.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = (
        g.Client.builder()
        .provider("openai", auth=g.AuthData.key("sk-openai"))
        .build()
    )

    with pytest.raises(g.AuthError):
        client.chat(g.ModelIden("gemini", "gemini-2.0-flash"), request_obj)


def test_default_headers_reach_the_provider(fake_provider, request_obj):
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-test")
        )
        .default_headers({"x-trace-id": "abc123"})
        .build()
    )

    client.chat(g.ModelIden("openai", "fake-model"), request_obj)

    assert fake_provider.requests[0]["headers"]["x-trace-id"] == "abc123"


def test_client_level_chat_options_apply(fake_provider, request_obj):
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-test")
        )
        .chat_options(g.ChatOptions(temperature=0.25))
        .build()
    )

    client.chat(g.ModelIden("openai", "fake-model"), request_obj)

    assert fake_provider.requests[0]["body"]["temperature"] == 0.25


def test_per_call_options_beat_client_defaults(fake_provider, request_obj):
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-test")
        )
        .chat_options(g.ChatOptions(temperature=0.25))
        .build()
    )

    client.chat(
        g.ModelIden("openai", "fake-model"), request_obj, g.ChatOptions(temperature=0.9)
    )

    assert fake_provider.requests[0]["body"]["temperature"] == 0.9


def test_timeouts_replace_the_whole_configuration(fake_provider, request_obj):
    # Setting only a read timeout must leave the connect default in place
    # rather than clearing it.
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-test")
        )
        .timeouts(read_seconds=30.0)
        .build()
    )

    assert client.chat(g.ModelIden("openai", "fake-model"), request_obj).text == "pong"


def test_builder_repr_lists_configured_providers():
    builder = (
        g.Client.builder()
        .adapter_kind("openai")
        .provider("openai")
        .provider("anthropic")
    )
    assert repr(builder) == (
        'ClientBuilder(adapter_kind=Some("openai"), providers=[anthropic, openai])'
    )
