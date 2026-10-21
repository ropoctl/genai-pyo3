"""Answering "where would this call actually go?" without spending a request."""

from __future__ import annotations

import asyncio

import genai_pyo3 as g
import pytest


class TestResolveTarget:
    def test_reports_the_configured_endpoint_and_adapter(self, fake_provider):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .build()
        )

        target = client.resolve_service_target(g.ModelIden("openai", "fake-model"))

        assert target.endpoint.base_url == fake_provider.base_url
        assert target.model.adapter_kind.name == "openai"
        assert target.model.model_name == "fake-model"
        # Resolution alone must not talk to the provider.
        assert fake_provider.requests == []

    def test_shows_the_effect_of_a_model_mapper(self, fake_provider):
        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .model_mapper(lambda iden: "mapped-model")
            .build()
        )

        target = client.resolve_service_target("gpt-4o")

        assert target.model.model_name == "mapped-model"

    def test_shows_the_effect_of_a_service_target_resolver(self, fake_provider):
        client = (
            g.Client.builder()
            .service_target_resolver(
                lambda target: g.ServiceTarget(
                    "https://rerouted.example/v1/", target.auth, target.model
                )
            )
            .build()
        )

        target = client.resolve_service_target(g.ModelIden("openai", "gpt-4o"))

        assert target.endpoint.base_url == "https://rerouted.example/v1/"

    def test_async_form(self, fake_provider):
        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .build()
        )

        async def run():
            return await client.aresolve_service_target(g.ModelIden("openai", "fake-model"))

        assert asyncio.run(run()).endpoint.base_url == fake_provider.base_url

    def test_env_credentials_resolve_lazily(self, monkeypatch):
        # Resolution records *where* the key will come from; the env lookup
        # itself happens at request time, so resolving is safe with no key
        # set and still tells you what would be read.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = g.Client()

        target = client.resolve_service_target(g.ModelIden("openai", "gpt-4o"))

        assert target.auth.kind == "from_env"
        assert target.auth.env_name == "OPENAI_API_KEY"

    def test_a_service_target_passes_straight_through(self):
        client = g.Client()
        target = g.ServiceTarget(
            "https://api.example/v1/",
            g.AuthData.key("sk-test"),
            g.ModelIden("openai", "gpt-4o"),
        )

        resolved = client.resolve_service_target(target)

        assert resolved.endpoint.base_url == "https://api.example/v1/"


class TestDefaultModel:
    def test_infers_the_adapter_from_the_name(self):
        client = g.Client()
        assert client.default_model("gpt-4o").adapter_kind.name == "openai"
        assert client.default_model("claude-sonnet-4-6").adapter_kind.name == "anthropic"

    def test_honours_a_namespace_prefix(self):
        client = g.Client()
        assert client.default_model("groq::llama-3").adapter_kind.name == "groq"


class TestListModels:
    def test_returns_what_the_provider_is_serving(self, fake_provider):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .build()
        )

        async def run():
            return await client.aall_model_names("openai")

        assert asyncio.run(run()) == ["model-a", "model-b"]

    def test_sync_form_returns_models_and_guards_running_event_loop(self, fake_provider):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .build()
        )

        assert client.all_model_names("openai") == ["model-a", "model-b"]

        async def run():
            return client.all_model_names("openai")

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(run())
        assert "running async runtime" in str(excinfo.value)

    def test_endpoint_and_auth_can_be_overridden_per_lookup(self, fake_provider):
        client = g.Client()

        async def run():
            return await client.aall_model_names(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )

        assert asyncio.run(run()) == ["model-a", "model-b"]

    def test_provider_failure_is_a_typed_error(self, fake_provider):
        fake_provider.model_list_reply = type(fake_provider.reply)(
            status=401, body="nope"
        )
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .build()
        )

        async def run():
            return await client.aall_model_names("openai")

        with pytest.raises(g.GenaiError):
            asyncio.run(run())


def test_list_models_never_falls_back_to_the_public_api(fake_provider, monkeypatch):
    # Regression: all_model_names takes a ProviderConfig and does not consult
    # the service-target resolver, so a client whose endpoint was configured
    # declaratively used to list models from api.openai.com instead — a real
    # network call from what should be an offline path.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-used")
    client = (
        g.Client.builder()
        .provider(
            "openai", endpoint=fake_provider.base_url, auth=g.AuthData.key("sk-test")
        )
        .build()
    )

    async def run():
        return await client.aall_model_names("openai")

    assert asyncio.run(run()) == ["model-a", "model-b"]
    assert fake_provider.requests, "the lookup did not reach the configured endpoint"
