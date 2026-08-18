"""Transport-level configuration and the schema utility."""

from __future__ import annotations

import asyncio

import genai_pyo3 as g
import pytest


class TestProxy:
    def test_invalid_scheme_is_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            g.Client.builder().proxy("http://localhost:3128", scheme="socks")
        assert "all, http, https" in str(excinfo.value)

    def test_invalid_url_is_rejected_at_build(self):
        builder = g.Client.builder().proxy("not a url at all")
        with pytest.raises(ValueError):
            builder.build()

    def test_a_valid_proxy_builds(self, fake_provider):
        # The proxy is not exercised here (no proxy to run against); this
        # pins that configuring one does not break client construction.
        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .proxy("http://localhost:3128", scheme="https")
            .build()
        )
        assert client.adapter_kind is None


class TestTransportToggles:
    def test_gzip_and_nodelay_can_be_disabled(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .gzip(False)
            .tcp_nodelay(False)
            .build()
        )

        assert client.chat(g.ModelIden("openai", "fake-model"), request_obj).text == "pong"


class TestClientLevelEmbedOptions:
    def test_defaults_apply_and_per_call_wins(self, fake_provider):
        client = (
            g.Client.builder()
            .adapter_kind("openai")
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .embed_options(g.EmbedOptions(dimensions=64))
            .build()
        )

        async def run(options=None):
            return await client.aembed("fake-embed", "hello", options)

        asyncio.run(run())
        assert fake_provider.requests[0]["body"]["dimensions"] == 64

        asyncio.run(run(g.EmbedOptions(dimensions=512)))
        assert fake_provider.requests[1]["body"]["dimensions"] == 512

    def test_embed_options_do_not_clobber_the_rest_of_the_config(self, fake_provider):
        # ClientBuilder::with_config replaces the whole ClientConfig, so
        # applying embed options at the wrong point silently discarded the
        # adapter binding and routed calls to the inferred provider instead.
        client = (
            g.Client.builder()
            .adapter_kind("openai")
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .embed_options(g.EmbedOptions(dimensions=64))
            .build()
        )

        assert client.adapter_kind == g.AdapterKind("openai")

        async def run():
            return await client.aembed("fake-embed", "hello")

        asyncio.run(run())
        assert fake_provider.requests, "the call did not reach the configured endpoint"


class TestSanitizeJsonSchema:
    def test_openai_strict_requires_every_property(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
        }

        sanitized = g.sanitize_json_schema(schema, "openai_strict")

        assert sorted(sanitized["required"]) == ["a", "b"]
        assert sanitized["additionalProperties"] is False

    def test_dialects_are_validated(self):
        with pytest.raises(ValueError) as excinfo:
            g.sanitize_json_schema({"type": "object"}, "gemini_strict")
        assert "openai_strict" in str(excinfo.value)

    def test_non_json_input_is_rejected(self):
        with pytest.raises(ValueError):
            g.sanitize_json_schema(object(), "openai_strict")
