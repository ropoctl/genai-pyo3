"""Routing: one client, many providers, chosen per call.

`ServiceTarget` states the endpoint, credential, and model outright, so a
registry of named endpoints is just a dict — no client per provider.
"""

from __future__ import annotations

import genai_pyo3 as g
import pytest


def test_public_names_are_all_importable():
    # __all__ drifting from the extension is a silent packaging break.
    for name in g.__all__:
        assert hasattr(g, name), f"{name} in __all__ but missing from the module"


class TestAdapterKind:
    def test_parses_case_insensitively(self):
        assert g.AdapterKind("openai") == g.AdapterKind("OpenAI")
        assert g.AdapterKind(" Anthropic ").name == "anthropic"

    def test_rejects_unknown_names(self):
        with pytest.raises(ValueError) as excinfo:
            g.AdapterKind("not-a-provider")
        # The message should list the valid options, not just complain.
        assert "openai" in str(excinfo.value)

    def test_exposes_display_name_and_key_env(self):
        kind = g.AdapterKind("openai")
        assert kind.name == "openai"
        assert kind.display_name == "OpenAI"
        assert kind.default_key_env_name == "OPENAI_API_KEY"

    def test_all_covers_the_known_adapters(self):
        names = {kind.name for kind in g.AdapterKind.all()}
        assert {"openai", "anthropic", "gemini", "ollama", "open_router"} <= names

    def test_infers_adapter_from_model_name(self):
        assert g.AdapterKind.from_model("gpt-4o").name == "openai"
        assert g.AdapterKind.from_model("claude-sonnet-4-6").name == "anthropic"

    def test_is_hashable_and_comparable(self):
        assert len({g.AdapterKind("openai"), g.AdapterKind("openai")}) == 1
        assert g.AdapterKind("openai") != g.AdapterKind("gemini")


class TestModelIden:
    def test_accepts_a_string_or_an_adapter_kind(self):
        from_str = g.ModelIden("openai", "gpt-4o")
        from_kind = g.ModelIden(g.AdapterKind("openai"), "gpt-4o")
        assert from_str == from_kind
        assert from_str.model_name == "gpt-4o"
        assert from_str.adapter_kind.name == "openai"

    def test_is_hashable_for_use_as_a_dict_key(self):
        registry = {g.ModelIden("openai", "gpt-4o"): "fast"}
        assert registry[g.ModelIden("openai", "gpt-4o")] == "fast"


class TestAuthData:
    def test_kinds_round_trip(self):
        assert g.AuthData.from_env("OPENAI_API_KEY").kind == "from_env"
        assert g.AuthData.from_env("OPENAI_API_KEY").env_name == "OPENAI_API_KEY"
        assert g.AuthData.key("sk-secret").kind == "key"
        assert g.AuthData.none().kind == "none"
        assert g.AuthData.multi({"a": "b"}).kind == "multi_keys"

        override = g.AuthData.request_override("https://gw/v1/", {"x-key": "v"})
        assert override.kind == "request_override"
        assert override.url == "https://gw/v1/"

    def test_secrets_never_appear_in_repr(self):
        # A ServiceTarget repr lands in logs; the key must not ride along.
        assert "sk-secret" not in repr(g.AuthData.key("sk-secret"))
        assert "sk-secret" not in repr(
            g.ServiceTarget(
                "https://api.example/v1/",
                g.AuthData.key("sk-secret"),
                g.ModelIden("openai", "gpt-4o"),
            )
        )

    def test_empty_override_url_is_rejected(self):
        with pytest.raises(ValueError):
            g.AuthData.request_override("   ", {})


class TestServiceTarget:
    def test_accepts_a_string_or_an_endpoint(self):
        model = g.ModelIden("openai", "gpt-4o")
        auth = g.AuthData.none()
        from_str = g.ServiceTarget("https://api.example/v1/", auth, model)
        from_obj = g.ServiceTarget(g.Endpoint("https://api.example/v1/"), auth, model)
        assert from_str.endpoint == from_obj.endpoint

    def test_exposes_its_parts(self):
        target = g.ServiceTarget(
            "https://api.example/v1/",
            g.AuthData.from_env("K"),
            g.ModelIden("openai", "gpt-4o"),
        )
        assert target.endpoint.base_url == "https://api.example/v1/"
        assert target.auth.kind == "from_env"
        assert target.model.model_name == "gpt-4o"
        assert target.to_dict()["adapter_kind"] == "openai"

    def test_empty_endpoint_is_rejected(self):
        with pytest.raises(ValueError):
            g.Endpoint("  ")


class TestRoutingPerCall:
    def test_service_target_routes_a_bare_client(self, fake_provider, request_obj):
        # No provider, key, or base URL configured on the client at all —
        # the target carries everything the call needs.
        client = g.Client()
        target = g.ServiceTarget(
            fake_provider.base_url,
            g.AuthData.key("sk-test"),
            g.ModelIden("openai", "fake-model"),
        )

        response = client.chat(target, request_obj)

        assert response.text == "pong"
        assert fake_provider.requests[0]["body"]["model"] == "fake-model"

    def test_one_client_reaches_several_targets(self, fake_provider, request_obj):
        client = g.Client()
        registry = {
            name: g.ServiceTarget(
                fake_provider.base_url,
                g.AuthData.key("sk-test"),
                g.ModelIden("openai", name),
            )
            for name in ("flash", "pro")
        }

        for name in registry:
            client.chat(registry[name], request_obj)

        sent = [req["body"]["model"] for req in fake_provider.requests]
        assert sent == ["flash", "pro"]

    def test_model_iden_still_resolves_auth_from_the_client(
        self, fake_provider, request_obj
    ):
        # ModelIden pins the adapter but leaves endpoint/auth to the client.
        client = g.Client.with_api_key_and_base_url(
            "openai", "sk-test", fake_provider.base_url
        )

        response = client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert response.text == "pong"

    def test_plain_string_model_still_works(self, client, request_obj):
        assert client.chat("fake-model", request_obj).text == "pong"

    def test_async_path_accepts_a_target(self, fake_provider, request_obj):
        import asyncio

        client = g.Client()
        target = g.ServiceTarget(
            fake_provider.base_url,
            g.AuthData.key("sk-test"),
            g.ModelIden("openai", "fake-model"),
        )

        async def run():
            return await client.achat(target, request_obj)

        assert asyncio.run(run()).text == "pong"

    def test_unusable_model_argument_is_rejected(self, client, request_obj):
        with pytest.raises(TypeError):
            client.chat(object(), request_obj)
