"""Python callbacks standing in for rust-genai's resolver hooks.

These run on a Tokio worker thread while holding the GIL, which is what
makes the async-callable rejection and the re-entrancy guard matter.
"""

from __future__ import annotations

import asyncio
import threading

import genai_pyo3 as g
import pytest


class TestRegistrationChecks:
    def test_async_callables_are_rejected_up_front(self):
        # A coroutine function would be called, never awaited, and its
        # coroutine object silently treated as the answer.
        async def resolver(model_iden):
            return g.AuthData.key("sk-test")

        with pytest.raises(TypeError) as excinfo:
            g.Client.builder().auth_resolver(resolver)

        assert "async" in str(excinfo.value)

    def test_async_generators_are_rejected(self):
        async def resolver(model_iden):
            yield g.AuthData.key("sk-test")

        with pytest.raises(TypeError):
            g.Client.builder().auth_resolver(resolver)

    def test_non_callables_are_rejected(self):
        with pytest.raises(TypeError) as excinfo:
            g.Client.builder().model_mapper("not a function")
        assert "callable" in str(excinfo.value)

    def test_each_hook_names_itself_in_its_error(self):
        for method in ("auth_resolver", "model_mapper", "service_target_resolver"):
            with pytest.raises(TypeError) as excinfo:
                getattr(g.Client.builder(), method)(42)
            assert method in str(excinfo.value)


class TestAuthResolver:
    def test_receives_the_model_and_supplies_the_key(self, fake_provider, request_obj):
        seen = []

        def resolver(model_iden):
            seen.append((model_iden.adapter_kind.name, model_iden.model_name))
            return g.AuthData.key("sk-from-callback")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )

        client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert seen == [("openai", "fake-model")]
        assert (
            fake_provider.requests[0]["headers"]["authorization"]
            == "Bearer sk-from-callback"
        )

    def test_callback_overrides_the_declared_credential(
        self, fake_provider, request_obj
    ):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-declared"),
            )
            .auth_resolver(lambda iden: g.AuthData.key("sk-callback"))
            .build()
        )

        client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert (
            fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-callback"
        )

    def test_returning_none_falls_back_to_the_declared_credential(
        self, fake_provider, request_obj
    ):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-declared"),
            )
            .auth_resolver(lambda iden: None)
            .build()
        )

        client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert (
            fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-declared"
        )

    def test_exception_surfaces_as_resolver_error(self, fake_provider, request_obj):
        def resolver(model_iden):
            raise ValueError("no key for tenant-7")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )

        with pytest.raises(g.ResolverError) as excinfo:
            client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        message = str(excinfo.value)
        # The type and message survive as text even though the exception
        # object itself does not cross the boundary.
        assert "ValueError" in message
        assert "no key for tenant-7" in message
        assert "auth_resolver" in message

    def test_wrong_return_type_is_reported_clearly(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(lambda iden: "sk-a-bare-string")
            .build()
        )

        with pytest.raises(g.ResolverError) as excinfo:
            client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert "must return an AuthData or None" in str(excinfo.value)


class TestModelMapper:
    def test_a_returned_string_renames_and_keeps_the_adapter(
        self, fake_provider, request_obj
    ):
        aliases = {"flash": "deepseek-ai/DeepSeek-V4-Flash-0731"}
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .model_mapper(lambda iden: aliases.get(iden.model_name, iden.model_name))
            .build()
        )

        client.chat(g.ModelIden("openai", "flash"), request_obj)

        assert (
            fake_provider.requests[0]["body"]["model"]
            == "deepseek-ai/DeepSeek-V4-Flash-0731"
        )

    def test_a_returned_model_iden_can_switch_adapters(
        self, fake_provider, request_obj
    ):
        client = (
            g.Client.builder()
            .provider(
                "open_router",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-router"),
            )
            .model_mapper(lambda iden: g.ModelIden("open_router", iden.model_name))
            .build()
        )

        client.chat("fake-model", request_obj)

        assert fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-router"

    def test_wrong_return_type_is_reported_clearly(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .model_mapper(lambda iden: 42)
            .build()
        )

        with pytest.raises(g.ResolverError) as excinfo:
            client.chat("fake-model", request_obj)

        assert "must return a ModelIden or str" in str(excinfo.value)


class TestServiceTargetResolver:
    def test_sees_the_declarative_endpoint_and_has_the_last_word(
        self, fake_provider, request_obj
    ):
        seen = []

        def resolver(target):
            seen.append(target.endpoint.base_url)
            # Redirect to the real fake server, overriding what was declared.
            return g.ServiceTarget(
                fake_provider.base_url, g.AuthData.key("sk-final"), target.model
            )

        client = (
            g.Client.builder()
            .provider("openai", endpoint="http://127.0.0.1:1/v1/")
            .service_target_resolver(resolver)
            .build()
        )

        response = client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert response.text == "pong"
        assert seen == ["http://127.0.0.1:1/v1/"]
        assert fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-final"

    def test_can_rewrite_the_model(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .service_target_resolver(
                lambda target: g.ServiceTarget(
                    fake_provider.base_url,
                    g.AuthData.key("sk-test"),
                    g.ModelIden("openai", "rewritten-model"),
                )
            )
            .build()
        )

        client.chat("fake-model", request_obj)

        assert fake_provider.requests[0]["body"]["model"] == "rewritten-model"

    def test_wrong_return_type_is_reported_clearly(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .provider(
                "openai",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .service_target_resolver(lambda target: None)
            .build()
        )

        with pytest.raises(g.ResolverError) as excinfo:
            client.chat("fake-model", request_obj)

        assert "must return a ServiceTarget" in str(excinfo.value)


class TestThreadingContract:
    def test_sync_path_runs_the_callback_on_the_calling_thread(
        self, fake_provider, request_obj
    ):
        # Runtime::block_on drives the root future inline, so the blocking
        # entry point calls back on the caller's own thread (after the GIL
        # was released and re-acquired around it).
        threads = []

        def resolver(model_iden):
            threads.append(threading.get_ident())
            return g.AuthData.key("sk-test")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )

        client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert threads == [threading.get_ident()]

    def test_async_path_runs_the_callback_off_the_loop_thread(
        self, fake_provider, request_obj
    ):
        # Here the future is spawned, so the callback lands on a Tokio
        # worker and has to attach to the interpreter itself.
        threads = []

        def resolver(model_iden):
            threads.append(threading.get_ident())
            return g.AuthData.key("sk-test")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )

        async def run():
            return await client.achat(g.ModelIden("openai", "fake-model"), request_obj)

        asyncio.run(run())

        assert threads and threads[0] != threading.get_ident()

    def test_reentering_the_blocking_client_is_refused_not_a_panic(
        self, fake_provider, request_obj
    ):
        # Calling the sync entry point from inside a Tokio worker would
        # panic deep in tokio; it must surface as a clean Python error.
        client_box = {}

        def resolver(model_iden):
            client_box["client"].chat("fake-model", request_obj)
            return g.AuthData.key("sk-test")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )
        client_box["client"] = client

        with pytest.raises(g.ResolverError) as excinfo:
            client.chat(g.ModelIden("openai", "fake-model"), request_obj)

        assert "running async runtime" in str(excinfo.value)

    def test_callbacks_work_on_the_async_path_too(self, fake_provider, request_obj):
        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(lambda iden: g.AuthData.key("sk-async"))
            .build()
        )

        async def run():
            return await client.achat(g.ModelIden("openai", "fake-model"), request_obj)

        assert asyncio.run(run()).text == "pong"
        assert fake_provider.requests[0]["headers"]["authorization"] == "Bearer sk-async"

    def test_concurrent_calls_do_not_deadlock(self, fake_provider, request_obj):
        # Several in-flight requests each attaching to the interpreter from
        # a different worker: the GIL must not be held across the awaits.
        calls = []

        def resolver(model_iden):
            calls.append(model_iden.model_name)
            return g.AuthData.key("sk-test")

        client = (
            g.Client.builder()
            .provider("openai", endpoint=fake_provider.base_url)
            .auth_resolver(resolver)
            .build()
        )

        async def run():
            return await asyncio.gather(
                *(
                    client.achat(g.ModelIden("openai", f"model-{i}"), request_obj)
                    for i in range(8)
                )
            )

        results = asyncio.wait_for(run(), timeout=30)
        responses = asyncio.run(results)

        assert [r.text for r in responses] == ["pong"] * 8
        assert sorted(calls) == sorted(f"model-{i}" for i in range(8))
