"""Embeddings."""

from __future__ import annotations

import asyncio
import json

import genai_pyo3 as g
import pytest


@pytest.fixture
def embed_client(fake_provider):
    return (
        g.Client.builder()
        .adapter_kind("openai")
        .provider(
            "openai",
            endpoint=fake_provider.base_url,
            auth=g.AuthData.key("sk-test"),
        )
        .build()
    )


def test_single_input(embed_client, fake_provider):
    async def run():
        return await embed_client.aembed("fake-embed", "hello")

    response = asyncio.run(run())

    assert response.vector == pytest.approx([0.1, 0.2, 0.3])
    assert len(response.embeddings[0]) == 3
    assert response.embeddings[0].dimensions == 3
    assert fake_provider.requests[0]["body"]["input"] == "hello"


def test_batch_input(embed_client, fake_provider):
    async def run():
        return await embed_client.aembed_batch("fake-embed", ["a", "b"])

    response = asyncio.run(run())

    assert len(response) == 2
    assert response.vectors == [
        pytest.approx([0.1, 0.2, 0.3]),
        pytest.approx([0.4, 0.5, 0.6]),
    ]
    assert fake_provider.requests[0]["body"]["input"] == ["a", "b"]


def test_vectors_are_sorted_by_input_index(embed_client, fake_provider):
    # Providers do not promise response order, so the binding must not
    # either — vectors sorts by the index each embedding carries.
    fake_provider.embeddings_reply = type(fake_provider.reply)(
        body=json.dumps(
            {
                "object": "list",
                "model": "fake-embed",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.4]},
                    {"object": "embedding", "index": 0, "embedding": [0.1]},
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        )
    )

    async def run():
        return await embed_client.aembed_batch("fake-embed", ["a", "b"])

    response = asyncio.run(run())

    assert response.vectors == [pytest.approx([0.1]), pytest.approx([0.4])]


def test_usage_is_reported(embed_client):
    async def run():
        return await embed_client.aembed("fake-embed", "hello")

    assert asyncio.run(run()).usage.prompt_tokens == 5


def test_options_reach_the_wire(embed_client, fake_provider):
    async def run():
        return await embed_client.aembed(
            "fake-embed",
            "hello",
            g.EmbedOptions(dimensions=256, encoding_format="float", user="u-1"),
        )

    asyncio.run(run())
    body = fake_provider.requests[0]["body"]

    assert body["dimensions"] == 256
    assert body["encoding_format"] == "float"
    assert body["user"] == "u-1"


def test_options_dict_form(embed_client, fake_provider):
    async def run():
        return await embed_client.aembed("fake-embed", "hello", {"dimensions": 128})

    asyncio.run(run())

    assert fake_provider.requests[0]["body"]["dimensions"] == 128


def test_service_target_routes_embeddings_too(fake_provider):
    client = g.Client()
    target = g.ServiceTarget(
        fake_provider.base_url,
        g.AuthData.key("sk-test"),
        g.ModelIden("openai", "fake-embed"),
    )

    async def run():
        return await client.aembed(target, "hello")

    assert asyncio.run(run()).vector == pytest.approx([0.1, 0.2, 0.3])


def test_failures_are_typed(embed_client, fake_provider):
    fake_provider.embeddings_reply = type(fake_provider.reply)(status=429, body="slow")

    async def run():
        return await embed_client.aembed("fake-embed", "hello")

    with pytest.raises(g.RateLimitError) as excinfo:
        asyncio.run(run())

    assert excinfo.value.retryable is True


def test_response_to_dict(embed_client):
    async def run():
        return await embed_client.aembed("fake-embed", "hello")

    as_dict = asyncio.run(run()).to_dict()

    assert as_dict["embeddings"][0]["index"] == 0
    assert as_dict["model_name"] == "fake-embed"
