"""Response detail: why generation stopped, and the raw body behind it."""

from __future__ import annotations

import asyncio
import json

import genai_pyo3 as g
import pytest
from conftest import CHAT_COMPLETION_OK, sse


def completion_with(finish_reason: str) -> str:
    body = json.loads(json.dumps(CHAT_COMPLETION_OK))
    body["choices"][0]["finish_reason"] = finish_reason
    return json.dumps(body)


@pytest.mark.parametrize(
    ("provider_reason", "normalised"),
    [
        ("stop", "completed"),
        ("end_turn", "completed"),
        ("length", "max_tokens"),
        ("max_tokens", "max_tokens"),
        ("tool_calls", "tool_call"),
        ("content_filter", "content_filter"),
        ("stop_sequence", "stop_sequence"),
        ("something_new", "other"),
    ],
)
def test_stop_reason_is_normalised(
    client, request_obj, fake_provider, provider_reason, normalised
):
    fake_provider.respond(body=completion_with(provider_reason))

    response = client.chat("fake-model", request_obj)

    assert response.stop_reason == normalised
    # The provider's own wording survives — normalising to "other" without
    # it would leave nothing to debug against.
    assert response.stop_reason_raw == provider_reason


def test_truncation_is_distinguishable_from_a_short_answer(
    client, request_obj, fake_provider
):
    fake_provider.respond(body=completion_with("length"))

    response = client.chat("fake-model", request_obj)

    assert response.stop_reason == "max_tokens"
    assert response.text == "pong"


def test_raw_body_is_absent_unless_requested(client, request_obj):
    assert client.chat("fake-model", request_obj).captured_raw_body is None


def test_raw_body_is_captured_on_request(client, request_obj):
    response = client.chat(
        "fake-model", request_obj, g.ChatOptions(capture_raw_body=True)
    )

    assert response.captured_raw_body["id"] == "chatcmpl-fake"


def test_to_dict_carries_the_new_fields(client, request_obj, fake_provider):
    fake_provider.respond(body=completion_with("length"))

    as_dict = client.chat("fake-model", request_obj).to_dict()

    assert as_dict["stop_reason"] == "max_tokens"
    assert as_dict["stop_reason_raw"] == "length"
    assert as_dict["captured_raw_body"] is None


class TestStreaming:
    def test_events_arrive_in_order(self, client, request_obj, fake_provider):
        fake_provider.respond(
            body=sse([{"content": "po"}, {"content": "ng"}]),
            content_type="text/event-stream",
        )

        async def run():
            stream = await client.astream_chat(
                "fake-model",
                request_obj,
                g.ChatOptions(capture_content=True),
            )
            return [event async for event in stream]

        events = asyncio.run(run())
        kinds = [event.kind for event in events]

        assert kinds[0] == "start"
        assert kinds[-1] == "end"
        assert "".join(e.content for e in events if e.kind == "chunk") == "pong"

    def test_stream_end_carries_the_stop_reason(
        self, client, request_obj, fake_provider
    ):
        fake_provider.respond(
            body=sse([{"content": "pong"}], finish_reason="length"),
            content_type="text/event-stream",
        )

        async def run():
            stream = await client.astream_chat(
                "fake-model",
                request_obj,
                g.ChatOptions(capture_content=True),
            )
            return [event async for event in stream][-1]

        end = asyncio.run(run()).end

        assert end.captured_stop_reason == "max_tokens"
        assert end.captured_stop_reason_raw == "length"
        assert end.captured_text == "pong"

    def test_collected_stream_reports_truncation_too(
        self, client, request_obj, fake_provider
    ):
        # achat_via_stream returns a ChatResponse, so it must surface the
        # stop reason the same way the non-streaming path does.
        fake_provider.respond(
            body=sse([{"content": "pong"}], finish_reason="length"),
            content_type="text/event-stream",
        )

        async def run():
            return await client.achat_via_stream("fake-model", request_obj)

        response = asyncio.run(run())

        assert response.stop_reason == "max_tokens"
        assert response.text == "pong"
