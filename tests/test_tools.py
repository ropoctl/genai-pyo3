"""Tool definitions and mixed text/binary content."""

from __future__ import annotations

import base64
import json

import genai_pyo3 as g
import pytest


def sent_tools(fake_provider):
    return fake_provider.requests[0]["body"]["tools"]


class TestToolDefinition:
    def test_strict_reaches_the_wire(self, client, fake_provider):
        request = g.ChatRequest(
            messages=[g.ChatMessage("user", "hi")],
            tools=[g.Tool("f", "does f", {"type": "object"}, strict=True)],
        )

        client.chat("fake-model", request)

        assert sent_tools(fake_provider)[0]["function"]["strict"] is True

    def test_cache_control_is_validated_at_construction(self):
        assert g.Tool("f", cache_control="ephemeral_1h").cache_control == "ephemeral_1h"
        with pytest.raises(ValueError):
            g.Tool("f", cache_control="forever")

    def test_custom_config_round_trips(self):
        tool = g.Tool("f", config={"vendor_flag": 1})
        assert tool.config == {"vendor_flag": 1}
        assert tool.is_builtin is False

    def test_eager_input_streaming_is_settable(self):
        assert g.Tool("f", eager_input_streaming=True).eager_input_streaming is True


class TestWebSearch:
    def test_builds_the_builtin_tool(self):
        tool = g.Tool.web_search(max_uses=3, allowed_domains=["example.com"])
        assert tool.is_builtin is True
        assert tool.name == "WebSearch"
        assert tool.config["max_uses"] == 3
        assert tool.config["allowed_domains"] == ["example.com"]

    def test_allowed_and_blocked_are_mutually_exclusive(self):
        # Providers reject the combination; catching it here beats a 400.
        with pytest.raises(ValueError) as excinfo:
            g.Tool.web_search(allowed_domains=["a.com"], blocked_domains=["b.com"])
        assert "not both" in str(excinfo.value)

    def test_reaches_the_wire_as_a_builtin(self, fake_provider):
        # The OpenAI Responses adapter is the one that emits a built-in
        # web-search tool rather than a function tool.
        client = (
            g.Client.builder()
            .provider(
                "openai_resp",
                endpoint=fake_provider.base_url,
                auth=g.AuthData.key("sk-test"),
            )
            .build()
        )
        request = g.ChatRequest(
            messages=[g.ChatMessage("user", "hi")],
            tools=[g.Tool.web_search(max_uses=2)],
        )

        # The fake returns a chat-completions body the Responses adapter
        # cannot parse; the request itself is what this asserts on.
        with pytest.raises(g.GenaiError):
            client.chat(g.ModelIden("openai_resp", "fake-model"), request)

        tools = fake_provider.requests[0]["body"]["tools"]
        assert any(tool.get("type") == "web_search" for tool in tools), tools


class TestBinary:
    def test_from_base64(self):
        binary = g.Binary.from_base64("image/png", "aGVsbG8=", "hello.png")
        assert binary.content_type == "image/png"
        assert binary.name == "hello.png"
        assert binary.source_kind == "base64"
        assert binary.is_image is True
        assert binary.is_pdf is False

    def test_from_url(self):
        binary = g.Binary.from_url("application/pdf", "https://example.com/a.pdf")
        assert binary.source_kind == "url"
        assert binary.is_pdf is True

    def test_from_path_reads_and_encodes(self, tmp_path):
        path = tmp_path / "pixel.png"
        payload = b"\x89PNG\r\n\x1a\n-not-really-a-png"
        path.write_bytes(payload)

        binary = g.Binary.from_path(str(path))

        assert binary.content_type == "image/png"
        assert binary.name == "pixel.png"
        assert binary.source_kind == "base64"
        assert binary.to_dict()["binary"]["source"]["Base64"] == base64.b64encode(
            payload
        ).decode()

    def test_missing_file_is_a_value_error(self, tmp_path):
        with pytest.raises(ValueError):
            g.Binary.from_path(str(tmp_path / "nope.png"))


class TestMixedContent:
    def test_text_and_image_in_one_message(self, client, fake_provider):
        message = g.ChatMessage(
            "user",
            content_parts=[
                "what is in this image?",
                g.Binary.from_base64("image/png", "aGVsbG8="),
            ],
        )

        client.chat("fake-model", g.ChatRequest(messages=[message]))

        content = fake_provider.requests[0]["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "what is in this image?"
        assert content[1]["type"] == "image_url"
        assert "aGVsbG8=" in json.dumps(content[1])

    def test_explicit_part_dicts_are_accepted(self, client, fake_provider):
        message = g.ChatMessage("user", content_parts=[{"text": "hello"}])

        client.chat("fake-model", g.ChatRequest(messages=[message]))

        # A text-only message flattens to a bare string on the OpenAI wire;
        # the multi-part array only appears once there is something to mix.
        content = fake_provider.requests[0]["body"]["messages"][0]["content"]
        assert content == "hello"

    def test_content_parts_round_trip_through_the_getter(self):
        message = g.ChatMessage("user", content_parts=["hi"])
        assert message.content_parts == [{"text": "hi"}]
        assert message.to_dict()["content_parts"] == [{"text": "hi"}]

    def test_plain_text_messages_are_unchanged(self, client, fake_provider):
        client.chat(
            "fake-model", g.ChatRequest(messages=[g.ChatMessage("user", "ping")])
        )
        assert fake_provider.requests[0]["body"]["messages"][0]["content"] == "ping"
        assert g.ChatMessage("user", "ping").content_parts is None

    def test_a_message_needs_some_content(self):
        with pytest.raises(ValueError) as excinfo:
            g.ChatMessage("user")
        assert "content" in str(excinfo.value)

    def test_unusable_part_is_rejected_with_its_index(self):
        with pytest.raises(ValueError) as excinfo:
            g.ChatMessage("user", content_parts=["ok", object()])
        assert "content_parts[1]" in str(excinfo.value)


def test_binary_exposes_every_media_predicate():
    image = g.Binary.from_base64("image/png", "aGVsbG8=")
    audio = g.Binary.from_base64("audio/mpeg", "aGVsbG8=")
    video = g.Binary.from_base64("video/mp4", "aGVsbG8=")
    pdf = g.Binary.from_url("application/pdf", "https://example.com/a.pdf")

    assert (image.is_image, image.is_audio, image.is_video, image.is_pdf) == (
        True,
        False,
        False,
        False,
    )
    assert audio.is_audio is True
    assert video.is_video is True
    assert pdf.is_pdf is True
    assert image.size > 0
