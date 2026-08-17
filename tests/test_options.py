"""ChatOptions must be able to express everything rust-genai accepts."""

from __future__ import annotations

import genai_pyo3 as g
import pytest


def send(client, request_obj, options):
    client.chat("fake-model", request_obj, options)


class TestToolChoice:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("auto", "auto"), ("none", "none"), ("required", "required")],
    )
    def test_keywords_reach_the_wire(
        self, client, request_obj, fake_provider, value, expected
    ):
        send(client, request_obj, g.ChatOptions(tool_choice=value))
        assert fake_provider.requests[0]["body"]["tool_choice"] == expected

    def test_any_other_value_names_a_specific_tool(
        self, client, request_obj, fake_provider
    ):
        send(client, request_obj, g.ChatOptions(tool_choice="get_weather"))
        assert fake_provider.requests[0]["body"]["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }


class TestVerbosity:
    def test_reaches_the_wire(self, client, request_obj, fake_provider):
        send(client, request_obj, g.ChatOptions(verbosity="low"))
        assert fake_provider.requests[0]["body"]["verbosity"] == "low"

    def test_typos_fail_at_construction(self):
        # Not a silent no-op: a mistyped option that quietly does nothing
        # is worse than one that refuses to be built.
        with pytest.raises(ValueError) as excinfo:
            g.ChatOptions(verbosity="lowish")
        assert "low, medium, high" in str(excinfo.value)


class TestServiceTier:
    def test_reaches_the_wire(self, client, request_obj, fake_provider):
        send(client, request_obj, g.ChatOptions(service_tier="flex"))
        assert fake_provider.requests[0]["body"]["service_tier"] == "flex"

    def test_typos_fail_at_construction(self):
        with pytest.raises(ValueError):
            g.ChatOptions(service_tier="cheap")


class TestCacheControl:
    def test_accepted_values_construct(self):
        for value in (
            "ephemeral",
            "memory",
            "ephemeral_5m",
            "ephemeral_1h",
            "ephemeral_24h",
        ):
            assert g.ChatOptions(cache_control=value).cache_control == value

    def test_typos_fail_at_construction(self):
        with pytest.raises(ValueError):
            g.ChatOptions(cache_control="forever")


class TestExistingKnobsStillWork:
    def test_core_sampling_options(self, client, request_obj, fake_provider):
        send(
            client,
            request_obj,
            g.ChatOptions(
                temperature=0.3,
                max_tokens=128,
                top_p=0.9,
                stop_sequences=["STOP"],
                seed=7,
            ),
        )
        body = fake_provider.requests[0]["body"]
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 128
        assert body["top_p"] == 0.9
        assert body["stop"] == ["STOP"]
        assert body["seed"] == 7

    def test_reasoning_effort(self, client, request_obj, fake_provider):
        send(client, request_obj, g.ChatOptions(reasoning_effort="high"))
        assert fake_provider.requests[0]["body"]["reasoning_effort"] == "high"

    def test_extra_body_merges_provider_specific_fields(
        self, client, request_obj, fake_provider
    ):
        send(client, request_obj, g.ChatOptions(extra_body={"custom_flag": True}))
        assert fake_provider.requests[0]["body"]["custom_flag"] is True

    def test_dict_form_is_equivalent(self, client, request_obj, fake_provider):
        client.chat("fake-model", request_obj, {"temperature": 0.4})
        assert fake_provider.requests[0]["body"]["temperature"] == 0.4


def test_dead_sanitize_schema_attribute_is_gone():
    # It was ignored by rust-genai and read as a constant True; keeping it
    # only invited callers to branch on a value that meant nothing.
    assert not hasattr(g.ChatOptions(), "sanitize_schema")
    with pytest.raises(TypeError):
        g.ChatOptions(sanitize_schema=False)


def test_partial_options_dict_is_accepted(client, request_obj, fake_provider):
    # rust-genai's stop_sequences is a bare Vec with no serde default, so a
    # partial dict used to fail with "missing field `stop_sequences`" even
    # though every key of the dict form is documented as optional.
    client.chat("fake-model", request_obj, {"max_tokens": 64})
    assert fake_provider.requests[0]["body"]["max_tokens"] == 64


def test_empty_options_dict_is_accepted(client, request_obj):
    assert client.chat("fake-model", request_obj, {}).text == "pong"
