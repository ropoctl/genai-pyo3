"""Type stubs for genai-pyo3.

Covers both the Rust-backed pyclasses (``ChatMessage``, ``ChatRequest``, ...)
and the TypedDict shapes that ``Client.chat`` / ``Client.achat`` /
``Client.astream_chat`` accept as an alternative input form.

The dict shapes are the native ``rust-genai`` serde representations, so
``TypedDict`` values flow through ``pythonize → serde_json::Value →
serde::from_value`` with no runtime translation layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

# ---------------------------------------------------------------------------
# TypedDict shapes (dict form accepted by Client.{chat,achat,astream_chat})
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]

class TextPartDict(TypedDict):
    """``{"text": "hello"}`` — lowercase alias for the ``Text`` variant."""

    text: str

class BinaryPartDict(TypedDict):
    binary: Any

class ToolCallPartDict(TypedDict):
    tool_call: Any

class ToolResponsePartDict(TypedDict):
    tool_response: Any

ContentPartDict = (
    TextPartDict | BinaryPartDict | ToolCallPartDict | ToolResponsePartDict
)

class ChatMessageDict(TypedDict):
    role: Role
    content: list[ContentPartDict]
    options: NotRequired[Any]

class ToolDict(TypedDict):
    name: str
    description: NotRequired[str]
    schema: NotRequired[dict[str, Any]]

class ChatRequestDict(TypedDict, total=False):
    system: str | None
    messages: list[ChatMessageDict]
    tools: list[ToolDict] | None
    previous_response_id: str | None
    store: bool | None

class JsonSpecDict(TypedDict):
    name: str
    schema: dict[str, Any]
    description: NotRequired[str]

class ChatOptionsDict(TypedDict, total=False):
    """Mirror of ``genai::chat::ChatOptions``.

    Only a subset of the option knobs is surfaced here — the dict is
    deserialized by serde, so any field accepted by the underlying
    ``ChatOptions`` struct will be honored even if it is not typed in this
    stub.
    """

    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    stop_sequences: list[str] | None
    capture_usage: bool | None
    capture_content: bool | None
    capture_reasoning_content: bool | None
    capture_tool_calls: bool | None
    capture_raw_body: bool | None
    response_json_spec: JsonSpecDict | None
    response_json_mode: bool | None
    normalize_reasoning_content: bool | None
    seed: int | None
    extra_headers: dict[str, str] | None
    reasoning_effort: str | None

# ---------------------------------------------------------------------------
# Rust-backed pyclasses
# ---------------------------------------------------------------------------

class ChatMessage:
    role: str
    content: str
    tool_calls: list[ToolCall] | None
    tool_response_call_id: str | None
    #: Per-message cache hint: "ephemeral" (default 5m), "ephemeral_5m",
    #: "ephemeral_1h", "ephemeral_24h", "memory". Anthropic maps these
    #: to ``cache_control`` at the content-part level. OpenAI maps a
    #: subset to request-level caching and ignores the rest.
    cache_control: str | None
    #: Encrypted reasoning blobs from a prior assistant turn (OpenAI
    #: Responses ``type:"reasoning"`` items). When set on an assistant
    #: message, the ``openai_resp`` adapter emits each blob as a
    #: top-level ``{type:"reasoning", encrypted_content:...}`` input
    #: item before the message it belongs to. Required to keep the
    #: Responses-API prefix cache warm across turns on reasoning models.
    thought_signatures: list[str] | None

    def __new__(
        cls,
        role: str,
        content: str,
        *,
        tool_calls: list[ToolCall] | None = ...,
        tool_response_call_id: str | None = ...,
        cache_control: str | None = ...,
        thought_signatures: list[str] | None = ...,
    ) -> ChatMessage: ...
    @staticmethod
    def from_python(obj: Any) -> ChatMessage: ...

class Tool:
    name: str
    description: str | None
    schema_json: str | None

    def __new__(
        cls,
        name: str,
        description: str | None = ...,
        schema_json: str | dict[str, Any] | None = ...,
    ) -> Tool: ...

class ChatRequest:
    system: str | None
    messages: list[ChatMessage]
    tools: list[Tool] | None
    #: OpenAI Responses API: chain a server-stored prior response so the
    #: backend only processes the new turn's delta. Set after capturing
    #: ``ChatResponse.response_id`` from a request that had ``store=True``.
    previous_response_id: str | None
    #: OpenAI Responses API: ask the backend to persist this response so
    #: its ``response_id`` is reusable on a follow-up request.
    store: bool | None

    def __new__(
        cls,
        messages: list[ChatMessage],
        system: str | None = ...,
        tools: list[Tool] | None = ...,
        previous_response_id: str | None = ...,
        store: bool | None = ...,
    ) -> ChatRequest: ...
    def add_message(self, message: ChatMessage) -> None: ...
    def message_count(self) -> int: ...

class JsonSpec:
    name: str
    schema_json: str
    description: str | None

    def __new__(
        cls,
        name: str,
        schema_json: str | dict[str, Any],
        description: str | None = ...,
    ) -> JsonSpec: ...

class ChatOptions:
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    stop_sequences: list[str]
    capture_usage: bool | None
    capture_content: bool | None
    capture_reasoning_content: bool | None
    capture_tool_calls: bool | None
    capture_raw_body: bool | None
    response_json_spec: JsonSpec | None
    response_json_mode: bool | None
    normalize_reasoning_content: bool | None
    seed: int | None
    extra_headers: dict[str, str] | None
    #: Reasoning effort hint. Accepted: "none" | "minimal" | "low" |
    #: "medium" | "high" | "xhigh" | "max" | "budget:<n>". When set
    #: alongside `capture_reasoning_content=True`, the OpenAI Responses
    #: adapter opts into `reasoning.summary="detailed"` — required
    #: to get summaries back from the API.
    reasoning_effort: str | None
    #: OpenAI Responses API prefix-cache key. Keep stable across requests
    #: in a session to maximise cache hits — surfaced as
    #: ``Usage.prompt_tokens_details.cached_tokens`` on subsequent calls.
    prompt_cache_key: str | None

    def __new__(cls) -> ChatOptions: ...

class ToolCall:
    call_id: str
    fn_name: str
    fn_arguments_json: str
    thought_signatures: list[str] | None

    def __new__(
        cls,
        call_id: str,
        fn_name: str,
        fn_arguments_json: str,
        thought_signatures: list[str] | None = ...,
    ) -> ToolCall: ...
    @property
    def fn_arguments(self) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_python(obj: Any) -> ToolCall: ...

class CacheCreationDetails:
    ephemeral_5m_tokens: int | None
    ephemeral_1h_tokens: int | None

    def __new__(
        cls,
        ephemeral_5m_tokens: int | None = ...,
        ephemeral_1h_tokens: int | None = ...,
    ) -> CacheCreationDetails: ...
    def to_dict(self) -> dict[str, Any]: ...

class PromptTokensDetails:
    #: Tokens written to the cache on this turn. Anthropic
    #: ``cache_creation_input_tokens``. May incur a surcharge; subsequent
    #: requests benefit via ``cached_tokens``.
    cache_creation_tokens: int | None
    cache_creation_details: CacheCreationDetails | None
    #: Tokens served from the cache on this turn — the headline number for
    #: verifying prompt-caching is live (Anthropic
    #: ``cache_read_input_tokens``, OpenAI ``cached_tokens``).
    cached_tokens: int | None
    audio_tokens: int | None

    def __new__(
        cls,
        cache_creation_tokens: int | None = ...,
        cache_creation_details: CacheCreationDetails | None = ...,
        cached_tokens: int | None = ...,
        audio_tokens: int | None = ...,
    ) -> PromptTokensDetails: ...
    def to_dict(self) -> dict[str, Any]: ...

class CompletionTokensDetails:
    accepted_prediction_tokens: int | None
    rejected_prediction_tokens: int | None
    reasoning_tokens: int | None
    audio_tokens: int | None

    def __new__(
        cls,
        accepted_prediction_tokens: int | None = ...,
        rejected_prediction_tokens: int | None = ...,
        reasoning_tokens: int | None = ...,
        audio_tokens: int | None = ...,
    ) -> CompletionTokensDetails: ...
    def to_dict(self) -> dict[str, Any]: ...

class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_tokens_details: PromptTokensDetails | None
    completion_tokens_details: CompletionTokensDetails | None

    def __new__(
        cls,
        prompt_tokens: int | None = ...,
        completion_tokens: int | None = ...,
        total_tokens: int | None = ...,
        prompt_tokens_details: PromptTokensDetails | None = ...,
        completion_tokens_details: CompletionTokensDetails | None = ...,
    ) -> Usage: ...
    def to_dict(self) -> dict[str, Any]: ...

class ChatResponse:
    """Pythonic wrapper for ``genai::chat::ChatResponse``.

    ``content`` is the single source of truth. ``text``, ``first_text``,
    ``texts``, and ``tool_calls`` are read-only derived properties.
    """

    def __new__(
        cls,
        content: list[ContentPartDict] | None = ...,
        reasoning_content: str | None = ...,
        model_adapter_kind: str | None = ...,
        model_name: str | None = ...,
        provider_model_adapter_kind: str | None = ...,
        provider_model_name: str | None = ...,
        usage: Usage | None = ...,
        response_id: str | None = ...,
    ) -> ChatResponse: ...
    @property
    def content(self) -> list[ContentPartDict]: ...
    reasoning_content: str | None
    model_adapter_kind: str
    model_name: str
    provider_model_adapter_kind: str
    provider_model_name: str
    usage: Usage
    #: OpenAI Responses API: the server-stored response id, present when
    #: the request was sent with ``store=True``. Feed back as
    #: ``previous_response_id`` on the next request to chain calls.
    response_id: str | None

    @property
    def text(self) -> str | None: ...
    @property
    def first_text(self) -> str | None: ...
    @property
    def texts(self) -> list[str]: ...
    @property
    def tool_calls(self) -> list[ToolCall]: ...
    def to_dict(self) -> dict[str, Any]: ...

class StreamEnd:
    """Pythonic wrapper for ``genai::chat::StreamEnd``.

    ``captured_content`` is the single source of truth. ``captured_text``,
    ``captured_first_text``, ``captured_texts``, and ``captured_tool_calls``
    are read-only derived properties.
    """

    @property
    def captured_content(self) -> list[ContentPartDict] | None: ...
    captured_reasoning_content: str | None
    captured_usage: Usage | None
    #: OpenAI Responses API: the server-stored response id captured from
    #: the terminal stream event. Feed back as ``previous_response_id``
    #: on the next request to chain calls.
    captured_response_id: str | None
    #: Encrypted reasoning blobs (``type:"reasoning"`` items) from the
    #: streamed response. On OpenAI Responses-API reasoning models
    #: these must be carried back into the next turn's input — as
    #: ``ChatMessage(thought_signatures=...)`` on the assistant turn —
    #: to keep the prefix cache warm.
    captured_thought_signatures: list[str] | None

    @property
    def captured_text(self) -> str | None: ...
    @property
    def captured_first_text(self) -> str | None: ...
    @property
    def captured_texts(self) -> list[str]: ...
    @property
    def captured_tool_calls(self) -> list[ToolCall]: ...
    def to_dict(self) -> dict[str, Any]: ...

class ChatStreamEvent:
    kind: str
    delta: str | None
    reasoning_delta: str | None
    tool_calls: list[ToolCall] | None
    end: StreamEnd | None

class _ChatStream:
    def __aiter__(self) -> _ChatStream: ...
    async def __anext__(self) -> ChatStreamEvent: ...

# ---------------------------------------------------------------------------
# Client — accepts either the pyclass shapes or the TypedDict forms above
# ---------------------------------------------------------------------------

ChatRequestLike = ChatRequest | ChatRequestDict
ChatOptionsLike = ChatOptions | ChatOptionsDict

class Client:
    def __new__(cls) -> Client: ...
    @staticmethod
    def with_api_key(
        provider: str,
        api_key: str,
        *,
        connect_timeout_seconds: float | None = ...,
        read_timeout_seconds: float | None = ...,
        timeout_seconds: float | None = ...,
    ) -> Client:
        """Construct a client scoped to *provider* with *api_key*.

        Timeouts default to a 30s connect-timeout so a stalled TLS
        handshake or unresponsive proxy can't block forever. Pass
        ``connect_timeout_seconds=None`` to disable. ``read_timeout_seconds``
        and ``timeout_seconds`` are opt-in (default ``None`` → unbounded).
        """
    @staticmethod
    def with_api_key_and_base_url(
        provider: str,
        api_key: str,
        base_url: str,
        *,
        connect_timeout_seconds: float | None = ...,
        read_timeout_seconds: float | None = ...,
        timeout_seconds: float | None = ...,
    ) -> Client: ...
    @staticmethod
    def with_base_url(
        provider: str,
        base_url: str,
        *,
        connect_timeout_seconds: float | None = ...,
        read_timeout_seconds: float | None = ...,
        timeout_seconds: float | None = ...,
    ) -> Client: ...
    @staticmethod
    def with_request_override(
        provider: str,
        url: str,
        headers: dict[str, str],
    ) -> Client: ...
    def chat(
        self,
        model: str,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...
    async def achat(
        self,
        model: str,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...
    async def astream_chat(
        self,
        model: str,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> _ChatStream: ...
    async def achat_via_stream(
        self,
        model: str,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...

__all__ = [
    "CacheCreationDetails",
    "ChatMessage",
    "ChatMessageDict",
    "ChatOptions",
    "ChatOptionsDict",
    "ChatOptionsLike",
    "ChatRequest",
    "ChatRequestDict",
    "ChatRequestLike",
    "ChatResponse",
    "ChatStreamEvent",
    "Client",
    "CompletionTokensDetails",
    "ContentPartDict",
    "JsonSpec",
    "JsonSpecDict",
    "PromptTokensDetails",
    "Role",
    "StreamEnd",
    "Tool",
    "ToolCall",
    "ToolDict",
    "Usage",
]
