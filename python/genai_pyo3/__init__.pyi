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

    def __new__(
        cls,
        role: str,
        content: str,
        *,
        tool_calls: list[ToolCall] | None = ...,
        tool_response_call_id: str | None = ...,
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
        schema_json: str | None = ...,
    ) -> Tool: ...

class ChatRequest:
    system: str | None
    messages: list[ChatMessage]
    tools: list[Tool] | None

    def __new__(
        cls,
        messages: list[ChatMessage],
        system: str | None = ...,
        tools: list[Tool] | None = ...,
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
        schema_json: str,
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

class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def __new__(
        cls,
        prompt_tokens: int | None = ...,
        completion_tokens: int | None = ...,
        total_tokens: int | None = ...,
    ) -> Usage: ...
    def to_dict(self) -> dict[str, Any]: ...

class ChatResponse:
    """Isomorphic to ``genai::chat::ChatResponse``.

    ``content`` is the single source of truth. ``first_text()``, ``texts()``,
    and ``tool_calls()`` are derived views, matching rust-genai's getter
    method names and semantics.
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
    ) -> ChatResponse: ...
    @property
    def content(self) -> list[ContentPartDict]: ...
    reasoning_content: str | None
    model_adapter_kind: str
    model_name: str
    provider_model_adapter_kind: str
    provider_model_name: str
    usage: Usage

    def first_text(self) -> str | None: ...
    def texts(self) -> list[str]: ...
    def tool_calls(self) -> list[ToolCall]: ...
    def to_dict(self) -> dict[str, Any]: ...

class StreamEnd:
    """Isomorphic to ``genai::chat::StreamEnd``.

    ``captured_content`` is the single source of truth. ``captured_first_text``,
    ``captured_texts``, and ``captured_tool_calls`` are derived views.
    """

    @property
    def captured_content(self) -> list[ContentPartDict] | None: ...
    captured_reasoning_content: str | None
    captured_usage: Usage | None

    def captured_first_text(self) -> str | None: ...
    def captured_texts(self) -> list[str]: ...
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
    "ContentPartDict",
    "JsonSpec",
    "JsonSpecDict",
    "Role",
    "StreamEnd",
    "Tool",
    "ToolCall",
    "ToolDict",
    "Usage",
]
