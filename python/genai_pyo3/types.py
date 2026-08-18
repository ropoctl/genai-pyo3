"""Runtime-importable TypedDict shapes for genai-pyo3.

``Client.chat`` / ``Client.achat`` / ``Client.astream_chat`` accept either a
Rust-backed pyclass (``ChatRequest`` / ``ChatOptions``) or a plain Python
dict that depythonizes into the same serde shape. These TypedDicts let
callers annotate the dict form for mypy / pyright without reaching into
private modules.

Keep in sync with ``__init__.pyi`` — the type stubs are the authoritative
reference for static checkers; this module exists so the same names are
importable at runtime (e.g. for isinstance-free duck-typing or for
libraries that re-export our types).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict, Union

# NotRequired is in typing only on 3.11+. typing_extensions back-ports it to
# 3.9 and is a transitive dep of most of the Python ecosystem already.
from typing_extensions import NotRequired

if False:  # TYPE_CHECKING
    # Only imported by stub-aware tools; keeps runtime import cycle out.
    from . import ChatOptions, ChatRequest, EmbedOptions, ModelIden, ServiceTarget

Role = Literal["system", "user", "assistant", "tool"]
"""Lowercase role names — isomorphic to genai's ``ChatRole`` variants.

The Rust coercion layer title-cases these (``"user"`` → ``"User"``) before
handing the value to serde, so callers get the Python-idiomatic spelling
without fighting serde's default external tagging.
"""


class TextPartDict(TypedDict):
    """Single text segment — ``{"text": "hello"}``."""

    text: str


class BinaryPartDict(TypedDict):
    binary: Any


class ToolCallPartDict(TypedDict):
    tool_call: Any


class ToolResponsePartDict(TypedDict):
    tool_response: Any


class CustomPartDict(TypedDict):
    custom: Any


class ThoughtSignaturePartDict(TypedDict):
    thought_signature: str


class ReasoningContentPartDict(TypedDict):
    reasoning_content: str


ContentPartDict = Union[
    TextPartDict,
    BinaryPartDict,
    ToolCallPartDict,
    ToolResponsePartDict,
    ThoughtSignaturePartDict,
    ReasoningContentPartDict,
    CustomPartDict,
]


class ChatMessageDict(TypedDict):
    role: Role
    content: list[ContentPartDict]
    options: NotRequired[Any]
    cache_control: NotRequired[str]
    thought_signatures: NotRequired[list[str]]
    reasoning_content: NotRequired[str]
    raw_content: NotRequired[list[dict[str, Any]]]


class ToolDict(TypedDict):
    name: str
    description: NotRequired[str]
    schema: NotRequired[dict[str, Any]]
    custom_format: NotRequired[dict[str, Any]]
    strict: NotRequired[bool]
    cache_control: NotRequired[str]
    eager_input_streaming: NotRequired[bool]
    config: NotRequired[Any]


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
    prompt_cache_key: str | None
    extra_body: dict[str, Any] | None
    tool_choice: str | None
    verbosity: str | None
    service_tier: str | None
    cache_control: str | None


class EmbedOptionsDict(TypedDict, total=False):
    headers: dict[str, str] | None
    capture_raw_body: bool | None
    capture_usage: bool | None
    dimensions: int | None
    encoding_format: str | None
    user: str | None
    embedding_type: str | None
    truncate: str | None


#: Normalised stop reasons, comparable across providers.
StopReason = Literal[
    "completed", "max_tokens", "tool_call", "content_filter", "stop_sequence", "other"
]

#: Stream event kinds, in the order they can occur.
StreamEventKind = Literal[
    "start",
    "chunk",
    "reasoning_chunk",
    "thought_signature_chunk",
    "tool_call_chunk",
    #: Keep-alive ping — no content, the stream is simply still open.
    "heartbeat",
    "end",
]


ChatRequestLike = Union["ChatRequest", ChatRequestDict]
ChatOptionsLike = Union["ChatOptions", ChatOptionsDict]
EmbedOptionsLike = Union["EmbedOptions", EmbedOptionsDict]

#: What a call's ``model`` argument may be — the three levels of routing
#: detail: infer everything, pin the adapter, or state the whole target.
ModelLike = Union[str, "ModelIden", "ServiceTarget"]


__all__ = [
    "BinaryPartDict",
    "ChatMessageDict",
    "ChatOptionsDict",
    "ChatOptionsLike",
    "ChatRequestDict",
    "ChatRequestLike",
    "ContentPartDict",
    "CustomPartDict",
    "ReasoningContentPartDict",
    "EmbedOptionsDict",
    "EmbedOptionsLike",
    "JsonSpecDict",
    "ModelLike",
    "Role",
    "StopReason",
    "StreamEventKind",
    "TextPartDict",
    "ThoughtSignaturePartDict",
    "ToolCallPartDict",
    "ToolDict",
    "ToolResponsePartDict",
]
