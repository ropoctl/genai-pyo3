import json
from collections.abc import Mapping

from ._genai_pyo3 import (
    AdapterKind,
    ApiError,
    AuthData,
    AuthError,
    BadRequestError,
    Binary,
    CacheCreationDetails,
    ChatMessage,
    ChatOptions,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    Client,
    ClientBuilder,
    CompletionTokensDetails,
    ConfigError,
    EmbedOptions,
    EmbedResponse,
    Embedding,
    Endpoint,
    GenaiError,
    JsonSpec,
    ModelIden,
    PromptTokensDetails,
    RateLimitError,
    ResolverError,
    ResponseError,
    ServerError,
    sanitize_json_schema,
    ServiceTarget,
    StreamEnd,
    StreamError,
    Tool,
    ToolCall,
    TransportError,
    UnsupportedError,
    Usage,
)
from .types import (
    BinaryPartDict,
    ChatMessageDict,
    ChatOptionsDict,
    ChatOptionsLike,
    ChatRequestDict,
    ChatRequestLike,
    ContentPartDict,
    CustomPartDict,
    EmbedOptionsDict,
    EmbedOptionsLike,
    JsonSpecDict,
    ModelLike,
    ReasoningContentPartDict,
    Role,
    StopReason,
    StreamEventKind,
    TextPartDict,
    ThoughtSignaturePartDict,
    ToolCallPartDict,
    ToolDict,
    ToolResponsePartDict,
)


_MISSING = object()
_CONTENT_PART_KEYS = {
    "text",
    "binary",
    "tool_call",
    "tool_response",
    "thought_signature",
    "reasoning_content",
    "custom",
}


def _mapping_or_attr(obj, key, default=_MISSING):
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    return getattr(obj, key, default)


def _is_content_part_mapping(item):
    if not isinstance(item, Mapping) or len(item) != 1:
        return False
    key = next(iter(item))
    return key in _CONTENT_PART_KEYS


def _has_structured_content_parts(items):
    for item in items:
        if isinstance(item, Binary) or _is_content_part_mapping(item):
            return True
    return False


def _coerce_item_to_content_part(item):
    if isinstance(item, (str, Binary)):
        return item
    if isinstance(item, Mapping):
        if _is_content_part_mapping(item):
            return dict(item)
        item_type = item.get("type")
        if item_type in {"text", "text-plain", "input_text", "output_text"}:
            text = item.get("text")
            if isinstance(text, str):
                return {"text": text}
        elif item_type == "reasoning":
            reasoning = item.get("reasoning")
            if isinstance(reasoning, str):
                return {"reasoning_content": reasoning}
    return item


def _content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type in {"text", "text-plain", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif item_type == "reasoning":
                reasoning = item.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    parts.append(reasoning)
            elif item_type is None and "text" in item:
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
            elif item_type is None and "reasoning_content" in item:
                reasoning = item.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    parts.append(reasoning)
        return "\n".join(part for part in parts if part)
    return str(content)


def _tool_call_from_python(obj):
    if isinstance(obj, ToolCall):
        return obj

    call_id = _mapping_or_attr(obj, "call_id", _MISSING)
    if call_id is _MISSING:
        call_id = _mapping_or_attr(obj, "id", "")

    fn_name = _mapping_or_attr(obj, "fn_name", _MISSING)
    if fn_name is _MISSING:
        fn_name = _mapping_or_attr(obj, "name", "")

    fn_arguments_json = _mapping_or_attr(obj, "fn_arguments_json", _MISSING)
    if fn_arguments_json is _MISSING:
        args = _mapping_or_attr(obj, "fn_arguments", _MISSING)
        if args is _MISSING:
            args = _mapping_or_attr(obj, "arguments", _MISSING)
        if args is _MISSING:
            args = _mapping_or_attr(obj, "args", {})
        fn_arguments_json = json.dumps(args)

    thought_signatures = _mapping_or_attr(obj, "thought_signatures", None)
    return ToolCall(
        call_id=str(call_id or ""),
        fn_name=str(fn_name or ""),
        fn_arguments_json=str(fn_arguments_json or "{}"),
        thought_signatures=thought_signatures,
    )


def _chat_message_from_python(obj):
    if isinstance(obj, ChatMessage):
        return obj
    if isinstance(obj, str):
        return ChatMessage("user", obj)

    role = _mapping_or_attr(obj, "role", _MISSING)
    if role is _MISSING:
        message_type = str(_mapping_or_attr(obj, "type", "user"))
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }.get(message_type, "user")
    else:
        role = str(role)

    raw_content_field = _mapping_or_attr(obj, "content", _MISSING)
    content_parts = _mapping_or_attr(obj, "content_parts", None)
    if content_parts is None and isinstance(raw_content_field, list):
        if _has_structured_content_parts(raw_content_field):
            content_parts = [
                _coerce_item_to_content_part(item) for item in raw_content_field
            ]
    cache_control = _mapping_or_attr(obj, "cache_control", None)
    thought_signatures = _mapping_or_attr(obj, "thought_signatures", None)
    reasoning_content = _mapping_or_attr(obj, "reasoning_content", None)
    raw_content = _mapping_or_attr(obj, "raw_content", None)
    if raw_content_field is _MISSING:
        content = None if (content_parts is not None or raw_content is not None) else ""
    else:
        content = _content_to_text(raw_content_field)

    if role == "tool":
        tool_response_call_id = _mapping_or_attr(obj, "tool_response_call_id", _MISSING)
        if tool_response_call_id is _MISSING:
            tool_response_call_id = _mapping_or_attr(obj, "tool_call_id", None)
        return ChatMessage(
            "tool",
            content,
            tool_response_call_id=tool_response_call_id,
            cache_control=cache_control,
            raw_content=raw_content,
            content_parts=content_parts,
        )

    tool_calls = _mapping_or_attr(obj, "tool_calls", None)
    if tool_calls:
        return ChatMessage(
            role,
            content,
            tool_calls=[ToolCall.from_python(tool_call) for tool_call in tool_calls],
            cache_control=cache_control,
            thought_signatures=thought_signatures,
            raw_content=raw_content,
            content_parts=content_parts,
            reasoning_content=reasoning_content,
        )
    return ChatMessage(
        role,
        content,
        cache_control=cache_control,
        thought_signatures=thought_signatures,
        raw_content=raw_content,
        content_parts=content_parts,
        reasoning_content=reasoning_content,
    )


ToolCall.from_python = staticmethod(_tool_call_from_python)
ChatMessage.from_python = staticmethod(_chat_message_from_python)

__all__ = [
    "AdapterKind",
    "ApiError",
    "AuthData",
    "AuthError",
    "BadRequestError",
    "Binary",
    "BinaryPartDict",
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
    "ClientBuilder",
    "CompletionTokensDetails",
    "ConfigError",
    "ContentPartDict",
    "CustomPartDict",
    "ReasoningContentPartDict",
    "EmbedOptions",
    "EmbedOptionsDict",
    "EmbedOptionsLike",
    "EmbedResponse",
    "Embedding",
    "Endpoint",
    "GenaiError",
    "JsonSpec",
    "JsonSpecDict",
    "ModelIden",
    "ModelLike",
    "PromptTokensDetails",
    "RateLimitError",
    "ResolverError",
    "ResponseError",
    "Role",
    "ServerError",
    "ServiceTarget",
    "StopReason",
    "StreamEnd",
    "StreamError",
    "StreamEventKind",
    "TextPartDict",
    "ThoughtSignaturePartDict",
    "Tool",
    "ToolCall",
    "ToolCallPartDict",
    "ToolDict",
    "ToolResponsePartDict",
    "TransportError",
    "UnsupportedError",
    "Usage",
    "sanitize_json_schema",
]
