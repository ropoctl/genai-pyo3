import json
from collections.abc import Mapping

from ._genai_pyo3 import (
    ChatMessage,
    ChatOptions,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    Client,
    JsonSpec,
    StreamEnd,
    Tool,
    ToolCall,
    Usage,
)


_MISSING = object()


def _mapping_or_attr(obj, key, default=_MISSING):
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    return getattr(obj, key, default)


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

    content = _content_to_text(_mapping_or_attr(obj, "content", ""))
    if role == "tool":
        tool_response_call_id = _mapping_or_attr(obj, "tool_response_call_id", _MISSING)
        if tool_response_call_id is _MISSING:
            tool_response_call_id = _mapping_or_attr(obj, "tool_call_id", None)
        return ChatMessage(
            "tool",
            content,
            tool_response_call_id=tool_response_call_id,
        )

    tool_calls = _mapping_or_attr(obj, "tool_calls", None)
    if tool_calls:
        return ChatMessage(
            role,
            content,
            tool_calls=[ToolCall.from_python(tool_call) for tool_call in tool_calls],
        )
    return ChatMessage(role, content)


ToolCall.from_python = staticmethod(_tool_call_from_python)
ChatMessage.from_python = staticmethod(_chat_message_from_python)

__all__ = [
    "Client",
    "ChatMessage",
    "ChatRequest",
    "ChatOptions",
    "ChatResponse",
    "ChatStreamEvent",
    "StreamEnd",
    "Tool",
    "ToolCall",
    "Usage",
    "JsonSpec",
]
