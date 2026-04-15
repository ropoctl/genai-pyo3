from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from typing import Any

import aiohttp

from ._genai_pyo3 import (
    ChatMessage,
    ChatOptions,
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    Client as _RustClient,
    StreamEnd,
    Tool,
    ToolCall,
    Usage,
)

_DEFAULT_SYSTEM = "You are a helpful assistant."


def _run_coro_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive relay
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "value" in error:
        raise error["value"]
    return result.get("value")


def _normalize_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    return base_url.rstrip("/")


def _normalize_headers(headers: dict[str, Any] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {str(name): str(value) for name, value in headers.items()}


def _merge_headers(*header_sets: dict[str, Any] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for header_set in header_sets:
        merged.update(_normalize_headers(header_set))
    return merged


def _json_loads_or_none(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _message_to_responses_input(message: ChatMessage) -> list[dict[str, Any]]:
    role = message.role

    if role == "assistant" and message.tool_calls:
        items: list[dict[str, Any]] = []
        if message.content:
            items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": message.content}],
                }
            )
        for tool_call in message.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.call_id,
                    "name": tool_call.fn_name,
                    "arguments": tool_call.fn_arguments_json,
                }
            )
        return items

    if role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_response_call_id or "",
                "output": message.content,
            }
        ]

    if not message.content:
        return []

    return [
        {
            "role": role,
            "content": [{"type": "input_text", "text": message.content}],
        }
    ]


def _request_to_responses_payload(
    model: str,
    request: ChatRequest,
    options: ChatOptions | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": request.system or _DEFAULT_SYSTEM,
        "store": False,
        "input": [],
        "stream": True,
    }

    messages = request.messages() if callable(request.messages) else request.messages
    for message in messages:
        payload["input"].extend(_message_to_responses_input(message))

    if not payload["input"]:
        payload["input"] = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": ""}],
            }
        ]

    if request.tools:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            item: dict[str, Any] = {
                "type": "function",
                "name": tool.name,
            }
            if tool.description:
                item["description"] = tool.description
            schema = _json_loads_or_none(tool.schema_json)
            if schema is not None:
                item["parameters"] = schema
            tools.append(item)
        payload["tools"] = tools

    if options is not None:
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.max_tokens is not None:
            payload["max_output_tokens"] = options.max_tokens
        if options.seed is not None:
            payload["seed"] = options.seed

    return payload


def _usage_from_completed_response(response_payload: dict[str, Any]) -> Usage:
    usage = response_payload.get("usage") or {}
    return Usage(
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


class Client:
    def __init__(
        self,
        inner: _RustClient | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> None:
        self._inner = inner or _RustClient()
        self._provider = provider
        self._api_key = api_key
        self._base_url = _normalize_base_url(base_url)
        self._headers = _normalize_headers(headers)

    @staticmethod
    def with_api_key(
        provider: str,
        api_key: str,
        headers: dict[str, Any] | None = None,
    ) -> "Client":
        return Client(
            _RustClient.with_api_key(provider, api_key),
            provider=provider,
            api_key=api_key,
            headers=headers,
        )

    @staticmethod
    def with_api_key_and_base_url(
        provider: str,
        api_key: str,
        base_url: str,
        headers: dict[str, Any] | None = None,
    ) -> "Client":
        return Client(
            _RustClient.with_api_key_and_base_url(provider, api_key, base_url),
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
        )

    @staticmethod
    def with_base_url(
        provider: str,
        base_url: str,
        headers: dict[str, Any] | None = None,
    ) -> "Client":
        return Client(
            _RustClient.with_base_url(provider, base_url),
            provider=provider,
            base_url=base_url,
            headers=headers,
        )

    def _use_python_openai_responses(self) -> bool:
        return (self._provider or "").strip().lower() == "openai_resp" and bool(self._base_url)

    def _options_with_headers(self, options: ChatOptions | None) -> ChatOptions | None:
        if not self._headers:
            return options

        option_headers = (
            _normalize_headers(options.extra_headers)
            if options is not None and getattr(options, "extra_headers", None)
            else {}
        )
        merged_headers = _merge_headers(self._headers, option_headers)
        if not merged_headers:
            return options

        if options is None:
            return ChatOptions(extra_headers=merged_headers)

        return ChatOptions(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p,
            stop_sequences=list(options.stop_sequences),
            capture_usage=options.capture_usage,
            capture_content=options.capture_content,
            capture_reasoning_content=options.capture_reasoning_content,
            capture_tool_calls=options.capture_tool_calls,
            capture_raw_body=options.capture_raw_body,
            normalize_reasoning_content=options.normalize_reasoning_content,
            seed=options.seed,
            extra_headers=merged_headers,
        )

    def chat(self, model: str, request: ChatRequest, options: ChatOptions | None = None):
        options = self._options_with_headers(options)
        if not self._use_python_openai_responses():
            return self._inner.chat(model, request, options)
        return _run_coro_sync(self.achat(model, request, options))

    async def achat(self, model: str, request: ChatRequest, options: ChatOptions | None = None):
        options = self._options_with_headers(options)
        if not self._use_python_openai_responses():
            return await self._inner.achat(model, request, options)

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        end_event: StreamEnd | None = None

        stream = await self.astream_chat(model, request, options)
        async for event in stream:
            if event.kind == "chunk" and event.content:
                texts.append(event.content)
            elif event.kind == "tool_call_chunk" and event.tool_call is not None:
                tool_calls.append(event.tool_call)
            elif event.kind == "end" and event.end is not None:
                end_event = event.end

        if end_event is not None:
            final_texts = list(end_event.captured_texts or [])
            if not final_texts and end_event.captured_first_text:
                final_texts = [end_event.captured_first_text]
            final_tool_calls = list(end_event.captured_tool_calls or tool_calls)
            usage = end_event.captured_usage or Usage()
            text = end_event.captured_first_text
            if text is None and final_texts:
                text = final_texts[0]
        else:
            final_texts = ["".join(texts)] if texts else []
            final_tool_calls = tool_calls
            usage = Usage()
            text = final_texts[0] if final_texts else None

        return ChatResponse(
            text=text,
            texts=final_texts,
            reasoning_content=None,
            model_adapter_kind="openai_resp",
            model_name=model,
            provider_model_adapter_kind="openai_resp",
            provider_model_name=model,
            usage=usage,
            tool_calls=final_tool_calls,
        )

    async def astream_chat(self, model: str, request: ChatRequest, options: ChatOptions | None = None):
        options = self._options_with_headers(options)
        if not self._use_python_openai_responses():
            return await self._inner.astream_chat(model, request, options)
        return _astream_openai_responses(
            model=model,
            request=request,
            options=options,
            api_key=self._api_key,
            base_url=self._base_url,
            extra_headers=self._headers,
        )


def _astream_openai_responses(
    *,
    model: str,
    request: ChatRequest,
    options: ChatOptions | None,
    api_key: str | None,
    base_url: str | None,
    extra_headers: dict[str, Any] | None = None,
):
    if not api_key:
        raise RuntimeError("openai_resp client requires an API key")
    if not base_url:
        raise RuntimeError("openai_resp client requires a base_url")

    payload = _request_to_responses_payload(model, request, options)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    headers = _merge_headers(headers, extra_headers, getattr(options, "extra_headers", None))
    url = f"{base_url}/responses"

    async def generator():
        saw_start = False
        streamed_tool_calls: list[ToolCall] = []
        function_calls_by_id: dict[str, dict[str, Any]] = defaultdict(dict)
        text_buffer: list[str] = []
        final_usage = Usage()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(
                        f"openai_resp request failed with status {response.status}: {detail}"
                    )

                event_name: str | None = None
                data_lines: list[str] = []

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if not data_lines:
                            event_name = None
                            continue

                        data = "\n".join(data_lines)
                        data_lines = []
                        if data == "[DONE]":
                            break

                        payload_obj = json.loads(data)
                        event_type = event_name or payload_obj.get("type", "")

                        if event_type == "response.created" and not saw_start:
                            saw_start = True
                            yield ChatStreamEvent(kind="start")
                        elif event_type == "response.output_text.delta":
                            delta = payload_obj.get("delta")
                            if delta:
                                text_buffer.append(delta)
                                yield ChatStreamEvent(kind="chunk", content=delta)
                        elif event_type == "response.function_call_arguments.delta":
                            item_id = payload_obj.get("item_id") or ""
                            delta = payload_obj.get("delta") or ""
                            if item_id:
                                call = function_calls_by_id[item_id]
                                call["arguments"] = f"{call.get('arguments', '')}{delta}"
                        elif event_type in {
                            "response.output_item.added",
                            "response.output_item.done",
                        }:
                            item = payload_obj.get("item") or {}
                            if item.get("type") == "function_call":
                                item_id = item.get("id") or item.get("call_id") or ""
                                if item_id:
                                    call = function_calls_by_id[item_id]
                                    call["call_id"] = item.get("call_id") or item_id
                                    call["name"] = item.get("name") or call.get("name") or ""
                                    call["arguments"] = item.get("arguments") or call.get("arguments") or ""
                                if event_type == "response.output_item.done" and item_id:
                                    final_call = function_calls_by_id.pop(item_id, {})
                                    tool_call = ToolCall(
                                        call_id=final_call.get("call_id") or item_id,
                                        fn_name=final_call.get("name") or "",
                                        fn_arguments_json=final_call.get("arguments") or "{}",
                                    )
                                    streamed_tool_calls.append(tool_call)
                                    yield ChatStreamEvent(
                                        kind="tool_call_chunk",
                                        tool_call=tool_call,
                                    )
                            elif event_type == "response.output_item.done" and item.get("type") == "message":
                                for content_part in item.get("content") or []:
                                    if content_part.get("type") == "output_text":
                                        text = content_part.get("text")
                                        if text and not text_buffer:
                                            text_buffer.append(text)
                        elif event_type == "response.completed":
                            response_obj = payload_obj.get("response") or {}
                            final_usage = _usage_from_completed_response(response_obj)
                            final_text = "".join(text_buffer)
                            captured_texts = [final_text] if final_text else []
                            end = StreamEnd(
                                captured_usage=final_usage,
                                captured_first_text=final_text or None,
                                captured_texts=captured_texts or None,
                                captured_reasoning_content=None,
                                captured_tool_calls=streamed_tool_calls or None,
                            )
                            yield ChatStreamEvent(kind="end", end=end)
                            return
                        elif event_type == "response.failed":
                            response_obj = payload_obj.get("response") or {}
                            error = response_obj.get("error") or payload_obj.get("error")
                            raise RuntimeError(f"openai_resp request failed: {error}")

                        event_name = None
                        continue

                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())

        final_text = "".join(text_buffer)
        end = StreamEnd(
            captured_usage=final_usage,
            captured_first_text=final_text or None,
            captured_texts=[final_text] if final_text else None,
            captured_reasoning_content=None,
            captured_tool_calls=streamed_tool_calls or None,
        )
        yield ChatStreamEvent(kind="end", end=end)

    return generator()

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
]
