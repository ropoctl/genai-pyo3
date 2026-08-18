"""Type stubs for genai-pyo3.

Covers the Rust-backed pyclasses and the TypedDict shapes that the client
accepts as an alternative input form.

The dict shapes are the native ``rust-genai`` serde representations, so
``TypedDict`` values flow through ``pythonize → serde_json::Value →
serde::from_value`` with no runtime translation layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

# ---------------------------------------------------------------------------
# TypedDict shapes (dict form accepted wherever a typed object is)
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

class CustomPartDict(TypedDict):
    custom: Any

ContentPartDict = (
    TextPartDict | BinaryPartDict | ToolCallPartDict | ToolResponsePartDict | CustomPartDict
)

class ChatMessageDict(TypedDict):
    role: Role
    content: list[ContentPartDict]
    options: NotRequired[Any]
    cache_control: NotRequired[str]
    thought_signatures: NotRequired[list[str]]
    raw_content: NotRequired[list[dict[str, Any]]]

class ToolDict(TypedDict):
    name: str
    description: NotRequired[str]
    schema: NotRequired[dict[str, Any]]
    #: Provider-native freeform custom-tool format. OpenAI Responses accepts
    #: grammar formats such as {"type": "grammar", "syntax": "lark", ...}.
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
    """Mirror of ``genai::chat::ChatOptions``.

    Any field accepted by the underlying Rust struct is honored even if it
    is not typed here; missing keys fall back to the struct's defaults.
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

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class AdapterKind:
    """Which provider protocol to speak."""

    def __new__(cls, name: str) -> AdapterKind:
        """Parse a provider name case-insensitively (``"openai"``, ``"OpenAI"``)."""

    @staticmethod
    def custom(index: int) -> AdapterKind:
        """The ``genai_<n>`` family, configured from ``GENAI_<n>_*`` env vars."""

    @staticmethod
    def all() -> list[AdapterKind]:
        """Every built-in adapter kind (excludes the parameterized ``custom``)."""

    @staticmethod
    def from_model(model: str) -> AdapterKind:
        """Infer the adapter from a model name, as a bare model string would."""

    @property
    def name(self) -> str:
        """Canonical lowercase name, the form the constructor accepts back."""

    @property
    def display_name(self) -> str: ...
    @property
    def default_key_env_name(self) -> str | None: ...

class ModelIden:
    """An adapter paired with a model name."""

    def __new__(cls, adapter_kind: AdapterKind | str, model_name: str) -> ModelIden: ...
    @property
    def adapter_kind(self) -> AdapterKind: ...
    @property
    def model_name(self) -> str: ...

class Endpoint:
    def __new__(cls, base_url: str) -> Endpoint: ...
    @property
    def base_url(self) -> str: ...

class AuthData:
    """How to authenticate against a provider.

    Secret material is never returned to Python and never appears in
    ``repr()`` — an ``AuthData`` that printed its own key would leak into
    every log line that logged a ``ServiceTarget``.
    """

    @staticmethod
    def from_env(env_name: str) -> AuthData:
        """Read the key from an environment variable at request time."""

    @staticmethod
    def key(value: str) -> AuthData: ...
    @staticmethod
    def request_override(url: str, headers: dict[str, str]) -> AuthData:
        """Replace the request URL and headers wholesale."""

    @staticmethod
    def multi(values: dict[str, str]) -> AuthData: ...
    @staticmethod
    def none() -> AuthData: ...
    @property
    def kind(self) -> Literal["from_env", "key", "request_override", "multi_keys", "none"]: ...
    @property
    def env_name(self) -> str | None: ...
    @property
    def url(self) -> str | None: ...

class ServiceTarget:
    """A named inference endpoint: where, with what credential, for which model.

    Pass one wherever a model name is accepted to route a single call
    without touching the client's configuration.
    """

    def __new__(
        cls, endpoint: Endpoint | str, auth: AuthData, model: ModelIden
    ) -> ServiceTarget: ...
    @property
    def endpoint(self) -> Endpoint: ...
    @property
    def auth(self) -> AuthData: ...
    @property
    def model(self) -> ModelIden: ...
    def to_dict(self) -> dict[str, Any]: ...

#: What a call's ``model`` argument may be. The three forms line up with
#: rust-genai's ``ModelSpec`` and pick how much resolution still has to
#: happen: infer everything, pin the adapter, or state the whole target.
ModelLike = str | ModelIden | ServiceTarget

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class GenaiError(RuntimeError):
    """Base class for every error raised by genai-pyo3.

    Derives from ``RuntimeError`` because that is what every failure used
    to be; existing ``except RuntimeError:`` handlers keep working.
    """

    #: Resolved model name, when the failure happened after resolution.
    model_name: str | None
    #: Resolved adapter, in rust-genai's display spelling (``"OpenAI"``).
    adapter_kind: str | None
    #: HTTP status, when the provider answered with one.
    status: int | None
    #: Response body, when there was one.
    body: str | None
    #: ``Retry-After`` in seconds, when the provider sent one.
    retry_after: float | None
    #: Conservative hint: true only for statuses and transport failures
    #: worth retrying. The retry and backoff policy is yours.
    is_retryable: bool

class ConfigError(GenaiError): ...
class AuthError(GenaiError): ...
class ResolverError(GenaiError): ...
class ApiError(GenaiError): ...
class BadRequestError(ApiError): ...
class RateLimitError(ApiError): ...
class ServerError(ApiError): ...
class TransportError(GenaiError): ...
class StreamError(GenaiError): ...
class ResponseError(GenaiError): ...
class UnsupportedError(GenaiError): ...

# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class Binary:
    """A binary attachment — an image, a PDF, an audio clip."""

    @staticmethod
    def from_base64(content_type: str, content: str, name: str | None = ...) -> Binary: ...
    @staticmethod
    def from_url(content_type: str, url: str, name: str | None = ...) -> Binary:
        """Reference a remote file. Only some providers fetch URLs."""

    @staticmethod
    def from_path(path: str) -> Binary:
        """Read a local file, guessing its MIME type and base64-encoding it."""

    @property
    def content_type(self) -> str: ...
    @property
    def name(self) -> str | None: ...
    @property
    def source_kind(self) -> Literal["base64", "url"]: ...
    @property
    def is_image(self) -> bool: ...
    @property
    def is_pdf(self) -> bool: ...
    @property
    def is_audio(self) -> bool: ...
    @property
    def is_video(self) -> bool: ...
    @property
    def size(self) -> int:
        """Approximate payload size in bytes."""

    def to_dict(self) -> dict[str, Any]: ...

class ChatMessage:
    role: str
    content: str
    tool_calls: list[ToolCall] | None
    tool_response_call_id: str | None
    cache_control: str | None
    thought_signatures: list[str] | None
    raw_content_json: str | None

    def __new__(
        cls,
        role: Role,
        content: str | None = ...,
        tool_calls: list[ToolCall] | None = ...,
        tool_response_call_id: str | None = ...,
        cache_control: str | None = ...,
        thought_signatures: list[str] | None = ...,
        raw_content: Any | None = ...,
        content_parts: list[str | Binary | ContentPartDict] | None = ...,
    ) -> ChatMessage:
        """A message. Supply ``content`` for plain text, or ``content_parts``
        to mix text with images and PDFs."""

    @property
    def content_parts(self) -> list[ContentPartDict] | None: ...
    def to_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_python(obj: Any) -> ChatMessage: ...

class Tool:
    name: str
    description: str | None
    schema_json: str | None
    strict: bool | None
    cache_control: str | None
    eager_input_streaming: bool | None

    def __new__(
        cls,
        name: str,
        description: str | None = ...,
        schema_json: Any | None = ...,
        custom_format: Any | None = ...,
        *,
        strict: bool | None = ...,
        cache_control: str | None = ...,
        eager_input_streaming: bool | None = ...,
        config: Any | None = ...,
    ) -> Tool: ...
    @staticmethod
    def web_search(
        *,
        max_uses: int | None = ...,
        allowed_domains: list[str] | None = ...,
        blocked_domains: list[str] | None = ...,
    ) -> Tool:
        """The provider's built-in, provider-run web-search tool."""

    @property
    def custom_format(self) -> Any | None: ...
    @property
    def config(self) -> Any | None: ...
    @property
    def is_builtin(self) -> bool: ...

class ChatRequest:
    system: str | None
    messages: list[ChatMessage]
    tools: list[Tool] | None
    previous_response_id: str | None
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
        cls, name: str, schema_json: Any, description: str | None = ...
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
    #: to get summaries back from the API. Reasoning capture is opt-in
    #: everywhere (including `achat_via_stream`): some models reject
    #: `reasoning.summary` outright (gpt-5.3-codex-spark returns
    #: `unsupported_parameter`), so only set
    #: `capture_reasoning_content=True` on models that support it.
    reasoning_effort: str | None
    #: OpenAI Responses API prefix-cache key. Keep stable across requests
    #: in a session to maximise cache hits — surfaced as
    #: ``Usage.prompt_tokens_details.cached_tokens`` on subsequent calls.
    prompt_cache_key: str | None
    #: Provider-specific top-level request payload fields.
    extra_body_json: str | None
    #: "auto" | "none" | "required", or the name of a tool the model must call.
    tool_choice: str | None
    #: "low" | "medium" | "high" (OpenAI gpt-5 family).
    verbosity: str | None
    #: "flex" | "auto" | "default" (OpenAI processing tier).
    service_tier: str | None
    #: "ephemeral" | "ephemeral_5m" | "ephemeral_1h" | "ephemeral_24h" | "memory".
    cache_control: str | None

    def __new__(
        cls,
        temperature: float | None = ...,
        max_tokens: int | None = ...,
        top_p: float | None = ...,
        stop_sequences: list[str] | None = ...,
        capture_usage: bool | None = ...,
        capture_content: bool | None = ...,
        capture_reasoning_content: bool | None = ...,
        capture_tool_calls: bool | None = ...,
        capture_raw_body: bool | None = ...,
        response_json_spec: JsonSpec | None = ...,
        response_json_mode: bool | None = ...,
        normalize_reasoning_content: bool | None = ...,
        seed: int | None = ...,
        extra_headers: dict[str, str] | None = ...,
        reasoning_effort: str | None = ...,
        prompt_cache_key: str | None = ...,
        extra_body: dict[str, Any] | None = ...,
        tool_choice: str | None = ...,
        verbosity: str | None = ...,
        service_tier: str | None = ...,
        cache_control: str | None = ...,
    ) -> ChatOptions: ...

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

# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class CacheCreationDetails:
    ephemeral_5m_tokens: int | None
    ephemeral_1h_tokens: int | None
    def to_dict(self) -> dict[str, Any]: ...

class PromptTokensDetails:
    cache_creation_tokens: int | None
    cache_creation_details: CacheCreationDetails | None
    cached_tokens: int | None
    audio_tokens: int | None
    def to_dict(self) -> dict[str, Any]: ...

class CompletionTokensDetails:
    accepted_prediction_tokens: int | None
    rejected_prediction_tokens: int | None
    reasoning_tokens: int | None
    audio_tokens: int | None
    def to_dict(self) -> dict[str, Any]: ...

class Usage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    prompt_tokens_details: PromptTokensDetails | None
    completion_tokens_details: CompletionTokensDetails | None
    def to_dict(self) -> dict[str, Any]: ...

#: Normalised stop reasons, comparable across providers.
StopReason = Literal[
    "completed", "max_tokens", "tool_call", "content_filter", "stop_sequence", "other"
]

class ChatResponse:
    reasoning_content: str | None
    model_adapter_kind: str
    model_name: str
    provider_model_adapter_kind: str
    provider_model_name: str
    usage: Usage
    response_id: str | None
    #: Why generation stopped. ``"max_tokens"`` is the only way to tell a
    #: truncated answer from a short one.
    stop_reason: StopReason | None
    #: The provider's own stop-reason string, before normalisation.
    stop_reason_raw: str | None

    @property
    def content(self) -> list[ContentPartDict]: ...
    @property
    def text(self) -> str | None: ...
    @property
    def first_text(self) -> str | None: ...
    @property
    def texts(self) -> list[str]: ...
    @property
    def tool_calls(self) -> list[ToolCall]: ...
    @property
    def captured_raw_body(self) -> Any | None:
        """The provider's raw body — only when ``capture_raw_body=True``."""

    def to_dict(self) -> dict[str, Any]: ...

class StreamEnd:
    captured_usage: Usage | None
    captured_reasoning_content: str | None
    captured_response_id: str | None
    captured_thought_signatures: list[str] | None
    captured_stop_reason: StopReason | None
    captured_stop_reason_raw: str | None

    @property
    def captured_content(self) -> list[ContentPartDict] | None: ...
    @property
    def captured_text(self) -> str | None: ...
    @property
    def captured_first_text(self) -> str | None: ...
    @property
    def captured_texts(self) -> list[str]: ...
    @property
    def captured_tool_calls(self) -> list[ToolCall]: ...
    def to_dict(self) -> dict[str, Any]: ...

#: Stream event kinds, in the order they can occur.
StreamEventKind = Literal[
    "start",
    "chunk",
    "reasoning_chunk",
    "thought_signature_chunk",
    "tool_call_chunk",
    "end",
]

class ChatStreamEvent:
    kind: StreamEventKind
    #: Text for ``chunk`` / ``reasoning_chunk`` / ``thought_signature_chunk``.
    content: str | None
    tool_call: ToolCall | None
    end: StreamEnd | None
    def to_dict(self) -> dict[str, Any]: ...

class _ChatStream:
    def __aiter__(self) -> _ChatStream: ...
    async def __anext__(self) -> ChatStreamEvent: ...

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbedOptions:
    headers: dict[str, str] | None
    capture_raw_body: bool | None
    capture_usage: bool | None
    dimensions: int | None
    encoding_format: str | None
    user: str | None
    #: Provider-specific vocabulary. Cohere: "search_document" /
    #: "search_query" / "classification" / "clustering". Gemini:
    #: "SEMANTIC_SIMILARITY" / "RETRIEVAL_QUERY" / "RETRIEVAL_DOCUMENT" /
    #: "CLASSIFICATION". Getting it wrong quietly costs retrieval quality.
    embedding_type: str | None
    truncate: str | None

    def __new__(
        cls,
        headers: dict[str, str] | None = ...,
        capture_raw_body: bool | None = ...,
        capture_usage: bool | None = ...,
        dimensions: int | None = ...,
        encoding_format: str | None = ...,
        user: str | None = ...,
        embedding_type: str | None = ...,
        truncate: str | None = ...,
    ) -> EmbedOptions: ...

class Embedding:
    @property
    def vector(self) -> list[float]: ...
    @property
    def index(self) -> int:
        """Position of the corresponding input in the request."""

    @property
    def dimensions(self) -> int: ...
    def __len__(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...

class EmbedResponse:
    embeddings: list[Embedding]
    model_adapter_kind: str
    model_name: str
    provider_model_adapter_kind: str
    provider_model_name: str
    usage: Usage

    @property
    def vector(self) -> list[float] | None:
        """The first vector — the common case for a single input."""

    @property
    def vectors(self) -> list[list[float]]:
        """Every vector, sorted by input index."""

    def __len__(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

ChatRequestLike = ChatRequest | ChatRequestDict
ChatOptionsLike = ChatOptions | ChatOptionsDict
EmbedOptionsLike = EmbedOptions | EmbedOptionsDict

class ClientBuilder:
    def __new__(cls) -> ClientBuilder: ...
    def adapter_kind(self, adapter_kind: AdapterKind | str) -> ClientBuilder:
        """Pin routing to one adapter, so bare model names need no prefix."""

    def provider(
        self,
        adapter_kind: AdapterKind | str,
        *,
        endpoint: Endpoint | str | None = ...,
        auth: AuthData | None = ...,
    ) -> ClientBuilder:
        """Configure one provider. Repeatable — call once per provider."""

    def chat_options(self, options: ChatOptionsLike) -> ClientBuilder:
        """Defaults for every call. Per-call options win."""

    def timeouts(
        self,
        *,
        connect_seconds: float | None = ...,
        read_seconds: float | None = ...,
        total_seconds: float | None = ...,
    ) -> ClientBuilder:
        """Replace the whole timeout configuration (connect defaults to 30s)."""

    def default_headers(self, headers: dict[str, str]) -> ClientBuilder: ...
    def embed_options(self, options: EmbedOptionsLike) -> ClientBuilder:
        """Defaults for every embedding call. Per-call options win."""

    def proxy(
        self, url: str, *, scheme: Literal["all", "http", "https"] = ...
    ) -> ClientBuilder:
        """Route requests through an HTTP proxy."""

    def gzip(self, enabled: bool) -> ClientBuilder: ...
    def tcp_nodelay(self, enabled: bool) -> ClientBuilder: ...
    def auth_resolver(self, callback: Any) -> ClientBuilder:
        """``fn(ModelIden) -> AuthData | None``, consulted before the
        declarative provider table. Must be a regular ``def``: it is called
        synchronously while holding the GIL, so blocking in it stalls every
        other Python thread."""

    def model_mapper(self, callback: Any) -> ClientBuilder:
        """``fn(ModelIden) -> ModelIden | str``. A bare string renames the
        model and keeps the adapter."""

    def service_target_resolver(self, callback: Any) -> ClientBuilder:
        """``fn(ServiceTarget) -> ServiceTarget``, run after the declarative
        endpoint overrides, with the last word on the whole target."""

    def build(self) -> Client: ...

class Client:
    def __new__(cls) -> Client:
        """A client with no configuration: every call carries its own routing."""

    @staticmethod
    def builder() -> ClientBuilder: ...
    @property
    def adapter_kind(self) -> AdapterKind | None: ...
    def chat(
        self,
        model: ModelLike,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...
    async def achat(
        self,
        model: ModelLike,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...
    async def astream_chat(
        self,
        model: ModelLike,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> _ChatStream: ...
    async def achat_via_stream(
        self,
        model: ModelLike,
        request: ChatRequestLike,
        options: ChatOptionsLike | None = ...,
    ) -> ChatResponse: ...
    async def aembed(
        self, model: ModelLike, input: str, options: EmbedOptionsLike | None = ...
    ) -> EmbedResponse: ...
    async def aembed_batch(
        self, model: ModelLike, inputs: list[str], options: EmbedOptionsLike | None = ...
    ) -> EmbedResponse: ...
    def resolve_target(self, model: ModelLike) -> ServiceTarget:
        """Where this call would go, without making it."""

    async def aresolve_target(self, model: ModelLike) -> ServiceTarget: ...
    def default_model(self, model_name: str) -> ModelIden: ...
    async def alist_models(
        self,
        adapter_kind: AdapterKind | str,
        *,
        endpoint: Endpoint | str | None = ...,
        auth: AuthData | None = ...,
    ) -> list[str]:
        """Ask a provider which models it is currently serving."""

def sanitize_json_schema(
    schema: Any,
    dialect: Literal["openai_strict", "openai_strict_tool", "anthropic_structured"],
) -> Any:
    """Apply a provider's constrained-decoding schema normalization.

    rust-genai does this automatically at the adapter boundary; this is for
    inspecting what a provider will enforce before a request discovers it.
    The rewrite can be lossy.
    """

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
