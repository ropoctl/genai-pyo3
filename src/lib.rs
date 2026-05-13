use futures::StreamExt;
use genai::Headers;
use genai::adapter::AdapterKind;
use genai::chat::{
    CacheControl, CacheCreationDetails, ChatMessage, ChatOptions, ChatRequest, ChatResponseFormat,
    ChatRole, ChatStream, ChatStreamEvent, CompletionTokensDetails, JsonSpec, MessageOptions,
    PromptTokensDetails, ReasoningEffort, StreamEnd, Tool, ToolCall, ToolResponse, Usage,
};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Builder;
use tokio::sync::Mutex;

/// Default connect-timeout applied by `PyClient::with_*` builders when the
/// caller doesn't override it. A 30-second cap is long enough to cover any
/// realistic TCP+TLS handshake and short enough to surface a hung proxy or
/// an unreachable endpoint before it eats through an entire run. Pass
/// ``connect_timeout_seconds=None`` explicitly to disable it.
const DEFAULT_CONNECT_TIMEOUT_SECONDS: f64 = 30.0;

fn web_config_from_overrides(
    connect_timeout_seconds: Option<f64>,
    read_timeout_seconds: Option<f64>,
    timeout_seconds: Option<f64>,
    default_headers: Option<HashMap<String, String>>,
) -> PyResult<Option<genai::WebConfig>> {
    if connect_timeout_seconds.is_none()
        && read_timeout_seconds.is_none()
        && timeout_seconds.is_none()
        && default_headers.as_ref().map_or(true, |h| h.is_empty())
    {
        return Ok(None);
    }
    let mut web_config = genai::WebConfig::default();
    if let Some(secs) = connect_timeout_seconds {
        web_config = web_config.with_connect_timeout(Duration::from_secs_f64(secs));
    }
    if let Some(secs) = read_timeout_seconds {
        web_config.read_timeout = Some(Duration::from_secs_f64(secs));
    }
    if let Some(secs) = timeout_seconds {
        web_config = web_config.with_timeout(Duration::from_secs_f64(secs));
    }
    if let Some(headers) = default_headers
        && !headers.is_empty()
    {
        web_config = web_config.with_default_headers(build_header_map(&headers)?);
    }
    Ok(Some(web_config))
}

fn build_header_map(headers: &HashMap<String, String>) -> PyResult<HeaderMap> {
    let mut map = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|err| PyValueError::new_err(format!("invalid header name {name:?}: {err}")))?;
        let header_value = HeaderValue::from_str(value).map_err(|err| {
            PyValueError::new_err(format!("invalid header value for {name:?}: {err}"))
        })?;
        map.insert(header_name, header_value);
    }
    Ok(map)
}

/// Python-facing wrapper around `genai::Client`.
///
/// A `Client` built via `with_api_key` / `with_base_url` /
/// `with_api_key_and_base_url` is bound to a single provider at
/// construction time — rust-genai's `ClientBuilder::with_adapter_kind`
/// drives routing directly from that binding, so callers hand bare
/// model names to `chat`, `achat`, and friends. The zero-arg
/// `Client()` path leaves routing unbound and falls back to
/// rust-genai's model-name prefix inference.
#[pyclass(name = "Client")]
struct PyClient {
    inner: genai::Client,
    /// Canonical provider string the Client was built with
    /// (`"openai"`, `"openai_resp"`, `"anthropic"`, …), or `None`
    /// for a zero-arg unbound `Client()`. Exposed as a read-only
    /// `.provider` attribute so callers can introspect a Client
    /// without keeping parallel state alongside it.
    #[pyo3(get)]
    provider: Option<String>,
}

#[pyclass(name = "ChatMessage", from_py_object)]
#[derive(Clone)]
struct PyChatMessage {
    #[pyo3(get, set)]
    role: String,
    #[pyo3(get, set)]
    content: String,
    /// For assistant messages that contain tool calls (serialized as JSON array of tool calls)
    #[pyo3(get, set)]
    tool_calls: Option<Vec<PyToolCall>>,
    /// For tool response messages
    #[pyo3(get, set)]
    tool_response_call_id: Option<String>,
    /// Per-message cache hint. Accepted: "ephemeral", "ephemeral_5m",
    /// "ephemeral_1h", "ephemeral_24h", "memory". Anthropic applies these
    /// as `cache_control: {type: ephemeral, ttl: ...}` at the content-part
    /// level; OpenAI maps a subset to request-level caching and ignores
    /// the rest. Unknown values are silently dropped.
    #[pyo3(get, set)]
    cache_control: Option<String>,
    /// Encrypted reasoning blobs carried back from a prior assistant
    /// turn (OpenAI Responses `type: "reasoning"` items). When set on an
    /// assistant message, the openai_resp adapter emits each blob as a
    /// top-level `{type: "reasoning", encrypted_content: ...}` input
    /// item before the message it belongs to. Required to keep the
    /// Responses-API prefix cache warm across turns on reasoning models.
    #[pyo3(get, set)]
    thought_signatures: Option<Vec<String>>,
}

#[pymethods]
impl PyChatMessage {
    #[new]
    #[pyo3(signature = (
        role,
        content,
        tool_calls = None,
        tool_response_call_id = None,
        cache_control = None,
        thought_signatures = None,
    ))]
    fn new(
        role: String,
        content: String,
        tool_calls: Option<Vec<PyToolCall>>,
        tool_response_call_id: Option<String>,
        cache_control: Option<String>,
        thought_signatures: Option<Vec<String>>,
    ) -> PyResult<Self> {
        validate_role(&role)?;
        Ok(Self {
            role,
            content,
            tool_calls,
            tool_response_call_id,
            cache_control,
            thought_signatures,
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("role", self.role.clone())?;
        dict.set_item("content", self.content.clone())?;
        match &self.tool_calls {
            Some(tool_calls) => {
                dict.set_item("tool_calls", py_tool_calls_to_py(py, tool_calls)?)?
            }
            None => dict.set_item("tool_calls", py.None())?,
        }
        match &self.tool_response_call_id {
            Some(call_id) => dict.set_item("tool_response_call_id", call_id.clone())?,
            None => dict.set_item("tool_response_call_id", py.None())?,
        }
        match &self.cache_control {
            Some(cc) => dict.set_item("cache_control", cc.clone())?,
            None => dict.set_item("cache_control", py.None())?,
        }
        match &self.thought_signatures {
            Some(sigs) => dict.set_item("thought_signatures", sigs.clone())?,
            None => dict.set_item("thought_signatures", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "ChatRequest", from_py_object)]
#[derive(Clone)]
struct PyChatRequest {
    #[pyo3(get, set)]
    system: Option<String>,
    #[pyo3(get, set)]
    messages: Vec<PyChatMessage>,
    #[pyo3(get, set)]
    tools: Option<Vec<PyTool>>,
    /// OpenAI Responses API: chain a previous server-stored response so the
    /// backend only needs to process the new turn's delta. Pair with `store=True`
    /// on the previous request to make its `response_id` reusable here.
    #[pyo3(get, set)]
    previous_response_id: Option<String>,
    /// OpenAI Responses API: ask the backend to persist this response so its
    /// `response_id` can be used as `previous_response_id` on a follow-up call.
    #[pyo3(get, set)]
    store: Option<bool>,
}

#[pymethods]
impl PyChatRequest {
    #[new]
    #[pyo3(signature = (messages, system = None, tools = None, previous_response_id = None, store = None))]
    fn new(
        messages: Vec<PyChatMessage>,
        system: Option<String>,
        tools: Option<Vec<PyTool>>,
        previous_response_id: Option<String>,
        store: Option<bool>,
    ) -> Self {
        Self {
            system,
            messages,
            tools,
            previous_response_id,
            store,
        }
    }

    fn add_message(&mut self, message: PyChatMessage) {
        self.messages.push(message);
    }

    fn message_count(&self) -> usize {
        self.messages.len()
    }

    fn messages(&self) -> Vec<PyChatMessage> {
        self.messages.clone()
    }
}

#[pyclass(name = "Tool", from_py_object)]
#[derive(Clone)]
struct PyTool {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    description: Option<String>,
    /// JSON Schema for parameters, stored as a JSON string. Accepts either a
    /// `str` (used as-is) or any JSON-serializable Python object (typically a
    /// `dict`, e.g. ``pydantic.BaseModel.model_json_schema()``).
    #[pyo3(get)]
    schema_json: Option<String>,
}

#[pymethods]
impl PyTool {
    #[new]
    #[pyo3(signature = (name, description = None, schema_json = None))]
    fn new(
        name: String,
        description: Option<String>,
        schema_json: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema_json = schema_json.map(coerce_schema_to_json_string).transpose()?;
        Ok(Self {
            name,
            description,
            schema_json,
        })
    }

    #[setter]
    fn set_schema_json(&mut self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.schema_json = value.map(coerce_schema_to_json_string).transpose()?;
        Ok(())
    }
}

#[pyclass(name = "ChatOptions", from_py_object)]
#[derive(Clone, Default)]
struct PyChatOptions {
    #[pyo3(get, set)]
    temperature: Option<f64>,
    #[pyo3(get, set)]
    max_tokens: Option<u32>,
    #[pyo3(get, set)]
    top_p: Option<f64>,
    #[pyo3(get, set)]
    stop_sequences: Vec<String>,
    #[pyo3(get, set)]
    capture_usage: Option<bool>,
    #[pyo3(get, set)]
    capture_content: Option<bool>,
    #[pyo3(get, set)]
    capture_reasoning_content: Option<bool>,
    #[pyo3(get, set)]
    capture_tool_calls: Option<bool>,
    #[pyo3(get, set)]
    capture_raw_body: Option<bool>,
    #[pyo3(get, set)]
    response_json_spec: Option<PyJsonSpec>,
    #[pyo3(get, set)]
    response_json_mode: Option<bool>,
    #[pyo3(get, set)]
    normalize_reasoning_content: Option<bool>,
    #[pyo3(get, set)]
    seed: Option<u64>,
    #[pyo3(get, set)]
    extra_headers: Option<HashMap<String, String>>,
    /// Reasoning effort hint — tells the provider to produce a
    /// reasoning trace. Accepted values: "none", "minimal", "low",
    /// "medium", "high", "xhigh", "max", or "budget:<n>" (token
    /// budget). Without this, capture_reasoning_content=True only
    /// captures reasoning the provider volunteers on its own, which
    /// on OpenAI Responses is typically empty.
    #[pyo3(get, set)]
    reasoning_effort: Option<String>,
    /// OpenAI Responses API prefix-cache key. Holding this constant across
    /// requests in a session lets the backend recognise the cacheable
    /// prefix even when its content hashes shift in cheap ways
    /// (timestamps, formatting). Returned as `cached_tokens` in
    /// `Usage.prompt_tokens_details`.
    #[pyo3(get, set)]
    prompt_cache_key: Option<String>,
}

#[pymethods]
impl PyChatOptions {
    #[new]
    #[pyo3(signature = (
		temperature = None,
		max_tokens = None,
		top_p = None,
		stop_sequences = None,
		capture_usage = None,
		capture_content = None,
		capture_reasoning_content = None,
			capture_tool_calls = None,
			capture_raw_body = None,
			response_json_spec = None,
			response_json_mode = None,
			normalize_reasoning_content = None,
			seed = None,
            extra_headers = None,
            reasoning_effort = None,
            prompt_cache_key = None,
		))]
    fn new(
        temperature: Option<f64>,
        max_tokens: Option<u32>,
        top_p: Option<f64>,
        stop_sequences: Option<Vec<String>>,
        capture_usage: Option<bool>,
        capture_content: Option<bool>,
        capture_reasoning_content: Option<bool>,
        capture_tool_calls: Option<bool>,
        capture_raw_body: Option<bool>,
        response_json_spec: Option<PyJsonSpec>,
        response_json_mode: Option<bool>,
        normalize_reasoning_content: Option<bool>,
        seed: Option<u64>,
        extra_headers: Option<HashMap<String, String>>,
        reasoning_effort: Option<String>,
        prompt_cache_key: Option<String>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            top_p,
            stop_sequences: stop_sequences.unwrap_or_default(),
            capture_usage,
            capture_content,
            capture_reasoning_content,
            capture_tool_calls,
            capture_raw_body,
            response_json_spec,
            response_json_mode,
            normalize_reasoning_content,
            seed,
            extra_headers,
            reasoning_effort,
            prompt_cache_key,
        }
    }
}

#[pyclass(name = "JsonSpec", from_py_object)]
#[derive(Clone)]
struct PyJsonSpec {
    #[pyo3(get, set)]
    name: String,
    /// JSON Schema for the response, stored as a JSON string. Accepts either a
    /// `str` (used as-is) or any JSON-serializable Python object (typically a
    /// `dict`, e.g. ``pydantic.BaseModel.model_json_schema()``).
    #[pyo3(get)]
    schema_json: String,
    #[pyo3(get, set)]
    description: Option<String>,
}

#[pymethods]
impl PyJsonSpec {
    #[new]
    #[pyo3(signature = (name, schema_json, description = None))]
    fn new(
        name: String,
        schema_json: Bound<'_, PyAny>,
        description: Option<String>,
    ) -> PyResult<Self> {
        let schema_json = coerce_schema_to_json_string(schema_json)?;
        Ok(Self {
            name,
            schema_json,
            description,
        })
    }

    #[setter]
    fn set_schema_json(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.schema_json = coerce_schema_to_json_string(value)?;
        Ok(())
    }
}

#[pyclass(name = "ToolCall", from_py_object)]
#[derive(Clone)]
struct PyToolCall {
    #[pyo3(get, set)]
    call_id: String,
    #[pyo3(get, set)]
    fn_name: String,
    #[pyo3(get, set)]
    fn_arguments_json: String,
    #[pyo3(get, set)]
    thought_signatures: Option<Vec<String>>,
}

#[pymethods]
impl PyToolCall {
    #[new]
    #[pyo3(signature = (call_id, fn_name, fn_arguments_json, thought_signatures = None))]
    fn new(
        call_id: String,
        fn_name: String,
        fn_arguments_json: String,
        thought_signatures: Option<Vec<String>>,
    ) -> Self {
        Self {
            call_id,
            fn_name,
            fn_arguments_json,
            thought_signatures,
        }
    }

    #[getter]
    fn fn_arguments(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let parsed = serde_json::from_str::<serde_json::Value>(&self.fn_arguments_json)
            .unwrap_or(serde_json::Value::Null);
        json_value_to_py(py, &parsed)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("call_id", self.call_id.clone())?;
        dict.set_item("fn_name", self.fn_name.clone())?;
        dict.set_item("fn_arguments", self.fn_arguments(py)?)?;
        dict.set_item("fn_arguments_json", self.fn_arguments_json.clone())?;
        match &self.thought_signatures {
            Some(signatures) => dict.set_item("thought_signatures", signatures.clone())?,
            None => dict.set_item("thought_signatures", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "CacheCreationDetails", from_py_object)]
#[derive(Clone, Default)]
struct PyCacheCreationDetails {
    /// Anthropic: tokens written to the 5-minute ephemeral cache on this turn.
    #[pyo3(get, set)]
    ephemeral_5m_tokens: Option<i32>,
    /// Anthropic: tokens written to the 1-hour ephemeral cache on this turn.
    #[pyo3(get, set)]
    ephemeral_1h_tokens: Option<i32>,
}

#[pymethods]
impl PyCacheCreationDetails {
    #[new]
    #[pyo3(signature = (ephemeral_5m_tokens = None, ephemeral_1h_tokens = None))]
    fn new(ephemeral_5m_tokens: Option<i32>, ephemeral_1h_tokens: Option<i32>) -> Self {
        Self {
            ephemeral_5m_tokens,
            ephemeral_1h_tokens,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        match self.ephemeral_5m_tokens {
            Some(v) => dict.set_item("ephemeral_5m_tokens", v)?,
            None => dict.set_item("ephemeral_5m_tokens", py.None())?,
        }
        match self.ephemeral_1h_tokens {
            Some(v) => dict.set_item("ephemeral_1h_tokens", v)?,
            None => dict.set_item("ephemeral_1h_tokens", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "PromptTokensDetails", from_py_object)]
#[derive(Clone, Default)]
struct PyPromptTokensDetails {
    /// Tokens written to the cache on this turn (Anthropic
    /// `cache_creation_input_tokens`). May incur a small surcharge;
    /// subsequent requests benefit via `cached_tokens`.
    #[pyo3(get, set)]
    cache_creation_tokens: Option<i32>,
    /// Per-TTL breakdown of `cache_creation_tokens`, populated only when
    /// the provider returns it (currently Anthropic).
    #[pyo3(get, set)]
    cache_creation_details: Option<PyCacheCreationDetails>,
    /// Tokens served from the cache on this turn — the headline number for
    /// verifying prompt-caching is working (Anthropic
    /// `cache_read_input_tokens`, OpenAI `cached_tokens`).
    #[pyo3(get, set)]
    cached_tokens: Option<i32>,
    #[pyo3(get, set)]
    audio_tokens: Option<i32>,
}

#[pymethods]
impl PyPromptTokensDetails {
    #[new]
    #[pyo3(signature = (
        cache_creation_tokens = None,
        cache_creation_details = None,
        cached_tokens = None,
        audio_tokens = None,
    ))]
    fn new(
        cache_creation_tokens: Option<i32>,
        cache_creation_details: Option<PyCacheCreationDetails>,
        cached_tokens: Option<i32>,
        audio_tokens: Option<i32>,
    ) -> Self {
        Self {
            cache_creation_tokens,
            cache_creation_details,
            cached_tokens,
            audio_tokens,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        match self.cache_creation_tokens {
            Some(v) => dict.set_item("cache_creation_tokens", v)?,
            None => dict.set_item("cache_creation_tokens", py.None())?,
        }
        match &self.cache_creation_details {
            Some(d) => dict.set_item("cache_creation_details", d.to_dict(py)?)?,
            None => dict.set_item("cache_creation_details", py.None())?,
        }
        match self.cached_tokens {
            Some(v) => dict.set_item("cached_tokens", v)?,
            None => dict.set_item("cached_tokens", py.None())?,
        }
        match self.audio_tokens {
            Some(v) => dict.set_item("audio_tokens", v)?,
            None => dict.set_item("audio_tokens", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "CompletionTokensDetails", from_py_object)]
#[derive(Clone, Default)]
struct PyCompletionTokensDetails {
    #[pyo3(get, set)]
    accepted_prediction_tokens: Option<i32>,
    #[pyo3(get, set)]
    rejected_prediction_tokens: Option<i32>,
    /// Reasoning tokens used to produce this turn's answer (Gemini
    /// `thoughts_token_count`, OpenAI o-series reasoning, etc.). Already
    /// included in `completion_tokens`.
    #[pyo3(get, set)]
    reasoning_tokens: Option<i32>,
    #[pyo3(get, set)]
    audio_tokens: Option<i32>,
}

#[pymethods]
impl PyCompletionTokensDetails {
    #[new]
    #[pyo3(signature = (
        accepted_prediction_tokens = None,
        rejected_prediction_tokens = None,
        reasoning_tokens = None,
        audio_tokens = None,
    ))]
    fn new(
        accepted_prediction_tokens: Option<i32>,
        rejected_prediction_tokens: Option<i32>,
        reasoning_tokens: Option<i32>,
        audio_tokens: Option<i32>,
    ) -> Self {
        Self {
            accepted_prediction_tokens,
            rejected_prediction_tokens,
            reasoning_tokens,
            audio_tokens,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        match self.accepted_prediction_tokens {
            Some(v) => dict.set_item("accepted_prediction_tokens", v)?,
            None => dict.set_item("accepted_prediction_tokens", py.None())?,
        }
        match self.rejected_prediction_tokens {
            Some(v) => dict.set_item("rejected_prediction_tokens", v)?,
            None => dict.set_item("rejected_prediction_tokens", py.None())?,
        }
        match self.reasoning_tokens {
            Some(v) => dict.set_item("reasoning_tokens", v)?,
            None => dict.set_item("reasoning_tokens", py.None())?,
        }
        match self.audio_tokens {
            Some(v) => dict.set_item("audio_tokens", v)?,
            None => dict.set_item("audio_tokens", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "Usage", from_py_object)]
#[derive(Clone)]
struct PyUsage {
    #[pyo3(get, set)]
    prompt_tokens: Option<i32>,
    #[pyo3(get, set)]
    completion_tokens: Option<i32>,
    #[pyo3(get, set)]
    total_tokens: Option<i32>,
    /// Breakdown of prompt tokens: cached vs cache-creation vs other.
    /// Inspect ``prompt_tokens_details.cached_tokens`` to verify a
    /// prompt-cache hit on the current turn.
    #[pyo3(get, set)]
    prompt_tokens_details: Option<PyPromptTokensDetails>,
    /// Breakdown of completion tokens: reasoning vs prediction vs audio.
    #[pyo3(get, set)]
    completion_tokens_details: Option<PyCompletionTokensDetails>,
}

#[pymethods]
impl PyUsage {
    #[new]
    #[pyo3(signature = (
        prompt_tokens = None,
        completion_tokens = None,
        total_tokens = None,
        prompt_tokens_details = None,
        completion_tokens_details = None,
    ))]
    fn new(
        prompt_tokens: Option<i32>,
        completion_tokens: Option<i32>,
        total_tokens: Option<i32>,
        prompt_tokens_details: Option<PyPromptTokensDetails>,
        completion_tokens_details: Option<PyCompletionTokensDetails>,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_tokens_details,
            completion_tokens_details,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        match self.prompt_tokens {
            Some(value) => dict.set_item("prompt_tokens", value)?,
            None => dict.set_item("prompt_tokens", py.None())?,
        }
        match self.completion_tokens {
            Some(value) => dict.set_item("completion_tokens", value)?,
            None => dict.set_item("completion_tokens", py.None())?,
        }
        match self.total_tokens {
            Some(value) => dict.set_item("total_tokens", value)?,
            None => dict.set_item("total_tokens", py.None())?,
        }
        match &self.prompt_tokens_details {
            Some(d) => dict.set_item("prompt_tokens_details", d.to_dict(py)?)?,
            None => dict.set_item("prompt_tokens_details", py.None())?,
        }
        match &self.completion_tokens_details {
            Some(d) => dict.set_item("completion_tokens_details", d.to_dict(py)?)?,
            None => dict.set_item("completion_tokens_details", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "ChatResponse", from_py_object)]
/// Pythonic wrapper for rust-genai's `ChatResponse`:
///   - `content` is the single source of truth (list of lowercase-keyed
///     content-part dicts).
///   - `text`, `first_text`, `texts`, and `tool_calls` are read-only derived
///     properties.
#[derive(Clone)]
struct PyChatResponse {
    inner_content: genai::chat::MessageContent,
    #[pyo3(get)]
    reasoning_content: Option<String>,
    #[pyo3(get)]
    model_adapter_kind: String,
    #[pyo3(get)]
    model_name: String,
    #[pyo3(get)]
    provider_model_adapter_kind: String,
    #[pyo3(get)]
    provider_model_name: String,
    #[pyo3(get)]
    usage: PyUsage,
    /// OpenAI Responses API: the server-stored response id, present when the
    /// request was sent with `store=True`. Pass it as `previous_response_id`
    /// on the next request to chain calls server-side.
    #[pyo3(get)]
    response_id: Option<String>,
}

#[pymethods]
impl PyChatResponse {
    /// Build a `ChatResponse` from Python — primarily for test fixtures.
    ///
    /// ``content`` accepts the same lowercase-keyed content-part dict form
    /// the client emits (``[{"text": "..."}, {"tool_call": {...}}, ...]``)
    /// and runs through ``pythonize → serde_json::Value → MessageContent``.
    /// Normal response objects come back from ``client.achat`` /
    /// ``client.achat_via_stream``; production code should not need this.
    #[new]
    #[pyo3(signature = (
        content = None,
        reasoning_content = None,
        model_adapter_kind = None,
        model_name = None,
        provider_model_adapter_kind = None,
        provider_model_name = None,
        usage = None,
        response_id = None,
    ))]
    fn new(
        content: Option<Bound<'_, PyAny>>,
        reasoning_content: Option<String>,
        model_adapter_kind: Option<String>,
        model_name: Option<String>,
        provider_model_adapter_kind: Option<String>,
        provider_model_name: Option<String>,
        usage: Option<PyUsage>,
        response_id: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner_content: coerce_message_content(content)?,
            reasoning_content,
            model_adapter_kind: model_adapter_kind.unwrap_or_default(),
            model_name: model_name.unwrap_or_default(),
            provider_model_adapter_kind: provider_model_adapter_kind.unwrap_or_default(),
            provider_model_name: provider_model_name.unwrap_or_default(),
            usage: usage.unwrap_or(PyUsage {
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            response_id,
        })
    }

    /// Full content as a list of lowercase-keyed content-part dicts
    /// (``[{"text": "..."}, {"tool_call": {...}}, ...]``).
    #[getter]
    fn content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut value = serde_json::to_value(&self.inner_content)
            .map_err(|err| PyRuntimeError::new_err(format!("serialize content: {err}")))?;
        lowercase_content_part_keys(&mut value);
        pythonize::pythonize(py, &value)
            .map_err(|err| PyRuntimeError::new_err(format!("pythonize content: {err}")))
    }

    /// Return the first text segment in the response content, if any.
    #[getter]
    fn text(&self) -> Option<String> {
        self.inner_content.first_text().map(str::to_string)
    }

    /// Alias for ``text`` with rust-genai's explicit naming.
    #[getter]
    fn first_text(&self) -> Option<String> {
        self.text()
    }

    /// Return every text segment in the response content, in order.
    #[getter]
    fn texts(&self) -> Vec<String> {
        self.inner_content
            .texts()
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    /// Return every tool call in the response content, in order.
    #[getter]
    fn tool_calls(&self) -> PyResult<Vec<PyToolCall>> {
        self.inner_content
            .tool_calls()
            .into_iter()
            .map(to_py_tool_call)
            .collect()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("content", self.content(py)?)?;
        match self.text() {
            Some(text) => dict.set_item("text", text)?,
            None => dict.set_item("text", py.None())?,
        }
        match self.first_text() {
            Some(text) => dict.set_item("first_text", text)?,
            None => dict.set_item("first_text", py.None())?,
        }
        dict.set_item("texts", self.texts())?;
        dict.set_item("tool_calls", py_tool_calls_to_py(py, &self.tool_calls()?)?)?;
        match &self.reasoning_content {
            Some(reasoning) => dict.set_item("reasoning_content", reasoning.clone())?,
            None => dict.set_item("reasoning_content", py.None())?,
        }
        dict.set_item("model_adapter_kind", self.model_adapter_kind.clone())?;
        dict.set_item("model_name", self.model_name.clone())?;
        dict.set_item(
            "provider_model_adapter_kind",
            self.provider_model_adapter_kind.clone(),
        )?;
        dict.set_item("provider_model_name", self.provider_model_name.clone())?;
        dict.set_item("usage", self.usage.to_dict(py)?)?;
        match &self.response_id {
            Some(id) => dict.set_item("response_id", id.clone())?,
            None => dict.set_item("response_id", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "StreamEnd", from_py_object)]
/// Pythonic wrapper for rust-genai's `StreamEnd`: one source of truth
/// (`captured_content: Option<MessageContent>`) plus read-only derived
/// properties.
#[derive(Clone)]
struct PyStreamEnd {
    captured_content: Option<genai::chat::MessageContent>,
    #[pyo3(get)]
    captured_usage: Option<PyUsage>,
    #[pyo3(get)]
    captured_reasoning_content: Option<String>,
    /// OpenAI Responses API: the server-stored response id captured from
    /// the terminal stream event (when the request was sent with
    /// `store=True`). Feed back as `previous_response_id` on the next
    /// request to chain calls.
    #[pyo3(get)]
    captured_response_id: Option<String>,
    /// Encrypted reasoning blobs (`type:"reasoning"` items) from the
    /// streamed response. On OpenAI Responses-API reasoning models these
    /// must be carried back into the next turn's input — as
    /// `ChatMessage(thought_signatures=...)` on the assistant turn — to
    /// keep the prefix cache warm. (Originally Gemini's "thought
    /// signatures" terminology; rust-genai reuses the same slot for the
    /// OpenAI encrypted_content blob.)
    #[pyo3(get)]
    captured_thought_signatures: Option<Vec<String>>,
}

#[pymethods]
impl PyStreamEnd {
    /// Captured content as ``list[dict]`` (same shape as
    /// ``ChatResponse.content``), or ``None`` when the stream carried no
    /// content capture.
    #[getter]
    fn captured_content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(content) = self.captured_content.as_ref() else {
            return Ok(py.None().into_bound(py));
        };
        let mut value = serde_json::to_value(content)
            .map_err(|err| PyRuntimeError::new_err(format!("serialize content: {err}")))?;
        lowercase_content_part_keys(&mut value);
        pythonize::pythonize(py, &value)
            .map_err(|err| PyRuntimeError::new_err(format!("pythonize content: {err}")))
    }

    #[getter]
    fn captured_text(&self) -> Option<String> {
        self.captured_content
            .as_ref()
            .and_then(|c| c.first_text().map(str::to_string))
    }

    #[getter]
    fn captured_first_text(&self) -> Option<String> {
        self.captured_text()
    }

    #[getter]
    fn captured_texts(&self) -> Vec<String> {
        self.captured_content
            .as_ref()
            .map(|c| c.texts().into_iter().map(str::to_string).collect())
            .unwrap_or_default()
    }

    #[getter]
    fn captured_tool_calls(&self) -> PyResult<Vec<PyToolCall>> {
        let Some(content) = self.captured_content.as_ref() else {
            return Ok(Vec::new());
        };
        content
            .tool_calls()
            .into_iter()
            .map(to_py_tool_call)
            .collect()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("captured_content", self.captured_content(py)?)?;
        match self.captured_text() {
            Some(text) => dict.set_item("captured_text", text)?,
            None => dict.set_item("captured_text", py.None())?,
        }
        match self.captured_first_text() {
            Some(text) => dict.set_item("captured_first_text", text)?,
            None => dict.set_item("captured_first_text", py.None())?,
        }
        dict.set_item("captured_texts", self.captured_texts())?;
        dict.set_item(
            "captured_tool_calls",
            py_tool_calls_to_py(py, &self.captured_tool_calls()?)?,
        )?;
        match &self.captured_usage {
            Some(usage) => dict.set_item("captured_usage", usage.to_dict(py)?)?,
            None => dict.set_item("captured_usage", py.None())?,
        }
        match &self.captured_reasoning_content {
            Some(reasoning) => dict.set_item("captured_reasoning_content", reasoning.clone())?,
            None => dict.set_item("captured_reasoning_content", py.None())?,
        }
        match &self.captured_response_id {
            Some(id) => dict.set_item("captured_response_id", id.clone())?,
            None => dict.set_item("captured_response_id", py.None())?,
        }
        match &self.captured_thought_signatures {
            Some(sigs) => dict.set_item("captured_thought_signatures", sigs.clone())?,
            None => dict.set_item("captured_thought_signatures", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pyclass(name = "ChatStreamEvent", from_py_object)]
#[derive(Clone)]
struct PyChatStreamEvent {
    #[pyo3(get, set)]
    kind: String,
    #[pyo3(get, set)]
    content: Option<String>,
    #[pyo3(get, set)]
    tool_call: Option<PyToolCall>,
    #[pyo3(get, set)]
    end: Option<PyStreamEnd>,
}

#[pymethods]
impl PyChatStreamEvent {
    #[new]
    #[pyo3(signature = (kind, content = None, tool_call = None, end = None))]
    fn new(
        kind: String,
        content: Option<String>,
        tool_call: Option<PyToolCall>,
        end: Option<PyStreamEnd>,
    ) -> Self {
        Self {
            kind,
            content,
            tool_call,
            end,
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("kind", self.kind.clone())?;
        match &self.content {
            Some(content) => dict.set_item("content", content.clone())?,
            None => dict.set_item("content", py.None())?,
        }
        match &self.tool_call {
            Some(tool_call) => dict.set_item("tool_call", tool_call.to_dict(py)?)?,
            None => dict.set_item("tool_call", py.None())?,
        }
        match &self.end {
            Some(end) => dict.set_item("end", end.to_dict(py)?)?,
            None => dict.set_item("end", py.None())?,
        }
        Ok(dict.into_any())
    }
}

#[pymethods]
impl PyClient {
    #[new]
    fn new() -> Self {
        Self {
            inner: genai::Client::default(),
            provider: None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.provider {
            Some(provider) => format!("Client(provider={provider:?})"),
            None => "Client(unbound)".to_string(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        api_key,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_api_key(
        provider: String,
        api_key: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            Some(api_key),
            None,
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        api_key,
        base_url,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_api_key_and_base_url(
        provider: String,
        api_key: String,
        base_url: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            Some(api_key),
            Some(base_url),
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (
        provider,
        base_url,
        *,
        connect_timeout_seconds = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        read_timeout_seconds = None,
        timeout_seconds = None,
        default_headers = None,
    ))]
    fn with_base_url(
        provider: String,
        base_url: String,
        connect_timeout_seconds: Option<f64>,
        read_timeout_seconds: Option<f64>,
        timeout_seconds: Option<f64>,
        default_headers: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        build_client_with_overrides(
            provider,
            None,
            Some(base_url),
            connect_timeout_seconds,
            read_timeout_seconds,
            timeout_seconds,
            default_headers,
        )
    }

    #[staticmethod]
    fn with_request_override(
        provider: String,
        url: String,
        headers: HashMap<String, String>,
    ) -> PyResult<Self> {
        build_client_with_request_override(provider, url, headers)
    }

    #[pyo3(signature = (model, request, options = None))]
    fn chat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyChatResponse>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;

        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|err| {
                PyRuntimeError::new_err(format!("failed to build Tokio runtime: {err}"))
            })?;

        let chat_res = runtime
            .block_on(self.inner.exec_chat(model, chat_req, chat_options.as_ref()))
            .map_err(to_runtime_error)?;

        to_py_chat_response(py, chat_res)
    }

    #[pyo3(signature = (model, request, options = None))]
    fn achat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let chat_res = client
                .exec_chat(model, chat_req, chat_options.as_ref())
                .await
                .map_err(to_runtime_error)?;

            Python::attach(|py| to_py_chat_response(py, chat_res))
        })
    }

    #[pyo3(signature = (model, request, options = None))]
    fn astream_chat<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let chat_options = coerce_chat_options(options)?;
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model, chat_req, chat_options.as_ref())
                .await
                .map_err(to_runtime_error)?;

            Python::attach(|py| {
                Py::new(py, PyChatStream::new(stream_res.stream)).map_err(|err| {
                    PyRuntimeError::new_err(format!("failed to build stream object: {err}"))
                })
            })
        })
    }

    /// Run the request as a stream and return a fully-collected
    /// :class:`ChatResponse`. Useful for providers whose backend only
    /// accepts ``stream=true`` (e.g. OpenAI's Responses API) but whose
    /// callers still want a single synchronous result.
    ///
    /// `capture_content`, `capture_reasoning_content`,
    /// `capture_tool_calls`, and `capture_usage` are forced on so the
    /// terminal `StreamEnd` carries the aggregated content — callers do
    /// not need to configure capture themselves.
    #[pyo3(signature = (model, request, options = None))]
    fn achat_via_stream<'py>(
        &self,
        py: Python<'py>,
        model: String,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chat_req = coerce_chat_request(request)?;
        let mut chat_options = coerce_chat_options(options)?.unwrap_or_default();
        chat_options.capture_content = Some(true);
        chat_options.capture_reasoning_content = Some(true);
        chat_options.capture_tool_calls = Some(true);
        chat_options.capture_usage = Some(true);
        let client = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream_res = client
                .exec_chat_stream(model, chat_req, Some(&chat_options))
                .await
                .map_err(to_runtime_error)?;

            let model_iden = stream_res.model_iden.clone();
            let mut stream = stream_res.stream;
            let mut end: Option<StreamEnd> = None;
            while let Some(event) = stream.next().await {
                let event = event.map_err(to_runtime_error)?;
                if let ChatStreamEvent::End(e) = event {
                    end = Some(e);
                    break;
                }
            }

            let end = end.ok_or_else(|| {
                PyRuntimeError::new_err("stream ended without a terminal End event")
            })?;
            let content = end
                .captured_content
                .unwrap_or_else(genai::chat::MessageContent::default);
            let reasoning_content = end.captured_reasoning_content;
            let usage = end.captured_usage.unwrap_or_default();
            let response_id = end.captured_response_id;

            Python::attach(|py| {
                Py::new(
                    py,
                    PyChatResponse {
                        inner_content: content,
                        reasoning_content,
                        model_adapter_kind: model_iden.adapter_kind.to_string(),
                        model_name: model_iden.model_name.to_string(),
                        provider_model_adapter_kind: model_iden.adapter_kind.to_string(),
                        provider_model_name: model_iden.model_name.to_string(),
                        usage: to_py_usage(&usage),
                        response_id,
                    },
                )
            })
        })
    }
}

#[pyclass(name = "_ChatStream")]
struct PyChatStream {
    inner: Arc<Mutex<Option<ChatStream>>>,
}

impl PyChatStream {
    fn new(stream: ChatStream) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(stream))),
        }
    }
}

#[pymethods]
impl PyChatStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;

            let stream = match guard.as_mut() {
                Some(stream) => stream,
                None => {
                    return Err(PyStopAsyncIteration::new_err("stream exhausted"));
                }
            };

            match stream.next().await {
                Some(Ok(event)) => Python::attach(|py| to_py_stream_event(py, event)),
                Some(Err(err)) => Err(to_runtime_error(err)),
                None => {
                    *guard = None;
                    Err(PyStopAsyncIteration::new_err("stream exhausted"))
                }
            }
        })
    }
}

#[pymodule]
fn _genai_pyo3(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyClient>()?;
    module.add_class::<PyChatMessage>()?;
    module.add_class::<PyChatRequest>()?;
    module.add_class::<PyChatOptions>()?;
    module.add_class::<PyJsonSpec>()?;
    module.add_class::<PyTool>()?;
    module.add_class::<PyToolCall>()?;
    module.add_class::<PyUsage>()?;
    module.add_class::<PyPromptTokensDetails>()?;
    module.add_class::<PyCompletionTokensDetails>()?;
    module.add_class::<PyCacheCreationDetails>()?;
    module.add_class::<PyChatResponse>()?;
    module.add_class::<PyStreamEnd>()?;
    module.add_class::<PyChatStreamEvent>()?;
    module.add_class::<PyChatStream>()?;
    Ok(())
}

fn parse_chat_role(role: &str) -> PyResult<ChatRole> {
    match role {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        "tool" => Ok(ChatRole::Tool),
        _ => Err(PyValueError::new_err(format!(
            "invalid chat role '{role}'. expected one of: system, user, assistant, tool"
        ))),
    }
}

/// Coerce a Python value into a JSON Schema string. Accepts either a Python
/// `str` (used as-is) or any JSON-serializable object (typically a `dict`
/// such as ``pydantic.BaseModel.model_json_schema()``).
fn coerce_schema_to_json_string(value: Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = value.extract::<String>() {
        return Ok(s);
    }
    let json_value: serde_json::Value = depythonize(&value).map_err(|err| {
        PyValueError::new_err(format!(
            "schema must be a JSON string or a JSON-serializable object: {err}"
        ))
    })?;
    serde_json::to_string(&json_value).map_err(|err| {
        PyValueError::new_err(format!("failed to serialize schema to JSON: {err}"))
    })
}

fn json_value_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(value) => {
            Ok((*value).into_pyobject(py)?.to_owned().unbind().into_any())
        }
        serde_json::Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                Ok(integer.into_pyobject(py)?.unbind().into_any())
            } else if let Some(integer) = value.as_u64() {
                Ok(integer.into_pyobject(py)?.unbind().into_any())
            } else if let Some(float) = value.as_f64() {
                Ok(float.into_pyobject(py)?.unbind().into_any())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(value) => {
            Ok(value.clone().into_pyobject(py)?.unbind().into_any())
        }
        serde_json::Value::Array(values) => {
            let list = PyList::empty(py);
            for value in values {
                list.append(json_value_to_py(py, value)?)?;
            }
            Ok(list.unbind().into_any())
        }
        serde_json::Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, json_value_to_py(py, value)?)?;
            }
            Ok(dict.unbind().into_any())
        }
    }
}

fn py_tool_calls_to_py<'py>(
    py: Python<'py>,
    tool_calls: &[PyToolCall],
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::empty(py);
    for tool_call in tool_calls {
        list.append(tool_call.to_dict(py)?)?;
    }
    Ok(list.into_any())
}

fn validate_role(role: &str) -> PyResult<()> {
    parse_chat_role(role).map(|_| ())
}

impl TryFrom<PyChatMessage> for ChatMessage {
    type Error = PyErr;

    fn try_from(message: PyChatMessage) -> Result<Self, Self::Error> {
        let role = parse_chat_role(&message.role)?;
        let options = message
            .cache_control
            .as_deref()
            .and_then(parse_cache_control)
            .map(|cc| MessageOptions { cache_control: Some(cc) });
        let thought_signatures = message.thought_signatures.clone();

        let mut chat_message = match role {
            ChatRole::Assistant if message.tool_calls.is_some() => {
                // Assistant message with tool calls
                let py_calls = message.tool_calls.unwrap();
                let calls: Vec<ToolCall> = py_calls
                    .into_iter()
                    .map(|tc| {
                        let args: serde_json::Value = serde_json::from_str(&tc.fn_arguments_json)
                            .unwrap_or(serde_json::Value::Null);
                        ToolCall {
                            call_id: tc.call_id,
                            fn_name: tc.fn_name,
                            fn_arguments: args,
                            thought_signatures: tc.thought_signatures,
                        }
                    })
                    .collect();
                ChatMessage::from(calls)
            }
            ChatRole::Tool => {
                // Tool response message
                let call_id = message.tool_response_call_id.unwrap_or_default();
                let response = ToolResponse::new(call_id, message.content);
                ChatMessage::from(response)
            }
            _ => {
                // Regular text message (system, user, assistant without tool calls)
                ChatMessage {
                    role,
                    content: message.content.into(),
                    options: None,
                }
            }
        };

        // Attach encrypted reasoning blobs as ThoughtSignature parts so the
        // openai_resp adapter can emit them as top-level `type: "reasoning"`
        // input items in the next request. Order: signatures go FIRST in the
        // content vector so they precede text / tool calls when serialized.
        if let Some(sigs) = thought_signatures
            && !sigs.is_empty()
        {
            let parts = chat_message.content.parts().clone();
            let mut new_parts: Vec<genai::chat::ContentPart> = sigs
                .into_iter()
                .map(genai::chat::ContentPart::ThoughtSignature)
                .collect();
            new_parts.extend(parts);
            chat_message.content = genai::chat::MessageContent::from_parts(new_parts);
        }

        Ok(match options {
            Some(opts) => chat_message.with_options(opts),
            None => chat_message,
        })
    }
}

/// Accept either a `PyChatRequest` pyclass instance or any Python object that
/// depythonizes into a `serde_json::Value` matching the rust-genai
/// `ChatRequest` serde shape.
///
/// The shape is isomorphic to the native serde form; as an ergonomic
/// affordance, callers may write the enum variant names in Python-idiomatic
/// lowercase (`"user"`, `"text"`, `"tool_call"`, ...) and the coercion
/// layer title-cases them back into the `ChatRole` / `ContentPart` variant
/// names serde expects. Native title-case keys still pass through unchanged.
fn coerce_chat_request(obj: Bound<'_, PyAny>) -> PyResult<ChatRequest> {
    if let Ok(typed) = obj.extract::<PyChatRequest>() {
        return ChatRequest::try_from(typed);
    }
    let mut value: serde_json::Value = depythonize(&obj).map_err(|err| {
        PyValueError::new_err(format!(
            "chat request must be a ChatRequest or a JSON-compatible dict: {err}"
        ))
    })?;
    normalize_chat_request_value(&mut value)?;
    serde_json::from_value(value).map_err(|err| {
        PyValueError::new_err(format!("chat request dict does not match schema: {err}"))
    })
}

const ROLES_LOWER: [&str; 4] = ["system", "user", "assistant", "tool"];
const CONTENT_PART_KEYS_LOWER: [&str; 6] = [
    "text",
    "binary",
    "tool_call",
    "tool_response",
    "thought_signature",
    "reasoning_content",
];

fn title_case_role(s: &str) -> Option<&'static str> {
    match s {
        "system" => Some("System"),
        "user" => Some("User"),
        "assistant" => Some("Assistant"),
        "tool" => Some("Tool"),
        _ => None,
    }
}

fn title_case_content_part(s: &str) -> Option<&'static str> {
    match s {
        "text" => Some("Text"),
        "binary" => Some("Binary"),
        "tool_call" => Some("ToolCall"),
        "tool_response" => Some("ToolResponse"),
        "thought_signature" => Some("ThoughtSignature"),
        "reasoning_content" => Some("ReasoningContent"),
        _ => None,
    }
}

/// Inverse of `title_case_content_part`: rewrite the externally-tagged
/// `ContentPart` variant names in a serialized `MessageContent` value from
/// Rust title-case to the Python-facing lowercase form. Variants we don't
/// know are left untouched (forward-compat with upstream rust-genai adding
/// new parts).
fn lowercase_content_part_keys(value: &mut serde_json::Value) {
    let Some(parts) = value.as_array_mut() else {
        return;
    };
    for part in parts {
        let Some(obj) = part.as_object_mut() else {
            continue;
        };
        let Some(title_key) = obj.keys().next().cloned() else {
            continue;
        };
        let lower = match title_key.as_str() {
            "Text" => "text",
            "Binary" => "binary",
            "ToolCall" => "tool_call",
            "ToolResponse" => "tool_response",
            "ThoughtSignature" => "thought_signature",
            "ReasoningContent" => "reasoning_content",
            _ => continue,
        };
        let inner = obj.remove(&title_key).unwrap();
        obj.insert(lower.to_string(), inner);
    }
}

/// Strictly require lowercase variant names and title-case them into the
/// serde form. Any other spelling (including native title-case) is a
/// ValueError — rust-genai's enum names are an implementation detail; the
/// dict shape is documented lowercase-only.
fn normalize_chat_request_value(value: &mut serde_json::Value) -> PyResult<()> {
    let Some(messages) = value.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    for (msg_idx, msg) in messages.iter_mut().enumerate() {
        let Some(obj) = msg.as_object_mut() else {
            continue;
        };
        if let Some(role_value) = obj.get("role") {
            if let Some(role_str) = role_value.as_str() {
                let title = title_case_role(role_str).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "messages[{msg_idx}].role: expected one of {:?}, got {role_str:?}",
                        ROLES_LOWER
                    ))
                })?;
                obj.insert("role".to_string(), serde_json::Value::String(title.into()));
            }
        }
        let Some(parts) = obj.get_mut("content").and_then(|v| v.as_array_mut()) else {
            continue;
        };
        normalize_content_parts(parts, &format!("messages[{msg_idx}].content"))?;
    }
    Ok(())
}

/// Title-case the externally-tagged variant keys on a top-level content-part
/// list (lowercase ``text`` → ``Text``, etc.) so serde can deserialize it
/// into ``Vec<ContentPart>`` / ``MessageContent``. Errors mention the given
/// path (``messages[N].content`` for request-side, ``content`` for the
/// response-side constructor) so callers get a clear pointer on malformed
/// input.
fn normalize_content_parts(parts: &mut [serde_json::Value], path: &str) -> PyResult<()> {
    for (part_idx, part) in parts.iter_mut().enumerate() {
        let Some(part_obj) = part.as_object_mut() else {
            continue;
        };
        if part_obj.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "{path}[{part_idx}]: expected a single-key dict ({:?}), got {} keys",
                CONTENT_PART_KEYS_LOWER,
                part_obj.len()
            )));
        }
        let lower_key = part_obj.keys().next().cloned().unwrap();
        let inner = part_obj.remove(&lower_key).unwrap();
        let title = title_case_content_part(&lower_key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{path}[{part_idx}]: expected one of {:?}, got {:?}",
                CONTENT_PART_KEYS_LOWER, lower_key
            ))
        })?;
        part_obj.insert(title.into(), inner);
    }
    Ok(())
}

/// Depythonize a Python object (or None) into a ``MessageContent``. Accepts
/// the same lowercase-keyed content-part dict form the client emits.
fn coerce_message_content(obj: Option<Bound<'_, PyAny>>) -> PyResult<genai::chat::MessageContent> {
    let Some(obj) = obj else {
        return Ok(genai::chat::MessageContent::default());
    };
    if obj.is_none() {
        return Ok(genai::chat::MessageContent::default());
    }
    let mut value: serde_json::Value = depythonize(&obj).map_err(|err| {
        PyValueError::new_err(format!(
            "content must be a list of content-part dicts: {err}"
        ))
    })?;
    let parts = value
        .as_array_mut()
        .ok_or_else(|| PyValueError::new_err("content must be a list of content-part dicts"))?;
    normalize_content_parts(parts, "content")?;
    serde_json::from_value(value)
        .map_err(|err| PyValueError::new_err(format!("content list does not match schema: {err}")))
}

/// Accept an optional `PyChatOptions` pyclass or a JSON-compatible dict.
fn coerce_chat_options(obj: Option<Bound<'_, PyAny>>) -> PyResult<Option<ChatOptions>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(typed) = obj.extract::<PyChatOptions>() {
        return Ok(Some(ChatOptions::from(typed)));
    }
    let value: serde_json::Value = depythonize(&obj).map_err(|err| {
        PyValueError::new_err(format!(
            "chat options must be a ChatOptions or a JSON-compatible dict: {err}"
        ))
    })?;
    serde_json::from_value(value).map(Some).map_err(|err| {
        PyValueError::new_err(format!("chat options dict does not match schema: {err}"))
    })
}

impl TryFrom<PyChatRequest> for ChatRequest {
    type Error = PyErr;

    fn try_from(request: PyChatRequest) -> Result<Self, Self::Error> {
        let messages = request
            .messages
            .into_iter()
            .map(ChatMessage::try_from)
            .collect::<PyResult<Vec<_>>>()?;

        let tools = request.tools.map(|py_tools| {
            py_tools
                .into_iter()
                .map(|t| {
                    let mut tool = Tool::new(t.name);
                    if let Some(desc) = t.description {
                        tool = tool.with_description(desc);
                    }
                    if let Some(schema_str) = t.schema_json {
                        if let Ok(schema) = serde_json::from_str::<serde_json::Value>(&schema_str) {
                            tool = tool.with_schema(schema);
                        }
                    }
                    tool
                })
                .collect()
        });

        Ok(Self {
            system: request.system,
            messages,
            tools,
            previous_response_id: request.previous_response_id,
            store: request.store,
        })
    }
}

impl From<PyChatOptions> for ChatOptions {
    fn from(options: PyChatOptions) -> Self {
        let response_format = if let Some(spec) = options.response_json_spec {
            let schema = serde_json::from_str::<serde_json::Value>(&spec.schema_json)
                .unwrap_or(serde_json::Value::Null);
            let mut json_spec = JsonSpec::new(spec.name, schema);
            if let Some(description) = spec.description {
                json_spec = json_spec.with_description(description);
            }
            Some(ChatResponseFormat::JsonSpec(json_spec))
        } else if options.response_json_mode.unwrap_or(false) {
            Some(ChatResponseFormat::JsonMode)
        } else {
            None
        };

        Self {
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            top_p: options.top_p,
            stop_sequences: options.stop_sequences,
            capture_usage: options.capture_usage,
            capture_content: options.capture_content,
            capture_reasoning_content: options.capture_reasoning_content,
            capture_tool_calls: options.capture_tool_calls,
            capture_raw_body: options.capture_raw_body,
            response_format,
            normalize_reasoning_content: options.normalize_reasoning_content,
            seed: options.seed,
            extra_headers: options.extra_headers.map(Into::into),
            reasoning_effort: options
                .reasoning_effort
                .as_deref()
                .and_then(parse_reasoning_effort),
            prompt_cache_key: options.prompt_cache_key,
            ..Default::default()
        }
    }
}

/// Parse a case-insensitive effort keyword into a `ReasoningEffort`.
///
/// Accepted: "none", "minimal", "low", "medium", "high", "xhigh",
/// "max", or "budget:<n>" where `<n>` is a non-negative u32 token
/// budget. Returns `None` for unrecognized values so callers get a
/// silent no-op rather than a hard error — matches the rest of
/// ChatOptions' "everything is optional" pattern.
fn parse_reasoning_effort(raw: &str) -> Option<ReasoningEffort> {
    let trimmed = raw.trim().to_ascii_lowercase();
    if let Some(budget) = trimmed.strip_prefix("budget:") {
        return budget
            .trim()
            .parse::<u32>()
            .ok()
            .map(ReasoningEffort::Budget);
    }
    match trimmed.as_str() {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// Parse a case-insensitive cache-control keyword into a `CacheControl`.
///
/// Accepted: "ephemeral" (default 5m), "memory", "ephemeral_5m",
/// "ephemeral_1h", "ephemeral_24h". Returns `None` for unrecognized
/// values so callers get a silent no-op rather than a hard error —
/// matches the rest of ChatOptions' "everything is optional" pattern.
fn parse_cache_control(raw: &str) -> Option<CacheControl> {
    let trimmed = raw.trim().to_ascii_lowercase();
    match trimmed.as_str() {
        "ephemeral" => Some(CacheControl::Ephemeral),
        "memory" => Some(CacheControl::Memory),
        "ephemeral_5m" | "5m" => Some(CacheControl::Ephemeral5m),
        "ephemeral_1h" | "1h" => Some(CacheControl::Ephemeral1h),
        "ephemeral_24h" | "24h" => Some(CacheControl::Ephemeral24h),
        _ => None,
    }
}

fn to_py_cache_creation_details(d: &CacheCreationDetails) -> PyCacheCreationDetails {
    PyCacheCreationDetails {
        ephemeral_5m_tokens: d.ephemeral_5m_tokens,
        ephemeral_1h_tokens: d.ephemeral_1h_tokens,
    }
}

fn to_py_prompt_tokens_details(d: &PromptTokensDetails) -> PyPromptTokensDetails {
    PyPromptTokensDetails {
        cache_creation_tokens: d.cache_creation_tokens,
        cache_creation_details: d.cache_creation_details.as_ref().map(to_py_cache_creation_details),
        cached_tokens: d.cached_tokens,
        audio_tokens: d.audio_tokens,
    }
}

fn to_py_completion_tokens_details(d: &CompletionTokensDetails) -> PyCompletionTokensDetails {
    PyCompletionTokensDetails {
        accepted_prediction_tokens: d.accepted_prediction_tokens,
        rejected_prediction_tokens: d.rejected_prediction_tokens,
        reasoning_tokens: d.reasoning_tokens,
        audio_tokens: d.audio_tokens,
    }
}

fn to_py_usage(usage: &Usage) -> PyUsage {
    PyUsage {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        prompt_tokens_details: usage
            .prompt_tokens_details
            .as_ref()
            .map(to_py_prompt_tokens_details),
        completion_tokens_details: usage
            .completion_tokens_details
            .as_ref()
            .map(to_py_completion_tokens_details),
    }
}

fn to_py_tool_call(tool_call: &ToolCall) -> PyResult<PyToolCall> {
    let fn_arguments_json = serde_json::to_string(&tool_call.fn_arguments).map_err(|err| {
        PyRuntimeError::new_err(format!("failed to serialize tool call arguments: {err}"))
    })?;

    Ok(PyToolCall {
        call_id: tool_call.call_id.clone(),
        fn_name: tool_call.fn_name.clone(),
        fn_arguments_json,
        thought_signatures: tool_call.thought_signatures.clone(),
    })
}

fn to_py_chat_response(
    py: Python<'_>,
    response: genai::chat::ChatResponse,
) -> PyResult<Py<PyChatResponse>> {
    Py::new(
        py,
        PyChatResponse {
            inner_content: response.content,
            reasoning_content: response.reasoning_content,
            model_adapter_kind: response.model_iden.adapter_kind.to_string(),
            model_name: response.model_iden.model_name.to_string(),
            provider_model_adapter_kind: response.provider_model_iden.adapter_kind.to_string(),
            provider_model_name: response.provider_model_iden.model_name.to_string(),
            usage: to_py_usage(&response.usage),
            response_id: response.response_id,
        },
    )
}

fn to_py_stream_end(end: StreamEnd) -> PyResult<PyStreamEnd> {
    // Lift encrypted reasoning blobs into their own field so Python callers
    // can pluck them out without scanning content parts. Rust-genai stores
    // them as `ContentPart::ThoughtSignature` parts inside `captured_content`.
    let captured_thought_signatures: Option<Vec<String>> = end
        .captured_thought_signatures()
        .map(|v| v.into_iter().map(String::from).collect());
    Ok(PyStreamEnd {
        captured_content: end.captured_content,
        captured_usage: end.captured_usage.as_ref().map(to_py_usage),
        captured_reasoning_content: end.captured_reasoning_content,
        captured_response_id: end.captured_response_id,
        captured_thought_signatures,
    })
}

fn to_py_stream_event(py: Python<'_>, event: ChatStreamEvent) -> PyResult<Py<PyChatStreamEvent>> {
    let (kind, content, tool_call, end) = match event {
        ChatStreamEvent::Start => ("start".to_string(), None, None, None),
        ChatStreamEvent::Chunk(chunk) => ("chunk".to_string(), Some(chunk.content), None, None),
        ChatStreamEvent::ReasoningChunk(chunk) => (
            "reasoning_chunk".to_string(),
            Some(chunk.content),
            None,
            None,
        ),
        ChatStreamEvent::ThoughtSignatureChunk(chunk) => (
            "thought_signature_chunk".to_string(),
            Some(chunk.content),
            None,
            None,
        ),
        ChatStreamEvent::ToolCallChunk(chunk) => (
            "tool_call_chunk".to_string(),
            None,
            Some(to_py_tool_call(&chunk.tool_call)?),
            None,
        ),
        ChatStreamEvent::End(stream_end) => (
            "end".to_string(),
            None,
            None,
            Some(to_py_stream_end(stream_end)?),
        ),
    };

    Py::new(
        py,
        PyChatStreamEvent {
            kind,
            content,
            tool_call,
            end,
        },
    )
}

fn parse_adapter_kind(provider: &str) -> PyResult<AdapterKind> {
    let normalized = provider.trim().to_ascii_lowercase();
    AdapterKind::from_lower_str(&normalized).ok_or_else(|| {
		PyValueError::new_err(format!(
			"invalid provider '{provider}'. expected one of: openai, openai_resp, gemini, anthropic, fireworks, together, groq, mimo, nebius, xai, deepseek, zai, bigmodel, aliyun, cohere, ollama"
		))
	})
}

fn build_client_with_overrides(
    provider: String,
    api_key: Option<String>,
    base_url: Option<String>,
    connect_timeout_seconds: Option<f64>,
    read_timeout_seconds: Option<f64>,
    timeout_seconds: Option<f64>,
    default_headers: Option<HashMap<String, String>>,
) -> PyResult<PyClient> {
    let adapter_kind = parse_adapter_kind(&provider)?;
    // Bind the Client to the adapter up front. rust-genai's
    // `with_adapter_kind` drives routing directly from this, so every
    // `exec_chat` call hands a bare model name through without any
    // `"{provider}::{model}"` trickery on the Python side. Mismatched
    // namespaces in the caller's model string surface as
    // `Error::AdapterKindMismatch` from the Rust side.
    let mut builder = genai::Client::builder().with_adapter_kind(adapter_kind);

    if let Some(web_config) = web_config_from_overrides(
        connect_timeout_seconds,
        read_timeout_seconds,
        timeout_seconds,
        default_headers,
    )? {
        builder = builder.with_web_config(web_config);
    }

    if let Some(api_key) = api_key {
        let auth_resolver = AuthResolver::from_resolver_fn(move |model_iden: genai::ModelIden| {
            if model_iden.adapter_kind == adapter_kind {
                Ok(Some(AuthData::from_single(api_key.clone())))
            } else {
                Ok(None)
            }
        });
        builder = builder.with_auth_resolver(auth_resolver);
    }

    if let Some(base_url) = base_url {
        let service_target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |mut service_target: genai::ServiceTarget| {
                if service_target.model.adapter_kind == adapter_kind {
                    service_target.endpoint = Endpoint::from_owned(base_url.clone());
                }
                Ok(service_target)
            },
        );
        builder = builder.with_service_target_resolver(service_target_resolver);
    }

    Ok(PyClient {
        inner: builder.build(),
        provider: Some(provider),
    })
}

fn build_client_with_request_override(
    provider: String,
    url: String,
    headers: HashMap<String, String>,
) -> PyResult<PyClient> {
    let adapter_kind = parse_adapter_kind(&provider)?;
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err(PyValueError::new_err(
            "request override url cannot be empty",
        ));
    }

    let headers = Headers::from(headers.into_iter().collect::<Vec<_>>());
    let auth_resolver = AuthResolver::from_resolver_fn(move |model_iden: genai::ModelIden| {
        if model_iden.adapter_kind == adapter_kind {
            Ok(Some(AuthData::RequestOverride {
                url: url.clone(),
                headers: headers.clone(),
            }))
        } else {
            Ok(None)
        }
    });

    Ok(PyClient {
        inner: genai::Client::builder()
            .with_adapter_kind(adapter_kind)
            .with_auth_resolver(auth_resolver)
            .build(),
        provider: Some(provider),
    })
}

fn to_runtime_error(error: genai::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
