//! Request-side pyclasses: messages, requests, tools, and options.

use genai::chat::{
    Binary, BinarySource, ChatMessage, ChatOptions, ChatRequest, ChatResponseFormat, ChatRole,
    JsonSpec, MessageOptions, ServiceTier, Tool, ToolCall, ToolChoice, ToolConfig, ToolResponse,
    Verbosity, WebSearchConfig,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use std::collections::HashMap;

use crate::convert::{
    coerce_custom_format_to_json_string, coerce_json_to_string, coerce_schema_to_json_string,
    json_value_to_py, lowercase_content_part_keys, normalize_chat_request_value,
    normalize_content_parts, parse_cache_control, parse_chat_role, parse_reasoning_effort,
    raw_content_json_to_message_content, validate_role,
};

#[pyclass(name = "ChatMessage", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyChatMessage {
    #[pyo3(get, set)]
    pub(crate) role: String,
    #[pyo3(get, set)]
    pub(crate) content: String,
    /// For assistant messages that contain tool calls (serialized as JSON array of tool calls)
    #[pyo3(get, set)]
    pub(crate) tool_calls: Option<Vec<PyToolCall>>,
    /// For tool response messages
    #[pyo3(get, set)]
    pub(crate) tool_response_call_id: Option<String>,
    /// Per-message cache hint. Accepted: "ephemeral", "ephemeral_5m",
    /// "ephemeral_1h", "ephemeral_24h", "memory". Anthropic applies these
    /// as `cache_control: {type: ephemeral, ttl: ...}` at the content-part
    /// level; OpenAI maps a subset to request-level caching and ignores
    /// the rest. Unknown values are silently dropped.
    #[pyo3(get, set)]
    pub(crate) cache_control: Option<String>,
    /// Encrypted reasoning blobs carried back from a prior assistant
    /// turn (OpenAI Responses `type: "reasoning"` items). When set on an
    /// assistant message, the openai_resp adapter emits each blob as a
    /// top-level `{type: "reasoning", encrypted_content: ...}` input
    /// item before the message it belongs to. Required to keep the
    /// Responses-API prefix cache warm across turns on reasoning models.
    #[pyo3(get, set)]
    pub(crate) thought_signatures: Option<Vec<String>>,
    /// Provider-native content blocks serialized as JSON. When present, the
    /// blocks are converted to `ContentPart::Custom` values and passed through
    /// adapters that support custom content.
    #[pyo3(get)]
    pub(crate) raw_content_json: Option<String>,
    /// Mixed content — text plus images, PDFs, audio — stored in the serde
    /// form. When present it replaces `content`, which can only carry text.
    pub(crate) content_parts_json: Option<String>,
}

#[pymethods]
impl PyChatMessage {
    #[new]
    #[pyo3(signature = (
        role,
        content = None,
        tool_calls = None,
        tool_response_call_id = None,
        cache_control = None,
        thought_signatures = None,
        raw_content = None,
        content_parts = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        role: String,
        content: Option<String>,
        tool_calls: Option<Vec<PyToolCall>>,
        tool_response_call_id: Option<String>,
        cache_control: Option<String>,
        thought_signatures: Option<Vec<String>>,
        raw_content: Option<Bound<'_, PyAny>>,
        content_parts: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        validate_role(&role)?;
        let raw_content_json = raw_content.map(coerce_schema_to_json_string).transpose()?;
        let content_parts_json = content_parts.map(coerce_content_parts).transpose()?;
        if content.is_none() && content_parts_json.is_none() && raw_content_json.is_none() {
            return Err(PyValueError::new_err(
                "ChatMessage needs content, content_parts, or raw_content",
            ));
        }
        Ok(Self {
            role,
            content: content.unwrap_or_default(),
            tool_calls,
            tool_response_call_id,
            cache_control,
            thought_signatures,
            raw_content_json,
            content_parts_json,
        })
    }

    /// The mixed-content parts, as the lowercase-keyed dict form.
    #[getter]
    fn content_parts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(raw) = self.content_parts_json.as_deref() else {
            return Ok(py.None().into_bound(py));
        };
        let mut value = serde_json::from_str::<serde_json::Value>(raw)
            .map_err(|err| PyRuntimeError::new_err(format!("invalid content parts: {err}")))?;
        lowercase_content_part_keys(&mut value);
        Ok(json_value_to_py(py, &value)?.into_bound(py))
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
        match &self.raw_content_json {
            Some(raw) => {
                let value = serde_json::from_str::<serde_json::Value>(raw).map_err(|err| {
                    PyRuntimeError::new_err(format!("invalid raw_content_json: {err}"))
                })?;
                dict.set_item("raw_content", json_value_to_py(py, &value)?)?;
            }
            None => dict.set_item("raw_content", py.None())?,
        }
        dict.set_item("content_parts", self.content_parts(py)?)?;
        Ok(dict.into_any())
    }
}

/// A binary attachment — an image, a PDF, an audio clip.
///
/// Providers accept these as base64 payloads almost universally, and as
/// URLs only sometimes; `from_url` is there for the ones that do.
#[pyclass(name = "Binary", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyBinary {
    pub(crate) inner: Binary,
}

#[pymethods]
impl PyBinary {
    /// Wrap already-encoded base64 bytes.
    #[staticmethod]
    #[pyo3(signature = (content_type, content, name = None))]
    fn from_base64(content_type: String, content: String, name: Option<String>) -> Self {
        Self {
            inner: Binary::from_base64(content_type, content, name),
        }
    }

    /// Reference a remote file. Only some providers fetch URLs; the rest
    /// need the bytes inline.
    #[staticmethod]
    #[pyo3(signature = (content_type, url, name = None))]
    fn from_url(content_type: String, url: String, name: Option<String>) -> Self {
        Self {
            inner: Binary::from_url(content_type, url, name),
        }
    }

    /// Read a local file, guessing its MIME type from the extension and
    /// base64-encoding the contents.
    #[staticmethod]
    fn from_path(path: String) -> PyResult<Self> {
        Binary::from_file(&path)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn content_type(&self) -> String {
        self.inner.content_type.clone()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// Where the bytes come from: `"base64"` or `"url"`.
    #[getter]
    fn source_kind(&self) -> &'static str {
        match self.inner.source {
            BinarySource::Base64(_) => "base64",
            BinarySource::Url(_) => "url",
        }
    }

    #[getter]
    fn is_image(&self) -> bool {
        self.inner.is_image()
    }

    #[getter]
    fn is_pdf(&self) -> bool {
        self.inner.is_pdf()
    }

    /// The content-part dict form, ready to drop into `content_parts`.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let value = serde_json::to_value(&self.inner)
            .map_err(|err| PyRuntimeError::new_err(format!("serialize binary: {err}")))?;
        let part = serde_json::json!({ "binary": value });
        Ok(json_value_to_py(py, &part)?.into_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Binary(content_type={:?}, source={:?}, name={:?})",
            self.inner.content_type,
            self.source_kind(),
            self.inner.name,
        )
    }
}

#[pyclass(name = "ChatRequest", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyChatRequest {
    #[pyo3(get, set)]
    pub(crate) system: Option<String>,
    #[pyo3(get, set)]
    pub(crate) messages: Vec<PyChatMessage>,
    #[pyo3(get, set)]
    pub(crate) tools: Option<Vec<PyTool>>,
    /// OpenAI Responses API: chain a previous server-stored response so the
    /// backend only needs to process the new turn's delta. Pair with `store=True`
    /// on the previous request to make its `response_id` reusable here.
    #[pyo3(get, set)]
    pub(crate) previous_response_id: Option<String>,
    /// OpenAI Responses API: ask the backend to persist this response so its
    /// `response_id` can be used as `previous_response_id` on a follow-up call.
    #[pyo3(get, set)]
    pub(crate) store: Option<bool>,
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
#[derive(Clone, Default)]
pub(crate) struct PyTool {
    #[pyo3(get, set)]
    pub(crate) name: String,
    #[pyo3(get, set)]
    pub(crate) description: Option<String>,
    /// JSON Schema for parameters, stored as a JSON string. Accepts either a
    /// `str` (used as-is) or any JSON-serializable Python object (typically a
    /// `dict`, e.g. ``pydantic.BaseModel.model_json_schema()``).
    #[pyo3(get)]
    pub(crate) schema_json: Option<String>,
    /// Provider-native freeform custom-tool format, stored internally as JSON.
    /// Exposed to Python as a JSON-compatible value through the custom getter.
    pub(crate) custom_format_json: Option<String>,
    /// Ask the provider to enforce the schema on tool-call arguments
    /// (OpenAI and Anthropic constrained decoding).
    #[pyo3(get, set)]
    pub(crate) strict: Option<bool>,
    /// Prompt-cache hint. On Anthropic, setting this on the *last* tool
    /// caches the whole tool-list prefix.
    #[pyo3(get, set)]
    pub(crate) cache_control: Option<String>,
    /// Anthropic fine-grained tool streaming: stream tool-call input
    /// token-by-token instead of buffering the whole JSON server-side.
    #[pyo3(get, set)]
    pub(crate) eager_input_streaming: Option<bool>,
    /// Provider-specific tool configuration, stored internally as JSON.
    pub(crate) config_json: Option<String>,
    /// True when this is the provider's built-in web-search tool rather
    /// than a user-defined function.
    pub(crate) is_web_search: bool,
}

#[pymethods]
impl PyTool {
    #[new]
    #[pyo3(signature = (
        name,
        description = None,
        schema_json = None,
        custom_format = None,
        *,
        strict = None,
        cache_control = None,
        eager_input_streaming = None,
        config = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        description: Option<String>,
        schema_json: Option<Bound<'_, PyAny>>,
        custom_format: Option<Bound<'_, PyAny>>,
        strict: Option<bool>,
        cache_control: Option<String>,
        eager_input_streaming: Option<bool>,
        config: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema_json = schema_json.map(coerce_schema_to_json_string).transpose()?;
        let custom_format_json = custom_format
            .map(coerce_custom_format_to_json_string)
            .transpose()?;
        let config_json = config
            .map(|value| coerce_json_to_string(value, "config"))
            .transpose()?;
        if let Some(value) = cache_control.as_deref() {
            parse_cache_control_strict(value)?;
        }
        Ok(Self {
            name,
            description,
            schema_json,
            custom_format_json,
            strict,
            cache_control,
            eager_input_streaming,
            config_json,
            is_web_search: false,
        })
    }

    /// The provider's built-in web-search tool.
    ///
    /// Not a function you implement — the provider runs the search itself
    /// (Google Search on Gemini, the web-search tool on OpenAI Responses),
    /// so there is no schema and no tool response to send back.
    #[staticmethod]
    #[pyo3(signature = (*, max_uses = None, allowed_domains = None, blocked_domains = None))]
    fn web_search(
        max_uses: Option<u32>,
        allowed_domains: Option<Vec<String>>,
        blocked_domains: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if allowed_domains.is_some() && blocked_domains.is_some() {
            // Providers reject the combination, so catch it here rather
            // than letting it become a remote 400.
            return Err(PyValueError::new_err(
                "web_search accepts allowed_domains or blocked_domains, not both",
            ));
        }
        let config = serde_json::json!({
            "max_uses": max_uses,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
        });
        Ok(Self {
            name: "WebSearch".to_string(),
            description: None,
            schema_json: None,
            custom_format_json: None,
            strict: None,
            cache_control: None,
            eager_input_streaming: None,
            config_json: Some(config.to_string()),
            is_web_search: true,
        })
    }

    /// True for the built-in web-search tool.
    #[getter]
    fn is_builtin(&self) -> bool {
        self.is_web_search
    }

    /// Provider-specific configuration as a Python value.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.config_json
            .as_deref()
            .map(|raw| {
                let value = serde_json::from_str(raw).map_err(|err| {
                    PyRuntimeError::new_err(format!("invalid tool config JSON: {err}"))
                })?;
                json_value_to_py(py, &value)
            })
            .transpose()
    }

    #[setter]
    fn set_schema_json(&mut self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.schema_json = value.map(coerce_schema_to_json_string).transpose()?;
        Ok(())
    }

    #[getter]
    fn custom_format(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.custom_format_json
            .as_deref()
            .map(|raw| {
                let value = serde_json::from_str(raw).map_err(|err| {
                    PyRuntimeError::new_err(format!("invalid custom_format JSON: {err}"))
                })?;
                json_value_to_py(py, &value)
            })
            .transpose()
    }

    #[setter]
    fn set_custom_format(&mut self, value: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.custom_format_json = value.map(coerce_custom_format_to_json_string).transpose()?;
        Ok(())
    }
}

#[pyclass(name = "ChatOptions", from_py_object)]
#[derive(Clone, Default)]
pub(crate) struct PyChatOptions {
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
    /// Tool selection preference: "auto", "none", "required", or the name
    /// of a specific tool the model must call. Unrecognised values are a
    /// ValueError rather than a silent no-op — a mistyped tool name here
    /// would otherwise quietly become "the model may do as it likes".
    #[pyo3(get, set)]
    tool_choice: Option<String>,
    /// Response length preference where the provider has one (OpenAI
    /// gpt-5 family): "low", "medium", or "high".
    #[pyo3(get, set)]
    verbosity: Option<String>,
    /// OpenAI processing tier: "flex" (cheaper, slower), "auto", or
    /// "default".
    #[pyo3(get, set)]
    service_tier: Option<String>,
    /// Request-level prompt-cache hint. Accepted: "ephemeral",
    /// "ephemeral_5m", "ephemeral_1h", "ephemeral_24h", "memory".
    /// Anthropic auto-marks the static tools+system prefix; OpenAI maps a
    /// subset onto its request-level cache settings.
    #[pyo3(get, set)]
    cache_control: Option<String>,
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
    /// Provider-specific top-level request payload fields serialized as JSON.
    #[pyo3(get)]
    extra_body_json: Option<String>,
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
            extra_body = None,
            tool_choice = None,
            verbosity = None,
            service_tier = None,
            cache_control = None,
		))]
    #[allow(clippy::too_many_arguments)]
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
        extra_body: Option<Bound<'_, PyAny>>,
        tool_choice: Option<String>,
        verbosity: Option<String>,
        service_tier: Option<String>,
        cache_control: Option<String>,
    ) -> PyResult<Self> {
        let extra_body_json = extra_body.map(coerce_schema_to_json_string).transpose()?;
        // Validate the enum-shaped strings here, at construction, so a typo
        // is a ValueError at the call site rather than an option that
        // silently does nothing on the wire.
        if let Some(value) = verbosity.as_deref() {
            parse_verbosity(value)?;
        }
        if let Some(value) = service_tier.as_deref() {
            parse_service_tier(value)?;
        }
        if let Some(value) = cache_control.as_deref() {
            parse_cache_control_strict(value)?;
        }
        Ok(Self {
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
            extra_body_json,
            tool_choice,
            verbosity,
            service_tier,
            cache_control,
        })
    }
}

#[pyclass(name = "JsonSpec", from_py_object)]
#[derive(Clone)]
pub(crate) struct PyJsonSpec {
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
pub(crate) struct PyToolCall {
    #[pyo3(get, set)]
    pub(crate) call_id: String,
    #[pyo3(get, set)]
    pub(crate) fn_name: String,
    #[pyo3(get, set)]
    pub(crate) fn_arguments_json: String,
    #[pyo3(get, set)]
    pub(crate) thought_signatures: Option<Vec<String>>,
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

    pub(crate) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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

pub(crate) fn py_tool_calls_to_py<'py>(
    py: Python<'py>,
    tool_calls: &[PyToolCall],
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::empty(py);
    for tool_call in tool_calls {
        list.append(tool_call.to_dict(py)?)?;
    }
    Ok(list.into_any())
}

pub(crate) fn to_py_tool_call(tool_call: &ToolCall) -> PyResult<PyToolCall> {
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

impl TryFrom<PyChatMessage> for ChatMessage {
    type Error = PyErr;

    fn try_from(message: PyChatMessage) -> Result<Self, Self::Error> {
        let role = parse_chat_role(&message.role)?;
        let options = message
            .cache_control
            .as_deref()
            .and_then(parse_cache_control)
            .map(|cc| MessageOptions {
                cache_control: Some(cc),
            });
        let thought_signatures = message.thought_signatures.clone();

        let mut chat_message = match role {
            // Structured content wins over the plain `content` string:
            // raw_content first (provider-native blocks), then the mixed
            // text/binary parts, then the text-shaped forms.
            _ if message.raw_content_json.is_some() => {
                let raw_content_json = message.raw_content_json.as_deref().unwrap_or("null");
                ChatMessage {
                    role,
                    content: raw_content_json_to_message_content(raw_content_json)?,
                    options: None,
                }
            }
            _ if message.content_parts_json.is_some() => {
                let raw = message.content_parts_json.as_deref().unwrap_or("[]");
                ChatMessage {
                    role,
                    content: serde_json::from_str(raw).map_err(|err| {
                        PyValueError::new_err(format!("invalid content parts: {err}"))
                    })?,
                    options: None,
                }
            }
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

impl TryFrom<PyChatRequest> for ChatRequest {
    type Error = PyErr;

    fn try_from(request: PyChatRequest) -> Result<Self, Self::Error> {
        let messages = request
            .messages
            .into_iter()
            .map(ChatMessage::try_from)
            .collect::<PyResult<Vec<_>>>()?;

        let tools = request
            .tools
            .map(|py_tools| py_tools.into_iter().map(Tool::from).collect());

        Ok(Self {
            system: request.system,
            messages,
            tools,
            previous_response_id: request.previous_response_id,
            store: request.store,
        })
    }
}

impl From<PyTool> for Tool {
    fn from(py_tool: PyTool) -> Self {
        let mut tool = if py_tool.is_web_search {
            Tool::new_web_search()
        } else {
            Tool::new(py_tool.name)
        };
        if let Some(description) = py_tool.description {
            tool = tool.with_description(description);
        }
        if let Some(schema) = py_tool
            .schema_json
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        {
            tool = tool.with_schema(schema);
        }
        if let Some(custom_format) = py_tool
            .custom_format_json
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        {
            tool = tool.with_custom_format(custom_format);
        }
        if let Some(strict) = py_tool.strict {
            tool = tool.with_strict(strict);
        }
        if let Some(eager) = py_tool.eager_input_streaming {
            tool = tool.with_eager_input_streaming(eager);
        }
        if let Some(cache_control) = py_tool
            .cache_control
            .as_deref()
            .and_then(parse_cache_control)
        {
            tool = tool.with_cache_control(cache_control);
        }
        if let Some(config) = py_tool
            .config_json
            .as_deref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        {
            // The built-in tools take their own typed config; everything
            // else is passed through as an opaque provider blob.
            let config = if py_tool.is_web_search {
                serde_json::from_value::<WebSearchConfig>(config)
                    .map(ToolConfig::WebSearch)
                    .unwrap_or_else(|_| ToolConfig::Custom(serde_json::Value::Null))
            } else {
                ToolConfig::Custom(config)
            };
            tool = tool.with_config(config);
        }
        tool
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

        let extra_body = options
            .extra_body_json
            .as_deref()
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());

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
            extra_body,
            tool_choice: options.tool_choice.as_deref().map(parse_tool_choice),
            verbosity: options
                .verbosity
                .as_deref()
                .and_then(|v| parse_verbosity(v).ok()),
            service_tier: options
                .service_tier
                .as_deref()
                .and_then(|v| parse_service_tier(v).ok()),
            cache_control: options
                .cache_control
                .as_deref()
                .and_then(parse_cache_control),
            // Deliberately no `..Default::default()`: every field of
            // rust-genai's ChatOptions is spelled out here, so a field
            // added upstream becomes a compile error rather than an option
            // that silently does nothing. Letting that happen quietly is
            // how this binding fell behind in the first place.
        }
    }
}

/// Map a tool-choice keyword onto rust-genai's enum. Anything that is not
/// one of the three keywords names a specific tool the model must call.
fn parse_tool_choice(raw: &str) -> ToolChoice {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        "required" => ToolChoice::Required,
        _ => ToolChoice::tool(raw.trim()),
    }
}

fn parse_verbosity(raw: &str) -> PyResult<Verbosity> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "low" => Ok(Verbosity::Low),
        "medium" => Ok(Verbosity::Medium),
        "high" => Ok(Verbosity::High),
        other => Err(PyValueError::new_err(format!(
            "invalid verbosity {other:?}. expected one of: low, medium, high"
        ))),
    }
}

fn parse_service_tier(raw: &str) -> PyResult<ServiceTier> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "flex" => Ok(ServiceTier::Flex),
        "auto" => Ok(ServiceTier::Auto),
        "default" => Ok(ServiceTier::Default),
        other => Err(PyValueError::new_err(format!(
            "invalid service_tier {other:?}. expected one of: flex, auto, default"
        ))),
    }
}

fn parse_cache_control_strict(raw: &str) -> PyResult<()> {
    parse_cache_control(raw).map(|_| ()).ok_or_else(|| {
        PyValueError::new_err(format!(
            "invalid cache_control {raw:?}. expected one of: ephemeral, memory, \
             ephemeral_5m, ephemeral_1h, ephemeral_24h"
        ))
    })
}

/// Coerce a heterogeneous `content_parts` list into the serde form.
///
/// Accepts, per element: a `str` (becomes a text part), a `Binary`, or an
/// already-shaped content-part dict such as `{"text": "..."}`.
fn coerce_content_parts(parts: Vec<Bound<'_, PyAny>>) -> PyResult<String> {
    let mut values = Vec::with_capacity(parts.len());
    for (index, part) in parts.into_iter().enumerate() {
        if let Ok(text) = part.extract::<String>() {
            values.push(serde_json::json!({ "text": text }));
            continue;
        }
        if let Ok(binary) = part.extract::<PyBinary>() {
            let value = serde_json::to_value(&binary.inner)
                .map_err(|err| PyValueError::new_err(format!("content_parts[{index}]: {err}")))?;
            values.push(serde_json::json!({ "binary": value }));
            continue;
        }
        let value: serde_json::Value = depythonize(&part).map_err(|err| {
            PyValueError::new_err(format!(
                "content_parts[{index}] must be a str, a Binary, or a content-part dict: {err}"
            ))
        })?;
        values.push(value);
    }
    normalize_content_parts(&mut values, "content_parts")?;
    serde_json::to_string(&values)
        .map_err(|err| PyValueError::new_err(format!("failed to serialize content_parts: {err}")))
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
pub(crate) fn coerce_chat_request(obj: Bound<'_, PyAny>) -> PyResult<ChatRequest> {
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

/// Accept an optional `PyChatOptions` pyclass or a JSON-compatible dict.
pub(crate) fn coerce_chat_options(obj: Option<Bound<'_, PyAny>>) -> PyResult<Option<ChatOptions>> {
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
    serde_json::from_value(fill_chat_options_defaults(value))
        .map(Some)
        .map_err(|err| {
            PyValueError::new_err(format!("chat options dict does not match schema: {err}"))
        })
}

/// Overlay a caller's option dict onto a serialized `ChatOptions::default()`.
///
/// Every field of the dict form is meant to be optional, but not every
/// field of the Rust struct carries a serde default — `stop_sequences` is a
/// bare `Vec<String>`, so deserializing `{"temperature": 0.4}` fails with
/// "missing field `stop_sequences`". Starting from the default and layering
/// the caller's keys on top keeps the dict form genuinely partial, and does
/// so for any other non-defaulted field upstream may add.
///
/// The specific `stop_sequences` case is fixed upstream by
/// jeremychone/rust-genai#285; this stays regardless, since it also covers
/// whatever field is added next without a default.
fn fill_chat_options_defaults(value: serde_json::Value) -> serde_json::Value {
    let Some(provided) = value.as_object() else {
        return value;
    };
    let Ok(mut base) = serde_json::to_value(ChatOptions::default()) else {
        return value;
    };
    let Some(base_obj) = base.as_object_mut() else {
        return value;
    };
    for (key, entry) in provided {
        base_obj.insert(key.clone(), entry.clone());
    }
    base
}

#[cfg(test)]
mod tests {
    use super::{PyTool, Tool};
    use serde_json::json;

    #[test]
    fn py_tool_custom_format_reaches_genai_tool() {
        let expected = json!({
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: \"PING\""
        });
        let py_tool = PyTool {
            name: "emit_ping".to_string(),
            description: Some("Emit the constrained token".to_string()),
            schema_json: None,
            custom_format_json: Some(expected.to_string()),
            ..PyTool::default()
        };

        let tool = Tool::from(py_tool);

        assert_eq!(tool.name.to_string(), "emit_ping");
        assert_eq!(
            tool.description.as_deref(),
            Some("Emit the constrained token")
        );
        assert_eq!(tool.custom_format, Some(expected));
        assert!(tool.schema.is_none());
    }
}
