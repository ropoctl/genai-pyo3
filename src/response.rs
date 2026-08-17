//! Response-side pyclasses: usage, chat responses, and streaming.

use futures::StreamExt;
use genai::chat::{
    CacheCreationDetails, ChatStream, ChatStreamEvent, CompletionTokensDetails,
    PromptTokensDetails, StreamEnd, Usage,
};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::chat::{PyToolCall, py_tool_calls_to_py, to_py_tool_call};
use crate::convert::{coerce_message_content, lowercase_content_part_keys};
use crate::errors::to_runtime_error;

#[pyclass(name = "CacheCreationDetails", from_py_object)]
#[derive(Clone, Default)]
pub(crate) struct PyCacheCreationDetails {
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
pub(crate) struct PyPromptTokensDetails {
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
pub(crate) struct PyCompletionTokensDetails {
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
pub(crate) struct PyUsage {
    #[pyo3(get, set)]
    pub(crate) prompt_tokens: Option<i32>,
    #[pyo3(get, set)]
    pub(crate) completion_tokens: Option<i32>,
    #[pyo3(get, set)]
    pub(crate) total_tokens: Option<i32>,
    /// Breakdown of prompt tokens: cached vs cache-creation vs other.
    /// Inspect ``prompt_tokens_details.cached_tokens`` to verify a
    /// prompt-cache hit on the current turn.
    #[pyo3(get, set)]
    pub(crate) prompt_tokens_details: Option<PyPromptTokensDetails>,
    /// Breakdown of completion tokens: reasoning vs prediction vs audio.
    #[pyo3(get, set)]
    pub(crate) completion_tokens_details: Option<PyCompletionTokensDetails>,
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

    pub(crate) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
pub(crate) struct PyChatResponse {
    pub(crate) inner_content: genai::chat::MessageContent,
    #[pyo3(get)]
    pub(crate) reasoning_content: Option<String>,
    #[pyo3(get)]
    pub(crate) model_adapter_kind: String,
    #[pyo3(get)]
    pub(crate) model_name: String,
    #[pyo3(get)]
    pub(crate) provider_model_adapter_kind: String,
    #[pyo3(get)]
    pub(crate) provider_model_name: String,
    #[pyo3(get)]
    pub(crate) usage: PyUsage,
    /// OpenAI Responses API: the server-stored response id, present when the
    /// request was sent with `store=True`. Pass it as `previous_response_id`
    /// on the next request to chain calls server-side.
    #[pyo3(get)]
    pub(crate) response_id: Option<String>,
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
    #[allow(clippy::too_many_arguments)]
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
pub(crate) struct PyStreamEnd {
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
pub(crate) struct PyChatStreamEvent {
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

#[pyclass(name = "_ChatStream")]
pub(crate) struct PyChatStream {
    inner: Arc<Mutex<Option<ChatStream>>>,
}

impl PyChatStream {
    pub(crate) fn new(stream: ChatStream) -> Self {
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

fn to_py_cache_creation_details(d: &CacheCreationDetails) -> PyCacheCreationDetails {
    PyCacheCreationDetails {
        ephemeral_5m_tokens: d.ephemeral_5m_tokens,
        ephemeral_1h_tokens: d.ephemeral_1h_tokens,
    }
}

fn to_py_prompt_tokens_details(d: &PromptTokensDetails) -> PyPromptTokensDetails {
    PyPromptTokensDetails {
        cache_creation_tokens: d.cache_creation_tokens,
        cache_creation_details: d
            .cache_creation_details
            .as_ref()
            .map(to_py_cache_creation_details),
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

pub(crate) fn to_py_usage(usage: &Usage) -> PyUsage {
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

pub(crate) fn to_py_chat_response(
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
