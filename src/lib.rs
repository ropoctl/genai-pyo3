use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{
	ChatMessage, ChatOptions, ChatRequest, ChatRole, ChatStream, ChatStreamEvent, StreamEnd, ToolCall, Usage,
};
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Builder;
use tokio::sync::Mutex;

#[pyclass(name = "Client")]
struct PyClient {
	inner: genai::Client,
}

#[pyclass(name = "ChatMessage", from_py_object)]
#[derive(Clone)]
struct PyChatMessage {
	#[pyo3(get, set)]
	role: String,
	#[pyo3(get, set)]
	content: String,
}

#[pymethods]
impl PyChatMessage {
	#[new]
	fn new(role: String, content: String) -> PyResult<Self> {
		validate_role(&role)?;
		Ok(Self { role, content })
	}
}

#[pyclass(name = "ChatRequest", from_py_object)]
#[derive(Clone)]
struct PyChatRequest {
	#[pyo3(get, set)]
	system: Option<String>,
	#[pyo3(get, set)]
	messages: Vec<PyChatMessage>,
}

#[pymethods]
impl PyChatRequest {
	#[new]
	#[pyo3(signature = (messages, system = None))]
	fn new(messages: Vec<PyChatMessage>, system: Option<String>) -> Self {
		Self { system, messages }
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
	normalize_reasoning_content: Option<bool>,
	#[pyo3(get, set)]
	seed: Option<u64>,
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
		normalize_reasoning_content = None,
		seed = None,
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
		normalize_reasoning_content: Option<bool>,
		seed: Option<u64>,
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
			normalize_reasoning_content,
			seed,
		}
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

#[pyclass(name = "Usage", from_py_object)]
#[derive(Clone)]
struct PyUsage {
	#[pyo3(get, set)]
	prompt_tokens: Option<i32>,
	#[pyo3(get, set)]
	completion_tokens: Option<i32>,
	#[pyo3(get, set)]
	total_tokens: Option<i32>,
}

#[pyclass(name = "ChatResponse", from_py_object)]
#[derive(Clone)]
struct PyChatResponse {
	#[pyo3(get, set)]
	text: Option<String>,
	#[pyo3(get, set)]
	texts: Vec<String>,
	#[pyo3(get, set)]
	reasoning_content: Option<String>,
	#[pyo3(get, set)]
	model_adapter_kind: String,
	#[pyo3(get, set)]
	model_name: String,
	#[pyo3(get, set)]
	provider_model_adapter_kind: String,
	#[pyo3(get, set)]
	provider_model_name: String,
	#[pyo3(get, set)]
	usage: PyUsage,
	#[pyo3(get, set)]
	tool_calls: Vec<PyToolCall>,
}

#[pyclass(name = "StreamEnd", from_py_object)]
#[derive(Clone)]
struct PyStreamEnd {
	#[pyo3(get, set)]
	captured_usage: Option<PyUsage>,
	#[pyo3(get, set)]
	captured_first_text: Option<String>,
	#[pyo3(get, set)]
	captured_texts: Option<Vec<String>>,
	#[pyo3(get, set)]
	captured_reasoning_content: Option<String>,
	#[pyo3(get, set)]
	captured_tool_calls: Option<Vec<PyToolCall>>,
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
impl PyClient {
	#[new]
	fn new() -> Self {
		Self {
			inner: genai::Client::default(),
		}
	}

	#[staticmethod]
	fn with_api_key(provider: String, api_key: String) -> PyResult<Self> {
		build_client_with_overrides(provider, Some(api_key), None)
	}

	#[staticmethod]
	fn with_api_key_and_base_url(provider: String, api_key: String, base_url: String) -> PyResult<Self> {
		build_client_with_overrides(provider, Some(api_key), Some(base_url))
	}

	#[staticmethod]
	fn with_base_url(provider: String, base_url: String) -> PyResult<Self> {
		build_client_with_overrides(provider, None, Some(base_url))
	}

	#[pyo3(signature = (model, request, options = None))]
	fn chat(
		&self,
		py: Python<'_>,
		model: String,
		request: PyChatRequest,
		options: Option<PyChatOptions>,
	) -> PyResult<Py<PyChatResponse>> {
		let chat_req = to_rust_chat_request(request)?;
		let chat_options = options.map(to_rust_chat_options);

		let runtime = Builder::new_multi_thread()
			.enable_all()
			.build()
			.map_err(|err| PyRuntimeError::new_err(format!("failed to build Tokio runtime: {err}")))?;

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
		request: PyChatRequest,
		options: Option<PyChatOptions>,
	) -> PyResult<Bound<'py, PyAny>> {
		let chat_req = to_rust_chat_request(request)?;
		let chat_options = options.map(to_rust_chat_options);
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
		request: PyChatRequest,
		options: Option<PyChatOptions>,
	) -> PyResult<Bound<'py, PyAny>> {
		let chat_req = to_rust_chat_request(request)?;
		let chat_options = options.map(to_rust_chat_options);
		let client = self.inner.clone();

		pyo3_async_runtimes::tokio::future_into_py(py, async move {
			let stream_res = client
				.exec_chat_stream(model, chat_req, chat_options.as_ref())
				.await
				.map_err(to_runtime_error)?;

			Python::attach(|py| {
				Py::new(py, PyChatStream::new(stream_res.stream))
					.map_err(|err| PyRuntimeError::new_err(format!("failed to build stream object: {err}")))
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
	module.add_class::<PyToolCall>()?;
	module.add_class::<PyUsage>()?;
	module.add_class::<PyChatResponse>()?;
	module.add_class::<PyStreamEnd>()?;
	module.add_class::<PyChatStreamEvent>()?;
	module.add_class::<PyChatStream>()?;
	Ok(())
}

fn validate_role(role: &str) -> PyResult<()> {
	match role {
		"system" | "user" | "assistant" | "tool" => Ok(()),
		_ => Err(PyValueError::new_err(format!(
			"invalid chat role '{role}'. expected one of: system, user, assistant, tool"
		))),
	}
}

fn to_rust_role(role: &str) -> PyResult<ChatRole> {
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

fn to_rust_chat_request(request: PyChatRequest) -> PyResult<ChatRequest> {
	let mut messages: Vec<ChatMessage> = Vec::with_capacity(request.messages.len());
	for message in request.messages {
		let role = to_rust_role(&message.role)?;
		messages.push(ChatMessage {
			role,
			content: message.content.into(),
			options: None,
		});
	}

	Ok(ChatRequest {
		system: request.system,
		messages,
		tools: None,
		previous_response_id: None,
		store: None,
	})
}

fn to_rust_chat_options(options: PyChatOptions) -> ChatOptions {
	ChatOptions {
		temperature: options.temperature,
		max_tokens: options.max_tokens,
		top_p: options.top_p,
		stop_sequences: options.stop_sequences,
		capture_usage: options.capture_usage,
		capture_content: options.capture_content,
		capture_reasoning_content: options.capture_reasoning_content,
		capture_tool_calls: options.capture_tool_calls,
		capture_raw_body: options.capture_raw_body,
		response_format: None,
		normalize_reasoning_content: options.normalize_reasoning_content,
		reasoning_effort: None,
		verbosity: None,
		seed: options.seed,
		service_tier: None,
		extra_headers: None,
		cache_control: None,
		prompt_cache_key: None,
	}
}

fn to_py_usage(usage: &Usage) -> PyUsage {
	PyUsage {
		prompt_tokens: usage.prompt_tokens,
		completion_tokens: usage.completion_tokens,
		total_tokens: usage.total_tokens,
	}
}

fn to_py_tool_call(tool_call: &ToolCall) -> PyResult<PyToolCall> {
	let fn_arguments_json = serde_json::to_string(&tool_call.fn_arguments)
		.map_err(|err| PyRuntimeError::new_err(format!("failed to serialize tool call arguments: {err}")))?;

	Ok(PyToolCall {
		call_id: tool_call.call_id.clone(),
		fn_name: tool_call.fn_name.clone(),
		fn_arguments_json,
		thought_signatures: tool_call.thought_signatures.clone(),
	})
}

fn to_py_chat_response(py: Python<'_>, response: genai::chat::ChatResponse) -> PyResult<Py<PyChatResponse>> {
	let text = response.first_text().map(|t| t.to_string());
	let texts = response.texts().into_iter().map(|t| t.to_string()).collect::<Vec<_>>();
	let tool_calls = response
		.tool_calls()
		.into_iter()
		.map(to_py_tool_call)
		.collect::<PyResult<Vec<_>>>()?;

	Py::new(
		py,
		PyChatResponse {
			text,
			texts,
			reasoning_content: response.reasoning_content,
			model_adapter_kind: response.model_iden.adapter_kind.to_string(),
			model_name: response.model_iden.model_name.to_string(),
			provider_model_adapter_kind: response.provider_model_iden.adapter_kind.to_string(),
			provider_model_name: response.provider_model_iden.model_name.to_string(),
			usage: to_py_usage(&response.usage),
			tool_calls,
		},
	)
}

fn to_py_stream_end(end: StreamEnd) -> PyResult<PyStreamEnd> {
	let captured_usage = end.captured_usage.as_ref().map(to_py_usage);
	let captured_first_text = end.captured_first_text().map(|t| t.to_string());
	let captured_texts = end
		.captured_texts()
		.map(|values| values.into_iter().map(|value| value.to_string()).collect::<Vec<_>>());
	let captured_tool_calls = end
		.captured_tool_calls()
		.map(|calls| calls.into_iter().map(to_py_tool_call).collect::<PyResult<Vec<_>>>())
		.transpose()?;

	Ok(PyStreamEnd {
		captured_usage,
		captured_first_text,
		captured_texts,
		captured_reasoning_content: end.captured_reasoning_content,
		captured_tool_calls,
	})
}

fn to_py_stream_event(py: Python<'_>, event: ChatStreamEvent) -> PyResult<Py<PyChatStreamEvent>> {
	let (kind, content, tool_call, end) = match event {
		ChatStreamEvent::Start => ("start".to_string(), None, None, None),
		ChatStreamEvent::Chunk(chunk) => ("chunk".to_string(), Some(chunk.content), None, None),
		ChatStreamEvent::ReasoningChunk(chunk) => ("reasoning_chunk".to_string(), Some(chunk.content), None, None),
		ChatStreamEvent::ThoughtSignatureChunk(chunk) => {
			("thought_signature_chunk".to_string(), Some(chunk.content), None, None)
		}
		ChatStreamEvent::ToolCallChunk(chunk) => (
			"tool_call_chunk".to_string(),
			None,
			Some(to_py_tool_call(&chunk.tool_call)?),
			None,
		),
		ChatStreamEvent::End(stream_end) => ("end".to_string(), None, None, Some(to_py_stream_end(stream_end)?)),
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
) -> PyResult<PyClient> {
	let adapter_kind = parse_adapter_kind(&provider)?;
	let mut builder = genai::Client::builder();

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
		let service_target_resolver =
			ServiceTargetResolver::from_resolver_fn(move |mut service_target: genai::ServiceTarget| {
				if service_target.model.adapter_kind == adapter_kind {
					service_target.endpoint = Endpoint::from_owned(base_url.clone());
				}
				Ok(service_target)
			});
		builder = builder.with_service_target_resolver(service_target_resolver);
	}

	Ok(PyClient { inner: builder.build() })
}

fn to_runtime_error(error: genai::Error) -> PyErr {
	PyRuntimeError::new_err(error.to_string())
}
