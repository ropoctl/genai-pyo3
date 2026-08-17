//! Routing types: who to talk to, with which credential, about which model.
//!
//! rust-genai already models this as [`ServiceTarget`] — endpoint + auth +
//! [`ModelIden`] — and `exec_chat` accepts one directly via `ModelSpec`.
//! Exposing these lets a single `Client` reach every configured provider:
//! keep a `dict[str, ServiceTarget]` of named endpoints and pass one per
//! call, instead of building a separate `Client` per provider.

use genai::ModelIden;
use genai::adapter::AdapterKind;
use genai::resolver::{AuthData, Endpoint};
use genai::{ModelSpec, ServiceTarget};
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Canonical lowercase names for every non-`Custom` adapter, mirroring the
/// `adapter_kind_str_maps!` table in rust-genai. Kept as data here because
/// upstream exposes `from_lower_str` but no way to enumerate; the unit test
/// at the bottom of this file round-trips every entry, so a rename upstream
/// breaks the build rather than silently yielding a bogus list.
const ADAPTER_KIND_NAMES: [&str; 29] = [
    "openai",
    "openai_resp",
    "gemini",
    "anthropic",
    "fireworks",
    "together",
    "groq",
    "aihubmix",
    "kimi",
    "mimo",
    "moonshot",
    "nebius",
    "xai",
    "deepseek",
    "zai",
    "bigmodel",
    "aliyun",
    "baidu",
    "cohere",
    "ollama",
    "ollama_cloud",
    "omlx",
    "vertex",
    "github_copilot",
    "opencode_go",
    "bedrock_api",
    "open_router",
    "atlascloud",
    "minimax",
];

/// Which provider protocol to speak.
#[pyclass(name = "AdapterKind", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub(crate) struct PyAdapterKind {
    pub(crate) inner: AdapterKind,
}

#[pymethods]
impl PyAdapterKind {
    /// Parse a provider name, case-insensitively (`"openai"`, `"OpenAI"`,
    /// `"open_router"`, ...).
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_adapter_kind(name)?,
        })
    }

    /// The `genai_<n>` custom-adapter family, which resolves its endpoint
    /// and key from `GENAI_<n>_ENDPOINT` / `GENAI_<n>_API_KEY`.
    #[staticmethod]
    fn custom(index: u8) -> Self {
        Self {
            inner: AdapterKind::Custom(index),
        }
    }

    /// Every built-in adapter kind. Excludes the `Custom(n)` family, which
    /// is parameterised rather than enumerable.
    #[staticmethod]
    fn all() -> Vec<Self> {
        ADAPTER_KIND_NAMES
            .iter()
            .filter_map(|name| AdapterKind::from_lower_str(name))
            .map(|inner| Self { inner })
            .collect()
    }

    /// Infer the adapter from a model name the way rust-genai does when
    /// handed a bare string (`"gpt-4o"` → openai, `"claude-…"` → anthropic,
    /// namespaced `"groq::…"` → groq, anything unrecognised → ollama).
    #[staticmethod]
    fn from_model(model: &str) -> PyResult<Self> {
        AdapterKind::from_model(model)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Canonical lowercase name, the form `AdapterKind(...)` accepts back.
    #[getter]
    fn name(&self) -> String {
        self.inner.as_lower_str().to_string()
    }

    /// rust-genai's display spelling (`"OpenAI"`, `"OpenAIResp"`, ...).
    #[getter]
    fn display_name(&self) -> String {
        self.inner.as_str().to_string()
    }

    /// Environment variable this adapter reads a key from by default, if
    /// it has one (`"OPENAI_API_KEY"`, ...). `None` for adapters that need
    /// no credential, such as a local Ollama.
    #[getter]
    fn default_key_env_name(&self) -> Option<String> {
        self.inner.default_key_env_name().map(str::to_string)
    }

    fn __str__(&self) -> String {
        self.inner.as_lower_str().to_string()
    }

    fn __repr__(&self) -> String {
        format!("AdapterKind({:?})", self.inner.as_lower_str())
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyValueError::new_err(
                "AdapterKind supports only == and != comparisons",
            )),
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// An adapter paired with a model name — the fully-resolved identity of a
/// model, independent of where it is served from.
#[pyclass(name = "ModelIden", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyModelIden {
    pub(crate) inner: ModelIden,
}

#[pymethods]
impl PyModelIden {
    #[new]
    fn new(adapter_kind: AdapterKindLike, model_name: String) -> PyResult<Self> {
        Ok(Self {
            inner: ModelIden::new(adapter_kind.resolve()?, model_name),
        })
    }

    #[getter]
    fn adapter_kind(&self) -> PyAdapterKind {
        PyAdapterKind {
            inner: self.inner.adapter_kind,
        }
    }

    #[getter]
    fn model_name(&self) -> String {
        self.inner.model_name.to_string()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelIden({:?}, {:?})",
            self.inner.adapter_kind.as_lower_str(),
            self.inner.model_name.to_string()
        )
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyValueError::new_err(
                "ModelIden supports only == and != comparisons",
            )),
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

/// A provider base URL.
#[pyclass(name = "Endpoint", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEndpoint {
    pub(crate) inner: Endpoint,
}

#[pymethods]
impl PyEndpoint {
    #[new]
    fn new(base_url: String) -> PyResult<Self> {
        let trimmed = base_url.trim();
        if trimmed.is_empty() {
            return Err(PyValueError::new_err("endpoint base_url cannot be empty"));
        }
        Ok(Self {
            inner: Endpoint::from_owned(trimmed.to_string()),
        })
    }

    #[getter]
    fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    fn __str__(&self) -> String {
        self.inner.base_url().to_string()
    }

    fn __repr__(&self) -> String {
        format!("Endpoint({:?})", self.inner.base_url())
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner.base_url() == other.inner.base_url()),
            CompareOp::Ne => Ok(self.inner.base_url() != other.inner.base_url()),
            _ => Err(PyValueError::new_err(
                "Endpoint supports only == and != comparisons",
            )),
        }
    }
}

/// How to authenticate against a provider.
///
/// Secret material is never returned to Python and never appears in
/// `repr()` — an `AuthData` that printed its own key would leak into every
/// log line that logged a `ServiceTarget`.
#[pyclass(name = "AuthData", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyAuthData {
    pub(crate) inner: AuthData,
}

#[pymethods]
impl PyAuthData {
    /// Read the key from an environment variable at request time.
    #[staticmethod]
    fn from_env(env_name: String) -> Self {
        Self {
            inner: AuthData::from_env(env_name),
        }
    }

    /// Use a literal key.
    #[staticmethod]
    fn key(value: String) -> Self {
        Self {
            inner: AuthData::from_single(value),
        }
    }

    /// Replace the request URL and headers wholesale — for gateways whose
    /// auth scheme is not "put a key in the usual header".
    #[staticmethod]
    fn request_override(url: String, headers: HashMap<String, String>) -> PyResult<Self> {
        let url = url.trim().to_string();
        if url.is_empty() {
            return Err(PyValueError::new_err(
                "request override url cannot be empty",
            ));
        }
        Ok(Self {
            inner: AuthData::RequestOverride {
                url,
                headers: genai::Headers::from(headers.into_iter().collect::<Vec<_>>()),
            },
        })
    }

    /// Multiple named credential parts, for adapters that need more than a
    /// single key.
    #[staticmethod]
    fn multi(values: HashMap<String, String>) -> Self {
        Self {
            inner: AuthData::from_multi(values),
        }
    }

    /// No credential at all — a local Ollama or vLLM with auth disabled.
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: AuthData::None,
        }
    }

    /// Which form this is: `"from_env"`, `"key"`, `"request_override"`,
    /// `"multi_keys"`, or `"none"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            AuthData::FromEnv(_) => "from_env",
            AuthData::Key(_) => "key",
            AuthData::RequestOverride { .. } => "request_override",
            AuthData::MultiKeys(_) => "multi_keys",
            AuthData::None => "none",
        }
    }

    /// The environment variable name, for `from_env` only.
    #[getter]
    fn env_name(&self) -> Option<String> {
        match &self.inner {
            AuthData::FromEnv(name) => Some(name.clone()),
            _ => None,
        }
    }

    /// The override URL, for `request_override` only.
    #[getter]
    fn url(&self) -> Option<String> {
        match &self.inner {
            AuthData::RequestOverride { url, .. } => Some(url.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            AuthData::FromEnv(name) => format!("AuthData.from_env({name:?})"),
            AuthData::Key(_) => "AuthData.key(***)".to_string(),
            AuthData::RequestOverride { url, .. } => {
                format!("AuthData.request_override({url:?}, ***)")
            }
            AuthData::MultiKeys(_) => "AuthData.multi(***)".to_string(),
            AuthData::None => "AuthData.none()".to_string(),
        }
    }
}

/// A named inference endpoint: where to send the request, how to
/// authenticate, and which model to ask for.
///
/// Pass one wherever a model name is accepted to route a single call
/// without touching the client's configuration.
#[pyclass(name = "ServiceTarget", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyServiceTarget {
    pub(crate) inner: ServiceTarget,
}

#[pymethods]
impl PyServiceTarget {
    #[new]
    #[pyo3(signature = (endpoint, auth, model))]
    fn new(endpoint: EndpointLike, auth: PyAuthData, model: PyModelIden) -> PyResult<Self> {
        Ok(Self {
            inner: ServiceTarget {
                endpoint: endpoint.resolve()?,
                auth: auth.inner,
                model: model.inner,
            },
        })
    }

    #[getter]
    fn endpoint(&self) -> PyEndpoint {
        PyEndpoint {
            inner: self.inner.endpoint.clone(),
        }
    }

    #[getter]
    fn auth(&self) -> PyAuthData {
        PyAuthData {
            inner: self.inner.auth.clone(),
        }
    }

    #[getter]
    fn model(&self) -> PyModelIden {
        PyModelIden {
            inner: self.inner.model.clone(),
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("endpoint", self.inner.endpoint.base_url())?;
        dict.set_item("auth", self.auth().__repr__())?;
        dict.set_item("adapter_kind", self.inner.model.adapter_kind.as_lower_str())?;
        dict.set_item("model_name", self.inner.model.model_name.to_string())?;
        Ok(dict.into_any())
    }

    fn __repr__(&self) -> String {
        format!(
            "ServiceTarget(endpoint={:?}, auth={}, model={:?}/{:?})",
            self.inner.endpoint.base_url(),
            self.auth().__repr__(),
            self.inner.model.adapter_kind.as_lower_str(),
            self.inner.model.model_name.to_string(),
        )
    }
}

/// A `str` or an `AdapterKind`, wherever an adapter is named.
#[derive(FromPyObject)]
pub(crate) enum AdapterKindLike {
    Kind(PyAdapterKind),
    Name(String),
}

impl AdapterKindLike {
    pub(crate) fn resolve(self) -> PyResult<AdapterKind> {
        match self {
            AdapterKindLike::Kind(kind) => Ok(kind.inner),
            AdapterKindLike::Name(name) => parse_adapter_kind(&name),
        }
    }
}

/// A `str` or an `Endpoint`, wherever a base URL is accepted.
#[derive(FromPyObject)]
pub(crate) enum EndpointLike {
    Endpoint(PyEndpoint),
    Url(String),
}

impl EndpointLike {
    pub(crate) fn resolve(self) -> PyResult<Endpoint> {
        match self {
            EndpointLike::Endpoint(endpoint) => Ok(endpoint.inner),
            EndpointLike::Url(url) => Ok(PyEndpoint::new(url)?.inner),
        }
    }
}

/// What a call's `model` argument may be.
///
/// The three forms line up with rust-genai's `ModelSpec` and pick how much
/// resolution still has to happen:
///
/// - `str` — infer the adapter from the name, resolve auth and endpoint
///   through the client's configuration;
/// - `ModelIden` — adapter stated outright, auth and endpoint still
///   resolved through the client;
/// - `ServiceTarget` — everything stated; only the service-target resolver
///   still runs.
#[derive(FromPyObject)]
pub(crate) enum ModelLike {
    Target(PyServiceTarget),
    Iden(PyModelIden),
    Name(String),
}

impl ModelLike {
    pub(crate) fn into_spec(self) -> ModelSpec {
        match self {
            ModelLike::Target(target) => ModelSpec::from_target(target.inner),
            ModelLike::Iden(iden) => ModelSpec::from_iden(iden.inner),
            ModelLike::Name(name) => ModelSpec::from_name(name),
        }
    }
}

pub(crate) fn parse_adapter_kind(provider: &str) -> PyResult<AdapterKind> {
    let normalized = provider.trim().to_ascii_lowercase();
    AdapterKind::from_lower_str(&normalized).ok_or_else(|| {
        PyValueError::new_err(format!(
            "invalid provider {provider:?}. expected one of: {}",
            ADAPTER_KIND_NAMES.join(", ")
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ADAPTER_KIND_NAMES` is hand-maintained against upstream's
    /// `adapter_kind_str_maps!`. If upstream renames a variant this fails
    /// loudly instead of `AdapterKind.all()` quietly dropping an entry.
    #[test]
    fn every_listed_adapter_name_round_trips() {
        for name in ADAPTER_KIND_NAMES {
            let kind = AdapterKind::from_lower_str(name)
                .unwrap_or_else(|| panic!("unknown adapter kind name: {name}"));
            assert_eq!(kind.as_lower_str(), name);
        }
        assert_eq!(PyAdapterKind::all().len(), ADAPTER_KIND_NAMES.len());
    }
}
