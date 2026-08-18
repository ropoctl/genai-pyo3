//! Embedding pyclasses and their coercions.

use genai::embed::{EmbedOptions, EmbedResponse, Embedding};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::depythonize;
use std::collections::HashMap;

use crate::response::{PyUsage, to_py_usage};

#[pyclass(name = "EmbedOptions", from_py_object)]
#[derive(Clone, Default)]
pub(crate) struct PyEmbedOptions {
    /// Additional HTTP headers for the request.
    #[pyo3(get, set)]
    headers: Option<HashMap<String, String>>,
    #[pyo3(get, set)]
    capture_raw_body: Option<bool>,
    #[pyo3(get, set)]
    capture_usage: Option<bool>,
    /// Requested vector width, where the provider supports shortening.
    #[pyo3(get, set)]
    dimensions: Option<usize>,
    /// Wire encoding for the vectors ("float", "base64", ...).
    #[pyo3(get, set)]
    encoding_format: Option<String>,
    /// End-user identifier passed through to the provider.
    #[pyo3(get, set)]
    user: Option<String>,
    /// What the embedding is for. The vocabulary is provider-specific —
    /// Cohere takes "search_document" / "search_query" / "classification" /
    /// "clustering"; Gemini takes "SEMANTIC_SIMILARITY" /
    /// "RETRIEVAL_QUERY" / "RETRIEVAL_DOCUMENT" / "CLASSIFICATION" — and
    /// getting it wrong quietly costs retrieval quality, since documents
    /// and queries are meant to be embedded differently.
    #[pyo3(get, set)]
    embedding_type: Option<String>,
    /// How to handle over-long input, where the provider supports it
    /// (Cohere: "NONE", "START", "END").
    #[pyo3(get, set)]
    truncate: Option<String>,
}

#[pymethods]
impl PyEmbedOptions {
    #[new]
    #[pyo3(signature = (
        headers = None,
        capture_raw_body = None,
        capture_usage = None,
        dimensions = None,
        encoding_format = None,
        user = None,
        embedding_type = None,
        truncate = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        headers: Option<HashMap<String, String>>,
        capture_raw_body: Option<bool>,
        capture_usage: Option<bool>,
        dimensions: Option<usize>,
        encoding_format: Option<String>,
        user: Option<String>,
        embedding_type: Option<String>,
        truncate: Option<String>,
    ) -> Self {
        Self {
            headers,
            capture_raw_body,
            capture_usage,
            dimensions,
            encoding_format,
            user,
            embedding_type,
            truncate,
        }
    }
}

impl From<PyEmbedOptions> for EmbedOptions {
    fn from(options: PyEmbedOptions) -> Self {
        Self {
            headers: options.headers.map(Into::into),
            capture_raw_body: options.capture_raw_body,
            capture_usage: options.capture_usage,
            dimensions: options.dimensions,
            encoding_format: options.encoding_format,
            user: options.user,
            embedding_type: options.embedding_type,
            truncate: options.truncate,
        }
    }
}

/// Accept an optional `PyEmbedOptions` pyclass or a JSON-compatible dict.
pub(crate) fn coerce_embed_options(
    obj: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<EmbedOptions>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(typed) = obj.extract::<PyEmbedOptions>() {
        return Ok(Some(EmbedOptions::from(typed)));
    }
    let value: serde_json::Value = depythonize(&obj).map_err(|err| {
        PyValueError::new_err(format!(
            "embed options must be an EmbedOptions or a JSON-compatible dict: {err}"
        ))
    })?;
    serde_json::from_value(value).map(Some).map_err(|err| {
        PyValueError::new_err(format!("embed options dict does not match schema: {err}"))
    })
}

/// One embedding vector and where it came from in the request.
#[pyclass(name = "Embedding", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEmbedding {
    /// The vector itself.
    #[pyo3(get)]
    vector: Vec<f32>,
    /// Position of the corresponding input in the request — batch
    /// responses are not guaranteed to arrive in order.
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    dimensions: usize,
}

#[pymethods]
impl PyEmbedding {
    fn __len__(&self) -> usize {
        self.dimensions
    }

    fn __repr__(&self) -> String {
        format!(
            "Embedding(index={}, dimensions={})",
            self.index, self.dimensions
        )
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("vector", self.vector.clone())?;
        dict.set_item("index", self.index)?;
        dict.set_item("dimensions", self.dimensions)?;
        Ok(dict.into_any())
    }
}

#[pyclass(name = "EmbedResponse", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyEmbedResponse {
    #[pyo3(get)]
    embeddings: Vec<PyEmbedding>,
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
}

#[pymethods]
impl PyEmbedResponse {
    /// The first vector — the common case for a single-input request.
    #[getter]
    fn vector(&self) -> Option<Vec<f32>> {
        self.embeddings.first().map(|e| e.vector.clone())
    }

    /// Every vector, in request order.
    #[getter]
    fn vectors(&self) -> Vec<Vec<f32>> {
        let mut sorted = self.embeddings.clone();
        sorted.sort_by_key(|e| e.index);
        sorted.into_iter().map(|e| e.vector).collect()
    }

    fn __len__(&self) -> usize {
        self.embeddings.len()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = PyDict::new(py);
        let embeddings = PyList::empty(py);
        for embedding in &self.embeddings {
            embeddings.append(embedding.to_dict(py)?)?;
        }
        dict.set_item("embeddings", embeddings)?;
        dict.set_item("model_adapter_kind", self.model_adapter_kind.clone())?;
        dict.set_item("model_name", self.model_name.clone())?;
        dict.set_item(
            "provider_model_adapter_kind",
            self.provider_model_adapter_kind.clone(),
        )?;
        dict.set_item("provider_model_name", self.provider_model_name.clone())?;
        dict.set_item("usage", self.usage.to_dict(py)?)?;
        Ok(dict.into_any())
    }
}

pub(crate) fn to_py_embed_response(response: EmbedResponse) -> PyEmbedResponse {
    PyEmbedResponse {
        embeddings: response.embeddings.iter().map(to_py_embedding).collect(),
        model_adapter_kind: response.model_iden.adapter_kind.to_string(),
        model_name: response.model_iden.model_name.to_string(),
        provider_model_adapter_kind: response.provider_model_iden.adapter_kind.to_string(),
        provider_model_name: response.provider_model_iden.model_name.to_string(),
        usage: to_py_usage(&response.usage),
    }
}

fn to_py_embedding(embedding: &Embedding) -> PyEmbedding {
    PyEmbedding {
        vector: embedding.vector.clone(),
        index: embedding.index,
        dimensions: embedding.dimensions,
    }
}
