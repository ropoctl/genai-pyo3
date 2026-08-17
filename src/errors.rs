//! Translation from rust-genai's error type into Python exceptions.
//!
//! Currently a flat mapping onto `RuntimeError`; the typed exception
//! hierarchy replaces this in a follow-up commit.

use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

pub(crate) fn to_runtime_error(error: genai::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
