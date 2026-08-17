//! Python callbacks bridged into rust-genai's resolver hooks.
//!
//! rust-genai lets an application intercept three points in request
//! resolution: which credential to use, what a model name maps to, and
//! where the whole request should go. Each hook is a Rust closure; these
//! wrappers let a plain Python function stand in for one.
//!
//! # Constraints, and why they exist
//!
//! **Sync callables only.** The hooks used here are rust-genai's
//! synchronous variants. Handing them an `async def` would produce a
//! coroutine object that is never awaited and silently treated as the
//! answer, so coroutine functions are rejected at registration time
//! instead of misbehaving at request time.
//!
//! **The callback runs while holding the GIL, on whichever thread is
//! driving the request.** For the async entry points that is a Tokio
//! worker, which must attach to the interpreter to make the call; for the
//! blocking `Client.chat()` it is the caller's own thread, since
//! `Runtime::block_on` drives the root future inline. Either way, blocking
//! inside the callback — a vault round-trip, a subprocess — stalls the
//! request *and* holds the GIL, which stalls every other Python thread.
//! Keep callbacks fast; cache in Python.
//!
//! **Do not re-enter the client.** Calling the blocking `Client.chat()`
//! from a callback is refused with a clear error rather than panicking
//! inside Tokio; the async entry points are the way back in.
//!
//! **Exception type is not preserved.** A Python exception crosses the
//! boundary as text (`"ValueError: no key for tenant-7"` plus traceback)
//! and re-emerges as `ResolverError`. Carrying the original exception
//! object across would need a per-call side channel, which concurrent
//! requests would race on.

use genai::ModelIden;
use genai::ServiceTarget;
use genai::resolver::{AuthData, Error as ResolverError, Result as ResolverResult};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::target::{PyAuthData, PyModelIden, PyServiceTarget};

/// A Python callable held for invocation from Tokio worker threads.
///
/// The handle is an `Arc<Py<PyAny>>` rather than a bare `Py<PyAny>`
/// because rust-genai's mapper and service-target hooks require `Clone`,
/// and pyo3 only implements `Clone for Py<T>` under the `py-clone`
/// feature — where it panics if the calling thread is not attached to the
/// interpreter, which a Tokio worker generally is not.
#[derive(Clone)]
pub(crate) struct PyCallback {
    func: Arc<Py<PyAny>>,
    /// Hook name, used to make failures say which callback misbehaved.
    role: &'static str,
}

impl PyCallback {
    /// Accept a callable, rejecting the forms that cannot work here.
    pub(crate) fn new(func: Bound<'_, PyAny>, role: &'static str) -> PyResult<Self> {
        if !func.is_callable() {
            return Err(PyTypeError::new_err(format!("{role} must be callable")));
        }
        let py = func.py();
        let inspect = py.import("inspect")?;
        for probe in ["iscoroutinefunction", "isasyncgenfunction"] {
            if inspect
                .call_method1(probe, (&func,))?
                .is_truthy()
                .unwrap_or(false)
            {
                return Err(PyTypeError::new_err(format!(
                    "{role} must be a regular function, not an async one — it is \
                     called synchronously from a Tokio worker thread, so a \
                     coroutine would never be awaited. Do the async work ahead \
                     of time and close over the result."
                )));
            }
        }
        Ok(Self {
            func: Arc::new(func.unbind()),
            role,
        })
    }

    fn call_one<'py, T>(&self, py: Python<'py>, arg: T) -> ResolverResult<Bound<'py, PyAny>>
    where
        (T,): pyo3::call::PyCallArgs<'py>,
    {
        self.func
            .bind(py)
            .call1((arg,))
            .map_err(|err| self.failed(py, err))
    }

    fn failed(&self, py: Python<'_>, err: PyErr) -> ResolverError {
        ResolverError::Custom(format!("{} raised {}", self.role, format_py_err(py, err)))
    }

    fn bad_return(&self, expected: &str, got: &Bound<'_, PyAny>) -> ResolverError {
        let type_name = got
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        ResolverError::Custom(format!(
            "{} must return {expected}, got {type_name}",
            self.role
        ))
    }
}

/// Render a Python exception as `"Type: message"` plus its traceback.
fn format_py_err(py: Python<'_>, err: PyErr) -> String {
    let type_name = err
        .get_type(py)
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "Exception".to_string());
    let message = err.value(py).to_string();
    let mut rendered = format!("{type_name}: {message}");
    if let Some(traceback) = err.traceback(py)
        && let Ok(formatted) = traceback.format()
    {
        rendered.push('\n');
        rendered.push_str(formatted.trim_end());
    }
    rendered
}

/// `fn(ModelIden) -> AuthData | None`
///
/// Returning `None` means "not mine" and falls through to whatever the
/// client was configured with declaratively, then to the adapter's default
/// credential environment variable.
pub(crate) fn auth_resolver_fn(
    callback: PyCallback,
) -> impl Fn(ModelIden) -> ResolverResult<Option<AuthData>> + Send + Sync + Clone + 'static {
    move |model_iden: ModelIden| {
        Python::attach(|py| {
            let arg = PyModelIden { inner: model_iden };
            let result = callback.call_one(py, arg)?;
            if result.is_none() {
                return Ok(None);
            }
            let auth = result
                .extract::<PyAuthData>()
                .map_err(|_| callback.bad_return("an AuthData or None", &result))?;
            Ok(Some(auth.inner))
        })
    }
}

/// `fn(ModelIden) -> ModelIden | str`
///
/// A bare string is taken as a rename that keeps the adapter, which is
/// what an alias table wants: `lambda iden: ALIASES.get(iden.model_name,
/// iden.model_name)`.
pub(crate) fn model_mapper_fn(
    callback: PyCallback,
) -> impl Fn(ModelIden) -> ResolverResult<ModelIden> + Send + Sync + Clone + 'static {
    move |model_iden: ModelIden| {
        Python::attach(|py| {
            let original = model_iden.clone();
            let arg = PyModelIden { inner: model_iden };
            let result = callback.call_one(py, arg)?;
            if let Ok(iden) = result.extract::<PyModelIden>() {
                return Ok(iden.inner);
            }
            if let Ok(name) = result.extract::<String>() {
                return Ok(ModelIden::new(original.adapter_kind, name));
            }
            Err(callback.bad_return("a ModelIden or str", &result))
        })
    }
}

/// `fn(ServiceTarget) -> ServiceTarget`
///
/// Receives the target as already resolved — including any endpoint the
/// client was configured with — and may replace any part of it.
pub(crate) fn service_target_resolver_fn(
    callback: PyCallback,
    declarative: impl Fn(ServiceTarget) -> ServiceTarget + Send + Sync + Clone + 'static,
) -> impl Fn(ServiceTarget) -> ResolverResult<ServiceTarget> + Send + Sync + Clone + 'static {
    move |target: ServiceTarget| {
        let target = declarative(target);
        Python::attach(|py| {
            let arg = PyServiceTarget { inner: target };
            let result = callback.call_one(py, arg)?;
            let resolved = result
                .extract::<PyServiceTarget>()
                .map_err(|_| callback.bad_return("a ServiceTarget", &result))?;
            Ok(resolved.inner)
        })
    }
}
