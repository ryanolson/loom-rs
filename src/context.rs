//! Thread-local runtime context.
//!
//! This module provides thread-local storage for accessing the current loom runtime
//! from any thread managed by the runtime. This enables ergonomic access to runtime
//! methods without passing the runtime reference explicitly.
//!
//! # Example
//!
//! ```ignore
//! use loom_rs::LoomBuilder;
//!
//! let runtime = LoomBuilder::new().build()?;
//!
//! runtime.block_on(async {
//!     // Works anywhere in the runtime's threads
//!     let result = loom_rs::spawn_compute(|| {
//!         expensive_work()
//!     }).await;
//! });
//! ```

use std::cell::RefCell;
use std::sync::Weak;

use crate::runtime::{LoomRuntime, LoomRuntimeInner};

thread_local! {
    static CURRENT_RUNTIME: RefCell<Option<Weak<LoomRuntimeInner>>> = const { RefCell::new(None) };
}

/// Get the current loom runtime from thread-local storage.
///
/// Returns `Some` when called from a thread managed by a `LoomRuntime`
/// (tokio workers, rayon workers, or within `block_on`).
///
/// Returns `None` when called from outside a loom runtime context.
///
/// # Example
///
/// ```ignore
/// runtime.block_on(async {
///     // Works anywhere in the runtime's threads
///     let rt = loom_rs::current_runtime().unwrap();
///     rt.spawn_compute(|| work()).await;
/// });
/// ```
pub fn current_runtime() -> Option<LoomRuntime> {
    CURRENT_RUNTIME.with(|rt| {
        rt.borrow()
            .as_ref()
            .and_then(|weak| weak.upgrade())
            .map(LoomRuntime::from_inner)
    })
}

/// Set the current runtime for this thread.
///
/// This is called internally by the runtime when starting tokio and rayon threads.
pub(crate) fn set_current_runtime(runtime: Weak<LoomRuntimeInner>) {
    CURRENT_RUNTIME.with(|rt| {
        *rt.borrow_mut() = Some(runtime);
    });
}

/// Clear the current runtime for this thread.
///
/// This is called internally by the runtime when threads are stopping.
pub(crate) fn clear_current_runtime() {
    CURRENT_RUNTIME.with(|rt| {
        *rt.borrow_mut() = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_runtime_none_initially() {
        // Should be None when not in a runtime context
        assert!(current_runtime().is_none());
    }

    #[test]
    fn test_set_and_clear_runtime() {
        // Create a dummy inner runtime for testing
        // We can't easily test this without a real runtime, so just test the API
        assert!(current_runtime().is_none());

        // After clear, should still be None
        clear_current_runtime();
        assert!(current_runtime().is_none());
    }
}
