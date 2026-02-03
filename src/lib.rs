//! # loom-rs
//!
//! **Weaving multiple threads together**
//!
//! A bespoke thread pool runtime combining tokio and rayon with CPU pinning capabilities.
//!
//! ## Features
//!
//! - **Hybrid Runtime**: Combines tokio for async I/O with rayon for CPU-bound parallel work
//! - **CPU Pinning**: Automatically pins threads to specific CPUs for consistent performance
//! - **Zero Allocation**: `spawn_compute()` uses per-type pools for zero allocation after warmup
//! - **Scoped Compute**: `scope_compute()` allows borrowing local data for parallel work
//! - **Flexible Configuration**: Configure via files (TOML/YAML/JSON), environment variables, or code
//! - **CLI Integration**: Built-in clap support for command-line overrides
//! - **CUDA NUMA Awareness**: Optional feature for selecting CPUs local to a GPU (Linux only)
//!
//! ## Quick Start
//!
//! ```ignore
//! use loom_rs::LoomBuilder;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let runtime = LoomBuilder::new()
//!         .prefix("myapp")
//!         .tokio_threads(2)
//!         .rayon_threads(6)
//!         .build()?;
//!
//!     runtime.block_on(async {
//!         // Spawn tracked async I/O task
//!         let io_handle = runtime.spawn_async(async {
//!             // Async I/O work
//!             42
//!         });
//!
//!         // Spawn tracked compute task and await result (zero alloc after warmup)
//!         let result = runtime.spawn_compute(|| {
//!             // CPU-bound work on rayon
//!             (0..1000000).sum::<i64>()
//!         }).await;
//!         println!("Compute result: {}", result);
//!
//!         // Zero-overhead parallel iterators
//!         let _sum = runtime.install(|| {
//!             use rayon::prelude::*;
//!             (0..1000).into_par_iter().sum::<i64>()
//!         });
//!
//!         // Wait for async task
//!         let io_result = io_handle.await.unwrap();
//!         println!("I/O result: {}", io_result);
//!     });
//!
//!     // Graceful shutdown
//!     runtime.block_until_idle();
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Ergonomic Access
//!
//! Use `current_runtime()` or `spawn_compute()` from anywhere in the runtime:
//!
//! ```ignore
//! use loom_rs::LoomBuilder;
//!
//! let runtime = LoomBuilder::new().build()?;
//!
//! runtime.block_on(async {
//!     // No need to pass &runtime around
//!     let result = loom_rs::spawn_compute(|| expensive_work()).await;
//!
//!     // Or get the runtime explicitly
//!     let rt = loom_rs::current_runtime().unwrap();
//!     rt.spawn_async(async { /* ... */ });
//! });
//! ```
//!
//! ## Configuration
//!
//! Configuration sources are merged in order (later sources override earlier):
//!
//! 1. Default values
//! 2. Config files (via `.file()`)
//! 3. Environment variables (via `.env_prefix()`)
//! 4. Programmatic overrides
//! 5. CLI arguments (via `.with_cli_args()`)
//!
//! ### Config File Example (TOML)
//!
//! ```toml
//! prefix = "myapp"
//! cpuset = "0-7,16-23"
//! tokio_threads = 2
//! rayon_threads = 14
//! compute_pool_size = 64
//! ```
//!
//! ### Environment Variables
//!
//! With `.env_prefix("LOOM")`:
//! - `LOOM_PREFIX=myapp`
//! - `LOOM_CPUSET=0-7`
//! - `LOOM_TOKIO_THREADS=2`
//! - `LOOM_RAYON_THREADS=6`
//!
//! ### CLI Arguments
//!
//! ```ignore
//! use clap::Parser;
//! use loom_rs::{LoomBuilder, LoomArgs};
//!
//! #[derive(Parser)]
//! struct MyArgs {
//!     #[command(flatten)]
//!     loom: LoomArgs,
//! }
//!
//! let args = MyArgs::parse();
//! let runtime = LoomBuilder::new()
//!     .file("config.toml")
//!     .env_prefix("LOOM")
//!     .with_cli_args(&args.loom)
//!     .build()?;
//! ```
//!
//! ## CPU Set Format
//!
//! The `cpuset` option accepts a string in Linux taskset/numactl format:
//! - Single CPUs: `"0"`, `"5"`
//! - Ranges: `"0-7"`, `"16-23"`
//! - Mixed: `"0-3,8-11"`, `"0,2,4,6-8"`
//!
//! ## CUDA Support
//!
//! With the `cuda` feature enabled (Linux only), you can configure the runtime
//! to use CPUs local to a specific CUDA GPU:
//!
//! ```ignore
//! let runtime = LoomBuilder::new()
//!     .cuda_device_id(0)  // Use CPUs near GPU 0
//!     .build()?;
//! ```
//!
//! ## Thread Naming
//!
//! Threads are named with the configured prefix:
//! - Tokio threads: `{prefix}-tokio-0000`, `{prefix}-tokio-0001`, ...
//! - Rayon threads: `{prefix}-rayon-0000`, `{prefix}-rayon-0001`, ...

pub(crate) mod affinity;
pub(crate) mod bridge;
pub mod builder;
pub mod config;
pub(crate) mod context;
pub mod cpuset;
pub mod error;
pub mod mab;
pub mod metrics;
pub(crate) mod pool;
pub mod runtime;
pub mod stream;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use builder::{LoomArgs, LoomBuilder};
pub use config::LoomConfig;
pub use context::current_runtime;
pub use error::{LoomError, Result};
pub use mab::{Arm, ComputeHint, ComputeHintProvider, MabKnobs, MabScheduler};
pub use metrics::LoomMetrics;
pub use runtime::LoomRuntime;
pub use stream::ComputeStreamExt;

// Re-export rayon::Scope for ergonomic use with scope_compute
pub use rayon::Scope;

/// Spawn compute work using the current runtime.
///
/// This is a convenience function for `loom_rs::current_runtime().unwrap().spawn_compute(f)`.
/// It allows spawning compute work from anywhere within a loom runtime without
/// explicitly passing the runtime reference.
///
/// # Panics
///
/// Panics if called outside a loom runtime context (i.e., not within `block_on`,
/// a tokio worker thread, or a rayon worker thread managed by the runtime).
///
/// # Performance
///
/// Same as `LoomRuntime::spawn_compute()`:
/// - 0 bytes allocation after warmup (pool hit)
/// - ~100-500ns overhead
///
/// # Example
///
/// ```ignore
/// use loom_rs::LoomBuilder;
///
/// let runtime = LoomBuilder::new().build()?;
///
/// runtime.block_on(async {
///     // No need to pass &runtime around
///     let result = loom_rs::spawn_compute(|| {
///         expensive_work()
///     }).await;
/// });
/// ```
pub async fn spawn_compute<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    current_runtime()
        .expect("spawn_compute called outside loom runtime")
        .spawn_compute(f)
        .await
}

/// Try to spawn compute work using the current runtime.
///
/// Like `spawn_compute()`, but returns `None` if not in a runtime context
/// instead of panicking.
///
/// # Example
///
/// ```ignore
/// if let Some(future) = loom_rs::try_spawn_compute(|| work()) {
///     let result = future.await;
/// }
/// ```
pub fn try_spawn_compute<F, R>(f: F) -> Option<impl std::future::Future<Output = R>>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    current_runtime().map(|rt| {
        let rt = rt;
        async move { rt.spawn_compute(f).await }
    })
}

/// Spawn adaptive work using the current runtime.
///
/// This is a convenience function for `loom_rs::current_runtime().unwrap().spawn_adaptive(f)`.
/// Uses MAB (Multi-Armed Bandit) to learn whether to inline or offload work.
///
/// # Panics
///
/// Panics if called outside a loom runtime context.
///
/// # Example
///
/// ```ignore
/// use loom_rs::LoomBuilder;
///
/// let runtime = LoomBuilder::new().build()?;
///
/// runtime.block_on(async {
///     // MAB adaptively decides inline vs offload
///     let result = loom_rs::spawn_adaptive(|| {
///         process_work()
///     }).await;
/// });
/// ```
pub async fn spawn_adaptive<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    current_runtime()
        .expect("spawn_adaptive called outside loom runtime")
        .spawn_adaptive(f)
        .await
}

/// Spawn adaptive work with hint using the current runtime.
///
/// Like `spawn_adaptive()`, but provides a hint to guide cold-start behavior.
///
/// # Panics
///
/// Panics if called outside a loom runtime context.
///
/// # Example
///
/// ```ignore
/// use loom_rs::{LoomBuilder, ComputeHint};
///
/// let runtime = LoomBuilder::new().build()?;
///
/// runtime.block_on(async {
///     // Hint that this is likely expensive work
///     let result = loom_rs::spawn_adaptive_with_hint(
///         ComputeHint::High,
///         || expensive_work()
///     ).await;
/// });
/// ```
pub async fn spawn_adaptive_with_hint<F, R>(hint: ComputeHint, f: F) -> R
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    current_runtime()
        .expect("spawn_adaptive_with_hint called outside loom runtime")
        .spawn_adaptive_with_hint(hint, f)
        .await
}

/// Execute a scoped parallel computation using the current runtime.
///
/// This is a convenience function for `loom_rs::current_runtime().unwrap().scope_compute(f)`.
/// It allows borrowing local variables from the async context for use in parallel work.
///
/// # Panics
///
/// Panics if called outside a loom runtime context (i.e., not within `block_on`,
/// a tokio worker thread, or a rayon worker thread managed by the runtime).
///
/// # Performance
///
/// | Aspect | Value |
/// |--------|-------|
/// | Allocation | ~96 bytes per call (not pooled) |
/// | Overhead | Comparable to `spawn_compute()` |
///
/// # Cancellation Safety
///
/// If the future is dropped before completion (e.g., via `select!` or timeout),
/// the drop will **block** until the rayon scope finishes. This is necessary
/// to prevent use-after-free of borrowed data.
///
/// # Example
///
/// ```ignore
/// use loom_rs::LoomBuilder;
/// use std::sync::atomic::{AtomicI32, Ordering};
///
/// let runtime = LoomBuilder::new().build()?;
///
/// runtime.block_on(async {
///     let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
///     let sum = AtomicI32::new(0);
///
///     // Borrow `data` and `sum` for parallel processing - no need to pass &runtime
///     loom_rs::scope_compute(|s| {
///         let (left, right) = data.split_at(data.len() / 2);
///         let sum_ref = &sum;
///
///         s.spawn(move |_| {
///             sum_ref.fetch_add(left.iter().sum::<i32>(), Ordering::Relaxed);
///         });
///         s.spawn(move |_| {
///             sum_ref.fetch_add(right.iter().sum::<i32>(), Ordering::Relaxed);
///         });
///     }).await;
///
///     // data and sum are still valid here
///     println!("Sum of {:?} = {}", data, sum.load(Ordering::Relaxed));
/// });
/// ```
pub async fn scope_compute<'env, F, R>(f: F) -> R
where
    F: FnOnce(&Scope<'env>) -> R + Send + 'env,
    R: Send + 'env,
{
    current_runtime()
        .expect("scope_compute called outside loom runtime")
        .scope_compute(f)
        .await
}
