//! Loom runtime implementation.
//!
//! The runtime combines a tokio async runtime with a rayon thread pool,
//! both configured with CPU pinning.
//!
//! # Performance
//!
//! This module is designed for zero unnecessary overhead:
//! - `spawn_async()`: ~10ns overhead (TaskTracker token only)
//! - `spawn_compute()`: ~100-500ns (cross-thread signaling, 0 bytes after warmup)
//! - `install()`: ~0ns (zero overhead, direct rayon access)
//!
//! # Thread Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     LoomRuntime                              │
//! │  pools: ComputePoolRegistry (per-type lock-free pools)      │
//! │  (One pool per result type, shared across all threads)      │
//! └─────────────────────────────────────────────────────────────┘
//!          │ on_thread_start           │ start_handler
//!          ▼                           ▼
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │   Tokio Workers     │     │   Rayon Workers     │
//! │  thread_local! {    │     │  thread_local! {    │
//! │    RUNTIME: Weak<>  │     │    RUNTIME: Weak<>  │
//! │  }                  │     │  }                  │
//! └─────────────────────┘     └─────────────────────┘
//! ```

use crate::affinity::{pin_to_cpu, CpuAllocator};
use crate::bridge::{
    PooledRayonTask, ScopedCompletion, ScopedComputeFuture, ScopedTaskState, TaskState,
};
use crate::config::LoomConfig;
use crate::context::{clear_current_runtime, set_current_runtime};
use crate::cpuset::{available_cpus, format_cpuset, parse_and_validate_cpuset};
use crate::error::{LoomError, Result};
use crate::mab::{Arm, ComputeHint, Context, FunctionKey, MabKnobs, MabScheduler};
use crate::metrics::LoomMetrics;
use crate::pool::ComputePoolRegistry;

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Instant;
use tokio::sync::Notify;
use tokio_util::task::TaskTracker;
use tracing::{debug, info, warn};

/// State for tracking in-flight compute tasks.
///
/// Combines the task counter with a notification mechanism for efficient
/// shutdown waiting (avoids spin loops).
struct ComputeTaskState {
    /// Number of tasks currently executing on rayon
    count: AtomicUsize,
    /// Notified when count reaches 0
    notify: Notify,
}

impl ComputeTaskState {
    fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
            notify: Notify::new(),
        }
    }
}

/// Guard for tracking async task metrics.
///
/// Panic-safe: task_completed is called even if the future panics.
struct AsyncMetricsGuard {
    inner: Arc<LoomRuntimeInner>,
}

impl AsyncMetricsGuard {
    fn new(inner: Arc<LoomRuntimeInner>) -> Self {
        inner.prometheus_metrics.task_started();
        Self { inner }
    }
}

impl Drop for AsyncMetricsGuard {
    fn drop(&mut self) {
        self.inner.prometheus_metrics.task_completed();
    }
}

/// Guard for tracking compute task state and metrics.
///
/// Panic-safe: executes even if the task closure panics.
///
/// SAFETY: The state lives in LoomRuntimeInner which outlives all rayon tasks
/// because block_until_idle waits for compute_tasks to reach 0.
struct ComputeTaskGuard {
    state: *const ComputeTaskState,
    metrics: *const LoomMetrics,
}

unsafe impl Send for ComputeTaskGuard {}

impl ComputeTaskGuard {
    /// Create a new guard, tracking submission in MAB metrics.
    ///
    /// This should be called BEFORE spawning on rayon.
    fn new(state: &ComputeTaskState, metrics: &LoomMetrics) -> Self {
        state.count.fetch_add(1, Ordering::Relaxed);
        metrics.rayon_submitted();
        Self {
            state: state as *const ComputeTaskState,
            metrics: metrics as *const LoomMetrics,
        }
    }

    /// Mark that the rayon task has started executing.
    ///
    /// This should be called at the START of the rayon closure.
    fn started(&self) {
        // SAFETY: metrics outlives rayon tasks
        unsafe {
            (*self.metrics).rayon_started();
        }
    }
}

impl Drop for ComputeTaskGuard {
    fn drop(&mut self) {
        // SAFETY: state and metrics outlive rayon tasks due to shutdown waiting
        unsafe {
            // Track MAB metrics completion (panic-safe)
            (*self.metrics).rayon_completed();

            let prev = (*self.state).count.fetch_sub(1, Ordering::Release);
            if prev == 1 {
                // Count just went from 1 to 0, notify waiters
                (*self.state).notify.notify_waiters();
            }
        }
    }
}

/// A bespoke thread pool runtime combining tokio and rayon with CPU pinning.
///
/// The runtime provides:
/// - A tokio async runtime for I/O-bound work
/// - A rayon thread pool for CPU-bound parallel work
/// - Automatic CPU pinning for both runtimes
/// - A task tracker for graceful shutdown
/// - Zero-allocation compute spawning after warmup
///
/// # Performance Guarantees
///
/// | Method | Overhead | Allocations | Tracked |
/// |--------|----------|-------------|---------|
/// | `spawn_async()` | ~10ns | Token only | Yes |
/// | `spawn_compute()` | ~100-500ns | 0 bytes (after warmup) | Yes |
/// | `install()` | ~0ns | None | No |
/// | `rayon_pool()` | 0ns | None | No |
/// | `tokio_handle()` | 0ns | None | No |
///
/// # Examples
///
/// ```ignore
/// use loom_rs::LoomBuilder;
///
/// let runtime = LoomBuilder::new()
///     .prefix("myapp")
///     .tokio_threads(2)
///     .rayon_threads(6)
///     .build()?;
///
/// runtime.block_on(async {
///     // Spawn tracked async I/O task
///     let io_handle = runtime.spawn_async(async {
///         fetch_data().await
///     });
///
///     // Spawn tracked compute task and await result
///     let result = runtime.spawn_compute(|| {
///         expensive_computation()
///     }).await;
///
///     // Zero-overhead parallel iterators (within tracked context)
///     let processed = runtime.install(|| {
///         data.par_iter().map(|x| process(x)).collect()
///     });
/// });
///
/// // Graceful shutdown from main thread
/// runtime.block_until_idle();
/// ```
pub struct LoomRuntime {
    pub(crate) inner: Arc<LoomRuntimeInner>,
}

/// Inner state shared with thread-locals.
///
/// This is Arc-wrapped and shared with tokio/rayon worker threads via thread-local
/// storage, enabling `current_runtime()` to work from any managed thread.
pub(crate) struct LoomRuntimeInner {
    config: LoomConfig,
    tokio_runtime: tokio::runtime::Runtime,
    pub(crate) rayon_pool: rayon::ThreadPool,
    task_tracker: TaskTracker,
    /// Track in-flight rayon tasks for graceful shutdown
    compute_state: ComputeTaskState,
    /// Per-type object pools for zero-allocation spawn_compute
    pub(crate) pools: ComputePoolRegistry,
    /// Number of tokio worker threads
    pub(crate) tokio_threads: usize,
    /// Number of rayon worker threads
    pub(crate) rayon_threads: usize,
    /// CPUs allocated to tokio workers
    pub(crate) tokio_cpus: Vec<usize>,
    /// CPUs allocated to rayon workers
    pub(crate) rayon_cpus: Vec<usize>,
    /// Lazily initialized shared MAB scheduler
    mab_scheduler: OnceLock<Arc<MabScheduler>>,
    /// MAB knobs configuration
    pub(crate) mab_knobs: MabKnobs,
    /// Prometheus metrics - single source of truth for all metrics
    /// (serves both Prometheus exposition and MAB scheduling)
    pub(crate) prometheus_metrics: LoomMetrics,
}

impl LoomRuntime {
    /// Create a LoomRuntime from an existing inner reference.
    ///
    /// This does **not** create a new runtime; it only creates another
    /// handle that points at the same `LoomRuntimeInner`. As a result,
    /// multiple `LoomRuntime` values may refer to the same underlying
    /// runtime state.
    ///
    /// This is intended for internal use by `current_runtime()` to wrap the
    /// thread-local inner reference. Callers must **not** treat the returned
    /// handle as an independently owned runtime for the purpose of shutdown
    /// or teardown. Invoking shutdown-related methods from multiple wrappers
    /// that share the same inner state may lead to unexpected behavior.
    pub(crate) fn from_inner(inner: Arc<LoomRuntimeInner>) -> Self {
        Self { inner }
    }

    /// Create a runtime from a configuration.
    ///
    /// This is typically called via `LoomBuilder::build()`.
    pub(crate) fn from_config(config: LoomConfig) -> Result<Self> {
        let pool_size = config.compute_pool_size;
        // Determine available CPUs
        // Priority: CUDA device cpuset > user cpuset > all available CPUs
        // Error if both cuda_device and cpuset are specified (mutually exclusive)
        let cpus = {
            #[cfg(feature = "cuda")]
            {
                // Check for conflicting configuration first
                if config.cuda_device.is_some() && config.cpuset.is_some() {
                    return Err(LoomError::CudaCpusetConflict);
                }

                if let Some(ref selector) = config.cuda_device {
                    match crate::cuda::cpuset_for_cuda_device(selector)? {
                        Some(cuda_cpus) => cuda_cpus,
                        None => {
                            // Could not determine CUDA locality, fall back to all CPUs
                            available_cpus()
                        }
                    }
                } else if let Some(ref cpuset_str) = config.cpuset {
                    parse_and_validate_cpuset(cpuset_str)?
                } else {
                    available_cpus()
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                if let Some(ref cpuset_str) = config.cpuset {
                    parse_and_validate_cpuset(cpuset_str)?
                } else {
                    available_cpus()
                }
            }
        };

        if cpus.is_empty() {
            return Err(LoomError::NoCpusAvailable);
        }

        let total_cpus = cpus.len();
        let tokio_threads = config.effective_tokio_threads();
        let rayon_threads = config.effective_rayon_threads(total_cpus);

        // Validate we have enough CPUs
        let total_threads = tokio_threads + rayon_threads;
        if total_threads > total_cpus {
            return Err(LoomError::InsufficientCpus {
                requested: total_threads,
                available: total_cpus,
            });
        }

        // Split CPUs between tokio and rayon
        let (tokio_cpus, rayon_cpus) = cpus.split_at(tokio_threads.min(cpus.len()));
        let tokio_cpus = tokio_cpus.to_vec();
        let rayon_cpus = if rayon_cpus.is_empty() {
            // If we don't have dedicated rayon CPUs, share with tokio
            tokio_cpus.clone()
        } else {
            rayon_cpus.to_vec()
        };

        info!(
            prefix = %config.prefix,
            tokio_threads,
            rayon_threads,
            total_cpus,
            pool_size,
            "building loom runtime"
        );

        // Use Arc<str> for prefix to avoid cloning on each thread start
        let prefix: Arc<str> = config.prefix.as_str().into();

        // Create the inner runtime first (without tokio/rayon)
        // We'll use a two-phase approach with OnceCell-like pattern
        let inner = Arc::new_cyclic(|weak: &Weak<LoomRuntimeInner>| {
            let weak_clone = weak.clone();

            // Build tokio runtime with thread-local injection
            let tokio_runtime = Self::build_tokio_runtime(
                &prefix,
                tokio_threads,
                tokio_cpus.clone(),
                weak_clone.clone(),
            )
            .expect("failed to build tokio runtime");

            // Build rayon pool with thread-local injection
            let rayon_pool =
                Self::build_rayon_pool(&prefix, rayon_threads, rayon_cpus.clone(), weak_clone)
                    .expect("failed to build rayon pool");

            // Extract MAB knobs, using defaults if not configured
            let mab_knobs = config.mab_knobs.clone().unwrap_or_default();

            // Create Prometheus metrics with the runtime's prefix
            let prometheus_metrics = LoomMetrics::with_prefix(&config.prefix);

            // Register with provided registry if available
            if let Some(ref registry) = config.prometheus_registry {
                if let Err(e) = prometheus_metrics.register(registry) {
                    warn!(%e, "failed to register prometheus metrics");
                }
            }

            LoomRuntimeInner {
                config,
                tokio_runtime,
                rayon_pool,
                task_tracker: TaskTracker::new(),
                compute_state: ComputeTaskState::new(),
                pools: ComputePoolRegistry::new(pool_size),
                tokio_threads,
                rayon_threads,
                tokio_cpus,
                rayon_cpus,
                mab_scheduler: OnceLock::new(),
                mab_knobs,
                prometheus_metrics,
            }
        });

        Ok(Self { inner })
    }

    fn build_tokio_runtime(
        prefix: &Arc<str>,
        num_threads: usize,
        cpus: Vec<usize>,
        runtime_weak: Weak<LoomRuntimeInner>,
    ) -> Result<tokio::runtime::Runtime> {
        let allocator = Arc::new(CpuAllocator::new(cpus));
        let prefix_clone = Arc::clone(prefix);

        // Thread name counter
        let thread_counter = Arc::new(AtomicUsize::new(0));
        let name_prefix = Arc::clone(prefix);

        let start_weak = runtime_weak.clone();
        let start_allocator = allocator.clone();
        let start_prefix = prefix_clone.clone();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_threads)
            .thread_name_fn(move || {
                let id = thread_counter.fetch_add(1, Ordering::SeqCst);
                format!("{}-tokio-{:04}", name_prefix, id)
            })
            .on_thread_start(move || {
                // Pin CPU
                let cpu_id = start_allocator.allocate();
                if let Err(e) = pin_to_cpu(cpu_id) {
                    warn!(%e, %start_prefix, cpu_id, "failed to pin tokio thread");
                } else {
                    debug!(cpu_id, %start_prefix, "pinned tokio thread to CPU");
                }

                // Inject runtime reference into thread-local
                set_current_runtime(start_weak.clone());
            })
            .on_thread_stop(|| {
                clear_current_runtime();
            })
            .enable_all()
            .build()?;

        Ok(runtime)
    }

    fn build_rayon_pool(
        prefix: &Arc<str>,
        num_threads: usize,
        cpus: Vec<usize>,
        runtime_weak: Weak<LoomRuntimeInner>,
    ) -> Result<rayon::ThreadPool> {
        let allocator = Arc::new(CpuAllocator::new(cpus));
        let name_prefix = Arc::clone(prefix);

        let start_weak = runtime_weak.clone();
        let start_allocator = allocator.clone();
        let start_prefix = Arc::clone(prefix);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("{}-rayon-{:04}", name_prefix, i))
            .start_handler(move |thread_index| {
                // Pin CPU
                let cpu_id = start_allocator.allocate();
                debug!(thread_index, cpu_id, %start_prefix, "rayon thread starting");
                if let Err(e) = pin_to_cpu(cpu_id) {
                    warn!(%e, %start_prefix, cpu_id, thread_index, "failed to pin rayon thread");
                }

                // Inject runtime reference into thread-local
                set_current_runtime(start_weak.clone());
            })
            .exit_handler(|_thread_index| {
                clear_current_runtime();
            })
            .build()?;

        Ok(pool)
    }

    /// Get the resolved configuration.
    pub fn config(&self) -> &LoomConfig {
        &self.inner.config
    }

    /// Get the tokio runtime handle.
    ///
    /// This can be used to spawn untracked tasks or enter the runtime context.
    /// For tracked async tasks, prefer `spawn_async()`.
    ///
    /// # Performance
    ///
    /// Zero overhead - returns a reference.
    pub fn tokio_handle(&self) -> &tokio::runtime::Handle {
        self.inner.tokio_runtime.handle()
    }

    /// Get the rayon thread pool.
    ///
    /// This can be used to execute parallel iterators or spawn untracked work directly.
    /// For tracked compute tasks, prefer `spawn_compute()`.
    /// For zero-overhead parallel iterators, prefer `install()`.
    ///
    /// # Performance
    ///
    /// Zero overhead - returns a reference.
    pub fn rayon_pool(&self) -> &rayon::ThreadPool {
        &self.inner.rayon_pool
    }

    /// Get the task tracker for graceful shutdown.
    ///
    /// Use this to track spawned tasks and wait for them to complete.
    pub fn task_tracker(&self) -> &TaskTracker {
        &self.inner.task_tracker
    }

    /// Block on a future using the tokio runtime.
    ///
    /// This is the main entry point for running async code from the main thread.
    /// The current runtime is available via `loom_rs::current_runtime()` within
    /// the block_on scope.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     // Async code here
    ///     // loom_rs::current_runtime() works here
    /// });
    /// ```
    pub fn block_on<F: Future>(&self, f: F) -> F::Output {
        // Set current runtime for the main thread during block_on
        set_current_runtime(Arc::downgrade(&self.inner));
        let result = self.inner.tokio_runtime.block_on(f);
        clear_current_runtime();
        result
    }

    /// Spawn a tracked async task on tokio.
    ///
    /// The task is tracked for graceful shutdown via `block_until_idle()`.
    ///
    /// # Performance
    ///
    /// Overhead: ~10ns (TaskTracker token only).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     let handle = runtime.spawn_async(async {
    ///         // I/O-bound async work
    ///         fetch_data().await
    ///     });
    ///
    ///     let result = handle.await.unwrap();
    /// });
    /// ```
    #[inline]
    pub fn spawn_async<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        // Track task for MAB metrics (panic-safe via guard)
        let metrics_guard = AsyncMetricsGuard::new(Arc::clone(&self.inner));
        let token = self.inner.task_tracker.token();
        self.inner.tokio_runtime.spawn(async move {
            let _tracker = token;
            let _metrics = metrics_guard;
            future.await
        })
    }

    /// Spawn CPU-bound work on rayon and await the result.
    ///
    /// The task is tracked for graceful shutdown via `block_until_idle()`.
    /// Automatically uses per-type object pools for zero allocation after warmup.
    ///
    /// # Performance
    ///
    /// | State | Allocations | Overhead |
    /// |-------|-------------|----------|
    /// | Pool hit | 0 bytes | ~100-500ns |
    /// | Pool miss | ~32 bytes | ~100-500ns |
    /// | First call per type | Pool + state | ~1µs |
    ///
    /// For zero-overhead parallel iterators (within an already-tracked context),
    /// use `install()` instead.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     let result = runtime.spawn_compute(|| {
    ///         // CPU-intensive work
    ///         expensive_computation()
    ///     }).await;
    /// });
    /// ```
    #[inline]
    pub async fn spawn_compute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.inner.spawn_compute(f).await
    }

    /// Spawn work with adaptive inline/offload decision.
    ///
    /// Uses MAB (Multi-Armed Bandit) to learn whether this function type should
    /// run inline on tokio or offload to rayon. Good for handler patterns where
    /// work duration varies by input.
    ///
    /// Unlike `spawn_compute()` which always offloads, this adaptively chooses
    /// based on learned behavior and current system pressure.
    ///
    /// # Performance
    ///
    /// | Scenario | Behavior | Overhead |
    /// |----------|----------|----------|
    /// | Fast work | Inlines after learning | ~100ns (decision only) |
    /// | Slow work | Offloads after learning | ~100-500ns (+ offload) |
    /// | Cold start | Explores both arms | Variable |
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     // MAB will learn whether this is fast or slow
    ///     let result = runtime.spawn_adaptive(|| {
    ///         process_item(item)
    ///     }).await;
    /// });
    /// ```
    pub async fn spawn_adaptive<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.spawn_adaptive_with_hint(ComputeHint::Unknown, f).await
    }

    /// Spawn with hint for cold-start guidance.
    ///
    /// The hint helps the scheduler make better initial decisions before it has
    /// learned the actual execution time of this function type.
    ///
    /// # Hints
    ///
    /// - `ComputeHint::Low` - Expected < 50µs (likely inline-safe)
    /// - `ComputeHint::Medium` - Expected 50-500µs (borderline)
    /// - `ComputeHint::High` - Expected > 500µs (should test offload early)
    /// - `ComputeHint::Unknown` - No hint (default exploration)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use loom_rs::ComputeHint;
    ///
    /// runtime.block_on(async {
    ///     // Hint that this is likely slow work
    ///     let result = runtime.spawn_adaptive_with_hint(
    ///         ComputeHint::High,
    ///         || expensive_computation()
    ///     ).await;
    /// });
    /// ```
    pub async fn spawn_adaptive_with_hint<F, R>(&self, hint: ComputeHint, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let ctx = self.collect_context();
        let key = FunctionKey::from_type::<F>();
        let scheduler = self.mab_scheduler();

        let (id, arm) = scheduler.choose_with_hint(key, &ctx, hint);
        let start = Instant::now();

        let result = match arm {
            Arm::InlineTokio => f(),
            Arm::OffloadRayon => self.inner.spawn_compute(f).await,
        };

        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        scheduler.finish(id, elapsed_us, Some(elapsed_us));
        result
    }

    /// Execute work on rayon with zero overhead (sync, blocking).
    ///
    /// This installs the rayon pool for the current scope, allowing direct use
    /// of rayon's parallel iterators.
    ///
    /// **NOT tracked** - use within an already-tracked task (e.g., inside
    /// `spawn_async` or `spawn_compute`) for proper shutdown tracking.
    ///
    /// # Performance
    ///
    /// Zero overhead - direct rayon access.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     // This is a tracked context (we're in block_on)
    ///     let processed = runtime.install(|| {
    ///         use rayon::prelude::*;
    ///         data.par_iter().map(|x| process(x)).collect::<Vec<_>>()
    ///     });
    /// });
    /// ```
    #[inline]
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.inner.rayon_pool.install(f)
    }

    /// Execute a scoped parallel computation, allowing borrowed data.
    ///
    /// Unlike `spawn_compute()` which requires `'static` bounds, `scope_compute`
    /// allows borrowing local variables from the async context for use in parallel
    /// work. This is safe because:
    ///
    /// 1. The `.await` suspends the async task
    /// 2. `rayon::scope` blocks until ALL spawned work completes
    /// 3. Only then does the future resolve
    /// 4. Therefore, borrowed references remain valid throughout
    ///
    /// # Performance
    ///
    /// | Aspect | Value |
    /// |--------|-------|
    /// | Allocation | ~96 bytes per call (not pooled) |
    /// | Overhead | Comparable to `spawn_compute()` |
    ///
    /// State cannot be pooled because the result type R may contain borrowed
    /// references tied to the calling scope. Benchmarks show performance is
    /// within noise of `spawn_compute()` - the overhead is dominated by
    /// cross-thread communication, not state management.
    ///
    /// # Cancellation Safety
    ///
    /// If the future is dropped before completion (e.g., via `select!` or timeout),
    /// the drop will **block** until the rayon scope finishes. This is necessary
    /// to prevent use-after-free of borrowed data. In normal usage (awaiting to
    /// completion), there is no blocking overhead.
    ///
    /// # Panic Safety
    ///
    /// If the closure or any spawned work panics, the panic is captured and
    /// re-raised when the future is polled. This ensures panics propagate to
    /// the async context as expected.
    ///
    /// # Leaking the Future
    ///
    /// **Important:** Do not leak this future via `std::mem::forget` or similar.
    /// The safety of borrowed data relies on the future's `Drop` implementation
    /// blocking until the rayon scope completes. Leaking the future would allow
    /// the rayon work to continue accessing borrowed data after it goes out of
    /// scope, leading to undefined behavior. This is a known limitation shared
    /// by other scoped async APIs (e.g., `async-scoped`).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::sync::atomic::{AtomicI32, Ordering};
    ///
    /// runtime.block_on(async {
    ///     let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    ///     let sum = AtomicI32::new(0);
    ///
    ///     // Borrow `data` and `sum` for parallel processing
    ///     runtime.scope_compute(|s| {
    ///         let (left, right) = data.split_at(data.len() / 2);
    ///
    ///         s.spawn(|_| {
    ///             sum.fetch_add(left.iter().sum(), Ordering::Relaxed);
    ///         });
    ///         s.spawn(|_| {
    ///             sum.fetch_add(right.iter().sum(), Ordering::Relaxed);
    ///         });
    ///     }).await;
    ///
    ///     // `data` and `sum` are still valid here
    ///     println!("Sum of {:?} = {}", data, sum.load(Ordering::Relaxed));
    /// });
    /// ```
    pub async fn scope_compute<'env, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'env>) -> R + Send + 'env,
        R: Send + 'env,
    {
        self.inner.scope_compute(f).await
    }

    /// Execute scoped work with adaptive sync/async decision.
    ///
    /// Uses MAB (Multi-Armed Bandit) to learn whether this function type should:
    /// - Run synchronously via `install()` (blocks tokio worker, lower overhead)
    /// - Run asynchronously via `scope_compute()` (frees tokio worker, higher overhead)
    ///
    /// Unlike `spawn_adaptive()` which chooses between inline execution and rayon offload,
    /// `scope_adaptive` always uses `rayon::scope` (needed for parallel spawning with
    /// borrowed data), but chooses whether to block the tokio worker or use the async bridge.
    ///
    /// # Performance
    ///
    /// | Scenario | Behavior | Overhead |
    /// |----------|----------|----------|
    /// | Fast scoped work | Sync after learning | ~0ns (install overhead only) |
    /// | Slow scoped work | Async after learning | ~100-500ns (+ bridge) |
    /// | Cold start | Explores both arms | Variable |
    ///
    /// # When to Use
    ///
    /// Use `scope_adaptive` when:
    /// - You need to borrow local data (`'env` lifetime)
    /// - You want parallel spawning via `rayon::scope`
    /// - Work duration varies and you want the runtime to learn the best strategy
    ///
    /// Use `scope_compute` directly when:
    /// - Work is always slow (> 500µs)
    /// - You want consistent async behavior
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use std::sync::atomic::{AtomicI32, Ordering};
    ///
    /// runtime.block_on(async {
    ///     let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    ///     let sum = AtomicI32::new(0);
    ///
    ///     // MAB learns whether this is fast or slow scoped work
    ///     runtime.scope_adaptive(|s| {
    ///         let (left, right) = data.split_at(data.len() / 2);
    ///         let sum_ref = &sum;
    ///
    ///         s.spawn(move |_| {
    ///             sum_ref.fetch_add(left.iter().sum(), Ordering::Relaxed);
    ///         });
    ///         s.spawn(move |_| {
    ///             sum_ref.fetch_add(right.iter().sum(), Ordering::Relaxed);
    ///         });
    ///     }).await;
    ///
    ///     println!("Sum: {}", sum.load(Ordering::Relaxed));
    /// });
    /// ```
    pub async fn scope_adaptive<'env, F, R>(&self, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'env>) -> R + Send + 'env,
        R: Send + 'env,
    {
        self.scope_adaptive_with_hint(ComputeHint::Unknown, f).await
    }

    /// Execute scoped work with hint for cold-start guidance.
    ///
    /// The hint helps the scheduler make better initial decisions before it has
    /// learned the actual execution time of this function type.
    ///
    /// # Hints
    ///
    /// - `ComputeHint::Low` - Expected < 50µs (likely sync-safe)
    /// - `ComputeHint::Medium` - Expected 50-500µs (borderline)
    /// - `ComputeHint::High` - Expected > 500µs (should test async early)
    /// - `ComputeHint::Unknown` - No hint (default exploration)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use loom_rs::ComputeHint;
    /// use std::sync::atomic::{AtomicI32, Ordering};
    ///
    /// runtime.block_on(async {
    ///     let data = vec![1, 2, 3, 4];
    ///     let sum = AtomicI32::new(0);
    ///
    ///     // Hint that this is likely fast work
    ///     runtime.scope_adaptive_with_hint(ComputeHint::Low, |s| {
    ///         let sum_ref = &sum;
    ///         for &val in &data {
    ///             s.spawn(move |_| {
    ///                 sum_ref.fetch_add(val, Ordering::Relaxed);
    ///             });
    ///         }
    ///     }).await;
    /// });
    /// ```
    pub async fn scope_adaptive_with_hint<'env, F, R>(&self, hint: ComputeHint, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'env>) -> R + Send + 'env,
        R: Send + 'env,
    {
        let ctx = self.collect_context();
        // Use from_type_name since F may capture non-'static references
        let key = FunctionKey::from_type_name::<F>();
        let scheduler = self.mab_scheduler();

        let (id, arm) = scheduler.choose_with_hint(key, &ctx, hint);
        let start = Instant::now();

        let result = match arm {
            Arm::InlineTokio => {
                // Sync: blocks tokio but no async bridge overhead
                self.install(|| rayon::scope(|s| f(s)))
            }
            Arm::OffloadRayon => {
                // Async: frees tokio worker during execution
                self.scope_compute(f).await
            }
        };

        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        scheduler.finish(id, elapsed_us, Some(elapsed_us));
        result
    }

    /// Stop accepting new tasks.
    ///
    /// After calling this, `spawn_async()` and `spawn_compute()` will still
    /// work, but the shutdown process has begun. Use `is_idle()` or
    /// `wait_for_shutdown()` to check/wait for completion.
    pub fn shutdown(&self) {
        self.inner.task_tracker.close();
    }

    /// Check if all tracked tasks have completed.
    ///
    /// Returns `true` if `shutdown()` has been called and all tracked async
    /// tasks and compute tasks have finished.
    ///
    /// # Performance
    ///
    /// Zero overhead - single atomic load.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.inner.task_tracker.is_closed()
            && self.inner.task_tracker.is_empty()
            && self.inner.compute_state.count.load(Ordering::Acquire) == 0
    }

    /// Get the number of compute tasks currently in flight.
    ///
    /// Useful for debugging shutdown issues or monitoring workload.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if runtime.compute_tasks_in_flight() > 0 {
    ///     tracing::warn!("Still waiting for {} compute tasks",
    ///         runtime.compute_tasks_in_flight());
    /// }
    /// ```
    #[inline]
    pub fn compute_tasks_in_flight(&self) -> usize {
        self.inner.compute_state.count.load(Ordering::Relaxed)
    }

    /// Wait for all tracked tasks to complete (async).
    ///
    /// Call from within `block_on()`. Requires `shutdown()` to be called first,
    /// otherwise this will wait forever.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     runtime.spawn_async(work());
    ///     runtime.shutdown();
    ///     runtime.wait_for_shutdown().await;
    /// });
    /// ```
    pub async fn wait_for_shutdown(&self) {
        self.inner.task_tracker.wait().await;

        // Wait for compute tasks efficiently (no spin loop)
        let mut logged = false;
        loop {
            let count = self.inner.compute_state.count.load(Ordering::Acquire);
            if count == 0 {
                break;
            }
            if !logged {
                debug!(count, "waiting for compute tasks to complete");
                logged = true;
            }
            self.inner.compute_state.notify.notified().await;
        }
    }

    /// Block until all tracked tasks complete (from main thread).
    ///
    /// This is the primary shutdown method. It:
    /// 1. Calls `shutdown()` to close the task tracker
    /// 2. Waits for all tracked async and compute tasks to finish
    ///
    /// # Examples
    ///
    /// ```ignore
    /// runtime.block_on(async {
    ///     runtime.spawn_async(background_work());
    ///     runtime.spawn_compute(|| cpu_work());
    /// });
    ///
    /// // Graceful shutdown from main thread
    /// runtime.block_until_idle();
    /// ```
    pub fn block_until_idle(&self) {
        self.shutdown();
        self.block_on(self.wait_for_shutdown());
    }

    /// Get the shared MAB scheduler for handler patterns.
    ///
    /// The scheduler is lazily initialized on first call. Use this when you
    /// need to make manual scheduling decisions in handler code.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::mab::{FunctionKey, Arm};
    ///
    /// let sched = runtime.mab_scheduler();
    /// let key = FunctionKey::from_type::<MyHandler>();
    /// let ctx = runtime.collect_context();
    ///
    /// let (id, arm) = sched.choose(key, &ctx);
    /// let result = match arm {
    ///     Arm::InlineTokio => my_work(),
    ///     Arm::OffloadRayon => runtime.block_on(async {
    ///         runtime.spawn_compute(|| my_work()).await
    ///     }),
    /// };
    /// sched.finish(id, elapsed_us, Some(fn_us));
    /// ```
    pub fn mab_scheduler(&self) -> Arc<MabScheduler> {
        self.inner
            .mab_scheduler
            .get_or_init(|| {
                Arc::new(MabScheduler::with_metrics(
                    self.inner.mab_knobs.clone(),
                    self.inner.prometheus_metrics.clone(),
                ))
            })
            .clone()
    }

    /// Collect current runtime context for MAB scheduling decisions.
    ///
    /// Returns a snapshot of current metrics including inflight tasks,
    /// spawn rate, and queue depth.
    pub fn collect_context(&self) -> Context {
        self.inner.prometheus_metrics.collect_context(
            self.inner.tokio_threads as u32,
            self.inner.rayon_threads as u32,
        )
    }

    /// Get the number of tokio worker threads.
    pub fn tokio_threads(&self) -> usize {
        self.inner.tokio_threads
    }

    /// Get the number of rayon threads.
    pub fn rayon_threads(&self) -> usize {
        self.inner.rayon_threads
    }

    /// Get the Prometheus metrics.
    ///
    /// The metrics are always collected (zero overhead atomic operations).
    /// If a Prometheus registry was provided via `LoomBuilder::prometheus_registry()`,
    /// the metrics are also registered for exposition.
    pub fn prometheus_metrics(&self) -> &LoomMetrics {
        &self.inner.prometheus_metrics
    }

    /// Get the CPUs allocated to tokio workers.
    pub fn tokio_cpus(&self) -> &[usize] {
        &self.inner.tokio_cpus
    }

    /// Get the CPUs allocated to rayon workers.
    pub fn rayon_cpus(&self) -> &[usize] {
        &self.inner.rayon_cpus
    }
}

impl LoomRuntimeInner {
    /// Spawn CPU-bound work on rayon and await the result.
    ///
    /// Uses per-type object pools for zero allocation after warmup.
    #[inline]
    pub async fn spawn_compute<F, R>(self: &Arc<Self>, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let pool = self.pools.get_or_create::<R>();

        // Try to get state from pool, or allocate new
        let state = pool.pop().unwrap_or_else(|| Arc::new(TaskState::new()));

        // Create the pooled task
        let (task, completion, state_for_return) = PooledRayonTask::new(state);

        // Create guard BEFORE spawning - it increments counter and tracks MAB metrics
        let guard = ComputeTaskGuard::new(&self.compute_state, &self.prometheus_metrics);

        self.rayon_pool.spawn(move || {
            // Track rayon task start for queue depth calculation
            guard.started();

            // Execute work inside guard scope so counter decrements BEFORE completing.
            // This ensures the async future sees count=0 when it wakes up.
            let result = {
                let _guard = guard;
                f()
            };
            completion.complete(result);
        });

        let result = task.await;

        // Return state to pool for reuse
        state_for_return.reset();
        pool.push(state_for_return);

        result
    }

    /// Execute a scoped parallel computation with borrowed data.
    ///
    /// # Safety Argument
    ///
    /// The lifetime erasure via transmute is sound because:
    /// 1. The async task is suspended at `.await`
    /// 2. The future only completes after `rayon::scope` returns
    /// 3. `rayon::scope` blocks until ALL spawned work completes
    /// 4. Therefore, `'env` references outlive all accesses to borrowed data
    ///
    /// This is the same safety argument used by `std::thread::scope`.
    pub async fn scope_compute<'env, F, R>(self: &Arc<Self>, f: F) -> R
    where
        F: FnOnce(&rayon::Scope<'env>) -> R + Send + 'env,
        R: Send + 'env,
    {
        // SAFETY: Lifetime erasure is sound because:
        // 1. The future holds the async task suspended at .await
        // 2. Future only completes after rayon::scope returns
        // 3. rayon::scope blocks until all spawned work completes
        // 4. Therefore 'env outlives all accesses to borrowed data
        //
        // Additionally, if the future is dropped before completion (cancellation),
        // ScopedComputeFuture::drop blocks until the scope finishes, maintaining
        // the safety invariant.

        let state = Arc::new(ScopedTaskState::<R>::new());
        let future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state);

        // Create guard BEFORE spawning - it increments counter and tracks MAB metrics
        let guard = ComputeTaskGuard::new(&self.compute_state, &self.prometheus_metrics);

        // Create a closure that captures f, completion, and guard.
        // We use catch_unwind to ensure completion is always signaled, even on panic.
        let work = move || {
            guard.started();
            let result = {
                let _guard = guard;
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rayon::scope(|s| f(s))))
            };
            match result {
                Ok(value) => completion.complete(value),
                Err(payload) => completion.complete_with_panic(payload),
            }
        };

        // SAFETY: We erase the lifetime to 'static here. This is safe because:
        // 1. The ScopedComputeFuture we return will block in Drop if the work
        //    hasn't completed (cancellation safety)
        // 2. Normal await returns only after the work completes
        // 3. Therefore, all 'env references remain valid for the duration of
        //    the work execution
        let erased: Box<dyn FnOnce() + Send + 'static> = unsafe {
            std::mem::transmute::<Box<dyn FnOnce() + Send + 'env>, Box<dyn FnOnce() + Send + 'static>>(
                Box::new(work),
            )
        };

        self.rayon_pool.spawn(move || {
            erased();
        });

        future.await
    }
}

impl std::fmt::Debug for LoomRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoomRuntime")
            .field("config", &self.inner.config)
            .field(
                "compute_tasks_in_flight",
                &self.inner.compute_state.count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for LoomRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoomRuntime[{}]: tokio({}, cpus={}) rayon({}, cpus={})",
            self.inner.config.prefix,
            self.inner.tokio_threads,
            format_cpuset(&self.inner.tokio_cpus),
            self.inner.rayon_threads,
            format_cpuset(&self.inner.rayon_cpus),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::DEFAULT_POOL_SIZE;

    fn test_config() -> LoomConfig {
        LoomConfig {
            prefix: "test".to_string(),
            cpuset: None,
            tokio_threads: Some(1),
            rayon_threads: Some(1),
            compute_pool_size: DEFAULT_POOL_SIZE,
            #[cfg(feature = "cuda")]
            cuda_device: None,
            mab_knobs: None,
            calibration: None,
            prometheus_registry: None,
        }
    }

    #[test]
    fn test_runtime_creation() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();
        assert_eq!(runtime.config().prefix, "test");
    }

    #[test]
    fn test_block_on() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_spawn_compute() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result =
            runtime.block_on(async { runtime.spawn_compute(|| (0..100).sum::<i32>()).await });
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_spawn_async() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let handle = runtime.spawn_async(async { 42 });
            handle.await.unwrap()
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_install() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.install(|| {
            use rayon::prelude::*;
            (0..100).into_par_iter().sum::<i32>()
        });
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_shutdown_and_idle() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        // Initially not idle (tracker not closed)
        assert!(!runtime.is_idle());

        // After shutdown with no tasks, should be idle
        runtime.shutdown();
        assert!(runtime.is_idle());
    }

    #[test]
    fn test_block_until_idle() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            runtime.spawn_async(async { 42 });
            runtime.spawn_compute(|| 100).await;
        });

        runtime.block_until_idle();
        assert!(runtime.is_idle());
    }

    #[test]
    fn test_insufficient_cpus_error() {
        let mut config = test_config();
        config.cpuset = Some("0".to_string()); // Only 1 CPU
        config.tokio_threads = Some(2);
        config.rayon_threads = Some(2);

        let result = LoomRuntime::from_config(config);
        assert!(matches!(result, Err(LoomError::InsufficientCpus { .. })));
    }

    #[test]
    fn test_current_runtime_in_block_on() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            // current_runtime should work inside block_on
            let current = crate::context::current_runtime();
            assert!(current.is_some());
        });

        // Outside block_on, should be None
        assert!(crate::context::current_runtime().is_none());
    }

    #[test]
    fn test_spawn_compute_pooling() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        // Warmup - first call allocates
        runtime.block_on(async {
            runtime.spawn_compute(|| 1i32).await;
        });

        // Subsequent calls should reuse pooled state (we can't easily verify this
        // without internal access, but we can verify it works)
        runtime.block_on(async {
            for i in 0..100 {
                let result = runtime.spawn_compute(move || i).await;
                assert_eq!(result, i);
            }
        });
    }

    #[test]
    fn test_spawn_compute_guard_drops_on_scope_exit() {
        // This test verifies the guard's Drop implementation works correctly.
        // We can't easily test panic behavior in rayon (panics abort by default),
        // but we can verify the guard decrements the counter when it goes out of scope.
        use crate::metrics::LoomMetrics;
        use std::sync::atomic::Ordering;

        let state = super::ComputeTaskState::new();
        let metrics = LoomMetrics::new();

        // Create a guard (increments counter)
        {
            let _guard = super::ComputeTaskGuard::new(&state, &metrics);
            assert_eq!(state.count.load(Ordering::Relaxed), 1);
        }
        // Guard dropped, counter should be 0
        assert_eq!(state.count.load(Ordering::Relaxed), 0);

        // Test multiple guards
        let state = super::ComputeTaskState::new();

        let guard1 = super::ComputeTaskGuard::new(&state, &metrics);
        assert_eq!(state.count.load(Ordering::Relaxed), 1);

        let guard2 = super::ComputeTaskGuard::new(&state, &metrics);
        assert_eq!(state.count.load(Ordering::Relaxed), 2);

        drop(guard1);
        assert_eq!(state.count.load(Ordering::Relaxed), 1);

        drop(guard2);
        assert_eq!(state.count.load(Ordering::Relaxed), 0);

        // The notification mechanism is verified by the fact that wait_for_shutdown
        // doesn't spin-loop forever when compute tasks complete
    }

    #[test]
    fn test_compute_tasks_in_flight() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        // Initially no tasks
        assert_eq!(runtime.compute_tasks_in_flight(), 0);

        // After spawning and completing, should be back to 0
        runtime.block_on(async {
            runtime.spawn_compute(|| 42).await;
        });
        assert_eq!(runtime.compute_tasks_in_flight(), 0);
    }

    #[test]
    fn test_display() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let display = format!("{}", runtime);
        assert!(display.starts_with("LoomRuntime[test]:"));
        assert!(display.contains("tokio(1, cpus="));
        assert!(display.contains("rayon(1, cpus="));
    }

    #[test]
    fn test_cpuset_only() {
        let mut config = test_config();
        config.cpuset = Some("0".to_string());
        config.tokio_threads = Some(1);
        config.rayon_threads = Some(0);

        let runtime = LoomRuntime::from_config(config).unwrap();
        // Should use the user-provided cpuset
        assert_eq!(runtime.inner.tokio_cpus, vec![0]);
    }

    /// Test that CUDA cpuset conflict error is properly detected.
    /// This test requires actual CUDA hardware to verify the conflict.
    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_cpuset_conflict_error() {
        let mut config = test_config();
        config.cuda_device = Some(crate::cuda::CudaDeviceSelector::DeviceId(0));
        config.cpuset = Some("0".to_string()); // Conflict: both specified

        let result = LoomRuntime::from_config(config);
        assert!(
            matches!(result, Err(LoomError::CudaCpusetConflict)),
            "expected CudaCpusetConflict error, got {:?}",
            result
        );
    }

    /// Test that CUDA device alone (without cpuset) works.
    #[cfg(feature = "cuda-tests")]
    #[test]
    fn test_cuda_device_only() {
        let mut config = test_config();
        config.cuda_device = Some(crate::cuda::CudaDeviceSelector::DeviceId(0));
        config.cpuset = None;

        let runtime = LoomRuntime::from_config(config).unwrap();
        // Should have found CUDA-local CPUs
        assert!(!runtime.inner.tokio_cpus.is_empty());
    }

    // =============================================================================
    // spawn_adaptive Tests
    // =============================================================================

    #[test]
    fn test_spawn_adaptive_runs_work() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async { runtime.spawn_adaptive(|| 42).await });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_spawn_adaptive_with_hint() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            runtime
                .spawn_adaptive_with_hint(crate::ComputeHint::Low, || 100)
                .await
        });

        assert_eq!(result, 100);
    }

    #[test]
    fn test_spawn_adaptive_multiple_calls() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            // Run many fast tasks to let MAB learn
            for i in 0..50 {
                let result = runtime.spawn_adaptive(move || i * 2).await;
                assert_eq!(result, i * 2);
            }
        });
    }

    #[test]
    fn test_spawn_adaptive_records_metrics() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            // Run some adaptive tasks
            for _ in 0..10 {
                runtime.spawn_adaptive(|| std::hint::black_box(42)).await;
            }
        });

        // Check that metrics were recorded
        let metrics = runtime.prometheus_metrics();
        let total_decisions = metrics.inline_decisions.get() + metrics.offload_decisions.get();
        assert!(
            total_decisions >= 10,
            "Should have recorded at least 10 decisions, got {}",
            total_decisions
        );
    }

    #[test]
    fn test_prometheus_metrics_use_prefix() {
        let mut config = test_config();
        config.prefix = "myapp".to_string();
        let runtime = LoomRuntime::from_config(config).unwrap();

        // The metrics should use the prefix from config
        // We can verify by checking the registry if one was provided
        let registry = prometheus::Registry::new();
        runtime
            .prometheus_metrics()
            .register(&registry)
            .expect("registration should succeed");

        let families = registry.gather();
        // Find a metric with our prefix
        let myapp_metric = families.iter().find(|f| f.get_name().starts_with("myapp_"));
        assert!(
            myapp_metric.is_some(),
            "Should find metrics with 'myapp_' prefix"
        );

        // Should not find metrics with default 'loom_' prefix
        let loom_metric = families.iter().find(|f| f.get_name().starts_with("loom_"));
        assert!(
            loom_metric.is_none(),
            "Should not find metrics with 'loom_' prefix"
        );
    }

    // =============================================================================
    // scope_compute Tests
    // =============================================================================

    #[test]
    fn test_scope_compute_basic() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            runtime
                .scope_compute(|_s| {
                    // Simple computation without spawning
                    42
                })
                .await
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_scope_compute_borrow_local_data() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];

            let sum = runtime
                .scope_compute(|_s| {
                    // Borrow data inside the scope
                    data.iter().sum::<i32>()
                })
                .await;

            // data is still valid after scope_compute
            assert_eq!(data.len(), 8);
            sum
        });

        assert_eq!(result, 36);
    }

    #[test]
    fn test_scope_compute_parallel_with_atomic() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let mut config = test_config();
        config.rayon_threads = Some(2);
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];
            let sum = AtomicI32::new(0);

            runtime
                .scope_compute(|s| {
                    let (left, right) = data.split_at(data.len() / 2);
                    let sum_ref = &sum;

                    s.spawn(move |_| {
                        let partial: i32 = left.iter().sum();
                        sum_ref.fetch_add(partial, Ordering::Relaxed);
                    });
                    s.spawn(move |_| {
                        let partial: i32 = right.iter().sum();
                        sum_ref.fetch_add(partial, Ordering::Relaxed);
                    });
                })
                .await;

            sum.load(Ordering::Relaxed)
        });

        assert_eq!(result, 36);
    }

    #[test]
    fn test_scope_compute_nested_spawns() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];
            let sum = AtomicI32::new(0);

            runtime
                .scope_compute(|s| {
                    let data_ref = &data;
                    let sum_ref = &sum;

                    s.spawn(move |s| {
                        // Nested spawn
                        s.spawn(move |_| {
                            sum_ref
                                .fetch_add(data_ref[0..2].iter().sum::<i32>(), Ordering::Relaxed);
                        });
                        s.spawn(move |_| {
                            sum_ref
                                .fetch_add(data_ref[2..4].iter().sum::<i32>(), Ordering::Relaxed);
                        });
                    });
                    s.spawn(move |_| {
                        sum_ref.fetch_add(data_ref[4..8].iter().sum::<i32>(), Ordering::Relaxed);
                    });
                })
                .await;

            // rayon::scope guarantees all spawned work completes before returning
            sum.load(Ordering::Relaxed)
        });

        assert_eq!(result, 36);
    }

    #[test]
    fn test_scope_compute_with_rayon_par_iter() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

            runtime
                .scope_compute(|_s| {
                    use rayon::prelude::*;
                    // Use parallel iterators inside the scope
                    data.par_iter().map(|x| x * 2).sum::<i32>()
                })
                .await
        });

        assert_eq!(result, 110);
    }

    #[test]
    fn test_scope_compute_tracks_compute_tasks() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        // Initially no tasks
        assert_eq!(runtime.compute_tasks_in_flight(), 0);

        runtime.block_on(async {
            runtime.scope_compute(|_s| 42).await;
        });

        // After completion, should be back to 0
        assert_eq!(runtime.compute_tasks_in_flight(), 0);
    }

    #[test]
    fn test_scope_compute_data_still_valid_after() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            let mut data = vec![1, 2, 3, 4, 5];

            let sum = runtime.scope_compute(|_s| data.iter().sum::<i32>()).await;

            assert_eq!(sum, 15);

            // data is still valid and can be modified
            data.push(6);
            assert_eq!(data.len(), 6);

            // Can use scope_compute again with the same data
            let new_sum = runtime.scope_compute(|_s| data.iter().sum::<i32>()).await;
            assert_eq!(new_sum, 21);
        });
    }

    /// Test that scope_compute properly yields to the async executor and doesn't block.
    /// This validates the future correctly returns Poll::Pending and wakes up when done.
    #[test]
    fn test_scope_compute_yields_to_executor() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            let concurrent_task_ran = Arc::new(AtomicBool::new(false));
            let concurrent_task_ran_clone = concurrent_task_ran.clone();

            // Spawn a concurrent async task that should run while scope_compute is waiting
            let concurrent_handle = runtime.spawn_async(async move {
                // Wait a bit for the scope_compute to start
                tokio::time::sleep(Duration::from_millis(5)).await;
                concurrent_task_ran_clone.store(true, Ordering::Release);
                100
            });

            // Run scope_compute with work that takes some time
            let result = runtime
                .scope_compute(|_s| {
                    // Simulate some work
                    std::thread::sleep(Duration::from_millis(20));
                    42
                })
                .await;

            assert_eq!(result, 42);

            // Verify the concurrent task actually ran during scope_compute
            // (check BEFORE awaiting the handle to prove it ran concurrently)
            assert!(
                concurrent_task_ran.load(Ordering::Acquire),
                "Concurrent task should have run while scope_compute was in progress"
            );

            // The concurrent task should have completed
            let concurrent_result = concurrent_handle.await.unwrap();
            assert_eq!(concurrent_result, 100);
        });
    }

    /// Test that mirrors the documentation example for scope_compute.
    /// This ensures the example code actually compiles and works.
    #[test]
    fn test_scope_compute_doc_example() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];
            let sum = AtomicI32::new(0);

            runtime
                .scope_compute(|s| {
                    let (left, right) = data.split_at(data.len() / 2);
                    let sum_ref = &sum;

                    s.spawn(move |_| {
                        sum_ref.fetch_add(left.iter().sum::<i32>(), Ordering::Relaxed);
                    });
                    s.spawn(move |_| {
                        sum_ref.fetch_add(right.iter().sum::<i32>(), Ordering::Relaxed);
                    });
                })
                .await;

            // Verify data is still accessible after scope_compute
            assert_eq!(data.len(), 8);
            sum.load(Ordering::Relaxed)
        });

        assert_eq!(result, 36); // 1+2+3+4+5+6+7+8 = 36
    }

    /// Test that scope_compute works correctly with tokio::select! (cancellation scenario).
    /// The future should block on drop until the scope completes.
    #[test]
    fn test_scope_compute_cancellation_safety() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            let scope_completed = Arc::new(AtomicBool::new(false));
            let scope_completed_clone = scope_completed.clone();

            // Use select! to race scope_compute against a timeout
            // The scope_compute will "lose" but should still complete before we continue
            let result = tokio::select! {
                biased;

                _ = tokio::time::sleep(Duration::from_millis(5)) => {
                    // Timeout wins - this drops the scope_compute future
                    // But drop should block until scope completes
                    None
                }
                result = runtime.scope_compute(|_s| {
                    // This takes longer than the timeout
                    std::thread::sleep(Duration::from_millis(50));
                    scope_completed_clone.store(true, Ordering::Release);
                    42
                }) => {
                    Some(result)
                }
            };

            // The timeout should have won
            assert!(result.is_none(), "Timeout should have won the race");

            // But critically, the scope should have completed (drop blocked)
            assert!(
                scope_completed.load(Ordering::Acquire),
                "Scope should have completed even though future was cancelled"
            );
        });
    }

    /// Test that panics inside scope_compute are properly propagated to the awaiter.
    #[test]
    #[should_panic(expected = "intentional panic for testing")]
    fn test_scope_compute_panic_propagation() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            runtime
                .scope_compute(|_s| {
                    panic!("intentional panic for testing");
                })
                .await
        });
    }

    /// Test that panics in spawned work inside scope_compute are properly propagated.
    #[test]
    #[should_panic(expected = "panic in spawned work")]
    fn test_scope_compute_spawned_panic_propagation() {
        let mut config = test_config();
        config.rayon_threads = Some(2);
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            runtime
                .scope_compute(|s| {
                    s.spawn(|_| {
                        panic!("panic in spawned work");
                    });
                })
                .await
        });
    }

    // =============================================================================
    // scope_adaptive Tests
    // =============================================================================

    #[test]
    fn test_scope_adaptive_basic() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            runtime
                .scope_adaptive(|_s| {
                    // Simple computation without spawning
                    42
                })
                .await
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_scope_adaptive_with_hint() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            runtime
                .scope_adaptive_with_hint(crate::ComputeHint::Low, |_s| 100)
                .await
        });

        assert_eq!(result, 100);
    }

    #[test]
    fn test_scope_adaptive_borrow_local_data() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];

            let sum = runtime
                .scope_adaptive(|_s| {
                    // Borrow data inside the scope
                    data.iter().sum::<i32>()
                })
                .await;

            // data is still valid after scope_adaptive
            assert_eq!(data.len(), 8);
            sum
        });

        assert_eq!(result, 36);
    }

    #[test]
    fn test_scope_adaptive_parallel_with_atomic() {
        use std::sync::atomic::{AtomicI32, Ordering};

        let mut config = test_config();
        config.rayon_threads = Some(2);
        let runtime = LoomRuntime::from_config(config).unwrap();

        let result = runtime.block_on(async {
            let data = [1, 2, 3, 4, 5, 6, 7, 8];
            let sum = AtomicI32::new(0);

            runtime
                .scope_adaptive(|s| {
                    let (left, right) = data.split_at(data.len() / 2);
                    let sum_ref = &sum;

                    s.spawn(move |_| {
                        let partial: i32 = left.iter().sum();
                        sum_ref.fetch_add(partial, Ordering::Relaxed);
                    });
                    s.spawn(move |_| {
                        let partial: i32 = right.iter().sum();
                        sum_ref.fetch_add(partial, Ordering::Relaxed);
                    });
                })
                .await;

            sum.load(Ordering::Relaxed)
        });

        assert_eq!(result, 36);
    }

    #[test]
    fn test_scope_adaptive_records_metrics() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            // Run some adaptive tasks
            for _ in 0..10 {
                runtime.scope_adaptive(|_s| std::hint::black_box(42)).await;
            }
        });

        // Check that metrics were recorded
        let metrics = runtime.prometheus_metrics();
        let total_decisions = metrics.inline_decisions.get() + metrics.offload_decisions.get();
        assert!(
            total_decisions >= 10,
            "Should have recorded at least 10 decisions, got {}",
            total_decisions
        );
    }

    /// Test that panics inside scope_adaptive are properly propagated.
    #[test]
    #[should_panic(expected = "intentional panic in scope_adaptive")]
    fn test_scope_adaptive_panic_propagation() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config).unwrap();

        runtime.block_on(async {
            runtime
                .scope_adaptive(|_s| {
                    panic!("intentional panic in scope_adaptive");
                })
                .await
        });
    }
}
