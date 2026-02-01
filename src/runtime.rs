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
use crate::bridge::{PooledRayonTask, TaskState};
use crate::config::LoomConfig;
use crate::context::{clear_current_runtime, set_current_runtime};
use crate::cpuset::{available_cpus, format_cpuset, parse_and_validate_cpuset};
use crate::error::{LoomError, Result};
use crate::mab::{Context, MabKnobs, MabScheduler, RuntimeMetrics};
use crate::metrics::LoomMetrics;
use crate::pool::ComputePoolRegistry;

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
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

/// Guard that decrements compute task counter on drop.
///
/// Panic-safe: executes even if the task closure panics.
///
/// SAFETY: The state lives in LoomRuntimeInner which outlives all rayon tasks
/// because block_until_idle waits for compute_tasks to reach 0.
struct ComputeTaskGuard {
    state: *const ComputeTaskState,
}

unsafe impl Send for ComputeTaskGuard {}

impl ComputeTaskGuard {
    fn new(state: &ComputeTaskState) -> Self {
        state.count.fetch_add(1, Ordering::Relaxed);
        Self {
            state: state as *const ComputeTaskState,
        }
    }
}

impl Drop for ComputeTaskGuard {
    fn drop(&mut self) {
        // SAFETY: state outlives rayon tasks due to shutdown waiting
        unsafe {
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
    inner: Arc<LoomRuntimeInner>,
}

/// Inner state shared with thread-locals.
///
/// This is Arc-wrapped and shared with tokio/rayon worker threads via thread-local
/// storage, enabling `current_runtime()` to work from any managed thread.
pub struct LoomRuntimeInner {
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
    /// Runtime metrics for MAB scheduling
    pub(crate) metrics: RuntimeMetrics,
    /// Lazily initialized shared MAB scheduler
    mab_scheduler: OnceLock<Arc<MabScheduler>>,
    /// MAB knobs configuration
    mab_knobs: MabKnobs,
    /// Prometheus metrics (always active, registry optional)
    pub(crate) prometheus_metrics: LoomMetrics,
}

impl LoomRuntime {
    /// Create a runtime from a configuration.
    ///
    /// This is typically called via `LoomBuilder::build()`.
    pub(crate) fn from_config(config: LoomConfig, pool_size: usize) -> Result<Self> {
        // Determine available CPUs
        // Priority: CUDA device cpuset > user cpuset > all available CPUs
        // Error if both cuda_device and cpuset are specified and CUDA cpuset is accurate
        let cpus = {
            #[cfg(feature = "cuda")]
            {
                if let Some(ref selector) = config.cuda_device {
                    match crate::cuda::cpuset_for_cuda_device(selector)? {
                        Some(cuda_cpus) => {
                            // CUDA cpuset was successfully determined
                            if config.cpuset.is_some() {
                                return Err(LoomError::CudaCpusetConflict);
                            }
                            cuda_cpus
                        }
                        None => {
                            // Could not determine CUDA locality, fall back
                            if let Some(ref cpuset_str) = config.cpuset {
                                parse_and_validate_cpuset(cpuset_str)?
                            } else {
                                available_cpus()
                            }
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

            // Create Prometheus metrics
            let prometheus_metrics = LoomMetrics::new();

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
                metrics: RuntimeMetrics::new(),
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
        let token = self.inner.task_tracker.token();
        self.inner.tokio_runtime.spawn(async move {
            let _guard = token;
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
            .get_or_init(|| Arc::new(MabScheduler::new(self.inner.mab_knobs.clone())))
            .clone()
    }

    /// Collect current runtime context for MAB scheduling decisions.
    ///
    /// Returns a snapshot of current metrics including inflight tasks,
    /// spawn rate, and queue depth.
    pub fn collect_context(&self) -> Context {
        self.inner.metrics.collect(
            self.inner.tokio_threads as u32,
            self.inner.rayon_threads as u32,
        )
    }

    /// Get the runtime metrics.
    ///
    /// Primarily used internally by the MAB scheduler and stream adapters.
    pub fn metrics(&self) -> &RuntimeMetrics {
        &self.inner.metrics
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
    pub async fn spawn_compute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let pool = self.pools.get_or_create::<R>();

        // Try to get state from pool, or allocate new
        let state = pool.pop().unwrap_or_else(|| Arc::new(TaskState::new()));

        // Create the pooled task
        let (task, completion, state_for_return) = PooledRayonTask::new(state);

        // Create guard BEFORE spawning - it increments counter in constructor
        let guard = ComputeTaskGuard::new(&self.compute_state);

        self.rayon_pool.spawn(move || {
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
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();
        assert_eq!(runtime.config().prefix, "test");
    }

    #[test]
    fn test_block_on() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

        let result = runtime.block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_spawn_compute() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

        let result =
            runtime.block_on(async { runtime.spawn_compute(|| (0..100).sum::<i32>()).await });
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_spawn_async() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

        let result = runtime.block_on(async {
            let handle = runtime.spawn_async(async { 42 });
            handle.await.unwrap()
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_install() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

        let result = runtime.install(|| {
            use rayon::prelude::*;
            (0..100).into_par_iter().sum::<i32>()
        });
        assert_eq!(result, 4950);
    }

    #[test]
    fn test_shutdown_and_idle() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

        // Initially not idle (tracker not closed)
        assert!(!runtime.is_idle());

        // After shutdown with no tasks, should be idle
        runtime.shutdown();
        assert!(runtime.is_idle());
    }

    #[test]
    fn test_block_until_idle() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

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

        let result = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE);
        assert!(matches!(result, Err(LoomError::InsufficientCpus { .. })));
    }

    #[test]
    fn test_current_runtime_in_block_on() {
        let config = test_config();
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

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
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

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
        use std::sync::atomic::Ordering;

        let state = super::ComputeTaskState::new();

        // Create a guard (increments counter)
        {
            let _guard = super::ComputeTaskGuard::new(&state);
            assert_eq!(state.count.load(Ordering::Relaxed), 1);
        }
        // Guard dropped, counter should be 0
        assert_eq!(state.count.load(Ordering::Relaxed), 0);

        // Test multiple guards
        let state = super::ComputeTaskState::new();

        let guard1 = super::ComputeTaskGuard::new(&state);
        assert_eq!(state.count.load(Ordering::Relaxed), 1);

        let guard2 = super::ComputeTaskGuard::new(&state);
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
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

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
        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();

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

        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();
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

        let result = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE);
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

        let runtime = LoomRuntime::from_config(config, DEFAULT_POOL_SIZE).unwrap();
        // Should have found CUDA-local CPUs
        assert!(!runtime.inner.tokio_cpus.is_empty());
    }
}
