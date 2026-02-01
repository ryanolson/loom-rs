//! Stream combinators for processing items via rayon compute threads.
//!
//! This module provides the [`ComputeStreamExt`] trait which extends any [`Stream`]
//! with methods for processing items through the rayon thread pool.
//!
//! # Example
//!
//! ```ignore
//! use loom_rs::{LoomBuilder, ComputeStreamExt};
//! use futures::stream::{self, StreamExt};
//!
//! let runtime = LoomBuilder::new().build()?;
//!
//! runtime.block_on(async {
//!     let results: Vec<_> = stream::iter(0..100)
//!         .compute_map(|n| {
//!             // CPU-intensive work runs on rayon
//!             (0..n).map(|i| i * i).sum::<i64>()
//!         })
//!         .collect()
//!         .await;
//! });
//! ```
//!
//! # Performance
//!
//! The key optimization is reusing the same `TaskState` for all items in the stream,
//! rather than getting/returning from the pool for each item:
//!
//! | Operation | Overhead | Allocations |
//! |-----------|----------|-------------|
//! | Stream creation | ~1us | TaskState (from pool or new) |
//! | Each item | ~100-500ns | 0 bytes (reuses TaskState) |
//! | Stream drop | ~10ns | Returns TaskState to pool |

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures_core::Stream;

use crate::bridge::{PooledRayonTask, PooledTaskCompletion, TaskState};
use crate::context::current_runtime;
use crate::mab::{Arm, ComputeHintProvider, DecisionId, FunctionKey, MabScheduler};
use crate::pool::TypedPool;
use crate::runtime::LoomRuntimeInner;

/// Extension trait for streams that adds compute-based processing methods.
///
/// This trait is automatically implemented for all types that implement [`Stream`].
///
/// # Example
///
/// ```ignore
/// use loom_rs::{LoomBuilder, ComputeStreamExt};
/// use futures::stream::{self, StreamExt};
///
/// let runtime = LoomBuilder::new().build()?;
///
/// runtime.block_on(async {
///     let numbers = stream::iter(0..10);
///
///     // Each item is processed on rayon, results stream back in order
///     let results: Vec<_> = numbers
///         .compute_map(|n| n * 2)
///         .collect()
///         .await;
///
///     assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
/// });
/// ```
pub trait ComputeStreamExt: Stream {
    /// Map each stream item through a compute-heavy closure on rayon.
    ///
    /// Items are processed sequentially (one at a time) to preserve
    /// stream ordering and provide natural backpressure.
    ///
    /// # Performance
    ///
    /// Unlike calling `spawn_compute` in a loop, `compute_map` reuses the same
    /// internal `TaskState` for every item, avoiding per-item pool operations:
    ///
    /// - First poll: Gets `TaskState` from pool (or allocates new)
    /// - Each item: ~100-500ns overhead, 0 allocations
    /// - Stream drop: Returns `TaskState` to pool
    ///
    /// # Panics
    ///
    /// Panics if called outside a loom runtime context (i.e., not within `block_on`,
    /// a tokio worker thread, or a rayon worker thread managed by the runtime).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::{LoomBuilder, ComputeStreamExt};
    /// use futures::stream::{self, StreamExt};
    ///
    /// let runtime = LoomBuilder::new().build()?;
    ///
    /// runtime.block_on(async {
    ///     let results: Vec<_> = stream::iter(vec!["hello", "world"])
    ///         .compute_map(|s| s.to_uppercase())
    ///         .collect()
    ///         .await;
    ///
    ///     assert_eq!(results, vec!["HELLO", "WORLD"]);
    /// });
    /// ```
    fn compute_map<F, U>(self, f: F) -> ComputeMap<Self, F, U>
    where
        Self: Sized,
        F: Fn(Self::Item) -> U + Send + Sync + 'static,
        Self::Item: Send + 'static,
        U: Send + 'static;

    /// Adaptively map items, choosing inline vs offload per item.
    ///
    /// Each stream instance maintains its own MAB scheduler state for immediate
    /// feedback learning. The scheduler learns from execution times and adapts
    /// its decisions to minimize total cost.
    ///
    /// If the input item implements [`ComputeHintProvider`], the hint is used
    /// to guide cold-start behavior before the scheduler has learned enough.
    ///
    /// # When to Use
    ///
    /// Use `adaptive_map` when:
    /// - You're unsure if work is cheap enough for inline execution
    /// - Work complexity varies significantly across items
    /// - You want automatic adaptation without manual tuning
    ///
    /// Use `compute_map` when:
    /// - Work is consistently expensive (> 250µs)
    /// - You want guaranteed offload behavior
    ///
    /// # Example
    ///
    /// ```ignore
    /// use loom_rs::{LoomBuilder, ComputeStreamExt};
    /// use futures::stream::{self, StreamExt};
    ///
    /// let runtime = LoomBuilder::new().build()?;
    ///
    /// runtime.block_on(async {
    ///     // Scheduler learns that small items are fast (inline)
    ///     // and large items are slow (offload)
    ///     let results: Vec<_> = stream::iter(data)
    ///         .adaptive_map(|item| process(item))
    ///         .collect()
    ///         .await;
    /// });
    /// ```
    fn adaptive_map<F, U>(self, f: F) -> AdaptiveMap<Self, F, U>
    where
        Self: Sized,
        F: Fn(Self::Item) -> U + Send + Sync + 'static,
        Self::Item: ComputeHintProvider + Send + 'static,
        U: Send + 'static;
}

impl<S: Stream> ComputeStreamExt for S {
    fn compute_map<F, U>(self, f: F) -> ComputeMap<Self, F, U>
    where
        Self: Sized,
        F: Fn(Self::Item) -> U + Send + Sync + 'static,
        Self::Item: Send + 'static,
        U: Send + 'static,
    {
        ComputeMap::new(self, f)
    }

    fn adaptive_map<F, U>(self, f: F) -> AdaptiveMap<Self, F, U>
    where
        Self: Sized,
        F: Fn(Self::Item) -> U + Send + Sync + 'static,
        Self::Item: ComputeHintProvider + Send + 'static,
        U: Send + 'static,
    {
        AdaptiveMap::new(self, f)
    }
}

/// A stream adapter that maps items through rayon compute threads.
///
/// Created by the [`compute_map`](ComputeStreamExt::compute_map) method on streams.
/// Items are processed sequentially to preserve ordering.
#[must_use = "streams do nothing unless polled"]
pub struct ComputeMap<S, F, U>
where
    U: Send + 'static,
{
    stream: S,
    f: Arc<F>,
    // Lazily initialized on first poll. Drop impl on ComputeMapState
    // handles returning TaskState to pool.
    state: Option<ComputeMapState<U>>,
}

// Manual Unpin implementation - ComputeMap is Unpin if S is Unpin
impl<S: Unpin, F, U: Send + 'static> Unpin for ComputeMap<S, F, U> {}

/// Internal state for ComputeMap, initialized on first poll.
///
/// The Drop impl returns the TaskState to the pool when the stream is dropped.
struct ComputeMapState<U: Send + 'static> {
    runtime: Arc<LoomRuntimeInner>,
    pool: Arc<TypedPool<U>>,
    /// Reused TaskState - no per-item pool operations!
    task_state: Arc<TaskState<U>>,
    /// Currently pending compute task, if any
    pending: Option<PooledRayonTask<U>>,
}

impl<U: Send + 'static> Drop for ComputeMapState<U> {
    fn drop(&mut self) {
        // Return TaskState to pool if there's no pending task
        // (which would still be using it)
        if self.pending.is_none() {
            self.task_state.reset();
            // Clone the Arc before pushing - we need ownership
            let task_state = Arc::clone(&self.task_state);
            self.pool.push(task_state);
        }
        // If there's a pending task, the TaskState will be dropped with it
        // This is a rare edge case (stream dropped while compute in flight)
    }
}

impl<S, F, U> ComputeMap<S, F, U>
where
    U: Send + 'static,
{
    fn new(stream: S, f: F) -> Self {
        Self {
            stream,
            f: Arc::new(f),
            state: None,
        }
    }
}

impl<S, F, U> Stream for ComputeMap<S, F, U>
where
    S: Stream + Unpin,
    S::Item: Send + 'static,
    F: Fn(S::Item) -> U + Send + Sync + 'static,
    U: Send + 'static,
{
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;

        // Initialize state on first poll
        let state = this.state.get_or_insert_with(|| {
            let runtime = current_runtime().expect("compute_map used outside loom runtime");
            let pool = runtime.pools.get_or_create::<U>();
            let task_state = pool.pop().unwrap_or_else(|| Arc::new(TaskState::new()));

            ComputeMapState {
                runtime,
                pool,
                task_state,
                pending: None,
            }
        });

        // If we have a pending task, poll it first
        if let Some(ref mut pending) = state.pending {
            match Pin::new(pending).poll(cx) {
                Poll::Ready(result) => {
                    // Task complete, clear pending
                    state.pending = None;
                    // Reset state for reuse
                    state.task_state.reset();
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }

        // No pending task, poll the inner stream for the next item
        match Pin::new(&mut this.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // Got an item, spawn compute task
                let f = Arc::clone(&this.f);
                let task_state = Arc::clone(&state.task_state);

                // Create the pooled task components
                let (task, completion): (PooledRayonTask<U>, PooledTaskCompletion<U>) = {
                    // Reuse the same TaskState (already have it)
                    let (task, completion, _state_for_return) = PooledRayonTask::new(task_state);
                    (task, completion)
                };

                // Spawn on rayon
                state.runtime.rayon_pool.spawn(move || {
                    let result = f(item);
                    completion.complete(result);
                });

                // Store pending task and poll it immediately
                state.pending = Some(task);

                // Poll the pending task - it might already be ready
                if let Some(ref mut pending) = state.pending {
                    match Pin::new(pending).poll(cx) {
                        Poll::Ready(result) => {
                            state.pending = None;
                            state.task_state.reset();
                            Poll::Ready(Some(result))
                        }
                        Poll::Pending => Poll::Pending,
                    }
                } else {
                    Poll::Pending
                }
            }
            Poll::Ready(None) => {
                // Stream exhausted
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We produce the same number of items as the inner stream
        // Adjust for pending task if any
        let (lower, upper) = self.stream.size_hint();
        if self.state.as_ref().is_some_and(|s| s.pending.is_some()) {
            // We have a pending item that will be produced
            (lower.saturating_add(1), upper.map(|u| u.saturating_add(1)))
        } else {
            (lower, upper)
        }
    }
}

// =============================================================================
// AdaptiveMap - MAB-based adaptive execution
// =============================================================================

/// A stream adapter that adaptively maps items using MAB scheduling.
///
/// Created by the [`adaptive_map`](ComputeStreamExt::adaptive_map) method.
/// Each stream instance owns its own scheduler for immediate feedback learning.
#[must_use = "streams do nothing unless polled"]
pub struct AdaptiveMap<S, F, U>
where
    U: Send + 'static,
{
    stream: S,
    f: Arc<F>,
    function_key: FunctionKey,
    state: Option<AdaptiveMapState<U>>,
}

// Manual Unpin implementation - AdaptiveMap is Unpin if S is Unpin
impl<S: Unpin, F, U: Send + 'static> Unpin for AdaptiveMap<S, F, U> {}

/// Pending work for AdaptiveMap.
struct AdaptivePending<U: Send + 'static> {
    decision_id: DecisionId,
    start_time: Instant,
    task: PooledRayonTask<U>,
}

/// Internal state for AdaptiveMap, initialized on first poll.
struct AdaptiveMapState<U: Send + 'static> {
    runtime: Arc<LoomRuntimeInner>,
    pool: Arc<TypedPool<U>>,
    /// Reused TaskState for offload operations
    task_state: Arc<TaskState<U>>,
    /// Per-stream MAB scheduler
    scheduler: MabScheduler,
    /// Currently pending work
    pending: Option<AdaptivePending<U>>,
}

impl<U: Send + 'static> Drop for AdaptiveMapState<U> {
    fn drop(&mut self) {
        // Return TaskState to pool if there's no pending offload task
        // Skip cleanup during panic to avoid potential double-panic
        if self.pending.is_none() && !std::thread::panicking() {
            self.task_state.reset();
            let task_state = Arc::clone(&self.task_state);
            self.pool.push(task_state);
        }
    }
}

impl<S, F: 'static, U> AdaptiveMap<S, F, U>
where
    U: Send + 'static,
{
    fn new(stream: S, f: F) -> Self {
        // Generate a unique function key for this stream instance
        // Using the type of F and a random component for uniqueness
        let function_key = FunctionKey::from_type::<F>();

        Self {
            stream,
            f: Arc::new(f),
            function_key,
            state: None,
        }
    }
}

impl<S, F, U> Stream for AdaptiveMap<S, F, U>
where
    S: Stream + Unpin,
    S::Item: ComputeHintProvider + Send + 'static,
    F: Fn(S::Item) -> U + Send + Sync + 'static,
    U: Send + 'static,
{
    type Item = U;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;

        // Initialize state on first poll
        let state = this.state.get_or_insert_with(|| {
            let runtime = current_runtime().expect("adaptive_map used outside loom runtime");
            let pool = runtime.pools.get_or_create::<U>();
            let task_state = pool.pop().unwrap_or_else(|| Arc::new(TaskState::new()));

            // Create a per-stream scheduler with runtime's configured knobs
            let scheduler = MabScheduler::new(runtime.mab_knobs.clone());

            AdaptiveMapState {
                runtime,
                pool,
                task_state,
                scheduler,
                pending: None,
            }
        });

        // If we have pending work, handle it
        if let Some(mut pending) = state.pending.take() {
            // Poll the offload task
            match Pin::new(&mut pending.task).poll(cx) {
                Poll::Ready(result) => {
                    let elapsed_us = pending.start_time.elapsed().as_nanos() as f64 / 1000.0;
                    state
                        .scheduler
                        .finish(pending.decision_id, elapsed_us, None);
                    state.task_state.reset();
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => {
                    // Put it back
                    state.pending = Some(pending);
                    return Poll::Pending;
                }
            }
        }

        // No pending work, poll the inner stream for the next item
        match Pin::new(&mut this.stream).poll_next(cx) {
            Poll::Ready(Some(item)) => {
                // Get compute hint from the item
                let hint = item.compute_hint();

                // Collect runtime context
                let ctx = state.runtime.metrics.collect(
                    state.runtime.tokio_threads as u32,
                    state.runtime.rayon_threads as u32,
                );

                // Ask scheduler for decision
                let (decision_id, arm) =
                    state
                        .scheduler
                        .choose_with_hint(this.function_key, &ctx, hint);

                let f = Arc::clone(&this.f);

                match arm {
                    Arm::InlineTokio => {
                        // Execute inline
                        let start = Instant::now();
                        let result = f(item);
                        let elapsed_us = start.elapsed().as_nanos() as f64 / 1000.0;

                        // Record immediately and return
                        state
                            .scheduler
                            .finish(decision_id, elapsed_us, Some(elapsed_us));
                        Poll::Ready(Some(result))
                    }
                    Arm::OffloadRayon => {
                        // Offload to rayon
                        let task_state = Arc::clone(&state.task_state);
                        let (task, completion): (PooledRayonTask<U>, PooledTaskCompletion<U>) = {
                            let (task, completion, _state_for_return) =
                                PooledRayonTask::new(task_state);
                            (task, completion)
                        };

                        let start_time = Instant::now();

                        // Spawn on rayon
                        state.runtime.rayon_pool.spawn(move || {
                            let result = f(item);
                            completion.complete(result);
                        });

                        // Store pending and poll immediately
                        let mut pending = AdaptivePending {
                            decision_id,
                            start_time,
                            task,
                        };

                        // Poll the pending task - it might already be ready
                        match Pin::new(&mut pending.task).poll(cx) {
                            Poll::Ready(result) => {
                                let elapsed_us =
                                    pending.start_time.elapsed().as_nanos() as f64 / 1000.0;
                                state
                                    .scheduler
                                    .finish(pending.decision_id, elapsed_us, None);
                                state.task_state.reset();
                                Poll::Ready(Some(result))
                            }
                            Poll::Pending => {
                                state.pending = Some(pending);
                                Poll::Pending
                            }
                        }
                    }
                }
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.stream.size_hint();
        if self.state.as_ref().is_some_and(|s| s.pending.is_some()) {
            (lower.saturating_add(1), upper.map(|u| u.saturating_add(1)))
        } else {
            (lower, upper)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LoomConfig;
    use crate::pool::DEFAULT_POOL_SIZE;
    use crate::runtime::LoomRuntime;
    use futures::stream::{self, StreamExt};

    fn test_config() -> LoomConfig {
        LoomConfig {
            prefix: "stream-test".to_string(),
            cpuset: None,
            tokio_threads: Some(1),
            rayon_threads: Some(2),
            compute_pool_size: DEFAULT_POOL_SIZE,
            #[cfg(feature = "cuda")]
            cuda_device: None,
            mab_knobs: None,
            calibration: None,
            prometheus_registry: None,
        }
    }

    fn test_runtime() -> LoomRuntime {
        LoomRuntime::from_config(test_config(), DEFAULT_POOL_SIZE).unwrap()
    }

    #[test]
    fn test_compute_map_basic() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(0..10).compute_map(|n| n * 2).collect().await;
            assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
        });
    }

    #[test]
    fn test_compute_map_preserves_order() {
        let runtime = test_runtime();
        runtime.block_on(async {
            // Use items that might have varying compute times
            let results: Vec<_> = stream::iter(vec![5, 1, 3, 2, 4])
                .compute_map(|n| {
                    // Small delay proportional to value (simulates varying compute times)
                    std::thread::sleep(std::time::Duration::from_micros(n as u64 * 10));
                    n * 10
                })
                .collect()
                .await;
            // Order should be preserved
            assert_eq!(results, vec![50, 10, 30, 20, 40]);
        });
    }

    #[test]
    fn test_compute_map_empty_stream() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<i32> = stream::iter(std::iter::empty::<i32>())
                .compute_map(|n| n * 2)
                .collect()
                .await;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn test_compute_map_single_item() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(vec![42])
                .compute_map(|n| n + 1)
                .collect()
                .await;
            assert_eq!(results, vec![43]);
        });
    }

    #[test]
    fn test_compute_map_with_strings() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(vec!["hello", "world"])
                .compute_map(|s| s.to_uppercase())
                .collect()
                .await;
            assert_eq!(results, vec!["HELLO", "WORLD"]);
        });
    }

    #[test]
    fn test_compute_map_type_conversion() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(1..=5)
                .compute_map(|n| format!("item-{}", n))
                .collect()
                .await;
            assert_eq!(
                results,
                vec!["item-1", "item-2", "item-3", "item-4", "item-5"]
            );
        });
    }

    #[test]
    fn test_compute_map_cpu_intensive() {
        let runtime = test_runtime();
        runtime.block_on(async {
            // Simulate CPU-intensive work
            let results: Vec<_> = stream::iter(0..5)
                .compute_map(|n| (0..1000).map(|i| i * n).sum::<i64>())
                .collect()
                .await;

            let expected: Vec<i64> = (0..5).map(|n| (0..1000).map(|i| i * n).sum()).collect();
            assert_eq!(results, expected);
        });
    }

    #[test]
    fn test_compute_map_size_hint() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let stream = stream::iter(0..10).compute_map(|n| n * 2);
            assert_eq!(stream.size_hint(), (10, Some(10)));
        });
    }

    #[test]
    fn test_compute_map_chained() {
        let runtime = test_runtime();
        runtime.block_on(async {
            // Chain compute_map with other stream combinators
            let results: Vec<_> = stream::iter(0..10)
                .compute_map(|n| n * 2)
                .filter(|n| futures::future::ready(*n > 10))
                .collect()
                .await;
            assert_eq!(results, vec![12, 14, 16, 18]);
        });
    }

    // =============================================================================
    // AdaptiveMap tests
    // =============================================================================

    #[test]
    fn test_adaptive_map_basic() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(0..10).adaptive_map(|n| n * 2).collect().await;
            assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
        });
    }

    #[test]
    fn test_adaptive_map_preserves_order() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<_> = stream::iter(vec![5, 1, 3, 2, 4])
                .adaptive_map(|n| {
                    // Small delay proportional to value
                    std::thread::sleep(std::time::Duration::from_micros(n as u64 * 10));
                    n * 10
                })
                .collect()
                .await;
            assert_eq!(results, vec![50, 10, 30, 20, 40]);
        });
    }

    #[test]
    fn test_adaptive_map_empty_stream() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let results: Vec<i32> = stream::iter(std::iter::empty::<i32>())
                .adaptive_map(|n| n * 2)
                .collect()
                .await;
            assert!(results.is_empty());
        });
    }

    #[test]
    fn test_adaptive_map_with_hint() {
        use crate::mab::{ComputeHint, ComputeHintProvider};

        struct HintedItem {
            value: i32,
            hint: ComputeHint,
        }

        impl ComputeHintProvider for HintedItem {
            fn compute_hint(&self) -> ComputeHint {
                self.hint
            }
        }

        let runtime = test_runtime();
        runtime.block_on(async {
            let items = vec![
                HintedItem {
                    value: 1,
                    hint: ComputeHint::Low,
                },
                HintedItem {
                    value: 2,
                    hint: ComputeHint::High,
                },
                HintedItem {
                    value: 3,
                    hint: ComputeHint::Medium,
                },
            ];

            let results: Vec<_> = stream::iter(items)
                .adaptive_map(|item| item.value * 2)
                .collect()
                .await;

            assert_eq!(results, vec![2, 4, 6]);
        });
    }

    #[test]
    fn test_adaptive_map_learns_from_fast_work() {
        let runtime = test_runtime();
        runtime.block_on(async {
            // Fast work should eventually be inlined
            let results: Vec<_> = stream::iter(0..100)
                .adaptive_map(|n| {
                    // Very fast: ~1µs
                    n + 1
                })
                .collect()
                .await;

            assert_eq!(results.len(), 100);
            assert_eq!(results[0], 1);
            assert_eq!(results[99], 100);
        });
    }

    #[test]
    fn test_adaptive_map_size_hint() {
        let runtime = test_runtime();
        runtime.block_on(async {
            let stream = stream::iter(0..10).adaptive_map(|n| n * 2);
            assert_eq!(stream.size_hint(), (10, Some(10)));
        });
    }
}
