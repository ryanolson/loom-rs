//! Async-to-rayon bridge with minimal overhead.
//!
//! This module provides a custom future for spawning work on rayon and awaiting
//! the result from an async context. It replaces tokio-rayon with a more
//! efficient implementation using atomic wakers instead of channels.
//!
//! # Performance
//!
//! - `RayonTask`: ~32 bytes allocation (Arc + state) vs ~80 bytes for oneshot channels
//! - `PooledRayonTask`: 0 bytes allocation after warmup (reuses pooled state)
//! - Uses `diatomic-waker` for efficient cross-thread waking
//! - Uses `parking_lot::Mutex` for lower-overhead locking

#![allow(dead_code)]

use std::any::Any;
use std::future::Future;
use std::panic;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use diatomic_waker::DiatomicWaker;
use parking_lot::{Condvar, Mutex};

/// A future representing work spawned on rayon.
///
/// Uses atomic waker for minimal overhead (~32 bytes vs ~80 bytes for oneshot).
pub struct RayonTask<R> {
    state: Arc<TaskState<R>>,
}

/// Shared state between the task future and completion handle.
///
/// This is public for use by the pool module.
pub struct TaskState<R> {
    result: Mutex<Option<R>>,
    waker: DiatomicWaker,
}

impl<R> TaskState<R> {
    /// Create a new task state.
    pub fn new() -> Self {
        Self {
            result: Mutex::new(None),
            waker: DiatomicWaker::new(),
        }
    }

    /// Reset the state for reuse.
    ///
    /// This clears any pending result and prepares the state for a new task.
    pub fn reset(&self) {
        *self.result.lock() = None;
        // DiatomicWaker doesn't need explicit reset - it will be overwritten on next register
    }
}

impl<R> Default for TaskState<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> RayonTask<R> {
    /// Create a new task and return (task, completion_handle).
    #[inline]
    pub fn new() -> (Self, TaskCompletion<R>) {
        let state = Arc::new(TaskState {
            result: Mutex::new(None),
            waker: DiatomicWaker::new(),
        });
        (
            RayonTask {
                state: state.clone(),
            },
            TaskCompletion { state },
        )
    }
}

/// Handle used by rayon thread to complete the task.
pub struct TaskCompletion<R> {
    state: Arc<TaskState<R>>,
}

impl<R> TaskCompletion<R> {
    /// Complete the task with a result. Wakes the waiting future.
    #[inline]
    pub fn complete(self, result: R) {
        *self.state.result.lock() = Some(result);
        self.state.waker.notify();
    }
}

/// A future representing pooled work spawned on rayon.
///
/// Unlike `RayonTask`, this reuses a `TaskState` from a pool, achieving
/// zero allocation after warmup. The caller is responsible for returning
/// the state to the pool after the result is consumed.
pub struct PooledRayonTask<R: Send + 'static> {
    state: Arc<TaskState<R>>,
}

/// Handle used by rayon thread to complete a pooled task.
pub struct PooledTaskCompletion<R: Send + 'static> {
    state: Arc<TaskState<R>>,
}

impl<R: Send + 'static> PooledRayonTask<R> {
    /// Create a new pooled task from existing state.
    ///
    /// Returns (task, completion_handle, state_for_return).
    /// The caller should return `state_for_return` to the pool after awaiting the task.
    #[inline]
    pub fn new(state: Arc<TaskState<R>>) -> (Self, PooledTaskCompletion<R>, Arc<TaskState<R>>) {
        let state_for_return = state.clone();
        (
            PooledRayonTask {
                state: state.clone(),
            },
            PooledTaskCompletion { state },
            state_for_return,
        )
    }
}

impl<R: Send + 'static> PooledTaskCompletion<R> {
    /// Complete the task with a result. Wakes the waiting future.
    #[inline]
    pub fn complete(self, result: R) {
        *self.state.result.lock() = Some(result);
        self.state.waker.notify();
    }
}

impl<R: Send + 'static> Future for PooledRayonTask<R> {
    type Output = R;

    #[inline]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // Try to take result first
        if let Some(result) = self.state.result.lock().take() {
            return Poll::Ready(result);
        }

        // Register/update waker
        // SAFETY: PooledRayonTask is polled from a single task at a time
        unsafe {
            self.state.waker.register(cx.waker());
        }

        // Check again in case completion happened between our first check and registration
        if let Some(result) = self.state.result.lock().take() {
            Poll::Ready(result)
        } else {
            Poll::Pending
        }
    }
}

impl<R> Future for RayonTask<R> {
    type Output = R;

    #[inline]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // Try to take result first
        if let Some(result) = self.state.result.lock().take() {
            return Poll::Ready(result);
        }

        // Register/update waker
        // SAFETY: RayonTask is polled from a single task at a time (the async executor
        // guarantees this). The DiatomicWaker register() requires that it not be called
        // concurrently from multiple threads for the same waker, which is satisfied here.
        unsafe {
            self.state.waker.register(cx.waker());
        }

        // Check again in case completion happened between our first check and registration
        if let Some(result) = self.state.result.lock().take() {
            Poll::Ready(result)
        } else {
            Poll::Pending
        }
    }
}

// =============================================================================
// Scoped Compute Types
// =============================================================================

/// State for scoped compute tasks.
///
/// Unlike regular TaskState, this cannot be pooled because R may have lifetime
/// parameters tied to the calling scope ('env).
///
/// The completion signaling uses a condvar for cancellation safety - if the
/// future is dropped before completion (e.g., via `select!`), we must block
/// until the scope finishes to avoid dangling references.
///
/// The result stores `Result<R, Box<dyn Any + Send>>` to handle panics - if the
/// closure panics, we capture the payload and re-raise it when the future is polled.
pub struct ScopedTaskState<R> {
    /// Stores Ok(result) on success, Err(panic_payload) on panic
    result: Mutex<Option<Result<R, Box<dyn Any + Send>>>>,
    waker: DiatomicWaker,
    /// Set to true when the rayon scope completes (success or panic)
    completed: AtomicBool,
    /// Mutex + Condvar for blocking on cancellation
    completion_lock: Mutex<()>,
    completion_condvar: Condvar,
}

impl<R> ScopedTaskState<R> {
    /// Create a new scoped task state.
    pub fn new() -> Self {
        Self {
            result: Mutex::new(None),
            waker: DiatomicWaker::new(),
            completed: AtomicBool::new(false),
            completion_lock: Mutex::new(()),
            completion_condvar: Condvar::new(),
        }
    }
}

impl<R> Default for ScopedTaskState<R> {
    fn default() -> Self {
        Self::new()
    }
}

/// A future representing scoped work spawned on rayon.
///
/// # Cancellation Safety
///
/// If this future is dropped before completion (e.g., via `select!` or runtime
/// shutdown), the Drop impl will block until the rayon scope finishes. This is
/// necessary to prevent use-after-free of borrowed data.
///
/// In normal usage (awaiting to completion), Drop is a no-op since the scope
/// has already completed.
pub struct ScopedComputeFuture<R> {
    state: Arc<ScopedTaskState<R>>,
}

impl<R> ScopedComputeFuture<R> {
    /// Create a new scoped compute future.
    pub fn new(state: Arc<ScopedTaskState<R>>) -> Self {
        Self { state }
    }
}

impl<R> Future for ScopedComputeFuture<R> {
    type Output = R;

    #[inline]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        // Try to take result first
        if let Some(result) = self.state.result.lock().take() {
            return Poll::Ready(unwrap_or_resume_panic(result));
        }

        // Register/update waker
        // SAFETY: ScopedComputeFuture is polled from a single task at a time
        unsafe {
            self.state.waker.register(cx.waker());
        }

        // Check again in case completion happened between our first check and registration
        if let Some(result) = self.state.result.lock().take() {
            Poll::Ready(unwrap_or_resume_panic(result))
        } else {
            Poll::Pending
        }
    }
}

/// Unwrap a result or resume the panic if it was a panic payload.
#[inline]
fn unwrap_or_resume_panic<R>(result: Result<R, Box<dyn Any + Send>>) -> R {
    match result {
        Ok(value) => value,
        Err(payload) => panic::resume_unwind(payload),
    }
}

impl<R> Drop for ScopedComputeFuture<R> {
    fn drop(&mut self) {
        // If the scope hasn't completed, we MUST wait to avoid use-after-free.
        // This only triggers on cancellation (select!, timeout, runtime shutdown).
        // Normal .await returns AFTER completion, so this is a no-op in the common case.
        if !self.state.completed.load(Ordering::Acquire) {
            // Block until the scope completes
            let mut guard = self.state.completion_lock.lock();
            while !self.state.completed.load(Ordering::Acquire) {
                self.state.completion_condvar.wait(&mut guard);
            }
        }
    }
}

/// Handle used by rayon thread to complete a scoped task.
pub struct ScopedCompletion<R> {
    state: Arc<ScopedTaskState<R>>,
}

impl<R> ScopedCompletion<R> {
    /// Create a new scoped completion handle.
    pub fn new(state: Arc<ScopedTaskState<R>>) -> Self {
        Self { state }
    }

    /// Complete the task with a result. Wakes the waiting future.
    #[inline]
    pub fn complete(self, result: R) {
        self.complete_with_result(Ok(result));
    }

    /// Complete the task with a panic payload. Wakes the waiting future.
    ///
    /// The panic will be resumed when the future is polled.
    #[inline]
    pub fn complete_with_panic(self, payload: Box<dyn Any + Send>) {
        self.complete_with_result(Err(payload));
    }

    /// Internal: complete with either success or panic.
    #[inline]
    fn complete_with_result(self, result: Result<R, Box<dyn Any + Send>>) {
        // Store the result
        *self.state.result.lock() = Some(result);

        // Mark as completed BEFORE waking/notifying
        self.state.completed.store(true, Ordering::Release);

        // Wake the async task
        self.state.waker.notify();

        // Notify any blocked Drop (cancellation case)
        let _guard = self.state.completion_lock.lock();
        self.state.completion_condvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_queue::ArrayQueue;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    #[test]
    fn test_rayon_task_completion() {
        let (task, completion) = RayonTask::<i32>::new();

        // Complete from another thread
        let handle = thread::spawn(move || {
            completion.complete(42);
        });

        // May need to wait for completion
        handle.join().unwrap();

        // Poll manually (in real usage, tokio would do this)
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut task = task;

        // Should be ready now
        let pinned = Pin::new(&mut task);
        match pinned.poll(&mut cx) {
            Poll::Ready(val) => assert_eq!(val, 42),
            Poll::Pending => panic!("expected Ready after completion"),
        }
    }

    #[test]
    fn test_rayon_task_pending_before_completion() {
        let (mut task, _completion) = RayonTask::<i32>::new();

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let pinned = Pin::new(&mut task);

        // Should be pending before completion
        assert!(matches!(pinned.poll(&mut cx), Poll::Pending));
    }

    #[test]
    fn test_task_state_reset() {
        let state = TaskState::<i32>::new();
        *state.result.lock() = Some(42);

        state.reset();

        assert!(state.result.lock().is_none());
    }

    #[test]
    fn test_pooled_rayon_task_completion() {
        let state = Arc::new(TaskState::<i32>::new());

        let (task, completion, state_for_return) = PooledRayonTask::new(state);

        // Complete from another thread
        let handle = thread::spawn(move || {
            completion.complete(42);
        });

        handle.join().unwrap();

        // Poll the task
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut task = task;
        let pinned = Pin::new(&mut task);

        match pinned.poll(&mut cx) {
            Poll::Ready(val) => assert_eq!(val, 42),
            Poll::Pending => panic!("expected Ready after completion"),
        }

        // Caller returns state to pool
        state_for_return.reset();
        let pool = Arc::new(crossbeam_queue::ArrayQueue::new(4));
        let _ = pool.push(state_for_return);
        assert!(pool.pop().is_some());
    }

    #[test]
    fn test_pooled_task_reuse() {
        let pool = Arc::new(ArrayQueue::new(4));

        // First task
        let state1 = Arc::new(TaskState::<i32>::new());
        let state1_ptr = Arc::as_ptr(&state1);
        let (mut task1, completion1, state_for_return) = PooledRayonTask::new(state1);

        completion1.complete(1);

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(Pin::new(&mut task1).poll(&mut cx), Poll::Ready(1)));

        // Caller returns state to pool
        state_for_return.reset();
        let _ = pool.push(state_for_return);

        // State should be in pool
        let reused_state = pool.pop().expect("state should be in pool");
        assert_eq!(Arc::as_ptr(&reused_state), state1_ptr);

        // Reuse the state for a second task
        let (mut task2, completion2, _state_for_return2) = PooledRayonTask::new(reused_state);
        completion2.complete(2);
        assert!(matches!(Pin::new(&mut task2).poll(&mut cx), Poll::Ready(2)));
    }

    // =========================================================================
    // Scoped Compute Tests
    // =========================================================================

    #[test]
    fn test_scoped_compute_completion() {
        let state = Arc::new(ScopedTaskState::<i32>::new());
        let future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state);

        // Complete from another thread
        let handle = thread::spawn(move || {
            completion.complete(42);
        });

        handle.join().unwrap();

        // Poll the future
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut future = future;
        let pinned = Pin::new(&mut future);

        match pinned.poll(&mut cx) {
            Poll::Ready(val) => assert_eq!(val, 42),
            Poll::Pending => panic!("expected Ready after completion"),
        }
    }

    #[test]
    fn test_scoped_compute_pending_before_completion() {
        let state = Arc::new(ScopedTaskState::<i32>::new());
        let mut future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state);

        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        let pinned = Pin::new(&mut future);

        // Should be pending before completion
        assert!(matches!(pinned.poll(&mut cx), Poll::Pending));

        // Complete to avoid blocking on drop
        completion.complete(0);
    }

    #[test]
    fn test_scoped_compute_drop_blocks_until_complete() {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        let state = Arc::new(ScopedTaskState::<i32>::new());
        let future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state.clone());

        let completed_flag = Arc::new(AtomicBool::new(false));
        let completed_flag_clone = completed_flag.clone();

        // Spawn a thread that will complete after a delay
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            completed_flag_clone.store(true, Ordering::Release);
            completion.complete(42);
        });

        // Drop the future - this should block until completion
        drop(future);

        // By the time drop returns, completion should have happened
        assert!(
            completed_flag.load(Ordering::Acquire),
            "Drop should have blocked until completion"
        );

        handle.join().unwrap();
    }

    #[test]
    fn test_scoped_compute_drop_noop_after_completion() {
        let state = Arc::new(ScopedTaskState::<i32>::new());
        let future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state);

        // Complete immediately
        completion.complete(42);

        // Drop should be a no-op (not block)
        let start = std::time::Instant::now();
        drop(future);
        let elapsed = start.elapsed();

        // Should complete nearly instantly (< 1ms)
        assert!(
            elapsed.as_millis() < 10,
            "Drop after completion should be instant, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_scoped_compute_waker_is_called() {
        use std::sync::atomic::AtomicUsize;
        use std::time::Duration;

        let state = Arc::new(ScopedTaskState::<i32>::new());
        let mut future = ScopedComputeFuture::new(state.clone());
        let completion = ScopedCompletion::new(state);

        // Create a custom waker that tracks wake calls
        let wake_count = Arc::new(AtomicUsize::new(0));
        let wake_count_clone = wake_count.clone();

        let waker = futures::task::waker(Arc::new(CountingWaker {
            count: wake_count_clone,
        }));
        let mut cx = Context::from_waker(&waker);

        // First poll should return Pending and register the waker
        assert!(matches!(Pin::new(&mut future).poll(&mut cx), Poll::Pending));
        assert_eq!(
            wake_count.load(Ordering::Relaxed),
            0,
            "No wake before completion"
        );

        // Complete from another thread
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            completion.complete(42);
        });

        handle.join().unwrap();

        // Waker should have been called
        assert!(
            wake_count.load(Ordering::Relaxed) >= 1,
            "Waker should be called on completion"
        );

        // Second poll should return Ready
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(val) => assert_eq!(val, 42),
            Poll::Pending => panic!("expected Ready after completion"),
        }
    }

    /// Helper waker that counts wake calls
    struct CountingWaker {
        count: Arc<AtomicUsize>,
    }

    impl futures::task::ArcWake for CountingWaker {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.count.fetch_add(1, Ordering::Relaxed);
        }
    }
}
