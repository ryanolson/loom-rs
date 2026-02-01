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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use diatomic_waker::DiatomicWaker;
use parking_lot::Mutex;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_queue::ArrayQueue;
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
}
