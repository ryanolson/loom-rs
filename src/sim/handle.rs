use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use super::queue::{CancelHandle, EventQueue};
use super::time::SimTime;

/// Handle for simulated components to interact with the DES scheduler.
///
/// Cheaply cloneable. Every simulated component holds one.
///
/// # Two patterns
///
/// **Fire-and-forget** — schedule a closure at a future virtual time:
/// ```ignore
/// handle.schedule(Duration::from_millis(10), || {
///     channel.send(message).ok();
/// });
/// ```
///
/// **Suspending delay** — pause an async task until virtual time advances:
/// ```ignore
/// handle.delay(Duration::from_secs(1)).await;
/// ```
#[derive(Clone)]
pub struct SimHandle {
    pub(crate) queue: Arc<Mutex<EventQueue>>,
    pub(crate) clock: Arc<AtomicU64>,
}

impl SimHandle {
    fn assert_not_past(&self, api_name: &str, time: SimTime) {
        let now = self.now();
        assert!(
            time >= now,
            "loom_rs::sim::{api_name}() cannot schedule work in the past: requested {time:?}, current virtual time {now:?}"
        );
    }

    /// Schedule a closure to execute at `now() + delay`.
    ///
    /// Returns immediately. The closure runs when the simulation steps to its target time.
    pub fn schedule(&self, delay: Duration, action: impl FnOnce() + Send + 'static) {
        let time = self.now() + delay;
        self.queue.lock().unwrap().insert(time, Box::new(action));
    }

    /// Schedule a closure at an absolute virtual time.
    pub fn schedule_at(&self, time: SimTime, action: impl FnOnce() + Send + 'static) {
        self.assert_not_past("schedule_at", time);
        self.queue.lock().unwrap().insert(time, Box::new(action));
    }

    /// Suspend this async task for `duration` of virtual time.
    ///
    /// Returns a future backed by the DES queue. When first polled, it registers
    /// a waker callback at `now() + duration`. The future completes when the
    /// simulation's stepping loop advances past the target time.
    pub fn delay(&self, duration: Duration) -> DelayFuture {
        DelayFuture {
            queue: self.queue.clone(),
            clock: self.clock.clone(),
            target_time: self.now() + duration,
            waker_slot: Arc::new(Mutex::new(None)),
            cancel_handle: None,
            registered: false,
        }
    }

    /// Spawn an async task at `now() + delay`.
    ///
    /// The future is created immediately (capturing state) but not polled
    /// until the DES scheduler reaches the target time, at which point it's
    /// spawned onto tokio. Inside the future, `delay().await`, channels,
    /// and any async API work normally.
    ///
    /// ```ignore
    /// handle.spawn_after(network_delay, async move {
    ///     sim.delay(process_time).await;
    ///     send_response();
    /// });
    /// ```
    pub fn spawn_after<F>(&self, delay: Duration, fut: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let time = self.now() + delay;
        self.queue.lock().unwrap().insert(
            time,
            Box::new(move || {
                tokio::spawn(fut);
            }),
        );
    }

    /// Spawn an async task at an absolute virtual time.
    pub fn spawn_at<F>(&self, time: SimTime, fut: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.assert_not_past("spawn_at", time);
        self.queue.lock().unwrap().insert(
            time,
            Box::new(move || {
                tokio::spawn(fut);
            }),
        );
    }

    /// Current virtual time.
    pub fn now(&self) -> SimTime {
        Duration::from_nanos(self.clock.load(Ordering::Acquire))
    }
}

/// Future that completes when virtual time reaches the target.
///
/// On first poll, enqueues a waker callback in the DES queue at the target time.
/// The waker is stored in a shared slot so it is always up-to-date if the future
/// is re-polled with a different waker (e.g. after being moved between tasks).
pub struct DelayFuture {
    queue: Arc<Mutex<EventQueue>>,
    clock: Arc<AtomicU64>,
    target_time: SimTime,
    waker_slot: Arc<Mutex<Option<Waker>>>,
    cancel_handle: Option<CancelHandle>,
    registered: bool,
}

impl Future for DelayFuture {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let now = Duration::from_nanos(self.clock.load(Ordering::Acquire));
        if now >= self.target_time {
            return Poll::Ready(());
        }
        // Always refresh the waker so the DES action wakes the current task
        *self.waker_slot.lock().unwrap() = Some(cx.waker().clone());
        if !self.registered {
            self.registered = true;
            let slot = self.waker_slot.clone();
            let cancel_handle = self.queue.lock().unwrap().insert_cancellable(
                self.target_time,
                Box::new(move || {
                    if let Some(w) = slot.lock().unwrap().take() {
                        w.wake();
                    }
                }),
            );
            self.cancel_handle = Some(cancel_handle);
        }
        Poll::Pending
    }
}

impl Drop for DelayFuture {
    fn drop(&mut self) {
        if let Some(cancel_handle) = self.cancel_handle.take() {
            cancel_handle.cancel();
        }
    }
}
