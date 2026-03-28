use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::runtime::LoomRuntime;
use crate::LoomBuilder;

use super::handle::SimHandle;
use super::queue::EventQueue;
use super::time::SimTime;

/// Maximum actions at a single timestamp before livelock detection triggers.
const MAX_SAME_TIME_ACTIONS: usize = 100_000;

/// Outcome of a single simulation step.
pub enum StepOutcome {
    /// Advanced to this virtual time and processed all work there.
    Advanced(SimTime),
    /// No more events in the queue.
    NoMoreEvents,
}

/// Discrete-event simulation runtime.
///
/// Wraps a [`LoomRuntime`] configured for simulation (current-thread tokio,
/// paused virtual clock, rayon APIs disabled). Owns the DES event queue and
/// drives the stepping loop.
///
/// # Example
///
/// ```ignore
/// use loom_rs::sim::{SimulationRuntime, StepOutcome};
/// use std::time::Duration;
///
/// let mut sim = SimulationRuntime::new()?;
/// let handle = sim.handle();
///
/// // Schedule an action at T=1s
/// handle.schedule(Duration::from_secs(1), || {
///     println!("Hello from virtual time!");
/// });
///
/// sim.step()?; // advances to T=1s, executes the action
/// ```
pub struct SimulationRuntime {
    loom: LoomRuntime,
    queue: Arc<Mutex<EventQueue>>,
    clock: Arc<AtomicU64>,
    current_time: SimTime,
}

impl SimulationRuntime {
    /// Create a new simulation runtime.
    ///
    /// Builds a [`LoomRuntime`] with `current_thread` tokio and paused virtual clock.
    /// All rayon-backed APIs will panic if called.
    pub fn new() -> crate::Result<Self> {
        let loom = LoomBuilder::new().simulation_mode(true).build()?;

        // Pause tokio's clock — all tokio::time is now virtual
        loom.block_on(async {
            tokio::time::pause();
        });

        let queue = Arc::new(Mutex::new(EventQueue::new()));
        let clock = Arc::new(AtomicU64::new(0));

        let handle = SimHandle {
            queue: queue.clone(),
            clock: clock.clone(),
        };

        // Inject sim handle into the runtime so spawn_compute etc. can
        // delay results by measured wall-clock time in virtual time.
        loom.inner.sim_handle.set(handle).ok();

        Ok(Self {
            loom,
            queue,
            clock,
            current_time: Duration::ZERO,
        })
    }

    /// Get a [`SimHandle`] for use by simulated components.
    ///
    /// The handle is cheaply cloneable and can be shared across components.
    pub fn handle(&self) -> SimHandle {
        SimHandle {
            queue: self.queue.clone(),
            clock: self.clock.clone(),
        }
    }

    /// Access the underlying [`LoomRuntime`].
    ///
    /// Useful for `block_on`, `spawn_async`, `tokio_handle`, etc.
    /// Rayon-backed APIs will panic (simulation mode).
    pub fn loom(&self) -> &LoomRuntime {
        &self.loom
    }

    /// Current virtual time.
    pub fn now(&self) -> SimTime {
        self.current_time
    }

    /// Advance to the next scheduled event time and process all work there.
    ///
    /// Advance the simulation by one DES event timestamp.
    ///
    /// The DES contract:
    /// 1. Peek next time T from the event queue
    /// 2. Advance tokio's virtual clock to T (resolves framework timers)
    /// 3. Drain tokio tasks woken by timer resolution
    /// 4. Execute all DES actions scheduled at T (FIFO order)
    /// 5. Drain reactive async work spawned by those actions
    /// 6. Repeat 4-5 for zero-delay cascading (new actions at same T)
    /// 7. Livelock guard: error if > 100k actions at one timestamp
    ///
    /// # Clock ordering invariant
    ///
    /// The DES clock (`SimHandle::now()`) is updated **before** `tokio::time::advance()`
    /// is called. Tasks polled during `advance()` therefore observe the new simulation
    /// time via `SimHandle::now()`, but `tokio::time::Instant::now()` is updated
    /// concurrently by the tokio runtime internals.
    ///
    /// **Do not mix** `tokio::time::Instant::now()` with `SimHandle::now()` within a
    /// single `step()` call — the two clocks may momentarily disagree. Always use
    /// `SimHandle::now()` for simulation-time observations to avoid surprises.
    pub fn step(&mut self) -> crate::Result<StepOutcome> {
        // Drain first: spawned tasks may not have been polled yet, so their
        // DelayFutures/tokio timers haven't registered. This first poll
        // lets them insert wakers into the DES queue / timer wheel.
        self.drain();

        let next_time = match self.queue.lock().unwrap().peek_time() {
            Some(t) => t,
            None => return Ok(StepOutcome::NoMoreEvents),
        };
        assert!(
            next_time >= self.current_time,
            "simulation attempted to move backward in time: next event {next_time:?}, current time {:?}",
            self.current_time
        );

        let delta = next_time.saturating_sub(self.current_time);

        // Update the DES clock before advancing tokio time. `advance()` may poll
        // tasks before it returns, and those tasks must observe the stepped
        // virtual time when they call SimHandle::now()/schedule().
        self.current_time = next_time;
        self.clock
            .store(next_time.as_nanos() as u64, Ordering::Release);

        // 1. Advance tokio's virtual clock to next_time.
        //    advance() internally calls yield_now → runtime never parks → no auto-advance.
        if !delta.is_zero() {
            self.loom.block_on(async {
                tokio::time::advance(delta).await;
                tokio::time::sleep(Duration::ZERO).await;
            });
        }

        // 2. Drain tokio tasks woken by timer resolution
        self.drain();

        self.process_current_time_actions()?;

        Ok(StepOutcome::Advanced(self.current_time))
    }

    /// Step until virtual time reaches `deadline` or the queue is empty.
    pub fn step_until(&mut self, deadline: SimTime) -> crate::Result<()> {
        loop {
            self.drain();
            let should_step = self
                .queue
                .lock()
                .unwrap()
                .peek_time()
                .is_some_and(|t| t <= deadline);
            if should_step {
                self.step()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Advance simulation to `target_time`, processing all events along the way.
    ///
    /// Unlike `step_until` which stops when the DES queue empties, this method
    /// force-advances the tokio virtual clock to `target_time` even if no DES
    /// events remain. This resolves any pending `tokio::time::*` timers
    /// (framework timeouts, heartbeats, etc.) that expire before the target,
    /// and fully settles zero-delay DES work scheduled at `target_time`.
    pub fn run_until(&mut self, target_time: SimTime) -> crate::Result<()> {
        // Drain first: spawned tasks may not have registered timers yet
        self.drain();

        // Process DES events up to target_time
        self.step_until(target_time)?;

        // Force-advance tokio clock to target_time even if DES queue is empty.
        // This resolves framework timers that fire between the last DES event
        // and target_time.
        if self.current_time < target_time {
            let delta = target_time - self.current_time;

            self.current_time = target_time;
            self.clock
                .store(target_time.as_nanos() as u64, Ordering::Release);

            self.loom.block_on(async {
                tokio::time::advance(delta).await;
                tokio::time::sleep(Duration::ZERO).await;
            });
            self.step_until(target_time)?;
        }

        Ok(())
    }

    /// Run until no more events remain. Returns the final virtual time.
    pub fn run(&mut self) -> crate::Result<SimTime> {
        loop {
            match self.step()? {
                StepOutcome::Advanced(_) => continue,
                StepOutcome::NoMoreEvents => return Ok(self.current_time),
            }
        }
    }

    /// Run the simulation until a spawned task completes, returning its value.
    ///
    /// This is the simulation equivalent of SimPy's `env.run(until=process)`.
    /// The future is spawned as a tokio task and the simulation steps forward
    /// until it resolves.
    ///
    /// Returns `Err(SimStall)` if the DES queue empties before the task
    /// completes (the task is stuck waiting for an event that will never fire).
    /// If the monitored future panics, that panic is resumed in the caller.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut sim = SimulationRuntime::new()?;
    /// let handle = sim.handle();
    ///
    /// let result = sim.run_until_complete(async move {
    ///     handle.delay(Duration::from_secs(5)).await;
    ///     42
    /// })?;
    ///
    /// assert_eq!(result, 42);
    /// assert_eq!(sim.now(), Duration::from_secs(5));
    /// ```
    pub fn run_until_complete<F, T>(&mut self, future: F) -> crate::Result<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let mut task = Some(self.loom.spawn_async(future));

        loop {
            self.drain();
            if let Some(result) = self.try_take_completed_task(&mut task) {
                return Ok(result);
            }
            match self.step()? {
                StepOutcome::Advanced(_) => continue,
                StepOutcome::NoMoreEvents => {
                    self.drain();
                    if let Some(result) = self.try_take_completed_task(&mut task) {
                        return Ok(result);
                    }

                    let task = task.take().expect("monitored sim task must exist");
                    self.abort_task(task);
                    return Err(crate::LoomError::SimStall {
                        time: self.current_time,
                    });
                }
            }
        }
    }

    fn process_current_time_actions(&mut self) -> crate::Result<()> {
        let mut total_actions = 0usize;
        loop {
            let mut batch = Vec::new();
            {
                let mut q = self.queue.lock().unwrap();
                while let Some(action) = q.pull_if_at(self.current_time) {
                    batch.push(action);
                }
            }
            if batch.is_empty() {
                break;
            }

            total_actions += batch.len();
            if total_actions > MAX_SAME_TIME_ACTIONS {
                return Err(crate::LoomError::SimLivelock {
                    time: self.current_time,
                    actions: total_actions,
                });
            }

            // Execute inside block_on for:
            // - tokio runtime context (spawn_async works)
            // - current_runtime() thread-local is set (block_on sets it)
            // - drain reactive async work via yield_now
            self.loom.block_on(async {
                for action in batch {
                    action();
                }
                tokio::task::yield_now().await;
            });
        }

        Ok(())
    }

    fn try_take_completed_task<T>(
        &self,
        task: &mut Option<tokio::task::JoinHandle<T>>,
    ) -> Option<T> {
        if task.as_ref().is_some_and(|handle| handle.is_finished()) {
            let task = task.take().expect("finished sim task handle must exist");
            Some(self.join_completed_task(task))
        } else {
            None
        }
    }

    fn join_completed_task<T>(&self, task: tokio::task::JoinHandle<T>) -> T {
        match self.loom.block_on(task) {
            Ok(value) => value,
            Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
            Err(err) => panic!("simulation task unexpectedly cancelled: {err}"),
        }
    }

    fn abort_task<T>(&self, task: tokio::task::JoinHandle<T>) {
        task.abort();
        match self.loom.block_on(task) {
            Ok(_) => {}
            Err(err) if err.is_cancelled() => {}
            Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
            Err(err) => panic!("simulation task abort failed: {err}"),
        }
    }

    /// Drain all ready tokio tasks.
    ///
    /// On `current_thread`, `yield_now` processes the ENTIRE ready queue
    /// (including tasks woken by other tasks) before returning.
    fn drain(&self) {
        self.loom.block_on(async {
            tokio::task::yield_now().await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AOrdering};

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, AOrdering::SeqCst);
        }
    }

    #[test]
    fn test_delay_future_under_des() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let completed = Arc::new(AtomicBool::new(false));

        let c = completed.clone();
        sim.loom().spawn_async(async move {
            handle.delay(Duration::from_secs(1)).await;
            c.store(true, AOrdering::SeqCst);
        });

        // Not yet — no step taken
        sim.drain();
        assert!(!completed.load(AOrdering::SeqCst));

        // Step should advance to T=1s and complete the delay
        let outcome = sim.step().unwrap();
        assert!(matches!(outcome, StepOutcome::Advanced(t) if t == Duration::from_secs(1)));
        assert!(completed.load(AOrdering::SeqCst));
    }

    #[test]
    fn test_tokio_sleep_under_des() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let completed = Arc::new(AtomicBool::new(false));

        // Use a DES event to drive time, and a tokio::time::sleep that rides along
        let c = completed.clone();
        sim.loom().spawn_async(async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            c.store(true, AOrdering::SeqCst);
        });

        // Schedule a DES event at T=3s to drive advancement
        handle.schedule_at(Duration::from_secs(3), || {});

        // Step to T=3s — should also resolve the tokio sleep at T=2s
        let outcome = sim.step().unwrap();
        assert!(matches!(outcome, StepOutcome::Advanced(t) if t == Duration::from_secs(3)));
        assert!(completed.load(AOrdering::SeqCst));
    }

    #[test]
    fn test_same_time_fifo() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let order = Arc::new(AtomicUsize::new(0));

        let t = Duration::from_secs(1);
        for i in 0..3u8 {
            let o = order.clone();
            handle.schedule_at(t, move || {
                let prev = o.fetch_add(1, AOrdering::SeqCst);
                assert_eq!(prev, i as usize);
            });
        }

        sim.step().unwrap();
        assert_eq!(order.load(AOrdering::SeqCst), 3);
    }

    #[test]
    fn test_zero_delay_cascade() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let cascaded = Arc::new(AtomicBool::new(false));

        let h = handle.clone();
        let c = cascaded.clone();
        handle.schedule_at(Duration::from_secs(1), move || {
            // Schedule another action at the same time (zero-delay cascade)
            h.schedule(Duration::ZERO, move || {
                c.store(true, AOrdering::SeqCst);
            });
        });

        sim.step().unwrap();
        assert!(cascaded.load(AOrdering::SeqCst));
    }

    #[test]
    fn test_des_actions_use_current_runtime() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let spawned = Arc::new(AtomicBool::new(false));

        let s = spawned.clone();
        handle.schedule_at(Duration::from_secs(1), move || {
            // current_runtime() should work inside DES actions
            let rt = crate::current_runtime().expect("should have runtime context");
            let s2 = s.clone();
            rt.spawn_async(async move {
                s2.store(true, AOrdering::SeqCst);
            });
        });

        sim.step().unwrap();
        assert!(spawned.load(AOrdering::SeqCst));
    }

    #[test]
    fn test_unsupported_apis_panic() {
        let sim = SimulationRuntime::new().unwrap();

        // rayon_pool() should still panic — no simulated alternative
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            sim.loom().rayon_pool();
        }));
        assert!(result.is_err());

        // install() runs inline in sim mode — no panic
        let val = sim.loom().install(|| 42);
        assert_eq!(val, 42);
    }

    #[test]
    fn test_livelock_detection() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();

        // Create an infinite zero-delay cascade
        fn infinite_cascade(h: SimHandle) {
            let h2 = h.clone();
            h.schedule(Duration::ZERO, move || {
                infinite_cascade(h2);
            });
        }

        let h = handle.clone();
        handle.schedule_at(Duration::from_secs(1), move || {
            infinite_cascade(h);
        });

        let result = sim.step();
        assert!(matches!(result, Err(crate::LoomError::SimLivelock { .. })));
    }

    #[test]
    fn test_step_until() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let count = Arc::new(AtomicUsize::new(0));

        for i in 1..=5 {
            let c = count.clone();
            handle.schedule_at(Duration::from_secs(i), move || {
                c.fetch_add(1, AOrdering::SeqCst);
            });
        }

        sim.step_until(Duration::from_secs(3)).unwrap();
        assert_eq!(count.load(AOrdering::SeqCst), 3);
        assert_eq!(sim.now(), Duration::from_secs(3));
    }

    #[test]
    fn test_step_until_drains_pending_tasks_first() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let completed = Arc::new(AtomicBool::new(false));

        let c = completed.clone();
        sim.loom().spawn_async(async move {
            handle.delay(Duration::from_millis(10)).await;
            c.store(true, AOrdering::SeqCst);
        });

        sim.step_until(Duration::from_millis(50)).unwrap();
        assert!(completed.load(AOrdering::SeqCst));
        assert_eq!(sim.now(), Duration::from_millis(10));
    }

    #[test]
    fn test_schedule_and_spawn_at_past_time_panic() {
        let mut sim = SimulationRuntime::new().unwrap();
        sim.run_until(Duration::from_secs(1)).unwrap();
        let handle = sim.handle();

        let schedule_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            handle.schedule_at(Duration::from_millis(500), || {});
        }));
        assert!(schedule_result.is_err());

        let spawn_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            handle.spawn_at(Duration::from_millis(500), async {});
        }));
        assert!(spawn_result.is_err());
        assert_eq!(sim.now(), Duration::from_secs(1));
    }

    #[test]
    fn test_runtime_invariant_prevents_time_rewind() {
        let mut sim = SimulationRuntime::new().unwrap();
        sim.run_until(Duration::from_secs(1)).unwrap();

        sim.queue
            .lock()
            .unwrap()
            .insert(Duration::from_millis(500), Box::new(|| {}));

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            sim.step().unwrap();
        }));
        assert!(result.is_err());
        assert_eq!(sim.now(), Duration::from_secs(1));
    }

    #[test]
    fn test_run() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let count = Arc::new(AtomicUsize::new(0));

        for i in 1..=5 {
            let c = count.clone();
            handle.schedule_at(Duration::from_secs(i), move || {
                c.fetch_add(1, AOrdering::SeqCst);
            });
        }

        let final_time = sim.run().unwrap();
        assert_eq!(count.load(AOrdering::SeqCst), 5);
        assert_eq!(final_time, Duration::from_secs(5));
    }

    #[test]
    fn test_multiple_delay_futures() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let order = Arc::new(AtomicUsize::new(0));

        // Task 1 delays 2s, task 2 delays 1s — task 2 should complete first
        let h1 = handle.clone();
        let o1 = order.clone();
        sim.loom().spawn_async(async move {
            h1.delay(Duration::from_secs(2)).await;
            let prev = o1.fetch_add(1, AOrdering::SeqCst);
            assert_eq!(prev, 1); // Should be second
        });

        let h2 = handle.clone();
        let o2 = order.clone();
        sim.loom().spawn_async(async move {
            h2.delay(Duration::from_secs(1)).await;
            let prev = o2.fetch_add(1, AOrdering::SeqCst);
            assert_eq!(prev, 0); // Should be first
        });

        sim.run().unwrap();
        assert_eq!(order.load(AOrdering::SeqCst), 2);
    }

    #[test]
    fn test_mixed_des_and_tokio_timers() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let order = Arc::new(AtomicUsize::new(0));

        // DelayFuture at T=1s
        let h = handle.clone();
        let o1 = order.clone();
        sim.loom().spawn_async(async move {
            h.delay(Duration::from_secs(1)).await;
            let prev = o1.fetch_add(1, AOrdering::SeqCst);
            assert_eq!(prev, 0);
        });

        // tokio::time::sleep at T=2s — resolved when DES advances past 2s
        let o2 = order.clone();
        sim.loom().spawn_async(async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let prev = o2.fetch_add(1, AOrdering::SeqCst);
            assert_eq!(prev, 1);
        });

        // DES action at T=3s
        let o3 = order.clone();
        handle.schedule_at(Duration::from_secs(3), move || {
            let prev = o3.fetch_add(1, AOrdering::SeqCst);
            assert_eq!(prev, 2);
        });

        sim.run().unwrap();
        assert_eq!(order.load(AOrdering::SeqCst), 3);
    }

    #[test]
    fn test_spawn_after_with_inner_delay() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let completed = Arc::new(AtomicBool::new(false));

        // spawn_after: async task starts at T=10ms, then delays 5ms internally
        let h = handle.clone();
        let c = completed.clone();
        handle.spawn_after(Duration::from_millis(10), async move {
            // We're now at T=10ms
            h.delay(Duration::from_millis(5)).await;
            // Now at T=15ms
            c.store(true, AOrdering::SeqCst);
        });

        // Step to T=10ms — spawns the async task, which registers a delay at T=15ms
        sim.step().unwrap();
        assert_eq!(sim.now(), Duration::from_millis(10));
        assert!(!completed.load(AOrdering::SeqCst));

        // Step to T=15ms — delay resolves, task completes
        sim.step().unwrap();
        assert_eq!(sim.now(), Duration::from_millis(15));
        assert!(completed.load(AOrdering::SeqCst));
    }

    #[test]
    fn test_spawn_at_with_channels() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let result = Arc::new(Mutex::new(String::new()));

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);

        // Receiver task — runs from the start
        let r = result.clone();
        sim.loom().spawn_async(async move {
            if let Some(msg) = rx.recv().await {
                *r.lock().unwrap() = msg;
            }
        });

        // Sender task — spawned at T=100ms
        handle.spawn_at(Duration::from_millis(100), async move {
            tx.send("hello from T=100ms".to_string()).await.unwrap();
        });

        sim.run().unwrap();
        assert_eq!(*result.lock().unwrap(), "hello from T=100ms");
    }

    #[test]
    fn test_run_until_no_des_events() {
        let mut sim = SimulationRuntime::new().unwrap();
        let completed = Arc::new(AtomicBool::new(false));

        // Only a tokio::time::sleep — no DES events at all
        let c = completed.clone();
        sim.loom().spawn_async(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            c.store(true, AOrdering::SeqCst);
        });

        // run_until force-advances tokio clock to 100ms, resolving the 50ms sleep
        sim.run_until(Duration::from_millis(100)).unwrap();
        assert!(completed.load(AOrdering::SeqCst));
        assert_eq!(sim.now(), Duration::from_millis(100));
    }

    #[test]
    fn test_run_until_with_des_events() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let count = Arc::new(AtomicUsize::new(0));

        for i in 1..=3 {
            let c = count.clone();
            handle.schedule_at(Duration::from_millis(i * 10), move || {
                c.fetch_add(1, AOrdering::SeqCst);
            });
        }

        // run_until processes DES events at 10,20,30 then advances to 50ms
        sim.run_until(Duration::from_millis(50)).unwrap();
        assert_eq!(count.load(AOrdering::SeqCst), 3);
        assert_eq!(sim.now(), Duration::from_millis(50));
    }

    #[test]
    fn test_run_until_flushes_same_timestamp_des_work() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let fired = Arc::new(AtomicBool::new(false));

        let f = fired.clone();
        sim.loom().spawn_async(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            handle.schedule(Duration::ZERO, move || {
                f.store(true, AOrdering::SeqCst);
            });
        });

        sim.run_until(Duration::from_millis(50)).unwrap();
        assert!(fired.load(AOrdering::SeqCst));
        assert_eq!(sim.now(), Duration::from_millis(50));
    }

    #[test]
    fn test_run_until_flushes_same_timestamp_cascade() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let count = Arc::new(AtomicUsize::new(0));

        let h = handle.clone();
        let c = count.clone();
        sim.loom().spawn_async(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;

            let h2 = h.clone();
            let c2 = c.clone();
            h.schedule(Duration::ZERO, move || {
                c2.fetch_add(1, AOrdering::SeqCst);

                let c3 = c2.clone();
                h2.schedule(Duration::ZERO, move || {
                    c3.fetch_add(1, AOrdering::SeqCst);
                });
            });
        });

        sim.run_until(Duration::from_millis(50)).unwrap();
        assert_eq!(count.load(AOrdering::SeqCst), 2);
        assert_eq!(sim.now(), Duration::from_millis(50));
    }

    #[test]
    fn test_canceled_delay_does_not_advance_time() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();
        let short_won = Arc::new(AtomicBool::new(false));

        let won = short_won.clone();
        sim.loom().spawn_async(async move {
            tokio::select! {
                _ = handle.delay(Duration::from_secs(1)) => {}
                _ = handle.delay(Duration::from_millis(10)) => {
                    won.store(true, AOrdering::SeqCst);
                }
            }
        });

        let final_time = sim.run().unwrap();
        assert!(short_won.load(AOrdering::SeqCst));
        assert_eq!(final_time, Duration::from_millis(10));
    }

    #[test]
    fn test_simulated_spawn_compute() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();

        let result = Arc::new(Mutex::new(0u64));

        let r = result.clone();
        let h = handle.clone();
        sim.loom().spawn_async(async move {
            let before = h.now();
            // spawn_compute runs inline and delays by measured wall time
            let val = crate::current_runtime()
                .unwrap()
                .spawn_compute(|| {
                    // Do some trivial work
                    (0..1000u64).sum::<u64>()
                })
                .await;
            *r.lock().unwrap() = val;
            let after = h.now();
            // Virtual time should have advanced (at least 0, maybe 1+ μs)
            assert!(after >= before);
        });

        // Need a DES event to drive time past the delay future
        // The spawn_compute delay registers a DelayFuture in the DES queue
        sim.run().unwrap();
        assert_eq!(*result.lock().unwrap(), 499500);
    }

    #[test]
    fn test_simulated_install() {
        let sim = SimulationRuntime::new().unwrap();

        // install() runs inline in sim mode — no panic, no delay
        let result = sim.loom().block_on(async {
            let rt = crate::current_runtime().unwrap();
            rt.install(|| {
                use rayon::prelude::*;
                (0..100).into_par_iter().sum::<i32>()
            })
        });
        assert_eq!(result, 4950);
    }

    // =========================================================================
    // run_until_complete Tests
    // =========================================================================

    #[test]
    fn test_run_until_complete_basic() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();

        let result = sim
            .run_until_complete(async move {
                handle.delay(Duration::from_secs(5)).await;
                42
            })
            .unwrap();

        assert_eq!(result, 42);
        assert_eq!(sim.now(), Duration::from_secs(5));
    }

    #[test]
    fn test_run_until_complete_immediate() {
        let mut sim = SimulationRuntime::new().unwrap();

        let result = sim.run_until_complete(async { "hello" }).unwrap();

        assert_eq!(result, "hello");
        assert_eq!(sim.now(), Duration::ZERO);
    }

    #[test]
    fn test_run_until_complete_stall() {
        let mut sim = SimulationRuntime::new().unwrap();

        // Task waits on a channel that nobody sends to — DES queue will empty
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
        std::mem::forget(tx); // keep channel open so recv suspends forever

        let result = sim.run_until_complete(async move {
            rx.recv().await;
        });

        assert!(
            matches!(result, Err(crate::LoomError::SimStall { .. })),
            "expected SimStall, got: {result:?}"
        );
    }

    #[test]
    fn test_run_until_complete_stall_aborts_monitored_task() {
        let mut sim = SimulationRuntime::new().unwrap();
        let dropped = Arc::new(AtomicBool::new(false));

        // Task waits on a channel that nobody sends to — DES queue will empty
        let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(1);
        std::mem::forget(tx); // keep channel open so recv suspends forever

        let drop_flag = dropped.clone();
        let result = sim.run_until_complete(async move {
            let _guard = DropFlag(drop_flag);
            rx.recv().await;
        });

        assert!(
            matches!(result, Err(crate::LoomError::SimStall { .. })),
            "expected SimStall, got: {result:?}"
        );
        assert!(dropped.load(AOrdering::SeqCst));
        assert!(sim.loom().task_tracker().is_empty());
    }

    #[test]
    #[should_panic(expected = "run_until_complete panic")]
    fn test_run_until_complete_panic_propagates() {
        let mut sim = SimulationRuntime::new().unwrap();

        let _ = sim.run_until_complete(async {
            panic!("run_until_complete panic");
        });
    }

    #[test]
    fn test_run_until_complete_with_subtasks() {
        let mut sim = SimulationRuntime::new().unwrap();
        let handle = sim.handle();

        let result = sim
            .run_until_complete(async move {
                let (tx, mut rx) = tokio::sync::mpsc::channel(4);

                // Spawn subtasks that send results after different delays
                for i in 0..3u32 {
                    let h = handle.clone();
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        h.delay(Duration::from_millis((i as u64 + 1) * 10)).await;
                        tx.send(i).await.ok();
                    });
                }
                drop(tx);

                let mut values = Vec::new();
                while let Some(v) = rx.recv().await {
                    values.push(v);
                }
                values
            })
            .unwrap();

        assert_eq!(result, vec![0, 1, 2]);
        assert_eq!(sim.now(), Duration::from_millis(30));
    }
}
