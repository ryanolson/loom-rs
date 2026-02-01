//! Runtime metrics collection for MAB scheduling decisions.
//!
//! This module provides lightweight atomic counters for tracking:
//! - In-flight async tasks
//! - Spawn rate (tasks per second)
//! - Rayon queue depth estimates

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use super::types::Context;

/// Runtime metrics for MAB scheduling decisions.
///
/// All operations are lock-free using atomic primitives.
/// Overhead is minimal (~1-2ns per counter update).
pub struct RuntimeMetrics {
    /// Number of tracked async tasks currently in flight
    inflight_tasks: AtomicU32,

    /// Total number of spawned tasks (for rate calculation)
    spawn_count: AtomicU64,

    /// Timestamp of spawn window start (for rate calculation)
    spawn_window_start: AtomicU64,

    /// Number of tasks submitted to rayon
    rayon_submitted: AtomicU32,

    /// Number of tasks that have started on rayon
    rayon_started: AtomicU32,

    /// Epoch instant for timestamp calculations
    #[allow(dead_code)]
    epoch: Instant,
}

impl RuntimeMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            inflight_tasks: AtomicU32::new(0),
            spawn_count: AtomicU64::new(0),
            spawn_window_start: AtomicU64::new(0),
            rayon_submitted: AtomicU32::new(0),
            rayon_started: AtomicU32::new(0),
            epoch: now,
        }
    }

    /// Record that a tracked async task has started.
    #[inline]
    pub fn task_started(&self) {
        self.inflight_tasks.fetch_add(1, Ordering::Relaxed);
        self.spawn_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a tracked async task has completed.
    #[inline]
    pub fn task_completed(&self) {
        self.inflight_tasks.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record that a task has been submitted to rayon.
    #[inline]
    pub fn rayon_submitted(&self) {
        self.rayon_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a task has started executing on rayon.
    #[inline]
    pub fn rayon_started(&self) {
        self.rayon_started.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a rayon task has completed.
    /// This balances both submitted and started counters.
    #[inline]
    pub fn rayon_completed(&self) {
        // Decrement both counters to maintain queue depth accuracy
        self.rayon_submitted.fetch_sub(1, Ordering::Relaxed);
        self.rayon_started.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get the current number of in-flight tasks.
    #[inline]
    pub fn inflight_tasks(&self) -> u32 {
        self.inflight_tasks.load(Ordering::Relaxed)
    }

    /// Get the estimated rayon queue depth.
    ///
    /// This is the difference between submitted and started tasks.
    #[inline]
    pub fn rayon_queue_depth(&self) -> u32 {
        let submitted = self.rayon_submitted.load(Ordering::Relaxed);
        let started = self.rayon_started.load(Ordering::Relaxed);
        submitted.saturating_sub(started)
    }

    /// Calculate the spawn rate (tasks per second).
    ///
    /// Uses a rolling window approach with atomic updates.
    pub fn spawn_rate_per_s(&self) -> f32 {
        let now_micros = self.epoch.elapsed().as_micros() as u64;
        let window_start = self.spawn_window_start.load(Ordering::Relaxed);
        let count = self.spawn_count.load(Ordering::Relaxed);

        // Calculate elapsed time since window start
        let elapsed_micros = now_micros.saturating_sub(window_start);

        if elapsed_micros < 1000 {
            // Less than 1ms, not enough data
            return 0.0;
        }

        // Reset window if it's been more than 1 second
        if elapsed_micros > 1_000_000 {
            // Try to reset the window atomically
            if self
                .spawn_window_start
                .compare_exchange(
                    window_start,
                    now_micros,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                self.spawn_count.store(0, Ordering::Relaxed);
            }
            // Use the data we have for this calculation
            let rate = (count as f64 * 1_000_000.0) / (elapsed_micros as f64);
            return rate as f32;
        }

        // Normal case: calculate rate from current window
        let rate = (count as f64 * 1_000_000.0) / (elapsed_micros as f64);
        rate as f32
    }

    /// Collect current metrics into a Context for scheduling decisions.
    pub fn collect(&self, tokio_workers: u32, rayon_threads: u32) -> Context {
        Context {
            tokio_workers,
            inflight_tasks: self.inflight_tasks(),
            spawn_rate_per_s: self.spawn_rate_per_s(),
            rayon_threads,
            rayon_queue_depth: self.rayon_queue_depth(),
        }
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_tracking() {
        let metrics = RuntimeMetrics::new();

        assert_eq!(metrics.inflight_tasks(), 0);

        metrics.task_started();
        metrics.task_started();
        assert_eq!(metrics.inflight_tasks(), 2);

        metrics.task_completed();
        assert_eq!(metrics.inflight_tasks(), 1);

        metrics.task_completed();
        assert_eq!(metrics.inflight_tasks(), 0);
    }

    #[test]
    fn test_rayon_queue_depth() {
        let metrics = RuntimeMetrics::new();

        assert_eq!(metrics.rayon_queue_depth(), 0);

        // Submit 3 tasks
        metrics.rayon_submitted();
        metrics.rayon_submitted();
        metrics.rayon_submitted();
        assert_eq!(metrics.rayon_queue_depth(), 3);

        // 2 tasks start
        metrics.rayon_started();
        metrics.rayon_started();
        assert_eq!(metrics.rayon_queue_depth(), 1);

        // 1 more starts (all running)
        metrics.rayon_started();
        assert_eq!(metrics.rayon_queue_depth(), 0);

        // Complete tasks
        metrics.rayon_completed();
        metrics.rayon_completed();
        metrics.rayon_completed();
        assert_eq!(metrics.rayon_queue_depth(), 0);
    }

    #[test]
    fn test_collect() {
        let metrics = RuntimeMetrics::new();
        metrics.task_started();
        metrics.task_started();
        metrics.rayon_submitted();

        let ctx = metrics.collect(4, 8);

        assert_eq!(ctx.tokio_workers, 4);
        assert_eq!(ctx.rayon_threads, 8);
        assert_eq!(ctx.inflight_tasks, 2);
        assert_eq!(ctx.rayon_queue_depth, 1);
    }
}
