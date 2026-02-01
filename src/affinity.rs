//! Thread affinity utilities for CPU pinning.
//!
//! This module provides utilities for pinning threads to specific CPUs
//! using the `core_affinity` crate.

use crate::error::{LoomError, Result};
use core_affinity::CoreId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, warn};

/// Pin the current thread to the specified CPU.
///
/// # Errors
///
/// Returns `LoomError::AffinityFailed` if the thread cannot be pinned.
///
/// # Examples
///
/// ```ignore
/// use loom_rs::affinity::pin_to_cpu;
///
/// // Pin current thread to CPU 0
/// pin_to_cpu(0).expect("failed to pin thread");
/// ```
pub fn pin_to_cpu(cpu_id: usize) -> Result<()> {
    let core_id = CoreId { id: cpu_id };
    if core_affinity::set_for_current(core_id) {
        debug!(cpu_id, "pinned thread to CPU");
        Ok(())
    } else {
        warn!(cpu_id, "failed to pin thread to CPU");
        Err(LoomError::AffinityFailed(cpu_id))
    }
}

/// A CPU allocator that distributes CPUs to threads in round-robin fashion.
///
/// This is used internally by the runtime to assign CPUs to tokio and rayon
/// worker threads.
#[derive(Debug)]
pub struct CpuAllocator {
    cpus: Vec<usize>,
    next: AtomicUsize,
}

impl CpuAllocator {
    /// Create a new CPU allocator with the given CPU set.
    ///
    /// # Panics
    ///
    /// Panics if `cpus` is empty.
    pub fn new(cpus: Vec<usize>) -> Self {
        assert!(!cpus.is_empty(), "CPU allocator requires at least one CPU");
        Self {
            cpus,
            next: AtomicUsize::new(0),
        }
    }

    /// Allocate the next CPU in round-robin order.
    ///
    /// This method is thread-safe and can be called from multiple threads
    /// simultaneously.
    pub fn allocate(&self) -> usize {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.cpus.len();
        self.cpus[index]
    }

    /// Get the total number of CPUs in the allocator.
    pub fn len(&self) -> usize {
        self.cpus.len()
    }

    /// Check if the allocator is empty.
    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    /// Get a reference to the underlying CPU list.
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
}

/// Create a thread start handler that pins threads to CPUs.
///
/// Returns a closure suitable for use with tokio's `on_thread_start` or
/// rayon's `start_handler`.
///
/// # Arguments
///
/// * `allocator` - Arc-wrapped CPU allocator to distribute CPUs
/// * `prefix` - Thread name prefix for logging
pub fn make_pin_handler(
    allocator: Arc<CpuAllocator>,
    prefix: Arc<str>,
) -> impl Fn() + Send + Sync + Clone + 'static {
    move || {
        let cpu_id = allocator.allocate();
        if let Err(e) = pin_to_cpu(cpu_id) {
            warn!(%e, %prefix, cpu_id, "failed to pin thread");
        }
    }
}

/// Create a rayon-specific thread start handler that pins threads to CPUs.
///
/// Rayon's `start_handler` receives the thread index as an argument.
///
/// # Arguments
///
/// * `allocator` - Arc-wrapped CPU allocator to distribute CPUs
/// * `prefix` - Thread name prefix for logging
pub fn make_rayon_pin_handler(
    allocator: Arc<CpuAllocator>,
    prefix: Arc<str>,
) -> impl Fn(usize) + Send + Sync + Clone + 'static {
    move |thread_index: usize| {
        let cpu_id = allocator.allocate();
        debug!(thread_index, cpu_id, %prefix, "rayon thread starting");
        if let Err(e) = pin_to_cpu(cpu_id) {
            warn!(%e, %prefix, cpu_id, thread_index, "failed to pin rayon thread");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_allocator_round_robin() {
        let allocator = CpuAllocator::new(vec![0, 2, 4, 6]);

        assert_eq!(allocator.allocate(), 0);
        assert_eq!(allocator.allocate(), 2);
        assert_eq!(allocator.allocate(), 4);
        assert_eq!(allocator.allocate(), 6);
        // Wraps around
        assert_eq!(allocator.allocate(), 0);
        assert_eq!(allocator.allocate(), 2);
    }

    #[test]
    fn test_cpu_allocator_single_cpu() {
        let allocator = CpuAllocator::new(vec![5]);

        // Always returns the same CPU
        assert_eq!(allocator.allocate(), 5);
        assert_eq!(allocator.allocate(), 5);
        assert_eq!(allocator.allocate(), 5);
    }

    #[test]
    fn test_cpu_allocator_len() {
        let allocator = CpuAllocator::new(vec![0, 1, 2, 3]);
        assert_eq!(allocator.len(), 4);
        assert!(!allocator.is_empty());
    }

    #[test]
    #[should_panic(expected = "CPU allocator requires at least one CPU")]
    fn test_cpu_allocator_empty_panics() {
        let _ = CpuAllocator::new(vec![]);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_pin_to_cpu() {
        // Pin to CPU 0 should generally work
        let result = pin_to_cpu(0);
        assert!(result.is_ok());
    }

    // Note: We cannot test pinning to an invalid CPU because core_affinity
    // panics on Linux when given a CPU ID outside the valid range (due to
    // bounds checking in the libc sched_setaffinity wrapper).

    #[test]
    #[cfg(target_os = "linux")]
    fn test_make_pin_handler() {
        let allocator = Arc::new(CpuAllocator::new(vec![0]));
        let handler = make_pin_handler(allocator, "test".into());

        // Should be callable
        handler();
    }
}
