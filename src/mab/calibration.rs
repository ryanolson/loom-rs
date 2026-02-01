//! Optional startup calibration to measure offload overhead.
//!
//! Calibration runs during runtime construction (if enabled) and measures
//! the fixed overhead of offloading work to rayon. This helps the MAB make
//! better decisions for borderline workloads.
//!
//! # Default Behavior
//!
//! Calibration is **disabled by default** for fast unit test startup.
//! Enable it via `LoomBuilder::calibrate(true)` for production use.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::bridge::RayonTask;

/// Results from runtime calibration.
#[derive(Clone, Debug)]
pub struct CalibrationResult {
    /// Measured overhead of offloading to rayon (spawn + queue + wake) in microseconds.
    /// This is the median round-trip time for a no-op task.
    pub offload_overhead_us: f64,

    /// P50 (median) of measured samples
    pub p50_us: f64,

    /// P99 of measured samples (useful for understanding tail latency)
    pub p99_us: f64,
}

/// Configuration for the calibration phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Whether to run calibration at startup.
    /// Default: false (for fast unit test startup)
    #[serde(default)]
    pub enabled: bool,

    /// Number of warmup iterations before measuring.
    /// Default: 100
    #[serde(default = "default_warmup_iterations")]
    pub warmup_iterations: usize,

    /// Number of measurement samples.
    /// Default: 1000
    #[serde(default = "default_sample_count")]
    pub sample_count: usize,
}

fn default_warmup_iterations() -> usize {
    100
}

fn default_sample_count() -> usize {
    1000
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            warmup_iterations: default_warmup_iterations(),
            sample_count: default_sample_count(),
        }
    }
}

impl CalibrationConfig {
    /// Create a new calibration config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable calibration.
    pub fn enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Set the number of warmup iterations.
    pub fn warmup_iterations(mut self, count: usize) -> Self {
        self.warmup_iterations = count;
        self
    }

    /// Set the number of measurement samples.
    pub fn sample_count(mut self, count: usize) -> Self {
        self.sample_count = count;
        self
    }
}

/// Run calibration to measure offload overhead.
///
/// Should be called during LoomRuntime construction if enabled.
/// This is an async function that must be run within the tokio context.
pub async fn calibrate(
    rayon_pool: &rayon::ThreadPool,
    config: &CalibrationConfig,
) -> CalibrationResult {
    // Warmup: let thread pools settle, populate caches
    for _ in 0..config.warmup_iterations {
        let (task, completion) = RayonTask::<()>::new();
        rayon_pool.spawn(move || {
            completion.complete(());
        });
        task.await;
    }

    // Measure offload overhead (no-op task round-trip)
    let mut samples = Vec::with_capacity(config.sample_count);
    for _ in 0..config.sample_count {
        let start = Instant::now();
        let (task, completion) = RayonTask::<()>::new();
        rayon_pool.spawn(move || {
            // Use black_box to prevent the compiler from optimizing away the task
            std::hint::black_box(());
            completion.complete(());
        });
        task.await;
        samples.push(start.elapsed().as_nanos() as f64 / 1000.0);
    }

    // Compute statistics
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_us = percentile_sorted(&samples, 50.0);
    let p99_us = percentile_sorted(&samples, 99.0);

    CalibrationResult {
        offload_overhead_us: p50_us, // Use median as the representative overhead
        p50_us,
        p99_us,
    }
}

/// Calculate a percentile from a sorted slice.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_config_defaults() {
        let config = CalibrationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.warmup_iterations, 100);
        assert_eq!(config.sample_count, 1000);
    }

    #[test]
    fn test_calibration_config_builder() {
        let config = CalibrationConfig::new()
            .enabled()
            .warmup_iterations(50)
            .sample_count(500);

        assert!(config.enabled);
        assert_eq!(config.warmup_iterations, 50);
        assert_eq!(config.sample_count, 500);
    }

    #[test]
    fn test_percentile_sorted() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

        assert!((percentile_sorted(&data, 0.0) - 0.0).abs() < 0.5);
        assert!((percentile_sorted(&data, 50.0) - 50.0).abs() < 0.5);
        assert!((percentile_sorted(&data, 100.0) - 99.0).abs() < 0.5);
    }

    #[test]
    fn test_percentile_sorted_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(percentile_sorted(&data, 50.0), 0.0);
    }

    #[test]
    fn test_calibration_config_serialization() {
        let config = CalibrationConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CalibrationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.enabled, config.enabled);
        assert_eq!(parsed.warmup_iterations, config.warmup_iterations);
    }
}
