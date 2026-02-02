//! Prometheus metrics for loom-rs runtime observability.
//!
//! This module provides zero-overhead metrics collection using Prometheus counters
//! and gauges. Counters work standalone without a Registry - registration is only
//! needed for exposition (scraping).
//!
//! # Design Principles
//!
//! - **Always-on**: Counters are always incremented (zero overhead - just atomic ops)
//! - **Registry optional**: Users can optionally provide a Registry for exposition
//! - **Cached access**: Direct field access, no HashMap lookups in hot paths
//! - **Configurable prefix**: Metric names use `{prefix}_` prefix (default: "loom")
//!
//! # Usage
//!
//! ```ignore
//! // Pattern 1: No exposition (counters still work internally)
//! let runtime = LoomBuilder::new()
//!     .build()?;
//!
//! // Pattern 2: External registry for scraping
//! let registry = prometheus::Registry::new();
//! let runtime = LoomBuilder::new()
//!     .prometheus_registry(registry.clone())
//!     .build()?;
//!
//! // Later: expose via HTTP endpoint
//! let encoder = TextEncoder::new();
//! let metric_families = registry.gather();
//! encoder.encode(&metric_families, &mut buffer)?;
//! ```
//!
//! # Performance
//!
//! | Operation | Overhead |
//! |-----------|----------|
//! | Counter increment | ~1-2ns (atomic fetch_add) |
//! | Gauge set | ~1-2ns (atomic store) |
//! | Registry lookup | **NEVER** (direct field access) |
//! | Without registry | Same cost (counters work standalone) |

use prometheus::{Gauge, IntCounter, IntGauge, Opts, Registry};

/// Prometheus metrics for loom-rs runtime.
///
/// Counters are always incremented (zero overhead atomic ops).
/// Registration to a Registry is optional - only needed for exposition.
#[derive(Clone)]
pub struct LoomMetrics {
    // === Gauges (current values) ===
    /// Number of tracked async tasks currently in flight
    pub inflight_tasks: IntGauge,

    /// Estimated Rayon task queue depth
    pub rayon_queue_depth: IntGauge,

    /// Task spawn rate per second
    pub spawn_rate: Gauge,

    /// MAB pressure index (0-10)
    pub pressure_index: Gauge,

    // === Counters (cumulative) ===
    /// Total tasks spawned
    pub total_spawns: IntCounter,

    /// Starvation events detected (pressure > 3.0 or guardrail triggered)
    pub starvation_events: IntCounter,

    /// GR1 hard threshold activations (EMA > t_block_hard_us)
    pub gr1_activations: IntCounter,

    /// GR2 pressure threshold activations
    pub gr2_activations: IntCounter,

    /// GR3 strike suppression activations
    pub gr3_activations: IntCounter,

    /// MAB inline decisions
    pub inline_decisions: IntCounter,

    /// MAB offload decisions
    pub offload_decisions: IntCounter,
}

impl Default for LoomMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl LoomMetrics {
    /// Create metrics with default prefix "loom".
    ///
    /// Counters work fine - just not exposed until registered.
    pub fn new() -> Self {
        Self::with_prefix("loom")
    }

    /// Create metrics with a custom prefix.
    ///
    /// Metric names will be `{prefix}_inflight_tasks`, `{prefix}_total_spawns`, etc.
    /// The prefix is sanitized to be a valid Prometheus metric name: hyphens and other
    /// invalid characters are replaced with underscores.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let metrics = LoomMetrics::with_prefix("myapp");
    /// // Creates metrics like: myapp_inflight_tasks, myapp_total_spawns, etc.
    ///
    /// let metrics = LoomMetrics::with_prefix("my-app");
    /// // Creates metrics like: my_app_inflight_tasks, my_app_total_spawns, etc.
    /// ```
    pub fn with_prefix(prefix: &str) -> Self {
        // Sanitize prefix for Prometheus: replace invalid chars with underscores
        // Valid chars: [a-zA-Z_:] for first char, [a-zA-Z0-9_:] for rest
        let prefix: String = prefix
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    if c.is_ascii_alphabetic() || c == '_' || c == ':' {
                        c
                    } else {
                        '_'
                    }
                } else if c.is_ascii_alphanumeric() || c == '_' || c == ':' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Self {
            // Gauges
            inflight_tasks: IntGauge::with_opts(Opts::new(
                format!("{}_inflight_tasks", prefix),
                "Tracked async tasks in flight",
            ))
            .expect("metric creation should not fail"),

            rayon_queue_depth: IntGauge::with_opts(Opts::new(
                format!("{}_rayon_queue_depth", prefix),
                "Rayon task queue depth",
            ))
            .expect("metric creation should not fail"),

            spawn_rate: Gauge::with_opts(Opts::new(
                format!("{}_spawn_rate", prefix),
                "Task spawn rate per second",
            ))
            .expect("metric creation should not fail"),

            pressure_index: Gauge::with_opts(Opts::new(
                format!("{}_pressure_index", prefix),
                "MAB pressure index (0-10)",
            ))
            .expect("metric creation should not fail"),

            // Counters
            total_spawns: IntCounter::with_opts(Opts::new(
                format!("{}_total_spawns", prefix),
                "Total tasks spawned",
            ))
            .expect("metric creation should not fail"),

            starvation_events: IntCounter::with_opts(Opts::new(
                format!("{}_starvation_events", prefix),
                "Starvation events detected",
            ))
            .expect("metric creation should not fail"),

            gr1_activations: IntCounter::with_opts(Opts::new(
                format!("{}_gr1_activations", prefix),
                "GR1 hard threshold activations",
            ))
            .expect("metric creation should not fail"),

            gr2_activations: IntCounter::with_opts(Opts::new(
                format!("{}_gr2_activations", prefix),
                "GR2 pressure threshold activations",
            ))
            .expect("metric creation should not fail"),

            gr3_activations: IntCounter::with_opts(Opts::new(
                format!("{}_gr3_activations", prefix),
                "GR3 strike suppression activations",
            ))
            .expect("metric creation should not fail"),

            inline_decisions: IntCounter::with_opts(Opts::new(
                format!("{}_inline_decisions", prefix),
                "MAB inline decisions",
            ))
            .expect("metric creation should not fail"),

            offload_decisions: IntCounter::with_opts(Opts::new(
                format!("{}_offload_decisions", prefix),
                "MAB offload decisions",
            ))
            .expect("metric creation should not fail"),
        }
    }

    /// Register all metrics with a Registry for exposition.
    ///
    /// Call this if you want Prometheus scraping.
    ///
    /// # Errors
    ///
    /// Returns an error if any metric fails to register (e.g., duplicate names).
    pub fn register(&self, registry: &Registry) -> Result<(), prometheus::Error> {
        registry.register(Box::new(self.inflight_tasks.clone()))?;
        registry.register(Box::new(self.rayon_queue_depth.clone()))?;
        registry.register(Box::new(self.spawn_rate.clone()))?;
        registry.register(Box::new(self.pressure_index.clone()))?;
        registry.register(Box::new(self.total_spawns.clone()))?;
        registry.register(Box::new(self.starvation_events.clone()))?;
        registry.register(Box::new(self.gr1_activations.clone()))?;
        registry.register(Box::new(self.gr2_activations.clone()))?;
        registry.register(Box::new(self.gr3_activations.clone()))?;
        registry.register(Box::new(self.inline_decisions.clone()))?;
        registry.register(Box::new(self.offload_decisions.clone()))?;
        Ok(())
    }

    /// Record a MAB decision.
    #[inline]
    pub fn record_decision(&self, inline: bool) {
        if inline {
            self.inline_decisions.inc();
        } else {
            self.offload_decisions.inc();
        }
    }

    /// Record a guardrail activation.
    #[inline]
    pub fn record_guardrail(&self, guardrail: &str) {
        match guardrail {
            "GR1" => self.gr1_activations.inc(),
            "GR2" => self.gr2_activations.inc(),
            "GR3" => self.gr3_activations.inc(),
            _ => {}
        }
    }

    /// Record a starvation event.
    #[inline]
    pub fn record_starvation(&self) {
        self.starvation_events.inc();
    }

    /// Update the pressure index gauge.
    #[inline]
    pub fn set_pressure_index(&self, pressure: f64) {
        self.pressure_index.set(pressure);
    }

    /// Update the inflight tasks gauge.
    #[inline]
    pub fn set_inflight_tasks(&self, count: i64) {
        self.inflight_tasks.set(count);
    }

    /// Update the spawn rate gauge.
    #[inline]
    pub fn set_spawn_rate(&self, rate: f64) {
        self.spawn_rate.set(rate);
    }

    /// Increment total spawns counter.
    #[inline]
    pub fn inc_total_spawns(&self) {
        self.total_spawns.inc();
    }
}

impl std::fmt::Debug for LoomMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoomMetrics")
            .field("inflight_tasks", &self.inflight_tasks.get())
            .field("total_spawns", &self.total_spawns.get())
            .field("inline_decisions", &self.inline_decisions.get())
            .field("offload_decisions", &self.offload_decisions.get())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = LoomMetrics::new();
        assert_eq!(metrics.inflight_tasks.get(), 0);
        assert_eq!(metrics.total_spawns.get(), 0);
    }

    #[test]
    fn test_counter_increments() {
        let metrics = LoomMetrics::new();

        metrics.inc_total_spawns();
        assert_eq!(metrics.total_spawns.get(), 1);

        metrics.record_decision(true);
        assert_eq!(metrics.inline_decisions.get(), 1);
        assert_eq!(metrics.offload_decisions.get(), 0);

        metrics.record_decision(false);
        assert_eq!(metrics.inline_decisions.get(), 1);
        assert_eq!(metrics.offload_decisions.get(), 1);
    }

    #[test]
    fn test_guardrail_recording() {
        let metrics = LoomMetrics::new();

        metrics.record_guardrail("GR1");
        assert_eq!(metrics.gr1_activations.get(), 1);

        metrics.record_guardrail("GR2");
        assert_eq!(metrics.gr2_activations.get(), 1);

        metrics.record_guardrail("GR3");
        assert_eq!(metrics.gr3_activations.get(), 1);

        // Unknown guardrail should be ignored
        metrics.record_guardrail("UNKNOWN");
        assert_eq!(metrics.gr1_activations.get(), 1);
    }

    #[test]
    fn test_gauge_updates() {
        let metrics = LoomMetrics::new();

        metrics.set_pressure_index(5.5);
        assert!((metrics.pressure_index.get() - 5.5).abs() < 0.001);

        metrics.set_inflight_tasks(42);
        assert_eq!(metrics.inflight_tasks.get(), 42);

        metrics.set_spawn_rate(1000.0);
        assert!((metrics.spawn_rate.get() - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_registry_integration() {
        let metrics = LoomMetrics::new();
        let registry = Registry::new();

        metrics
            .register(&registry)
            .expect("registration should succeed");

        // Increment a counter
        metrics.inc_total_spawns();

        // Gather metrics from registry
        let families = registry.gather();
        assert!(!families.is_empty());

        // Find our metric
        let total_spawns = families
            .iter()
            .find(|f| f.get_name() == "loom_total_spawns");
        assert!(total_spawns.is_some());
    }

    #[test]
    fn test_custom_prefix() {
        let metrics = LoomMetrics::with_prefix("myapp");
        let registry = Registry::new();

        metrics
            .register(&registry)
            .expect("registration should succeed");

        metrics.inc_total_spawns();

        let families = registry.gather();
        // Find metric with custom prefix
        let total_spawns = families
            .iter()
            .find(|f| f.get_name() == "myapp_total_spawns");
        assert!(total_spawns.is_some());

        // Should not find metric with default prefix
        let loom_total_spawns = families
            .iter()
            .find(|f| f.get_name() == "loom_total_spawns");
        assert!(loom_total_spawns.is_none());
    }

    #[test]
    fn test_metrics_clone() {
        let metrics = LoomMetrics::new();
        metrics.inc_total_spawns();

        let cloned = metrics.clone();
        assert_eq!(cloned.total_spawns.get(), 1);

        // Cloned metrics share the same underlying counters
        cloned.inc_total_spawns();
        assert_eq!(metrics.total_spawns.get(), 2);
    }
}
