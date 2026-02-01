//! Metrics collection for loom-rs runtime observability.
//!
//! This module provides Prometheus-compatible metrics for monitoring the loom runtime.
//! Metrics are always collected (zero overhead atomic operations) and can optionally
//! be exposed via a Prometheus registry for scraping.
//!
//! # Architecture
//!
//! The metrics system is designed for zero overhead in the common case:
//!
//! - **Counters**: Atomic increments (~1-2ns)
//! - **Gauges**: Atomic stores (~1-2ns)
//! - **No lookups**: Direct field access to all metrics
//! - **Registry optional**: Works without exposition
//!
//! # Available Metrics
//!
//! ## Gauges (current values)
//!
//! - `loom_inflight_tasks` - Tracked async tasks in flight
//! - `loom_rayon_queue_depth` - Rayon task queue depth
//! - `loom_spawn_rate` - Task spawn rate per second
//! - `loom_pressure_index` - MAB pressure index (0-10)
//!
//! ## Counters (cumulative)
//!
//! - `loom_total_spawns` - Total tasks spawned
//! - `loom_starvation_events` - Starvation events detected
//! - `loom_gr1_activations` - GR1 hard threshold activations
//! - `loom_gr2_activations` - GR2 pressure threshold activations
//! - `loom_gr3_activations` - GR3 strike suppression activations
//! - `loom_inline_decisions` - MAB inline decisions
//! - `loom_offload_decisions` - MAB offload decisions
//!
//! # Example
//!
//! ```ignore
//! use prometheus::Registry;
//! use loom_rs::LoomBuilder;
//!
//! // Create a registry for exposition
//! let registry = Registry::new();
//!
//! // Build runtime with metrics registered
//! let runtime = LoomBuilder::new()
//!     .prometheus_registry(registry.clone())
//!     .build()?;
//!
//! // Later: expose metrics via HTTP
//! use prometheus::TextEncoder;
//! let encoder = TextEncoder::new();
//! let mut buffer = Vec::new();
//! encoder.encode(&registry.gather(), &mut buffer)?;
//! ```

mod prometheus;

pub use prometheus::LoomMetrics;
