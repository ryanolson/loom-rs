//! Configuration knobs for the MAB scheduler.
//!
//! All knobs have sensible defaults tuned for typical workloads.
//! Most users won't need to modify these values.

use serde::{Deserialize, Serialize};

use super::calibration::CalibrationResult;

/// Configuration knobs for the Multi-Armed Bandit scheduler.
///
/// These control the decision-making logic, guardrails, and learning rates.
/// Defaults are tuned for typical server workloads.
///
/// # Cost Model
///
/// The scheduler models the "adjusted cost" of each arm as:
/// ```text
/// adjusted_cost = fn_time_us + k_starve * pressure_index * fn_time_us
/// ```
///
/// Where `pressure_index` measures how stressed Tokio workers are.
///
/// # Guardrails
///
/// Four guardrails prevent Tokio starvation:
/// - **GR0**: Single-worker protection (very conservative when tokio_workers=1)
/// - **GR1**: Hard blocking threshold (never inline if fn > t_block_hard_us)
/// - **GR2**: Pressure-sensitive threshold (tighter inline limit under pressure)
/// - **GR3**: Strike suppression (suppress inline after repeated slow executions)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MabKnobs {
    // === Cost composition ===
    /// Weight for starvation cost component.
    /// Higher values penalize inline execution more when Tokio is under pressure.
    /// Default: 0.15
    pub k_starve: f64,

    // === Pressure index weights ===
    /// Weight for inflight task count in pressure calculation.
    /// Default: 0.7
    pub w_inflight: f64,

    /// Weight for spawn rate in pressure calculation.
    /// Default: 0.3
    pub w_spawn: f64,

    /// Maximum pressure index value (prevents runaway).
    /// Default: 10.0
    pub pressure_clip: f64,

    // === Decay parameters ===
    /// Per-observation decay factor for exponential weighting.
    /// Default: 0.999653 (half-life of ~2000 observations)
    pub decay: f64,

    /// Decay factor for strike counter (per observation).
    /// Default: 0.993 (half-life of ~100 observations)
    pub strike_decay: f64,

    /// Smoothing factor for EMA of function time.
    /// Default: 0.1 (gives ~90% weight to recent 23 observations)
    pub ema_alpha: f64,

    // === Guardrail thresholds ===
    /// GR0/GR2: Threshold for "tiny" work that's safe to inline even under pressure.
    /// Default: 50.0 microseconds
    pub t_tiny_inline_us: f64,

    /// GR1: Hard threshold above which inline is never allowed.
    /// Default: 250.0 microseconds
    pub t_block_hard_us: f64,

    /// GR2: Threshold for inline under high pressure.
    /// Default: 100.0 microseconds
    pub t_inline_under_pressure_us: f64,

    /// Pressure threshold below which GR2 doesn't apply.
    /// Default: 0.5
    pub p_low: f64,

    /// Pressure threshold above which GR2 applies strictly.
    /// Default: 3.0
    pub p_high: f64,

    // === Strike suppression (GR3) ===
    /// Whether to enable strike-based suppression.
    /// Default: true
    pub enable_strikes: bool,

    /// Threshold above which an inline execution counts as a "strike".
    /// Default: 1000.0 microseconds (1ms)
    pub t_strike_us: f64,

    /// Maximum strike value that triggers suppression.
    /// Default: 1.0
    pub s_max: f64,

    // === Compute hint settings ===
    /// Effective sample count below which we're in "cold start" mode.
    /// Hints are only used during cold start.
    /// Default: 5.0
    pub hint_trust_threshold: f64,

    /// Number of forced offload explorations for High hint.
    /// Default: 3
    pub hint_exploration_count: u32,

    /// Minimum samples before trusting hint-specific EMA.
    /// Below this threshold, falls back to global `ema_fn_us`.
    /// Default: 3.0
    pub hint_min_samples: f64,

    // === Initial EMA seeds for hints ===
    /// Initial EMA estimate for Low hint (microseconds).
    /// Default: 30.0
    pub hint_low_ema_us: f64,

    /// Initial EMA estimate for Medium hint (microseconds).
    /// Default: 200.0
    pub hint_medium_ema_us: f64,

    /// Initial EMA estimate for High hint (microseconds).
    /// Default: 1000.0
    pub hint_high_ema_us: f64,

    // === Calibration-derived values ===
    /// Measured overhead of offloading to rayon (microseconds).
    /// Set by calibration, or None to use a conservative estimate.
    /// Default: None (uses ~300µs estimate)
    #[serde(default)]
    pub measured_offload_overhead_us: Option<f64>,
}

impl Default for MabKnobs {
    fn default() -> Self {
        Self {
            // Cost composition
            k_starve: 0.15,

            // Pressure index weights
            w_inflight: 0.7,
            w_spawn: 0.3,
            pressure_clip: 10.0,

            // Decay: half-life ~2000 observations
            // 0.5^(1/2000) ≈ 0.999653
            decay: 0.999653,
            // Strike decay: half-life ~100 observations
            // 0.5^(1/100) ≈ 0.993
            strike_decay: 0.993,
            // EMA alpha
            ema_alpha: 0.1,

            // Guardrail thresholds
            t_tiny_inline_us: 50.0,
            t_block_hard_us: 250.0,
            t_inline_under_pressure_us: 100.0,
            p_low: 0.5,
            p_high: 3.0,

            // Strike suppression
            enable_strikes: true,
            t_strike_us: 1000.0,
            s_max: 1.0,

            // Compute hints
            hint_trust_threshold: 5.0,
            hint_exploration_count: 3,
            hint_min_samples: 3.0,
            hint_low_ema_us: 30.0,
            hint_medium_ema_us: 200.0,
            hint_high_ema_us: 1000.0,

            // Calibration
            measured_offload_overhead_us: None,
        }
    }
}

impl MabKnobs {
    /// Create knobs with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply calibration results to set the measured offload overhead.
    ///
    /// The MAB uses this to understand the "fixed cost" of offloading,
    /// which helps make better decisions for borderline workloads.
    pub fn apply_calibration(&mut self, cal: &CalibrationResult) {
        self.measured_offload_overhead_us = Some(cal.offload_overhead_us);
    }

    /// Get the effective offload overhead estimate (microseconds).
    ///
    /// Uses the measured value if available, otherwise a conservative default.
    pub fn offload_overhead_us(&self) -> f64 {
        self.measured_offload_overhead_us.unwrap_or(300.0)
    }

    /// Builder method to set k_starve.
    pub fn with_k_starve(mut self, k_starve: f64) -> Self {
        self.k_starve = k_starve;
        self
    }

    /// Builder method to disable strikes (GR3).
    pub fn without_strikes(mut self) -> Self {
        self.enable_strikes = false;
        self
    }

    /// Builder method to set guardrail thresholds.
    pub fn with_thresholds(mut self, tiny: f64, block_hard: f64, under_pressure: f64) -> Self {
        self.t_tiny_inline_us = tiny;
        self.t_block_hard_us = block_hard;
        self.t_inline_under_pressure_us = under_pressure;
        self
    }

    /// Builder method to set hint exploration count.
    pub fn with_hint_exploration(mut self, count: u32) -> Self {
        self.hint_exploration_count = count;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let knobs = MabKnobs::default();
        assert!((knobs.k_starve - 0.15).abs() < 0.001);
        assert!((knobs.decay - 0.999653).abs() < 0.0001);
        assert!(knobs.enable_strikes);
        assert_eq!(knobs.hint_exploration_count, 3);
    }

    #[test]
    fn test_offload_overhead() {
        let knobs = MabKnobs::default();
        // Default uses conservative estimate
        assert!((knobs.offload_overhead_us() - 300.0).abs() < 0.001);

        let knobs = MabKnobs {
            measured_offload_overhead_us: Some(150.0),
            ..Default::default()
        };
        assert!((knobs.offload_overhead_us() - 150.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_calibration() {
        let mut knobs = MabKnobs::default();
        let cal = CalibrationResult {
            offload_overhead_us: 180.0,
            p50_us: 175.0,
            p99_us: 450.0,
        };
        knobs.apply_calibration(&cal);
        assert!((knobs.offload_overhead_us() - 180.0).abs() < 0.001);
    }

    #[test]
    fn test_builder_methods() {
        let knobs = MabKnobs::default()
            .with_k_starve(0.2)
            .without_strikes()
            .with_hint_exploration(5);

        assert!((knobs.k_starve - 0.2).abs() < 0.001);
        assert!(!knobs.enable_strikes);
        assert_eq!(knobs.hint_exploration_count, 5);
    }

    #[test]
    fn test_serialization() {
        let knobs = MabKnobs::default();
        let json = serde_json::to_string(&knobs).unwrap();
        let parsed: MabKnobs = serde_json::from_str(&json).unwrap();
        assert!((parsed.k_starve - knobs.k_starve).abs() < 0.001);
    }
}
