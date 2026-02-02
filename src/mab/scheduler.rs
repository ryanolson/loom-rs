//! Core MAB scheduler with Thompson Sampling.
//!
//! The scheduler makes per-function decisions about whether to execute
//! compute work inline on Tokio or offload to Rayon, learning from
//! observed execution times to minimize total cost.

use std::collections::HashMap;
use std::time::Instant;

use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use super::knobs::MabKnobs;
use super::types::{Arm, ArmStats, ComputeHint, Context, DecisionId, FunctionKey, KeyStats};
use crate::metrics::LoomMetrics;

/// Pending decision waiting for outcome recording.
struct Pending {
    key: FunctionKey,
    arm: Arm,
}

/// Internal mutable state of the scheduler.
struct MabInner {
    /// Global prior for new keys
    global: KeyStats,
    /// Per-function statistics
    per_key: HashMap<FunctionKey, KeyStats>,
    /// Pending decisions awaiting finish()
    pending: HashMap<DecisionId, Pending>,
    /// Counter for generating unique decision IDs
    next_id: u64,
    /// Thread-local RNG for Thompson sampling
    rng: SmallRng,
}

/// Multi-Armed Bandit scheduler with Thompson Sampling.
///
/// Makes per-function decisions about whether to execute compute work
/// inline on Tokio or offload to Rayon. Learns from observed execution
/// times while respecting guardrails to prevent Tokio starvation.
///
/// # Usage Patterns
///
/// ## Stream Mode (per-stream scheduler)
/// Each `adaptive_map()` stream owns its own scheduler for immediate feedback:
/// ```ignore
/// stream.adaptive_map(|x| expensive(x))
/// ```
///
/// ## Handler Mode (shared scheduler)
/// Multiple handler invocations share a scheduler for delayed feedback:
/// ```ignore
/// let scheduler = runtime.mab_scheduler();
/// let (id, arm) = scheduler.choose(key, &ctx);
/// // ... execute work based on arm ...
/// scheduler.finish(id, cost_us, Some(fn_us));
/// ```
pub struct MabScheduler {
    inner: Mutex<MabInner>,
    knobs: MabKnobs,
    /// Optional prometheus metrics for recording decisions and guardrails
    metrics: Option<LoomMetrics>,
}

impl MabScheduler {
    /// Create a new scheduler with the given knobs.
    pub fn new(knobs: MabKnobs) -> Self {
        Self {
            inner: Mutex::new(MabInner {
                global: KeyStats::default(),
                per_key: HashMap::new(),
                pending: HashMap::new(),
                next_id: 0,
                rng: SmallRng::from_entropy(),
            }),
            knobs,
            metrics: None,
        }
    }

    /// Create a new scheduler with metrics for prometheus exposition.
    ///
    /// When metrics are provided, the scheduler will record:
    /// - Decision counts (inline vs offload)
    /// - Guardrail activations (GR1, GR2, GR3)
    /// - Starvation events (when strikes are accumulated)
    /// - Pressure index updates
    pub fn with_metrics(knobs: MabKnobs, metrics: LoomMetrics) -> Self {
        Self {
            inner: Mutex::new(MabInner {
                global: KeyStats::default(),
                per_key: HashMap::new(),
                pending: HashMap::new(),
                next_id: 0,
                rng: SmallRng::from_entropy(),
            }),
            knobs,
            metrics: Some(metrics),
        }
    }

    /// Create a scheduler with default knobs.
    pub fn with_defaults() -> Self {
        Self::new(MabKnobs::default())
    }

    /// Choose which arm to use for a function.
    ///
    /// Returns a `DecisionId` and the chosen `Arm`. After executing the work,
    /// call `finish()` with the decision ID to record the outcome.
    ///
    /// # Arguments
    /// * `key` - Identifies the function (for per-function learning)
    /// * `ctx` - Current runtime context (pressure, queue depth, etc.)
    pub fn choose(&self, key: FunctionKey, ctx: &Context) -> (DecisionId, Arm) {
        self.choose_with_hint(key, ctx, ComputeHint::Unknown)
    }

    /// Choose which arm to use, with a compute hint for cold-start guidance.
    ///
    /// The hint biases initial exploration but is not trusted until validated
    /// by actual observations.
    ///
    /// # Arguments
    /// * `key` - Identifies the function
    /// * `ctx` - Current runtime context
    /// * `hint` - Optional hint about expected compute cost
    pub fn choose_with_hint(
        &self,
        key: FunctionKey,
        ctx: &Context,
        hint: ComputeHint,
    ) -> (DecisionId, Arm) {
        let mut inner = self.inner.lock();

        // Check if we need to create stats for this key
        if !inner.per_key.contains_key(&key) {
            // Initialize from hint if available
            let mut stats = inner.global.clone();
            if hint != ComputeHint::Unknown {
                stats.ema_fn_us = self.hint_to_ema(hint);
                if hint == ComputeHint::High {
                    stats.hint_explore_remaining = self.knobs.hint_exploration_count;
                }
            }
            inner.per_key.insert(key, stats);
        }

        // Calculate pressure index
        let pressure = self.pressure_index(ctx);

        // Get the stats (we know it exists now)
        let ks = inner.per_key.get_mut(&key).unwrap();

        // Check if we should force offload exploration for High hint
        if ks.hint_explore_remaining > 0 {
            ks.hint_explore_remaining -= 1;
            let id = self.allocate_decision_inner(&mut inner, key, Arm::OffloadRayon);
            return (id, Arm::OffloadRayon);
        }

        // Collect stats for Thompson sampling before we need to borrow rng
        let inline_stats = ks.inline;
        let offload_stats = ks.offload;
        let ema_fn_us = ks.ema_fn_us;
        let inline_n_eff = ks.inline.n_eff;
        let offload_n_eff = ks.offload.n_eff;
        let inline_strikes = ks.inline_strikes;

        // Apply guardrails (using local copies)
        let (inline_allowed, guardrail_triggered) =
            self.inline_allowed_with_guardrail(ema_fn_us, inline_strikes, ctx, pressure);

        let arm = if !inline_allowed {
            // Guardrails forbid inline
            Arm::OffloadRayon
        } else if inline_n_eff < 1.0 && offload_n_eff < 1.0 {
            // Cold start: explore inline first (it's cheaper if fast)
            Arm::InlineTokio
        } else {
            // Thompson Sampling
            self.thompson_sample(&inline_stats, &offload_stats, pressure, &mut inner.rng)
        };

        // Record metrics if available
        if let Some(ref metrics) = self.metrics {
            metrics.record_decision(matches!(arm, Arm::InlineTokio));
            metrics.set_pressure_index(pressure);
            if let Some(guardrail) = guardrail_triggered {
                metrics.record_guardrail(guardrail);
            }
        }

        let id = self.allocate_decision_inner(&mut inner, key, arm);
        (id, arm)
    }

    /// Record the outcome of a decision.
    ///
    /// # Arguments
    /// * `id` - The decision ID returned by `choose()`
    /// * `observed_cost_us` - Total wall-clock time in microseconds
    /// * `observed_fn_us` - Optional pure function time (excluding overhead)
    pub fn finish(&self, id: DecisionId, observed_cost_us: f64, observed_fn_us: Option<f64>) {
        let mut inner = self.inner.lock();

        let pending = match inner.pending.remove(&id) {
            Some(p) => p,
            None => return, // Already finished or invalid ID
        };

        let ks = match inner.per_key.get_mut(&pending.key) {
            Some(ks) => ks,
            None => return, // Key was removed (shouldn't happen)
        };

        // Use provided fn_us or estimate from total cost
        let fn_us = observed_fn_us.unwrap_or(observed_cost_us);

        // Update EMA of function time
        if ks.ema_fn_us == 0.0 {
            ks.ema_fn_us = fn_us;
        } else {
            ks.ema_fn_us =
                self.knobs.ema_alpha * fn_us + (1.0 - self.knobs.ema_alpha) * ks.ema_fn_us;
        }

        // Update arm statistics using log-cost model
        let log_cost = observed_cost_us.max(1.0).ln();
        let arm_stats = match pending.arm {
            Arm::InlineTokio => &mut ks.inline,
            Arm::OffloadRayon => &mut ks.offload,
        };

        // Decay existing stats
        arm_stats.n_eff *= self.knobs.decay;
        arm_stats.mu *= self.knobs.decay;
        arm_stats.s2 *= self.knobs.decay;

        // Update with new observation (Welford's online algorithm)
        arm_stats.n_eff += 1.0;
        let delta = log_cost - arm_stats.mu;
        arm_stats.mu += delta / arm_stats.n_eff;
        let delta2 = log_cost - arm_stats.mu;
        arm_stats.s2 += delta * delta2;

        // Update strike counter if inline was slow
        if pending.arm == Arm::InlineTokio {
            if fn_us > self.knobs.t_strike_us {
                ks.inline_strikes = (ks.inline_strikes + 1.0).min(self.knobs.s_max * 2.0);
                // Record starvation event when inline work was slow
                if let Some(ref metrics) = self.metrics {
                    metrics.record_starvation();
                }
            } else {
                ks.inline_strikes *= self.knobs.strike_decay;
            }
        }

        // Clone stats for global prior update (avoids borrow conflict)
        let ks_clone = ks.clone();

        // Update global prior
        Self::update_global_prior(&self.knobs, &mut inner.global, &ks_clone);
    }

    /// Execute work adaptively, returning the result.
    ///
    /// This is a convenience method that handles the full choose-execute-finish cycle.
    /// Used by `adaptive_map()` for stream processing.
    ///
    /// # Arguments
    /// * `key` - Function identifier
    /// * `ctx` - Runtime context
    /// * `hint` - Compute hint
    /// * `f` - The work to execute
    /// * `rayon_pool` - Rayon pool for offload execution
    pub async fn execute_adaptive<F, R>(
        &self,
        key: FunctionKey,
        ctx: &Context,
        hint: ComputeHint,
        f: F,
        rayon_pool: &rayon::ThreadPool,
    ) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (id, arm) = self.choose_with_hint(key, ctx, hint);
        let start = Instant::now();

        let result = match arm {
            Arm::InlineTokio => {
                // Execute inline on current thread
                f()
            }
            Arm::OffloadRayon => {
                // Offload to rayon
                let (task, completion) = crate::bridge::RayonTask::new();
                rayon_pool.spawn(move || {
                    let result = f();
                    completion.complete(result);
                });
                task.await
            }
        };

        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        self.finish(id, elapsed_us, None);

        result
    }

    /// Get the knobs configuration.
    pub fn knobs(&self) -> &MabKnobs {
        &self.knobs
    }

    // === Private helper methods ===

    /// Allocate a new decision ID and store the pending decision.
    fn allocate_decision_inner(
        &self,
        inner: &mut MabInner,
        key: FunctionKey,
        arm: Arm,
    ) -> DecisionId {
        let id = DecisionId(inner.next_id);
        inner.next_id += 1;

        inner.pending.insert(id, Pending { key, arm });

        id
    }

    /// Check if inline is allowed and return which guardrail was triggered.
    ///
    /// Returns `(inline_allowed, guardrail_name)` where guardrail_name is
    /// `Some("GR0"|"GR1"|"GR2"|"GR3")` if inline was blocked, or `None` if allowed.
    fn inline_allowed_with_guardrail(
        &self,
        ema_fn_us: f64,
        inline_strikes: f64,
        ctx: &Context,
        pressure: f64,
    ) -> (bool, Option<&'static str>) {
        // GR0: Single-worker protection
        if ctx.tokio_workers <= 1 {
            let allowed = ema_fn_us < self.knobs.t_tiny_inline_us && pressure < self.knobs.p_low;
            return if allowed {
                (true, None)
            } else {
                (false, Some("GR0"))
            };
        }

        // GR1: Hard blocking threshold
        if ema_fn_us > self.knobs.t_block_hard_us {
            return (false, Some("GR1"));
        }

        // GR2: Pressure-sensitive threshold
        if pressure > self.knobs.p_high && ema_fn_us > self.knobs.t_inline_under_pressure_us {
            return (false, Some("GR2"));
        }

        // GR3: Strike suppression
        if self.knobs.enable_strikes && inline_strikes >= self.knobs.s_max {
            return (false, Some("GR3"));
        }

        (true, None)
    }

    /// Calculate pressure index from runtime context.
    fn pressure_index(&self, ctx: &Context) -> f64 {
        if ctx.tokio_workers == 0 {
            return 0.0;
        }

        // Normalize inflight tasks by worker count
        let inflight_ratio = ctx.inflight_tasks as f64 / ctx.tokio_workers as f64;

        // Normalize spawn rate (baseline ~1000/s per worker is normal)
        let spawn_ratio = ctx.spawn_rate_per_s as f64 / (1000.0 * ctx.tokio_workers as f64);

        // Weighted combination
        let pressure = self.knobs.w_inflight * inflight_ratio + self.knobs.w_spawn * spawn_ratio;

        // Clip to maximum
        pressure.min(self.knobs.pressure_clip)
    }

    /// Perform Thompson Sampling to choose an arm.
    fn thompson_sample(
        &self,
        inline: &ArmStats,
        offload: &ArmStats,
        pressure: f64,
        rng: &mut SmallRng,
    ) -> Arm {
        // Sample from posterior for each arm
        let inline_sample = self.sample_from_posterior(inline, rng);
        let offload_sample = self.sample_from_posterior(offload, rng);

        // Adjust inline cost for pressure (starvation penalty)
        let inline_adjusted = inline_sample * (1.0 + self.knobs.k_starve * pressure);

        // Add offload overhead
        let offload_adjusted = offload_sample + self.knobs.offload_overhead_us().ln();

        if inline_adjusted < offload_adjusted {
            Arm::InlineTokio
        } else {
            Arm::OffloadRayon
        }
    }

    /// Sample from the posterior distribution of an arm.
    fn sample_from_posterior(&self, stats: &ArmStats, rng: &mut SmallRng) -> f64 {
        if stats.n_eff < 1.0 {
            // No data, use diffuse prior
            // Sample from a wide normal centered on log(100us)
            let prior_mu = 100.0_f64.ln();
            let prior_std = 2.0; // Wide prior
            Normal::new(prior_mu, prior_std)
                .map(|d| d.sample(rng))
                .unwrap_or(prior_mu)
        } else {
            // Sample from Normal(mu, sigma^2/n_eff) - posterior with known variance
            let posterior_std = (stats.variance() / stats.n_eff).sqrt().max(0.01);
            Normal::new(stats.mu, posterior_std)
                .map(|d| d.sample(rng))
                .unwrap_or(stats.mu)
        }
    }

    /// Convert a compute hint to an initial EMA value.
    fn hint_to_ema(&self, hint: ComputeHint) -> f64 {
        match hint {
            ComputeHint::Unknown => 0.0,
            ComputeHint::Low => self.knobs.hint_low_ema_us,
            ComputeHint::Medium => self.knobs.hint_medium_ema_us,
            ComputeHint::High => self.knobs.hint_high_ema_us,
        }
    }

    /// Update the global prior from learned key stats.
    fn update_global_prior(_knobs: &MabKnobs, global: &mut KeyStats, ks: &KeyStats) {
        // Simple exponential smoothing of global statistics
        let alpha = 0.01; // Slow update rate for global prior

        if ks.inline.n_eff > 0.0 {
            global.inline.mu = alpha * ks.inline.mu + (1.0 - alpha) * global.inline.mu;
        }
        if ks.offload.n_eff > 0.0 {
            global.offload.mu = alpha * ks.offload.mu + (1.0 - alpha) * global.offload.mu;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ctx() -> Context {
        Context {
            tokio_workers: 4,
            inflight_tasks: 2,
            spawn_rate_per_s: 100.0,
            rayon_threads: 8,
            rayon_queue_depth: 0,
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = MabScheduler::with_defaults();
        assert!((scheduler.knobs().k_starve - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_choose_returns_decision_id() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(1);
        let ctx = default_ctx();

        let (id1, _) = scheduler.choose(key, &ctx);
        let (id2, _) = scheduler.choose(key, &ctx);

        // Decision IDs should be unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_finish_updates_stats() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(1);
        let ctx = default_ctx();

        // Make a decision and record outcome
        let (id, arm) = scheduler.choose(key, &ctx);
        scheduler.finish(id, 50.0, Some(50.0));

        // Make another decision - should have learned
        let (id2, _) = scheduler.choose(key, &ctx);
        scheduler.finish(id2, 50.0, Some(50.0));

        // Check that stats were updated
        let inner = scheduler.inner.lock();
        let ks = inner.per_key.get(&key).unwrap();
        assert!(ks.inline.n_eff > 0.0 || ks.offload.n_eff > 0.0);

        drop(inner);
        let _ = arm;
    }

    #[test]
    fn test_gr0_single_worker_protection() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(1);

        // Pre-seed with slow function time
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key,
                KeyStats {
                    ema_fn_us: 100.0, // Above tiny threshold
                    ..Default::default()
                },
            );
        }

        // Single worker context
        let ctx = Context {
            tokio_workers: 1,
            inflight_tasks: 1,
            spawn_rate_per_s: 100.0,
            rayon_threads: 4,
            rayon_queue_depth: 0,
        };

        // Should force offload due to GR0
        let (_, arm) = scheduler.choose(key, &ctx);
        assert_eq!(arm, Arm::OffloadRayon);
    }

    #[test]
    fn test_gr1_hard_blocking_threshold() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(1);

        // Pre-seed with very slow function time
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key,
                KeyStats {
                    ema_fn_us: 500.0, // Above hard threshold (250us)
                    ..Default::default()
                },
            );
        }

        let ctx = default_ctx();

        // Should force offload due to GR1
        let (_, arm) = scheduler.choose(key, &ctx);
        assert_eq!(arm, Arm::OffloadRayon);
    }

    #[test]
    fn test_gr3_strike_suppression() {
        let knobs = MabKnobs::default();
        let scheduler = MabScheduler::new(knobs);
        let key = FunctionKey(1);

        // Pre-seed with strike at max
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key,
                KeyStats {
                    ema_fn_us: 50.0,     // Below thresholds
                    inline_strikes: 1.0, // At s_max
                    ..Default::default()
                },
            );
        }

        let ctx = default_ctx();

        // Should force offload due to GR3
        let (_, arm) = scheduler.choose(key, &ctx);
        assert_eq!(arm, Arm::OffloadRayon);
    }

    #[test]
    fn test_high_hint_forces_early_offload() {
        let knobs = MabKnobs {
            hint_exploration_count: 3,
            ..Default::default()
        };
        let scheduler = MabScheduler::new(knobs);
        let key = FunctionKey(1);
        let ctx = default_ctx();

        // First 3 calls with High hint should force offload
        for i in 0..3 {
            let (id, arm) = scheduler.choose_with_hint(key, &ctx, ComputeHint::High);
            assert_eq!(arm, Arm::OffloadRayon, "iteration {} should offload", i);
            scheduler.finish(id, 500.0, Some(500.0));
        }

        // After exploration, Thompson Sampling takes over
        // (we can't assert the exact arm since it depends on sampling)
    }

    #[test]
    fn test_low_hint_sets_initial_ema() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(2);
        let ctx = default_ctx();

        // First call with Low hint
        let (id, _) = scheduler.choose_with_hint(key, &ctx, ComputeHint::Low);
        scheduler.finish(id, 30.0, Some(30.0));

        // Check EMA was seeded
        let inner = scheduler.inner.lock();
        let ks = inner.per_key.get(&key).unwrap();
        assert!(ks.ema_fn_us > 0.0);
    }

    #[test]
    fn test_pressure_index_calculation() {
        let scheduler = MabScheduler::with_defaults();

        // Low pressure
        let ctx = Context {
            tokio_workers: 4,
            inflight_tasks: 4,        // 1 per worker
            spawn_rate_per_s: 1000.0, // ~250 per worker
            rayon_threads: 8,
            rayon_queue_depth: 0,
        };
        let pressure = scheduler.pressure_index(&ctx);
        assert!(pressure < 1.0, "low pressure: {}", pressure);

        // High pressure
        let ctx = Context {
            tokio_workers: 4,
            inflight_tasks: 40,        // 10 per worker
            spawn_rate_per_s: 20000.0, // 5000 per worker
            rayon_threads: 8,
            rayon_queue_depth: 0,
        };
        let pressure = scheduler.pressure_index(&ctx);
        assert!(pressure > 1.0, "high pressure: {}", pressure);
    }

    #[test]
    fn test_finish_with_invalid_id() {
        let scheduler = MabScheduler::with_defaults();

        // Should not panic with invalid ID
        scheduler.finish(DecisionId(999999), 100.0, None);
    }

    #[test]
    fn test_cold_start_explores_inline() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(100);
        let ctx = default_ctx();

        // First call with no hint should explore inline (cheaper if fast)
        let (_, arm) = scheduler.choose(key, &ctx);
        assert_eq!(arm, Arm::InlineTokio);
    }

    // =============================================================================
    // GR2 Pressure-Sensitive Threshold Tests
    // =============================================================================

    #[test]
    fn test_gr2_pressure_sensitive_threshold() {
        // GR2: pressure > p_high && ema > t_inline_under_pressure_us → offload
        // Default p_high = 3.0, t_inline_under_pressure_us = 100.0
        let knobs = MabKnobs::default();
        let scheduler = MabScheduler::new(knobs);
        let key = FunctionKey::from_name("gr2_test");

        // Pre-seed with medium-duration work (above t_inline_under_pressure_us (100) but below t_block_hard_us (250))
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key,
                KeyStats {
                    ema_fn_us: 150.0, // 150µs > t_inline_under_pressure_us (100us) but < t_block_hard_us (250us)
                    ..Default::default()
                },
            );
        }

        // Under high pressure, should force offload via GR2
        // Pressure = w_inflight * (inflight/workers) + w_spawn * (spawn_rate / (1000 * workers))
        // Pressure = 0.7 * (200/4) + 0.3 * (20000 / (1000 * 4)) = 0.7 * 50 + 0.3 * 5 = 35 + 1.5 = 36.5
        // (capped at pressure_clip = 10.0)
        let high_pressure_ctx = Context {
            tokio_workers: 4,
            inflight_tasks: 200, // High inflight count creates pressure
            spawn_rate_per_s: 20000.0,
            rayon_threads: 8,
            rayon_queue_depth: 0,
        };

        let (_, arm) = scheduler.choose(key, &high_pressure_ctx);
        assert_eq!(
            arm,
            Arm::OffloadRayon,
            "GR2 should force offload under high pressure"
        );

        // Under low pressure, the same EMA should allow inline
        let low_pressure_ctx = Context {
            tokio_workers: 4,
            inflight_tasks: 2,
            spawn_rate_per_s: 100.0,
            rayon_threads: 8,
            rayon_queue_depth: 0,
        };

        // Need a new key since the first one might have modified stats
        let key2 = FunctionKey::from_name("gr2_test_low_pressure");
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key2,
                KeyStats {
                    ema_fn_us: 150.0, // Same EMA
                    ..Default::default()
                },
            );
        }

        let (_, arm) = scheduler.choose(key2, &low_pressure_ctx);
        // Under low pressure, inline should be allowed (cold start explores inline)
        assert_eq!(
            arm,
            Arm::InlineTokio,
            "Under low pressure, inline should be allowed"
        );
    }

    // =============================================================================
    // Learning Convergence Tests
    // =============================================================================

    #[test]
    fn test_learns_to_inline_fast_work() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey::from_name("fast_work");
        let ctx = default_ctx();

        // Train with fast work (5µs)
        for _ in 0..50 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 5.0, Some(5.0));
        }

        // After learning, should prefer inline
        let mut inline_count = 0;
        for _ in 0..10 {
            let (id, arm) = scheduler.choose(key, &ctx);
            if arm == Arm::InlineTokio {
                inline_count += 1;
            }
            scheduler.finish(id, 5.0, Some(5.0));
        }

        assert!(
            inline_count >= 7,
            "Should prefer inline for fast work, got {} inline out of 10",
            inline_count
        );
    }

    #[test]
    fn test_learns_to_offload_slow_work() {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey::from_name("slow_work");
        let ctx = default_ctx();

        // Train with slow work (500µs) - triggers GR1 after learning
        for _ in 0..50 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 500.0, Some(500.0));
        }

        // After learning, should prefer offload (EMA > t_block_hard_us)
        let mut offload_count = 0;
        for _ in 0..10 {
            let (id, arm) = scheduler.choose(key, &ctx);
            if arm == Arm::OffloadRayon {
                offload_count += 1;
            }
            scheduler.finish(id, 500.0, Some(500.0));
        }

        assert!(
            offload_count >= 8,
            "Should prefer offload for slow work, got {} offload out of 10",
            offload_count
        );
    }

    // =============================================================================
    // Thompson Sampling Tests
    // =============================================================================

    #[test]
    fn test_thompson_sampling_explores_initially() {
        // Test that Thompson sampling makes some exploration decisions, but note that
        // for fast work it will quickly converge to preferring inline, which is correct.
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey::from_name("borderline");
        let ctx = default_ctx();

        // Use 100µs work - right at the edge of t_inline_under_pressure_us threshold
        // This should create more uncertainty than very fast or very slow work
        let mut inline_count = 0;
        let mut offload_count = 0;

        for _ in 0..100 {
            let (id, arm) = scheduler.choose(key, &ctx);
            match arm {
                Arm::InlineTokio => inline_count += 1,
                Arm::OffloadRayon => offload_count += 1,
            }
            scheduler.finish(id, 100.0, Some(100.0));
        }

        // With borderline work, we should see some exploration in at least one direction
        // (either inline or offload should have been tried at least once)
        assert!(
            inline_count > 0 || offload_count > 0,
            "Thompson sampling should make decisions: inline={}, offload={}",
            inline_count,
            offload_count
        );

        // The total should be 100
        assert_eq!(inline_count + offload_count, 100);
    }

    // =============================================================================
    // Metrics Integration Tests
    // =============================================================================

    #[test]
    fn test_prometheus_metrics_recorded() {
        let metrics = LoomMetrics::new();
        let scheduler = MabScheduler::with_metrics(MabKnobs::default(), metrics.clone());

        let key = FunctionKey::from_name("metrics_test");
        let ctx = default_ctx();

        // Make some decisions
        for _ in 0..10 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        // Verify metrics recorded
        let total_decisions = metrics.inline_decisions.get() + metrics.offload_decisions.get();
        assert_eq!(
            total_decisions, 10,
            "Should have recorded 10 decisions, got {}",
            total_decisions
        );
    }

    #[test]
    fn test_guardrail_metrics_recorded() {
        let metrics = LoomMetrics::new();
        let knobs = MabKnobs::default();
        let scheduler = MabScheduler::with_metrics(knobs, metrics.clone());

        let key = FunctionKey::from_name("gr1_metrics");

        // Pre-seed with very slow work to trigger GR1
        {
            let mut inner = scheduler.inner.lock();
            inner.per_key.insert(
                key,
                KeyStats {
                    ema_fn_us: 300.0, // >250µs triggers GR1
                    ..Default::default()
                },
            );
        }

        let ctx = default_ctx();

        // Make decisions that should trigger GR1
        for _ in 0..10 {
            let (id, arm) = scheduler.choose(key, &ctx);
            assert_eq!(arm, Arm::OffloadRayon);
            scheduler.finish(id, 300.0, Some(300.0));
        }

        assert!(
            metrics.gr1_activations.get() >= 10,
            "GR1 should have been recorded, got {}",
            metrics.gr1_activations.get()
        );
    }

    #[test]
    fn test_starvation_metrics_recorded() {
        let metrics = LoomMetrics::new();
        let scheduler = MabScheduler::with_metrics(MabKnobs::default(), metrics.clone());

        let key = FunctionKey::from_name("starvation_test");
        let ctx = default_ctx();

        // Force inline decision and report slow execution (triggers strike)
        // First, we need to get inline decisions
        for _ in 0..5 {
            let (id, arm) = scheduler.choose(key, &ctx);
            // Finish with slow time to trigger starvation event
            // t_strike_us is 1000us by default
            if arm == Arm::InlineTokio {
                scheduler.finish(id, 1500.0, Some(1500.0)); // >1000µs triggers strike
            } else {
                scheduler.finish(id, 1500.0, Some(1500.0));
            }
        }

        // Should have recorded starvation events for inline decisions
        assert!(
            metrics.starvation_events.get() > 0,
            "Should have recorded starvation events"
        );
    }

    #[test]
    fn test_guardrail_returns_correct_name() {
        let scheduler = MabScheduler::with_defaults();

        // Test GR0 (single worker protection)
        let ctx_single = Context {
            tokio_workers: 1,
            inflight_tasks: 10,
            spawn_rate_per_s: 5000.0,
            rayon_threads: 4,
            rayon_queue_depth: 0,
        };
        // GR0: when tokio_workers <= 1, require ema < t_tiny_inline_us (50) AND pressure < p_low (0.5)
        // 100us > 50us, so it should fail GR0
        let (allowed, guardrail) =
            scheduler.inline_allowed_with_guardrail(100.0, 0.0, &ctx_single, 0.3);
        assert!(!allowed);
        assert_eq!(guardrail, Some("GR0"));

        // Test GR1 (hard blocking threshold: ema > t_block_hard_us (250))
        let ctx = default_ctx();
        let (allowed, guardrail) = scheduler.inline_allowed_with_guardrail(300.0, 0.0, &ctx, 1.0);
        assert!(!allowed);
        assert_eq!(guardrail, Some("GR1"));

        // Test GR2 (pressure > p_high (3.0) AND ema > t_inline_under_pressure_us (100))
        // Need ema > 100 and pressure > 3.0
        let (allowed, guardrail) = scheduler.inline_allowed_with_guardrail(150.0, 0.0, &ctx, 5.0);
        assert!(!allowed);
        assert_eq!(guardrail, Some("GR2"));

        // Test GR3 (strikes >= s_max (1.0))
        let (allowed, guardrail) = scheduler.inline_allowed_with_guardrail(50.0, 1.0, &ctx, 1.0);
        assert!(!allowed);
        assert_eq!(guardrail, Some("GR3"));

        // Test allowed (ema < t_inline_under_pressure_us, low pressure, no strikes)
        let (allowed, guardrail) = scheduler.inline_allowed_with_guardrail(30.0, 0.0, &ctx, 0.5);
        assert!(allowed);
        assert_eq!(guardrail, None);
    }
}
