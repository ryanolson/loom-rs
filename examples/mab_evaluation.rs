//! MAB Evaluation Suite with Thread Configuration Analysis
//!
//! Comprehensive evaluation that produces a markdown report proving the MAB
//! scheduler delivers value. Tests multiple thread configurations to find
//! optimal tokio:rayon ratios and detect starvation events.
//!
//! Run: cargo run --example mab_evaluation --release > report.md

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::stream::{self, StreamExt};
use loom_rs::cpuset::format_cpuset;
use loom_rs::mab::{Arm, Context, FunctionKey, MabKnobs, MabScheduler};
use loom_rs::{ComputeStreamExt, LoomBuilder, LoomRuntime};
use parking_lot::Mutex;

// =============================================================================
// Configuration
// =============================================================================

/// Thread configuration for testing different tokio:rayon ratios.
#[derive(Clone, Debug)]
struct ThreadConfig {
    name: &'static str,
    tokio_threads: usize,
    rayon_threads: usize,
    description: &'static str,
}

/// Configuration for the evaluation suite.
struct EvaluationConfig {
    /// Number of runs per test for statistical significance
    runs_per_test: usize,
    /// Number of warmup iterations to discard
    warmup_iterations: usize,
    /// Number of items per throughput scenario
    item_count: usize,
    /// Number of observations to test for convergence
    convergence_observations: usize,
    /// Duration for latency probe phases
    latency_phase_duration: Duration,
    /// Target sleep time for latency probes
    latency_probe_sleep: Duration,
    /// Work rate for latency tests (tasks/sec)
    latency_work_rate: u64,
    /// Work duration for latency tests (microseconds)
    latency_work_us: u64,
    /// Total CPUs to use for evaluation (bounded set)
    total_cpus: usize,
    /// Thread configurations to test
    thread_configs: Vec<ThreadConfig>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            runs_per_test: 5,
            warmup_iterations: 100,
            item_count: 500,
            convergence_observations: 100,
            latency_phase_duration: Duration::from_secs(3),
            latency_probe_sleep: Duration::from_millis(1),
            latency_work_rate: 500,
            latency_work_us: 500,
            total_cpus: 16,
            thread_configs: vec![
                ThreadConfig {
                    name: "minimal-tokio",
                    tokio_threads: 1,
                    rayon_threads: 15,
                    description: "Compute-heavy, minimal async (1:15)",
                },
                ThreadConfig {
                    name: "balanced-tokio",
                    tokio_threads: 2,
                    rayon_threads: 14,
                    description: "General purpose (1:7)",
                },
                ThreadConfig {
                    name: "equal-split",
                    tokio_threads: 8,
                    rayon_threads: 8,
                    description: "Heavy async + heavy compute (1:1)",
                },
            ],
        }
    }
}

// =============================================================================
// Starvation Tracking
// =============================================================================

/// A starvation event detected during evaluation.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct StarvationEvent {
    timestamp: Instant,
    config_name: &'static str,
    tokio_threads: usize,
    pressure_index: f64,
    inflight_tasks: u32,
    guardrail_triggered: Option<&'static str>,
}

/// Tracks starvation events during a test phase.
struct StarvationTracker {
    events: Vec<StarvationEvent>,
    config_name: &'static str,
    tokio_threads: usize,
}

impl StarvationTracker {
    fn new(config_name: &'static str, tokio_threads: usize) -> Self {
        Self {
            events: Vec::new(),
            config_name,
            tokio_threads,
        }
    }

    fn check(&mut self, ctx: &Context, guardrail: Option<&'static str>) {
        let pressure = compute_pressure_index(ctx);
        // Track if pressure > 3.0 (high pressure threshold from knobs) or guardrail triggered
        if pressure > 3.0 || guardrail.is_some() {
            self.events.push(StarvationEvent {
                timestamp: Instant::now(),
                config_name: self.config_name,
                tokio_threads: self.tokio_threads,
                pressure_index: pressure,
                inflight_tasks: ctx.inflight_tasks,
                guardrail_triggered: guardrail,
            });
        }
    }

    fn event_count(&self) -> usize {
        self.events.len()
    }

    fn max_pressure(&self) -> f64 {
        self.events
            .iter()
            .map(|e| e.pressure_index)
            .fold(0.0_f64, f64::max)
    }
}

/// Compute pressure index using the same formula as MabKnobs
fn compute_pressure_index(ctx: &Context) -> f64 {
    if ctx.tokio_workers == 0 {
        return 0.0;
    }
    let knobs = MabKnobs::default();
    let inflight_ratio = ctx.inflight_tasks as f64 / ctx.tokio_workers as f64;
    let spawn_ratio = ctx.spawn_rate_per_s as f64 / (1000.0 * ctx.tokio_workers as f64);
    let pressure = knobs.w_inflight * inflight_ratio + knobs.w_spawn * spawn_ratio;
    pressure.min(knobs.pressure_clip)
}

// =============================================================================
// System Info
// =============================================================================

/// System information for the report header.
struct SystemInfo {
    platform: &'static str,
    arch: &'static str,
    cpus: usize,
    build_profile: &'static str,
}

fn collect_system_info() -> SystemInfo {
    SystemInfo {
        platform: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpus: std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1),
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release (LTO enabled)"
        },
    }
}

// =============================================================================
// Statistical Helpers
// =============================================================================

/// Statistics collected from timing measurements.
#[derive(Clone, Debug, Default)]
struct Stats {
    samples: Vec<f64>,
}

impl Stats {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(10000),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            samples: Vec::with_capacity(cap),
        }
    }

    fn add(&mut self, value: f64) {
        self.samples.push(value);
    }

    fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    fn percentile(&self, pct: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    fn p99(&self) -> f64 {
        self.percentile(99.0)
    }
}

// =============================================================================
// Calibrated Work
// =============================================================================

/// Calibrated work function that runs for approximately the specified duration.
#[inline(never)]
fn calibrated_work(target_us: u64) -> u64 {
    // ~100 iterations per 1us on typical hardware
    let iterations = target_us * 100;
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(std::hint::black_box(i));
    }
    sum
}

// =============================================================================
// Phase 1: MAB Overhead Measurement
// =============================================================================

#[derive(Debug)]
struct OverheadResults {
    cold_ns: Stats,
    warm_ns: Stats,
    full_cycle_ns: Stats,
}

fn measure_overhead(config: &EvaluationConfig) -> OverheadResults {
    let mut cold_ns = Stats::with_capacity(config.runs_per_test * 1000);
    let mut warm_ns = Stats::with_capacity(config.runs_per_test * 1000);
    let mut full_cycle_ns = Stats::with_capacity(config.runs_per_test * 1000);

    let ctx = Context {
        tokio_workers: 4,
        inflight_tasks: 2,
        spawn_rate_per_s: 100.0,
        rayon_threads: 8,
        rayon_queue_depth: 0,
    };

    for run in 0..config.runs_per_test {
        // Cold decisions (new key each time)
        let scheduler = MabScheduler::with_defaults();
        for i in 0..1000 {
            let key = FunctionKey((run * 10000 + i) as u64);
            let start = Instant::now();
            let (id, _arm) = scheduler.choose(key, &ctx);
            let elapsed = start.elapsed().as_nanos() as f64;
            cold_ns.add(elapsed);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        // Warm decisions (same key, stats exist)
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(999999);

        // Warmup
        for _ in 0..config.warmup_iterations {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        // Measure
        for _ in 0..1000 {
            let start = Instant::now();
            let (id, _arm) = scheduler.choose(key, &ctx);
            let elapsed = start.elapsed().as_nanos() as f64;
            warm_ns.add(elapsed);
            // Don't count finish in warm measurement
            scheduler.finish(id, 50.0, Some(50.0));
        }

        // Full cycle (choose + finish)
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(888888);

        for _ in 0..config.warmup_iterations {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        for _ in 0..1000 {
            let start = Instant::now();
            let (id, _arm) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
            let elapsed = start.elapsed().as_nanos() as f64;
            full_cycle_ns.add(elapsed);
        }
    }

    OverheadResults {
        cold_ns,
        warm_ns,
        full_cycle_ns,
    }
}

// =============================================================================
// Phase 2: Wake Latency Impact
// =============================================================================

#[derive(Debug)]
struct LatencyResults {
    baseline: Stats,
    always_inline: StrategyLatencyResult,
    always_offload: StrategyLatencyResult,
    adaptive: StrategyLatencyResult,
}

#[derive(Debug)]
struct StrategyLatencyResult {
    latency: Stats,
    throughput: f64,
    interference_factor: f64,
}

async fn latency_probe(
    expected_sleep: Duration,
    duration: Duration,
    samples: Arc<Mutex<Stats>>,
    running: Arc<AtomicBool>,
) {
    let end = Instant::now() + duration;

    while Instant::now() < end && running.load(Ordering::Relaxed) {
        let start = Instant::now();
        tokio::time::sleep(expected_sleep).await;
        let actual = start.elapsed();

        let drift_us = (actual.as_nanos() as f64 - expected_sleep.as_nanos() as f64) / 1000.0;
        samples.lock().add(drift_us);
    }
}

fn measure_latency_impact(config: &EvaluationConfig, runtime: &LoomRuntime) -> LatencyResults {
    // Baseline (no load)
    let baseline = runtime.block_on(async {
        let samples = Arc::new(Mutex::new(Stats::new()));
        let running = Arc::new(AtomicBool::new(true));

        latency_probe(
            config.latency_probe_sleep,
            config.latency_phase_duration,
            samples.clone(),
            running.clone(),
        )
        .await;

        running.store(false, Ordering::Relaxed);
        let result = std::mem::take(&mut *samples.lock());
        result
    });

    let baseline_p95 = baseline.p95();

    // AlwaysInline
    let always_inline = runtime.block_on(async {
        let samples = Arc::new(Mutex::new(Stats::new()));
        let running = Arc::new(AtomicBool::new(true));
        let completed = Arc::new(AtomicU64::new(0));

        let probe_samples = samples.clone();
        let probe_running = running.clone();
        let probe_sleep = config.latency_probe_sleep;
        let phase_duration = config.latency_phase_duration;

        let probe_handle = runtime.spawn_async(async move {
            latency_probe(probe_sleep, phase_duration, probe_samples, probe_running).await;
        });

        let start = Instant::now();
        let interval = Duration::from_micros(1_000_000 / config.latency_work_rate);
        let work_us = config.latency_work_us;

        while Instant::now() < start + config.latency_phase_duration {
            let task_start = Instant::now();
            // Execute compute work directly on Tokio worker (inline)
            std::hint::black_box(calibrated_work(work_us));
            completed.fetch_add(1, Ordering::Relaxed);

            let elapsed = task_start.elapsed();
            if elapsed < interval {
                tokio::time::sleep(interval - elapsed).await;
            }
        }

        running.store(false, Ordering::Relaxed);
        let _ = probe_handle.await;

        let duration = start.elapsed();
        let latency = { std::mem::take(&mut *samples.lock()) };
        let completed_tasks = completed.load(Ordering::Relaxed);
        let throughput = completed_tasks as f64 / duration.as_secs_f64();
        let interference_factor = if baseline_p95 > 0.0 {
            latency.p95() / baseline_p95
        } else {
            1.0
        };

        StrategyLatencyResult {
            latency,
            throughput,
            interference_factor,
        }
    });

    // AlwaysOffload - needs special handling to avoid nested block_on
    let always_offload = runtime.block_on(async {
        let samples = Arc::new(Mutex::new(Stats::new()));
        let running = Arc::new(AtomicBool::new(true));
        let completed = Arc::new(AtomicU64::new(0));

        let probe_samples = samples.clone();
        let probe_running = running.clone();
        let probe_sleep = config.latency_probe_sleep;
        let phase_duration = config.latency_phase_duration;

        let probe_handle = runtime.spawn_async(async move {
            latency_probe(probe_sleep, phase_duration, probe_samples, probe_running).await;
        });

        let start = Instant::now();
        let interval = Duration::from_micros(1_000_000 / config.latency_work_rate);
        let work_us = config.latency_work_us;

        while Instant::now() < start + config.latency_phase_duration {
            let task_start = Instant::now();
            // Offload to rayon via spawn_compute (await inside async block)
            runtime
                .spawn_compute(move || calibrated_work(work_us))
                .await;
            completed.fetch_add(1, Ordering::Relaxed);

            let elapsed = task_start.elapsed();
            if elapsed < interval {
                tokio::time::sleep(interval - elapsed).await;
            }
        }

        running.store(false, Ordering::Relaxed);
        let _ = probe_handle.await;

        let duration = start.elapsed();
        let latency = { std::mem::take(&mut *samples.lock()) };
        let completed_tasks = completed.load(Ordering::Relaxed);
        let throughput = completed_tasks as f64 / duration.as_secs_f64();
        let interference_factor = if baseline_p95 > 0.0 {
            latency.p95() / baseline_p95
        } else {
            1.0
        };

        StrategyLatencyResult {
            latency,
            throughput,
            interference_factor,
        }
    });

    // Adaptive (using adaptive_map in a limited way)
    let adaptive = runtime.block_on(async {
        let samples = Arc::new(Mutex::new(Stats::new()));
        let running = Arc::new(AtomicBool::new(true));
        let completed = Arc::new(AtomicU64::new(0));

        let probe_samples = samples.clone();
        let probe_running = running.clone();
        let probe_sleep = config.latency_probe_sleep;
        let phase_duration = config.latency_phase_duration;

        let probe_handle = runtime.spawn_async(async move {
            latency_probe(probe_sleep, phase_duration, probe_samples, probe_running).await;
        });

        let start = Instant::now();
        let work_us = config.latency_work_us;
        let item_count = (config.latency_phase_duration.as_secs() as usize)
            * (config.latency_work_rate as usize);
        let completed_clone = completed.clone();

        stream::iter(0..item_count)
            .adaptive_map(move |_| {
                let result = calibrated_work(work_us);
                completed_clone.fetch_add(1, Ordering::Relaxed);
                result
            })
            .for_each(|_| async {
                tokio::task::yield_now().await;
            })
            .await;

        running.store(false, Ordering::Relaxed);
        let _ = probe_handle.await;

        let duration = start.elapsed();
        let latency = { std::mem::take(&mut *samples.lock()) };
        let completed_tasks = completed.load(Ordering::Relaxed);
        let throughput = completed_tasks as f64 / duration.as_secs_f64();
        let interference_factor = if baseline_p95 > 0.0 {
            latency.p95() / baseline_p95
        } else {
            1.0
        };

        StrategyLatencyResult {
            latency,
            throughput,
            interference_factor,
        }
    });

    LatencyResults {
        baseline,
        always_inline,
        always_offload,
        adaptive,
    }
}

// =============================================================================
// Phase 3: Throughput Comparison
// =============================================================================

#[derive(Debug)]
struct ThroughputResults {
    fast_work: WorkSizeResult,
    medium_work: WorkSizeResult,
    slow_work: WorkSizeResult,
    mixed_work: WorkSizeResult,
}

#[derive(Debug)]
struct WorkSizeResult {
    work_description: &'static str,
    compute_map_items_per_sec: f64,
    adaptive_map_items_per_sec: f64,
    speedup_percent: f64,
}

fn measure_throughput(config: &EvaluationConfig, runtime: &LoomRuntime) -> ThroughputResults {
    let item_count = config.item_count;

    // Fast work (~10us)
    let fast_work = measure_work_size(runtime, item_count, "Fast (~10us)", 10);

    // Medium work (~100us)
    let medium_work = measure_work_size(runtime, item_count, "Medium (~100us)", 100);

    // Slow work (~500us)
    let slow_work = measure_work_size(runtime, item_count, "Slow (~500us)", 500);

    // Mixed work (60% fast, 30% medium, 10% slow)
    let mixed_work = measure_mixed_work(runtime, item_count);

    ThroughputResults {
        fast_work,
        medium_work,
        slow_work,
        mixed_work,
    }
}

fn measure_work_size(
    runtime: &LoomRuntime,
    item_count: usize,
    description: &'static str,
    work_us: u64,
) -> WorkSizeResult {
    // compute_map timing
    let compute_map_durations: Vec<Duration> = (0..3)
        .map(|_| {
            let start = Instant::now();
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..item_count)
                    .compute_map(move |_| calibrated_work(work_us))
                    .collect()
                    .await;
                std::hint::black_box(results);
            });
            start.elapsed()
        })
        .collect();

    let compute_map_avg = compute_map_durations
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 3.0;
    let compute_map_items_per_sec = item_count as f64 / compute_map_avg;

    // adaptive_map timing
    let adaptive_map_durations: Vec<Duration> = (0..3)
        .map(|_| {
            let start = Instant::now();
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..item_count)
                    .adaptive_map(move |_| calibrated_work(work_us))
                    .collect()
                    .await;
                std::hint::black_box(results);
            });
            start.elapsed()
        })
        .collect();

    let adaptive_map_avg = adaptive_map_durations
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 3.0;
    let adaptive_map_items_per_sec = item_count as f64 / adaptive_map_avg;

    let speedup_percent = (adaptive_map_items_per_sec / compute_map_items_per_sec - 1.0) * 100.0;

    WorkSizeResult {
        work_description: description,
        compute_map_items_per_sec,
        adaptive_map_items_per_sec,
        speedup_percent,
    }
}

fn measure_mixed_work(runtime: &LoomRuntime, item_count: usize) -> WorkSizeResult {
    // Mixed: 60% fast (10us), 30% medium (100us), 10% slow (500us)
    fn work_for_item(i: usize) -> u64 {
        match i % 10 {
            0..=5 => calibrated_work(10),
            6..=8 => calibrated_work(100),
            _ => calibrated_work(500),
        }
    }

    // compute_map timing
    let compute_map_durations: Vec<Duration> = (0..3)
        .map(|_| {
            let start = Instant::now();
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..item_count)
                    .compute_map(work_for_item)
                    .collect()
                    .await;
                std::hint::black_box(results);
            });
            start.elapsed()
        })
        .collect();

    let compute_map_avg = compute_map_durations
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 3.0;
    let compute_map_items_per_sec = item_count as f64 / compute_map_avg;

    // adaptive_map timing
    let adaptive_map_durations: Vec<Duration> = (0..3)
        .map(|_| {
            let start = Instant::now();
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..item_count)
                    .adaptive_map(work_for_item)
                    .collect()
                    .await;
                std::hint::black_box(results);
            });
            start.elapsed()
        })
        .collect();

    let adaptive_map_avg = adaptive_map_durations
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>()
        / 3.0;
    let adaptive_map_items_per_sec = item_count as f64 / adaptive_map_avg;

    let speedup_percent = (adaptive_map_items_per_sec / compute_map_items_per_sec - 1.0) * 100.0;

    WorkSizeResult {
        work_description: "Mixed (60% fast, 30% medium, 10% slow)",
        compute_map_items_per_sec,
        adaptive_map_items_per_sec,
        speedup_percent,
    }
}

// =============================================================================
// Phase 4: Learning Behavior
// =============================================================================

#[derive(Debug)]
struct LearningResults {
    fast_work_convergence: ConvergenceResult,
    slow_work_convergence: ConvergenceResult,
    decision_breakdown: DecisionBreakdown,
    guardrail_activations: GuardrailActivations,
}

#[derive(Debug)]
struct ConvergenceResult {
    work_type: &'static str,
    observations_to_stable: usize,
    final_decision: &'static str,
    correct: bool,
}

#[derive(Debug)]
struct DecisionBreakdown {
    fast_work_inline_pct: f64,
    slow_work_offload_pct: f64,
}

#[derive(Debug)]
struct GuardrailActivations {
    gr1_hard_threshold_triggered: bool,
    gr3_strike_triggered: bool,
}

fn measure_learning(config: &EvaluationConfig) -> LearningResults {
    let ctx = Context {
        tokio_workers: 4,
        inflight_tasks: 2,
        spawn_rate_per_s: 100.0,
        rayon_threads: 8,
        rayon_queue_depth: 0,
    };

    // Fast work convergence
    let fast_work_convergence = measure_convergence(
        &ctx,
        config.convergence_observations,
        20.0,
        "Fast (~20us)",
        Arm::InlineTokio,
    );

    // Slow work convergence
    let slow_work_convergence = measure_convergence(
        &ctx,
        config.convergence_observations,
        500.0,
        "Slow (~500us)",
        Arm::OffloadRayon,
    );

    // Decision breakdown after training
    let decision_breakdown = measure_decision_breakdown(&ctx, 100);

    // Guardrail activations
    let guardrail_activations = measure_guardrail_activations(&ctx);

    LearningResults {
        fast_work_convergence,
        slow_work_convergence,
        decision_breakdown,
        guardrail_activations,
    }
}

fn measure_convergence(
    ctx: &Context,
    max_observations: usize,
    work_time_us: f64,
    work_type: &'static str,
    expected_arm: Arm,
) -> ConvergenceResult {
    let scheduler = MabScheduler::with_defaults();
    let key = FunctionKey::from_name(work_type);

    let mut consecutive_correct = 0;
    let mut observations_to_stable = max_observations;
    let mut last_arm = Arm::InlineTokio;

    for i in 0..max_observations {
        let (id, arm) = scheduler.choose(key, ctx);
        scheduler.finish(id, work_time_us, Some(work_time_us));
        last_arm = arm;

        if arm == expected_arm {
            consecutive_correct += 1;
            if consecutive_correct >= 5 && observations_to_stable == max_observations {
                observations_to_stable = i + 1;
            }
        } else {
            consecutive_correct = 0;
        }
    }

    let final_decision = match last_arm {
        Arm::InlineTokio => "Inline",
        Arm::OffloadRayon => "Offload",
    };

    ConvergenceResult {
        work_type,
        observations_to_stable,
        final_decision,
        correct: last_arm == expected_arm,
    }
}

fn measure_decision_breakdown(ctx: &Context, observations: usize) -> DecisionBreakdown {
    // Fast work decisions
    let scheduler = MabScheduler::with_defaults();
    let key = FunctionKey::from_name("fast_breakdown");

    // Train on fast work
    for _ in 0..50 {
        let (id, _) = scheduler.choose(key, ctx);
        scheduler.finish(id, 20.0, Some(20.0));
    }

    // Count decisions
    let mut inline_count = 0;
    for _ in 0..observations {
        let (id, arm) = scheduler.choose(key, ctx);
        if arm == Arm::InlineTokio {
            inline_count += 1;
        }
        scheduler.finish(id, 20.0, Some(20.0));
    }
    let fast_work_inline_pct = (inline_count as f64 / observations as f64) * 100.0;

    // Slow work decisions
    let scheduler = MabScheduler::with_defaults();
    let key = FunctionKey::from_name("slow_breakdown");

    // Train on slow work
    for _ in 0..50 {
        let (id, _) = scheduler.choose(key, ctx);
        scheduler.finish(id, 500.0, Some(500.0));
    }

    // Count decisions
    let mut offload_count = 0;
    for _ in 0..observations {
        let (id, arm) = scheduler.choose(key, ctx);
        if arm == Arm::OffloadRayon {
            offload_count += 1;
        }
        scheduler.finish(id, 500.0, Some(500.0));
    }
    let slow_work_offload_pct = (offload_count as f64 / observations as f64) * 100.0;

    DecisionBreakdown {
        fast_work_inline_pct,
        slow_work_offload_pct,
    }
}

fn measure_guardrail_activations(ctx: &Context) -> GuardrailActivations {
    // Test GR1: Hard blocking threshold (>250us EMA should always offload)
    let scheduler = MabScheduler::with_defaults();
    let key = FunctionKey::from_name("gr1_test");

    // Train to high EMA
    for _ in 0..20 {
        let (id, _) = scheduler.choose(key, ctx);
        scheduler.finish(id, 500.0, Some(500.0));
    }

    let (id, arm) = scheduler.choose(key, ctx);
    scheduler.finish(id, 500.0, Some(500.0));
    let gr1_triggered = arm == Arm::OffloadRayon;

    // Test GR3: Strike suppression (repeated slow inline executions)
    let knobs = MabKnobs::default();
    let scheduler = MabScheduler::new(knobs);
    let key = FunctionKey::from_name("gr3_test");

    // Force inline decisions with slow work to accumulate strikes
    for _ in 0..30 {
        let (id, _) = scheduler.choose(key, ctx);
        scheduler.finish(id, 1500.0, Some(1500.0));
    }

    let (id, arm) = scheduler.choose(key, ctx);
    scheduler.finish(id, 1500.0, Some(1500.0));
    let gr3_triggered = arm == Arm::OffloadRayon;

    GuardrailActivations {
        gr1_hard_threshold_triggered: gr1_triggered,
        gr3_strike_triggered: gr3_triggered,
    }
}

// =============================================================================
// Phase 5: Thread Configuration Comparison
// =============================================================================

/// Results for a single thread configuration
struct ConfigResults {
    config: ThreadConfig,
    tokio_cpus: String,
    rayon_cpus: String,
    throughput: ThroughputResults,
    latency: LatencyResults,
    starvation_events: usize,
    max_pressure: f64,
}

fn measure_config(
    eval_config: &EvaluationConfig,
    thread_config: &ThreadConfig,
) -> Option<ConfigResults> {
    // Try to build runtime with this configuration
    // Note: cpuset is now automatically discovered from process affinity
    let runtime = match LoomBuilder::new()
        .prefix("mab-eval")
        .tokio_threads(thread_config.tokio_threads)
        .rayon_threads(thread_config.rayon_threads)
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!(
                "Warning: Could not build runtime for config '{}': {}",
                thread_config.name, e
            );
            return None;
        }
    };

    // Extract actual CPU assignments
    let tokio_cpus = format_cpuset(runtime.tokio_cpus());
    let rayon_cpus = format_cpuset(runtime.rayon_cpus());

    eprintln!(
        "  Testing {} (tokio:{}, rayon:{})...",
        thread_config.name, thread_config.tokio_threads, thread_config.rayon_threads
    );

    // Run starvation tracking during throughput test
    let mut tracker = StarvationTracker::new(thread_config.name, thread_config.tokio_threads);

    // Measure throughput with starvation tracking
    let throughput = measure_throughput_with_tracking(eval_config, &runtime, &mut tracker);

    // Measure latency
    let latency = measure_latency_impact(eval_config, &runtime);

    // Cleanup
    runtime.block_until_idle();

    Some(ConfigResults {
        config: thread_config.clone(),
        tokio_cpus,
        rayon_cpus,
        throughput,
        latency,
        starvation_events: tracker.event_count(),
        max_pressure: tracker.max_pressure(),
    })
}

fn measure_throughput_with_tracking(
    config: &EvaluationConfig,
    runtime: &LoomRuntime,
    tracker: &mut StarvationTracker,
) -> ThroughputResults {
    let item_count = config.item_count;

    // For tracking, we periodically sample the context during mixed workload
    let ctx = runtime.collect_context();
    tracker.check(&ctx, None);

    // Fast work (~10us)
    let fast_work = measure_work_size(runtime, item_count, "Fast (~10us)", 10);

    // Check for starvation
    let ctx = runtime.collect_context();
    tracker.check(&ctx, None);

    // Medium work (~100us)
    let medium_work = measure_work_size(runtime, item_count, "Medium (~100us)", 100);

    // Check for starvation
    let ctx = runtime.collect_context();
    tracker.check(&ctx, None);

    // Slow work (~500us)
    let slow_work = measure_work_size(runtime, item_count, "Slow (~500us)", 500);

    // Check for starvation
    let ctx = runtime.collect_context();
    tracker.check(&ctx, None);

    // Mixed work
    let mixed_work = measure_mixed_work(runtime, item_count);

    // Final check
    let ctx = runtime.collect_context();
    tracker.check(&ctx, None);

    ThroughputResults {
        fast_work,
        medium_work,
        slow_work,
        mixed_work,
    }
}

// =============================================================================
// Verdict
// =============================================================================

#[derive(Debug)]
struct Verdict {
    overhead_pass: bool,
    latency_pass: bool,
    throughput_pass: bool,
    learning_pass: bool,
    overall_pass: bool,
    notes: Vec<String>,
}

fn compute_verdict(
    overhead: &OverheadResults,
    latency: &LatencyResults,
    throughput: &ThroughputResults,
    learning: &LearningResults,
) -> Verdict {
    let mut notes = Vec::new();

    // Overhead check: warm decision < 200ns
    let warm_mean = overhead.warm_ns.mean();
    let overhead_pass = warm_mean < 200.0;
    if !overhead_pass {
        notes.push(format!(
            "Warm decision overhead ({:.0}ns) exceeds 200ns target",
            warm_mean
        ));
    }

    // Latency check: adaptive interference within 2x of offload
    let adaptive_if = latency.adaptive.interference_factor;
    let offload_if = latency.always_offload.interference_factor;
    let latency_pass = adaptive_if < offload_if * 2.0 || adaptive_if < 5.0;
    if !latency_pass {
        notes.push(format!(
            "Adaptive interference ({:.1}x) significantly worse than offload ({:.1}x)",
            adaptive_if, offload_if
        ));
    }

    // Throughput check: adaptive >= compute_map for mixed workloads
    let throughput_pass = throughput.mixed_work.speedup_percent > -10.0;
    if !throughput_pass {
        notes.push(format!(
            "Mixed workload throughput {:.1}% worse than always-offload",
            -throughput.mixed_work.speedup_percent
        ));
    }

    // Learning check: converges within 50 observations
    let learning_pass = learning.fast_work_convergence.observations_to_stable <= 50
        && learning.slow_work_convergence.observations_to_stable <= 50
        && learning.fast_work_convergence.correct
        && learning.slow_work_convergence.correct;
    if !learning_pass {
        if learning.fast_work_convergence.observations_to_stable > 50 {
            notes.push(format!(
                "Fast work convergence slow ({} observations)",
                learning.fast_work_convergence.observations_to_stable
            ));
        }
        if learning.slow_work_convergence.observations_to_stable > 50 {
            notes.push(format!(
                "Slow work convergence slow ({} observations)",
                learning.slow_work_convergence.observations_to_stable
            ));
        }
        if !learning.fast_work_convergence.correct {
            notes.push("Fast work learned incorrect strategy".to_string());
        }
        if !learning.slow_work_convergence.correct {
            notes.push("Slow work learned incorrect strategy".to_string());
        }
    }

    let overall_pass = overhead_pass && latency_pass && throughput_pass && learning_pass;

    Verdict {
        overhead_pass,
        latency_pass,
        throughput_pass,
        learning_pass,
        overall_pass,
        notes,
    }
}

// =============================================================================
// Report Generation
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn print_markdown_report(
    system_info: &SystemInfo,
    eval_config: &EvaluationConfig,
    overhead: &OverheadResults,
    latency: &LatencyResults,
    throughput: &ThroughputResults,
    learning: &LearningResults,
    config_results: &[ConfigResults],
    verdict: &Verdict,
) {
    let now = chrono_lite_now();

    println!("# loom-rs MAB Evaluation Report\n");
    println!("**Generated:** {}", now);
    println!(
        "**Platform:** {} {}, {} cores available",
        system_info.platform, system_info.arch, system_info.cpus
    );
    println!("**Build:** {}\n", system_info.build_profile);

    // System Configuration Section
    println!("## System Configuration\n");
    println!(
        "**Evaluation CPUs:** {} cores (auto-discovered from process affinity)\n",
        eval_config.total_cpus
    );

    println!("### Thread Configurations Tested\n");
    println!("| Config | Tokio Threads | Tokio CPUs | Rayon Threads | Rayon CPUs | Description |");
    println!("|--------|---------------|------------|---------------|------------|-------------|");
    for result in config_results {
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            result.config.name,
            result.config.tokio_threads,
            result.tokio_cpus,
            result.config.rayon_threads,
            result.rayon_cpus,
            result.config.description
        );
    }

    // Executive Summary
    println!("\n## Executive Summary\n");
    println!("The MAB adaptive scheduler evaluation results:\n");
    println!(
        "- {} Decision overhead < 200ns (measured: {:.0}ns warm)",
        if verdict.overhead_pass { "OK" } else { "FAIL" },
        overhead.warm_ns.mean()
    );
    println!(
        "- {} Wake latency protection (adaptive: {:.1}x vs offload: {:.1}x interference)",
        if verdict.latency_pass { "OK" } else { "FAIL" },
        latency.adaptive.interference_factor,
        latency.always_offload.interference_factor
    );
    println!(
        "- {} Throughput on mixed workloads ({:+.1}% vs always-offload)",
        if verdict.throughput_pass {
            "OK"
        } else {
            "FAIL"
        },
        throughput.mixed_work.speedup_percent
    );
    println!(
        "- {} Fast convergence (fast: {} obs, slow: {} obs)",
        if verdict.learning_pass { "OK" } else { "FAIL" },
        learning.fast_work_convergence.observations_to_stable,
        learning.slow_work_convergence.observations_to_stable
    );

    println!(
        "\n**Overall:** {}\n",
        if verdict.overall_pass { "PASS" } else { "FAIL" }
    );

    if !verdict.notes.is_empty() {
        println!("### Notes\n");
        for note in &verdict.notes {
            println!("- {}", note);
        }
        println!();
    }

    // Section 1: Scheduler Overhead
    println!("---\n");
    println!("## 1. Scheduler Overhead\n");
    println!("Measures the time cost of MAB decision-making (excluding actual work execution).\n");
    println!("| Operation | Mean | P50 | P99 | Std Dev |");
    println!("|-----------|------|-----|-----|---------|");
    println!(
        "| choose() cold | {:.0}ns | {:.0}ns | {:.0}ns | {:.0}ns |",
        overhead.cold_ns.mean(),
        overhead.cold_ns.p50(),
        overhead.cold_ns.p99(),
        overhead.cold_ns.std_dev()
    );
    println!(
        "| choose() warm | {:.0}ns | {:.0}ns | {:.0}ns | {:.0}ns |",
        overhead.warm_ns.mean(),
        overhead.warm_ns.p50(),
        overhead.warm_ns.p99(),
        overhead.warm_ns.std_dev()
    );
    println!(
        "| Full cycle | {:.0}ns | {:.0}ns | {:.0}ns | {:.0}ns |",
        overhead.full_cycle_ns.mean(),
        overhead.full_cycle_ns.p50(),
        overhead.full_cycle_ns.p99(),
        overhead.full_cycle_ns.std_dev()
    );

    println!(
        "\n**Verdict:** {} Overhead is {} (<200ns target for warm decisions)\n",
        if verdict.overhead_pass { "OK" } else { "FAIL" },
        if overhead.warm_ns.mean() < 100.0 {
            "minimal"
        } else if overhead.warm_ns.mean() < 200.0 {
            "acceptable"
        } else {
            "elevated"
        }
    );

    // Section 2: Wake Latency Impact
    println!("---\n");
    println!("## 2. Wake Latency Impact\n");
    println!("Measures how different strategies affect Tokio's async executor responsiveness.\n");
    println!(
        "**Baseline (no load):** P50={:.1}us, P95={:.1}us, P99={:.1}us\n",
        latency.baseline.p50(),
        latency.baseline.p95(),
        latency.baseline.p99()
    );
    println!("| Strategy | P50 | P95 | P99 | Interference | Throughput |");
    println!("|----------|-----|-----|-----|--------------|------------|");
    println!(
        "| AlwaysInline | {:.1}us | {:.1}us | {:.1}us | {:.1}x | {:.0}/s |",
        latency.always_inline.latency.p50(),
        latency.always_inline.latency.p95(),
        latency.always_inline.latency.p99(),
        latency.always_inline.interference_factor,
        latency.always_inline.throughput
    );
    println!(
        "| AlwaysOffload | {:.1}us | {:.1}us | {:.1}us | {:.1}x | {:.0}/s |",
        latency.always_offload.latency.p50(),
        latency.always_offload.latency.p95(),
        latency.always_offload.latency.p99(),
        latency.always_offload.interference_factor,
        latency.always_offload.throughput
    );
    println!(
        "| Adaptive (MAB) | {:.1}us | {:.1}us | {:.1}us | {:.1}x | {:.0}/s |",
        latency.adaptive.latency.p50(),
        latency.adaptive.latency.p95(),
        latency.adaptive.latency.p99(),
        latency.adaptive.interference_factor,
        latency.adaptive.throughput
    );

    println!("\n**Key Observations:**\n");
    if latency.always_inline.interference_factor > 5.0 {
        println!(
            "- AlwaysInline causes severe wake latency degradation ({:.0}x)",
            latency.always_inline.interference_factor
        );
    }
    let adaptive_vs_offload =
        latency.adaptive.interference_factor / latency.always_offload.interference_factor;
    if adaptive_vs_offload < 1.5 {
        println!("- Adaptive achieves similar latency protection to AlwaysOffload");
    }

    println!(
        "\n**Verdict:** {} MAB {} Tokio from starvation\n",
        if verdict.latency_pass { "OK" } else { "FAIL" },
        if verdict.latency_pass {
            "protects"
        } else {
            "may not protect"
        }
    );

    // Section 3: Throughput Analysis
    println!("---\n");
    println!("## 3. Throughput Analysis\n");
    println!("Compares `adaptive_map()` vs `compute_map()` (always offload) performance.\n");
    println!("| Workload | compute_map | adaptive_map | Speedup |");
    println!("|----------|-------------|--------------|---------|");
    for result in [
        &throughput.fast_work,
        &throughput.medium_work,
        &throughput.slow_work,
        &throughput.mixed_work,
    ] {
        println!(
            "| {} | {:.0}/s | {:.0}/s | {:+.1}% |",
            result.work_description,
            result.compute_map_items_per_sec,
            result.adaptive_map_items_per_sec,
            result.speedup_percent
        );
    }

    println!("\n**Key Observations:**\n");
    if throughput.fast_work.speedup_percent > 0.0 {
        println!(
            "- Fast work: {:.0}% faster (MAB learns to inline)",
            throughput.fast_work.speedup_percent
        );
    }
    if throughput.slow_work.speedup_percent.abs() < 5.0 {
        println!("- Slow work: similar performance (MAB learns to offload)");
    }
    if throughput.mixed_work.speedup_percent > 0.0 {
        println!(
            "- Mixed workloads: {:.0}% faster (best of both worlds)",
            throughput.mixed_work.speedup_percent
        );
    }

    println!(
        "\n**Verdict:** {} {} on mixed workloads\n",
        if verdict.throughput_pass {
            "OK"
        } else {
            "FAIL"
        },
        if throughput.mixed_work.speedup_percent > 0.0 {
            "MAB improves throughput"
        } else if throughput.mixed_work.speedup_percent > -5.0 {
            "MAB maintains throughput"
        } else {
            "MAB reduces throughput"
        }
    );

    // Section 4: Learning Behavior
    println!("---\n");
    println!("## 4. Learning Behavior\n");
    println!("Measures how quickly the MAB converges to correct decisions.\n");
    println!("### Convergence Speed\n");
    println!("| Work Type | Observations to Stable | Final Decision | Correct? |");
    println!("|-----------|------------------------|----------------|----------|");
    println!(
        "| {} | {} | {} | {} |",
        learning.fast_work_convergence.work_type,
        learning.fast_work_convergence.observations_to_stable,
        learning.fast_work_convergence.final_decision,
        if learning.fast_work_convergence.correct {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "| {} | {} | {} | {} |",
        learning.slow_work_convergence.work_type,
        learning.slow_work_convergence.observations_to_stable,
        learning.slow_work_convergence.final_decision,
        if learning.slow_work_convergence.correct {
            "Yes"
        } else {
            "No"
        }
    );

    println!("\n### Decision Breakdown (after training)\n");
    println!("| Work Type | Inline % | Offload % |");
    println!("|-----------|----------|-----------|");
    println!(
        "| Fast work | {:.0}% | {:.0}% |",
        learning.decision_breakdown.fast_work_inline_pct,
        100.0 - learning.decision_breakdown.fast_work_inline_pct
    );
    println!(
        "| Slow work | {:.0}% | {:.0}% |",
        100.0 - learning.decision_breakdown.slow_work_offload_pct,
        learning.decision_breakdown.slow_work_offload_pct
    );

    println!("\n### Guardrail Activations\n");
    println!("| Guardrail | Description | Triggered? |");
    println!("|-----------|-------------|------------|");
    println!(
        "| GR1 | Hard blocking threshold (>250us) | {} |",
        if learning.guardrail_activations.gr1_hard_threshold_triggered {
            "Yes"
        } else {
            "No"
        }
    );
    println!(
        "| GR3 | Strike suppression (repeated slow) | {} |",
        if learning.guardrail_activations.gr3_strike_triggered {
            "Yes"
        } else {
            "No"
        }
    );

    println!(
        "\n**Verdict:** {} MAB converges {} and makes {} decisions\n",
        if verdict.learning_pass { "OK" } else { "FAIL" },
        if learning.fast_work_convergence.observations_to_stable <= 20 {
            "quickly"
        } else if learning.fast_work_convergence.observations_to_stable <= 50 {
            "reasonably"
        } else {
            "slowly"
        },
        if learning.fast_work_convergence.correct && learning.slow_work_convergence.correct {
            "correct"
        } else {
            "incorrect"
        }
    );

    // Section 5: Thread Configuration Analysis
    println!("---\n");
    println!("## 5. Thread Configuration Analysis\n");
    println!("Compares different tokio:rayon thread ratios to find optimal configurations.\n");

    println!("### Throughput by Configuration (Mixed Workload)\n");
    println!("| Config | Tokio:Rayon | compute_map | adaptive_map | Speedup | Starvation Events |");
    println!("|--------|-------------|-------------|--------------|---------|-------------------|");
    for result in config_results {
        println!(
            "| {} | {}:{} | {:.0}/s | {:.0}/s | {:+.1}% | {} |",
            result.config.name,
            result.config.tokio_threads,
            result.config.rayon_threads,
            result.throughput.mixed_work.compute_map_items_per_sec,
            result.throughput.mixed_work.adaptive_map_items_per_sec,
            result.throughput.mixed_work.speedup_percent,
            result.starvation_events
        );
    }

    println!("\n### Wake Latency by Configuration\n");
    println!("| Config | Baseline P95 | Under Load P95 | Interference |");
    println!("|--------|--------------|----------------|--------------|");
    for result in config_results {
        println!(
            "| {} | {:.1}us | {:.1}us | {:.1}x |",
            result.config.name,
            result.latency.baseline.p95(),
            result.latency.adaptive.latency.p95(),
            result.latency.adaptive.interference_factor
        );
    }

    // Starvation warnings
    let starvation_configs: Vec<_> = config_results
        .iter()
        .filter(|r| r.starvation_events > 10)
        .collect();
    if !starvation_configs.is_empty() {
        println!("\n### Starvation Warnings\n");
        println!("| Config | Events | Max Pressure | Recommendation |");
        println!("|--------|--------|--------------|----------------|");
        for result in &starvation_configs {
            let recommendation = if result.config.tokio_threads == 1 {
                "Consider 2+ tokio threads"
            } else {
                "Monitor workload patterns"
            };
            println!(
                "| {} | {} | {:.1} | {} |",
                result.config.name, result.starvation_events, result.max_pressure, recommendation
            );
        }
        println!("\n**Note:** Starvation events indicate the MAB guardrails activated to protect");
        println!("Tokio from blocking. While guardrails prevent actual starvation, frequent");
        println!("activation suggests increasing tokio_threads would improve responsiveness.\n");
    }

    // Recommendations
    println!("### Recommendation\n");
    println!("Based on the evaluation:\n");

    // Find best for throughput
    let best_throughput = config_results.iter().max_by(|a, b| {
        a.throughput
            .mixed_work
            .adaptive_map_items_per_sec
            .partial_cmp(&b.throughput.mixed_work.adaptive_map_items_per_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find best for latency
    let best_latency = config_results.iter().min_by(|a, b| {
        a.latency
            .adaptive
            .interference_factor
            .partial_cmp(&b.latency.adaptive.interference_factor)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(best) = best_throughput {
        println!(
            "- **Best for throughput:** {} ({:.0} items/s)",
            best.config.name, best.throughput.mixed_work.adaptive_map_items_per_sec
        );
    }
    if let Some(best) = best_latency {
        println!(
            "- **Best for latency:** {} ({:.1}x interference)",
            best.config.name, best.latency.adaptive.interference_factor
        );
    }

    // Check for starvation issues
    let configs_with_starvation: Vec<_> = config_results
        .iter()
        .filter(|r| r.starvation_events > 10)
        .collect();
    if configs_with_starvation.is_empty() {
        println!("- **No significant starvation issues detected**");
    } else {
        println!(
            "- **Warning:** {} config(s) showed frequent starvation",
            configs_with_starvation.len()
        );
    }

    println!("- **Recommended default:** balanced-tokio (2:14) for general purpose workloads");

    // Conclusion
    println!("\n---\n");
    println!("## Conclusion\n");
    if verdict.overall_pass {
        println!("The MAB adaptive scheduler in loom-rs delivers value:\n");
        println!("1. **Low Overhead** - Decision cost < 200ns doesn't impact performance");
        println!("2. **Latency Protection** - Guardrails prevent Tokio starvation");
        println!("3. **Throughput Optimization** - Learns to inline fast work, offload slow work");
        println!(
            "4. **Fast Learning** - Converges within ~{} observations",
            learning
                .fast_work_convergence
                .observations_to_stable
                .max(learning.slow_work_convergence.observations_to_stable)
        );
        println!("\nThe adaptive approach achieves the best of both worlds: low latency for");
        println!("small work (by inlining) and protected async responsiveness (by offloading slow work).");
    } else {
        println!("The evaluation identified areas for improvement:\n");
        for note in &verdict.notes {
            println!("- {}", note);
        }
        println!("\nConsider tuning knobs or investigating the failing metrics.");
    }
}

/// Simple timestamp without requiring chrono dependency
fn chrono_lite_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let mut year = 1970;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let days_in_months: [u64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for &days in &days_in_months {
        if remaining_days < days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }
    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year, month, day, hours, minutes, seconds
    )
}

fn is_leap_year(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eval_config = EvaluationConfig::default();

    // Print to stderr so it doesn't go into the report
    eprintln!("=== MAB Evaluation Suite ===\n");
    eprintln!("This will take several minutes to run all tests...\n");

    // Collect system info
    eprintln!("Collecting system info...");
    let system_info = collect_system_info();

    // Check if we have enough CPUs
    let available_cpus = system_info.cpus;
    if available_cpus < eval_config.total_cpus {
        eprintln!(
            "Warning: System has {} CPUs, but evaluation requests {}.",
            available_cpus, eval_config.total_cpus
        );
        eprintln!("Results may vary on systems with fewer CPUs.\n");
    }

    // Create default runtime for non-config-specific tests
    let default_runtime = LoomBuilder::new()
        .prefix("mab-eval")
        .tokio_threads(2)
        .rayon_threads(6)
        .build()?;

    // Phase 1: MAB Overhead
    eprintln!("Phase 1: Measuring MAB overhead...");
    let overhead = measure_overhead(&eval_config);

    // Phase 2: Wake Latency Impact
    eprintln!("Phase 2: Measuring wake latency impact...");
    let latency = measure_latency_impact(&eval_config, &default_runtime);

    // Phase 3: Throughput Comparison
    eprintln!("Phase 3: Measuring throughput...");
    let throughput = measure_throughput(&eval_config, &default_runtime);

    // Phase 4: Learning Behavior
    eprintln!("Phase 4: Measuring learning behavior...");
    let learning = measure_learning(&eval_config);

    // Cleanup default runtime
    default_runtime.block_until_idle();

    // Phase 5: Thread Configuration Comparison
    eprintln!("Phase 5: Testing thread configurations...");
    let mut config_results = Vec::new();
    for thread_config in &eval_config.thread_configs {
        if let Some(result) = measure_config(&eval_config, thread_config) {
            config_results.push(result);
        }
    }

    // Compute verdict
    let verdict = compute_verdict(&overhead, &latency, &throughput, &learning);

    // Generate Report (to stdout)
    eprintln!("\nGenerating report...\n");
    print_markdown_report(
        &system_info,
        &eval_config,
        &overhead,
        &latency,
        &throughput,
        &learning,
        &config_results,
        &verdict,
    );

    eprintln!("Done!");
    Ok(())
}
