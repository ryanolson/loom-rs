//! Wake Latency Benchmark
//!
//! Measures the impact of compute work on Tokio's async executor by tracking
//! the drift between expected and actual sleep times under different strategies.
//!
//! This benchmark demonstrates:
//! 1. **AlwaysInline**: Severe wake latency degradation when compute blocks Tokio workers
//! 2. **AlwaysOffload**: Low interference but constant offload overhead
//! 3. **Adaptive (MAB)**: Learns to offload slow work, achieving best of both worlds
//!
//! Run: cargo run --example wake_latency_benchmark --release

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use loom_rs::LoomBuilder;
use parking_lot::Mutex;

/// Strategy for handling compute work
#[derive(Clone, Copy, Debug)]
enum Strategy {
    /// Execute compute work directly on Tokio worker thread
    AlwaysInline,
    /// Always offload compute work to Rayon
    AlwaysOffload,
    /// Use MAB to adaptively choose
    Adaptive,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::AlwaysInline => write!(f, "AlwaysInline"),
            Strategy::AlwaysOffload => write!(f, "AlwaysOffload"),
            Strategy::Adaptive => write!(f, "Adaptive (MAB)"),
        }
    }
}

/// Configuration for the benchmark
struct BenchConfig {
    /// Number of Tokio worker threads
    tokio_threads: usize,
    /// Number of Rayon threads
    rayon_threads: usize,
    /// How long each compute task takes (microseconds)
    work_duration_us: u64,
    /// Target rate of compute tasks per second
    tasks_per_sec: u64,
    /// How long to run each phase
    phase_duration: Duration,
    /// Expected sleep time for latency probes
    probe_sleep: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            tokio_threads: 2,
            rayon_threads: 6,
            work_duration_us: 500,
            tasks_per_sec: 500,
            phase_duration: Duration::from_secs(5),
            probe_sleep: Duration::from_millis(1),
        }
    }
}

/// Statistics collected from latency measurements
#[derive(Clone, Debug)]
struct LatencyStats {
    samples: Vec<f64>,
}

impl LatencyStats {
    fn new() -> Self {
        Self {
            samples: Vec::with_capacity(10000),
        }
    }

    fn add(&mut self, drift_us: f64) {
        self.samples.push(drift_us);
    }

    fn percentile(&self, pct: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    fn count(&self) -> usize {
        self.samples.len()
    }
}

/// Results from running a strategy
struct StrategyResult {
    strategy: Strategy,
    latency: LatencyStats,
    completed_tasks: u64,
    duration: Duration,
}

impl StrategyResult {
    fn throughput(&self) -> f64 {
        self.completed_tasks as f64 / self.duration.as_secs_f64()
    }

    fn interference_factor(&self, baseline_p95: f64) -> f64 {
        if baseline_p95 == 0.0 {
            1.0
        } else {
            self.latency.p95() / baseline_p95
        }
    }
}

/// Calibrated work function that runs for approximately the specified duration
#[inline(never)]
fn calibrated_work(target_us: u64) -> u64 {
    // Empirically calibrated: ~1 iteration ≈ 10ns on typical hardware
    // This will vary by CPU, but relative comparisons remain valid
    let iterations = target_us * 100;
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(std::hint::black_box(i));
    }
    sum
}

/// Async probe that measures wake latency drift
async fn latency_probe(
    expected_sleep: Duration,
    duration: Duration,
    samples: Arc<Mutex<LatencyStats>>,
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

/// Run the baseline measurement (no compute load)
async fn run_baseline(config: &BenchConfig) -> LatencyStats {
    println!("  Running baseline (no compute load)...");

    let samples = Arc::new(Mutex::new(LatencyStats::new()));
    let running = Arc::new(AtomicBool::new(true));

    // Just run probes, no compute load
    latency_probe(
        config.probe_sleep,
        config.phase_duration,
        samples.clone(),
        running.clone(),
    )
    .await;

    running.store(false, Ordering::Relaxed);
    let stats = samples.lock().clone();
    println!(
        "    {} samples, P50={:.1}µs, P95={:.1}µs, P99={:.1}µs",
        stats.count(),
        stats.p50(),
        stats.p95(),
        stats.p99()
    );
    stats
}

/// Run a strategy and measure its impact
async fn run_strategy(
    runtime: &loom_rs::LoomRuntime,
    config: &BenchConfig,
    strategy: Strategy,
) -> StrategyResult {
    println!("  Running {:}...", strategy);

    let samples = Arc::new(Mutex::new(LatencyStats::new()));
    let running = Arc::new(AtomicBool::new(true));
    let completed = Arc::new(AtomicU64::new(0));

    let probe_samples = samples.clone();
    let probe_running = running.clone();
    let probe_sleep = config.probe_sleep;
    let phase_duration = config.phase_duration;

    // Start the latency probe
    let probe_handle = runtime.spawn_async(async move {
        latency_probe(probe_sleep, phase_duration, probe_samples, probe_running).await;
    });

    // Generate compute load
    let start = Instant::now();
    let interval = Duration::from_micros(1_000_000 / config.tasks_per_sec);
    let work_us = config.work_duration_us;

    match strategy {
        Strategy::AlwaysInline => {
            // Execute compute work directly on Tokio worker
            while Instant::now() < start + config.phase_duration {
                let task_start = Instant::now();
                std::hint::black_box(calibrated_work(work_us));
                completed.fetch_add(1, Ordering::Relaxed);

                let elapsed = task_start.elapsed();
                if elapsed < interval {
                    tokio::time::sleep(interval - elapsed).await;
                }
            }
        }
        Strategy::AlwaysOffload => {
            // Always offload to Rayon via spawn_compute
            while Instant::now() < start + config.phase_duration {
                let task_start = Instant::now();
                runtime
                    .spawn_compute(move || calibrated_work(work_us))
                    .await;
                completed.fetch_add(1, Ordering::Relaxed);

                let elapsed = task_start.elapsed();
                if elapsed < interval {
                    tokio::time::sleep(interval - elapsed).await;
                }
            }
        }
        Strategy::Adaptive => {
            // Use spawn_adaptive which employs MAB scheduling
            while Instant::now() < start + config.phase_duration {
                let task_start = Instant::now();
                runtime
                    .spawn_adaptive(move || calibrated_work(work_us))
                    .await;
                completed.fetch_add(1, Ordering::Relaxed);

                let elapsed = task_start.elapsed();
                if elapsed < interval {
                    tokio::time::sleep(interval - elapsed).await;
                }
            }
        }
    }

    running.store(false, Ordering::Relaxed);
    let _ = probe_handle.await;

    let duration = start.elapsed();
    let latency = samples.lock().clone();
    let completed_tasks = completed.load(Ordering::Relaxed);

    println!(
        "    {} samples, P50={:.1}µs, P95={:.1}µs, P99={:.1}µs, throughput={:.0}/s",
        latency.count(),
        latency.p50(),
        latency.p95(),
        latency.p99(),
        completed_tasks as f64 / duration.as_secs_f64()
    );

    StrategyResult {
        strategy,
        latency,
        completed_tasks,
        duration,
    }
}

fn print_results(config: &BenchConfig, baseline: &LatencyStats, results: &[StrategyResult]) {
    println!("\n{}", "=".repeat(85));
    println!("Wake Latency Benchmark Results");
    println!("{}", "=".repeat(85));
    println!(
        "\nConfig: {} tokio + {} rayon threads, compute work ~{}µs, rate {}/sec\n",
        config.tokio_threads, config.rayon_threads, config.work_duration_us, config.tasks_per_sec
    );

    println!("Baseline (no load):");
    println!(
        "  P50: {:.1}µs, P95: {:.1}µs, P99: {:.1}µs\n",
        baseline.p50(),
        baseline.p95(),
        baseline.p99()
    );

    println!("Under Load:");
    println!(
        "| {:<15} | {:>9} | {:>9} | {:>9} | {:>12} | {:>10} |",
        "Strategy", "P50", "P95", "P99", "Interference", "Throughput"
    );
    println!(
        "|{:-<17}|{:-<11}|{:-<11}|{:-<11}|{:-<14}|{:-<12}|",
        "", "", "", "", "", ""
    );

    for result in results {
        let interference = result.interference_factor(baseline.p95());
        println!(
            "| {:<15} | {:>7.1}µs | {:>7.1}µs | {:>7.1}µs | {:>10.1}x | {:>8.0}/s |",
            format!("{}", result.strategy),
            result.latency.p50(),
            result.latency.p95(),
            result.latency.p99(),
            interference,
            result.throughput()
        );
    }

    println!("\nKey Insights:");
    if let Some(inline) = results
        .iter()
        .find(|r| matches!(r.strategy, Strategy::AlwaysInline))
    {
        let interference = inline.interference_factor(baseline.p95());
        if interference > 10.0 {
            println!(
                "  - AlwaysInline causes {:.0}x wake latency degradation",
                interference
            );
        }
    }

    if let (Some(offload), Some(adaptive)) = (
        results
            .iter()
            .find(|r| matches!(r.strategy, Strategy::AlwaysOffload)),
        results
            .iter()
            .find(|r| matches!(r.strategy, Strategy::Adaptive)),
    ) {
        let offload_if = offload.interference_factor(baseline.p95());
        let adaptive_if = adaptive.interference_factor(baseline.p95());
        println!(
            "  - AlwaysOffload: {:.1}x interference, {:.0}/s throughput",
            offload_if,
            offload.throughput()
        );
        println!(
            "  - Adaptive MAB: {:.1}x interference, {:.0}/s throughput",
            adaptive_if,
            adaptive.throughput()
        );

        if (offload_if - adaptive_if).abs() < 1.0 {
            println!(
                "  - MAB achieves similar latency to AlwaysOffload (learned to offload slow work)"
            );
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse optional CLI arguments for configuration
    let config = BenchConfig::default();

    println!("=== Wake Latency Benchmark ===\n");
    println!("This benchmark measures how compute work impacts Tokio's async executor");
    println!("by tracking the drift between expected and actual sleep times.\n");

    // Create the runtime
    let runtime = LoomBuilder::new()
        .prefix("wake-bench")
        .tokio_threads(config.tokio_threads)
        .rayon_threads(config.rayon_threads)
        .build()?;

    println!(
        "Runtime: {} tokio threads, {} rayon threads",
        config.tokio_threads, config.rayon_threads
    );
    println!(
        "Workload: ~{}µs compute tasks at {}/sec",
        config.work_duration_us, config.tasks_per_sec
    );
    println!("Phase duration: {:?}\n", config.phase_duration);

    // Run the benchmark
    let (baseline, results) = runtime.block_on(async {
        // Phase 1: Baseline
        println!("Phase 1: Baseline measurement");
        let baseline = run_baseline(&config).await;

        // Phase 2: Strategy comparison
        println!("\nPhase 2: Strategy comparison");
        let mut results = Vec::new();

        for strategy in [
            Strategy::AlwaysInline,
            Strategy::AlwaysOffload,
            Strategy::Adaptive,
        ] {
            let result = run_strategy(&runtime, &config, strategy).await;
            results.push(result);
        }

        (baseline, results)
    });

    // Print results
    print_results(&config, &baseline, &results);

    runtime.block_until_idle();
    Ok(())
}
