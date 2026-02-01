//! Benchmarks for the Multi-Armed Bandit (MAB) adaptive scheduler.
//!
//! Run with: cargo bench -- mab
//!
//! These benchmarks measure:
//! - MAB decision overhead (choose + finish)
//! - Strategy comparison across work sizes
//! - MAB convergence speed
//! - Adaptive stream performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::stream::{self, StreamExt};
use loom_rs::mab::{Arm, ComputeHint, Context, FunctionKey, MabKnobs, MabScheduler};
use loom_rs::{ComputeStreamExt, LoomBuilder};

/// Default context for benchmarks
fn default_ctx() -> Context {
    Context {
        tokio_workers: 4,
        inflight_tasks: 2,
        spawn_rate_per_s: 100.0,
        rayon_threads: 8,
        rayon_queue_depth: 0,
    }
}

/// Calibrated work function that runs for approximately the specified duration
#[inline(never)]
fn calibrated_work(target_us: u64) -> u64 {
    // ~100 iterations ≈ 1µs on typical hardware
    let iterations = target_us * 100;
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(std::hint::black_box(i));
    }
    sum
}

/// Small work (~10µs)
#[inline(never)]
fn small_work() -> u64 {
    calibrated_work(10)
}

/// Medium work (~100µs)
#[inline(never)]
fn medium_work() -> u64 {
    calibrated_work(100)
}

/// Large work (~500µs)
#[inline(never)]
fn large_work() -> u64 {
    calibrated_work(500)
}

fn create_runtime() -> loom_rs::LoomRuntime {
    LoomBuilder::new()
        .prefix("mab-bench")
        .tokio_threads(2)
        .rayon_threads(4)
        .build()
        .expect("failed to create runtime")
}

// =============================================================================
// MAB Overhead Benchmarks
// =============================================================================

/// Benchmark MAB decision overhead
fn bench_mab_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("mab_overhead");

    // Cold decision (new key each time)
    group.bench_function("choose_cold", |b| {
        let scheduler = MabScheduler::with_defaults();
        let ctx = default_ctx();
        let mut key_counter = 0u64;

        b.iter(|| {
            let key = FunctionKey(key_counter);
            key_counter += 1;
            let (id, arm) = scheduler.choose(key, &ctx);
            black_box((id, arm))
        });
    });

    // Warm decision (same key, stats exist)
    group.bench_function("choose_warm", |b| {
        let scheduler = MabScheduler::with_defaults();
        let ctx = default_ctx();
        let key = FunctionKey(1);

        // Warmup: create stats for this key
        for _ in 0..100 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        b.iter(|| {
            let (id, arm) = scheduler.choose(key, &ctx);
            black_box((id, arm))
        });
    });

    // Full cycle: choose + finish
    group.bench_function("choose_and_finish", |b| {
        let scheduler = MabScheduler::with_defaults();
        let ctx = default_ctx();
        let key = FunctionKey(2);

        // Warmup
        for _ in 0..100 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        }

        b.iter(|| {
            let (id, _arm) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 50.0, Some(50.0));
        });
    });

    // Choose with hint (cold)
    group.bench_function("choose_with_hint_cold", |b| {
        let scheduler = MabScheduler::with_defaults();
        let ctx = default_ctx();
        let mut key_counter = 1000u64;

        b.iter(|| {
            let key = FunctionKey(key_counter);
            key_counter += 1;
            let (id, arm) = scheduler.choose_with_hint(key, &ctx, ComputeHint::Medium);
            black_box((id, arm))
        });
    });

    // Choose with High hint (forces initial offloads)
    group.bench_function("choose_with_high_hint", |b| {
        let scheduler = MabScheduler::with_defaults();
        let ctx = default_ctx();
        let mut key_counter = 2000u64;

        b.iter(|| {
            let key = FunctionKey(key_counter);
            key_counter += 1;
            let (id, arm) = scheduler.choose_with_hint(key, &ctx, ComputeHint::High);
            black_box((id, arm))
        });
    });

    group.finish();
}

// =============================================================================
// Strategy Comparison Benchmarks
// =============================================================================

/// Benchmark different strategies across work sizes
fn bench_strategy_comparison(c: &mut Criterion) {
    let runtime = create_runtime();
    let mut group = c.benchmark_group("strategy_comparison");

    // Test with different work durations
    for work_us in [10u64, 50, 100, 250, 500, 1000] {
        // Always inline (direct execution)
        group.bench_with_input(
            BenchmarkId::new("always_inline", work_us),
            &work_us,
            |b, &work_us| {
                b.iter(|| runtime.block_on(async { black_box(calibrated_work(work_us)) }));
            },
        );

        // Always offload (spawn_compute)
        group.bench_with_input(
            BenchmarkId::new("always_offload", work_us),
            &work_us,
            |b, &work_us| {
                b.iter(|| {
                    runtime.block_on(async {
                        runtime
                            .spawn_compute(move || calibrated_work(work_us))
                            .await
                    })
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// MAB Convergence Benchmarks
// =============================================================================

/// Benchmark MAB convergence speed for different workload types
fn bench_mab_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("mab_convergence");
    group.sample_size(50); // Fewer samples since we're measuring convergence

    // Measure how many observations until MAB converges on fast work (should inline)
    group.bench_function("converge_fast_work", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;

            for _ in 0..iters {
                let scheduler = MabScheduler::with_defaults();
                let ctx = default_ctx();
                let key = FunctionKey(999);

                let start = std::time::Instant::now();

                // Feed fast observations until we see consistent inline decisions
                let mut consecutive_inline = 0;
                let mut observations = 0;

                while consecutive_inline < 5 && observations < 100 {
                    let (id, arm) = scheduler.choose(key, &ctx);
                    // Simulate fast work (~20µs)
                    scheduler.finish(id, 20.0, Some(20.0));
                    observations += 1;

                    if arm == Arm::InlineTokio {
                        consecutive_inline += 1;
                    } else {
                        consecutive_inline = 0;
                    }
                }

                total += start.elapsed();
            }

            total
        });
    });

    // Measure convergence for slow work (should offload)
    group.bench_function("converge_slow_work", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;

            for _ in 0..iters {
                let scheduler = MabScheduler::with_defaults();
                let ctx = default_ctx();
                let key = FunctionKey(888);

                let start = std::time::Instant::now();

                // Feed slow observations until we see consistent offload decisions
                let mut consecutive_offload = 0;
                let mut observations = 0;

                while consecutive_offload < 5 && observations < 100 {
                    let (id, arm) = scheduler.choose(key, &ctx);
                    // Simulate slow work (~500µs)
                    scheduler.finish(id, 500.0, Some(500.0));
                    observations += 1;

                    if arm == Arm::OffloadRayon {
                        consecutive_offload += 1;
                    } else {
                        consecutive_offload = 0;
                    }
                }

                total += start.elapsed();
            }

            total
        });
    });

    // Measure convergence for borderline work
    group.bench_function("converge_borderline_work", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;

            for _ in 0..iters {
                let scheduler = MabScheduler::with_defaults();
                let ctx = default_ctx();
                let key = FunctionKey(777);

                let start = std::time::Instant::now();

                // Feed borderline observations (~100µs)
                for _ in 0..50 {
                    let (id, _arm) = scheduler.choose(key, &ctx);
                    scheduler.finish(id, 100.0, Some(100.0));
                }

                total += start.elapsed();
            }

            total
        });
    });

    group.finish();
}

// =============================================================================
// Adaptive Stream Benchmarks
// =============================================================================

/// Benchmark adaptive_map vs compute_map
fn bench_adaptive_map(c: &mut Criterion) {
    let runtime = create_runtime();
    let mut group = c.benchmark_group("adaptive_map");

    let item_counts = [100, 500];

    for count in item_counts {
        group.throughput(Throughput::Elements(count as u64));

        // compute_map (always offload)
        group.bench_with_input(
            BenchmarkId::new("compute_map_small_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .compute_map(|_| small_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );

        // adaptive_map (MAB decides)
        group.bench_with_input(
            BenchmarkId::new("adaptive_map_small_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .adaptive_map(|_| small_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );

        // compute_map with medium work
        group.bench_with_input(
            BenchmarkId::new("compute_map_medium_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .compute_map(|_| medium_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );

        // adaptive_map with medium work
        group.bench_with_input(
            BenchmarkId::new("adaptive_map_medium_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .adaptive_map(|_| medium_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );

        // compute_map with large work
        group.bench_with_input(
            BenchmarkId::new("compute_map_large_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .compute_map(|_| large_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );

        // adaptive_map with large work
        group.bench_with_input(
            BenchmarkId::new("adaptive_map_large_work", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    runtime.block_on(async {
                        let results: Vec<_> = stream::iter(0..count)
                            .adaptive_map(|_| large_work())
                            .collect()
                            .await;
                        black_box(results)
                    })
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Mixed Workload Benchmark
// =============================================================================

/// Benchmark with mixed workload (realistic scenario)
fn bench_mixed_workload(c: &mut Criterion) {
    let runtime = create_runtime();
    let mut group = c.benchmark_group("mixed_workload");

    // 100 items: 60% fast, 30% medium, 10% slow
    let count = 100usize;
    group.throughput(Throughput::Elements(count as u64));

    // Generate pattern: determines work size per item
    fn work_for_item(i: usize) -> u64 {
        match i % 10 {
            0..=5 => calibrated_work(10),  // 60% fast
            6..=8 => calibrated_work(100), // 30% medium
            _ => calibrated_work(500),     // 10% slow
        }
    }

    group.bench_function("compute_map_mixed", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..count)
                    .compute_map(work_for_item)
                    .collect()
                    .await;
                black_box(results)
            })
        });
    });

    group.bench_function("adaptive_map_mixed", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let results: Vec<_> = stream::iter(0..count)
                    .adaptive_map(work_for_item)
                    .collect()
                    .await;
                black_box(results)
            })
        });
    });

    group.finish();
}

// =============================================================================
// Knobs Configuration Benchmark
// =============================================================================

/// Benchmark different knob configurations
fn bench_knobs_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("knobs_configs");

    let ctx = default_ctx();

    // Default knobs
    group.bench_function("default_knobs", |b| {
        let scheduler = MabScheduler::with_defaults();
        let key = FunctionKey(100);

        // Warmup
        for _ in 0..50 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
        }

        b.iter(|| {
            let (id, arm) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
            black_box(arm)
        });
    });

    // No strikes (GR3 disabled)
    group.bench_function("no_strikes", |b| {
        let knobs = MabKnobs::default().without_strikes();
        let scheduler = MabScheduler::new(knobs);
        let key = FunctionKey(101);

        for _ in 0..50 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
        }

        b.iter(|| {
            let (id, arm) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
            black_box(arm)
        });
    });

    // Higher k_starve (more aggressive offload under pressure)
    group.bench_function("high_k_starve", |b| {
        let knobs = MabKnobs::default().with_k_starve(0.5);
        let scheduler = MabScheduler::new(knobs);
        let key = FunctionKey(102);

        for _ in 0..50 {
            let (id, _) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
        }

        b.iter(|| {
            let (id, arm) = scheduler.choose(key, &ctx);
            scheduler.finish(id, 100.0, Some(100.0));
            black_box(arm)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mab_overhead,
    bench_strategy_comparison,
    bench_mab_convergence,
    bench_adaptive_map,
    bench_mixed_workload,
    bench_knobs_configs,
);

criterion_main!(benches);
