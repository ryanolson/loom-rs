//! Benchmarks for loom-rs runtime performance.
//!
//! Run with: cargo bench
//!
//! These benchmarks verify the performance characteristics documented in the API:
//! - `spawn_async()`: ~10ns overhead (TaskTracker token only)
//! - `spawn_compute()`: ~100-500ns (cross-thread signaling, ~32 bytes allocation)
//! - `install()`: ~0ns (zero overhead, direct rayon access)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use loom_rs::LoomBuilder;

/// Minimal work that won't be optimized away but is fast enough to measure overhead.
#[inline(never)]
fn minimal_work() -> u64 {
    black_box(42)
}

/// Small compute work (~100ns).
#[inline(never)]
fn small_work() -> u64 {
    let mut sum = 0u64;
    for i in 0..100 {
        sum = sum.wrapping_add(black_box(i));
    }
    sum
}

/// Medium compute work (~1µs).
#[inline(never)]
fn medium_work() -> u64 {
    let mut sum = 0u64;
    for i in 0..1000 {
        sum = sum.wrapping_add(black_box(i));
    }
    sum
}

/// Large compute work (~10µs).
#[inline(never)]
fn large_work() -> u64 {
    let mut sum = 0u64;
    for i in 0..10000 {
        sum = sum.wrapping_add(black_box(i));
    }
    sum
}

fn create_runtime() -> loom_rs::LoomRuntime {
    LoomBuilder::new()
        .prefix("bench")
        .tokio_threads(2)
        .rayon_threads(4)
        .build()
        .expect("failed to create runtime")
}

/// Benchmark spawn_compute() overhead with varying workload sizes.
fn bench_spawn_compute(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("spawn_compute");

    // Minimal work - shows pure overhead
    group.bench_function("minimal_work", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(minimal_work).await
            })
        });
    });

    // Small work (~100ns) - overhead still visible
    group.bench_function("small_work", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(small_work).await
            })
        });
    });

    // Medium work (~1µs) - overhead becoming amortized
    group.bench_function("medium_work", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(medium_work).await
            })
        });
    });

    // Large work (~10µs) - overhead fully amortized
    group.bench_function("large_work", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(large_work).await
            })
        });
    });

    group.finish();
}

/// Benchmark install() to verify zero overhead.
fn bench_install(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("install");

    // Compare direct rayon install vs our wrapper
    group.bench_function("loom_install", |b| {
        b.iter(|| runtime.install(minimal_work));
    });

    group.bench_function("direct_rayon_install", |b| {
        b.iter(|| runtime.rayon_pool().install(minimal_work));
    });

    group.finish();
}

/// Benchmark spawn_async() overhead.
fn bench_spawn_async(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("spawn_async");

    // Measure spawn + await overhead
    group.bench_function("spawn_and_await", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let handle = runtime.spawn_async(async { minimal_work() });
                handle.await.unwrap()
            })
        });
    });

    // Compare to direct tokio spawn
    group.bench_function("direct_tokio_spawn", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let handle = runtime.tokio_handle().spawn(async { minimal_work() });
                handle.await.unwrap()
            })
        });
    });

    group.finish();
}

/// Benchmark the RayonTask bridge in isolation.
fn bench_bridge(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("bridge");

    // Measure bridge overhead by comparing spawn_compute to direct rayon + block
    group.bench_function("spawn_compute_bridge", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(minimal_work).await
            })
        });
    });

    // Direct rayon spawn with manual sync (simulates what tokio-rayon did)
    group.bench_function("rayon_with_oneshot", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let (tx, rx) = tokio::sync::oneshot::channel();
                runtime.rayon_pool().spawn(move || {
                    let result = minimal_work();
                    let _ = tx.send(result);
                });
                rx.await.unwrap()
            })
        });
    });

    group.finish();
}

/// Benchmark parallel workloads to show real-world performance.
fn bench_parallel_workloads(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("parallel_workloads");

    // Parallel sum using install
    let data: Vec<u64> = (0..10000).collect();

    group.throughput(Throughput::Elements(data.len() as u64));

    group.bench_function("par_iter_via_install", |b| {
        b.iter(|| {
            runtime.install(|| {
                use rayon::prelude::*;
                data.par_iter().sum::<u64>()
            })
        });
    });

    group.bench_function("par_iter_direct_rayon", |b| {
        b.iter(|| {
            runtime.rayon_pool().install(|| {
                use rayon::prelude::*;
                data.par_iter().sum::<u64>()
            })
        });
    });

    group.finish();
}

/// Benchmark multiple concurrent spawn_compute calls.
fn bench_concurrent_spawn_compute(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("concurrent_spawn_compute");

    for num_tasks in [1, 4, 16, 64] {
        group.bench_with_input(
            BenchmarkId::new("tasks", num_tasks),
            &num_tasks,
            |b, &num_tasks| {
                b.iter(|| {
                    runtime.block_on(async {
                        let mut handles = Vec::with_capacity(num_tasks);
                        for _ in 0..num_tasks {
                            handles.push(runtime.spawn_compute(small_work));
                        }
                        let mut sum = 0u64;
                        for handle in handles {
                            sum = sum.wrapping_add(handle.await);
                        }
                        sum
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark shutdown overhead.
fn bench_shutdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("shutdown");

    // Measure block_until_idle with no tasks
    group.bench_function("block_until_idle_empty", |b| {
        b.iter(|| {
            let runtime = create_runtime();
            runtime.block_until_idle();
        });
    });

    // Measure is_idle check
    group.bench_function("is_idle_check", |b| {
        let runtime = create_runtime();
        runtime.shutdown();
        b.iter(|| black_box(runtime.is_idle()));
    });

    group.finish();
}

/// Benchmark task tracker overhead in isolation.
fn bench_task_tracker(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("task_tracker");

    // Measure token acquisition and drop
    group.bench_function("token_lifecycle", |b| {
        b.iter(|| {
            let token = runtime.task_tracker().token();
            drop(black_box(token));
        });
    });

    group.finish();
}

/// Benchmark spawn_compute pooling behavior.
fn bench_spawn_compute_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_compute_pooling");

    // Benchmark cold start (first call per type)
    group.bench_function("cold_start", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                // Create fresh runtime for each iteration to test cold start
                let runtime = LoomBuilder::new()
                    .prefix("bench")
                    .tokio_threads(1)
                    .rayon_threads(2)
                    .build()
                    .expect("failed to create runtime");

                let start = std::time::Instant::now();
                runtime.block_on(async {
                    runtime.spawn_compute(minimal_work).await
                });
                total += start.elapsed();
            }
            total
        });
    });

    // Benchmark warm (pool hit) - reusing the same runtime
    let runtime = create_runtime();

    // Warmup the pool
    runtime.block_on(async {
        runtime.spawn_compute(minimal_work).await;
    });

    group.bench_function("warm_pool_hit", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(minimal_work).await
            })
        });
    });

    // Benchmark pool exhaustion scenario
    group.bench_function("concurrent_4_tasks", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut handles = Vec::with_capacity(4);
                for _ in 0..4 {
                    handles.push(runtime.spawn_compute(minimal_work));
                }
                for handle in handles {
                    black_box(handle.await);
                }
            })
        });
    });

    group.finish();
}

/// Benchmark current_runtime() overhead.
fn bench_current_runtime(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("current_runtime");

    // Measure current_runtime() lookup time inside block_on
    group.bench_function("lookup_in_block_on", |b| {
        runtime.block_on(async {
            b.iter(|| {
                black_box(loom_rs::current_runtime())
            });
        });
    });

    // Measure current_runtime() outside runtime (should return None quickly)
    group.bench_function("lookup_outside_runtime", |b| {
        b.iter(|| {
            black_box(loom_rs::current_runtime())
        });
    });

    group.finish();
}

/// Benchmark spawn_compute via current_runtime (ergonomic API).
fn bench_ergonomic_spawn_compute(c: &mut Criterion) {
    let runtime = create_runtime();

    let mut group = c.benchmark_group("ergonomic_spawn_compute");

    // Warmup
    runtime.block_on(async {
        loom_rs::spawn_compute(minimal_work).await;
    });

    // Compare direct vs ergonomic API
    group.bench_function("via_runtime_ref", |b| {
        b.iter(|| {
            runtime.block_on(async {
                runtime.spawn_compute(minimal_work).await
            })
        });
    });

    group.bench_function("via_current_runtime", |b| {
        b.iter(|| {
            runtime.block_on(async {
                loom_rs::spawn_compute(minimal_work).await
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_spawn_compute,
    bench_install,
    bench_spawn_async,
    bench_bridge,
    bench_parallel_workloads,
    bench_concurrent_spawn_compute,
    bench_shutdown,
    bench_task_tracker,
    bench_spawn_compute_pooling,
    bench_current_runtime,
    bench_ergonomic_spawn_compute,
);

criterion_main!(benches);
