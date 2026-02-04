# loom-rs

**Weaving threads together**

A Rust crate providing a bespoke thread pool runtime combining tokio and rayon with CPU pinning capabilities.

## Features

- **Hybrid Runtime**: Combines tokio for async I/O with rayon for CPU-bound parallel work
- **CPU Pinning**: Automatically pins threads to specific CPUs for consistent performance
- **Flexible Configuration**: Configure via files (TOML/YAML/JSON), environment variables, or code
- **CLI Integration**: Built-in clap support for command-line overrides
- **CUDA NUMA Awareness**: Optional feature for selecting CPUs local to a GPU (Linux only)
- **Adaptive Scheduling**: [MAB-based scheduler](docs/mab.md) learns optimal inline vs offload decisions

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | **Full support** | All features including CPU pinning and CUDA |
| macOS | Partial | Compiles and runs, but CPU pinning may silently fail |
| Windows | Partial | Compiles and runs, but CPU pinning may silently fail |

**Note**: CPU affinity (thread pinning) is a Linux-focused feature. On macOS and Windows, pinning calls may return failure or have no effect. The library remains functional for development and testing, but production deployments targeting performance should use Linux.

## Installation

```bash
cargo add loom-rs
```

For CUDA support (Linux only):

```bash
cargo add loom-rs --features cuda
```

## Quick Start

```rust
use loom_rs::LoomBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = LoomBuilder::new()
        .prefix("myapp")
        .tokio_threads(2)
        .rayon_threads(6)
        .build()?;

    runtime.block_on(async {
        // Spawn tracked async I/O task
        let io_handle = runtime.spawn_async(async {
            // Async I/O work
            fetch_data().await
        });

        // Spawn tracked compute task and await result
        let result = runtime.spawn_compute(|| {
            // CPU-bound work on rayon
            (0..1000000).sum::<i64>()
        }).await;
        println!("Compute result: {}", result);

        // Zero-overhead parallel iterators
        let processed = runtime.install(|| {
            use rayon::prelude::*;
            data.par_iter().map(|x| process(x)).collect::<Vec<_>>()
        });

        // Wait for async task
        let data = io_handle.await?;
    });

    // Graceful shutdown - waits for all tracked tasks
    runtime.block_until_idle();

    Ok(())
}
```

## Configuration

Configuration sources are merged in order (later sources override earlier):

1. Default values
2. Config files (via `.file()`)
3. Environment variables (via `.env_prefix()`)
4. Programmatic overrides
5. CLI arguments (via `.with_cli_args()`)

### Config File (TOML)

```toml
prefix = "myapp"
tokio_threads = 2
rayon_threads = 14
```

### Environment Variables

With `.env_prefix("LOOM")`:

```bash
export LOOM_PREFIX=myapp
export LOOM_TOKIO_THREADS=2
export LOOM_RAYON_THREADS=6
```

### CLI Arguments

```rust
use clap::Parser;
use loom_rs::{LoomBuilder, LoomArgs};

#[derive(Parser)]
struct MyArgs {
    #[command(flatten)]
    loom: LoomArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = MyArgs::parse();
    let runtime = LoomBuilder::new()
        .file("config.toml")
        .env_prefix("LOOM")
        .with_cli_args(&args.loom)
        .build()?;
    Ok(())
}
```

Available CLI arguments:
- `--loom-prefix`: Thread name prefix
- `--loom-tokio-threads`: Number of tokio threads
- `--loom-rayon-threads`: Number of rayon threads
- `--loom-cuda-device`: CUDA device ID or UUID (requires `cuda` feature)

## CPU Affinity

When `pin_threads = true` (the default), loom automatically discovers the
process's allowed CPU set via `sched_getaffinity` on Linux. This respects:

- **cgroup CPU constraints**: Docker `--cpuset-cpus`, Kubernetes CPU limits
- **taskset restrictions**: `taskset -c 0-3 ./myapp`
- **NUMA policies**: `numactl --cpunodebind=0`

When using CUDA (`cuda_device` option), the CUDA device's NUMA-local CPUs
are intersected with the process affinity mask to ensure optimal data locality.

Use `runtime.effective_cpuset()` to inspect which CPUs the runtime is using:

```rust
let runtime = LoomBuilder::new().build()?;
println!("Using CPUs: {:?}", runtime.effective_cpuset());
```

## CUDA Support

With the `cuda` feature enabled (Linux only), configure the runtime to use CPUs local to a specific GPU.

### System Dependencies

```bash
sudo apt-get install libhwloc-dev libudev-dev
```

### Usage

```rust
let runtime = LoomBuilder::new()
    .cuda_device_id(0)  // Use CPUs near GPU 0
    .build()?;

// Or by UUID
let runtime = LoomBuilder::new()
    .cuda_device_uuid("GPU-12345678-1234-1234-1234-123456789012")
    .build()?;
```

This is useful for GPU-accelerated workloads where data needs to be transferred between CPU and GPU memory, as it minimizes NUMA-related latency.

## Thread Naming

Threads are named with the configured prefix:

- Tokio threads: `{prefix}-tokio-0000`, `{prefix}-tokio-0001`, ...
- Rayon threads: `{prefix}-rayon-0000`, `{prefix}-rayon-0001`, ...

## API Reference

### Task Spawning

| Method | Use Case | Overhead | Tracked |
|--------|----------|----------|---------|
| `spawn_async()` | I/O-bound async tasks | ~10ns | Yes |
| `spawn_compute()` | CPU-bound work (always offload) | ~100-500ns | Yes |
| `spawn_adaptive()` | CPU work (MAB decides inline/offload) | ~50-200ns | Yes |
| `scoped_compute()` | Borrow local data for parallel work | ~100-500ns | Yes |
| `scoped_adaptive()` | Scoped work with MAB decisions | ~50-200ns | Yes |
| `try_spawn_compute()` | Non-panicking compute spawn | ~100-500ns | Yes |
| `compute_map()` | Stream -> rayon -> stream | ~100-500ns/item | No |
| `adaptive_map()` | Stream with MAB decisions | ~50-200ns/item | No |
| `install()` | Zero-overhead parallel iterators | ~0ns | No |

**Note**: `scoped_*` methods allocate ~96 bytes per call (cannot use object pooling since results may contain borrowed references).

### Shutdown

```rust
// Option 1: Simple shutdown from main thread
runtime.block_until_idle();

// Option 2: Manual control from async context
runtime.block_on(async {
    runtime.spawn_async(background_work());

    // Signal shutdown
    runtime.shutdown();

    // Wait for completion
    runtime.wait_for_shutdown().await;
});

// Option 3: Check status without blocking
if runtime.is_idle() {
    println!("All tasks complete");
}
```

### Direct Access (Untracked)

For advanced use cases requiring untracked access:

```rust
// Direct tokio handle
let handle = runtime.tokio_handle();
handle.spawn(untracked_task());

// Direct rayon pool
let pool = runtime.rayon_pool();
pool.spawn(|| untracked_work());
```

## Ergonomic Access

Use `current_runtime()` or `spawn_compute()` from anywhere in the runtime:

```rust
use loom_rs::LoomBuilder;

let runtime = LoomBuilder::new().build()?;

runtime.block_on(async {
    // No need to pass &runtime around
    let result = loom_rs::spawn_compute(|| expensive_work()).await;

    // Or get the runtime explicitly
    let rt = loom_rs::current_runtime().unwrap();
    rt.spawn_async(async { /* ... */ });
});
```

## Scoped Compute

Use `scoped_compute()` to borrow local (non-`'static`) data in parallel work. This is useful when you want to process data in parallel without cloning:

```rust
use loom_rs::LoomBuilder;
use std::sync::atomic::{AtomicI32, Ordering};

let runtime = LoomBuilder::new().build()?;

runtime.block_on(async {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let sum = AtomicI32::new(0);

    // Borrow `data` and `sum` for parallel processing
    runtime.scoped_compute(|s| {
        let (left, right) = data.split_at(data.len() / 2);

        s.spawn(|_| sum.fetch_add(left.iter().sum::<i32>(), Ordering::Relaxed));
        s.spawn(|_| sum.fetch_add(right.iter().sum::<i32>(), Ordering::Relaxed));
    }).await;

    // `data` and `sum` are still valid here
    println!("Sum of {:?} = {}", data, sum.load(Ordering::Relaxed));
});
```

There's also `scoped_adaptive()` which uses MAB to decide whether to run synchronously via `install()` or asynchronously via `scoped_compute()`.

**Cancellation Safety**: If a scoped future is dropped before completion (e.g., via `select!` or timeout), the drop will block until the rayon scope finishes. This prevents use-after-free of borrowed data.

## Stream Processing

Use `ComputeStreamExt` to process async stream items on rayon:

```rust
use loom_rs::{LoomBuilder, ComputeStreamExt};
use futures::stream::{self, StreamExt};

let runtime = LoomBuilder::new().build()?;

runtime.block_on(async {
    let numbers = stream::iter(0..100);

    // Each item is processed on rayon, results stream back
    let results: Vec<_> = numbers
        .compute_map(|n| {
            // CPU-intensive work runs on rayon
            (0..n).map(|i| i * i).sum::<i64>()
        })
        .collect()
        .await;
});
```

This is ideal for pipelines where you:
1. Await values from an async source (network, channel, file)
2. Process each value with CPU-intensive work
3. Continue the async pipeline with the results

Items are processed sequentially to preserve ordering and provide natural backpressure.

## Adaptive Scheduling (MAB)

loom-rs includes a Multi-Armed Bandit (MAB) scheduler that learns whether to run
compute work inline on Tokio or offload to Rayon. This eliminates the need to
manually tune offload decisions - the scheduler adapts to your actual workload.

### Stream Mode

```rust
use loom_rs::ComputeStreamExt;

// MAB learns optimal strategy per-closure
let results: Vec<_> = stream
    .adaptive_map(|item| process(item))
    .collect()
    .await;
```

### One-Shot Mode

```rust
// For request handlers - MAB adapts per function type
let result = runtime.spawn_adaptive(|| handle_request(data)).await;
```

### Key Features

- **Thompson Sampling**: Balances exploration vs exploitation
- **Guardrails**: 4 layers of Tokio starvation protection (GR0-GR3)
- **Pressure-Aware**: Adjusts decisions based on runtime load
- **Low Overhead**: ~50-200ns per decision

### Adaptive Hints

Use `ComputeHint` to guide cold-start behavior when the scheduler hasn't learned yet:

```rust
use loom_rs::ComputeHint;

// Hint that this is likely expensive work - prefer offload initially
let result = runtime.spawn_adaptive_with_hint(ComputeHint::High, || heavy_work()).await;

// Hint that this is likely fast work - prefer inline initially
let result = runtime.spawn_adaptive_with_hint(ComputeHint::Low, || quick_work()).await;
```

See [docs/mab.md](docs/mab.md) for the complete design and configuration options.

## Performance

loom-rs is designed for zero unnecessary overhead:

- **Thread pinning**: One-time cost at runtime creation only
- **Zero allocation after warmup**: `spawn_compute()` uses per-type object pools
- **Custom async-rayon bridge**: Uses atomic wakers (~32 bytes) instead of channels (~80 bytes)
- **Main thread is separate**: Not part of worker pools

### spawn_compute Performance

| State | Allocations | Overhead |
|-------|-------------|----------|
| Pool hit | 0 bytes | ~100-500ns |
| Pool miss | ~32 bytes | ~100-500ns |
| First call per type | Pool + state | ~1µs |

Configure pool size for high-concurrency workloads:

```rust
let runtime = LoomBuilder::new()
    .compute_pool_size(128)  // Default is 64
    .build()?;
```

## Patterns to Avoid

### 1. Nested spawn_compute (Deadlock Risk)

```rust
// BAD: Can deadlock if all rayon threads are waiting
runtime.spawn_compute(|| {
    runtime.block_on(runtime.spawn_compute(|| work()))
}).await;

// GOOD: Use install() for nested parallelism
runtime.spawn_compute(|| {
    runtime.install(|| {
        data.par_iter().map(|x| process(x)).collect()
    })
}).await;
```

### 2. Blocking I/O in spawn_compute

```rust
// BAD: Blocks rayon thread
runtime.spawn_compute(|| {
    std::fs::read_to_string("file.txt")
}).await;

// GOOD: I/O in async, compute in rayon
let data = tokio::fs::read_to_string("file.txt").await?;
runtime.spawn_compute(|| process(&data)).await;
```

### 3. spawn_compute in Tight Loops

```rust
// OK (auto-pooling): Each call reuses pooled state
for item in items {
    results.push(runtime.spawn_compute(|| process(item)).await);
}

// STILL BETTER for batch: Single cross-thread trip
let results = runtime.install(|| {
    items.par_iter().map(|item| process(item)).collect()
});
```

### 4. Holding Locks Across spawn_compute

```rust
// BAD: Lock held during async gap
let guard = mutex.lock();
runtime.spawn_compute(|| use(&guard)).await;

// GOOD: Clone data, release lock
let data = mutex.lock().clone();
runtime.spawn_compute(move || process(data)).await;
```

### 5. install() Blocks the Thread

```rust
// CAUTION in async context: blocks tokio worker
runtime.spawn_async(async {
    runtime.install(|| heavy_par_iter());  // Blocks!
}).await;

// BETTER: spawn_compute for async-safe bridge
runtime.spawn_async(async {
    runtime.spawn_compute(|| heavy_par_iter()).await;
}).await;
```

### 6. Manual spawn_compute Loop on Streams

```rust
// WORKS but slower: Pool get/return for each item
while let Some(item) = stream.next().await {
    let result = runtime.spawn_compute(|| process(item)).await;
    results.push(result);
}

// BETTER: compute_map reuses internal state
let results: Vec<_> = stream
    .compute_map(|item| process(item))
    .collect()
    .await;
```

## Testing

loom-rs provides a `#[loom_rs::test]` macro for writing tests that run within a LoomRuntime:

```rust
#[loom_rs::test]
async fn test_spawn_compute() {
    let result = loom_rs::spawn_compute(|| 42).await;
    assert_eq!(result, 42);
}
```

### Default Configuration

- 1 tokio thread
- 2 rayon threads
- Thread pinning disabled

### Custom Thread Counts

```rust
#[loom_rs::test(tokio_thread_count = 2, rayon_thread_count = 4)]
async fn test_parallel_work() {
    // Test with more threads
}
```

### Result Return Types

The macro supports returning `Result` for integration with test frameworks like `anyhow`:

```rust
#[loom_rs::test]
async fn test_with_result() -> anyhow::Result<()> {
    let result = loom_rs::spawn_compute(|| 42).await;
    assert_eq!(result, 42);
    Ok(())
}
```

The macro automatically:
- Creates a runtime with a unique prefix based on the test name
- Runs the async test body in `block_on()`
- Waits for all tracked tasks to complete via `block_until_idle()`

## Workspace Structure

loom-rs is organized as a Cargo workspace:

- `loom-rs` - Main runtime crate
- `loom-macros` - Procedural macros (`#[loom_rs::test]`), automatically included as a dependency

## License

MIT
