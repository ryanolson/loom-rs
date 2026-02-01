# CLAUDE.md - Development Guidelines

## Build Commands

- `cargo build` - Build without CUDA feature
- `cargo build --features cuda` - Build with CUDA feature (Linux only)
- `cargo test` - Run tests
- `cargo bench` - Run all benchmarks
- `cargo bench -- <filter>` - Run specific benchmarks (e.g., `cargo bench -- bridge`)
- `cargo clippy -- -D warnings -A deprecated` - Lint check
- `cargo deny check` - License compliance check
- `cargo doc --open` - Generate and view documentation

## Setup

Install git hooks to auto-format on commit:
```bash
./hooks/install.sh
```

### System Dependencies (for `cuda` feature)

The `cuda` feature requires hwloc system libraries (Linux only):
```bash
sudo apt-get install libhwloc-dev libudev-dev
```

## Code Style

- Run `cargo clippy -- -D warnings -A deprecated` before committing
- Run `cargo deny check` to verify license compliance
- Never use `.unwrap()` in library code - use proper error handling with `?` and `Result`
- All public APIs must have doc comments with examples where appropriate
- Use `thiserror` for error types
- Thread names follow pattern: `{prefix}-tokio-{NNNN}` or `{prefix}-rayon-{NNNN}`

## Architecture

- `config.rs` - Configuration structs with serde support
- `error.rs` - Error types using thiserror
- `cpuset.rs` - CPU set parsing (e.g., "0-7,16-23")
- `affinity.rs` - Thread pinning utilities using core_affinity
- `bridge.rs` - Minimal-overhead async-to-rayon bridge (replaces tokio-rayon)
- `builder.rs` - LoomBuilder with figment integration and LoomArgs for CLI
- `runtime.rs` - LoomRuntime combining tokio + rayon
- `cuda.rs` - CUDA NUMA selection (feature-gated, Linux only)

## Features

- `default` - No optional features
- `cuda` - Enable CUDA device selection for NUMA-aware CPU pinning (requires hwlocality and nvml-wrapper, Linux only)
- `cuda-tests` - Enable hardware-dependent CUDA tests (implies `cuda`)

## Testing

- Unit tests are in each module
- CUDA hardware tests require the `cuda-tests` feature: `cargo test --features cuda-tests`

## Performance Guidelines

loom-rs is designed for zero unnecessary overhead. Follow these rules:

### API Performance Characteristics

| Method | Overhead | Allocations | Tracked |
|--------|----------|-------------|---------|
| `spawn_async()` | ~10ns | Token only | Yes |
| `spawn_compute()` | ~100-500ns | ~32 bytes (Arc state) | Yes |
| `install()` | ~0ns | None | No |
| `rayon_pool()` | 0ns | None | No |
| `tokio_handle()` | 0ns | None | No |

### When to Use Each Method

- **`spawn_async()`**: For I/O-bound async tasks that need tracking for graceful shutdown
- **`spawn_compute()`**: For CPU-bound work when you need to await the result from async context
- **`install()`**: For zero-overhead parallel iterators within an already-tracked context
- **`rayon_pool()`/`tokio_handle()`**: For direct, untracked access when needed

### Performance Rules

- Thread pinning happens once at runtime creation - no per-task overhead
- Use `Arc<str>` instead of `String` cloning in hot paths
- The bridge module uses atomic wakers (~32 bytes) instead of channels (~80 bytes)
- No hidden allocations - all overhead is documented

### Benchmarks

Run `cargo bench` to verify performance characteristics. Key benchmark groups:

- `spawn_compute` - Measures async-to-rayon bridge overhead with various workload sizes
- `install` - Verifies zero overhead vs direct rayon access
- `spawn_async` - Measures task tracker overhead
- `bridge` - Compares our custom bridge to oneshot channel approach
- `parallel_workloads` - Real-world parallel iterator performance
- `concurrent_spawn_compute` - Multiple concurrent compute tasks
- `shutdown` - Shutdown and idle check overhead
