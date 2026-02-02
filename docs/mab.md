# Multi-Armed Bandit Adaptive Scheduler

The MAB (Multi-Armed Bandit) adaptive scheduler in loom-rs decides per-function whether to execute compute work **inline on Tokio** or **offload to Rayon**. It uses Thompson Sampling to learn optimal strategies while protecting the async runtime from starvation.

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
  - [Thompson Sampling](#thompson-sampling)
  - [Cost Model](#cost-model)
  - [Pressure Index](#pressure-index)
- [Guardrails](#guardrails)
- [Usage Patterns](#usage-patterns)
  - [Stream Mode](#stream-mode-adaptive_map)
  - [Handler Mode](#handler-mode-mab_scheduler)
- [Configuration](#configuration-mabknobs)
- [Compute Hints](#compute-hints)
- [When to Use What](#when-to-use-what)
- [Performance Characteristics](#performance-characteristics)

## Problem Statement

When processing compute work in an async application, you face a classic tradeoff:

| Strategy | Pros | Cons |
|----------|------|------|
| **Always Inline** | Zero overhead (~0ns) | Blocks Tokio worker threads, increases async task latency |
| **Always Offload** | Never blocks Tokio | Fixed overhead (~100-500ns per task) |

The optimal strategy depends on:
- **Function execution time**: Fast work (<50µs) is better inline; slow work (>250µs) is better offloaded
- **System pressure**: Under high load, even moderate work should offload
- **Per-function characteristics**: Each function has different timing profiles

The MAB scheduler **learns the optimal strategy per-function** while respecting guardrails that prevent Tokio starvation.

## How It Works

### Thompson Sampling

The scheduler models the decision as a 2-arm bandit:

| Arm | Description |
|-----|-------------|
| `InlineTokio` | Execute synchronously on the current Tokio worker thread |
| `OffloadRayon` | Spawn work on Rayon thread pool and await completion |

**Thompson Sampling** balances exploration (trying both arms) with exploitation (using the apparently better arm):

1. Each arm maintains a posterior distribution over its cost
2. Sample from each arm's posterior distribution
3. Choose the arm with the lower sampled cost
4. Update the chosen arm's statistics with the observed cost

This naturally handles the explore-exploit tradeoff: uncertain arms (few observations) have wide posteriors that occasionally sample lower than well-known arms.

### Cost Model

The scheduler uses a **log-cost model** where observed costs are transformed as `ln(cost_us)`. This handles the log-normal distribution of execution times common in real workloads.

**Adjusted Cost Formula:**

```
adjusted_cost = fn_time + k_starve * pressure * fn_time
```

Where:
- `fn_time` is the raw execution time
- `k_starve` (default: 0.15) penalizes inline execution under pressure
- `pressure` is the computed pressure index

For offload, an overhead term is added:

```
offload_adjusted = sampled_cost + ln(offload_overhead_us)
```

### Pressure Index

Pressure quantifies how stressed the Tokio runtime is:

```
pressure = 0.7 * (inflight_tasks / tokio_workers) + 0.3 * (spawn_rate / baseline)
```

| Pressure Level | Interpretation |
|----------------|----------------|
| < 0.5 | Low pressure, inline is usually safe |
| 0.5 - 3.0 | Normal, MAB decides |
| > 3.0 | High pressure, guardrails restrict inline |

### Decayed Sample Count

Statistics use an **effective sample count** with exponential decay:

```
n_eff *= decay  // Per observation
```

With `decay = 0.999653`, the half-life is ~2000 observations. This means:
- Recent observations have more weight
- The scheduler adapts to changing workload characteristics
- Old data naturally fades out

## Guardrails

Four guardrails protect Tokio from starvation, even when the MAB is still learning:

### GR0: Single-Worker Protection

When `tokio_workers == 1`, any blocking is catastrophic. Only allow inline if:
- EMA < `t_tiny_inline_us` (50µs)
- AND pressure < `p_low` (0.5)

### GR1: Hard Blocking Threshold

**Never** allow inline if:
- EMA > `t_block_hard_us` (250µs)

This is a hard ceiling regardless of MAB statistics.

### GR2: Pressure-Sensitive Threshold

Under high pressure (`pressure > p_high`), restrict inline to:
- EMA < `t_inline_under_pressure_us` (100µs)

### GR3: Strike Suppression

Track slow inline executions as "strikes":
- If `fn_time > t_strike_us` (1ms), increment strike counter
- If strikes >= `s_max` (1.0), force offload
- Strikes decay over time (`strike_decay = 0.993`, half-life ~100 observations)

**Guardrail Summary:**

| Guardrail | Condition | Effect |
|-----------|-----------|--------|
| GR0 | `tokio_workers == 1` | Very conservative inline |
| GR1 | `ema > 250µs` | Never inline |
| GR2 | `pressure > 3.0 && ema > 100µs` | Never inline |
| GR3 | `strikes >= 1.0` | Never inline |

## Usage Patterns

### Stream Mode: `adaptive_map()`

Each stream instance owns its own MAB scheduler for immediate feedback learning:

```rust
use loom_rs::{LoomBuilder, ComputeStreamExt};
use futures::stream::{self, StreamExt};

let runtime = LoomBuilder::new()
    .tokio_threads(2)
    .rayon_threads(6)
    .build()?;

runtime.block_on(async {
    let results: Vec<_> = stream::iter(items)
        .adaptive_map(|item| process(item))  // MAB decides per-item
        .collect()
        .await;
});
```

**Characteristics:**
- Scheduler learns from each item immediately
- Maintains state across the stream lifetime
- Dropped with the stream

### Handler Mode: `mab_scheduler()`

For handler patterns where decisions and outcomes are separated:

```rust
use loom_rs::mab::{FunctionKey, Arm};

let sched = runtime.mab_scheduler();  // Shared, lazily initialized
let key = FunctionKey::from_type::<MyHandler>();

async fn handle_request(runtime: &LoomRuntime, req: Request) -> Response {
    let ctx = runtime.collect_context();
    let (id, arm) = sched.choose(key, &ctx);

    let start = std::time::Instant::now();
    let result = match arm {
        Arm::InlineTokio => process(req),
        Arm::OffloadRayon => runtime.spawn_compute(|| process(req)).await,
    };
    let elapsed_us = start.elapsed().as_micros() as f64;

    sched.finish(id, elapsed_us, Some(elapsed_us));
    result
}
```

**Characteristics:**
- Shared across all handlers of the same type
- Delayed feedback via `finish()`
- Persistent for runtime lifetime

## Configuration (MabKnobs)

All parameters have sensible defaults. Override via `LoomBuilder`:

```rust
use loom_rs::{LoomBuilder, MabKnobs};

let knobs = MabKnobs::default()
    .with_k_starve(0.2)           // Higher starvation penalty
    .with_thresholds(40.0, 200.0, 80.0)  // Custom thresholds
    .without_strikes();            // Disable GR3

let runtime = LoomBuilder::new()
    .mab_knobs(knobs)
    .build()?;
```

### Cost Composition

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_starve` | 0.15 | Starvation cost multiplier |

### Pressure Index

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_inflight` | 0.7 | Weight for inflight task count |
| `w_spawn` | 0.3 | Weight for spawn rate |
| `pressure_clip` | 10.0 | Maximum pressure value |

### Decay Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay` | 0.999653 | Per-observation decay (half-life ~2000) |
| `strike_decay` | 0.993 | Strike counter decay (half-life ~100) |
| `ema_alpha` | 0.1 | EMA smoothing for function time |

### Guardrail Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t_tiny_inline_us` | 50.0 | Safe inline threshold (GR0/GR2) |
| `t_block_hard_us` | 250.0 | Hard block threshold (GR1) |
| `t_inline_under_pressure_us` | 100.0 | Under-pressure threshold (GR2) |
| `p_low` | 0.5 | Low pressure boundary |
| `p_high` | 3.0 | High pressure boundary |

### Strike Suppression

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_strikes` | true | Enable GR3 |
| `t_strike_us` | 1000.0 | Strike threshold (1ms) |
| `s_max` | 1.0 | Max strikes before suppression |

### Compute Hints

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hint_trust_threshold` | 5.0 | n_eff below which hints matter |
| `hint_exploration_count` | 3 | Forced offload count for High hint |
| `hint_low_ema_us` | 30.0 | Initial EMA for Low hint |
| `hint_medium_ema_us` | 200.0 | Initial EMA for Medium hint |
| `hint_high_ema_us` | 1000.0 | Initial EMA for High hint |

### Calibration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `measured_offload_overhead_us` | None | Measured offload overhead (or ~300µs default) |

Enable calibration to measure actual overhead:

```rust
let runtime = LoomBuilder::new()
    .calibrate(true)  // Measure offload overhead at startup
    .build()?;
```

## Compute Hints

Implement `ComputeHintProvider` on input types to guide cold-start behavior:

```rust
use loom_rs::mab::{ComputeHint, ComputeHintProvider};

struct WorkItem {
    data: Vec<u8>,
    complexity: Complexity,
}

enum Complexity {
    Simple,    // ~5µs
    Moderate,  // ~100µs
    Complex,   // ~1ms
}

impl ComputeHintProvider for WorkItem {
    fn compute_hint(&self) -> ComputeHint {
        match self.complexity {
            Complexity::Simple => ComputeHint::Low,
            Complexity::Moderate => ComputeHint::Medium,
            Complexity::Complex => ComputeHint::High,
        }
    }
}
```

### Hint Levels

| Hint | Expected Duration | Cold-Start Behavior |
|------|-------------------|---------------------|
| `Unknown` | No information | Default exploration (try inline first) |
| `Low` | < 50µs | Seeds EMA at 30µs, likely inline |
| `Medium` | 50-500µs | Seeds EMA at 200µs, explores both |
| `High` | > 500µs | Seeds EMA at 1000µs, forces 3 offloads |

**Important:** Hints only affect cold-start behavior. Once the scheduler has enough observations (`n_eff > hint_trust_threshold`), it relies entirely on learned data.

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Unknown complexity | `adaptive_map()` |
| Consistently fast (<50µs) | Direct inline execution |
| Consistently slow (>250µs) | `spawn_compute()` |
| Variable complexity | `adaptive_map()` with `ComputeHintProvider` |
| Request handlers | `mab_scheduler()` with `FunctionKey` |
| Parallel iterators | `install()` (zero overhead) |

### Decision Flow

```
Is work duration known and consistent?
├── Yes, always <50µs → Execute inline (no MAB needed)
├── Yes, always >250µs → spawn_compute() (no MAB needed)
└── No or variable → Use MAB
    ├── Stream processing → adaptive_map()
    └── Handler pattern → mab_scheduler()
```

## Performance Characteristics

### MAB Overhead

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| `choose()` cold | ~200ns | New function key |
| `choose()` warm | ~50ns | Existing statistics |
| `finish()` | ~30ns | Update statistics |
| Full cycle | ~80ns | Warm choose + finish |

### Convergence

| Workload Type | Typical Convergence |
|---------------|---------------------|
| Clearly fast (<20µs) | ~5-10 observations |
| Clearly slow (>500µs) | ~10-20 observations |
| Borderline (100-200µs) | ~50-100 observations |

The MAB will:
1. **Cold start**: Explore inline first (cheaper if fast)
2. **Learning**: Thompson Sampling explores both arms
3. **Converged**: Strongly prefer the better arm

### Memory Usage

| Component | Size |
|-----------|------|
| `MabScheduler` | ~200 bytes + per-key stats |
| Per `FunctionKey` | ~150 bytes |
| Per pending decision | ~24 bytes |

## Example: Observing MAB Behavior

```rust
use loom_rs::{LoomBuilder, ComputeStreamExt};
use futures::stream::{self, StreamExt};

// Work with variable duration
fn variable_work(n: u64) -> u64 {
    let iterations = if n % 10 == 0 { 50_000 } else { 100 };
    let mut sum = 0u64;
    for i in 0..iterations {
        sum = sum.wrapping_add(i);
    }
    sum
}

let runtime = LoomBuilder::new()
    .tokio_threads(2)
    .rayon_threads(4)
    .build()?;

runtime.block_on(async {
    // MAB will learn:
    // - Most items (n % 10 != 0) are fast → inline
    // - Every 10th item is slow → offload
    let results: Vec<_> = stream::iter(0..1000)
        .adaptive_map(variable_work)
        .collect()
        .await;
});
```

## Debugging

Enable tracing to see MAB decisions:

```rust
// In your application setup
tracing_subscriber::fmt()
    .with_env_filter("loom_rs::mab=debug")
    .init();
```

This logs:
- Guardrail activations
- Arm selections
- Learning updates
