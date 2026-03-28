# Simulation Runtime Guide

`loom-rs` can run application code under virtual time with the `sim` feature. This is useful for testing protocols, pipelines, retries, and time-driven async flows without waiting on wall clock time.

This mode is intentionally pragmatic. It is not a full discrete-event simulator for arbitrary compute. Async timing and explicit simulated events are virtual; synchronous compute is still measured on the host and translated into virtual delay as a heuristic.

## Enable It

```toml
loom-rs = { version = "0.4", features = ["sim"] }
```

Create a simulation runtime instead of a normal `LoomRuntime`:

```rust
use loom_rs::sim::SimulationRuntime;

let mut sim = SimulationRuntime::new()?;
let handle = sim.handle();
```

Examples in this repo:

- `cargo run --features sim --example sim_ping_pong`
- `cargo run --features sim --example sim_pipeline`

## Mental Model

Simulation mode combines two time domains:

- DES events in the internal sim queue
- Tokio virtual time from `tokio::time::pause()` and `advance()`

`SimulationRuntime::step()` does this:

1. Poll ready Tokio tasks so they can register timers or `delay()` wakeups.
2. Find the next live simulated event time.
3. Advance Tokio's virtual clock to that time.
4. Execute all simulated actions at that timestamp.
5. Drain follow-on async work before moving to a later timestamp.

Time only moves when the runtime is driven by `step()`, `step_until()`, `run()`, or `run_until()`.

## Core APIs

### `SimHandle::schedule()` and `schedule_at()`

Use these for fire-and-forget simulated events such as message delivery, retries, or hardware interrupts.

```rust
handle.schedule(Duration::from_millis(10), move || {
    deliver_packet();
});
```

Guidance:

- Prefer `schedule(delay, ...)` unless you already have an absolute virtual timestamp.
- `schedule_at()` must not target a time earlier than `now()`. That is treated as a modeling bug and will panic.
- Use closures for state transitions that should happen at a discrete virtual instant.

### `SimHandle::delay().await`

Use this inside async tasks to suspend work in virtual time.

```rust
handle.delay(Duration::from_secs(1)).await;
```

Use it for:

- service time
- backoff
- network latency inside async flows
- protocol timers owned by a task

Prefer `delay().await` over real sleeps in tests and examples.

### `spawn_after()` and `spawn_at()`

Use these when the thing that should happen later is itself async.

```rust
handle.spawn_after(Duration::from_millis(50), async move {
    handle.delay(Duration::from_millis(5)).await;
    send_response().await;
});
```

Guidance:

- Use `spawn_after()` for relative delays.
- Use `spawn_at()` only with absolute virtual times at or after `now()`.

### `SimulationRuntime::step()`, `step_until()`, `run()`, and `run_until()`

- `step()` advances to the next live simulated event and processes everything at that timestamp.
- `step_until(deadline)` keeps stepping while live events exist at or before `deadline`.
- `run()` keeps stepping until no live simulated events remain.
- `run_until(target)` processes events up to `target`, then force-advances Tokio virtual time to `target` even if the sim queue is empty.

Rule of thumb:

- Use `run()` when explicit sim events or `delay().await` drive progress.
- Use `run_until()` when you need pure Tokio timers like `tokio::time::sleep()` to expire even if nothing is in the sim queue.
- Use `step()` and `step_until()` when the test needs checkpoints and assertions between virtual timestamps.

## Human Usage Patterns

### Model external systems as events

For networks, transports, queues, or device callbacks, schedule explicit events:

```rust
handle.schedule(network_delay, move || {
    peer.receive(msg);
});
```

This makes causality obvious and keeps tests easy to reason about.

### Model in-task waiting with `delay().await`

For processing time owned by an async task, prefer:

```rust
handle.delay(process_time).await;
```

That keeps task-local control flow readable and composes well with channels, `select!`, and Tokio synchronization primitives.

### Assert virtual time, not wall time

Good sim tests assert:

- `sim.now()`
- returned final time from `run()`
- event ordering
- state at specific checkpoints

Always use `SimHandle::now()` for simulation-time observations. Do not use `tokio::time::Instant::now()` — the DES clock is updated before `tokio::time::advance()` is called inside `step()`, so the two clocks can momentarily disagree during the same advance. Code that mixes them may observe an inconsistent snapshot.

Avoid assertions on host runtime duration unless you are explicitly measuring simulator overhead.

### Treat simulated compute as approximate

`spawn_compute`, `spawn_adaptive`, `scope_compute`, and `scope_adaptive` run inline in sim mode and translate measured wall-clock duration into virtual delay. This is useful for rough modeling, but it is not deterministic enough to be the primary assertion surface for protocol tests.

## Coding-Agent Guidance

When an agent adds or modifies sim usage, it should follow these rules:

- Never introduce `std::thread::sleep` or wall-clock waits in sim tests or examples.
- Prefer `schedule()` or `delay().await` over ad hoc time bookkeeping.
- Keep absolute virtual times monotonic. Use `now() + delta` unless the test truly needs a fixed timestamp.
- Add tests that prove virtual-time behavior, especially ordering, cancellation, and same-tick cascades.
- Use `run_until()` when a test depends on Tokio timers with no explicit sim event to drive the clock.
- Treat wall-clock-measured compute delays as modeling hints, not exact truth.

## Unsupported Or Special-Case Behavior

Simulation mode is a constrained runtime:

- `rayon_pool()` is not supported and will panic.
- `ComputeStreamExt::compute_map()` is not supported and will panic.
- `ComputeStreamExt::adaptive_map()` is not supported and will panic.
- `install()` still works, but it runs on the sim runtime's single-thread Rayon pool and does not add virtual-time delay by itself.

## Common Mistakes

- Scheduling in the past with `schedule_at()` or `spawn_at()`.
- Expecting `run()` to advance pure Tokio timers when there are no live sim events.
- Using wall-clock sleeps instead of virtual-time delays.
- Writing tests that only assert outputs and never assert virtual-time progression.
- Mixing `tokio::time::Instant::now()` with `SimHandle::now()` — use `SimHandle::now()` exclusively for simulation-time observations.

## Recommended Test Shapes

Use small, explicit scenarios:

- one event schedules another event at the same timestamp
- two delays race under `tokio::select!`
- channel send and receive happen across virtual network delay
- checkpoint with `step()` before and after a specific timestamp

The existing examples are a good template for both human-written code and agent-generated patches.
