# loom-rs MAB Evaluation Report

**Generated:** 2026-02-04 05:52:11 UTC
**Platform:** linux x86_64, 30 cores available
**Build:** release (LTO enabled)

## Measurement Journey

This report follows a progression of measurements designed to answer:
*Does the MAB adaptive scheduler protect Tokio from starvation while*
*optimizing throughput for mixed workloads?*

1. **Overhead** - Can we afford to make a decision on every task?
2. **Latency Impact** - Does inlining cause measurable starvation?
3. **Throughput** - Do we leave performance on the table?
4. **Learning** - Does the MAB converge to correct decisions?
5. **Thread Configurations** - How do different ratios affect behavior?
6. **Workload Shift** - Does the MAB adapt when workloads change?
7. **Pressure Escalation** - Do guardrails activate under load?

## System Configuration

**Evaluation CPUs:** 16 cores (auto-discovered from process affinity)

### Thread Configurations Tested

| Config | Tokio Threads | Tokio CPUs | Rayon Threads | Rayon CPUs | Description |
|--------|---------------|------------|---------------|------------|-------------|
| minimal-tokio | 1 | 0 | 15 | 1-29 | Compute-heavy, minimal async (1:15) |
| balanced-tokio | 2 | 0-1 | 14 | 2-29 | General purpose (1:7) |
| equal-split | 8 | 0-7 | 8 | 8-29 | Heavy async + heavy compute (1:1) |

## Executive Summary

The MAB adaptive scheduler evaluation results:

- OK Decision overhead < 200ns (measured: 113ns warm)
- OK Wake latency protection (adaptive: 0.8x vs offload: 0.8x interference)
- OK Starvation differentiation (inline: 1.5x vs adaptive: 0.8x interference)
- OK Throughput on mixed workloads (+324.7% vs always-offload)
- OK Fast convergence (fast: 5 obs, slow: 6 obs)

**Overall:** PASS

---

## 1. Scheduler Overhead

Measures the time cost of MAB decision-making (excluding actual work execution).

| Operation | Mean | P50 | P99 | Std Dev |
|-----------|------|-----|-----|---------|
| choose() cold | 211ns | 104ns | 254ns | 2192ns |
| choose() warm | 113ns | 109ns | 148ns | 79ns |
| Full cycle | 182ns | 172ns | 210ns | 374ns |

**Verdict:** OK Overhead is acceptable (<200ns target for warm decisions)

---

## 2. Wake Latency Impact

Measures how different strategies affect Tokio's async executor responsiveness.
Uses a dedicated **1-tokio-thread** runtime with **3000us work** at **1500/s** rate
to ensure blocking work directly competes with the latency probe.

**Baseline (no load):** P50=981.0us, P95=1980.6us, P99=1982.5us

| Strategy | P50 | P95 | P99 | Interference | Throughput |
|----------|-----|-----|-----|--------------|------------|
| AlwaysInline | 2919.7us | 2923.7us | 2934.3us | 1.5x | 331/s |
| AlwaysOffload | 962.1us | 1682.5us | 1868.8us | 0.8x | 697/s |
| Adaptive (MAB) | 963.1us | 1680.8us | 1867.7us | 0.8x | 686/s |

**Key Observations:**

- AlwaysInline causes 1.5x wake latency degradation (starvation risk)
- AlwaysOffload maintains low interference (0.8x) but pays throughput cost
- Adaptive achieves similar latency protection to AlwaysOffload (0.8x vs 0.8x)

**Verdict:** OK MAB protects Tokio from starvation

---

## 3. Throughput Analysis

Compares `adaptive_map()` vs `compute_map()` (always offload) performance.

| Workload | compute_map | adaptive_map | Speedup |
|----------|-------------|--------------|---------|
| Fast (~10us) | 85761/s | 1392286/s | +1523.4% |
| Medium (~100us) | 55222/s | 209818/s | +280.0% |
| Slow (~500us) | 27802/s | 44509/s | +60.1% |
| Mixed (60% fast, 30% medium, 10% slow) | 58673/s | 249178/s | +324.7% |

**Key Observations:**

- Fast work: 1523% faster (MAB learns to inline)
- Mixed workloads: 325% faster (best of both worlds)

**Verdict:** OK MAB improves throughput on mixed workloads

---

## 4. Learning Behavior

Measures how quickly the MAB converges to correct decisions.

### Convergence Speed

| Work Type | Observations to Stable | Final Decision | Correct? |
|-----------|------------------------|----------------|----------|
| Fast (~20us) | 5 | Inline | Yes |
| Slow (~500us) | 6 | Offload | Yes |

### Decision Breakdown (after training)

| Work Type | Inline % | Offload % |
|-----------|----------|-----------|
| Fast work | 100% | 0% |
| Slow work | 0% | 100% |

### Guardrail Activations

| Guardrail | Description | Triggered? |
|-----------|-------------|------------|
| GR1 | Hard blocking threshold (>250us) | Yes |
| GR3 | Strike suppression (repeated slow) | Yes |

**Verdict:** OK MAB converges quickly and makes correct decisions

---

## 5. Thread Configuration Analysis

Compares different tokio:rayon thread ratios to find optimal configurations.

### Throughput by Configuration (Mixed Workload)

| Config | Tokio:Rayon | compute_map | adaptive_map | Speedup | Starvation Events |
|--------|-------------|-------------|--------------|---------|-------------------|
| minimal-tokio | 1:15 | 59333/s | 242489/s | +308.7% | 0 |
| balanced-tokio | 2:14 | 54140/s | 241836/s | +346.7% | 0 |
| equal-split | 8:8 | 63619/s | 242915/s | +281.8% | 0 |

### Wake Latency by Configuration

| Config | Baseline P95 | Under Load P95 | Interference |
|--------|--------------|----------------|--------------|
| minimal-tokio | 1962.0us | 1860.3us | 0.9x |
| balanced-tokio | 1996.2us | 1677.9us | 0.8x |
| equal-split | 1961.9us | 1672.6us | 0.9x |
### Recommendation

Based on the evaluation:

- **Best for throughput:** equal-split (242915 items/s)
- **Best for latency:** balanced-tokio (0.8x interference)
- **No significant starvation issues detected**
- **Recommended default:** balanced-tokio (2:14) for general purpose workloads

---

## 6. Workload Shift Adaptation

Tests the MAB's ability to adapt when workload characteristics change abruptly.
200 fast observations (20us) followed by 200 slow observations (500us).

### Decision Trajectory

| Observation Range | Inline % | Offload % |
|-------------------|----------|-----------|
| Fast 1-50 | 100% | 0% |
| Fast 51-100 | 100% | 0% |
| Fast 101-150 | 100% | 0% |
| Fast 151-200 | 100% | 0% |
| Slow 1-50 | 14% | 86% |
| Slow 51-100 | 0% | 100% |
| Slow 101-150 | 0% | 100% |
| Slow 151-200 | 0% | 100% |

**Observations to switch after shift:** 10
**Guardrail protection during transition:** No (MAB learned)

---

## 7. Pressure Escalation

Gradually increases concurrent load on a 4-tokio-thread runtime to show
guardrail activation under pressure.

| Load Level | Spawn Rate | Pressure | GR Activations | Inline % | Starvation Events |
|------------|------------|----------|----------------|----------|-------------------|
| Low | 100/s | 0.7 | 0 | 99% | 0 |
| Medium | 1000/s | 2.2 | 0 | 100% | 0 |
| High | 4000/s | 3.8 | 99 | 1% | 0 |
| Very High | 8000/s | 7.6 | 99 | 1% | 0 |
| Extreme | 20000/s | 10.0 | 99 | 1% | 0 |

**Key Finding:** 297 guardrail activations at pressure > 3.0

---

## 8. Runtime Metrics Summary

Cumulative Prometheus counters across all evaluation phases.

| Metric | Value |
|--------|-------|
| Inline Decisions | 50 |
| Offload Decisions | 50 |
| GR1 Activations | 0 |
| GR2 Activations | 0 |
| GR3 Activations | 0 |
| Starvation Events | 0 |
| Last Pressure Index | 10.00 |

---

## Conclusion

The MAB adaptive scheduler in loom-rs delivers value:

1. **Low Overhead** - Decision cost < 200ns doesn't impact performance
2. **Starvation Protection** - Inline causes 1.5x interference; adaptive keeps it at 0.8x
3. **Throughput Optimization** - Learns to inline fast work, offload slow work
4. **Fast Learning** - Converges within ~6 observations
5. **Adaptive** - Detects workload shifts within ~10 observations
6. **Guardrails** - Activate under pressure to prevent starvation

The adaptive approach achieves the best of both worlds: low latency for
small work (by inlining) and protected async responsiveness (by offloading slow work).
