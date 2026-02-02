//! Adaptive Workload Example
//!
//! Demonstrates the MAB scheduler adapting to items with varying compute costs.
//! Shows `ComputeHintProvider` implementation and `adaptive_map()` usage.
//!
//! The MAB learns to:
//! - Inline fast work (trivial/light complexity)
//! - Offload slow work (heavy/massive complexity)
//!
//! Run: cargo run --example adaptive_workload --release

use std::time::Instant;

use futures::stream::{self, StreamExt};
use loom_rs::mab::{ComputeHint, ComputeHintProvider};
use loom_rs::{ComputeStreamExt, LoomBuilder};

/// Work item complexity levels
#[derive(Clone, Copy, Debug)]
enum Complexity {
    /// ~5µs - should be inlined
    Trivial,
    /// ~50µs - borderline, scheduler decides
    Light,
    /// ~500µs - should be offloaded
    Heavy,
    /// ~5ms - definitely offloaded
    Massive,
}

impl Complexity {
    fn target_us(&self) -> u64 {
        match self {
            Complexity::Trivial => 5,
            Complexity::Light => 50,
            Complexity::Heavy => 500,
            Complexity::Massive => 5000,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Complexity::Trivial => "Trivial (~5µs)",
            Complexity::Light => "Light (~50µs)",
            Complexity::Heavy => "Heavy (~500µs)",
            Complexity::Massive => "Massive (~5ms)",
        }
    }
}

/// Work item with variable complexity
#[derive(Clone)]
struct WorkItem {
    id: u64,
    complexity: Complexity,
    data: Vec<u8>,
}

impl WorkItem {
    fn new(id: u64, complexity: Complexity) -> Self {
        // Create data proportional to complexity for a realistic scenario
        let data_size = match complexity {
            Complexity::Trivial => 64,
            Complexity::Light => 1024,
            Complexity::Heavy => 16 * 1024,
            Complexity::Massive => 256 * 1024,
        };
        Self {
            id,
            complexity,
            data: vec![0u8; data_size],
        }
    }
}

/// Implement ComputeHintProvider to guide cold-start behavior
impl ComputeHintProvider for WorkItem {
    fn compute_hint(&self) -> ComputeHint {
        match self.complexity {
            Complexity::Trivial | Complexity::Light => ComputeHint::Low,
            Complexity::Heavy => ComputeHint::Medium,
            Complexity::Massive => ComputeHint::High,
        }
    }
}

/// Processed result
struct ProcessedResult {
    #[allow(dead_code)]
    id: u64,
    checksum: u64,
    elapsed_us: f64,
}

/// Process a work item - cost varies by complexity
fn process_item(item: WorkItem) -> ProcessedResult {
    let start = Instant::now();

    // Calibrated work based on complexity
    // ~100 iterations ≈ 1µs on typical hardware
    let iterations = item.complexity.target_us() * 100;
    let mut checksum = 0u64;

    // Include data in computation to prevent optimization
    for byte in item.data.iter().take(64) {
        checksum = checksum.wrapping_add(*byte as u64);
    }

    // Main compute work
    for i in 0..iterations {
        checksum = checksum.wrapping_add(std::hint::black_box(i));
    }

    let elapsed_us = start.elapsed().as_nanos() as f64 / 1000.0;

    ProcessedResult {
        id: item.id,
        checksum,
        elapsed_us,
    }
}

/// Generate a mixed workload with various complexities
fn generate_mixed_workload(count: usize) -> Vec<WorkItem> {
    let mut items = Vec::with_capacity(count);

    // Distribution: 40% trivial, 30% light, 20% heavy, 10% massive
    for i in 0..count {
        let complexity = match i % 10 {
            0..=3 => Complexity::Trivial,
            4..=6 => Complexity::Light,
            7..=8 => Complexity::Heavy,
            _ => Complexity::Massive,
        };
        items.push(WorkItem::new(i as u64, complexity));
    }

    items
}

/// Statistics per complexity level
struct ComplexityStats {
    count: usize,
    total_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
}

impl Default for ComplexityStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_time_us: 0.0,
            min_time_us: f64::MAX, // Start high so first value becomes minimum
            max_time_us: 0.0,
        }
    }
}

impl ComplexityStats {
    fn add(&mut self, time_us: f64) {
        self.count += 1;
        self.total_time_us += time_us;
        self.min_time_us = self.min_time_us.min(time_us);
        self.max_time_us = self.max_time_us.max(time_us);
    }

    fn avg(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_time_us / self.count as f64
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Workload Example ===\n");
    println!("Demonstrating MAB scheduler with varying compute costs.\n");

    // Create runtime with typical configuration
    let runtime = LoomBuilder::new()
        .prefix("adaptive")
        .tokio_threads(2)
        .rayon_threads(6)
        .build()?;

    println!("Runtime: 2 tokio threads, 6 rayon threads\n");

    // Generate mixed workload
    let item_count = 1000;
    let items = generate_mixed_workload(item_count);

    // Count complexity distribution
    let mut trivial = 0;
    let mut light = 0;
    let mut heavy = 0;
    let mut massive = 0;
    for item in &items {
        match item.complexity {
            Complexity::Trivial => trivial += 1,
            Complexity::Light => light += 1,
            Complexity::Heavy => heavy += 1,
            Complexity::Massive => massive += 1,
        }
    }

    println!("Workload distribution ({} items):", item_count);
    println!(
        "  - Trivial (~5µs):  {} items ({}%)",
        trivial,
        trivial * 100 / item_count
    );
    println!(
        "  - Light (~50µs):   {} items ({}%)",
        light,
        light * 100 / item_count
    );
    println!(
        "  - Heavy (~500µs):  {} items ({}%)",
        heavy,
        heavy * 100 / item_count
    );
    println!(
        "  - Massive (~5ms):  {} items ({}%)",
        massive,
        massive * 100 / item_count
    );
    println!();

    // Store complexity for later analysis
    let complexities: Vec<Complexity> = items.iter().map(|i| i.complexity).collect();

    // Process with adaptive_map
    println!("Processing with adaptive_map()...");
    let start = Instant::now();

    let results: Vec<ProcessedResult> = runtime.block_on(async {
        stream::iter(items)
            .adaptive_map(process_item)
            .collect()
            .await
    });

    let total_duration = start.elapsed();

    // Analyze results by complexity
    let mut stats: [ComplexityStats; 4] = Default::default();
    for (result, complexity) in results.iter().zip(complexities.iter()) {
        let idx = match complexity {
            Complexity::Trivial => 0,
            Complexity::Light => 1,
            Complexity::Heavy => 2,
            Complexity::Massive => 3,
        };
        stats[idx].add(result.elapsed_us);
    }

    // Print results
    println!("\n{}", "=".repeat(70));
    println!("Results");
    println!("{}", "=".repeat(70));
    println!("\nTotal processing time: {:?}", total_duration);
    println!(
        "Throughput: {:.1} items/sec\n",
        results.len() as f64 / total_duration.as_secs_f64()
    );

    println!("Timing by complexity:");
    println!(
        "| {:<20} | {:>8} | {:>10} | {:>10} | {:>10} |",
        "Complexity", "Count", "Avg (µs)", "Min (µs)", "Max (µs)"
    );
    println!(
        "|{:-<22}|{:-<10}|{:-<12}|{:-<12}|{:-<12}|",
        "", "", "", "", ""
    );

    let names = [
        Complexity::Trivial.name(),
        Complexity::Light.name(),
        Complexity::Heavy.name(),
        Complexity::Massive.name(),
    ];
    for (i, name) in names.iter().enumerate() {
        let s = &stats[i];
        if s.count > 0 {
            println!(
                "| {:<20} | {:>8} | {:>10.1} | {:>10.1} | {:>10.1} |",
                name,
                s.count,
                s.avg(),
                s.min_time_us,
                s.max_time_us
            );
        }
    }

    // Verify correctness
    let total_checksum: u64 = results.iter().map(|r| r.checksum).sum();
    println!("\nTotal checksum: {} (for verification)", total_checksum);

    // Explain MAB behavior
    println!("\n{}", "=".repeat(70));
    println!("MAB Behavior Analysis");
    println!("{}", "=".repeat(70));
    println!("\nThe adaptive scheduler uses Multi-Armed Bandit (Thompson Sampling) to learn:");
    println!("  - Trivial/Light items (<50µs): Likely inlined on Tokio worker");
    println!("  - Heavy items (~500µs): Likely offloaded to Rayon");
    println!("  - Massive items (~5ms): Definitely offloaded (guardrails enforce this)");
    println!("\nKey points:");
    println!("  1. ComputeHintProvider guides cold-start behavior");
    println!("  2. MAB learns from actual execution times");
    println!("  3. Guardrails prevent Tokio starvation (never inline >250µs EMA)");
    println!("  4. Each adaptive_map() stream has its own scheduler for fast learning");

    // Compare with always-offload baseline
    println!("\n{}", "=".repeat(70));
    println!("Comparison: Always Offload");
    println!("{}", "=".repeat(70));

    let items2 = generate_mixed_workload(item_count);
    let start2 = Instant::now();

    let _results2: Vec<ProcessedResult> = runtime.block_on(async {
        stream::iter(items2)
            .compute_map(process_item) // Always offloads
            .collect()
            .await
    });

    let offload_duration = start2.elapsed();

    println!("\ncompute_map() (always offload): {:?}", offload_duration);
    println!("adaptive_map() (MAB):           {:?}", total_duration);

    let speedup = offload_duration.as_secs_f64() / total_duration.as_secs_f64();
    if speedup > 1.0 {
        println!(
            "\nAdaptive is {:.1}% faster by inlining fast work",
            (speedup - 1.0) * 100.0
        );
    } else {
        println!(
            "\nAlways-offload is {:.1}% faster (work is mostly heavy)",
            (1.0 / speedup - 1.0) * 100.0
        );
    }

    runtime.block_until_idle();
    Ok(())
}
