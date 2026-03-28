//! Simulated ping-pong between two nodes.
//!
//! Two nodes exchange messages through a simulated network with propagation
//! delay. Each node waits a processing delay before replying. Demonstrates:
//!
//! - `SimHandle::schedule()` for fire-and-forget message delivery
//! - `SimHandle::now()` for reading virtual time
//! - `SimulationRuntime::run()` / `step()` for advancing simulation
//! - DES event ordering: all work at time T completes before advancing
//!
//! ```text
//!   Node A                    Network (10ms)                  Node B
//!     |---- ping #1 ------>      ~~~delay~~~     ------>        |
//!     |                                           process 5ms   |
//!     |<---- pong #1 ------      ~~~delay~~~     <------        |
//!     |  process 5ms                                            |
//!     |---- ping #2 ------>      ...                            |
//! ```
//!
//! Run with: `cargo run --features sim --example sim_ping_pong`

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use loom_rs::sim::{SimHandle, SimulationRuntime};

/// A message exchanged between nodes.
#[derive(Debug, Clone)]
struct Message {
    text: String,
    seq: usize,
}

/// Simulated node that receives messages and replies after a processing delay.
struct Node {
    name: &'static str,
    sim: SimHandle,
    process_time: Duration,
    network_delay: Duration,
    reply_count: Arc<AtomicUsize>,
}

impl Node {
    fn receive(&self, msg: Message, reply_to: Arc<Node>) {
        let now = self.sim.now();
        println!("  [{:>6.1?}] {} received: {:?}", now, self.name, msg.text);

        self.reply_count.fetch_add(1, Ordering::SeqCst);

        // After processing, send a reply back
        let reply = Message {
            text: format!("pong #{}", msg.seq),
            seq: msg.seq,
        };
        let delay = self.process_time + self.network_delay;
        let name = self.name;
        let sim = self.sim.clone();

        self.sim.schedule(delay, move || {
            let delivery_time = sim.now();
            println!(
                "  [{:>6.1?}] {} -> reply delivered: {:?}",
                delivery_time, name, reply.text
            );
            reply_to.receive_reply(reply);
        });
    }

    fn receive_reply(&self, msg: Message) {
        let now = self.sim.now();
        println!("  [{:>6.1?}] {} got reply: {:?}", now, self.name, msg.text);
        self.reply_count.fetch_add(1, Ordering::SeqCst);
    }
}

fn main() -> loom_rs::Result<()> {
    let mut sim = SimulationRuntime::new()?;
    let handle = sim.handle();

    let network_delay = Duration::from_millis(10);
    let process_time = Duration::from_millis(5);

    // Create two nodes
    let node_a = Arc::new(Node {
        name: "Alice",
        sim: handle.clone(),
        process_time,
        network_delay,
        reply_count: Arc::new(AtomicUsize::new(0)),
    });

    let node_b = Arc::new(Node {
        name: "Bob",
        sim: handle.clone(),
        process_time,
        network_delay,
        reply_count: Arc::new(AtomicUsize::new(0)),
    });

    // Schedule 3 pings from Alice to Bob at different times
    for seq in 1..=3 {
        let send_time = Duration::from_millis(seq as u64 * 50);
        let msg = Message {
            text: format!("ping #{}", seq),
            seq,
        };
        let b = node_b.clone();
        let a = node_a.clone();
        let h = handle.clone();

        handle.schedule_at(send_time, move || {
            println!("  [{:>6.1?}] Alice sends: {:?}", h.now(), msg.text);

            // Schedule delivery after network delay
            let deliver_msg = msg.clone();
            let b2 = b.clone();
            b.sim.schedule(network_delay, move || {
                b2.receive(deliver_msg, a);
            });
        });
    }

    println!("=== Ping-Pong Simulation ===\n");

    // Run the full simulation
    let final_time = sim.run()?;

    println!("\n=== Simulation Complete ===");
    println!("  Final virtual time: {:?}", final_time);
    println!(
        "  Alice messages: {}",
        node_a.reply_count.load(Ordering::SeqCst)
    );
    println!(
        "  Bob messages:   {}",
        node_b.reply_count.load(Ordering::SeqCst)
    );

    Ok(())
}
